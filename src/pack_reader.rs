use {
    crate::*,
    minilz4::*,
    std::{
        collections::hash_map::{Entry, HashMap},
        hash::{BuildHasher, Hasher},
        num::NonZeroU64,
        ops::Deref,
        path::PathBuf,
    },
};

/// Result of the successful file data lookup returned by [`PackReader::lookup`] and [`PackReader::lookup_alloc`].
pub enum LookupResult<'b, O> {
    /// The file's data is uncompressed and backed directly by a memory-mapped data pack file.
    Uncompressed(&'b [u8]),
    /// The file's data is compressed and is backed by a dedicated memory allocation.
    /// Either a boxed slice (individual allocation), or an allocated byte slice from a custom allocator.
    Compressed(O),
}

impl<'b, O: AsRef<[u8]>> LookupResult<'b, O> {
    fn as_ref(&self) -> &[u8] {
        match self {
            LookupResult::Uncompressed(data) => *data,
            LookupResult::Compressed(data) => data.as_ref(),
        }
    }
}

impl<'b, O: AsRef<[u8]>> AsRef<[u8]> for LookupResult<'b, O> {
    fn as_ref(&self) -> &[u8] {
        self.as_ref()
    }
}

impl<'b, O: AsRef<[u8]>> Deref for LookupResult<'b, O> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

/// Trait representing an allocator interface used by [`PackReader::lookup_alloc`]
/// to allocate backing memory for decompressed resource files.
pub trait Alloc {
    /// Allocates `len` bytes and returns a mutable reference to the allocated byte slice
    /// to be filled with the decompressed resource file's data.
    ///
    /// Implementation guarantees the length of the returned byte slice is exactly `len`.
    ///
    /// TODO: alignment.
    /// TODO: out of memory handling.
    fn alloc(&self, len: NonZeroU64) -> &mut [u8];
}

/// Provides an interface to lookup resource file data from the resource pack.
pub struct PackReader {
    index: IndexReader,
    packs: Vec<DataPackReader>,
}

impl PackReader {
    /// Attempts to open a resource pack for reading.
    ///
    /// `path` - absolute path to the folder which contains the resource pack's files.
    ///
    /// `checksum_hasher` - hasher used to validate the index checksum,
    /// and, if `validate_checksum` is `true`, to validate the individual resource file data checksums.
    /// Must match the `checksum_hasher` used when [`packing`](pack()); otherwise an error will be returned
    /// ([`PackReaderError::CorruptData`] if `validate_checksum` is `true`, [`PackReaderError::InvalidChecksum`] otherwise).
    ///
    /// `expected_checksum` - if `Some`, this declares the expected checksum / "version" of the opened pack
    /// (as provided by the library when the pack was created) for a quick validity check.
    /// A [`PackReaderError::UnexpectedChecksum`] is returned on mismatch.
    /// E.g., an application executable might be bound to a certain version of the resource pack
    /// (e.g. embedded in the executable itself when building/cooking), treating other versions as invalid.
    ///
    /// `validate_checksum` - if `true`, individual file's checksums are calculated to verify data integrity.
    /// This is (relatively) expensive, memory- and computationally-wise, scaling with total pack size,
    /// as it requires reading and hashing the entire pack's data set.
    /// Other, less expensive integrity checks are still performed even if `validate_checksum` is `false`
    /// (e.g. the index file's checksum is always validated, as well as bound checks on all file data offsets and lengths).
    pub fn new<H: BuildHasher>(
        mut path: PathBuf,
        checksum_hasher: H,
        expected_checksum: Option<Checksum>,
        validate_checksum: bool,
    ) -> Result<Self, PackReaderError> {
        // Read the index file.
        let index = IndexReader::new(&mut path, expected_checksum)?;

        let (lookup_keys, lookup_values) = index.lookup_keys_and_values();

        let mut packs: HashMap<PackIndex, DataPackReader> = HashMap::new();

        let mut index_checksum = checksum_hasher.build_hasher();

        for (path_hash, index_entry) in lookup_keys
            .iter()
            .map(|k| u64_from_bin(*k))
            .zip(lookup_values.iter().map(PackedIndexEntry::unpack))
        {
            // Get or open the pack file.
            let pack: &DataPackReader = match packs.entry(index_entry.pack_index) {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(entry) => entry.insert(DataPackReader::new(
                    &mut path,
                    index_entry.pack_index,
                    index.checksum(),
                )?),
            };

            // Do the bounds checks on the index entrie's offset and length.
            if (index_entry.offset + index_entry.len) > pack.data().len() as _ {
                return Err(PackReaderError::CorruptData);
            }

            // Make sure the uncompressed length is valid.
            if index_entry.uncompressed_len <= index_entry.len && index_entry.uncompressed_len != 0
            {
                return Err(PackReaderError::CorruptData);
            }

            // If required, validate the file data checksum.
            if validate_checksum {
                let file_data = unsafe { pack.slice(index_entry.offset, index_entry.len) };
                let file_checksum = {
                    let mut file_hasher = checksum_hasher.build_hasher();
                    file_hasher.write(file_data);
                    file_hasher.finish()
                };

                if file_checksum != index_entry.checksum {
                    return Err(PackReaderError::CorruptData);
                }
            }

            // Update the index checksum.
            hash_index_entry(
                &mut index_checksum,
                path_hash,
                index_entry.pack_index,
                index_entry.checksum,
                index_entry.len,
            );
        }

        let actual_checksum = index.checksum();
        if index_checksum.finish() != actual_checksum {
            return Err(PackReaderError::InvalidChecksum(actual_checksum));
        }

        let packs = {
            let mut packs_ = Vec::with_capacity(packs.len());

            // Valid pack indices are contiguous integers in range [0 .. <num data packs>].
            // So if any index is missing, we have a malformed index file.
            for pack_index in 0..packs.len() as PackIndex {
                packs_.push(match packs.remove(&pack_index) {
                    Some(data_pack_reader) => data_pack_reader,
                    None => {
                        return Err(PackReaderError::CorruptData);
                    }
                });
            }
            packs_
        };

        Ok(Self { index, packs })
    }

    /// (Optionally) builds the hashmap to accelerate lookups.
    /// Call once after creating the [`PackReader`].
    ///
    /// Provides `O(1)` lookups at the cost of extra memory.
    /// Otherwise lookups are `O(n)`.
    pub fn build_lookup(&mut self) {
        self.index.build_lookup();
    }

    /// Attempts to look up the data for the resource file with `path_hash`,
    /// uncompressing (and allocating the buffer as a boxed slice) if necessary.
    pub fn lookup(&self, path_hash: PathHash) -> Option<LookupResult<'_, Box<[u8]>>> {
        self.lookup_impl(path_hash, |src, uncompressed_len| {
            decompress_to_vec(src, uncompressed_len)
                .expect("failed to decompress")
                .into_boxed_slice()
        })
    }

    /// Attempts to look up the data for the resource file with `path_hash`,
    /// uncompressing (and allocating the buffer using the user-provided allocator `alloc`) if necessary.
    pub fn lookup_alloc<'a, A: Alloc>(
        &self,
        path_hash: PathHash,
        alloc: &'a A,
    ) -> Option<LookupResult<'_, &'a [u8]>> {
        self.lookup_impl(path_hash, |src, uncompressed_len| {
            let dst = alloc.alloc(uncompressed_len);
            debug_assert_eq!(dst.len() as FileSize, uncompressed_len.get());
            decompress(src, dst).expect("failed to decompress");
            dst as _
        })
    }

    fn lookup_impl<O, F: FnOnce(&[u8], NonZeroU64) -> O>(
        &self,
        path_hash: PathHash,
        f: F,
    ) -> Option<LookupResult<'_, O>> {
        if let Some(entry) = self.index.lookup(path_hash) {
            debug_assert!(entry.pack_index < self.packs.len() as _);
            let pack = unsafe { self.packs.get_unchecked(entry.pack_index as usize) };
            let data = unsafe { pack.slice(entry.offset, entry.len) };

            Some(if let Some(uncompressed_len) = entry.is_compressed() {
                LookupResult::Compressed(f(data, uncompressed_len))
            } else {
                LookupResult::Uncompressed(data)
            })
        } else {
            None
        }
    }
}
