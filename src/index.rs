use {
    crate::*,
    memmap2::*,
    miniunchecked::*,
    static_assertions::*,
    std::{
        collections::HashMap,
        fs::File,
        hash::Hasher,
        io::{self, Seek, SeekFrom, Write},
        iter::Iterator,
        mem,
        num::NonZeroU64,
        path::PathBuf,
    },
};

const INDEX_FILE_NAME: &str = "index";
const INDEX_HEADER_MAGIC: u32 = 0x696b6170; // `paki`, little endian.

/// Header of the pack's index file.
///
/// Index file provides a lookup from the resource file's path hash
/// to information about the file's data physical location in the data pack file(s).
///
/// Pack index file has the following layout:
///
/// | Header                    | `IndexHeader`              | 16b                      |
/// | Path hashes               | `[PathHash]`               | 8b * `<len>`             |
/// | Index entries             | `[PackedIndexEntry]`       | 32b * `<len>`            |
///
/// The length `len` of the path hash and index entry arrays is implicit, determined by the index file size minus the header.
/// Path hash and index entry arrays are the keys and values, respectively, of the lookup data structure.
#[repr(C, packed)]
pub(crate) struct IndexHeader {
    /// INDEX_HEADER_MAGIC
    magic: u32,
    /// Always `0`.
    _padding: u32,
    /// The pack's checksum, and its authoritative/content-driven "version".
    /// Calculated as the checksum/hash of all tuples of files' (path hash, pack index, checksum, len),
    /// in order they are found in the index.
    checksum: Checksum,
}

impl IndexHeader {
    pub(crate) fn check_magic(&self) -> bool {
        u32_from_bin(self.magic) == INDEX_HEADER_MAGIC && self._padding == 0
    }

    pub(crate) fn checksum(&self) -> Checksum {
        u64_from_bin(self.checksum)
    }

    pub(crate) fn write<W: Write>(checksum: Checksum, w: &mut W) -> Result<usize, io::Error> {
        let mut written = 0;

        written += write_u32(w, INDEX_HEADER_MAGIC)?;
        written += write_u32(w, 0)?;
        written += write_u64(w, checksum)?;

        debug_assert_eq!(written, std::mem::size_of::<IndexHeader>());

        Ok(written)
    }
}

/// Information about a single resource file stored in the pack, as located in the index file.
///
/// This is the value part of the key-value pair of the pack index, where the key is the file's path hash.
///
/// Also see `IndexEntry`.
#[repr(C, packed)]
pub(crate) struct PackedIndexEntry {
    /// Index of the data pack the file is in (top two bytes),
    /// and size in bytes of the file's data within the containing data pack (bottom six bytes).
    pack_index_and_size: PackIndexAndSize,
    /// Offset in bytes to the start of the file's data within the containing data pack (starting past the data pack's header).
    offset: Offset,
    /// Checksum / hash of the file's data (as stored in the data pack, i.e. optionally compressed).
    checksum: Checksum,
    /// Either
    /// - `0`, in which case the file's data is stored within the containing data pack uncompressed, or
    /// - `> len`, in which case the file's data is stored compressed, and this is its original uncompressed length in bytes.
    uncompressed_len: FileSize,
}

impl PackedIndexEntry {
    pub(crate) fn unpack(&self) -> IndexEntry {
        let (pack_index, size) = unpack_pack_index_and_size(u64_from_bin(self.pack_index_and_size));

        let uncompressed_len = u64_from_bin(self.uncompressed_len);

        if uncompressed_len == 0 {
            IndexEntry::new_uncompressed(
                pack_index,
                u64_from_bin(self.offset),
                size,
                u64_from_bin(self.checksum),
            )
        } else {
            debug_assert!(uncompressed_len > size);
            IndexEntry::new_compressed(
                pack_index,
                u64_from_bin(self.offset),
                size,
                u64_from_bin(self.checksum),
                uncompressed_len,
            )
        }
    }
}

/// Two highest bytes: pack index, lower bytes: data length in bytes.
type PackIndexAndSize = u64;

const LEN_BITS: PackIndexAndSize = 48;
const SIZE_OFFSET: PackIndexAndSize = 0;
const MAX_FILE_SIZE: PackIndexAndSize = (1 << LEN_BITS) - 1;
const SIZE_MASK: PackIndexAndSize = MAX_FILE_SIZE << SIZE_OFFSET;

const PACK_INDEX_BITS: PackIndexAndSize = 16;
const PACK_INDEX_OFFSET: PackIndexAndSize = LEN_BITS;
const MAX_PACK_INDEX: PackIndexAndSize = (1 << PACK_INDEX_BITS) - 1;
const PACK_INDEX_MASK: PackIndexAndSize = MAX_PACK_INDEX << PACK_INDEX_OFFSET;

const_assert!(
    PACK_INDEX_BITS + LEN_BITS == (mem::size_of::<PackIndexAndSize>() as PackIndexAndSize) * 8
);

// See `pack_pack_index_and_size`.
fn unpack_pack_index_and_size(pack_index_and_size: PackIndexAndSize) -> (PackIndex, FileSize) {
    (
        ((pack_index_and_size & PACK_INDEX_MASK) >> PACK_INDEX_OFFSET) as PackIndex,
        ((pack_index_and_size & SIZE_MASK) >> SIZE_OFFSET) as FileSize,
    )
}

// See `unpack_pack_index_and_size`.
fn pack_pack_index_and_size(pack_index: PackIndex, size: FileSize) -> PackIndexAndSize {
    debug_assert!(size > 0);
    // Maximum file size we can encode is 48 bits, or 256 terabytes, which is more than enough.
    debug_assert!(size <= MAX_FILE_SIZE);
    // Maximum `pack_index` value we can encode is 16 bits, or 64k packs, which is more than enough.
    debug_assert!((pack_index as PackIndexAndSize) <= MAX_PACK_INDEX);

    (((pack_index as PackIndexAndSize) << PACK_INDEX_OFFSET) & PACK_INDEX_MASK)
        | (((size as PackIndexAndSize) << SIZE_OFFSET) & SIZE_MASK)
}

/// See `PackedIndexEntry`.
#[derive(Clone, Copy)]
pub(crate) struct IndexEntry {
    pub(crate) pack_index: PackIndex,
    pub(crate) offset: Offset,
    pub(crate) size: FileSize,
    pub(crate) checksum: Checksum,
    pub(crate) uncompressed_len: FileSize,
}

impl IndexEntry {
    pub(crate) fn new_compressed(
        pack_index: PackIndex,
        offset: Offset,
        size: FileSize,
        checksum: Checksum,
        uncompressed_len: FileSize,
    ) -> Self {
        debug_assert!(uncompressed_len > size);
        Self {
            pack_index,
            offset,
            size,
            checksum,
            uncompressed_len,
        }
    }

    pub(crate) fn new_uncompressed(
        pack_index: PackIndex,
        offset: Offset,
        size: FileSize,
        checksum: Checksum,
    ) -> Self {
        Self {
            pack_index,
            offset,
            size,
            checksum,
            uncompressed_len: 0,
        }
    }

    pub(crate) fn write<W: Write>(&self, w: &mut W) -> Result<usize, std::io::Error> {
        let mut written = 0;

        written += write_u64(w, pack_pack_index_and_size(self.pack_index, self.size))?;
        written += write_u64(w, self.offset)?;
        written += write_u64(w, self.checksum)?;
        written += write_u64(w, self.uncompressed_len)?;

        debug_assert_eq!(written, std::mem::size_of::<PackedIndexEntry>());

        Ok(written)
    }

    /// Returns the uncompressed size in bytes of the source file
    /// represented by this index entry if it is compressed; otherwise returns `None`.
    pub(crate) fn is_compressed(&self) -> Option<NonZeroU64> {
        debug_assert!(self.uncompressed_len == 0 || (self.uncompressed_len > self.size));
        NonZeroU64::new(self.uncompressed_len)
    }
}

pub(crate) struct IndexWriter<H: Hasher> {
    file: File,
    /// The pack's total checksum, updated via `hash_index_entry()` with each added index entry
    /// and finalized on `write`.
    checksum: H,
}

impl<H: Hasher> IndexWriter<H> {
    pub(crate) fn new(checksum: H, path: &mut PathBuf) -> Result<Self, PackError> {
        // Create the index file.
        let path = PathPopGuard::push(path, INDEX_FILE_NAME);
        let file = File::create(&path).map_err(PackError::FailedToWriteIndexFile)?;

        Ok(Self { file, checksum })
    }

    /// Takes an iterator builder over the gathered (`PathHash`, `IndexEntry`) tuples,
    /// consumes the writer and writes all its data to the index file.
    /// Returns the calculated pack's total checksum.
    pub(crate) fn write<F, I>(mut self, path_hashes_and_entries: F) -> Result<Checksum, PackError>
    where
        F: Fn() -> I,
        I: Iterator<Item = (PathHash, IndexEntry)>,
    {
        // Write the index file.
        || -> _ {
            // Write the dummy header.
            IndexHeader::write(0, &mut self.file)?;

            // Write all file path hashes.
            for (path_hash, _) in path_hashes_and_entries() {
                write_u64(&mut self.file, path_hash)?;
            }

            // Write all the index entries, while hashing them and generating the index checksum.
            for (path_hash, entry) in path_hashes_and_entries() {
                hash_index_entry(
                    &mut self.checksum,
                    path_hash,
                    entry.pack_index,
                    entry.checksum,
                    entry.size,
                );

                entry.write(&mut self.file)?;
            }

            // Get the final checksum and re-write the header with the now correct checksum value.
            let total_checksum = self.checksum.finish();

            self.file.seek(SeekFrom::Start(0))?;

            IndexHeader::write(total_checksum, &mut self.file)?;

            self.file.flush()?;

            Ok(total_checksum)
        }()
        .map_err(PackError::FailedToWriteIndexFile)
    }
}

pub(crate) struct IndexReader {
    map: Mmap,
    lookup: Option<HashMap<PathHash, IndexEntry>>,
}

impl IndexReader {
    pub(crate) fn new(
        path: &mut PathBuf,
        expected_checksum: Option<Checksum>,
    ) -> Result<Self, PackReaderError> {
        // Open and map the index file.
        let map = || -> _ {
            let path = PathPopGuard::push(path, INDEX_FILE_NAME);
            let file = File::open(&path)?;
            unsafe { Mmap::map(&file) }
        }()
        .map_err(PackReaderError::FailedToOpenIndexFile)?;

        if !Self::validate_blob(&map) {
            return Err(PackReaderError::InvalidIndexFile);
        }

        // Check whether the index checksum matches the expected value.
        if let Some(expected_checksum) = expected_checksum {
            let actual_checksum = unsafe { Self::header(&map) }.checksum();
            if actual_checksum != expected_checksum {
                return Err(PackReaderError::UnexpectedChecksum(actual_checksum));
            }
        }

        Ok(Self { map, lookup: None })
    }

    pub(crate) fn checksum(&self) -> Checksum {
        unsafe { Self::header(&self.map) }.checksum()
    }

    fn lookup_keys_and_values(&self) -> (&[PathHash], &[PackedIndexEntry]) {
        // We've made sure the index file is valid.
        unsafe { Self::lookup_keys_and_values_impl(&self.map) }
    }

    pub(crate) fn path_hashes_and_index_entries(
        &self,
    ) -> impl Iterator<Item = (PathHash, IndexEntry)> + '_ {
        let (lookup_keys, lookup_values) = self.lookup_keys_and_values();

        lookup_keys
            .iter()
            .cloned()
            .map(u64_from_bin)
            .zip(lookup_values.iter().map(PackedIndexEntry::unpack))
    }

    /// (Optionally) builds the hashmap to accelerate lookups.
    /// Call once after creating the [`IndexReader`].
    ///
    /// Provides `O(1)` lookups at the cost of extra memory.
    /// Otherwise lookups are `O(n)`.
    pub(crate) fn build_lookup(&mut self) {
        if self.lookup.is_none() {
            // We've made sure the index file is valid.
            let (lookup_keys, lookup_values) =
                unsafe { Self::lookup_keys_and_values_impl(&self.map) };

            self.lookup.replace(
                lookup_keys
                    .iter()
                    .cloned()
                    .map(u64_from_bin)
                    .zip(lookup_values.iter().map(PackedIndexEntry::unpack))
                    .collect(),
            );
        }
    }

    pub(crate) fn lookup(&self, path_hash: PathHash) -> Option<IndexEntry> {
        if let Some(lookup) = self.lookup.as_ref() {
            lookup.get(&path_hash).copied()
        } else {
            let (lookup_keys, lookup_values) = self.lookup_keys_and_values();

            // NOTE - binary search relies on path hash sorting when packing.
            if let Ok(idx) = lookup_keys.binary_search_by(|&key| u64_from_bin(key).cmp(&path_hash))
            {
                Some(unsafe { lookup_values.get_unchecked_dbg(idx) }.unpack())
            } else {
                None
            }
        }
    }

    /// The caller guarantees the `data` is at least large enough for a header and one key-value pair.
    unsafe fn lookup_keys_and_values_impl(data: &[u8]) -> (&[PathHash], &[PackedIndexEntry]) {
        let len = Self::len(data.len() as _);
        (
            Self::slice(data, mem::size_of::<IndexHeader>() as _, len),
            Self::slice(
                data,
                mem::size_of::<IndexHeader>() as Offset
                    + len * mem::size_of::<PathHash>() as Offset,
                len,
            ),
        )
    }

    fn validate_blob(data: &[u8]) -> bool {
        // Check if the data is at least large enough to hold the smallest possible index blob (header + one key-value pair).
        let header_size = mem::size_of::<IndexHeader>();
        let lookup_pair_size = mem::size_of::<PathHash>() + mem::size_of::<PackedIndexEntry>();
        let min_size = header_size + lookup_pair_size;

        if data.len() < min_size {
            return false;
        }

        let header = unsafe { Self::header(data) };

        // Check the header magic.
        if !header.check_magic() {
            return false;
        }

        let payload_size = data.len() - header_size;

        // Must contain an integer number of lookup pairs.
        if payload_size % lookup_pair_size != 0 {
            return false;
        }

        true
    }

    /// The caller guarantees the `data` is at least large enough to contain an `IndexHeader`.
    unsafe fn header(data: &[u8]) -> &IndexHeader {
        &*(data.as_ptr() as *const _)
    }

    /// Calculates the length of the index lookup in elements (i.e. pairs of (path hash, index entry)).
    /// The caller guarantees `data_size` contains an integer number of lookup elements.
    fn len(data_size: FileSize) -> FileSize {
        let header_size = mem::size_of::<IndexHeader>() as FileSize;
        let lookup_pair_size =
            (mem::size_of::<PathHash>() + mem::size_of::<PackedIndexEntry>()) as FileSize;
        debug_assert!(data_size > header_size);
        let payload_size = data_size - header_size;
        payload_size / lookup_pair_size
    }

    /// Returns a subslice within the `data` blob at `offset` (in bytes) from the start, with `len` `T` elements.
    /// The caller guarantees `offset` and `len` are valid.
    unsafe fn slice<T>(data: &[u8], offset: Offset, len: FileSize) -> &[T] {
        debug_assert!(offset + len * (mem::size_of::<T>() as Offset) <= data.len() as _);

        std::slice::from_raw_parts(data.as_ptr().offset(offset as _) as *const _, len as _)
    }
}
