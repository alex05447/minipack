use {
    crate::*,
    memmap2::*,
    miniunchecked::*,
    std::{
        collections::hash_map::HashMap,
        fs::{File, OpenOptions},
        io::{self, Seek, SeekFrom, Write},
        mem,
        path::PathBuf,
    },
};

const PACK_FILE_NAME: &str = "pack";
const PACK_HEADER_MAGIC: u32 = 0x646b6170; // `pakd`, little endian.

/// Header of the data pack file.
///
/// Data pack files simply contain the (potentially compressed) data of the packed resource files.
/// Index into (via pack index and data pack file byte offset) by the entries in the index file (see `PackedIndexEntry`).
///
/// NOTE: data is stored contiguously, but nothing prevents an otherwise valid data pack to contain holes / unused space.
/// This might appear, e.g., during patching, and is removed during a separate defragmentation step.
#[repr(C, packed)]
pub(crate) struct DataPackHeader {
    /// PACK_HEADER_MAGIC
    magic: u32,
    /// Always `0`.
    _padding: u32,
    /// Must match `IndexHeader::checksum`.
    checksum: u64,
}

impl DataPackHeader {
    fn check_magic(&self) -> bool {
        u32_from_bin(self.magic) == PACK_HEADER_MAGIC && self._padding == 0
    }

    fn checksum(&self) -> Checksum {
        u64_from_bin(self.checksum)
    }

    fn write<W: Write>(checksum: u64, w: &mut W) -> Result<usize, io::Error> {
        let mut written = 0;

        written += write_u32(w, PACK_HEADER_MAGIC)?;
        written += write_u32(w, 0)?;
        written += write_u64(w, checksum)?;

        debug_assert_eq!(written, std::mem::size_of::<DataPackHeader>());

        Ok(written)
    }
}

/// The smallest valid data pack size contains at least a header an a single one-byte file.
fn min_data_pack_size() -> u64 {
    let header_size = mem::size_of::<DataPackHeader>();
    let min_size = header_size + 1;

    min_size as _
}

/// A single data pack file, as written to by the `DataPackWriter`.
struct DataPack {
    file: File,
    /// Offset to the next written resource file's data in bytes,
    /// relative to the start of the data pack file's payload, past the `DataPackHeader` (so it starts at `0`).
    offset: Offset,
    /// Amount of space in bytes reserved in this pack for to-be-compressed source data files.
    /// See [`DataPackWriter::reserve`] / [`DataPackWriter::write_reserved`].
    reserved: Offset,
    // Only needed to (potentially) delete a now-empty data pack file on compaction after compression.
    path: PathBuf,
}

impl DataPack {
    fn new(index: PackIndex, path: &PathBuf) -> Result<Self, PackError> {
        // Create the data pack's file.
        let (file, path) = || -> _ {
            let mut path = path.clone();
            path.push(format!("{}{}", PACK_FILE_NAME, index));
            // Same as `File::create()`, but also need the read access for (potential) compaction after compression.
            let mut file = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .read(true)
                .open(&path)?;

            // Write the dummy data pack file header.
            // Header will be updated with the correct checksum later,
            // after the total pack checksum is calculated when all source files are processed.
            DataPackHeader::write(0, &mut file)?;

            Ok((file, path))
        }()
        .map_err(|err| PackError::FailedToWritePackFile((index, err)))?;

        Ok(Self {
            file,
            offset: 0,
            reserved: 0,
            path,
        })
    }

    fn remaining(&self, max_pack_size: Offset) -> Offset {
        let header_size = mem::size_of::<DataPackHeader>() as Offset;

        debug_assert!(max_pack_size > header_size);
        let payload_size = max_pack_size - header_size;

        // NOTE: `offset` might be larger than maximum payload size if the data pack file contains a single large file.
        payload_size.saturating_sub(self.offset + self.reserved)
    }

    fn is_empty(&self) -> bool {
        self.offset == 0
    }

    /// Writes the resource file's (optionally compressed) `data` to the data pack file.
    /// Returns the offset in bytes to the start of the file's data in the data pack file past the `DataPackHeader`.
    fn write(&mut self, data: &[u8]) -> Result<Offset, io::Error> {
        self.file.write_all(data)?;

        let offset = self.offset;
        self.offset += data.len() as Offset;

        Ok(offset)
    }

    /// Reserves `data_size` bytes in this pack for a to-be-compressed resource file.
    /// Followed by an eventual call to `write_reserved()`, whether the file was successfully compressed or not.
    fn reserve(&mut self, data_size: Offset) {
        self.reserved += data_size;
    }

    /// Writes the resource file's (optionally compressed) `data` to the data pack file.
    /// Returns the offset in bytes to the start of the file's data in the data pack file past the `DataPackHeader`.
    ///
    /// The caller guarantees space was previously reserved in this pack for this file's uncompressed data.
    fn write_reserved(&mut self, data: &[u8]) -> Result<Offset, io::Error> {
        debug_assert!(self.reserved >= data.len() as _);
        self.reserved -= data.len() as Offset;

        self.write(data)
    }

    /// Updates the data pack file's header with the correct total pack checksum after all resource files have been processed.
    fn write_header(&mut self, checksum: Checksum) -> Result<(), io::Error> {
        self.file.seek(SeekFrom::Start(0))?;
        DataPackHeader::write(checksum, &mut self.file)?;
        self.file.flush()
    }

    /// User guarantees the `offset` and `size` are valid for this data pack file.
    fn read(data: &[u8], mut offset: Offset, size: FileSize) -> &[u8] {
        let header_size = mem::size_of::<DataPackHeader>() as Offset;
        // Offset is relative to the data pack's payload (after the header), so need to offset by header size.
        offset += header_size;
        unsafe { data.get_unchecked_dbg(offset as usize..(offset + size) as usize) }
    }

    /// Compacts the data pack file.
    ///
    /// Given
    /// - `pack_index`, the current data pack's index,
    /// - `packs`, array of all existing data packs
    /// (can't pass a mutable slice because of borrow checker; we pinky swear not to modify the current pack through it),
    /// - `pack_index_lookup`, the lookup for the source file's deterministic pack index
    /// (i.e. one which would be assigned with singlethreaded compression),
    /// - `index`, the current source file path hash to index entry lookup,
    ///
    /// we find all source files currently in this data pack (using `index`) but should be in a different pack (using `pack_index_lookup`),
    /// read their data from the current data pack,
    /// append the data to the correct data pack,
    /// and then defragment the current data pack, filling the holes
    /// potentially remaining after some of the files were moved to a different pack.
    ///
    /// This modifies the index entries in `index` for moved source files with new pack indices and data offsets.
    ///
    /// `pack_files_buf` is a helper buffer to avoid allocating a `Vec` of source files in the current pack for each pack.
    fn compact(
        &mut self,
        pack_index: PackIndex,
        packs: PacksWrapper,
        pack_index_lookup: &HashMap<PathHash, PackIndex>,
        index: &mut [(PathHash, IndexEntry)],
        pack_files_buf: VecAllocation,
    ) -> Result<VecAllocation, PackError> {
        // Map the current data pack file for reading.
        // The file must have been open with read acces.
        let pack_data = unsafe { Mmap::map(&self.file) }
            .map_err(|err| PackError::FailedToWritePackFile((pack_index, err)))?;

        // Gather from `index` all the source files which are currently located in this data pack.
        let mut pack_files = empty_vec_from_allocation(pack_files_buf);
        pack_files.extend(index.iter_mut().filter_map(|(path_hash, index_entry)| {
            (index_entry.pack_index == pack_index).then(|| (*path_hash, index_entry))
        }));

        // For each source file in the current data pack ...
        for (path_hash, pack_file) in pack_files.iter_mut() {
            // ... determine the pack index this file should be in.
            let dst_pack_index = *unsafe {
                pack_index_lookup
                    .get(path_hash)
                    .unwrap_unchecked_dbg_msg("invalid path hash")
            };

            // If the file should be in a different pack ...
            if dst_pack_index != pack_index {
                // ... append the data for the current file in this pack to the found destination pack.
                let dst_pack = packs.get(dst_pack_index);

                let new_offset = dst_pack
                    .write(Self::read(&pack_data, pack_file.offset, pack_file.size))
                    .map_err(|err| PackError::FailedToWritePackFile((dst_pack_index as _, err)))?;

                // Patch the entry for the moved file with the new pack index and offset within it.
                // Length/checksum are of course unchanged.
                pack_file.pack_index = dst_pack_index as PackIndex;
                pack_file.offset = new_offset;
            }
        }

        // Unmap the current data pack file for reading.
        mem::drop(pack_data);

        // Defragment the current data pack file.
        self.defragment(pack_index, &mut pack_files)
            .map_err(|err| PackError::FailedToWritePackFile((pack_index, err)))?;

        // Return the helper buffer to be reused.
        Ok(unsafe { empty_vec_into_allocation(pack_files) })
    }

    /// Defragments the data pack file.
    ///
    /// Given
    /// - `pack_index`, the current data pack's index,
    /// - `pack_files`, the list of source files which were present in this data pack file
    /// but might have been moved to a different data pack file,
    ///
    /// moves/compacts/defragments the data for all source files which are still located in this data pack file
    /// such that there are no unoccupied holes in the data pack file.
    ///
    /// This modifies the index entries in `pack_files` for moved source files with new data offsets within this data pack file.
    ///
    /// Truncates the data pack file, modifies the file pointer.
    fn defragment<T>(
        &mut self,
        pack_index: PackIndex,
        pack_files: &mut [(T, &mut IndexEntry)],
    ) -> Result<(), io::Error> {
        // Map the current data pack file for writing.
        let mut pack_data = unsafe { MmapMut::map_mut(&self.file) }?;
        let data = pack_data.as_mut_ptr();

        let header_size = mem::size_of::<DataPackHeader>() as Offset;

        // First, must sort source files according to their offset (even those from the wrong packs, they will be ignored below).
        pack_files.sort_by(|(_, l), (_, r)| l.offset.cmp(&r.offset));

        // For all source files which remained in this data pack file and which need to be moved/compacted/defragmented
        // (i.e. at least one source file located before them in the data pack was moved to a different data pack, leaving a hole) ...
        let mut new_offset = 0;
        for (pack_file, new_offset_) in pack_files.iter_mut().filter_map(|(_, pack_file)| {
            if pack_file.pack_index == pack_index {
                let offset = new_offset;
                new_offset += pack_file.size;
                (pack_file.offset != offset).then(|| {
                    // Source files only move upwards when defragmenting.
                    debug_assert!(pack_file.offset > offset);
                    (pack_file, offset)
                })
            } else {
                None
            }
        }) {
            debug_assert!(new_offset_ <= pack_file.offset);

            // ... copy the file's data from its current location in the data pack file to the new offset.
            let src = Self::read(&pack_data, pack_file.offset, pack_file.size).as_ptr();
            // Must account for the header size as offsets are relative to the data pack file's payload (past the header).
            let dst = unsafe { data.offset(header_size as isize + new_offset_ as isize) };

            // May overlap.
            unsafe { std::ptr::copy(src, dst, pack_file.size as _) };

            // Patch the entry for the moved file with the new offset within the curent data pack file.
            // Length/checksum are of course unchanged.
            pack_file.offset = new_offset_;
        }

        // Unmap the current data pack file for writing.
        mem::drop(pack_data);

        // Truncate the data pack file, modify the file pointer.
        self.offset = new_offset;
        self.file.set_len((header_size + new_offset) as _)?;
        let _pos = self.file.seek(SeekFrom::Start(header_size + new_offset))?;
        debug_assert_eq!(_pos, header_size + new_offset);

        Ok(())
    }
}

/// Writes resource file data to data pack files during packing.
///
/// Also responsible for determining which data pack file a resource file goes to.
/// Ideally this should be completely determined by the user application,
/// but for now a very simple greedy model is used with only one parameter - maximum data pack file size.
///
/// TODO: revisit this.
pub(crate) struct DataPackWriter {
    //packs2: VirtualDataPackWriter,
    packs: Vec<DataPack>,
    // Maximum size of written data pack files (including the `DataPackHeader`), which determines the number of data pack files
    // created during packing, and, by a greedy first-fit algorithm, which data pack file a resource file goes to.
    //
    // NOTE: a data pack file may be larger than this value in case a resource file exceeds the limit
    // (in which case the data pack file will only contain this file's data).
    //
    // TODO: revisit this.
    max_pack_size: Option<Offset>,
}

impl DataPackWriter {
    pub(crate) fn new(max_pack_size: Option<u64>) -> Self {
        Self {
            packs: Vec::new(),
            max_pack_size: max_pack_size
                .map(|max_pack_size| max_pack_size.max(min_data_pack_size())),
        }
    }

    /// Writes the resource file's (optionally compressed) `data` to one of the data pack files.
    ///
    /// Uses the first one with enough space, or adds a new one.
    /// TODO: revisit this.
    ///
    /// Returns the data pack index and offset in bytes to the start of the file's data in the data pack file past the header.
    pub(crate) fn write(
        &mut self,
        data: &[u8],
        path: &PathBuf,
    ) -> Result<(PackIndex, Offset), PackError> {
        let (pack_index, pack) = self.find_or_add_pack(data.len() as _, path)?;

        // Write the resource file's data to the data pack file and return the pack index and offset.
        let offset = pack
            .write(data)
            .map_err(|err| PackError::FailedToWritePackFile((pack_index as _, err)))?;

        Ok((pack_index, offset))
    }

    /// Reserves space for `data_size` bytes in one of the data pack files.
    ///
    /// This is done to support multithreaded source file data compression.
    /// We know we won't compress files which are same size or larger when compressed.
    /// So we reserve some space in the data pack file, return its index and move on.
    /// When the source file is compressed, we'll write the data to the data pack file
    /// we reserved space in with [`DataPackWriter::write_reserved`].
    ///
    /// This does result in data pack files generally smaller than requested size,
    /// which is fixed with a compaction post-process.
    ///
    /// Uses the first one with enough space, or adds a new one.
    /// TODO: revisit this.
    ///
    /// Returns the data pack index and offset in bytes to the start of the file's data in the data pack file past the header.
    pub(crate) fn reserve(
        &mut self,
        data_size: FileSize,
        path: &PathBuf,
    ) -> Result<PackIndex, PackError> {
        self.find_or_add_pack(data_size, path)
            .map(|(pack_index, pack)| {
                pack.reserve(data_size);
                pack_index
            })
    }

    /// See [`DataPackWriter::reserve`].
    /// Same as [`DataPackWriter::write`], but we already know the `pack_index`.
    ///
    /// Caller guarantees `pack_index` is actually the one we [`DataPackWriter::reserve`]'d space in.
    pub(crate) fn write_reserved(
        &mut self,
        data: &[u8],
        pack_index: PackIndex,
    ) -> Result<Offset, PackError> {
        debug_assert!(pack_index < self.packs.len() as _);
        let pack = unsafe { self.packs.get_unchecked_mut(pack_index as usize) };

        // Write the (potentially compressed) resource file's data to the data pack file and return the offset.
        pack.write_reserved(data)
            .map_err(|err| PackError::FailedToWritePackFile((pack_index as _, err)))
    }

    /// Updates all the data pack files' headers with the correct total pack checksum after all resource files have been processed.
    pub(crate) fn write_headers(&mut self, checksum: Checksum) -> Result<(), PackError> {
        Ok(
            for (pack_index, pack) in self.packs.iter_mut().enumerate() {
                pack.write_header(checksum)
                    .map_err(|err| PackError::FailedToWritePackFile((pack_index as _, err)))?;
            },
        )
    }

    /// Compacts the data pack files.
    ///
    /// Given `index`, the current source file path hash to index entry lookup
    /// (up-to-date after all files have just been compressed),
    /// we determine the deterministic pack index for each source file
    /// (i.e. one which would be assigned with singlethreaded compression),
    /// and move the source file data to the "correct" data pack,
    /// thus compacting the data packs and removing any empty ones.
    ///
    /// I.e. turns this: ...         ... into this:
    ///
    /// xxxx  xxxx  xxxx  xxxx      xxxx  xxxx  xxxx
    /// xxxx  xxxx  xxxx  xxxx      xxxx  xxxx  0000
    /// xxxx  0000  xxxx  0000      xxxx  xxxx  0000
    /// 0000  0000  xxxx  0000      xxxx  xxxx  0000
    /// 0000  0000  0000  0000      xxxx  xxxx  0000
    ///
    /// pack0 pack1 pack2 pack3     pack0 pack1 pack2
    ///
    /// The other reason this must be done is packing determinism between single- and multithreaded compression.
    /// Remember that currently, for better or worse, the index entrie's data pack file index is part the data
    /// which determines the checksum / "version" of the generated data pack (see `hash_index_entry()`).
    /// I.e. identical packs in terms of source file contents, but with at least one file differing in its pack index
    /// are going to have different checksums.
    ///
    /// Multithreaded compression necessarily results in different pack indices reserved for source files
    /// (and smaller packs after compression) when that decision is based on their uncompressed size.
    /// A naive post-process compaction scheme, simply moving source file data to any available data pack with a smaller index
    /// will not result in the same pack index selection as the singlethreaded packing process.
    ///
    /// This modifies the index entries in `index` for moved source files with new pack indices and data offsets.
    ///
    /// NOTE: does nothing if we don't use a data pack file size limit, or if there's only one data pack file.
    pub(crate) fn compact(
        &mut self,
        index: &mut [(PathHash, IndexEntry)],
        before_compaction: Option<Box<BeforeCompactionCallback>>,
        mut on_pack_compacted: Option<Box<OnPackCompactedCallback>>,
        after_compaction: Option<Box<AfterCompactionCallback>>,
    ) -> Result<(), PackError> {
        // There's only one data pack file if we don't use a data pack file size limit.
        let max_pack_size = if let Some(max_pack_size) = self.max_pack_size {
            max_pack_size
        } else {
            debug_assert_eq!(self.packs.len(), 1);
            return Ok(());
        };

        let prev_num_packs = self.num_packs();

        // Call the progress callback.
        before_compaction.map(|before_compaction| {
            before_compaction(BeforeCompaction {
                num_packs: prev_num_packs,
            })
        });

        // No need for compaction if there's only one data pack file (or even none at all).
        if prev_num_packs <= 1 {
            return Ok(());
        }

        // Generate the deterministic pack index lookup for the file path hashes from `index`,
        // using their now-known compressed sizes.
        let pack_index_lookup = {
            // This is just the `DataPack` / `DataPackWriter` space reservation / data pack index determining logic ripped out.
            // TODO: get rid of `VirtualPack` and `VirtualPacks` and somehow reuse the pack allocation logic.

            struct VirtualPack {
                offset: Offset,
            }

            impl VirtualPack {
                fn new() -> Self {
                    Self { offset: 0 }
                }

                fn remaining(&self, max_pack_size: Offset) -> Offset {
                    let header_size = mem::size_of::<DataPackHeader>() as Offset;

                    debug_assert!(max_pack_size > header_size);
                    let payload_size = max_pack_size - header_size;

                    payload_size.saturating_sub(self.offset)
                }

                fn reserve(&mut self, len: FileSize) {
                    self.offset += len;
                }
            }

            struct VirtualPacks {
                packs: Vec<VirtualPack>,
                max_pack_size: Offset,
            }

            impl VirtualPacks {
                fn new(max_pack_size: Offset) -> Self {
                    Self {
                        packs: Vec::new(),
                        max_pack_size,
                    }
                }

                /// Needs to match [`DataPackWriter::find_or_add_pack`].
                fn reserve(&mut self, len: FileSize) -> PackIndex {
                    let pack_index = self
                        .packs
                        .iter()
                        .position(|pack| pack.remaining(self.max_pack_size) >= len)
                        .unwrap_or_else(|| {
                            let pack_index = self.packs.len();
                            self.packs.push(VirtualPack::new());
                            pack_index
                        });

                    debug_assert!(pack_index < self.packs.len() as _);
                    let pack = unsafe { self.packs.get_unchecked_mut(pack_index as usize) };

                    pack.reserve(len);

                    pack_index as _
                }
            }

            let mut virtual_packs = VirtualPacks::new(max_pack_size);

            index
                .iter()
                .map(|(hash, index_entry)| (*hash, virtual_packs.reserve(index_entry.size)))
                .collect::<HashMap<_, _>>()
        };

        let packs = PacksWrapper::new(&mut self.packs);

        // Reuse the allocation for the files-in-data-pack list for all data packs.
        let mut pack_files_buf =
            unsafe { empty_vec_into_allocation(Vec::<(PathHash, &mut IndexEntry)>::new()) };

        // Compact each data pack file in order, moving source file data as necessary from it to other data pack files
        // and defragmenting the moved-from pack.
        let num_packs = self.packs.len() as PackIndex;
        for pack_index in 0..num_packs {
            pack_files_buf = packs.get(pack_index).compact(
                pack_index,
                packs,
                &pack_index_lookup,
                index,
                pack_files_buf,
            )?;

            // Call the progress callback.
            on_pack_compacted.as_mut().map(|on_pack_compacted| {
                on_pack_compacted(OnPackCompacted {
                    pack_index,
                    num_packs,
                })
            });
        }

        // Remove data pack files which were left empty after compaction.
        self.packs.retain(|pack| {
            if pack.is_empty() {
                std::fs::remove_file(&pack.path).unwrap();
                false
            } else {
                true
            }
        });

        // Call the progress callback.
        after_compaction.map(|after_compaction| {
            after_compaction(AfterCompaction {
                num_packs_before_compaction: prev_num_packs,
                num_packs: self.num_packs(),
            })
        });

        Ok(())
    }

    /// Tries to find a (first) data pack with enough space for `data_size` bytes of source file data,
    /// or creates a new one.
    /// Returns the found/added pack index and the pack itself.
    ///
    /// Needs to match [`VirtualPacks::reserve`].
    fn find_or_add_pack(
        &mut self,
        data_size: FileSize,
        path: &PathBuf,
    ) -> Result<(PackIndex, &mut DataPack), PackError> {
        let pack_index = if let Some(pack_index) = self.packs.iter().position(|pack| {
            self.max_pack_size
                .map(|max_pack_size| pack.remaining(max_pack_size) >= data_size)
                // Use the first pack if no data pack file size limit.
                .unwrap_or(true)
        }) {
            pack_index as PackIndex
        } else {
            let pack_index = self.packs.len();
            self.packs.push(DataPack::new(pack_index as _, path)?);
            pack_index as PackIndex
        };

        debug_assert!(pack_index < self.packs.len() as _);
        let pack = unsafe { self.packs.get_unchecked_mut(pack_index as usize) };

        Ok((pack_index, pack))
    }

    pub(crate) fn num_packs(&self) -> PackIndex {
        self.packs.len() as _
    }
}

/// Wrapper around a `&mut [DataPack]` (pointer + length) to make the borrow checker happy
/// while having the ability to have simultaneous mutable access (unique of course)
/// to multiple elements of the slice
/// (i.e. reading source file data from one pack, defragmenting and truncating it,
/// while writing it to the other pack(s)).
///
/// TODO: does this trigger UB? Technically I never create, let alone access,
/// more than one mutable reference to the same data pack at the same time, but who knows.
#[derive(Clone, Copy)]
struct PacksWrapper(*mut DataPack, PackIndex);

impl PacksWrapper {
    fn new(packs: &mut [DataPack]) -> Self {
        Self(packs.as_mut_ptr(), packs.len() as _)
    }

    fn get(&self, index: PackIndex) -> &mut DataPack {
        debug_assert!(index < self.1);
        unsafe { &mut *self.0.offset(index as _) }
    }
}

pub(crate) struct DataPackReader {
    map: Mmap,
}

impl DataPackReader {
    pub(crate) fn new(
        path: &mut PathBuf,
        index: PackIndex,
        index_checksum: Checksum,
    ) -> Result<Self, PackReaderError> {
        // Open and map the data pack file.
        let map = || -> _ {
            let path = PathPopGuard::push(path, format!("{}{}", PACK_FILE_NAME, index));
            let file = File::open(&path)?;
            unsafe { Mmap::map(&file) }
        }()
        .map_err(|err| PackReaderError::FailedToOpenPackFile((index, err)))?;

        if !|| -> bool {
            // Check if the data is at least large enough to hold the smallest possible data pack blob.
            if (map.len() as u64) < min_data_pack_size() {
                return false;
            }

            let header = unsafe { Self::header(&map) };

            // Check the header magic.
            if !header.check_magic() {
                return false;
            }

            // Check whether the data pack's checksum / "version" matches the index file.
            if header.checksum() != index_checksum {
                return false;
            }

            true
        }() {
            return Err(PackReaderError::InvalidPackFile(index));
        }

        Ok(Self { map })
    }

    /// Returns the data pack's payload (all data past the data pack header).
    /// All byte offsets in the index file are relative to this.
    pub(crate) fn data(&self) -> &[u8] {
        // The caller guarantees the data is at least large enough for a `PackHeader` and a single byte of payload.
        unsafe {
            self.map
                .get_unchecked_dbg(mem::size_of::<DataPackHeader>()..)
        }
    }

    /// Returns a subslice within the data pack's payload `data()` blob at `offset` (in bytes) from the start, with `size` byte elements.
    /// The caller guarantees `offset` and `size` are valid.
    pub(crate) unsafe fn slice(&self, offset: Offset, size: FileSize) -> &[u8] {
        let data = self.data();
        data.get_unchecked_dbg(offset as usize..(offset + size) as usize)
    }

    /// The caller guarantees `data` is at least large enough for a `PackHeader`.
    unsafe fn header(data: &[u8]) -> &DataPackHeader {
        &*(data.as_ptr() as *const _)
    }
}
