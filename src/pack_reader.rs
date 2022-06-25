use {
    crate::*,
    memmap2::*,
    minifilepath::*,
    minifiletree::{Reader as FileTreeReader, ReaderError as FileTreeReaderError},
    minilz4::*,
    miniunchecked::*,
    std::{
        collections::hash_map::{Entry, HashMap},
        fs::{self, File},
        hash::{BuildHasher, Hasher},
        io::Write,
        num::NonZeroU64,
        ops::Deref,
        path::PathBuf,
    },
    tracing::info_span,
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
    /// Allocates `size` bytes and returns a mutable reference to the allocated byte slice
    /// to be filled with the decompressed resource file's data.
    ///
    /// Implementation guarantees the length of the returned byte slice is exactly `size`.
    ///
    /// TODO: alignment.
    /// TODO: out of memory handling.
    fn alloc(&self, size: NonZeroU64) -> &mut [u8];
}

/// Provides an interface to lookup resource file data from the resource pack.
pub struct PackReader {
    index: IndexReader,
    packs: Vec<DataPackReader>,
    path: PathBuf,
}

impl PackReader {
    /// Attempts to open a resource pack for reading.
    ///
    /// `path` - absolute path to the folder which contains the resource pack's files.
    ///
    /// `checksum_hasher` - hasher used to validate the index checksum,
    /// and, if `validate_checksum` is `true`, to validate the individual resource file data checksums.
    /// Must match the `checksum_hasher` used when [`packing`](PackOptions::pack); otherwise an error will be returned
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

        //let (lookup_keys, lookup_values) = index.lookup_keys_and_values();

        let mut packs: HashMap<PackIndex, DataPackReader> = HashMap::new();

        let mut index_checksum = checksum_hasher.build_hasher();

        for (path_hash, index_entry) in index.path_hashes_and_index_entries() {
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
            if (index_entry.offset + index_entry.size) > pack.data().len() as _ {
                return Err(PackReaderError::CorruptData);
            }

            // Make sure the uncompressed length is valid.
            if index_entry.uncompressed_len <= index_entry.size && index_entry.uncompressed_len != 0
            {
                return Err(PackReaderError::CorruptData);
            }

            // If required, validate the file data checksum.
            if validate_checksum {
                let file_data = unsafe { pack.slice(index_entry.offset, index_entry.size) };
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
                index_entry.size,
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

        Ok(Self { index, packs, path })
    }

    pub fn checksum(&self) -> Checksum {
        self.index.checksum()
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

    /// Tries to unpack the pack's contents into the folder at `dst_path`.
    ///
    /// Requires the pack to be written with support for unpacking (see [`PackOptions::write_file_tree`]).
    ///
    /// `num_workers` determines the number of worker threads which will be spawned to decompress and write the source file data.
    /// If `0`, no worker threads are spawned, all processing is done on the current thread.    ///
    /// By default, if `None`, the number of worker threads is determined by [`std::thread::available_parallelism`]
    /// (minus `1` for the current thread, which is also used for compression).
    ///
    /// `memory_limit` determines the attempted best-effort limit on the amount of temporary memory
    /// allocated at any given time for source file decompression purposes if using multiple threads
    /// (i.e. `num_workers` has a non-zero value, or if `num_workers` is `None`
    /// but [`std::thread::available_parallelism`] returned a positive value).    ///
    /// NOTE: the limit will be exceeded if the required amount of memory to decompress a single source file exceeds it in the first place.
    /// NOTE: this value, if too low, may limit processing parallellism, as worker threads cannot
    /// proceed while waiting for temporary memory to become available.
    /// As a rule of thumb, `memory_limit` should be approximately equal to `(num_workers + 1) * average_decompressed_source_file_size`.
    /// By default, if `None`, temporary memory use is not limited.
    /// Ignored if not using worker threads.
    pub fn unpack(
        &self,
        dst_path: PathBuf,
        num_workers: Option<usize>,
        memory_limit: Option<u64>,
    ) -> Result<(), UnpackError> {
        // Open and map the strings file.
        let map = || -> _ {
            let mut path = self.path.clone();
            let path = PathPopGuard::push(&mut path, STRINGS_FILE_NAME);
            let file = File::open(&path)?;
            unsafe { Mmap::map(&file) }
        }()
        .map_err(UnpackError::FailedToOpenStringsFile)?;

        let file_tree =
            FileTreeReader::new(&map, Some(self.checksum())).map_err(|err| match err {
                FileTreeReaderError::InvalidData => UnpackError::InvalidStringsFile,
                FileTreeReaderError::UnexpectedVersion(version) => {
                    UnpackError::UnexpectedChecksum(version)
                }
            })?;

        // Create the root output directory, if necessary.
        fs::create_dir_all(&dst_path)
            .map_err(|err| UnpackError::FailedToCreateOutputFolder((None, err)))?;

        if let Some(num_workers) = calc_num_workers(num_workers) {
            // Allocator we'll use to allocate temp buffers for decompression.
            let allocator = Allocator::new(memory_limit);

            let context_dst_path = dst_path.clone();
            let context_absolute_file_path = dst_path.clone();

            let mut thread_pool = ThreadPool::new(
                num_workers,
                move |_| WorkerContext {
                    file_path_builder: FilePathBuilder::new(),
                    dst_path: context_dst_path,
                    absolute_file_path: context_absolute_file_path,
                },
                |context, task: UnpackTask| -> Option<UnpackResult> {
                    let file_path_builder =
                        std::mem::replace(&mut context.file_path_builder, FilePathBuilder::new());

                    let absolute_file_path =
                        std::mem::replace(&mut context.absolute_file_path, PathBuf::new());

                    let mut reset_paths = |file_path_builder_, mut absolute_file_path_: PathBuf| {
                        let _ =
                            std::mem::replace(&mut context.file_path_builder, file_path_builder_);
                        absolute_file_path_.clear();
                        absolute_file_path_.push(&context.dst_path);
                        let _ =
                            std::mem::replace(&mut context.absolute_file_path, absolute_file_path_);
                    };

                    match self.process_unpack_task(
                        &file_tree,
                        file_path_builder,
                        absolute_file_path,
                        &allocator,
                        task.path_hash,
                        task.index_entry,
                    ) {
                        Ok(res) => match res {
                            Some((file_path_builder_, absolute_file_path_)) => {
                                reset_paths(file_path_builder_, absolute_file_path_);
                                Some(UnpackResult { error: None })
                            }
                            None => None,
                        },
                        Err((error, file_path_builder_, absolute_file_path_)) => {
                            reset_paths(file_path_builder_, absolute_file_path_);
                            Some(UnpackResult { error: Some(error) })
                        }
                    }
                },
                // Wake up the worker threads blocked on an allocator on error/panic and cancel further allocations.
                || {
                    allocator.cancel();
                },
            );

            let mut file_path_builder = FilePathBuilder::new();
            let mut absolute_file_path = dst_path.clone();

            thread_pool.push_tasks(self.index.path_hashes_and_index_entries().map(
                |(path_hash, index_entry)| UnpackTask {
                    path_hash,
                    index_entry,
                },
            ));

            thread_pool.finish();

            while let Some(result_or_task) = thread_pool.pop_result_or_task() {
                match result_or_task {
                    ResultOrTask::Result(result) => {
                        if let Some(err) = result.error {
                            return Err(err);
                        }
                    }
                    ResultOrTask::Task(task) => {
                        match self
                            .process_unpack_task(
                                &file_tree,
                                file_path_builder,
                                absolute_file_path,
                                &allocator,
                                task.path_hash,
                                task.index_entry,
                            )
                            .map_err(|(err, _, _)| err)?
                        {
                            Some((file_path_builder_, mut absolute_file_path_)) => {
                                file_path_builder = file_path_builder_;
                                absolute_file_path_.clear();
                                absolute_file_path_.push(&dst_path);
                                absolute_file_path = absolute_file_path_;
                            }
                            // Must not happen - only ever returns `None` if the main thread `cancel()`'s the allocator, i.e. drops the thread pool,
                            // and we *are* in the main thread.
                            None => unsafe {
                                unreachable_dbg!("failed to allocate on the main thread")
                            },
                        }
                    }
                }
            }
        } else {
            // Scratch buffer (re)used to decompress compressed file data.
            let mut uncompressed_buffer = Vec::new();

            let mut file_path_builder = FilePathBuilder::new();
            let mut absolute_file_path = dst_path.clone();

            for (path_hash, index_entry) in self.index.path_hashes_and_index_entries() {
                let _span = info_span!("unpack task").entered();

                // Lookup the file path.
                let file_path = {
                    let _span = info_span!("lookup file path").entered();

                    file_tree
                        .lookup_into(path_hash, file_path_builder)
                        .map_err(|_| UnpackError::InvalidStringsFile)?
                };

                if let Some(FilePathAndName {
                    file_path,
                    file_name,
                }) = file_path_and_name(&file_path)
                {
                    let _span = info_span!("create output folder").entered();

                    // Build the absolute folder path.
                    absolute_file_path.push(file_path.as_path());

                    // Create the output file directory, if necessary.
                    fs::create_dir_all(&absolute_file_path).map_err(|err| {
                        UnpackError::FailedToCreateOutputFolder((Some(file_path.into()), err))
                    })?;

                    // Build the absolute file path.
                    absolute_file_path.push(file_name.as_str());
                } else {
                    // Build the absolute file path.
                    absolute_file_path.push(file_path.as_path());
                }

                // Create the output file.
                let mut out_file = {
                    let _span = info_span!("create output file").entered();

                    File::create(&absolute_file_path).map_err(|err| {
                        UnpackError::FailedToCreateOutputFile((file_path.clone(), err))
                    })?
                };

                // Lookup the file data.
                let data = {
                    let _span = info_span!("lookup file data").entered();

                    self.lookup_file_data(index_entry)
                };

                // Decompress the file data if necessary.
                let data = if let Some(uncompressed_size) = index_entry.is_compressed() {
                    let _span = info_span!("decompress").entered();

                    decompress_into_vec(data, uncompressed_size, &mut uncompressed_buffer)
                        .expect("failed to decompress");
                    &uncompressed_buffer
                } else {
                    data
                };

                // Write the file data to the output file.
                {
                    let _span = info_span!("write to output file").entered();

                    out_file.write_all(data).map_err(|err| {
                        UnpackError::FailedToWriteOutputFile((file_path.clone(), err))
                    })?;
                }

                {
                    let _span = info_span!("close output file").entered();

                    std::mem::drop(out_file);
                }

                // Reset the relative and absolute paths.
                file_path_builder = file_path.into_builder();
                absolute_file_path.clear();
                absolute_file_path.push(&dst_path);
            }
        }

        Ok(())
    }

    fn process_unpack_task(
        &self,
        file_tree: &FileTreeReader<'_>,
        file_path_builder: FilePathBuilder,
        mut absolute_file_path: PathBuf,
        allocator: &Allocator,
        path_hash: PathHash,
        index_entry: IndexEntry,
    ) -> Result<Option<(FilePathBuilder, PathBuf)>, (UnpackError, FilePathBuilder, PathBuf)> {
        let _span = info_span!("unpack task").entered();

        // Lookup the file path.
        let file_path = {
            let _span = info_span!("lookup file path").entered();

            match file_tree.lookup_into(path_hash, file_path_builder) {
                Ok(file_path) => file_path,
                Err(file_path_builder) => {
                    return Err((
                        UnpackError::InvalidStringsFile,
                        file_path_builder,
                        absolute_file_path,
                    ))
                }
            }
        };

        if let Some(FilePathAndName {
            file_path: file_path_,
            file_name,
        }) = file_path_and_name(&file_path)
        {
            let _span = info_span!("create output folder").entered();

            // Build the absolute folder path.
            absolute_file_path.push(file_path_.as_path());

            // Create the output file directory, if necessary.
            if let Err(err) = fs::create_dir_all(&absolute_file_path) {
                return Err((
                    UnpackError::FailedToCreateOutputFolder((Some(file_path_.into()), err)),
                    file_path.into_builder(),
                    absolute_file_path,
                ));
            }

            // Build the absolute file path.
            absolute_file_path.push(file_name.as_str());
        } else {
            // Build the absolute file path.
            absolute_file_path.push(file_path.as_path());
        }

        // Create the output file.
        let mut out_file = {
            let _span = info_span!("create output file").entered();

            match File::create(&absolute_file_path) {
                Ok(out_file) => out_file,
                Err(err) => {
                    return Err((
                        UnpackError::FailedToCreateOutputFile((file_path.clone(), err)),
                        file_path.into_builder(),
                        absolute_file_path,
                    ));
                }
            }
        };

        // Lookup the file data.
        let data = {
            let _span = info_span!("lookup file data").entered();

            self.lookup_file_data(index_entry)
        };

        if let Some(uncompressed_size) = index_entry.is_compressed() {
            let mut allocation = match allocator.allocate(uncompressed_size.get()) {
                Some(allocation) => allocation,
                None => return Ok(None),
            };

            // Decompress the file data.
            {
                let _span = info_span!("decompress").entered();

                decompress(data, &mut allocation).expect("failed to decompress");
            }

            // Write the file data to the output file.
            {
                let _span = info_span!("write to output file").entered();

                if let Err(err) = out_file.write_all(&allocation) {
                    return Err((
                        UnpackError::FailedToWriteOutputFile((file_path.clone(), err)),
                        file_path.into_builder(),
                        absolute_file_path,
                    ));
                }
            }
        } else {
            {
                let _span = info_span!("write to output file").entered();

                // Write the file data to the output file.
                if let Err(err) = out_file.write_all(data) {
                    return Err((
                        UnpackError::FailedToWriteOutputFile((file_path.clone(), err)),
                        file_path.into_builder(),
                        absolute_file_path,
                    ));
                }
            }
        }

        {
            let _span = info_span!("close output file").entered();

            std::mem::drop(out_file);
        }

        Ok(Some((file_path.into_builder(), absolute_file_path)))
    }

    fn lookup_file_data(&self, entry: IndexEntry) -> &[u8] {
        let pack = unsafe { self.packs.get_unchecked_dbg(entry.pack_index as usize) };
        unsafe { pack.slice(entry.offset, entry.size) }
    }

    fn lookup_impl<O, F: FnOnce(&[u8], NonZeroU64) -> O>(
        &self,
        path_hash: PathHash,
        f: F,
    ) -> Option<LookupResult<'_, O>> {
        if let Some(entry) = self.index.lookup(path_hash) {
            let data = self.lookup_file_data(entry);
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

struct WorkerContext {
    file_path_builder: FilePathBuilder,
    dst_path: PathBuf,
    absolute_file_path: PathBuf,
}

unsafe impl Send for WorkerContext {}

struct UnpackTask {
    path_hash: PathHash,
    index_entry: IndexEntry,
}

struct UnpackResult {
    error: Option<UnpackError>,
}
