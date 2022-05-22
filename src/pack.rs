use {
    crate::*,
    memmap::*,
    minifilepath::*,
    minifiletree::*,
    minilz4::*,
    std::{
        fs::{self, File},
        hash::{BuildHasher, Hasher},
        mem,
        num::{NonZeroU64, NonZeroUsize},
        path::PathBuf,
        thread,
    },
};

/// Called once for each source file to determine whether it should be compressed.
///
/// The callback is passed the file's relative path in `src_dir` and its data.
/// If the callback returns `true`, the file data will be compressed (except if the compressed data is (slightly) larger in size than uncompressed,
/// which is possible due to compression format overhead when the data is incompressible).
/// If the callback returns `false`, source file data is not compressed.
pub type CompressionCallback = dyn FnMut(&FilePath, &[u8]) -> bool;

/// Argument passed to [`OnSourceFilesGatheredCallback`].
pub struct OnSourceFilesGathered {
    /// Total number of (non-empty) files found in the source directory.
    pub num_files: usize,
    /// Total size in bytes of all (non-empty) files found in the source directory.
    pub total_file_size: u64,
}

/// Callback called once after the source directory was scanned and all (non-empty) source files gathered.
///
/// Returns `true` if processing should continue, or `false` if it should be cancelled
/// (e.g. to enforce a limit on total source directory size to process to prevent patological cases / accidents).
///
/// [`PackError::Cancelled`] is returned if this callback returns `false`.
pub type OnSourceFilesGatheredCallback<'a> = dyn FnOnce(OnSourceFilesGathered) -> bool + 'a;

/// Argument passed to [`OnSourceFileProcessedCallback`].
pub struct OnSourceFileProcessed {
    /// Original size in bytes of the processed source file's data.
    /// These add up to [`OnSourceFilesGathered::total_file_size`].
    pub src_file_size: u64,
    /// Equal to `src_file_size` if the source file was not compressed;
    /// strictly less than `src_file_size` if the source file was compressed.
    pub packed_file_size: u64,
    /// Current total number of created data pack files.
    pub num_packs: PackIndex,
}

/// Called once for each processed source file
/// (so [`OnSourceFilesGathered::num_files`] times total, unless an error occurs).
pub type OnSourceFileProcessedCallback<'a> = dyn FnMut(OnSourceFileProcessed) + 'a;

/// Argument passed to [`BeforePackingCallback`].
pub struct BeforePacking {
    /// The number of worker threads which will be used for packing.
    pub num_workers: usize,
}

/// Called once immediately before beginning packing / compressing of all gathered source data files.
pub type BeforePackingCallback<'a> = dyn FnOnce(BeforePacking) + 'a;

/// Argument passed to [`BeforeCompactionCallback`].
pub struct BeforeCompaction {
    /// Current total number of created data pack files before their compaction.
    pub num_packs: PackIndex,
}

/// If using at least one worker thread and a data pack file limit,
/// is called once after all gathered source data files were packed / compressed,
/// before compacting the created data pack files.
pub type BeforeCompactionCallback<'a> = dyn FnOnce(BeforeCompaction) + 'a;

/// Argument passed to [`OnPackCompactedCallback`].
pub struct OnPackCompacted {
    /// Index of the compacted pack, in range `[0 .. BeforeCompaction::num_packs]`.
    pub pack_index: PackIndex,
    /// Current total number of created data pack files before their compaction.
    pub num_packs: PackIndex,
}

/// If using at least one worker thread and a data pack file limit,
/// and if there's more than one data pack file,
/// is called once after each data pack file was compacted
/// (so [`BeforeCompaction::num_packs`], or [`OnPackCompacted::num_packs`] times total, unless an error occurs).
pub type OnPackCompactedCallback<'a> = dyn FnMut(OnPackCompacted) + 'a;

/// Argument passed to [`AfterCompactionCallback`].
pub struct AfterCompaction {
    /// Total number of created data pack files before their compaction (equal to [`BeforeCompaction::num_packs`]).
    pub num_packs_before_compaction: PackIndex,
    /// Current total number of created data pack files after their compaction.
    pub num_packs: PackIndex,
}

/// If using at least one worker thread and a data pack file limit,
/// and if there's more than one data pack file,
/// is called after all data pack files were compacted.
pub type AfterCompactionCallback<'a> = dyn FnOnce(AfterCompaction) + 'a;

#[derive(Default)]
struct ProgressCallbacks<'a> {
    on_source_files_gathered: Option<Box<OnSourceFilesGatheredCallback<'a>>>,
    before_packing: Option<Box<BeforePackingCallback<'a>>>,
    on_source_file_processed: Option<Box<OnSourceFileProcessedCallback<'a>>>,
    before_compaction: Option<Box<BeforeCompactionCallback<'a>>>,
    on_pack_compacted: Option<Box<OnPackCompactedCallback<'a>>>,
    after_compaction: Option<Box<AfterCompactionCallback<'a>>>,
}

/// Used to configure the packing options and perform packing (see [`PackOptions::pack`]).
pub struct PackOptions<'a, F, H>
where
    F: BuildHasher,
    F::Hasher: Clone,
    H: BuildHasher,
{
    src_dir: PathBuf,
    dst_dir: PathBuf,
    filepath_hasher: F,
    checksum_hasher: H,
    max_pack_size: Option<u64>,
    compression_callback: Option<Box<CompressionCallback>>,
    num_workers: Option<usize>,
    memory_limit: Option<u64>,
    write_file_tree: bool,
    progress_callbacks: ProgressCallbacks<'a>,
}

impl<'a, F, H> PackOptions<'a, F, H>
where
    F: BuildHasher,
    F::Hasher: Clone,
    H: BuildHasher,
{
    /// Create the [`PackOptions`] with non-optional parameters necessary for packing.
    ///
    /// `src_dir` is the path to the directory containing the source files to pack.
    ///
    /// `dst_dir` is the path to the directory (created if necessary) which will contain the resulting pack files.
    ///
    /// `filepath_hasher` is used to create [`PathHash`]'es from the source file relative paths whithin `src_dir`
    /// to create the lookup data structure.
    /// This same hasher must then be used when looking up the source file data in the created pack.
    ///
    /// `checksum_hasher` is used to calculate source file checksums and the pack's checksum / "version", returned by the call to [`PackOptions::pack`].
    /// This same hasher must then be used when opening the pack for reading.
    pub fn new(src_dir: PathBuf, dst_dir: PathBuf, filepath_hasher: F, checksum_hasher: H) -> Self {
        Self {
            src_dir,
            dst_dir,
            filepath_hasher,
            checksum_hasher,
            max_pack_size: None,
            compression_callback: None,
            write_file_tree: false,
            num_workers: None,
            memory_limit: None,
            progress_callbacks: Default::default(),
        }
    }

    /// The source file data is split into data pack files no larger then `max_pack_size` bytes
    /// (unless an individual source file is larger than this value, in which case the data pack file
    /// will be as large as the source file (and only contain this source file)).
    ///
    /// By default, if this is not called, all source file data will be put in a single data pack file.
    pub fn max_pack_size(mut self, max_pack_size: u64) -> Self {
        self.max_pack_size.replace(max_pack_size);
        self
    }

    /// See [`CompressionCallback`].
    pub fn compression_callback(mut self, compression_callback: Box<CompressionCallback>) -> Self {
        self.compression_callback.replace(compression_callback);
        self
    }

    /// `num_workers` determines the number of worker threads which will be spawned to compress source file data, if necessary.
    /// If `0`, no worker threads are spawned, all processing is done on the current thread.
    ///
    /// By default, if this is not called, the number of worker threads is determined by [`std::thread::available_parallelism`]
    /// (minus `1` for the current thread, which is also used for compression).
    ///
    /// Ignored if [`PackOptions::compression_callback`] is not specified.
    pub fn num_workers(mut self, num_workers: usize) -> Self {
        self.num_workers.replace(num_workers);
        self
    }

    /// `memory_limit` determines the attempted best-effort limit on the amount of temporary memory
    /// allocated at any given time for source file compression purposes
    /// if using multiple threads for compression
    /// (i.e. [`PackOptions::num_workers`] was called with a non-zero value, or if [`PackOptions::num_workers`] was not called
    /// but [`std::thread::available_parallelism`] returned a positive value).
    ///
    /// NOTE: the limit will be exceeded if the required amount of memory to compress a single source file exceeds it in the first place.
    /// NOTE: this value, if too low, may limit compression parallellism, as worker threads cannot
    /// proceed with compression while waiting for temporary memory to become available.
    /// As a rule of thumb, `memory_limit` should be approximately equal to `(num_workers + 1) * average_source_file_size`.
    ///
    /// By default, if this is not called, temporary memory use is not limited.
    ///
    /// Ignored if not using worker threads for compression.
    /// Ignored if [`PackOptions::compression_callback`] is not specified.
    pub fn memory_limit(mut self, memory_limit: u64) -> Self {
        self.memory_limit.replace(memory_limit);
        self
    }

    /// If `write_file_tree` is `true`, data necessary to unpack the file pack is serialized along with it.
    ///
    /// By default, if this is not called, the file tree is not written.
    pub fn write_file_tree(mut self, write_file_tree: bool) -> Self {
        self.write_file_tree = write_file_tree;
        self
    }

    /// See [`OnSourceFilesGatheredCallback`].
    pub fn on_source_files_gathered(
        mut self,
        on_source_files_gathered: Box<OnSourceFilesGatheredCallback<'a>>,
    ) -> Self {
        self.progress_callbacks
            .on_source_files_gathered
            .replace(on_source_files_gathered);
        self
    }

    /// See [`BeforePackingCallback`].
    pub fn before_packing(mut self, before_packing: Box<BeforePackingCallback<'a>>) -> Self {
        self.progress_callbacks
            .before_packing
            .replace(before_packing);
        self
    }

    /// See [`OnSourceFileProcessedCallback`].
    pub fn on_source_file_processed(
        mut self,
        on_source_file_processed: Box<OnSourceFileProcessedCallback<'a>>,
    ) -> Self {
        self.progress_callbacks
            .on_source_file_processed
            .replace(on_source_file_processed);
        self
    }

    /// See [`BeforeCompactionCallback`].
    pub fn before_compaction(
        mut self,
        before_compaction: Box<BeforeCompactionCallback<'a>>,
    ) -> Self {
        self.progress_callbacks
            .before_compaction
            .replace(before_compaction);
        self
    }

    /// See [`OnPackCompactedCallback`].
    pub fn on_pack_compacted(
        mut self,
        on_pack_compacted: Box<OnPackCompactedCallback<'a>>,
    ) -> Self {
        self.progress_callbacks
            .on_pack_compacted
            .replace(on_pack_compacted);
        self
    }

    /// See [`AfterCompactionCallback`].
    pub fn after_compaction(mut self, after_compaction: Box<AfterCompactionCallback<'a>>) -> Self {
        self.progress_callbacks
            .after_compaction
            .replace(after_compaction);
        self
    }

    /// Consumes the [`PackOptions`] and begins packing.
    ///
    /// Iterates recursively over all (non-empty) source files in `src_dir`,
    /// (optionally) compresses the source file data and packs it into the data pack files in `dst_dir`.
    ///
    /// Returns the [`Checksum`] of the generated pack.
    pub fn pack(self) -> Result<Checksum, PackError> {
        pack(
            self.src_dir,
            self.dst_dir,
            self.max_pack_size,
            self.filepath_hasher,
            self.checksum_hasher,
            self.compression_callback,
            self.num_workers,
            self.memory_limit,
            self.write_file_tree,
            self.progress_callbacks,
        )
    }
}

fn pack<F, H>(
    mut src_dir: PathBuf,
    mut dst_dir: PathBuf,
    max_pack_size: Option<u64>,
    filepath_hasher: F,
    checksum_hasher: H,
    compression_callback: Option<Box<CompressionCallback>>,
    num_workers: Option<usize>,
    memory_limit: Option<u64>,
    write_file_tree: bool,
    progress_callbacks: ProgressCallbacks,
) -> Result<Checksum, PackError>
where
    F: BuildHasher,
    F::Hasher: Clone,
    H: BuildHasher,
{
    use FileTreeWriterError::*;

    // Gather all the non-empty files in the source directory and save their path hashes / string paths.
    let (mut path_hashes, file_tree, total_file_size) = {
        let mut path_hashes = Vec::new();
        let mut file_tree = FileTreeWriter::new(filepath_hasher);
        let mut total_file_size = 0;

        iterate_files_in_dir(&mut src_dir, &mut |relative_path, file_size| {
            path_hashes.push(file_tree.insert(relative_path).map_err(|err| match err {
                FolderAlreadyExistsAtFilePath => {
                    PackError::FolderAlreadyExistsAtFilePath(relative_path.into())
                }
                FileAlreadyExistsAtFolderPath(path) => {
                    PackError::FileAlreadyExistsAtFolderPath(path)
                }
                PathHashCollision(path) => {
                    PackError::PathHashCollision((relative_path.into(), path))
                }
                PathAlreadyExists => {
                    panic!("somehow processed the same file path twice")
                }
            })?);

            total_file_size += file_size;

            Ok(())
        })?;

        (path_hashes, file_tree, total_file_size)
    };

    if !progress_callbacks
        .on_source_files_gathered
        .map(|on_source_files_gathered| {
            on_source_files_gathered(OnSourceFilesGathered {
                num_files: path_hashes.len(),
                total_file_size,
            })
        })
        .unwrap_or(true)
    {
        return Err(PackError::Cancelled);
    }

    // Create the output directory, if necessary.
    fs::create_dir_all(&dst_dir).map_err(|err| PackError::FailedToCreateOutputDirectory(err))?;

    let index = IndexWriter::new(checksum_hasher.build_hasher(), &mut dst_dir)?;
    let packs = DataPackWriter::new(max_pack_size);

    // Sort the path hashes by file path string before processing to determine file data location in the data pack files.
    //
    // NOTE: this is arbitrary.
    // Advantages: file data locality - i.e. files in the same folder will be usually located contiguously in the same data pack file
    // (but only when combined with the current greedy first-fit logic of data pack file selection -
    // so will not be the case if/when using custom data pack index selection logic).
    // Disadvantages (w.r.t. file path hash sorting): sorting performance; cannot use binary search in index lookups.
    //
    // NOTE: nothing prevents us from using a different file path hash sorting when writing to the pack index,
    // e.g. sort by file path hash to allow binary searching.
    {
        let mut left_path = FilePathBuilder::new();
        let mut right_path = FilePathBuilder::new();

        path_hashes.sort_by(|l, r| {
            let lp = lookup_path(
                &file_tree,
                *l,
                mem::replace(&mut left_path, Default::default()),
            );
            let rp = lookup_path(
                &file_tree,
                *r,
                mem::replace(&mut right_path, Default::default()),
            );

            let res = lp.as_str().cmp(rp.as_str());

            let _ = mem::replace(&mut left_path, lp.into_builder());
            let _ = mem::replace(&mut right_path, rp.into_builder());

            res
        });
    }

    // Calculate the number of worker threads we'll use,
    // and do single- or multi-threaded processing based on the result.
    //
    // Don't need worker threads if we won't be using compression.
    let num_workers = compression_callback.as_ref().and_then(|_| {
        match num_workers {
            Some(num_workers) => NonZeroUsize::new(num_workers),
            // Subtract `1` for the main thread.
            None => thread::available_parallelism()
                .ok()
                .and_then(|num_workers| NonZeroUsize::new(num_workers.get() - 1)),
        }
    });

    let (mut index_, mut packs) = if let Some(num_workers) = num_workers {
        progress_callbacks.before_packing.map(|before_packing| {
            before_packing(BeforePacking {
                num_workers: num_workers.get(),
            })
        });

        pack_multithreaded(
            src_dir,
            &dst_dir,
            packs,
            &file_tree,
            &checksum_hasher,
            unsafe {
                debug_unwrap_option(
                    compression_callback,
                    "multithreaded packing requires a compression callback",
                )
            },
            num_workers,
            memory_limit,
            path_hashes,
            progress_callbacks.on_source_file_processed,
            progress_callbacks.before_compaction,
            progress_callbacks.on_pack_compacted,
            progress_callbacks.after_compaction,
        )
    } else {
        progress_callbacks
            .before_packing
            .map(|before_packing| before_packing(BeforePacking { num_workers: 0 }));

        pack_singlethreaded(
            src_dir,
            &dst_dir,
            packs,
            &file_tree,
            &checksum_hasher,
            compression_callback,
            path_hashes,
            progress_callbacks.on_source_file_processed,
        )
    }?;

    // Re-sort the index entries in file path hash order.
    // NOTE: sorting order is relied on by the pack reader for binary search.
    index_.sort_by(|l, r| l.0.cmp(&r.0));

    let total_checksum = index.write(|| index_.iter().cloned())?;

    // Write all the pack headers.
    packs.write_headers(total_checksum)?;

    // Write the file tree, if required.
    if write_file_tree {
        let file_tree_path = PathPopGuard::push(&mut dst_dir, STRINGS_FILE_NAME);

        || -> _ {
            let mut file_tree_file = File::create(&file_tree_path)?;

            file_tree
                .write(total_checksum, &mut file_tree_file)
                .map(|_| ())
        }()
        .map_err(PackError::FailedToWriteStringsFile)?;
    }

    Ok(total_checksum)
}

fn pack_singlethreaded<F, H>(
    src_dir: PathBuf,
    dst_dir: &PathBuf,
    mut packs: DataPackWriter,
    file_tree: &FileTreeWriter<F>,
    checksum_hasher: &H,
    mut compression_callback: Option<Box<CompressionCallback>>,
    path_hashes: Vec<PathHash>,
    mut on_source_file_processed: Option<Box<OnSourceFileProcessedCallback>>,
) -> Result<(Vec<(PathHash, IndexEntry)>, DataPackWriter), PackError>
where
    F: BuildHasher,
    F::Hasher: Clone,
    H: BuildHasher,
{
    let mut file_path_builder = FilePathBuilder::new();
    let mut absolute_file_path = src_dir.clone();

    let mut compressor = Compressor::new().expect("failed to create an LZ4 compressor");
    // Scratch buffer used for compression.
    let mut compressed_data = Vec::new();

    let mut hashes_and_index_entries = Vec::new();

    for path_hash in path_hashes {
        // Lookup the relative file path. Must succeed - all path hashes have been inserted.
        let file_path = lookup_path(&file_tree, path_hash, file_path_builder);

        // Build the absolute file path.
        absolute_file_path.push(file_path.as_path());

        // Open and map the source file.
        let src_file = || -> _ { unsafe { Mmap::map(&File::open(&absolute_file_path)?) } }()
            .map_err(|err| PackError::FailedToOpenSourceFile((file_path.clone(), err)))?;

        // Call the compression callback, passing the relative file path and the file data.
        let compress = compression_callback
            .as_mut()
            .map(|compression_callback| compression_callback(&file_path, &src_file))
            .unwrap_or(false);

        // Try to compress the file data if the compression callback specified it.
        let (compressed, file_data) = if compress {
            compressor
                .compress_into_vec(&src_file, &mut compressed_data, false)
                .map_err(|_| PackError::FailedToCompress(file_path.clone()))?;

            // Do not compress if the source file data is incompressible.
            if compressed_data.len() >= src_file.len() {
                (false, src_file.as_ref())
            } else {
                (true, compressed_data.as_ref())
            }
        } else {
            (false, src_file.as_ref())
        };

        // Find the data pack file for this source file.
        // Write the source file to the data pack file.
        // TODO: user-determined splitting into data packs.
        let (pack_index, offset) = packs.write(file_data, dst_dir)?;

        // Calculate the checksum of the source file's data.
        let src_file_checksum = hash_file(file_data, checksum_hasher);

        // Add the file to the index.
        hashes_and_index_entries.push((
            path_hash,
            if compressed {
                IndexEntry::new_compressed(
                    pack_index,
                    offset,
                    file_data.len() as _,
                    src_file_checksum,
                    src_file.len() as _,
                )
            } else {
                IndexEntry::new_uncompressed(
                    pack_index,
                    offset,
                    file_data.len() as _,
                    src_file_checksum,
                )
            },
        ));

        // Call the progress callback.
        on_source_file_processed
            .as_mut()
            .map(|on_source_file_processed| {
                on_source_file_processed(OnSourceFileProcessed {
                    src_file_size: src_file.len() as _,
                    packed_file_size: file_data.len() as _,
                    num_packs: packs.num_packs(),
                })
            });

        // Reset the relative and absolute paths.
        file_path_builder = file_path.into_builder();
        absolute_file_path.clear();
        absolute_file_path.push(&src_dir);
    }

    Ok((hashes_and_index_entries, packs))
}

fn pack_multithreaded<F, H>(
    src_dir: PathBuf,
    dst_dir: &PathBuf,
    mut packs: DataPackWriter,
    file_tree: &FileTreeWriter<F>,
    checksum_hasher: &H,
    mut compression_callback: Box<CompressionCallback>,
    num_workers: NonZeroUsize,
    memory_limit: Option<u64>,
    path_hashes: Vec<PathHash>,
    mut on_source_file_processed: Option<Box<OnSourceFileProcessedCallback>>,
    before_compaction: Option<Box<BeforeCompactionCallback>>,
    on_pack_compacted: Option<Box<OnPackCompactedCallback>>,
    after_compaction: Option<Box<AfterCompactionCallback>>,
) -> Result<(Vec<(PathHash, IndexEntry)>, DataPackWriter), PackError>
where
    F: BuildHasher,
    F::Hasher: Clone,
    H: BuildHasher,
{
    let mut file_path_builder = FilePathBuilder::new();
    let mut absolute_file_path = src_dir.clone();

    let mut index = vec![
        (
            0,
            IndexEntry {
                pack_index: 0,
                offset: 0,
                len: 0,
                checksum: 0,
                uncompressed_len: 0
            }
        );
        path_hashes.len()
    ];

    // Files which require compression and which are larger than half the memory limit, if any,
    // cannot be processed in parallel anyway, so handle them separately on the main thread.
    let mut files_too_large_to_compress_in_parallel = Vec::new();

    // Each worker thread gets its own compressor.
    struct WorkerContext(Compressor);

    unsafe impl Send for WorkerContext {}

    // Allocator we'll use to allocate temp buffers for compression.
    let allocator = Allocator::new(memory_limit);

    // Allocates a buffer large enough to hold compressed data for a `len` bytes sized source file.
    let allocate = |len: FileSize, block: bool| -> Option<Allocation<'_>> {
        let bound = Compressor::compressed_size_bound(unsafe { NonZeroU64::new_unchecked(len) });

        if block {
            allocator.allocate(bound.get())
        } else {
            allocator.try_allocate(bound.get())
        }
    };

    // Create the compression thread pool.
    let mut thread_pool = ThreadPool::new(
        num_workers,
        |_| -> WorkerContext {
            WorkerContext(Compressor::new().expect("failed to create an LZ4 compressor"))
        },
        |context, task: CompressionTask| -> Option<CompressionResult> {
            // Allocator only ever returns `None` when we cancelled further processing on error/panic.
            // Return `None` to exit the worker loop.
            let mut allocation = allocate(task.src_file.len() as _, /* block */ true)?;

            let compressed = match context.0.compress(&task.src_file, &mut allocation) {
                Ok(compressed_size) => {
                    // Do not compress if the source file data is incompressible.
                    if compressed_size.get() >= task.src_file.len() as FileSize {
                        Ok(None)
                    } else {
                        Ok(Some(CompressedBuffer {
                            allocation,
                            compressed_size,
                        }))
                    }
                }
                Err(err) => Err(err),
            };

            Some(CompressionResult { task, compressed })
        },
        // Wake up the worker threads blocked on an allocator on error/panic and cancel further allocations.
        || {
            allocator.cancel();
        },
    );

    for (insert_index, &path_hash) in path_hashes.iter().enumerate() {
        // Lookup the relative file path. Must succeed - all path hashes have been inserted.
        let file_path = lookup_path(&file_tree, path_hash, file_path_builder);

        // Build the absolute file path.
        absolute_file_path.push(file_path.as_path());

        // Open and map the source file.
        let src_file = || -> _ { unsafe { Mmap::map(&File::open(&absolute_file_path)?) } }()
            .map_err(|err| PackError::FailedToOpenSourceFile((file_path.clone(), err)))?;

        // Call the compression callback, passing the relative file path and the file data.
        if compression_callback(&file_path, &src_file) {
            let uncompressed_size = src_file.len() as FileSize;
            let task = CompressionTask {
                path_hash,
                src_file,
                pack_index: packs.reserve(uncompressed_size, dst_dir)?,
                insert_index,
            };

            // If the file size is half of memory limit (if any) or larger, it cannot be compressed in parallel,
            // because the second (and other) thread(s) will always be blocked on the allocator until the first one finishes processing.
            if memory_limit
                .map(|memory_limit| uncompressed_size >= (memory_limit / 2))
                .unwrap_or(false)
            {
                files_too_large_to_compress_in_parallel.push(task);
            // Otherwise spawn a compression task.
            } else {
                thread_pool.push(task);
            }
        } else {
            process_uncompressed_file(
                dst_dir,
                checksum_hasher,
                &mut packs,
                &mut index,
                insert_index,
                path_hash,
                src_file,
                None,
                on_source_file_processed.as_mut(),
            )?;
        }

        // Reset the relative and absolute paths.
        file_path_builder = file_path.into_builder();
        absolute_file_path.clear();
        absolute_file_path.push(&src_dir);

        // If we use a memory limit, try to process a single compression result each iteration
        // to relieve allocator pressure.
        if memory_limit.is_some() {
            if let Some(result) = thread_pool.try_pop_result() {
                process_result(
                    dst_dir,
                    &file_tree,
                    checksum_hasher,
                    &mut packs,
                    &mut index,
                    result,
                    &file_path_builder,
                    on_source_file_processed.as_mut(),
                )?;
            }
        }
    }

    // Signal the thread pool there will be no more tasks.
    thread_pool.finish();

    // Create the compressor the main thread will use.
    let mut compressor = Compressor::new().expect("failed to create an LZ4 compressor");

    // Process the results and help out with tasks if there are no results ready.
    while let Some(result_or_task) = thread_pool.pop_result_or_task() {
        match result_or_task {
            ResultOrTask::Result(result) => {
                process_result(
                    dst_dir,
                    &file_tree,
                    checksum_hasher,
                    &mut packs,
                    &mut index,
                    result,
                    &file_path_builder,
                    on_source_file_processed.as_mut(),
                )?;
            }
            ResultOrTask::Task(task) => {
                // Try to allocate memory for the compression task.
                // Do not block, return the task back to the queue on failure.
                // NOTE - task queue is FIFO, so we won't just pick up the same task again on the next iteration of this loop.
                if let Some(allocation) = allocate(task.src_file.len() as _, /* block */ false) {
                    compress_file(
                        checksum_hasher,
                        &mut packs,
                        &mut index,
                        task,
                        &file_tree,
                        &file_path_builder,
                        allocation,
                        &mut compressor,
                        on_source_file_processed.as_mut(),
                    )?;
                } else {
                    thread_pool.push(task);
                }
            }
        }
    }

    // Wait for all worker threads to exit.
    mem::drop(thread_pool);

    // Process the large files.
    for task in files_too_large_to_compress_in_parallel {
        // Must succeed - only the main thread is allocating and all files are above the memory limit in size.
        let allocation = unsafe {
            debug_unwrap_option(
                allocate(task.src_file.len() as _, /* block */ false),
                "allocation from the main thread must succeed",
            )
        };

        compress_file(
            checksum_hasher,
            &mut packs,
            &mut index,
            task,
            &file_tree,
            &file_path_builder,
            allocation,
            &mut compressor,
            on_source_file_processed.as_mut(),
        )?;
    }

    // Compact the data pack files if necessary.
    packs.compact(
        &mut index,
        before_compaction,
        on_pack_compacted,
        after_compaction,
    )?;

    Ok((index, packs))
}

/// Handles files which do not require compression (in which case `pack_index` is `None`),
/// or which where incomressible (in which case `pack_index` is `Some`, as returned by `packs.reserve()`).
fn process_uncompressed_file<H>(
    dst_dir: &PathBuf,
    checksum_hasher: &H,
    packs: &mut DataPackWriter,
    index: &mut Vec<(PathHash, IndexEntry)>,
    insert_index: usize,
    path_hash: PathHash,
    src_file: Mmap,
    pack_index: Option<PackIndex>,
    on_source_file_processed: Option<&mut Box<OnSourceFileProcessedCallback>>,
) -> Result<(), PackError>
where
    H: BuildHasher,
{
    // If we have reserved space in a pack, write to it.
    let (pack_index, offset) = if let Some(pack_index) = pack_index {
        (pack_index, packs.write_reserved(&src_file, pack_index)?)
    // Otherwise find a pack and write to it.
    // TODO: user-determined splitting into packs.
    } else {
        packs.write(&src_file, dst_dir)?
    };

    debug_assert!(insert_index < index.len());
    let index_entry = unsafe { index.get_unchecked_mut(insert_index) };

    debug_assert_eq!(index_entry.0, 0);
    debug_assert_eq!(index_entry.1.len, 0);
    *index_entry = (
        path_hash,
        IndexEntry::new_uncompressed(
            pack_index,
            offset,
            src_file.len() as _,
            // Calculate the checksum of the source file's data.
            hash_file(&src_file, checksum_hasher),
        ),
    );

    // Call the progress callback.
    on_source_file_processed.map(|on_source_file_processed| {
        on_source_file_processed(OnSourceFileProcessed {
            src_file_size: src_file.len() as _,
            packed_file_size: src_file.len() as _,
            num_packs: packs.num_packs(),
        })
    });

    Ok(())
}

/// A compression task sent to the thread pool from the main thread.
struct CompressionTask {
    /// Mapped source file data.
    src_file: Mmap,
    /// Reserved data pack file index for this source file.
    pack_index: PackIndex,
    /// Needed for error reporting (to lookup the file name).
    path_hash: PathHash,
    /// Needed to avoid re-sorting the (path hash, index entry) result index array in file path alphabetic order
    /// after non-deterministic multithreaded compression.
    /// Index must be sorted in alphabetic order, same as for singlethreaded processing,
    /// for deterministic results.
    insert_index: usize,
}

/// Compressed source file data buffer allocated by the worker thread.
struct CompressedBuffer<'a> {
    /// The entire allocation for this buffer.
    allocation: Allocation<'a>,
    /// Actual compressed source file data size in bytes within the buffer.
    compressed_size: NonZeroU64,
}

impl<'a> CompressedBuffer<'a> {
    /// Returns the subslice of the allocation with the compressed source file data.
    fn compressed(&self) -> &[u8] {
        debug_assert!(self.compressed_size.get() <= self.allocation.len() as FileSize);
        unsafe {
            self.allocation
                .get_unchecked(0..self.compressed_size.get() as usize)
        }
    }
}

struct CompressionResult<'a> {
    task: CompressionTask,
    /// `Ok(Some)` if succesfully compressed.
    /// `Ok(None)` if source file data was incompressible.
    /// `Err` if failed to compress.
    compressed: Result<Option<CompressedBuffer<'a>>, CompressorError>,
}

fn lookup_path<H>(
    file_tree: &FileTreeWriter<H>,
    path_hash: PathHash,
    builder: FilePathBuilder,
) -> FilePathBuf
where
    H: BuildHasher,
    H::Hasher: Clone,
{
    unsafe {
        debug_unwrap_result(
            file_tree.lookup_into(path_hash, builder),
            "failed to look up an inserted path",
        )
    }
}

fn compress_file<H, F>(
    checksum_hasher: &H,
    packs: &mut DataPackWriter,
    index: &mut Vec<(PathHash, IndexEntry)>,
    task: CompressionTask,
    file_tree: &FileTreeWriter<F>,
    file_path_builder: &FilePathBuilder,
    mut allocation: Allocation<'_>,
    compressor: &mut Compressor,
    on_source_file_processed: Option<&mut Box<OnSourceFileProcessedCallback>>,
) -> Result<(), PackError>
where
    H: BuildHasher,
    F: BuildHasher,
    F::Hasher: Clone,
{
    // Try to compress the file data.
    let compressed_size = compressor
        .compress(&task.src_file, &mut allocation)
        .map_err(|_| {
            PackError::FailedToCompress(lookup_path(
                file_tree,
                task.path_hash,
                file_path_builder.clone(),
            ))
        })?;

    // Do not compress if the source file data is incompressible.
    let (compressed, file_data) = if compressed_size.get() >= task.src_file.len() as _ {
        (false, task.src_file.as_ref())
    } else {
        debug_assert!(compressed_size.get() <= allocation.len() as _);
        (true, unsafe {
            allocation
                .as_ref()
                .get_unchecked(0..compressed_size.get() as usize)
        })
    };

    // Write the source file data, wether compressed or not, to the reserved data pack file.
    let offset = packs.write_reserved(file_data, task.pack_index)?;

    // Calculate the checksum of the source file's data.
    let checksum = hash_file(file_data, checksum_hasher);

    // Add the file to the index.
    debug_assert!(task.insert_index < index.len());
    let index_entry = unsafe { index.get_unchecked_mut(task.insert_index) };

    debug_assert_eq!(index_entry.0, 0);
    debug_assert_eq!(index_entry.1.len, 0);
    *index_entry = (
        task.path_hash,
        if compressed {
            IndexEntry::new_compressed(
                task.pack_index,
                offset,
                file_data.len() as _,
                checksum,
                task.src_file.len() as _,
            )
        } else {
            IndexEntry::new_uncompressed(task.pack_index, offset, file_data.len() as _, checksum)
        },
    );

    // Call the progress callback.
    on_source_file_processed.map(|on_source_file_processed| {
        on_source_file_processed(OnSourceFileProcessed {
            src_file_size: task.src_file.len() as _,
            packed_file_size: file_data.len() as _,
            num_packs: packs.num_packs(),
        })
    });

    Ok(())
}

fn process_result<F, H>(
    dst_dir: &PathBuf,
    file_tree: &FileTreeWriter<F>,
    checksum_hasher: &H,
    packs: &mut DataPackWriter,
    index: &mut Vec<(PathHash, IndexEntry)>,
    result: CompressionResult,
    file_path_builder: &FilePathBuilder,
    on_source_file_processed: Option<&mut Box<OnSourceFileProcessedCallback>>,
) -> Result<(), PackError>
where
    F: BuildHasher,
    F::Hasher: Clone,
    H: BuildHasher,
{
    // Write the source file's data (compressed or original) to the reserved data pack file.
    if let Some(compressed) = result.compressed.map_err(|_| {
        PackError::FailedToCompress(lookup_path(
            file_tree,
            result.task.path_hash,
            file_path_builder.clone(),
        ))
    })? {
        // Source file data was succesfully compressed.
        let compressed = compressed.compressed();

        debug_assert!(result.task.insert_index < index.len());
        let index_entry = unsafe { index.get_unchecked_mut(result.task.insert_index) };

        debug_assert_eq!(index_entry.0, 0);
        debug_assert_eq!(index_entry.1.len, 0);
        *index_entry = (
            result.task.path_hash,
            IndexEntry::new_compressed(
                result.task.pack_index,
                packs.write_reserved(compressed, result.task.pack_index)?,
                compressed.len() as _,
                hash_file(compressed, checksum_hasher),
                result.task.src_file.len() as _,
            ),
        );

        // Call the progress callback.
        on_source_file_processed.map(|on_source_file_processed| {
            on_source_file_processed(OnSourceFileProcessed {
                src_file_size: result.task.src_file.len() as _,
                packed_file_size: compressed.len() as _,
                num_packs: packs.num_packs(),
            })
        });
    } else {
        // Source file was incompressible.
        process_uncompressed_file(
            dst_dir,
            checksum_hasher,
            packs,
            index,
            result.task.insert_index,
            result.task.path_hash,
            result.task.src_file,
            Some(result.task.pack_index),
            on_source_file_processed,
        )?;
    }

    Ok(())
}

fn hash_file<C: BuildHasher>(file_data: &[u8], hash_builder: &C) -> Checksum {
    let mut file_hasher = hash_builder.build_hasher();
    file_hasher.write(file_data);
    file_hasher.finish()
}

/// Call `f` for each file in the directory at `dir_path`, recursively,
/// passing it the relative `FilePath` w.r.t. `dir_path`.
fn iterate_files_in_dir<F>(dir_path: &mut PathBuf, f: &mut F) -> Result<(), PackError>
where
    F: FnMut(&FilePath, FileSize) -> Result<(), PackError>,
{
    iterate_files_in_dir_impl(dir_path, FilePathBuilder::new(), f).map(|_| ())
}

fn iterate_files_in_dir_impl<F>(
    absolute_path: &mut PathBuf,
    mut relative_path: FilePathBuilder,
    f: &mut F,
) -> Result<FilePathBuilder, PackError>
where
    F: FnMut(&FilePath, FileSize) -> Result<(), PackError>,
{
    for entry in fs::read_dir(&absolute_path).map_err(|err| {
        PackError::FailedToIterateSourceDirectory((relative_path.clone().build(), err))
    })? {
        let entry = entry.map_err(|err| {
            PackError::FailedToIterateSourceDirectory((relative_path.clone().build(), err))
        })?;
        let name = entry.file_name();

        absolute_path.push(&name);
        relative_path.push(&name).map_err(|err| {
            PackError::InvalidSourceFileName((
                {
                    let mut path = PathBuf::from(relative_path.clone().into_inner());
                    path.push(&name);
                    path
                },
                err,
            ))
        })?;

        let metadata = entry.metadata().map_err(|err| {
            PackError::FailedToOpenSourceFile((
                // Must succeed - path is not empty.
                unsafe {
                    debug_unwrap_option(relative_path.clone().build(), "path must not be empty")
                },
                err,
            ))
        })?;

        if metadata.is_file() {
            // Skip empty files.
            if metadata.len() > 0 {
                relative_path = {
                    // Must succeed - path is not empty.
                    let relative_path = unsafe {
                        debug_unwrap_option(relative_path.build(), "path must not be empty")
                    };
                    f(&relative_path, metadata.len())?;
                    relative_path.into_builder()
                };
            }
        } else {
            relative_path = iterate_files_in_dir_impl(absolute_path, relative_path, f)?;
        }

        let popped = absolute_path.pop();
        debug_assert!(popped);
        let popped = relative_path.pop();
        debug_assert!(popped);
    }

    Ok(relative_path)
}
