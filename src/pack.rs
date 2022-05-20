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

/// Argument passed to the [`ProgressCallbacks::on_source_files_gathered`] callback.
pub struct OnSourceFilesGathered {
    /// Total number of (non-empty) files found in the source directory.
    pub num_files: usize,
    /// Total size in bytes of all (non-empty) files found in the source directory.
    pub total_file_size: u64,
}

/// Argument passed to the [`ProgressCallbacks::on_source_file_processed`] callback.
pub struct OnSourceFileProcessed {
    /// Original size in bytes of the processed file's data.
    /// These add up to [`OnSourceFilesGathered::total_file_size`].
    pub src_file_size: u64,
    /// Equal to `src_file_size` if the source file was not compressed;
    /// strictly less than `src_file_size` if the source file was compressed;
    pub packed_file_size: u64,
}

/// User-provided progress-tracking callbacks passed to [`pack()`].
pub struct ProgressCallbacks<PCb1, PCb2> {
    /// Called once after the source directory was processed and all (non-empty) source data files gathered.
    /// Is passed [`OnSourceFilesGathered`].
    /// Returns `true` if processing should continue, or `false` if it should be cancelled.
    pub on_source_files_gathered: PCb1,
    /// Called after `on_source_files_gathered`, once for each processed source data file
    /// (so [`OnSourceFilesGathered::num_files`] times total, unless an error occurs).
    /// Is passed [`OnSourceFileProcessed`].
    pub on_source_file_processed: PCb2,
}

/// Creates a pack in `dst_dir` from the contents of the `src_dir`.
///
/// Iterates recursively over all (non-empty) files in `src_dir`.
///
/// Splits the file data into "data pack" files no larger then `max_pack_len` bytes
/// (unless an individual resource file is larger than this value, in which case the "data pack" file
/// will be as large as the resource file (and only contain this resource file)).
///
/// `filepath_hasher` is used to create [`PathHash`]'es from the resource file relative paths whithin `src_dir`
/// to create the lookup data structure.
/// This same hasher must then be used when looking up the resource file data in the created pack.
///
/// `checksum_hasher` is used to calculate resource file checksums and the pack's checksum / "version", returned by this function.
/// This same hasher must then be used when opening the pack for reading.
///
/// `compression_callback` is called for each processed resource file to determine whether it must be compressed.
/// The callback is passed the file's relative path in `src_dir` and its data.
/// If the callback returns `true`, the file data will be compressed (except if the compressed data is (slightly) larger in size than uncompressed,
/// which is possible due to compression format overhead when the data is incompressible).
/// If the callback returns `false`, source file data is not compressed.
///
/// If `write_file_tree` is `true`, data necessary to unpack the file pack is serialized along with it.
///
/// `num_workers` determines the number of worker threads which will be spawned to compress source file data, if necessary.
/// If `None`, the number of threads is determined by `std::thread::available_parallelism()` (minus 1 for the current thread, which is also used for compression).
/// If `Some(0)`, no worker threads are spawned, all processing is done on the current thread.
/// Otherwise one or more worker threads are used.
///
/// If `memory_limit` is `Some`, this value will determine the attempted best-effort limit
/// on the amount of temporary memory allocated at any given time for source file compression purposes.
/// NOTE: the limit will be exceeded if the required amount of memory to compress a single source file exceeds it in the first place.
/// NOTE: this value, if too low, also potentially limits compression parallellism, as worker threads cannot
/// proceed with compression while waiting for temporary memory to become available.
/// As a rule of thumb, `memory_limit` should be approximately equal to `num_threads * average_file_size`,
/// where `average_file_size` is the average/representative source file size.
///
/// `progress_callbacks` struct contains the callbacks called to track packing progress.
pub fn pack<F, H, CCb, PCb1, PCb2>(
    mut src_dir: PathBuf,
    mut dst_dir: PathBuf,
    max_pack_len: u64,
    filepath_hasher: F,
    checksum_hasher: H,
    compression_callback: CCb,
    write_file_tree: bool,
    num_workers: Option<usize>,
    memory_limit: Option<u64>,
    progress_callbacks: ProgressCallbacks<PCb1, PCb2>,
) -> Result<Checksum, PackError>
where
    F: BuildHasher,
    F::Hasher: Clone,
    H: BuildHasher + Clone,
    CCb: FnMut(&FilePath, &[u8]) -> bool,
    PCb1: FnOnce(OnSourceFilesGathered) -> bool,
    PCb2: FnMut(OnSourceFileProcessed),
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

    if !(progress_callbacks.on_source_files_gathered)(OnSourceFilesGathered {
        num_files: path_hashes.len(),
        total_file_size,
    }) {
        return Err(PackError::Cancelled);
    }

    // Create the output directory, if necessary.
    fs::create_dir_all(&dst_dir).map_err(|err| PackError::FailedToCreateOutputDirectory(err))?;

    let index = IndexWriter::new(checksum_hasher.build_hasher(), &mut dst_dir)?;
    let packs = DataPackWriter::new(max_pack_len);

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
    let num_workers = match num_workers {
        Some(num_workers) => NonZeroUsize::new(num_workers),
        // Subtract `1` for the main thread.
        None => thread::available_parallelism()
            .ok()
            .and_then(|num_workers| NonZeroUsize::new(num_workers.get() - 1)),
    };

    let (mut index_, mut packs) = if let Some(num_workers) = num_workers {
        compress_multithreaded(
            src_dir,
            &dst_dir,
            packs,
            &file_tree,
            &checksum_hasher,
            compression_callback,
            num_workers,
            memory_limit,
            path_hashes,
            progress_callbacks.on_source_file_processed,
        )
    } else {
        compress_singlethreaded(
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

fn compress_singlethreaded<F, H, CCbb>(
    src_dir: PathBuf,
    dst_dir: &PathBuf,
    mut packs: DataPackWriter,
    file_tree: &FileTreeWriter<F>,
    checksum_hasher: &H,
    mut compression_callback: CCbb,
    path_hashes: Vec<PathHash>,
    mut on_source_file_processed: impl FnMut(OnSourceFileProcessed),
) -> Result<(Vec<(PathHash, IndexEntry)>, DataPackWriter), PackError>
where
    F: BuildHasher,
    F::Hasher: Clone,
    H: BuildHasher + Clone,
    CCbb: FnMut(&FilePath, &[u8]) -> bool,
{
    let mut file_path_builder = FilePathBuilder::new();
    let mut absolute_file_path = src_dir.clone();

    let mut compressor = Compressor::new().expect("failed to create an LZ4 compressor");
    // Scratch buffer used for compression.
    let mut compressed_data = Vec::new();

    let mut hashes_and_index_entries = Vec::new();

    for path_hash in path_hashes {
        // Lookup the relative file path. Must succeed - all path hashes have been inserted.
        let relative_file_path = lookup_path(&file_tree, path_hash, file_path_builder);

        // Build the absolute file path.
        absolute_file_path.push(relative_file_path.as_path());

        // Open and map the source file.
        let src_file = || -> _ { unsafe { Mmap::map(&File::open(&absolute_file_path)?) } }()
            .map_err(|err| PackError::FailedToOpenSourceFile((relative_file_path.clone(), err)))?;

        // Call the compression callback, passing the relative file path and the file data.
        let compress = compression_callback(&relative_file_path, &src_file);

        // Try to compress the file data if the compression callback specified it.
        let (compressed, file_data) = if compress {
            compressor
                .compress_into_vec(&src_file, &mut compressed_data, false)
                .map_err(|_| PackError::FailedToCompress(relative_file_path.clone()))?;

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
        (on_source_file_processed)(OnSourceFileProcessed {
            src_file_size: src_file.len() as _,
            packed_file_size: file_data.len() as _,
        });

        // Reset the relative and absolute paths.
        file_path_builder = relative_file_path.into_builder();
        absolute_file_path.clear();
        absolute_file_path.push(&src_dir);
    }

    Ok((hashes_and_index_entries, packs))
}

fn compress_multithreaded<F, H, CCb, PCb>(
    src_dir: PathBuf,
    dst_dir: &PathBuf,
    mut packs: DataPackWriter,
    file_tree: &FileTreeWriter<F>,
    checksum_hasher: &H,
    mut compression_callback: CCb,
    num_workers: NonZeroUsize,
    memory_limit: Option<u64>,
    path_hashes: Vec<PathHash>,
    mut on_source_file_processed: PCb,
) -> Result<(Vec<(PathHash, IndexEntry)>, DataPackWriter), PackError>
where
    F: BuildHasher,
    F::Hasher: Clone,
    H: BuildHasher + Clone,
    CCb: FnMut(&FilePath, &[u8]) -> bool,
    PCb: FnMut(OnSourceFileProcessed),
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
        let relative_file_path = lookup_path(&file_tree, path_hash, file_path_builder);

        // Build the absolute file path.
        absolute_file_path.push(relative_file_path.as_path());

        // Open and map the source file.
        let src_file = || -> _ { unsafe { Mmap::map(&File::open(&absolute_file_path)?) } }()
            .map_err(|err| PackError::FailedToOpenSourceFile((relative_file_path.clone(), err)))?;

        // Call the compression callback, passing the relative file path and the file data.
        if compression_callback(&relative_file_path, &src_file) {
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
                &mut on_source_file_processed,
            )?;
        }

        // Reset the relative and absolute paths.
        file_path_builder = relative_file_path.into_builder();
        absolute_file_path.clear();
        absolute_file_path.push(&src_dir);

        // Try to process a single compression result each iteration
        // to relieve allocator pressure if we use a memory limit.
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
                    &mut on_source_file_processed,
                )?;
            }
        }
    }

    let mut compressor = Compressor::new().expect("failed to create an LZ4 compressor");

    // Signal the thread pool there will be no more tasks.
    thread_pool.finish();

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
                    &mut on_source_file_processed,
                )?;
            }
            ResultOrTask::Task(task) => {
                // Try to allocate memory for the compression task.
                // Do not block, return the task to the queue on failure.
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
                        &mut on_source_file_processed,
                    )?;
                } else {
                    thread_pool.push(task);
                }
            }
        }
    }

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
            &mut on_source_file_processed,
        )?;
    }

    // Compact the data packs.
    packs.compact(&mut index)?;

    Ok((index, packs))
}

/// Handles files which do not require compression (in which case `pack_index` is `None`),
/// or which where incomressible (in which case `pack_index` is `Some`, as returned by `packs.reserve()`).
fn process_uncompressed_file<H, PCb>(
    dst_dir: &PathBuf,
    checksum_hasher: &H,
    packs: &mut DataPackWriter,
    index: &mut Vec<(PathHash, IndexEntry)>,
    insert_index: usize,
    path_hash: PathHash,
    src_file: Mmap,
    pack_index: Option<PackIndex>,
    on_source_file_processed: &mut PCb,
) -> Result<(), PackError>
where
    H: BuildHasher,
    PCb: FnMut(OnSourceFileProcessed),
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
    on_source_file_processed(OnSourceFileProcessed {
        src_file_size: src_file.len() as _,
        packed_file_size: src_file.len() as _,
    });

    Ok(())
}

// A compression task sent to the thread pool from the main thread.
struct CompressionTask {
    // Mapped source file data.
    src_file: Mmap,
    // Reserved data pack file index for this source file.
    pack_index: PackIndex,
    // Needed for error reporting (to lookup the file name).
    path_hash: PathHash,
    // Needed to avoid re-sorting the (path hash, index entry) result index array in file path alphabetic order
    // after non-deterministic multithreaded compression.
    // Index must be sorted in alphabetic order, same as for singlethreaded processing
    // for deterministic results.
    insert_index: usize,
}

// Compressed source file data buffer allocated by the worker thread.
struct CompressedBuffer<'a> {
    // The entire allocation for this buffer.
    allocation: Allocation<'a>,
    // Actual compressed source file data size in bytes within the buffer.
    compressed_size: NonZeroU64,
}

impl<'a> CompressedBuffer<'a> {
    // Returns the subslice of the allocation with the compressed source file data.
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
    // `Ok(Some)` if succesfully compressed.
    // `Ok(None)` if source file data was incompressible.
    // `Err` if failed to compress.
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

fn compress_file<H, F, PCb>(
    checksum_hasher: &H,
    packs: &mut DataPackWriter,
    index: &mut Vec<(PathHash, IndexEntry)>,
    task: CompressionTask,
    file_tree: &FileTreeWriter<F>,
    file_path_builder: &FilePathBuilder,
    mut allocation: Allocation<'_>,
    compressor: &mut Compressor,
    on_source_file_processed: &mut PCb,
) -> Result<(), PackError>
where
    H: BuildHasher,
    F: BuildHasher,
    F::Hasher: Clone,
    PCb: FnMut(OnSourceFileProcessed),
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
    on_source_file_processed(OnSourceFileProcessed {
        src_file_size: task.src_file.len() as _,
        packed_file_size: file_data.len() as _,
    });

    Ok(())
}

fn process_result<F, H, PCb>(
    dst_dir: &PathBuf,
    file_tree: &FileTreeWriter<F>,
    checksum_hasher: &H,
    packs: &mut DataPackWriter,
    index: &mut Vec<(PathHash, IndexEntry)>,
    result: CompressionResult,
    file_path_builder: &FilePathBuilder,
    on_source_file_processed: &mut PCb,
) -> Result<(), PackError>
where
    F: BuildHasher,
    F::Hasher: Clone,
    H: BuildHasher,
    PCb: FnMut(OnSourceFileProcessed),
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
        on_source_file_processed(OnSourceFileProcessed {
            src_file_size: result.task.src_file.len() as _,
            packed_file_size: compressed.len() as _,
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
