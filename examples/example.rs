use {
    minifilepath::*,
    minifilepath_macro::*,
    minipack::*,
    seahash::*,
    std::{
        env,
        hash::{BuildHasher, Hash, Hasher},
    },
    tracing_chrome::*,
    tracing_subscriber::prelude::*,
};

#[derive(Default, Clone, Copy)]
struct BuildSeaHasher;

impl BuildHasher for BuildSeaHasher {
    type Hasher = SeaHasher;

    fn build_hasher(&self) -> Self::Hasher {
        SeaHasher::new()
    }
}

#[derive(Clone, Copy)]
struct FNV1AHasher(u64);

impl FNV1AHasher {
    fn new() -> Self {
        const FNV1A64_SEED: u64 = 0xcbf2_9ce4_8422_2325;
        Self(FNV1A64_SEED)
    }
}

impl Hasher for FNV1AHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        const FNV1A64_PRIME: u64 = 0x0000_0100_0000_01B3;

        for byte in bytes {
            self.0 = (self.0 ^ *byte as u64).wrapping_mul(FNV1A64_PRIME);
        }
    }
}

struct BuildFNV1AHasher;

impl BuildHasher for BuildFNV1AHasher {
    type Hasher = FNV1AHasher;

    fn build_hasher(&self) -> Self::Hasher {
        FNV1AHasher::new()
    }
}

fn main() {
    // "... minifiletree"
    let root_dir = env::current_dir().expect("failed to get the current directory");
    // "... minifiletree/target"
    let mut target_dir = root_dir.clone();
    target_dir.push("target_");
    let mut dst_dir = root_dir.clone();
    dst_dir.push("packed");

    let pack_impl = |max_pack_size: u64,
                     num_workers: Option<usize>,
                     memory_limit: Option<u64>|
     -> (Checksum, std::time::Duration, Option<std::time::Duration>) {
        let num_files = std::cell::RefCell::new(0);
        let mut num_processed_files = 0;
        let mut num_compressed_files = 0;
        let src_data_size = std::cell::RefCell::new(0);
        let mut processed_data_size = 0;
        let mut packed_data_size = 0;

        let start = std::time::Instant::now();
        let mut compaction_start = None;
        let mut compaction_end = None;

        let mut pack_options = PackOptions::new(
            target_dir.clone(),
            dst_dir.clone(),
            BuildFNV1AHasher,
            BuildSeaHasher
        )
        .max_pack_size(
            max_pack_size
        )
        .compression_callback(Box::new(
            |_path: &FilePath, _file_data: &[u8]| {
                    true
            }
        )).write_file_tree(
            true)
        .on_source_files_gathered(Box::new(
            |arg: OnSourceFilesGathered| {
                *num_files.borrow_mut() = arg.num_files;
                *src_data_size.borrow_mut()= arg.total_file_size;
                println!("Found {} files, {:.2} Mb total, in \"{}\"", arg.num_files, (arg.total_file_size as f64) / 1024.0 / 1024.0, target_dir.as_os_str().to_string_lossy());
                true
            })
        )
        .before_packing(Box::new(|arg: BeforePacking| {
                if arg.num_workers > 0 {
                    println!("Packing with {} worker threads.", arg.num_workers);
                } else {
                    println!("Packing in singlethreaded mode.");
                }
            })
        )
        .on_source_file_processed(Box::new(
                |arg: OnSourceFileProcessed| {
                    processed_data_size += arg.src_file_size;
                    packed_data_size += arg.packed_file_size;
                    num_processed_files += 1;

                    let compressed = arg.packed_file_size < arg.src_file_size;
                    if compressed {
                        num_compressed_files += 1;
                    }

                    print!(
                        "\rProcessed {:4} / {:4} files ({:.2} / {:.2} Mb), wrote {:.2} Mb in {:2} packs (compressed {:4} files to {:.2}%)\t",
                        num_processed_files,
                        *num_files.borrow(),
                        (processed_data_size as f64) / 1024.0 / 1024.0,
                        (*src_data_size.borrow() as f64) / 1024.0 / 1024.0,
                        (packed_data_size as f64) / 1024.0 / 1024.0,
                        arg.num_packs,
                        num_compressed_files,
                        (packed_data_size as f64) / (processed_data_size as f64) * 100.0
                    );
                    use std::io::Write;
                    std::io::stdout().flush().unwrap();

                    if num_processed_files == *num_files.borrow() {
                        println!();
                    }
                })
        )
        .before_compaction(Box::new(|arg: BeforeCompaction| {
                    compaction_start.replace(std::time::Instant::now());
                    println!("Compacting {} packs.", arg.num_packs);
                })
            )
        .on_pack_compacted(Box::new(
                |arg: OnPackCompacted| {
                    print!("\rCompacted pack {:2} of {:2}.", arg.pack_index, arg.num_packs);
                    use std::io::Write;
                    std::io::stdout().flush().unwrap();

                    if arg.pack_index == arg.num_packs - 1 {
                        println!();
                    }
                })
            ).after_compaction(Box::new(|arg: AfterCompaction| {
                    compaction_end.replace(std::time::Instant::now());
                    println!("Compacted {} packs into {}.", arg.num_packs_before_compaction, arg.num_packs);
                }));

        if let Some(num_workers) = num_workers {
            pack_options = pack_options.num_workers(num_workers);
        }

        if let Some(memory_limit) = memory_limit {
            pack_options = pack_options.memory_limit(memory_limit);
        }

        let checksum = pack_options.pack().unwrap();

        let duration = std::time::Instant::now().duration_since(start);

        let compaction_duration = if let Some(compaction_start) = compaction_start {
            if let Some(compaction_end) = compaction_end {
                Some(compaction_end.duration_since(compaction_start))
            } else {
                None
            }
        } else {
            None
        };

        (checksum, duration, compaction_duration)
    };

    let max_pack_size = 4 * 1024 * 1024;
    let num_workers = None;
    let memory_limit = None;

    let (checksum_mt, duration, compaction_duration) =
        pack_impl(max_pack_size, num_workers, memory_limit);

    if let Some(compaction_duration) = compaction_duration {
        println!(
            "Finished (multithreaded) in {:.2} sec (incl. {:.2} sec for compaction)",
            duration.as_secs_f32(),
            compaction_duration.as_secs_f32()
        );
    } else {
        println!(
            "Finished (multithreaded) in {:.2} sec",
            duration.as_secs_f32(),
        );
    }

    let pack_singlethreaded = false;

    if pack_singlethreaded {
        let (checksum_st, duration, _) = pack_impl(max_pack_size, Some(0), None);

        println!(
            "Finished (singlethreaded) in {:.2} sec",
            duration.as_secs_f32()
        );

        assert_eq!(checksum_mt, checksum_st);
    }

    let reader = {
        let mut reader =
            PackReader::new(dst_dir.clone(), BuildSeaHasher, Some(checksum_mt), true).unwrap();
        reader.build_lookup();
        reader
    };

    let path = filepath!(".rustdoc_fingerprint.json");
    let path_hash = {
        let mut path_hasher = FNV1AHasher::new();
        path.hash(&mut path_hasher);
        path_hasher.finish()
    };
    let data = reader.lookup(path_hash).unwrap();
    let data = std::str::from_utf8(&data).unwrap();
    println!("{}", data);

    let (chrome_layer, _guard) = ChromeLayerBuilder::new().file("./trace.json").build();
    tracing_subscriber::registry().with(chrome_layer).init();

    {
        let mut unpack_dir = root_dir.clone();
        unpack_dir.push("unpacked_st");

        let start = std::time::Instant::now();
        reader.unpack(unpack_dir, Some(0), None).unwrap();
        let duration = std::time::Instant::now().duration_since(start);

        println!(
            "Unpacked (singlethreaded) in {:.2} sec",
            duration.as_secs_f32()
        );
    }

    {
        let mut unpack_dir = root_dir.clone();
        unpack_dir.push("unpacked_mt");

        let start = std::time::Instant::now();
        reader.unpack(unpack_dir, Some(7), None).unwrap();
        let duration = std::time::Instant::now().duration_since(start);

        println!(
            "Unpacked (multithreaded) in {:.2} sec",
            duration.as_secs_f32()
        );
    }
}
