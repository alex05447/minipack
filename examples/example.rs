use {
    minifilepath::*,
    minifilepath_macro::*,
    minipack::*,
    seahash::*,
    std::{
        env,
        hash::{BuildHasher, Hash, Hasher},
    },
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
    target_dir.push("target");
    let mut dst_dir = root_dir;
    dst_dir.push("packed");

    let do_packing = |num_workers: Option<usize>, memory_limit: Option<u64>| -> Checksum {
        let num_files = std::cell::RefCell::new(0);
        let mut num_processed_files = 0;
        let mut num_compressed_files = 0;
        let src_data_size = std::cell::RefCell::new(0);
        let mut processed_data_size = 0;
        let mut packed_data_size = 0;

        pack(
            target_dir.clone(),
            dst_dir.clone(),
            4 * 1024 * 1024,
            BuildFNV1AHasher,
            BuildSeaHasher,
            |_path: &FilePath, _file_data: &[u8]| {
                    true
            },
            true,
            num_workers,
            memory_limit,
            ProgressCallbacks {
                on_source_files_gathered: |arg: OnSourceFilesGathered| {
                    *num_files.borrow_mut() = arg.num_files;
                    *src_data_size.borrow_mut()= arg.total_file_size;
                    println!("Found {} files, {:.2} Mb total, in \"{}\"", arg.num_files, (arg.total_file_size as f64) / 1024.0 / 1024.0, target_dir.as_os_str().to_string_lossy());
                    true
                },
                on_source_file_processed: |arg: OnSourceFileProcessed| {
                    processed_data_size += arg.src_file_size;
                    packed_data_size += arg.packed_file_size;
                    num_processed_files += 1;

                    let compressed = arg.packed_file_size < arg.src_file_size;
                    if compressed {
                        num_compressed_files += 1;
                    }

                    print!(
                        "\rProcessed {:4} / {:4} files ({:.2} / {:.2} Mb), wrote {:.2} Mb (compressed {:4} to {:.2}%)\t",
                        num_processed_files,
                        *num_files.borrow(),
                        (processed_data_size as f64) / 1024.0 / 1024.0,
                        (*src_data_size.borrow() as f64) / 1024.0 / 1024.0,
                        (packed_data_size as f64) / 1024.0 / 1024.0,
                        num_compressed_files,
                        (packed_data_size as f64) / (processed_data_size as f64) * 100.0
                    );
                    use std::io::Write;
                    std::io::stdout().flush().unwrap();

                    if num_processed_files == *num_files.borrow() {
                        println!();
                    }
                },
            }
        )
        .unwrap()
    };

    // Use automatic detection.
    let num_workers = None;
    let memory_limit = None;//Some(8 * 1024 * 1024);

    let start = std::time::Instant::now();
    let checksum_mt = do_packing(num_workers, memory_limit);
    let duration = std::time::Instant::now().duration_since(start);

    println!("Finished (multithreaded) in {} sec", duration.as_secs_f32());

    let start = std::time::Instant::now();
    let checksum_st = do_packing(Some(0), None);
    let duration = std::time::Instant::now().duration_since(start);

    println!(
        "Finished (singlethreaded) in {} sec",
        duration.as_secs_f32()
    );

    assert_eq!(checksum_mt, checksum_st);

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
}
