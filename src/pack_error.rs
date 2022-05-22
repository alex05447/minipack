use {
    crate::*,
    minifilepath::*,
    std::{
        error::Error,
        fmt::{Display, Formatter},
        io,
        path::PathBuf,
    },
};

/// An error returned by [`PackOptions::pack`].
#[derive(Debug)]
pub enum PackError {
    /// Failed to iterate files in the source directory.
    /// Contains the relative path to the directory (or `None` if it is the root source directory)
    /// and the actual IO error.
    FailedToIterateSourceDirectory((Option<FilePathBuf>, io::Error)),
    /// Encountered an invalid source file name.
    /// Contains the relative path to the invalid source file and the actual error.
    InvalidSourceFileName((PathBuf, FilePathError)),
    /// Failed to open the source file.
    /// Contains the relative path to the source file and the actual IO error.
    ///
    /// This might happen if the source directory was modified after it was scanned for source files
    /// and before the source file was processed.
    FailedToOpenSourceFile((FilePathBuf, io::Error)),
    /// Encountered a file path corresponding to an already processed folder.
    /// Contains the relative path to the processed folder.
    FolderAlreadyExistsAtFilePath(FilePathBuf),
    /// Encountered a folder path corresponding to an already processed file.
    /// Contains the relative path to the processed file.
    FileAlreadyExistsAtFolderPath(FilePathBuf),
    /// Source file path hash collides with an already processed file path.
    /// Contains the colliding relative paths.
    PathHashCollision((FilePathBuf, FilePathBuf)),
    /// Failed to create the pack output directory.
    /// Contains the actual IO error.
    FailedToCreateOutputDirectory(io::Error),
    /// Failed to compress the source file.
    /// Contains the relative path to the source file.
    FailedToCompress(FilePathBuf),
    /// Failed to write to the index file.
    /// Contains the actual IO error.
    FailedToWriteIndexFile(io::Error),
    /// Failed to write to the data pack file.
    /// Contains the index of the data pack file and the actual IO error.
    FailedToWritePackFile((PackIndex, io::Error)),
    /// Failed to write the strings file.
    /// Contains the actual IO error.
    FailedToWriteStringsFile(io::Error),
    /// Packing was cancelled by the user.
    Cancelled,
}

impl Display for PackError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use PackError::*;

        match self {
            FailedToIterateSourceDirectory((dir, err)) => match dir {
                Some(dir) => write!(f, "failed to iterate the source directory \"{}\": {}", dir, err),
                None => write!(f, "failed to iterate the root source directory: {}", err),
            },
            InvalidSourceFileName((path, err)) => write!(f, "encountered an invalid source file name: \"{}\", {}", path.display(), err),
            FailedToOpenSourceFile((path, err)) => write!(f, "failed to open the source file \"{}\": {}", path, err),
            FolderAlreadyExistsAtFilePath(path) => write!(f, "encountered a file path corresponding to an already processed file or folder \"{}\"", path),
            FileAlreadyExistsAtFolderPath(path) => write!(f, "encountered a file or folder path corresponding to an already processed file \"{}\"", path),
            PathHashCollision((path0, path1)) => write!(f, "source file path \"{}\" hash collides with an already processed file path \"{}\"", path0, path1),
            FailedToCreateOutputDirectory(err) => write!(f, "failed to create the pack output directory: {}", err),
            FailedToCompress(path) => write!(f, "failed to compress the source file \"{}\"", path),
            FailedToWriteIndexFile(err) => write!(f, "failed to write to the index file: {}", err),
            FailedToWritePackFile((idx, err)) => write!(f, "failed to write to the data pack file {}: {}", idx, err),
            FailedToWriteStringsFile(err) => write!(f, "failed to write to the strings file: {}", err),
            Cancelled => "packing was cancelled by the user".fmt(f),
        }
    }
}

impl Error for PackError {}
