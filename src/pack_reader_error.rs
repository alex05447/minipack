use {
    crate::*,
    minifilepath::*,
    std::{
        error::Error,
        fmt::{Display, Formatter},
        io,
    },
};

/// An error returned by the [`PackReader::new`].
#[derive(Debug)]
pub enum PackReaderError {
    /// Failed to open the pack's index file.
    /// Contains the actual IO error.
    FailedToOpenIndexFile(io::Error),
    /// The pack's index file is invalid or corrupted.
    InvalidIndexFile,
    /// The pack's index file has an unexpected declared checksum value.
    /// Contains the found checksum value.
    UnexpectedChecksum(Checksum),
    /// The pack's index file's calculated checksum mismatches the declared / expected value.
    /// Contains the calculated index checksum value.
    InvalidChecksum(Checksum),
    /// Failed to open one of the pack's data pack files.
    /// Contains the data pack index and the actual IO error.
    FailedToOpenPackFile((PackIndex, io::Error)),
    /// One of the pack's data pack files is invalid or corrupted.
    /// Contains the data pack index.
    InvalidPackFile(PackIndex),
    /// The pack is invalid or corrupted.
    /// The pack's index file or one of the data pack files contain invalid data.
    CorruptData,
}

impl Display for PackReaderError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use PackReaderError::*;

        match self {
            FailedToOpenIndexFile(err) => write!(f, "failed to open the pack's index file: {}", err),
            InvalidIndexFile => "the pack's index file is invalid or corrupted".fmt(f),
            UnexpectedChecksum(checksum) => write!(f, "the pack's index file has an unexpected checksum value ({})", checksum),
            InvalidChecksum(checksum) => write!(f, "the pack's index file's calculated checksum ({}) mismatches the declared / expected value", checksum),
            FailedToOpenPackFile((idx, err)) => write!(f, "failed to open the pack's data pack file {}: {}", idx, err),
            InvalidPackFile(idx) => write!(f, "the pack's data pack file {} is invalid", idx),
            CorruptData => "the pack is invalid or corrupted".fmt(f),
        }
    }
}

impl Error for PackReaderError {}

/// An error returned by [`PackReader::unpack`].
#[derive(Debug)]
pub enum UnpackError {
    /// Failed to open the strings file.
    /// Contains the actual IO error.
    FailedToOpenStringsFile(io::Error),
    /// The pack's strings file is invalid or corrupted.
    InvalidStringsFile,
    /// The pack's strings file has an unexpected declared checksum value.
    /// Contains the found checksum value.
    UnexpectedChecksum(Checksum),
    /// Failed to create an output folder.
    /// Contains the relative path to the folder (or `None` if it is the root output folder)
    /// and the actual IO error.
    FailedToCreateOutputFolder((Option<FilePathBuf>, io::Error)),
    /// Failed to create an output file.
    /// Contains the relative path to the file and the actual IO error.
    FailedToCreateOutputFile((FilePathBuf, io::Error)),
    /// Failed to write to an output file.
    /// Contains the relative path to the file and the actual IO error.
    FailedToWriteOutputFile((FilePathBuf, io::Error)),
}

impl Display for UnpackError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use UnpackError::*;

        match self {
            FailedToOpenStringsFile(err) => {
                write!(f, "failed to open the pack's strings file: {}", err)
            }
            InvalidStringsFile => "the pack's strings file is invalid or corrupted".fmt(f),
            UnexpectedChecksum(checksum) => write!(
                f,
                "the pack's strings file has an unexpected checksum value ({})",
                checksum
            ),
            FailedToCreateOutputFolder((dir, err)) => match dir {
                Some(dir) => write!(f, "failed to create output foler \"{}\": {}", dir, err),
                None => write!(f, "failed to create the root output foler: {}", err),
            },
            FailedToCreateOutputFile((file, err)) => {
                write!(f, "failed to create output file \"{}\": {}", file, err)
            }
            FailedToWriteOutputFile((file, err)) => {
                write!(f, "failed to write to output file \"{}\": {}", file, err)
            }
        }
    }
}

impl Error for UnpackError {}
