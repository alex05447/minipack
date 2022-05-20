use {
    crate::*,
    std::{
        io::{self, Write},
        mem::{self, ManuallyDrop},
        path::{Path, PathBuf},
    },
};

pub(crate) fn u16_to_bin_bytes(val: u16) -> [u8; mem::size_of::<u16>()] {
    u16::to_le_bytes(val)
    //u16::to_be_bytes(val)
}

pub(crate) fn u32_from_bin(bin: u32) -> u32 {
    u32::from_le(bin)
    //u32::from_be(bin)
}

pub(crate) fn u32_to_bin_bytes(val: u32) -> [u8; mem::size_of::<u32>()] {
    u32::to_le_bytes(val)
    //u32::to_be_bytes(val)
}

pub(crate) fn u64_from_bin(bin: u64) -> u64 {
    u64::from_le(bin)
    //u64::from_be(bin)
}

pub(crate) fn u64_to_bin_bytes(val: u64) -> [u8; mem::size_of::<u64>()] {
    u64::to_le_bytes(val)
    //u64::to_be_bytes(val)
}

fn write_all<W: Write>(w: &mut W, buf: &[u8]) -> Result<usize, io::Error> {
    w.write_all(buf).map(|_| buf.len())
}

pub(crate) fn write_u64<W: Write>(w: &mut W, val: u64) -> Result<usize, io::Error> {
    write_all(w, &u64_to_bin_bytes(val))
}

pub(crate) fn write_u32<W: Write>(w: &mut W, val: u32) -> Result<usize, io::Error> {
    write_all(w, &u32_to_bin_bytes(val))
}

pub(crate) fn debug_unreachable(msg: &'static str) -> ! {
    if cfg!(debug_assertions) {
        unreachable!("{}", msg)
    } else {
        unsafe { std::hint::unreachable_unchecked() }
    }
}

pub(crate) unsafe fn debug_unwrap_result<T, E>(val: Result<T, E>, msg: &'static str) -> T {
    if let Ok(val) = val {
        val
    } else {
        debug_unreachable(msg)
    }
}

pub(crate) unsafe fn debug_unwrap_option<T>(val: Option<T>, msg: &'static str) -> T {
    if let Some(val) = val {
        val
    } else {
        debug_unreachable(msg)
    }
}

/// The pack's checksum/"version" is calculated by consecutively hashing tuples of
/// (path hash, pack index, file checksum, file length)
/// for all pack files / index entries.
pub(crate) fn hash_index_entry<H: std::hash::Hasher>(
    h: &mut H,
    path_hash: PathHash,
    pack_index: PackIndex,
    file_checksum: Checksum,
    file_len: FileSize,
) {
    // NOTE: not using `Hasher::write_u64()`, because it uses `u64::to_ne_bytes()` internally,
    // but we want deterministic hashing regardless of platform endianness, so using `u64_to_bin_bytes()` for consistency.
    h.write(&u64_to_bin_bytes(path_hash));
    h.write(&u16_to_bin_bytes(pack_index));
    h.write(&u64_to_bin_bytes(file_checksum));
    h.write(&u64_to_bin_bytes(file_len));
}

pub(crate) struct PathPopGuard<'a> {
    path: &'a mut PathBuf,
}

impl<'a> PathPopGuard<'a> {
    pub(crate) fn push<P: AsRef<Path>>(path: &'a mut PathBuf, component: P) -> Self {
        path.push(component.as_ref());
        Self { path }
    }
}

impl<'a> Drop for PathPopGuard<'a> {
    fn drop(&mut self) {
        self.path.pop();
    }
}

impl<'a> AsRef<Path> for PathPopGuard<'a> {
    fn as_ref(&self) -> &Path {
        self.path
    }
}

pub(crate) struct VecAllocation {
    allocation: *mut (),
    /// In elements of unknown erased type. It's up to the caller to track this information.
    capacity: usize,
}

/// Clears the `vec` and returns its memory allocation and capacity.
///
/// Motivation: to reuse vectors (i.e. their allocations) of mutable borrows to minimize allocations while keeping the borrowchecker happy
/// (because it cannot prove the reused vec is empty and does not contain lingering mutable borrows when we're done with it and want to reuse it
/// on the next iteration of some loop, which is fair).
/// I know what I'm doing, just let me.
///
/// # Safety
///
/// See the `Vec::into_raw_parts`, gated by an unstable feature;
/// but this is also type-erased in addition, so even more unsafe.
/// It's up to the caller to ensure `empty_vec_from_allocation()` is only ever called
/// with an allocation made for a vector of appropriate type (i.e. correct alignment), or things will probably explode.
pub(crate) unsafe fn empty_vec_into_allocation<T>(mut vec: Vec<T>) -> VecAllocation {
    vec.clear();
    let mut vec = ManuallyDrop::new(vec);
    VecAllocation {
        allocation: vec.as_mut_ptr() as *mut (),
        capacity: vec.capacity(),
    }
}

/// See `empty_vec_into_raw_parts()`.
///
/// # Safety
///
/// Caller guarantees `parts` where created by a previous call to `empty_vec_into_raw_parts()` for a empty vector of matching element type `T`.
pub(crate) fn empty_vec_from_allocation<T>(parts: VecAllocation) -> Vec<T> {
    unsafe { Vec::from_raw_parts(parts.allocation as *mut T, 0, parts.capacity) }
}
