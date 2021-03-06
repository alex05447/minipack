use {
    miniunchecked::*,
    std::{
        ops::{Deref, DerefMut},
        sync::{Condvar, Mutex},
    },
};

type MemBuffer = Box<[u8]>;

/// A memory allocation returned by the [`Allocator`].
/// Derefs to a slice of bytes.
/// Borrows the allocator, so cannot outlive it.
/// This works fine for our use case.
pub(crate) struct Allocation<'a> {
    // `Option` to allow moving out of it in the `Drop` handler.
    // Always `Some` as long as the allocation lives, taken only in `Drop` when the allocation is no longer accessible.
    buffer: Option<MemBuffer>,
    // Reference to the allocator, needed to free this allocation on `Drop`.
    allocator: &'a Allocator,
}

impl<'a> Allocation<'a> {
    fn as_ref(&self) -> &[u8] {
        unsafe {
            self.buffer
                .as_ref()
                .unwrap_unchecked_dbg_msg("invalid allocation state")
        }
        .as_ref()
    }

    fn as_mut(&mut self) -> &mut [u8] {
        unsafe {
            self.buffer
                .as_mut()
                .unwrap_unchecked_dbg_msg("invalid allocation state")
        }
        .as_mut()
    }

    fn take_buffer(&mut self) -> MemBuffer {
        unsafe {
            self.buffer
                .take()
                .unwrap_unchecked_dbg_msg("invalid allocation state")
        }
    }
}

impl<'a> Deref for Allocation<'a> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

impl<'a> DerefMut for Allocation<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut()
    }
}

impl<'a> Drop for Allocation<'a> {
    fn drop(&mut self) {
        self.allocator.free(self.take_buffer());
    }
}

struct AllocatorState {
    /// If `Some`, this is the limit for how much memory we may allocate at any given time
    /// (except if the allocation size is larger than the limit in the first place,
    /// in which case we allow the allocation, but only if the allocator is otherwise completely empty).
    limit: Option<u64>,
    // How many bytes of memory we have currently allocated.
    allocated: u64,
    /// When `true`, it's a signal for the worker thread from the main thread
    /// to fail any following allocations.
    /// Happens when an error occurs and we need to exit ASAP, skipping any additional work.
    cancel_flag: bool,
}

impl AllocatorState {
    fn new(limit: Option<u64>) -> Self {
        Self {
            limit,
            allocated: 0,
            cancel_flag: false,
        }
    }

    /// Called from any thread.
    fn allocate(&mut self, len: u64) -> Option<MemBuffer> {
        // If we have a memory limit.
        if let Some(&limit) = self.limit.as_ref() {
            // Only satisfy requests above the limit if the allocator is otherwise empty.
            if len > limit {
                if self.allocated == 0 {
                    self.allocated += len;
                    Some(alloc(len))
                } else {
                    None
                }
            // For allocations below the limit, satisfy them if we actually have enough free space.
            } else {
                let remaining = limit.saturating_sub(self.allocated);
                if remaining >= len {
                    self.allocated += len;
                    Some(alloc(len))
                } else {
                    None
                }
            }
        // No memory limit - simply allocate everything.
        } else {
            self.allocated += len;
            Some(alloc(len))
        }
    }

    /// Called from any thread.
    fn free(&mut self, mem: MemBuffer) {
        debug_assert!(self.allocated >= mem.len() as u64);
        self.allocated -= mem.len() as u64;

        free(mem);
    }
}

/// An "allocator" which server these purposes:
/// - allocate temporary memory for multithreaded source file data compression;
/// - track such memory use;
/// - enforce a maximum in-flight memory limit, if any, by blocking allocating threads
/// if necessary until memory becomes available;
/// - (TODO) minimize calls to the system allocator by using a preallocated arena.
///
/// Created [`Allocation`]'s use RAII to free memory (and thus borrow the [`Allocator`]).
/// This works fine for our use case.
pub(crate) struct Allocator {
    allocator: Mutex<AllocatorState>,
    condvar: Condvar,
}

impl Allocator {
    pub(crate) fn new(limit: Option<u64>) -> Self {
        Self {
            allocator: Mutex::new(AllocatorState::new(limit)),
            condvar: Condvar::new(),
        }
    }

    /// Called from the worker threads (and also the main thread when unpacking).
    ///
    /// Tries to allocate `size` bytes respecting the memory limit,
    /// blocking while waiting for free memory if necessary.
    ///
    /// Only returns `None` in the worker threads when the main thread has called `cancel()`.
    pub(crate) fn allocate(&self, size: u64) -> Option<Allocation<'_>> {
        self.allocate_impl(size).map(|buffer| Allocation {
            buffer: Some(buffer),
            allocator: &self,
        })
    }

    /// Called from the main thread.
    ///
    /// Tries to allocate `size` bytes respecting the memory limit,
    /// without blocking while waiting for free memory.
    ///
    /// Only returns `None` if failed to allocate `size` bytes because of the memory limit.
    pub(crate) fn try_allocate(&self, size: u64) -> Option<Allocation<'_>> {
        self.try_allocate_impl(size).map(|buffer| Allocation {
            buffer: Some(buffer),
            allocator: &self,
        })
    }

    /// Called from the main thread.
    pub(crate) fn cancel(&self) {
        self.cancel_impl()
    }

    /// Called from the worker threads (and also the main thread when unpacking).
    fn allocate_impl(&self, len: u64) -> Option<MemBuffer> {
        let mut allocator_ = self.allocator.lock().unwrap();

        loop {
            // Fail all allocations after the `cancel_flag` was set by the main thread.
            if allocator_.cancel_flag {
                return None;
            }

            if let Some(mem) = allocator_.allocate(len) {
                return Some(mem);
            }

            // Wait for free memory if we hit the memory limit.
            // We'll be woken up by a call to `free()` or `cancel()`.
            allocator_ = self.condvar.wait(allocator_).unwrap();
        }
    }

    /// Called from the main thread.
    /// Tries to allocate `size` bytes respecting the memory limit,
    /// but does not block waiting for free memory on failure.
    fn try_allocate_impl(&self, size: u64) -> Option<MemBuffer> {
        self.allocator.lock().unwrap().allocate(size)
    }

    /// Called from any thread.
    fn free(&self, mem: MemBuffer) {
        self.allocator.lock().unwrap().free(mem);
        self.condvar.notify_one();
    }

    /// Called from the main thread.
    /// Wakes all wating (worker) threads and signals the allocator to fail all following allocations
    /// (e.g. an error occured and we want to return to the caller ASAP).
    pub(crate) fn cancel_impl(&self) {
        self.allocator.lock().unwrap().cancel_flag = true;
        self.condvar.notify_all();
    }
}

impl Drop for Allocator {
    fn drop(&mut self) {
        let _allocator = self.allocator.lock().unwrap();
        debug_assert_eq!(_allocator.allocated, 0);
    }
}

fn alloc(len: u64) -> MemBuffer {
    vec![0; len as _].into_boxed_slice()
}

fn free(mem: MemBuffer) {
    std::mem::drop(mem)
}
