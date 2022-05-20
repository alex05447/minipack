use std::{
    collections::VecDeque,
    sync::{Condvar, Mutex},
};

/// `Arc`-shared result queue state.
pub(super) struct ResultQueue<R> {
    queue: Mutex<VecDeque<R>>,
    condvar: Condvar,
}

impl<R> ResultQueue<R> {
    pub(super) fn new() -> Self {
        Self {
            queue: Mutex::new(VecDeque::new()),
            condvar: Condvar::new(),
        }
    }

    /// Called from the worker threads.
    /// Adds the `result` to the queue, wakes up the main thread.
    pub(super) fn push(&self, result: R) {
        self.queue.lock().unwrap().push_back(result);
        self.condvar.notify_one();
    }

    /// Called from the main thread.
    /// Tries to pop a result from the queue. Never blocks.
    pub(super) fn try_pop(&self) -> Option<R> {
        self.queue.lock().unwrap().pop_front()
    }

    /// Called from the main thread.
    /// Tries to pop a result from the queue, blocking on a condvar if the queue is empty.
    ///
    /// NOTE: the caller guarantees there either already is at least one result in the queue,
    /// or there will eventually be one pushed by the worker thread.
    /// Otherwise the main thread will deadlock blocked on an empty result queue.
    pub(super) fn pop(&self) -> R {
        let mut queue = self.queue.lock().unwrap();

        if let Some(result) = queue.pop_back() {
            return result;
        }

        loop {
            queue = self.condvar.wait(queue).unwrap();

            if let Some(result) = queue.pop_back() {
                return result;
            }
        }
    }
}
