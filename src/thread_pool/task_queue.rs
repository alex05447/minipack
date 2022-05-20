use std::{
    collections::VecDeque,
    sync::{Condvar, Mutex},
};

/// `Mutex`-protected task queue state.
struct TaskQueueState<T> {
    /// FIFO task queue.
    /// Pushed to (to the back) by the main thread.
    /// Popped from (from the front) by the worker threads.
    queue: VecDeque<T>,
    /// When `true`, it's a signal for the worker thread from the main thread
    /// to first process the remainder of the tasks in the queue,
    /// then exit the worker thread loop.
    /// Happens when no new tasks will be pushed to the queue,
    /// either because the main thread explicitly signaled so, or because the thread pool is being destroyed.
    finish_flag: bool,
    /// When `true`, it's a signal for the worker thread from the main thread
    /// to exit the worker thread loop without processing the remainder of the tasks in the queue.
    /// Happens when an error occurs and we need to exit ASAP, skipping any additional work.
    cancel_flag: bool,
}

impl<T> TaskQueueState<T> {
    fn new() -> Self {
        Self {
            queue: VecDeque::new(),
            finish_flag: false,
            cancel_flag: false,
        }
    }

    /// Called from the main thread.
    fn push(&mut self, task: T) {
        self.queue.push_back(task);
    }

    /// Called from any thread.
    fn pop(&mut self) -> Option<T> {
        self.queue.pop_front()
    }

    /// Called from the main thread.
    fn finish(&mut self) {
        self.finish_flag = true;
    }

    /// Called from the main thread.
    fn cancel(&mut self) {
        self.cancel_flag = true;
    }
}

/// `Arc`-shared task queue state.
pub(super) struct TaskQueue<T> {
    queue: Mutex<TaskQueueState<T>>,
    condvar: Condvar,
}

impl<T> TaskQueue<T> {
    pub(super) fn new() -> Self {
        Self {
            queue: Mutex::new(TaskQueueState::new()),
            condvar: Condvar::new(),
        }
    }

    /// See `ThreadPool::push`
    pub(super) fn push(&self, task: T) {
        self.queue.lock().unwrap().push(task);
        self.condvar.notify_one();
    }

    /*
    /// See `ThreadPool::push_tasks`
    pub(super) fn push_tasks<I: Iterator<Item = T>>(&self, tasks: I) -> usize {
        let mut queue = self.queue.lock().unwrap();
        let mut num_tasks = 0;
        for task in tasks {
            queue.push(task);
            num_tasks += 1;
        }
        self.condvar.notify_all();
        num_tasks
    }
    */

    /// Called from the worker threads.
    /// Tries to pop a task from the queue, blocking on a condvar if the queue is empty.
    /// Returns `None` if the main thread signaled for the worker thread to exit
    /// (either after all tasks have been popped and processed, or after we've been told to exit immediately).
    pub(super) fn pop(&self) -> Option<T> {
        let mut queue_state = self.queue.lock().unwrap();

        loop {
            // If the cancel flag is set, exit immediately, abandoning any remaining tasks.
            if queue_state.cancel_flag {
                return None;

            // Popped a task - return it, release the lock.
            } else if let Some(val) = queue_state.pop() {
                return Some(val);
            // Otherwise the queue is empty.
            } else {
                // If the finish flag is set, we're done - signal the worker thread to exit (by returning `None`).
                if queue_state.finish_flag {
                    return None;
                // If the finish flag is not set, wait for a new task by blocking on a condvar.
                // The main thread will wake us up by either calling `push()`, or `finish()` / `cancel()`.
                } else {
                    queue_state = self.condvar.wait(queue_state).unwrap();
                }
            }
        }
    }

    /// Called from the main thread.
    /// Tries to pop a task from the queue. Never blocks.
    pub(super) fn try_pop(&self) -> Option<T> {
        self.queue.lock().unwrap().pop()
    }

    /// Called from the main thread.
    /// See `ThreadPool::finish`
    pub(super) fn finish(&self) {
        self.queue.lock().unwrap().finish();
        self.condvar.notify_all();
    }

    /// Called from the main thread.
    /// Wakes the worker threads and signals them to abandon any remaining tasks and exit, as no new tasks will be added
    /// and there's no need to process the remaining tasks (e.g. an error occured and we want to return to the caller ASAP).
    /// Does not wait for the worker threads to exit.
    /// NOTE: does not cancel already in-flight tasks.
    pub(super) fn cancel(&self) {
        self.queue.lock().unwrap().cancel();
        self.condvar.notify_all();
    }
}
