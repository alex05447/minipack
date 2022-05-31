use {
    super::{result_queue::ResultQueue, task_queue::TaskQueue},
    miniclosure::ClosureHolder,
    std::{marker::PhantomData, num::NonZeroUsize, sync::Arc, thread},
};

struct ClosureWrapper(ClosureHolder);

impl ClosureWrapper {
    fn execute_mut<C>(&mut self, context: &mut C) {
        unsafe { self.0.execute_mut(context) };
    }
}

unsafe impl Send for ClosureWrapper {}

pub(crate) enum ResultOrTask<R, T> {
    Result(R),
    Task(T),
}

/// A simple, hacky thread pool.
///
/// User pushes tasks `T` to the task queue from the main thread.
/// Worker threads pop the tasks `T` from the task queue, blocking if empty, and push results `R` to the result queue.
/// Main thread pops the results `R` from the ready queue, or tasks `T` from the task queue, blocking on the result queue if empty.
///
/// Worker threads exit when the user calls [`ThreadPool::finish`], or the thread pool is `Drop`'ped.
///
/// The thread pool's `Drop` handler ensures the user-provided closures' borrows outlive the thread pool
/// by canceling unfinished tasks, if any, and waiting for the spawned worker threads to exit.
/// Thus it is unsafe to `mem::forget` the thread pool when using stack borrows in provided closures.
/// Making it fully safe while allowing stack borrows would require a solution similar to `crossbeam::scope`.
///
/// Handles unprocessed results, if any, in the `Drop` handler by calling the user-provided drop handler for results `R`.
/// TODO: implement same for unprocessed tasks.
pub(crate) struct ThreadPool<'env, T, R, W>
where
    W: FnMut() + 'env,
{
    /// Spawned worker thread join handles.
    /// Joined when the thread pool is dropped,
    /// ensuring all borrows by the worker thread entry point must outlive the thread pool.
    /// Obviously `mem::forget`'ting the thread pool will result in bad things, but we're not doing that.
    workers: Option<Vec<thread::JoinHandle<()>>>,
    /// Tasks are pushed to the task queue by the main thread, popped by the worker threads
    /// (and also by the main thread to help out when waiting for results but none are awailable).
    task_queue: Arc<TaskQueue<T>>,
    /// Results are pushed to the result queue by the worker threads, popped by the main thread.
    result_queue: Arc<ResultQueue<R>>,
    /// The number of tasks pushed so far, and the corresponding number of results we expect.
    /// Incremented for each pushed task, decremented for each popped result.
    /// Needed to track when to block the main thread in `pop_result_or_task()`.
    num_tasks: usize,
    /// Optional user-provided closure called in the thread pool's `Drop` handler to wake up the worker threads.
    waker: W,
    _marker: PhantomData<&'env ()>,
}

impl<'env, T, R, W> Drop for ThreadPool<'env, T, R, W>
where
    W: FnMut() + 'env,
{
    fn drop(&mut self) {
        // Cancel processing all remaining tasks.
        // If we are dropping the thread pool and there are unfinished tasks,
        // we've encountered an error or a panic and want to exit ASAP.
        self.task_queue.cancel();

        // Call the user-provided waker to wake up the workers, if necessary.
        (self.waker)();

        // Wait for the worker threads to exit.
        self.workers
            .take()
            .map(Vec::into_iter)
            .map(|workers| workers.for_each(|worker| worker.join().unwrap()));
    }
}

impl<'env, T, R, W> ThreadPool<'env, T, R, W>
where
    W: FnMut() + 'env,
{
    /// Creates a thread pool with `num_workers` worker threads.
    /// A thread context `C` is created per spawned thread by the `init_context` closure, passed the worker thread index.
    /// Closure `f` is called for each processed task, passed the mutable reference to thread's context `&mut C` and the task `T`.
    /// If the closure returns `None`, the worker thread exits.
    ///
    /// When `Drop`'ped, cancels unfinished tasks, if any.
    /// Calls the `waker`, if any, to wake up the worker threads before waiting for them to exit.
    pub(crate) fn new<I, C, F>(num_workers: NonZeroUsize, init_context: I, f: F, waker: W) -> Self
    where
        I: FnOnce(usize) -> C + Clone + Send + 'static,
        F: FnMut(&mut C, T) -> Option<R> + Send + Sync + Clone + 'env,
        T: 'env + Send,
        R: 'env + Send,
    {
        let task_queue = Arc::new(TaskQueue::new());
        let result_queue = Arc::new(ResultQueue::new());

        let workers = (0..num_workers.get())
            .map(|worker_index| {
                let task_queue = Arc::clone(&task_queue);
                let result_queue = Arc::clone(&result_queue);
                let mut f = f.clone();

                // Safe (but allocates on the heap):
                // let f: Box<dyn FnMut(&mut C, T) -> Option<R> + Send + 'env> = Box::new(f);
                // let mut f: Box<dyn FnMut(&mut C, T) -> Option<R> + Send + 'static> =
                //         unsafe { std::mem::transmute(f) };

                let mut worker_loop_impl = move |context: &mut C| {
                    // Exit the worker thread loop if
                    // - the main thread signalled there will be no more tasks and there are none, or if
                    // - the main thread signalled us to exit immediately.
                    while let Some(task) = task_queue.pop() {
                        if let Some(result) = f(context, task) {
                            result_queue.push(result);
                        // Exit the worker thread loop if the user closure returned `None`.
                        } else {
                            break;
                        }
                    }
                };

                // Unsafe (but uses the small function optimization under the hood):
                let mut f = ClosureWrapper(unsafe {
                    ClosureHolder::new_mut(move |context: &mut C| {
                        worker_loop_impl(context);
                    })
                });

                let init_context = init_context.clone();

                thread::Builder::new()
                    .name(format!("Worker thread {}", worker_index))
                    .spawn(move || {
                        let mut context = init_context(worker_index);

                        // Safe:
                        //worker_loop_impl(&mut context);

                        // Unsafe:
                        f.execute_mut(&mut context);
                    })
                    .unwrap()
            })
            .collect::<Vec<_>>();

        Self {
            workers: Some(workers),
            task_queue,
            result_queue,
            num_tasks: 0,
            waker,
            _marker: PhantomData,
        }
    }

    /// Called from the main thread.
    /// Adds the `task` to the queue, wakes up a single worker.
    pub(crate) fn push(&mut self, task: T) {
        self.task_queue.push(task);
        self.num_tasks += 1;
    }

    // Called from the main thread.
    // Adds all the `tasks` from the iterator to the queue, wakes up all workers.
    pub(crate) fn push_tasks<I: Iterator<Item = T>>(&mut self, tasks: I) {
        let num_tasks = self.task_queue.push_tasks(tasks);
        self.num_tasks += num_tasks;
    }

    /// Called from the main thread.
    /// Signals the worker threads to finish any remaining tasks and exit, as no new tasks will be added.
    /// Does not wait for the worker threads to exit.
    ///
    /// NOTE: the caller must guarantee no new tasks will be pushed to the task queue after this call,
    /// as those are not guaranteed to ever be processed as the worker threads might have already exited by that point.
    /// Alternatively, the main thread might drain the task queue itself by calling `pop_result_or_task()` in a loop.
    pub(crate) fn finish(&self) {
        self.task_queue.finish();
    }

    /// Called from the main thread.
    /// Tries to pop a result from the queue.
    /// If there are no results, tries to pop a task from the queue.
    /// If there are no tasks, blocks on the result queue waiting for a result to be pushed by the worker thread.
    ///
    /// Returns `None` if we have processed all results and tasks.
    pub(crate) fn pop_result_or_task(&mut self) -> Option<ResultOrTask<R, T>> {
        if self.num_tasks == 0 {
            return None;
        }

        self.num_tasks -= 1;

        Some(if let Some(result) = self.result_queue.try_pop() {
            ResultOrTask::Result(result)
        } else if let Some(task) = self.task_queue.try_pop() {
            ResultOrTask::Task(task)
        } else {
            ResultOrTask::Result(self.result_queue.pop())
        })
    }

    /// Called from the main thread.
    /// Tries to pop a result from the result queue.
    /// Never blocks.
    pub(crate) fn try_pop_result(&mut self) -> Option<R> {
        self.result_queue.try_pop().map(|result| {
            debug_assert!(self.num_tasks > 0);
            self.num_tasks -= 1;
            result
        })
    }
}
