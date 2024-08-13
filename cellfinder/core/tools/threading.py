"""
Provides classes that can run a function in another thread or process and
allow passing data to and from the threads/processes. It also passes on any
exceptions that occur in the secondary thread/sub-process in the main thread
or when it exits.

If using a sub-process and pytorch Tensors are sent from/to the main
process, pytorch will memory map the tensor so the same data is shared and
edits in the main process will be visible in the sub-process. However, there
are limitations (such so not re-sharing a tensor shared with us). See
https://pytorch.org/docs/stable/multiprocessing.html#sharing-cuda-tensors for
details.

Typical example::

    from cellfinder.core.tools.threading import ThreadWithException, \\
        EOFSignal, ProcessWithException
    import torch


    def worker(thread: ThreadWithException, power: float):
        while True:
            # if the main thread wants us to exit, it'll wake us up
            msg = thread.get_msg_from_mainthread()
            # we were asked to exit
            if msg == EOFSignal:
                return

            tensor_id, tensor, add = msg
            # tensors are memory mapped for subprocess (and obv threads) so we
            # can do the work inplace and result will be visible in main thread
            tensor += add
            tensor.pow_(power)

            # we should not share a tensor shared with us as per pytorch docs,
            # just send the id back
            thread.send_msg_to_mainthread(tensor_id)

        # we can also handle errors here, which will be re-raised in the main
        # process
            if tensor_id == 7:
                raise ValueError("I fell asleep")


    if __name__ == "__main__":
        data = torch.rand((5, 10))
        thread = ThreadWithException(
            target=worker, args=(2.5,), pass_self=True
        )
        # use thread or sub-process
        # thread = ProcessWithException(
        #     target=worker, args=(2.5,), pass_self=True
        # )
        thread.start()

        try:
            for i in range(10):
                thread.send_msg_to_thread((i, data, i / 2))
                # if the thread raises an exception, get_msg_from_thread will
                # re-raise it here
                msg = thread.get_msg_from_thread()
                if msg == EOFSignal:
                    # thread exited for whatever reason (not exception)
                    break

                print(f"Thread processed tensor {i}")
        finally:
            # whatever happens, make sure thread is told to finish so it
            # doesn't get stuck
            thread.notify_to_end_thread()
        thread.join()

When run, this prints::

    Thread processed tensor 0
    Thread processed tensor 1
    Thread processed tensor 2
    Thread processed tensor 3
    Thread processed tensor 4
    Thread processed tensor 5
    Thread processed tensor 6
    Thread processed tensor 7
    Traceback (most recent call last):
      File "threading.py", line 139, in user_func_runner
        self.user_func(self, *self.args)
      File "play.py", line 24, in worker
        raise ValueError("I fell asleep")
    ValueError: I fell asleep

    The above exception was the direct cause of the following exception:

    Traceback (most recent call last):
      File "play.py", line 38, in <module>
        msg = thread.get_msg_from_thread()
      File "threading.py", line 203, in get_msg_from_thread
        raise ExecutionFailure(
    ExecutionFailure: Reporting failure from other thread/process
"""

from queue import Queue
from threading import Thread
from typing import Any, Callable, Optional

import torch.multiprocessing as mp


class EOFSignal:
    """
    This class object (not instance) is returned by the thread/process as an
    indicator that someone exited or that you should exit.
    """

    pass


class ExecutionFailure(Exception):
    """
    Exception class raised in the main thread when the function running in the
    thread/process raises an exception.

    Get the original exception using the `__cause__` property of this
    exception.
    """

    pass


class ExceptionWithQueueMixIn:
    """
    A mix-in that can be used with a secondary thread or sub-process to
    facilitate communication with them and the passing back of exceptions and
    signals such as when the thread/sub-process exits or when the main
    thread/process want the sub-process/thread to end.

    Communication happens bi-directionally via queues. These queues are not
    limited in buffer size, so if there's potential for the queue to be backed
    up because the main thread or sub-thread/process is unable to read and
    process messages quickly enough, you must implement some kind of
    back-pressure to prevent the queue from growing unlimitedly in size.
    E.g. using a limited number of tokens that must be held to send a message
    to prevent a sender from thrashing the queue. And these tokens can be
    passed between main thread to sub-thread/process etc on each message and
    when the message is done processing.

    The main process uses `get_msg_from_thread` to raise exceptions in the
    main thread that occurred in the sub-thread/process. So that must be
    called in the main thread if you want to know if the sub-thread/process
    exited.
    """

    to_thread_queue: Queue
    """
    Internal queue used to send messages to the thread from the main thread.
    """

    from_thread_queue: Queue
    """
    Internal queue used to send messages from the thread to the main thread.
    """

    args: tuple = ()
    """
    The args provided by the caller that will be passed to the function running
    in the thread/process when it starts.
    """

    # user_func_runner must end with an eof msg to the main thread. This tracks
    # whether the main thread saw the eof. If it didn't, we know there are more
    # msgs in the queue waiting to be read
    _saw_eof: bool = False

    def __init__(self, target: Callable, pass_self: bool = False):
        self.user_func = target
        self.pass_self = pass_self

    def user_func_runner(self) -> None:
        """
        The internal function that runs the target function provided by the
        user.
        """
        try:
            if self.user_func is not None:
                if self.pass_self:
                    self.user_func(self, *self.args)
                else:
                    self.user_func(*self.args)
        except BaseException as e:
            self.from_thread_queue.put(
                ("exception", e), block=True, timeout=None
            )
        finally:
            self.from_thread_queue.put(("eof", None), block=True, timeout=None)

    def send_msg_to_mainthread(self, value: Any) -> None:
        """
        Sends the `value` to the main thread, from the sub-thread/process.

        The value must be pickleable if it's running in a sub-process. The main
        thread then can read it using `get_msg_from_thread`.

        The queue is not limited in size, so if there's potential for the
        queue to be backed up because the main thread is unable to read quickly
        enough, you must implement some kind of back-pressure to prevent
        the queue from growing unlimitedly in size.
        """
        self.from_thread_queue.put(("user", value), block=True, timeout=None)

    def clear_remaining(self) -> None:
        """
        Celled in the main-thread as part of cleanup when we expect the
        secondary thread to have exited (e.g. we sent it a message telling it
        to).

        It will drop any waiting messages sent by the secondary thread, but
        more importantly, it will handle exceptions raised in the secondary
        thread before it exited, that may not have yet been processed in the
        main thread (e.g. we stopped listening to messages from the secondary
        thread before we got an eof from it).
        """
        while not self._saw_eof:
            self.get_msg_from_thread()

    def get_msg_from_thread(self, timeout: Optional[float] = None) -> Any:
        """
        Gets a message from the sub-thread/process sent to the main thread.

        This blocks forever until the message is sent by the sub-thread/process
        and received by us. If `timeout` is not None, that's how long we block
        here, in seconds, before raising an `Empty` Exception if no message was
        received by then.

        If the return value is the `EOFSignal` object, it means the
        sub-thread/process has or is about to exit.

        If the sub-thread/process has raised an exception, that exception is
        caught and re-raised in the main thread when this method is called.
        The exception raised is an `ExecutionFailure` and its `__cause__`
        property is the original exception raised in the sub-thread/process.

        A typical pattern is::

            >>> try:
            ...     msg = thread.get_msg_from_thread()
            ...     if msg == EOFSignal:
            ...         # thread exited
            ...         pass
            ...     else:
            ...         # do something with the msg
            ...         pass
            ... except ExecutionFailure as e:
            ...     print(f"got exception {type(e.__cause__)}")
            ...     print(f"with message {e.__cause__.args[0]}")
        """
        msg, value = self.from_thread_queue.get(block=True, timeout=timeout)
        if msg == "eof":
            self._saw_eof = True
            return EOFSignal
        if msg == "exception":
            raise ExecutionFailure(
                "Reporting failure from other thread/process"
            ) from value

        return value

    def send_msg_to_thread(self, value: Any) -> None:
        """
        Sends the `value` to the sub-thread/process, from the main thread.

        The value must be pickleable if it's sent to a sub-process. The thread
        then can read it using `get_msg_from_mainthread`.

        The queue is not limited in size, so if there's potential for the
        queue to be backed up because the thread is unable to read quickly
        enough, you must implement some kind of back-pressure to prevent
        the queue from growing unlimitedly in size.
        """
        self.to_thread_queue.put(("user", value), block=True, timeout=None)

    def notify_to_end_thread(self) -> None:
        """
        Sends a message to the sub-process/thread that the main process wants
        them to end. The sub-process/thread sees it by receiving an `EOFSignal`
        message from `get_msg_from_mainthread` and it should exit asap.
        """
        self.to_thread_queue.put(("eof", None), block=True, timeout=None)

    def get_msg_from_mainthread(self, timeout: Optional[float] = None) -> Any:
        """
        Gets a message from the main thread sent to the sub-thread/process.

        This blocks forever until the message is sent by the main thread
        and received by us. If `timeout` is not None, that's how long we block
        here, in seconds, before raising an `Empty` Exception if no message was
        received by then.

        If the return value is the `EOFSignal` object, it means the
        main thread has sent us an EOF message using `notify_to_end_thread`
        because it wants us to exit.

        A typical pattern is::

            >>> msg = thread.get_msg_from_mainthread()
            ... if msg == EOFSignal:
            ...     # we should exit asap
            ...     return
            ... # do something with the msg
        """
        msg, value = self.to_thread_queue.get(block=True, timeout=timeout)
        if msg == "eof":
            return EOFSignal

        return value


class ThreadWithException(ExceptionWithQueueMixIn):
    """
    Runs a target function in a secondary thread.
    """

    thread: Thread = None
    """The thread running the function."""

    def __init__(self, target, args=(), **kwargs):
        super().__init__(target=target, **kwargs)
        self.to_thread_queue = Queue(maxsize=0)
        self.from_thread_queue = Queue(maxsize=0)
        self.args = args
        self.thread = Thread(target=self.user_func_runner)

    def start(self) -> None:
        """Starts the thread that runs the target function."""
        self.thread.start()

    def join(self, timeout: Optional[float] = None) -> None:
        """
        Waits and blocks until the thread exits. If timeout is given,
        it's the duration to wait, in seconds, before returning.

        To know if it exited, you need to check `is_alive` of the `thread`.
        """
        self.thread.join(timeout=timeout)


class ProcessWithException(ExceptionWithQueueMixIn):
    """
    Runs a target function in a sub-process.

    Any data sent between the processes must be pickleable.

    We run the function using `torch.multiprocessing`. Any tensors sent between
    the main process and sub-process is memory mapped so that it doesn't copy
    the tensor. So any edits in the main process/sub-process is seen in the
    other as well. See https://pytorch.org/docs/stable/multiprocessing.html
    for more details on this.
    """

    process: mp.Process = None
    """The sub-process running the function."""

    def __init__(self, target, args=(), **kwargs):
        super().__init__(target=target, **kwargs)
        ctx = mp.get_context("spawn")
        self.to_thread_queue = ctx.Queue(maxsize=0)
        self.from_thread_queue = ctx.Queue(maxsize=0)
        self.process = ctx.Process(target=self.user_func_runner)

        self.args = args

    def start(self) -> None:
        """Starts the sub-process that runs the target function."""
        self.process.start()

    def join(self, timeout: Optional[float] = None) -> None:
        """
        Waits and blocks until the process exits. If timeout is given,
        it's the duration to wait, in seconds, before returning.

        To know if it exited, you need to check `is_alive` of the `process`.
        """
        self.process.join(timeout=timeout)
