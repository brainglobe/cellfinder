import pytest

from cellfinder.core.tools.threading import (
    EOFSignal,
    ExceptionWithQueueMixIn,
    ExecutionFailure,
    ProcessWithException,
    ThreadWithException,
)

cls_to_test = [ThreadWithException, ProcessWithException]


class ExceptionTest(Exception):
    pass


def raise_exc(*args):
    raise ExceptionTest("I'm a test")


def raise_exc_about_self(*args):
    arg_msg = "No arg"
    if args and args[-1] == "7":
        arg_msg = "Got 7"

    if args and isinstance(args[0], ExceptionWithQueueMixIn):
        raise ExceptionTest("Got self" + arg_msg)
    raise ExceptionTest("No self" + arg_msg)


def do_nothing(*args):
    pass


def send_back_msg(thread: ExceptionWithQueueMixIn):
    # do this single op and exit
    thread.send_msg_to_mainthread(("back", thread.get_msg_from_mainthread()))


def send_multiple_msgs(thread: ExceptionWithQueueMixIn):
    thread.send_msg_to_mainthread("hello")
    thread.send_msg_to_mainthread("to")
    thread.send_msg_to_mainthread("you")


@pytest.mark.parametrize("cls", cls_to_test)
def test_reraise_exception_in_main_thread_from_thread(cls):
    # exception in thread will show up in main thread
    thread = cls(target=raise_exc, args=(1, "4"))
    thread.start()

    with pytest.raises(ExecutionFailure) as exc_info:
        thread.get_msg_from_thread()
    assert type(exc_info.value.__cause__) is ExceptionTest
    assert exc_info.value.__cause__.args[0] == "I'm a test"
    # thread will have exited
    thread.join()


@pytest.mark.parametrize("cls", cls_to_test)
def test_get_eof_in_main_thread_from_thread(cls):
    # we should get eof when thread exits
    thread = cls(target=do_nothing, args=(1, "4"))
    thread.start()

    assert thread.get_msg_from_thread() is EOFSignal
    # thread will have exited
    thread.join()


@pytest.mark.parametrize("cls", cls_to_test)
def test_get_eof_in_thread_from_main_thread(cls):
    # thread should get eof when we send it
    thread = cls(target=send_back_msg, pass_self=True)
    thread.start()

    thread.notify_to_end_thread()
    assert thread.get_msg_from_thread() == ("back", EOFSignal)
    # thread will have exited
    thread.join()


@pytest.mark.parametrize("args", [(), (55, "7")])
@pytest.mark.parametrize("pass_self", [(True, "Got self"), (False, "No self")])
@pytest.mark.parametrize("cls", cls_to_test)
def test_pass_self_arg_to_func(cls, pass_self, args):
    # check that passing self to the function works with/without args
    # without passing self, there's no way for the thread to respond
    # other than by the text of an error
    thread = cls(
        target=raise_exc_about_self, pass_self=pass_self[0], args=args
    )
    thread.start()

    arg_msg = "No arg"
    if args:
        arg_msg = "Got 7"

    with pytest.raises(ExecutionFailure) as exc_info:
        thread.get_msg_from_thread()
    assert type(exc_info.value.__cause__) is ExceptionTest
    assert exc_info.value.__cause__.args[0] == pass_self[1] + arg_msg
    thread.join()


@pytest.mark.parametrize("cls", cls_to_test)
def test_send_to_and_recv_from_thread(cls):
    # tests sending to the thread and receiving a message from the thread
    thread = cls(target=send_back_msg, pass_self=True)
    thread.start()

    msg = thread.get_msg_from_thread(thread.send_msg_to_thread("hello"))
    assert msg == ("back", "hello")
    thread.join()


@pytest.mark.parametrize("cls", cls_to_test)
def test_get_multiple_messages(cls):
    # tests getting multiple msgs from thread
    thread = cls(target=send_multiple_msgs, pass_self=True)
    thread.start()

    assert thread.get_msg_from_thread() == "hello"
    assert thread.get_msg_from_thread() == "to"
    assert thread.get_msg_from_thread() == "you"
    assert thread.get_msg_from_thread() == EOFSignal
    thread.join()


@pytest.mark.parametrize("cls", cls_to_test)
def test_skip_until_eof(cls):
    # tests skipping reading everything queued until we get eof
    thread = cls(target=send_multiple_msgs, pass_self=True)
    thread.start()

    thread.clear_remaining()
    # it knows that there are no further messages because the thread sent an
    # eof to main-thread, which is the last thing thread does before exiting
    assert thread._saw_eof
    # there shouldn't be more messages
    assert not thread.from_thread_queue.qsize()
    thread.join()
