import pytest

from grunnur import Context, MultiQueue, Queue


def test_queue(mock_or_real_context: tuple[Context, bool]) -> None:
    context, _mocked = mock_or_real_context
    queue = Queue(context.device)
    assert queue.device == context.devices[0]

    queue.synchronize()


def test_queue_on_multi_device_context(
    mock_or_real_multi_device_context: tuple[Context, bool],
) -> None:
    context, _mocked = mock_or_real_multi_device_context
    queue = Queue(context.devices[1])
    assert queue.device == context.devices[1]


def test_multi_queue(mock_or_real_multi_device_context: tuple[Context, bool]) -> None:
    context, _mocked = mock_or_real_multi_device_context
    mqueue = MultiQueue.on_devices(context.devices)
    assert set(mqueue.queues.keys()) == {context.devices[0], context.devices[1]}
    assert mqueue.devices == context.devices[[0, 1]]

    mqueue.synchronize()

    mqueue = MultiQueue.on_devices(context.devices[[1]])
    assert set(mqueue.queues.keys()) == {context.devices[1]}
    assert mqueue.devices == context.devices[[1]]

    mqueue.synchronize()


def test_multi_queue_out_of_queues(mock_4_device_context: Context) -> None:
    context = mock_4_device_context
    queue0 = Queue(context.devices[0])
    queue1 = Queue(context.devices[1])

    mqueue = MultiQueue([queue0, queue1])
    assert set(mqueue.queues.keys()) == {context.devices[0], context.devices[1]}
    assert mqueue.devices == context.devices[[0, 1]]
