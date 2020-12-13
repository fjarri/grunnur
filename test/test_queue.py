import pytest

from grunnur import Context, Queue, MultiQueue


def test_queue(mock_or_real_context):
    context, _mocked = mock_or_real_context
    queue = Queue(context)
    assert queue.device_idx == 0
    assert queue.device == context.devices[0]

    queue.synchronize()


def test_queue_on_multi_device_context(mock_or_real_multi_device_context):
    context, _mocked = mock_or_real_multi_device_context
    queue = Queue(context, device_idx=1)
    assert queue.device_idx == 1
    assert queue.device == context.devices[1]


def test_queue_on_multi_device_context_needs_device_idx(mock_4_device_context):
    context = mock_4_device_context
    with pytest.raises(ValueError, match="For multi-device contexts, device_idx must be set explicitly"):
        queue = Queue(context)


def test_multi_queue(mock_or_real_multi_device_context):
    context, _mocked = mock_or_real_multi_device_context
    mqueue = MultiQueue(context)
    assert set(mqueue.queues.keys()) == {0, 1}
    assert mqueue.devices == {0: context.devices[0], 1: context.devices[1]}
    assert mqueue.device_idxs == {0, 1}

    mqueue.synchronize()

    mqueue = MultiQueue.on_device_idxs(context, device_idxs=[1])
    assert set(mqueue.queues.keys()) == {1}
    assert mqueue.devices == {1: context.devices[1]}
    assert mqueue.device_idxs == {1}

    mqueue.synchronize()


def test_multi_queue_out_of_queues(mock_4_device_context):
    context = mock_4_device_context
    queue0 = Queue(context, device_idx=0)
    queue0_2 = Queue(context, device_idx=0)
    queue1 = Queue(context, device_idx=1)

    with pytest.raises(ValueError, match="Queues in a MultiQueue must belong to distinct devices"):
        MultiQueue(context, [queue0, queue0_2])

    # this works
    mqueue = MultiQueue(context, [queue0, queue1])
    assert set(mqueue.queues.keys()) == {0, 1}
    assert mqueue.devices == {0: context.devices[0], 1: context.devices[1]}
    assert mqueue.device_idxs == {0, 1}
