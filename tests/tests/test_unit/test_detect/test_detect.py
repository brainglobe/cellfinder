import multiprocessing

from cellfinder_core.detect.detect import _map_with_locks


def add_one(a: int) -> int:
    return a + 1


def test_map_with_locks():
    args = [1, 2, 3, 2, 10]

    with multiprocessing.Pool(2) as worker_pool:
        result_queue, locks = _map_with_locks(add_one, args, worker_pool)

        async_results = [result_queue.get() for _ in range(len(args))]
        assert len(async_results) == len(locks) == len(args)

        for lock in locks:
            lock.release()

        results = [res.get() for res in async_results]
        assert results == [2, 3, 4, 3, 11]
