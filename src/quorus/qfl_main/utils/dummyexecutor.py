"""### DummyExecutor Class"""
from concurrent.futures import Future, Executor
from threading import Lock
# copied from https://stackoverflow.com/questions/10434593/dummyexecutor-for-pythons-futures
class DummyExecutor(Executor):

    def __init__(self, max_workers=0):
        self._shutdown = False
        self._shutdownLock = Lock()
        self.max_workers = max_workers

    def submit(self, fn, *args, **kwargs):
        with self._shutdownLock:
            if self._shutdown:
                raise RuntimeError('cannot schedule new futures after shutdown')

            f = Future()
            try:
                result = fn(*args, **kwargs)
            except BaseException as e:
                f.set_exception(e)
            else:
                f.set_result(result)

            return f

    def shutdown(self, wait=True):
        with self._shutdownLock:
            self._shutdown = True