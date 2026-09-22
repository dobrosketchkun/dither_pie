"""One running preview and one replaceable pending request.

Submit, poll, and close on the UI thread. Worker threads only compute and put
results in a queue; they never touch Tk variables, widgets, or callbacks.
"""

from queue import Empty, SimpleQueue
from threading import Thread


class LatestPreviewJob:
    def __init__(self):
        self._results = SimpleQueue()
        self._generation = 0
        self._running = False
        self._pending = None
        self._closed = False

    def submit(self, compute, on_result, on_error):
        if self._closed:
            return
        self._generation += 1
        self._pending = (self._generation, compute, on_result, on_error)
        self._start_pending()

    def invalidate(self):
        """Discard pending/stale work, for example when using a cached result."""
        self._generation += 1
        self._pending = None

    def _start_pending(self):
        if self._running or self._pending is None or self._closed:
            return
        job = self._pending
        self._pending = None
        self._running = True

        def run():
            generation, compute, on_result, on_error = job
            try:
                result = compute()
                error = None
            except Exception as exc:
                result, error = None, exc
            self._results.put((generation, result, error, on_result, on_error))

        Thread(target=run, daemon=True).start()

    def poll(self):
        """Deliver only current results, and start the newest pending request."""
        try:
            generation, result, error, on_result, on_error = self._results.get_nowait()
        except Empty:
            return
        self._running = False
        try:
            if not self._closed and generation == self._generation:
                if error is None:
                    on_result(result)
                else:
                    on_error(error)
        finally:
            self._start_pending()

    def close(self):
        self._closed = True
        self.invalidate()
