from threading import Event, get_ident
import time
import unittest

from ui.preview_jobs import LatestPreviewJob


class PreviewJobTests(unittest.TestCase):
    def wait_for(self, runner, condition):
        deadline = time.monotonic() + 3
        while not condition() and time.monotonic() < deadline:
            runner.poll()
            time.sleep(0.005)
        self.assertTrue(condition(), 'Preview did not finish')

    def test_newest_pending_request_wins_and_callbacks_run_on_owner(self):
        runner = LatestPreviewJob()
        release, started = Event(), Event()
        self.addCleanup(release.set)
        executed, displayed, callback_threads = [], [], []
        owner = get_ident()

        def slow():
            started.set()
            release.wait(3)
            return 'old'

        def display(value):
            displayed.append(value)
            callback_threads.append(get_ident())

        runner.submit(slow, display, self.fail)
        self.assertTrue(started.wait(3))
        runner.submit(lambda: executed.append('intermediate'), display, self.fail)
        runner.submit(lambda: 'newest', display, self.fail)
        release.set()
        self.wait_for(runner, lambda: bool(displayed))
        self.assertEqual(displayed, ['newest'])
        self.assertEqual(executed, [])
        self.assertEqual(callback_threads, [owner])

    def test_close_discards_completed_and_pending_work(self):
        runner = LatestPreviewJob()
        release = Event()
        self.addCleanup(release.set)
        displayed = []
        runner.submit(lambda: release.wait(3), displayed.append, displayed.append)
        runner.submit(lambda: 'pending', displayed.append, displayed.append)
        runner.close()
        release.set()
        self.wait_for(runner, lambda: not runner._running)
        self.assertEqual(displayed, [])

    def test_cached_selection_invalidates_inflight_result(self):
        runner = LatestPreviewJob()
        displayed = []
        runner.submit(lambda: 'stale', displayed.append, displayed.append)
        runner.invalidate()
        self.wait_for(runner, lambda: not runner._running)
        self.assertEqual(displayed, [])

    def test_worker_error_is_delivered_and_next_request_can_run(self):
        runner = LatestPreviewJob()
        errors, displayed = [], []

        def fail():
            raise ValueError('bad preview')

        runner.submit(fail, displayed.append, errors.append)
        self.wait_for(runner, lambda: bool(errors))
        self.assertIsInstance(errors[0], ValueError)
        runner.submit(lambda: 'recovered', displayed.append, errors.append)
        self.wait_for(runner, lambda: bool(displayed))
        self.assertEqual(displayed, ['recovered'])
