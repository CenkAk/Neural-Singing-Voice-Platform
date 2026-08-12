from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nsvp.contracts import JobState
from nsvp.jobs import JobStore, Worker
from nsvp.storage import LocalArtifactStore


class JobsStorageTests(unittest.TestCase):
    def test_job_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = JobStore(Path(directory) / "jobs.sqlite3")
            job = store.enqueue("add", {"left": 2, "right": 3})
            worker = Worker(store, {"add": lambda payload, progress: {"value": payload["left"] + payload["right"]}})
            self.assertTrue(worker.run_once())
            completed = store.get(job.id)
            self.assertEqual(completed.state, JobState.SUCCEEDED)
            self.assertEqual(store.result(job.id), {"value": 5})

    def test_queued_job_cancellation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = JobStore(Path(directory) / "jobs.sqlite3")
            job = store.enqueue("noop", {})
            self.assertEqual(store.cancel(job.id).state, JobState.CANCELLED)

    def test_artifact_path_escape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LocalArtifactStore(Path(directory) / "artifacts")
            with self.assertRaises(ValueError):
                store.resolve("../secret")


if __name__ == "__main__":
    unittest.main()

