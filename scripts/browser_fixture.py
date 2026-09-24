"""Serve disposable synthetic conversions and the production UI for browser tests."""
import os
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np

from nsvp.audio.io import save_audio
from nsvp.components import ComponentFactory
from nsvp.config import AppConfig
from nsvp.contracts import AudioBuffer, ConversionRequest
from nsvp.jobs import JobStore, Worker
from nsvp.runtime import build_handlers
from nsvp.testing_backends import IdentityVoiceConverter


def main() -> None:
    import uvicorn
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles

    with tempfile.TemporaryDirectory(prefix="nsvp-browser-test-") as temporary:
        root = Path(temporary)
        os.environ["NSVP_ARTIFACT_ROOT"] = str(root / "artifacts")
        os.environ["NSVP_DATABASE_PATH"] = str(root / "jobs.db")
        from nsvp.api import create_app

        config = AppConfig(artifact_root=root / "artifacts", database_path=root / "jobs.db")
        config.components.voice_converter.provider = "synthetic-a"
        factory = ComponentFactory(config, converters={"synthetic-a": IdentityVoiceConverter, "synthetic-b": IdentityVoiceConverter})
        times = np.arange(48000, dtype=np.float32) / 16000
        audio = AudioBuffer(waveform=(0.2 * np.sin(2 * np.pi * 220 * times))[None, :], sample_rate=16000)
        source = root / "fixture.wav"
        save_audio(source, audio)
        jobs = JobStore(config.database_path)
        worker = Worker(jobs, build_handlers(config, factory))
        for name in ("synthetic-a", "synthetic-b"):
            request = ConversionRequest(song_path=source, target_reference_path=source,
                output_name=name, voice_converter=name, input_kind="vocal")
            jobs.enqueue("conversion", request.model_dump(mode="json"))
            worker.run_once()
        @asynccontextmanager
        async def lifespan(app):
            stop = threading.Event()

            def process_jobs():
                while not stop.is_set():
                    if not worker.run_once():
                        stop.wait(0.05)

            thread = threading.Thread(target=process_jobs, daemon=True)
            thread.start()
            try:
                yield
            finally:
                stop.set()
                thread.join(timeout=10)

        app = FastAPI(lifespan=lifespan)
        @app.get("/__fixture__")
        def fixture_identity() -> dict[str, str]:
            return {"fixture": "nsvp-disposable-browser-v03"}

        app.mount("/api", create_app(config, factory))
        app.mount("/", StaticFiles(directory=Path(__file__).resolve().parents[1] / "apps/web/dist", html=True))
        uvicorn.run(app, host="127.0.0.1", port=8873)


if __name__ == "__main__":
    main()
