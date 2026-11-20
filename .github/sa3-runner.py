import hashlib
import io
import json
import math
import platform
import socket
import statistics
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import psutil

root = Path.cwd()
models = root / "models"
results = root / "results"
results.mkdir(exist_ok=True)
assets = json.loads((root / ".github/sa3-qa-assets.json").read_text(encoding="utf-8-sig"))
query = json.loads((root / ".github/sa3-qa-request.json").read_text(encoding="utf-8-sig"))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


if sys.argv[1] == "download":
    for asset in assets:
        target = models / asset["dest"]
        target.parent.mkdir(exist_ok=True, parents=True)
        url = f"https://huggingface.co/{asset['repo']}/resolve/{asset['revision']}/{asset['file']}"
        subprocess.run(["curl.exe", "-fsSL", "--retry", "3", "-o", str(target), url], check=True)
        assert target.stat().st_size == asset["bytes"] and digest(target) == asset["sha256"], target
    sys.exit(0)

revision = sys.argv[1]
build = root / ("build-" + revision)
exe, = build.rglob("ace-server.exe")
with socket.socket() as available:
    available.bind(("127.0.0.1", 0))
    port = available.getsockname()[1]
api = f"http://127.0.0.1:{port}"
rows = []
report = dict(revision=subprocess.check_output(["git", "-C", "source-" + revision, "rev-parse", "HEAD"], text=True).strip(),
              binary_sha256=digest(exe), os=platform.platform(), cpu=platform.processor(),
              host_ram_bytes=psutil.virtual_memory().total, gpu_backend=False, input=query,
              sample_interval_seconds=.2, weights=assets, runs=rows)
with (results / f"{revision}.log").open("wb") as log:
    started = time.monotonic()
    process = subprocess.Popen([exe, "--models", models, "--host", "127.0.0.1", "--port", str(port)],
                               cwd=exe.parent, stdout=log, stderr=log)
    monitored = psutil.Process(process.pid)
    samples = []
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            try:
                rss = sum(p.memory_info().rss for p in [monitored, *monitored.children(recursive=True)])
                samples.append((time.monotonic(), rss))
            except psutil.NoSuchProcess:
                if process.poll() is not None:
                    return
            stop.wait(.2)

    worker = threading.Thread(target=sample)
    worker.start()
    try:
        for attempt in range(240):
            if process.poll() is not None:
                raise RuntimeError(f"Server exited with {process.returncode}")
            try:
                with urlopen(api + "/sa3/props", timeout=5) as res:
                    props = json.load(res)
                break
            except OSError:
                time.sleep(.25)
        else:
            raise TimeoutError("SA3 discovery")
        assert props["synth_ready"], props
        report["properties"] = props
        for iteration in range(4):
            request = Request(api + "/sa3-synth?" + urlencode(query), data=b"")
            before = time.monotonic()
            with urlopen(request, timeout=1200) as response:
                body = response.read()
            end = time.monotonic()
            with wave.open(io.BytesIO(body)) as wav:
                assert (wav.getnchannels(), wav.getframerate(), wav.getsampwidth()) == (2, 44100, 2)
                duration = wav.getnframes()/wav.getframerate()
                samples16 = np.frombuffer(wav.readframes(wav.getnframes()), "<i2").astype(float)
            rms = math.sqrt(float(np.mean(samples16**2)))
            assert abs(duration - query["duration"]) < .1 and rms > 1, (duration, rms)
            window_start = before if iteration else started
            rows.append(dict(round=iteration, wall_seconds=end-before, audio_seconds=duration, rms_pcm16=rms,
                             pcm_sha256=hashlib.sha256(samples16.astype("<i2").tobytes()).hexdigest(),
                             peak_tree_rss=max(rss for t,rss in samples if window_start <= t <= end)))
            if not iteration:
                report["start_to_first_wav_seconds"] = end-started
                (results / f"{revision}.wav").write_bytes(body)
            print(revision, rows[-1], flush=True)
        report["warm_median_seconds"] = statistics.median(row["wall_seconds"] for row in rows[1:])
        with urlopen(api + "/health", timeout=5) as response:
            assert response.status == 200
        (results / f"{revision}.json").write_text(json.dumps(report, indent=2))
    finally:
        stop.set()
        worker.join()
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=30)
