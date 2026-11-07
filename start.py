"""Small process supervisor: keeps the prediction engine separate from Gunicorn."""
import os
import signal
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
STOP = False


def request_stop(signum, frame):
    global STOP
    STOP = True
    print(f"[SUPERVISOR] Received signal {signum}; shutting down", flush=True)


signal.signal(signal.SIGTERM, request_stop)
signal.signal(signal.SIGINT, request_stop)

port = os.getenv("PORT", "10000")
worker = None
web = None
next_worker_start = 0

try:
    worker = subprocess.Popen([sys.executable, "-u", "worker.py"], cwd=ROOT)
    web = subprocess.Popen([
        sys.executable, "-m", "gunicorn", "main:app",
        "--bind", f"0.0.0.0:{port}",
        "--workers", "1", "--threads", "2", "--timeout", "120",
        "--access-logfile", "-", "--error-logfile", "-",
    ], cwd=ROOT)
    print(f"[SUPERVISOR] Started worker pid={worker.pid} and web pid={web.pid} on port {port}", flush=True)

    while not STOP:
        web_code = web.poll()
        if web_code is not None:
            print(f"[SUPERVISOR] Gunicorn exited with code {web_code}; stopping service", flush=True)
            break

        worker_code = worker.poll()
        if worker_code is not None and time.time() >= next_worker_start:
            print(f"[SUPERVISOR] Engine exited with code {worker_code}; restarting in 5 seconds", flush=True)
            time.sleep(5)
            if not STOP:
                worker = subprocess.Popen([sys.executable, "-u", "worker.py"], cwd=ROOT)
                print(f"[SUPERVISOR] Restarted engine pid={worker.pid}", flush=True)
                next_worker_start = time.time() + 5
        time.sleep(1)
finally:
    for process in (web, worker):
        if process is not None and process.poll() is None:
            process.terminate()
    deadline = time.time() + 20
    for process in (web, worker):
        if process is not None and process.poll() is None:
            try:
                process.wait(timeout=max(1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                process.kill()
    print("[SUPERVISOR] Shutdown complete", flush=True)

if web is not None and web.returncode not in (0, None):
    sys.exit(web.returncode)
