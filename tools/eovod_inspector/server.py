"""EOVOD frame inspector: a small web server for ``dump.py``'s per-frame dumps.

Runs on an Isambard-AI compute node (``tools/isambard/inspect.sbatch``) and is
reached from a laptop through an SSH tunnel via the login node. It serves the
page beside this file, the dumps under ``--root`` and a JSON API; new dumps
run on this node's GPU, one at a time, as ``dump.py`` subprocesses.

    GET  /api/runs          summaries of ROOT/runs/*.json
    GET  /api/checkpoints   *.pth under --work-dirs
    GET  /api/diagnostics   ROOT/runs/diagnostics.json
    GET  /api/speed         ROOT/speed/*.json (tools/eovod_speed.py output)
    GET  /api/dumps         dumps started by this server: state and log tail
    POST /api/dumps         start one: {ckpt, run, note, cfg_options, videos,
                            max_frames, heavy_every}

Every request needs the access token printed at start-up: once as
``?token=...`` (which sets a cookie), then the cookie. Standard library only.
"""

from __future__ import annotations

import argparse
import glob
import http.cookies
import json
import mimetypes
import os
import os.path as osp
import queue
import re
import secrets
import socket
import subprocess
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = osp.dirname(osp.abspath(__file__))
REPO = osp.dirname(osp.dirname(HERE))
COOKIE = "eovod_inspector"
RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
CFG_OPTION = re.compile(r"^[A-Za-z0-9_.]+=[A-Za-z0-9_.,:+\-\[\]() '\"]*$")
VIDEOS = re.compile(r"^\d+(,\d+)*$")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True, help="dump directory (runs/, img/)")
    ap.add_argument("--work-dirs", required=True, help="where training checkpoints live")
    ap.add_argument("--config", default="configs/vid/eovod/eovod_fcos_r101_fpn_3x.py")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765, help="first port to try")
    ap.add_argument("--token", help="default: a random one")
    return ap.parse_args()


class Dumps:
    """A queue of ``dump.py`` runs on this node's GPU, one at a time."""

    def __init__(self, root: str, work_dirs: str, config: str):
        self.root, self.work_dirs, self.config = root, work_dirs, config
        self.jobs: list[dict] = []
        self.lock = threading.Lock()
        self.pending: queue.Queue[dict] = queue.Queue()
        threading.Thread(target=self._worker, daemon=True).start()

    def submit(self, req: dict) -> dict:
        run = str(req.get("run", ""))
        ckpt = str(req.get("ckpt", ""))
        cfg_options = [str(o) for o in req.get("cfg_options", [])]
        videos = str(req.get("videos", "0,45,120,230,340,480")).replace(" ", "")
        if not RUN_NAME.match(run):
            raise ValueError("run: letters, digits, '_', '.', '-'; at most 64 characters")
        real = osp.realpath(ckpt)
        if not (real.endswith(".pth") and osp.isfile(real)
                and real.startswith(osp.realpath(self.work_dirs) + os.sep)):
            raise ValueError(f"no checkpoint at {ckpt!r} under {self.work_dirs}")
        bad = [o for o in cfg_options if not CFG_OPTION.match(o)]
        if bad:
            raise ValueError(f"not KEY=VALUE: {bad}")
        if not VIDEOS.match(videos):
            raise ValueError("videos: comma-separated val video indices")
        max_frames = int(req.get("max_frames", 150))
        heavy_every = int(req.get("heavy_every", 15))
        if not (1 <= max_frames <= 2000 and 1 <= heavy_every <= max_frames):
            raise ValueError("max_frames in [1, 2000] and heavy_every in [1, max_frames]")
        with self.lock:
            if any(j["run"] == run and j["state"] in ("queued", "running") for j in self.jobs):
                raise ValueError(f"{run!r} is already queued")
            job = dict(id=len(self.jobs) + 1, run=run, ckpt=ckpt, cfg_options=cfg_options,
                       videos=videos, state="queued", created=time.time(),
                       log=osp.join(self.root, "logs", f"{run}.log"))
            argv = [sys.executable, osp.join(HERE, "dump.py"), "--config", self.config,
                    "--ckpt", ckpt, "--run", run, "--note", str(req.get("note", ""))[:200],
                    "--out", self.root, "--videos", videos, "--max-frames", str(max_frames),
                    "--heavy-every", str(heavy_every)]
            if cfg_options:
                argv += ["--cfg-options", *cfg_options]
            job["argv"] = argv
            self.jobs.append(job)
        self.pending.put(job)
        return self._public(job)

    def _worker(self) -> None:
        while True:
            job = self.pending.get()
            os.makedirs(osp.dirname(job["log"]), exist_ok=True)
            with self.lock:
                job.update(state="running", started=time.time())
            with open(job["log"], "w") as log:
                proc = subprocess.run(job["argv"], cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            with self.lock:
                job.update(state="done" if proc.returncode == 0 else "failed",
                           ended=time.time(), returncode=proc.returncode)

    @staticmethod
    def _tail(path: str, n: int = 12) -> list[str]:
        try:
            with open(path, errors="replace") as f:
                lines = f.read().replace("\r", "\n").splitlines()
        except OSError:
            return []
        return [ln for ln in lines if ln.strip()][-n:]

    def _public(self, job: dict) -> dict:
        out = {k: v for k, v in job.items() if k not in ("argv", "log")}
        out["tail"] = self._tail(job["log"]) if job["state"] != "queued" else []
        return out

    def list(self) -> list[dict]:
        with self.lock:
            return [self._public(j) for j in reversed(self.jobs)]


class RunIndex:
    """Summaries of the dumps, re-read only when a file changes."""

    def __init__(self, root: str):
        self.root = root
        self.cache: dict[str, tuple[float, dict]] = {}

    def list(self) -> list[dict]:
        out = []
        for path in glob.glob(osp.join(self.root, "runs", "*.json")):
            if osp.basename(path) in ("index.json", "diagnostics.json"):
                continue
            mtime = osp.getmtime(path)
            hit = self.cache.get(path)
            if hit is None or hit[0] != mtime:
                try:
                    with open(path) as f:
                        run = json.load(f)
                except (OSError, json.JSONDecodeError):
                    continue  # being written
                hit = (mtime, dict(file=f"runs/{osp.basename(path)}", run=run["run"],
                                   note=run.get("note", ""), summary=run.get("summary", {}),
                                   created=run.get("created", ""),
                                   cfg_options=run.get("cfg_options", []),
                                   ckpt=run.get("ckpt", "")))
                self.cache[path] = hit
            out.append(dict(hit[1], mtime=mtime))
        return sorted(out, key=lambda r: -r["mtime"])


def make_handler(args, token: str, runs: RunIndex, dumps: Dumps):
    static = {"/": osp.join(HERE, "index.html"), "/index.html": osp.join(HERE, "index.html")}

    class Handler(BaseHTTPRequestHandler):
        server_version = "EOVODInspector/1"

        def log_message(self, fmt, *a):  # one line per API call, none per image
            if self.path.startswith("/api/"):
                sys.stderr.write(f"{time.strftime('%H:%M:%S')} {self.command} {self.path}\n")

        def _authorised(self, query: dict) -> bool:
            if secrets.compare_digest(query.get("token", [""])[0], token):
                return True
            jar = http.cookies.SimpleCookie(self.headers.get("Cookie", ""))
            return COOKIE in jar and secrets.compare_digest(jar[COOKIE].value, token)

        def _send(self, status, body: bytes, ctype: str, extra: dict | None = None):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store" if ctype.startswith(
                ("application/json", "text/html")) else "max-age=3600")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, status=HTTPStatus.OK):
            self._send(status, json.dumps(obj).encode(), "application/json; charset=utf-8")

        def _file(self, path: str, extra: dict | None = None):
            try:
                with open(path, "rb") as f:
                    body = f.read()
            except OSError:
                return self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
            if ctype.startswith("text/") or ctype == "application/json":
                ctype += "; charset=utf-8"
            self._send(HTTPStatus.OK, body, ctype, extra)

        def _gate(self):
            url = urlparse(self.path)
            query = parse_qs(url.query)
            if not self._authorised(query):
                self._send(HTTPStatus.FORBIDDEN,
                           b"Open the link printed by the server, with its ?token=...\n",
                           "text/plain; charset=utf-8")
                return None, None
            return url, query

        def do_GET(self):
            url, query = self._gate()
            if url is None:
                return
            path = url.path
            if path in static:
                cookie = f"{COOKIE}={token}; Path=/; HttpOnly; SameSite=Strict"
                return self._file(static[path], {"Set-Cookie": cookie})
            if path == "/api/runs":
                return self._json(runs.list())
            if path == "/api/dumps":
                return self._json(dumps.list())
            if path == "/api/diagnostics":
                return self._file(osp.join(args.root, "runs", "diagnostics.json"))
            if path == "/api/speed":
                out = []
                for f in sorted(glob.glob(osp.join(args.root, "speed", "*.json"))):
                    try:
                        with open(f) as fh:
                            out.append(dict(json.load(fh), file=osp.basename(f)))
                    except (OSError, json.JSONDecodeError):
                        continue
                return self._json(out)
            if path == "/api/checkpoints":
                ckpts = sorted(glob.glob(osp.join(args.work_dirs, "*", "*.pth")),
                               key=osp.getmtime, reverse=True)
                return self._json([dict(path=c, name=osp.relpath(c, args.work_dirs),
                                        mtime=osp.getmtime(c)) for c in ckpts
                                   if not osp.islink(c)])
            if path.startswith(("/runs/", "/img/")):
                full = osp.realpath(osp.join(args.root, path.lstrip("/")))
                if not full.startswith(osp.realpath(args.root) + os.sep):
                    return self._json({"error": "outside the dump directory"},
                                      HTTPStatus.FORBIDDEN)
                return self._file(full)
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self):
            url, _ = self._gate()
            if url is None:
                return
            if url.path != "/api/dumps":
                return self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                req = json.loads(self.rfile.read(min(length, 1 << 16)) or b"{}")
                return self._json(dumps.submit(req), HTTPStatus.CREATED)
            except (ValueError, TypeError) as e:
                return self._json({"error": str(e)}, HTTPStatus.BAD_REQUEST)

    return Handler


def main():
    args = parse_args()
    args.root, args.work_dirs = osp.abspath(args.root), osp.abspath(args.work_dirs)
    for sub in ("runs", "img", "logs", "speed"):
        os.makedirs(osp.join(args.root, sub), exist_ok=True)
    token = args.token or secrets.token_urlsafe(18)
    handler = make_handler(args, token, RunIndex(args.root),
                           Dumps(args.root, args.work_dirs, args.config))
    server = None
    for port in range(args.port, args.port + 20):
        try:
            server = ThreadingHTTPServer((args.host, port), handler)
            break
        except OSError:
            continue
    if server is None:
        raise SystemExit(f"no free port in {args.port}..{args.port + 19}")
    host = socket.gethostname().split(".")[0]
    print(f"EOVOD inspector on {host}:{port}, serving {args.root}\n"
          f"On your laptop:\n"
          f"    ssh -N -L {port}:{host}:{port} b5cs.aip2.isambard\n"
          f"then open:\n"
          f"    http://localhost:{port}/?token={token}\n", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
