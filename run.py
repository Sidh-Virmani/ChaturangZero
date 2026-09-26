"""Portable local launcher. Python 3.12 recommended; no global package changes.

    python run.py --setup --open     # first run: isolated dependencies + UI build
    python run.py --open             # subsequent runs; serves UI and API together
    python run.py --check            # verify installation without starting a server
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request
import webbrowser

ROOT = Path(__file__).resolve().parent
UI = ROOT / 'shared' / 'ui'
VENV = ROOT / '.venv'
ENV_PYTHON = VENV / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def command(args, cwd=ROOT):
    subprocess.run([str(arg) for arg in args], cwd=cwd, check=True)


def ensure_environment(setup):
    if sys.version_info < (3, 11):
        raise RuntimeError('Python 3.11 or newer is required; Python 3.12 is recommended.')
    if not ENV_PYTHON.exists() and setup:
        print('Creating the project Python environment...', flush=True)
        command([sys.executable, '-m', 'venv', VENV])
    if ENV_PYTHON.exists() and Path(sys.prefix).resolve() != VENV.resolve():
        raise SystemExit(subprocess.call([str(ENV_PYTHON), str(ROOT / 'run.py'), *sys.argv[1:]], cwd=ROOT))
    missing = [name for name in ('torch', 'numpy') if importlib.util.find_spec(name) is None]
    if missing:
        if not setup:
            raise RuntimeError('AI dependencies are missing. Run: python run.py --setup')
        print('Installing CPU AI dependencies inside .venv...', flush=True)
        args = [sys.executable, '-m', 'pip', 'install', '-r', ROOT / 'requirements.txt']
        if sys.platform in ('win32', 'linux'):
            args += ['--index-url', 'https://download.pytorch.org/whl/cpu']
        command(args)


def ensure_ui(force=False, setup=False):
    built = UI / 'dist' / 'index.html'
    if built.exists() and not force:
        return
    if not setup and not force:
        raise RuntimeError('The UI has not been built. Run: python run.py --setup')
    npm = shutil.which('npm.cmd' if os.name == 'nt' else 'npm')
    if not npm:
        raise RuntimeError('Install Node.js 22.12+ (or 24+) to build the interface, then rerun --setup.')
    command([npm, 'ci'], UI)
    command([npm, 'run', 'build'], UI)


def main():
    parser = argparse.ArgumentParser(description='ChaturangZero: local AI and 3D arena')
    parser.add_argument('--setup', action='store_true', help='Install missing project dependencies and build the UI')
    parser.add_argument('--build', action='store_true', help='Rebuild the UI from source')
    parser.add_argument('--open', action='store_true', help='Open the arena in your browser')
    parser.add_argument('--check', action='store_true', help='Check installation, then exit')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('--port must be between 1 and 65535')
    os.chdir(ROOT)
    ensure_environment(args.setup)
    ensure_ui(args.build, args.setup)
    from output.server.api_server import Handler, get_engine
    from http.server import ThreadingHTTPServer
    info = get_engine().info()
    print(f"Model: {info['name']} | {info['parameters']:,} parameters | {'trained' if info['trained'] else 'untrained'}", flush=True)
    if args.check:
        print('READY: Python dependencies, model inference, and built UI are available.')
        return
    url = f'http://127.0.0.1:{args.port}'
    try:
        server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    except OSError:
        try:
            with urllib.request.urlopen(url + '/api/health', timeout=5) as response:
                health = json.load(response)
            if health.get('app') == 'ChaturangZero':
                print(f'ChaturangZero is already running at {url}')
                if args.open:
                    webbrowser.open(url)
                return
        except Exception:
            pass
        raise RuntimeError(f'Port {args.port} is in use. Choose another port: python run.py --port 8001')
    print(f'\nChaturangZero is ready at {url}\nPress Ctrl+C to stop.\n', flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nStopping ChaturangZero.')
    finally:
        server.server_close()


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f'\nCould not start ChaturangZero: {error}', file=sys.stderr)
        raise SystemExit(1)
