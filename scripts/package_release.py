"""Package a reviewable release without local environments or private files.

Run from any directory: python scripts/package_release.py
The release includes source, built UI, model/data, docs, and verification logs.
"""
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
NAME = 'ChaturangZero-v1.0.0'
ROOT_FILES = ['README.md', 'Architecture_plan.md', 'requirements.txt',
              'run.py', 'start-chaturangzero.bat', '.gitignore']
FOLDERS = ['p1_representation', 'p2_neural_network', 'p3_mcts', 'p4_training',
           'integration', 'shared/game_core', 'shared/ui/src', 'shared/ui/public',
           'shared/ui/dist', 'tests', 'docs', 'scripts', 'output/server',
           'output/verification']
EXTRA = ['shared/__init__.py', 'output/README.md', 'shared/ui/package.json',
         'shared/ui/package-lock.json', 'shared/ui/index.html', 'shared/ui/vite.config.ts',
         'shared/ui/tsconfig.json', 'shared/ui/tsconfig.app.json', 'shared/ui/tsconfig.node.json',
         'checkpoints/best.pt', 'checkpoints/candidate.pt', 'data/replay.npz',
         'data/training_metrics.json', 'data/evaluation_report.json']


def main():
    if (ROOT / 'data/training.lock').exists():
        raise SystemExit('Finish the active training job before packaging.')
    report = json.loads((ROOT / 'data/evaluation_report.json').read_text(encoding='utf-8'))
    if not report.get('complete'):
        raise SystemExit('Finish the held-out evaluation before packaging.')
    if report['sha256'] != hashlib.sha256((ROOT / 'checkpoints/best.pt').read_bytes()).hexdigest():
        raise SystemExit('The benchmark does not match best.pt; evaluate the current model first.')
    paths = {ROOT / name for name in ROOT_FILES + EXTRA}
    for folder in FOLDERS:
        paths.update(path for path in (ROOT / folder).rglob('*') if path.is_file()
                     and '__pycache__' not in path.parts and path.suffix not in {'.pyc', '.pyo', '.tsbuildinfo'}
                     and path.name != '.gitkeep')
    assert (ROOT / 'shared/ui/dist/index.html').is_file(), 'Build the frontend first.'
    output = ROOT / 'output/releases'
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f'{NAME}.zip'
    manifest = {}
    with ZipFile(archive, 'w', compression=ZIP_DEFLATED, compresslevel=9) as bundle:
        for path in sorted(paths):
            data = path.read_bytes()
            relative = path.relative_to(ROOT).as_posix()
            manifest[relative] = hashlib.sha256(data).hexdigest()
            bundle.writestr(f'{NAME}/{relative}', data)
        bundle.writestr(f'{NAME}/SHA256SUMS.json', json.dumps(manifest, indent=2))
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix('.sha256').write_text(f'{checksum}  {archive.name}\n', encoding='utf-8')
    print(f'{archive}\n{len(manifest)} files; {archive.stat().st_size:,} bytes\nSHA256 {checksum}')


if __name__ == '__main__':
    main()
