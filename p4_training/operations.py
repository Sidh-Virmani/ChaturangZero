"""Small terminal-run safeguards; no training or evaluation algorithms."""
from contextlib import contextmanager
import os
from pathlib import Path
from time import perf_counter


@contextmanager
def training_lock(directory: Path):
    """OS releases the writer lock even after a crash; never delete its inode."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.training.lock').open('a+b') as stream:
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f'Another trainer holds the writer lock for {directory}') from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def run_with_progress(trainer, print_iteration, print_game, print_wait):
    """ETA uses completed wall-clock iterations, including checkpoint writes."""
    started = previous = perf_counter()
    completed = 0
    target = trainer.completed_iterations + trainer.config.iterations
    print(f'Training through iteration {target}; resume state: '
          f'{trainer.config.checkpoint_dir / "training_state.pt"}', flush=True)

    def report(metrics):
        nonlocal previous, completed
        now = perf_counter()
        completed += 1
        elapsed = now - started
        eta = elapsed / completed * (target - metrics.iteration)
        print(f'iteration={metrics.iteration}/{target} wall_seconds={now - previous:.1f} '
              f'elapsed_minutes={elapsed / 60:.1f} eta_minutes={eta / 60:.1f}', flush=True)
        previous = now
        print_iteration(metrics)

    trainer.run(on_game=print_game, on_iteration=report, on_wait=print_wait)
