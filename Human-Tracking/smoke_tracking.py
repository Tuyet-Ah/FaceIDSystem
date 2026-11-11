"""Exercise the real tracking pipeline on a short local-video segment."""
import argparse
import json
import time
from pathlib import Path

import legacy_app as app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=int, default=30)
    parser.add_argument('--require-gpu', action='store_true')
    args = parser.parse_args()
    if args.frames < 2:
        parser.error('--frames must be at least 2')
    source = next((app.BASE_DIR / 'data').glob('*Chain*.mp4'))
    # Process the same consecutive frames regardless of hardware speed.
    app.REALTIME_SYNC = False
    worker = app.TrackingWorker(str(source))
    samples = []
    started = time.perf_counter()

    def publish(frame):
        samples.append(time.perf_counter())
        if len(samples) >= args.frames:
            worker.running = False

    worker._publish = publish
    worker.run()
    result = {
        'frames': len(samples), 'error': worker.error,
        'reid_provider': worker.embedding_backend,
        'reid_model': app.REID_LABEL,
        'torch_cuda': app.torch.cuda.is_available(),
        'people_last_frame': worker.current_count,
        'confirmed_tracks': len(worker._gallery.profiles) if hasattr(worker, '_gallery') else len(getattr(worker, '_confirmed_ids', {})),
        'wall_seconds': time.perf_counter() - started,
        'fps_after_first_frame': ((len(samples)-1)/(samples[-1]-samples[0])) if len(samples)>1 else 0,
    }
    print(json.dumps(result, indent=2))
    assert worker.error is None and len(samples) == args.frames, result
    if args.require_gpu:
        assert app.torch.cuda.is_available(), result
        assert worker.embedding_backend in {'CUDAExecutionProvider', 'PyTorch CUDA'}, result


if __name__ == '__main__':
    main()
