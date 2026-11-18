"""CPU verification of downloaded aligned .bin pairs; no app/database imports.

10 contiguous folds in original pair order; fit thresholds on other 9 folds.
No detection, realignment, flip TTA or training. Outputs are resumable per cell.
"""
import argparse
import codecs
import csv
import hashlib
import io
import json
import os
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('buffalo_s', 'ghostfacenet', 'buffalo_l', 'adaface')
DATASETS = ('lfw', 'cfp_fp', 'agedb_30', 'cplfw')


class PairUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Python 2 byte strings sometimes use this constructor. Never import
        # arbitrary classes/functions from a downloaded pickle.
        if (module, name) == ('_codecs', 'encode'):
            return codecs.encode
        allowed = {('numpy.core.multiarray', '_reconstruct'): np.core.multiarray._reconstruct,
                   ('numpy.core.multiarray', 'scalar'): np.core.multiarray.scalar,
                   ('numpy', 'ndarray'): np.ndarray, ('numpy', 'dtype'): np.dtype}
        if (module, name) in allowed:
            return allowed[module, name]
        raise pickle.UnpicklingError(f'Unsupported dataset object: {module}.{name}')


def load_pairs(path):
    raw = path.read_bytes()
    bins, labels = PairUnpickler(io.BytesIO(raw), encoding='bytes').load()
    if not isinstance(bins, (list, tuple)) or not isinstance(labels, (list, tuple, np.ndarray)):
        raise ValueError('Expected image-byte list and boolean-label list')
    if not all(isinstance(v, (bool, int, np.bool_, np.integer)) and v in (0, 1) for v in labels):
        raise ValueError('Invalid pair labels')
    if len(bins) != 2*len(labels) or len(labels) < 10:
        raise ValueError('Invalid pair count')
    if not all(isinstance(v, bytes) or (isinstance(v, np.ndarray) and v.dtype == np.uint8) for v in bins):
        raise ValueError('Invalid image bytes')
    bins = [v if isinstance(v, bytes) else v.tobytes() for v in bins]
    return bins, np.asarray(labels, dtype=bool), hashlib.sha256(raw).hexdigest()


def score_pairs(scores, labels):
    if scores.shape != labels.shape or not np.isfinite(scores).all():
        raise ValueError('Invalid similarity scores')
    thresholds = np.linspace(-1, 1, 2001)
    predictions = scores[:, None] >= thresholds[None, :]
    correct = predictions == labels[:, None]
    total = correct.sum(axis=0)
    out = np.empty(len(labels), dtype=bool)
    folds = []
    for test in np.array_split(np.arange(len(labels)), 10):
        train_correct = total - correct[test].sum(axis=0)
        selected = int(np.argmax(train_correct))
        out[test] = predictions[test, selected]
        folds.append({'threshold': float(thresholds[selected]),
                      'accuracy': float(correct[test, selected].mean()),
                      'testPairs': len(test)})
    tp = int(np.sum(out & labels)); tn = int(np.sum(~out & ~labels))
    fp = int(np.sum(out & ~labels)); fn = int(np.sum(~out & labels))
    return {'accuracyPct': 100*float((out == labels).mean()),
            'foldStdPct': 100*float(np.std([f['accuracy'] for f in folds])),
            'fixedThreshold05AccuracyPct': 100*float(((scores >= .5) == labels).mean()),
            'falseAcceptRatePct': 100*fp/(fp+tn), 'falseRejectRatePct': 100*fn/(fn+tp),
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'folds': folds}


def make_model(name, threads):
    import onnxruntime as ort
    options = ort.SessionOptions()
    # Legacy InsightFace output metadata says batch=1 even though the graph
    # supports dynamic batches; actual output shape is checked below.
    options.log_severity_level = 3
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    if name == 'adaface':
        import torch
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        sys.path.insert(0, str(ROOT/'BE/backend/AdaFace'))
        import net
        path = ROOT/'BE/backend/AdaFace/pretrained/adaface_ir50_webface4m.ckpt'
        model = net.build_model('ir_50').eval()
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        model.load_state_dict({k[6:]: v for k, v in checkpoint['state_dict'].items() if k.startswith('model.')})
        del checkpoint
        def infer(images):
            batch = (np.stack(images).astype(np.float32)/255 - .5)/.5
            tensor = torch.from_numpy(batch.transpose(0, 3, 1, 2).copy())
            with torch.inference_mode():
                features, _ = model(tensor)
            return features.numpy()
        batch_size = 16
    else:
        path = (ROOT/'BE/backend/models/ghostfacenet_v1_w1.3_s1_arcface_ms1mv3.onnx'
                if name == 'ghostfacenet' else Path.home()/'.insightface/models'/name/
                ('w600k_mbf.onnx' if name == 'buffalo_s' else 'w600k_r50.onnx'))
        session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
        if name == 'ghostfacenet':
            def infer(images):
                rgb = np.stack(images)[..., ::-1].astype(np.float32)
                return session.run(None, {session.get_inputs()[0].name: (rgb-127.5)/128})[0]
            batch_size = 1
        else:
            from insightface.model_zoo.arcface_onnx import ArcFaceONNX
            model = ArcFaceONNX(str(path), session=session)
            infer = model.get_feat
            batch_size = 16
    return infer, batch_size, {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def write_reports(outdir):
    results = [json.loads(p.read_text()) for p in sorted(outdir.glob('*__*.json'))]
    fields = ['model', 'dataset', 'pairs', 'accuracyPct', 'foldStdPct',
              'falseAcceptRatePct', 'falseRejectRatePct', 'fixedThreshold05AccuracyPct', 'elapsedSeconds']
    with (outdir/'summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(results)
    lines = ['# Face verification benchmark (CPU)', '',
             'Aligned 112x112 crops; no detector/realignment; no flip augmentation.',
             '10 contiguous folds, original pair order. Threshold grid -1..1 step .001; fit only on other 9 folds.',
             'Accuracy is pair verification, NOT employee identification or camera end-to-end accuracy.',
             'Std is between-fold standard deviation, not a confidence interval. Timing includes decode/preprocess/inference; batch sizes differ.',
             'Different pretrained training datasets/backbones/runtimes: this does not isolate the effect of the training loss.', '',
             '|Model|Dataset|Pairs|Accuracy %|Fold std %|FAR %|FRR %|Accuracy at 0.5 %|',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in results:
        lines.append(f"|{r['model']}|{r['dataset']}|{r['pairs']}|{r['accuracyPct']:.3f}|{r['foldStdPct']:.3f}|{r['falseAcceptRatePct']:.3f}|{r['falseRejectRatePct']:.3f}|{r['fixedThreshold05AccuracyPct']:.3f}|")
    lines += ['', f'Completed cells: {len(results)}/16. Per-cell JSON contains model/dataset hashes and fold thresholds.']
    (outdir/'report.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', choices=MODELS, default=MODELS)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=DATASETS)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--output', type=Path, default=ROOT/'eval/results/verification-2026-10-28')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    for name in args.models:
        infer, batch_size, model_info = make_model(name, args.threads)
        for dataset in args.datasets:
            bins, labels, dataset_hash = load_pairs(ROOT/'eval/val_sets'/f'{dataset}.bin')
            target = args.output/f'{name}__{dataset}.json'
            if target.exists():
                previous = json.loads(target.read_text())
                if previous['datasetSha256'] == dataset_hash and previous['modelInfo'] == model_info:
                    print(f'SKIP {name}/{dataset} completed', flush=True); continue
                raise ValueError(f'Stale result: choose a new output directory: {target}')
            print(f'START {name}/{dataset}: {len(labels)} pairs, {len(bins)} images', flush=True)
            infer([np.zeros((112,112,3), np.uint8)]*batch_size)
            features = []; started = time.perf_counter(); last_report = started
            for i in range(0, len(bins), batch_size):
                images = [cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR) for b in bins[i:i+batch_size]]
                if any(im is None or im.shape != (112,112,3) for im in images):
                    raise ValueError(f'{dataset}: invalid/non-aligned image at batch {i}; no samples silently skipped')
                feature = infer(images)
                norms = np.linalg.norm(feature, axis=1, keepdims=True)
                if feature.shape != (len(images),512) or not np.isfinite(feature).all() or (norms == 0).any():
                    raise ValueError('Invalid embeddings')
                features.append(feature/norms)
                now = time.perf_counter()
                if now-last_report >= 30:
                    done = min(i+batch_size, len(bins))
                    print(f'PROGRESS {name}/{dataset} {done}/{len(bins)} elapsed={now-started:.0f}s ETA={(len(bins)-done)*(now-started)/done:.0f}s', flush=True)
                    last_report = now
            elapsed = time.perf_counter()-started
            embeddings = np.concatenate(features)
            scores = np.sum(embeddings[0::2]*embeddings[1::2], axis=1)
            result = dict(model=name, dataset=dataset, pairs=len(labels),
                          samePairs=int(labels.sum()), differentPairs=int((~labels).sum()),
                          datasetSha256=dataset_hash, modelInfo=model_info, threads=args.threads,
                          batchSize=batch_size, flip=False, alignedInput=True, elapsedSeconds=elapsed,
                          **score_pairs(scores, labels))
            target.write_text(json.dumps(result, indent=2), encoding='utf-8')
            np.savez_compressed(args.output/f'{name}__{dataset}.npz', scores=scores, labels=labels)
            write_reports(args.output)
            print(f"DONE {name}/{dataset}: {result['accuracyPct']:.3f}% +/- {result['foldStdPct']:.3f}% ({elapsed:.1f}s)", flush=True)
        del infer
    print('ALL REQUESTED CELLS COMPLETE', flush=True)


if __name__ == '__main__':
    main()
