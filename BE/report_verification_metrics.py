"""Summarize saved pair scores; never run models or modify original results."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODELS = ('buffalo_l', 'buffalo_s', 'adaface', 'ghostfacenet')
DATASETS = ('lfw', 'cfp_fp', 'agedb_30', 'cplfw')


def calculate(scores, labels, saved):
    if scores.ndim != 1 or scores.shape != labels.shape or not np.isfinite(scores).all():
        raise ValueError('Invalid score/label arrays')
    if set(np.unique(labels).tolist()) != {False, True}:
        raise ValueError('Both genuine and impostor pairs required')
    labels = labels.astype(bool)
    prediction = np.zeros(len(labels), dtype=bool)
    tests = np.array_split(np.arange(len(labels)), 10)
    if len(saved['folds']) != 10 or len(labels) != saved['pairs']:
        raise ValueError('Fold or pair count mismatch')
    for test, fold in zip(tests, saved['folds']):
        assert len(test) == fold['testPairs']
        prediction[test] = scores[test] >= fold['threshold']
    counts = dict(tp=int(np.sum(prediction & labels)), tn=int(np.sum(~prediction & ~labels)),
                  fp=int(np.sum(prediction & ~labels)), fn=int(np.sum(~prediction & labels)))
    for key, value in counts.items():
        if value != saved[key]:
            raise ValueError(f'Saved scores disagree with report: {key}')
    far = 100*counts['fp']/(counts['fp']+counts['tn'])
    frr = 100*counts['fn']/(counts['fn']+counts['tp'])
    accuracy = 100*(counts['tp']+counts['tn'])/len(labels)
    assert np.isclose(far, saved['falseAcceptRatePct'])
    assert np.isclose(frr, saved['falseRejectRatePct'])
    assert np.isclose(accuracy, saved['accuracyPct'])
    fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    auc = float(roc_auc_score(labels, scores))
    return dict(model=saved['model'], dataset=saved['dataset'], pairs=len(labels),
                farPct=far, frrPct=frr, tarPct=100-frr, accuracyPct=accuracy,
                pooledRocAuc=auc, **counts), (fpr, tpr, thresholds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=Path(__file__).resolve().parents[1]/'eval/results/verification-2026-10-28')
    args = parser.parse_args()
    out = args.input/'metrics'
    out.mkdir(exist_ok=True)
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for dataset, ax in zip(DATASETS, axes.flat):
        for model in MODELS:
            stem = f'{model}__{dataset}'
            saved = json.loads((args.input/f'{stem}.json').read_text(encoding='utf-8'))
            with np.load(args.input/f'{stem}.npz', allow_pickle=False) as data:
                row, curve = calculate(data['scores'], data['labels'], saved)
            rows.append(row)
            fpr, tpr, thresholds = curve
            np.savez_compressed(out/f'{stem}_roc.npz', far=fpr, tar=tpr, thresholds=thresholds)
            ax.plot(100*fpr, 100*tpr, label=f'{model} (AUC {row["pooledRocAuc"]:.4f})')
        ax.set(title=dataset.upper(), xlabel='FAR (%)', ylabel='TAR (%)', xlim=(0,10), ylim=(50,100.2))
        ax.grid(alpha=.3); ax.legend(loc='lower right', fontsize=8)
    fig.suptitle('Pooled empirical ROC - zoom FAR 0-10%; no flip TTA, aligned face pairs')
    fig.savefig(out/'roc_comparison.png', dpi=180)
    plt.close(fig)
    (out/'metrics.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    lines = ['# FAR / FRR / TAR / ROC-AUC', '',
             'FAR, FRR, TAR, Accuracy: held-out predictions using the saved threshold for each of 10 folds.',
             'ROC/AUC: descriptive pooled empirical curves across all saved pair scores; not a separately calibrated deployment threshold.',
             'AUC is on a 0-1 scale. Other metrics below are percentages.',
             'Only 3,000 (CFP-FP: 3,500) impostor pairs per dataset; do not infer reliable FAR=0.01% performance.',
             'These are aligned-image pair verification results, not end-to-end camera identification or spoof resistance.', '',
             '|Dataset|Model|Accuracy %|FAR %|FRR %|TAR %|Pooled ROC-AUC|FP|FN|',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"|{r['dataset']}|{r['model']}|{r['accuracyPct']:.3f}|{r['farPct']:.3f}|{r['frrPct']:.3f}|{r['tarPct']:.3f}|{r['pooledRocAuc']:.6f}|{r['fp']}|{r['fn']}|")
    lines += ['', '![ROC comparison](roc_comparison.png)', '',
              'All 16 confusion matrices were recomputed from NPZ scores and checked against the original JSON results.',
              'Full ROC points (not only the plotted zoom) are saved in each *_roc.npz. No source results were overwritten.']
    (out/'far_frr_tar_roc.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__': main()
