# check_data_set.py  (chạy trong BE/backend: python check_data_set.py)
import pickle, json, time
import numpy as np, cv2
from pathlib import Path
from insightface.model_zoo import get_model

VAL_DIR = Path("../../eval/val_sets")
DATASETS = ["lfw"]   # chạy thử nhanh; khi ổn đổi thành ["lfw", "cfp_fp", "agedb_30", "cplfw"]

def load_bin(path):
    with open(path, "rb") as f:
        bins, issame = pickle.load(f, encoding="bytes")
    imgs = [cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR) for b in bins]  # BGR 112x112
    return imgs, np.array(issame, dtype=bool)

def subset(imgs, issame, n_pairs=1000):
    # giữ n_pairs cặp đầu; mỗi cặp là 2 ảnh liên tiếp
    return imgs[:2 * n_pairs], issame[:n_pairs]

def embed(imgs, rec, bs=64, flip=True):
    out = []
    for i in range(0, len(imgs), bs):
        batch = imgs[i:i + bs]
        e = np.asarray(rec(batch), dtype=np.float32)
        if flip:
            e = e + np.asarray(rec([cv2.flip(x, 1) for x in batch]), dtype=np.float32)
        out.append(e)
    E = np.concatenate(out)
    return E / np.linalg.norm(E, axis=1, keepdims=True)

def kfold_acc(E, issame, k=10):
    sims = (E[0::2] * E[1::2]).sum(1)
    n = len(sims)
    folds = np.array_split(np.arange(n), k)
    thr_grid = np.linspace(-1, 1, 2001)
    accs = []
    for i in range(k):
        te = folds[i]
        tr = np.concatenate([folds[j] for j in range(k) if j != i])
        tr_acc = [((sims[tr] > t) == issame[tr]).mean() for t in thr_grid]
        t_best = thr_grid[int(np.argmax(tr_acc))]
        accs.append(((sims[te] > t_best) == issame[te]).mean())
    return float(np.mean(accs)), float(np.std(accs))

def make_insightface_rec(pack, onnx_name):
    path = Path.home() / ".insightface" / "models" / pack / onnx_name
    if not path.exists():
        raise FileNotFoundError(path)
    rec = get_model(str(path), providers=["CPUExecutionProvider"])
    rec.prepare(ctx_id=-1)
    return lambda batch: rec.get_feat(batch)

MODELS = {
    "buffalo_l": lambda: make_insightface_rec("buffalo_l", "w600k_r50.onnx"),
    # "antelopev2": lambda: make_insightface_rec("antelopev2", "glintr100.onnx"),
}

if __name__ == "__main__":
    print("Dang nap du lieu tu", VAL_DIR.resolve())
    data = {d: load_bin(VAL_DIR / f"{d}.bin") for d in DATASETS}
    results = {}
    for name, factory in MODELS.items():
        rec = factory()
        results[name] = {}
        for d, (imgs, issame) in data.items():
            t0 = time.time()
            acc, std = kfold_acc(embed(imgs, rec), issame)
            results[name][d] = {"acc": round(acc * 100, 3), "std": round(std * 100, 3),
                                "sec": round(time.time() - t0, 1)}
            print(f"{name:12s} {d:9s} {acc*100:.2f} ± {std*100:.2f}  ({time.time()-t0:.0f}s)")
    Path("benchmark-verification.json").write_text(json.dumps(results, indent=2))
    print("Xong, da luu benchmark-verification.json")