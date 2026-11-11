# Paste this whole file into Colab in place of the old preparation cell.
from pathlib import Path
import yaml

root = Path(globals().get("DATASET_ROOT", "/content/dataset"))
configs = list(root.rglob("data.yaml"))
if len(configs) != 1:
    raise ValueError(f"Expected one data.yaml under {root}, found: {configs}")
data_yaml = configs[0]
base = data_yaml.parent
config = yaml.safe_load(data_yaml.read_text(encoding="utf-8-sig"))
names = config["names"]
if isinstance(names, dict):
    indexed_names = {int(key): value for key, value in names.items()}
    names = [indexed_names[index] for index in range(len(indexed_names))]
if names not in (["DESKTOP PC", "Desktop-PC", "Monitors"], ["pc_case"]):
    raise ValueError(f"Unexpected classes; do not train before checking them: {names}")
pc_ids = {0, 1} if len(names) == 3 else {0}

for split, original in (("train", "train"), ("val", "valid"), ("test", "test")):
    choices = [base / "images" / split, base / original / "images"]
    images = next((path for path in choices if path.is_dir()), None)
    if images is None:
        raise FileNotFoundError(f"Missing {split} images; checked {choices}")
    labels = base / "labels" / split if images.name != "images" else images.parent / "labels"
    if not labels.is_dir():
        raise FileNotFoundError(f"Missing labels directory: {labels}")
    for label in labels.rglob("*.txt"):
        rows = []
        for line in label.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 5 or int(fields[0]) not in range(len(names)):
                raise ValueError(f"Invalid annotation in {label}: {line}")
            if int(fields[0]) in pc_ids:
                rows.append("0 " + " ".join(fields[1:]))
        temp_label = label.with_suffix(".tmp")
        temp_label.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        temp_label.replace(label)
    config[split] = str(images.resolve())

config.update(path=str(base.resolve()), nc=1, names={0: "pc_case"})
temp_yaml = data_yaml.with_suffix(".tmp")
temp_yaml.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
temp_yaml.replace(data_yaml)
print("Dataset:", data_yaml)
print("Classes:", config["names"])
print("Ready. Run the YOLO11n training cell next.")
