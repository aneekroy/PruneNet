from datasets import list_datasets, load_dataset
from pathlib import Path
import json

RAW_DIR = Path("data/raw")


def download_all():
    for dataset_name in list_datasets():
        ds = load_dataset(dataset_name)
        for split in ds.keys():
            out_dir = RAW_DIR / dataset_name / split
            out_dir.mkdir(parents=True, exist_ok=True)
            for i, item in enumerate(ds[split]):
                with open(out_dir / f"{i}.json", "w") as f:
                    json.dump(item, f)


if __name__ == "__main__":
    download_all()
