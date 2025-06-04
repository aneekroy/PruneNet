import json
import random
from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
STD_DIR = Path("data/standardized")


def standardize_task(task_dir: Path):
    split_files = list(task_dir.glob("*.json"))
    records = [json.load(open(f)) for f in split_files]
    if "question" in records[0]:
        df = pd.DataFrame(records)
        df["choices"] = df["choices"].apply(json.dumps)
    else:
        df = pd.DataFrame(records)
    random.shuffle(records)
    mid = int(0.5 * len(records))
    pool, test = records[:mid], records[mid:]
    pd.DataFrame(pool).to_csv(STD_DIR / f"{task_dir.name}_pool.csv", index=False)
    pd.DataFrame(test).to_csv(STD_DIR / f"{task_dir.name}_test.csv", index=False)


def main():
    for task_dir in RAW_DIR.iterdir():
        STD_DIR.mkdir(parents=True, exist_ok=True)
        standardize_task(task_dir)


if __name__ == "__main__":
    main()
