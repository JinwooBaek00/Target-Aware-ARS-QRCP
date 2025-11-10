import json
from pathlib import Path

DATA_FILE = Path("names.json")


def add_name(name):
    names = load_names()
    names.append(name)
    with open(DATA_FILE, "w") as f:
        json.dump(names, f)


def load_names():
    if DATA_FILE.exists():
        with open(DATA_FILE) as f:
            return json.load(f)
    return []
