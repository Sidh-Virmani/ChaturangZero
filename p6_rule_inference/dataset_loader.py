import os
import json
from pathlib import Path
from typing import List

from dotenv import load_dotenv

# Load HF token from .env if present
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN not set in environment. Please create a .env file with HF_TOKEN=...")

# Name of the dataset on HuggingFace – replace with an actual public dataset if exists
DATASET_REPO = "anonymous/chaturanga-games"  # placeholder name
CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "chaturanga"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def download_dataset(split: str = "train") -> List[List[str]]:
    """Download the dataset from HuggingFace and return a list of games.

    Each game is represented as a list of SAN‑style move strings, e.g. ["e2e3", "e7e6", ...].
    The function caches the raw JSON locally to avoid repeated downloads.
    """
    import requests

    cache_file = CACHE_DIR / f"{split}.json"
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    # Build the download URL – using the Hub API
    url = f"https://huggingface.co/api/datasets/{DATASET_REPO}/download/{split}.json"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()
    # Assume data is a list of dicts with a "moves" field (space‑separated string)
    games = []
    for entry in data:
        moves_str = entry.get("moves", "")
        if not moves_str:
            continue
        moves = moves_str.split()
        games.append(moves)
    # Cache locally
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(games, f)
    return games

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download Chaturanga dataset from HuggingFace")
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"], help="Dataset split to download")
    args = parser.parse_args()
    ds = download_dataset(args.split)
    print(f"Downloaded {len(ds)} games for split {args.split} into {CACHE_DIR}")
