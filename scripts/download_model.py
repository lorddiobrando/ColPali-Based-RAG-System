"""
Direct HTTP model downloader with resume support.
Bypasses HF client entirely to avoid xet protocol issues.
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

import requests
from tqdm import tqdm

HF_TOKEN = os.environ.get("HF_TOKEN", "")
CACHE_DIR = Path(os.environ.get("USERPROFILE", "~")) / ".cache" / "huggingface" / "hub"


def download_file(repo_id: str, filename: str, dest_dir: Path):
    """Download a file from HF using plain HTTP with resume support."""
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    dest = dest_dir / filename
    dest.parent.mkdir(parents=True, exist_ok=True)

    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    # Check existing size for resume
    existing_size = 0
    if dest.exists():
        existing_size = dest.stat().st_size
        headers["Range"] = f"bytes={existing_size}-"

    print(f"  Downloading {filename} ...")
    if existing_size > 0:
        print(f"    Resuming from {existing_size / 1024 / 1024:.1f} MB")

    try:
        resp = requests.get(url, headers=headers, stream=True, timeout=60, allow_redirects=True)

        if resp.status_code == 416:
            # Range not satisfiable = file already complete
            print(f"  [OK] {filename} already complete ({existing_size / 1024 / 1024:.1f} MB)")
            return dest

        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0))
        if existing_size > 0 and resp.status_code == 206:
            total += existing_size

        mode = "ab" if resp.status_code == 206 else "wb"
        if mode == "wb":
            existing_size = 0

        with open(dest, mode) as f, tqdm(
            total=total,
            initial=existing_size,
            unit="B",
            unit_scale=True,
            desc=f"    {filename}",
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=8192 * 16):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

        print(f"  [OK] {filename} -> {dest}")
        return dest

    except Exception as e:
        print(f"  [X] {filename} failed: {e}")
        return None


def main():
    base_dest = Path("data/model_cache/colqwen2-base")
    adapter_dest = Path("data/model_cache/colqwen2-v1.0")

    base_files = [
        "config.json",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "generation_config.json",
        "special_tokens_map.json",
    ]

    print("=== Downloading vidore/colqwen2-base ===\n")
    for f in base_files:
        result = download_file("vidore/colqwen2-base", f, base_dest)
        if result is None and "safetensors" in f:
            # Retry once for big files
            print("    Retrying ...")
            download_file("vidore/colqwen2-base", f, base_dest)

    print("\n=== Downloading vidore/colqwen2-v1.0 adapter ===\n")
    adapter_files = ["adapter_config.json", "adapter_model.safetensors"]
    for f in adapter_files:
        download_file("vidore/colqwen2-v1.0", f, adapter_dest)

    print("\n[OK] Done. Files saved to:")
    print(f"  Base:    {base_dest.resolve()}")
    print(f"  Adapter: {adapter_dest.resolve()}")


if __name__ == "__main__":
    main()
