

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def read_prompt_csv(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_filename(text: str, max_len: int = 120) -> str:
    text = str(text).strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-zA-Z0-9_\-\.\u4e00-\u9fff]+", "", text)
    text = text.strip("_")
    return text[:max_len] if text else "prompt"


def prompt_dir_name(prompt_id: str, prompt: str) -> str:
    return f"{prompt_id}__{safe_filename(prompt)}"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def image_complete(path: Path, min_image_bytes: int = 1024) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size >= min_image_bytes


def metadata_complete(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def external_sample_complete(
    prompt_dir: Path,
    seed: int,
    min_image_bytes: int = 1024,
) -> bool:
    image_path = prompt_dir / f"seed{seed}.png"
    meta_path = prompt_dir / f"seed{seed}_metadata.json"
    return image_complete(image_path, min_image_bytes) and metadata_complete(meta_path)


def copy_or_move_image(src: Path, dst: Path, move: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))


def find_first_image(folder: Path) -> Optional[Path]:
    if not folder.exists():
        return None
    candidates = []
    for suffix in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        candidates.extend(folder.rglob(suffix))
    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]
