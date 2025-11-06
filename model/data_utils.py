from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


def _iter_candidates(filename: str, fold: Any, label: Any, audio_root: Path) -> Iterable[Path]:
    if fold is not None and str(fold).strip() not in {"", "nan", "NaN"}:
        try:
            fold_idx = int(float(fold))
            yield audio_root / f"fold{fold_idx}" / filename
            yield audio_root / f"{fold_idx}" / filename
        except (TypeError, ValueError):
            pass

    if isinstance(label, str) and label:
        yield audio_root / label / filename
        yield audio_root / label.lower() / filename
        yield audio_root / label.replace(" ", "_") / filename
        yield audio_root / label.lower().replace(" ", "_") / filename

    yield audio_root / filename


def _safe_get(row: Any, key: str, default: Any = None) -> Any:
    if hasattr(row, "get"):
        return row.get(key, default)
    try:
        return row[key]
    except Exception:
        return default


def resolve_audio_path(row: Any, audio_root: Path) -> Path:
    """
    Retourne le chemin absolu du wav correspondant à une ligne de métadonnées.
    Compatible UrbanSound8K (foldx/...) et datasets rangés par classe.
    """
    filename = row["slice_file_name"]
    fold = _safe_get(row, "fold")
    label = _safe_get(row, "class")

    seen: set[Path] = set()
    for candidate in _iter_candidates(filename, fold, label, audio_root):
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Impossible de localiser {filename} (fold={fold}, class={label}) sous {audio_root}"
    )
