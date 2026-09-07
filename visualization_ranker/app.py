"""
FastAPI backend for ranking visualization images.
"""
import json
import random
import re
from pathlib import Path
from collections import defaultdict

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

_APP_DIR = Path(__file__).resolve().parent
VIZ_DIR = _APP_DIR.parent / "visualizations"
VIZ_DIR= Path("C:/Users/PahnkeLab/Desktop/maldi_msi/outputs/2026-03-07/19-26-02/visualizations")
RANKINGS_FILE = _APP_DIR / "visualization_rankings.json"

# Matches __r0__, __r4__, __r12__, etc. (repeat index, optionally followed by __s42__ seed)
_R_RE = re.compile(r"__r(\d+)__(?:s(\d+)__)?")
_R_PREFIX_RE = re.compile(r"__r\d+__")


def get_dataset_key(filename: str) -> str:
    """Extract dataset key: prefix before __rN__ (repeat index). Handles __r0__, __r4__s46__, etc."""
    m = _R_PREFIX_RE.search(filename)
    if m:
        return filename[: m.start()]
    return filename


def get_seed_key(filename: str) -> str:
    """Extract seed key: rN_sM or rN. E.g. __r4__s46__ -> r4_s46, __r0__ -> r0."""
    m = _R_RE.search(filename)
    if m:
        r, s = m.group(1), m.group(2)
        return f"r{r}_s{s}" if s else f"r{r}"
    return "r0"


def get_sampling_mode(filename: str) -> str | None:
    # Handles __r0__superpixel__, __r0__s42__superpixel__, __r4__s46__random__, etc.
    if "__superpixel__" in filename:
        return "superpixel"
    if "__random__" in filename:
        return "random"
    return None


def load_groups(viz_dir: Path) -> dict[str, dict[str, list[str]]]:
    """Group by (mode, dataset_key__seed_key). Each group has images for one seed only."""
    if not viz_dir.exists():
        return {}
    groups = defaultdict(lambda: defaultdict(list))
    for f in viz_dir.glob("*.png"):
        name = f.name
        mode = get_sampling_mode(name)
        if mode:
            dataset_key = get_dataset_key(name)
            seed_key = get_seed_key(name)
            combined_key = f"{dataset_key}__{seed_key}"
            groups[mode][combined_key].append(name)
    for mode in groups:
        for key in groups[mode]:
            groups[mode][key].sort()
    return dict(groups)


app = FastAPI(title="Visualization Ranker")


@app.get("/api/config")
def api_config():
    """Return paths used by the app."""
    return {"rankingsFile": str(RANKINGS_FILE.resolve())}


@app.get("/api/groups")
def api_groups(viz_dir: str | None = None):
    """Return { mode: { dataset_key: [filenames] } }"""
    vd = Path(viz_dir) if viz_dir else VIZ_DIR
    groups = load_groups(vd)
    # Convert to serializable format
    return {
        mode: {k: v for k, v in datasets.items()}
        for mode, datasets in groups.items()
    }


@app.get("/api/dataset/{mode}/{dataset_key}")
def api_dataset(mode: str, dataset_key: str, viz_dir: str | None = None):
    """Return shuffled list of image filenames for this dataset."""
    vd = Path(viz_dir) if viz_dir else VIZ_DIR
    groups = load_groups(vd)
    if mode not in groups or dataset_key not in groups[mode]:
        raise HTTPException(404, "Dataset not found")
    filenames = groups[mode][dataset_key].copy()
    random.shuffle(filenames)
    return {"filenames": filenames}


@app.get("/api/image/{filename:path}")
def api_image(filename: str, viz_dir: str | None = None):
    """Serve an image file."""
    vd = Path(viz_dir) if viz_dir else VIZ_DIR
    path = vd / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(404, "Image not found")
    return FileResponse(path, media_type="image/png")


@app.get("/api/rankings")
def api_get_rankings():
    """Load current rankings."""
    if not RANKINGS_FILE.exists():
        return {}
    with open(RANKINGS_FILE, encoding="utf-8") as f:
        return json.load(f)


@app.post("/api/rankings")
def api_save_rankings(data: dict):
    """Merge and save rankings. Expects { key: [{filename, rank}, ...] }"""
    existing = {}
    if RANKINGS_FILE.exists():
        with open(RANKINGS_FILE, encoding="utf-8") as f:
            existing = json.load(f)
    existing.update(data)
    with open(RANKINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
    return {"ok": True}


# Serve SPA at root
static_dir = _APP_DIR / "static"


@app.get("/")
def index():
    return FileResponse(static_dir / "index.html")
