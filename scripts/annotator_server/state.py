"""
State for the annotator server: loaded image, model, current visualization, and annotations.
All heavy work (load, fit, predict) is done synchronously; the FastAPI app runs it in a thread.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

# Lazy imports for msi_visual and train script logic to avoid loading torch on import
_state: Optional[AnnotatorState] = None


class AnnotatorState:
    """Holds current image, model, visualization, and annotations."""

    def __init__(self) -> None:
        self.config: Optional[Any] = None
        self.config_path: Optional[Path] = None
        self.config_overrides: dict = {}  # UI overrides, e.g. {"model": {"number_of_points": 1500}}
        self.img: Optional[np.ndarray] = None
        self.model: Optional[Any] = None
        self.current_viz: Optional[np.ndarray] = None
        self.annotations: list[dict[str, Any]] = []  # [{ "id", "polygon", "category?" }]
        self.last_roi: Optional[list] = None  # last ROI used for recompute; used by "Apply and reload" when frontend sends no roi
        self.last_category_annotations: Optional[list] = None  # last category annotations used for recompute
        self.current_input_path: Optional[str] = None  # path of the currently loaded .npy (reload only if changed)
        self.busy: bool = False
        self.error: Optional[str] = None
        self.mode: str = "train"  # "train" | "inference"

    @property
    def image_shape(self) -> Optional[tuple[int, int]]:
        if self.img is None:
            return None
        return (int(self.img.shape[0]), int(self.img.shape[1]))

    def reset(self) -> None:
        self.config = None
        self.config_path = None
        self.config_overrides = {}
        self.img = None
        self.model = None
        self.current_viz = None
        self.annotations = []
        self.last_roi = None
        self.last_category_annotations = None
        self.current_input_path = None
        self.busy = False
        self.error = None
        self.mode = "train"


def get_state() -> AnnotatorState:
    global _state
    if _state is None:
        _state = AnnotatorState()
    return _state
