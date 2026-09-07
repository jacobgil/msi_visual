"""NMF-based 3-component visualization, similar to PCA3D."""
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import NMF

from msi_visual.utils import normalize


class NMF3D:
    """NMF decomposition to 3 components for RGB visualization."""

    def __init__(
        self,
        n_components: int = 3,
        init: str = "nndsvda",
        max_iter: int = 200,
        random_state: int | None = 42,
        start_bin: int = 0,
        end_bin: int | None = None,
    ):
        self.n_components = n_components
        self.init = init
        self.max_iter = max_iter
        self.random_state = random_state
        self.start_bin = start_bin
        self.end_bin = end_bin
        self._trained = False

    def _ensure_legacy_attrs(self) -> None:
        """Fill defaults for instances loaded from old joblib pickles (no __init__)."""
        if not hasattr(self, "n_components"):
            self.n_components = 3
        if not hasattr(self, "init"):
            self.init = "nndsvda"
        if not hasattr(self, "max_iter"):
            self.max_iter = 200
        if not hasattr(self, "random_state"):
            self.random_state = 42
        if not hasattr(self, "start_bin"):
            self.start_bin = 0
        if not hasattr(self, "end_bin"):
            self.end_bin = None
        if not hasattr(self, "_trained"):
            self._trained = bool(getattr(self, "nmf", None) is not None)

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._ensure_legacy_attrs()

    def __repr__(self) -> str:
        self._ensure_legacy_attrs()
        return f"NMF3D n_components={self.n_components} max_iter={self.max_iter} random_state={self.random_state}"

    def fit(self, images: list[np.ndarray]) -> "NMF3D":
        if self.end_bin is None:
            self.end_bin = images[0].shape[-1]
        n_bins = self.end_bin - self.start_bin
        vector = np.concatenate(
            [
                img[:, :, self.start_bin : self.end_bin].reshape(-1, n_bins)
                for img in images
            ],
            axis=0,
        )
        vector = np.clip(vector, 0.0, None).astype(np.float32)
        self.nmf = NMF(
            n_components=self.n_components,
            init=self.init,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self.nmf.fit(vector)
        self._trained = True
        return self

    def predict(self, img: np.ndarray) -> np.ndarray:
        n_bins = self.end_bin - self.start_bin
        vector = img[:, :, self.start_bin : self.end_bin].reshape((-1, n_bins))
        vector = np.clip(vector, 0.0, None).astype(np.float32)
        result = self.nmf.transform(vector)
        result = result.reshape(img.shape[0], img.shape[1], self.n_components)
        return np.uint8(255 * normalize(result))

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if not self._trained:
            self.fit([img])
        return self.predict(img)

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)
