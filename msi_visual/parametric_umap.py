from umap.parametric_umap import ParametricUMAP 
from msi_visual.utils import normalize, segment_visualization
from msi_visual.parametric_mics_lmc import random_pixel_sampling, uniform_superpixel_sampling
import tensorflow as tf
import numpy as np
import os
import cv2
from pathlib import Path
import joblib


def _configure_tensorflow_gpu() -> bool:
    """Enable GPU memory growth when a CUDA device is visible to TensorFlow."""
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return False
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    # ParametricUMAP on TF 2.10 + GPU needs global eager mode (see keras empty-logs error).
    try:
        tf.config.run_functions_eagerly(True)
    except Exception:
        pass
    return True


def _require_keras2_parametric_umap() -> None:
    """Parametric UMAP (umap-learn 0.5.x) is incompatible with Keras 3."""
    try:
        import keras

        major = int(str(getattr(keras, "__version__", "0")).split(".")[0])
    except Exception:
        return
    if major >= 3:
        raise RuntimeError(
            "Parametric UMAP needs Keras 2 (TensorFlow <=2.15, umap-learn 0.5.5). "
            f"This env has keras {keras.__version__}. "
            "Activate the GPU env: conda activate maldi-pumap-gpu"
        )


class UMAPVirtualStain:
    def __init__(
            self,
            n_components=1,
            n_training_epochs=1,
            n_epochs=None,
            n_neighbors=50,
            pixel_sampling="superpixel",
            num_samples=5000,
            pca_fit_step=4,
            random_state=42,
            start_bin=0,
            end_bin=None):
        self.n_components = n_components
        self.n_training_epochs = n_training_epochs
        self.n_epochs = n_epochs
        self.n_neighbors = n_neighbors
        self.pixel_sampling = pixel_sampling
        self.num_samples = num_samples
        self.pca_fit_step = pca_fit_step
        self.random_state = random_state
        self.UMAP = MSIParametricUMAP(
            n_components=n_components,
            n_training_epochs=n_training_epochs,
            n_epochs=n_epochs,
            n_neighbors=n_neighbors,
            pixel_sampling=pixel_sampling,
            num_samples=num_samples,
            pca_fit_step=pca_fit_step,
            random_state=random_state,
            start_bin=start_bin,
            end_bin=end_bin)

    def fit(self, images, keras_fit_kwargs, roi_mask=None):
        _require_keras2_parametric_umap()
        self.encoder = self.UMAP.fit(images, keras_fit_kwargs, roi_mask=roi_mask)

    def save(self, output_folder):
        os.makedirs(output_folder, exist_ok=True)
        self.encoder.save(Path(output_folder) / "UMAP.keras")
        print("saved encoder")
        joblib.dump(self.UMAP , Path(output_folder) / "msi_UMAP.joblib")

    def load(self, output_folder):
        self.encoder = tf.keras.models.load_model(
            Path(output_folder) / "UMAP.keras")

        self.UMAP  = joblib.load(Path(output_folder) / "msi_UMAP.joblib")

    def predict(self, img):
        mask = img.sum(axis=-1) > 0
        result = normalize(self.UMAP.predict(img, self.encoder), mask=mask)
        result = np.uint8(255 * result)
        result[~mask] = 0
        result = np.float32(result) / 255
        return result

    def segment_visualization(self,
                              img,
                              visualization,
                              method='spatial_norm'):

        return segment_visualization(visualization)


class MSIParametricUMAP :
    def __init__(
            self,
            n_components=1,
            n_training_epochs=1,
            n_epochs=None,
            n_neighbors=50,
            pixel_sampling="superpixel",
            num_samples=5000,
            pca_fit_step=4,
            random_state=42,
            start_bin=0,
            end_bin=None):
        self.start_bin = start_bin
        self.end_bin = end_bin
        self.n_components = n_components
        self.n_training_epochs = n_training_epochs
        self.n_epochs = n_epochs
        self.n_neighbors = n_neighbors
        self.pixel_sampling = pixel_sampling
        self.num_samples = num_samples
        self.pca_fit_step = pca_fit_step
        self.random_state = random_state

    def predict(self, img, encoder):
        vector = img[:, :, self.start_bin:self.end_bin]
        vector = vector.reshape(-1, vector.shape[-1])
        output = encoder.predict(vector, batch_size=1000)
        output = output.reshape(img.shape[0], img.shape[1], output.shape[-1])
        return output

    def fit(self, images, keras_fit_kwargs={}, roi_mask=None):
        sampled_vectors = []
        for img in images:
            height, width = img.shape[0], img.shape[1]
            reshaped_full = img.reshape(height * width, -1)
            if self.pixel_sampling == "random":
                _, sampled_flat_indices, _ = random_pixel_sampling(
                    reshaped_full,
                    self.num_samples,
                    height,
                    width,
                    roi_mask=roi_mask,
                    rng_seed=self.random_state,
                )
            else:
                _, sampled_flat_indices, _ = uniform_superpixel_sampling(
                    img,
                    reshaped_full,
                    approx_budget=self.num_samples,
                    init_superpixels=512,
                    init_points_per_superpixel=8,
                    compactness=10,
                    max_iters=8,
                    verbose=False,
                    visualize=False,
                    rng_seed=self.random_state,
                    roi_mask=roi_mask,
                    pca_fit_step=max(1, int(self.pca_fit_step)),
                )
            if sampled_flat_indices is None or len(sampled_flat_indices) == 0:
                raise ValueError("Sampling produced no points for Parametric UMAP")
            vector_full = img[:, :, self.start_bin:self.end_bin].reshape(height * width, -1)
            sampled_vectors.append(vector_full[sampled_flat_indices])
            self.sampled_flat_indices = sampled_flat_indices
        vector = np.concatenate(sampled_vectors, axis=0)
        vector = vector.reshape((-1, vector.shape[-1]))
        n_samples = int(vector.shape[0])
        if n_samples < 2:
            raise ValueError("Parametric UMAP needs at least 2 samples; increase num_samples or ROI size")

        # Avoid steps_per_epoch=0 inside ParametricUMAP when batches exceed samples
        fit_kwargs = dict(keras_fit_kwargs) if keras_fit_kwargs else {}
        if "steps_per_epoch" in fit_kwargs and fit_kwargs["steps_per_epoch"] is not None:
            if int(fit_kwargs["steps_per_epoch"]) <= 0:
                raise ValueError("steps_per_epoch must be >= 1 for Parametric UMAP")
        # Remove steps_per_epoch from fit kwargs to avoid duplicates in Keras
        if "steps_per_epoch" in fit_kwargs:
            fit_kwargs.pop("steps_per_epoch", None)
        if "batch_size" in fit_kwargs:
            fit_kwargs.pop("batch_size", None)

        # Choose a batch size that guarantees steps_per_epoch >= 1 in ParametricUMAP
        n_neighbors_effective = max(1, min(self.n_neighbors, n_samples - 1))
        n_edges_estimate = n_samples * n_neighbors_effective
        max_batch_for_steps = max(1, n_edges_estimate // 100)  # loss_report_frequency=100
        batch_size = int(max(1, min(256, n_samples, max_batch_for_steps)))
        print(
            f"[ParametricUMAP] n_samples={n_samples} n_neighbors={self.n_neighbors} "
            f"n_edges_estimate={n_edges_estimate} batch_size={batch_size}"
        )
        # Avoid duplicating steps_per_epoch; ParametricUMAP supplies it internally.
        if "steps_per_epoch" in fit_kwargs:
            fit_kwargs.pop("steps_per_epoch", None)

        dims = vector.shape[-1]
        use_gpu = _configure_tensorflow_gpu()
        if use_gpu:
            print("[ParametricUMAP] TensorFlow GPU enabled")
        else:
            print("[ParametricUMAP] TensorFlow CPU only")

        encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(dims,)),
            tf.keras.layers.Dense(units=256, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(units=128, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(units=64, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(units=self.n_components)
        ])

        umap_kwargs = {
            "encoder": encoder,
            "verbose": True,
            "n_components": self.n_components,
            "n_neighbors": self.n_neighbors,
            "n_training_epochs": self.n_training_epochs,
            "loss_report_frequency": 100,
            "dims": (dims,),
            "run_eagerly": True,
            "batch_size": batch_size,
            "keras_fit_kwargs": fit_kwargs,
        }
        if self.n_epochs is not None:
            umap_kwargs["n_epochs"] = int(self.n_epochs)

        UMAP_model = ParametricUMAP(**umap_kwargs)

        UMAP_model.fit(vector)
        return encoder
