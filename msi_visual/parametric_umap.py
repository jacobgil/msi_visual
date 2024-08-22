from umap.parametric_umap import ParametricUMAP 
from msi_visual.utils import normalize, segment_visualization
import tensorflow as tf
import numpy as np
import os
import cv2
from pathlib import Path
import joblib


class UMAPVirtualStain:
    def __init__(self, n_components=1, start_bin=0, end_bin=None):
        self.n_components = n_components
        self.UMAP  = MSIParametricUMAP (
            n_components=n_components,
            start_bin=start_bin,
            end_bin=end_bin)

    def fit(self, images, keras_fit_kwargs):
        self.encoder = self.UMAP.fit(images, keras_fit_kwargs)

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
        result = normalize(self.UMAP.predict(img, self.encoder))
        result = np.uint8(255 * result)
        result = np.float32(result) / 255
        return result

    def segment_visualization(self,
                              img,
                              visualization,
                              method='spatial_norm'):

        return segment_visualization(img, visualization)


class MSIParametricUMAP :
    def __init__(
            self,
            n_components=1,
            n_training_epochs=1,
            n_neighbors=50,
            start_bin=0,
            end_bin=None):
        self.start_bin = start_bin
        self.end_bin = end_bin
        self.n_components = n_components
        self.n_training_epochs = n_training_epochs
        self.n_neighbors = n_neighbors

    def predict(self, img, encoder):
        vector = img[:, :, self.start_bin:self.end_bin]
        vector = vector.reshape(-1, vector.shape[-1])
        output = encoder.predict(vector, batch_size=1000)
        output = output.reshape(img.shape[0], img.shape[1], output.shape[-1])
        return output

    def fit(self, images, keras_fit_kwargs={}):
        vector = np.concatenate([img[:, :, self.start_bin:self.end_bin].reshape(
            img.shape[0] * img.shape[1], -1) for img in images], axis=0)
        vector = vector.reshape((-1, vector.shape[-1]))

        dims = vector.shape[-1]

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

        UMAP_model = ParametricUMAP (encoder=encoder,
                                    verbose=True,
                                    n_components=self.n_components,
                                    n_neighbors=self.n_neighbors,
                                    n_training_epochs=self.n_training_epochs,
                                    loss_report_frequency=100,
                                    dims=(dims,),
                                    run_eagerly=True,
                                    keras_fit_kwargs=keras_fit_kwargs)

        UMAP_model.fit(vector)
        return encoder
