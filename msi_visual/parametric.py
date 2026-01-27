import xgboost as xgb
import numpy as np

class ParametricModel:
    def __init__(self, model, downsampling):
        self.model = model
        self._trained = False
        self.model._trained = False
        self.downsampling = downsampling
        self.parametric_model = xgb.XGBRegressor(
            n_estimators=30,
            learning_rate=0.1,
            max_depth=7,
            multi_strategy="multi_output_tree",
            random_state=42
        )

    def __repr__(self):
        return f"Parametric {self.model} {self.downsampling}"

    def __call__(self, img):        
        if not self._trained:
            downsampled = img[::self.downsampling, ::self.downsampling, :]
            print(downsampled.shape, "downsampled")
            output = self.model(downsampled)
            if isinstance(output, list):
                output = output[0]
            
            mask = np.uint8(downsampled.max(axis=-1) > 0) * 255
            print(mask.shape, downsampled.shape)
            vector = downsampled[mask > 0].reshape(-1, downsampled.shape[-1])
            output_float = np.float32(output) / 255 - 0.5
            print(output_float.shape, mask.shape, "output")
            output_float = output_float[mask > 0]
            output_float = output_float.reshape(-1, 3)
            print(vector.shape, output_float.shape)
            self.parametric_model.fit(vector, output_float)
        
        vector = img.reshape(-1, img.shape[-1])
        output = self.parametric_model.predict(vector) + 0.5
        output = output.reshape(img.shape[0], img.shape[1], 3)
        output = np.clip(output, 0, 1)
        output = np.uint8(output * 255)
        return output