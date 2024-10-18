from msi_visual.base_dim_reduction import BaseDimReductionWithoutFit
from sklearn.manifold import SpectralEmbedding

class Spectral3D(BaseDimReductionWithoutFit):
    def __init__(self):
        super().__init__(model=SpectralEmbedding(n_components=3), name="SpectralEmbedding3D")