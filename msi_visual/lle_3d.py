from msi_visual.base_dim_reduction import BaseDimReductionWithoutFit
from sklearn.manifold import LocallyLinearEmbedding

class LLE3D(BaseDimReductionWithoutFit):
    def __init__(self):
        super().__init__(model=LocallyLinearEmbedding(n_components=3), name="LLE3D")