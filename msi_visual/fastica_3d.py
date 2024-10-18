from msi_visual.base_dim_reduction import BaseDimReductionWithoutFit
from sklearn.decomposition import FastICA

class FastICA3D(BaseDimReductionWithoutFit):
    def __init__(self):
        super().__init__(model=FastICA(n_components=3), name="FastICA3D")