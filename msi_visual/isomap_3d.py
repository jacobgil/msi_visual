from msi_visual.base_dim_reduction import BaseDimReductionWithoutFit
from sklearn.manifold import Isomap

class Isomap3D(BaseDimReductionWithoutFit):
    def __init__(self):
        super().__init__(model=Isomap(n_components=3), name="Isomap")