from msi_visual.base_dim_reduction import BaseDimReductionWithoutFit
from sklearn.manifold import MDS

class MDS3D(BaseDimReductionWithoutFit):
    def __init__(self):
        super().__init__(model=MDS(n_components=3), name="MDS3D")