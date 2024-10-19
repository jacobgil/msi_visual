from msi_visual.base_dim_reduction import BaseDimReductionWithoutFit
from sklearn.decomposition import LatentDirichletAllocation
class LDA3D(BaseDimReductionWithoutFit):
    def __init__(self):
        super().__init__(model=LatentDirichletAllocation(n_components=3), name="LDA3D")