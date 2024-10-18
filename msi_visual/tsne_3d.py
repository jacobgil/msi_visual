from msi_visual.base_dim_reduction import BaseDimReductionWithoutFit
from sklearn.manifold import TSNE

class TSNE3D(BaseDimReductionWithoutFit):
    def __init__(self):
        super().__init__(model=TSNE(n_components=3), name="TSNE3D")