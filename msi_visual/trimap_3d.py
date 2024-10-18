from msi_visual.base_dim_reduction import BaseDimReductionWithoutFit
import trimap


class Trimap3D(BaseDimReductionWithoutFit):
    def __init__(self):
        super().__init__(model=trimap.TRIMAP(n_dims=3), name="Trimap")