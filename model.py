import mygrad as mg
from mynn.layers.dense import dense
from mynn.initializers.glorot_normal import glorot_normal
class Model:
    def __init__(self, d_full=512, d_embed=200):
        self.dense = dense(d_full, d_embed, weight_initializer=glorot_normal, bias=False)

    def __call__(self, descriptors):
        w = self.dense(descriptors)
        norm = mg.sqrt(mg.sum(w ** 2, axis=1, keepdims=True))
        return w / norm
    @property
    def parameters(self):
        return self.dense.parameters
