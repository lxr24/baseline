import jittor as jt
jt.flags.use_cuda = 0 # force CPU for test
from omegaconf import OmegaConf
from src.model.parse import get_model
cfg = OmegaConf.load("configs/model/cvm.yaml")
model = get_model(OmegaConf.to_container(cfg), transform_config={})
print(model)

cfg2 = OmegaConf.load("configs/model/straightpcf.yaml")
model2 = get_model(OmegaConf.to_container(cfg2), transform_config={})
print(model2)
