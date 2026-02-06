import torch
from importlib_resources import files
import sgpo
from pathlib import Path
path = files(sgpo)
ckpt = torch.load(path / Path('checkpoints/continuous_ESM/CreiLOV/best_model.ckpt'))
ckpt['hyper_parameters']
hps = ckpt['hyper_parameters']
for key, value in hps.items():
    if '_target_' in value and value['_target_'].split('.')[0] not in ['sgpo', 'torch', 'transformers']:
        value['_target_'] = 'sgpo.' + value['_target_']
    # TODO recurse
torch.save(ckpt, path / Path('checkpoints/continuous_ESM/CreiLOV/best_model.ckpt'))
