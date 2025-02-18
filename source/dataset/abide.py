import numpy as np
import torch
from .preprocess import StandardScaler
from omegaconf import DictConfig, open_dict
# from source.utils import get_ITS_yeo

def load_abide_data(cfg: DictConfig):

    data = np.load(cfg.dataset.path, allow_pickle=True).item()
    final_pearson = data["FC"]
    final_timedelay = data["TD"]
    final_its = data["ITS"]
    labels = data["label"] -1  # 0: ASD, 1: TD
    site = data["site"]
    age = data["age"]
    sex = ["sex"]
    
    # cc200Toyeo7_idx = np.load("V:/XXX/Area/template/craddock/cc200Toyeo7.npy", allow_pickle=True).item()["cc200Toyeo7_idx"]
    # final_its_yeo = get_ITS_yeo(final_its, cc200Toyeo7_idx)
    
    final_pearson, final_timedelay, final_its, labels = [torch.from_numpy(
        data).float() for data in (final_pearson, final_timedelay, final_its, labels)]

    with open_dict(cfg):

        cfg.dataset.node_sz, cfg.dataset.node_feature_sz = final_pearson.shape[1:]

    return final_pearson, final_timedelay, final_its, labels, site
