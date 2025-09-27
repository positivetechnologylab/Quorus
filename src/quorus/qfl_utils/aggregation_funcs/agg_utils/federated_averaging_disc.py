from quorus.logging.custom_slog import print_cust
import copy
import torch

"""#### Discriminator Aggregation Functions

##### Discriminator Aggregation Helper
"""

# import copy, torch
from collections import OrderedDict

def fedavg_disc(models, weights=None):
    """
    models  : List[nn.Module]  – client models after local training
    weights : List[int|float] – sample counts or None for plain mean
    returns : nn.Module       – new global model
    """
    # this assumes that models[0] exists
    global_model = copy.deepcopy(models[0])      # keep architecture
    global_dict  = OrderedDict()
    print_cust(f"fedavg_disc, global_model: {global_model}")
    print_cust(f"fedavg_disc, models: {models}")
    print_cust(f"fedavg_disc, weights: {weights}")

    # 1. stack same-shaped tensors from every client
    for key in global_model.state_dict().keys():
        stacked = torch.stack([m.state_dict()[key].float().cpu()
                               for m in models], dim=0)
        print_cust(f"fedavg_disc, stacked: {stacked}")
        if weights is None:                          # unweighted
            global_dict[key] = stacked.mean(dim=0)
        else:                                        # weighted
            w = torch.tensor(weights, dtype=stacked.dtype)
            print_cust(f"fedavg_disc, w: {w}")
            w = w / w.sum()
            print_cust(f"fedavg_disc, after summing, w: {w}")                          # normalise
            # TODO: test this!!!!!
            global_dict[key] = (stacked * w.view(-1, *([1]*
                                      (stacked.dim()-1)))).sum(dim=0)

        print_cust(f"fedavg_disc, key: {key}, global_dict[key]: {global_dict[key]}")

    # 2. load averaged parameters and return
    global_model.load_state_dict(global_dict, strict=True)
    print_cust(f"fedavg_disc, global_model after loading state dict, global_model: {global_model}")
    return global_model