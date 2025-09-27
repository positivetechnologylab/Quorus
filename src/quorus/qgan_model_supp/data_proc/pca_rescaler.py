from quorus.logging.custom_slog import print_cust
import torch

import torch.nn.functional as F

"""## Dynamic parameter initialization

### Definition of PCARescaler
"""
class PCARescaler:
    def __init__(self, pca_obj, pca_mins, pca_maxs, inv_pca_mins, inv_pca_maxs, device):
        self.pca_obj = pca_obj
        self.pca_mins = pca_mins
        self.pca_maxs = pca_maxs
        self.inv_pca_min = inv_pca_mins
        self.inv_pca_max = inv_pca_maxs
        self.device = device

    def __str__(self):
        return f"""
        PCARescaler:
        self.pca_obj: {self.pca_obj}
        self.pca_mins: {self.pca_mins}
        self.pca_maxs: {self.pca_maxs}
        self.inv_pca_min: {self.inv_pca_min}
        self.inv_pca_max: {self.inv_pca_max}
        self.device: {self.device}
        """
    __repr__ = __str__

    def rescale_pca_comps(self, gen_data):
        # it is assumed that gen_data is b/w 0 and 1.|
        print_cust(f"PCARescaler, gen_data.shape: {gen_data.shape}")
        gen_data_dim = gen_data.shape[1]
        pca_maxs = self.pca_maxs
        pca_mins = self.pca_mins
        if isinstance(pca_maxs, torch.return_types.max):
            pca_maxs = pca_maxs.values
        if isinstance(pca_mins, torch.return_types.min):
            pca_mins = pca_mins.values
        pca_maxs = pca_maxs[:gen_data_dim]
        pca_mins = pca_mins[:gen_data_dim]

        print_cust(f"PCARescaler, pca_maxs: {pca_maxs}, pca_mins: {pca_mins}")

        pca_comps = gen_data * (pca_maxs - pca_mins) + pca_mins
        print_cust(f"PCARescaler, rescale_pca_comps, pca_comps: {pca_comps}")
        print_cust(f"PCARescaler, type(pca_comps): {type(pca_comps)}")
        print_cust(f"PCARescaler, rescale_pca_comps, torch.min(pca_comps, dim=0): {torch.min(pca_comps, dim=0)}, torch.max(pca_comps, dim=0): {torch.max(pca_comps, dim=0)}")
        # print_cust("[DBG]", os.getpid(), "about to print; rss", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, file=sys.stderr, flush=True)
        # try:
        #   print_cust(f"PCARescaler, rescale_pca_comps, torch.min(pca_comps, dim=0): {torch.min(pca_comps, dim=0)}, torch.max(pca_comps, dim=0): {torch.max(pca_comps, dim=0)}")
        # except Exception as e:
        #   print_cust(f"PCARescaler, exception e: {e}")
        #   raise e
        print_cust(f"PCARescaler, about to return pca_comps")
        # gen_imgs_reconstr = self.pca_obj.inverse_transform(pca_comps)

        # print_cust(f"PCARescaler, rescale_pca_comps, gen_imgs_reconstr: {gen_imgs_reconstr}")

        # print_cust(f"PCARescaler, rescale_pca_comps, gen_imgs_reconstr.min(): {gen_imgs_reconstr.min()}, gen_imgs_reconstr.max(): {gen_imgs_reconstr.max()}")

        # gen_imgs_reconstr_rescaled = (gen_imgs_reconstr - self.inv_pca_min) / (self.inv_pca_max - self.inv_pca_min)

        # print_cust(f"PCARescaler, rescale_pca_comps, after min-max, gen_imgs_reconstr_rescaled.min(): {gen_imgs_reconstr_rescaled.min()}, gen_imgs_reconstr_rescaled.max(): {gen_imgs_reconstr_rescaled.max()}")

        return pca_comps

    def invpca_to_img(self, gen_pca_comps, resc_invpca=True):
        pca_dim = self.pca_obj.n_components
        print_cust(f"PCARescaler, invpca_to_img, resc_invpca: {resc_invpca}")
        print_cust(f"PCARescaler, invpca_to_img, pca_dim: {pca_dim}")

        cur_dim = gen_pca_comps.size(-1)             # length of the last (fast-changing) axis
        if cur_dim < pca_dim:
            pad_right = pca_dim - cur_dim
            # (left, right) for 1-D padding of the last dimension
            gen_pca_comps = F.pad(gen_pca_comps, (0, pad_right), mode="constant", value=0.0)
            print_cust(f"PCARescaler, invpca_to_img, gen_pca_comps.shape: {gen_pca_comps.shape}")
            print_cust(f"PCARescaler, invpca_to_img, after padding, gen_pca_comps: {gen_pca_comps}")

        gen_pca_comps_numpy = gen_pca_comps.detach().cpu().numpy()

        gen_imgs_reconstr_numpy = self.pca_obj.inverse_transform(gen_pca_comps_numpy)

        gen_imgs_reconstr = torch.from_numpy(gen_imgs_reconstr_numpy).to(self.device)

        print_cust(f"PCARescaler, invpca_to_img, gen_imgs_reconstr: {gen_imgs_reconstr}")

        print_cust(f"PCARescaler, invpca_to_img, gen_imgs_reconstr.min(): {gen_imgs_reconstr.min()}, gen_imgs_reconstr.max(): {gen_imgs_reconstr.max()}")

        if resc_invpca:
          print_cust(f"PCARescaler, invpca_to_img, rescaling generated images")
          gen_imgs_reconstr_rescaled = (gen_imgs_reconstr - self.inv_pca_min) / (self.inv_pca_max - self.inv_pca_min)
        else:
          print_cust(f"PCARescaler, invpca_to_img, leaving generated images alone")
          gen_imgs_reconstr_rescaled = gen_imgs_reconstr

        print_cust(f"PCARescaler, invpca_to_img, after min-max, gen_imgs_reconstr_rescaled.min(): {gen_imgs_reconstr_rescaled.min()}, gen_imgs_reconstr_rescaled.max(): {gen_imgs_reconstr_rescaled.max()}")

        return gen_imgs_reconstr_rescaled

    def get_data_components(self):
      return self.pca_obj, self.pca_mins, self.pca_maxs, self.inv_pca_min, self.inv_pca_max, self.device