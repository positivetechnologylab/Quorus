import torch.nn as nn
import torch

from quorus.logging.custom_slog import print_cust
from quorus.quantum_circuit_funcs.circuit_data_extraction.qgan_extraction_funcs import get_expectations

"""### Definition of PatchQuantumGenerator"""

class PatchQuantumGenerator(nn.Module):
    """Quantum generator class for the patch method"""

    def __init__(self, n_generators, q_delta, n_qubits_gen, n_a_qubits_gen, qnode, qubit_depth_dict, device, img_size, gen_pca=True, pca_rescaler=None):
        """
        Args:
            n_generators (int): Number of sub-generators to be used in the patch method.
            q_delta (float, optional): Spread of the random distribution for parameter initialisation.
        """

        super().__init__()

        # NOTE: getting rid of patch for now; can adapt code for improvement later.

        # if existing_params is None:

        #     self.q_params = nn.ParameterList(
        #         [
        #             nn.Parameter(q_delta * torch.rand(q_depth_layer * n_qubits_layer), requires_grad=True)
        #             for n_qubits_layer, q_depth_layer in qubit_depth_dict.items()
        #         ]
        #     )
        # else:
        #     # TODO: ensure that existing_params is indeed a valid PyTorch tensor
        #     self.q_params = nn.ParameterList(
        #         [
        #             nn.Parameter(existing_params, requires_grad=True)
        #             for _ in range(n_generators)
        #         ]
        #     )

        # will assume that existing_params is a list.
        # existing_num_qubits_params = [params.shape[1] for params in existing_params]
        # print_cust(f"PatchQuantumGenerator, existing_num_qubits_params: {existing_num_qubits_params}")
        params_qnode = []

        for n_qubits_layer in sorted(qubit_depth_dict.keys()):
            q_depth_layer = qubit_depth_dict[n_qubits_layer]
            # if n_qubits_layer not in existing_num_qubits_params:
            cur_q_params = nn.Parameter(q_delta * torch.rand(q_depth_layer, n_qubits_layer, 3), requires_grad=True)
            # else:
            #     existing_params_idx = existing_num_qubits_params.index(n_qubits_layer)
            #     cur_q_params = nn.Parameter(existing_params[existing_params_idx], requires_grad=True)
            params_qnode.append(cur_q_params)

        print_cust(f"PatchQuantumGenerator, params_qnode: {params_qnode}")

        self.q_params = nn.ParameterList(params_qnode)

        self.n_generators = n_generators
        self.n_qubits_gen = n_qubits_gen
        self.n_a_qubits_gen = n_a_qubits_gen
        self.qnode = qnode
        self.qubit_depth_dict = qubit_depth_dict
        self.device = device
        self.img_size = img_size
        self.gen_pca = gen_pca
        self.pca_rescaler = pca_rescaler
        self.q_delta = q_delta

    def __str__(self):
        return f"""
        self.q_params: {self.q_params}
        self.n_generators: {self.n_generators}
        self.q_delta: {self.q_delta}
        self.n_qubits_gen: {self.n_qubits_gen}
        self.n_a_qubits_gen: {self.n_a_qubits_gen}
        self.qnode: {self.qnode}
        self.qubit_depth_dict: {self.qubit_depth_dict}
        self.device: {self.device}
        self.img_size: {self.img_size}
        self.gen_pca: {self.gen_pca}
        self.pca_rescaler: {self.pca_rescaler}
        self.state_dict(): {self.state_dict()}
        """
    __repr__ = __str__

    def initialize_existing_parameters(self, existing_params):
      # assume that existing params is NOT larger than current param sizes.
      if existing_params is None:
        return
      current_params_qubits = [params.shape[1] for params in self.q_params]
      for existing_params_tens in existing_params:
        print_cust(f"initialize_existing_parameters, type(existing_params_tens): {type(existing_params_tens)}")
        existing_params_tens_qub = existing_params_tens.shape[1]
        if existing_params_tens_qub in current_params_qubits:
          current_param_idx = current_params_qubits.index(existing_params_tens_qub)
          existing_param_nn = nn.Parameter(existing_params_tens, requires_grad=True)
          self.q_params[current_param_idx] = existing_param_nn

      print_cust(f"PatchQuantumGenerator, initialize_existing_parameters, self.q_params: {self.q_params}")
      print_cust(f"PatchQuantumGenerator, initialize_existing_parameters, self.state_dict(): {self.state_dict()}")

    def forward(self, x, n_qubits_gen_forward=None, alpha=1.0):
        # Size of each sub-generator output
        if n_qubits_gen_forward is None:
            n_qubits_gen_forward = self.n_qubits_gen
        patch_size = self.n_qubits_gen * self.n_generators
        print_cust(f"PatchQuantumGenerator, forward, len(self.q_params): {len(self.q_params)}")

        # Create a Tensor to 'catch' a batch of images from the for loop. x.size(0) is the batch size.
        images = torch.Tensor(x.size(0), 0).to(self.device)
        print_cust(f"PatchQuantumGenerator, forward, images.shape: {images.shape}")

        # # Iterate over all sub-generators
        # for params in self.q_params:

        # def tape(name):
        #     return lambda g: grad_log_glob.setdefault(name, g.norm().item())
        # def save_norm(name):
        #     def hook(g):
        #         grad_log_glob[name] = (g.norm().item(), g)
        #         return g
        #     return hook

        # TODO: adapt this code for multiple subgenerators
        # Create a Tensor to 'catch' a batch of the patches from a single sub-generator
        patches = torch.Tensor(0, patch_size).to(self.device)
        print_cust(f"PatchQuantumGenerator, forward, patches.shape: {patches.shape}")
        # for elem in x:
        #     if self.gen_pca:
        #         q_out = get_expectations(elem, self.qnode, n_qubits_gen_forward, self.q_params).float().unsqueeze(0)
        #         q_out = (q_out + 1.0) / 2.0
        #         print_cust(f"PatchQuantumGenerator, forward, q_out: {q_out}")
        #         # q_out.retain_grad()
        #         # q_out.register_hook(save_norm("before"))
        #     # else:
        #     #     q_out = partial_measure(elem, self.q_params, self.qnode, n_qubits_func=n_qubits_gen_forward, n_a_qubits_func=self.n_a_qubits_gen, qubit_depth_dict=self.qubit_depth_dict, alpha=alpha).float().unsqueeze(0)

        #     patches = torch.cat((patches, q_out))
        #     print_cust(f"PatchQuantumGenerator, forward, patches.shape: {patches.shape}")

        if self.gen_pca:
          q_out = get_expectations(x, self.qnode, n_qubits_gen_forward, self.q_params).float()
          q_out = (q_out + 1.0) / 2.0
          print_cust(f"PatchQuantumGenerator, forward, q_out: {q_out}")
          patches = q_out.T

        # Each batch of patches is concatenated with each other to create a batch of images
        print_cust(f"PatchQuantumGenerator, patches.shape: {patches.shape}")
        print_cust(f"PatchQuantumGenerator, patches.device: {patches.device}")
        print_cust(f"PatchQuantumGenerator, forward, patches: {patches}")
        images = torch.cat((images, patches), 1)
        # print_cust(f"PatchQuantumGenerator, forward, images.shape: {patches.shape}")

        # images = (images + 1) / 2
        print_cust(f"PatchQuantumGenerator, forward, images.shape: {images.shape}")
        if self.gen_pca:
            print_cust(f"PatchQuantumGenerator, is gen_pca, so rescaling expectations to PCA components")
            reconstr_images = self.pca_rescaler.rescale_pca_comps(images)
            print_cust(f"PatchQuantumGenerator, reconstr_images.min(): {reconstr_images.min()}, reconstr_images.max(): {reconstr_images.max()}")
            reconstr_images_abs = reconstr_images.abs()
            print_cust(f"PatchQuantumGenerator, reconstr_images_abs.min(): {reconstr_images_abs.min()}, reconstr_images_abs.max(): {reconstr_images_abs.max()}")
            reconstr_images_sum = reconstr_images_abs.sum()
            print_cust(f"PatchQuantumGenerator, reconstr_images_sum: {reconstr_images_sum}")
            # self.q_params.zero_grad()
            # print_cust(f"PatchQuantumGenerator, forward, self.q_params[0].grad: {self.q_params[0].grad}")
            # reconstr_images_sum.backward()
            # self.q_params.zero_grad()
            # print_cust(f'PatchQuantumGenerator, forward, after backward(), self.q_params[0].grad: {self.q_params[0].grad}')
            # print_cust(f"PatchQuanutmGenerator, forward, after backward(), np.linalg.norm(self.q_params[0].grad): {np.linalg.norm(self.q_params[0].grad)}")
            # reconstr_images.retain_grad()
            # reconstr_images.register_hook(save_norm("after"))
        else:
            reconstr_images = images
        return reconstr_images

    def get_data_components(self):
      return (self.n_generators, self.q_delta, self.n_qubits_gen, self.n_a_qubits_gen, None, self.qubit_depth_dict, self.device, self.img_size, self.gen_pca, self.pca_rescaler.get_data_components())