from quorus.logging.custom_slog import print_cust
import torch
import torch.nn as nn

"""## Initialize Client Optimizers"""

def initialize_client_optimizers(client_params_dict, lr_gen=0.004, lr_disc=0.001, gen_betas=(0.5, 0.9), disc_betas=(0.5, 0.9), optim_type="sgd", existing_optims_dict=None, generative=False, is_qcnn=False, opt_layers=[-1]):
  """
  Initializes the optimizer dictionary storing the optimizers for each client.

  Parameters:
    client_params_dict: A dictionary containing the parameters for all the clients.
    lr_gen: A float that is the learning rate for the optimizer for the generator.
    lr_disc: A float that is the learning rate for the optimizer for the discriminator (or the classifier, in the classifier case).
    gen_betas: A tuple of floats that are (beta1, beta2) for the generator optimizer.
    disc_betas: A tuple of floats that are (beta1, beta2) for the discriminator optimizer.
    optim_type: A string representing the optimizer type. Supported are "sgd" or "adam".
    existing_optims_dict: A dictionary containing the optimizer information for each client, or None if it does not yet exist.
    generative: A boolean indicating whether or not the QGAN setup is used.
    is_qcnn: A boolean indicating whether or not the QCNN setup is used.
    opt_layers: A list of integers specifying the indices of layers to optimize over.

  Returns:
    client_optims_dict, a dictionary mapping client types (integers, representing the numbers of qubits that each client contains) to a list of parameters for each client
    (the i-th element of the list are the optimizers for the i-th client)
  """
  print_cust(f"initialize_client_optimizers, generative: {generative}, is_qcnn: {is_qcnn}, opt_layers: {opt_layers}")

  client_optimizers_dict = {}
  for client_type, client_params_list in client_params_dict.items():
    client_optimizers_dict[client_type] = []
    for client_idx, client_params in enumerate(client_params_list):
      if generative:
        curr_cli_gen = client_params[5][0]
        curr_cli_disc = client_params[5][1]
        if optim_type == "sgd":
          curr_cli_gen_opt = torch.optim.SGD(nn.ParameterList([curr_cli_gen.q_params[-1]]), lr=lr_gen)
          curr_cli_disc_opt = torch.optim.SGD(curr_cli_disc.parameters(), lr=lr_disc)
        elif optim_type == "adam":
          curr_cli_gen_opt = torch.optim.Adam(nn.ParameterList([curr_cli_gen.q_params[-1]]), lr=lr_gen, betas=gen_betas)
          curr_cli_disc_opt = torch.optim.Adam(curr_cli_disc.parameters(), lr=lr_disc, betas=disc_betas)
        if existing_optims_dict is not None:
          existing_cli_disc_opt = existing_optims_dict[client_type][client_idx][5][1]
          print_cust(f"initialize_client_optimizers, existing_cli_disc_opt: {existing_cli_disc_opt}")
          curr_cli_disc_opt = existing_cli_disc_opt
        # client_optimizers_dict[client_type].append([curr_cli_gen_opt, curr_cli_disc_opt])
        cli_optims = [[] for _ in range(len(client_params))]
        print_cust(f"initialize_client_optimizers, cli_optims: {cli_optims}")
        cli_optims[5].append(curr_cli_gen_opt)
        cli_optims[5].append(curr_cli_disc_opt)
        print_cust(f"initialize_client_optimizers, cli_optims: {cli_optims}")
      elif not is_qcnn:
        curr_cli_blockparams = client_params[5]
        list_params = []
        if opt_layers is None:
          list_params = [nn.Parameter(block_param, requires_grad=True) for block_param in curr_cli_blockparams]
        else:
          for layer_idx in opt_layers:
            layer_opt_param = nn.Parameter(curr_cli_blockparams[layer_idx], requires_grad=True)
            list_params.append(layer_opt_param)
        if optim_type == "sgd":
          # NOTE: these are technically pointing to the WRONG model; BUT, because during train_client, I recreate
          # the optimizers, the actual optimizers themselves during optimization should point to the right actual
          # parameter objects. These are mainly just to store the 'architecture' of the optimizer based on the model
          # params.
          # TODO, layers, 10:49 PM 8/1: continue here
          curr_cli_blockparams_opt = torch.optim.SGD(nn.ParameterList(list_params), lr=lr_disc)
        elif optim_type == "adam":
          curr_cli_blockparams_opt = torch.optim.Adam(nn.ParameterList(list_params), lr=lr_disc, betas=disc_betas)
        # TOMODIFY, layers: need to have better expansion logic here, IF you want to add a param group; need to compute
        # differences and stuff.
        # if existing_optims_dict is not None:
        #   existing_cli_blockparams_opt = existing_optims_dict[client_type][client_idx][5]
        #   print_cust(f"initialize_client_optimizers, existing_cli_blockparams_opt: {existing_cli_blockparams_opt}")
        #   curr_cli_blockparams_opt = existing_cli_blockparams_opt
        cli_optims = [None for _ in range(len(client_params))]
        print_cust(f"initialize_client_optimizers, cli_optims: {cli_optims}")
        cli_optims[5] = curr_cli_blockparams_opt
        print_cust(f"initialize_client_optimizers, cli_optims: {cli_optims}")

      else:
        # not implemented
        cli_optims = [None for _ in range(len(client_params))]

      client_optimizers_dict[client_type].append(cli_optims)
  print_cust(f"initialize_client_optimizers, client_optimizers_dict: {client_optimizers_dict}")
  return client_optimizers_dict