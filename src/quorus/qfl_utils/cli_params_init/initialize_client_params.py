from quorus.logging.custom_slog import print_cust
from quorus.quantum_circuit_funcs.utils.qcnn_code import compute_conv_layers
from quorus.parameter_initialization.qcnn_params_init.qcnn_params_init_func import init_dynamic_qcnn_params
from quorus.qgan_model_supp.models.pca_discriminator import PCADiscriminator
import copy
from quorus.qgan_model_supp.data_proc.pca_rescaler import PCARescaler
from quorus.qgan_model_supp.models.patchquantumgenerator import PatchQuantumGenerator

"""## Initialize client params function"""

def initialize_client_params(clients_config_arg, model_size=4, cur_client_params_dict=None, debug=False, qubits_and_layers_to_add_block_params=[],
                             train_models_parallel=False, n_output_qubits=1, generative=False, use_torch=False, qnode_func=None, device=None, pca_info=None, is_qcnn=True, alt_zeros_init=""):
    """
    Initializes the parameter dictionary storing the parameters for each client.

    Parameters:
      clients_config_arg: The client configuration dictionary used to determine the size and types of parameters that each client will have.
      model_size: Integer representing the shared model size (if that exists) for the subnet initialization case.
      cur_client_params_dict: A dictionary representing a client parameters dictionary, used for previous parameter initialization.
      debug: A Boolean indicating whether or not the parameters should be initialized in debug mode.
      qubits_and_layers_to_add_block_params: A dictionary mapping client sizes to a list of (n_qubits, n_layers) specifying the block parameters
      that this client contains.
      train_models_parallel: A Boolean indicating whether or not the models are trained in parallel (and thus, whether or not we should employ subnet initialization).
      n_output_qubits: An integer specifying the number of output qubits for these models.

    Returns:
      client_params_dict, a dictionary mapping client types (integers, representing the numbers of qubits that each client contains) to a list of parameters for each client
      (the i-th element of the list are the parameters for the i-th client)
    """
    # NOTE: for train_models_parallel, this function only needs to be called ONCE. There is no notion of 'expansion'; so the
    # subsequent checks for 'expansion' in this function should NOT be strictly necessary.
    client_params_dict = {}
    max_client_size = max(clients_config_arg.keys())
    # Iterate over each client type
    for client_type, cfg in clients_config_arg.items():
        # If the client type is smaller than the model size in expansion, or the number of clients is 0, then don't create
        # parameters for models of this type
        if (client_type < model_size and not train_models_parallel) or cfg["num_clients"] == 0:
            continue
        num_clients = cfg["num_clients"]
        client_params_dict[client_type] = []
        # For each client that is to be created of this type, initialize the parameters for this client.
        for client_idx in range(num_clients):
            # NOCHANGE: TOMODIFY, depthFL: inject some argument here to indicate that client_model_size should be dynamically found based on the largest qubits needed for its block params,
            # if not qcnn.
            # ^ client_model_size is never manifested for the non-QCNN case, so is OK to use client_model_size
            # as an alias to diff't client types in terms of number of layers.
            if train_models_parallel:
              client_model_size = client_type
            else:
              client_model_size = model_size
            is_expansion = (cur_client_params_dict is not None)
            # Filter block config: for each (num_qubits, num_layers) tuple, if num_qubits <= client_type, include it.
            qubits_and_layers_to_add_block_params_client = []
            if client_model_size in qubits_and_layers_to_add_block_params:
              qubits_and_layers_to_add_block_params_client = qubits_and_layers_to_add_block_params[client_model_size]
              if generative:
                print_cust(f"initialize_client_params, generative, filtering qubits_and_layers_to_add_block_params_client")
                qubits_and_layers_to_add_block_params_client_filt = []
                for qubits_and_layers_tuple in sorted(qubits_and_layers_to_add_block_params_client, key=lambda x:x[0]):
                  if qubits_and_layers_tuple[0] <= client_model_size:
                    qubits_and_layers_to_add_block_params_client_filt.append(qubits_and_layers_tuple)
                print_cust(f"initialize_client_params, generative, qubits_and_layers_to_add_block_params_client_filt: {qubits_and_layers_to_add_block_params_client_filt}")
                qubits_and_layers_to_add_block_params_client = qubits_and_layers_to_add_block_params_client_filt


            print_cust(f"initialize_client_params, client_idx: {client_idx}, client_model_size: {client_model_size}, n_output_qubits: {n_output_qubits}")
            print_cust(f"initialize_client_params, generative: {generative}")
            conv_layers = compute_conv_layers(client_model_size, n_output_qubits, generative=generative, is_qcnn=is_qcnn)

            conv_params_tuple, pool_params_tuple, final_pool_param, final_params, bias_param, block_params_list = \
                init_dynamic_qcnn_params(client_model_size, conv_layers, debug=debug, zeros_init=is_expansion,
                                         qubits_and_layers_to_add_block_params=qubits_and_layers_to_add_block_params_client, generative=generative, use_torch=use_torch, alt_zeros_init=alt_zeros_init)

            if generative:
              client_discriminator = PCADiscriminator(max_client_size, 4)
              client_qgan_qnode = qnode_func(client_model_size)
              client_qubits_depth_dict = {}
              for qubit_depth_tuple in qubits_and_layers_to_add_block_params_client:
                client_qubits_depth_dict[qubit_depth_tuple[0]] = qubit_depth_tuple[1]
              pca_info_used = copy.deepcopy(pca_info)
              client_pcarescaler = PCARescaler(*pca_info_used)
              client_generator = PatchQuantumGenerator(1, 1.0, client_model_size, 0, client_qgan_qnode, client_qubits_depth_dict, device, 0, True, client_pcarescaler)
              client_gan_models = [client_generator, client_discriminator]

            # TODO: continue here. initialize previous disc/generator.
            # If we want to pre-initialize the parameters for this client, do so.
            if cur_client_params_dict is not None and not train_models_parallel:
                init_trained_params = copy.deepcopy(cur_client_params_dict[client_type][client_idx])
                new_conv_params = list(conv_params_tuple)
                new_conv_params[1:] = list(init_trained_params[0])
                conv_params_tuple = tuple(new_conv_params)
                new_pool_params_tuple = list(pool_params_tuple)
                new_pool_params_tuple[1:] = list(init_trained_params[1])
                pool_params_tuple = tuple(new_pool_params_tuple)
                final_pool_param = init_trained_params[2]
                final_params = init_trained_params[3]
                bias_param = init_trained_params[4]
                # block_params_list = init_trained_params[5]
                init_block_params_list = init_trained_params[5]
                if not generative:
                  for init_block_param_idx, block_params in enumerate(init_block_params_list):
                    block_params_shape = block_params.shape
                    print_cust(f"initialize_client_params, block_params_shape: {block_params_shape}")
                    num_layers_blockparams = block_params_shape[0]
                    num_qubits_blockparams = block_params_shape[1]
                    existing_block_params = block_params_list[init_block_param_idx]
                    if existing_block_params.shape == block_params_shape:
                      block_params_list[init_block_param_idx] = block_params
                  print_cust(f"initialize_client_params, block_params_list: {block_params_list}")
                else:
                  # first time the function is called, we assume we get a block params list. but that is effectively ignored and replaced with our gen and disc.
                  # next time this function is called, I can assume that I'll get passed in a generator and discriminator model. so assume that's the format,
                  # if cur_client_params_dict is not None.
                  print_cust(f"initialize_client_params, generative, init_block_params_list: {init_block_params_list}")
                  existing_generator, existing_discriminator = init_block_params_list[0], init_block_params_list[1]
                  client_discriminator = existing_discriminator
                  client_generator = PatchQuantumGenerator(1, 0.0, client_model_size, 0, client_qgan_qnode, client_qubits_depth_dict, device, 0, True, client_pcarescaler)
                  client_generator.initialize_existing_parameters(existing_generator.q_params)
                  client_gan_models = [client_generator, client_discriminator]

            if generative:
              block_params_list = client_gan_models
              print_cust(f"initialize_client_params, generative, block_params_list: {block_params_list}")

            client_params = (conv_params_tuple, pool_params_tuple, final_pool_param, final_params, bias_param, block_params_list)
            client_params_dict[client_type].append(client_params)
    return client_params_dict