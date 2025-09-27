from quorus.logging.custom_slog import print_cust
from quorus.qnode_funcs.qnode_creation_funcs.create_qnode_qgan import create_qnode_qgan
from quorus.qgan_model_supp.models.patchquantumgenerator import PatchQuantumGenerator
from quorus.qgan_model_supp.models.pca_discriminator import PCADiscriminator

"""## Broadcast parameters function

#### Helper Function, Convert Block Params to Generator
"""

def convert_agg_block_params_generator(aggregated_block_params):
    # This function assumes that aggregated_block_params is meant to be converted to a PatchQuantumGenerator (list of two elements; second element is PCADiscriminator)
    # This function mutates aggregated_block_params.


    # assumed to be a sequence-like obj containing first a dict, and then a PCADiscriminator.
    # focusing on the dict.
    # effectively below code, converts the dict for generator only to a model class, and then
    # sets agg_block_params to be [gen_model, disc_model] (instead of [agg_block_params_dict, disc_model])
    generator_block_params_dict = aggregated_block_params[0]
    block_list = []
    if isinstance(generator_block_params_dict, dict):
      for block_type in sorted(generator_block_params_dict.keys()):
        agg_blocks = generator_block_params_dict[block_type]
        if isinstance(agg_blocks, tuple):
          agg_blocks = agg_blocks[0]
        # assumed to be a list of tensors
        block_list.append(agg_blocks)
    # debug
    for block in block_list:
      print_cust(f"convert_agg_block_params_generator, type(block): {type(block)}")
      print_cust(f"convert_agg_block_params_generator, block.shape: {block.shape}")

    n_qubits_gen_agg = max([block.shape[1] for block in block_list])
    # NOTE: I am using global functions here which is NOT GOOD. can replace later; I am just doing this b/c I don't want to deal with
    # weight copying mutations issues.
    qgan_qnode_agg = create_qnode_qgan(n_qubits_gen_agg)
    qubits_depth_dict_agg = {}
    for block in block_list:
      # assumed that each num_qubits of each block is unique
      qubits_depth_dict_agg[block.shape[1]] = block.shape[0]
    # device_agg = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_agg = "cpu"
    aggregated_generator = PatchQuantumGenerator(1, 1.0, n_qubits_gen_agg, 0, qgan_qnode_agg, qubits_depth_dict_agg, device_agg, 0, True, None)

    aggregated_generator.initialize_existing_parameters(block_list)

    aggregated_block_params[0] = aggregated_generator
    assert isinstance(aggregated_block_params[1], PCADiscriminator), f"broadcast_param_updates, when agg_block_params is NOT a dict, type(aggregated_block_params[1]): {type(aggregated_block_params[1])}"

    return aggregated_block_params