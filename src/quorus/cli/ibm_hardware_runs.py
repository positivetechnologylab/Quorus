import pennylane as qml
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
import torch

from quorus.logging.custom_slog import print_cust
from quorus.metrics_funcs.agg_metrics_func.agg_metrics_func_eval import compute_metrics_angle_param_batch
from quorus.qnode_funcs.qnode_creation_funcs.create_qnode_cheating import create_qnode_qcnn_multieval_cheating
from quorus.qnode_funcs.qnode_creation_funcs.create_qnode_multirun import create_qnode_qcnn_multieval
from quorus.qnode_funcs.qnode_creation_funcs.create_qnode_qcnn import create_qnode_qcnn
from quorus.qnode_funcs.qnode_creation_funcs.create_qnode_singlerun_multiprobs import create_qnode_qcnn_singleeval
from quorus.qnode_funcs.qnode_creation_funcs.create_qnode_singlerun_tunnel import create_qnode_qcnn_singleeval_tunneldown
from quorus.qnode_funcs.qnode_run_funcs.multieval_run import run_multiprob_qnode
from quorus.qnode_funcs.qnode_run_funcs.singleeval_run_tunnel import run_singleeval_tunneldown_qnode
from quorus.qnode_funcs.template_funcs.qcnn_template import qcnn_template

# Make a simple Qiskit Aer noise model
noise_model = NoiseModel()
noise_model.add_all_qubit_quantum_error(depolarizing_error(0.001, 1), ['rz','sx','x'])
noise_model.add_all_qubit_quantum_error(depolarizing_error(0.01, 2), ['cx'])

# Option A: build an Aer backend with noise, pass it to the PL device
aer_backend = AerSimulator(method="density_matrix", noise_model=noise_model)

dev_ibm = qml.device("qiskit.aer", wires=2, backend=aer_backend, shots=10_000)

@qml.qnode(dev_ibm)
def ibm_noisy_qnode(theta):
    qml.RX(theta, 0)
    qml.CNOT([0, 1])
    return qml.probs(wires=[0, 1])

print(ibm_noisy_qnode(0.0))

# %%




# %%
import importlib

# %%
import qiskit

# %%
importlib.reload(qiskit)

# %%
import qiskit, qiskit_aer, pennylane, pennylane_qiskit
print("qiskit:", qiskit.__version__)
print("aer:", qiskit_aer.__version__)
print("pennylane:", pennylane.__version__)
print("pennylane-qiskit:", pennylane_qiskit.__version__)

# %%
from dotenv import load_dotenv

# %%
DOTENV_PATH = ".env"

# %%
load_dotenv(dotenv_path=DOTENV_PATH)

# %%
import os

# %%
token = os.getenv("IBMQ_TOKEN")

# %%
ibmq_crn = os.getenv("IBMQ_CRN")

# %%
from qiskit_ibm_runtime import QiskitRuntimeService


# %%
service = QiskitRuntimeService(channel='ibm_cloud', token=token, instance=ibmq_crn)  # reads saved account; see section 3

# %%
service.backends()

backend = service.least_busy(operational=True, simulator=False, min_num_qubits=2)
noise_model = NoiseModel.from_backend(backend)


backend

aer_backend_realnoise = AerSimulator(noise_model=noise_model)
dev_ibm_realnoise = qml.device("qiskit.aer", wires=2, backend=aer_backend_realnoise, shots=10_000)

@qml.qnode(dev_ibm_realnoise)
def ibm_noisy_qnode_realnoise(theta):
    qml.RX(theta, 0)
    qml.CNOT([0,1])
    return qml.probs(wires=[0,1])

print("IBM/Aer probs:", ibm_noisy_qnode_realnoise(0.0))

from datetime import datetime
import pickle, os

service = QiskitRuntimeService(channel='ibm_cloud', token=token, instance=ibmq_crn)  # uses your saved account (new platform)
backend = service.backend("ibm_fez")

# %%
backend


aer = AerSimulator.from_backend(
    backend,
    noise_model=NoiseModel.from_backend(backend)
)


dev_qiskit_comp = qml.device(
    "qiskit.aer",
    wires=2,
    backend=aer,
    shots=20_000,
    optimization_level=3,      # Qiskit transpiler preset level
    layout_method="sabre",     # good general purpose choices
    routing_method="sabre",
)


@qml.qnode(dev_qiskit_comp)
def circuit(theta):
    qml.RX(theta, 0)
    qml.CNOT([0, 1])
    return qml.probs(wires=[0, 1])

print(circuit(0.0))


from pathlib import Path


def run_singleeval_qnode_cust(input_angles, params, qnode):
  print_cust(f"run_singleeval_qnode_cust, params: {params}")
  output_probs_list = qnode(input_angles, params)
  # print(qml.draw(qnode, level="device")(input_angles, params))
  # assumed to be a list of probabilities for each qubit output, because the QC should be using ancillas and return
  # a list of torch tensors.
  output_probs_tensors = torch.stack(output_probs_list)
  print_cust(f"run_singleeval_qnode_cust, output_probs_tensors: {output_probs_tensors}, output_probs_tensors.shape: {output_probs_tensors.shape}")
  return output_probs_tensors


def create_qnode_qcnn_noisy(n_data, conv_layers, expansion_data, n_classes=2, pennylane_interface="autograd", num_ancillas=0, layer_types_list=[], cheating=False, tunn_down=False, aer_backend=None):
    """
    Creates a QNode of the specified size.

    Parameters:
      n_data: an integer representing the number of qubits for this QNode
      conv_layers: an integer representing the number of convolutional layers in this QNode
      expansion_data: a list of (encoded_angles, reversed_indices) that is used for identity initialization
      n_classes: an integer representing the number of output classes for this QCNN

    Returns:
      qnode: a QNode object satisfying the above constraints.
    """
    n_qubits = n_data
    # dev = qml.device("default.qubit", wires=n_qubits)
    # aer_backend.set_options(noise_model=None)
    dev = qml.device(
        "qiskit.aer",
        wires=n_qubits,
        backend=aer_backend,
        shots=None,                # TOMODIFY: Hardcoded for now
        optimization_level=0,      # forwarded to qiskit.transpile
        seed_transpiler=1,
        scheduling_method="asap",
        precision="single",
        runtime_parameter_bind_enable=True
    )
    # print_cust(f"create_qnode_qcnn, pennylane_interface: {pennylane_interface}, num_ancillas: {num_ancillas}, cheating: {cheating}, tunn_down: {tunn_down}")
    circuit = partial(qcnn_template, n_qubits=n_qubits, expansion_data=expansion_data, n_classes=n_classes, num_ancillas=num_ancillas, layer_types_list=layer_types_list, cheating=cheating, tunn_down=tunn_down)

    return qml.qnode(dev, interface=pennylane_interface)(circuit)


def create_qnode_qcnn_realhardware(n_data, conv_layers, expansion_data, n_classes=2, pennylane_interface="autograd", num_ancillas=0, layer_types_list=[], cheating=False, tunn_down=False, aer_backend=None):
    """
    Creates a QNode of the specified size.

    Parameters:
      n_data: an integer representing the number of qubits for this QNode
      conv_layers: an integer representing the number of convolutional layers in this QNode
      expansion_data: a list of (encoded_angles, reversed_indices) that is used for identity initialization
      n_classes: an integer representing the number of output classes for this QCNN

    Returns:
      qnode: a QNode object satisfying the above constraints.
    """
    n_qubits = n_data
    # dev = qml.device("default.qubit", wires=n_qubits)
    # aer_backend.set_options(noise_model=None)
    dev = qml.device(
        "qiskit.remote",
        wires=n_qubits,
        backend=aer_backend,
        shots=1000,                # TOMODIFY: Hardcoded for now
        optimization_level=0,      # forwarded to qiskit.transpile
        # resilience_level=0,
        seed_transpiler=1,
        scheduling_method="asap",
        # runtime_parameter_bind_enable=True
    )
    # print_cust(f"create_qnode_qcnn, pennylane_interface: {pennylane_interface}, num_ancillas: {num_ancillas}, cheating: {cheating}, tunn_down: {tunn_down}")
    circuit = partial(qcnn_template, n_qubits=n_qubits, expansion_data=expansion_data, n_classes=n_classes, num_ancillas=num_ancillas, layer_types_list=layer_types_list, cheating=cheating, tunn_down=tunn_down)

    return qml.qnode(dev, interface=pennylane_interface)(circuit)

# %%

def create_qnode_qcnn_singleeval_noisysim(n_data, conv_layers, expansion_data, n_classes=2, pennylane_interface="autograd", block_layers=5, layer_types_list=[], aer_backend=None):
    """
    Creates a multi-evaluation QNode of the specified size.

    Parameters:
      n_data: an integer representing the number of qubits for this QNode
      conv_layers: an integer representing the number of convolutional layers in this QNode
      expansion_data: a list of (encoded_angles, reversed_indices) that is used for identity initialization
      n_classes: an integer representing the number of output classes for this QCNN

    Returns:
      qnode: a QNode object satisfying the above constraints.
    """
    # TOMODIFY, DepthFL, HACK: inject the number of additional ancillas that I want my qnode to have.
    # ^ currently, is hardcoded as a default arg to see if it works.
    print_cust(f"create_qnode_qcnn_singleeval, n_data: {n_data}, block_layers: {block_layers}")
    # TOMODIFY, DepthFL: for now, assuming (or maybe not? it's just named this way) that number of
    # block layers = number of ancillas in the circuit.
    qcnn_qnode = create_qnode_qcnn_noisy(n_data + block_layers, conv_layers, expansion_data, n_classes=n_classes, pennylane_interface=pennylane_interface, num_ancillas=block_layers, layer_types_list=layer_types_list, aer_backend=aer_backend)
    # return a partial to run_multiprob_qnode
    multiprob_fn = partial(run_singleeval_qnode_cust, qnode=qcnn_qnode)
    return multiprob_fn

def create_qnode_qcnn_multieval_noisysim(n_data, conv_layers, expansion_data, n_classes=2, pennylane_interface="autograd", layer_types_list=[], aer_backend=None):
    """
    Creates a multi-evaluation QNode of the specified size.

    Parameters:
      n_data: an integer representing the number of qubits for this QNode
      conv_layers: an integer representing the number of convolutional layers in this QNode
      expansion_data: a list of (encoded_angles, reversed_indices) that is used for identity initialization
      n_classes: an integer representing the number of output classes for this QCNN

    Returns:
      qnode: a QNode object satisfying the above constraints.
    """
    qcnn_qnode = create_qnode_qcnn_noisy(n_data, conv_layers, expansion_data, n_classes=n_classes, pennylane_interface=pennylane_interface, layer_types_list=layer_types_list, aer_backend=aer_backend)
    # return a partial to run_multiprob_qnode
    multiprob_fn = partial(run_multiprob_qnode, qnode=qcnn_qnode)
    return multiprob_fn

def create_qnode_qcnn_singleeval_tunneldown_noisysim(n_data, conv_layers, expansion_data, n_classes=2, pennylane_interface="autograd", layer_types_list=[], aer_backend=None):
    """
    Creates a multi-evaluation QNode of the specified size.

    Parameters:
      n_data: an integer representing the number of qubits for this QNode
      conv_layers: an integer representing the number of convolutional layers in this QNode
      expansion_data: a list of (encoded_angles, reversed_indices) that is used for identity initialization
      n_classes: an integer representing the number of output classes for this QCNN

    Returns:
      qnode: a QNode object satisfying the above constraints.
    """
    # TOMODIFY, DepthFL, HACK: inject the number of additional ancillas that I want my qnode to have.
    # ^ currently, is hardcoded as a default arg to see if it works.
    print_cust(f"create_qnode_qcnn_singleeval_tunneldown_noisysim, n_data: {n_data}, block_layers")
    # TOMODIFY, DepthFL: for now, assuming (or maybe not? it's just named this way) that number of
    # block layers = number of ancillas in the circuit.
    # create_qnode_qcnn(n_data, conv_layers, expansion_data, n_classes=2, pennylane_interface="autograd", num_ancillas=0, layer_types_list=[], cheating=False):
    qcnn_qnode = create_qnode_qcnn_noisy(n_data, conv_layers, expansion_data, n_classes=n_classes, pennylane_interface=pennylane_interface, layer_types_list=layer_types_list, tunn_down=True, aer_backend=aer_backend)
    # return a partial to run_multiprob_qnode
    multiprob_fn = partial(run_singleeval_tunneldown_qnode, qnode=qcnn_qnode)
    return multiprob_fn

# %%
def create_qnode_qcnn_singleeval_tunneldown_realhardware(n_data, conv_layers, expansion_data, n_classes=2, pennylane_interface="autograd", layer_types_list=[], aer_backend=None):
    """
    Creates a multi-evaluation QNode of the specified size.

    Parameters:
      n_data: an integer representing the number of qubits for this QNode
      conv_layers: an integer representing the number of convolutional layers in this QNode
      expansion_data: a list of (encoded_angles, reversed_indices) that is used for identity initialization
      n_classes: an integer representing the number of output classes for this QCNN

    Returns:
      qnode: a QNode object satisfying the above constraints.
    """
    # TOMODIFY, DepthFL, HACK: inject the number of additional ancillas that I want my qnode to have.
    # ^ currently, is hardcoded as a default arg to see if it works.
    print_cust(f"create_qnode_qcnn_singleeval_tunneldown_realhardware, n_data: {n_data}, block_layers")
    # TOMODIFY, DepthFL: for now, assuming (or maybe not? it's just named this way) that number of
    # block layers = number of ancillas in the circuit.
    # create_qnode_qcnn(n_data, conv_layers, expansion_data, n_classes=2, pennylane_interface="autograd", num_ancillas=0, layer_types_list=[], cheating=False):
    qcnn_qnode = create_qnode_qcnn_realhardware(n_data, conv_layers, expansion_data, n_classes=n_classes, pennylane_interface=pennylane_interface, layer_types_list=layer_types_list, tunn_down=True, aer_backend=aer_backend)
    # return a partial to run_multiprob_qnode
    multiprob_fn = partial(run_singleeval_tunneldown_qnode, qnode=qcnn_qnode)
    return multiprob_fn

# %%


# %% [markdown]
# ## Evaluate params for a qnode

# %%
from functools import partial

# %%
def eval_params_qnode(agg_params_used, testing_data, multiclassifier_type, num_classes=2, local_pca=True, shared_pca=True, shots=1000, aer_backend=None):
    # TODO: put this in an alternative function
    # NOTE, layers: this is a logical override.
    (X_test_client_angle, y_test) = testing_data

    num_qubits_bps = []
    for block_param in agg_params_used[5]:
        num_qubits_bps.append(block_param.shape[1])

    cur_shared_model_size = max(num_qubits_bps)
    if local_pca or shared_pca:
        print_cust(f"run_qfl_experiments_parallel_multiprocess, not generative and not is_qcnn, cur_shared_model_size: {cur_shared_model_size}")
        X_test_client_angle = X_test_client_angle[:, :cur_shared_model_size]
    # TOMODIFY, depthFL: change the name of this variable (for key into qubits_and_layer_types_block_params)


    # NOTE: hardcoded
    # TODO: allow for diff circ_type 's
    qubits_and_layer_types_block_params = {10: ["v_shape"],
     11: ["v_shape", "v_shape"],
     12: ["v_shape", "v_shape", "v_shape"],
     13: ["v_shape", "v_shape", "v_shape", "v_shape"],
     14: ["v_shape", "v_shape", "v_shape", "v_shape", "v_shape"]}

    qnode_builder = create_qnode_qcnn      # tiny alias for pickling friendliness
    if multiclassifier_type == "multirun":
        # NOTE, depthFL: this is NOT a real qnode; its a caller that calls a qnode MULTIPLE times.
        qnode_builder = create_qnode_qcnn_multieval
    elif multiclassifier_type == "ancilla_endmeas":
        qnode_builder = create_qnode_qcnn_singleeval
    elif multiclassifier_type == "cheating":
        qnode_builder = create_qnode_qcnn_multieval_cheating
    elif multiclassifier_type == "tunnel_down":
        qnode_builder = create_qnode_qcnn_singleeval_tunneldown
    elif multiclassifier_type == "ancilla_endmeas_noisy":
        qnode_builder = partial(create_qnode_qcnn_singleeval_noisysim, aer_backend=aer_backend)
    elif multiclassifier_type == "multirun_noisy":
        qnode_builder = partial(create_qnode_qcnn_multieval_noisysim, aer_backend=aer_backend)
    elif multiclassifier_type == "tunnel_down_noisy":
        qnode_builder = partial(create_qnode_qcnn_singleeval_tunneldown_noisysim, aer_backend=aer_backend)
    elif multiclassifier_type == "tunnel_down_realhardware":
        qnode_builder = partial(create_qnode_qcnn_singleeval_tunneldown_realhardware, aer_backend=aer_backend)

    largest_clisize_layertypes = max(qubits_and_layer_types_block_params)
    print(f"load_mod_params_testingdata: {qnode_builder}")
    print_cust(f"run_qfl_experiments_parallel_multiprocess, largest_clisize_layertypes: {largest_clisize_layertypes}")
    layer_types_list_largest = qubits_and_layer_types_block_params[largest_clisize_layertypes]
    # TOMODIFY, layers: supply layer_types_list to this qnode_builder
    conv_layers = 0
    pennylane_interface = "torch"
    qnode_test = qnode_builder(cur_shared_model_size, conv_layers, [], n_classes=num_classes, pennylane_interface=pennylane_interface, layer_types_list=layer_types_list_largest)

    print(f"run_qfl_experiments_parallel_multiprocess, X_test_client_angle: {X_test_client_angle}")
    print(f"run_qfl_experiments_parallel, X_test_client_angle.shape: {X_test_client_angle.shape}, type(X_test_client_angle): {type(X_test_client_angle)}, y_test.shape: {y_test.shape}, type(y_test): {type(y_test)}")
    print(f"run_qfl_experiments_parallel, X_test_client_angle.min(): {X_test_client_angle.min()}, X_test_client_angle.max(): {X_test_client_angle.max()}, y_test.min(): {y_test.min()}, y_test.max(): {y_test.max()}")

    local_batch_size = 32

    math_int = torch
    
    test_acc, test_acc_stdev, test_acc_topk, test_loss, avg_acc_classifiers, std_acc_classifiers, top_k_accuracies_classifiers, avg_loss_classifiers, gen_all_probs = compute_metrics_angle_param_batch(agg_params_used, X_test_client_angle, y_test, layers=conv_layers, shots=shots, batch_size=local_batch_size, qnode=qnode_test, math_int=math_int)


    print(f"load_mod_params_testingdata, test_acc, test_acc_stdev, test_acc_topk, test_loss, avg_acc_classifiers, std_acc_classifiers, top_k_accuracies_classifiers, avg_loss_classifiers, gen_all_probs: {test_acc, test_acc_stdev, test_acc_topk, test_loss, avg_acc_classifiers, std_acc_classifiers, top_k_accuracies_classifiers, avg_loss_classifiers, gen_all_probs}")
    
    print(f"load_mod_params_testingdata, test_acc: {test_acc}")

    return test_acc, test_acc_stdev, test_acc_topk, test_loss, avg_acc_classifiers, std_acc_classifiers, top_k_accuracies_classifiers, avg_loss_classifiers, gen_all_probs

# %% [markdown]
# ## Load in the params and testing data

# %%
def load_mod_params_testingdata(dataset, classes, multiclassifier_type, seed, model_size=14, round_num=999, base_model_size=10, circ_type="v_shape", shots=1000, folder_prefix="."):
    # NOTE: I can use this same code to compute the ensemble test acc for HeteroFL, if possible.

    classes_str = '_'.join(classes)
    # multiclassifier_type = "ancilla_endmeas"
    folder_name = f"{folder_prefix}/data_logs_{dataset}_classes_{classes_str}_n_samples_3640_n_train_640_qfl_gen_1000rounds_le_1_bs_32_opt_adam_mpd_mlt_lg_0.004_ld_0.001_dqdm_ba_sp_qcnn_False_ba_sm_ae_False_dr_nce_mc_{multiclassifier_type}_mp_True_tclip_aggshared_{seed}_{circ_type}_ldd_1.0_2_6l"
    print(f"load_mod_params_testingdata, folder_name: {folder_name}")
    # if circ_type == "revstair_vshape":
    #     folder_name += "_bf"
    if multiclassifier_type == "tunnel_down":
        folder_name += "_bfc"
    pickle_path = Path(f"{folder_name}/result_datalogs.pkl")
    pickle_exists = pickle_path.is_file()
    if not pickle_exists:
        print(f"load_mod_params_testingdata, dataset: {dataset}, circ_type: {circ_type}, classes: {classes}, seed: {seed}: folder_name: {folder_name}, result_datalogs.pkl doesn't exist, continuing")
        return
    with open(f"{folder_name}/result_datalogs.pkl", "rb") as file:
        data_logs_loaded_test_depthfl = pickle.load(file)
    print(f"load_mod_params_testingdata, data_logs_loaded_test_depthfl: {data_logs_loaded_test_depthfl}")
    # first, get the aggregated model of interest
    agg_params = data_logs_loaded_test_depthfl[round_num]["aggregated_params"]
    agg_params_used = list(agg_params)
    bp_list = []
    keys_sorted = sorted(list(agg_params[5].keys()), key=lambda x:x[1])
    print(f"load_mod_params_testingdata, keys_sorted: {keys_sorted}")
    for key_val in keys_sorted:
        layer_num = key_val[1]
        if (base_model_size + layer_num) <= model_size:
            bp_list.append(agg_params[5][key_val])
    print(f"load_mod_params_testingdata, bp_list: {bp_list}")
    # sanity
    for block_param in bp_list:
        print(f"load_mod_params_testingdata, type(block_param): {type(block_param)}")
    agg_params_used[5] = bp_list
    agg_params_used = tuple(agg_params_used)

    # now, set up the other stuff for running

    # NOTE, layers: only doing this for the not is_qcnn case.
    # TOMODIFY, layers: change cur_shared_model_size here. and also, subset the testing data here.
    num_qubits_bps = []
    for block_param in agg_params_used[5]:
        num_qubits_bps.append(block_param.shape[1])

    print(f"load_mod_params_testingdata, not generative and not is_qcnn, num_qubits_bps: {num_qubits_bps}")

    # I have to get X_test_client_angle
    pca_obj = data_logs_loaded_test_depthfl['clients_data_dict'][model_size][0][2][0]

    print(f"load_mod_params_testingdata, pca_obj: {pca_obj}")

    local_pca = True
    generative = False
    use_torch = True
    shared_pca = True
    math_int = torch
    cur_model_size = max(num_qubits_bps)

    (X_test, y_test) = data_logs_loaded_test_depthfl["testing_data"]

    if math_int == torch:
      # TODO, layers: convert y_test to a PyTorch tensor as well.
      X_test = torch.tensor(X_test, dtype=torch.float32)
      y_test = torch.tensor(y_test, dtype=torch.float32)

    device = "cpu"

    
    # get shared_max_comps, shared_min_comps
    agg_data_size = max(data_logs_loaded_test_depthfl['clients_data_dict'])
    agg_client_data = math_int.empty(0, agg_data_size)

    print(f"load_mod_params_testingdata, agg_client_data.shape: {agg_client_data.shape}")

    for client_type, client_data in data_logs_loaded_test_depthfl['clients_data_dict'].items():
        for client_data_indiv in client_data:
            agg_client_data = math_int.concatenate((agg_client_data, client_data_indiv[2][1]), axis=0)
    
    # TODO: make sure that shared_max_comps and shared_min_comps is correct.
    shared_max_comps = math_int.max(agg_client_data, axis=0)
    shared_min_comps = math_int.min(agg_client_data, axis=0)
        

    

    # TODO: 9/18: continue here
    # build client‑specific testing data once here
    if local_pca and not generative:
        # NOTE: why was client_data_indiv used here?? doesn't make that much sense.
        # NOTE, layers: changed client_pca_info here to use index 2 instead of 1; not sure if it's right
        # client_pca_info = client_data_list[client_idx][2]
        # print_cust(f"run_qfl_experiments_parallel_multiprocess, client_pca_info: {client_pca_info}")
        client_pca = pca_obj
        # client_data_pca = client_pca_info[1]
        # DONE: TOMODIFY, layers: note that the PCA is for numpy, and X_test is a pytorch tensors, so I'll need to do some data conversions (applies for this entire block of code below)
        print(f"run_qfl_experiments_parallel_multiprocess, X_test.shape: {X_test.shape}, type(X_test): {type(X_test)}")
        if use_torch:
            X_test_np = X_test.detach().cpu().numpy()
        else:
            X_test_np = X_test
        print_cust(f"run_qfl_experiments_parallel_multiprocess, X_test_np.shape: {X_test_np.shape}, type(X_test_np): {type(X_test_np)}")
        X_test_client_pca = client_pca.transform(X_test_np)
        # Scale each PCA component independently to [0, π]
        # MODIFIED, layers: changed np -> math_int
        if use_torch:
            # MODIFIED, layers: converted to torch after PCA transform
            # NOTE, layers: I might get dtype issues here, in which case I'd need to change to torch.float32 explicitly.
            X_test_client_pca = torch.from_numpy(X_test_client_pca).to(device)
        X_test_client_angle = math_int.zeros_like(X_test_client_pca)
        # assuming that the number of components is simply the client's data type
        # TODO: rescale the testing data to match the PCA scale for each client
        print_cust(f"run_qfl_experiments_parallel_multiprocess, X_test_client_pca.shape: {X_test_client_pca.shape}, type(X_test_client_pca): {type(X_test_client_pca)}")
        print_cust(f"run_qfl_experiments_parallel_multiprocess, X_test_client_angle.shape: {X_test_client_angle.shape}, type(X_test_client_angle): {type(X_test_client_angle)}")
        print(f"run_qfl_experiments_parallel_multiprocess, shared_min_comps: {shared_min_comps}, shared_max_comps: {shared_max_comps}, type(shared_min_comps): {type(shared_min_comps)}, type(shared_max_comps): {type(shared_max_comps)}")
        for i in range(cur_model_size):
            comp = X_test_client_pca[:, i]
            if shared_pca:
                # NOTE, layers: technically I should do instanceof check here to make sure that
                # these are torch return types
                lo, hi = shared_min_comps.values[i], shared_max_comps.values[i]
                print_cust(f"run_qfl_experiments_parallel_multiprocess, comp.shape: {comp.shape}, type(comp): {type(comp)}, lo.shape: {lo.shape}, hi.shape: {hi.shape}, type(lo): {type(lo)}, type(hi): {type(hi)}")
                comp_norm = ( (comp - lo) / (hi - lo + 1e-8) )
                # NOTE, depthFL: need to clip comp_norm????
                # comp_norm = np.clip(comp_norm, 0, 1)
            # else:
            #     orig_comp = client_data_pca[:, i]
            #     comp_norm = (comp - orig_comp.min()) / (orig_comp.max() - orig_comp.min() + 1e-8)
            # MODIFIED, layers: changed np -> math_int
            X_test_client_angle[:, i] = comp_norm * math_int.pi
    else:
        X_test_client_angle = X_test

    print_cust(f"run_qfl_experiments_parallel, X_test_client_angle.shape: {X_test_client_angle.shape}, type(X_test_client_angle): {type(X_test_client_angle)}, y_test.shape: {y_test.shape}, type(y_test): {type(y_test)}")
    print_cust(f"run_qfl_experiments_parallel, X_test_client_angle.min(): {X_test_client_angle.min()}, X_test_client_angle.max(): {X_test_client_angle.max()}, y_test.min(): {y_test.min()}, y_test.max(): {y_test.max()}")
    if local_pca and not generative:
        X_test_client_angle = math_int.clip(X_test_client_angle, 0.0, math_int.pi)
    print_cust(f"run_qfl_experiments_parallel, X_test_client_angle.shape: {X_test_client_angle.shape}, type(X_test_client_angle): {type(X_test_client_angle)}, y_test.shape: {y_test.shape}, type(y_test): {type(y_test)}")
    print_cust(f"run_qfl_experiments_parallel, X_test_client_angle.min(): {X_test_client_angle.min()}, X_test_client_angle.max(): {X_test_client_angle.max()}, y_test.min(): {y_test.min()}, y_test.max(): {y_test.max()}")
    testing_data = (X_test_client_angle, y_test)
    
    # get the target params
    return agg_params_used, testing_data

# %%


# %% [markdown]
# ## Side usecase: Compute HeteroFL Ensemble Accs

# %%
def get_mean_stddev_dict(dataset_names, classes_list_all, seed_list, circ_type_list, num_clients_1cli, num_classifiers_cli, final_round_idx, client_labels, multiclassifier_type, multiclassifier_type_run, aer_backend=None, folder_prefix=".", n_testing_data=None, overall_client_size=14):
    final_metrics_dict = {}
    print(f"get_mean_stddev_dict, overall_client_size: {overall_client_size}")
    for dataset_idx, dataset in enumerate(dataset_names):
        for circ_type in circ_type_list:
            classes_list = classes_list_all[dataset_idx]
            for classes in classes_list:
                print(f"get_mean_stddev_dict, classes: {classes}")
                # TODO: inject client size here
                client_size = overall_client_size
                print(f"get_mean_stddev_dict, client_size: {client_size}")
                testing_acc_dict = {}
                for cli_idx_val, client_num in enumerate(range(num_clients_1cli)):
                    for classifier_idx in range(num_classifiers_cli):
                    #   # if classifier_idx == num_classifiers_cli - 1:
                        testing_acc_dict[f"Dataset {dataset}, Circ Type {circ_type}, Classes {classes}, {client_size} Qubit Client {client_labels[client_num]} Testing Accuracies, 1 Client, Classifier {classifier_idx}"] = []
                        testing_acc_dict[f"Dataset {dataset}, Circ Type {circ_type}, Classes {classes}, {client_size} Qubit Client {client_labels[client_num]} Testing Accuracies, 1 Client, Ensemble Classifier {classifier_idx}"] = []
                        testing_acc_dict[f"Dataset {dataset}, Circ Type {circ_type}, Classes {classes}, {client_size} Qubit Client {client_labels[client_num]} Testing Accuracies, 1 Client"] = []
                for seed in seed_list:
                    agg_params_used, testing_data = load_mod_params_testingdata(dataset, classes, multiclassifier_type, seed, client_size, round_num=final_round_idx, folder_prefix=folder_prefix)
                    if n_testing_data is not None:
                        testing_data_list = []
                        testing_data_list.append(testing_data[0][:n_testing_data, :])
                        testing_data_list.append(testing_data[1][:n_testing_data])
                        testing_data = tuple(testing_data_list)
                    print(f"get_mean_stddev_dict, testing_data[0].shape: {testing_data[0].shape}, testing_data[1].shape: {testing_data[1].shape}")
                    test_acc, test_acc_stdev, test_acc_topk, test_loss, avg_acc_classifiers, std_acc_classifiers, top_k_accuracies_classifiers, avg_loss_classifiers, gen_all_probs = eval_params_qnode(agg_params_used, testing_data, multiclassifier_type_run, aer_backend=aer_backend)
                    testing_acc_dict[f"Dataset {dataset}, Circ Type {circ_type}, Classes {classes}, {client_size} Qubit Client {client_labels[client_num]} Testing Accuracies, 1 Client"].append(test_acc)
                    for classifier_idx in range(num_classifiers_cli):
                    #   # print(f"round: {round}, client_size: {client_size}, client_num: {client_num}, client_idx: {client_idx}, classifier_idx: {classifier_idx}")

                        if avg_acc_classifiers is not None:
                            testing_acc_dict[f"Dataset {dataset}, Circ Type {circ_type}, Classes {classes}, {client_size} Qubit Client {client_labels[client_num]} Testing Accuracies, 1 Client, Classifier {classifier_idx}"].append(avg_acc_classifiers[classifier_idx])
                        else:
                            testing_acc_dict[f"Dataset {dataset}, Circ Type {circ_type}, Classes {classes}, {client_size} Qubit Client {client_labels[client_num]} Testing Accuracies, 1 Client, Classifier {classifier_idx}"].append(None)

                        if gen_all_probs is not None:
                          gen_all_probs = gen_all_probs
                          all_probs = gen_all_probs[:classifier_idx + 1].mean(axis=0)
                          preds = []
                          math_int = torch
                          container_creator = torch.tensor
                          float_dtype = torch.float32
                          y = testing_data[1]
                          for i, probs in enumerate(all_probs):
                              # probs = qnode(input_angles, params)
                              preds.append(torch.argmax(probs))
                          
                          preds = container_creator(preds)

                          avg_acc = math_int.mean(preds == container_creator(y), dtype=float_dtype)

                          testing_acc_dict[f"Dataset {dataset}, Circ Type {circ_type}, Classes {classes}, {client_size} Qubit Client {client_labels[client_num]} Testing Accuracies, 1 Client, Ensemble Classifier {classifier_idx}"].append(avg_acc)
                        else:
                          testing_acc_dict[f"Dataset {dataset}, Circ Type {circ_type}, Classes {classes}, {client_size} Qubit Client {client_labels[client_num]} Testing Accuracies, 1 Client, Ensemble Classifier {classifier_idx}"].append(None)
                print(f"get_mean_stddev_dict, testing_acc_dict: {testing_acc_dict}")
                for key_name in testing_acc_dict:
                   # TODO: can check that key_name doesn't already exist in final_metrics_dict
                   if len(testing_acc_dict[key_name]) == 0:
                    print(f"get_mean_stddev_dict, skipping key_name {key_name}, empty metrics list")
                    continue
                   metrics_tens = torch.stack(testing_acc_dict[key_name], dim=0)
                   final_metrics_dict[key_name] = (metrics_tens.mean(dim=0), metrics_tens.std(dim=0, correction=1))
    return final_metrics_dict

# %%


# %% [markdown]
# ## Get backends

# %%
def get_backends(channel, token, ibmq_crn):
    service = QiskitRuntimeService(channel=channel, token=token, instance=ibmq_crn)
    return service.backends()

# %% [markdown]
# ## Func that takes in a backend name, and returns the noise model

# %%
def get_aer_sim_backend(backend_name_to_filename, backend=None):
    backend_name = backend.name
    backend_filename = backend_name_to_filename[backend_name]
    with open(f"{backend_filename}", "rb") as file:
        noise_dict = pickle.load(file)
    print(f"get_aer_sim_backend, backend_filename: {backend_filename}")
    noise = noise_dict['noise_model']
    aer_backend = AerSimulator.from_backend(backend)                  # copies basis_gates + coupling_map
    aer_backend.set_options(noise_model=noise)                        # make noise explicit
    return aer_backend

def get_real_backend(backend=None):
    # backend_name = backend.name
    # backend_filename = backend_name_to_filename[backend_name]
    # with open(f"{backend_filename}", "rb") as file:
    #     noise_dict = pickle.load(file)
    # print(f"get_aer_sim_backend, backend_filename: {backend_filename}")
    # noise = noise_dict['noise_model']
    # aer_backend = AerSimulator.from_backend(backend)                  # copies basis_gates + coupling_map
    # aer_backend.set_options(noise_model=noise)                        # make noise explicit
    return backend

# %% [markdown]
# ## "Main" function to run same model on different noise models

# %%
def run_same_model_diffnm(backends_func, backends_retrieval_func, metrics_func, folder_path=".", valid_backend_names=None, multiclassifier_type="multirun", n_testing_data=None, overall_client_size=14):
    print(f"run_same_model_diffnm, valid_backend_names: {valid_backend_names}")
    res_backend_dict = {}
    valid_backends = backends_func()
    for backend in valid_backends:
        print(f"run_same_model_diffnm, backend.name: {backend.name}")
        if valid_backend_names is not None:
            if backend.name not in valid_backend_names:
                continue
        print(f"run_same_model_diffnm, running for backend.name: {backend.name}")
        conv_aer_backend = backends_retrieval_func(backend=backend)
        resulting_metrics = metrics_func(aer_backend=conv_aer_backend)
        res_backend_dict[backend.name] = resulting_metrics
        with open(f"{folder_path}/{backend.name}_5cli_10q_6l_{multiclassifier_type}_metrics_dict_realhardware_fast_24_ntd_{n_testing_data}_csize_{overall_client_size}.pkl", "wb") as file:
            pickle.dump(resulting_metrics, file)
    return res_backend_dict

# %%

    

# %% [markdown]
# ## Running the "main" function

def main(argv=None):

    # %%
    dataset_names = ["Fashion-MNIST"]
    classes_list_all = [
        # [
        #   ["4", "9"],
        #   ["3", "4"],
        #   ["0", "1"]
        # ],
        [
        ["2", "4"],
        #   ["5", "8"],
        #   ["1", "9"]
        ]
    ]
    seed_list = [
        # 12,
        30,
        # 50,
        # 70,
        # 400
    ]
    circ_type_list = ["v_shape"]
    num_clients_1cli = 1
    final_round_idx = 999
    client_labels = ['A', 'B', 'C', 'D', 'E']
    multiclassifier_type = "tunnel_down"
    multiclassifier_type_run = "tunnel_down_realhardware"
    folder_prefix = "."

    # %%
    backends_func = partial(get_backends, channel="ibm_cloud", token=token, ibmq_crn=ibmq_crn)

    # %%
    backend_name_to_filename = {
        "ibm_brisbane": f"{folder_prefix}/noise_snapshots/ibm_brisbane_noise_20250918T122043Z.pkl",
        "ibm_fez": f"{folder_prefix}/noise_snapshots/ibm_fez_noise_20250918T122057Z.pkl",
        "ibm_kingston": f"{folder_prefix}/noise_snapshots/ibm_kingston_noise_20250918T122140Z.pkl",
        "ibm_marrakesh": f"{folder_prefix}/noise_snapshots/ibm_marrakesh_noise_20250918T122125Z.pkl",
        "ibm_pittsburgh": f"{folder_prefix}/noise_snapshots/ibm_pittsburgh_noise_20250918T122037Z.pkl",
        "ibm_torino": f"{folder_prefix}/noise_snapshots/ibm_torino_noise_20250918T122110Z.pkl"
    }

    # %%
    # backend_retrieval_func = partial(get_aer_sim_backend, backend_name_to_filename=backend_name_to_filename)
    backend_retrieval_func = get_real_backend

    n_testing_data = 100

    list_client_sizes = [14]
    list_classifier_nums = [5]

    # overall_client_size = 14
    # num_classifiers_cli = 5

    for overall_client_size, num_classifiers_cli in zip(list_client_sizes, list_classifier_nums):
        metrics_func = partial(get_mean_stddev_dict, dataset_names=dataset_names, classes_list_all=classes_list_all, seed_list=seed_list, circ_type_list=circ_type_list, num_clients_1cli=num_clients_1cli, num_classifiers_cli=num_classifiers_cli, final_round_idx=final_round_idx, client_labels=client_labels, multiclassifier_type=multiclassifier_type, multiclassifier_type_run=multiclassifier_type_run, folder_prefix=folder_prefix, n_testing_data=n_testing_data, overall_client_size=overall_client_size)

        # %%
        print(f"Running noise simulations")

        valid_backend_names = ["ibm_kingston"]

        import sys, contextlib
        with open(f"{folder_prefix}/noise_results/5cli_10q_6l_{multiclassifier_type}_metrics_dict_realhardware_stdout_fashmnist_24_{"_".join(valid_backend_names)}_fast_ntd_{n_testing_data}_csize_{overall_client_size}.txt", "w") as fout, open(f"{folder_prefix}/noise_results/5cli_10q_6l_{multiclassifier_type}_metrics_dict_realhardware_stderr_fashmnist_24_{"_".join(valid_backend_names)}_fast_ntd_{n_testing_data}_csize_{overall_client_size}.txt", "w") as ferr:
            with contextlib.redirect_stdout(fout), contextlib.redirect_stderr(ferr):
                res_backend_dict_diffnm = run_same_model_diffnm(backends_func, backend_retrieval_func, metrics_func, folder_path=f"{folder_prefix}/noise_results", valid_backend_names=valid_backend_names, multiclassifier_type=multiclassifier_type, n_testing_data=n_testing_data, overall_client_size=overall_client_size)

        print(f"Finished noise simulations")
        with open(f"{folder_prefix}/noise_results/5cli_10q_6l_{multiclassifier_type}_metrics_dict_realhardware_fashmnist_24_{"_".join(valid_backend_names)}_fast_ntd_{n_testing_data}_csize_{overall_client_size}.pkl", "wb") as file:
            pickle.dump(res_backend_dict_diffnm, file)

# %%
if __name__ == "__main__":
    main()