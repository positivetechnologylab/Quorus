import contextlib
import pickle
from quorus.misc_utils.count_numcli_clitype import count_numcli_clitype
from quorus.misc_utils.count_qubits_layers import count_qubits_layers
from quorus.qfl_main.run_qfl_exp_multiproc import run_qfl_experiments_parallel_multiprocess
from quorus.qgan_model_supp.imggen_funcs.latent_noise_gen import generate_latent_noise
import torch.nn as nn


def main(argv=None):
  print("Running qfl_main_test.py")

  # import torch.multiprocessing as mp
  import multiprocessing as mp

  import os, random

  from pennylane import numpy as np

  import torch

  def set_seed(seed: int, deterministic: bool = False):
    """
    Sets the seed for the current run.

    Parameters:
      seed (int): The seed to set.
      deterministic (Boolean): Boolean indicating whether or not deterministic torch algorithms should be used.

    Returns:
      Nothing. Modifies seed for the current run.
    """
    # Python + NumPy
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)                          # seeds NumPy's *global* RNG

    # PyTorch (CPU + CUDA/MPS)
    torch.manual_seed(seed)                       # seeds RNG on all devices
    # Optional (esp. multi-GPU): torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Make CUDA ops deterministic where possible
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

  
  # Define the client configuration of interest.

  dataset_types = ["Fashion-MNIST"]

  # Specifies the ansatz type for each client. A dictionary mapping client types to a list of strings, where each string
  # determines the ansatz type used for that layer. Supported strings are "reversed_staircase", "v_shape", and "revstair_vshape".
  ansatz_types_list = [
    # {10: ["reversed_staircase"],
    #  11: ["reversed_staircase", "reversed_staircase"],
    #  12: ["reversed_staircase", "reversed_staircase", "reversed_staircase"],
    #  13: ["reversed_staircase", "reversed_staircase", "reversed_staircase", "reversed_staircase"],
    #  14: ["reversed_staircase", "reversed_staircase", "reversed_staircase", "reversed_staircase", "reversed_staircase"]},

    {10: ["v_shape"],
     11: ["v_shape", "v_shape"],
     12: ["v_shape", "v_shape", "v_shape"],
     13: ["v_shape", "v_shape", "v_shape", "v_shape"],
     14: ["v_shape", "v_shape", "v_shape", "v_shape", "v_shape"]},

    # {10: ["revstair_vshape"],
    #  11: ["revstair_vshape", "revstair_vshape"],
    #  12: ["revstair_vshape", "revstair_vshape", "revstair_vshape"],
    #  13: ["revstair_vshape", "revstair_vshape", "revstair_vshape", "revstair_vshape"],
    #  14: ["revstair_vshape", "revstair_vshape", "revstair_vshape", "revstair_vshape", "revstair_vshape"]}
  ]

  class_types_list = [
    [
      ["4", "9"],
      ["3", "4"],
      ["0", "1"]
    ],
    [
      ["2", "4"],
      ["5", "8"],
      ["1", "9"]
    ]
  ]

  random_seeds = [
    12,
    30,
    50,
    70,
    400
  ]

  # Iterate over all datasets, ansatz types, classes, and seeds to run ALL specified experiments.
  for dataset_idx, dataset_type_exper in enumerate(dataset_types):
    for qubits_and_layer_types_block_params in ansatz_types_list:
      # Get the classes associated with this dataset.
      dataset_classes = class_types_list[dataset_idx]
      print(f"dataset_classes: {dataset_classes}")
      for classes_exper in dataset_classes:
        for random_state in random_seeds:
          
          # Set the seed for this run.
          set_seed(random_state)

          # Get the client configurations for this run.
          client_config_exper_parallel = {
              10: {
                  "percentage_data": 0.20,
                  "num_clients": 1,
                  "local_epochs": 1,
                  "communication_rounds": 0
              },
              11: {
                  "percentage_data": 0.20,
                  "num_clients": 1,
                  "local_epochs": 1,
                  "communication_rounds": 0
              },
              12: {
                  "percentage_data": 0.20,
                  "num_clients": 1,
                  "local_epochs": 1,
                  "communication_rounds": 0
              },
              13: {
                  "percentage_data": 0.20,
                  "num_clients": 1,
                  "local_epochs": 1,
                  "communication_rounds": 0
              },
              14: {
                  "percentage_data": 0.20,
                  "num_clients": 1,
                  "local_epochs": 1,
                  "communication_rounds": 2
              }
              # 16: {
              #     "percentage_data": 1.0,
              #     "num_clients": 4,
              #     "local_epochs": 1,
              #     "communication_rounds": 10
              # }
          }
          # Count the number of clients per client type.
          cli_type_counts_str = count_numcli_clitype(client_config_exper_parallel)
          print(f"cli_type_counts_str: {cli_type_counts_str}")
          # Specify the number of qubits and layer types used per client type. This is used for loading in initial client parameters/data.
          qubits_layers_list = [(10, 2), (11, 4), (12, 6), (13, 8), (14, 10)]
          # Convert the above list to a string, also used for loading initial parameters/data.
          qubits_layers_str = count_qubits_layers(qubits_layers_list)
          print(f"qubits_layers_str: {qubits_layers_str}")
          # Find the maximum local epochs for all clients. This is used for logging.
          local_epochs =  max([client_config_exper_parallel[key]["local_epochs"] for key in client_config_exper_parallel])
          # Specifies the path to store the folder containnig the logs for this run.
          IMG_PATH = "."
          print(f"dataset_type_exper: {dataset_type_exper}")
          # classes_exper = ["4", "9"]
          # Specifies the total amount of data, the number of clients, and the amount of data per client.
          n_samples_exper = 3640
          n_clients_tot = 5
          datapoints_per_cli = 128
          n_train_samples_exper = n_clients_tot * datapoints_per_cli
          print(f"n_train_samples_exper: {n_train_samples_exper}")
          # Find the total number of rounds to run.
          num_total_rounds_glob = max([client_config_exper_parallel[key]["communication_rounds"] for key in client_config_exper_parallel])
          # Specify the cutoff at which testing accuracy is computed every 10 rounds, for speed.
          testacc_rd_cutoff = 101
          print(f"testacc_rd_cutoff: {testacc_rd_cutoff}")
          is_multiproc = True
          # Specify the local batch size and optimizer type.
          local_batch_size = 32
          optim_type = "adam"

          # Specify learning rate for generator/discriminator for QGAN, and only discriminator for QNN.
          lr_gen = 0.004
          lr_disc = 0.001
          # Specify the decay rate of the discriminator LR.
          lr_disc_decay = 1.0
          # Specify whether or not the optimizer state should persist across rounds.
          cont_optim_state = False

          # Specify whether or not FID is computed (for QGAN).
          compute_fid = False

          # DONE: TOMODIFY, layers: have an indicator to specify that this is the classification, layers, no QCNN case.
          # DONE: TOMODIFY, layers, currently mod: use is_qcnn in the below log_data_folder name.
          # Specify whether or not to use QCNN (i.e., variational convolutional layers + feedforward) (NOTE: this is
          # an experimental feature)
          is_qcnn = False

          # Specifiy whether or not amplitude embedding is to be used (NOTE: this is an experimental feature)
          amp_embed = False
          # Specify whether clients should share their PCA or not.
          shared_pca = True
          # Specify whether or not clients should perform local PCA.
          local_pca = True

          # Specifiy whether or not parameters in later layers should have a specifal initialization, for identity initialization (NOTE:
          # this is an experimental feature).
          # This should either be "", "posneg", or "random"
          alt_zeros_init = ""

          # Specify the quantum model type to be used.
          # Should either be "", "multirun", "ancilla_endmeas", "cheating", or "tunnel_down". Details found in README.
          multiclassifier_type = "multirun"

          # Specifies whether or not the models should be trained in parallel.
          train_models_parallel = True
          # Specifies whether or not heirarchical training should be performed.
          heirarchical_train = False

          # Specifies the type of aggregation that should be performed.
          # morepers is "aggshared" (circular averaging) or "mocked_bcast" (no aggregation). Other supported types are experimental.
          morepers = "aggshared"

          # Finds the ansatz type used in the experiments.
          ansatz_type = qubits_and_layer_types_block_params[list(qubits_and_layer_types_block_params.keys())[0]][0]
          print(f"ansatz_type: {ansatz_type}")

          # random_state = 12
          # Specify the log folder where results should be stored. This string is a function of the previous configurations, in string format.
          log_data_folder = f"{IMG_PATH}/data_logs_{dataset_type_exper}_classes_{'_'.join(classes_exper)}_n_samples_{n_samples_exper}_n_train_{n_train_samples_exper}_qfl_gen_{num_total_rounds_glob}rounds_le_{local_epochs}_bs_{local_batch_size}_opt_{optim_type}_mpd_mlt_lg_{lr_gen}_ld_{lr_disc}_dqdm_ba_sp_qcnn_{is_qcnn}_ba_sm_ae_{amp_embed}_dr_nce_mc_{multiclassifier_type}_mp_{train_models_parallel}_tclip_{morepers}_{random_state}_{ansatz_type}_ldd_{lr_disc_decay}_2_10l_nos"



          # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
          # Set the device used.
          device = "cpu"

          print(f"device: {device}")

          # NOTE: tech, I don't want to mask grads...

          # import dill as pickle

          max_workers = os.cpu_count()

          mp_ctx = mp.get_context("spawn")

          # # # # TODO: redirect stdout to see what the problem is.
          # TODO: move all the stuff into if __name__ == "__main__".
          # Make the log_data_folder, if it doesn't yet exist.
          if not os.path.exists(log_data_folder):
              os.makedirs(log_data_folder)

          print("_".join(classes_exper))
          print(f"log_data_folder: {log_data_folder}")

          # import cProfile, pstats

          # Redirect output to stdout and stderr.
          with open(f"{log_data_folder}/main_stdout.txt", "w") as fout, open(f"{log_data_folder}/main_stderr.txt", "w") as ferr:
              with contextlib.redirect_stdout(fout), contextlib.redirect_stderr(ferr):

                  print(f"qubits_and_layer_types_block_params: {qubits_and_layer_types_block_params}")
                  print(f"classes_exper: {classes_exper}")
                  print(f"random_state: {random_state}")

                  mp.set_start_method("spawn", force=True)
                  # torch.multiprocessing.set_sharing_strategy('file_system')

                  # import multiprocessing as mp

                  mp_ctx = mp.get_context("spawn")

                  print(f"max_workers: {max_workers}")

                  # Specify the betas for the generator and discriminator. For the classifier case, the discriminator betas are used.
                  gen_betas = (0.5, 0.9)
                  disc_betas = (0.9, 0.99)

                  # Run the QFL workflow.
                  data_logs_prev = None
                  initial_supp_params = None
                  # TODO, public code: have a boolean indicating whether or not to use previous data.
                  with open(f"{IMG_PATH}/initial_configs_{n_clients_tot}cli_{datapoints_per_cli}tr_{n_samples_exper - (datapoints_per_cli * n_clients_tot)}test_{qubits_layers_str}/data_logs_n_samples_{n_samples_exper}_dataset_type_{dataset_type_exper}_classes_{'_'.join(classes_exper)}_train_models_parallel_{train_models_parallel}_feature_skew_0.0_label_skew_None_local_pca_{local_pca}_shared_pca_{shared_pca}_gen_False_qcnn_{is_qcnn}_{random_state}.pkl", "rb") as file:
                    data_logs_prev = pickle.load(file)
                  print(f"data_logs_prev.keys(): {data_logs_prev.keys()}")
                  print(f"data_logs_prev['clients_data_dict'].keys(): {data_logs_prev['clients_data_dict'].keys()}")
                  print(f"data_logs_prev: {data_logs_prev}")
                  # TODO, public code: have a boolean indicating whether or not to use previous parameters.
                  with open(f"{IMG_PATH}/initial_configs_{n_clients_tot}cli_{datapoints_per_cli}tr_{n_samples_exper - (datapoints_per_cli * n_clients_tot)}test_{qubits_layers_str}/client_params_dict_n_samples_{n_samples_exper}_dataset_type_{dataset_type_exper}_classes_{'_'.join(classes_exper)}_train_models_parallel_{train_models_parallel}_feature_skew_0.0_label_skew_None_local_pca_{local_pca}_shared_pca_{shared_pca}_gen_False_qcnn_{is_qcnn}_{random_state}.pkl", "rb") as file:
                    initial_supp_params = pickle.load(file)
                  print(f"initial_supp_params.keys(): {initial_supp_params.keys()}")
                  print(f"initial_supp_params: {initial_supp_params}")

                  # Commented code that is used to trim the parameters, and thus the model size. The dimensions of the supplied parameters
                  # determine the architecture/number of layers of the model.

                  # SHARED_MODEL_NUMLAYERS = 2
                  # for cli_type in initial_supp_params:
                  #   for cli_idx, cli_params in enumerate(initial_supp_params[cli_type]):
                  #     # DONE: TOMODIFY, layers: this is for the generative case; for the nongenerative case, have the code. should be pretty straightforward; just replace block params entirely
                  #     # ^ and later, can do validation to make sure that the structure, order of the supplied block params is consistent with qubits_and_layers_to_add_block_params.
                  #     # if generative:
                  #     #   cli_gen = cli_params[5][0]
                  #     #   supp_cli_gen = initial_supp_params[cli_type][cli_idx][5][0]
                  #     #   cli_gen.load_state_dict(supp_cli_gen.state_dict())
                  #     #   cli_params[5][1].load_state_dict(initial_supp_params[cli_type][cli_idx][5][1].state_dict())
                  #     # elif not is_qcnn:
                  #     cli_params_list = list(cli_params)
                  #     # new_params_list = []
                  #     # # TOMODIFY, depthFL, HACK: heuristic used
                  #     # # NOTE: assumes that there's at least one block param in the list
                  #     # num_tot_qubits = initial_supp_params[cli_type][cli_idx][5][0].shape[1]
                  #     # print(f"cli_type: {cli_type}, cli_idx: {cli_idx}, num_tot_qubits: {num_tot_qubits}")
                  #     # for orig_tens_idx, orig_param_tens in enumerate(initial_supp_params[cli_type][cli_idx][5]):
                  #     #   new_params_list.append(orig_param_tens[:, -(num_tot_qubits - orig_tens_idx):, :])
                  #     # cli_params_list[5] = new_params_list
                  #     cli_params_list[5] = initial_supp_params[cli_type][cli_idx][5][:SHARED_MODEL_NUMLAYERS]
                  #     initial_supp_params[cli_type][cli_idx] = tuple(cli_params_list)
                  
                  # print(f"initial_supp_params, after filtering extra layers: {initial_supp_params}")
                  # print(f"initial_supp_params, after filtering extra qubits: {initial_supp_params}")

                  # Specifies whether or not PCA values should be rescaled after inverse PCA, for the QGAN case (this is experimental).
                  resc_invpca = False

                  # Specifies the interface that pennylane should use.
                  pennylane_interface = "torch"

                  # Specifies the layers that each client optimizes over (NOTE: experimental; should be kept to None.)
                  opt_layers = None

                  print(f"random_state: {random_state}")

                  # Run the QFL workflow and obtain the resulting metrics, in data_logs.
                  data_logs = run_qfl_experiments_parallel_multiprocess(client_config_exper_parallel, classes=classes_exper, n_samples=n_samples_exper, dataset_type=dataset_type_exper, agg_strategy="fedavg_circ", test_frac=((n_samples_exper - n_train_samples_exper)/n_samples_exper), val_frac=0.0, random_state=random_state, pool_in=True,
                                          local_batch_size=local_batch_size, local_lr=0.01, shots=1024, debug=True, save_pkl=True, mask_grads=True, init_client_data_dict=data_logs_prev, qubits_and_layers_to_add_block_params={10: [(10, 2)], 11: [(10, 2), (10, 1)], 12: [(10, 2), (10, 1), (10, 1)], 13: [(10, 2), (10, 1), (10, 1), (10, 1)], 14: [(10, 1), (10, 1), (10, 1), (10, 1), (10, 1)]},
                                                          train_models_parallel=train_models_parallel, same_init=True, feature_skew=0.0, label_skew=None, local_pca=local_pca, do_lda=False, feat_sel_type="top", amp_embed=amp_embed, feat_ordering="same", morepers=morepers, custom_debug=True,
                                                          shared_pca=shared_pca, heirarchical_train=heirarchical_train, generative=False, use_torch=True, fed_pca_mocked=True, lr_gen=lr_gen, lr_disc=lr_disc, noise_func=generate_latent_noise, criterion_func=nn.BCELoss,
                                                              targ_data_folder_prefix="testing_gen_imgs", gen_data_folder_prefix="qgan_gen_imgs", device=device, fid_batch_size=None, max_workers=max_workers, mp_ctx=mp_ctx, log_data_folder=log_data_folder,
                                                                          initial_supp_params=initial_supp_params, optim_type=optim_type, gen_betas=gen_betas, disc_betas=disc_betas, resc_invpca=resc_invpca, compute_fid=compute_fid, is_qcnn=is_qcnn,
                                                                          pennylane_interface=pennylane_interface, opt_layers=opt_layers, alt_zeros_init=alt_zeros_init, multiclassifier_type=multiclassifier_type, testacc_rd_cutoff=testacc_rd_cutoff,
                                                                          qubits_and_layer_types_block_params=qubits_and_layer_types_block_params, lr_disc_decay=lr_disc_decay, cont_optim_state=cont_optim_state)
                  print("running after the main() function.")

                  print(f"data_logs.keys(): {data_logs.keys()}")

                  print(f"data_logs['clients_data_dict']: {data_logs['clients_data_dict']}")

                  # Save the resulting data metrics.
                  with open(f"{log_data_folder}/result_datalogs.pkl", "wb") as file:
                      pickle.dump(data_logs, file)

if __name__ == "__main__":
  main()