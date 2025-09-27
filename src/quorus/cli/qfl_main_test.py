import argparse
import contextlib
import json
import os
import pickle
import random
import re
from typing import Any, Dict, List, Tuple

from pennylane import numpy as np
import torch
import torch.nn as nn
import multiprocessing as mp

from quorus.logging.custom_slog import set_glob_loglevel
from quorus.misc_utils.count_numcli_clitype import count_numcli_clitype
from quorus.misc_utils.count_qubits_layers import count_qubits_layers
from quorus.qfl_main.run_qfl_exp_multiproc import run_qfl_experiments_parallel_multiprocess
from quorus.qgan_model_supp.imggen_funcs.latent_noise_gen import generate_latent_noise
import quorus.logging.custom_slog as slog


def load_json_with_comments(path: str) -> Dict[str, Any]:
    """
    Loads JSON or JSONC (// and /* */ comments allowed) into a dict.
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    # Strip /* ... */ block comments
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    # Strip // line comments
    text = re.sub(r"(?m)^\s*//.*?$", "", text)
    # Also strip trailing // comments after values
    text = re.sub(r"(?m)\/\/.*$", "", text)
    return json.loads(text)


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep-merge override into base (mutates base).
    """
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def set_seed(seed: int, deterministic: bool = False):
    """
    Sets seeds for Python, NumPy, and PyTorch, with optional deterministic Torch algorithms.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def coerce_int_keys(obj: Any):
    """
    Recursively convert dict keys that look like integers to int.
    """
    if isinstance(obj, dict):
        new_d = {}
        for k, v in obj.items():
            if isinstance(k, str) and k.strip().lstrip("-").isdigit():
                new_k = int(k)
            else:
                new_k = k
            new_d[new_k] = coerce_int_keys(v)
        return new_d
    elif isinstance(obj, list):
        return [coerce_int_keys(x) for x in obj]
    else:
        return obj


def to_tuple_list(lst2d: List[List[int]]) -> List[Tuple[int, int]]:
    return [tuple(x) for x in lst2d]


def ensure_classes_are_str(class_groups: List[List[str]]) -> List[List[str]]:
    """
    Classes may be provided as ints; convert to str to match original code's behavior.
    """
    out = []
    for pair in class_groups:
        out.append([str(x) for x in pair])
    return out


def build_qal_add_params(
    qal_block_params: Dict[int, List[str]],
    default_block_param: Tuple[int, int]
) -> Dict[int, List[Tuple[int, int]]]:
    """
    Build qubits_and_layers_to_add_block_params from the ansatz definition list lengths
    if not explicitly provided in config. Mirrors your previous static mapping where
    each layer reuses the same (qubits, layer_depth) tuple.
    """
    out = {}
    for cli_type, layer_kinds in qal_block_params.items():
        num_layers = len(layer_kinds)
        out[cli_type] = [default_block_param for _ in range(num_layers)]
    return out


def get_default_config() -> Dict[str, Any]:
    """
    Defaults derived from your current script, so running with no overrides reproduces
    prior behavior.
    """
    return {
        "dataset_types": ["Fashion-MNIST"],
        "ansatz_types_list": [
            {10: ["v_shape"], 11: ["v_shape", "v_shape"], 12: ["v_shape", "v_shape", "v_shape"],
             13: ["v_shape", "v_shape", "v_shape", "v_shape"],
             14: ["v_shape", "v_shape", "v_shape", "v_shape", "v_shape"]}
        ],
        "class_types_list": [
            [["1", "9"]]
        ],
        "random_seeds": [12, 30, 50, 70, 400],

        "client_config_exper_parallel": {
            "10": {"percentage_data": 0.20, "num_clients": 1, "local_epochs": 1, "communication_rounds": 0},
            "11": {"percentage_data": 0.20, "num_clients": 1, "local_epochs": 1, "communication_rounds": 0},
            "12": {"percentage_data": 0.20, "num_clients": 1, "local_epochs": 1, "communication_rounds": 0},
            "13": {"percentage_data": 0.20, "num_clients": 1, "local_epochs": 1, "communication_rounds": 0},
            "14": {"percentage_data": 0.20, "num_clients": 1, "local_epochs": 1, "communication_rounds": 2}
        },

        "qubits_layers_list": [[10, 2], [11, 4], [12, 6], [13, 8], [14, 10]],
        "IMG_PATH": ".",

        "n_samples_exper": 3640,
        "n_clients_tot": 5,
        "datapoints_per_cli": 128,

        "testacc_rd_cutoff": 101,
        "is_multiproc": True,

        "local_batch_size": 32,
        "optim_type": "adam",

        "lr_gen": 0.004,
        "lr_disc": 0.001,
        "lr_disc_decay": 1.0,
        "cont_optim_state": False,

        "compute_fid": False,
        "is_qcnn": False,
        "amp_embed": False,
        "shared_pca": True,
        "local_pca": True,
        "alt_zeros_init": "",  # "", "posneg", or "random"

        "multiclassifier_type": "multirun",
        "train_models_parallel": True,
        "heirarchical_train": False,  # spelling preserved from original

        "morepers": "aggshared",      # "aggshared" or "mocked_bcast"
        "device": "cpu",

        # Advanced toggles used inside run_qfl_experiments_parallel_multiprocess (kept from your call)
        "agg_strategy": "fedavg_circ",
        "pool_in": True,
        "local_lr": 0.01,
        "shots": 1024,
        "debug": True,
        "save_pkl": True,
        "mask_grads": True,
        "same_init": True,
        "feature_skew": 0.0,
        "label_skew": None,
        "do_lda": False,
        "feat_sel_type": "top",
        "feat_ordering": "same",
        "custom_debug": True,
        "generative": False,
        "use_torch": True,
        "fed_pca_mocked": True,
        "targ_data_folder_prefix": "testing_gen_imgs",
        "gen_data_folder_prefix": "qgan_gen_imgs",
        "fid_batch_size": None,
        "gen_betas": [0.5, 0.9],
        "disc_betas": [0.9, 0.99],
        "resc_invpca": False,
        "pennylane_interface": "torch",
        "opt_layers": None,

        # Build qubits_and_layers_to_add_block_params automatically unless provided.
        "default_block_param": [10, 2],
        "qubits_and_layers_to_add_block_params": None,

        # PKL load controls
        "use_prev_data": True,
        "use_prev_params": True,
        "prev_data_path_override": None,
        "prev_params_path_override": None,

        # Seeding
        "deterministic_torch": False,

        # Multiprocessing start method
        "mp_start_method": "spawn",

        # log level
        "log_level": "ERROR"
    }


def main(argv=None):
    print(f"Running main for qfl_main_test, JSON config")
    parser = argparse.ArgumentParser(description="QFL entrypoint (config-driven).")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to JSON/JSONC config file with experiment settings."
    )
    args = parser.parse_args(argv)
    print(f"main, args.config: {args.config}")

    user_cfg = load_json_with_comments(args.config)
    default_cfg = get_default_config()
    cfg = deep_update(default_cfg, user_cfg)

    # Normalize types from JSON
    cfg["client_config_exper_parallel"] = coerce_int_keys(cfg["client_config_exper_parallel"])
    cfg["ansatz_types_list"] = [coerce_int_keys(d) for d in cfg["ansatz_types_list"]]
    qubits_layers_list = to_tuple_list(cfg["qubits_layers_list"])

    # Safety: validate lengths
    if len(cfg["class_types_list"]) != len(cfg["dataset_types"]):
        raise ValueError("class_types_list must have one entry per dataset in dataset_types.")

    for dataset_idx, dataset_type_exper in enumerate(cfg["dataset_types"]):
        # Block params (per-experiment dict)
        for qal_block_params in cfg["ansatz_types_list"]:
            dataset_classes = ensure_classes_are_str(cfg["class_types_list"][dataset_idx])

            for classes_exper in dataset_classes:
                for random_state in cfg["random_seeds"]:
                    set_seed(random_state, deterministic=cfg.get("deterministic_torch", False))

                    client_config_exper_parallel = cfg["client_config_exper_parallel"]

                    # Count client types (used for info/logs)
                    cli_type_counts_str = count_numcli_clitype(client_config_exper_parallel)
                    print(f"cli_type_counts_str: {cli_type_counts_str}")

                    qubits_layers_str = count_qubits_layers(qubits_layers_list)
                    print(f"qubits_layers_str: {qubits_layers_str}")

                    local_epochs = max([client_config_exper_parallel[k]["local_epochs"]
                                        for k in client_config_exper_parallel])
                    IMG_PATH = cfg["IMG_PATH"]

                    n_samples_exper = cfg["n_samples_exper"]
                    n_clients_tot = cfg["n_clients_tot"]
                    datapoints_per_cli = cfg["datapoints_per_cli"]
                    n_train_samples_exper = n_clients_tot * datapoints_per_cli

                    print(f"dataset_type_exper: {dataset_type_exper}")
                    print(f"n_train_samples_exper: {n_train_samples_exper}")

                    num_total_rounds_glob = max([client_config_exper_parallel[k]["communication_rounds"]
                                                 for k in client_config_exper_parallel])

                    testacc_rd_cutoff = cfg["testacc_rd_cutoff"]
                    print(f"testacc_rd_cutoff: {testacc_rd_cutoff}")

                    local_batch_size = cfg["local_batch_size"]
                    optim_type = cfg["optim_type"]

                    lr_gen = cfg["lr_gen"]
                    lr_disc = cfg["lr_disc"]
                    lr_disc_decay = cfg["lr_disc_decay"]
                    cont_optim_state = cfg["cont_optim_state"]

                    compute_fid = cfg["compute_fid"]
                    is_qcnn = cfg["is_qcnn"]
                    amp_embed = cfg["amp_embed"]
                    shared_pca = cfg["shared_pca"]
                    local_pca = cfg["local_pca"]
                    alt_zeros_init = cfg["alt_zeros_init"]
                    multiclassifier_type = cfg["multiclassifier_type"]
                    train_models_parallel = cfg["train_models_parallel"]
                    heirarchical_train = cfg["heirarchical_train"]
                    morepers = cfg["morepers"]

                    # Ansätze: pick the "type name" from first client just like your original logging string
                    first_cli_type = list(qal_block_params.keys())[0]
                    ansatz_type = qal_block_params[first_cli_type][0]
                    print(f"ansatz_type: {ansatz_type}")

                    device = cfg["device"]
                    print(f"device: {device}")

                    max_workers = os.cpu_count()
                    mp_ctx = mp.get_context(cfg.get("mp_start_method", "spawn"))

                    # Build log folder name to match your original scheme
                    log_data_folder = (
                        f"{IMG_PATH}/data_logs_{dataset_type_exper}_classes_{'_'.join(classes_exper)}"
                        f"_n_samples_{n_samples_exper}_n_train_{n_train_samples_exper}"
                        f"_qfl_gen_{num_total_rounds_glob}rounds_le_{local_epochs}_bs_{local_batch_size}"
                        f"_opt_{optim_type}_mpd_mlt_lg_{lr_gen}_ld_{lr_disc}_dqdm_ba_sp_qcnn_{is_qcnn}"
                        f"_ba_sm_ae_{amp_embed}_dr_nce_mc_{multiclassifier_type}_mp_{train_models_parallel}"
                        f"_tclip_{morepers}_{random_state}_{ansatz_type}_ldd_{lr_disc_decay}_2_10l_nos"
                    )

                    if not os.path.exists(log_data_folder):
                        os.makedirs(log_data_folder, exist_ok=True)

                    print("_".join(classes_exper))
                    print(f"log_data_folder: {log_data_folder}")

                    with open(f"{log_data_folder}/main_stdout.txt", "w") as fout, open(f"{log_data_folder}/main_stderr.txt", "w") as ferr:
                        with contextlib.redirect_stdout(fout), contextlib.redirect_stderr(ferr):

                            print(f"qubits_and_layer_types_block_params: {qal_block_params}")
                            print(f"classes_exper: {classes_exper}")
                            print(f"random_state: {random_state}")

                            mp.set_start_method(cfg.get("mp_start_method", "spawn"), force=True)
                            mp_ctx = mp.get_context(cfg.get("mp_start_method", "spawn"))

                            gen_betas = tuple(cfg["gen_betas"])
                            disc_betas = tuple(cfg["disc_betas"])

                            # Optional: load previous data/params (gracefully skip if file missing)
                            data_logs_prev = None
                            initial_supp_params = None

                            # Build default filenames used in your original code
                            base_init_dir = (
                                f"{IMG_PATH}/initial_configs_{n_clients_tot}cli_{datapoints_per_cli}tr_"
                                f"{n_samples_exper - (datapoints_per_cli * n_clients_tot)}test_{qubits_layers_str}"
                            )
                            default_prev_data = (
                                f"{base_init_dir}/data_logs_n_samples_{n_samples_exper}"
                                f"_dataset_type_{dataset_type_exper}_classes_{'_'.join(classes_exper)}"
                                f"_train_models_parallel_{train_models_parallel}_feature_skew_0.0_label_skew_None"
                                f"_local_pca_{local_pca}_shared_pca_{shared_pca}_gen_False_qcnn_{is_qcnn}"
                                f"_{random_state}.pkl"
                            )
                            default_prev_params = (
                                f"{base_init_dir}/client_params_dict_n_samples_{n_samples_exper}"
                                f"_dataset_type_{dataset_type_exper}_classes_{'_'.join(classes_exper)}"
                                f"_train_models_parallel_{train_models_parallel}_feature_skew_0.0_label_skew_None"
                                f"_local_pca_{local_pca}_shared_pca_{shared_pca}_gen_False_qcnn_{is_qcnn}"
                                f"_{random_state}.pkl"
                            )

                            if cfg["use_prev_data"]:
                                prev_data_path = cfg.get("prev_data_path_override") or default_prev_data
                                if os.path.exists(prev_data_path):
                                    with open(prev_data_path, "rb") as f:
                                        data_logs_prev = pickle.load(f)
                                    print(f"Loaded previous data_logs from: {prev_data_path}")
                                else:
                                    print(f"[WARN] Previous data file not found: {prev_data_path} (continuing with None)")

                            if cfg["use_prev_params"]:
                                prev_params_path = cfg.get("prev_params_path_override") or default_prev_params
                                if os.path.exists(prev_params_path):
                                    with open(prev_params_path, "rb") as f:
                                        initial_supp_params = pickle.load(f)
                                    print(f"Loaded previous client params from: {prev_params_path}")
                                else:
                                    print(f"[WARN] Previous params file not found: {prev_params_path} (continuing with None)")

                            resc_invpca = cfg["resc_invpca"]
                            pennylane_interface = cfg["pennylane_interface"]
                            opt_layers = cfg["opt_layers"]

                            # Prepare qubits_and_layers_to_add_block_params
                            qal_add_params_cfg = cfg.get("qubits_and_layers_to_add_block_params")
                            if qal_add_params_cfg:
                                qal_add_params = {int(k): [tuple(p) for p in v] for k, v in qal_add_params_cfg.items()}
                            else:
                                qal_add_params = build_qal_add_params(
                                    qal_block_params, tuple(cfg["default_block_param"])
                                )

                            # Set log level
                            # log_level = cfg["log_level"]

                            slog.set_glob_loglevel(cfg.get("log_level", "INFO"))

                            # Run the QFL workflow (same signature you had, but fed by config)
                            data_logs = run_qfl_experiments_parallel_multiprocess(
                                client_config_exper_parallel,
                                classes=classes_exper,
                                n_samples=n_samples_exper,
                                dataset_type=dataset_type_exper,
                                agg_strategy=cfg["agg_strategy"],
                                test_frac=((n_samples_exper - n_train_samples_exper) / n_samples_exper),
                                val_frac=0.0,
                                random_state=random_state,
                                pool_in=cfg["pool_in"],
                                local_batch_size=local_batch_size,
                                local_lr=cfg["local_lr"],
                                shots=cfg["shots"],
                                debug=cfg["debug"],
                                save_pkl=cfg["save_pkl"],
                                mask_grads=cfg["mask_grads"],
                                init_client_data_dict=data_logs_prev,
                                qubits_and_layers_to_add_block_params=qal_add_params,
                                train_models_parallel=train_models_parallel,
                                same_init=cfg["same_init"],
                                feature_skew=cfg["feature_skew"],
                                label_skew=cfg["label_skew"],
                                local_pca=local_pca,
                                do_lda=cfg["do_lda"],
                                feat_sel_type=cfg["feat_sel_type"],
                                amp_embed=amp_embed,
                                feat_ordering=cfg["feat_ordering"],
                                morepers=morepers,
                                custom_debug=cfg["custom_debug"],
                                shared_pca=shared_pca,
                                heirarchical_train=heirarchical_train,
                                generative=cfg["generative"],
                                use_torch=cfg["use_torch"],
                                fed_pca_mocked=cfg["fed_pca_mocked"],
                                lr_gen=lr_gen,
                                lr_disc=lr_disc,
                                noise_func=generate_latent_noise,
                                criterion_func=nn.BCELoss,
                                targ_data_folder_prefix=cfg["targ_data_folder_prefix"],
                                gen_data_folder_prefix=cfg["gen_data_folder_prefix"],
                                device=device,
                                fid_batch_size=cfg["fid_batch_size"],
                                max_workers=max_workers,
                                mp_ctx=mp_ctx,
                                log_data_folder=log_data_folder,
                                initial_supp_params=initial_supp_params,
                                optim_type=optim_type,
                                gen_betas=gen_betas,
                                disc_betas=disc_betas,
                                resc_invpca=resc_invpca,
                                compute_fid=compute_fid,
                                is_qcnn=is_qcnn,
                                pennylane_interface=pennylane_interface,
                                opt_layers=opt_layers,
                                alt_zeros_init=alt_zeros_init,
                                multiclassifier_type=multiclassifier_type,
                                testacc_rd_cutoff=testacc_rd_cutoff,
                                qubits_and_layer_types_block_params=qal_block_params,
                                lr_disc_decay=lr_disc_decay,
                                cont_optim_state=cont_optim_state,
                            )

                            print("running after the main() function.")
                            print(f"data_logs.keys(): {list(data_logs.keys())}")
                            print(f"data_logs['clients_data_dict']: {data_logs.get('clients_data_dict')}")

                            with open(f"{log_data_folder}/result_datalogs.pkl", "wb") as file:
                                pickle.dump(data_logs, file)


if __name__ == "__main__":
    main()
