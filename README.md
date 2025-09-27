# Layerwise Federated Learning for Heterogeneous Quantum Clients using Quorus

## Overview
This repository provides an open-source, flexible package for federated learning using quantum clients with heterogenous quantum models.

## Key Features
1. Provides a ready-to-use executable file that can be configured by supplying your own `.jsonc` file.
2. No outside FL packages are used in this repository. The FL framework is implemented from scratch (although heavily uses Pennylane for quantum circuit evaluation/gradient computation, and PyTorch for gradient calculation).

The package can also be adapted adapted by injecting your own functions from `src/quorus/cli/qfl_main_test.py`.

## Usage
1. In your terminal, run `git clone https://github.com/positivetechnologylab/quorus.git` to clone the repository.
2. Run `cd quorus`.
3. Create a virtual environment if necessary (our code uses Python 3.12.8), and run `python -m pip install -e .` to install the `quorus` package.
4. Run `quorus-exp --config <path_to_config_jsonc>`.
5. The results will be stored in a generated log folder.

## Configurations
Please see `json_configs` for example configuration files to pass in. In a future version, we hope to add more complete documentation on the configurations.

## Requirements
The requirements and specific versions are provided in `pyproject.toml`. In future versions of this package, we hope to make these requirements more loose (for now, we provide the specific versions that were used in our experiments.)
`src/quorus/cli/ibm_hardware_runs.py` require a variable "IBMQ_TOKEN" and "IBMQ_CRN" in a `.env` file to connect to the IBM Quantum Cloud. In a future work, we hope to include the hardware runs as an easy-to-use executable.

## Side Effects
The scripts will create folders containing the logs for each experiment. In addition, `.pkl` files may be created (for debugging).

## Repository Structure
- [**`README.md`**](README.md): Repository readme with setup and execution instructions.
- [**`src`**](src): Contains the source code for the repository.
- [**`pyproject.toml`**](pyproject.toml): The .toml file for the repository.
- [**`json_configs`**](json_configs): Sample .jsonc configuration files to use. Please refer to these when writing your own configurations. In a future version, we hope to provide more robust input validation.
- [**`.gitignore`**](.gitignore): Gitignore for the repository. By default, the generated log folders will be ignored by git (as they can be quite large).
- [**`initial_configs_5cli_128tr_3000test_10_1_11_2_12_3_13_4_14_5`**](initial_configs_5cli_128tr_3000test_10_1_11_2_12_3_13_4_14_5): The initial parameters, as well as training/testing data, for where one client has 1 layer, one client has 2 layers, one has 3, one has 4, and one has 5.
- [**`initial_configs_5cli_128tr_3000test_10_2_11_2_12_2_13_2_14_2`**](initial_configs_5cli_128tr_3000test_10_2_11_2_12_2_13_2_14_2): The initial parameters, as well as training/testing data, for where all clients have 2 layers.
- [**`initial_configs_5cli_128tr_3000test_10_2_11_3_12_4_13_5_14_6`**](initial_configs_5cli_128tr_3000test_10_2_11_3_12_4_13_5_14_6): The initial parameters, as well as training/testing data, for where one client has 2 layers, one client has 3 layers, one has 4, one has 5, and one has 6.
- [**`initial_configs_5cli_128tr_3000test_10_2_11_4_12_6_13_8_14_10`**](initial_configs_5cli_128tr_3000test_10_2_11_4_12_6_13_8_14_10): The initial parameters, as well as training/testing data, for where one client has 2 layers, one client has 4 layers, one has 6, one has 8, and one has 10.

## Future Work
In a future version, we hope to add unit and integration tests and integrate with a CI/CD pipeline for automated deployment.

## Copyright
Copyright © 2025 Positive Technology Lab. All rights reserved. For permissions, contact ptl@rice.edu.