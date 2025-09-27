from quorus.logging.custom_slog import print_cust
import torch

"""### QGAN Expectations Helper"""

def get_expectations(noise, qnode, n_qubits_func, block_params_list):
    # Non-linear Transform
    ret_expvals = qnode(noise, block_params_list)
    # print_cust(f"get_expectations, type(ret_expvals): {type(ret_expvals)}")
    # print_cust(f"get_expectations, type(weights): {type(weights)}")
    # print_cust(f"get_expectations, len(weights): {len(weights)}")
    # for block_param in block_params_list:
    #     print_cust(f"get_expectations, block_param.shape: {block_param.shape}")
    ret_expvals_sum = sum(ret_expvals)
    print_cust(f"get_expectations, type(ret_expvals_sum): {type(ret_expvals_sum)}")
    print_cust(f"get_expectations, ret_expvals_sum: {ret_expvals_sum}")
    # weights.zero_grad()
    # ret_expvals_sum.backward()
    # print_cust(f"get_expectations, weights.grad: {weights.grad}")
    # print_cust(f"get_expectations, np.linalg.norm(weights.grad): {np.linalg.norm(weights.grad)}")
    # grad_fn_debug = qml.grad(qnode, argnum=0)
    # grads_test = grad_fn_debug(weights, n_qubits_circ=n_qubits_func, qubit_depth_dict=qubit_depth_dict, alpha=alpha, ret_exp=True)
    # print_cust(f"get_expectations, grads_test: {grads_test}")
    # print_cust(f"get_expectations, np.linalg.norm(grads_test): {np.linalg.norm(grads_test)}")
    # print_cust(f"partial_measure, probs: {probs}")
    # print_cust(f"partial_measure, torch.sum(probs): {torch.sum(probs)}")
    # probsgiven0 = probs[: (2 ** (n_qubits_func - n_a_qubits_func))]
    # # print_cust(f"partial_measure, probsgiven0: {probsgiven0}")
    # # print_cust(f"partial_measure, torch.sum(probsgiven0): {torch.sum(probsgiven0)}")
    # probsgiven0 /= torch.sum(probsgiven0)
    # # print_cust(f"partial_measure, torch.max(probsgiven0): {torch.max(probsgiven0)}")

    # # Post-Processing
    # probsgiven = probsgiven0 / torch.max(probsgiven0)
    # print_cust(f"partial_measure, probsgiven: {probsgiven}")
    torch_ret_expvals = torch.stack(ret_expvals)
    print_cust(f"get_expectations, type(torch_ret_expvals): {type(torch_ret_expvals)}")
    print_cust(f"get_expectations, torch_ret_expvals.shape: {torch_ret_expvals.shape}")
    return torch_ret_expvals