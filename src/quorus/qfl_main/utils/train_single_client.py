import copy
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn

from quorus.factory_funcs.patchquantumgen_factory import build_patchquantumgen
from quorus.factory_funcs.pca_discriminator_factory import build_pca_discriminator
from quorus.factory_funcs.variationalquantumclassif_factory import build_variationalquantumclassifier
from quorus.logging.custom_slog import print_cust
from quorus.metrics_funcs.agg_metrics_func.agg_metrics_func_eval import compute_metrics_angle_param_batch
from quorus.misc_utils.create_feat_list_expansion_data import create_feat_list_expansion_data
from quorus.training_funcs.pennylane_training.train_mod_pennylane import train_epochs_angle_param_adam
from quorus.training_funcs.torch_training.qgan_training.qgan_training_funcs import train_models
from quorus.training_funcs.torch_training.varquantumclassifier_training.train_models_classifier import train_models_classifier

"""### QFL Experiment Train Single Client Helper Function"""

import contextlib

def _train_single_client(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker executed in a separate process.

    Returns a dict with:
        client_type, client_idx,
        trained_params, minibatch_losses, validation_losses,
        train_acc, train_acc_stdev, train_acc_topk,
        test_acc,  test_acc_stdev,  test_acc_topk, test_loss
    """
    # --- unpack ------------------------------------------------------------
    # NOTE, layers, for client_params_indiv: it is ALL OF THE CLIENT PARAMS. list of block params is also there.
    (client_type, client_idx, client_params_indiv,
     client_train_data, client_val_data, testing_data,
     cur_model_size, min_size_clients, pool_in, feat_sel_type,
     train_models_parallel, amp_embed, shots, local_batch_size,
     local_lr, num_local_epochs, conv_layers, feat_ordering, classes,
     qnode_builder, num_total_rounds, round_num, grad_mask, generative, lr_gen, lr_disc,
     noise_func, criterion_func, log_data_folder, device, client_optims_indiv, optim_type,
     gen_betas, disc_betas, use_torch, pennylane_interface, opt_layers, layer_types_list,
     loss_type, lr_disc_decay, cont_optim_state) = (job[k] for k in
         ("client_type","client_idx","client_params_indiv",
          "client_train_data","client_val_data","testing_data",
          "cur_model_size","min_size_clients","pool_in","feat_sel_type",
          "train_models_parallel","amp_embed","shots","local_batch_size",
          "local_lr","num_local_epochs","conv_layers","feat_ordering","classes",
          "qnode_builder","num_total_rounds","round_num","grad_mask","generative",
          "lr_gen","lr_disc","noise_func","criterion_func","log_data_folder","device",
          "client_optims_indiv","optim_type","gen_betas","disc_betas","use_torch","pennylane_interface","opt_layers",
          "layer_types_list", "loss_type", "lr_disc_decay", "cont_optim_state"))

    with open(f"{log_data_folder}/cli_type_{client_type}_client_idx_{client_idx}_round_num_{round_num}_stdout.txt", "w") as fout_loc, open(f"{log_data_folder}/cli_type_{client_type}_client_idx_{client_idx}_round_num_{round_num}_stderr.txt", "w") as ferr_loc:
      with contextlib.redirect_stdout(fout_loc), contextlib.redirect_stderr(ferr_loc):

        print_cust(f"_train_single_client, opt_layers: {opt_layers}")

        print_cust(f"_train_single_client, device: {device}")

        print_cust(f"_train_single_client, qnode_builder: {qnode_builder}")

        print_cust(f"_train_single_client, lr_gen: {lr_gen}, lr_disc: {lr_disc}")

        # TODO: won't work for nongenerative case, but not going to worry about that for now.
        print_cust(f"_train_single_client, client_optims_indiv: {client_optims_indiv}")

        print_cust(f"_train_single_client, optim_type: {optim_type}")

        print_cust(f"_train_single_client, gen_betas: {gen_betas}, disc_betas: {disc_betas}")

        print_cust(f"_train_single_client, grad_mask: {grad_mask}, generative: {generative}")

        print_cust(f"_train_single_client, client_train_data: {client_train_data}")

        print_cust(f"_train_single_client, conv_layers: {conv_layers}")

        print_cust(f"_train_single_client, round_num: {round_num}, lr_disc_decay: {lr_disc_decay}")

        print_cust(f"_train_single_client, cont_optim_state: {cont_optim_state}")

        print_cust(f"_train_single_client, feat_ordering: {feat_ordering}, num_total_rounds: {num_total_rounds}")

        if not generative and conv_layers == 0:
          # NOTE, layers: override cur_model_size
          num_qubits_bps = []
          for block_param in client_params_indiv[5]:
            num_qubits_bps.append(block_param.shape[1])

          print_cust(f"_train_single_client, num_qubits_bps: {num_qubits_bps}")

          # NOTE, layers: this is a logical override.
          cur_model_size = max(num_qubits_bps)

          print_cust(f"_train_single_client, not generative and conv_layers == 0, cur_model_size: {cur_model_size}")

        # ----------------------------------------------------------------------
        # build feature list + QNode locally (nothing un‑pickle‑able is sent in)
        # DONE (ish): TOMODIFY, Layers: don't have any expansion data for the layers expansion (for now)
        # ^ TOMODIFY, LAYERS, HACK (ish): not called if conv_layers 0.
        if not generative and conv_layers > 0:
          print_cust(f"_train_single_client, calling create_feat_list_expansion_data")
          feature_list, expansion_data = create_feat_list_expansion_data(
              cur_model_size, conv_layers, expand=cur_model_size>min_size_clients,
              pool_in=pool_in, min_qubits_noexpand=min_size_clients,
              train_models_parallel=train_models_parallel,
              feat_sel_type=feat_sel_type)
        else:
          feature_list, expansion_data = list(range(cur_model_size)), []
        print_cust(f"_train_single_client, feature_list: {feature_list}, expansion_data: {expansion_data}")
        if not generative and conv_layers > 0:
          # DONE: TOMODIFY, layers: inject pennylane_interface here.
          qnode = qnode_builder(cur_model_size, conv_layers,
                                expansion_data, n_classes=len(classes), pennylane_interface=pennylane_interface)
        # DONE: TOMODIFY, layers: have another option for creating a quantumclassifer using the builder, *BUT*
        # note that the input now is the list of tensors representing the variational parameters.
        # DONE: TOMODIFY, layers: inject pennylane_interface here.
        elif not generative:
          print_cust(f"_train_single_client, conv_layers should be 0, conv_layers: {conv_layers}")

          # NOCHANGE: TOMODIFY, depthFL: add an arg for number of classifiers to create.
          # ^ TOMODIFY, depthFL: have some kind of argument to CONTROL the number/amount of classifiers I want to create.
          # classifier_data_comps = [cur_model_size, conv_layers, expansion_data, len(classes), pennylane_interface, layer_types_list, device]
          # DONE: BUG, depthFL: classifier_data_comps needs to include device somehow.... but doesn't work if it's now a dictionary.. probably, change build_vqc
          # function.
          classifier_data_comps = {
            "n_data": cur_model_size,
            "conv_layers": conv_layers,
            "expansion_data": expansion_data,
            "n_classes": len(classes),
            "pennylane_interface": pennylane_interface,
            "layer_types_list": layer_types_list,
            "device": device
          }
          cur_classifier_params = client_params_indiv
          print_cust(f"_train_single_client, cur_classifier_params: {cur_classifier_params}")
          cli_classifier_loc = build_variationalquantumclassifier(classifier_data_comps, cur_classifier_params, qnode_builder)
          print_cust(f"_train_single_client, cli_classifier_loc: {cli_classifier_loc}")
        else:
          (gen_state_dict, gen_metadata) = client_params_indiv[5][0]
          (disc_state_dict, disc_metadata) = client_params_indiv[5][1]
          print_cust(f"_train_single_client, gen_state_dict: {gen_state_dict}, gen_metadata: {gen_metadata}")
          print_cust(f"_train_single_client, disc_state_dict: {disc_state_dict}, disc_metadata: {disc_metadata}")
          cli_generator_loc = build_patchquantumgen(gen_metadata, gen_state_dict, qnode_builder)
          cli_disc_loc = build_pca_discriminator(disc_metadata, disc_state_dict)
          print_cust(f"_train_single_client, cli_generator_loc: {cli_generator_loc}")
          print_cust(f"_train_single_client, cli_disc_loc: {cli_disc_loc}")




        # ----------------------------------------------------------------------
        # slice / reorder data --------------------------------------------------
        if amp_embed:
            # TOMODIFY, layers, amp embed: comment this out. hopefully, don't need to do this; just do vanilla ampencode.
            # TOMODIFY, layers: what I should do is have a condition; if feature_list is just the range of qubits, then
            # don't do anything. Otherwise, call reorder_amplitude_data.
            print_cust(f"_train_single_client, amp_embed is True")
            # X_train = reorder_amplitude_data(client_train_data[0], feature_list)
            # X_val   = reorder_amplitude_data(client_val_data[0],  feature_list)
            # X_test  = reorder_amplitude_data(testing_data[0],      feature_list)
            X_train = client_train_data[0]
            X_val = client_val_data[0]
            X_test = testing_data[0]
        else:
            X_train = client_train_data[0][:, feature_list]
            X_val   = client_val_data[0][:,  feature_list]
            if not generative:
              X_test  = testing_data[0][:,     feature_list]
              # DONE (NO CHANGE): TOMODIFY, layers: X_test needs to be ordered in terms of feature_list.
              # ^ I mean, variational QNN is not generative

        # ----------------------------------------------------------------------
        # local training --------------------------------------------------------
        if not generative and conv_layers > 0:
          trained_params, minibatch_losses, validation_losses = train_epochs_angle_param_adam(
              copy.deepcopy(client_params_indiv),
              X_train, client_train_data[1],
              X_val,   client_val_data[1],
              n_epochs=num_local_epochs, shots=shots,
              batch_size=local_batch_size, lr=local_lr,
              qnode=qnode, trainable_mask=grad_mask)           # grad_mask handled outside

          # ----------------------------------------------------------------------
          # metrics ---------------------------------------------------------------
          if use_torch:
            math_int = torch
          else:
            math_int = np

          print_cust(f"_train_single_client, math_int: {math_int}")

          train_acc, train_acc_stdev, train_acc_topk, _ = \
              compute_metrics_angle_param_batch(trained_params, X_train,
                                                client_train_data[1],
                                                layers=conv_layers, shots=shots,
                                                batch_size=local_batch_size,
                                                qnode=qnode, math_int=math_int)
        elif not generative:
          # DONE: TOMODIFY, layers: call a different training function
          # NOT IMPLEMENTED (should be implemented in the main func): and use the updated compute_metrics_angle_param_batch function.
          # DONE: TOMODIFY, layers: initialize my optimizer from my optimizer state_dict, as before.
          print_cust(f"_train_single_client, conv_layers should be 0, conv_layers: {conv_layers}")
          client_classifier = cli_classifier_loc
          print_cust(f"_train_single_client, client_classifier: {client_classifier}")
          curr_cli_blockparams = client_classifier.q_params
          list_params = []
          if opt_layers is None:
            list_params = [block_param_torch for block_param_torch in curr_cli_blockparams]
          else:
            for layer_idx in opt_layers:
              layer_opt_param = curr_cli_blockparams[layer_idx]
              list_params.append(layer_opt_param)

          if optim_type == "sgd":
            client_optim_classifier = torch.optim.SGD(nn.ParameterList(list_params), lr=lr_disc * (lr_disc_decay ** round_num))
          elif optim_type == "adam":
            client_optim_classifier = torch.optim.Adam(nn.ParameterList(list_params), lr=lr_disc * (lr_disc_decay ** round_num), betas=disc_betas)

          if client_optims_indiv is not None and cont_optim_state:
            # NOTE: this is a SHALLOW copy of loading state dict from previous client optimizer.
            # HACK, depthFL: loading in state dict into client_optim_classifier only in non-SGD is a hack; figure out
            # how to dynamically change LR.
            print_cust(f"_train_single_client, loading state dict for client_optim_classifier")
            client_optim_classifier.load_state_dict(client_optims_indiv[5])

          print_cust(f"_train_single_client, client_optim_classifier: {client_optim_classifier}")

          print_cust(f"_train_single_client, X_train: {X_train}")

          # TODO, 8/4, 6:38 PM: continue here
          train_metrics_dict = train_models_classifier(local_batch_size, client_classifier, client_optim_classifier, criterion_func(), X_train, client_train_data[1], device, num_local_epochs, loss_type=loss_type)

          trained_classifier_params = client_classifier.q_params

          print_cust(f"_train_single_client, trained_classifier_params: {trained_classifier_params}")

          trained_optim_classifier = client_optim_classifier.state_dict()

          print_cust(f"_train_single_client, trained_optim_classifier: {trained_optim_classifier}")

          # MODIFY, layers: call compute_metrics_angle_param_batch to get the training accuracies.

        else:
          # I mean, I could return the minibatch_losses, validation_losses, but idt I will.
          print_cust(f"_train_single_client, generative is True (QGAN training)")
          client_generator = cli_generator_loc
          print_cust(f"_train_single_client, client_generator: {client_generator}")
          client_discriminator = cli_disc_loc
          print_cust(f"_train_single_client, client_discriminator: {client_discriminator}")
          # client_optim_gen = torch.optim.SGD(nn.ParameterList([client_generator.q_params[-1]]), lr=lr_gen)
          # client_optim_disc = torch.optim.SGD(client_discriminator.parameters(), lr=lr_disc)
          if optim_type == "sgd":
            client_optim_gen = torch.optim.SGD(nn.ParameterList([client_generator.q_params[-1]]), lr=lr_gen)
            client_optim_disc = torch.optim.SGD(client_discriminator.parameters(), lr=lr_disc * (lr_disc_decay ** round_num))
          elif optim_type == "adam":
            client_optim_gen = torch.optim.Adam(nn.ParameterList([client_generator.q_params[-1]]), lr=lr_gen, betas=gen_betas)
            client_optim_disc = torch.optim.Adam(client_discriminator.parameters(), lr=lr_disc * (lr_disc_decay ** round_num), betas=disc_betas)

          if client_optims_indiv is not None:
            client_optim_gen.load_state_dict(client_optims_indiv[5][0])
            client_optim_disc.load_state_dict(client_optims_indiv[5][1])

          print_cust(f"_train_single_client, client_optim_gen: {client_optim_gen}")
          print_cust(f"_train_single_client, client_optim_disc: {client_optim_disc}")
          # maybe concatenate in the train with val data? not using val data otherwise
          # can use test_result_imgs for visualization, later, if I'd like.
          print_cust(f"_train_single_client, X_train: {X_train}")
          test_result_imgs, train_metrics_dict = train_models(client_generator.n_qubits_gen, local_batch_size, client_generator, client_discriminator, client_optim_disc, client_optim_gen,
                                          noise_func=noise_func, criterion=criterion_func(), train_data=X_train, device=device, image_size=0, compressed_img_size=0, max_num_epochs=num_local_epochs,
                                          n_qubits_small=0, gen_pcas=True, disc_img_size=None, pca_disc=True)
          print_cust(f"_train_single_client, train_metrics_dict: {train_metrics_dict}")
          trained_gen_params = client_generator.state_dict()
          trained_disc_params = client_discriminator.state_dict()

          print_cust(f"_train_single_client, trained_gen_params: {trained_gen_params}")
          print_cust(f"_train_single_client, trained_disc_params: {trained_disc_params}")

          trained_optim_gen = client_optim_gen.state_dict()
          trained_optim_disc = client_optim_disc.state_dict()

          print_cust(f"_train_single_client, trained_optim_gen: {trained_optim_gen}")
          print_cust(f"_train_single_client, trained_optim_disc: {trained_optim_disc}")

        test_acc, test_acc_stdev, test_acc_topk, test_loss = (None, None, None, None)

        # test_acc, test_acc_stdev, test_acc_topk, test_loss = \
        #     compute_metrics_angle_param_batch(trained_params, X_test,
        #                                       testing_data[1],
        #                                       layers=conv_layers, shots=shots,
        #                                       batch_size=local_batch_size,
        #                                       qnode=qnode)

        # if num_total_rounds > 10:
        #   if (round_num + 1) % 10 == 0:
        #     test_acc, test_acc_stdev, test_acc_topk, test_loss = compute_metrics_angle_param_batch(trained_params, X_test, testing_data[1], layers=conv_layers, shots=shots, batch_size=local_batch_size, qnode=qnode)
        #   else:
        #     test_acc, test_acc_stdev, test_acc_topk, test_loss = (None, None, None, None)
        # else:
        #   test_acc, test_acc_stdev, test_acc_topk, test_loss = compute_metrics_angle_param_batch(trained_params, X_test, testing_data[1], layers=conv_layers, shots=shots, batch_size=local_batch_size, qnode=qnode)
        # ----------------------------------------------------------------------
        if not generative and conv_layers > 0:
          ret_dict = dict(client_type=client_type,
                    client_idx=client_idx,
                    trained_params=trained_params,
                    minibatch_losses=minibatch_losses,
                    validation_losses=validation_losses,
                    train_acc=train_acc,
                    test_acc=test_acc,
                    test_loss=test_loss,
                    train_acc_stdev=train_acc_stdev,
                    test_acc_stdev=test_acc_stdev,
                    train_acc_topk=train_acc_topk,
                    test_acc_topk=test_acc_topk)
        elif not generative:
          # NOTE, layers: trying to match the same interface as in the generative case.
          # TOMODIFY, layers: add additional training metrics (training accuracies) here.
          ret_dict = dict(client_type=client_type,
                          client_idx=client_idx,
                          trained_disc_params=trained_classifier_params,
                          trained_optim_disc=trained_optim_classifier,
                          train_metrics_dict=train_metrics_dict)
        else:
          ret_dict = dict(client_type=client_type,
                          client_idx=client_idx,
                          trained_gen_params=trained_gen_params,
                          trained_disc_params=trained_disc_params,
                          trained_optim_gen=trained_optim_gen,
                          trained_optim_disc=trained_optim_disc,
                          train_metrics_dict=train_metrics_dict)
        # DONE: TOMODIFY, layers: return new object, with trained layers params (send back the list of block params for consistency), and send back the
        # torch optimizer state dict. (Later -- can send back a training metrics dictionary.)
        # DONE (just did the point above): TOMODIFY, layers: for minimum viable things, send back whatever metrics are necessary.

    return ret_dict