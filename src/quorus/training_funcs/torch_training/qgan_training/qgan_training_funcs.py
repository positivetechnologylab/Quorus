"""## Core Training Functions, QGAN

### Core Training Batch Function
"""

from quorus.logging.custom_slog import print_cust
import copy
from quorus.param_processing.torch_funcs.state_dict_diffs import l2_state_dict_difference
from pennylane import numpy as np
import torch

# from torchviz import make_dot

# from itertools import chain

# import uuid

# import time

def train_step(generator, discriminator, optD, optG, criterion, results, real_labels, fake_labels, counter, fixed_noise, real_data, fake_data, compressed_img_size, alpha=1.0, disc_img_size=None, pca_disc=True):

    print_cust(f"train_step, disc_img_size: {disc_img_size}")
    # Training the discriminator
    discriminator.zero_grad()

    # TODO: add assertions here about gradient magnitudes; should all be zero (disc at least, should)
    for name, p in generator.named_parameters():
        print_cust(f"train_step, generator param name: {name}, p.requires_grad: {p.requires_grad}")
        if p.grad is None:
            print_cust(f"train_step, generator param name: {name}, no grad")
        else:
            print_cust(f"train_step, generator param name: {name}, p.grad.norm(): {p.grad.norm()}")
            # print_cust(name, p.grad.norm())

    for name, p in discriminator.named_parameters():
        print_cust(f"train_step, discriminator param name: {name}, p.requires_grad: {p.requires_grad}")
        if p.grad is None:
            print_cust(f"train_step, discriminator param name: {name}, no grad")
        else:
            print_cust(f"train_step, discriminator param name: {name}, p.grad.norm(): {p.grad.norm()}")



    print_cust(f"train_step, real_data.shape: {real_data.shape}")
    print_cust(f"train_step, fake_data.shape: {fake_data.shape}")

    disc_beforeinf_statedict = copy.deepcopy(discriminator.state_dict())
    gen_beforeinf_statedict = copy.deepcopy(generator.state_dict())

    # if not pca_disc:
    #     real_data = pad_images_newdim(real_data, disc_img_size ** 2)
    #     fake_data = pad_images_newdim(fake_data, disc_img_size ** 2)

    # print_cust(f"train_step, after padding, real_data.shape: {real_data.shape}")
    # print_cust(f"train_step, after padding, fake_data.shape: {fake_data.shape}")

    # TODO: detach real_data too???
    outD_real = discriminator(real_data.detach(), alpha).view(-1)

    # TODO: see why in the world we detach????
    outD_fake = discriminator(fake_data.detach(), alpha).view(-1)

    disc_afterinf_statedict = copy.deepcopy(discriminator.state_dict())
    gen_afterinf_statedict = copy.deepcopy(generator.state_dict())

    print_cust(f"train_step, real_data.shape: {real_data.shape}")
    print_cust(f"train_step, fake_data.shape: {fake_data.shape}")
    print_cust(f"train_step, outD_real.shape: {outD_real.shape}")
    print_cust(f"train_step, outD_real: {outD_real}")
    print_cust(f"train_step, outD_fake: {outD_fake}")
    print_cust(f"train_step, real_labels.shape: {real_labels.shape}")
    print_cust(f"train_step, real_labels: {real_labels}")
    print_cust(f"train_step, fake_labels.shape: {fake_labels.shape}")
    print_cust(f"train_step, fake_labels: {fake_labels}")

    # could do some assertions here, but can do it later
    print_cust(f"train_step, change in disc params after inf: {l2_state_dict_difference(disc_beforeinf_statedict, disc_afterinf_statedict)}")
    print_cust(f"train_step, change in gen params after inf: {l2_state_dict_difference(gen_beforeinf_statedict, gen_afterinf_statedict)}")

    errD_real = criterion(outD_real, real_labels)

    errD_fake = criterion(outD_fake, fake_labels)

    # err_intermed = errD_real + errD_fake

    # param_dict_intermed = {
    #     **{f"gen.{k}": v for k, v in generator.named_parameters()},
    #     **{f"disc.{k}": v for k, v in discriminator.named_parameters()}
    # }

    # n_qubits_stamp = generator.n_qubits_gen

    # stamp_disc = time.strftime("%Y%m%d-%H%M%S")

    # dot_disc = make_dot(err_intermed,
    #             params=param_dict_intermed,
    #             show_attrs=True,
    #             show_saved=True)

    # dot_disc.render(f"discriminator_autograd_graph_batchsize_one_pcadisc_n_qubits_{n_qubits_stamp}_super_{stamp_disc}", format="png")

    # Propagate gradients
    errD_real.backward()
    errD_fake.backward()

    print_cust(f"train_step, after discriminator .backward()")

    # Didn't clear grads for generator here, so that's why they r non zero I think.

    for name, p in generator.named_parameters():
        print_cust(f"train_step, generator param name: {name}, p.requires_grad: {p.requires_grad}")
        if p.grad is None:
            print_cust(f"train_step, generator param name: {name}, no grad")
        else:
            print_cust(f"train_step, generator param name: {name}, p.grad.norm(): {p.grad.norm()}")
            # print_cust(name, p.grad.norm())


    disc_param_grad_norms = []

    for name, p in discriminator.named_parameters():
        print_cust(f"train_step, discriminator param name: {name}, p.requires_grad: {p.requires_grad}")
        if p.grad is None:
            print_cust(f"train_step, discriminator param name: {name}, no grad")
        else:
            print_cust(f"train_step, discriminator param name: {name}, p.grad.norm(): {p.grad.norm()}")
            disc_param_grad_norms.append(p.grad.norm())

    print_cust(f"train_step, disc_param_grad_norms: {disc_param_grad_norms}")

    errD = errD_real + errD_fake
    orig_generator_params = generator.q_params[0].clone().detach()
    optD.step()
    print_cust(f"train_step, change in generator parameters: {np.linalg.norm(generator.q_params[0].clone().detach() - orig_generator_params)}")
    disc_afterdiscstep_statedict = copy.deepcopy(discriminator.state_dict())
    gen_afterdiscstep_statedict = copy.deepcopy(generator.state_dict())
    print_cust(f"train_step, change in disc params after disc step: {l2_state_dict_difference(disc_afterinf_statedict, disc_afterdiscstep_statedict)}")
    print_cust(f"train_step, change in gen params after disc step: {l2_state_dict_difference(gen_afterinf_statedict, gen_afterdiscstep_statedict)}")

    # Training the generator
    generator.zero_grad()
    outD_fake = discriminator(fake_data, alpha).view(-1)
    print_cust(f"train_step, generator loss, outD_fake: {outD_fake}")
    errG = criterion(outD_fake, real_labels)

    # dot = make_dot(errG,
    #             params=dict(generator.named_parameters()),
    #             show_attrs=True,
    #             show_saved=True)

    # stamp_gen = time.strftime("%Y%m%d-%H%M%S")

    # dot.render(f"generator_autograd_graph_batchsize_one_pcadisc_n_qubits_{n_qubits_stamp}_super_{stamp_gen}", format="png")

    errG.backward()

    # print_cust(f"train_step, grad_log_glob: {grad_log_glob}")

    print_cust(f"train_step, generator gradient norms")

    generator_param_grad_norms = []

    for name, p in generator.named_parameters():
        print_cust(f"train_step, generator param name: {name}, p.requires_grad: {p.requires_grad}")
        if p.grad is None:
            print_cust(f"train_step, generator param name: {name}, no grad")
        else:
            print_cust(f"train_step, generator param name: {name}, p.grad.norm(): {p.grad.norm()}")
            generator_param_grad_norms.append(p.grad.norm())
            # print_cust(name, p.grad.norm())

    print_cust(f"train_step, generator_param_grad_norms: {generator_param_grad_norms}")
    print_cust(f"train_step, discriminator gradient norms")

    for name, p in discriminator.named_parameters():
        print_cust(f"train_step, generator param name: {name}, p.requires_grad: {p.requires_grad}")
        if p.grad is None:
            print_cust(f"train_step, discriminator param name: {name}, no grad")
        else:
            print_cust(f"train_step, discriminator param name: {name}, p.grad.norm(): {p.grad.norm()}")

    optG.step()
    disc_aftergenstep_statedict = copy.deepcopy(discriminator.state_dict())
    gen_aftergenstep_statedict = copy.deepcopy(generator.state_dict())

    print_cust(f"train_step, change in disc params after gen step: {l2_state_dict_difference(disc_afterdiscstep_statedict, disc_aftergenstep_statedict)}")
    print_cust(f"train_step, change in gen params after gen step: {l2_state_dict_difference(gen_afterdiscstep_statedict, gen_aftergenstep_statedict)}")
    # TODO: verify that the discriminator parameters do not change here.

    # Show loss values
    print_cust(f'train_step, Iteration: {counter}, Discriminator Loss: {errD:0.3f}, Generator Loss: {errG:0.3f}')
    if (counter + 1) % 10 == 0:
        # TODO: add some kind of adapter to indicate that I need to transform the size here.
        if not pca_disc:
            test_images = generator(fixed_noise, alpha).view(8,1,compressed_img_size,compressed_img_size).cpu().detach()
            # visualize_generator_imgs(generator, fixed_noise, compressed_img_size, alpha=alpha)
        else:
            test_images = generator(fixed_noise, alpha).cpu().detach()
        # Save images every 50 iterations
        if (counter + 1) % 50 == 0:
            results.append(test_images)

    # verify that testimgs is not anything wild?

    return errD.detach(), errG.detach(), disc_param_grad_norms, generator_param_grad_norms
    # verify that testimgs is not anything wild?

"""### Core Training Epochs Function"""

# from re import A
# TODO: add disc_img_size
def train_models(n_qubits, batch_size, generator, discriminator, optD, optG, noise_func, criterion, train_data, device, image_size, compressed_img_size, max_num_epochs, n_qubits_small=0, gen_pcas=True, disc_img_size=None, pca_disc=True):
    # NOTE: this function mutates the inputs.
    print_cust(f"train_models, train_data.shape: {train_data.shape}")
    n_train_data = train_data.shape[0]

    print_cust(f"train_models, gen_pcas: {gen_pcas}, pca_disc: {pca_disc}")
    print_cust(f"train_models, disc_img_size: {disc_img_size}")
    real_labels = torch.full((batch_size,), 1.0, dtype=torch.float, device=device)
    fake_labels = torch.full((batch_size,), 0.0, dtype=torch.float, device=device)

    # Fixed noise allows us to visually track the generated images throughout training
    # TODO: make the fixed noise dim a variable
    fixed_noise = noise_func(8, n_qubits, device)

    # Collect images for plotting later
    results = []

    log_metrics = {
      "disc_grad_norms": [],
      "gen_grad_norms": [],
      "disc_loss": [],
      "gen_loss": []
    }

    print_cust(f"train_models, log_metrics: {log_metrics}")

    counter = 0

    # print_cust(f"train_models, len(dataloader): {len(dataloader)}")

    print_cust(f"train_models, n_train_data: {n_train_data}")

    # max_num_epochs = math.ceil(num_iter / n_train_data)

    print_cust(f"train_models, max_num_epochs: {max_num_epochs}")

    for epoch_num in range(max_num_epochs):

        print_cust(f"train_models, [SGD] Epoch {epoch_num+1}/{max_num_epochs}")
        indices = np.arange(n_train_data)
        np.random.shuffle(indices)
        for start in range(0, n_train_data, batch_size):

            end = min(start + batch_size, n_train_data)
            batch_indices = indices[start:end]
            if len(batch_indices) < batch_size:
              print_cust(f"train_models, skipping batch because too small. len(batch_indices): {len(batch_indices)}")
              continue
            X_batch = train_data[batch_indices, :]
            print_cust(f"train_models, batch_indices: {batch_indices}")
            print_cust(f"train_models, X_batch.shape: {X_batch.shape}")

            # Data for training the discriminator
            # TODO: add an image processor adapter here.
            # alpha = counter / num_iter

            # FOR NOW
            if gen_pcas:
                alpha = 1.0

            print_cust(f"train_models, alpha: {alpha}")

            if not pca_disc:
                X_batch = X_batch.reshape(-1, image_size * image_size)
            # print_cust(f"data.shape: {data.shape}")
            # print_cust(f"np.linalg.norm(data[0]): {np.linalg.norm(data[0])}")
            # print_cust(f"data[0].min(): {data[0].min()}, data[0].max(): {data[0].max()}")
            # print_cust(f"data[0]: {data[0]}")
            real_data = X_batch.to(device)

            # Noise follwing a uniform distribution in range [0,pi/2)
            noise = noise_func(batch_size, n_qubits, device, n_qubits_small=n_qubits_small)
            # print_cust(f"train_models, noise.shape: {noise.shape}")
            # print_cust(f"train_models, noise: {noise}")
            # print_cust(f"train_models, generator: {generator}")

            fake_data = generator(noise, alpha=alpha)

            # fake_data = morton_permute_batched(fake_data)

            # print_cust(f"fake_data.shape: {fake_data.shape}")
            # print_cust(f"np.linalg.norm(fake_data[0]): {np.linalg.norm(fake_data[0].detach().numpy())}")
            # print_cust(f"fake_data[0]: {fake_data[0]}")

            # print_cust(f"fake_data.shape: {fake_data.shape}")
            # print_cust(f"real_data.shape: {real_data.shape}")

            # if not gen_pcas:
            #     real_data = progressive_resize_batch(real_data, fake_data.shape[1])
            #     real_data /= real_data.max()

            print_cust(f"X_batch.shape: {X_batch.shape}")
            print_cust(f"np.linalg.norm(real_data[0]): {np.linalg.norm(real_data[0])}")
            print_cust(f"real_data[0].min(): {real_data[0].min()}, real_data[0].max(): {real_data[0].max()}")
            print_cust(f"real_data[0]: {real_data[0]}")

            # if ((counter + 1) % 10 == 0) and not pca_disc:
            #     print_cust(f"real_data[0].shape: {real_data[0].shape}")
            #     print_cust(f"real_data[0].reshape(compressed_img_size, compressed_img_size): {real_data[0].reshape(compressed_img_size, compressed_img_size)}")
            #     real_data_reshaped = real_data[0].reshape(compressed_img_size, compressed_img_size)
            #     # 1. convert to NumPy (detach if the tensor tracks gradients)
            #     # --- 2. convert to NumPy for matplotlib --------------------------------------
            #     img4 = real_data_reshaped.numpy()                       # shape (4, 4)

            #     # --- 3. nearest-neighbour up-sampling to 8×8 ---------------------------------
            #     # duplicate every row, then every column
            #     img8 = np.repeat(np.repeat(img4, 2, axis=0), 2, axis=1)   # shape (8, 8)

            #     # --- 4. plot both images side by side ----------------------------------------
            #     fig, axes = plt.subplots(1, 2, figsize=(6, 3))

            #     print_cust(f"img4.shape: {img4.shape}")
            #     print_cust(f"img4: {img4}")
            #     axes[0].imshow(img4, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            #     axes[0].set_title('4×4 original')
            #     axes[0].axis('off')

            #     print_cust(f"img8.shape: {img8.shape}")
            #     print_cust(f"img8: {img8}")
            #     axes[1].imshow(img8, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            #     axes[1].set_title('8×8 nearest-neighbour')
            #     axes[1].axis('off')

            #     plt.tight_layout()
            #     plt.show()

            #     plt.imshow(img8, cmap='gray', vmin=0, vmax=1)

            # NOTE: technically not 'semantically' correct to increment counter here, but functionally, it is the same (FOR NOW)
            print_cust(f"train_models, before counter {counter}, generator: {generator}, discriminator: {discriminator}")
            errD, errG, disc_param_grad_norms, generator_param_grad_norms = train_step(generator, discriminator, optD, optG, criterion, results, real_labels, fake_labels, counter, fixed_noise, real_data, fake_data, compressed_img_size, alpha=alpha, disc_img_size=disc_img_size, pca_disc=pca_disc)
            print_cust(f"train_models, errD: {errD}, errG: {errG}, disc_param_grad_norms: {disc_param_grad_norms}, generator_param_grad_norms: {generator_param_grad_norms}")
            print_cust(f"train_models, after counter {counter}, generator: {generator}, discriminator: {discriminator}")
            counter += 1

            print_cust(f"train_models, counter: {counter}")

            log_metrics["disc_loss"].append(errD)
            log_metrics["gen_loss"].append(errG)
            log_metrics["disc_grad_norms"].append(disc_param_grad_norms)
            log_metrics["gen_grad_norms"].append(generator_param_grad_norms)
            # if counter == num_iter:
            #     break

    print_cust(f"train_models, about to return, log_metrics: {log_metrics}")
    return results, log_metrics