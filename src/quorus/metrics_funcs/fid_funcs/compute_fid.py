from quorus.logging.custom_slog import print_cust
from quorus.qgan_model_supp.imggen_funcs.gen_imgs import generate_images_generator
import torch
from pytorch_fid.fid_score import calculate_fid_given_paths
from quorus.qgan_model_supp.imggen_funcs.save_imgs import save_tensors_to_folder

"""### FID Calculation Function"""


def compute_fid_to_data(generator, noise_func, targ_data_folder_name, gen_data_folder_name, n_samples_gen, device, fid_batch_size=1000, resc_invpca=True):
  # n_samples_noise = targ_data.shape[0]
  print_cust(f"compute_fid_to_data, resc_invpca: {resc_invpca}")
  print_cust(f"compute_fid_to_data, fid_batch_size: {fid_batch_size}")

  print_cust(f"compute_fid_to_data, n_samples_noise: {n_samples_gen}")
  n_qubits_noise = generator.n_qubits_gen
  noise_samples = noise_func(n_samples_gen, n_qubits_noise, device)
  pca_rescaler = generator.pca_rescaler
  generator_gen_imgs = generate_images_generator(generator, noise_samples, pca_rescaler, resc_invpca=resc_invpca)

  print_cust(f"compute_fid_to_data, generator_gen_imgs: {generator_gen_imgs}")

  if generator_gen_imgs.max() >= 1.0 or generator_gen_imgs.min() < 0.0:
    print_cust(f"compute_fid_to_data, generator_gen_imgs exceeds range of pixel values. generator_gen_imgs.max(): {generator_gen_imgs.max()}, generator_gen_imgs.min(): {generator_gen_imgs.min()}")
    # maybe print out the number of pixels that needed to be clamped?
    generator_gen_imgs = torch.clamp(generator_gen_imgs, max=1.0, min=0.0)
    print_cust(f"compute_fid_to_data, after clamping, generator_gen_imgs.max(): {generator_gen_imgs.max()}, generator_gen_imgs.min(): {generator_gen_imgs.min()}")

  save_tensors_to_folder(generator_gen_imgs, gen_data_folder_name, "img")

  # temporary override. the main time bottleneck appears to be FID calculation.
  # I suppose if it is still taking a long time, then reduce the testing data size.
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  fid_score = calculate_fid_given_paths([gen_data_folder_name, targ_data_folder_name], fid_batch_size, device, 2048)

  return fid_score