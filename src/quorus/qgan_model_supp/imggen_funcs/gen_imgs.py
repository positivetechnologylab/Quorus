import math
from quorus.logging.custom_slog import print_cust

"""### Generate Images from Generator Function"""

def generate_images_generator(generator_obj, many_samples_noise, pca_rescaler, ret_pca_feats=False, resc_invpca=True):
    # many_samples_noise = latent_noise_func(n_imgs, generator_obj.n_qubits_gen, generator_obj.device)
    print_cust(f"generate_images_generator, resc_invpca: {resc_invpca}")
    many_samples_pcafeats = generator_obj(many_samples_noise)
    many_samples_imgs_glob = pca_rescaler.invpca_to_img(many_samples_pcafeats, resc_invpca=resc_invpca)
    if not ret_pca_feats:
        return many_samples_imgs_glob.view(many_samples_imgs_glob.shape[0], math.isqrt(many_samples_imgs_glob.shape[1]), math.isqrt(many_samples_imgs_glob.shape[1]))
    else:
        return many_samples_imgs_glob.view(many_samples_imgs_glob.shape[0], math.isqrt(many_samples_imgs_glob.shape[1]), math.isqrt(many_samples_imgs_glob.shape[1])), many_samples_pcafeats