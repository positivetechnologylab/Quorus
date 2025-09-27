"""### Save Tensors to Folder Function"""

from PIL import Image
from quorus.logging.custom_slog import print_cust
import os
import torch
from pennylane import numpy as np

def save_tensors_to_folder(img_tensors, folder_name, input_prefix, img_ext="png"):
    img_tensors = img_tensors.detach()
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    print_cust(f"save_tensors_to_folder, img_tensors.dtype: {img_tensors.dtype}")
    print_cust(f"save_tensors_to_folder, img_tensors.max(): {img_tensors.max()}")
    print_cust(f"save_tensors_to_folder, img_tensors.min(): {img_tensors.min()}")
    assert img_tensors.max() <= 1.0, f"save_tensors_to_folder, img_tensors.max() > 1.0: {img_tensors.max()}"
    assert img_tensors.min() >= 0.0, f"save_tensors_to_folder, img_tensors.min() < 0.0: {img_tensors.min()}"
    if not img_tensors.dtype == torch.uint8 and img_tensors.max() <= 1.0:
        print_cust(f'save_tensors_to_folder: renormalizing data')
        img_tensors = (img_tensors * 255).round()
    img_tensors = img_tensors.numpy().astype(np.uint8)
    for img_idx, img_arr in enumerate(img_tensors):
        img_obj = Image.fromarray(img_arr)
        img_obj.save(f"{folder_name}/{input_prefix}_{img_idx}.{img_ext}")