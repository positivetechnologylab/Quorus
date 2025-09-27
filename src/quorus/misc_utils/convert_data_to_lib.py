import torch
from pennylane import numpy as np

"""## Convert Data to Tensors"""

def convert_data_to_lib(clients_data_dict, math_int=np):
  """
  Function that converts the data dictionariy for each client into the specified math interface.

  Parameters:
      clients_data_dict (dict): A dictionary containing the data for each client.
      math_int (module): Either np or torch.

  Returns:
      clients_data_dict, where all data is converted to the types in math_int. Note that clients_data_dict is
      also mutated.
  """
  for client_type, clients_data in clients_data_dict.items():
    for data in clients_data:
      train_data = data[0][0]
      train_labels = data[0][1]
      # NOTE: if I'm getting datatype issues, I can change this to float32 (b/c converting from np arrs, these are float64.)
      if math_int == torch:
        data[0][0] = torch.tensor(train_data, dtype=torch.float32)
        data[0][1] = torch.tensor(train_labels, dtype=torch.float32)
      val_data = data[1][0]
      val_labels = data[1][1]
      if math_int == torch:
        data[1][0] = torch.tensor(val_data, dtype=torch.float32)
        data[1][1] = torch.tensor(val_labels, dtype=torch.float32)
  return clients_data_dict