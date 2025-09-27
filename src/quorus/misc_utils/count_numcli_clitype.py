def count_numcli_clitype(config_dict):
  """
  Function that counts the number of clients of each specified client type.

  Parameters:
      config_dict (dict): The configuration dictionary for the run.

  Returns:
      A string specifying the counts of each client type.
  """
  list_counts = []
  for cli_type, cli_configs in config_dict.items():
    list_counts.append(str(cli_configs["num_clients"]))
    list_counts.append(str(cli_type))
  return "_".join(list_counts)
