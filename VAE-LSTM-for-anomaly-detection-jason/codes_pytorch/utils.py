import json
import os
import argparse
from datetime import datetime
import torch


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(dictionary)
    """
    # Parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    return config_dict


def save_config(config):
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%d-%b-%Y-%H-%M")
    filename = config['result_dir'] + f'training_config_{timestampStr}.txt'
    config_to_save = json.dumps(config)
    with open(filename, "w") as f:
        f.write(config_to_save)


def process_config(json_file):
    config = get_config_from_json(json_file)

    # Create directories to save experiment results and trained models
    if config['load_dir'] == "default":
        save_dir = f"../experiments/local-results/{config['exp_name']}/{config['dataset']}/batch-{config['batch_size']}"
    else:
        save_dir = config['load_dir']

    # Specify the saving folder name for this experiment
    if config['TRAIN_sigma'] == 1:
        save_name = f"{config['exp_name']}-{config['dataset']}-{config['l_win']}-{config['l_seq']}-{config['code_size']}-trainSigma"
    else:
        save_name = f"{config['exp_name']}-{config['dataset']}-{config['l_win']}-{config['l_seq']}-{config['code_size']}-fixedSigma-{config['sigma']}"

    config['summary_dir'] = os.path.join(save_dir, save_name, "summary/")
    config['result_dir'] = os.path.join(save_dir, save_name, "result/")
    config['checkpoint_dir'] = os.path.join(save_dir, save_name, "checkpoint/")
    config['checkpoint_dir_lstm'] = os.path.join(save_dir, save_name, "checkpoint/lstm/")

    return config


def create_dirs(dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return exit_code: 0:success -1:failed
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print(f"Creating directories error: {err}")
        exit(-1)


def count_trainable_parameters(model):
    """Count the trainable parameters in a PyTorch model"""
    total_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The total number of trainable parameters is: {total_parameters}')
    return total_parameters


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file'
    )
    args = argparser.parse_args()
    return args