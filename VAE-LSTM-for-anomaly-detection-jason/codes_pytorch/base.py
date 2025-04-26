import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt


class BaseDataGenerator:
    def __init__(self, config):
        self.config = config

    # Separate training and val sets
    def separate_train_and_val_set(self, n_win):
        n_train = int(np.floor((n_win * 0.9)))
        n_val = n_win - n_train
        idx_train = random.sample(range(n_win), n_train)
        idx_val = list(set(range(n_win)) - set(idx_train))
        return idx_train, idx_val, n_train, n_val


class BaseModel(nn.Module):
    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.config = config
        # Initialize counters
        self.global_step = 0
        self.cur_epoch = 0

    # Save function that saves the checkpoint in the path defined in the config file
    def save(self, path=None):
        if path is None:
            path = f"{self.config['checkpoint_dir']}model_{self.global_step}.pth"
        print("Saving model...")
        torch.save({
            'model_state_dict': self.state_dict(),
            'global_step': self.global_step,
            'cur_epoch': self.cur_epoch,
        }, path)
        print("Model saved.")

    # Load latest checkpoint from the experiment path defined in the config file
    def load(self, path=None):
        if path is None:
            # Find all checkpoint files
            import os
            checkpoint_files = [f for f in os.listdir(self.config['checkpoint_dir'])
                                if
                                f.endswith('.pth') and os.path.isfile(os.path.join(self.config['checkpoint_dir'], f))]

            if not checkpoint_files:
                print("No model loaded.")
                return False

            # Get the latest checkpoint file
            latest_checkpoint = sorted(checkpoint_files, key=lambda x: int(x.split('_')[1].split('.')[0]))[-1]
            path = os.path.join(self.config['checkpoint_dir'], latest_checkpoint)

        if os.path.exists(path):
            print(f"Loading model checkpoint {path} ...")
            checkpoint = torch.load(path)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.global_step = checkpoint['global_step']
            self.cur_epoch = checkpoint['cur_epoch']
            print("Model loaded.")
            return True
        else:
            print("No model loaded.")
            return False


class BaseTrain:
    def __init__(self, model, data, config, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.config = config
        self.data = data
        self.device = device

        # Initialize records for training progress
        self.train_loss = []
        self.val_loss = []
        self.train_loss_ave_epoch = []
        self.val_loss_ave_epoch = []
        self.recons_loss_train = []
        self.recons_loss_val = []
        self.KL_loss_train = []
        self.KL_loss_val = []
        self.sample_std_dev_train = []
        self.sample_std_dev_val = []
        self.iter_epochs_list = []
        self.test_sigma2 = []

    def train(self):
        """
        This is the main training loop
        :return:
        """
        raise NotImplementedError

    def train_step(self):
        """
        Implementation of the train step
        :return:
        """
        raise NotImplementedError

    def train_epoch(self):
        """
        Implement the logic of each epoch
        :return:
        """
        raise NotImplementedError