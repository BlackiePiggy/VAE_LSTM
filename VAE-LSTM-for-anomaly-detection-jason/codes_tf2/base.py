import tensorflow as tf
import random
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pyplot import savefig
import time


class BaseDataGenerator:
    def __init__(self, config):
        self.config = config

    def separate_train_and_val_set(self, n_win):
        n_train = int(np.floor((n_win * 0.9)))
        n_val = n_win - n_train
        idx_train = random.sample(range(n_win), n_train)
        idx_val = list(set(idx_train) ^ set(range(n_win)))
        return idx_train, idx_val, n_train, n_val


class BaseModel(tf.keras.Model):
    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.config = config
        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.cur_epoch = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.two_pi = tf.constant(2 * np.pi)

    def increment_global_step(self):
        self.global_step.assign_add(1)

    def increment_epoch(self):
        self.cur_epoch.assign_add(1)

    def save(self, checkpoint_path):
        """Save model weights"""
        self.save_weights(checkpoint_path)
        print("Model saved.")

    def load(self, checkpoint_path):
        """Load model weights"""
        try:
            self.load_weights(checkpoint_path)
            print("Model loaded.")
        except:
            print("No model loaded.")

    def define_loss(self, original_signal, encoded_output, decoded_output):
        """Define loss computation"""
        # Extract mean and std from encoder output
        code_mean, code_std_dev = encoded_output

        # KL divergence loss - analytical result
        KL_loss = 0.5 * (tf.reduce_sum(tf.square(code_mean), 1)
                         + tf.reduce_sum(tf.square(code_std_dev), 1)
                         - tf.reduce_sum(tf.math.log(tf.square(code_std_dev)), 1)
                         - self.config['code_size'])
        KL_loss = tf.reduce_mean(KL_loss)

        # Standard deviation norm
        std_dev_norm = tf.reduce_mean(code_std_dev, axis=0)

        # Reconstruction errors
        ls_reconstruction_error = tf.reduce_mean(
            tf.reduce_sum(tf.square(original_signal - decoded_output), [1, 2])
        )

        # Weighted reconstruction error (assuming sigma2 is a trainable parameter)
        sigma2 = getattr(self, 'sigma2', 1.0)  # Default to 1.0 if not defined
        weighted_reconstruction_error = ls_reconstruction_error / (2 * sigma2)

        # ELBO loss
        input_dims = original_signal.shape[1] * original_signal.shape[2]
        sigma_regularizer = (input_dims / 2) * tf.math.log(sigma2)
        two_pi_term = (input_dims / 2) * tf.constant(2 * np.pi)

        elbo_loss = (two_pi_term + sigma_regularizer +
                     0.5 * weighted_reconstruction_error + KL_loss)

        return {
            'elbo_loss': elbo_loss,
            'KL_loss': KL_loss,
            'ls_reconstruction_error': ls_reconstruction_error,
            'weighted_reconstruction_error': weighted_reconstruction_error,
            'std_dev_norm': std_dev_norm
        }

    def clip_gradients(self, gradients, clip_value=1.0):
        """Clip gradients to prevent exploding gradients"""
        clipped_gradients = []
        for grad in gradients:
            if grad is not None:
                clipped_gradients.append(tf.clip_by_value(grad, -clip_value, clip_value))
            else:
                clipped_gradients.append(grad)
        return clipped_gradients


class BaseTrain:
    def __init__(self, model, data, config):
        self.model = model
        self.config = config
        self.data = data

        # Keep a record of the training result
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
        self.start_time = time.time()
        for cur_epoch in range(0, self.config['num_epochs_vae'], 1):
            self.train_epoch()

            # Compute current execution time
            self.current_time = time.time()
            elapsed_time = (self.current_time - self.start_time) / 60
            est_remaining_time = ((self.current_time - self.start_time) / (cur_epoch + 1) *
                                  (self.config['num_epochs_vae'] - cur_epoch - 1)) / 60
            print(f"Already trained for {elapsed_time:.2f} min; Remaining {est_remaining_time:.2f} min.")

            self.model.increment_epoch()

    def save_variables_VAE(self):
        """Save training variables for later inspection"""
        file_name = f"{self.config['result_dir']}{self.config['exp_name']}-batch-{self.config['batch_size']}-epoch-{self.config['num_epochs_vae']}-code-{self.config['code_size']}-lr-{self.config['learning_rate_vae']}.npz"

        np.savez(file_name,
                 iter_list_val=self.iter_epochs_list,
                 train_loss=self.train_loss,
                 val_loss=self.val_loss,
                 n_train_iter=self.n_train_iter,
                 n_val_iter=self.n_val_iter,
                 recons_loss_train=self.recons_loss_train,
                 recons_loss_val=self.recons_loss_val,
                 KL_loss_train=self.KL_loss_train,
                 KL_loss_val=self.KL_loss_val,
                 num_para_all=self.model.count_params(),
                 sigma2=self.test_sigma2)

    def plot_train_and_val_loss(self):
        """Plot training and validation loss"""
        # Plot training and validation loss over epochs
        plt.clf()
        plt.figure(num=1, figsize=(8, 6))
        plt.plot(self.train_loss, 'b-')
        plt.plot(self.iter_epochs_list, self.val_loss_ave_epoch, 'r-')
        plt.legend(('training loss (total)', 'validation loss'))
        plt.title('training loss over iterations (val @ epochs)')
        plt.ylabel('total loss')
        plt.xlabel('iterations')
        plt.grid(True)
        plt.savefig(f"{self.config['result_dir']}/loss.png")

        # Plot individual components of validation loss
        plt.clf()
        plt.figure(num=1, figsize=(8, 6))
        plt.plot(self.recons_loss_val, 'b-')
        plt.plot(self.KL_loss_val, 'r-')
        plt.legend(('Reconstruction loss', 'KL loss'))
        plt.title('validation loss breakdown')
        plt.ylabel('loss')
        plt.xlabel('num of batch')
        plt.grid(True)
        plt.savefig(f"{self.config['result_dir']}/val-loss.png")

        # Plot sigma2 over training
        plt.clf()
        plt.figure(num=1, figsize=(8, 6))
        plt.plot(self.test_sigma2, 'b-')
        plt.title('sigma2 over training')
        plt.ylabel('sigma2')
        plt.xlabel('iter')
        plt.grid(True)
        plt.savefig(f"{self.config['result_dir']}/sigma2.png")