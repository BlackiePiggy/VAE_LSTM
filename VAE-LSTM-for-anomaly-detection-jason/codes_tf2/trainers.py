import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pyplot import savefig
from scipy.stats import multivariate_normal
from base import BaseTrain


class VAETrainer(BaseTrain):
    def __init__(self, model, data, config):
        super(VAETrainer, self).__init__(model, data, config)
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=config['learning_rate_vae']
        )

    def train_epoch(self):
        """Train for one epoch"""
        self.cur_epoch = self.model.cur_epoch.numpy()

        # Create dataset for training
        train_dataset = tf.data.Dataset.from_tensor_slices(self.data.train_set_vae['data'])
        train_dataset = train_dataset.shuffle(
            buffer_size=60000,
            seed=self.cur_epoch
        ).batch(self.config['batch_size'])

        # Training loop
        self.n_train_iter = self.data.n_train_vae // self.config['batch_size']
        train_loss_cur_epoch = 0.0

        for i, batch in enumerate(train_dataset.take(self.n_train_iter)):
            loss = self.train_step(batch)
            self.model.increment_global_step()
            self.train_loss.append(np.squeeze(loss))
            train_loss_cur_epoch += loss

        self.train_loss_ave_epoch.append(train_loss_cur_epoch / self.n_train_iter)

        # Validation
        val_dataset = tf.data.Dataset.from_tensor_slices(self.data.val_set_vae['data'])
        val_dataset = val_dataset.batch(self.config['batch_size'])

        self.n_val_iter = self.data.n_val_vae // self.config['batch_size']
        val_loss_cur_epoch = 0.0

        for i, batch in enumerate(val_dataset.take(self.n_val_iter)):
            val_loss = self.val_step(batch)
            val_loss_cur_epoch += val_loss

        self.val_loss_ave_epoch.append(val_loss_cur_epoch / self.n_val_iter)

        # Test metrics
        test_metrics = self.test_step()

        # Save model
        checkpoint_path = f"{self.config['checkpoint_dir']}model_epoch_{self.cur_epoch}"
        self.model.save(checkpoint_path)

        # Print epoch summary
        print(f"{self.cur_epoch}/{self.config['num_epochs_vae'] - 1}, "
              f"test loss: -elbo: {test_metrics['elbo_loss']:.4f}, "
              f"recons_loss_weighted: {test_metrics['weighted_reconstruction_error']:.4f}, "
              f"recons_loss_ls: {test_metrics['ls_reconstruction_error']:.4f}, "
              f"KL_loss: {test_metrics['KL_loss']:.4f}, "
              f"code_std_dev: {test_metrics['std_dev_norm']}")

        print(f"Loss on training and val sets:\n"
              f"train: {self.train_loss_ave_epoch[self.cur_epoch]:.4f}, "
              f"val: {self.val_loss_ave_epoch[self.cur_epoch]:.4f}")

        print(f"Current sigma2: {self.model.sigma2:.7f}")

        # Save variables and plot results
        self.save_variables_VAE()
        self.plot_reconstructed_signal()
        self.generate_samples_from_prior()
        self.plot_train_and_val_loss()

    @tf.function
    def train_step(self, batch):
        """Single training step"""
        with tf.GradientTape() as tape:
            # Forward pass
            decoded, code_mean, code_std_dev, code_sample = self.model(batch, training=True)

            # Compute losses
            losses = self.model.define_loss(batch, (code_mean, code_std_dev), decoded)
            loss = losses['elbo_loss']

        # Compute gradients and update weights
        gradients = tape.gradient(loss, self.model.trainable_variables)
        clipped_gradients = self.model.clip_gradients(gradients)
        self.optimizer.apply_gradients(zip(clipped_gradients, self.model.trainable_variables))

        return loss

    def val_step(self, batch):
        """Single validation step"""
        # Forward pass without training
        decoded, code_mean, code_std_dev, _ = self.model(batch, training=False)

        # Compute losses
        losses = self.model.define_loss(batch, (code_mean, code_std_dev), decoded)

        # Store validation metrics
        self.val_loss.append(np.squeeze(losses['elbo_loss']))
        self.recons_loss_val.append(np.squeeze(losses['ls_reconstruction_error']))
        self.KL_loss_val.append(losses['KL_loss'])

        return losses['elbo_loss']

    def test_step(self):
        """Run test evaluation"""
        test_batch = self.data.test_set_vae['data']

        # Forward pass
        decoded, code_mean, code_std_dev, _ = self.model(test_batch, training=False)

        # Compute losses
        losses = self.model.define_loss(test_batch, (code_mean, code_std_dev), decoded)

        # Store test outputs
        self.output_test = decoded
        self.test_sigma2.append(np.squeeze(self.model.sigma2.numpy()))

        return losses

    def plot_reconstructed_signal(self):
        """Plot reconstructed signals"""
        input_images = np.squeeze(self.data.test_set_vae['data'])
        decoded_images = np.squeeze(self.output_test.numpy())
        n_images = 20

        for j in range(self.config['n_channel']):
            fig, axs = plt.subplots(4, 5, figsize=(18, 10))
            fig.subplots_adjust(hspace=.4, wspace=.4)
            axs = axs.ravel()

            for i in range(n_images):
                if self.config['n_channel'] == 1:
                    axs[i].plot(input_images[i])
                    axs[i].plot(decoded_images[i])
                else:
                    axs[i].plot(input_images[i, :, j])
                    axs[i].plot(decoded_images[i, :, j])

                axs[i].grid(True)
                axs[i].set_xlim(0, self.config['l_win'])
                axs[i].set_ylim(-5, 5)

                if i == 19:
                    axs[i].legend(('original', 'reconstructed'))

            plt.suptitle(f'Channel {j}')
            plt.savefig(f"{self.config['result_dir']}test_reconstructed_{self.cur_epoch}_{j}.pdf")
            plt.close()

    def generate_samples_from_prior(self):
        """Generate samples from prior distribution"""
        rv = multivariate_normal(
            np.zeros(self.config['code_size']),
            np.diag(np.ones(self.config['code_size']))
        )

        n_images = 20
        samples_code_prior = rv.rvs(n_images)
        sampled_images = self.model.decode(samples_code_prior)
        sampled_images = np.squeeze(sampled_images.numpy())

        for j in range(self.config['n_channel']):
            fig, axs = plt.subplots(4, 5, figsize=(18, 10))
            fig.subplots_adjust(hspace=.4, wspace=.4)
            axs = axs.ravel()

            for i in range(n_images):
                if self.config['n_channel'] == 1:
                    axs[i].plot(sampled_images[i])
                else:
                    axs[i].plot(sampled_images[i, :, j])

                axs[i].grid(True)
                axs[i].set_xlim(0, self.config['l_win'])
                axs[i].set_ylim(-5, 5)

            plt.suptitle(f'Channel {j}')
            plt.savefig(f"{self.config['result_dir']}generated_samples_{self.cur_epoch}_{j}.pdf")
            plt.close()