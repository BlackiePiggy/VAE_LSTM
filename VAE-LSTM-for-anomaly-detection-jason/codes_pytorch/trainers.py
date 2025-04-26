import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig, figure
import torch
from scipy.stats import multivariate_normal
import time
from base import BaseTrain


class vaeTrainer(BaseTrain):
    def __init__(self, model, data, config, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(vaeTrainer, self).__init__(model, data, config, device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate_vae'])

    def train(self):
        self.start_time = time.time()

        for epoch in range(self.config['num_epochs_vae']):
            self.cur_epoch = epoch
            self.train_epoch()

            # Compute current execution time
            self.current_time = time.time()
            elapsed_time = (self.current_time - self.start_time) / 60
            est_remaining_time = (self.current_time - self.start_time) / (epoch + 1) * (
                    self.config['num_epochs_vae'] - epoch - 1) / 60

            print(f"Already trained for {elapsed_time:.2f} min; Remaining {est_remaining_time:.2f} min.")

            # Update epoch counter in model
            self.model.cur_epoch += 1

    def train_epoch(self):
        # Setup data loaders
        train_tensor = torch.tensor(self.data.train_set_vae['data'], dtype=torch.float32)
        val_tensor = torch.tensor(self.data.val_set_vae['data'], dtype=torch.float32)
        test_tensor = torch.tensor(self.data.test_set_vae['data'], dtype=torch.float32)

        train_dataset = torch.utils.data.TensorDataset(train_tensor)
        val_dataset = torch.utils.data.TensorDataset(val_tensor)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False
        )

        # Training
        self.model.train()
        self.n_train_iter = len(train_loader)
        train_loss_cur_epoch = 0.0

        for i, (batch_image,) in enumerate(train_loader):
            batch_image = batch_image.to(self.device)
            loss = self.train_step(batch_image)
            self.train_loss.append(loss)
            train_loss_cur_epoch += loss

            if i == len(train_loader) - 1:
                test_results = self.test_step(test_tensor.to(self.device))

        self.train_loss_ave_epoch.append(train_loss_cur_epoch / self.n_train_iter)

        # Validation
        self.model.eval()
        self.iter_epochs_list.append(self.n_train_iter * (self.cur_epoch + 1))
        self.n_val_iter = len(val_loader)
        val_loss_cur_epoch = 0.0

        with torch.no_grad():
            for batch_image, in val_loader:
                batch_image = batch_image.to(self.device)
                val_loss = self.val_step(batch_image)
                val_loss_cur_epoch += val_loss

        self.val_loss_ave_epoch.append(val_loss_cur_epoch / self.n_val_iter)

        # Save the model
        self.model.save()

        # Print progress
        print(
            f"{self.cur_epoch}/{self.config['num_epochs_vae'] - 1}, "
            f"test loss: -elbo: {test_results['elbo_loss']:.4f}, "
            f"recons_loss_weighted: {test_results['weighted_recon_loss']:.4f}, "
            f"recons_loss_ls: {test_results['recon_loss']:.4f}, "
            f"KL_loss: {test_results['kl_loss']:.4f}, "
            f"sigma_regularisor: {test_results['sigma_regularizer']:.4f}, "
            f"code_std_dev: {test_results['code_std_norm']}"
        )

        print(f"Loss on training and val sets:\n"
              f"train: {self.train_loss_ave_epoch[-1]:.4f}, "
              f"val: {self.val_loss_ave_epoch[-1]:.4f}")

        print(f"Current sigma2: {test_results['sigma2']:.7f}")

        # Save variables and generate plots
        self.save_variables_VAE()
        self.plot_reconstructed_signal(test_tensor)
        self.generate_samples_from_prior()
        self.plot_train_and_val_loss()

    def train_step(self, batch_image):
        self.optimizer.zero_grad()

        # Learning rate decay
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.config['learning_rate_vae'] * (0.98 ** self.cur_epoch)

        # Forward pass
        x_recon, mean, std_dev, _ = self.model(batch_image)

        # Compute loss
        loss_dict = self.model.compute_loss(batch_image, x_recon, mean, std_dev)

        # Backward pass
        loss_dict['elbo_loss'].backward()
        self.optimizer.step()

        return loss_dict['elbo_loss'].item()

    def val_step(self, batch_image):
        # Forward pass
        x_recon, mean, std_dev, _ = self.model(batch_image)

        # Compute loss
        loss_dict = self.model.compute_loss(batch_image, x_recon, mean, std_dev)

        self.val_loss.append(loss_dict['elbo_loss'].item())
        self.recons_loss_val.append(loss_dict['recon_loss'].item())
        self.KL_loss_val.append(loss_dict['kl_loss'].item())

        return loss_dict['elbo_loss'].item()

    def test_step(self, test_data):
        self.model.eval()
        with torch.no_grad():
            # Forward pass
            x_recon, mean, std_dev, _ = self.model(test_data)

            # Compute loss
            loss_dict = self.model.compute_loss(test_data, x_recon, mean, std_dev)

            # Get sigma2
            sigma2 = self.model.get_sigma2().item()

            # Calculate code_std_norm
            code_std_norm = torch.mean(std_dev, dim=0).cpu().numpy()

            # Store outputs for plotting
            self.output_test = x_recon.cpu().detach().numpy()
            self.test_sigma2.append(sigma2)

        return {
            'elbo_loss': loss_dict['elbo_loss'].item(),
            'weighted_recon_loss': loss_dict['weighted_recon_loss'].item(),
            'recon_loss': loss_dict['recon_loss'].item(),
            'kl_loss': loss_dict['kl_loss'].item(),
            'sigma_regularizer': loss_dict['sigma_regularizer'].item(),
            'code_std_norm': code_std_norm,
            'sigma2': sigma2
        }

    def plot_reconstructed_signal(self, test_data):
        self.model.eval()
        with torch.no_grad():
            input_images = test_data.cpu().numpy().squeeze()
            decoded_images = self.output_test.squeeze()

            n_images = 20

            # Plot reconstructed images for each channel
            for j in range(self.config['n_channel']):
                fig, axs = plt.subplots(4, 5, figsize=(18, 10), edgecolor='k')
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
                savefig(self.config['result_dir'] + f'test_reconstructed_{self.cur_epoch}_{j}.pdf')
                fig.clf()
                plt.close()

    def generate_samples_from_prior(self):
        self.model.eval()
        with torch.no_grad():
            # Generate samples from prior
            n_images = 20
            sampled_images = self.model.sample_from_prior(n_images).cpu().numpy()
            sampled_images = np.squeeze(sampled_images)

            # Plot samples for each channel
            for j in range(self.config['n_channel']):
                fig, axs = plt.subplots(4, 5, figsize=(18, 10), edgecolor='k')
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
                savefig(self.config['result_dir'] + f'generated_samples_{self.cur_epoch}_{j}.pdf')
                fig.clf()
                plt.close()

    def save_variables_VAE(self):
        # Save variables for later inspection
        file_name = f"{self.config['result_dir']}{self.config['exp_name']}-batch-{self.config['batch_size']}-epoch-{self.config['num_epochs_vae']}-code-{self.config['code_size']}-lr-{self.config['learning_rate_vae']}.npz"

        np.savez(
            file_name,
            iter_list_val=self.iter_epochs_list,
            train_loss=self.train_loss,
            val_loss=self.val_loss,
            n_train_iter=self.n_train_iter,
            n_val_iter=self.n_val_iter,
            recons_loss_train=self.recons_loss_train,
            recons_loss_val=self.recons_loss_val,
            KL_loss_train=self.KL_loss_train,
            KL_loss_val=self.KL_loss_val,
            # Count trainable parameters
            num_para_all=sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            sigma2=self.test_sigma2
        )

    def plot_train_and_val_loss(self):
        # Plot training and validation loss over epochs
        plt.clf()
        figure(num=1, figsize=(8, 6))
        plt.plot(self.train_loss, 'b-')
        plt.plot(self.iter_epochs_list, self.val_loss_ave_epoch, 'r-')
        plt.legend(('training loss (total)', 'validation loss'))
        plt.title('training loss over iterations (val @ epochs)')
        plt.ylabel('total loss')
        plt.xlabel('iterations')
        plt.grid(True)
        savefig(self.config['result_dir'] + '/loss.png')

        # Plot individual components of validation loss over epochs
        plt.clf()
        figure(num=1, figsize=(8, 6))
        plt.plot(self.recons_loss_val, 'b-')
        plt.plot(self.KL_loss_val, 'r-')
        plt.legend(('Reconstruction loss', 'KL loss'))
        plt.title('validation loss breakdown')
        plt.ylabel('loss')
        plt.xlabel('num of batch')
        plt.grid(True)
        savefig(self.config['result_dir'] + '/val-loss.png')

        # Plot sigma2 over training
        plt.clf()
        figure(num=1, figsize=(8, 6))
        plt.plot(self.test_sigma2, 'b-')
        plt.title('sigma2 over training')
        plt.ylabel('sigma2')
        plt.xlabel('iter')
        plt.grid(True)
        savefig(self.config['result_dir'] + '/sigma2.png')