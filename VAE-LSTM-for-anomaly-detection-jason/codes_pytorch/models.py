import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from base import BaseModel


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        init_filters = config['num_hidden_units'] // 16

        if config['l_win'] == 24:
            self.conv1 = nn.Conv2d(1, init_filters, kernel_size=(3, config['n_channel']), stride=(2, 1), padding=(1, 0))
            self.conv2 = nn.Conv2d(init_filters, init_filters * 2, kernel_size=(3, config['n_channel']), stride=(2, 1),
                                   padding=(1, 0))
            self.conv3 = nn.Conv2d(init_filters * 2, init_filters * 4, kernel_size=(3, config['n_channel']),
                                   stride=(2, 1), padding=(1, 0))
            self.conv4 = nn.Conv2d(init_filters * 4, config['num_hidden_units'], kernel_size=(4, config['n_channel']),
                                   stride=1, padding=0)

        elif config['l_win'] == 48:
            self.conv1 = nn.Conv2d(1, init_filters, kernel_size=(3, config['n_channel']), stride=(2, 1), padding=(1, 0))
            self.conv2 = nn.Conv2d(init_filters, init_filters * 2, kernel_size=(3, config['n_channel']), stride=(2, 1),
                                   padding=(1, 0))
            self.conv3 = nn.Conv2d(init_filters * 2, init_filters * 4, kernel_size=(3, config['n_channel']),
                                   stride=(2, 1), padding=(1, 0))
            self.conv4 = nn.Conv2d(init_filters * 4, config['num_hidden_units'], kernel_size=(6, config['n_channel']),
                                   stride=1, padding=0)

        elif config['l_win'] == 144:
            self.conv1 = nn.Conv2d(1, init_filters, kernel_size=(3, config['n_channel']), stride=(4, 1), padding=(1, 0))
            self.conv2 = nn.Conv2d(init_filters, init_filters * 2, kernel_size=(3, config['n_channel']), stride=(4, 1),
                                   padding=(1, 0))
            self.conv3 = nn.Conv2d(init_filters * 2, init_filters * 4, kernel_size=(3, config['n_channel']),
                                   stride=(3, 1), padding=(1, 0))
            self.conv4 = nn.Conv2d(init_filters * 4, config['num_hidden_units'], kernel_size=(3, config['n_channel']),
                                   stride=1, padding=0)

        self.fc = nn.Linear(config['num_hidden_units'], config['code_size'] * 4)
        self.fc_mean = nn.Linear(config['code_size'] * 4, config['code_size'])
        self.fc_logvar = nn.Linear(config['code_size'] * 4, config['code_size'])

    def forward(self, x):
        # Add channel dimension if needed
        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        # Apply appropriate padding for l_win == 24
        if self.config['l_win'] == 24:
            # Symmetric padding on the width dimension
            x = F.pad(x, (0, 0, 4, 4), mode='reflect')

        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))

        x = x.view(x.size(0), -1)  # Flatten
        x = F.leaky_relu(self.fc(x))

        mean = self.fc_mean(x)
        # Ensure std_dev is positive using relu + small constant
        std_dev = F.relu(self.fc_logvar(x)) + 1e-2

        return mean, std_dev


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.config = config

        self.fc = nn.Linear(config['code_size'], config['num_hidden_units'])

        if config['l_win'] == 24:
            self.conv1 = nn.Conv2d(config['num_hidden_units'], config['num_hidden_units'], kernel_size=1, stride=1,
                                   padding=0)
            self.conv2 = nn.Conv2d(config['num_hidden_units'] // 4, config['num_hidden_units'] // 4, kernel_size=(3, 1),
                                   stride=1, padding=(1, 0))
            self.conv3 = nn.Conv2d(config['num_hidden_units'] // 8, config['num_hidden_units'] // 8, kernel_size=(3, 1),
                                   stride=1, padding=(1, 0))
            self.conv4 = nn.Conv2d(config['num_hidden_units'] // 16, config['num_hidden_units'] // 16,
                                   kernel_size=(3, 1), stride=1, padding=(1, 0))
            self.conv5 = nn.Conv2d(config['num_hidden_units'] // 16, config['n_channel'], kernel_size=(9, 1), stride=1,
                                   padding=0)

        elif config['l_win'] == 48:
            self.conv1 = nn.Conv2d(config['num_hidden_units'], 256 * 3, kernel_size=1, stride=1, padding=0)
            self.conv2 = nn.Conv2d(256, 256, kernel_size=(3, 1), stride=1, padding=(1, 0))
            self.conv3 = nn.Conv2d(128, 128, kernel_size=(3, 1), stride=1, padding=(1, 0))
            self.conv4 = nn.Conv2d(32, 32, kernel_size=(3, 1), stride=1, padding=(1, 0))
            self.conv5 = nn.Conv2d(16, 1, kernel_size=(5, config['n_channel']), stride=1, padding=(2, 0))

        elif config['l_win'] == 144:
            self.conv1 = nn.Conv2d(config['num_hidden_units'], 32 * 27, kernel_size=1, stride=1, padding=0)
            self.conv2 = nn.Conv2d(32 * 9, 32 * 9, kernel_size=(3, 1), stride=1, padding=(1, 0))
            self.conv3 = nn.Conv2d(32 * 3, 32 * 3, kernel_size=(3, 1), stride=1, padding=(1, 0))
            self.conv4 = nn.Conv2d(24, 24, kernel_size=(3, 1), stride=1, padding=(1, 0))
            self.conv5 = nn.Conv2d(6, 1, kernel_size=(9, config['n_channel']), stride=1, padding=(4, 0))

    def depth_to_space(self, x, block_size):
        """PyTorch implementation of TensorFlow's depth_to_space operation"""
        batch, depth, height, width = x.size()
        new_depth = depth // (block_size ** 2)
        new_height = height * block_size
        new_width = width * block_size

        x = x.view(batch, block_size, block_size, new_depth, height, width)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
        x = x.view(batch, new_depth, new_height, new_width)

        return x

    def forward(self, x):
        x = F.leaky_relu(self.fc(x))
        x = x.view(-1, x.size(1), 1, 1)

        if self.config['l_win'] == 24:
            x = F.leaky_relu(self.conv1(x))
            x = x.view(-1, self.config['num_hidden_units'] // 4, 4, 1)

            x = F.leaky_relu(self.conv2(x))
            # depth_to_space with block_size=2
            x = self.depth_to_space(x, 2)
            x = x.view(-1, self.config['num_hidden_units'] // 8, 8, 1)

            x = F.leaky_relu(self.conv3(x))
            # depth_to_space with block_size=2
            x = self.depth_to_space(x, 2)
            x = x.view(-1, self.config['num_hidden_units'] // 16, 16, 1)

            x = F.leaky_relu(self.conv4(x))
            # depth_to_space with block_size=2
            x = self.depth_to_space(x, 2)
            x = x.view(-1, self.config['num_hidden_units'] // 16, 16, 1)

            x = self.conv5(x)
            output = x.view(-1, self.config['l_win'], self.config['n_channel'])

        elif self.config['l_win'] == 48:
            x = F.leaky_relu(self.conv1(x))
            x = x.view(-1, 256, 3, 1)

            x = F.leaky_relu(self.conv2(x))
            # depth_to_space with block_size=2
            x = self.depth_to_space(x, 2)
            x = x.view(-1, 128, 6, 1)

            x = F.leaky_relu(self.conv3(x))
            # depth_to_space with block_size=2
            x = self.depth_to_space(x, 2)
            x = x.view(-1, 32, 24, 1)

            x = F.leaky_relu(self.conv4(x))
            # depth_to_space with block_size=2
            x = self.depth_to_space(x, 2)
            x = x.view(-1, 16, 48, 1)

            x = self.conv5(x)
            output = x.view(-1, self.config['l_win'], self.config['n_channel'])

        elif self.config['l_win'] == 144:
            x = F.leaky_relu(self.conv1(x))
            x = x.view(-1, 32 * 9, 3, 1)

            x = F.leaky_relu(self.conv2(x))
            # depth_to_space with block_size=3
            x = self.depth_to_space(x, 3)
            x = x.view(-1, 32 * 3, 9, 1)

            x = F.leaky_relu(self.conv3(x))
            # depth_to_space with block_size=2
            x = self.depth_to_space(x, 2)
            x = x.view(-1, 24, 36, 1)

            x = F.leaky_relu(self.conv4(x))
            # depth_to_space with block_size=2
            x = self.depth_to_space(x, 2)
            x = x.view(-1, 6, 144, 1)

            x = self.conv5(x)
            output = x.view(-1, self.config['l_win'], self.config['n_channel'])

        return output


class VAEmodel(BaseModel):
    def __init__(self, config, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(VAEmodel, self).__init__(config)

        self.device = device
        self.input_dims = self.config['l_win'] * self.config['n_channel']

        # Initialize encoder and decoder
        self.encoder = Encoder(config).to(device)
        self.decoder = Decoder(config).to(device)

        # Initialize sigma parameter
        if self.config['TRAIN_sigma'] == 1:
            self.log_sigma = nn.Parameter(torch.tensor(np.log(self.config['sigma']), dtype=torch.float32))
        else:
            self.register_buffer('sigma', torch.tensor(self.config['sigma'], dtype=torch.float32))

        self.sigma2_offset = self.config['sigma2_offset']

        self.to(device)

    def forward(self, x, is_code_input=False, code_input=None):
        if is_code_input:
            z = code_input
            x_recon = self.decoder(z)
            return x_recon, None, None, None
        else:
            # Encode
            mean, std_dev = self.encoder(x)

            # Sample
            eps = torch.randn_like(std_dev)
            z = mean + eps * std_dev

            # Decode
            x_recon = self.decoder(z)

            return x_recon, mean, std_dev, z

    def get_sigma2(self):
        if self.config['TRAIN_sigma'] == 1:
            sigma = torch.exp(self.log_sigma)
            sigma2 = sigma ** 2 + self.sigma2_offset
        else:
            sigma2 = self.sigma ** 2
        return sigma2

    def compute_loss(self, x, x_recon, mean, std_dev):
        # Reconstruction loss (weighted by 1/sigma^2)
        sigma2 = self.get_sigma2()
        recon_loss = torch.sum((x - x_recon) ** 2, dim=[1, 2])
        weighted_recon_loss = torch.mean(recon_loss / (2 * sigma2))

        # KL divergence
        kl_loss = 0.5 * torch.sum(mean ** 2 + std_dev ** 2 - torch.log(std_dev ** 2) - 1, dim=1)
        kl_loss = torch.mean(kl_loss)

        # Sigma regularizer
        sigma_regularizer = self.input_dims / 2 * torch.log(sigma2)

        # ELBO loss
        two_pi = self.input_dims / 2 * torch.tensor(2 * np.pi, device=self.device)
        elbo_loss = two_pi + sigma_regularizer + weighted_recon_loss + kl_loss

        return {
            'elbo_loss': elbo_loss,
            'weighted_recon_loss': weighted_recon_loss,
            'recon_loss': torch.mean(recon_loss),
            'kl_loss': kl_loss,
            'sigma_regularizer': sigma_regularizer
        }

    def sample_from_prior(self, n_samples):
        z = torch.randn(n_samples, self.config['code_size'], device=self.device)
        samples = self.decoder(z)
        return samples

    def encode(self, x):
        mean, std_dev = self.encoder(x)
        return mean


class LSTMModel(nn.Module):
    def __init__(self, config):
        super(LSTMModel, self).__init__()
        self.config = config

        self.lstm1 = nn.LSTM(
            input_size=config['code_size'],
            hidden_size=config['num_hidden_units_lstm'],
            batch_first=True
        )

        self.lstm2 = nn.LSTM(
            input_size=config['num_hidden_units_lstm'],
            hidden_size=config['num_hidden_units_lstm'],
            batch_first=True
        )

        self.lstm3 = nn.LSTM(
            input_size=config['num_hidden_units_lstm'],
            hidden_size=config['code_size'],
            batch_first=True
        )

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        return x


# Helper for compatibility with original code
class lstmKerasModel:
    def __init__(self, data, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.embedding_lstm_train = None
        self.embedding_lstm_test = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def create_lstm_model(self, config):
        model = LSTMModel(config).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate_lstm'])
        return model, optimizer

    def produce_embeddings(self, config, model_vae, data, device='cuda' if torch.cuda.is_available() else 'cpu'):
        model_vae.eval()

        self.embedding_lstm_train = np.zeros((data.n_train_lstm, config['l_seq'], config['code_size']))
        for i in range(data.n_train_lstm):
            with torch.no_grad():
                input_tensor = torch.tensor(data.train_set_lstm['data'][i], dtype=torch.float32).to(device)
                self.embedding_lstm_train[i] = model_vae.encode(input_tensor).cpu().numpy()

        print("Finish processing the embeddings of the entire dataset.")
        print(f"The first a few embeddings are\n{self.embedding_lstm_train[0, 0:5]}")

        self.x_train = torch.tensor(self.embedding_lstm_train[:, :config['l_seq'] - 1], dtype=torch.float32)
        self.y_train = torch.tensor(self.embedding_lstm_train[:, 1:], dtype=torch.float32)

        self.embedding_lstm_test = np.zeros((data.n_val_lstm, config['l_seq'], config['code_size']))
        for i in range(data.n_val_lstm):
            with torch.no_grad():
                input_tensor = torch.tensor(data.val_set_lstm['data'][i], dtype=torch.float32).to(device)
                self.embedding_lstm_test[i] = model_vae.encode(input_tensor).cpu().numpy()

        self.x_test = torch.tensor(self.embedding_lstm_test[:, :config['l_seq'] - 1], dtype=torch.float32)
        self.y_test = torch.tensor(self.embedding_lstm_test[:, 1:], dtype=torch.float32)

    def load_model(self, lstm_model, config, checkpoint_path):
        if os.path.isfile(checkpoint_path):
            lstm_model.load_state_dict(torch.load(checkpoint_path))
            print("LSTM model loaded.")
            return True
        else:
            print("No LSTM model loaded.")
            return False

    def train(self, config, lstm_model, optimizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
        lstm_model.train()

        # Create DataLoader objects
        train_dataset = torch.utils.data.TensorDataset(self.x_train.to(device), self.y_train.to(device))
        val_dataset = torch.utils.data.TensorDataset(self.x_test.to(device), self.y_test.to(device))

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config['batch_size_lstm'],
            shuffle=True
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config['batch_size_lstm'],
            shuffle=False
        )

        best_val_loss = float('inf')

        for epoch in range(config['num_epochs_lstm']):
            # Training
            lstm_model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = lstm_model(batch_x)
                loss = F.mse_loss(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            lstm_model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    outputs = lstm_model(batch_x)
                    loss = F.mse_loss(outputs, batch_y)
                    val_loss += loss.item()

            print(f"Epoch {epoch + 1}/{config['num_epochs_lstm']} - "
                  f"Train loss: {train_loss / len(train_loader):.4f}, "
                  f"Val loss: {val_loss / len(val_loader):.4f}")

            # Save checkpoint if validation loss improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(lstm_model.state_dict(), os.path.join(config['checkpoint_dir_lstm'], "cp.pth"))
                print(f"Model saved at epoch {epoch + 1}")

    def plot_reconstructed_lt_seq(self, idx_test, config, model_vae, data, lstm_embedding_test,
                                  device='cuda' if torch.cuda.is_available() else 'cpu'):
        model_vae.eval()

        with torch.no_grad():
            # Get VAE reconstruction
            vae_embedding = torch.tensor(self.embedding_lstm_test[idx_test], dtype=torch.float32).to(device)
            decoded_seq_vae, _, _, _ = model_vae(None, True, vae_embedding)
            decoded_seq_vae = decoded_seq_vae.cpu().numpy()

            # Get LSTM prediction reconstruction
            lstm_embedding_tensor = torch.tensor(lstm_embedding_test[idx_test], dtype=torch.float32).to(device)
            decoded_seq_lstm, _, _, _ = model_vae(None, True, lstm_embedding_tensor)
            decoded_seq_lstm = decoded_seq_lstm.cpu().numpy()

        # Plot results
        fig, axs = plt.subplots(config['n_channel'], 2, figsize=(15, 4.5 * config['n_channel']), edgecolor='k')
        fig.subplots_adjust(hspace=.4, wspace=.4)
        axs = axs.ravel()

        for j in range(config['n_channel']):
            for i in range(2):
                axs[i + j * 2].plot(np.arange(0, config['l_seq'] * config['l_win']),
                                    np.reshape(data.val_set_lstm['data'][idx_test, :, :, j],
                                               (config['l_seq'] * config['l_win'])))
                axs[i + j * 2].grid(True)
                axs[i + j * 2].set_xlim(0, config['l_seq'] * config['l_win'])
                axs[i + j * 2].set_xlabel('samples')

            if config['n_channel'] == 1:
                axs[0 + j * 2].plot(np.arange(0, config['l_seq'] * config['l_win']),
                                    np.reshape(decoded_seq_vae, (config['l_seq'] * config['l_win'])), 'r--')
                axs[1 + j * 2].plot(np.arange(config['l_win'], config['l_seq'] * config['l_win']),
                                    np.reshape(decoded_seq_lstm, ((config['l_seq'] - 1) * config['l_win'])), 'g--')
            else:
                axs[0 + j * 2].plot(np.arange(0, config['l_seq'] * config['l_win']),
                                    np.reshape(decoded_seq_vae[:, :, j], (config['l_seq'] * config['l_win'])), 'r--')
                axs[1 + j * 2].plot(np.arange(config['l_win'], config['l_seq'] * config['l_win']),
                                    np.reshape(decoded_seq_lstm[:, :, j], ((config['l_seq'] - 1) * config['l_win'])),
                                    'g--')

            axs[0 + j * 2].set_title('VAE reconstruction - channel {}'.format(j))
            axs[1 + j * 2].set_title('LSTM reconstruction - channel {}'.format(j))

            for i in range(2):
                axs[i + j * 2].legend(('ground truth', 'reconstruction'))

                savefig(config['result_dir'] + "lstm_long_seq_recons_{}.pdf".format(idx_test))
                fig.clf()
                plt.close()

            def plot_lstm_embedding_prediction(self, idx_test, config, model_vae, data, lstm_embedding_test,
                                               device='cuda' if torch.cuda.is_available() else 'cpu'):
                # 先绘制重建序列
                self.plot_reconstructed_lt_seq(idx_test, config, model_vae, data, lstm_embedding_test, device)

                # 然后绘制嵌入
                fig, axs = plt.subplots(2, config['code_size'] // 2, figsize=(15, 5.5), edgecolor='k')
                fig.subplots_adjust(hspace=.4, wspace=.4)
                axs = axs.ravel()

                for i in range(config['code_size']):
                    axs[i].plot(np.arange(1, config['l_seq']), np.squeeze(self.embedding_lstm_test[idx_test, 1:, i]))
                    axs[i].plot(np.arange(1, config['l_seq']), np.squeeze(lstm_embedding_test[idx_test, :, i]))
                    axs[i].set_xlim(1, config['l_seq'] - 1)
                    axs[i].set_ylim(-2.5, 2.5)
                    axs[i].grid(True)
                    axs[i].set_title(f'Embedding dim {i}')
                    axs[i].set_xlabel('windows')

                    if i == config['code_size'] - 1:
                        axs[i].legend(('VAE\nembedding', 'LSTM\nembedding'))

                savefig(config['result_dir'] + f"lstm_seq_embedding_{idx_test}.pdf")
                fig.clf()
                plt.close()