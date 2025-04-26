import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMPredictor(nn.Module):
    def __init__(self, config):
        super(LSTMPredictor, self).__init__()
        self.config = config

        # LSTM层
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


class LSTMHandler:
    def __init__(self, data, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.embedding_train = None
        self.embedding_test = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def create_lstm_model(self, config):
        model = LSTMPredictor(config).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate_lstm'])
        return model, optimizer

    def prepare_embeddings(self, config, vae_model, data, device='cuda' if torch.cuda.is_available() else 'cpu'):
        vae_model.eval()

        self.embedding_train = np.zeros((data.n_train_lstm, config['l_seq'], config['code_size']))
        for i in range(data.n_train_lstm):
            with torch.no_grad():
                input_tensor = torch.tensor(data.train_set_lstm['data'][i], dtype=torch.float32).to(device)
                self.embedding_train[i] = vae_model.encode(input_tensor).cpu().numpy()

        print("Finished processing embeddings for the entire dataset.")
        print(f"First few embeddings:\n{self.embedding_train[0, 0:5]}")

        self.x_train = torch.tensor(self.embedding_train[:, :config['l_seq'] - 1], dtype=torch.float32)
        self.y_train = torch.tensor(self.embedding_train[:, 1:], dtype=torch.float32)

        self.embedding_test = np.zeros((data.n_val_lstm, config['l_seq'], config['code_size']))
        for i in range(data.n_val_lstm):
            with torch.no_grad():
                input_tensor = torch.tensor(data.val_set_lstm['data'][i], dtype=torch.float32).to(device)
                self.embedding_test[i] = vae_model.encode(input_tensor).cpu().numpy()

        self.x_test = torch.tensor(self.embedding_test[:, :config['l_seq'] - 1], dtype=torch.float32)
        self.y_test = torch.tensor(self.embedding_test[:, 1:], dtype=torch.float32)

    def load_model(self, lstm_model, checkpoint_path):
        if os.path.isfile(checkpoint_path):
            lstm_model.load_state_dict(torch.load(checkpoint_path))
            print("LSTM model loaded successfully.")
            return True
        else:
            print("No LSTM model found to load.")
            return False

    def train(self, config, lstm_model, optimizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
        lstm_model.train()

        # 创建DataLoader对象
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
            # 训练
            lstm_model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = lstm_model(batch_x)
                loss = F.mse_loss(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # 验证
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

            # 如果验证损失改善，保存模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(lstm_model.state_dict(), os.path.join(config['checkpoint_dir_lstm'], "best_model.pth"))
                print(f"Model saved at epoch {epoch + 1}")

    def plot_reconstructed_sequences(self, idx_test, config, vae_model, data, lstm_predictions,
                                     device='cuda' if torch.cuda.is_available() else 'cpu'):
        vae_model.eval()

        with torch.no_grad():
            # 获取VAE重建
            vae_embedding = torch.tensor(self.embedding_test[idx_test], dtype=torch.float32).to(device)
            decoded_seq_vae, _, _, _ = vae_model(None, True, vae_embedding)
            decoded_seq_vae = decoded_seq_vae.cpu().numpy()

            # 获取LSTM预测的重建
            lstm_embedding_tensor = torch.tensor(lstm_predictions[idx_test], dtype=torch.float32).to(device)
            decoded_seq_lstm, _, _, _ = vae_model(None, True, lstm_embedding_tensor)
            decoded_seq_lstm = decoded_seq_lstm.cpu().numpy()

        # 绘制结果
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

            axs[0 + j * 2].set_title(f'VAE reconstruction - channel {j}')
            axs[1 + j * 2].set_title(f'LSTM reconstruction - channel {j}')

            for i in range(2):
                axs[i + j * 2].legend(('ground truth', 'reconstruction'))

        savefig(config['result_dir'] + f"lstm_long_seq_recons_{idx_test}.pdf")
        fig.clf()
        plt.close()

    def plot_embeddings(self, idx_test, config, vae_model, data, lstm_predictions,
                        device='cuda' if torch.cuda.is_available() else 'cpu'):
        # 首先绘制重建序列
        self.plot_reconstructed_sequences(idx_test, config, vae_model, data, lstm_predictions, device)

        # 然后绘制嵌入
        fig, axs = plt.subplots(2, config['code_size'] // 2, figsize=(15, 5.5), edgecolor='k')
        fig.subplots_adjust(hspace=.4, wspace=.4)
        axs = axs.ravel()

        for i in range(config['code_size']):
            axs[i].plot(np.arange(1, config['l_seq']), np.squeeze(self.embedding_test[idx_test, 1:, i]))
            axs[i].plot(np.arange(1, config['l_seq']), np.squeeze(lstm_predictions[idx_test, :, i]))
            axs[i].set_xlim(1, config['l_seq'] - 1)
            axs[i].set_ylim(-2.5, 2.5)
            axs[i].grid(True)
            axs[i].set_title(f'Embedding dim {i}')
            axs[i].set_xlabel('windows')

            if i == config['code_size'] - 1:
                axs[i].legend(('VAE embedding', 'LSTM prediction'))

        savefig(config['result_dir'] + f"lstm_seq_embedding_{idx_test}.pdf")
        fig.clf()
        plt.close()