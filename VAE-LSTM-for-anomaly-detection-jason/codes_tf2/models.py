import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pyplot import savefig
from base import BaseModel

tfd = tfp.distributions


class VAEmodel(BaseModel):
    def __init__(self, config):
        super(VAEmodel, self).__init__(config)
        self.input_dims = config['l_win'] * config['n_channel']
        self.batch_size = config['batch_size']

        # Define model architecture
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

        # Define sigma2 parameter
        if config['TRAIN_sigma'] == 1:
            self.sigma = tf.Variable(
                tf.cast(config['sigma'], tf.float32),
                trainable=True,
                name='sigma'
            )
        else:
            self.sigma = tf.constant(config['sigma'], dtype=tf.float32)

        self.sigma2_offset = tf.constant(config['sigma2_offset'])

    @property
    def sigma2(self):
        sigma2_value = tf.square(self.sigma)
        if self.config['TRAIN_sigma'] == 1:
            sigma2_value += self.sigma2_offset
        return sigma2_value

    def build_encoder(self):
        init = tf.keras.initializers.GlorotUniform()

        inputs = tf.keras.Input(shape=(self.config['l_win'], self.config['n_channel']))
        x = tf.expand_dims(inputs, -1)

        if self.config['l_win'] == 24:
            # Use padding layer for symmetrical padding
            x = tf.keras.layers.ZeroPadding2D(padding=((4, 4), (0, 0)))(x)
            x = tf.keras.layers.Conv2D(
                filters=self.config['num_hidden_units'] // 16,
                kernel_size=(3, self.config['n_channel']),
                strides=(2, 1),
                padding='same',
                activation='relu',
                kernel_initializer=init
            )(x)

            x = tf.keras.layers.Conv2D(
                filters=self.config['num_hidden_units'] // 8,
                kernel_size=(3, self.config['n_channel']),
                strides=(2, 1),
                padding='same',
                activation='relu',
                kernel_initializer=init
            )(x)

            x = tf.keras.layers.Conv2D(
                filters=self.config['num_hidden_units'] // 4,
                kernel_size=(3, self.config['n_channel']),
                strides=(2, 1),
                padding='same',
                activation='relu',
                kernel_initializer=init
            )(x)

            x = tf.keras.layers.Conv2D(
                filters=self.config['num_hidden_units'],
                kernel_size=(4, self.config['n_channel']),
                strides=1,
                padding='valid',
                activation='relu',
                kernel_initializer=init
            )(x)

        elif self.config['l_win'] == 48:
            # Similar conv layers for l_win=48
            x = tf.keras.layers.Conv2D(
                filters=self.config['num_hidden_units'] // 16,
                kernel_size=(3, self.config['n_channel']),
                strides=(2, 1),
                padding='same',
                activation='relu',
                kernel_initializer=init
            )(x)
            # Continue with other conv layers...

        elif self.config['l_win'] == 144:
            # Similar conv layers for l_win=144
            x = tf.keras.layers.Conv2D(
                filters=self.config['num_hidden_units'] // 16,
                kernel_size=(3, self.config['n_channel']),
                strides=(4, 1),
                padding='same',
                activation='relu',
                kernel_initializer=init
            )(x)
            # Continue with other conv layers...

        # Flatten and dense layers
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(
            units=self.config['code_size'] * 4,
            activation='relu',
            kernel_initializer=init
        )(x)

        # Output layers for mean and stddev
        code_mean = tf.keras.layers.Dense(
            units=self.config['code_size'],
            activation=None,
            kernel_initializer=init,
            name='code_mean'
        )(x)

        code_std_dev = tf.keras.layers.Dense(
            units=self.config['code_size'],
            activation='relu',
            kernel_initializer=init,
            name='code_std_dev'
        )(x)
        code_std_dev = code_std_dev + 1e-2

        return tf.keras.Model(inputs=inputs, outputs=[code_mean, code_std_dev])

    def build_decoder(self):
        init = tf.keras.initializers.GlorotUniform()

        inputs = tf.keras.Input(shape=(self.config['code_size'],))
        x = tf.keras.layers.Dense(
            units=self.config['num_hidden_units'],
            activation='relu',
            kernel_initializer=init
        )(inputs)

        x = tf.keras.layers.Reshape((1, 1, self.config['num_hidden_units']))(x)

        if self.config['l_win'] == 24:
            # Decoder architecture for l_win=24
            x = self._build_decoder_24(x, init)
        elif self.config['l_win'] == 48:
            # Decoder architecture for l_win=48
            x = self._build_decoder_48(x, init)
        elif self.config['l_win'] == 144:
            # Decoder architecture for l_win=144
            x = self._build_decoder_144(x, init)

        return tf.keras.Model(inputs=inputs, outputs=x)

    def _build_decoder_24(self, x, init):
        # Implementation for l_win=24 decoder
        x = tf.keras.layers.Conv2D(
            filters=self.config['num_hidden_units'],
            kernel_size=1,
            padding='same',
            activation='relu'
        )(x)

        x = tf.keras.layers.Reshape((4, 1, self.config['num_hidden_units'] // 4))(x)

        # Continue with transpose convolutions and depth_to_space operations
        # ... (similar pattern as original but using Keras layers)

        return x

    def _build_decoder_48(self, x, init):
        # Implementation for l_win=48 decoder
        # ... similar pattern to _build_decoder_24
        return x

    def _build_decoder_144(self, x, init):
        # Implementation for l_win=144 decoder
        # ... similar pattern to _build_decoder_24
        return x

    def call(self, inputs, training=False):
        """Forward pass of the VAE"""
        code_mean, code_std_dev = self.encoder(inputs)

        # Sample from the latent distribution
        if training:
            eps = tf.random.normal(shape=tf.shape(code_mean))
            code_sample = code_mean + code_std_dev * eps
        else:
            code_sample = code_mean

        decoded = self.decoder(code_sample)

        return decoded, code_mean, code_std_dev, code_sample

    def encode(self, inputs):
        """Encode inputs to latent space"""
        code_mean, code_std_dev = self.encoder(inputs)
        return code_mean, code_std_dev

    def decode(self, code):
        """Decode from latent space"""
        return self.decoder(code)


class LSTMModel(tf.keras.Model):
    def __init__(self, config):
        super(LSTMModel, self).__init__()
        self.config = config

        # Define LSTM layers
        self.lstm1 = tf.keras.layers.LSTM(
            config['num_hidden_units_lstm'],
            return_sequences=True
        )
        self.lstm2 = tf.keras.layers.LSTM(
            config['num_hidden_units_lstm'],
            return_sequences=True
        )
        self.lstm3 = tf.keras.layers.LSTM(
            config['code_size'],
            return_sequences=True,
            activation=None
        )

    def call(self, inputs):
        x = self.lstm1(inputs)
        x = self.lstm2(x)
        return self.lstm3(x)


class LSTMKerasModelWrapper:
    def __init__(self, data):
        self.data = data

    def create_lstm_model(self, config):
        """Create LSTM model using functional API"""
        lstm_input = tf.keras.layers.Input(shape=(config['l_seq'] - 1, config['code_size']))
        x = tf.keras.layers.LSTM(config['num_hidden_units_lstm'], return_sequences=True)(lstm_input)
        x = tf.keras.layers.LSTM(config['num_hidden_units_lstm'], return_sequences=True)(x)
        lstm_output = tf.keras.layers.LSTM(config['code_size'], return_sequences=True, activation=None)(x)

        lstm_model = tf.keras.Model(lstm_input, lstm_output)
        lstm_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate_lstm']),
            loss='mse',
            metrics=['mse']
        )
        return lstm_model

    def produce_embeddings(self, config, model_vae, data):
        """Produce embeddings for LSTM training"""
        self.embedding_lstm_train = np.zeros((data.n_train_lstm, config['l_seq'], config['code_size']))

        for i in range(data.n_train_lstm):
            embedding = model_vae.encode(data.train_set_lstm['data'][i])[0]  # Get mean only
            self.embedding_lstm_train[i] = embedding

        print("Finish processing the embeddings of the entire dataset.")
        print(f"The first a few embeddings are\n{self.embedding_lstm_train[0, 0:5]}")

        self.x_train = self.embedding_lstm_train[:, :config['l_seq'] - 1]
        self.y_train = self.embedding_lstm_train[:, 1:]

        # Process validation embeddings
        self.embedding_lstm_test = np.zeros((data.n_val_lstm, config['l_seq'], config['code_size']))

        for i in range(data.n_val_lstm):
            embedding = model_vae.encode(data.val_set_lstm['data'][i])[0]
            self.embedding_lstm_test[i] = embedding

        self.x_test = self.embedding_lstm_test[:, :config['l_seq'] - 1]
        self.y_test = self.embedding_lstm_test[:, 1:]

    def train(self, config, lstm_model):
        """Train LSTM model"""
        checkpoint_path = config['checkpoint_dir_lstm'] + "cp.ckpt"
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            verbose=1
        )

        lstm_model.fit(
            self.x_train,
            self.y_train,
            validation_data=(self.x_test, self.y_test),
            batch_size=config['batch_size_lstm'],
            epochs=config['num_epochs_lstm'],
            callbacks=[cp_callback]
        )

    def plot_reconstructed_lt_seq(self, idx_test, config, model_vae, data, lstm_embedding_test):
        """Plot reconstructed long sequences"""
        # Decode VAE sequences
        decoded_seq_vae = model_vae.decode(self.embedding_lstm_test[idx_test])
        decoded_seq_vae = tf.squeeze(decoded_seq_vae).numpy()

        # Decode LSTM sequences
        decoded_seq_lstm = model_vae.decode(lstm_embedding_test[idx_test])
        decoded_seq_lstm = tf.squeeze(decoded_seq_lstm).numpy()

        fig, axs = plt.subplots(config['n_channel'], 2, figsize=(15, 4.5 * config['n_channel']))
        fig.subplots_adjust(hspace=.4, wspace=.4)
        axs = axs.ravel()

        # Plotting logic...
        # (Implementation similar to original but updated for TF 2.x)

        plt.savefig(f"{config['result_dir']}lstm_long_seq_recons_{idx_test}.pdf")
        plt.close()

    def plot_lstm_embedding_prediction(self, idx_test, config, model_vae, data, lstm_embedding_test):
        """Plot LSTM embedding predictions"""
        self.plot_reconstructed_lt_seq(idx_test, config, model_vae, data, lstm_embedding_test)

        fig, axs = plt.subplots(2, config['code_size'] // 2, figsize=(15, 5.5))
        fig.subplots_adjust(hspace=.4, wspace=.4)
        axs = axs.ravel()

        # Plotting logic...
        # (Implementation similar to original)

        plt.savefig(f"{config['result_dir']}lstm_seq_embedding_{idx_test}.pdf")
        plt.close()
