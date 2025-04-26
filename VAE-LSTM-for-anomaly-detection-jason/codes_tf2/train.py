import os
import tensorflow as tf
from data_loader import DataGenerator
from models import VAEmodel, LSTMKerasModelWrapper
from trainers import VAETrainer
from utils import process_config, create_dirs, get_args, save_config

# Set GPU memory growth to avoid OOM errors
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def main():
    # Capture the config path from the run arguments
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    # Create the experiments directories
    create_dirs([config['result_dir'], config['checkpoint_dir'], config['checkpoint_dir_lstm']])

    # Save the config in a txt file
    save_config(config)

    # Create data generator
    data = DataGenerator(config)

    # Create VAE model
    model_vae = VAEmodel(config)

    # Build model by calling it once with dummy data
    dummy_input = tf.zeros((1, config['l_win'], config['n_channel']))
    _ = model_vae(dummy_input)

    # Load model if checkpoint exists
    checkpoint_path = tf.train.latest_checkpoint(config['checkpoint_dir'])
    if checkpoint_path:
        model_vae.load(checkpoint_path)

    # Train VAE if needed
    if config['TRAIN_VAE'] and config['num_epochs_vae'] > 0:
        trainer_vae = VAETrainer(model_vae, data, config)
        trainer_vae.train()

    # Train LSTM if needed
    if config['TRAIN_LSTM']:
        # Create LSTM model wrapper instance
        lstm_model_wrapper = LSTMKerasModelWrapper(data)

        # Produce embeddings for LSTM training
        lstm_model_wrapper.produce_embeddings(config, model_vae, data)

        # Create LSTM model
        lstm_nn_model = lstm_model_wrapper.create_lstm_model(config)
        lstm_nn_model.summary()

        # Load weights if available
        checkpoint_path = config['checkpoint_dir_lstm'] + "cp.ckpt"
        if os.path.exists(checkpoint_path + '.index'):
            lstm_nn_model.load_weights(checkpoint_path)
            print("LSTM model loaded.")

        # Train LSTM if needed
        if config['num_epochs_lstm'] > 0:
            lstm_model_wrapper.train(config, lstm_nn_model)

        # Make predictions on test set
        lstm_embedding = lstm_nn_model.predict(
            lstm_model_wrapper.x_test,
            batch_size=config['batch_size_lstm']
        )
        print(f"LSTM embedding shape: {lstm_embedding.shape}")

        # Visualize predictions
        for i in range(10):
            lstm_model_wrapper.plot_lstm_embedding_prediction(
                i, config, model_vae, data, lstm_embedding
            )


if __name__ == '__main__':
    main()
