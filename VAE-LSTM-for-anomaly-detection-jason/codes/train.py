import os
import tensorflow as tf
from data_loader import DataGenerator
from models import VAEmodel, lstmKerasModel
from trainers import vaeTrainer
from utils import process_config, create_dirs, get_args, save_config
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs 创建实验目录
    create_dirs([config['result_dir'], config['checkpoint_dir'], config['checkpoint_dir_lstm']])
    # save the config in a txt file 保存配置到一个txt文件中
    save_config(config)
    # create tensorflow session  创建tensorflow会话
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    # create your data generator  创建数据生成器
    data = DataGenerator(config)
    # create a CNN model  实例  创建一个VAE模型实例
    model_vae = VAEmodel(config)
    # create a trainer for VAE model  创建一个VAE模型的训练器
    trainer_vae = vaeTrainer(sess, model_vae, data, config)
    model_vae.load(sess)  # load the model if possible  加载模型
    # here you train your model
    if config['TRAIN_VAE']:  # 训练VAE模型
        if config['num_epochs_vae'] > 0:
            trainer_vae.train()  # 训练模型

    if config['TRAIN_LSTM']:
        # create a lstm model class instance  创建一个LSTM模型实例
        lstm_model = lstmKerasModel(data)

        # produce the embedding of all sequences for training of lstm model  为LSTM模型训练生成所有序列的embedding
        # process the windows in sequence to get their VAE embeddings  依次处理窗口，以获得其 VAE embedding
        lstm_model.produce_embeddings(config, model_vae, data, sess)

        # Create a basic model instance  创建一个基本LSTM模型实例
        lstm_nn_model = lstm_model.create_lstm_model(config)
        lstm_nn_model.summary()   # Display the model's architecture  显示LSTM模型的架构
        # checkpoint path  创建检查点路径
        checkpoint_path = config['checkpoint_dir_lstm']\
                          + "cp.ckpt"
        # Create a callback that saves the model's weights  创建一个回调函数，用于保存模型的权重
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1)
        # load weights if possible  加载权重
        lstm_model.load_model(lstm_nn_model, config, checkpoint_path)

        # start training  训练
        if config['num_epochs_lstm'] > 0:
            lstm_model.train(config, lstm_nn_model, cp_callback)

        # make a prediction on the test set using the trained model  使用训练好的模型对测试集进行预测
        lstm_embedding = lstm_nn_model.predict(lstm_model.x_test, batch_size=config['batch_size_lstm'])
        print(lstm_embedding.shape)

        # visualise the first 10 test sequences  可视化前10个测试序列
        for i in range(10):
            lstm_model.plot_lstm_embedding_prediction(i, config, model_vae, sess, data, lstm_embedding)


if __name__ == '__main__':
    main()
