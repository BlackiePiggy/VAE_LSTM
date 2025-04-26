import os
import torch
from data_loader import DataGenerator
from models import VAEmodel
from lstm_model import LSTMPredictor, LSTMHandler
from trainers import vaeTrainer
from utils import process_config, create_dirs, get_args, save_config, count_trainable_parameters

# 设置CUDA设备（如果可用）
if torch.cuda.is_available():
    device = torch.device('cuda')
    # 可选：设置特定的GPU设备
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    device = torch.device('cpu')

print(f"使用设备: {device}")


def main():
    # 获取配置路径，并处理json配置文件
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("参数缺失或无效")
        exit(0)

    # 创建实验目录
    create_dirs([config['result_dir'], config['checkpoint_dir'], config['checkpoint_dir_lstm']])

    # 保存配置到txt文件
    save_config(config)

    # 创建数据生成器
    data = DataGenerator(config)

    # 创建VAE模型
    model_vae = VAEmodel(config, device=device)
    print(f"VAE模型创建完成，共有 {count_trainable_parameters(model_vae)} 个可训练参数")

    # 创建VAE模型的训练器
    trainer_vae = vaeTrainer(model_vae, data, config, device=device)

    # 尝试加载模型
    model_vae.load()

    # 如果配置为训练VAE
    if config['TRAIN_VAE']:
        if config['num_epochs_vae'] > 0:
            trainer_vae.train()

    # 如果配置为训练LSTM
    if config['TRAIN_LSTM']:
        # 创建LSTM模型类实例
        lstm_handler = LSTMHandler(data, device=device)

        # 为LSTM模型训练准备所有序列的嵌入
        lstm_handler.prepare_embeddings(config, model_vae, data, device)

        # 创建基础LSTM模型实例
        lstm_model, optimizer = lstm_handler.create_lstm_model(config)
        print(lstm_model)

        # 检查点路径
        checkpoint_path = os.path.join(config['checkpoint_dir_lstm'], "best_model.pth")

        # 如果可能，加载权重
        lstm_handler.load_model(lstm_model, checkpoint_path)

        # 开始训练
        if config['num_epochs_lstm'] > 0:
            lstm_handler.train(config, lstm_model, optimizer, device)

        # 使用训练好的模型对测试集进行预测
        lstm_model.eval()
        with torch.no_grad():
            lstm_predictions = lstm_model(lstm_handler.x_test.to(device))
            lstm_predictions = lstm_predictions.cpu().numpy()

        print(f"LSTM预测结果形状: {lstm_predictions.shape}")

        # 可视化前10个测试序列
        for i in range(min(10, lstm_predictions.shape[0])):
            lstm_handler.plot_embeddings(i, config, model_vae, data, lstm_predictions, device)


if __name__ == '__main__':
    main()