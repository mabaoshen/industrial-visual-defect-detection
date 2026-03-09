from ultralytics import YOLO
import os
import time
import torch
from scripts.utils.check_dataset import check_dataset  # 注意相对导入路径是否正确


def train_crack_model():
    if not check_dataset():
        return

    # 检查GPU是否可用
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {'GPU' if device == '0' else 'CPU'}")

    # 加载本地模型（相对路径）
    model = YOLO('../../models/pretrained/yolov8n.pt')
    print("已加载本地YOLOv8模型，开始处理尺寸不一致的图片...")

    start_time = time.time()
    training_results = model.train(
        data='../../config/crack_dataset.yaml',
        epochs=50,
        imgsz=640,
        batch=4,
        device=device,  # 使用GPU或CPU
        project='../../models',
        name='crack_detector_optimized',
        augment=True,
        # 数据增强参数优化（针对裂缝检测特点）
        mosaic=1.0,          # 马赛克增强（混合4张图）
        mixup=0.2,           # 提高混合增强比例，增强泛化能力
        scale=0.6,           # 扩大缩放范围，适应不同尺寸裂缝
        flipud=0.3,          # 上下翻转概率30%
        fliplr=0.5,          # 左右翻转概率50%
        degrees=10.0,        # 随机旋转±10度
        perspective=0.001,   # 轻微透视变换，增强视角鲁棒性
        # 学习率调整（适合小数据集）
        lr0=0.001,           # 初始学习率（默认0.01的1/10）
        lrf=0.01,            # 最终学习率因子
        # 过拟合抑制
        patience=20,         # 延长早停阈值
        weight_decay=0.0005, # 权重衰减，减少过拟合
        rect=False
    )

    end_time = time.time()
    print(f"\n===== 训练完成 =====")
    print(f"训练耗时: {end_time - start_time:.2f}秒")
    # 兼容不同版本的结果字典键名
    try:
        map50 = training_results.results_dict['metrics/mAP50(B)']
    except KeyError:
        map50 = training_results.results_dict.get('metrics/mAP50', 0)
    print(f"最佳模型mAP50: {map50:.3f}")
    print(f"模型保存路径: {model.ckpt_path}")


if __name__ == "__main__":
    train_crack_model()