from ultralytics import YOLO
import os
import time
from pathlib import Path
import torch


def detect_cracks():
    # 检查GPU是否可用
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {'GPU' if device == '0' else 'CPU'}")

    model_path = '../../models/crack_detector_optimized/weights/best.pt'

    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在 - {model_path}")
        return

    model = YOLO(model_path)
    print(f"已加载模型: {model_path}")

    # 配置参数
    test_dir = '../../test_images'
    output_dir = '../../detection_results'
    os.makedirs(output_dir, exist_ok=True)

    # 关键参数调整（解决漏检）
    conf_threshold = 0.15  # 降低置信度阈值（更敏感）
    iou_threshold = 0.25  # 降低IOU阈值（适应重叠裂缝）
    imgsz = 640  # 与训练一致

    start_time = time.time()
    image_count = 0

    for img_path in Path(test_dir).glob('*.*'):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_count += 1
            results = model(
                str(img_path),
                conf=conf_threshold,
                iou=iou_threshold,
                imgsz=imgsz,
                show=False,
                device=device  # 使用配置的设备
            )

            save_path = os.path.join(output_dir, img_path.name)
            results[0].save(save_path)

            orig_h, orig_w = results[0].orig_shape
            crack_count = len(results[0].boxes)
            print(f"处理完成: {img_path.name} - 检测到{crack_count}个裂缝")

    end_time = time.time()
    print(f"\n===== 检测完成 =====")
    print(f"共处理{image_count}张图片，耗时{end_time - start_time:.2f}秒")
    print(f"结果保存: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    detect_cracks()