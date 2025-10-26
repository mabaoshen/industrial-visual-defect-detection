import os
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.applications.mobilenet_v2 import preprocess_input

# 配置参数（与项目结构保持一致）
THRESHOLD = 0.35  # 一阶段检测阈值
RESCUE_THRESHOLD = 0.2  # 二阶段救援阈值
MIN_AREA = 80  # 一阶段最小缺陷面积
RESCUE_MIN_AREA = 30  # 二阶段最小缺陷面积
TARGET_SIZE = (576, 576)
MODEL_PATH = "D:/pythonCode/1/best_model.h5"
INPUT_DIR = "D:/pythonCode/1/images"  # 测试图片目录
OUTPUT_DIR = "D:/pythonCode/1/segmented_results"
OVERLAY_COLOR = (0, 255, 0)  # 绿色 (BGR格式)
OVERLAY_ALPHA = 0.5  # 叠加透明度
BORDER_COLOR = (0, 0, 255)  # 边界颜色
BORDER_THICKNESS = 1  # 边界厚度


def create_output_dir():
    """创建输出目录"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    print(f"✅ 输出目录已准备：{OUTPUT_DIR}")


def load_segmentation_model():
    """加载分割模型"""
    try:
        model = load_model(MODEL_PATH, compile=False)
        print("✅ 模型加载成功！")
        return model
    except Exception as e:
        print(f"❌ 模型加载失败：{str(e)}")
        return None


def morph_ops(mask, op_type="open", kernel_size=3):
    """形态学操作：去噪或填补缺陷"""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if op_type == "open":
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # 去噪
    elif op_type == "close":
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # 填补
    return mask


def filter_small_regions(mask, min_area):
    """过滤小面积区域"""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] < min_area:
            mask[labels == label] = 0
    return mask


def tta_inference(model, img):
    """测试时数据增强：水平翻转提高稳健性"""
    # 原图预测
    pred1 = model.predict(img, verbose=0)[0]
    # 水平翻转预测
    img_flipped = tf.image.flip_left_right(img)
    pred2 = model.predict(img_flipped, verbose=0)[0]
    pred2 = tf.image.flip_left_right(pred2).numpy()  # 翻转回来
    return (pred1 + pred2) / 2  # 平均结果


def detect_defects(model, img_path, threshold, min_area, morph_op="open"):
    """检测缺陷的核心函数"""
    # 读取原图
    original_img = cv2.imread(img_path)
    if original_img is None:
        return None, None

    h, w = original_img.shape[:2]

    # 预处理
    img_resized = cv2.resize(original_img, TARGET_SIZE)
    img_preprocessed = preprocess_input(img_resized)
    img_input = np.expand_dims(img_preprocessed, axis=0)

    # 预测（带TTA）
    pred_mask = tta_inference(model, img_input)

    # 恢复掩码到原图尺寸
    pred_mask = cv2.resize(pred_mask, (w, h))

    # 后处理
    binary_mask = (pred_mask > threshold).astype(np.uint8) * 255
    binary_mask = morph_ops(binary_mask, morph_op)
    binary_mask = filter_small_regions(binary_mask, min_area)

    # 判断是否有缺陷
    has_defect = np.sum(binary_mask) > 0

    return original_img, binary_mask, has_defect


def draw_overlay(original_img, mask):
    """在原图上绘制缺陷叠加效果"""
    # 创建彩色掩码
    colored_mask = np.zeros_like(original_img)
    colored_mask[mask == 255] = OVERLAY_COLOR

    # 叠加半透明效果
    overlay_img = cv2.addWeighted(
        colored_mask, OVERLAY_ALPHA,
        original_img, 1 - OVERLAY_ALPHA,
        0
    )

    # 绘制缺陷边界
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay_img, contours, -1, BORDER_COLOR, BORDER_THICKNESS)

    return overlay_img


def segment_images(model):
    """批量处理图片"""
    image_files = [f for f in os.listdir(INPUT_DIR)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not image_files:
        print(f"❌ 未在{INPUT_DIR}找到图片文件")
        return

    print(f"开始分割{len(image_files)}张图片（检测defects）...")

    for i, img_file in enumerate(image_files, 1):
        img_path = os.path.join(INPUT_DIR, img_file)
        print(f"\n处理 {img_file} ({i}/{len(image_files)})")

        # 一阶段检测
        original_img, mask, has_defect = detect_defects(
            model, img_path, THRESHOLD, MIN_AREA, "open"
        )

        if original_img is None:
            print(f"❌ 无法处理图片：{img_file}")
            continue

        # 二阶段救援（如果一阶段未检测到缺陷）
        if not has_defect:
            print(f"⚠️ 一阶段未检测到缺陷，启动二阶段救援...")
            _, rescue_mask, rescue_has_defect = detect_defects(
                model, img_path, RESCUE_THRESHOLD, RESCUE_MIN_AREA, "close"
            )
            if rescue_has_defect:
                has_defect = True
                mask = rescue_mask

        # 绘制并保存结果
        overlay_img = draw_overlay(original_img, mask)
        output_path = os.path.join(OUTPUT_DIR, f"seg_defects_{img_file}")

        if cv2.imwrite(output_path, overlay_img):
            status = "检测到缺陷" if has_defect else "未检测到缺陷"
            print(f"✅ 已保存（{i}/{len(image_files)}）：{output_path} - {status}")
        else:
            print(f"❌ 保存失败：{output_path}")


def main():
    create_output_dir()
    model = load_segmentation_model()
    if model:
        segment_images(model)
    print("\n分割完成！")


if __name__ == "__main__":
    main()