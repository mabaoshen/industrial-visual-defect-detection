import json
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
import re

# 配置参数（保持项目结构）
TARGET_LABEL = 'defects'
SECOND_LEVEL_FOLDER = 'defects'
MODEL_INPUT_SIZE = (128, 128)  # 与训练模型输入尺寸一致
VALID_MASK_MIN_AREA = 50  # 最小缺陷面积阈值


def normalize_filename(filename):
    """标准化文件名，确保一致性"""
    normalized = re.sub(r'[^\w\-\.]', '', filename)
    name_part, ext_part = os.path.splitext(normalized)
    return f"{name_part}{ext_part.lower()}"


def check_mask_quality(mask, min_area=VALID_MASK_MIN_AREA):
    """检查掩码质量，过滤无效数据"""
    if np.sum(mask) == 0:
        return False, "全黑掩码"

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, "无有效轮廓"

    max_area = max(cv2.contourArea(contour) for contour in contours)
    if max_area < min_area:
        return False, f"缺陷面积过小（最大{max_area}像素）"

    return True, "质量合格"


def json_to_mask(image_dir, json_dir, mask_dir):
    """将JSON标注转换为掩码图像"""
    os.makedirs(mask_dir, exist_ok=True)
    valid_count = 0
    total_count = 0

    for json_file in os.listdir(json_dir):
        if not json_file.endswith(".json"):
            continue
        total_count += 1

        # 查找对应图片
        img_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        base_name = os.path.splitext(json_file)[0]
        img_name = None

        for ext in img_extensions:
            candidate = f"{base_name}{ext}"
            if os.path.exists(os.path.join(image_dir, candidate)):
                img_name = candidate
                break

        if not img_name:
            print(f"❌ {json_file}未找到对应图片，跳过")
            continue

        # 读取图片
        img_path = os.path.join(image_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            print(f"❌ 无法读取图片{img_path}，跳过")
            continue
        orig_height, orig_width = image.shape[:2]

        # 解析JSON标注
        json_path = os.path.join(json_dir, json_file)
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"❌ {json_file}解析失败 - {str(e)}，跳过")
            continue

        # 检查是否有defects标签
        shapes = data.get('shapes', [])
        has_defects = any(shape.get('label', '').lower() == TARGET_LABEL.lower() for shape in shapes)
        if not has_defects:
            print(f"❌ {json_file}中未找到'{TARGET_LABEL}'标签，跳过")
            continue

        # 绘制掩码
        mask = np.zeros((orig_height, orig_width), dtype=np.uint8)
        mask_updated = False

        for shape in shapes:
            shape_label = shape.get('label', '').lower()
            if shape_label == TARGET_LABEL.lower():
                try:
                    points = np.array(shape['points'], dtype=np.int32)
                    if points.ndim == 2 and points.shape[1] == 2 and len(points) >= 3:
                        points = np.clip(points, 0, [orig_width - 1, orig_height - 1])
                        cv2.fillPoly(mask, [points], 255)
                        mask_updated = True
                    else:
                        print(f"⚠️ {json_file}中点集格式无效")
                except Exception as e:
                    print(f"⚠️ {json_file}中点集处理失败 - {str(e)}")

        if not mask_updated:
            print(f"❌ {json_file}未生成有效标注，跳过")
            continue

        # 掩码质量检查
        mask_qualified, quality_msg = check_mask_quality(mask)
        if not mask_qualified:
            print(f"⚠️ {json_file}掩码质量不合格 - {quality_msg}，跳过")
            continue

        # 保存掩码
        normalized_img_name = normalize_filename(img_name)
        mask_path = os.path.join(mask_dir, normalized_img_name)

        if normalized_img_name != img_name:
            print(f"ℹ️ 文件名标准化 '{img_name}' → '{normalized_img_name}'")

        if cv2.imwrite(mask_path, mask):
            valid_count += 1
            print(f"✅ 生成掩码（{valid_count}/{total_count}）：{mask_path}")
        else:
            print(f"❌ 无法保存掩码{mask_path}")

    print(f"\n掩码生成完成：共处理{total_count}个文件，成功生成{valid_count}个有效掩码")
    return valid_count > 0


def split_dataset(image_dir, mask_dir, train_dir, val_dir):
    """划分训练集和验证集"""
    # 创建目录结构
    dirs_to_create = [
        os.path.join(train_dir, "images", SECOND_LEVEL_FOLDER),
        os.path.join(train_dir, "masks", SECOND_LEVEL_FOLDER),
        os.path.join(val_dir, "images", SECOND_LEVEL_FOLDER),
        os.path.join(val_dir, "masks", SECOND_LEVEL_FOLDER)
    ]
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)

    # 收集有效样本
    mask_files = [f for f in os.listdir(mask_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    valid_images = []
    filename_mismatches = []

    for mask_file in mask_files:
        normalized_mask = normalize_filename(mask_file)
        img_path = os.path.join(image_dir, normalized_mask)

        if not os.path.exists(img_path):
            img_path_original = os.path.join(image_dir, mask_file)
            if os.path.exists(img_path_original):
                print(f"ℹ️ 标准化图片文件名 '{mask_file}' → '{normalized_mask}'")
                shutil.move(img_path_original, os.path.join(image_dir, normalized_mask))
                img_path = os.path.join(image_dir, normalized_mask)
            else:
                filename_mismatches.append(mask_file)
                continue

        # 再次校验掩码质量
        mask_path = os.path.join(mask_dir, mask_file)
        mask = cv2.imread(mask_path, 0)
        mask_qualified, _ = check_mask_quality(mask)
        if mask is not None and mask_qualified:
            valid_images.append(normalized_mask)
        else:
            print(f"⚠️ 掩码{mask_file}质量不合格，已排除")

    if filename_mismatches:
        print(f"❌ 以下掩码无对应图片：")
        for f in filename_mismatches[:5]:
            print(f"  - {f}")

    print(f"找到{len(valid_images)}个有效样本")
    if len(valid_images) == 0:
        print("❌ 未找到有效样本")
        return False

    # 划分数据集
    train_images, val_images = train_test_split(
        valid_images,
        test_size=0.3,
        random_state=42,
        shuffle=True
    )

    # 复制训练集文件
    for img in train_images:
        shutil.copy2(
            os.path.join(image_dir, img),
            os.path.join(train_dir, "images", SECOND_LEVEL_FOLDER, img)
        )
        shutil.copy2(
            os.path.join(mask_dir, img),
            os.path.join(train_dir, "masks", SECOND_LEVEL_FOLDER, img)
        )

    # 复制验证集文件
    for img in val_images:
        shutil.copy2(
            os.path.join(image_dir, img),
            os.path.join(val_dir, "images", SECOND_LEVEL_FOLDER, img)
        )
        shutil.copy2(
            os.path.join(mask_dir, img),
            os.path.join(val_dir, "masks", SECOND_LEVEL_FOLDER, img)
        )

    print(f"✅ 数据集划分完成：训练集{len(train_images)}张，验证集{len(val_images)}张")
    return True


if __name__ == "__main__":
    # 路径配置（保持原有项目结构）
    IMAGE_DIR = "D:/pythonCode/1/original_images"
    JSON_DIR = "D:/pythonCode/1/annotations"
    MASK_DIR = "D:/pythonCode/1/masks"
    TRAIN_DIR = "D:/pythonCode/1/dataset/train"
    VAL_DIR = "D:/pythonCode/1/dataset/val"

    # 生成掩码
    print("=" * 50)
    print("开始生成缺陷掩码...")
    if not json_to_mask(IMAGE_DIR, JSON_DIR, MASK_DIR):
        print("❌ 掩码生成失败，程序退出")
        exit(1)

    # 划分数据集
    print("\n" + "=" * 50)
    print("开始划分训练集/验证集...")
    if not split_dataset(IMAGE_DIR, MASK_DIR, TRAIN_DIR, VAL_DIR):
        print("❌ 数据集划分失败，程序退出")
        exit(1)

    print("\n" + "=" * 50)
    print("✅ 所有数据准备工作完成！")
