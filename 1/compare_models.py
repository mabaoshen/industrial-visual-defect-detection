import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import confusion_matrix

# 配置
DEEPLAB_MODEL_PATH = "best_model.h5"
UNET_MODEL_PATH = "best_unet_model.h5"
TEST_IMAGES_DIR = "images"
TEST_MASKS_DIR = "masks"
OUTPUT_DIR = "model_comparison_results"

# 两种模型的输入尺寸
DEEPLAB_SIZE = (576, 576)
UNET_SIZE = (320, 320)

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_model(model_path, input_size):
    """加载模型并设置正确的输入尺寸"""
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print(f"✅ 成功加载模型: {model_path}")
        return model
    except Exception as e:
        print(f"❌ 加载模型失败 {model_path}: {str(e)}")
        return None


def preprocess_image(image_path, target_size, model_type):
    """根据模型类型预处理图像"""
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # 转换为RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 调整大小
    img_resized = cv2.resize(img, target_size)
    
    # 根据模型类型进行预处理
    if model_type == "deeplab":
        # MobileNetV2预处理
        img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_resized)
    else:  # unet
        # 简单的0-1归一化
        img_preprocessed = img_resized.astype(np.float32) / 255.0
    
    # 添加批次维度
    img_preprocessed = np.expand_dims(img_preprocessed, axis=0)
    
    return img, img_preprocessed


def preprocess_mask(mask_path, original_size):
    """预处理掩码"""
    # 读取掩码
    mask = cv2.imread(mask_path, 0)
    if mask is None:
        return None
    
    # 调整回原始图像尺寸
    mask_resized = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)
    
    # 二值化
    mask_binary = (mask_resized > 128).astype(np.uint8)
    
    return mask_binary


def postprocess_prediction(prediction, original_size, threshold=0.5):
    """后处理模型预测结果"""
    # 移除批次维度
    pred = np.squeeze(prediction)
    
    # 应用阈值
    pred_binary = (pred > threshold).astype(np.uint8)
    
    # 调整回原始图像尺寸
    pred_resized = cv2.resize(pred_binary, original_size, interpolation=cv2.INTER_NEAREST)
    
    return pred_resized


def calculate_metrics(true_mask, pred_mask):
    """计算评估指标"""
    # 展平数组
    true_flat = true_mask.flatten()
    pred_flat = pred_mask.flatten()
    
    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(true_flat, pred_flat).ravel()
    
    # 计算各种指标
    # 精确率 (Precision)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # 召回率 (Recall)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # F1分数
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # IoU (Intersection over Union)
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    
    # Dice系数
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    
    # 计算目标区域占比
    target_ratio = np.sum(true_flat) / len(true_flat)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "dice": dice,
        "target_ratio": target_ratio,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn
    }


def visualize_comparison(original_img, true_mask, deeplab_pred, unet_pred, image_name):
    """可视化比较结果"""
    plt.figure(figsize=(20, 5))
    
    # 原始图像
    plt.subplot(1, 4, 1)
    plt.imshow(original_img)
    plt.title("原始图像")
    plt.axis('off')
    
    # 真实掩码
    plt.subplot(1, 4, 2)
    plt.imshow(true_mask, cmap='gray')
    plt.title("真实掩码")
    plt.axis('off')
    
    # DeepLabV3+预测
    plt.subplot(1, 4, 3)
    plt.imshow(deeplab_pred, cmap='gray')
    plt.title("DeepLabV3+预测")
    plt.axis('off')
    
    # U-Net预测
    plt.subplot(1, 4, 4)
    plt.imshow(unet_pred, cmap='gray')
    plt.title("U-Net预测")
    plt.axis('off')
    
    # 保存图像
    output_path = os.path.join(OUTPUT_DIR, f"comparison_{image_name}.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存单独的预测结果
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"deeplab_{image_name}.png"), deeplab_pred * 255)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"unet_{image_name}.png"), unet_pred * 255)


def analyze_small_target_performance(all_metrics):
    """分析小目标性能"""
    # 根据目标区域比例分组
    small_targets = []  # 小于5%
    medium_targets = []  # 5% - 15%
    large_targets = []  # 大于15%
    
    for metrics in all_metrics:
        target_ratio = metrics["target_ratio"]
        if target_ratio < 0.05:
            small_targets.append(metrics)
        elif target_ratio < 0.15:
            medium_targets.append(metrics)
        else:
            large_targets.append(metrics)
    
    # 计算每组的平均指标
    def calculate_average(group, name):
        if not group:
            return None
        
        avg_metrics = {
            "category": name,
            "count": len(group),
            "avg_target_ratio": np.mean([m["target_ratio"] for m in group]),
            "deeplab": {
                "avg_iou": np.mean([m["deeplab"]["iou"] for m in group]),
                "avg_dice": np.mean([m["deeplab"]["dice"] for m in group]),
                "avg_precision": np.mean([m["deeplab"]["precision"] for m in group]),
                "avg_recall": np.mean([m["deeplab"]["recall"] for m in group]),
                "avg_f1": np.mean([m["deeplab"]["f1"] for m in group])
            },
            "unet": {
                "avg_iou": np.mean([m["unet"]["iou"] for m in group]),
                "avg_dice": np.mean([m["unet"]["dice"] for m in group]),
                "avg_precision": np.mean([m["unet"]["precision"] for m in group]),
                "avg_recall": np.mean([m["unet"]["recall"] for m in group]),
                "avg_f1": np.mean([m["unet"]["f1"] for m in group])
            },
            "improvement": {
                "iou": np.mean([m["unet"]["iou"] - m["deeplab"]["iou"] for m in group]),
                "dice": np.mean([m["unet"]["dice"] - m["deeplab"]["dice"] for m in group]),
                "precision": np.mean([m["unet"]["precision"] - m["deeplab"]["precision"] for m in group]),
                "recall": np.mean([m["unet"]["recall"] - m["deeplab"]["recall"] for m in group]),
                "f1": np.mean([m["unet"]["f1"] - m["deeplab"]["f1"] for m in group])
            }
        }
        return avg_metrics
    
    small_results = calculate_average(small_targets, "小目标 (<5%)")
    medium_results = calculate_average(medium_targets, "中目标 (5%-15%)")
    large_results = calculate_average(large_targets, "大目标 (>15%)")
    
    return small_results, medium_results, large_results


def visualize_comparison_no_mask(original_img, deeplab_pred, unet_pred, image_name):
    """不使用真实掩码的可视化比较结果"""
    plt.figure(figsize=(15, 5))
    
    # 原始图像
    plt.subplot(1, 3, 1)
    plt.imshow(original_img)
    plt.title("原始图像")
    plt.axis('off')
    
    # DeepLabV3+预测
    plt.subplot(1, 3, 2)
    plt.imshow(deeplab_pred, cmap='gray')
    plt.title("DeepLabV3+预测")
    plt.axis('off')
    
    # U-Net预测
    plt.subplot(1, 3, 3)
    plt.imshow(unet_pred, cmap='gray')
    plt.title("U-Net预测")
    plt.axis('off')
    
    # 保存图像
    output_path = os.path.join(OUTPUT_DIR, f"comparison_{image_name}.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    print("开始比较DeepLabV3+和U-Net模型性能...")
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 加载模型
    deeplab_model = load_model(DEEPLAB_MODEL_PATH, DEEPLAB_SIZE)
    unet_model = load_model(UNET_MODEL_PATH, UNET_SIZE)
    
    if not deeplab_model or not unet_model:
        print("❌ 模型加载失败，无法继续比较")
        return
    
    # 获取图像列表
    image_files = glob.glob(os.path.join(TEST_IMAGES_DIR, "*.png"))
    if not image_files:
        print(f"❌ 在目录 {TEST_IMAGES_DIR} 中未找到测试图像")
        return
    
    print(f"找到 {len(image_files)} 张测试图像")
    
    # 处理每张图像
    for i, image_path in enumerate(image_files, 1):
        image_name = os.path.basename(image_path)
        print(f"\n处理图像 {i}/{len(image_files)}: {image_name}")
        
        # 读取原始图像以获取原始尺寸
        original_img = cv2.imread(image_path)
        if original_img is None:
            print(f"  ❌ 无法读取图像: {image_path}")
            continue
        original_height, original_width = original_img.shape[:2]
        original_size = (original_width, original_height)
        
        # DeepLabV3+模型推理
        deeplab_img, deeplab_preprocessed = preprocess_image(image_path, DEEPLAB_SIZE, "deeplab")
        if deeplab_preprocessed is None:
            print(f"  ❌ DeepLabV3+预处理失败")
            continue
        
        print("  正在进行DeepLabV3+推理...")
        deeplab_prediction = deeplab_model.predict(deeplab_preprocessed, verbose=0)
        deeplab_result = postprocess_prediction(deeplab_prediction, original_size)
        
        # U-Net模型推理
        unet_img, unet_preprocessed = preprocess_image(image_path, UNET_SIZE, "unet")
        if unet_preprocessed is None:
            print(f"  ❌ U-Net预处理失败")
            continue
        
        print("  正在进行U-Net推理...")
        unet_prediction = unet_model.predict(unet_preprocessed, verbose=0)
        unet_result = postprocess_prediction(unet_prediction, original_size)
        
        # 查找对应的掩码文件
        mask_path = os.path.join(TEST_MASKS_DIR, image_name)
        has_mask = os.path.exists(mask_path)
        
        if has_mask:
            # 预处理真实掩码
            true_mask = preprocess_mask(mask_path, original_size)
            if true_mask is not None:
                # 使用有掩码的可视化
                visualize_comparison(
                    original_img=cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB),
                    true_mask=true_mask,
                    deeplab_pred=deeplab_result,
                    unet_pred=unet_result,
                    image_name=os.path.splitext(image_name)[0]
                )
                print("  ✅ 已生成包含真实掩码的比较图像")
            else:
                # 使用无掩码的可视化
                visualize_comparison_no_mask(
                    original_img=cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB),
                    deeplab_pred=deeplab_result,
                    unet_pred=unet_result,
                    image_name=os.path.splitext(image_name)[0]
                )
                print("  ✅ 已生成无真实掩码的比较图像")
        else:
            # 使用无掩码的可视化
            visualize_comparison_no_mask(
                original_img=cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB),
                deeplab_pred=deeplab_result,
                unet_pred=unet_result,
                image_name=os.path.splitext(image_name)[0]
            )
            print("  ✅ 已生成无真实掩码的比较图像")
        
        # 保存单独的预测结果
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"deeplab_{os.path.splitext(image_name)[0]}.png"), deeplab_result * 255)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"unet_{os.path.splitext(image_name)[0]}.png"), unet_result * 255)
        print("  ✅ 已保存单独的预测结果")
    
    # 保存比较说明
    with open(os.path.join(OUTPUT_DIR, "comparison_guide.txt"), "w", encoding="utf-8") as f:
        f.write("====== 模型预测比较指南 ======\n\n")
        f.write("本目录包含DeepLabV3+和U-Net模型在相同测试图像上的分割结果比较。\n\n")
        f.write("文件说明：\n")
        f.write("1. comparison_*.png - 包含原始图像、DeepLabV3+预测和U-Net预测的并排比较图\n")
        f.write("2. deeplab_*.png - DeepLabV3+模型的二值分割预测结果\n")
        f.write("3. unet_*.png - U-Net模型的二值分割预测结果\n\n")
        f.write("视觉评估要点：\n")
        f.write("1. 小目标检测能力 - 观察较小的目标区域是否被准确分割\n")
        f.write("2. 边界准确性 - 比较分割边界与实际目标边界的吻合程度\n")
        f.write("3. 假阳性/假阴性 - 注意是否存在过多的误检测或漏检测\n")
        f.write("4. 整体一致性 - 评估整个图像中分割结果的稳定性和一致性\n")
    
    print(f"\n比较完成！所有视觉比较结果已保存在 {OUTPUT_DIR} 目录中")
    print(f"请查看 {OUTPUT_DIR}\comparison_guide.txt 获取详细的评估指南")


if __name__ == "__main__":
    main()
