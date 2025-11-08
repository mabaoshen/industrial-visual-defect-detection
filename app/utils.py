import os
import cv2
import numpy as np
import tensorflow as tf
from keras.applications.mobilenet_v2 import preprocess_input
from app.model_loader import get_segmentation_model, get_detection_model

# 分割相关参数
SEGMENTATION_THRESHOLD = 0.35  # 一阶段检测阈值
RESCUE_THRESHOLD = 0.2  # 二阶段救援阈值
MIN_AREA = 80  # 一阶段最小缺陷面积
RESCUE_MIN_AREA = 30  # 二阶段最小缺陷面积
TARGET_SIZE = (576, 576)
OVERLAY_COLOR = (0, 255, 0)  # 绿色 (BGR格式)
OVERLAY_ALPHA = 0.5  # 叠加透明度
BORDER_COLOR = (0, 0, 255)  # 边界颜色
BORDER_THICKNESS = 1  # 边界厚度

# 检测相关参数
DETECTION_CONF_THRESHOLD = 0.15  # 更低的置信度阈值，与单独运行的代码保持一致
DETECTION_IOU_THRESHOLD = 0.25   # 降低IOU阈值，适应重叠裂缝
DETECTION_IMGSZ = 640            # 与训练一致的图像大小

def preprocess_image_for_segmentation(img_path, target_size=(576, 576)):
    """预处理图像用于分割模型，使用模型期望的576x576尺寸"""
    # 读取原图并调整大小
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, target_size)
    # 使用与训练时相同的预处理方法
    img_preprocessed = preprocess_input(img_resized)
    img_array = np.expand_dims(img_preprocessed, axis=0)
    return img_array

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
        return None, None, False

    h, w = original_img.shape[:2]

    # 预处理
    processed_img = preprocess_image_for_segmentation(img_path)

    # 预测（带TTA）
    pred_mask = tta_inference(model, processed_img)

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

def apply_segmentation(img_path, result_path):
    """应用图像分割 - 优化版本，与单独运行的代码保持一致的效果"""
    model = get_segmentation_model()
    if model is None:
        return False, "分割模型未加载"
    
    try:
        # 一阶段检测
        original_img, mask, has_defect = detect_defects(
            model, img_path, SEGMENTATION_THRESHOLD, MIN_AREA, "open"
        )

        if original_img is None:
            return False, "无法读取图像"

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
        
        # 保存结果
        success = cv2.imwrite(result_path, overlay_img)
        
        # 计算缺陷区域数量
        defect_count = 0
        if has_defect:
            # 查找轮廓来计算缺陷区域数量
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            defect_count = len(contours)
        
        if success:
            # 返回结构化结果
            return True, {
                'status': "检测到缺陷" if has_defect else "未检测到缺陷",
                'defect_count': defect_count
            }
        else:
            return False, "保存结果失败"
    except Exception as e:
        return False, f"分割过程中出错: {str(e)}"

def apply_detection(img_path, result_path):
    """应用目标检测 - 优化版本，与单独运行的代码保持一致的效果"""
    model = get_detection_model()
    if model is None:
        # 如果模型未加载，创建一个简单的替代图像作为结果
        try:
            # 复制原始图像到结果路径，并添加文字说明
            original_img = cv2.imread(img_path)
            if original_img is not None:
                # 添加文字说明
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(original_img, '检测模型未加载', (50, 50), 
                           font, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imwrite(result_path, original_img)
                return False, "检测模型未加载，但已生成标记的图像"
            else:
                return False, "检测模型未加载且无法读取图像"
        except Exception as inner_e:
            return False, f"检测模型未加载且处理失败: {str(inner_e)}"
    
    try:
        # 执行检测，使用与单独运行代码相同的参数
        print(f"执行裂缝检测，置信度阈值: {DETECTION_CONF_THRESHOLD}, IOU阈值: {DETECTION_IOU_THRESHOLD}, 图像大小: {DETECTION_IMGSZ}")
        
        # 检查GPU是否可用
        import torch
        device = '0' if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {'GPU' if device == '0' else 'CPU'}")
        
        # 执行检测
        results = model(
            img_path,
            conf=DETECTION_CONF_THRESHOLD,
            iou=DETECTION_IOU_THRESHOLD,
            imgsz=DETECTION_IMGSZ,
            device=device
        )
        
        # 保存结果
        results[0].save(result_path)
        
        # 获取检测信息
        detections = []
        for box in results[0].boxes:
            detections.append({
                'class_id': int(box.cls),
                'confidence': float(box.conf),
                'bbox': box.xyxy[0].tolist()
            })
        
        defect_count = len(detections)
        print(f"检测完成，发现 {defect_count} 个潜在裂缝")
        
        # 返回结构化结果
        return True, {
            'defect_count': defect_count,
            'detections': detections
        }
    except Exception as e:
        # 如果检测过程出错，提供详细的错误信息
        return False, f"检测过程中出错: {str(e)}"

def allowed_file(filename):
    """检查文件类型是否允许"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS