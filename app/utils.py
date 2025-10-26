import os
import cv2
import numpy as np
import tensorflow as tf
from app.model_loader import get_segmentation_model, get_detection_model

def preprocess_image_for_segmentation(img_path, target_size=(576, 576)):
    """预处理图像用于分割模型，使用模型期望的576x576尺寸"""
    # 使用TensorFlow的现代API替代过时的keras.preprocessing
    img = tf.keras.utils.load_img(img_path, target_size=target_size)
    img_array = tf.keras.utils.img_to_array(img) / 255.0  # 归一化
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def apply_segmentation(img_path, result_path):
    """应用图像分割"""
    model = get_segmentation_model()
    if model is None:
        return False, "分割模型未加载"
    
    try:
        # 预处理图像
        processed_img = preprocess_image_for_segmentation(img_path)
        
        # 进行预测
        prediction = model.predict(processed_img)
        
        # 后处理
        pred_mask = (prediction[0, :, :, 0] > 0.5).astype(np.uint8) * 255
        
        # 读取原始图像
        original_img = cv2.imread(img_path)
        original_img = cv2.resize(original_img, (576, 576))
        
        # 创建彩色掩码
        colored_mask = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR)
        colored_mask[:, :, 0] = 0  # 蓝色通道设为0
        colored_mask[:, :, 1] = 0  # 绿色通道设为0
        
        # 将掩码叠加到原始图像上
        overlay = cv2.addWeighted(original_img, 0.7, colored_mask, 0.3, 0)
        
        # 保存结果
        cv2.imwrite(result_path, overlay)
        
        return True, "分割成功"
    except Exception as e:
        return False, f"分割过程中出错: {str(e)}"

def apply_detection(img_path, result_path, confidence=0.3):
    """应用目标检测，降低置信度阈值以提高检测灵敏度"""
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
        # 执行检测，降低置信度阈值
        print(f"执行裂缝检测，置信度阈值: {confidence}")
        results = model(img_path, conf=confidence)
        
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
        
        print(f"检测完成，发现 {len(detections)} 个潜在裂缝")
        return True, detections
    except Exception as e:
        # 如果检测过程出错，提供详细的错误信息
        return False, f"检测过程中出错: {str(e)}"

def allowed_file(filename):
    """检查文件类型是否允许"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS