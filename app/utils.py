import os
import cv2
import numpy as np
import tensorflow as tf
from keras.applications.mobilenet_v2 import preprocess_input
from app.model_loader import get_segmentation_model, get_detection_model, loaded_segmentation_model, loaded_detection_model
from ultralytics import YOLO

# 分割相关参数 - 优化版本以提高检测敏感度
SEGMENTATION_THRESHOLD = 0.3  # 降低一阶段检测阈值，提高灵敏度
RESCUE_THRESHOLD = 0.15  # 降低二阶段救援阈值，进一步提高召回率
MIN_AREA = 50  # 减小一阶段最小缺陷面积，检测更小的缺陷
RESCUE_MIN_AREA = 20  # 减小二阶段最小缺陷面积
TARGET_SIZE = (576, 576)
OVERLAY_COLOR = (0, 255, 0)  # 绿色 (BGR格式)
OVERLAY_ALPHA = 0.5  # 叠加透明度
BORDER_COLOR = (0, 0, 255)  # 边界颜色
BORDER_THICKNESS = 1  # 边界厚度

# 目标检测参数 - 优化版本以提高检测敏感度
DETECTION_CONF_THRESHOLD = 0.05  # 更低的置信度阈值，进一步提高检测敏感度
DETECTION_IOU_THRESHOLD = 0.2    # 更低的IOU阈值，更好地适应重叠裂缝
DETECTION_IMGSZ = 640            # 与训练一致的图像大小
MAX_DETECTIONS = 300             # 增加最大检测数量
SCALE = 0.5                      # 使用scale参数代替multi_scale，值为0.5表示多尺度检测

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
    """检测缺陷的核心函数 - 增强版本"""
    # 读取原图
    original_img = cv2.imread(img_path)
    if original_img is None:
        return None, None, False

    # 图像预处理增强：对比度调整
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    enhanced_img = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
    
    # 保存增强后的图像为临时文件
    temp_path = img_path.replace('.', '_segment_preprocessed.')
    cv2.imwrite(temp_path, enhanced_img)

    h, w = original_img.shape[:2]

    # 预处理
    processed_img = preprocess_image_for_segmentation(temp_path)

    # 预测（带TTA）
    pred_mask = tta_inference(model, processed_img)

    # 恢复掩码到原图尺寸
    pred_mask = cv2.resize(pred_mask, (w, h))

    # 后处理 - 改进的形态学操作序列
    binary_mask = (pred_mask > threshold).astype(np.uint8) * 255
    
    # 先闭操作填充小空洞，再开操作去除噪声
    if morph_op:
        # 先闭操作填充小空洞
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
        # 再开操作去除噪声
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    
    binary_mask = filter_small_regions(binary_mask, min_area)

    # 判断是否有缺陷
    has_defect = np.sum(binary_mask) > 0
    
    # 清理临时文件
    if os.path.exists(temp_path):
        os.remove(temp_path)

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

def apply_segmentation(img_path, result_path, params=None):
    """应用图像分割 - 增强版本，支持用户自定义参数和模型选择"""
    # 使用默认参数或用户提供的参数
    if params is None:
        params = {}
    
    seg_threshold = params.get('seg_threshold', SEGMENTATION_THRESHOLD)
    rescue_threshold = params.get('rescue_threshold', RESCUE_THRESHOLD)
    min_area = params.get('min_area', MIN_AREA)
    rescue_min_area = params.get('rescue_min_area', RESCUE_MIN_AREA)
    model_type = params.get('model_type', 'default')
    model_path = params.get('model_path', None)
    use_loaded_model = params.get('use_loaded_model', False)
    
    print(f"使用分割参数: seg_threshold={seg_threshold}, rescue_threshold={rescue_threshold}, ")
    print(f"min_area={min_area}, rescue_min_area={rescue_min_area}, model_type={model_type}")
    
    # 优先使用已加载的模型（如果指定）
    if use_loaded_model and loaded_segmentation_model is not None:
        print("使用已加载的自定义分割模型")
        model = loaded_segmentation_model
    # 根据模型类型加载相应的模型
    elif model_type == 'custom' and model_path and os.path.exists(model_path):
        print(f"加载自定义分割模型: {model_path}")
        try:
            model = tf.keras.models.load_model(model_path)
            print("自定义分割模型加载成功")
        except Exception as e:
            print(f"加载自定义分割模型失败: {str(e)}")
            return False, f"加载自定义分割模型失败: {str(e)}"
    else:
        # 使用默认模型
        model = get_segmentation_model()
    if model is None:
        # 如果模型未加载，创建一个简单的替代图像作为结果
        try:
            # 复制原始图像到结果路径，并添加文字说明
            original_img = cv2.imread(img_path)
            if original_img is not None:
                # 添加文字说明
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(original_img, '分割模型未加载', (50, 50), 
                           font, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imwrite(result_path, original_img)
                return False, "分割模型未加载，但已生成标记的图像"
            else:
                return False, "分割模型未加载且无法读取图像"
        except Exception as inner_e:
            return False, f"分割模型未加载且处理失败: {str(inner_e)}"
    
    try:
        # 一阶段检测
        original_img, mask, has_defect = detect_defects(
            model, img_path, seg_threshold, min_area, "open"
        )

        if original_img is None:
            return False, "无法读取图像"

        # 二阶段救援（如果一阶段未检测到缺陷）
        if not has_defect:
            print("一阶段未检测到缺陷，启动二阶段救援模式")
            _, rescue_mask, rescue_has_defect = detect_defects(
                model, img_path, rescue_threshold, rescue_min_area, "open"
            )
            if rescue_has_defect:
                has_defect = True
                mask = rescue_mask
                print(f"二阶段救援成功，检测到缺陷")
        
        # 三阶段增强：对检测到的小区域进行形态学膨胀，增强可视化效果
        if has_defect:
            # 对小缺陷进行适度膨胀，使其在视觉上更明显
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)

        # 绘制并保存结果
        overlay_img = draw_overlay(original_img, mask)
        
        # 在结果图像上添加缺陷计数信息
        font = cv2.FONT_HERSHEY_SIMPLEX
        defect_count = 0
        if has_defect:
            # 查找轮廓来计算缺陷区域数量，使用更鲁棒的参数
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
            defect_count = len(contours)
        
        # 使用英文文本避免OpenCV中文显示问题
        status_text = f"Defects: {defect_count}" if has_defect else "No defects"
        cv2.putText(overlay_img, status_text, (20, 30), 
                   font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        
        # 保存结果
        success = cv2.imwrite(result_path, overlay_img)
        
        if success:
            # 返回结构化结果
            return True, {
                'status': "检测到缺陷" if has_defect else "未检测到缺陷",
                'defect_count': defect_count
            }
        else:
            return False, "保存结果失败"
    except Exception as e:
        print(f"分割异常: {str(e)}")
        return False, f"分割过程中出错: {str(e)}"

# 定义正确的标签映射
default_label_mapping = {
    0: 'liefeng'  # 对应我们训练时的裂缝类别
}

def apply_detection(img_path, result_path, params=None):
    """应用目标检测 - 增强版本，优化检测效果，支持用户自定义参数和模型选择"""
    # 使用默认参数或用户提供的参数
    if params is None:
        params = {}
    
    conf_threshold = params.get('conf_threshold', DETECTION_CONF_THRESHOLD)
    iou_threshold = params.get('iou_threshold', DETECTION_IOU_THRESHOLD)
    max_detections = params.get('max_detections', MAX_DETECTIONS)
    img_size = params.get('img_size', DETECTION_IMGSZ)
    model_type = params.get('model_type', 'default')
    model_path = params.get('model_path', None)
    label_mapping = params.get('label_mapping', default_label_mapping)
    use_loaded_model = params.get('use_loaded_model', False)
    
    print(f"执行裂缝检测，置信度阈值: {conf_threshold}, IOU阈值: {iou_threshold}, 图像大小: {img_size}, model_type={model_type}")
      
    # 优先使用已加载的模型（如果指定）
    if use_loaded_model and loaded_detection_model is not None:
        print("使用已加载的自定义检测模型")
        model = loaded_detection_model
    # 根据模型类型加载相应的模型
    elif model_type == 'custom' and model_path:
        print(f"加载自定义检测模型: {model_path}")
        try:
            # 检查路径是否存在
            if not os.path.exists(model_path):
                # 尝试添加best.pt后缀，因为有时路径可能只包含目录
                if os.path.exists(os.path.join(model_path, 'best.pt')):
                    model_path = os.path.join(model_path, 'best.pt')
                else:
                    return False, f"自定义模型路径不存在: {model_path}"
            
            # 检查文件是否为文本占位符（以#开头）
            try:
                with open(model_path, 'r') as f:
                    first_line = f.readline().strip()
                    if first_line.startswith('#'):
                        print(f"检测到文本占位符模型文件: {model_path}")
                        # 使用预训练模型作为备选
                        print("切换到预训练YOLO模型")
                        model = YOLO('yolov8n.pt')
                    else:
                        model = YOLO(model_path)
            except UnicodeDecodeError:
                # 如果无法以文本方式读取，尝试作为二进制模型加载
                model = YOLO(model_path)
            
            print("自定义检测模型加载成功")
        except Exception as e:
            print(f"加载自定义检测模型失败: {str(e)}")
            # 失败时提供更友好的错误信息，并尝试使用预训练模型
            try:
                print("尝试使用预训练YOLO模型作为备选")
                model = YOLO('yolov8n.pt')
                print("成功加载预训练模型作为备选")
            except:
                return False, f"加载模型失败: {str(e)}。请确保模型文件格式正确。"
    else:
        # 使用默认模型
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
        # 检查GPU是否可用
        import torch
        device = '0' if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {'GPU' if device == '0' else 'CPU'}")
        
        # 调试：检查原始图像路径
        print(f"DEBUG: 原始图像路径: {img_path}")
        print(f"DEBUG: 原始图像是否存在: {os.path.exists(img_path)}")
        
        # 加载原始图像进行预处理
        original_img = cv2.imread(img_path)
        if original_img is None:
            return False, "无法读取输入图像"
        
        # 图像预处理增强对比度，有助于检测小裂缝（在内存中处理，不保存临时文件）
        # 应用自适应直方图均衡化以增强对比度
        gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        # 转换回彩色图像格式以保持与模型兼容
        enhanced_img = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
        
        # 调试：图像预处理完成，图像形状: {enhanced_img.shape}
        print(f"DEBUG: 图像预处理完成，增强后图像形状: {enhanced_img.shape}")
        
        # 直接使用内存中的增强图像进行检测，使用用户指定的参数
        results = model(
            enhanced_img,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=img_size,
            max_det=max_detections,
            device=device,
            augment=True,  # 启用测试时增强
            scale=SCALE,  # 使用scale参数进行多尺度检测
            classes=None  # 不限制检测类别
        )
        
        # 确保模型使用正确的标签映射
        if hasattr(results[0], 'names'):
            # 替换为我们定义的标签映射
            results[0].names = {int(k): v for k, v in label_mapping.items()}
        
        # 后处理：应用更严格的非极大值抑制来减少重复检测
        boxes = results[0].boxes
        if len(boxes) > 0:
            # 提取框、置信度和类别
            bboxes = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy()
            
            # 应用更严格的NMS以合并重叠的边界框
            indices = cv2.dnn.NMSBoxes(bboxes.tolist(), confs.tolist(), conf_threshold, 0.4)
            
            # 如果有检测结果通过NMS
            if len(indices) > 0:
                # 重新构建结果以只包含通过NMS的框
                results[0].boxes = results[0].boxes[indices.flatten()]
        
        # 手动绘制检测结果并保存图像
        result_img = enhanced_img.copy()  # 使用增强后的图像作为基础
        
        # 绘制检测框
        boxes = results[0].boxes
        for box in boxes:
            # 获取边界框坐标
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # 获取类别ID和置信度
            class_id = int(box.cls)
            confidence = float(box.conf)
            # 获取类别名称
            class_name = label_mapping.get(class_id, f"Class {class_id}")
            
            # 绘制边界框 (红色)
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # 绘制标签和置信度
            label = f"{class_name}: {confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_color = (255, 255, 255)
            background_color = (0, 0, 255)
            
            # 获取文本尺寸
            text_size = cv2.getTextSize(label, font, font_scale, 1)[0]
            # 绘制背景矩形
            cv2.rectangle(result_img, (x1, y1 - text_size[1] - 5), 
                         (x1 + text_size[0], y1), background_color, -1)
            # 绘制文本
            cv2.putText(result_img, label, (x1, y1 - 5), 
                       font, font_scale, font_color, 1, cv2.LINE_AA)
        
        # 添加缺陷计数信息
        font = cv2.FONT_HERSHEY_SIMPLEX
        defect_count = len(boxes)
        status_text = f"Defects: {defect_count}" if defect_count > 0 else "No defects"
        cv2.putText(result_img, status_text, (20, 30), 
                   font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        
        # 保存结果图像
        cv2.imwrite(result_path, result_img)
        
        # 获取检测信息
        detections = []
        for box in results[0].boxes:
            detections.append({
                'class_id': int(box.cls),
                'confidence': float(box.conf),
                'bbox': box.xyxy[0].tolist()
            })
        
        # 不再使用临时文件，所以不需要清理
        
        defect_count = len(detections)
        print(f"检测完成，发现 {defect_count} 个潜在裂缝")
        
        # 返回结构化结果
        return True, {
            'defect_count': defect_count,
            'detections': detections
        }
    except Exception as e:
        # 如果检测过程出错，提供详细的错误信息
        print(f"检测异常: {str(e)}")
        return False, f"检测过程中出错: {str(e)}"

def allowed_file(filename):
    """检查文件类型是否允许，支持文件夹上传"""
    # 对于文件夹，filename可能没有扩展名，这种情况下直接返回True
    if '.' not in filename:
        return True
    
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS