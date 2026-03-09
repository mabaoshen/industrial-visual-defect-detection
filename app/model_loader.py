import os
import tensorflow as tf
from ultralytics import YOLO
import numpy as np
from flask import current_app

# 全局模型变量
segmentation_model = None
detection_model = None

# 用户上传的自定义模型变量
loaded_segmentation_model = None
loaded_detection_model = None

# 模型路径
SEGMENTATION_MODEL_PATH = os.path.join('1', 'best_model.h5')
DETECTION_MODEL_PATH = os.path.join('2', 'models', 'crack_detector_optimized', 'weights','best.pt')

def load_segmentation_model():
    """加载图像分割模型"""
    global segmentation_model
    try:
        # 确保路径正确
        if os.path.exists(SEGMENTATION_MODEL_PATH):
            # 验证HDF5文件格式
            try:
                with open(SEGMENTATION_MODEL_PATH, 'rb') as f:
                    # HDF5文件的魔数是b'\x89HDF\r\n\x1a\n'
                    magic_number = f.read(8)
                    if magic_number != b'\x89HDF\r\n\x1a\n':
                        raise ValueError("不是有效的HDF5文件格式")
            except Exception as e:
                print(f"HDF5文件验证失败: {str(e)}")
                return
                
            # 直接使用TensorFlow加载模型，并设置compile=False避免自定义损失函数问题
            import tensorflow as tf
            segmentation_model = tf.keras.models.load_model(SEGMENTATION_MODEL_PATH, compile=False)
            print("图像分割模型加载成功")
        else:
            print(f"未找到分割模型文件: {SEGMENTATION_MODEL_PATH}")
    except Exception as e:
        # 提供更友好的错误信息
        if 'invalid load key' in str(e) or 'HDF5' in str(e):
            print(f"加载分割模型时出错: 模型文件可能已损坏或不是有效的HDF5格式")
        else:
            print(f"加载分割模型时出错: {str(e)}")

def load_detection_model():
    """加载目标检测模型 - 完全使用Ultralytics官方API"""
    global detection_model
    
    # 首先尝试加载默认模型路径
    default_model_path = os.path.join('2', 'models', 'crack_detector_optimized', 'weights', 'best.pt')
    
    # 定义所有可能的模型路径，优先使用最新的优化模型
    model_paths = [
        default_model_path,
        os.path.join('2', 'models', 'crack_detector_optimized3', 'weights', 'best.pt'),
        os.path.join('2', 'models', 'crack_detector_optimized2', 'weights', 'best.pt'),
        os.path.join('2', 'models', 'pretrained', 'yolov8n.pt')  # 使用预训练模型作为最后的备选
    ]
    
    # 遍历所有可能的模型路径，尝试加载第一个可用的模型
    for model_path in model_paths:
        # 检查路径是否存在
        if not os.path.exists(model_path):
            print(f"❌ 模型路径不存在: {model_path}")
            continue
        
        # 验证.pt文件格式（检查文件是否为空）
        if os.path.getsize(model_path) == 0:
            print(f"❌ 模型文件为空: {model_path}")
            continue
        
        print(f"✅ 发现模型文件: {model_path}")
        
        try:
            # 使用Ultralytics官方YOLO类加载模型
            print("正在使用Ultralytics官方API加载YOLO模型...")
            detection_model = YOLO(model_path, task='detect')
            print(f"✅ 成功加载模型: {model_path}")
            
            # 验证模型是否可用
            try:
                model_info = detection_model.info()
                print(f"✅ 模型验证成功")
            except Exception as verify_error:
                print(f"⚠️  模型加载成功但验证时出错: {str(verify_error)}")
                # 即使验证出错，我们仍然使用模型
            
            return detection_model
        except Exception as e:
            error_msg = str(e)
            if 'invalid load key' in error_msg:
                print(f"❌ 加载模型失败: 模型文件可能已损坏或格式错误 ({model_path})")
            else:
                print(f"❌ 加载模型失败: {error_msg}")
            continue
    
    print("❌ 所有模型路径都无法加载成功")
    return None

def load_models():
    """加载所有模型，确保即使部分模型加载失败应用也能运行"""
    print("开始加载模型...")
    
    # 加载分割模型
    load_segmentation_model()
    
    # 加载检测模型
    detection_result = load_detection_model()
    if detection_result is None:
        print("⚠️  注意: 检测模型加载失败，但不影响分割功能使用")
    
    print("模型加载过程完成")
    # 打印当前模型状态摘要
    if segmentation_model is not None:
        print("✅ 分割模型: 已就绪")
    else:
        print("❌ 分割模型: 未就绪")
        
    if detection_model is not None:
        print("✅ 检测模型: 已就绪")
    else:
        print("❌ 检测模型: 未就绪 (分割功能仍可正常使用)")

def get_segmentation_model():
    """获取分割模型实例"""
    return segmentation_model

def get_detection_model():
    """获取检测模型实例"""
    return detection_model