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
            # 直接使用TensorFlow加载模型，并设置compile=False避免自定义损失函数问题
            import tensorflow as tf
            segmentation_model = tf.keras.models.load_model(SEGMENTATION_MODEL_PATH, compile=False)
            print("图像分割模型加载成功")
        else:
            print(f"未找到分割模型文件: {SEGMENTATION_MODEL_PATH}")
    except Exception as e:
        print(f"加载分割模型时出错: {str(e)}")

def load_detection_model():
    """加载目标检测模型 - 完全使用Ultralytics官方API"""
    global detection_model
    
    # 使用绝对路径确保文件定位准确
    model_path = os.path.join('2', 'models', 'crack_detector_optimized', 'weights', 'best.pt')
    
    # 第一步：检查路径是否存在，输出详细信息帮助调试
    if not os.path.exists(model_path):
        print(f"❌ 模型路径不存在: {model_path}")
        print(f"当前工作目录: {os.getcwd()}")
        # 尝试检查不同可能的路径变体
        alt_path1 = os.path.join('d:', 'pythonCode', '工业检测', '2', 'models', 'crack_detector_optimized', 'weights', 'best.pt')
        if os.path.exists(alt_path1):
            print(f"✅ 发现替代路径: {alt_path1}")
            model_path = alt_path1
        else:
            print(f"❌ 替代路径也不存在: {alt_path1}")
        return None
    
    print(f"✅ 模型文件存在: {model_path}")
    
    # 第二步：完全使用Ultralytics官方API加载模型
    try:
        # 使用Ultralytics官方YOLO类，这是推荐的官方API使用方式
        # YOLO类内部会处理所有与PyTorch版本兼容性相关的问题
        print("正在使用Ultralytics官方API加载YOLO模型...")
        detection_model = YOLO(model_path, task='detect')
        print("✅ YOLO检测模型加载成功")
        
        # 验证模型是否可用
        try:
            # 进行简单的模型验证
            model_info = detection_model.info()
            print(f"✅ 模型验证成功: {model_info}")
        except Exception as verify_error:
            print(f"⚠️  模型加载成功但验证时出错: {str(verify_error)}")
            # 即使验证出错，我们仍然返回模型，因为可能只是某些方法不可用
            
        return detection_model
    except Exception as e:
        print(f"❌ YOLO模型加载失败: {str(e)}")
        
        # 尝试加载模型的不同版本
        print("尝试加载替代模型变体...")
        model_variants = [
            os.path.join('2', 'models', 'crack_detector_optimized2', 'weights', 'best.pt'),
            os.path.join('2', 'models', 'crack_detector_optimized3', 'weights', 'best.pt'),
            os.path.join('2', 'models', 'pretrained', 'best.pt')
        ]
        
        for variant in model_variants:
            if os.path.exists(variant):
                print(f"尝试加载替代模型: {variant}")
                try:
                    detection_model = YOLO(variant, task='detect')
                    print(f"✅ 成功加载替代模型: {variant}")
                    return detection_model
                except Exception as variant_error:
                    print(f"❌ 替代模型加载失败: {str(variant_error)}")
        
        print("❌ 所有模型变体都加载失败")
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