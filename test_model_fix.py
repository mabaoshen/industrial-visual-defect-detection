import os
from ultralytics import YOLO
import sys

def test_model_loading(model_path):
    """测试模型加载功能"""
    print(f"测试模型路径: {model_path}")
    
    try:
        # 检查路径是否存在
        if not os.path.exists(model_path):
            print(f"路径不存在: {model_path}")
            return False
        
        # 检查是否为目录
        if os.path.isdir(model_path):
            print(f"路径是目录，尝试查找best.pt")
            potential_pt_path = os.path.join(model_path, 'best.pt')
            if os.path.exists(potential_pt_path):
                model_path = potential_pt_path
                print(f"找到best.pt: {model_path}")
            else:
                print(f"目录中未找到best.pt文件")
                return False
        
        # 尝试加载模型
        model = YOLO(model_path)
        print("模型加载成功")
        
        # 验证模型
        try:
            info = model.info()
            print(f"模型信息: {info}")
        except Exception as verify_error:
            print(f"模型验证警告: {verify_error}")
            # 继续，因为某些模型可能无法调用info()
        
        return True
    except Exception as e:
        print(f"模型加载失败: {e}")
        return False

if __name__ == "__main__":
    # 测试常见的模型路径场景
    print("=== 测试模型加载修复 ===")
    
    # 场景1：直接指定.pt文件路径
    if len(sys.argv) > 1:
        test_path = sys.argv[1]
        print(f"\n测试用户提供的路径: {test_path}")
        success = test_model_loading(test_path)
        print(f"测试结果: {'成功' if success else '失败'}")
    else:
        # 场景2：测试可能的模型目录
        test_dirs = [
            '2/models/crack_detector_optimized',
            '2/models/crack_detector_optimized/weights',
            '2/models/crack_detector_optimized/weights/best.pt',
            '2/models/crack_detector_20251110_202517',
            '2/models/crack_detector_20251110_202517/weights'
        ]
        
        for test_dir in test_dirs:
            print(f"\n测试路径: {test_dir}")
            success = test_model_loading(test_dir)
            print(f"测试结果: {'成功' if success else '失败'}")
    
    print("\n=== 测试完成 ===")