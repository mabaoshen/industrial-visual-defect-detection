import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from train_model import enhanced_focal_loss, weighted_dice_loss, boundary_aware_loss, lovasz_softmax_loss, small_target_combined_loss, focal_loss, dice_loss, combined_loss

def create_test_data(target_size=0.01):
    """
    创建测试数据，包含不同大小的目标
    target_size: 目标区域占总图像的比例
    """
    batch_size = 4
    img_height = 256
    img_width = 256
    
    # 创建空白图像和掩码
    images = np.zeros((batch_size, img_height, img_width, 3), dtype=np.float32)
    masks = np.zeros((batch_size, img_height, img_width, 1), dtype=np.float32)
    
    for i in range(batch_size):
        # 随机位置和大小创建目标
        if i == 0:  # 小目标 (1%)
            size = int(np.sqrt(img_height * img_width * 0.01))
        elif i == 1:  # 中目标 (5%)
            size = int(np.sqrt(img_height * img_width * 0.05))
        elif i == 2:  # 大目标 (15%)
            size = int(np.sqrt(img_height * img_width * 0.15))
        else:  # 非常小的目标 (0.1%)
            size = int(np.sqrt(img_height * img_width * 0.001))
            
        # 确保大小至少为2
        size = max(2, size)
        
        # 随机位置
        x_start = np.random.randint(0, img_width - size)
        y_start = np.random.randint(0, img_height - size)
        
        # 在掩码上创建目标
        masks[i, y_start:y_start+size, x_start:x_start+size, 0] = 1.0
        
        # 在图像上标记目标位置（可视化用）
        images[i, y_start:y_start+size, x_start:x_start+size, :] = 1.0
    
    # 创建一些噪声预测
    np.random.seed(42)
    predictions = masks + np.random.normal(0, 0.2, masks.shape)
    predictions = np.clip(predictions, 0.0, 1.0)
    
    return images, masks, predictions

def test_loss_functions():
    """
    测试不同损失函数在不同大小目标上的表现
    """
    print("===== 测试损失函数在不同大小目标上的表现 =====")
    
    # 创建测试数据
    images, masks, predictions = create_test_data()
    
    # 转换为tf张量
    masks_tf = tf.convert_to_tensor(masks, dtype=tf.float32)
    predictions_tf = tf.convert_to_tensor(predictions, dtype=tf.float32)
    
    # 初始化损失函数
    original_focal = focal_loss()
    original_dice = dice_loss
    original_combined = combined_loss
    
    new_focal = enhanced_focal_loss()
    new_dice = weighted_dice_loss()
    new_boundary = boundary_aware_loss()
    new_lovasz = lovasz_softmax_loss
    new_combined = small_target_combined_loss
    
    # 计算每个样本的损失值
    print("\n各样本的损失值比较:")
    print("-" * 80)
    print(f"{'样本':<10}{'目标大小':<12}{'原始Focal':<12}{'新Focal':<12}{'原始Dice':<12}{'新Dice':<12}{'原始组合':<12}{'新组合':<12}")
    print("-" * 80)
    
    target_sizes = ["小目标(1%)", "中目标(5%)", "大目标(15%)", "极小目标(0.1%)"]
    
    for i in range(4):
        # 提取单个样本
        single_mask = masks_tf[i:i+1]
        single_pred = predictions_tf[i:i+1]
        
        # 计算各种损失值
        original_focal_val = float(original_focal(single_mask, single_pred).numpy())
        new_focal_val = float(new_focal(single_mask, single_pred).numpy())
        
        original_dice_val = float(original_dice(single_mask, single_pred).numpy())
        new_dice_val = float(new_dice(single_mask, single_pred).numpy())
        
        original_combined_val = float(original_combined(single_mask, single_pred).numpy())
        new_combined_val = float(new_combined(single_mask, single_pred).numpy())
        
        # 打印结果
        print(f"{i+1:<10}{target_sizes[i]:<12}{original_focal_val:<12.6f}{new_focal_val:<12.6f}{original_dice_val:<12.6f}{new_dice_val:<12.6f}{original_combined_val:<12.6f}{new_combined_val:<12.6f}")
    
    print("\n边界感知损失和Lovász损失:")
    for i in range(4):
        single_mask = masks_tf[i:i+1]
        single_pred = predictions_tf[i:i+1]
        
        boundary_val = float(new_boundary(single_mask, single_pred).numpy())
        lovasz_val = float(new_lovasz(single_mask, single_pred).numpy())
        
        print(f"样本 {i+1} ({target_sizes[i]}): 边界损失 = {boundary_val:.6f}, Lovász损失 = {lovasz_val:.6f}")
    
    # 计算整体性能提升
    original_total = float(original_combined(masks_tf, predictions_tf).numpy())
    new_total = float(new_combined(masks_tf, predictions_tf).numpy())
    improvement = ((original_total - new_total) / original_total) * 100 if original_total > 0 else 0
    
    print("\n整体性能比较:")
    print(f"原始组合损失: {original_total:.6f}")
    print(f"新组合损失: {new_total:.6f}")
    print(f"损失值减少: {(original_total - new_total):.6f} ({improvement:.2f}%)")
    
    print("\n===== 测试完成 =====")
    print("\n结论:")
    if improvement > 10:
        print("✓ 新的损失函数组合显著提高了性能!")
    elif improvement > 0:
        print("✓ 新的损失函数组合提高了性能。")
    else:
        print("! 新的损失函数组合在测试数据上没有显示出性能提升，建议调整参数。")
    
    print("\n注意: 这个测试仅使用了合成数据，在实际训练中可能会有不同的结果。")
    print("建议在实际数据集上进行完整的训练实验来验证改进效果。")

if __name__ == "__main__":
    test_loss_functions()
