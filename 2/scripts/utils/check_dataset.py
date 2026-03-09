import os
from pathlib import Path


def check_dataset():
    """检查数据集的完整性和一致性"""
    # 数据集路径（相对路径）
    train_img_dir = "../../dataset/images/train"
    train_label_dir = "../../dataset/labels/train"
    val_img_dir = "../../dataset/images/val"
    val_label_dir = "../../dataset/labels/val"

    # 检查函数
    def check_split(img_dir, label_dir, split_name):
        """检查单个数据集分割（训练集或验证集）"""
        print(f"\n检查{split_name}集...")

        # 获取所有图片和标注文件
        img_files = {f.stem for f in Path(img_dir).glob('*.*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']}
        label_files = {f.stem for f in Path(label_dir).glob('*.txt')}

        # 统计数量
        print(f"图片数量: {len(img_files)}")
        print(f"标注文件数量: {len(label_files)}")

        # 检查差异
        img_without_label = img_files - label_files
        label_without_img = label_files - img_files

        # 输出检查结果
        if not img_without_label and not label_without_img:
            print(f"{split_name}集检查通过: 所有图片都有对应的标注文件")
            return True
        else:
            if img_without_label:
                print(f"警告: 以下图片没有对应的标注文件:")
                for f in sorted(img_without_label):
                    print(f"  - {f}")
            if label_without_img:
                print(f"警告: 以下标注文件没有对应的图片:")
                for f in sorted(label_without_img):
                    print(f"  - {f}")
            return False

    # 检查训练集和验证集
    train_ok = check_split(train_img_dir, train_label_dir, "训练")
    val_ok = check_split(val_img_dir, val_label_dir, "验证")

    # 总体结果
    if train_ok and val_ok:
        print("\n===== 数据集检查通过 =====")
        print("可以开始训练模型")
        return True
    else:
        print("\n===== 数据集检查未通过 =====")
        print("请先修复上述问题再训练模型")
        return False


if __name__ == "__main__":
    check_dataset()