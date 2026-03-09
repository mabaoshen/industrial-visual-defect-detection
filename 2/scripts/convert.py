import json
import os
from pathlib import Path


def convert_labelme_to_yolo(json_dir, label_dst_dir):
    """
    只转换标注文件（不复制图片）
    json_dir: JSON标注文件所在目录
    label_dst_dir: YOLO格式标注文件的保存目录
    """
    os.makedirs(label_dst_dir, exist_ok=True)  # 确保输出目录存在
    class_id = 0  # 裂缝类别固定为0

    for json_file in Path(json_dir).glob("*.json"):
        try:
            # 读取JSON内容
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 提取图像尺寸（用于坐标归一化）
            img_width = data["imageWidth"]
            img_height = data["imageHeight"]

            # 生成YOLO格式标注文件（与JSON同名，后缀为txt）
            label_txt_path = os.path.join(label_dst_dir, f"{json_file.stem}.txt")

            with open(label_txt_path, 'w', encoding='utf-8') as f:
                for shape in data["shapes"]:
                    if shape["label"] == "liefeng":  # 只处理裂缝标注
                        # 顶点坐标归一化（转换为YOLO要求的0-1范围）
                        points = shape["points"]
                        normalized = []
                        for x, y in points:
                            normalized.append(x / img_width)  # x归一化
                            normalized.append(y / img_height)  # y归一化
                        # 写入标注：class_id + 归一化坐标
                        f.write(f"{class_id} {' '.join(map(str, normalized))}\n")

            print(f"已生成标注：{label_txt_path}")

        except Exception as e:
            print(f"处理{json_file.name}出错：{str(e)}")


# --------------------------
# 配置路径（相对路径）
# --------------------------
# 训练集JSON和输出标注目录
TRAIN_JSON_DIR = "../scripts/temp_annotations/train"
TRAIN_LABEL_DST = "../dataset/labels/train"

# 验证集JSON和输出标注目录
VAL_JSON_DIR = "../scripts/temp_annotations/val"
VAL_LABEL_DST = "../dataset/labels/val"

# 执行转换
convert_labelme_to_yolo(TRAIN_JSON_DIR, TRAIN_LABEL_DST)
convert_labelme_to_yolo(VAL_JSON_DIR, VAL_LABEL_DST)