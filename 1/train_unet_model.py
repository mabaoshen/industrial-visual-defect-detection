import os
import tensorflow as tf
import numpy as np
from keras import layers, Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
import time
import glob


# 固定随机种子，确保结果可复现
def set_random_seeds(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


set_random_seeds()

# 环境配置
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 使用CPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

# 核心配置 - 确保尺寸能被2^5=32整除，因为U-Net有5次下采样
# 320是32的倍数，同时保持较小的模型大小
TARGET_SIZE = (320, 320)  # 修改为更适合U-Net的尺寸
BATCH_SIZE = 1
EPOCHS = 30
NUM_CLASSES = 1
TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"
MODEL_SAVE_PATH = "best_unet_model.h5"


# 从train_model.py复制相同的损失函数定义
def enhanced_focal_loss(alpha=0.75, gamma=2.0, small_target_weight=2.0):
    """改进的Focal Loss，增加对小目标的关注度"""
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        # 标准交叉熵
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        
        # Focal权重
        focal_weight = alpha * tf.math.pow(1 - y_pred, gamma) * y_true + \
                       (1 - alpha) * tf.math.pow(y_pred, gamma) * (1 - y_true)
        
        # 计算目标区域大小并为小目标添加额外权重
        # 计算每个样本中目标区域的比例
        target_area = tf.reduce_sum(y_true, axis=[1, 2, 3], keepdims=True)
        total_area = tf.cast(tf.reduce_prod(y_true.shape[1:3]), dtype=tf.float32)
        target_ratio = target_area / (total_area + 1e-7)
        
        # 对小目标区域应用额外权重
        # 当目标占比小于10%时，给予额外权重
        small_target_mask = tf.cast(target_ratio < 0.1, dtype=tf.float32)
        area_based_weight = 1.0 + (small_target_weight - 1.0) * small_target_mask
        
        # 组合权重
        combined_weight = focal_weight * area_based_weight
        
        return tf.reduce_mean(combined_weight * cross_entropy)

    return loss


def weighted_dice_loss(small_target_weight=3.0):
    """加权Dice Loss，对小目标区域进行加权"""
    def loss(y_true, y_pred):
        smooth = 1e-6
        y_true = tf.clip_by_value(y_true, 0.0, 1.0)
        y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
        
        # 计算目标区域大小
        target_area = tf.reduce_sum(y_true, axis=[1, 2, 3], keepdims=True)
        total_area = tf.cast(tf.reduce_prod(y_true.shape[1:3]), dtype=tf.float32)
        target_ratio = target_area / (total_area + 1e-7)
        
        # 根据目标大小计算权重
        # 目标越小，权重越大
        weight = 1.0 + (small_target_weight - 1.0) * tf.exp(-10.0 * target_ratio)
        
        # 计算加权的交集和并集
        intersection = tf.reduce_sum(weight * y_true * y_pred)
        union = tf.reduce_sum(weight * y_true) + tf.reduce_sum(weight * y_pred)
        
        return 1 - (2 * intersection + smooth) / (union + smooth)

    return loss


def small_target_combined_loss(y_true, y_pred):
    """简化高效的小目标损失函数组合"""
    # 计算目标区域比例，用于确定权重
    target_area = tf.reduce_sum(y_true, axis=[1, 2, 3])
    total_area = tf.cast(y_true.shape[1] * y_true.shape[2], dtype=tf.float32)
    target_ratio = target_area / (total_area + 1e-7)
    
    # 对于小目标(占比小于5%)，增加Dice Loss的权重
    # 对于大目标，增加Focal Loss的权重
    is_small = tf.cast(target_ratio < 0.05, dtype=tf.float32)
    
    # 动态调整权重
    # 小目标: 0.3*Focal + 0.7*Dice
    # 大目标: 0.6*Focal + 0.4*Dice
    focal_weight = 0.3 * is_small + 0.6 * (1 - is_small)
    dice_weight = 0.7 * is_small + 0.4 * (1 - is_small)
    
    # 计算两种主要损失
    enhanced_focal = enhanced_focal_loss(alpha=0.75, gamma=2.0, small_target_weight=2.0)(y_true, y_pred)
    weighted_dice = weighted_dice_loss(small_target_weight=2.5)(y_true, y_pred)
    
    # 组合损失并确保返回标量
    combined_loss = focal_weight * enhanced_focal + dice_weight * weighted_dice
    
    return tf.reduce_mean(combined_loss)


# 构建U-Net模型
def build_unet(input_shape):
    """构建U-Net架构模型，适合小目标分割"""
    inputs = layers.Input(shape=input_shape)
    
    # 下采样路径（编码器）
    # 第一层 - 输入尺寸: (320, 320, 3) -> (320, 320, 64)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    c1 = layers.BatchNormalization()(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)  # (160, 160, 64)
    
    # 第二层
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    c2 = layers.BatchNormalization()(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)  # (80, 80, 128)
    
    # 第三层
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    c3 = layers.BatchNormalization()(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)  # (40, 40, 256)
    
    # 第四层
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    c4 = layers.BatchNormalization()(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)  # (20, 20, 512)
    
    # 第五层（瓶颈层）
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)
    c5 = layers.BatchNormalization()(c5)
    
    # 上采样路径（解码器）
    # 第六层
    u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)  # (40, 40, 512)
    # 跳跃连接 - 从c4层获取特征
    u6 = layers.concatenate([u6, c4], axis=3)  # (40, 40, 1024)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)
    c6 = layers.BatchNormalization()(c6)
    
    # 第七层
    u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)  # (80, 80, 256)
    u7 = layers.concatenate([u7, c3], axis=3)  # (80, 80, 512)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.BatchNormalization()(c7)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)
    c7 = layers.BatchNormalization()(c7)
    
    # 第八层
    u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)  # (160, 160, 128)
    u8 = layers.concatenate([u8, c2], axis=3)  # (160, 160, 256)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.BatchNormalization()(c8)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)
    c8 = layers.BatchNormalization()(c8)
    
    # 第九层
    u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)  # (320, 320, 64)
    u9 = layers.concatenate([u9, c1], axis=3)  # (320, 320, 128)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.BatchNormalization()(c9)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)
    c9 = layers.BatchNormalization()(c9)
    
    # 输出层 - 1个通道用于二值分割
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model


# 数据加载与预处理 - 与train_model.py基本相同，但修改了TARGET_SIZE
def load_image_paths(data_dir):
    image_dir = os.path.join(data_dir, "images", "defects")
    mask_dir = os.path.join(data_dir, "masks", "defects")

    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.*")))
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.*")))

    if len(image_paths) != len(mask_paths):
        raise ValueError(f"图像和掩码数量不匹配: {len(image_paths)} vs {len(mask_paths)}")

    return list(zip(image_paths, mask_paths))


def create_augmentation_layers():
    rotate_layer = layers.RandomRotation(
        factor=15 / 360,
        fill_mode="constant",
        fill_value=0.0,
        interpolation="bilinear",
        seed=42
    )
    flip_layer = layers.RandomFlip(mode="horizontal", seed=42)
    return rotate_layer, flip_layer


def load_and_preprocess(image_path, mask_path, augment=False):
    # 加载图像
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, TARGET_SIZE)
    # 使用简单的归一化而非MobileNetV2预处理
    image = tf.cast(image, tf.float32) / 255.0

    # 加载掩码
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_image(mask, channels=1, expand_animations=False)
    mask = tf.image.resize(mask, TARGET_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    mask = tf.cast(mask, tf.float32) / 255.0
    mask = tf.clip_by_value(mask, 0, 1)

    # 数据增强
    if augment:
        rotate_layer, flip_layer = create_augmentation_layers()
        image = tf.expand_dims(image, 0)
        mask = tf.expand_dims(mask, 0)

        # 随机旋转
        if tf.random.uniform(()) > 0.5:
            seed = tf.random.uniform(shape=[2], minval=0, maxval=1000, dtype=tf.int32)
            tf.random.set_seed(seed[0])
            image = rotate_layer(image)
            tf.random.set_seed(seed[0])
            mask = layers.RandomRotation(
                factor=15 / 360,
                fill_mode="constant",
                fill_value=0.0,
                interpolation="nearest"
            )(mask)

        # 随机翻转
        if tf.random.uniform(()) > 0.5:
            seed = tf.random.uniform(shape=[2], minval=0, maxval=1000, dtype=tf.int32)
            tf.random.set_seed(seed[1])
            image = flip_layer(image)
            tf.random.set_seed(seed[1])
            mask = flip_layer(mask)

        image = tf.squeeze(image, 0)
        mask = tf.squeeze(mask, 0)

    return image, mask


def create_data_loaders():
    train_pairs = load_image_paths(TRAIN_DIR)
    val_pairs = load_image_paths(VAL_DIR)

    if not train_pairs:
        raise ValueError("训练集数据为空")
    if not val_pairs:
        raise ValueError("验证集数据为空")

    print(f"✅ 数据加载成功：训练集{len(train_pairs)}张，验证集{len(val_pairs)}张")

    # 训练数据集
    train_dataset = tf.data.Dataset.from_tensor_slices(train_pairs)
    train_dataset = train_dataset.shuffle(len(train_pairs), seed=42)
    train_dataset = train_dataset.map(
        lambda pair: load_and_preprocess(pair[0], pair[1], augment=True),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # 验证数据集
    val_dataset = tf.data.Dataset.from_tensor_slices(val_pairs)
    val_dataset = val_dataset.map(
        lambda pair: load_and_preprocess(pair[0], pair[1], augment=False),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset, len(train_pairs) // BATCH_SIZE, len(val_pairs) // BATCH_SIZE


# 训练主函数
def train_model():
    print(f"[{time.strftime('%H:%M:%S')}] 启动U-Net训练流程")
    # 确保模型保存路径存在
    model_dir = os.path.dirname(MODEL_SAVE_PATH)
    if model_dir:  # 只有当目录不为空时才创建目录
        os.makedirs(model_dir, exist_ok=True)
    input_shape = (*TARGET_SIZE, 3)

    try:
        # 构建U-Net模型
        print(f"[{time.strftime('%H:%M:%S')}] 构建U-Net模型...")
        model = build_unet(input_shape)
        
        # 显示模型结构摘要
        model.summary()

        # 编译模型 - 使用专为小目标设计的组合损失函数
        print(f"[{time.strftime('%H:%M:%S')}] 编译模型...")
        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss=small_target_combined_loss,
            metrics=["accuracy"]
        )
        
        # 打印信息说明使用了新的损失函数
        print("\n使用专为小目标设计的组合损失函数: small_target_combined_loss")
        print("包含: 增强版Focal Loss和加权Dice Loss，根据目标大小动态调整权重")

        # 回调函数
        callbacks = [
            ModelCheckpoint(
                MODEL_SAVE_PATH,
                monitor="val_loss",
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor="val_loss",
                patience=20,
                min_delta=1e-4,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]

        # 加载数据
        print(f"[{time.strftime('%H:%M:%S')}] 加载数据...")
        train_dataset, val_dataset, train_steps, val_steps = create_data_loaders()

        # 测试数据加载
        for test_img, test_mask in train_dataset.take(1):
            print(
                f"[{time.strftime('%H:%M:%S')}] 数据测试通过 - 图像形状: {test_img.shape}, 掩码形状: {test_mask.shape}")

        # 开始训练
        print(f"[{time.strftime('%H:%M:%S')}] 开始训练...")
        history = model.fit(
            train_dataset,
            steps_per_epoch=train_steps,
            validation_data=val_dataset,
            validation_steps=val_steps,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1
        )

        print(f"[{time.strftime('%H:%M:%S')}] 训练完成，U-Net模型保存至: {MODEL_SAVE_PATH}")
        return history

    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] 训练失败：{str(e)}")
        raise
    finally:
        print(f"[{time.strftime('%H:%M:%S')}] 训练流程结束")


if __name__ == "__main__":
    train_model()
