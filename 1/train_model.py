import os
import tensorflow as tf
import numpy as np
from keras import layers, Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.applications import MobileNetV2
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

# 核心配置 - 576x576确保所有尺寸兼容性
# 576能被16（MobileNetV2下采样倍数）整除：576÷16=36
# 36能被所有空洞卷积膨胀率整除：36÷1=36, 36÷6=6, 36÷12=3, 36÷18=2
TARGET_SIZE = (576, 576)
BATCH_SIZE = 1
EPOCHS = 30
NUM_CLASSES = 1
TRAIN_DIR = "D:/pythonCode/1/dataset/train"
VAL_DIR = "D:/pythonCode/1/dataset/val"
MODEL_SAVE_PATH = "D:/pythonCode/1/best_model.h5"
LOCAL_WEIGHTS_PATH = "D:/pythonCode/1/weights/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.5_128_no_top.h5"


# 损失函数定义
def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        focal_weight = alpha * tf.math.pow(1 - y_pred, gamma) * y_true + \
                       (1 - alpha) * tf.math.pow(y_pred, gamma) * (1 - y_true)
        return tf.reduce_mean(focal_weight * cross_entropy)

    return loss


def dice_loss(y_true, y_pred):
    smooth = 1e-6
    y_true = tf.clip_by_value(y_true, 0.0, 1.0)
    y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return 1 - (2 * intersection + smooth) / (union + smooth)


def combined_loss(y_true, y_pred):
    return 0.7 * focal_loss()(y_true, y_pred) + 0.3 * dice_loss(y_true, y_pred)


# 构建模型（修复所有尺寸问题）
def build_deeplabv3_plus(input_shape):
    if not os.path.exists(LOCAL_WEIGHTS_PATH):
        raise FileNotFoundError(f"本地权重文件未找到：{LOCAL_WEIGHTS_PATH}")

    # 骨干网络
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=None,
        alpha=0.5
    )
    base_model.load_weights(LOCAL_WEIGHTS_PATH)
    base_model.trainable = False

    # 获取高级特征（下采样16倍）
    high_level_feat = base_model.get_layer("block_13_expand_relu").output
    target_h, target_w = high_level_feat.shape[1], high_level_feat.shape[2]  # 应为36,36

    # ASPP分支 - 确保所有输出尺寸一致
    def aspp_branch(x, filters, rate):
        x = layers.Conv2D(filters, 3, dilation_rate=rate, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        # 强制尺寸匹配
        if x.shape[1] != target_h or x.shape[2] != target_w:
            x = layers.Resizing(target_h, target_w, interpolation="bilinear")(x)
        return x

    # 生成ASPP各分支
    aspp1 = aspp_branch(high_level_feat, 256, rate=1)
    aspp6 = aspp_branch(high_level_feat, 256, rate=6)
    aspp12 = aspp_branch(high_level_feat, 256, rate=12)
    aspp18 = aspp_branch(high_level_feat, 256, rate=18)

    # 全局池化分支（确保尺寸匹配）
    global_pool = layers.GlobalAveragePooling2D()(high_level_feat)
    global_pool = layers.Reshape((1, 1, high_level_feat.shape[-1]))(global_pool)
    global_pool = layers.Conv2D(256, 1, padding="same")(global_pool)
    global_pool = layers.BatchNormalization()(global_pool)
    global_pool = layers.Activation("relu")(global_pool)
    global_pool = layers.Resizing(target_h, target_w, interpolation="bilinear")(global_pool)

    # ASPP特征拼接（所有输入尺寸均为36x36）
    x = layers.Concatenate()([aspp1, aspp6, aspp12, aspp18, global_pool])
    x = layers.Conv2D(256, 1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # 低级特征处理
    low_level_feat = base_model.get_layer("block_3_expand_relu").output  # 下采样4倍，尺寸应为144x144
    low_level_feat = layers.Conv2D(48, 1, padding="same")(low_level_feat)
    low_level_feat = layers.BatchNormalization()(low_level_feat)
    low_level_feat = layers.Activation("relu")(low_level_feat)

    # 上采样高级特征以匹配低级特征尺寸（36x36 -> 144x144，上采样4倍）
    upsample_factor = (low_level_feat.shape[1] // x.shape[1], low_level_feat.shape[2] // x.shape[2])
    x = layers.UpSampling2D(size=upsample_factor, interpolation="bilinear")(x)

    # 确保拼接前尺寸完全匹配
    if x.shape[1] != low_level_feat.shape[1] or x.shape[2] != low_level_feat.shape[2]:
        x = layers.Resizing(low_level_feat.shape[1], low_level_feat.shape[2])(x)

    # 拼接高低级特征
    x = layers.Concatenate()([x, low_level_feat])

    # 最终处理与上采样
    x = layers.Conv2D(256, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(256, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # 上采样到输入尺寸（144x144 -> 576x576，上采样4倍）
    final_upsample_factor = (input_shape[0] // x.shape[1], input_shape[1] // x.shape[2])
    x = layers.UpSampling2D(size=final_upsample_factor, interpolation="bilinear")(x)

    # 输出层
    outputs = layers.Conv2D(1, 1, activation="sigmoid")(x)
    return Model(inputs=base_model.input, outputs=outputs)


# 数据加载与预处理
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
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

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
    print(f"[{time.strftime('%H:%M:%S')}] 启动训练流程")
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    input_shape = (*TARGET_SIZE, 3)

    try:
        # 构建模型
        print(f"[{time.strftime('%H:%M:%S')}] 构建模型...")
        model = build_deeplabv3_plus(input_shape)

        # 编译模型
        print(f"[{time.strftime('%H:%M:%S')}] 编译模型...")
        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss=combined_loss,
            metrics=["accuracy"]
        )

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

        print(f"[{time.strftime('%H:%M:%S')}] 训练完成，模型保存至: {MODEL_SAVE_PATH}")
        return history

    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] 训练失败：{str(e)}")
        raise
    finally:
        print(f"[{time.strftime('%H:%M:%S')}] 训练流程结束")


if __name__ == "__main__":
    train_model()
