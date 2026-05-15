from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# =========================
# CONFIG
# =========================

Path("results").mkdir(parents=True, exist_ok=True)
Path("results/logs").mkdir(parents=True, exist_ok=True)

DATASET_DIR = Path("dataset")

IMG_SIZE = 256
IMG_CHANNELS = 1
NUM_MASK_CHANNELS = 3

BATCH_SIZE = 128
EPOCHS = 200
LEARNING_RATE = 1e-4

# Loss weight cho A, B, C
# C cao hơn vì vùng overlap thường nhỏ và khó học hơn
CHANNEL_WEIGHTS = tf.constant([1.0, 1.0, 3.0], dtype=tf.float32)

AUTOTUNE = tf.data.AUTOTUNE


# =========================
# DATA LOADER
# =========================

def get_file_lists(split):
    """
    Trả về danh sách path:
    image_paths, mask_A_paths, mask_B_paths, mask_C_paths
    """
    split_dir = DATASET_DIR / split

    image_dir = split_dir / "images"
    mask_A_dir = split_dir / "masks_A"
    mask_B_dir = split_dir / "masks_B"
    mask_C_dir = split_dir / "masks_C"

    image_paths = sorted(image_dir.glob("*.png"))

    final_image_paths = []
    final_mask_A_paths = []
    final_mask_B_paths = []
    final_mask_C_paths = []

    for img_path in image_paths:
        name = img_path.name

        mask_A_path = mask_A_dir / name
        mask_B_path = mask_B_dir / name
        mask_C_path = mask_C_dir / name

        if mask_A_path.exists() and mask_B_path.exists() and mask_C_path.exists():
            final_image_paths.append(str(img_path))
            final_mask_A_paths.append(str(mask_A_path))
            final_mask_B_paths.append(str(mask_B_path))
            final_mask_C_paths.append(str(mask_C_path))
        else:
            print(f"Missing mask for {name}, skipped.")

    return final_image_paths, final_mask_A_paths, final_mask_B_paths, final_mask_C_paths


def read_image(path):
    """
    Đọc image grayscale PNG.
    Output shape: 256 x 256 x 1
    Pixel: 0-1
    """
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)  # 0-255 -> 0-1
    img.set_shape([IMG_SIZE, IMG_SIZE, 1])
    return img


def read_mask(path):
    """
    Đọc mask grayscale PNG.
    Output shape: 256 x 256 x 1
    Pixel: 0 hoặc 1
    """
    mask = tf.io.read_file(path)
    mask = tf.image.decode_png(mask, channels=1)

    # Mask gốc thường là 0 hoặc 255.
    # Chuyển thành 0 hoặc 1.
    mask = tf.cast(mask > 127, tf.float32)
    mask.set_shape([IMG_SIZE, IMG_SIZE, 1])
    return mask


def load_sample(image_path, mask_A_path, mask_B_path, mask_C_path):
    image = read_image(image_path)

    mask_A = read_mask(mask_A_path)
    mask_B = read_mask(mask_B_path)
    mask_C = read_mask(mask_C_path)

    # Stack thành output 3 channel: A, B, C
    mask = tf.concat([mask_A, mask_B, mask_C], axis=-1)
    mask.set_shape([IMG_SIZE, IMG_SIZE, 3])

    return image, mask


def make_dataset(split, batch_size=16, shuffle=False):
    image_paths, mask_A_paths, mask_B_paths, mask_C_paths = get_file_lists(split)

    print(f"{split}: {len(image_paths)} samples")

    ds = tf.data.Dataset.from_tensor_slices(
        (image_paths, mask_A_paths, mask_B_paths, mask_C_paths)
    )

    if shuffle:
        ds = ds.shuffle(buffer_size=len(image_paths), reshuffle_each_iteration=True)

    ds = ds.map(load_sample, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)

    return ds


# =========================
# MODEL: U-NET
# =========================

def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, 3, padding="same", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    return x


def encoder_block(x, filters):
    skip = conv_block(x, filters)
    pooled = layers.MaxPooling2D(pool_size=(2, 2))(skip)
    return skip, pooled


def decoder_block(x, skip, filters):
    x = layers.Conv2DTranspose(filters, 2, strides=2, padding="same")(x)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, filters)
    return x


def build_unet(input_shape=(256, 256, 1), output_channels=3):
    inputs = keras.Input(shape=input_shape)

    # Encoder
    s1, p1 = encoder_block(inputs, 32)
    s2, p2 = encoder_block(p1, 64)
    s3, p3 = encoder_block(p2, 128)
    s4, p4 = encoder_block(p3, 256)

    # Bridge
    b1 = conv_block(p4, 512)

    # Decoder
    d1 = decoder_block(b1, s4, 256)
    d2 = decoder_block(d1, s3, 128)
    d3 = decoder_block(d2, s2, 64)
    d4 = decoder_block(d3, s1, 32)

    # Vì output là multi-label 3 channel:
    # A, B, C có thể cùng = 1 tại overlap.
    # Nên dùng sigmoid, không dùng softmax.
    outputs = layers.Conv2D(output_channels, 1, padding="same", activation="sigmoid", dtype="float32")(d4)

    model = keras.Model(inputs, outputs, name="UNet_Chromosome_Segmentation")
    return model


# =========================
# LOSS + METRICS
# =========================

def weighted_bce(y_true, y_pred):
    """
    BCE có weight riêng cho A, B, C.
    C được weight cao hơn.
    """
    bce = keras.backend.binary_crossentropy(y_true, y_pred)

    # bce shape: batch, h, w, channels
    weighted = bce * CHANNEL_WEIGHTS
    return tf.reduce_mean(weighted)


def dice_loss(y_true, y_pred, smooth=1e-6):
    """
    Dice loss trung bình 3 channel.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
    denominator = tf.reduce_sum(y_true + y_pred, axis=[1, 2])

    dice = (2.0 * intersection + smooth) / (denominator + smooth)

    # Apply channel weights
    weighted_dice = dice * CHANNEL_WEIGHTS
    weighted_dice = tf.reduce_sum(weighted_dice, axis=-1) / tf.reduce_sum(CHANNEL_WEIGHTS)

    return 1.0 - tf.reduce_mean(weighted_dice)


def bce_dice_loss(y_true, y_pred):
    return weighted_bce(y_true, y_pred) + dice_loss(y_true, y_pred)


def dice_channel(index, name):
    def metric(y_true, y_pred):
        smooth = 1e-6

        yt = y_true[..., index]
        yp = y_pred[..., index]

        # Threshold để đo metric dễ hiểu hơn
        yp = tf.cast(yp > 0.5, tf.float32)

        intersection = tf.reduce_sum(yt * yp, axis=[1, 2])
        denominator = tf.reduce_sum(yt + yp, axis=[1, 2])

        dice = (2.0 * intersection + smooth) / (denominator + smooth)
        return tf.reduce_mean(dice)

    metric.__name__ = name
    return metric


def iou_channel(index, name):
    def metric(y_true, y_pred):
        smooth = 1e-6

        yt = y_true[..., index]
        yp = y_pred[..., index]

        yp = tf.cast(yp > 0.5, tf.float32)

        intersection = tf.reduce_sum(yt * yp, axis=[1, 2])
        union = tf.reduce_sum(yt + yp, axis=[1, 2]) - intersection

        iou = (intersection + smooth) / (union + smooth)
        return tf.reduce_mean(iou)

    metric.__name__ = name
    return metric


# =========================
# TRAIN
# =========================

def main():
    train_ds = make_dataset("train", batch_size=BATCH_SIZE, shuffle=True)
    val_ds = make_dataset("val", batch_size=BATCH_SIZE, shuffle=False)
    test_ds = make_dataset("test", batch_size=BATCH_SIZE, shuffle=False)

    model = build_unet(
        input_shape=(IMG_SIZE, IMG_SIZE, IMG_CHANNELS),
        output_channels=NUM_MASK_CHANNELS
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=bce_dice_loss,
        metrics=[
            dice_channel(0, "dice_A"),
            dice_channel(1, "dice_B"),
            dice_channel(2, "dice_C"),
            iou_channel(0, "iou_A"),
            iou_channel(1, "iou_B"),
            iou_channel(2, "iou_C"),
        ]
    )

    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="results/best_unet.keras",
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.CSVLogger(
            filename="results/logs/train_log.csv"
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    print("Evaluating on test set...")
    test_results = model.evaluate(test_ds, return_dict=True)
    print(test_results)

    model.save("results/final_unet.keras")
    print("Saved final model to results/final_unet.keras")


if __name__ == "__main__":
    main()