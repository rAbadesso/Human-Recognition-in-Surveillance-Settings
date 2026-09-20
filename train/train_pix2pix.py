import os
import argparse

# ==========================================
# 1. ARGUMENT PARSING & GPU SELECTION
# ==========================================
# We parse arguments first to set the CUDA visibility BEFORE TensorFlow initializes.
parser = argparse.ArgumentParser(description="Pix2Pix Multi-GPU Training")
parser.add_argument('--gpus', type=str, default=None,
                    help='Comma-separated GPUs to use (e.g., "0", "0,1,2"). Leave empty to use all.')
parser.add_argument('--resume', action='store_true', help='Flag to resume training from the latest checkpoint.')
parser.add_argument('--epochs', type=int, default=150, help='Total number of epochs to train.')
parser.add_argument('--ckpt_dir', type=str, default='training_checkpoints', help='Directory to save/load checkpoints.')
parser.add_argument('--log_dir', type=str, default='logs/fit', help='Directory to save TensorBoard logs.')
args = parser.parse_args()

if args.gpus is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)

# Now we can safely import TensorFlow
import glob
import time
import datetime
import tensorflow as tf

# ==========================================
# 2. HYPERPARAMETERS & CONFIG (Globals)
# ==========================================
BUFFER_SIZE = 400
BATCH_SIZE_PER_REPLICA = 1  # Pix2Pix Instance Norm requirement
IMG_WIDTH = 256
IMG_HEIGHT = 256
OUTPUT_CHANNELS = 3
LAMBDA_L1 = 100
LAMBDA_SSIM = 50

DATA_DIR = "../Dataset_img"


# ==========================================
# 3. DATASET & AUGMENTATION FUNCTIONS
# ==========================================
def get_paired_dataset_paths(base_dir):
    input_paths = []
    target_paths = []

    for u_dir in glob.glob(os.path.join(base_dir, "*_U_*")):
        dir_name = os.path.basename(u_dir)
        parts = dir_name.split('_')
        person_id, session_id = parts[0], parts[2]

        e_dir_name = f"{person_id}_E_{session_id}"
        e_dir = os.path.join(base_dir, e_dir_name)

        for u_img_path in glob.glob(os.path.join(u_dir, "*.jpg")):
            frame_idx = os.path.basename(u_img_path).split('_')[-1]
            e_img_name = f"{person_id}_E_{session_id}_{frame_idx}"
            e_img_path = os.path.join(e_dir, e_img_name)

            if os.path.exists(e_img_path):
                input_paths.append(u_img_path)
                target_paths.append(e_img_path)

    return input_paths, target_paths


def load_images(input_img_path, target_img_path):
    input_img = tf.cast(tf.io.decode_jpeg(tf.io.read_file(input_img_path), channels=3), tf.float32)
    target_img = tf.cast(tf.io.decode_jpeg(tf.io.read_file(target_img_path), channels=3), tf.float32)
    return input_img, target_img


@tf.function
def random_jitter(input_image, target_image):
    input_image = tf.image.resize(input_image, [286, 286], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    target_image = tf.image.resize(target_image, [286, 286], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    stacked_image = tf.stack([input_image, target_image], axis=0)
    cropped_image = tf.image.random_crop(stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])
    input_image, target_image = cropped_image[0], cropped_image[1]

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        target_image = tf.image.flip_left_right(target_image)

    input_image = tf.image.random_brightness(input_image, max_delta=0.2)
    input_image = tf.image.random_contrast(input_image, lower=0.8, upper=1.2)
    input_image = tf.clip_by_value(input_image, 0.0, 255.0)

    return input_image, target_image


def normalize(input_image, target_image):
    return (input_image / 127.5) - 1, (target_image / 127.5) - 1


def load_image_train(input_path, target_path):
    input_image, target_image = load_images(input_path, target_path)
    input_image, target_image = random_jitter(input_image, target_image)
    return normalize(input_image, target_image)


def load_image_val(input_path, target_path):
    input_image, target_image = load_images(input_path, target_path)
    input_image = tf.image.resize(input_image, [IMG_HEIGHT, IMG_WIDTH])
    target_image = tf.image.resize(target_image, [IMG_HEIGHT, IMG_WIDTH])
    return normalize(input_image, target_image)


# ==========================================
# 4. MODEL ARCHITECTURE FUNCTIONS
# ==========================================
def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                      kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                                               kernel_initializer=initializer, use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result


def Generator():
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])
    down_stack = [
        downsample(64, 4, apply_batchnorm=False),
        downsample(128, 4), downsample(256, 4), downsample(512, 4),
        downsample(512, 4), downsample(512, 4), downsample(512, 4), downsample(512, 4),
    ]
    up_stack = [
        upsample(512, 4, apply_dropout=True), upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True), upsample(512, 4),
        upsample(256, 4), upsample(128, 4), upsample(64, 4),
    ]
    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4, strides=2, padding='same',
                                           kernel_initializer=initializer, activation='tanh')
    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])
    return tf.keras.Model(inputs=inputs, outputs=last(x))


def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)
    inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')
    x = tf.keras.layers.concatenate([inp, tar])

    down1 = downsample(64, 4, False)(x)
    down2 = downsample(128, 4)(down1)
    down3 = downsample(256, 4)(down2)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)(zero_pad1)
    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)

    last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)
    return tf.keras.Model(inputs=[inp, tar], outputs=last)


# ==========================================
# 5. EXECUTION & DISTRIBUTION LOGIC
# ==========================================
if __name__ == '__main__':
    print(f"\n--> Looking for dataset in: {DATA_DIR}")
    input_paths, target_paths = get_paired_dataset_paths(DATA_DIR)

    if not input_paths:
        raise ValueError(f"No images found! Check if '{DATA_DIR}' exists and is accessible.")

    # Train/Validation Split
    split_idx = int(len(input_paths) * 0.9)
    train_input_paths, val_input_paths = input_paths[:split_idx], input_paths[split_idx:]
    train_target_paths, val_target_paths = target_paths[:split_idx], target_paths[split_idx:]

    # ---------------------------------------------
    # Setup Distributed Strategy
    # ---------------------------------------------
    strategy = tf.distribute.MirroredStrategy()
    print(f"--> Strategy Initialized. Synchronizing across {strategy.num_replicas_in_sync} GPU(s).")

    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

    # Prepare and distribute datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((train_input_paths, train_target_paths))
    train_dataset = train_dataset.map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE)
    train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)

    val_dataset = tf.data.Dataset.from_tensor_slices((val_input_paths, val_target_paths))
    val_dataset = val_dataset.map(load_image_val, num_parallel_calls=tf.data.AUTOTUNE).batch(GLOBAL_BATCH_SIZE)
    val_dist_dataset = strategy.experimental_distribute_dataset(val_dataset)

    # ---------------------------------------------
    # Build Models & Losses inside Strategy Scope
    # ---------------------------------------------
    with strategy.scope():
        generator = Generator()
        discriminator = Discriminator()

        # Proper distributed loss configuration
        loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)


        def compute_gan_loss(labels, predictions):
            per_example_loss = loss_object(labels, predictions)
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)


        def compute_l1_loss(target, gen_output):
            per_example_loss = tf.reduce_mean(tf.abs(target - gen_output), axis=[1, 2, 3])
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)


        def compute_ssim_loss(target, gen_output):
            target_denorm = (target + 1.0) / 2.0
            gen_denorm = (gen_output + 1.0) / 2.0
            ssim_val = tf.image.ssim(target_denorm, gen_denorm, max_val=1.0)
            per_example_loss = 1.0 - ssim_val
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)


        def generator_loss(disc_generated_output, gen_output, target):
            gan_loss = compute_gan_loss(tf.ones_like(disc_generated_output), disc_generated_output)
            l1_loss = compute_l1_loss(target, gen_output)
            ssim_loss = compute_ssim_loss(target, gen_output)
            total_gen_loss = gan_loss + (LAMBDA_L1 * l1_loss) + (LAMBDA_SSIM * ssim_loss)
            return total_gen_loss, gan_loss, l1_loss, ssim_loss


        def discriminator_loss(disc_real_output, disc_generated_output):
            real_loss = compute_gan_loss(tf.ones_like(disc_real_output), disc_real_output)
            generated_loss = compute_gan_loss(tf.zeros_like(disc_generated_output), disc_generated_output)
            return real_loss + generated_loss


        # Learning Rate Schedule & Optimizers
        steps_per_epoch = split_idx // GLOBAL_BATCH_SIZE
        boundaries = [100 * steps_per_epoch]
        values = [2e-4, 2e-5]
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)

        generator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.5)
        discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.5)

        # Stateful Checkpointing
        ckpt_epoch = tf.Variable(0, dtype=tf.int64)
        ckpt_step = tf.Variable(0, dtype=tf.int64)

        checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                         discriminator_optimizer=discriminator_optimizer,
                                         generator=generator,
                                         discriminator=discriminator,
                                         epoch=ckpt_epoch,
                                         step=ckpt_step)

        os.makedirs(args.ckpt_dir, exist_ok=True)
        checkpoint_prefix = os.path.join(args.ckpt_dir, "ckpt")

        if args.resume:
            latest_ckpt = tf.train.latest_checkpoint(args.ckpt_dir)
            if latest_ckpt:
                checkpoint.restore(latest_ckpt)
                print(f"--> Successfully resumed from Epoch {ckpt_epoch.numpy() + 1}, Step {ckpt_step.numpy()}")
            else:
                print("--> No checkpoint found. Starting from scratch.")
        else:
            print("--> Starting training from scratch (Epoch 1).")

    # ---------------------------------------------
    # Define Distributed Training Steps
    # ---------------------------------------------
    log_dir_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_summary_writer = tf.summary.create_file_writer(os.path.join(args.log_dir, log_dir_stamp, 'train'))
    val_summary_writer = tf.summary.create_file_writer(os.path.join(args.log_dir, log_dir_stamp, 'val'))


    @tf.function
    def distributed_train_step(input_image, target, step):
        def train_step(input_image, target):
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                gen_output = generator(input_image, training=True)
                disc_real_output = discriminator([input_image, target], training=True)
                disc_generated_output = discriminator([input_image, gen_output], training=True)

                gen_total_loss, gen_gan_loss, gen_l1_loss, gen_ssim_loss = generator_loss(disc_generated_output,
                                                                                          gen_output, target)
                disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

            generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
            discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

            generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

            return gen_total_loss, gen_l1_loss, gen_ssim_loss, disc_loss

        # Run on replicas
        per_replica_losses = strategy.run(train_step, args=(input_image, target))

        # Reduce losses to scalars for TensorBoard
        gen_total_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses[0], axis=None)
        gen_l1_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses[1], axis=None)
        gen_ssim_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses[2], axis=None)
        disc_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses[3], axis=None)

        with train_summary_writer.as_default():
            tf.summary.scalar('total_loss', gen_total_loss, step=step)
            tf.summary.scalar('L1_loss', gen_l1_loss, step=step)
            tf.summary.scalar('SSIM_loss', gen_ssim_loss, step=step)
            tf.summary.scalar('disc_loss', disc_loss, step=step)
            tf.summary.scalar('learning_rate', generator_optimizer.learning_rate, step=step)


    @tf.function
    def distributed_validate_step(val_dataset, epoch):
        def val_step(input_image, target):
            gen_output = generator(input_image, training=False)
            target_denorm = (target + 1.0) / 2.0
            gen_denorm = (gen_output + 1.0) / 2.0

            psnr = tf.reduce_mean(tf.image.psnr(target_denorm, gen_denorm, max_val=1.0))
            ssim = tf.reduce_mean(tf.image.ssim(target_denorm, gen_denorm, max_val=1.0))
            return psnr, ssim

        total_psnr = 0.0
        total_ssim = 0.0
        batches = 0.0

        for input_image, target in val_dataset:
            per_replica_psnr, per_replica_ssim = strategy.run(val_step, args=(input_image, target))

            psnr = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_psnr, axis=None)
            ssim = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_ssim, axis=None)

            total_psnr += psnr
            total_ssim += ssim
            batches += 1.0

        avg_psnr = total_psnr / batches
        avg_ssim = total_ssim / batches

        with val_summary_writer.as_default():
            tf.summary.scalar('PSNR', avg_psnr, step=epoch)
            tf.summary.scalar('SSIM', avg_ssim, step=epoch)

        return avg_psnr, avg_ssim


    # ---------------------------------------------
    # Execute Training Loop
    # ---------------------------------------------
    start_epoch = ckpt_epoch.numpy()
    print(f"\nTraining for {args.epochs} epochs... ({split_idx} Training Samples)")
    print(f"Global Batch Size: {GLOBAL_BATCH_SIZE}")

    for epoch in range(start_epoch, args.epochs):
        start = time.time()
        print(f"Epoch {epoch + 1}/{args.epochs}")

        for n, (input_image, target) in train_dist_dataset.enumerate():
            distributed_train_step(input_image, target, ckpt_step)
            ckpt_step.assign_add(1)

            if (n + 1) % 100 == 0:
                print(f"  Global Step {ckpt_step.numpy()}: Training in progress...")

        # Validate
        val_psnr, val_ssim = distributed_validate_step(val_dist_dataset, tf.cast(epoch, tf.int64))
        print(f"  Validation -> PSNR: {val_psnr:.2f} dB | SSIM: {val_ssim:.4f}")

        # State Saving
        ckpt_epoch.assign(epoch + 1)
        if (epoch + 1) % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
            print(f"  Checkpoint saved.")

        print(f"  Time taken: {time.time() - start:.2f} sec\n")

    checkpoint.save(file_prefix=checkpoint_prefix)
    print("Training complete and final checkpoint saved.")