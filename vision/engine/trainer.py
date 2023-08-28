
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import os, sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from vision.models.base_models import get_base_models
from vision.engine.build_models import unfreeze_model, building_models
from vision.core.utils import get_callbacks_list


def train(model_name, data, opt, loss, metrics, callbacks, imgs_aug, bs=32, ep=100, weights="imagenet", include_top=False):
    model_base = get_base_models(models=model_name, weights=weights, include_top=include_top)
    model_base = unfreeze_model(model=model_base, unfreeze=False)
    model_base = building_models(shape_input=data['shape'])
    # setup img aug
    aug = ImageDataGenerator(
        rotation_range=imgs_aug['rotation_range'],
        zoom_range=imgs_aug['zoom_range'],
        width_shift_range=imgs_aug['width_shift_range'],
        height_shift_range=imgs_aug['height_shift_range'],
        shear_range=imgs_aug['shear_range'],
        horizontal_flip=imgs_aug['horizontal_flip'],
        fill_mode=imgs_aug['fill_mode']
    )

    # setup callbacks
    print("Setup callbacks...")
    callbacks_list, path_save = get_callbacks_list(
        diractory=callbacks['diractory'],
        status_tensorboard=callbacks['status_tensorboard'],
        status_checkpoint=callbacks['status_checkpoint'],
        status_earlystop=callbacks['status_earlystop'],
        file_ckpt=callbacks['file_ckpt'],
        ckpt_monitor=callbacks['ckpt_monitor'],
        ckpt_mode=callbacks['ckpt_mode'],
        early_stop_monitor=callbacks['early_stop_monitor'],
        early_stop_mode=callbacks['early_stop_mode'],
        early_stop_patience=callbacks['early_stop_patience']
    )
    print("save callbacks to : ", path_save)
    print("[INFO] compiling model...")
    model_base.compile(loss=loss, optimizer=opt, metrics=metrics)

    # train the head of the network
    print("[INFO] training head...")
    history = model_base.fit(
        aug.flow(data['trainX'], data['trainY'], batch_size=bs),
        steps_per_epoch=len(data['trainX']) // bs,
        validation_data=(data['testX'], data['testY']),
        validation_steps=len(data['testY']) // bs,
        epochs=ep,
        shuffle=True,
        callbacks=callbacks_list
    )

    return history, model_base


def train_loop(model, x_train, y_train, x_val, y_val, save_dir):

    if not os.path.exists(save_dir, "weights"):
        os.mkdir(save_dir, "weights")
    
    if not os.path.exists(save_dir, "logs"):
        os.mkdir(save_dir, "logs")

    # Define loss function
    loss_fn = tf.keras.losses.categorical_crossentropy()

    # Define optimizer (Adam)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    # Define checkpoint callback
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(os.path.join(save_dir, 'weights'), "best_model_weights.h5"),
        save_best_only=True,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )

    # Define early stopping callback
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )

    # Define TensorBoard callback
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(save_dir, "logs"), histogram_freq=1)

    # Define data generators
    batch_size = 32
    train_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_generator = train_data_generator.flow(x_train, y_train, batch_size=batch_size, subset='training')
    validation_generator = train_data_generator.flow(x_val, y_val, batch_size=batch_size, subset='validation')

    # Set steps_per_epoch and validation_steps
    steps_per_epoch = len(train_generator) // batch_size
    validation_steps = len(validation_generator) // batch_size

    his_train = {
        'accuracy': [],
        'val_accuracy': [],
        'loss': [],
        'val_loss': []
    }

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        for batch_x, batch_y in train_generator:
            with tf.GradientTape() as tape:
                logits = model(batch_x, training=True)
                loss_value = loss_fn(batch_y, logits)

            gradients = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Calculate validation loss and accuracy at the end of each epoch
        val_loss, val_accuracy = model.evaluate(validation_generator, steps=validation_steps, verbose=0)
        train_loss, train_accuracy = model.evaluate(train_generator, steps=steps_per_epoch, verbose=0)
        his_train['accuracy'], his_train['loss'], his_train['val_accuracy'], his_train['val_loss'] = train_accuracy, train_loss, val_accuracy, val_loss

        print(f"Epoch {epoch}: "
            f"Train Loss = {train_loss:.4f}, Train Accuracy = {train_accuracy:.4f}, "
            f"Validation Loss = {val_loss:.4f}, Validation Accuracy = {val_accuracy:.4f}")

        # Checkpoint callback saves model weights at the end of each epoch
        checkpoint_callback.on_epoch_end(epoch)
        print("Best weights saved.")

        # Check for early stopping
        if early_stopping_callback.on_epoch_end(epoch, logs={'val_accuracy': val_accuracy}):
            print("Early stopping triggered.")
            break

        # Update TensorBoard logs
        tensorboard_callback.on_epoch_end(epoch, logs={'train_loss': train_loss, 'train_accuracy': train_accuracy,
                                                    'val_loss': val_loss, 'val_accuracy': val_accuracy})

        # Save last weights
        model.save_weights(os.path.join(os.path.join(save_dir, 'weights'), 'last_model_weights.h5'))
        print("Last weights saved.")
    return his_train
