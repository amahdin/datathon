import pandas as pd
import os
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the base directory where the 'archive-2' folder is located
base_dir = "dinosaur_dataset"

# Define the batch size and target image size for the model
batch_size = 32
target_image_size = (224, 224)

# Set up the ImageDataGenerators with a split for training and testing
data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2,
)

# Create the train generator
train_generator = data_gen.flow_from_directory(
    base_dir,
    target_size=target_image_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="training",
)

# Create the test generator
test_generator = data_gen.flow_from_directory(
    base_dir,
    target_size=target_image_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation",
)
# Load the base model
base_model = tf.keras.applications.ResNet152V2(
    weights="imagenet", include_top=False, input_shape=target_image_size + (3,)
)

# Start defining the custom classification head
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)

# Add a dropout layer before the dense layer
# Let's say you want to set 50% of the inputs to zero
x = tf.keras.layers.Dropout(0.5)(x)  # Adjust the dropout rate as needed

x = tf.keras.layers.Dense(512, activation="relu")(x)

# Continue with the final dense layer for classification
predictions = tf.keras.layers.Dense(15, activation="softmax")(x)

# Combine the base model with the custom classification head
model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

# Now proceed with compilation and training as before

print("Number of layers in the base model: ", len(base_model.layers))

# Freeze the base model layers
for layer in base_model.layers[:530]:
    layer.trainable = False

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=15,
    validation_data=test_generator,
    validation_steps=len(test_generator),
)