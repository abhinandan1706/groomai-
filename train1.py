import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

print("Step 1: Loading data...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    'dataset',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_data = train_datagen.flow_from_directory(
    'dataset',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# ✅ Batch test block — just to check data is being read correctly
print("Testing batch loading...")
x_batch, y_batch = next(train_data)
print(f"Batch shape: {x_batch.shape}, Labels shape: {y_batch.shape}")

print("Step 2: Building model...")

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(train_data.num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Step 3: Starting training...")

try:
    model.fit(
        train_data,
        validation_data=val_data,
        epochs=10
    )
except Exception as e:
    print(f"🔥 Training failed: {e}")
 # Replace with your trained model
model.save('groomai_skin_model.h5')
print("Model saved successfully ✅")
