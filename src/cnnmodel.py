import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils import class_weight

# Paths
cnn_saved_model_path = "cnn_model.h5"
cnn_saved_model_fit_results_path = "cnn_model_history.pkl"
cnn_load_model = True
cnn_train_again = False
cnn_save_model = True

# Load dataset
prepared_data_path = "/Users/vansh11/Downloads/prepared_dataset/"
batch_size = 64  # Increased batch size for faster training
img_size = (224, 224)

train_set = keras.preprocessing.image.ImageDataGenerator(rescale=1./255).flow_from_directory(
    os.path.join(prepared_data_path, "train"), target_size=img_size, batch_size=batch_size, class_mode="categorical"
)
validation_set = keras.preprocessing.image.ImageDataGenerator(rescale=1./255).flow_from_directory(
    os.path.join(prepared_data_path, "val"), target_size=img_size, batch_size=batch_size, class_mode="categorical"
)
test_set = keras.preprocessing.image.ImageDataGenerator(rescale=1./255).flow_from_directory(
    os.path.join(prepared_data_path, "test"), target_size=img_size, batch_size=batch_size, class_mode="categorical",
    shuffle=False
)

# Compute class weights
labels = train_set.classes
class_weights_dict = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = {i: class_weights_dict[i] for i in range(len(class_weights_dict))}
print("Class Weights:", class_weights)

# Load or train model
if os.path.exists(cnn_saved_model_path) and cnn_load_model:
    cnn_model = keras.models.load_model(cnn_saved_model_path)
    trained = True
else:
    trained = False
    cnn_model = keras.Sequential([
        Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", input_shape=(224, 224, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(64, activation='relu'),  # Reduced layer size for speed
        Dropout(0.3),
        Dense(4, activation="softmax")
    ])

    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    cnn_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Fast training settings
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)
reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=0.0001)

# Train or load history
if os.path.exists(cnn_saved_model_fit_results_path) and not cnn_train_again and trained:
    with open(cnn_saved_model_fit_results_path, "rb") as f:
        cnn_history = pickle.load(f)
else:
    train_set.reset()
    validation_set.reset()
    results = cnn_model.fit(
        train_set, epochs=20, validation_data=validation_set,
        callbacks=[early_stop, reduce_lr_callback], class_weight=class_weights
    )
    cnn_history = results.history
    with open(cnn_saved_model_fit_results_path, 'wb') as f:
        pickle.dump(cnn_history, f)

# Save model if needed
if cnn_save_model:
    cnn_model.save(cnn_saved_model_path)

# Evaluate on test set
test_set.reset()
cnn_results = cnn_model.evaluate(test_set)
print(f"The testing accuracy is: {cnn_results[1] * 100:.2f}%")
print(f"The testing loss is: {cnn_results[0]:.4f}")
