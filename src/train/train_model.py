# train_model.py (modified version)
"""
CNN Model Training for Handwritten Character Recognition
Supports IDX file format for EMNIST
"""
import os
import time
import sys

SRC_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.insert(0, SRC_ROOT)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report




# Import custom IDX loader
# NOTE: load_mnist_from_files / combine_mnist_emnist_digits are no longer
# imported - the MNIST idx files were deleted and this project now trains
# on EMNIST digits + EMNIST letters only.
from data.idx_loader import (
    load_emnist_digits,
    load_emnist_letters,
)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

class HCRModelTrainer:
    """Handwritten Character Recognition Model Trainer"""
    
    def __init__(self, model_type='digits'):
        """
        Initialize trainer
        model_type: 'digits' (0-9), 'letters' (A-Z), 'combined' (0-9 + A-Z)
        """
        self.model_type = model_type
        self.model = None
        self.history = None
        self.num_classes = self._get_num_classes()
        
    def _get_num_classes(self):
        """Determine number of classes based on model type"""
        if self.model_type == 'digits':
            return 10  # 0-9
        elif self.model_type == 'letters':
            return 26  # A-Z
        else:  # combined
            return 36  # 0-9 + A-Z
    
    def load_custom_data(self):
        """
        Load data from IDX files.

        FIX: this project only has EMNIST data on disk now (MNIST idx files
        were deleted). Digits load straight from EMNIST Digits instead of
        combine_mnist_emnist_digits()/load_mnist_from_files(), which
        referenced dataset/mnist/ files that no longer exist and would
        have crashed with a FileNotFoundError.
        """
        
        if self.model_type == 'digits':
            # EMNIST Digits (0-9), already 0-indexed
            (x_train, y_train), (x_test, y_test) = load_emnist_digits()
        
        elif self.model_type == 'letters':
            # Load EMNIST Letters (A-Z), 1-indexed (1-26)
            (x_train, y_train), (x_test, y_test) = load_emnist_letters()
        
        else:  # combined digits + letters
            print("\n" + "="*60)
            print("LOADING COMBINED DIGITS + LETTERS DATASET")
            print("="*60)
            
            # This requires EMNIST ByClass or Balanced, which you don't
            # currently have on disk. Falling back to letters only.
            print("Note: Combined model requires EMNIST ByClass dataset")
            print("If you don't have it, train digits and letters separately")
            
            (x_train, y_train), (x_test, y_test) = load_emnist_letters()
        
        return (x_train, y_train), (x_test, y_test)
    
    def prepare_data(self, x_train, x_test):
        """
        Prepare images for CNN input
        - Reshape to add channel dimension
        - Normalize pixel values
        """
        print("\n" + "="*60)
        print("PREPARING DATA FOR CNN")
        print("="*60)
        
        # Check current shape
        print(f"Before preparation - x_train shape: {x_train.shape}")
        print(f"Before preparation - x_test shape: {x_test.shape}")
        
        # Add channel dimension (for grayscale images)
        if len(x_train.shape) == 3:
            x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
            x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
        
        # Convert to float32 and normalize to [0, 1]
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train = x_train / 255.0
        x_test = x_test / 255.0
        
        print(f"After preparation - x_train shape: {x_train.shape}")
        print(f"After preparation - x_test shape: {x_test.shape}")
        print(f"Pixel value range: [{x_train.min()}, {x_train.max()}]")
        
        return x_train, x_test
    
    def prepare_labels(self, y_train, y_test):
        """
        Convert labels to categorical (one-hot encoding)

        FIX: the -1 shift only applies to EMNIST Letters, which is
        1-indexed (1='a' ... 26='z'). EMNIST Digits is already 0-indexed
        (0-9). The previous version unconditionally did `y_train - 1` for
        every model type, which for digits turned label 0 into -1 and
        silently corrupted one-hot encoding for the whole digits class.
        """
        if self.model_type == 'letters':
            y_train = y_train - 1
            y_test = y_test - 1
        # else: digits (and combined, which currently falls back to
        # letters data upstream) are already 0-indexed - no shift.

        y_train_cat = to_categorical(y_train, self.num_classes)
        y_test_cat = to_categorical(y_test, self.num_classes)
        return y_train_cat, y_test_cat
    
    def split_validation(self, x_train, y_train, val_split=0.1):
        """
        Split training data into train and validation sets
        """
        num_val = int(len(x_train) * val_split)
        indices = np.random.permutation(len(x_train))
        
        val_indices = indices[:num_val]
        train_indices = indices[num_val:]
        
        x_val = x_train[val_indices]
        y_val = y_train[val_indices]
        x_train = x_train[train_indices]
        y_train = y_train[train_indices]
        
        print(f"\nData Split:")
        print(f"  Training set: {len(x_train)} samples")
        print(f"  Validation set: {len(x_val)} samples")
        
        return x_train, y_train, x_val, y_val
    
    def augment_data(self, x_train, y_train):
        """Data augmentation to improve model generalization"""
        print("\n" + "="*60)
        print("DATA AUGMENTATION")
        print("="*60)
        
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            shear_range=0.1,
            fill_mode='nearest'
        )
        
        datagen.fit(x_train)
        
        print("Data augmentation parameters:")
        print(f"  Rotation range: ±10°")
        print(f"  Width shift: ±10%")
        print(f"  Height shift: ±10%")
        print(f"  Zoom range: ±10%")
        print(f"  Shear range: ±10%")
        
        return datagen
    
    def build_cnn_model(self, architecture='advanced'):
        """Build CNN model for character recognition"""
        print("\n" + "="*60)
        print(f"BUILDING CNN MODEL (Architecture: {architecture})")
        print("="*60)
        
        inputs = keras.Input(shape=(28, 28, 1), name='input_layer')
        
        if architecture == 'simple':
            x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Flatten()(x)
            x = layers.Dense(128, activation='relu')(x)
            x = layers.Dropout(0.5)(x)
            outputs = layers.Dense(self.num_classes, activation='softmax')(x)
            
        elif architecture == 'advanced':
            # Your existing advanced architecture
            x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
            x = layers.BatchNormalization()(x)
            x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Dropout(0.25)(x)
            
            x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Dropout(0.25)(x)
            
            x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Dropout(0.25)(x)
            
            x = layers.Flatten()(x)
            x = layers.Dense(256, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.5)(x)
            
            x = layers.Dense(128, activation='relu')(x)
            x = layers.Dropout(0.3)(x)
            outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs, name=f'HCR_{architecture}')
        self.model.summary()
        
        return self.model
    
    def compile_model(self, learning_rate=0.001, optimizer='adam'):
        """Compile the model"""
        print("\n" + "="*60)
        print("COMPILING MODEL")
        print("="*60)
        
        if optimizer == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'rmsprop':
            opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        
        self.model.compile(
            optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        print(f"Optimizer: {optimizer.upper()}")
        print(f"Learning rate: {learning_rate}")
        
        return self.model
    
    def train_model(self, x_train, y_train, x_val, y_val, 
                    epochs=20, batch_size=128, use_augmentation=True):
        """Train the CNN model"""
        print("\n" + "="*60)
        print("TRAINING MODEL")
        print("="*60)
        
        # Checkpoint saves inside model/ so app.py's load_model() can find
        # it - app.py looks for model/best_hcr_model_{type}.h5.
        os.makedirs('model', exist_ok=True)

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                f'model/best_hcr_model_{self.model_type}.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        print(f"Training configuration:")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Training samples: {len(x_train)}")
        print(f"  Validation samples: {len(x_val)}")
        
        start_time = time.time()
        
        if use_augmentation:
            datagen = self.augment_data(x_train, y_train)
            self.history = self.model.fit(
                datagen.flow(x_train, y_train, batch_size=batch_size),
                validation_data=(x_val, y_val),
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
        else:
            self.history = self.model.fit(
                x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(x_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        
        return self.history
    
    def save_model(self, model_path='hcr_model.h5'):
        """Save the trained model"""
        print("\n" + "="*60)
        print("SAVING MODEL")
        print("="*60)
        
        os.makedirs('model', exist_ok=True)
        full_path = os.path.join('model', model_path)
        self.model.save(full_path)
        print(f"✓ Model saved: {full_path}")
        
        # Save class mapping
        self.save_class_mapping()
    
    def save_class_mapping(self):
        """Save class to character mapping"""
        import json
        
        if self.model_type == 'digits':
            mapping = {i: str(i) for i in range(10)}
        elif self.model_type == 'letters':
            mapping = {i: chr(65 + i) for i in range(26)}
        else:
            mapping = {}
            for i in range(10):
                mapping[i] = str(i)
            for i in range(26):
                mapping[10 + i] = chr(65 + i)
        
        mapping_path = os.path.join('model', 'class_mapping.json')
        with open(mapping_path, 'w') as f:
            json.dump(mapping, f, indent=2)
        
        print(f"✓ Class mapping saved: {mapping_path}")

    def plot_training_history(self):
        """Save accuracy/loss curves so you can eyeball whether training went well"""
        if self.history is None:
            print("⚠️ No training history to plot")
            return

        os.makedirs('model', exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(self.history.history['accuracy'], label='train')
        axes[0].plot(self.history.history['val_accuracy'], label='val')
        axes[0].set_title('Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].legend()

        axes[1].plot(self.history.history['loss'], label='train')
        axes[1].plot(self.history.history['val_loss'], label='val')
        axes[1].set_title('Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].legend()

        plt.tight_layout()
        out_path = os.path.join('model', f'training_history_{self.model_type}.png')
        plt.savefig(out_path)
        print(f"✓ Training curves saved: {out_path}")

    def evaluate(self, x_test, y_test):
        """Print a classification report and confusion matrix summary on the test set"""
        y_pred_probs = self.model.predict(x_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_test, axis=1)

        print("\n" + "="*60)
        print("TEST SET EVALUATION")
        print("="*60)
        print(classification_report(y_true, y_pred))

        cm = confusion_matrix(y_true, y_pred)
        os.makedirs('model', exist_ok=True)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=False, cmap='Blues')
        plt.title(f'Confusion Matrix - {self.model_type}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        out_path = os.path.join('model', f'confusion_matrix_{self.model_type}.png')
        plt.savefig(out_path)
        print(f"✓ Confusion matrix saved: {out_path}")


def main():
    """
    Driver: wires together load -> prepare -> split -> build -> compile ->
    train -> evaluate -> save. This is what actually needs to run for a
    model file to be produced - the class above only defines the steps.

    Data source: EMNIST only (dataset/emnist/*.idx*-ubyte). MNIST idx
    files are gone, and load_custom_data() no longer references them.

    Orientation: idx_loader.py's load_emnist_digits()/load_emnist_letters()
    already apply the correct EMNIST orientation transpose - verified
    visually (see orientation_test.py). No changes needed there.
    """
    MODEL_TYPE = 'letters'   # change to 'letters' to train the letters model instead
    EPOCHS = 30
    BATCH_SIZE = 128

    trainer = HCRModelTrainer(model_type=MODEL_TYPE)

    # 1. Load raw data from IDX files (EMNIST only)
    (x_train, y_train), (x_test, y_test) = trainer.load_custom_data()

    # 2. Prepare images: add channel dim, normalize to [0, 1]
    x_train, x_test = trainer.prepare_data(x_train, x_test)

    # 3. Prepare labels: shift EMNIST Letters' 1-26 to 0-25 (digits
    #    already 0-indexed, no shift), one-hot encode
    y_train, y_test = trainer.prepare_labels(y_train, y_test)

    # 4. Carve out a validation split from training data
    x_train, y_train, x_val, y_val = trainer.split_validation(x_train, y_train, val_split=0.1)

    # 5. Build and compile the CNN
    trainer.build_cnn_model(architecture='advanced')
    trainer.compile_model(learning_rate=0.001, optimizer='adam')

    # 6. Train (this also saves the best checkpoint to
    #    model/best_hcr_model_{model_type}.h5 via ModelCheckpoint)
    trainer.train_model(x_train, y_train, x_val, y_val,
                         epochs=EPOCHS, batch_size=BATCH_SIZE, use_augmentation=True)

    # 7. Evaluate on the held-out test set and save diagnostic plots
    trainer.plot_training_history()
    trainer.evaluate(x_test, y_test)

    # 8. Save the final model + class_mapping.json
    #    (ModelCheckpoint already saved the BEST epoch to
    #    model/best_hcr_model_{model_type}.h5 - this saves the final state
    #    too, under a different filename, plus the class mapping.)
    trainer.save_model(model_path=f'{MODEL_TYPE}_final.h5')

    print("\n" + "="*60)
    print("DONE")
    print("="*60)
    print(f"Best checkpoint: model/best_hcr_model_{MODEL_TYPE}.h5  <- this is what app.py loads")
    print(f"Final epoch model: model/{MODEL_TYPE}_final.h5")
    print(f"Class mapping: model/class_mapping.json")


if __name__ == "__main__":
    main()