
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# ============ НАСТРОЙКИ ============
IMAGE_SIZE = 32
BATCH_SIZE = 32
EPOCHS = 50
NUM_CLASSES = 9

CLASSES = [
    'none',          # 0
    'one_finger',    # 1
    'two_fingers',   # 2
    'three_fingers', # 3
    'four_fingers',  # 4
    'five_fingers',  # 5
    'circle',        # 6
    'square',        # 7
    'triangle',      # 8
]

# ============ ЗАГРУЗКА ДАТАСЕТА ============
def load_dataset(data_path):
    X, y = [], []

    for class_id, class_name in enumerate(CLASSES):
        class_path = os.path.join(data_path, class_name)
        if not os.path.exists(class_path):
            print(f"Предупреждение: Папка {class_path} не найдена")
            continue

        images = [f for f in os.listdir(class_path)
                  if f.endswith(('.jpg', '.png', '.jpeg'))]
        print(f"Загрузка {len(images)} изображений для класса {class_name}")

        for img_name in images:
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

            # Нормализация: (x - 0.5) * 2 → диапазон [-1, 1]
            # Стабильнее чем /255 для ESP32 с шумной камерой
            img = img.astype(np.float32) / 255.0
            img = (img - 0.5) * 2.0

            X.append(img)
            y.append(class_id)

    X = np.array(X).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
    y = np.array(y)

    print(f"\nЗагружено {len(X)} изображений")
    print(f"Распределение по классам: {np.bincount(y)}")
    return X, y


# ============ МОДЕЛЬ (BATCH NORM ДЛЯ СТАБИЛЬНОСТИ) ============
def create_model():
    model = keras.Sequential([
        layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),

        # Блок 1
        layers.Conv2D(32, 3, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(2),

        # Блок 2
        layers.Conv2D(64, 3, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(2),

        # Блок 3
        layers.Conv2D(128, 3, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.GlobalAveragePooling2D(),

        # Классификатор
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation='softmax'),
    ])
    return model


# ============ ОБУЧЕНИЕ ============
def train_model(X, y):
    # ОДИН stratify split (не два!)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Class weights для несбалансированных классов
    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = dict(enumerate(weights))
    print(f"Class weights: {dict(enumerate(weights.round(2)))}")

    # Модель
    model = create_model()
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=7, restore_best_weights=True, monitor='val_accuracy'
        ),
        keras.callbacks.ReduceLROnPlateau(
            factor=0.5, patience=3, monitor='val_loss'
        ),
        keras.callbacks.ModelCheckpoint(
            'best_model.h5', save_best_only=True, monitor='val_accuracy'
        ),
    ]

    # БЕЗ online-аугментации (вся аугментация в augment_seeds.py)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    plot_training_history(history)
    return model, history


# ============ ВИЗУАЛИЗАЦИЯ ============
def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].plot(history.history['accuracy'], label='Train Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[1].set_title('Model Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()


# ============ КОНВЕРТАЦИЯ В TFLITE (INT8) ============
def convert_to_tflite(model, X_sample):
    def representative_dataset():
        for i in range(0, min(100, len(X_sample)), 10):
            img = X_sample[i].reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)
            yield [img.astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    # Фиксируем int8 вход/выход (важно для ESP32!)
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    with open('gesture_model.tflite', 'wb') as f:
        f.write(tflite_model)

    print(f"✅ Размер модели: {len(tflite_model) / 1024:.2f} КБ")

    # C++ заголовочный файл
    with open('gesture_model.h', 'w') as f:
        f.write('#ifndef GESTURE_MODEL_H\n#define GESTURE_MODEL_H\n\n')
        f.write(f'const unsigned char gesture_model[] = {{\n    ')
        for i, byte in enumerate(tflite_model):
            f.write(f'0x{byte:02x}')
            if i < len(tflite_model) - 1:
                f.write(', ')
            if (i + 1) % 12 == 0:
                f.write('\n    ')
        f.write(f'\n}};\nconst unsigned int gesture_model_len = {len(tflite_model)};\n\n')
        f.write('#endif')

    print("✅ gesture_model.h сохранён")


# ============ ТЕСТИРОВАНИЕ ============
def test_model(model, X_test, y_test):
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nТестовая точность: {test_acc * 100:.2f}%")

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    y_pred = np.argmax(model.predict(X_test), axis=1)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()


# ============ ГЛАВНАЯ ============
def main():
    print("=" * 50)
    print("ОБУЧЕНИЕ НЕЙРОСЕТИ ДЛЯ РАСПОЗНАВАНИЯ ЖЕСТОВ")
    print("=" * 50)

    DATASET_PATH = "synthetic_dataset"

    print("\n[1/4] Загрузка датасета...")
    X, y = load_dataset(DATASET_PATH)
    print(f"Форма данных: {X.shape}")
    print(f"Уникальных классов: {len(np.unique(y))}")

    # ОДИН split (stratify только здесь)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    print("\n[2/4] Обучение модели...")
    model, history = train_model(X_train, y_train)

    print("\n[3/4] Оценка модели...")
    model.load_weights('best_model.h5')
    test_model(model, X_test, y_test)

    print("\n[4/4] Конвертация в TFLite для ESP32...")
    convert_to_tflite(model, X_val)

    print("\nГОТОВО! Модель сохранена.")
    print("Файлы: best_model.h5, gesture_model.tflite, gesture_model.h")


if __name__ == "__main__":
    main()