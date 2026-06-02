
import cv2
import numpy as np
import os
import random
import hashlib

# =========================================================
# НАСТРОЙКИ
# =========================================================

IMAGE_SIZE = 32                     # размер выходных изображений (32x32)
SEED_DIR = "real_seeds"             # папка с исходными фото
OUTPUT_DIR = "synthetic_dataset"    # папка для сгенерированного датасета

SAMPLES_PER_CLASS = 2000            # сколько изображений генерировать на класс

CLASSES = [
    'none',
    'one_finger',
    'two_fingers',
    'three_fingers',
    'four_fingers',
    'five_fingers',
    'circle',
    'square',
    'triangle',
]

# =========================================================
# УТИЛИТЫ
# =========================================================

def ensure_size(img):
    """Приводит изображение к размеру IMAGE_SIZE (32x32)."""
    if img.shape != (IMAGE_SIZE, IMAGE_SIZE):
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    return img

def clamp(img):
    """Обрезает значения пикселей в диапазон [0, 255] и приводит к uint8."""
    return np.clip(img, 0, 255).astype(np.uint8)

# =========================================================
# БЕЗОПАСНЫЕ АУГМЕНТАЦИИ 
# =========================================================

def small_rotation(img):
    """Поворот на небольшой угол (до ±8°)."""
    h, w = img.shape
    angle = random.uniform(-8, 8)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

def small_shift(img):
    """Сдвиг по X и Y до ±2 пикселей."""
    h, w = img.shape
    dx = random.randint(-2, 2)
    dy = random.randint(-2, 2)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

def small_scale(img):
    """Изменение масштаба от 0.95 до 1.05 с центрированием."""
    h, w = img.shape
    scale = random.uniform(0.95, 1.05)
    nw = max(8, int(w * scale))
    nh = max(8, int(h * scale))
    scaled = cv2.resize(img, (nw, nh))

    if scale < 1.0:
        canvas = np.zeros((h, w), dtype=np.uint8)
        px = (w - nw) // 2
        py = (h - nh) // 2
        canvas[py:py+nh, px:px+nw] = scaled
        return canvas
    else:
        cx = (nw - w) // 2
        cy = (nh - h) // 2
        return scaled[cy:cy+h, cx:cx+w]

def random_crop_shift(img):
    """
    Случайный сдвиг изображения внутри чуть увеличенного холста (padding=2)
    с заполнением краёв отражением. Эквивалентно небольшому дрожанию камеры.
    """
    pad = 2
    padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    x = random.randint(0, pad * 2)
    y = random.randint(0, pad * 2)
    return padded[y:y+32, x:x+32]

def brightness_contrast(img):
    """Мягкое изменение яркости и контраста (не инвертирует)."""
    alpha = random.uniform(0.9, 1.1)   # контраст
    beta = random.randint(-10, 10)     # яркость
    img = img.astype(np.float32) * alpha + beta
    return clamp(img)


def random_augment(img):
    """
    Применяет случайный набор мягких преобразований.
    Никаких шумов, размытий, сжатий, бинаризации и т.п.
    """
    img = ensure_size(img)

    # Геометрические преобразования
    if random.random() < 0.7:
        img = small_rotation(img)
    if random.random() < 0.7:
        img = small_shift(img)
    if random.random() < 0.5:
        img = small_scale(img)
    if random.random() < 0.5:
        img = random_crop_shift(img)

    # Изменение яркости/контраста
    if random.random() < 0.5:
        img = brightness_contrast(img)

    # Горизонтальное отражение (безопасно для жестов)
    if random.random() < 0.3:
        img = cv2.flip(img, 1)

    return ensure_size(img)

# =========================================================
# ЗАГРУЗКА SEED-ИЗОБРАЖЕНИЙ (PNG, JPG, JPEG)
# =========================================================

def load_seed_images(class_name):
    """Собирает все исходные фото для указанного класса из real_seeds/."""
    seed_images = []

    # Путь 1: real_seeds/{gesture}/
    flat_path = os.path.join(SEED_DIR, class_name)
    if os.path.exists(flat_path):
        for fname in os.listdir(flat_path):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(flat_path, fname)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    seed_images.append(ensure_size(img))

    # Путь 2: real_seeds/{user}/{gesture}/
    if os.path.exists(SEED_DIR):
        for user_dir in os.listdir(SEED_DIR):
            user_gesture_path = os.path.join(SEED_DIR, user_dir, class_name)
            if os.path.exists(user_gesture_path):
                for fname in os.listdir(user_gesture_path):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        path = os.path.join(user_gesture_path, fname)
                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            seed_images.append(ensure_size(img))

    return seed_images

# =========================================================
# ХЕШ ДЛЯ ОТСЕИВАНИЯ ДУБЛИКАТОВ
# =========================================================

def image_hash(img):
    """Возвращает MD5-хеш изображения (используется для удаления дубликатов)."""
    return hashlib.md5(img.tobytes()).hexdigest()

# =========================================================
# ГЛАВНАЯ ФУНКЦИЯ
# =========================================================

def main():
    print("=" * 60)
    print("ESP32 GENTLE DATASET GENERATOR (NO EXTRA NOISE)")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total_generated = 0

    for class_name in CLASSES:
        print(f"\n[{class_name}]")
        out_path = os.path.join(OUTPUT_DIR, class_name)
        os.makedirs(out_path, exist_ok=True)

        seeds = load_seed_images(class_name)
        if not seeds:
            print("  ❌ seed-фото не найдены")
            continue

        print(f"  seed-фото: {len(seeds)}")

        # Для сложных классов (two_fingers, three_fingers) генерируем больше
        if class_name in ('two_fingers', 'three_fingers'):
            target = SAMPLES_PER_CLASS * 2
        else:
            target = SAMPLES_PER_CLASS

        print(f"  target: {target}")

        idx = 0
        hashes = set()

        # 1. Сохраняем оригинальные seed-фото (без изменений)
        for img in seeds:
            h = image_hash(img)
            if h in hashes:
                continue
            hashes.add(h)
            cv2.imwrite(os.path.join(out_path, f"{idx:05d}.jpg"), img)
            idx += 1

        print(f"  originals: {idx}")

        # 2. Генерируем аугментированные варианты до достижения target
        attempts = 0
        while idx < target:
            attempts += 1
            seed = random.choice(seeds)
            aug = random_augment(seed.copy())

            h = image_hash(aug)
            if h in hashes:
                continue
            hashes.add(h)

            cv2.imwrite(os.path.join(out_path, f"{idx:05d}.jpg"), aug)
            idx += 1

            if idx % 500 == 0:
                print(f"  {idx}/{target}")

            # Защита от бесконечного цикла (если всё время дубликаты)
            if attempts > target * 20:
                print("  ⚠ слишком много дубликатов, прерываем")
                break

        total_generated += idx
        print(f"  ✅ готово: {idx}")

    # 3. Итоговая статистика
    print("\n" + "=" * 60)
    print("ИТОГИ")
    print("=" * 60)
    grand_total = 0
    for cls in CLASSES:
        path = os.path.join(OUTPUT_DIR, cls)
        if os.path.exists(path):
            cnt = len([f for f in os.listdir(path) if f.endswith('.jpg')])
            grand_total += cnt
            print(f"{cls:15s}: {cnt}")
    print("-" * 60)
    print(f"ВСЕГО: {grand_total}")
    print("=" * 60)
    print("\nГотово.")
    print(f"Датасет сохранён в папку: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()