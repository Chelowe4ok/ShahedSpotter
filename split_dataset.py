import os
import random
import shutil
from pathlib import Path

def split_dataset(base_folder_name='data', train_percent=0.8, val_percent=0.1, test_percent=0.1):
    # 1. Отримуємо абсолютний шлях до базової папки (data)
    # Це прибере будь-які проблеми з відносними шляхами у Windows
    root_path = Path(__file__).parent.resolve()
    base_path = (root_path / base_folder_name).resolve()
    
    train_img_dir = base_path / 'train' / 'images'
    train_lbl_dir = base_path / 'train' / 'labels'
    
    if not train_img_dir.exists():
        print(f"❌ Помилка: Папка не знайдена за шляхом: {train_img_dir}")
        return

    # 2. Створюємо структуру папок для val та test (використовуємо абсолютні шляхи)
    for split in ['val', 'test']:
        (base_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (base_path / split / 'labels').mkdir(parents=True, exist_ok=True)

    # 3. Знаходимо всі зображення
    extensions = ['*.jpg', '*.jpeg', '*.png']
    images = []
    for ext in extensions:
        images.extend(list(train_img_dir.glob(ext)))
    
    if not images:
        print("❌ Зображень не знайдено в папці train/images")
        return

    random.shuffle(images)
    
    total_count = len(images)
    val_count = int(total_count * val_percent)
    test_count = int(total_count * test_percent)
    
    val_files = images[:val_count]
    test_files = images[val_count:val_count + test_count]
    
    def move_pairs(files, target_split):
        print(f"📦 Переміщення {len(files)} файлів у {target_split}...")
        for img_path in files:
            # Перевірка наявності файлу перед переміщенням
            if not img_path.exists():
                continue
                
            file_stem = img_path.stem  # назва без розширення
            label_path = train_lbl_dir / f"{file_stem}.txt"
            
            dest_img = base_path / target_split / 'images' / img_path.name
            dest_lbl = base_path / target_split / 'labels' / f"{file_stem}.txt"
            
            try:
                # Переміщуємо картинку
                shutil.move(str(img_path.absolute()), str(dest_img.absolute()))
                
                # Переміщуємо анотацію, якщо вона є
                if label_path.exists():
                    shutil.move(str(label_path.absolute()), str(dest_lbl.absolute()))
            except Exception as e:
                print(f"⚠️ Помилка при переміщенні {img_path.name}: {e}")

    # Виконуємо розподіл
    move_pairs(val_files, 'val')
    move_pairs(test_files, 'test')

    print("\n✅ Розподіл завершено успішно!")
    print(f"Train: {total_count - len(val_files) - len(test_files)} | Valid: {len(val_files)} | Test: {len(test_files)}")

if __name__ == "__main__":
    split_dataset('data')