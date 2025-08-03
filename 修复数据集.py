import os
import shutil
from pathlib import Path
import random

def fix_dataset():
    """修复数据集，正确分离训练集和验证集"""
    
    # 数据集路径
    dataset_root = Path(r"C:\Users\52953\Desktop\yolo_detection_dataset")
    train_images_path = dataset_root / "images" / "train"
    val_images_path = dataset_root / "images" / "val"
    train_labels_path = dataset_root / "labels" / "train"
    val_labels_path = dataset_root / "labels" / "val"
    
    print("开始修复数据集...")
    
    # 获取所有有标签的图像
    train_labels = list(train_labels_path.glob('*.txt'))
    labeled_image_names = {lbl.stem for lbl in train_labels}
    
    print(f"找到 {len(labeled_image_names)} 个有标签的图像")
    
    # 获取所有对应的图像文件
    all_images = []
    for name in labeled_image_names:
        # 查找对应的图像文件（可能是.jpg或.JPG）
        jpg_file = train_images_path / f"{name}.jpg"
        JPG_file = train_images_path / f"{name}.JPG"
        
        if jpg_file.exists():
            all_images.append((jpg_file, train_labels_path / f"{name}.txt"))
        elif JPG_file.exists():
            all_images.append((JPG_file, train_labels_path / f"{name}.txt"))
    
    print(f"找到 {len(all_images)} 对匹配的图像-标签对")
    
    # 随机打乱
    random.seed(42)  # 设置随机种子以确保可重复性
    random.shuffle(all_images)
    
    # 分割数据集 (80% 训练, 20% 验证)
    split_idx = int(len(all_images) * 0.8)
    train_pairs = all_images[:split_idx]
    val_pairs = all_images[split_idx:]
    
    print(f"训练集: {len(train_pairs)} 对")
    print(f"验证集: {len(val_pairs)} 对")
    
    # 清空验证集目录
    if val_images_path.exists():
        shutil.rmtree(val_images_path)
    if val_labels_path.exists():
        shutil.rmtree(val_labels_path)
    
    val_images_path.mkdir(parents=True, exist_ok=True)
    val_labels_path.mkdir(parents=True, exist_ok=True)
    
    # 移动验证集文件
    print("移动验证集文件...")
    for img_file, lbl_file in val_pairs:
        # 移动图像文件
        shutil.move(str(img_file), str(val_images_path / img_file.name))
        # 移动标签文件
        shutil.move(str(lbl_file), str(val_labels_path / lbl_file.name))
    
    print("数据集修复完成!")
    
    # 验证结果
    train_images_count = len(list(train_images_path.glob('*.jpg'))) + len(list(train_images_path.glob('*.JPG')))
    val_images_count = len(list(val_images_path.glob('*.jpg'))) + len(list(val_images_path.glob('*.JPG')))
    train_labels_count = len(list(train_labels_path.glob('*.txt')))
    val_labels_count = len(list(val_labels_path.glob('*.txt')))
    
    print(f"最终统计:")
    print(f"训练集图像: {train_images_count}")
    print(f"训练集标签: {train_labels_count}")
    print(f"验证集图像: {val_images_count}")
    print(f"验证集标签: {val_labels_count}")

if __name__ == "__main__":
    fix_dataset()
