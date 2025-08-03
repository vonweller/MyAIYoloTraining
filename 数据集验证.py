import os
import yaml
from pathlib import Path

def validate_dataset(data_yaml_path):
    """验证YOLO数据集的完整性"""
    print(f"验证数据集: {data_yaml_path}")
    
    # 读取data.yaml文件
    with open(data_yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    print(f"数据集配置: {data}")
    
    # 检查训练和验证路径
    train_path = data.get('train')
    val_path = data.get('val')
    
    print(f"训练图像路径: {train_path}")
    print(f"验证图像路径: {val_path}")
    
    # 检查路径是否存在
    if not os.path.exists(train_path):
        print(f"错误: 训练图像路径不存在: {train_path}")
        return False
    
    if not os.path.exists(val_path):
        print(f"错误: 验证图像路径不存在: {val_path}")
        return False
    
    # 检查标签路径
    train_labels_path = train_path.replace('images', 'labels')
    val_labels_path = val_path.replace('images', 'labels')
    
    print(f"训练标签路径: {train_labels_path}")
    print(f"验证标签路径: {val_labels_path}")
    
    if not os.path.exists(train_labels_path):
        print(f"错误: 训练标签路径不存在: {train_labels_path}")
        return False
    
    if not os.path.exists(val_labels_path):
        print(f"错误: 验证标签路径不存在: {val_labels_path}")
        return False
    
    # 统计文件数量
    train_images = list(Path(train_path).glob('*.jpg')) + list(Path(train_path).glob('*.JPG'))
    val_images = list(Path(val_path).glob('*.jpg')) + list(Path(val_path).glob('*.JPG'))
    
    train_labels = list(Path(train_labels_path).glob('*.txt'))
    val_labels = list(Path(val_labels_path).glob('*.txt'))
    
    print(f"训练图像数量: {len(train_images)}")
    print(f"训练标签数量: {len(train_labels)}")
    print(f"验证图像数量: {len(val_images)}")
    print(f"验证标签数量: {len(val_labels)}")
    
    # 检查图像和标签是否匹配
    train_image_names = {img.stem for img in train_images}
    train_label_names = {lbl.stem for lbl in train_labels}
    
    val_image_names = {img.stem for img in val_images}
    val_label_names = {lbl.stem for lbl in val_labels}
    
    train_missing_labels = train_image_names - train_label_names
    val_missing_labels = val_image_names - val_label_names
    
    if train_missing_labels:
        print(f"警告: 训练集中有 {len(train_missing_labels)} 个图像缺少标签")
        print(f"缺少标签的图像: {list(train_missing_labels)[:5]}...")  # 只显示前5个
    
    if val_missing_labels:
        print(f"警告: 验证集中有 {len(val_missing_labels)} 个图像缺少标签")
        print(f"缺少标签的图像: {list(val_missing_labels)[:5]}...")  # 只显示前5个
    
    # 检查标签文件格式
    print("检查标签文件格式...")
    sample_label = train_labels[0] if train_labels else None
    if sample_label:
        try:
            with open(sample_label, 'r') as f:
                lines = f.readlines()
                print(f"示例标签文件 {sample_label.name} 内容:")
                for i, line in enumerate(lines[:3]):  # 只显示前3行
                    print(f"  {line.strip()}")
                    parts = line.strip().split()
                    if len(parts) != 5:
                        print(f"  警告: 标签格式可能有问题，应该有5个值，实际有{len(parts)}个")
        except Exception as e:
            print(f"读取标签文件时出错: {e}")
    
    print("数据集验证完成!")
    return True

if __name__ == "__main__":
    data_yaml_path = r"C:\Users\52953\Desktop\yolo_detection_dataset\data.yaml"
    validate_dataset(data_yaml_path)
