from ultralytics import YOLO
import traceback
import torch

if __name__ == "__main__":
    try:
        print("开始YOLO训练测试...")
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU设备: {torch.cuda.get_device_name(0)}")
        
        # 加载模型
        print("加载YOLO11n模型...")
        model = YOLO("yolo11n.pt")
        
        # 最简单的训练配置
        print("开始训练...")
        results = model.train(
            data=r"C:\Users\52953\Desktop\yolo_detection_dataset\data.yaml",
            epochs=100,  # 只训练1轮
            imgsz=320,  # 小图像尺寸
            batch=4,   # 最小批次大小
            workers=4,  # 禁用多进程
            cache=True,  # 禁用缓存
            verbose=True,
            amp=True,  # 禁用AMP
            device=0,  # 使用GPU
            save=True,  # 保存模型
            plots=False,  # 禁用绘图以减少内存使用
            val=False  # 禁用验证以简化流程
        )
        print("训练完成！")
        print(f"结果保存在: {results.save_dir}")
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        traceback.print_exc()
