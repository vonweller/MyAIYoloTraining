# MyAIYoloTraining

一个基于YOLO11的AI目标检测训练工具集，提供完整的可视化界面和训练管理功能。

## 项目概述

这个项目包含了完整的YOLO模型训练生态系统，主要特性包括：

- 🎯 **可视化训练界面** - 基于PyQt5的现代化GUI界面
- 🔧 **完整的训练工具链** - 支持训练、测试、推理、数据集转换
- 📊 **实时监控** - 训练进度实时显示和日志管理
- 🎨 **多主题支持** - 包含科技风格主题
- 💾 **数据集管理** - 支持多种数据集格式转换
- ⚡ **GPU加速** - 支持CUDA GPU训练加速

## 项目结构

```
MyAIYoloTraining/
├── YOLO_UI/                    # 主要的UI应用程序
│   ├── main.py                 # 应用程序入口
│   ├── ui/                     # 用户界面组件
│   │   ├── main_window.py      # 主窗口
│   │   ├── components/         # UI组件
│   │   │   ├── training_tab.py     # 训练页面
│   │   │   ├── testing_tab.py      # 测试页面
│   │   │   ├── inference_tab.py    # 推理页面
│   │   │   ├── dataset_converter_tab.py # 数据集转换
│   │   │   └── settings_tab.py     # 设置页面
│   │   └── assets/             # 图标和资源文件
│   ├── utils/                  # 工具模块
│   │   ├── training_worker.py  # 训练工作线程
│   │   ├── testing_worker.py   # 测试工作线程
│   │   ├── inference_worker.py # 推理工作线程
│   │   ├── dataset_converter.py # 数据集转换工具
│   │   ├── theme_manager.py    # 主题管理
│   │   └── error_manager.py    # 错误处理
│   ├── data/                   # 数据和配置
│   │   ├── models/             # 模型文件存储
│   │   └── settings.json       # 应用程序设置
│   └── datasets/               # 示例数据集
├── yolo简单测试.py             # 简单的YOLO训练测试脚本
├── 数据集验证.py               # 数据集验证工具
├── 修复数据集.py               # 数据集修复工具
└── yolo11n.pt                  # 预训练模型文件
```

## 安装和使用

### 环境要求

- Python 3.8+
- PyTorch 1.10.0+
- CUDA (可选，用于GPU加速)

### 安装依赖

```bash
# 基础版本
pip install -r YOLO_UI/requirements.txt

# GPU版本（推荐）
pip install -r YOLO_UI/requirements-gpu.txt

# 开发版本
pip install -r YOLO_UI/requirements-gpu-dev.txt
```

### 启动应用

```bash
cd YOLO_UI
python main.py
```

## 功能特性

### 1. 训练模块
- 支持目标检测(detect)和分类(classify)任务
- 多种模型选择（YOLO11n/s/m/l/x）
- 实时训练进度监控
- 自动数据集验证
- 训练参数可视化配置

### 2. 测试模块
- 模型性能评估
- 可视化测试结果
- 自动生成测试报告
- 支持自定义测试数据集

### 3. 推理模块
- 单张图片推理
- 批量图片处理
- 实时摄像头检测
- 结果可视化和保存

### 4. 数据集工具
- 多格式数据集转换（COCO、VOC、YOLO等）
- 数据集验证和修复
- 自动标注质量检查
- 数据集统计分析

### 5. 设置管理
- GPU/CPU选择
- 主题切换
- 性能优化配置
- 快捷键管理

## 快速开始

1. **下载预训练模型**：项目已包含`yolo11n.pt`模型文件

2. **准备数据集**：将数据集放入`YOLO_UI/datasets/`目录，或使用内置的`coco8`示例数据集

3. **启动应用**：运行`python YOLO_UI/main.py`

4. **开始训练**：
   - 选择训练标签页
   - 配置训练参数
   - 选择数据集路径
   - 点击开始训练

## 技术特性

- **异步处理**：训练、测试、推理均在独立线程中执行，不阻塞UI
- **错误处理**：完善的错误捕获和用户友好的错误提示
- **日志管理**：详细的操作日志和终端输出重定向
- **主题系统**：可扩展的主题管理系统
- **快捷键支持**：常用操作的键盘快捷键
- **工具提示**：详细的操作说明和参数解释

## 开发信息

- **开发者**：Vonweller AI Tools
- **联系邮箱**：529538187@qq.com
- **技术栈**：Python, PyQt5, PyTorch, Ultralytics YOLO
- **支持平台**：Windows, Linux, macOS

## 版本历史

- 最新版本包含完整的GUI界面和所有核心功能
- 支持YOLO11系列模型
- 包含数据集转换和验证工具
- 添加了主题管理和用户体验优化

## 许可证

本项目用于学习和研究目的。请遵循相关开源协议。

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。

---

**注意**：首次运行前请确保安装了所有依赖项，特别是CUDA环境（如果使用GPU训练）。