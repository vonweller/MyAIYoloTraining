# -*- mode: python -*-
import os
block_cipher = None

# 添加必要的隐含导入
hidden_imports = [
    # PyQt5 相关
    'PyQt5.QtCore',
    'PyQt5.QtGui', 
    'PyQt5.QtWidgets',
    'PyQt5.QtSvg',
    'PyQt5.sip',
    
    # PyTorch 相关
    'torch',
    'torch.nn',
    'torch.nn.functional',
    'torchvision',
    'torchvision.transforms',
    
    # YOLO/Ultralytics 相关
    'ultralytics',
    'ultralytics.models',
    'ultralytics.engine',
    'ultralytics.utils',
    'ultralytics.nn',
    'ultralytics.data',
    
    # OpenCV 相关
    'cv2',
    
    # 其他依赖
    'numpy',
    'matplotlib',
    'matplotlib.pyplot',
    'PIL',
    'PIL.Image',
    'yaml',
    'tqdm',
    'pycocotools',
    
    # 自定义模块
    'ui.main_window',
    'ui.components.training_tab',
    'ui.components.testing_tab',
    'ui.components.inference_tab',
    'ui.components.settings_tab',
    'ui.components.dataset_converter_tab',
    'utils.training_worker',
    'utils.testing_worker',
    'utils.inference_worker',
    'utils.dataset_converter',
    'utils.error_manager',
    'utils.theme_manager',
    'utils.splash_screen',
    'utils.terminal_redirect',
    'utils.shortcut_manager',
    'utils.tooltip_manager',
    'utils.data_validator',
]

# 获取应用图标路径
icon_path = os.path.join('.', 'ui', 'assets', 'app_icon.svg') if os.path.exists('ui/assets/app_icon.svg') else None

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        # 包含UI资源
        ('ui', 'ui'),
        ('data', 'data'),
        ('utils', 'utils'),
        # 包含数据集示例（可选，如果太大可以移除）
        ('datasets', 'datasets'),
        # 包含模型文件
        ('*.pt', '.'),
        # 包含配置文件
        ('*.yaml', '.'),
        ('*.yml', '.'),
    ],
    hiddenimports=hidden_imports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        # 排除不需要的模块以减小文件大小
        'matplotlib.tests',
        'numpy.tests',
        'PIL.tests',
        'unittest',
        'test',
        'tests',
        'setuptools',
        'distutils',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='YOLO_Training_Tool',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # 隐藏控制台窗口（GUI应用）
    disable_windowed_traceback=False,
    icon=icon_path
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[
        # 排除压缩某些文件以避免问题
        'vcruntime140.dll',
        'msvcp140.dll',
        'api-ms-win-*.dll',
    ],
    name='YOLO_Training_Tool'
)