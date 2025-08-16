from PyQt5.QtCore import QThread, pyqtSignal
import os
import shutil
import numpy as np

class ONNXConverterWorker(QThread):
    """ONNX转换工作线程"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)
    
    def __init__(self, pt_model_path, output_dir):
        super().__init__()
        self.pt_model_path = pt_model_path
        self.output_dir = output_dir
    
    def run(self):
        try:
            self.progress.emit("🚀 开始ONNX模型转换...\n")
            self.progress.emit(f"输入模型: {self.pt_model_path}\n")
            self.progress.emit(f"输出目录: {self.output_dir}\n")
            
            model_name = os.path.splitext(os.path.basename(self.pt_model_path))[0]
            onnx_path = os.path.join(self.output_dir, f"{model_name}.onnx")
            
            self.progress.emit("正在加载YOLO模型...\n")
            from ultralytics import YOLO
            model = YOLO(self.pt_model_path)
            
            self.progress.emit("正在导出ONNX模型...\n")
            
            # 🔥 使用优化的导出参数
            export_params = {
                'format': 'onnx',
                'imgsz': 640,
                'half': False,        # 不使用半精度
                'int8': False,        # 不使用int8量化
                'dynamic': False,     # 不使用动态输入
                'simplify': False,    # 不简化模型
                'opset': 11,         # ONNX opset版本
                'nms': True,         # 启用NMS
                'device': 'cpu'      # 使用CPU导出
            }
            
            success = model.export(**export_params)
            
            if success:
                # 检查生成的ONNX文件
                current_dir_onnx = os.path.join(os.getcwd(), f"{model_name}.onnx")
                model_dir_onnx = os.path.join(os.path.dirname(self.pt_model_path), f"{model_name}.onnx")
                
                # 查找生成的ONNX文件位置
                source_onnx = None
                if os.path.exists(current_dir_onnx):
                    source_onnx = current_dir_onnx
                elif os.path.exists(model_dir_onnx):
                    source_onnx = model_dir_onnx
                
                if source_onnx and os.path.exists(source_onnx):
                    # 移动到指定输出目录
                    if source_onnx != onnx_path:
                        shutil.move(source_onnx, onnx_path)
                    
                    self.progress.emit("✅ ONNX模型导出成功！\n")
                    self.progress.emit(f"输出文件: {onnx_path}\n")
                    
                    # 验证ONNX模型
                    self.verify_onnx_model(onnx_path)
                    
                    success_msg = (
                        f"模型已成功转换为ONNX格式！\n\n"
                        f"输出文件: {onnx_path}\n\n"
                        f"导出参数:\n"
                        f"- 图像尺寸: 640x640\n"
                        f"- ONNX Opset: 11\n"
                        f"- NMS: 已启用\n"
                        f"- 精度: FP32"
                    )
                    self.finished.emit(True, success_msg)
                else:
                    self.progress.emit("❌ 未找到生成的ONNX文件\n")
                    self.finished.emit(False, "ONNX文件生成失败，未找到输出文件")
            else:
                self.progress.emit("❌ ONNX导出失败\n")
                self.finished.emit(False, "ONNX模型导出失败")
                
        except Exception as e:
            error_msg = f"转换过程中发生错误: {str(e)}"
            self.progress.emit(f"❌ {error_msg}\n")
            self.finished.emit(False, error_msg)
    
    def verify_onnx_model(self, onnx_path):
        """验证ONNX模型的有效性"""
        try:
            self.progress.emit("🔍 验证ONNX模型...\n")
            
            # 检查ONNX模型结构
            import onnx
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            self.progress.emit("✅ ONNX模型结构验证通过\n")
            
            # 检查输入输出信息
            import onnxruntime as ort
            session = ort.InferenceSession(onnx_path)
            input_info = session.get_inputs()[0]
            output_info = session.get_outputs()
            
            self.progress.emit("📊 模型信息:\n")
            self.progress.emit(f"   输入: {input_info.name} - {input_info.shape} - {input_info.type}\n")
            for i, output in enumerate(output_info):
                self.progress.emit(f"   输出{i}: {output.name} - {output.shape} - {output.type}\n")
            
            # 简单推理测试
            test_input = np.random.rand(*input_info.shape).astype(np.float32)
            outputs = session.run(None, {input_info.name: test_input})
            self.progress.emit("✅ ONNX推理测试通过\n")
            
        except ImportError as e:
            self.progress.emit(f"⚠️ 缺少验证依赖: {str(e)}\n")
            self.progress.emit("建议安装: pip install onnx onnxruntime\n")
        except Exception as e:
            self.progress.emit(f"⚠️ ONNX模型验证失败: {str(e)}\n")