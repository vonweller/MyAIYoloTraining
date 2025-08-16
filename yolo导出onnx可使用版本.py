# ===============================
# YOLO PT转ONNX完整解决方案
# ===============================

from ultralytics import YOLO
import onnx
import onnxruntime as ort
import numpy as np
import cv2
import os
import sys


def export_pt_to_onnx_fixed(pt_file, imgsz=640, opset_version=11):
    """
    修复的YOLO PT转ONNX导出函数
    """
    if not os.path.exists(pt_file):
        raise FileNotFoundError(f"模型文件 {pt_file} 不存在")

    model_name = os.path.splitext(os.path.basename(pt_file))[0]

    print(f"正在加载模型: {pt_file}")
    model = YOLO(pt_file)

    print(f"正在导出ONNX模型...")
    try:
        # 🔥 关键修复参数
        export_params = {
            'format': 'onnx',
            'imgsz': 640,
            'half': False,
            'int8': False,
            'dynamic': False,
            'simplify': False,
            'opset': 11,
            'nms': True,  # 🔥 明确启用 NMS
            'device': 'cpu'
        }

        success = model.export(**export_params)

        if success:
            # 🔧 修复：获取完整的ONNX文件路径
            pt_dir = os.path.dirname(pt_file)
            onnx_file = os.path.join(pt_dir, model_name + '.onnx')

            print(f"✅ 成功导出 ONNX 模型: {onnx_file}")

            # 验证ONNX模型
            verify_onnx_model(onnx_file)
            return onnx_file
        else:
            print("❌ 导出失败")
            return None

    except Exception as e:
        print(f"❌ 导出过程中发生错误: {str(e)}")
        raise


def verify_onnx_model(onnx_file):
    """验证ONNX模型的有效性"""
    try:
        # 检查ONNX模型
        onnx_model = onnx.load(onnx_file)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX模型结构验证通过")

        # 检查输入输出形状
        session = ort.InferenceSession(onnx_file)
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()

        print(f"📊 模型信息:")
        print(f"   输入: {input_info.name} - {input_info.shape} - {input_info.type}")
        for i, output in enumerate(output_info):
            print(f"   输出{i}: {output.name} - {output.shape} - {output.type}")

    except Exception as e:
        print(f"❌ ONNX模型验证失败: {str(e)}")


def preprocess_image_correctly(image_path, target_size=640):
    """
    正确的图像预处理函数 - 确保与PT模型完全一致
    """
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")

    # 保存原始尺寸用于后处理
    original_shape = img.shape[:2]

    # YOLOv8标准预处理
    # 1. 转换颜色空间 BGR -> RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2. 等比例缩放并填充
    h, w = img_rgb.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(h * scale), int(w * scale)

    # 缩放
    img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 创建640x640的画布并居中放置图像
    img_padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)  # 灰色填充

    # 计算偏移量（居中）
    dy = (target_size - new_h) // 2
    dx = (target_size - new_w) // 2

    img_padded[dy:dy + new_h, dx:dx + new_w] = img_resized

    # 3. 归一化到 [0, 1]
    img_normalized = img_padded.astype(np.float32) / 255.0

    # 4. 转换维度 HWC -> CHW
    img_chw = np.transpose(img_normalized, (2, 0, 1))

    # 5. 添加批次维度 CHW -> BCHW
    img_batch = np.expand_dims(img_chw, axis=0)

    # 返回预处理后的图像和缩放信息
    return img_batch, scale, (dx, dy), original_shape


def test_pt_vs_onnx_debug(pt_file, onnx_file, test_image, conf_threshold=0.25):
    """深度调试版本 - 分析ONNX真实输出格式"""
    print(f"\n🔧 深度调试ONNX输出...")
    
    # 1. PT模型推理
    print("\n--- PT模型推理 ---")
    pt_model = YOLO(pt_file)
    pt_results = pt_model(test_image, conf=conf_threshold, verbose=False)

    print(f"PT检测结果数量: {len(pt_results[0].boxes) if pt_results[0].boxes is not None else 0}")
    if pt_results[0].boxes is not None:
        for i, box in enumerate(pt_results[0].boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            print(f"  框{i}: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] conf={conf:.3f} cls={cls}")

    # 2. ONNX深度调试
    print("\n--- ONNX深度调试 ---")
    session = ort.InferenceSession(onnx_file)
    input_tensor, scale, padding, original_shape = preprocess_image_correctly(test_image)
    
    print(f"预处理信息:")
    print(f"  原图尺寸: {original_shape}")
    print(f"  缩放比例: {scale}")
    print(f"  填充偏移: {padding}")
    
    # 推理
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})
    
    # 详细分析输出
    detections = outputs[0][0]  # [300, 6]
    print(f"ONNX原始输出形状: {detections.shape}")
    
    # 查看前几个有效检测的原始数据
    valid_indices = np.where(detections[:, 4] > 0)[0]
    print(f"有效检测索引: {valid_indices[:10]}")  # 只看前10个
    
    for i, idx in enumerate(valid_indices[:5]):  # 只看前5个有效检测
        det = detections[idx]
        print(f"检测{i}原始数据: [{det[0]:.2f}, {det[1]:.2f}, {det[2]:.2f}, {det[3]:.2f}, {det[4]:.3f}, {det[5]:.0f}]")
    
    # 尝试不同的坐标解释方式
    print(f"\n🧪 测试不同坐标格式...")
    
    valid_detections = detections[detections[:, 4] > conf_threshold]
    if len(valid_detections) == 0:
        print("没有有效检测")
        return
        
    boxes = valid_detections[:, :4]
    confs = valid_detections[:, 4]
    
    print(f"\n方案1: 假设输出是xyxy格式 (直接使用)")
    test_boxes_1 = boxes.copy()
    dx, dy = padding
    test_boxes_1[:, [0, 2]] -= dx
    test_boxes_1[:, [1, 3]] -= dy
    test_boxes_1 /= scale
    
    print("结果:")
    for i in range(len(test_boxes_1)):
        x1, y1, x2, y2 = test_boxes_1[i]
        print(f"  框{i}: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] conf={confs[i]:.3f}")
    
    print(f"\n方案2: 假设输出是cxcywh格式")
    test_boxes_2 = boxes.copy()
    # 转换 cxcywh -> xyxy
    cx, cy, w, h = test_boxes_2[:, 0], test_boxes_2[:, 1], test_boxes_2[:, 2], test_boxes_2[:, 3]
    test_boxes_2[:, 0] = cx - w/2  # x1
    test_boxes_2[:, 1] = cy - h/2  # y1  
    test_boxes_2[:, 2] = cx + w/2  # x2
    test_boxes_2[:, 3] = cy + h/2  # y2
    
    # 坐标转换
    test_boxes_2[:, [0, 2]] -= dx
    test_boxes_2[:, [1, 3]] -= dy
    test_boxes_2 /= scale
    
    print("结果:")
    for i in range(len(test_boxes_2)):
        x1, y1, x2, y2 = test_boxes_2[i]
        print(f"  框{i}: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] conf={confs[i]:.3f}")


def test_pt_vs_onnx_v2(pt_file, onnx_file, test_image, conf_threshold=0.25, iou_threshold=0.5):
    """改进版本的PT vs ONNX对比测试"""
    print(f"\n🔍 开始对比测试（改进版）...")
    print(f"PT模型: {pt_file}")
    print(f"ONNX模型: {onnx_file}")
    print(f"测试图像: {test_image}")

    # 1. PT模型推理 - 保持原样
    print("\n--- PT模型推理 ---")
    pt_model = YOLO(pt_file)
    pt_results = pt_model(test_image, conf=conf_threshold, verbose=False)

    print(f"PT检测结果数量: {len(pt_results[0].boxes) if pt_results[0].boxes is not None else 0}")
    if pt_results[0].boxes is not None:
        for i, box in enumerate(pt_results[0].boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            print(f"  框{i}: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] conf={conf:.3f} cls={cls}")

    # 2. ONNX模型推理 - 改进版
    print("\n--- ONNX模型推理（改进版）---")
    session = ort.InferenceSession(onnx_file)

    # 预处理图像 - 使用与PT完全相同的预处理
    input_tensor, scale, padding, original_shape = preprocess_image_correctly(test_image)
    
    # 推理
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})

    # 解析输出
    if len(outputs) == 1 and outputs[0].shape[1] == 300:
        # 如果输出是 [1, 300, 6] 格式（带NMS的输出）
        detections = outputs[0][0]  # [300, 6]
        
        # 过滤有效检测
        valid_mask = detections[:, 4] > conf_threshold
        if not valid_mask.any():
            print("❌ ONNX 没有检测到任何目标")
            return
            
        valid_detections = detections[valid_mask]
        print(f"ONNX有效检测数量: {len(valid_detections)}")
        
        # 提取坐标和信息 - 修正：输出已经是xyxy格式
        boxes = valid_detections[:, :4]  # [x1, y1, x2, y2] 已经是xyxy格式！
        confs = valid_detections[:, 4]
        class_ids = valid_detections[:, 5].astype(int)
        
        # 坐标转换：从640x640转换回原图（不需要cxcywh转换）
        dx, dy = padding
        boxes[:, [0, 2]] -= dx  # x坐标减去x偏移
        boxes[:, [1, 3]] -= dy  # y坐标减去y偏移
        boxes /= scale  # 缩放回原图尺寸
        
        # 裁剪到图像范围
        h_ori, w_ori = original_shape
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w_ori)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h_ori)
        
        # 输出结果
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            conf = confs[i]
            cls = class_ids[i]
            print(f"  框{i}: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] conf={conf:.3f} cls={cls}")
            
    else:
        print("❌ 未识别的ONNX输出格式")
        for i, output in enumerate(outputs):
            print(f"输出{i}形状: {output.shape}")


def test_pt_vs_onnx(pt_file, onnx_file, test_image, conf_threshold=0.25, iou_threshold=0.5):
    print(f"\n🔍 开始对比测试...")
    print(f"PT模型: {pt_file}")
    print(f"ONNX模型: {onnx_file}")
    print(f"测试图像: {test_image}")

    # 1. PT模型推理
    print("\n--- PT模型推理 ---")
    pt_model = YOLO(pt_file)
    pt_results = pt_model(test_image, conf=conf_threshold, verbose=False)

    print(f"PT检测结果数量: {len(pt_results[0].boxes) if pt_results[0].boxes is not None else 0}")
    if pt_results[0].boxes is not None:
        for i, box in enumerate(pt_results[0].boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            print(f"  框{i}: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] conf={conf:.3f} cls={cls}")

    # 2. ONNX模型推理（适配 [1, 300, 6] 输出）
    print("\n--- ONNX模型推理（修正版）---")
    session = ort.InferenceSession(onnx_file)

    # 预处理图像
    input_tensor, scale, padding, original_shape = preprocess_image_correctly(test_image)
    h_ori, w_ori = original_shape

    # 推理
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})

    # 解析输出: [1, 300, 6] -> [300, 6]
    output = outputs[0][0]  # shape: (300, 6)
    print(f"ONNX原始输出形状: {output.shape}")

    # 过滤有效检测（conf > 0）
    valid_detections = output[output[:, 4] > 0]
    print(f"ONNX原始检测数量: {len(valid_detections)}")

    if len(valid_detections) == 0:
        print("❌ ONNX 没有检测到任何目标")
        return

    # 提取数据
    boxes = valid_detections[:, :4]  # [cx, cy, w, h]
    confs = valid_detections[:, 4]
    class_ids = valid_detections[:, 5]

    # 转换为 xyxy
    xyxy_boxes = np.empty_like(boxes)
    xyxy_boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
    xyxy_boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
    xyxy_boxes[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
    xyxy_boxes[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2

    # ⚠️ 关键：这些坐标是 640x640 上的，需要还原到原图尺寸
    # 正确的坐标转换方式：
    dx, dy = padding
    
    # 1. 先减去 padding 偏移（因为检测框是在 padded 图像上预测的）
    xyxy_boxes[:, [0, 2]] -= dx  # x坐标减去x偏移
    xyxy_boxes[:, [1, 3]] -= dy  # y坐标减去y偏移
    
    # 2. 然后缩放回原图尺寸
    xyxy_boxes /= scale

    # 裁剪到图像范围内
    xyxy_boxes[:, [0, 2]] = np.clip(xyxy_boxes[:, [0, 2]], 0, w_ori)
    xyxy_boxes[:, [1, 3]] = np.clip(xyxy_boxes[:, [1, 3]], 0, h_ori)

    # 再次过滤置信度
    conf_mask = confs >= conf_threshold
    xyxy_boxes = xyxy_boxes[conf_mask]
    confs = confs[conf_mask]
    class_ids = class_ids[conf_mask]

    if len(confs) == 0:
        print("❌ 过滤后无有效检测框")
        return

    # ✅ 执行 NMS（即使 ONNX 有 NMS，也可能不彻底）
    indices = cv2.dnn.NMSBoxes(xyxy_boxes.tolist(), confs.tolist(), conf_threshold, iou_threshold)
    if len(indices) > 0:
        indices = indices.flatten()
        nms_boxes = xyxy_boxes[indices]
        nms_confs = confs[indices]
        nms_classes = class_ids[indices]

        print(f"ONNX NMS后检测数量: {len(nms_boxes)}")
        for i in range(len(nms_boxes)):
            x1, y1, x2, y2 = nms_boxes[i]
            print(f"  框{i}: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] conf={nms_confs[i]:.3f} cls={int(nms_classes[i])}")
    else:
        print("❌ NMS后无有效检测框")

def main():
    # 🔧 配置参数
    pt_file = r"C:\Users\52953\Desktop\校园安全\校园火险与暴力全图训练\outputs\yolo_project_20250815-223855\yolo_project\weights\best.pt"
    test_image = r"C:\Users\52953\Desktop\fire_smoke\fire_smoke2标记好的数据集\images\train\0_9.jpg"  # 测试图像路径（请改为实际存在的图像）

    try:
        # 1. 导出ONNX模型
        print("🚀 第一步：导出ONNX模型")
        onnx_file = export_pt_to_onnx_fixed(pt_file, imgsz=640, opset_version=11)

        if onnx_file and os.path.exists(onnx_file):
            print(f"✅ ONNX模型已保存: {onnx_file}")

            # 2. 对比测试（如果有测试图像）
            if os.path.exists(test_image):
                print("\n🚀 第二步：深度调试测试")
                # 首先进行深度调试
                test_pt_vs_onnx_debug(pt_file, onnx_file, test_image)
                
                print("\n🚀 第三步：对比测试")
                # 使用改进版测试函数
                test_pt_vs_onnx_v2(pt_file, onnx_file, test_image)
            else:
                print(f"⚠️  测试图像不存在: {test_image}")
                print("跳过对比测试，但ONNX模型已成功导出")

                # 🔧 提供一个简单的ONNX推理测试
                print("\n🧪 执行简单的ONNX推理测试...")
                test_onnx_inference_only(onnx_file)
        else:
            print("❌ ONNX导出失败")

    except Exception as e:
        print(f"❌ 执行失败: {str(e)}")
        import traceback
        traceback.print_exc()


# 🔧 额外的调试函数
def test_onnx_inference_only(onnx_file):
    """仅测试ONNX模型推理是否正常工作"""
    try:
        session = ort.InferenceSession(onnx_file)

        # 创建随机输入测试
        input_shape = session.get_inputs()[0].shape
        print(f"📊 输入形状: {input_shape}")

        # 生成随机测试数据
        test_input = np.random.rand(*input_shape).astype(np.float32)

        # 推理测试
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: test_input})

        print(f"✅ ONNX推理测试成功!")
        for i, output in enumerate(outputs):
            print(f"   输出{i}形状: {output.shape}")
            print(f"   输出{i}数据类型: {output.dtype}")

        return True

    except Exception as e:
        print(f"❌ ONNX推理测试失败: {str(e)}")
        return False


def debug_onnx_output(onnx_file, test_image):
    """调试ONNX输出的详细信息"""
    session = ort.InferenceSession(onnx_file)
    input_tensor, _, _, _ = preprocess_image_correctly(test_image)

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})

    print(f"🔍 ONNX输出调试信息:")
    for i, output in enumerate(outputs):
        print(f"  输出{i}形状: {output.shape}")
        print(f"  输出{i}数据类型: {output.dtype}")
        print(f"  输出{i}值范围: [{output.min():.6f}, {output.max():.6f}]")
        if len(output.shape) >= 2:
            print(f"  输出{i}前5个值: {output.flatten()[:5]}")


if __name__ == '__main__':
    main()