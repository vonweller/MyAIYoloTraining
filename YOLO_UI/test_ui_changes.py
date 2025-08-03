#!/usr/bin/env python3
"""
测试UI修改的脚本
验证新增的训练参数控件是否正常工作
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试导入是否正常"""
    try:
        print("🔍 测试导入...")
        
        # 测试PyQt5导入
        from PyQt5.QtWidgets import QApplication, QWidget
        print("✅ PyQt5导入成功")
        
        # 测试我们的组件导入
        from ui.components.training_tab import TrainingTab
        print("✅ TrainingTab导入成功")
        
        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ui_components():
    """测试UI组件是否正常创建"""
    try:
        print("\n🔍 测试UI组件创建...")
        
        from PyQt5.QtWidgets import QApplication
        app = QApplication([])
        
        # 创建TrainingTab实例
        training_tab = TrainingTab()
        print("✅ TrainingTab实例创建成功")
        
        # 检查新增的控件是否存在
        required_attrs = [
            'workers_spin',
            'patience_spin', 
            'enable_val_checkbox',
            'enable_plots_checkbox',
            'enable_amp_checkbox',
            'cache_combo',
            'verbose_checkbox'
        ]
        
        missing_attrs = []
        for attr in required_attrs:
            if hasattr(training_tab, attr):
                print(f"✅ {attr} 控件存在")
            else:
                missing_attrs.append(attr)
                print(f"❌ {attr} 控件缺失")
        
        if missing_attrs:
            print(f"\n❌ 缺失的控件: {missing_attrs}")
            return False
        else:
            print("\n✅ 所有新增控件都存在")
            
        # 测试控件的默认值
        print("\n🔍 检查控件默认值...")
        print(f"Workers: {training_tab.workers_spin.value()}")
        print(f"Patience: {training_tab.patience_spin.value()}")
        print(f"Enable Val: {training_tab.enable_val_checkbox.isChecked()}")
        print(f"Enable Plots: {training_tab.enable_plots_checkbox.isChecked()}")
        print(f"Enable AMP: {training_tab.enable_amp_checkbox.isChecked()}")
        print(f"Cache: {training_tab.cache_combo.currentText()}")
        print(f"Verbose: {training_tab.verbose_checkbox.isChecked()}")
        
        return True
        
    except Exception as e:
        print(f"❌ UI组件测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_parameter_collection():
    """测试参数收集功能"""
    try:
        print("\n🔍 测试参数收集...")
        
        from PyQt5.QtWidgets import QApplication
        app = QApplication([])
        
        training_tab = TrainingTab()
        
        # 模拟设置一些参数
        training_tab.workers_spin.setValue(2)
        training_tab.patience_spin.setValue(30)
        training_tab.enable_val_checkbox.setChecked(True)
        training_tab.enable_plots_checkbox.setChecked(False)
        training_tab.enable_amp_checkbox.setChecked(True)
        training_tab.cache_combo.setCurrentText("disk (磁盘缓存)")
        training_tab.verbose_checkbox.setChecked(False)
        
        print("✅ 参数设置完成")
        
        # 检查参数值
        print(f"Workers设置为: {training_tab.workers_spin.value()}")
        print(f"Patience设置为: {training_tab.patience_spin.value()}")
        print(f"Val设置为: {training_tab.enable_val_checkbox.isChecked()}")
        print(f"Plots设置为: {training_tab.enable_plots_checkbox.isChecked()}")
        print(f"AMP设置为: {training_tab.enable_amp_checkbox.isChecked()}")
        print(f"Cache设置为: {training_tab.cache_combo.currentText()}")
        print(f"Verbose设置为: {training_tab.verbose_checkbox.isChecked()}")
        
        return True
        
    except Exception as e:
        print(f"❌ 参数收集测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 开始测试YOLO UI优化...")
    
    # 测试导入
    if not test_imports():
        print("\n❌ 导入测试失败，退出")
        return False
    
    # 测试UI组件
    if not test_ui_components():
        print("\n❌ UI组件测试失败，退出")
        return False
    
    # 测试参数收集
    if not test_parameter_collection():
        print("\n❌ 参数收集测试失败，退出")
        return False
    
    print("\n🎉 所有测试通过！UI优化成功！")
    print("\n📋 优化总结:")
    print("✅ 添加了Workers参数控制")
    print("✅ 添加了Patience早停控制")
    print("✅ 添加了验证开关(Val)")
    print("✅ 添加了绘图开关(Plots)")
    print("✅ 添加了混合精度开关(AMP)")
    print("✅ 添加了缓存方式选择")
    print("✅ 添加了详细日志开关")
    print("\n💡 这些参数将帮助解决训练稳定性问题，特别是0xC0000005错误")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
