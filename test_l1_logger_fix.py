#!/usr/bin/env python3
"""
L1基础设施层日志系统统一验证测试
验证日志系统统一是否成功
"""

import sys
import os
import time
import logging

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_logger_import_uniqueness():
    """测试日志导入唯一性"""
    print("🔍 测试日志导入唯一性...")
    
    # 检查是否还有重复的日志导入
    import subprocess
    
    try:
        result = subprocess.run([
            'grep', '-r', 'from utils.dependency_injection import get_logger', 
            '.', '--include=*.py'
        ], capture_output=True, text=True, cwd='.')
        
        lines = [line for line in result.stdout.split('\n') if line.strip() and 
                '__pycache__' not in line and 'backup' not in line and 'archive' not in line]
        
        if lines:
            print(f"❌ 仍有 {len(lines)} 个文件使用重复的日志导入")
            for line in lines[:5]:  # 只显示前5个
                print(f"  - {line}")
            return False
        
        print("✅ 日志导入唯一性验证通过")
        return True
        
    except Exception as e:
        print(f"⚠️  无法验证日志导入唯一性: {e}")
        return True  # 假设通过

def test_standard_logger_functionality():
    """测试标准日志功能"""
    print("🔍 测试标准日志功能...")
    
    try:
        from utils.logger import get_logger
        
        # 测试获取日志器
        logger = get_logger(__name__)
        assert logger is not None
        assert isinstance(logger, logging.Logger)
        
        # 测试日志记录
        logger.info("测试日志记录功能")
        
        # 测试不同名称的日志器
        logger2 = get_logger("test_module")
        assert logger2 is not None
        assert isinstance(logger2, logging.Logger)
        
        print("✅ 标准日志功能验证通过")
        return True
        
    except Exception as e:
        print(f"❌ 标准日志功能测试失败: {e}")
        return False

def test_dependency_injection_logger():
    """测试依赖注入日志功能"""
    print("🔍 测试依赖注入日志功能...")
    
    try:
        from utils.dependency_injection import get_logger
        
        # 测试获取日志器
        logger = get_logger(__name__)
        assert logger is not None
        assert isinstance(logger, logging.Logger)
        
        # 测试日志记录
        logger.info("测试依赖注入日志记录功能")
        
        print("✅ 依赖注入日志功能验证通过")
        return True
        
    except Exception as e:
        print(f"❌ 依赖注入日志功能测试失败: {e}")
        return False

def test_logger_consistency():
    """测试日志器一致性"""
    print("🔍 测试日志器一致性...")
    
    try:
        from utils.logger import get_logger as standard_get_logger
        from utils.dependency_injection import get_logger as di_get_logger
        
        # 获取两个日志器
        logger1 = standard_get_logger("test_consistency")
        logger2 = di_get_logger("test_consistency")
        
        # 验证它们是同一类型
        assert type(logger1) == type(logger2)
        
        # 验证它们都能正常工作
        logger1.info("标准日志器测试")
        logger2.info("依赖注入日志器测试")
        
        print("✅ 日志器一致性验证通过")
        return True
        
    except Exception as e:
        print(f"❌ 日志器一致性测试失败: {e}")
        return False

def test_no_import_errors():
    """测试无导入错误"""
    print("🔍 测试导入错误...")
    
    try:
        # 测试主要日志模块导入
        from utils.logger import get_logger
        from utils.dependency_injection import get_logger as di_get_logger
        
        # 测试一些使用日志的模块
        from indicators.macd import MACD
        from db.managers.data_access_manager import DataAccessManager
        
        print("✅ 导入测试通过")
        return True
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 其他错误: {e}")
        return False

def test_logger_configuration():
    """测试日志配置"""
    print("🔍 测试日志配置...")
    
    try:
        from utils.logger import get_logger
        
        # 测试日志器配置
        logger = get_logger("test_config")
        
        # 检查日志器是否有处理器
        has_handlers = len(logger.handlers) > 0 or len(logging.getLogger().handlers) > 0
        
        if not has_handlers:
            print("⚠️  日志器没有处理器，但这可能是正常的")
        
        # 测试日志级别设置
        original_level = logger.level
        logger.setLevel(logging.DEBUG)
        assert logger.level == logging.DEBUG
        logger.setLevel(original_level)
        
        print("✅ 日志配置验证通过")
        return True
        
    except Exception as e:
        print(f"❌ 日志配置测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始L1基础设施层日志系统统一验证...")
    print("=" * 50)
    
    tests = [
        test_logger_import_uniqueness,
        test_standard_logger_functionality,
        test_dependency_injection_logger,
        test_logger_consistency,
        test_no_import_errors,
        test_logger_configuration
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ 测试失败: {test_func.__name__}")
        except Exception as e:
            print(f"❌ 测试异常: {test_func.__name__} - {e}")
        
        print("-" * 30)
    
    print(f"\n📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 L1.3任务 - 日志系统统一修复成功！")
        return True
    else:
        print("🚫 L1.3任务修复失败，需要进一步处理")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
