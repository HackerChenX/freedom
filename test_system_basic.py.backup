#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基础系统功能测试脚本
不依赖虚拟环境，使用系统Python进行基础功能验证
"""

import sys
import os
import time
import json
from datetime import datetime

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_basic_imports():
    """测试基础导入功能"""
    print("🔍 测试基础导入功能...")
    
    try:
        # 测试基础Python模块
        import json
        import datetime
        import logging
        print("✅ 基础Python模块导入成功")
        
        # 测试项目结构
        expected_dirs = ['indicators', 'strategy', 'analysis', 'db', 'utils']
        missing_dirs = []
        
        for dir_name in expected_dirs:
            if not os.path.exists(dir_name):
                missing_dirs.append(dir_name)
        
        if missing_dirs:
            print(f"⚠️ 缺少目录: {missing_dirs}")
        else:
            print("✅ 项目目录结构完整")
            
        return True
        
    except Exception as e:
        print(f"❌ 基础导入测试失败: {e}")
        return False

def test_config_system():
    """测试配置系统"""
    print("\n🔍 测试配置系统...")
    
    try:
        # 检查配置文件是否存在
        config_files = [
            'config/config.json',
            'config/database.json',
            'config/buypoints_config.json'
        ]
        
        existing_configs = []
        for config_file in config_files:
            if os.path.exists(config_file):
                existing_configs.append(config_file)
        
        print(f"✅ 找到配置文件: {existing_configs}")
        
        # 尝试读取一个配置文件
        if existing_configs:
            with open(existing_configs[0], 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            print(f"✅ 配置文件读取成功: {existing_configs[0]}")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置系统测试失败: {e}")
        return False

def test_database_config():
    """测试数据库配置"""
    print("\n🔍 测试数据库配置...")
    
    try:
        # 检查数据库相关文件
        db_files = [
            'db/__init__.py',
            'db/clickhouse_db.py',
            'db/query_executor.py'
        ]
        
        existing_db_files = []
        for db_file in db_files:
            if os.path.exists(db_file):
                existing_db_files.append(db_file)
        
        print(f"✅ 找到数据库文件: {existing_db_files}")
        
        # 检查SQL文件
        sql_dir = 'sql'
        if os.path.exists(sql_dir):
            sql_files = []
            for root, dirs, files in os.walk(sql_dir):
                for file in files:
                    if file.endswith('.sql'):
                        sql_files.append(os.path.join(root, file))
            print(f"✅ 找到SQL文件: {sql_files}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据库配置测试失败: {e}")
        return False

def test_indicator_structure():
    """测试指标结构"""
    print("\n🔍 测试指标结构...")
    
    try:
        indicators_dir = 'indicators'
        if not os.path.exists(indicators_dir):
            print(f"❌ 指标目录不存在: {indicators_dir}")
            return False
        
        # 检查指标子目录
        indicator_subdirs = []
        for item in os.listdir(indicators_dir):
            item_path = os.path.join(indicators_dir, item)
            if os.path.isdir(item_path):
                indicator_subdirs.append(item)
        
        print(f"✅ 找到指标子目录: {indicator_subdirs}")
        
        # 检查指标文件
        indicator_files = []
        for root, dirs, files in os.walk(indicators_dir):
            for file in files:
                if file.endswith('.py') and not file.startswith('__'):
                    indicator_files.append(os.path.join(root, file))
        
        print(f"✅ 找到指标文件数量: {len(indicator_files)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 指标结构测试失败: {e}")
        return False

def test_strategy_structure():
    """测试策略结构"""
    print("\n🔍 测试策略结构...")
    
    try:
        strategy_dir = 'strategy'
        if not os.path.exists(strategy_dir):
            print(f"❌ 策略目录不存在: {strategy_dir}")
            return False
        
        # 检查策略文件
        strategy_files = []
        for item in os.listdir(strategy_dir):
            if item.endswith('.py') and not item.startswith('__'):
                strategy_files.append(item)
        
        print(f"✅ 找到策略文件: {strategy_files}")
        
        # 检查配置文件
        config_strategy_dir = 'config/strategies'
        if os.path.exists(config_strategy_dir):
            strategy_configs = []
            for item in os.listdir(config_strategy_dir):
                if item.endswith(('.yaml', '.json')):
                    strategy_configs.append(item)
            print(f"✅ 找到策略配置: {strategy_configs}")
        
        return True
        
    except Exception as e:
        print(f"❌ 策略结构测试失败: {e}")
        return False

def test_analysis_structure():
    """测试分析模块结构"""
    print("\n🔍 测试分析模块结构...")
    
    try:
        analysis_dir = 'analysis'
        if not os.path.exists(analysis_dir):
            print(f"❌ 分析目录不存在: {analysis_dir}")
            return False
        
        # 检查分析子目录
        analysis_subdirs = []
        for item in os.listdir(analysis_dir):
            item_path = os.path.join(analysis_dir, item)
            if os.path.isdir(item_path):
                analysis_subdirs.append(item)
        
        print(f"✅ 找到分析子目录: {analysis_subdirs}")
        
        # 检查买点分析
        buypoints_dir = 'analysis/buypoints'
        if os.path.exists(buypoints_dir):
            buypoint_files = []
            for item in os.listdir(buypoints_dir):
                if item.endswith('.py') and not item.startswith('__'):
                    buypoint_files.append(item)
            print(f"✅ 找到买点分析文件: {buypoint_files}")
        
        return True
        
    except Exception as e:
        print(f"❌ 分析结构测试失败: {e}")
        return False

def test_utils_structure():
    """测试工具模块结构"""
    print("\n🔍 测试工具模块结构...")
    
    try:
        utils_dir = 'utils'
        if not os.path.exists(utils_dir):
            print(f"❌ 工具目录不存在: {utils_dir}")
            return False
        
        # 检查工具文件
        util_files = []
        for item in os.listdir(utils_dir):
            if item.endswith('.py') and not item.startswith('__'):
                util_files.append(item)
        
        print(f"✅ 找到工具文件: {util_files}")
        
        return True
        
    except Exception as e:
        print(f"❌ 工具结构测试失败: {e}")
        return False

def run_basic_system_test():
    """运行基础系统测试"""
    print("=" * 80)
    print("🚀 开始基础系统功能测试")
    print("=" * 80)
    
    test_results = {
        'test_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'results': {},
        'summary': {}
    }
    
    tests = [
        ('基础导入', test_basic_imports),
        ('配置系统', test_config_system),
        ('数据库配置', test_database_config),
        ('指标结构', test_indicator_structure),
        ('策略结构', test_strategy_structure),
        ('分析结构', test_analysis_structure),
        ('工具结构', test_utils_structure)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            test_results['results'][test_name] = {
                'status': 'PASS' if result else 'FAIL',
                'success': result
            }
            if result:
                passed_tests += 1
        except Exception as e:
            test_results['results'][test_name] = {
                'status': 'ERROR',
                'error': str(e),
                'success': False
            }
    
    # 生成总结
    success_rate = (passed_tests / total_tests) * 100
    test_results['summary'] = {
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'failed_tests': total_tests - passed_tests,
        'success_rate': success_rate
    }
    
    print("\n" + "=" * 80)
    print("📋 测试结果总结")
    print("=" * 80)
    print(f"总测试数: {total_tests}")
    print(f"通过测试: {passed_tests}")
    print(f"失败测试: {total_tests - passed_tests}")
    print(f"成功率: {success_rate:.1f}%")
    
    # 保存测试结果
    result_file = f"basic_system_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n📄 测试结果已保存到: {result_file}")
    
    if success_rate >= 80:
        print("\n✅ 基础系统测试通过！系统结构完整。")
        return True
    else:
        print("\n⚠️ 基础系统测试发现问题，需要修复。")
        return False

if __name__ == "__main__":
    run_basic_system_test() 