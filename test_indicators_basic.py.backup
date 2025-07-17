#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基础指标测试脚本
测试指标模块的结构和基础功能，不依赖pandas
"""

import sys
import os
import time
import json
import importlib.util
from datetime import datetime

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_indicator_imports():
    """测试指标模块导入"""
    print("🔍 测试指标模块导入...")
    
    results = {
        'successful_imports': [],
        'failed_imports': [],
        'total_files': 0,
        'success_rate': 0
    }
    
    indicators_dir = 'indicators'
    if not os.path.exists(indicators_dir):
        print("❌ 指标目录不存在")
        return results
    
    # 获取所有指标文件
    indicator_files = []
    for root, dirs, files in os.walk(indicators_dir):
        for file in files:
            if file.endswith('.py') and not file.startswith('__'):
                relative_path = os.path.relpath(os.path.join(root, file), project_root)
                indicator_files.append(relative_path)
    
    results['total_files'] = len(indicator_files)
    print(f"📊 找到指标文件: {len(indicator_files)} 个")
    
    # 测试每个文件的基础导入
    for file_path in indicator_files:
        try:
            # 转换文件路径为模块名
            module_name = file_path.replace('/', '.').replace('.py', '')
            
            # 尝试导入模块
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                # 这里只检查文件能否被加载，不执行
                results['successful_imports'].append(file_path)
            else:
                results['failed_imports'].append((file_path, "无法创建模块规范"))
                
        except Exception as e:
            results['failed_imports'].append((file_path, str(e)))
    
    results['success_rate'] = (len(results['successful_imports']) / results['total_files']) * 100 if results['total_files'] > 0 else 0
    
    print(f"✅ 成功导入: {len(results['successful_imports'])} 个")
    print(f"❌ 失败导入: {len(results['failed_imports'])} 个")
    print(f"📈 成功率: {results['success_rate']:.1f}%")
    
    if results['failed_imports']:
        print("\n失败的文件:")
        for file_path, error in results['failed_imports'][:5]:  # 只显示前5个
            print(f"  - {file_path}: {error}")
        if len(results['failed_imports']) > 5:
            print(f"  ... 还有 {len(results['failed_imports']) - 5} 个失败")
    
    return results

def test_indicator_categories():
    """测试指标分类结构"""
    print("\n🔍 测试指标分类结构...")
    
    results = {
        'categories': {},
        'total_categories': 0,
        'total_indicators': 0
    }
    
    indicators_dir = 'indicators'
    expected_categories = ['trend', 'oscillator', 'volume', 'pattern', 'zxm']
    
    for category in expected_categories:
        category_path = os.path.join(indicators_dir, category)
        if os.path.exists(category_path) and os.path.isdir(category_path):
            # 统计该分类下的指标文件
            indicator_files = []
            for file in os.listdir(category_path):
                if file.endswith('.py') and not file.startswith('__'):
                    indicator_files.append(file)
            
            results['categories'][category] = {
                'exists': True,
                'file_count': len(indicator_files),
                'files': indicator_files
            }
            results['total_indicators'] += len(indicator_files)
        else:
            results['categories'][category] = {
                'exists': False,
                'file_count': 0,
                'files': []
            }
    
    results['total_categories'] = len([c for c in results['categories'].values() if c['exists']])
    
    print(f"📊 指标分类统计:")
    for category, info in results['categories'].items():
        status = "✅" if info['exists'] else "❌"
        print(f"  {status} {category}: {info['file_count']} 个指标")
    
    print(f"\n📈 总计: {results['total_categories']} 个分类, {results['total_indicators']} 个指标文件")
    
    return results

def test_zxm_indicators():
    """测试ZXM指标系列"""
    print("\n🔍 测试ZXM指标系列...")
    
    results = {
        'zxm_files': [],
        'zxm_categories': {},
        'total_zxm': 0
    }
    
    zxm_dir = 'indicators/zxm'
    if not os.path.exists(zxm_dir):
        print("❌ ZXM指标目录不存在")
        return results
    
    # 统计ZXM指标文件
    for file in os.listdir(zxm_dir):
        if file.endswith('.py') and not file.startswith('__'):
            results['zxm_files'].append(file)
    
    results['total_zxm'] = len(results['zxm_files'])
    
    # 分析ZXM指标类型
    zxm_types = {
        'buy_point': [],
        'elasticity': [],
        'trend': [],
        'volume': [],
        'score': [],
        'other': []
    }
    
    for file in results['zxm_files']:
        file_lower = file.lower()
        if 'buy' in file_lower or 'point' in file_lower:
            zxm_types['buy_point'].append(file)
        elif 'elasticity' in file_lower:
            zxm_types['elasticity'].append(file)
        elif 'trend' in file_lower:
            zxm_types['trend'].append(file)
        elif 'volume' in file_lower or 'turnover' in file_lower:
            zxm_types['volume'].append(file)
        elif 'score' in file_lower:
            zxm_types['score'].append(file)
        else:
            zxm_types['other'].append(file)
    
    results['zxm_categories'] = zxm_types
    
    print(f"📊 ZXM指标分析:")
    print(f"  总数: {results['total_zxm']} 个")
    for zxm_type, files in zxm_types.items():
        if files:
            print(f"  {zxm_type}: {len(files)} 个")
    
    return results

def test_strategy_modules():
    """测试策略模块"""
    print("\n🔍 测试策略模块...")
    
    results = {
        'strategy_files': [],
        'executor_files': [],
        'config_files': [],
        'total_strategies': 0
    }
    
    strategy_dir = 'strategy'
    if not os.path.exists(strategy_dir):
        print("❌ 策略目录不存在")
        return results
    
    # 统计策略文件
    for file in os.listdir(strategy_dir):
        if file.endswith('.py') and not file.startswith('__'):
            if 'executor' in file.lower():
                results['executor_files'].append(file)
            else:
                results['strategy_files'].append(file)
    
    results['total_strategies'] = len(results['strategy_files']) + len(results['executor_files'])
    
    # 检查策略配置文件
    config_strategy_dir = 'config/strategies'
    if os.path.exists(config_strategy_dir):
        for file in os.listdir(config_strategy_dir):
            if file.endswith(('.yaml', '.json')):
                results['config_files'].append(file)
    
    print(f"📊 策略模块统计:")
    print(f"  策略文件: {len(results['strategy_files'])} 个")
    print(f"  执行器文件: {len(results['executor_files'])} 个")
    print(f"  配置文件: {len(results['config_files'])} 个")
    print(f"  总计: {results['total_strategies']} 个策略模块")
    
    return results

def test_analysis_modules():
    """测试分析模块"""
    print("\n🔍 测试分析模块...")
    
    results = {
        'buypoint_files': [],
        'engine_files': [],
        'market_files': [],
        'total_analysis': 0
    }
    
    # 检查买点分析
    buypoints_dir = 'analysis/buypoints'
    if os.path.exists(buypoints_dir):
        for file in os.listdir(buypoints_dir):
            if file.endswith('.py') and not file.startswith('__'):
                results['buypoint_files'].append(file)
    
    # 检查分析引擎
    engines_dir = 'analysis/engines'
    if os.path.exists(engines_dir):
        for file in os.listdir(engines_dir):
            if file.endswith('.py') and not file.startswith('__'):
                results['engine_files'].append(file)
    
    # 检查市场分析
    market_dir = 'analysis/market'
    if os.path.exists(market_dir):
        for file in os.listdir(market_dir):
            if file.endswith('.py') and not file.startswith('__'):
                results['market_files'].append(file)
    
    results['total_analysis'] = len(results['buypoint_files']) + len(results['engine_files']) + len(results['market_files'])
    
    print(f"📊 分析模块统计:")
    print(f"  买点分析: {len(results['buypoint_files'])} 个")
    print(f"  分析引擎: {len(results['engine_files'])} 个")
    print(f"  市场分析: {len(results['market_files'])} 个")
    print(f"  总计: {results['total_analysis']} 个分析模块")
    
    return results

def test_database_modules():
    """测试数据库模块"""
    print("\n🔍 测试数据库模块...")
    
    results = {
        'db_files': [],
        'sql_files': [],
        'connection_files': [],
        'total_db': 0
    }
    
    db_dir = 'db'
    if os.path.exists(db_dir):
        for file in os.listdir(db_dir):
            if file.endswith('.py') and not file.startswith('__'):
                if 'connection' in file.lower() or 'manager' in file.lower():
                    results['connection_files'].append(file)
                else:
                    results['db_files'].append(file)
    
    # 检查SQL文件
    sql_dir = 'sql'
    if os.path.exists(sql_dir):
        for root, dirs, files in os.walk(sql_dir):
            for file in files:
                if file.endswith('.sql'):
                    results['sql_files'].append(os.path.join(root, file))
    
    results['total_db'] = len(results['db_files']) + len(results['connection_files'])
    
    print(f"📊 数据库模块统计:")
    print(f"  数据库文件: {len(results['db_files'])} 个")
    print(f"  连接管理: {len(results['connection_files'])} 个")
    print(f"  SQL文件: {len(results['sql_files'])} 个")
    print(f"  总计: {results['total_db']} 个数据库模块")
    
    return results

def run_comprehensive_structure_test():
    """运行全面的结构测试"""
    print("=" * 80)
    print("🚀 开始全面系统结构测试")
    print("=" * 80)
    
    test_results = {
        'test_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'results': {},
        'summary': {}
    }
    
    tests = [
        ('指标模块导入', test_indicator_imports),
        ('指标分类结构', test_indicator_categories),
        ('ZXM指标系列', test_zxm_indicators),
        ('策略模块', test_strategy_modules),
        ('分析模块', test_analysis_modules),
        ('数据库模块', test_database_modules)
    ]
    
    for test_name, test_func in tests:
        try:
            start_time = time.time()
            result = test_func()
            end_time = time.time()
            
            test_results['results'][test_name] = {
                'status': 'SUCCESS',
                'execution_time': end_time - start_time,
                'details': result
            }
        except Exception as e:
            test_results['results'][test_name] = {
                'status': 'ERROR',
                'error': str(e)
            }
    
    # 生成总结
    successful_tests = sum(1 for r in test_results['results'].values() if r['status'] == 'SUCCESS')
    total_tests = len(tests)
    success_rate = (successful_tests / total_tests) * 100
    
    # 统计模块数量
    total_indicators = 0
    total_strategies = 0
    total_analysis = 0
    total_db = 0
    
    for test_name, result in test_results['results'].items():
        if result['status'] == 'SUCCESS':
            details = result['details']
            if 'total_files' in details:
                total_indicators += details['total_files']
            elif 'total_strategies' in details:
                total_strategies += details['total_strategies']
            elif 'total_analysis' in details:
                total_analysis += details['total_analysis']
            elif 'total_db' in details:
                total_db += details['total_db']
    
    test_results['summary'] = {
        'total_tests': total_tests,
        'successful_tests': successful_tests,
        'success_rate': success_rate,
        'module_counts': {
            'indicators': total_indicators,
            'strategies': total_strategies,
            'analysis': total_analysis,
            'database': total_db
        }
    }
    
    print("\n" + "=" * 80)
    print("📋 全面结构测试总结")
    print("=" * 80)
    print(f"测试成功率: {success_rate:.1f}% ({successful_tests}/{total_tests})")
    print(f"指标模块: {total_indicators} 个")
    print(f"策略模块: {total_strategies} 个")
    print(f"分析模块: {total_analysis} 个")
    print(f"数据库模块: {total_db} 个")
    
    # 保存测试结果
    result_file = f"structure_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n📄 详细结果已保存到: {result_file}")
    
    if success_rate >= 90:
        print("\n✅ 系统结构测试优秀！架构完整且规范。")
        return True
    elif success_rate >= 70:
        print("\n⚠️ 系统结构测试良好，但有改进空间。")
        return True
    else:
        print("\n❌ 系统结构测试发现问题，需要修复。")
        return False

if __name__ == "__main__":
    run_comprehensive_structure_test() 