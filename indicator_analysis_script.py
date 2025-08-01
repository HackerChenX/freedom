#!/usr/bin/env python3
"""
Ultra Think方法论：全面指标分析脚本
系统性分析所有113个技术指标的状态
"""

import subprocess
import sys
import os
from pathlib import Path

def run_test(test_file):
    """运行单个测试文件并返回结果"""
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pytest', test_file, 
            '--tb=no', '-q', '--disable-warnings'
        ], capture_output=True, text=True, timeout=30)
        
        output = result.stdout + result.stderr
        
        # 解析测试结果
        if 'passed' in output and 'failed' not in output and 'error' not in output:
            # 提取通过的测试数量
            lines = output.split('\n')
            for line in lines:
                if 'passed' in line and ('failed' not in line or '0 failed' in line):
                    return 'PASS', line.strip()
        elif 'failed' in output or 'error' in output:
            # 提取失败信息
            lines = output.split('\n')
            for line in lines:
                if ('failed' in line and 'passed' in line) or 'error' in line:
                    return 'FAIL', line.strip()
        
        return 'UNKNOWN', output[:200] + '...' if len(output) > 200 else output
        
    except subprocess.TimeoutExpired:
        return 'TIMEOUT', 'Test timed out after 30 seconds'
    except Exception as e:
        return 'ERROR', str(e)

def analyze_all_indicators():
    """分析所有指标的测试状态"""
    
    # 获取所有测试文件
    test_files = []
    test_dir = Path('tests/unit')
    
    for test_file in test_dir.glob('test_*.py'):
        if test_file.name not in ['test_config_manager.py', 'test_data_manager.py', 
                                  'test_data_service_interfaces.py', 'test_result_validator.py',
                                  'test_strategy_parser.py']:  # 跳过已知有问题的非指标测试
            test_files.append(str(test_file))
    
    # 分析结果
    results = {
        'PASS': [],
        'FAIL': [],
        'TIMEOUT': [],
        'ERROR': [],
        'UNKNOWN': []
    }
    
    print("🎯 Ultra Think全面指标分析开始...")
    print(f"📊 总测试文件数: {len(test_files)}")
    print("=" * 80)
    
    for i, test_file in enumerate(sorted(test_files), 1):
        test_name = Path(test_file).stem.replace('test_', '')
        print(f"[{i:2d}/{len(test_files)}] 分析 {test_name}...", end=' ')
        
        status, details = run_test(test_file)
        results[status].append((test_name, details))
        
        # 状态图标
        status_icons = {
            'PASS': '✅',
            'FAIL': '❌', 
            'TIMEOUT': '⏰',
            'ERROR': '🔥',
            'UNKNOWN': '❓'
        }
        
        print(f"{status_icons.get(status, '❓')} {status}")
        if status != 'PASS':
            print(f"    详情: {details[:100]}...")
    
    print("=" * 80)
    print("📊 Ultra Think分析结果汇总:")
    print(f"✅ 完全通过: {len(results['PASS'])} 个")
    print(f"❌ 测试失败: {len(results['FAIL'])} 个") 
    print(f"⏰ 超时错误: {len(results['TIMEOUT'])} 个")
    print(f"🔥 系统错误: {len(results['ERROR'])} 个")
    print(f"❓ 状态未知: {len(results['UNKNOWN'])} 个")
    
    return results

if __name__ == "__main__":
    results = analyze_all_indicators()
    
    # 详细分类报告
    print("\n" + "=" * 80)
    print("🎯 Ultra Think详细分类分析:")
    
    categories = [
        ('✅ 完全健康指标', results['PASS']),
        ('❌ 测试失败指标', results['FAIL']),
        ('⏰ 超时错误指标', results['TIMEOUT']),
        ('🔥 系统错误指标', results['ERROR']),
        ('❓ 状态未知指标', results['UNKNOWN'])
    ]
    
    for category_name, indicators in categories:
        if indicators:
            print(f"\n{category_name} ({len(indicators)}个):")
            for name, details in indicators:
                print(f"  - {name}: {details[:80]}...")