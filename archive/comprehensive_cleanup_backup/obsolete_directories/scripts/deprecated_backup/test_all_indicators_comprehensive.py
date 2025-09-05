#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from db.query_executor import get_query_executor
from db.sql_manager import QueryType
"""
88个指标全面测试脚本
检查所有指标的评分状态，识别需要修复的指标
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import json
from datetime import datetime

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from indicators.complete_indicator_registry import complete_registry
from utils.dependency_injection import get_service
from utils.logger import get_logger

logger = get_logger(__name__)

def get_test_data_Comprehensive(code: str = "000001", limit: int = 100) -> pd.DataFrame:
    """获取测试数据"""
    try:
        container = get_container()
        data_access = container.get_data_access()
        
        query = f"""
        SELECT date, open, high, low, close, volume, turnover
        FROM stock_info WHERE 1=1
        WHERE code = '{code}'
          AND level = '日线'
        ORDER BY date DESC
        LIMIT {limit}
        """
        
        df = data_access.query_dataframe(query)
        if df.empty:
            logger.warning(f"未找到股票 {code} 的数据")
            return pd.DataFrame()
        
        # 按日期升序排列
        df = df.sort_values('date').reset_index(drop=True)
        
        # 确保数据类型正确
        df['date'] = pd.to_datetime(df['date'])
        for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
        
    except Exception as e:
        logger.error(f"获取测试数据失败: {e}")
        return pd.DataFrame()

def test_single_indicator_Comprehensive(indicator_name: str, test_data: pd.DataFrame) -> Dict:
    """测试单个指标"""
    result = {
        'name': indicator_name,
        'status': 'unknown',
        'error': None,
        'has_calculation': False,
        'has_result': False,
        'score_info': {
            'is_fixed_50': False,
            'unique_count': 0,
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'range': 0.0
        },
        'recommendation': 'unknown'
    }
    
    try:
        # 创建指标实例
        indicator = complete_registry.create_indicator(indicator_name)
        if indicator is None:
            result['error'] = "无法创建指标实例"
            result['status'] = 'error'
            result['recommendation'] = 'needs_fix'
            return result
        
        # 测试计算功能
        calc_result = indicator.calculate(test_data)
        result['has_calculation'] = True
        
        # 检查是否有计算结果
        result['has_result'] = indicator.has_result()
        
        if result['has_result']:
            # 计算评分
            scores = indicator.calculate_raw_score(test_data)
            
            if len(scores) > 0:
                valid_scores = scores.dropna()
                if len(valid_scores) > 0:
                    result['score_info']['unique_count'] = len(valid_scores.unique())
                    result['score_info']['mean'] = float(valid_scores.mean())
                    result['score_info']['std'] = float(valid_scores.std())
                    result['score_info']['min'] = float(valid_scores.min())
                    result['score_info']['max'] = float(valid_scores.max())
                    result['score_info']['range'] = float(valid_scores.max() - valid_scores.min())
                    
                    # 检查是否固定50分
                    if (result['score_info']['unique_count'] == 1 and 
                        abs(result['score_info']['mean'] - 50.0) < 0.01):
                        result['score_info']['is_fixed_50'] = True
                        result['status'] = 'fixed_50_score'
                        result['recommendation'] = 'needs_fix'
                    elif result['score_info']['std'] < 1.0:
                        result['status'] = 'low_variance'
                        result['recommendation'] = 'needs_optimization'
                    elif result['score_info']['unique_count'] < 10:
                        result['status'] = 'low_diversity'
                        result['recommendation'] = 'needs_optimization'
                    else:
                        result['status'] = 'working'
                        result['recommendation'] = 'good'
                else:
                    result['status'] = 'no_valid_scores'
                    result['recommendation'] = 'needs_fix'
            else:
                result['status'] = 'no_scores'
                result['recommendation'] = 'needs_fix'
        else:
            result['status'] = 'no_result'
            result['recommendation'] = 'needs_fix'
            
    except Exception as e:
        result['error'] = str(e)
        result['status'] = 'error'
        result['recommendation'] = 'needs_fix'
        logger.warning(f"测试指标 {indicator_name} 失败: {e}")
    
    return result

def test_all_indicators_Comprehensive() -> Dict:
    """测试所有88个指标"""
    print("=" * 80)
    print("88个指标全面测试")
    print("=" * 80)
    
    # 获取测试数据
    test_data = get_test_data_Comprehensive("000001", 100)
    if test_data.empty:
        print("❌ 无法获取测试数据")
        return {}
    
    print(f"✅ 获取到测试数据: {len(test_data)} 条记录")
    
    # 获取所有指标
    indicator_names = complete_registry.get_indicator_names()
    print(f"✅ 获取到指标注册表: {len(indicator_names)} 个指标")
    
    # 测试结果
    results = {
        'test_time': datetime.now().isoformat(),
        'data_info': {
            'stock_code': '000001',
            'data_count': len(test_data),
            'date_range': f"{test_data['date'].min()} - {test_data['date'].max()}"
        },
        'indicators': {},
        'summary': {
            'total': len(indicator_names),
            'working': 0,
            'fixed_50_score': 0,
            'low_variance': 0,
            'low_diversity': 0,
            'no_result': 0,
            'no_scores': 0,
            'no_valid_scores': 0,
            'error': 0,
            'needs_fix': 0,
            'needs_optimization': 0,
            'good': 0
        }
    }
    
    # 逐个测试指标
    for i, indicator_name in enumerate(indicator_names, 1):
        print(f"\n[{i:2d}/{len(indicator_names)}] 测试指标: {indicator_name}")
        
        result = test_single_indicator_Comprehensive(indicator_name, test_data)
        results['indicators'][indicator_name] = result
        
        # 更新统计
        results['summary'][result['status']] += 1
        results['summary'][result['recommendation']] += 1
        
        # 显示结果
        status_icon = {
            'working': '✅',
            'fixed_50_score': '❌',
            'low_variance': '⚠️',
            'low_diversity': '⚠️',
            'no_result': '❌',
            'no_scores': '❌',
            'no_valid_scores': '❌',
            'error': '💥',
            'unknown': '❓'
        }.get(result['status'], '❓')
        
        print(f"    {status_icon} 状态: {result['status']}")
        if result['error']:
            print(f"    💥 错误: {result['error']}")
        elif result['has_result']:
            score_info = result['score_info']
            print(f"    📊 评分: 唯一值={score_info['unique_count']}, "
                  f"均值={score_info['mean']:.2f}, 标准差={score_info['std']:.2f}, "
                  f"范围={score_info['min']:.2f}-{score_info['max']:.2f}")
    
    return results

def print_summary_Comprehensive(results: Dict):
    """打印测试摘要"""
    print("\n" + "=" * 80)
    print("测试摘要")
    print("=" * 80)
    
    summary = results['summary']
    total = summary['total']
    
    print(f"总指标数量: {total}")
    print(f"测试时间: {results['test_time']}")
    print(f"测试数据: {results['data_info']['stock_code']}, "
          f"{results['data_info']['data_count']} 条记录, "
          f"{results['data_info']['date_range']}")
    
    print(f"\n📊 指标状态分布:")
    print(f"  ✅ 正常工作: {summary['working']} ({summary['working']/total*100:.1f}%)")
    print(f"  ❌ 固定50分: {summary['fixed_50_score']} ({summary['fixed_50_score']/total*100:.1f}%)")
    print(f"  ⚠️ 低方差: {summary['low_variance']} ({summary['low_variance']/total*100:.1f}%)")
    print(f"  ⚠️ 低多样性: {summary['low_diversity']} ({summary['low_diversity']/total*100:.1f}%)")
    print(f"  ❌ 无结果: {summary['no_result']} ({summary['no_result']/total*100:.1f}%)")
    print(f"  ❌ 无评分: {summary['no_scores']} ({summary['no_scores']/total*100:.1f}%)")
    print(f"  ❌ 无有效评分: {summary['no_valid_scores']} ({summary['no_valid_scores']/total*100:.1f}%)")
    print(f"  💥 测试错误: {summary['error']} ({summary['error']/total*100:.1f}%)")
    
    print(f"\n🔧 修复建议:")
    print(f"  🚨 需要修复: {summary['needs_fix']} ({summary['needs_fix']/total*100:.1f}%)")
    print(f"  🔧 需要优化: {summary['needs_optimization']} ({summary['needs_optimization']/total*100:.1f}%)")
    print(f"  ✅ 状态良好: {summary['good']} ({summary['good']/total*100:.1f}%)")
    
    # 计算选股率
    working_rate = (summary['good'] + summary['needs_optimization']) / total * 100
    print(f"\n📈 当前选股率: {working_rate:.1f}% ({summary['good'] + summary['needs_optimization']}/{total})")

def print_detailed_issues(results: Dict):
    """打印详细问题列表"""
    print("\n" + "=" * 80)
    print("需要修复的指标详情")
    print("=" * 80)
    
    # 按问题类型分组
    issues = {
        'fixed_50_score': [],
        'low_variance': [],
        'low_diversity': [],
        'no_result': [],
        'no_scores': [],
        'no_valid_scores': [],
        'error': []
    }
    
    for indicator_name, result in results['indicators'].items():
        if result['recommendation'] in ['needs_fix', 'needs_optimization']:
            issues[result['status']].append((indicator_name, result))
    
    # 打印各类问题
    problem_descriptions = {
        'fixed_50_score': '❌ 固定50分问题（最高优先级）',
        'no_result': '❌ 无计算结果问题',
        'no_scores': '❌ 无评分问题',
        'no_valid_scores': '❌ 无有效评分问题',
        'error': '💥 计算错误问题',
        'low_variance': '⚠️ 低方差问题（需要优化）',
        'low_diversity': '⚠️ 低多样性问题（需要优化）'
    }
    
    for issue_type, description in problem_descriptions.items():
        if issues[issue_type]:
            print(f"\n{description} ({len(issues[issue_type])} 个):")
            for indicator_name, result in issues[issue_type]:
                if issue_type == 'fixed_50_score':
                    print(f"  • {indicator_name}: 固定50分，需要实现动态评分")
                elif issue_type == 'error':
                    print(f"  • {indicator_name}: {result['error']}")
                elif issue_type in ['low_variance', 'low_diversity']:
                    score_info = result['score_info']
                    print(f"  • {indicator_name}: 唯一值={score_info['unique_count']}, "
                          f"标准差={score_info['std']:.2f}")
                else:
                    print(f"  • {indicator_name}: {result['status']}")

def save_results_Comprehensive(results: Dict):
    """保存测试结果"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/validation/all_indicators_test_{timestamp}.json"
    
    # 确保目录存在
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 测试结果已保存到: {filename}")
    except Exception as e:
        print(f"\n❌ 保存结果失败: {e}")

def main_testallindicatorscomprehensive():
    """主函数"""
    try:
        # 运行全面测试
        results = test_all_indicators_Comprehensive()
        
        if results:
            # 打印摘要
            print_summary_Comprehensive(results)
            
            # 打印详细问题
            print_detailed_issues(results)
            
            # 保存结果
            save_results_Comprehensive(results)
            
            print("\n" + "=" * 80)
            print("测试完成")
            print("=" * 80)
        else:
            print("❌ 测试失败")
            
    except Exception as e:
        logger.error(f"测试过程中出现错误: {e}")
        print(f"❌ 测试失败: {e}")

if __name__ == "__main__":
    mainTestallindicatorscomprehensive() 