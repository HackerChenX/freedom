#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
技术指标性能分析器

专门用于分析和识别最耗时的技术指标
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import json
from datetime import datetime

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from utils.logger import get_logger
from analysis.buypoints.buypoint_batch_analyzer import BuyPointBatchAnalyzer

logger = get_logger(__name__)


class IndicatorPerformanceProfiler:
    """技术指标性能分析器"""
    
    def __init__(self):
        self.analyzer = BuyPointBatchAnalyzer()
        self.performance_data = {}
        
    def profile_all_indicators(self, stock_data: Dict[str, pd.DataFrame], 
                             target_rows: Dict[str, int],
                             iterations: int = 3) -> Dict[str, Dict[str, float]]:
        """
        分析所有技术指标的性能
        
        Args:
            stock_data: 股票数据
            target_rows: 目标行数
            iterations: 测试迭代次数
            
        Returns:
            Dict: 指标性能数据
        """
        logger.info(f"开始分析技术指标性能，迭代次数: {iterations}")
        
        all_indicators = self.analyzer.indicator_analyzer.all_indicators
        performance_results = {}
        
        for indicator_name in all_indicators:
            logger.info(f"分析指标: {indicator_name}")
            
            try:
                # 多次测试取平均值
                times = []
                success_count = 0
                
                for i in range(iterations):
                    start_time = time.time()
                    
                    try:
                        # 创建指标实例
                        indicator = self.analyzer.indicator_analyzer.indicator_registry.create_indicator(indicator_name)
                        if indicator is None:
                            indicator = self.analyzer.indicator_analyzer.indicator_factory.create_indicator(indicator_name)
                        
                        if indicator is None:
                            logger.warning(f"无法创建指标: {indicator_name}")
                            break
                        
                        # 计算指标（使用日线数据）
                        if 'daily' in stock_data and stock_data['daily'] is not None:
                            df_copy = stock_data['daily'].copy()
                            result = indicator.calculate(df_copy)
                            
                            if result is not None and not result.empty:
                                success_count += 1
                        
                        end_time = time.time()
                        times.append(end_time - start_time)
                        
                    except Exception as e:
                        logger.warning(f"指标 {indicator_name} 计算失败: {e}")
                        end_time = time.time()
                        times.append(end_time - start_time)
                
                if times:
                    avg_time = sum(times) / len(times)
                    min_time = min(times)
                    max_time = max(times)
                    success_rate = success_count / len(times)
                    
                    performance_results[indicator_name] = {
                        'avg_time': avg_time,
                        'min_time': min_time,
                        'max_time': max_time,
                        'success_rate': success_rate,
                        'total_iterations': len(times),
                        'successful_iterations': success_count
                    }
                    
                    logger.info(f"指标 {indicator_name}: 平均 {avg_time:.4f}s, 成功率 {success_rate:.1%}")
                
            except Exception as e:
                logger.error(f"分析指标 {indicator_name} 时出错: {e}")
        
        return performance_results
    
    def identify_slowest_indicators(self, performance_results: Dict[str, Dict[str, float]], 
                                  top_n: int = 10) -> List[Tuple[str, float]]:
        """
        识别最慢的指标
        
        Args:
            performance_results: 性能测试结果
            top_n: 返回前N个最慢的指标
            
        Returns:
            List: (指标名, 平均时间) 的列表
        """
        # 按平均时间排序
        sorted_indicators = sorted(
            [(name, data['avg_time']) for name, data in performance_results.items() 
             if data['success_rate'] > 0],
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_indicators[:top_n]
    
    def analyze_indicator_complexity(self, performance_results: Dict[str, Dict[str, float]]) -> Dict[str, str]:
        """
        分析指标复杂度分类
        
        Args:
            performance_results: 性能测试结果
            
        Returns:
            Dict: 指标复杂度分类
        """
        complexity_classification = {}
        
        # 计算时间阈值
        all_times = [data['avg_time'] for data in performance_results.values() if data['success_rate'] > 0]
        if not all_times:
            return complexity_classification
        
        avg_time = np.mean(all_times)
        std_time = np.std(all_times)
        
        high_threshold = avg_time + std_time
        low_threshold = avg_time - std_time
        
        for indicator_name, data in performance_results.items():
            if data['success_rate'] == 0:
                complexity_classification[indicator_name] = 'ERROR'
            elif data['avg_time'] > high_threshold:
                complexity_classification[indicator_name] = 'HIGH'
            elif data['avg_time'] < low_threshold:
                complexity_classification[indicator_name] = 'LOW'
            else:
                complexity_classification[indicator_name] = 'MEDIUM'
        
        return complexity_classification
    
    def generate_optimization_recommendations(self, performance_results: Dict[str, Dict[str, float]],
                                            slowest_indicators: List[Tuple[str, float]]) -> List[Dict[str, Any]]:
        """
        生成优化建议
        
        Args:
            performance_results: 性能测试结果
            slowest_indicators: 最慢的指标列表
            
        Returns:
            List: 优化建议列表
        """
        recommendations = []
        
        # 针对最慢的指标生成建议
        for indicator_name, avg_time in slowest_indicators[:5]:
            if avg_time > 0.1:  # 超过100ms的指标
                recommendations.append({
                    'indicator': indicator_name,
                    'current_time': avg_time,
                    'priority': 'HIGH' if avg_time > 0.5 else 'MEDIUM',
                    'optimization_type': 'VECTORIZATION',
                    'expected_improvement': '40-70%',
                    'description': f'指标 {indicator_name} 平均耗时 {avg_time:.3f}s，建议实施向量化优化'
                })
        
        # 针对中等耗时但使用频繁的指标
        medium_time_indicators = [
            (name, data['avg_time']) for name, data in performance_results.items()
            if 0.05 < data['avg_time'] < 0.1 and data['success_rate'] > 0.8
        ]
        
        for indicator_name, avg_time in sorted(medium_time_indicators, key=lambda x: x[1], reverse=True)[:3]:
            recommendations.append({
                'indicator': indicator_name,
                'current_time': avg_time,
                'priority': 'MEDIUM',
                'optimization_type': 'CACHING',
                'expected_improvement': '20-40%',
                'description': f'指标 {indicator_name} 适合实施缓存优化'
            })
        
        return recommendations
    
    def run_comprehensive_analysis(self, buypoints_csv: str, sample_size: int = 3) -> Dict[str, Any]:
        """
        运行综合性能分析
        
        Args:
            buypoints_csv: 买点数据CSV文件
            sample_size: 测试样本大小
            
        Returns:
            Dict: 综合分析结果
        """
        logger.info("开始综合指标性能分析")
        
        # 加载测试数据
        buypoints_df = self.analyzer.load_buypoints_from_csv(buypoints_csv)
        if buypoints_df.empty:
            logger.error("无法加载买点数据")
            return {}
        
        # 获取第一个买点的数据进行测试
        first_row = buypoints_df.iloc[0]
        stock_data = self.analyzer.data_processor.get_multi_period_data(
            stock_code=first_row['stock_code'],
            end_date=first_row['buypoint_date']
        )
        
        if not stock_data:
            logger.error("无法获取股票数据")
            return {}
        
        target_rows = {period: len(df) - 1 for period, df in stock_data.items() 
                      if df is not None and not df.empty}
        
        # 分析指标性能
        performance_results = self.profile_all_indicators(stock_data, target_rows)
        
        # 识别最慢的指标
        slowest_indicators = self.identify_slowest_indicators(performance_results)
        
        # 分析复杂度
        complexity_classification = self.analyze_indicator_complexity(performance_results)
        
        # 生成优化建议
        recommendations = self.generate_optimization_recommendations(performance_results, slowest_indicators)
        
        # 统计信息
        total_indicators = len(performance_results)
        successful_indicators = len([data for data in performance_results.values() if data['success_rate'] > 0])
        total_time = sum(data['avg_time'] for data in performance_results.values() if data['success_rate'] > 0)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'test_info': {
                'stock_code': first_row['stock_code'],
                'buypoint_date': first_row['buypoint_date'],
                'total_indicators': total_indicators,
                'successful_indicators': successful_indicators
            },
            'performance_results': performance_results,
            'slowest_indicators': dict(slowest_indicators),
            'complexity_classification': complexity_classification,
            'recommendations': recommendations,
            'summary': {
                'total_time': total_time,
                'avg_time_per_indicator': total_time / successful_indicators if successful_indicators > 0 else 0,
                'success_rate': successful_indicators / total_indicators if total_indicators > 0 else 0
            }
        }


def main():
    """主函数"""
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python indicator_performance_profiler.py <buypoints.csv> [sample_size]")
        sys.exit(1)
    
    buypoints_csv = sys.argv[1]
    sample_size = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    
    profiler = IndicatorPerformanceProfiler()
    results = profiler.run_comprehensive_analysis(buypoints_csv, sample_size)
    
    if not results:
        print("❌ 分析失败")
        return
    
    print("="*60)
    print("技术指标性能分析报告")
    print("="*60)
    
    # 显示基本信息
    test_info = results['test_info']
    print(f"测试股票: {test_info['stock_code']}")
    print(f"测试日期: {test_info['buypoint_date']}")
    print(f"总指标数: {test_info['total_indicators']}")
    print(f"成功指标数: {test_info['successful_indicators']}")
    
    # 显示最慢的指标
    print(f"\n最耗时的10个指标:")
    for i, (indicator, time_val) in enumerate(list(results['slowest_indicators'].items())[:10], 1):
        print(f"  {i:2d}. {indicator:20s}: {time_val:.4f}s")
    
    # 显示复杂度分类
    complexity = results['complexity_classification']
    high_complexity = [name for name, level in complexity.items() if level == 'HIGH']
    print(f"\n高复杂度指标 ({len(high_complexity)}个):")
    for indicator in high_complexity[:5]:
        time_val = results['performance_results'][indicator]['avg_time']
        print(f"  - {indicator}: {time_val:.4f}s")
    
    # 显示优化建议
    print(f"\n优化建议:")
    for i, rec in enumerate(results['recommendations'][:5], 1):
        print(f"  {i}. {rec['indicator']} ({rec['priority']})")
        print(f"     当前耗时: {rec['current_time']:.4f}s")
        print(f"     优化类型: {rec['optimization_type']}")
        print(f"     预期改进: {rec['expected_improvement']}")
    
    # 保存结果
    output_dir = "data/result/indicator_performance_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    results_path = os.path.join(output_dir, 'indicator_performance_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n分析结果已保存到: {results_path}")
    print("="*60)


if __name__ == "__main__":
    main()
