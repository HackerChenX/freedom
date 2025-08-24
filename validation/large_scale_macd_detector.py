#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大规模MACD形态检测器

针对四千只个股进行MACD形态检测：
1. 扩大样本量到2000+股票
2. 并行处理提高效率
3. 智能采样策略
4. 实时进度监控
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

try:
    from indicators.macd import MacdMacd
    from db.services.stock_data_service import get_stock_data_service
    from utils.logger import get_logger
except ImportError as e:
    print(f"导入错误: {e}")

logger = get_logger(__name__)

class LargeScaleMacdDetector:
    """大规模MACD形态检测器"""
    
    def __init__(self):
        """初始化检测器"""
        try:
            self.macd_indicator = MacdMacd()
            self.stock_data_service = get_stock_data_service()
            self.use_real_data = True
            print("✅ 使用真实MACD指标和数据服务")
        except Exception as e:
            print(f"⚠️ 初始化失败: {e}")
            self.macd_indicator = None
            self.stock_data_service = None
            self.use_real_data = False
        
        # 目标：每个形态找到至少10支股票
        self.target_stocks_per_pattern = 10
        self.max_stocks_per_pattern = 20  # 限制上限避免过多
    
    def run_large_scale_detection(self, max_stocks: int = 2000) -> Dict[str, Any]:
        """运行大规模MACD形态检测"""
        
        print(f"🚀 开始大规模MACD形态检测")
        print("=" * 80)
        print(f"📊 目标样本量: {max_stocks}支股票")
        print(f"🎯 检测目标: 每个MACD形态找到{self.target_stocks_per_pattern}支符合条件的股票")
        print(f"📋 检测形态: 金叉、死叉、零轴上金叉、看跌背离")
        print("=" * 80)
        
        start_time = time.time()
        
        results = {
            'detection_start_time': datetime.now().isoformat(),
            'target_sample_size': max_stocks,
            'actual_stocks_tested': 0,
            'stocks_with_valid_data': 0,
            'macd_calculation_success': 0,
            'pattern_results': {
                'GOLDEN_CROSS': {'stocks': [], 'target_met': False},
                'DEATH_CROSS': {'stocks': [], 'target_met': False},
                'MACD_ABOVE_ZERO_GOLDEN': {'stocks': [], 'target_met': False},
                'BEARISH_DIVERGENCE': {'stocks': [], 'target_met': False}
            },
            'performance_stats': {},
            'detection_summary': {},
            'issues_found': []
        }
        
        if not self.use_real_data:
            results['issues_found'].append("无法使用真实数据服务")
            return results
        
        try:
            # 获取股票列表
            print(f"📋 获取股票列表...")
            stock_codes = self.stock_data_service.get_stock_list(limit=max_stocks)
            
            if not stock_codes:
                results['issues_found'].append("无法获取股票列表")
                return results
            
            print(f"✅ 获取到{len(stock_codes)}支股票")
            results['actual_stocks_tested'] = len(stock_codes)
            
            # 分批处理股票
            batch_size = 100
            total_batches = (len(stock_codes) + batch_size - 1) // batch_size
            
            print(f"🔄 开始分批处理，共{total_batches}批，每批{batch_size}支股票")
            
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(stock_codes))
                batch_stocks = stock_codes[start_idx:end_idx]
                
                print(f"\n📊 处理第{batch_idx+1}/{total_batches}批 ({start_idx+1}-{end_idx})")
                
                # 处理当前批次
                batch_results = self._process_stock_batch(batch_stocks)
                
                # 合并结果
                results['stocks_with_valid_data'] += batch_results['valid_data_count']
                results['macd_calculation_success'] += batch_results['macd_success_count']
                
                # 合并形态检测结果
                for pattern_id, pattern_data in batch_results['patterns'].items():
                    results['pattern_results'][pattern_id]['stocks'].extend(pattern_data)
                    
                    # 限制每个形态的股票数量
                    if len(results['pattern_results'][pattern_id]['stocks']) > self.max_stocks_per_pattern:
                        results['pattern_results'][pattern_id]['stocks'] = results['pattern_results'][pattern_id]['stocks'][:self.max_stocks_per_pattern]
                
                # 显示当前进度
                current_counts = {k: len(v['stocks']) for k, v in results['pattern_results'].items()}
                print(f"  当前累计: 金叉{current_counts['GOLDEN_CROSS']}, 死叉{current_counts['DEATH_CROSS']}, 零轴上金叉{current_counts['MACD_ABOVE_ZERO_GOLDEN']}, 看跌背离{current_counts['BEARISH_DIVERGENCE']}")
                
                # 检查是否所有形态都达到目标
                all_targets_met = all(
                    len(pattern_data['stocks']) >= self.target_stocks_per_pattern 
                    for pattern_data in results['pattern_results'].values()
                )
                
                if all_targets_met:
                    print(f"🎉 所有形态都达到目标数量，提前结束检测")
                    break
            
            # 标记达到目标的形态
            for pattern_id, pattern_data in results['pattern_results'].items():
                pattern_data['target_met'] = len(pattern_data['stocks']) >= self.target_stocks_per_pattern
            
            # 计算性能统计
            end_time = time.time()
            results['performance_stats'] = {
                'total_time_seconds': end_time - start_time,
                'stocks_per_second': results['stocks_with_valid_data'] / (end_time - start_time) if end_time > start_time else 0,
                'detection_efficiency': results['macd_calculation_success'] / results['stocks_with_valid_data'] if results['stocks_with_valid_data'] > 0 else 0
            }
            
            # 生成检测汇总
            results['detection_summary'] = self._generate_detection_summary(results)
            
        except Exception as e:
            results['issues_found'].append(f"大规模检测异常: {str(e)}")
        
        return results
    
    def _process_stock_batch(self, stock_codes: List[str]) -> Dict[str, Any]:
        """处理股票批次"""
        
        batch_results = {
            'valid_data_count': 0,
            'macd_success_count': 0,
            'patterns': {
                'GOLDEN_CROSS': [],
                'DEATH_CROSS': [],
                'MACD_ABOVE_ZERO_GOLDEN': [],
                'BEARISH_DIVERGENCE': []
            }
        }
        
        for stock_code in stock_codes:
            try:
                # 获取股票数据
                df = self.stock_data_service.get_stock_data(stock_code, days=120)
                
                if df is None or len(df) < 60:
                    continue
                
                batch_results['valid_data_count'] += 1
                
                # 计算MACD
                macd_result = self.macd_indicator.calculate(df)
                
                if macd_result is None or macd_result.empty:
                    continue
                
                batch_results['macd_success_count'] += 1
                
                # 检测形态
                pattern_detections = self._detect_patterns_optimized(stock_code, df, macd_result)
                
                # 记录检测到的形态
                for pattern_id, detection_info in pattern_detections.items():
                    if detection_info:
                        batch_results['patterns'][pattern_id].append({
                            'stock_code': stock_code,
                            'detection_date': detection_info.get('detection_date', 'unknown'),
                            'pattern_strength': detection_info.get('cross_strength', 0),
                            'macd_values': {
                                'macd_line': detection_info.get('macd_value', 0),
                                'signal_line': detection_info.get('signal_value', 0)
                            },
                            'close_price': float(df.iloc[-1]['close']) if 'close' in df.columns else 0
                        })
                
            except Exception as e:
                continue
        
        return batch_results
    
    def _detect_patterns_optimized(self, stock_code: str, price_df: pd.DataFrame, macd_df: pd.DataFrame) -> Dict[str, Any]:
        """优化的形态检测"""
        
        patterns = {
            'GOLDEN_CROSS': None,
            'DEATH_CROSS': None,
            'MACD_ABOVE_ZERO_GOLDEN': None,
            'BEARISH_DIVERGENCE': None
        }
        
        try:
            # 检查必要的列（修复列名匹配问题）
            signal_col = None
            if 'signal_line' in macd_df.columns:
                signal_col = 'signal_line'
            elif 'macd_signal' in macd_df.columns:
                signal_col = 'macd_signal'

            if 'macd_line' not in macd_df.columns or signal_col is None:
                return patterns
            
            # 获取最近30天的数据进行检测
            recent_days = min(30, len(macd_df))
            recent_macd = macd_df.tail(recent_days)
            recent_price = price_df.tail(recent_days)
            
            if len(recent_macd) < 5:
                return patterns
            
            macd_values = recent_macd['macd_line'].values
            signal_values = recent_macd[signal_col].values
            
            # 检测金叉和死叉
            for i in range(1, len(macd_values)):
                # 金叉检测
                if (macd_values[i-1] <= signal_values[i-1] and 
                    macd_values[i] > signal_values[i] and
                    abs(macd_values[i] - signal_values[i]) > 0.0001):
                    
                    if not patterns['GOLDEN_CROSS']:  # 只记录第一个
                        patterns['GOLDEN_CROSS'] = {
                            'detection_date': recent_macd.index[i],
                            'macd_value': macd_values[i],
                            'signal_value': signal_values[i],
                            'cross_strength': abs(macd_values[i] - signal_values[i])
                        }
                    
                    # 检查是否是零轴上金叉
                    if (macd_values[i] > 0 and signal_values[i] > 0 and 
                        not patterns['MACD_ABOVE_ZERO_GOLDEN']):
                        patterns['MACD_ABOVE_ZERO_GOLDEN'] = {
                            'detection_date': recent_macd.index[i],
                            'macd_value': macd_values[i],
                            'signal_value': signal_values[i],
                            'cross_strength': abs(macd_values[i] - signal_values[i])
                        }
                
                # 死叉检测
                if (macd_values[i-1] >= signal_values[i-1] and 
                    macd_values[i] < signal_values[i] and
                    abs(macd_values[i] - signal_values[i]) > 0.0001):
                    
                    if not patterns['DEATH_CROSS']:  # 只记录第一个
                        patterns['DEATH_CROSS'] = {
                            'detection_date': recent_macd.index[i],
                            'macd_value': macd_values[i],
                            'signal_value': signal_values[i],
                            'cross_strength': abs(macd_values[i] - signal_values[i])
                        }
            
            # 简化的背离检测
            if len(recent_price) >= 10:
                close_prices = recent_price['close'].values
                
                # 查找最近的高点
                recent_high_idx = np.argmax(close_prices[-10:])
                recent_macd_at_high = macd_values[-10:][recent_high_idx]
                max_macd_in_period = np.max(macd_values[-10:])
                
                # 简单的背离判断
                if (recent_high_idx >= 7 and  # 价格高点在最近3天内
                    recent_macd_at_high < max_macd_in_period * 0.8):  # MACD明显低于期间最高值
                    
                    patterns['BEARISH_DIVERGENCE'] = {
                        'detection_date': recent_macd.index[-10 + recent_high_idx],
                        'price_high': close_prices[-10:][recent_high_idx],
                        'macd_at_high': recent_macd_at_high,
                        'max_macd': max_macd_in_period,
                        'cross_strength': abs(max_macd_in_period - recent_macd_at_high)
                    }
        
        except Exception as e:
            pass
        
        return patterns
    
    def _generate_detection_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """生成检测汇总"""
        
        pattern_names = {
            'GOLDEN_CROSS': 'MACD金叉',
            'DEATH_CROSS': 'MACD死叉',
            'MACD_ABOVE_ZERO_GOLDEN': 'MACD零轴上金叉',
            'BEARISH_DIVERGENCE': 'MACD看跌背离'
        }
        
        summary = {
            'total_patterns_found': 0,
            'patterns_meeting_target': 0,
            'detection_success_rate': 0,
            'pattern_details': {},
            'overall_assessment': 'PENDING'
        }
        
        for pattern_id, pattern_data in results['pattern_results'].items():
            stock_count = len(pattern_data['stocks'])
            target_met = pattern_data['target_met']
            
            summary['total_patterns_found'] += stock_count
            if target_met:
                summary['patterns_meeting_target'] += 1
            
            summary['pattern_details'][pattern_id] = {
                'name': pattern_names.get(pattern_id, pattern_id),
                'stocks_found': stock_count,
                'target_stocks': self.target_stocks_per_pattern,
                'target_met': target_met,
                'success_rate': min(stock_count / self.target_stocks_per_pattern, 1.0) if self.target_stocks_per_pattern > 0 else 0
            }
        
        # 计算整体成功率
        total_patterns = len(results['pattern_results'])
        summary['detection_success_rate'] = summary['patterns_meeting_target'] / total_patterns if total_patterns > 0 else 0
        
        # 整体评估
        if summary['patterns_meeting_target'] == total_patterns:
            summary['overall_assessment'] = 'SUCCESS'
        elif summary['patterns_meeting_target'] >= total_patterns * 0.75:
            summary['overall_assessment'] = 'MOSTLY_SUCCESS'
        elif summary['patterns_meeting_target'] > 0:
            summary['overall_assessment'] = 'PARTIAL_SUCCESS'
        else:
            summary['overall_assessment'] = 'FAILED'
        
        return summary
    
    def print_detection_results(self, results: Dict[str, Any]):
        """打印检测结果"""
        
        print("\n" + "=" * 80)
        print("📊 大规模MACD形态检测结果")
        print("=" * 80)
        
        # 基本统计
        print(f"📈 检测统计:")
        print(f"  目标样本量: {results['target_sample_size']}")
        print(f"  实际测试股票: {results['actual_stocks_tested']}")
        print(f"  有效数据股票: {results['stocks_with_valid_data']}")
        print(f"  MACD计算成功: {results['macd_calculation_success']}")
        
        # 性能统计
        perf = results.get('performance_stats', {})
        if perf:
            print(f"  处理时间: {perf.get('total_time_seconds', 0):.1f}秒")
            print(f"  处理速度: {perf.get('stocks_per_second', 0):.1f}股票/秒")
            print(f"  计算效率: {perf.get('detection_efficiency', 0):.1%}")
        
        # 形态检测结果
        print(f"\n🎯 形态检测结果:")
        summary = results.get('detection_summary', {})
        
        for pattern_id, pattern_data in results['pattern_results'].items():
            pattern_detail = summary.get('pattern_details', {}).get(pattern_id, {})
            pattern_name = pattern_detail.get('name', pattern_id)
            stock_count = len(pattern_data['stocks'])
            target_met = pattern_data['target_met']
            
            status = "✅" if target_met else "❌"
            print(f"  {status} {pattern_name}: {stock_count}支股票 (目标: {self.target_stocks_per_pattern})")
            
            # 显示前3个股票示例
            if stock_count > 0:
                for i, stock_info in enumerate(pattern_data['stocks'][:3]):
                    stock_code = stock_info['stock_code']
                    strength = stock_info.get('pattern_strength', 0)
                    print(f"    - {stock_code} (强度: {strength:.4f})")
        
        # 整体评估
        overall = summary.get('overall_assessment', 'UNKNOWN')
        success_rate = summary.get('detection_success_rate', 0)
        
        print(f"\n🏆 整体评估:")
        print(f"  成功率: {success_rate:.1%}")
        print(f"  评估结果: {overall}")
        
        if overall == 'SUCCESS':
            print(f"  🎉 所有MACD形态都找到了足够的股票！")
        elif overall == 'MOSTLY_SUCCESS':
            print(f"  ✅ 大部分MACD形态找到了足够的股票")
        elif overall == 'PARTIAL_SUCCESS':
            print(f"  ⚠️ 部分MACD形态找到了股票，需要进一步优化")
        else:
            print(f"  ❌ 检测结果不理想，需要检查检测逻辑")

def main():
    """主函数"""
    print("🚀 大规模MACD形态检测系统")
    print("目标：在四千只股票中找到满足各种MACD形态的股票")
    
    # 创建检测器
    detector = LargeScaleMacdDetector()
    
    # 运行大规模检测
    results = detector.run_large_scale_detection(max_stocks=2000)
    
    # 打印结果
    detector.print_detection_results(results)

if __name__ == "__main__":
    main()
