#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析MACD其他形态未检测到的原因

问题：
- MACD金叉：检测到2支股票 ✅
- MACD死叉：0支股票 ❌
- MACD零轴上金叉：0支股票 ❌  
- MACD看跌背离：0支股票 ❌

需要分析：
1. 检测条件是否过于严格
2. 2025-05-12这个特定日期是否不利于其他形态
3. 样本量是否足够
4. 检测逻辑是否有问题
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

try:
    from indicators.macd import MacdMacd
    from db.services.stock_data_service import get_stock_data_service
    from utils.logger import get_logger
except ImportError as e:
    print(f"导入错误: {e}")

logger = get_logger(__name__)

class MissingPatternsAnalyzer:
    """分析缺失形态的原因"""
    
    def __init__(self):
        """初始化分析器"""
        self.macd_indicator = MacdMacd()
        self.stock_data_service = get_stock_data_service()
        self.target_date = "2025-05-12"
        self.target_date_obj = pd.to_datetime(self.target_date).date()
        
        print("🔍 MACD形态缺失原因分析器初始化完成")
        print(f"📅 分析目标日期: {self.target_date}")
    
    def analyze_pattern_conditions(self, max_stocks: int = 100) -> Dict[str, Any]:
        """分析各种形态的检测条件"""
        
        print(f"\n🎯 开始分析MACD形态检测条件")
        print("=" * 80)
        
        analysis_results = {
            'analysis_timestamp': datetime.now().isoformat(),
            'target_date': self.target_date,
            'stocks_analyzed': 0,
            'stocks_with_valid_data': 0,
            'pattern_statistics': {
                'GOLDEN_CROSS': {'candidates': 0, 'near_misses': 0, 'detected': 0},
                'DEATH_CROSS': {'candidates': 0, 'near_misses': 0, 'detected': 0},
                'ABOVE_ZERO_GOLDEN': {'candidates': 0, 'near_misses': 0, 'detected': 0},
                'BEARISH_DIVERGENCE': {'candidates': 0, 'near_misses': 0, 'detected': 0}
            },
            'condition_analysis': {},
            'sample_cases': {},
            'recommendations': []
        }
        
        try:
            # 获取股票列表
            stock_codes = self.stock_data_service.get_stock_list(limit=max_stocks)
            
            if not stock_codes:
                return analysis_results
            
            print(f"📋 分析{len(stock_codes)}支股票的MACD形态条件...")
            
            for stock_code in stock_codes:
                analysis_results['stocks_analyzed'] += 1
                
                if analysis_results['stocks_analyzed'] % 20 == 0:
                    print(f"  📊 已分析{analysis_results['stocks_analyzed']}/{len(stock_codes)}支股票...")
                
                # 获取股票数据
                stock_data = self._get_stock_macd_data(stock_code)
                
                if stock_data is None:
                    continue
                
                analysis_results['stocks_with_valid_data'] += 1
                df, macd_result, target_idx, actual_date, price_data, macd_data = stock_data
                
                # 分析各种形态的条件
                self._analyze_golden_cross_conditions(
                    stock_code, df, macd_result, target_idx, macd_data, analysis_results
                )
                
                self._analyze_death_cross_conditions(
                    stock_code, df, macd_result, target_idx, macd_data, analysis_results
                )
                
                self._analyze_above_zero_golden_conditions(
                    stock_code, df, macd_result, target_idx, macd_data, analysis_results
                )
                
                self._analyze_bearish_divergence_conditions(
                    stock_code, df, macd_result, target_idx, price_data, macd_data, analysis_results
                )
            
            # 生成条件分析
            self._generate_condition_analysis(analysis_results)
            
            # 生成建议
            self._generate_recommendations(analysis_results)
            
            return analysis_results
            
        except Exception as e:
            analysis_results['error'] = f'分析过程异常: {str(e)}'
            return analysis_results
    
    def _get_stock_macd_data(self, stock_code: str) -> Optional[tuple]:
        """获取股票MACD数据"""
        
        try:
            df = self.stock_data_service.get_stock_data(stock_code, days=200)
            
            if df is None or len(df) < 120:
                return None
            
            # 查找目标日期
            target_rows = df[df['date'].dt.date == self.target_date_obj]
            
            if target_rows.empty:
                # 查找最接近的交易日
                df['date_diff'] = abs((df['date'].dt.date - self.target_date_obj).apply(lambda x: x.days))
                closest_idx = df['date_diff'].idxmin()
                
                if df.loc[closest_idx]['date_diff'] > 3:
                    return None
                
                target_idx = closest_idx
                actual_date = df.loc[closest_idx]['date'].date()
            else:
                target_idx = target_rows.index[0]
                actual_date = self.target_date_obj
            
            # 计算MACD
            macd_result = self.macd_indicator.calculate(df)
            
            if macd_result is None or macd_result.empty or target_idx >= len(macd_result):
                return None
            
            price_data = df.iloc[target_idx]
            macd_data = macd_result.iloc[target_idx]
            
            return (df, macd_result, target_idx, actual_date, price_data, macd_data)
            
        except Exception as e:
            return None
    
    def _analyze_golden_cross_conditions(self, stock_code: str, df: pd.DataFrame, 
                                       macd_df: pd.DataFrame, target_idx: int, 
                                       macd_data: pd.Series, analysis_results: Dict):
        """分析金叉条件"""
        
        try:
            if target_idx < 1:
                return
            
            curr_macd = macd_data['macd_line']
            curr_signal = macd_data['macd_signal']
            prev_macd = macd_df.iloc[target_idx - 1]['macd_line']
            prev_signal = macd_df.iloc[target_idx - 1]['macd_signal']
            
            # 检查是否接近金叉条件
            curr_diff = curr_macd - curr_signal
            prev_diff = prev_macd - prev_signal
            
            # 候选条件：MACD线接近信号线
            if abs(curr_diff) < 0.01:  # 差距小于0.01
                analysis_results['pattern_statistics']['GOLDEN_CROSS']['candidates'] += 1
            
            # 接近条件：前一天差距较大，当前差距较小
            if prev_diff <= 0 and curr_diff > 0:
                if abs(curr_diff) > 0.001:  # 满足强度要求
                    analysis_results['pattern_statistics']['GOLDEN_CROSS']['detected'] += 1
                else:  # 接近但强度不够
                    analysis_results['pattern_statistics']['GOLDEN_CROSS']['near_misses'] += 1
                    
                    # 记录接近案例
                    if len(analysis_results['sample_cases'].get('GOLDEN_CROSS_NEAR_MISS', [])) < 3:
                        if 'GOLDEN_CROSS_NEAR_MISS' not in analysis_results['sample_cases']:
                            analysis_results['sample_cases']['GOLDEN_CROSS_NEAR_MISS'] = []
                        
                        analysis_results['sample_cases']['GOLDEN_CROSS_NEAR_MISS'].append({
                            'stock_code': stock_code,
                            'curr_macd': curr_macd,
                            'curr_signal': curr_signal,
                            'cross_strength': abs(curr_diff),
                            'reason': '穿越强度不足'
                        })
        
        except Exception as e:
            pass
    
    def _analyze_death_cross_conditions(self, stock_code: str, df: pd.DataFrame, 
                                      macd_df: pd.DataFrame, target_idx: int, 
                                      macd_data: pd.Series, analysis_results: Dict):
        """分析死叉条件"""
        
        try:
            if target_idx < 1:
                return
            
            curr_macd = macd_data['macd_line']
            curr_signal = macd_data['macd_signal']
            prev_macd = macd_df.iloc[target_idx - 1]['macd_line']
            prev_signal = macd_df.iloc[target_idx - 1]['macd_signal']
            
            curr_diff = curr_macd - curr_signal
            prev_diff = prev_macd - prev_signal
            
            # 候选条件：MACD线接近信号线
            if abs(curr_diff) < 0.01:
                analysis_results['pattern_statistics']['DEATH_CROSS']['candidates'] += 1
            
            # 死叉条件：前一天MACD >= 信号线，当前MACD < 信号线
            if prev_diff >= 0 and curr_diff < 0:
                if abs(curr_diff) > 0.001:
                    analysis_results['pattern_statistics']['DEATH_CROSS']['detected'] += 1
                else:
                    analysis_results['pattern_statistics']['DEATH_CROSS']['near_misses'] += 1
                    
                    # 记录接近案例
                    if len(analysis_results['sample_cases'].get('DEATH_CROSS_NEAR_MISS', [])) < 3:
                        if 'DEATH_CROSS_NEAR_MISS' not in analysis_results['sample_cases']:
                            analysis_results['sample_cases']['DEATH_CROSS_NEAR_MISS'] = []
                        
                        analysis_results['sample_cases']['DEATH_CROSS_NEAR_MISS'].append({
                            'stock_code': stock_code,
                            'curr_macd': curr_macd,
                            'curr_signal': curr_signal,
                            'cross_strength': abs(curr_diff),
                            'reason': '穿越强度不足'
                        })
        
        except Exception as e:
            pass
    
    def _analyze_above_zero_golden_conditions(self, stock_code: str, df: pd.DataFrame, 
                                            macd_df: pd.DataFrame, target_idx: int, 
                                            macd_data: pd.Series, analysis_results: Dict):
        """分析零轴上金叉条件"""
        
        try:
            if target_idx < 1:
                return
            
            curr_macd = macd_data['macd_line']
            curr_signal = macd_data['macd_signal']
            prev_macd = macd_df.iloc[target_idx - 1]['macd_line']
            prev_signal = macd_df.iloc[target_idx - 1]['macd_signal']
            
            # 候选条件：MACD线和信号线都接近零轴上方
            if curr_macd > -0.01 and curr_signal > -0.01:  # 接近零轴
                analysis_results['pattern_statistics']['ABOVE_ZERO_GOLDEN']['candidates'] += 1
            
            # 零轴上金叉条件
            if (curr_macd > 0 and curr_signal > 0 and  # 都在零轴上方
                prev_macd <= prev_signal and curr_macd > curr_signal):  # 形成金叉
                
                if abs(curr_macd - curr_signal) > 0.001:
                    analysis_results['pattern_statistics']['ABOVE_ZERO_GOLDEN']['detected'] += 1
                else:
                    analysis_results['pattern_statistics']['ABOVE_ZERO_GOLDEN']['near_misses'] += 1
            
            # 记录接近零轴但未达到条件的案例
            elif curr_macd > -0.005 and curr_signal > -0.005:  # 非常接近零轴
                if len(analysis_results['sample_cases'].get('ABOVE_ZERO_NEAR_MISS', [])) < 3:
                    if 'ABOVE_ZERO_NEAR_MISS' not in analysis_results['sample_cases']:
                        analysis_results['sample_cases']['ABOVE_ZERO_NEAR_MISS'] = []
                    
                    analysis_results['sample_cases']['ABOVE_ZERO_NEAR_MISS'].append({
                        'stock_code': stock_code,
                        'curr_macd': curr_macd,
                        'curr_signal': curr_signal,
                        'reason': '接近零轴但未完全在上方'
                    })
        
        except Exception as e:
            pass
    
    def _analyze_bearish_divergence_conditions(self, stock_code: str, df: pd.DataFrame, 
                                             macd_df: pd.DataFrame, target_idx: int, 
                                             price_data: pd.Series, macd_data: pd.Series, 
                                             analysis_results: Dict):
        """分析看跌背离条件"""
        
        try:
            if target_idx < 20:
                return
            
            # 获取最近20天的数据
            recent_start = max(0, target_idx - 19)
            recent_price = df.iloc[recent_start:target_idx + 1]
            recent_macd = macd_df.iloc[recent_start:target_idx + 1]
            
            if len(recent_price) < 15:
                return
            
            close_prices = recent_price['close'].values
            macd_values = recent_macd['macd_line'].values
            
            current_price = close_prices[-1]
            current_macd = macd_values[-1]
            
            # 候选条件：当前价格是否接近最近的高点
            recent_max_price = max(close_prices[-10:])  # 最近10天最高价
            if current_price >= recent_max_price * 0.98:  # 接近最高价
                analysis_results['pattern_statistics']['BEARISH_DIVERGENCE']['candidates'] += 1
            
            # 查找前期高点进行背离分析
            for i in range(len(close_prices) - 5, 5, -1):
                if (close_prices[i] == max(close_prices[i-2:i+3]) and  # 局部高点
                    close_prices[i] < current_price):  # 当前价格更高
                    
                    prev_macd_high = macd_values[i]
                    
                    # 背离条件：价格创新高但MACD没有创新高
                    if current_macd < prev_macd_high * 0.9:  # MACD明显低于前期高点
                        analysis_results['pattern_statistics']['BEARISH_DIVERGENCE']['detected'] += 1
                        break
                    elif current_macd < prev_macd_high * 0.95:  # 接近背离条件
                        analysis_results['pattern_statistics']['BEARISH_DIVERGENCE']['near_misses'] += 1
                        
                        # 记录接近案例
                        if len(analysis_results['sample_cases'].get('BEARISH_DIVERGENCE_NEAR_MISS', [])) < 3:
                            if 'BEARISH_DIVERGENCE_NEAR_MISS' not in analysis_results['sample_cases']:
                                analysis_results['sample_cases']['BEARISH_DIVERGENCE_NEAR_MISS'] = []
                            
                            analysis_results['sample_cases']['BEARISH_DIVERGENCE_NEAR_MISS'].append({
                                'stock_code': stock_code,
                                'current_price': current_price,
                                'prev_high_price': close_prices[i],
                                'current_macd': current_macd,
                                'prev_macd_high': prev_macd_high,
                                'divergence_ratio': current_macd / prev_macd_high if prev_macd_high != 0 else 0,
                                'reason': '背离程度不足'
                            })
                        break
        
        except Exception as e:
            pass
    
    def _generate_condition_analysis(self, analysis_results: Dict):
        """生成条件分析"""
        
        condition_analysis = {}
        
        for pattern_id, stats in analysis_results['pattern_statistics'].items():
            candidates = stats['candidates']
            near_misses = stats['near_misses']
            detected = stats['detected']
            
            if candidates > 0:
                detection_rate = detected / candidates
                near_miss_rate = near_misses / candidates
            else:
                detection_rate = 0
                near_miss_rate = 0
            
            condition_analysis[pattern_id] = {
                'candidates_found': candidates,
                'near_misses': near_misses,
                'successfully_detected': detected,
                'detection_rate': detection_rate,
                'near_miss_rate': near_miss_rate,
                'condition_strictness': 'HIGH' if near_miss_rate > 0.5 else 'MEDIUM' if near_miss_rate > 0.2 else 'LOW'
            }
        
        analysis_results['condition_analysis'] = condition_analysis
    
    def _generate_recommendations(self, analysis_results: Dict):
        """生成建议"""
        
        recommendations = []
        
        for pattern_id, analysis in analysis_results['condition_analysis'].items():
            pattern_names = {
                'GOLDEN_CROSS': 'MACD金叉',
                'DEATH_CROSS': 'MACD死叉',
                'ABOVE_ZERO_GOLDEN': 'MACD零轴上金叉',
                'BEARISH_DIVERGENCE': 'MACD看跌背离'
            }
            
            pattern_name = pattern_names.get(pattern_id, pattern_id)
            candidates = analysis['candidates_found']
            detected = analysis['successfully_detected']
            near_misses = analysis['near_misses']
            
            if candidates == 0:
                recommendations.append(f"❌ {pattern_name}: 未找到候选股票，可能需要扩大样本量或调整检测时间窗口")
            elif detected == 0 and near_misses > 0:
                recommendations.append(f"⚠️ {pattern_name}: 有{near_misses}个接近案例但检测条件过严，建议放宽强度阈值")
            elif detected > 0:
                recommendations.append(f"✅ {pattern_name}: 检测正常，找到{detected}个符合条件的股票")
            else:
                recommendations.append(f"🔍 {pattern_name}: 有{candidates}个候选但无检测结果，需要检查检测逻辑")
        
        # 整体建议
        total_detected = sum(stats['detected'] for stats in analysis_results['pattern_statistics'].values())
        total_candidates = sum(stats['candidates'] for stats in analysis_results['pattern_statistics'].values())
        
        if total_detected < 4:  # 期望每个形态至少1个
            recommendations.append("💡 整体建议: 考虑扩大检测时间窗口（如前后3天）或放宽检测条件")
        
        if total_candidates < 10:
            recommendations.append("📊 数据建议: 样本量可能不足，建议增加检测股票数量")
        
        analysis_results['recommendations'] = recommendations

def main():
    """主函数"""
    print("🔍 分析MACD其他形态未检测到的原因")
    print("深入调查死叉、零轴上金叉、看跌背离的检测条件")
    
    # 创建分析器
    analyzer = MissingPatternsAnalyzer()
    
    # 运行分析
    results = analyzer.analyze_pattern_conditions(max_stocks=100)
    
    if 'error' in results:
        print(f"❌ 分析失败: {results['error']}")
        return
    
    # 显示分析结果
    print(f"\n📊 MACD形态条件分析结果")
    print("=" * 80)
    print(f"🕐 分析时间: {results['analysis_timestamp']}")
    print(f"📅 目标日期: {results['target_date']}")
    print(f"📈 分析股票: {results['stocks_analyzed']}支")
    print(f"📊 有效数据: {results['stocks_with_valid_data']}支")
    
    print(f"\n🎯 各形态条件分析:")
    print("-" * 80)
    
    pattern_names = {
        'GOLDEN_CROSS': 'MACD金叉',
        'DEATH_CROSS': 'MACD死叉',
        'ABOVE_ZERO_GOLDEN': 'MACD零轴上金叉',
        'BEARISH_DIVERGENCE': 'MACD看跌背离'
    }
    
    for pattern_id, analysis in results['condition_analysis'].items():
        pattern_name = pattern_names.get(pattern_id, pattern_id)
        
        print(f"\n📈 {pattern_name}:")
        print(f"  候选股票: {analysis['candidates_found']}")
        print(f"  接近案例: {analysis['near_misses']}")
        print(f"  成功检测: {analysis['successfully_detected']}")
        print(f"  检测率: {analysis['detection_rate']:.1%}")
        print(f"  条件严格度: {analysis['condition_strictness']}")
        
        # 显示样本案例
        sample_key = f"{pattern_id}_NEAR_MISS"
        if sample_key in results['sample_cases']:
            print(f"  接近案例:")
            for case in results['sample_cases'][sample_key]:
                print(f"    {case['stock_code']}: {case['reason']}")
    
    print(f"\n💡 改进建议:")
    print("-" * 80)
    for recommendation in results['recommendations']:
        print(f"  {recommendation}")
    
    print(f"\n🔍 关键发现:")
    print("-" * 80)
    
    # 分析主要问题
    zero_detection_patterns = [
        pattern_id for pattern_id, analysis in results['condition_analysis'].items()
        if analysis['successfully_detected'] == 0
    ]
    
    if zero_detection_patterns:
        print(f"  ❌ 未检测到的形态: {', '.join([pattern_names[p] for p in zero_detection_patterns])}")
        print(f"  🔍 主要原因分析:")
        
        for pattern_id in zero_detection_patterns:
            analysis = results['condition_analysis'][pattern_id]
            if analysis['candidates_found'] == 0:
                print(f"    - {pattern_names[pattern_id]}: 候选股票不足，可能需要扩大样本或调整时间")
            elif analysis['near_misses'] > 0:
                print(f"    - {pattern_names[pattern_id]}: 检测条件过严，有{analysis['near_misses']}个接近案例")
            else:
                print(f"    - {pattern_names[pattern_id]}: 检测逻辑可能需要调整")
    
    print(f"\n🎯 下一步行动建议:")
    print(f"  1. 考虑放宽穿越强度阈值（从0.001降低到0.0005）")
    print(f"  2. 扩大检测时间窗口（前后3天而不是单一日期）")
    print(f"  3. 增加样本量（从100支增加到200支股票）")
    print(f"  4. 调整零轴判断条件（从严格>0改为>-0.001）")

if __name__ == "__main__":
    main()
