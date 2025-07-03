#!/usr/bin/env python3
"""
历史信号验证器
检查指标在整个历史期间是否有买入信号，而不仅仅是最新日期
"""

import os
import sys
import pandas as pd
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from scripts.production_indicator_validator import ProductionIndicatorValidator
from utils.logger import get_logger

class HistoricalSignalValidator(ProductionIndicatorValidator):
    """历史信号验证器，继承自ProductionIndicatorValidator"""
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger(__name__)
    
    def validate_single_indicator_historical(self, indicator_name: str, test_date: str = None, 
                                           max_stocks: int = 500, history_days: int = 30) -> Dict[str, Any]:
        """
        验证单个指标在历史期间的选股效果
        
        Args:
            indicator_name: 指标名称
            test_date: 测试日期，None表示使用最新日期
            max_stocks: 最大测试股票数量
            history_days: 检查历史信号的天数
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        if indicator_name not in self.available_indicators:
            raise ValueError(f"未知指标: {indicator_name}，可用指标: {list(self.available_indicators.keys())}")
        
        if test_date is None:
            test_date = self.get_latest_trade_date()
        
        self.logger.info(f"🚀 开始历史信号验证指标: {indicator_name}")
        self.logger.info(f"📅 测试日期: {test_date}")
        self.logger.info(f"📊 最大股票数: {max_stocks}")
        self.logger.info(f"📈 历史天数: {history_days}")
        
        start_time = time.time()
        
        # 获取股票池
        stock_pool = self.get_stock_pool(test_date, max_stocks)
        if not stock_pool:
            return {
                'indicator_name': indicator_name,
                'test_date': test_date,
                'status': 'failed',
                'error': '无法获取股票池'
            }
        
        # 获取指标实例
        indicator = self.available_indicators[indicator_name]
        
        # 验证结果
        selected_stocks = []
        failed_stocks = []
        indicator_values = {}
        signal_details = {}
        
        self.logger.info(f"🔄 开始处理 {len(stock_pool)} 只股票...")
        
        for i, stock_code in enumerate(stock_pool):
            try:
                if (i + 1) % 50 == 0:
                    self.logger.info(f"📈 进度: {i+1}/{len(stock_pool)} ({(i+1)/len(stock_pool)*100:.1f}%)")
                
                # 获取股票数据（增加历史数据）
                stock_data = self.get_stock_data(stock_code, test_date, days=100 + history_days)
                
                if stock_data.empty or len(stock_data) < 20:
                    failed_stocks.append(stock_code)
                    continue
                
                # 计算指标
                result = indicator.calculate(stock_data)
                
                if result.empty:
                    failed_stocks.append(stock_code)
                    continue
                
                # 检查历史期间的买入信号
                has_buy_signal, signal_info = self._check_historical_signals(
                    result, indicator_name, history_days
                )
                
                if has_buy_signal:
                    selected_stocks.append(stock_code)
                    signal_details[stock_code] = signal_info
                
                # 保存指标值用于分析
                last_row = result.iloc[-1]
                indicator_values[stock_code] = {
                    'has_signal': bool(has_buy_signal),
                    'signal_count': signal_info['signal_count'],
                    'latest_signal_date': signal_info['latest_signal_date'],
                    'signal_dates': signal_info['signal_dates'][:5],  # 只保留最近5个信号日期
                    'last_close': float(last_row.get('close', stock_data['close'].iloc[-1]))
                }
                
            except Exception as e:
                self.logger.warning(f"⚠️ 处理股票 {stock_code} 失败: {e}")
                failed_stocks.append(stock_code)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 计算统计结果
        total_tested = len(stock_pool)
        success_count = len(indicator_values)
        selected_count = len(selected_stocks)
        failed_count = len(failed_stocks)
        
        selection_rate = selected_count / success_count if success_count > 0 else 0
        success_rate = success_count / total_tested if total_tested > 0 else 0
        
        # 分析信号分布
        total_signals = sum(v['signal_count'] for v in indicator_values.values())
        avg_signals_per_stock = total_signals / success_count if success_count > 0 else 0
        
        result = {
            'indicator_name': indicator_name,
            'test_date': test_date,
            'history_days': history_days,
            'validation_type': 'historical_signal',
            'status': 'success',
            'validation_time': datetime.now().isoformat(),
            'processing_time_seconds': round(processing_time, 2),
            'statistics': {
                'total_tested': total_tested,
                'success_processed': success_count,
                'failed_processed': failed_count,
                'selected_count': selected_count,
                'selection_rate': round(selection_rate, 4),
                'success_rate': round(success_rate, 4),
                'total_signals': total_signals,
                'avg_signals_per_stock': round(avg_signals_per_stock, 2)
            },
            'selected_stocks': selected_stocks,
            'failed_stocks': failed_stocks[:10],
            'signal_analysis': {
                'signal_distribution': self._analyze_signal_distribution_historical(indicator_values),
                'top_signals': self._get_top_signals_historical(indicator_values, 10)
            }
        }
        
        self.logger.info(f"✅ 指标 {indicator_name} 历史信号验证完成")
        self.logger.info(f"📊 选股结果: {selected_count}/{success_count} = {selection_rate:.2%}")
        self.logger.info(f"📈 总信号数: {total_signals}")
        self.logger.info(f"⏱️ 处理时间: {processing_time:.1f}秒")
        
        return result
    
    def _check_historical_signals(self, result: pd.DataFrame, indicator_name: str, 
                                history_days: int) -> tuple[bool, Dict[str, Any]]:
        """检查历史期间的买入信号"""
        
        # 确定信号列
        signal_columns = []
        if indicator_name in ['volume_shrink', 'bs_absorb', 'daily_macd', 'turnover', 'ma_callback']:
            # ZXM指标优先检查XG列
            if 'XG' in result.columns:
                signal_columns.append('XG')
            if 'buy_signal' in result.columns:
                signal_columns.append('buy_signal')
        else:
            # 其他指标检查buy_signal列
            if 'buy_signal' in result.columns:
                signal_columns.append('buy_signal')
            if 'XG' in result.columns:
                signal_columns.append('XG')
        
        # 如果没有找到标准信号列，尝试其他可能的列
        if not signal_columns:
            possible_signal_cols = ['signal', 'entry_signal', 'buy', 'long_signal']
            for col in possible_signal_cols:
                if col in result.columns:
                    signal_columns.append(col)
                    break
        
        if not signal_columns:
            return False, {
                'signal_count': 0,
                'latest_signal_date': None,
                'signal_dates': [],
                'error': f'未找到信号列，可用列: {list(result.columns)}'
            }
        
        # 检查历史期间的信号
        # 取最近的history_days天的数据
        recent_data = result.tail(min(history_days, len(result)))
        
        signal_dates = []
        total_signals = 0
        
        for signal_col in signal_columns:
            if signal_col in recent_data.columns:
                # 找到所有为True/1的信号
                signal_mask = recent_data[signal_col].astype(bool)
                signal_rows = recent_data[signal_mask]
                
                if not signal_rows.empty:
                    dates = [date.strftime('%Y-%m-%d') for date in signal_rows.index]
                    signal_dates.extend(dates)
                    total_signals += len(dates)
        
        # 去重并排序
        signal_dates = sorted(list(set(signal_dates)), reverse=True)
        
        has_signal = total_signals > 0
        latest_signal_date = signal_dates[0] if signal_dates else None
        
        return has_signal, {
            'signal_count': total_signals,
            'latest_signal_date': latest_signal_date,
            'signal_dates': signal_dates,
            'signal_columns_used': signal_columns
        }
    
    def _analyze_signal_distribution_historical(self, indicator_values: Dict[str, Dict]) -> Dict[str, Any]:
        """分析历史信号分布"""
        signal_count = sum(1 for v in indicator_values.values() if v['signal_count'] > 0)
        total_count = len(indicator_values)
        total_signals = sum(v['signal_count'] for v in indicator_values.values())
        
        return {
            'total_stocks': total_count,
            'signal_stocks': signal_count,
            'no_signal_stocks': total_count - signal_count,
            'signal_ratio': round(signal_count / total_count, 4) if total_count > 0 else 0,
            'total_signals': total_signals,
            'avg_signals_per_stock': round(total_signals / total_count, 2) if total_count > 0 else 0
        }
    
    def _get_top_signals_historical(self, indicator_values: Dict[str, Dict], top_n: int = 10) -> List[Dict[str, Any]]:
        """获取信号最强的股票（按信号数量排序）"""
        signal_stocks = [(code, data) for code, data in indicator_values.items() if data['signal_count'] > 0]
        
        # 按信号数量排序
        signal_stocks.sort(key=lambda x: x[1]['signal_count'], reverse=True)
        
        return [
            {
                'stock_code': code,
                'signal_count': data['signal_count'],
                'latest_signal_date': data['latest_signal_date'],
                'last_close': data['last_close']
            }
            for code, data in signal_stocks[:top_n]
        ]
    
    def test_no_selection_indicators(self, history_days: int = 30) -> Dict[str, Any]:
        """测试之前无选股能力的37个指标"""
        
        # 无选股的37个指标
        no_selection_indicators = [
            # 第1批-基础指标
            'macd', 'boll', 'kdj',
            # 第2批-趋势指标
            'sar',
            # 第3批-成交量指标
            'emv',
            # 第4批-波动率指标
            'stock_vix',
            # 第5批-ZXM专业指标
            'amplitude_elasticity', 'rise_elasticity', 'elasticity_score', 'market_breadth',
            # 第6批-增强指标
            'enhanced_rsi', 'enhanced_dmi', 'enhanced_macd_trend', 'enhanced_trix', 
            'enhanced_kdj_osc', 'enhanced_obv', 'enhanced_stochrsi', 'enhanced_wr', 'enhanced_macd_root',
            # 第7批-复合形态指标
            'composite', 'unified_ma', 'chip_distribution', 'institutional_behavior', 
            'candlestick_patterns', 'advanced_candlestick', 'patterns',
            # 第8批-工具公式指标
            'fibonacci_tools', 'gann_tools', 'elliott_wave', 'kdj_condition', 'macd_condition',
            # 第9批-多周期指标
            'monthly_kdj_trend_up', 'monthly_macd', 'weekly_kdj_d_or_dea_trend_up', 
            'weekly_kdj_d_trend_up', 'weekly_macd',
            # 第10批-震荡指标
            'stochrsi'
        ]
        
        print(f"🔬 开始历史信号测试")
        print(f"📋 测试指标数量: {len(no_selection_indicators)}")
        print(f"📈 历史检查天数: {history_days}")
        print("=" * 80)
        
        all_results = {}
        improved_indicators = []
        still_no_selection = []
        
        for i, indicator_name in enumerate(no_selection_indicators, 1):
            print(f"\n🔍 测试指标 {i}/{len(no_selection_indicators)}: {indicator_name}")
            
            try:
                result = self.validate_single_indicator_historical(
                    indicator_name, 
                    history_days=history_days,
                    max_stocks=50  # 使用较小的股票池进行快速测试
                )
                
                all_results[indicator_name] = result
                
                if result.get('statistics', {}).get('selected_count', 0) > 0:
                    selected_count = result['statistics']['selected_count']
                    total_signals = result['statistics']['total_signals']
                    print(f"   ✅ 改善成功！选出 {selected_count} 只股票，总信号数 {total_signals}")
                    improved_indicators.append(indicator_name)
                else:
                    print(f"   ❌ 仍无选股结果")
                    still_no_selection.append(indicator_name)
                    
            except Exception as e:
                print(f"   ❌ 测试失败: {e}")
                all_results[indicator_name] = {
                    'indicator_name': indicator_name,
                    'status': 'failed',
                    'error': str(e)
                }
                still_no_selection.append(indicator_name)
        
        # 生成总结报告
        print(f"\n📊 历史信号测试总结")
        print("=" * 80)
        print(f"🎯 总体改善情况:")
        print(f"   测试指标总数: {len(no_selection_indicators)}")
        print(f"   获得选股能力: {len(improved_indicators)} 个")
        print(f"   仍无选股能力: {len(still_no_selection)} 个")
        print(f"   改善率: {len(improved_indicators)/len(no_selection_indicators)*100:.1f}%")
        
        if improved_indicators:
            print(f"\n✅ 获得选股能力的指标 ({len(improved_indicators)}个):")
            for indicator in improved_indicators:
                if indicator in all_results and 'statistics' in all_results[indicator]:
                    stats = all_results[indicator]['statistics']
                    print(f"   {indicator}: {stats['selected_count']}只股票, {stats['total_signals']}个信号")
        
        if still_no_selection:
            print(f"\n❌ 仍无选股能力的指标 ({len(still_no_selection)}个):")
            for indicator in still_no_selection:
                print(f"   {indicator}")
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = f"results/validation/historical_signal_test_{timestamp}.json"
        
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'test_type': 'historical_signal_validation',
                'history_days': history_days,
                'test_time': datetime.now().isoformat(),
                'summary': {
                    'total_indicators': len(no_selection_indicators),
                    'improved_indicators': len(improved_indicators),
                    'still_no_selection': len(still_no_selection),
                    'improvement_rate': len(improved_indicators)/len(no_selection_indicators)
                },
                'improved_indicators': improved_indicators,
                'still_no_selection': still_no_selection,
                'detailed_results': all_results
            }, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\n💾 结果已保存到: {result_file}")
        
        return {
            'improved_indicators': improved_indicators,
            'still_no_selection': still_no_selection,
            'all_results': all_results
        }

if __name__ == "__main__":
    validator = HistoricalSignalValidator()
    validator.test_no_selection_indicators(history_days=30) 