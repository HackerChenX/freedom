#!/usr/bin/env python3
"""
买点批量分析器 - 生产级脚本
支持单股票和批量股票的买点分析
"""

import sys
import os
import argparse
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dependency_injection import get_service
from db.interfaces.data_access_interface import DataAccessInterface
from utils.logger import get_logger
from utils.decorators import exception_handler, performance_monitor
from indicators.complete_indicator_registry import CompleteIndicatorRegistry

logger = get_logger(__name__)

def convert_numpy_types(obj):
    """转换numpy类型为Python原生类型，用于JSON序列化"""
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

class BuypointBatchAnalyzer:
    """买点批量分析器"""
    
    def __init__(self):
        """初始化分析器"""
        self.data_access = get_service(DataAccessInterface)
        self.indicator_registry = CompleteIndicatorRegistry()
        self.indicator_registry.register_all_indicators()
        
    @exception_handler(reraise=True)
    @performance_monitor(threshold=30.0)
    def analyze_single_stock(self, stock_code: str, date: str, analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """
        分析单只股票的买点
        
        Args:
            stock_code: 股票代码
            date: 分析日期
            analysis_type: 分析类型
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        try:
            logger.info(f"开始分析股票 {stock_code} 在 {date} 的买点")
            
            # 获取股票数据
            query = f"""
            SELECT code, name, date, open, high, low, close, volume, turnover_rate
            FROM stock_info 
            WHERE code = '{stock_code}' 
            AND level = '日线'
            AND date <= '{date}'
            ORDER BY date DESC 
            LIMIT 200
            """
            
            stock_data = self.data_access.query_dataframe(query)
            
            if stock_data.empty:
                logger.warning(f"未找到股票 {stock_code} 的数据")
                return {
                    'stock_code': stock_code,
                    'analysis_date': date,
                    'status': 'NO_DATA',
                    'message': '未找到股票数据'
                }
            
            # 获取股票基本信息
            stock_info = stock_data.iloc[0]
            
            # 计算技术指标
            indicators_result = self._calculate_indicators(stock_data, analysis_type)
            
            # 生成买点信号
            buy_signal = self._generate_buy_signal(indicators_result)
            
            # 组装结果
            result = {
                'stock_code': stock_code,
                'stock_name': stock_info.get('name', ''),
                'analysis_date': date,
                'analysis_type': analysis_type,
                'status': 'SUCCESS',
                'indicators': indicators_result,
                'buy_signal': buy_signal,
                'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'data_points': len(stock_data)
            }
            
            logger.info(f"股票 {stock_code} 买点分析完成，信号: {buy_signal.get('signal', 'UNKNOWN')}")
            return result
            
        except Exception as e:
            logger.error(f"分析股票 {stock_code} 失败: {e}")
            return {
                'stock_code': stock_code,
                'analysis_date': date,
                'status': 'ERROR',
                'error': str(e),
                'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
    
    def _calculate_indicators(self, data, analysis_type: str) -> Dict[str, Any]:
        """计算技术指标"""
        indicators_result = {}
        
        try:
            # 获取所有注册的指标
            all_indicators = self.indicator_registry.get_all_indicators()
            indicator_names = list(all_indicators.keys())

            logger.info(f"开始计算所有 {len(indicator_names)} 个注册指标")

            
            for indicator_name in indicator_names:
                try:
                    indicator = self.indicator_registry.get_indicator(indicator_name)
                    if indicator:
                        # 计算指标
                        result = indicator.calculate(data)

                        # 尝试获取信号，适配不同的方法名
                        signal = self._get_indicator_signal(indicator, data)

                        indicators_result[indicator_name] = {
                            'signal': signal.get('signal', 'UNKNOWN'),
                            'value': signal.get('value', None),
                            'pattern': signal.get('pattern', ''),
                            'strength': signal.get('strength', 0.5)
                        }
                    else:
                        indicators_result[indicator_name] = {
                            'signal': 'NOT_AVAILABLE',
                            'error': f'指标 {indicator_name} 未找到'
                        }
                        
                except Exception as e:
                    logger.warning(f"计算指标 {indicator_name} 失败: {e}")
                    indicators_result[indicator_name] = {
                        'signal': 'ERROR',
                        'error': str(e)
                    }
            
            logger.info(f"成功计算 {len(indicators_result)} 个指标")
            return indicators_result
            
        except Exception as e:
            logger.error(f"计算指标失败: {e}")
            return {'error': str(e)}

    def _get_indicator_signal(self, indicator, data: pd.DataFrame) -> Dict[str, Any]:
        """
        获取指标信号，适配不同的方法名

        Args:
            indicator: 指标实例
            data: 股票数据

        Returns:
            Dict[str, Any]: 信号字典
        """
        try:
            # 尝试不同的信号获取方法
            if hasattr(indicator, 'get_signal'):
                # 单数形式的方法
                return indicator.get_signal(data)
            elif hasattr(indicator, 'get_signals'):
                # 复数形式的方法，需要转换格式
                signals_df = indicator.get_signals(data)
                if not signals_df.empty:
                    # 获取最新信号
                    latest_signals = signals_df.iloc[-1]

                    # 转换为标准格式
                    signal_value = 'HOLD'  # 默认值
                    strength = 0.5

                    # 根据不同指标的信号列名进行判断
                    for col in signals_df.columns:
                        if 'signal' in col.lower():
                            signal_val = latest_signals[col]
                            if signal_val > 0:
                                signal_value = 'BUY'
                                strength = 0.7
                            elif signal_val < 0:
                                signal_value = 'SELL'
                                strength = 0.7
                            break

                    return {
                        'signal': signal_value,
                        'strength': strength,
                        'value': latest_signals.get(signals_df.columns[0], None),
                        'pattern': f'{indicator.name}_信号'
                    }
                else:
                    return {'signal': 'NO_DATA', 'strength': 0.0}
            elif hasattr(indicator, 'generate_signals'):
                # 生成信号方法
                signals = indicator.generate_signals(data)
                if signals:
                    latest_signal = signals[-1] if isinstance(signals, list) else signals
                    return {
                        'signal': latest_signal.get('action', 'HOLD'),
                        'strength': latest_signal.get('strength', 0.5),
                        'value': latest_signal.get('value', None),
                        'pattern': latest_signal.get('pattern', '')
                    }
                else:
                    return {'signal': 'NO_SIGNAL', 'strength': 0.0}
            else:
                # 如果没有信号方法，返回默认值
                return {
                    'signal': 'NOT_IMPLEMENTED',
                    'strength': 0.5,
                    'value': None,
                    'pattern': f'{indicator.name}_未实现信号方法'
                }

        except Exception as e:
            return {
                'signal': 'ERROR',
                'strength': 0.0,
                'error': str(e)
            }

    def _generate_buy_signal(self, indicators_result: Dict[str, Any]) -> Dict[str, Any]:
        """生成买点信号"""
        try:
            buy_signals = 0
            total_signals = 0
            signal_details = []
            
            for indicator_name, indicator_data in indicators_result.items():
                if isinstance(indicator_data, dict) and 'signal' in indicator_data:
                    signal = indicator_data['signal']
                    total_signals += 1
                    
                    if signal in ['BUY', 'STRONG_BUY', 'BULLISH']:
                        buy_signals += 1
                        signal_details.append(f"{indicator_name}: {signal}")
            
            if total_signals == 0:
                buy_strength = 0.0
                overall_signal = 'NO_SIGNAL'
            else:
                buy_strength = buy_signals / total_signals
                
                if buy_strength >= 0.7:
                    overall_signal = 'STRONG_BUY'
                elif buy_strength >= 0.5:
                    overall_signal = 'BUY'
                elif buy_strength >= 0.3:
                    overall_signal = 'WEAK_BUY'
                else:
                    overall_signal = 'HOLD'
            
            return {
                'signal': overall_signal,
                'strength': buy_strength,
                'buy_signals_count': buy_signals,
                'total_signals_count': total_signals,
                'signal_details': signal_details,
                'confidence': min(buy_strength * 100, 100)
            }
            
        except Exception as e:
            logger.error(f"生成买点信号失败: {e}")
            return {
                'signal': 'ERROR',
                'error': str(e)
            }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='买点批量分析器')
    parser.add_argument('--stock-code', type=str, help='股票代码')
    parser.add_argument('--date', type=str, required=True, help='分析日期 (YYYY-MM-DD)')
    parser.add_argument('--analysis-type', type=str, default='comprehensive', 
                       choices=['basic', 'comprehensive', 'full_indicators'],
                       help='分析类型')
    parser.add_argument('--input', type=str, help='批量分析输入CSV文件')
    parser.add_argument('--output', type=str, required=True, help='输出文件或目录')
    parser.add_argument('--parallel', type=int, default=1, help='并行处理数量')
    
    args = parser.parse_args()
    
    try:
        analyzer = BuypointBatchAnalyzer()
        
        if args.stock_code:
            # 单股票分析
            result = analyzer.analyze_single_stock(args.stock_code, args.date, args.analysis_type)
            
            # 保存结果（转换numpy类型）
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            converted_result = convert_numpy_types(result)
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(converted_result, f, indent=2, ensure_ascii=False)
            
            print(f"✅ 股票 {args.stock_code} 分析完成，结果保存至: {args.output}")
            print(f"📊 买点信号: {result.get('buy_signal', {}).get('signal', 'UNKNOWN')}")
            
        elif args.input:
            # 批量分析
            print(f"🔄 批量分析功能开发中，输入文件: {args.input}")
            print(f"📁 输出目录: {args.output}")
            
        else:
            print("❌ 请指定 --stock-code 或 --input 参数")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"买点分析失败: {e}")
        print(f"❌ 分析失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
