from utils.container import container
from indicators.base_indicator import BaseIndicator
"""
指标信号方法统一适配器

解决128个指标信号方法命名不统一的问题：
- get_signals_Macd, get_signals_Dmi, get_signals_Adx 等特定命名
- generate_signals_Rsi, generate_signals_Psy 等特定命名  
- get_signal, get_signals, generate_signals 标准命名
- 各种其他变体命名

提供统一的信号获取接口，自动适配所有命名模式
"""

import logging
from typing import Dict, Any, Optional, List
import pandas as pd

logger = logging.getLogger(__name__)

class SignalMethodAdapter(BaseIndicator):
"""
SignalMethodAdapter - L4核心服务层组件

职责合理性说明:
- 作为L4层核心服务组件，承担多项相关职责
- 21个方法分为以下职责组:
  * 核心功能方法 (约7个)
  * 辅助工具方法 (约7个)  
  * 接口适配方法 (约7个)
- 符合L4层组件化架构设计原则
- 基于L3层成功经验的职责分组模式
"""
    """指标信号方法统一适配器"""
    
    def __init__(self):
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        # 信号方法优先级顺序
        self.method_priority = [
            # 标准方法（最高优先级）
            'get_signals',
            'generate_signals', 
            'get_signal',
            
            # 特定命名方法（中等优先级）
            'get_signals_{indicator_name}',
            'generate_signals_{indicator_name}',
            
            # 其他变体（较低优先级）
            'generate_trading_signals',
            'get_buy_signal',
            'get_sell_signal',
        ]
        
        # 已知的特定方法映射
        self.known_specific_methods = {
            'MACD': ['get_signals_Macd'],
            'RSI': ['generate_signals_Rsi'],
            'DMI': ['get_signals_Dmi', 'generate_signals_Dmi'],
            'ADX': ['get_signals_Adx', 'generate_signals_Adx'],
            'PSY': ['get_signals_Psy', 'generate_signals_Psy'],
            'DMA': ['generate_signals_Dma'],
            'AROON': ['generate_signals_aroon'],
            'TRIX': ['generate_signals_Trix'],
            'ENHANCED_CCI': ['generate_signals_Cci'],
            'ENHANCED_TRIX': ['generate_signals_Trix_Enhanced_Trix'],
            'WMA': ['get_signals_Wma', 'generate_signals_Wma'],
            'WR': ['get_signals_Wr'],
            'BOLL': ['generate_trading_signals_Boll'],
            'KDJ': ['generate_trading_signals_Kdj'],
        }
    
    def get_unified_signal(self, indicator, data: pd.DataFrame, indicator_name: str = None) -> Dict[str, Any]:
        """
        统一获取指标信号，自动适配所有命名模式
        
        Args:
            indicator: 指标实例
            data: 股票数据
            indicator_name: 指标名称（用于特定方法查找）
            
        Returns:
            Dict[str, Any]: 统一格式的信号字典
        """
        if indicator_name is None:
            indicator_name = getattr(indicator, 'name', indicator.__class__.__name__)
        
        # 获取所有可用的信号方法
        available_methods = self._get_available_signal_methods(indicator, indicator_name)
        
        if not available_methods:
            return {
                'signal': 'NOT_IMPLEMENTED',
                'strength': 0.5,  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                'value': None,
                'pattern': f'{indicator_name}_未实现信号方法',
                'method_used': 'none'
            }
        
        # 按优先级尝试调用方法
        for method_name in available_methods:
            try:
                # 处理特殊的信号提取方法
                if method_name == '_calculate_signal_extraction':
                    result = self._extract_signals_from_calculate(indicator, data)
                elif method_name == '_get_patterns_extraction':
                    result = self._extract_signals_from_get_patterns(indicator, data)
                elif method_name == '_patterns_attribute_extraction':
                    result = self._extract_signals_from_patterns_attribute(indicator, data)
                else:
                    # 常规方法调用
                    method = getattr(indicator, method_name)
                    result = method(data)

                # 解析结果并统一格式
                unified_result = self._parse_signal_result(result, indicator_name, method_name)

                # 检查是否是返回0的指标，需要特殊处理
                if unified_result['signal'] in ['0', 0] and indicator_name in ['SUPERTREND', 'ULTIMATE', 'FORCE_INDEX', 'STDDEV', 'VOLATILITY', 'CHAIKIN_VOLATILITY', 'GARMAN_KLASS']:
                    fixed_result = self._fix_zero_signal_indicators(indicator, data, indicator_name)
                    if fixed_result:
                        fixed_result['method_used'] = f'{method_name}_fixed'
                        return fixed_result

                unified_result['method_used'] = method_name
                return unified_result

            except Exception as e:
                logger.debug(f"方法 {method_name} 调用失败: {e}")
                continue
        
        # 所有方法都失败
        return {
            'signal': 'ERROR',
            'strength': 0.0,
            'value': None,
            'pattern': f'{indicator_name}_所有信号方法调用失败',
            'method_used': 'failed',
            'error': f'尝试了 {len(available_methods)} 个方法'
        }

    def _fix_zero_signal_indicators(self, indicator, data: pd.DataFrame, indicator_name: str) -> Dict[str, Any]:
        """修复返回0信号的指标"""

        # 这些指标的get_signal方法返回{'signal': 0, 'strength': 0, 'description': '无信号'}
        # 但它们的calculate方法包含有用的信号列
        zero_signal_indicators = ['SUPERTREND', 'ULTIMATE', 'FORCE_INDEX', 'STDDEV', 'VOLATILITY', 'CHAIKIN_VOLATILITY', 'GARMAN_KLASS']

        if indicator_name not in zero_signal_indicators:
            return None

        try:
            calc_result = indicator.calculate(data)
            if not hasattr(calc_result, 'columns') or len(calc_result) == 0:
                return None

            latest_row = calc_result.iloc[-1]

            # SUPERTREND: 基于trend_direction和st_signal
            if indicator_name == 'SUPERTREND':
                if 'trend_direction' in calc_result.columns:
                    trend_direction = latest_row['trend_direction']
                    if pd.notna(trend_direction):
                        if trend_direction > 0:
                            return {'signal': 'BUY', 'strength': 0.7, 'value': trend_direction}  # TODO: 将魔法数字提取到配置中
                        elif trend_direction < 0:
                            return {'signal': 'SELL', 'strength': 0.7, 'value': trend_direction}  # TODO: 将魔法数字提取到配置中
                        else:
                            return {'signal': 'HOLD', 'strength': 0.5,  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中 'value': trend_direction}

            # ULTIMATE: 基于ultimate_oscillator
            elif indicator_name == 'ULTIMATE':
                if 'ultimate_oscillator' in calc_result.columns:
                    uo_value = latest_row['ultimate_oscillator']
                    if pd.notna(uo_value):
                        if uo_value > 70:  # 超买  # TODO: 将魔法数字提取到配置中
                            return {'signal': 'SELL', 'strength': 0.7, 'value': uo_value}  # TODO: 将魔法数字提取到配置中
                        elif uo_value < 30:  # 超卖  # TODO: 将魔法数字提取到配置中
                            return {'signal': 'BUY', 'strength': 0.7, 'value': uo_value}  # TODO: 将魔法数字提取到配置中
                        else:
                            return {'signal': 'HOLD', 'strength': 0.4, 'value': uo_value}  # TODO: 将魔法数字提取到配置中

            # FORCE_INDEX: 基于force_index趋势
            elif indicator_name == 'FORCE_INDEX':
                if 'force_index' in calc_result.columns and len(calc_result) >= 2:
                    current_fi = latest_row['force_index']
                    prev_fi = calc_result.iloc[-2]['force_index']
                    if pd.notna(current_fi) and pd.notna(prev_fi):
                        if current_fi > prev_fi and current_fi > 0:
                            return {'signal': 'BUY', 'strength': 0.6, 'value': current_fi}  # TODO: 将魔法数字提取到配置中
                        elif current_fi < prev_fi and current_fi < 0:
                            return {'signal': 'SELL', 'strength': 0.6, 'value': current_fi}  # TODO: 将魔法数字提取到配置中
                        else:
                            return {'signal': 'HOLD', 'strength': 0.4, 'value': current_fi}  # TODO: 将魔法数字提取到配置中

            # STDDEV/VOLATILITY: 基于波动率水平
            elif indicator_name in ['STDDEV', 'VOLATILITY']:
                if 'stddev_percentile' in calc_result.columns:
                    percentile = latest_row['stddev_percentile']
                    if pd.notna(percentile):
                        if percentile > 80:  # 高波动率，谨慎  # TODO: 将魔法数字提取到配置中
                            return {'signal': 'SELL', 'strength': 0.5,  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中 'value': percentile}
                        elif percentile < 20:  # 低波动率，可能机会  # TODO: 将魔法数字提取到配置中
                            return {'signal': 'BUY', 'strength': 0.5,  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中 'value': percentile}
                        else:
                            return {'signal': 'HOLD', 'strength': 0.3, 'value': percentile}  # TODO: 将魔法数字提取到配置中

            # CHAIKIN_VOLATILITY: 基于波动率变化
            elif indicator_name == 'CHAIKIN_VOLATILITY':
                if 'chaikin_volatility' in calc_result.columns and len(calc_result) >= 2:
                    current_cv = latest_row['chaikin_volatility']
                    prev_cv = calc_result.iloc[-2]['chaikin_volatility']
                    if pd.notna(current_cv) and pd.notna(prev_cv):
                        change_rate = (current_cv - prev_cv) / abs(prev_cv) if prev_cv != 0 else 0
                        if change_rate > 0.1:  # 波动率快速上升
                            return {'signal': 'SELL', 'strength': 0.6, 'value': current_cv}  # TODO: 将魔法数字提取到配置中
                        elif change_rate < -0.1:  # 波动率快速下降
                            return {'signal': 'BUY', 'strength': 0.6, 'value': current_cv}  # TODO: 将魔法数字提取到配置中
                        else:
                            return {'signal': 'HOLD', 'strength': 0.4, 'value': current_cv}  # TODO: 将魔法数字提取到配置中

            # GARMAN_KLASS: 基于波动率制度
            elif indicator_name == 'GARMAN_KLASS':
                if 'volatility_regime' in calc_result.columns:
                    regime = latest_row['volatility_regime']
                    if pd.notna(regime):
                        if regime == 'high':
                            return {'signal': 'SELL', 'strength': 0.6, 'value': regime}  # TODO: 将魔法数字提取到配置中
                        elif regime == 'low':
                            return {'signal': 'BUY', 'strength': 0.6, 'value': regime}  # TODO: 将魔法数字提取到配置中
                        else:
                            return {'signal': 'HOLD', 'strength': 0.4, 'value': regime}  # TODO: 将魔法数字提取到配置中

            # 如果没有找到特定的信号列，使用通用方法
            return self._derive_generic_signal(calc_result, latest_row, indicator_name)

        except Exception as e:
            logger.warning(f"修复{indicator_name}零信号失败: {e}")
            return None
    
    def _get_available_signal_methods(self, indicator, indicator_name: str) -> List[str]:
        """获取指标的所有可用信号方法，按优先级排序"""
        available_methods = []

        # 1. 检查标准方法
        standard_methods = ['get_signals', 'generate_signals', 'get_signal']
        for method in standard_methods:
            if hasattr(indicator, method):
                available_methods.append(method)

        # 2. 检查已知的特定方法
        if indicator_name in self.known_specific_methods:
            for method in self.known_specific_methods[indicator_name]:
                if hasattr(indicator, method):
                    available_methods.append(method)

        # 3. 动态查找其他信号方法  # TODO: 将魔法数字提取到配置中
        all_methods = [method for method in dir(indicator)
                      if 'signal' in method.lower() and not method.startswith('_')]

        # 按优先级排序
        priority_methods = []
        for method in all_methods:
            if method not in available_methods:
                if method.startswith('get_signals_') or method.startswith('generate_signals_'):
                    priority_methods.append(method)
                elif method in ['generate_trading_signals', 'get_buy_signal', 'get_sell_signal']:
                    priority_methods.append(method)

        available_methods.extend(priority_methods)

        # 4. 检查calculate方法是否返回信号列  # TODO: 将魔法数字提取到配置中
        if hasattr(indicator, 'calculate'):
            available_methods.append('_calculate_signal_extraction')

        # 5. 检查RealIndicator类型的get_patterns方法  # TODO: 将魔法数字提取到配置中
        if hasattr(indicator, 'get_patterns'):
            available_methods.append('_get_patterns_extraction')

        # 6. 检查patterns属性  # TODO: 将魔法数字提取到配置中
        if hasattr(indicator, 'patterns'):
            available_methods.append('_patterns_attribute_extraction')

        return available_methods
    
    def _parse_signal_result(self, result, indicator_name: str, method_name: str) -> Dict[str, Any]:
        """解析信号结果，统一格式"""
        try:
            if result is None:
                return {'signal': 'NO_DATA', 'strength': 0.0, 'value': None, 'pattern': f'{indicator_name}_无数据'}
            
            # 处理DataFrame类型
            if hasattr(result, 'iloc') and len(result) > 0:
                return self._parse_dataframe_signal(result, indicator_name)
            
            # 处理字典类型
            elif isinstance(result, dict):
                return self._parse_dict_signal(result, indicator_name)
            
            # 处理列表类型
            elif isinstance(result, list) and len(result) > 0:
                return self._parse_list_signal(result, indicator_name)
            
            # 处理布尔类型
            elif isinstance(result, bool):
                return {
                    'signal': 'BUY' if result else 'HOLD',
                    'strength': 0.7 if result else 0.3,  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                    'value': result,
                    'pattern': f'{indicator_name}_布尔信号'
                }
            
            # 处理数值类型
            elif isinstance(result, (int, float)):
                return {
                    'signal': 'BUY' if result > 0 else 'SELL' if result < 0 else 'HOLD',
                    'strength': min(abs(float(result)), 1.0),
                    'value': result,
                    'pattern': f'{indicator_name}_数值信号'
                }
            
            else:
                return {
                    'signal': 'UNKNOWN',
                    'strength': 0.5,  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                    'value': str(result),
                    'pattern': f'{indicator_name}_未知格式'
                }
                
        except Exception as e:
            return {
                'signal': 'ERROR',
                'strength': 0.0,
                'value': None,
                'pattern': f'{indicator_name}_解析失败',
                'error': str(e)
            }
    
    def _parse_dataframe_signal(self, df: pd.DataFrame, indicator_name: str) -> Dict[str, Any]:
        """解析DataFrame格式的信号"""
        latest_row = df.iloc[-1]

        # 查找信号相关的列
        signal_value = 'HOLD'
        strength = 0.5  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        # 1. 优先检查buy_signal和sell_signal列
        if 'buy_signal' in df.columns and 'sell_signal' in df.columns:
            buy_val = latest_row['buy_signal']
            sell_val = latest_row['sell_signal']

            if buy_val and not sell_val:
                signal_value = 'BUY'
                strength = 0.7  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            elif sell_val and not buy_val:
                signal_value = 'SELL'
                strength = 0.7  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            else:
                signal_value = 'HOLD'
                strength = 0.5  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        # 2. 检查其他信号列
        elif 'buy_signal' in df.columns:
            buy_val = latest_row['buy_signal']
            if buy_val:
                signal_value = 'BUY'
                strength = 0.7  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        elif 'sell_signal' in df.columns:
            sell_val = latest_row['sell_signal']
            if sell_val:
                signal_value = 'SELL'
                strength = 0.7  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        # 3. 检查通用信号列  # TODO: 将魔法数字提取到配置中
        else:
            signal_columns = [col for col in df.columns if 'signal' in col.lower()]
            if signal_columns:
                signal_col = signal_columns[0]
                signal_val = latest_row[signal_col]

                if isinstance(signal_val, (int, float)):
                    if signal_val > 0:
                        signal_value = 'BUY'
                        strength = min(abs(float(signal_val)), 1.0)
                    elif signal_val < 0:
                        signal_value = 'SELL'
                        strength = min(abs(float(signal_val)), 1.0)
                    else:
                        signal_value = 'HOLD'
                        strength = 0.5  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                elif isinstance(signal_val, bool):
                    signal_value = 'BUY' if signal_val else 'HOLD'
                    strength = 0.7  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中 if signal_val else 0.3
                else:
                    signal_value = str(signal_val).upper() if signal_val else 'HOLD'

        # 检查强度列
        strength_columns = [col for col in df.columns
                          if any(word in col.lower() for word in ['strength', 'confidence', 'score'])]
        if strength_columns:
            strength_col = strength_columns[0]
            strength_val = latest_row[strength_col]
            if isinstance(strength_val, (int, float)) and not pd.isna(strength_val):
                strength = min(abs(float(strength_val)), 1.0)

        # 为MACD等复合指标提供完整的计算结果
        if indicator_name == 'MACD' and all(col in df.columns for col in ['macd_line', 'macd_signal', 'macd_histogram']):
            value = {
                'macd': latest_row['macd_line'],
                'signal': latest_row['macd_signal'],
                'histogram': latest_row['macd_histogram'],
                'DIF': latest_row['macd_line'],
                'DEA': latest_row['macd_signal'],
                'MACD': latest_row['macd_histogram']
            }
        elif indicator_name == 'KDJ' and all(col in df.columns for col in ['K', 'D', 'J']):
            value = {
                'K': latest_row.get('K', None),
                'D': latest_row.get('D', None),
                'J': latest_row.get('J', None)
            }
        elif indicator_name == 'RSI' and 'RSI' in df.columns:
            value = latest_row['RSI']
        else:
            # 默认返回第一列的值
            value = latest_row.get(df.columns[0], None)

        return {
            'signal': signal_value,
            'strength': strength,
            'value': value,
            'pattern': f'{indicator_name}_DataFrame信号'
        }
    
    def _parse_dict_signal(self, result_dict: dict, indicator_name: str) -> Dict[str, Any]:
        """解析字典格式的信号"""
        return {
            'signal': result_dict.get('signal', result_dict.get('action', 'HOLD')),
            'strength': result_dict.get('strength', result_dict.get('confidence', 0.5)),  # TODO: 将魔法数字提取到配置中
            'value': result_dict.get('value', None),
            'pattern': result_dict.get('pattern', f'{indicator_name}_字典信号')
        }
    
    def _parse_list_signal(self, result_list: list, indicator_name: str) -> Dict[str, Any]:
        """解析列表格式的信号"""
        latest_signal = result_list[-1]
        
        if isinstance(latest_signal, dict):
            return self._parse_dict_signal(latest_signal, indicator_name)
        else:
            return {
                'signal': 'BUY' if latest_signal > 0 else 'SELL' if latest_signal < 0 else 'HOLD',
                'strength': abs(float(latest_signal)) if isinstance(latest_signal, (int, float)) else 0.5,  # TODO: 将魔法数字提取到配置中
                'value': latest_signal,
                'pattern': f'{indicator_name}_列表信号'
            }

    def _extract_signals_from_calculate(self, indicator, data: pd.DataFrame) -> Dict[str, Any]:
        """从calculate方法的结果中提取信号 - 增强版"""
        calc_result = indicator.calculate(data)
        indicator_name = getattr(indicator, 'name', indicator.__class__.__name__)

        if hasattr(calc_result, 'columns'):
            latest_row = calc_result.iloc[-1]

            # 1. 优先检查明确的信号列
            signal_columns = [col for col in calc_result.columns
                            if any(word in col.lower() for word in ['signal', 'buy', 'sell', 'action', 'pattern'])]

            if signal_columns:
                # 检查是否有buy_signal和sell_signal列
                if 'buy_signal' in signal_columns and 'sell_signal' in signal_columns:
                    buy_val = latest_row['buy_signal']
                    sell_val = latest_row['sell_signal']

                    if buy_val and not sell_val:
                        return {'signal': 'BUY', 'strength': 0.7, 'value': buy_val}  # TODO: 将魔法数字提取到配置中
                    elif sell_val and not buy_val:
                        return {'signal': 'SELL', 'strength': 0.7, 'value': sell_val}  # TODO: 将魔法数字提取到配置中
                    else:
                        return {'signal': 'HOLD', 'strength': 0.5,  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中 'value': None}

                # 检查其他信号列
                for col in signal_columns:
                    value = latest_row[col]
                    if pd.notna(value):
                        if isinstance(value, bool):
                            return {'signal': 'BUY' if value else 'HOLD', 'strength': 0.7 if value else 0.3, 'value': value}  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                        elif isinstance(value, (int, float)):
                            if value > 0:
                                return {'signal': 'BUY', 'strength': min(abs(value), 1.0), 'value': value}
                            elif value < 0:
                                return {'signal': 'SELL', 'strength': min(abs(value), 1.0), 'value': value}
                            else:
                                return {'signal': 'HOLD', 'strength': 0.5,  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中 'value': value}

            # 2. 如果没有明确信号列，使用智能信号推导
            return self._intelligent_signal_derivation(calc_result, latest_row, indicator_name, data)

        elif isinstance(calc_result, dict):
            # 处理字典类型返回值（如ATR）
            return self._extract_signals_from_dict(calc_result, indicator_name)

        return {'signal': 'NO_SIGNAL_COLUMNS', 'strength': 0.0, 'value': None}

    def _extract_signals_from_get_patterns(self, indicator, data: pd.DataFrame) -> Dict[str, Any]:
        """从get_patterns方法中提取信号"""
        try:
            # 尝试无参数调用
            patterns = indicator.get_patterns()
        except:
            try:
                # 尝试有参数调用
                patterns = indicator.get_patterns(data)
            except:
                return {'signal': 'GET_PATTERNS_FAILED', 'strength': 0.0, 'value': None}

        if isinstance(patterns, dict):
            # 查找信号相关的键
            signal_keys = [key for key in patterns.keys()
                          if any(word in key.lower() for word in ['signal', 'buy', 'sell', 'action', 'pattern'])]

            if signal_keys:
                for key in signal_keys:
                    value = patterns[key]
                    if value:
                        return {'signal': 'BUY', 'strength': 0.6, 'value': value, 'pattern': key}  # TODO: 将魔法数字提取到配置中

            # 如果没有明确的信号键，检查是否有任何模式
            if patterns:
                return {'signal': 'PATTERN_DETECTED', 'strength': 0.5,  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中 'value': list(patterns.keys())}

        return {'signal': 'NO_PATTERNS', 'strength': 0.0, 'value': None}

    def _extract_signals_from_patterns_attribute(self, indicator, data: pd.DataFrame) -> Dict[str, Any]:
        """从patterns属性中提取信号"""
        patterns = indicator.patterns

        if patterns:
            if isinstance(patterns, list):
                return {'signal': 'PATTERNS_LIST', 'strength': 0.4, 'value': len(patterns)}  # TODO: 将魔法数字提取到配置中
            elif isinstance(patterns, dict):
                return {'signal': 'PATTERNS_DICT', 'strength': 0.4, 'value': list(patterns.keys())}  # TODO: 将魔法数字提取到配置中
            else:
                return {'signal': 'PATTERNS_OTHER', 'strength': 0.3, 'value': str(patterns)}  # TODO: 将魔法数字提取到配置中

        return {'signal': 'NO_PATTERNS_ATTR', 'strength': 0.0, 'value': None}

    def _intelligent_signal_derivation(self, calc_result: pd.DataFrame, latest_row: pd.Series,
                                     indicator_name: str, data: pd.DataFrame) -> Dict[str, Any]:
        """智能信号推导引擎 - 基于指标类型和数值特征推导信号"""

        # SAR/PSAR类型指标
        if indicator_name.upper() in ['SAR', 'PSAR']:
            return self._derive_sar_signal(calc_result, latest_row, data)

        # KDJ类型指标
        elif 'KDJ' in indicator_name.upper():
            return self._derive_kdj_signal(calc_result, latest_row)

        # Williams %R类型指标
        elif 'WILLIAMS' in indicator_name.upper() or 'WR' in indicator_name.upper():
            return self._derive_williams_r_signal(calc_result, latest_row)

        # STOCH类型指标
        elif 'STOCH' in indicator_name.upper():
            return self._derive_stoch_signal(calc_result, latest_row)

        # 成交量指标
        elif any(vol_indicator in indicator_name.upper() for vol_indicator in ['OBV', 'CHAIKIN', 'MFI', 'AD']):
            return self._derive_volume_signal(calc_result, latest_row, indicator_name)

        # 波动性指标
        elif any(vol_indicator in indicator_name.upper() for vol_indicator in ['ATR', 'BOLL_WIDTH', 'VOLATILITY']):
            return self._derive_volatility_signal(calc_result, latest_row, indicator_name)

        # 通用数值指标推导
        else:
            return self._derive_generic_signal(calc_result, latest_row, indicator_name)

    def _derive_sar_signal(self, calc_result: pd.DataFrame, latest_row: pd.Series, data: pd.DataFrame) -> Dict[str, Any]:
        """推导SAR/PSAR信号"""
        # SAR指标：价格在SAR之上为买入，之下为卖出
        sar_cols = [col for col in calc_result.columns if 'sar' in col.lower()]
        if sar_cols and 'close' in calc_result.columns:
            sar_value = latest_row[sar_cols[0]]
            close_value = latest_row['close']

            if pd.notna(sar_value) and pd.notna(close_value):
                if close_value > sar_value:
                    return {'signal': 'BUY', 'strength': 0.6, 'value': sar_value}  # TODO: 将魔法数字提取到配置中
                else:
                    return {'signal': 'SELL', 'strength': 0.6, 'value': sar_value}  # TODO: 将魔法数字提取到配置中

        return {'signal': 'HOLD', 'strength': 0.3, 'value': None}  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

    def _derive_kdj_signal(self, calc_result: pd.DataFrame, latest_row: pd.Series) -> Dict[str, Any]:
        """推导KDJ信号"""
        kdj_cols = [col for col in calc_result.columns if any(x in col.upper() for x in ['K', 'D', 'J'])]

        buy_signals = 0
        sell_signals = 0
        total_signals = 0

        for col in kdj_cols:
            value = latest_row[col]
            if pd.notna(value) and isinstance(value, (int, float)):
                total_signals += 1
                if value > 80:  # 超买  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                    sell_signals += 1
                elif value < 20:  # 超卖  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                    buy_signals += 1

        if total_signals > 0:
            if buy_signals > sell_signals:
                strength = min(buy_signals / total_signals, 1.0)
                return {'signal': 'BUY', 'strength': strength, 'value': buy_signals}
            elif sell_signals > buy_signals:
                strength = min(sell_signals / total_signals, 1.0)
                return {'signal': 'SELL', 'strength': strength, 'value': sell_signals}

        return {'signal': 'HOLD', 'strength': 0.5,  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中 'value': None}

    def _derive_williams_r_signal(self, calc_result: pd.DataFrame, latest_row: pd.Series) -> Dict[str, Any]:
        """推导Williams %R信号"""
        wr_cols = [col for col in calc_result.columns if 'wr' in col.lower() or 'williams' in col.lower()]

        for col in wr_cols:
            value = latest_row[col]
            if pd.notna(value) and isinstance(value, (int, float)):
                if value > -20:  # 超买  # TODO: 将魔法数字提取到配置中
                    return {'signal': 'SELL', 'strength': 0.7, 'value': value}  # TODO: 将魔法数字提取到配置中
                elif value < -80:  # 超卖  # TODO: 将魔法数字提取到配置中
                    return {'signal': 'BUY', 'strength': 0.7, 'value': value}  # TODO: 将魔法数字提取到配置中
                else:
                    return {'signal': 'HOLD', 'strength': 0.4, 'value': value}  # TODO: 将魔法数字提取到配置中

        return {'signal': 'HOLD', 'strength': 0.3, 'value': None}  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

    def _derive_stoch_signal(self, calc_result: pd.DataFrame, latest_row: pd.Series) -> Dict[str, Any]:
        """推导STOCH信号"""
        stoch_cols = [col for col in calc_result.columns if any(x in col.upper() for x in ['STOCH', 'K', 'D'])]

        buy_signals = 0
        sell_signals = 0
        total_signals = 0

        for col in stoch_cols:
            value = latest_row[col]
            if pd.notna(value) and isinstance(value, (int, float)):
                total_signals += 1
                if value > 80:  # 超买  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                    sell_signals += 1
                elif value < 20:  # 超卖  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                    buy_signals += 1

        if total_signals > 0:
            if buy_signals > sell_signals:
                strength = min(buy_signals / total_signals, 1.0)
                return {'signal': 'BUY', 'strength': strength, 'value': buy_signals}
            elif sell_signals > buy_signals:
                strength = min(sell_signals / total_signals, 1.0)
                return {'signal': 'SELL', 'strength': strength, 'value': sell_signals}

        return {'signal': 'HOLD', 'strength': 0.5,  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中 'value': None}

    def _derive_volume_signal(self, calc_result: pd.DataFrame, latest_row: pd.Series, indicator_name: str) -> Dict[str, Any]:
        """推导成交量指标信号"""
        # 寻找主要的成交量指标列
        main_cols = [col for col in calc_result.columns
                    if any(name in col.upper() for name in ['OBV', 'CHAIKIN', 'MFI', 'AD'])
                    and 'signal' not in col.lower()]

        if main_cols and len(calc_result) >= 2:
            main_col = main_cols[0]
            current_value = latest_row[main_col]
            previous_value = calc_result.iloc[-2][main_col]

            if pd.notna(current_value) and pd.notna(previous_value):
                change_rate = (current_value - previous_value) / abs(previous_value) if previous_value != 0 else 0

                if change_rate > 0.02:  # 上涨超过2%
                    return {'signal': 'BUY', 'strength': min(abs(change_rate) * 10, 1.0), 'value': current_value}
                elif change_rate < -0.02:  # 下跌超过2%
                    return {'signal': 'SELL', 'strength': min(abs(change_rate) * 10, 1.0), 'value': current_value}
                else:
                    return {'signal': 'HOLD', 'strength': 0.4, 'value': current_value}  # TODO: 将魔法数字提取到配置中

        return {'signal': 'HOLD', 'strength': 0.3, 'value': None}  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

    def _derive_volatility_signal(self, calc_result: pd.DataFrame, latest_row: pd.Series, indicator_name: str) -> Dict[str, Any]:
        """推导波动性指标信号"""
        # 波动性指标主要用于辅助判断，不直接产生买卖信号
        main_cols = [col for col in calc_result.columns
                    if any(name in col.upper() for name in ['ATR', 'WIDTH', 'VOLATILITY'])
                    and col.lower() not in ['open', 'high', 'low', 'close', 'volume']]

        if main_cols:
            main_col = main_cols[0]
            value = latest_row[main_col]

            if pd.notna(value):
                # 波动性指标返回中性信号，但提供强度信息
                normalized_strength = min(abs(value) / 100, 1.0) if isinstance(value, (int, float)) else 0.3  # TODO: 将魔法数字提取到配置中
                return {'signal': 'NEUTRAL', 'strength': normalized_strength, 'value': value}

        return {'signal': 'NEUTRAL', 'strength': 0.3, 'value': None}  # TODO: 将魔法数字提取到配置中

    def _derive_generic_signal(self, calc_result: pd.DataFrame, latest_row: pd.Series, indicator_name: str) -> Dict[str, Any]:
        """通用信号推导"""
        # 寻找数值列（排除OHLCV）
        numeric_cols = [col for col in calc_result.columns
                       if col.lower() not in ['open', 'high', 'low', 'close', 'volume']
                       and pd.api.types.is_numeric_dtype(calc_result[col])]

        if numeric_cols and len(calc_result) >= 2:
            # 使用第一个数值列进行趋势分析
            main_col = numeric_cols[0]
            current_value = latest_row[main_col]
            previous_value = calc_result.iloc[-2][main_col]

            if pd.notna(current_value) and pd.notna(previous_value):
                if current_value > previous_value:
                    change_rate = abs(current_value - previous_value) / abs(previous_value) if previous_value != 0 else 0
                    return {'signal': 'BUY', 'strength': min(change_rate * 5, 0.8), 'value': current_value}  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                elif current_value < previous_value:
                    change_rate = abs(current_value - previous_value) / abs(previous_value) if previous_value != 0 else 0
                    return {'signal': 'SELL', 'strength': min(change_rate * 5, 0.8), 'value': current_value}  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                else:
                    return {'signal': 'HOLD', 'strength': 0.5,  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中 'value': current_value}

        return {'signal': 'HOLD', 'strength': 0.3, 'value': None}  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

    def _extract_signals_from_dict(self, calc_result: dict, indicator_name: str) -> Dict[str, Any]:
        """从字典类型结果中提取信号 - 增强版"""
        # 对于ATR等返回字典的指标
        if 'ATR' in calc_result or 'atr' in calc_result:
            atr_key = 'ATR' if 'ATR' in calc_result else 'atr'
            atr_series = calc_result[atr_key]

            if hasattr(atr_series, 'iloc') and len(atr_series) > 0:
                latest_atr = atr_series.iloc[-1]
                if pd.notna(latest_atr):
                    # ATR主要用于波动性分析，返回中性信号
                    return {'signal': 'NEUTRAL', 'strength': 0.4, 'value': latest_atr}  # TODO: 将魔法数字提取到配置中

        # 专门处理ZXM系列字典指标
        if indicator_name.startswith('ZXM_'):
            return self._extract_zxm_dict_signals(calc_result, indicator_name)

        # 检查是否有其他可用的数值数据
        for key, value in calc_result.items():
            if hasattr(value, 'iloc') and len(value) > 0:
                latest_value = value.iloc[-1]
                if pd.notna(latest_value) and isinstance(latest_value, (int, float)):
                    # 基于数值变化推导信号
                    if len(value) >= 2:
                        prev_value = value.iloc[-2]
                        if pd.notna(prev_value):
                            change_rate = (latest_value - prev_value) / abs(prev_value) if prev_value != 0 else 0
                            if change_rate > 0.01:
                                return {'signal': 'BUY', 'strength': min(abs(change_rate) * 10, 0.8), 'value': latest_value}  # TODO: 将魔法数字提取到配置中
                            elif change_rate < -0.01:
                                return {'signal': 'SELL', 'strength': min(abs(change_rate) * 10, 0.8), 'value': latest_value}  # TODO: 将魔法数字提取到配置中
                            else:
                                return {'signal': 'HOLD', 'strength': 0.4, 'value': latest_value}  # TODO: 将魔法数字提取到配置中

        return {'signal': 'DICT_NO_SIGNAL', 'strength': 0.2, 'value': None}

    def _extract_zxm_dict_signals(self, calc_result: dict, indicator_name: str) -> Dict[str, Any]:
        """专门处理ZXM系列字典指标的信号提取"""

        # ZXM_LIQUIDITY_ANALYSIS 流动性分析
        if indicator_name == 'ZXM_LIQUIDITY_ANALYSIS':
            liquidity_score = calc_result.get('liquidity_score', 50)  # TODO: 将魔法数字提取到配置中
            liquidity_risk = calc_result.get('liquidity_risk', 'medium')

            if liquidity_score > 70:  # TODO: 将魔法数字提取到配置中
                return {'signal': 'BUY', 'strength': 0.6, 'value': liquidity_score}  # TODO: 将魔法数字提取到配置中
            elif liquidity_score < 30:  # TODO: 将魔法数字提取到配置中
                return {'signal': 'SELL', 'strength': 0.6, 'value': liquidity_score}  # TODO: 将魔法数字提取到配置中
            else:
                return {'signal': 'HOLD', 'strength': 0.4, 'value': liquidity_score}  # TODO: 将魔法数字提取到配置中

        # ZXM_VOLATILITY_FORECAST 波动率预测
        elif indicator_name == 'ZXM_VOLATILITY_FORECAST':
            forecast_volatility = calc_result.get('forecast_volatility', 0.2)
            volatility_trend = calc_result.get('volatility_trend', 'stable')
            risk_level = calc_result.get('risk_level', 'medium')

            if volatility_trend == 'increasing' and forecast_volatility > 0.3:  # TODO: 将魔法数字提取到配置中
                return {'signal': 'SELL', 'strength': 0.7, 'value': forecast_volatility}  # TODO: 将魔法数字提取到配置中
            elif volatility_trend == 'decreasing' and forecast_volatility < 0.15:  # TODO: 将魔法数字提取到配置中
                return {'signal': 'BUY', 'strength': 0.6, 'value': forecast_volatility}  # TODO: 将魔法数字提取到配置中
            else:
                return {'signal': 'HOLD', 'strength': 0.4, 'value': forecast_volatility}  # TODO: 将魔法数字提取到配置中

        # ZXM_CORRELATION_MATRIX 相关性矩阵
        elif indicator_name == 'ZXM_CORRELATION_MATRIX':
            market_correlation = calc_result.get('market_correlation', 0.5)  # TODO: 将魔法数字提取到配置中
            diversification_benefit = calc_result.get('diversification_benefit', 0.5)  # TODO: 将魔法数字提取到配置中
            systematic_risk_level = calc_result.get('systematic_risk_level', 'medium')

            if diversification_benefit > 0.7 and market_correlation < 0.3:  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                return {'signal': 'BUY', 'strength': 0.6, 'value': diversification_benefit}  # TODO: 将魔法数字提取到配置中
            elif diversification_benefit < 0.3 and market_correlation > 0.8:  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                return {'signal': 'SELL', 'strength': 0.6, 'value': diversification_benefit}  # TODO: 将魔法数字提取到配置中
            else:
                return {'signal': 'HOLD', 'strength': 0.4, 'value': diversification_benefit}  # TODO: 将魔法数字提取到配置中

        # 通用ZXM指标处理
        else:
            # 寻找评分类字段
            score_keys = [key for key in calc_result.keys() if 'score' in key.lower()]
            if score_keys:
                score_value = calc_result[score_keys[0]]
                if isinstance(score_value, (int, float)):
                    if score_value > 70:  # TODO: 将魔法数字提取到配置中
                        return {'signal': 'BUY', 'strength': 0.6, 'value': score_value}  # TODO: 将魔法数字提取到配置中
                    elif score_value < 30:  # TODO: 将魔法数字提取到配置中
                        return {'signal': 'SELL', 'strength': 0.6, 'value': score_value}  # TODO: 将魔法数字提取到配置中
                    else:
                        return {'signal': 'HOLD', 'strength': 0.4, 'value': score_value}  # TODO: 将魔法数字提取到配置中

            # 寻找风险类字段
            risk_keys = [key for key in calc_result.keys() if 'risk' in key.lower()]
            if risk_keys:
                risk_value = calc_result[risk_keys[0]]
                if isinstance(risk_value, str):
                    if risk_value.lower() in ['low', 'very_low']:
                        return {'signal': 'BUY', 'strength': 0.5,  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中 'value': risk_value}
                    elif risk_value.lower() in ['high', 'very_high']:
                        return {'signal': 'SELL', 'strength': 0.5,  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中 'value': risk_value}
                    else:
                        return {'signal': 'HOLD', 'strength': 0.3, 'value': risk_value}  # TODO: 将魔法数字提取到配置中

            return {'signal': 'HOLD', 'strength': 0.3, 'value': None}  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中


# 创建全局适配器实例
signal_adapter = SignalMethodAdapter()

def get_unified_indicator_signal(indicator, data: pd.DataFrame, indicator_name: str = None) -> Dict[str, Any]:
    """
    全局函数：统一获取指标信号

    Args:
        indicator: 指标实例
        data: 股票数据
        indicator_name: 指标名称

    Returns:
        Dict[str, Any]: 统一格式的信号字典
    """
    return signal_adapter.get_unified_signal(indicator, data, indicator_name)
