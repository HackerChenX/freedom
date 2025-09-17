#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
增强的测试数据生成器

专门为买点分析测试生成高质量的模拟股票数据
确保生成的数据完全符合stockInfo格式要求
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.logger import get_logger

logger = get_logger(__name__)


class EnhancedTestDataGenerator:
    """增强的测试数据生成器"""
    
    def __init__(self):
        """初始化生成器"""
        self.base_config = {
            'base_price': 10.0,
            'min_price': 1.0,
            'max_price': 100.0,
            'base_volume': 2000000,
            'min_volume': 100000,
            'max_volume': 50000000
        }
        
        # 技术形态参数配置
        self.pattern_configs = self._initialize_pattern_configs()
        
    def _initialize_pattern_configs(self) -> Dict[str, Dict[str, Any]]:
        """初始化形态配置参数"""
        return {
            # 趋势形态配置
            'MA_GOLDEN_CROSS': {
                'trend_direction': 'up',
                'trend_strength': 0.3,
                'volatility': 0.02,
                'cross_point': 45,  # 在第45个数据点发生金叉
                'ma_periods': [5, 20]
            },
            'MA_DEATH_CROSS': {
                'trend_direction': 'down',
                'trend_strength': 0.3,
                'volatility': 0.02,
                'cross_point': 45,
                'ma_periods': [5, 20]
            },
            'EMA_GOLDEN_CROSS': {
                'trend_direction': 'up',
                'trend_strength': 0.25,
                'volatility': 0.025,
                'cross_point': 40,
                'ema_periods': [12, 26]
            },
            'WMA_GOLDEN_CROSS': {
                'trend_direction': 'up',
                'trend_strength': 0.28,
                'volatility': 0.022,
                'cross_point': 42,  # 在第42个数据点发生金叉
                'wma_periods': [5, 14]  # 短期和长期WMA周期
            },
            'WMA_DEATH_CROSS': {
                'trend_direction': 'down',
                'trend_strength': 0.28,
                'volatility': 0.022,
                'cross_point': 42,
                'wma_periods': [5, 14]
            },
            
            # 振荡器形态配置
            'RSI_OVERBOUGHT': {
                'trend_direction': 'up',
                'trend_strength': 0.4,
                'volatility': 0.03,
                'target_rsi': 75,  # 目标RSI值
                'rsi_period': 14
            },
            'RSI_OVERSOLD': {
                'trend_direction': 'down',
                'trend_strength': 0.4,
                'volatility': 0.03,
                'target_rsi': 25,
                'rsi_period': 14
            },
            'KDJ_GOLDEN_CROSS': {
                'trend_direction': 'up',
                'trend_strength': 0.2,
                'volatility': 0.025,
                'cross_point': 40,
                'kdj_periods': [9, 3, 3]
            },
            'KDJ_OVERBOUGHT': {
                'trend_direction': 'up',
                'trend_strength': 0.3,
                'volatility': 0.03,
                'target_k': 85,
                'target_d': 80
            },
            
            # 动量形态配置
            'MACD_GOLDEN_CROSS': {
                'trend_direction': 'up',
                'trend_strength': 0.25,
                'volatility': 0.02,
                'cross_point': 42,
                'macd_periods': [12, 26, 9]
            },
            'MACD_DEATH_CROSS': {
                'trend_direction': 'down',
                'trend_strength': 0.25,
                'volatility': 0.02,
                'cross_point': 42,
                'macd_periods': [12, 26, 9]
            },

            # 🔧 CCI形态配置 (关键修复)
            'CCI_GOLDEN_CROSS': {
                'trend_direction': 'up',
                'trend_strength': 0.3,
                'volatility': 0.025,
                'cross_point': 45,  # 在第45个数据点发生金叉
                'cci_period': 20,
                'target_cci': 50  # 目标CCI值
            },
            'CCI_DEATH_CROSS': {
                'trend_direction': 'down',
                'trend_strength': 0.3,
                'volatility': 0.025,
                'cross_point': 45,
                'cci_period': 20,
                'target_cci': -50
            },

            # 🔧 ADX形态配置 (关键修复)
            'ADX_TREND_STRENGTH': {
                'trend_direction': 'up',
                'trend_strength': 0.35,
                'volatility': 0.03,
                'adx_period': 14,
                'target_adx': 30,  # 目标ADX值（强趋势）
                'trend_buildup_point': 35,  # 在第35个数据点开始建立趋势
                'pdi_mdi_spread': 5  # +DI和-DI的差值
            },
            'ADX_WEAK_TREND': {
                'trend_direction': 'sideways',
                'trend_strength': 0.1,
                'volatility': 0.02,
                'adx_period': 14,
                'target_adx': 15,  # 目标ADX值（弱趋势）
                'trend_buildup_point': 40,
                'pdi_mdi_spread': 2
            },
            'CCI_ZERO_CROSS_UP': {
                'trend_direction': 'up',
                'trend_strength': 0.2,
                'volatility': 0.02,
                'cross_point': 40,
                'cci_period': 20,
                'target_cci': 25
            },
            'CCI_ZERO_CROSS_DOWN': {
                'trend_direction': 'down',
                'trend_strength': 0.2,
                'volatility': 0.02,
                'cross_point': 40,
                'cci_period': 20,
                'target_cci': -25
            },
            'CCI_OVERBOUGHT': {
                'trend_direction': 'up',
                'trend_strength': 0.4,
                'volatility': 0.03,
                'target_cci': 150,  # CCI超买阈值
                'cci_period': 20
            },
            'CCI_OVERSOLD': {
                'trend_direction': 'down',
                'trend_strength': 0.4,
                'volatility': 0.03,
                'target_cci': -150,  # CCI超卖阈值
                'cci_period': 20
            },
            'CCI_EXTREME_OVERBOUGHT': {
                'trend_direction': 'up',
                'trend_strength': 0.5,
                'volatility': 0.04,
                'target_cci': 250,  # CCI极度超买阈值
                'cci_period': 20
            },
            'CCI_EXTREME_OVERSOLD': {
                'trend_direction': 'down',
                'trend_strength': 0.5,
                'volatility': 0.04,
                'target_cci': -250,  # CCI极度超卖阈值
                'cci_period': 20
            },
            'CCI_BULLISH_DIVERGENCE': {
                'trend_direction': 'up_reversal',
                'trend_strength': 0.25,
                'volatility': 0.03,
                'divergence_point': 40,
                'cci_period': 20
            },
            'CCI_BEARISH_DIVERGENCE': {
                'trend_direction': 'down_reversal',
                'trend_strength': 0.25,
                'volatility': 0.03,
                'divergence_point': 40,
                'cci_period': 20
            },
            
            # 波动性形态配置
            'BOLL_UPPER_BREAKOUT': {
                'trend_direction': 'up',
                'trend_strength': 0.35,
                'volatility': 0.04,
                'breakout_point': 50,
                'boll_period': 20,
                'boll_std': 2
            },
            'BOLL_LOWER_BREAKOUT': {
                'trend_direction': 'down',
                'trend_strength': 0.35,
                'volatility': 0.04,
                'breakout_point': 50,
                'boll_period': 20,
                'boll_std': 2
            },
            'BOLL_SQUEEZE': {
                'trend_direction': 'sideways',
                'trend_strength': 0.05,
                'volatility': 0.01,
                'squeeze_period': [30, 50],
                'boll_period': 20
            },
            
            # K线形态配置
            'DOJI': {
                'trend_direction': 'sideways',
                'trend_strength': 0.1,
                'volatility': 0.02,
                'doji_points': [55, 56, 57],  # 连续几个十字星
                'body_ratio': 0.1  # 实体与影线比例
            },
            'HAMMER': {
                'trend_direction': 'up_reversal',
                'trend_strength': 0.2,
                'volatility': 0.03,
                'hammer_point': 55,
                'lower_shadow_ratio': 2.0  # 下影线长度比例
            },
            'SHOOTING_STAR': {
                'trend_direction': 'down_reversal',
                'trend_strength': 0.2,
                'volatility': 0.03,
                'star_point': 55,
                'upper_shadow_ratio': 2.0
            }
        }

    def generate_pattern_data(self, pattern_type: str, data_points: int = 60, 
                            stock_code: str = "TEST001") -> Optional[pd.DataFrame]:
        """
        生成指定形态的测试数据
        
        Args:
            pattern_type: 形态类型
            data_points: 数据点数量
            stock_code: 股票代码
            
        Returns:
            pd.DataFrame: 符合stockInfo格式的测试数据
        """
        try:
            if pattern_type not in self.pattern_configs:
                logger.warning(f"不支持的形态类型: {pattern_type}")
                return None
            
            config = self.pattern_configs[pattern_type]
            
            # 生成基础价格序列
            prices = self._generate_base_price_series(data_points, config)
            
            # 生成OHLC数据
            ohlc_data = self._generate_ohlc_from_prices(prices, config)
            
            # 生成成交量数据
            volumes = self._generate_volume_data(data_points, config)
            
            # 生成日期序列
            dates = self._generate_date_series(data_points)
            
            # 构建DataFrame
            stock_data = pd.DataFrame({
                'date': dates,
                'code': [stock_code] * data_points,
                'name': [f'测试股票_{pattern_type}'] * data_points,
                'open': ohlc_data['open'],
                'high': ohlc_data['high'],
                'low': ohlc_data['low'],
                'close': ohlc_data['close'],
                'volume': volumes,
                'industry': ['测试行业'] * data_points
            })
            
            # 验证数据质量
            if self._validate_stock_data(stock_data):
                logger.debug(f"成功生成 {pattern_type} 形态数据: {len(stock_data)} 行")
                return stock_data
            else:
                logger.error(f"生成的数据质量验证失败: {pattern_type}")
                return None
                
        except Exception as e:
            logger.error(f"生成形态数据失败 {pattern_type}: {e}")
            return None

    def _generate_base_price_series(self, data_points: int, config: Dict[str, Any]) -> np.ndarray:
        """生成基础价格序列"""
        trend_direction = config.get('trend_direction', 'sideways')
        trend_strength = config.get('trend_strength', 0.1)
        volatility = config.get('volatility', 0.02)
        
        # 设置随机种子以确保可重复性
        np.random.seed(42)
        
        prices = [self.base_config['base_price']]
        
        for i in range(1, data_points):
            # 趋势分量
            if trend_direction == 'up':
                trend_component = trend_strength / data_points
            elif trend_direction == 'down':
                trend_component = -trend_strength / data_points
            elif trend_direction == 'up_reversal':
                # 前期下跌，后期上涨
                if i < data_points * 0.8:
                    trend_component = -trend_strength / data_points
                else:
                    trend_component = trend_strength * 2 / data_points
            elif trend_direction == 'down_reversal':
                # 前期上涨，后期下跌
                if i < data_points * 0.8:
                    trend_component = trend_strength / data_points
                else:
                    trend_component = -trend_strength * 2 / data_points
            else:  # sideways
                trend_component = 0
            
            # 随机波动分量
            random_component = np.random.normal(0, volatility)
            
            # 计算新价格
            price_change = trend_component + random_component
            new_price = prices[-1] * (1 + price_change)
            
            # 确保价格在合理范围内
            new_price = max(self.base_config['min_price'], 
                          min(self.base_config['max_price'], new_price))
            
            prices.append(new_price)
        
        return np.array(prices)

    def _generate_ohlc_from_prices(self, close_prices: np.ndarray, 
                                 config: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """从收盘价生成OHLC数据"""
        data_points = len(close_prices)
        volatility = config.get('volatility', 0.02)
        
        open_prices = np.zeros(data_points)
        high_prices = np.zeros(data_points)
        low_prices = np.zeros(data_points)
        
        for i in range(data_points):
            if i == 0:
                open_prices[i] = close_prices[i] * (1 + np.random.uniform(-0.01, 0.01))
            else:
                # 开盘价接近前一日收盘价
                open_prices[i] = close_prices[i-1] * (1 + np.random.uniform(-0.02, 0.02))
            
            # 生成当日高低价
            daily_range = close_prices[i] * volatility * np.random.uniform(0.5, 2.0)
            
            high_prices[i] = max(open_prices[i], close_prices[i]) + daily_range * np.random.uniform(0, 1)
            low_prices[i] = min(open_prices[i], close_prices[i]) - daily_range * np.random.uniform(0, 1)
            
            # 确保价格逻辑正确
            low_prices[i] = max(low_prices[i], self.base_config['min_price'])
            high_prices[i] = min(high_prices[i], self.base_config['max_price'])
            
            # 确保 low <= open,close <= high
            low_prices[i] = min(low_prices[i], min(open_prices[i], close_prices[i]))
            high_prices[i] = max(high_prices[i], max(open_prices[i], close_prices[i]))
        
        return {
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices
        }

    def _generate_volume_data(self, data_points: int, config: Dict[str, Any]) -> np.ndarray:
        """生成成交量数据"""
        base_volume = self.base_config['base_volume']
        
        volumes = []
        for i in range(data_points):
            # 成交量随机波动
            volume_multiplier = np.random.uniform(0.5, 2.0)
            
            # 趋势末期成交量放大
            if i > data_points * 0.8:
                volume_multiplier *= np.random.uniform(1.2, 2.5)
            
            volume = base_volume * volume_multiplier
            volume = max(self.base_config['min_volume'], 
                        min(self.base_config['max_volume'], volume))
            
            volumes.append(int(volume))
        
        return np.array(volumes)

    def _generate_date_series(self, data_points: int) -> List[datetime]:
        """生成日期序列"""
        # 为了兼容买点分析器，我们需要生成一个包含足够历史数据的日期范围
        # 买点分析器会要求买点日期前60天到后10天的数据

        # 设置一个固定的"买点日期"，确保测试的一致性
        buy_date = datetime(2025, 8, 1)  # 2025年8月1日作为买点日期

        # 计算开始日期：买点日期前70天（留一些余量）
        start_date = buy_date - timedelta(days=70)

        dates = []
        current_date = start_date

        while len(dates) < data_points:
            # 跳过周末
            if current_date.weekday() < 5:  # 0-4 是周一到周五
                dates.append(current_date)
            current_date += timedelta(days=1)

        return dates

    def _validate_stock_data(self, data: pd.DataFrame) -> bool:
        """验证股票数据质量"""
        try:
            # 检查必需列
            required_columns = ['date', 'code', 'name', 'open', 'high', 'low', 'close', 'volume', 'industry']
            if not all(col in data.columns for col in required_columns):
                logger.error("缺少必需的列")
                return False
            
            # 检查数据类型
            if not pd.api.types.is_numeric_dtype(data['open']):
                logger.error("开盘价数据类型错误")
                return False
            
            # 检查价格逻辑
            price_logic_check = (
                (data['low'] <= data['open']) & 
                (data['low'] <= data['close']) & 
                (data['high'] >= data['open']) & 
                (data['high'] >= data['close'])
            )
            
            if not price_logic_check.all():
                logger.error("价格逻辑错误")
                return False
            
            # 检查是否有空值
            if data[['open', 'high', 'low', 'close', 'volume']].isnull().any().any():
                logger.error("存在空值")
                return False
            
            # 检查价格是否为正数
            if (data[['open', 'high', 'low', 'close']] <= 0).any().any():
                logger.error("存在非正数价格")
                return False
            
            # 检查成交量是否为正数
            if (data['volume'] <= 0).any():
                logger.error("存在非正数成交量")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"数据验证异常: {e}")
            return False

    def generate_random_data(self, data_points: int = 60, stock_code: str = "RANDOM001") -> pd.DataFrame:
        """生成随机股票数据（用于负面测试）"""
        np.random.seed(None)  # 使用真正的随机种子
        
        # 生成随机价格序列
        base_price = np.random.uniform(5, 50)
        prices = [base_price]
        
        for i in range(1, data_points):
            change = np.random.normal(0, 0.02)  # 2%的随机波动
            new_price = prices[-1] * (1 + change)
            new_price = max(1.0, min(100.0, new_price))
            prices.append(new_price)
        
        close_prices = np.array(prices)
        
        # 生成OHLC
        config = {'volatility': 0.02}
        ohlc_data = self._generate_ohlc_from_prices(close_prices, config)
        
        # 生成成交量
        volumes = self._generate_volume_data(data_points, config)
        
        # 生成日期
        dates = self._generate_date_series(data_points)
        
        return pd.DataFrame({
            'date': dates,
            'code': [stock_code] * data_points,
            'name': [f'随机股票_{stock_code}'] * data_points,
            'open': ohlc_data['open'],
            'high': ohlc_data['high'],
            'low': ohlc_data['low'],
            'close': ohlc_data['close'],
            'volume': volumes,
            'industry': ['随机行业'] * data_points
        })
