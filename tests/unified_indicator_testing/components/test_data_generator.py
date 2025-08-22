#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试数据生成器 - 为其他指标生成专用测试数据

为ZXM、增强、形态、专业等指标类型生成定制化的测试数据
确保数据质量和格式符合指标计算要求

Author: AI Assistant
Date: 2025-08-22
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


class TestDataGenerator:
    """测试数据生成器 - 支持多种指标类型"""
    
    def __init__(self):
        """初始化生成器"""
        self.base_config = {
            'base_price': 10.0,
            'min_price': 1.0,
            'max_price': 100.0,
            'base_volume': 2000000,
            'min_volume': 100000,
            'max_volume': 50000000,
            'default_periods': 100
        }
    
    def generate_standard_test_data(self, periods: int = 100) -> pd.DataFrame:
        """生成标准测试数据"""
        try:
            dates = pd.date_range(end=datetime.now(), periods=periods, freq='D')
            
            # 生成价格数据
            np.random.seed(42)  # 确保可重复性
            returns = np.random.normal(0.001, 0.02, periods)  # 0.1%平均收益，2%波动率
            
            prices = [self.base_config['base_price']]
            for i in range(1, periods):
                price = prices[-1] * (1 + returns[i])
                price = max(self.base_config['min_price'], min(self.base_config['max_price'], price))
                prices.append(price)
            
            # 生成OHLC数据
            data = []
            for i, price in enumerate(prices):
                daily_volatility = abs(np.random.normal(0, 0.01))  # 日内波动率
                
                high = price * (1 + daily_volatility)
                low = price * (1 - daily_volatility)
                open_price = price + np.random.normal(0, price * 0.005)
                close_price = price
                
                # 生成成交量
                volume = self.base_config['base_volume'] * (1 + np.random.normal(0, 0.3))
                volume = max(self.base_config['min_volume'], min(self.base_config['max_volume'], volume))
                
                data.append({
                    'code': 'TEST001',
                    'name': 'TestStock',
                    'date': dates[i],
                    'open': round(open_price, 2),
                    'high': round(high, 2),
                    'low': round(low, 2),
                    'close': round(close_price, 2),
                    'volume': int(volume),
                    'turnover_rate': round(np.random.uniform(0.5, 5.0), 2)
                })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"生成标准测试数据失败: {e}")
            return pd.DataFrame()
    
    def generate_zxm_test_data(self, periods: int = 120) -> pd.DataFrame:
        """为ZXM指标生成测试数据 - 包含多周期特征"""
        try:
            dates = pd.date_range(end=datetime.now(), periods=periods, freq='D')
            
            # ZXM指标需要更复杂的市场行为模拟
            np.random.seed(42)
            
            # 模拟主力吸筹阶段
            phase1_len = periods // 3  # 吸筹阶段
            phase2_len = periods // 3  # 拉升阶段  
            phase3_len = periods - phase1_len - phase2_len  # 震荡阶段
            
            prices = [self.base_config['base_price']]
            volumes = []
            
            for i in range(1, periods):
                if i <= phase1_len:
                    # 吸筹阶段：价格缓慢上升，成交量逐步放大
                    price_change = np.random.normal(0.002, 0.015)  # 小幅上涨
                    volume_multiplier = 1.0 + (i / phase1_len) * 0.5  # 成交量逐步放大
                elif i <= phase1_len + phase2_len:
                    # 拉升阶段：价格快速上升，成交量大幅放大
                    price_change = np.random.normal(0.01, 0.025)  # 较大涨幅
                    volume_multiplier = 1.5 + np.random.uniform(0, 1.0)  # 成交量大幅放大
                else:
                    # 震荡阶段：价格震荡，成交量回落
                    price_change = np.random.normal(0, 0.02)  # 震荡
                    volume_multiplier = 0.8 + np.random.uniform(0, 0.4)  # 成交量回落
                
                price = prices[-1] * (1 + price_change)
                price = max(self.base_config['min_price'], min(self.base_config['max_price'], price))
                prices.append(price)
                
                # 计算成交量
                base_volume = self.base_config['base_volume'] * volume_multiplier
                volume = base_volume * (1 + np.random.normal(0, 0.2))
                volumes.append(max(self.base_config['min_volume'], int(volume)))
            
            # 生成完整的OHLC数据
            data = []
            for i, price in enumerate(prices):
                daily_volatility = abs(np.random.normal(0, 0.015))
                
                high = price * (1 + daily_volatility)
                low = price * (1 - daily_volatility * 0.8)
                open_price = price + np.random.normal(0, price * 0.003)
                close_price = price
                
                volume = volumes[i] if i < len(volumes) else volumes[-1]
                
                data.append({
                    'code': 'ZXM_TEST001',
                    'name': 'ZXMTestStock',
                    'date': dates[i],
                    'open': round(open_price, 2),
                    'high': round(high, 2),
                    'low': round(low, 2),
                    'close': round(close_price, 2),
                    'volume': volume,
                    'turnover_rate': round(volume / 100000000 * 100, 2)  # 简化的换手率计算
                })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"生成ZXM测试数据失败: {e}")
            return self.generate_standard_test_data(periods)
    
    def generate_candlestick_test_data(self, periods: int = 100) -> pd.DataFrame:
        """为蜡烛图形态识别生成测试数据"""
        try:
            dates = pd.date_range(end=datetime.now(), periods=periods, freq='D')
            
            np.random.seed(42)
            data = []
            base_price = self.base_config['base_price']
            
            for i in range(periods):
                # 在特定位置插入典型的蜡烛图形态
                if i == periods // 4:  # DOJI形态
                    open_price = base_price
                    close_price = base_price + np.random.uniform(-0.01, 0.01)
                    high = base_price + np.random.uniform(0.1, 0.3)
                    low = base_price - np.random.uniform(0.1, 0.3)
                elif i == periods // 2:  # HAMMER形态
                    open_price = base_price
                    close_price = base_price + np.random.uniform(0.05, 0.15)
                    high = close_price + np.random.uniform(0.01, 0.05)
                    low = base_price - np.random.uniform(0.2, 0.4)  # 长下影线
                elif i == periods * 3 // 4:  # SHOOTING_STAR形态
                    open_price = base_price
                    close_price = base_price - np.random.uniform(0.05, 0.15)
                    high = base_price + np.random.uniform(0.2, 0.4)  # 长上影线
                    low = close_price - np.random.uniform(0.01, 0.05)
                else:  # 正常K线
                    change = np.random.normal(0, 0.02)
                    open_price = base_price * (1 + change)
                    close_price = open_price * (1 + np.random.normal(0, 0.015))
                    high = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.01)))
                    low = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.01)))
                    base_price = close_price  # 更新基准价格
                
                volume = self.base_config['base_volume'] * (1 + np.random.normal(0, 0.3))
                volume = max(self.base_config['min_volume'], int(volume))
                
                data.append({
                    'code': 'PATTERN_TEST001',
                    'name': 'PatternTestStock',
                    'date': dates[i],
                    'open': round(open_price, 2),
                    'high': round(high, 2),
                    'low': round(low, 2),
                    'close': round(close_price, 2),
                    'volume': volume,
                    'turnover_rate': round(np.random.uniform(0.5, 5.0), 2)
                })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"生成蜡烛图测试数据失败: {e}")
            return self.generate_standard_test_data(periods)
    
    def generate_volatility_test_data(self, periods: int = 100) -> pd.DataFrame:
        """为波动性指标生成测试数据"""
        try:
            dates = pd.date_range(end=datetime.now(), periods=periods, freq='D')
            
            np.random.seed(42)
            data = []
            base_price = self.base_config['base_price']
            
            # 模拟不同的波动性环境
            for i in range(periods):
                if i < periods // 3:
                    # 低波动期
                    volatility = 0.01
                elif i < periods * 2 // 3:
                    # 高波动期
                    volatility = 0.05
                else:
                    # 正常波动期
                    volatility = 0.02
                
                # 生成价格变化
                change = np.random.normal(0, volatility)
                open_price = base_price * (1 + change)
                close_price = open_price * (1 + np.random.normal(0, volatility))
                
                # 根据波动率计算当日高低点
                intraday_vol = volatility * np.random.uniform(1.5, 3.0)
                high = max(open_price, close_price) * (1 + intraday_vol)
                low = min(open_price, close_price) * (1 - intraday_vol)
                
                base_price = close_price
                
                volume = self.base_config['base_volume'] * (1 + np.random.normal(0, 0.3))
                volume = max(self.base_config['min_volume'], int(volume))
                
                data.append({
                    'code': 'VOL_TEST001',
                    'name': 'VolatilityTestStock',
                    'date': dates[i],
                    'open': round(open_price, 2),
                    'high': round(high, 2),
                    'low': round(low, 2),
                    'close': round(close_price, 2),
                    'volume': volume,
                    'turnover_rate': round(np.random.uniform(0.5, 5.0), 2)
                })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"生成波动性测试数据失败: {e}")
            return self.generate_standard_test_data(periods)
    
    def generate_volume_test_data(self, periods: int = 100) -> pd.DataFrame:
        """为成交量指标生成测试数据"""
        try:
            dates = pd.date_range(end=datetime.now(), periods=periods, freq='D')
            
            np.random.seed(42)
            data = []
            base_price = self.base_config['base_price']
            
            for i in range(periods):
                # 价格变化
                price_change = np.random.normal(0.001, 0.02)
                close_price = base_price * (1 + price_change)
                open_price = close_price + np.random.normal(0, close_price * 0.005)
                
                # 根据价格变化调整成交量
                if price_change > 0.03:  # 大涨
                    volume_multiplier = 2.0 + np.random.uniform(0, 1.0)
                elif price_change < -0.03:  # 大跌
                    volume_multiplier = 1.8 + np.random.uniform(0, 0.8)
                elif abs(price_change) > 0.01:  # 中等变化
                    volume_multiplier = 1.2 + np.random.uniform(0, 0.5)
                else:  # 小幅变化
                    volume_multiplier = 0.8 + np.random.uniform(0, 0.4)
                
                volume = self.base_config['base_volume'] * volume_multiplier
                volume = max(self.base_config['min_volume'], int(volume))
                
                daily_volatility = abs(np.random.normal(0, 0.01))
                high = max(open_price, close_price) * (1 + daily_volatility)
                low = min(open_price, close_price) * (1 - daily_volatility)
                
                base_price = close_price
                
                data.append({
                    'code': 'VOLUME_TEST001',
                    'name': 'VolumeTestStock',
                    'date': dates[i],
                    'open': round(open_price, 2),
                    'high': round(high, 2),
                    'low': round(low, 2),
                    'close': round(close_price, 2),
                    'volume': volume,
                    'turnover_rate': round(volume / 100000000 * 100, 2)
                })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"生成成交量测试数据失败: {e}")
            return self.generate_standard_test_data(periods)
    
    def generate_enhanced_test_data(self, indicator_name: str, periods: int = 100) -> pd.DataFrame:
        """为增强指标生成测试数据"""
        try:
            # 根据指标类型生成特定数据
            if 'MACD' in indicator_name:
                return self._generate_trend_data(periods, trend_type='oscillating')
            elif 'RSI' in indicator_name:
                return self._generate_momentum_data(periods)
            elif 'BOLL' in indicator_name:
                return self._generate_volatility_data(periods)
            elif 'KDJ' in indicator_name:
                return self._generate_oscillator_data(periods)
            else:
                return self.generate_standard_test_data(periods)
                
        except Exception as e:
            logger.error(f"生成增强指标测试数据失败: {e}")
            return self.generate_standard_test_data(periods)
    
    def _generate_trend_data(self, periods: int, trend_type: str = 'up') -> pd.DataFrame:
        """生成趋势数据"""
        dates = pd.date_range(end=datetime.now(), periods=periods, freq='D')
        data = []
        base_price = self.base_config['base_price']
        
        trend_strength = 0.002 if trend_type == 'up' else -0.002 if trend_type == 'down' else 0
        
        for i in range(periods):
            # 添加趋势和噪声
            trend_component = trend_strength * i
            noise = np.random.normal(0, 0.015)
            oscillation = 0.01 * np.sin(i * 0.1) if trend_type == 'oscillating' else 0
            
            price_change = trend_component + noise + oscillation
            close_price = base_price * (1 + price_change)
            open_price = close_price + np.random.normal(0, close_price * 0.003)
            
            daily_volatility = abs(np.random.normal(0, 0.01))
            high = max(open_price, close_price) * (1 + daily_volatility)
            low = min(open_price, close_price) * (1 - daily_volatility)
            
            volume = self.base_config['base_volume'] * (1 + np.random.normal(0, 0.3))
            volume = max(self.base_config['min_volume'], int(volume))
            
            data.append({
                'code': 'TREND_TEST001',
                'name': 'TrendTestStock', 
                'date': dates[i],
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close_price, 2),
                'volume': volume,
                'turnover_rate': round(np.random.uniform(0.5, 5.0), 2)
            })
            
            base_price = close_price
        
        return pd.DataFrame(data)
    
    def _generate_momentum_data(self, periods: int) -> pd.DataFrame:
        """生成动量数据"""
        return self._generate_trend_data(periods, 'oscillating')
    
    def _generate_volatility_data(self, periods: int) -> pd.DataFrame:
        """生成波动性数据"""
        return self.generate_volatility_test_data(periods)
    
    def _generate_oscillator_data(self, periods: int) -> pd.DataFrame:
        """生成振荡器数据"""
        return self._generate_trend_data(periods, 'oscillating')
    
    def generate_technical_test_data(self, indicator_name: str, periods: int = 150) -> pd.DataFrame:
        """为技术指标生成专用测试数据"""
        try:
            # 根据指标名称生成相应的测试数据
            if indicator_name in ['SAR', 'TRIX', 'WMA', 'DMA', 'AROON']:
                return self.generate_trend_test_data(periods)
            elif indicator_name in ['CMO', 'ROC', 'MOMENTUM', 'MTM']:
                return self.generate_oscillator_test_data(periods)
            elif indicator_name in ['AD', 'EMV', 'VR', 'VOSC', 'MFI', 'CHAIKIN', 'PVT']:
                return self.generate_volume_test_data(periods)
            elif indicator_name in ['ATR', 'KC', 'VIX', 'STDDEV']:
                return self.generate_volatility_test_data(periods)
            elif indicator_name in ['VORTEX', 'RSIMA', 'COMPOSITE']:
                return self.generate_composite_test_data(periods)
            else:
                return self.generate_standard_test_data(periods)
                
        except Exception as e:
            logger.error(f"生成{indicator_name}技术测试数据失败: {e}")
            return self.generate_standard_test_data(periods)
    
    def generate_trend_test_data(self, periods: int = 120) -> pd.DataFrame:
        """为趋势指标生成测试数据"""
        try:
            dates = pd.date_range(end=datetime.now(), periods=periods, freq='D')
            data = []
            base_price = self.base_config['base_price']
            
            # 模拟明显的趋势变化
            for i in range(periods):
                if i < periods // 3:
                    # 上升趋势
                    trend = 0.005 + np.random.normal(0, 0.01)
                elif i < periods * 2 // 3:
                    # 下降趋势
                    trend = -0.003 + np.random.normal(0, 0.01)
                else:
                    # 震荡趋势
                    trend = np.random.normal(0, 0.015)
                
                close_price = base_price * (1 + trend)
                open_price = close_price + np.random.normal(0, close_price * 0.002)
                
                daily_volatility = abs(np.random.normal(0, 0.008))
                high = max(open_price, close_price) * (1 + daily_volatility)
                low = min(open_price, close_price) * (1 - daily_volatility)
                
                volume = self.base_config['base_volume'] * (1 + np.random.normal(0, 0.25))
                volume = max(self.base_config['min_volume'], int(volume))
                
                data.append({
                    'code': 'TREND_TEST001',
                    'name': 'TrendTestStock',
                    'date': dates[i],
                    'open': round(open_price, 2),
                    'high': round(high, 2),
                    'low': round(low, 2),
                    'close': round(close_price, 2),
                    'volume': volume,
                    'turnover_rate': round(np.random.uniform(0.5, 5.0), 2)
                })
                
                base_price = close_price
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"生成趋势测试数据失败: {e}")
            return self.generate_standard_test_data(periods)
    
    def generate_oscillator_test_data(self, periods: int = 120) -> pd.DataFrame:
        """为振荡器指标生成测试数据"""
        try:
            dates = pd.date_range(end=datetime.now(), periods=periods, freq='D')
            data = []
            base_price = self.base_config['base_price']
            
            # 模拟振荡行情
            for i in range(periods):
                # 创建周期性振荡
                cycle_component = 0.02 * np.sin(i * 0.15)  # 较长周期
                short_cycle = 0.01 * np.sin(i * 0.4)     # 较短周期
                noise = np.random.normal(0, 0.01)
                
                price_change = cycle_component + short_cycle + noise
                close_price = base_price * (1 + price_change)
                open_price = close_price + np.random.normal(0, close_price * 0.003)
                
                daily_volatility = abs(np.random.normal(0, 0.01))
                high = max(open_price, close_price) * (1 + daily_volatility)
                low = min(open_price, close_price) * (1 - daily_volatility)
                
                volume = self.base_config['base_volume'] * (1 + np.random.normal(0, 0.3))
                volume = max(self.base_config['min_volume'], int(volume))
                
                data.append({
                    'code': 'OSC_TEST001',
                    'name': 'OscillatorTestStock',
                    'date': dates[i],
                    'open': round(open_price, 2),
                    'high': round(high, 2),
                    'low': round(low, 2),
                    'close': round(close_price, 2),
                    'volume': volume,
                    'turnover_rate': round(np.random.uniform(0.5, 5.0), 2)
                })
                
                base_price = close_price
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"生成振荡器测试数据失败: {e}")
            return self.generate_standard_test_data(periods)
    
    def generate_composite_test_data(self, periods: int = 150) -> pd.DataFrame:
        """为复合指标生成测试数据"""
        try:
            dates = pd.date_range(end=datetime.now(), periods=periods, freq='D')
            data = []
            base_price = self.base_config['base_price']
            
            # 模拟复杂的市场行为
            for i in range(periods):
                # 多重因子影响
                trend = 0.001 * i / periods  # 长期趋势
                cycle1 = 0.015 * np.sin(i * 0.1)  # 长周期
                cycle2 = 0.008 * np.sin(i * 0.3)  # 中周期
                cycle3 = 0.005 * np.sin(i * 0.7)  # 短周期
                noise = np.random.normal(0, 0.012)
                
                price_change = trend + cycle1 + cycle2 + cycle3 + noise
                close_price = base_price * (1 + price_change)
                open_price = close_price + np.random.normal(0, close_price * 0.004)
                
                # 波动性也受多因子影响
                volatility_factor = 0.01 * (1 + 0.5 * abs(np.sin(i * 0.2)))
                high = max(open_price, close_price) * (1 + volatility_factor)
                low = min(open_price, close_price) * (1 - volatility_factor * 0.8)
                
                # 成交量与价格变化和波动性相关
                volume_factor = 1 + abs(price_change) * 10 + volatility_factor * 5
                volume = self.base_config['base_volume'] * volume_factor
                volume = max(self.base_config['min_volume'], int(volume))
                
                data.append({
                    'code': 'COMP_TEST001',
                    'name': 'CompositeTestStock',
                    'date': dates[i],
                    'open': round(open_price, 2),
                    'high': round(high, 2),
                    'low': round(low, 2),
                    'close': round(close_price, 2),
                    'volume': volume,
                    'turnover_rate': round(np.random.uniform(0.5, 8.0), 2)
                })
                
                base_price = close_price
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"生成复合指标测试数据失败: {e}")
            return self.generate_standard_test_data(periods)
    
    def generate_buypoint_test_data(self, periods: int = 120) -> pd.DataFrame:
        """为买点测试生成数据"""
        try:
            dates = pd.date_range(end=datetime.now(), periods=periods, freq='D')
            data = []
            base_price = self.base_config['base_price']
            
            # 模拟包含明显买点的行情
            for i in range(periods):
                # 在特定位置创建买点机会
                if i in [periods//4, periods//2, periods*3//4]:
                    # 创建超跌反弹买点
                    if i > 10:
                        # 先下跌
                        price_change = -0.08 + np.random.normal(0, 0.01)
                    else:
                        price_change = np.random.normal(0.02, 0.01)  # 反弹
                elif i in [periods//4 + 1, periods//2 + 1, periods*3//4 + 1]:
                    # 反弹确认
                    price_change = np.random.normal(0.03, 0.015)
                else:
                    # 正常波动
                    price_change = np.random.normal(0.001, 0.018)
                
                close_price = base_price * (1 + price_change)
                open_price = close_price + np.random.normal(0, close_price * 0.003)
                
                daily_volatility = abs(np.random.normal(0, 0.012))
                high = max(open_price, close_price) * (1 + daily_volatility)
                low = min(open_price, close_price) * (1 - daily_volatility)
                
                # 买点处成交量放大
                if abs(price_change) > 0.02:
                    volume_multiplier = 2.0 + np.random.uniform(0, 1.0)
                else:
                    volume_multiplier = 1.0 + np.random.uniform(0, 0.5)
                
                volume = self.base_config['base_volume'] * volume_multiplier
                volume = max(self.base_config['min_volume'], int(volume))
                
                data.append({
                    'code': 'BUYPOINT_TEST001',
                    'name': 'BuypointTestStock',
                    'date': dates[i],
                    'open': round(open_price, 2),
                    'high': round(high, 2),
                    'low': round(low, 2),
                    'close': round(close_price, 2),
                    'volume': volume,
                    'turnover_rate': round(np.random.uniform(0.5, 8.0), 2)
                })
                
                base_price = close_price
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"生成买点测试数据失败: {e}")
            return self.generate_standard_test_data(periods)