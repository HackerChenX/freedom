#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
StockInfo兼容数据生成器

生成与stockInfo对象结构完全兼容的模拟数据，支持各种技术指标形态的数据生成
确保生成的数据能被正式选股脚本正确处理
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional, Union
import random

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(root_dir)

# 首先导入logger
from utils.logger import getLogger
logger = getLogger(__name__)

try:
    from tests.buypoint_analysis.enhanced_test_data_generator import EnhancedTestDataGenerator
    from enums.period import Period
except ImportError as e:
    logger.warning(f"导入部分模块失败: {e}")
    # 如果导入失败，创建简单的占位符
    class EnhancedTestDataGenerator:
        def generate_pattern_data(self, pattern_type, data_points, stock_code):
            return None
    
    class Period:
        DAILY = "daily"


class StockInfoCompatibleDataGenerator:
    """
    StockInfo兼容数据生成器
    
    生成与stockInfo对象结构完全兼容的模拟数据
    支持各种技术指标形态的数据生成
    """
    
    def __init__(self):
        """初始化数据生成器"""
        self.base_generator = EnhancedTestDataGenerator()
        
        # stockInfo字段定义（基于数据库表结构）
        self.stockinfo_fields = {
            # 基础字段
            'date': 'Date',           # 交易日期
            'code': 'String',         # 股票代码
            'name': 'String',         # 股票名称
            'level': 'String',        # K线周期
            
            # 价格字段
            'open': 'Float64',        # 开盘价
            'high': 'Float64',        # 最高价
            'low': 'Float64',         # 最低价
            'close': 'Float64',       # 收盘价
            
            # 交易量字段
            'volume': 'Float64',      # 成交量
            'turnover_rate': 'Float64', # 换手率
            
            # 计算字段
            'price_change': 'Float64',  # 价格变动
            'price_range': 'Float64',   # 价格区间
            
            # 分类字段
            'industry': 'String',     # 行业
            
            # 时间戳字段
            'datetime': 'DateTime',   # 日期时间
            'seq': 'UInt32'          # 序号
        }
        
        # 指标历史数据需求映射
        self.indicator_history_requirements = {
            'MACD': 60,      # MACD需要足够数据计算EMA
            'RSI': 30,       # RSI计算需求
            'KDJ': 20,       # KDJ计算需求
            'BOLL': 40,      # 布林带计算需求
            'DMI': 30,       # DMI计算需求
            'CCI': 25,       # CCI计算需求
            'WR': 20,        # WR计算需求
            'BIAS': 30,      # BIAS计算需求
            'EMA': 50,       # EMA计算需求
            'VOL': 10,       # 成交量指标
            'ADX': 30,       # ADX计算需求
            'DMA': 40,       # DMA计算需求
            'WMA': 50,       # WMA计算需求
            'StochRSI': 35,  # StochRSI计算需求
            'MA': 60,        # MA计算需求
            'OBV': 30,       # OBV计算需求
            'MTM': 25,       # MTM计算需求
            'PVT': 30,       # PVT计算需求
            'MOMENTUM': 25,  # MOMENTUM计算需求
            'FIBONACCI': 60, # FIBONACCI计算需求
            'AROON': 30      # AROON计算需求
        }
        
        # 行业分类
        self.industries = [
            '银行', '证券', '保险', '房地产', '建筑材料', '建筑装饰',
            '钢铁', '有色金属', '煤炭', '石油石化', '化工', '基础化工',
            '机械设备', '电气设备', '国防军工', '汽车', '家用电器',
            '食品饮料', '纺织服装', '轻工制造', '医药生物', '农林牧渔',
            '公用事业', '交通运输', '电子', '计算机', '通信', '传媒',
            '综合', '商业贸易', '休闲服务', '测试行业'
        ]
        
        logger.info("StockInfo兼容数据生成器初始化完成")
    
    def generate_stockinfo_compatible_data(self, 
                                         indicator_name: str,
                                         pattern_type: str,
                                         stock_code: str,
                                         history_days: int = None) -> pd.DataFrame:
        """
        生成StockInfo兼容的数据
        
        Args:
            indicator_name: 指标名称
            pattern_type: 形态类型
            stock_code: 股票代码
            history_days: 历史数据天数
            
        Returns:
            pd.DataFrame: 符合stockInfo结构的数据
        """
        try:
            # 确定历史数据需求
            if history_days is None:
                history_days = self._get_indicator_history_requirement(indicator_name)
            
            logger.debug(f"生成StockInfo兼容数据: {stock_code} ({indicator_name}.{pattern_type}, {history_days}天)")
            
            # 尝试使用增强数据生成器
            base_data = self._generate_base_data_with_pattern(
                indicator_name, pattern_type, stock_code, history_days
            )
            
            if base_data is None or base_data.empty:
                # 如果失败，使用基础数据生成
                base_data = self._generate_basic_stockinfo_data(stock_code, history_days)
            
            # 确保数据结构完全兼容
            stockinfo_data = self._ensure_stockinfo_compatibility(base_data, stock_code)
            
            # 验证数据质量
            if not self._validate_stockinfo_structure(stockinfo_data):
                raise ValueError(f"生成的数据不符合stockInfo结构要求")
            
            logger.debug(f"成功生成StockInfo兼容数据: {len(stockinfo_data)} 行")
            return stockinfo_data
            
        except Exception as e:
            logger.error(f"生成StockInfo兼容数据失败: {e}")
            # 返回基础数据作为备选
            return self._generate_basic_stockinfo_data(stock_code, history_days or 60)
    
    def generate_random_stockinfo_data(self, stock_code: str, history_days: int) -> pd.DataFrame:
        """
        生成随机StockInfo数据
        
        Args:
            stock_code: 股票代码
            history_days: 历史数据天数
            
        Returns:
            pd.DataFrame: 随机的stockInfo数据
        """
        try:
            logger.debug(f"生成随机StockInfo数据: {stock_code} ({history_days}天)")
            
            # 生成随机基础数据
            base_data = self._generate_random_base_data(stock_code, history_days)
            
            # 确保数据结构兼容
            stockinfo_data = self._ensure_stockinfo_compatibility(base_data, stock_code)
            
            return stockinfo_data
            
        except Exception as e:
            logger.error(f"生成随机StockInfo数据失败: {e}")
            return self._generate_basic_stockinfo_data(stock_code, history_days)
    
    def _get_indicator_history_requirement(self, indicator_name: str) -> int:
        """获取指标历史数据需求"""
        return self.indicator_history_requirements.get(indicator_name, 60)
    
    def _generate_base_data_with_pattern(self, 
                                       indicator_name: str,
                                       pattern_type: str,
                                       stock_code: str,
                                       history_days: int) -> Optional[pd.DataFrame]:
        """使用增强数据生成器生成带形态的基础数据"""
        try:
            # 如果是目标股票，必须确保包含期望的形态
            if stock_code.startswith('TARGET'):
                return self._generate_target_pattern_data(indicator_name, pattern_type, stock_code, history_days)
            
            # 非目标股票使用原有逻辑
            if self.base_generator:
                pattern_key = f"{indicator_name}_{pattern_type}"
                data = self.base_generator.generate_pattern_data(
                    pattern_type=pattern_key,
                    data_points=history_days,
                    stock_code=stock_code
                )
                return data
        except Exception as e:
            logger.warning(f"增强数据生成器失败: {e}")
        
        return None

    def _generate_target_pattern_data(self, 
                                    indicator_name: str,
                                    pattern_type: str,
                                    stock_code: str,
                                    history_days: int) -> pd.DataFrame:
        """专门为目标股票生成包含特定形态的数据"""
        try:
            logger.debug(f"为目标股票 {stock_code} 生成 {indicator_name}.{pattern_type} 形态数据")
            
            # 生成基础数据
            base_data = self._generate_basic_stockinfo_data(stock_code, history_days)
            
            # 根据指标和形态类型调整数据，确保包含期望形态
            adjusted_data = self._adjust_data_for_pattern(base_data, indicator_name, pattern_type)
            
            return adjusted_data
            
        except Exception as e:
            logger.error(f"生成目标形态数据失败: {e}")
            # 返回基础数据作为备选
            return self._generate_basic_stockinfo_data(stock_code, history_days)

    def _adjust_data_for_pattern(self, 
                               data: pd.DataFrame, 
                               indicator_name: str, 
                               pattern_type: str) -> pd.DataFrame:
        """调整数据以确保包含特定形态"""
        try:
            adjusted_data = data.copy()
            
            if indicator_name == 'MACD':
                adjusted_data = self._create_macd_pattern(adjusted_data, pattern_type)
            elif indicator_name == 'RSI':
                adjusted_data = self._create_rsi_pattern(adjusted_data, pattern_type)
            elif indicator_name == 'VOL':
                adjusted_data = self._create_volume_pattern(adjusted_data, pattern_type)
            elif indicator_name == 'KDJ':
                adjusted_data = self._create_kdj_pattern(adjusted_data, pattern_type)
            elif indicator_name == 'BOLL':
                adjusted_data = self._create_boll_pattern(adjusted_data, pattern_type)
            elif indicator_name == 'CCI':
                adjusted_data = self._create_cci_pattern(adjusted_data, pattern_type)
            elif indicator_name == 'WMA':
                adjusted_data = self._create_wma_pattern(adjusted_data, pattern_type)
            elif indicator_name in ['MA', 'EMA', 'DMA', 'WMA']:
                # 移动平均线类指标统一处理
                adjusted_data = self._create_ma_pattern(adjusted_data, pattern_type)
            elif indicator_name in ['BIAS', 'STOCHRSI', 'WR', 'OBV', 'MTM', 'PVT', 'MOMENTUM', 'DMI', 'ADX']:
                # 其他技术指标统一处理GOLDEN_CROSS/DEATH_CROSS形态
                adjusted_data = self._create_cross_pattern(adjusted_data, pattern_type)
            elif indicator_name == 'FIBONACCI':
                adjusted_data = self._create_fibonacci_pattern(adjusted_data, pattern_type)
            elif indicator_name == 'AROON':
                adjusted_data = self._create_aroon_pattern(adjusted_data, pattern_type)
            else:
                # 对于其他指标，生成通用的趋势形态
                adjusted_data = self._create_generic_pattern(adjusted_data, pattern_type)
            
            return adjusted_data
            
        except Exception as e:
            logger.error(f"调整形态数据失败: {e}")
            return data

    def _create_macd_pattern(self, data: pd.DataFrame, pattern_type: str) -> pd.DataFrame:
        """创建MACD特定形态"""
        adjusted_data = data.copy()
        n = len(adjusted_data)
        
        if pattern_type == 'GOLDEN_CROSS':
            # 创建金叉形态：MACD线从下方向上穿越信号线
            # 在最后10个交易日制造金叉
            golden_cross_point = max(n - 10, n//2)
            
            base_price = adjusted_data['close'].iloc[0]
            
            # 先创建下跌趋势，然后上升形成金叉
            for i in range(n):
                if i < golden_cross_point:
                    # 下跌阶段
                    trend_factor = 1 - (0.15 * i / golden_cross_point)  # 最多下跌15%
                else:
                    # 上升阶段 - 形成金叉
                    recovery_ratio = (i - golden_cross_point) / (n - golden_cross_point)
                    trend_factor = 0.85 + (0.25 * recovery_ratio)  # 反弹25%
                
                new_price = base_price * trend_factor
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.99, 1.01)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.01, 1.05)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.95, 0.99)
                
        elif pattern_type == 'DEATH_CROSS':
            # 创建死叉形态：MACD线从上方向下穿越信号线
            death_cross_point = max(n - 10, n//2)
            
            base_price = adjusted_data['close'].iloc[0]
            
            # 先创建上升趋势，然后下跌形成死叉
            for i in range(n):
                if i < death_cross_point:
                    # 上升阶段
                    trend_factor = 1 + (0.20 * i / death_cross_point)  # 上涨20%
                else:
                    # 下跌阶段 - 形成死叉
                    decline_ratio = (i - death_cross_point) / (n - death_cross_point)
                    trend_factor = 1.20 - (0.30 * decline_ratio)  # 下跌30%
                
                new_price = base_price * trend_factor
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.99, 1.01)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.01, 1.05)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.95, 0.99)
        
        return adjusted_data

    def _create_rsi_pattern(self, data: pd.DataFrame, pattern_type: str) -> pd.DataFrame:
        """创建RSI特定形态"""
        adjusted_data = data.copy()
        n = len(adjusted_data)
        
        if pattern_type == 'OVERSOLD':
            # 创建超卖形态：连续下跌后反弹
            oversold_period = min(14, n//3)  # RSI计算一般用14天
            base_price = adjusted_data['close'].iloc[0]
            
            for i in range(n):
                if i < oversold_period:
                    # 持续下跌形成超卖
                    decline_factor = 1 - (0.25 * i / oversold_period)  # 下跌25%
                else:
                    # 超卖后反弹
                    recovery_ratio = min(1.0, (i - oversold_period) / oversold_period)
                    decline_factor = 0.75 + (0.15 * recovery_ratio)  # 反弹15%
                
                new_price = base_price * decline_factor
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.99, 1.01)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.01, 1.03)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.97, 0.99)
                
        elif pattern_type == 'OVERBOUGHT':
            # 创建超买形态：连续上涨后回调
            overbought_period = min(14, n//3)
            base_price = adjusted_data['close'].iloc[0]
            
            for i in range(n):
                if i < overbought_period:
                    # 持续上涨形成超买
                    rise_factor = 1 + (0.30 * i / overbought_period)  # 上涨30%
                else:
                    # 超买后回调
                    correction_ratio = min(1.0, (i - overbought_period) / overbought_period)
                    rise_factor = 1.30 - (0.15 * correction_ratio)  # 回调15%
                
                new_price = base_price * rise_factor
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.99, 1.01)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.01, 1.05)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.95, 0.99)
        
        return adjusted_data

    def _create_volume_pattern(self, data: pd.DataFrame, pattern_type: str) -> pd.DataFrame:
        """创建成交量特定形态"""
        adjusted_data = data.copy()
        n = len(adjusted_data)
        
        if pattern_type in ['VOLUME_SPIKE', 'VOLUME_SURGE']:
            # 创建成交量突增形态
            spike_point = max(n - 5, n//2)  # 在后半段出现突增
            base_volume = adjusted_data['volume'].mean()
            
            for i in range(n):
                if i >= spike_point:
                    # 成交量突增2-5倍
                    volume_multiplier = random.uniform(2.5, 5.0)
                    adjusted_data.loc[i, 'volume'] = base_volume * volume_multiplier
                    
                    # 成交量放大通常伴随价格突破
                    price_boost = random.uniform(1.02, 1.08)  # 2-8%的价格上涨
                    adjusted_data.loc[i, 'close'] *= price_boost
                    adjusted_data.loc[i, 'high'] *= price_boost
                    adjusted_data.loc[i, 'open'] *= price_boost
                    adjusted_data.loc[i, 'low'] *= price_boost
        
        return adjusted_data

    def _create_kdj_pattern(self, data: pd.DataFrame, pattern_type: str) -> pd.DataFrame:
        """创建KDJ特定形态"""
        # KDJ的形态与RSI类似，创建超买超卖或金叉死叉
        if pattern_type in ['OVERBOUGHT', 'OVERSOLD']:
            return self._create_rsi_pattern(data, pattern_type)
        else:
            return self._create_macd_pattern(data, pattern_type)

    def _create_boll_pattern(self, data: pd.DataFrame, pattern_type: str) -> pd.DataFrame:
        """创建布林带特定形态"""
        adjusted_data = data.copy()
        n = len(adjusted_data)
        
        if pattern_type == 'UPPER_BREAKOUT':
            # 创建上轨突破形态
            breakout_point = max(n - 8, n//2)
            base_price = adjusted_data['close'].iloc[0]
            
            for i in range(n):
                if i < breakout_point:
                    # 在布林带中轨附近波动
                    oscillation = random.uniform(-0.05, 0.05)
                    price_factor = 1 + oscillation
                else:
                    # 突破上轨
                    breakout_strength = (i - breakout_point) / (n - breakout_point)
                    price_factor = 1 + (0.15 * breakout_strength)  # 突破15%
                
                new_price = base_price * price_factor
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.99, 1.01)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.01, 1.05)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.95, 0.99)
        
        return adjusted_data

    def _create_cci_pattern(self, data: pd.DataFrame, pattern_type: str) -> pd.DataFrame:
        """创建CCI特定形态"""
        adjusted_data = data.copy()
        n = len(adjusted_data)
        
        if pattern_type == 'OVERSOLD':
            # 创建CCI超卖形态：价格持续在典型价格之下
            oversold_period = min(20, n//2)  # CCI计算一般用20天
            base_price = adjusted_data['close'].iloc[0]
            
            for i in range(n):
                if i < oversold_period:
                    # 持续下跌形成超卖，创造不规则的价格波动
                    decline_factor = 1 - (0.30 * i / oversold_period)  # 下跌30%
                    volatility = random.uniform(0.95, 1.05)  # 增加波动性
                    new_price = base_price * decline_factor * volatility
                else:
                    # 超卖后缓慢反弹
                    recovery_ratio = min(1.0, (i - oversold_period) / (n - oversold_period))
                    decline_factor = 0.70 + (0.20 * recovery_ratio)  # 反弹20%
                    new_price = base_price * decline_factor
                
                # 为了形成CCI超卖，需要创造正确的价格波动模式
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.99, 1.01)
                
                # CCI超卖需要收盘价持续低于典型价格(high+low+close)/3
                # 确保high和low的设置能让典型价格高于收盘价
                high_multiplier = random.uniform(1.03, 1.08)  # 高价比收盘价高3-8%
                low_multiplier = random.uniform(0.96, 1.00)   # 低价接近收盘价
                
                adjusted_data.loc[i, 'high'] = new_price * high_multiplier
                adjusted_data.loc[i, 'low'] = new_price * low_multiplier
                
        elif pattern_type == 'OVERBOUGHT':
            # 创建CCI超买形态：价格持续在典型价格之上
            overbought_period = min(20, n//2)
            base_price = adjusted_data['close'].iloc[0]
            
            for i in range(n):
                if i < overbought_period:
                    # 持续上涨形成超买
                    rise_factor = 1 + (0.35 * i / overbought_period)  # 上涨35%
                    volatility = random.uniform(0.95, 1.05)
                    new_price = base_price * rise_factor * volatility
                else:
                    # 超买后回调
                    correction_ratio = min(1.0, (i - overbought_period) / (n - overbought_period))
                    rise_factor = 1.35 - (0.20 * correction_ratio)  # 回调20%
                    new_price = base_price * rise_factor
                
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.99, 1.01)
                
                # CCI超买需要收盘价持续高于典型价格(high+low+close)/3
                # 确保high和low的设置能让典型价格低于收盘价
                high_multiplier = random.uniform(1.00, 1.04)  # 高价接近收盘价
                low_multiplier = random.uniform(0.92, 0.97)   # 低价比收盘价低3-8%
                
                adjusted_data.loc[i, 'high'] = new_price * high_multiplier
                adjusted_data.loc[i, 'low'] = new_price * low_multiplier
        
        return adjusted_data

    def _create_wma_pattern(self, data: pd.DataFrame, pattern_type: str) -> pd.DataFrame:
        """创建WMA（加权移动平均）特定形态"""
        adjusted_data = data.copy()
        n = len(adjusted_data)
        
        if pattern_type == 'GOLDEN_CROSS':
            # 创建WMA金叉形态：短期WMA上穿长期WMA
            cross_point = max(n - 12, n//2)
            base_price = adjusted_data['close'].iloc[0]
            
            for i in range(n):
                if i < cross_point:
                    # 下跌阶段，为金叉做准备
                    decline_factor = 1 - (0.18 * i / cross_point)  # 下跌18%
                    trend_noise = random.uniform(0.98, 1.02)  # 加噪音
                    new_price = base_price * decline_factor * trend_noise
                else:
                    # 上升阶段，形成WMA金叉
                    recovery_ratio = (i - cross_point) / (n - cross_point)
                    # 使用加速上升来形成明显的WMA金叉
                    acceleration_factor = 1 + (recovery_ratio ** 1.5) * 0.3  # 加速上涨30%
                    decline_factor = 0.82 * acceleration_factor
                    new_price = base_price * decline_factor
                
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.99, 1.01)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.01, 1.04)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.96, 0.99)
                
        elif pattern_type == 'DEATH_CROSS':
            # 创建WMA死叉形态：短期WMA下穿长期WMA
            cross_point = max(n - 12, n//2)
            base_price = adjusted_data['close'].iloc[0]
            
            for i in range(n):
                if i < cross_point:
                    # 上涨阶段，为死叉做准备
                    rise_factor = 1 + (0.25 * i / cross_point)  # 上涨25%
                    new_price = base_price * rise_factor
                else:
                    # 下跌阶段，形成WMA死叉
                    decline_ratio = (i - cross_point) / (n - cross_point)
                    # 加速下跌形成明显的WMA死叉
                    acceleration_factor = 1 + (decline_ratio ** 1.5) * 0.4
                    rise_factor = 1.25 - (0.35 * acceleration_factor)  # 下跌35%
                    new_price = base_price * rise_factor
                
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.99, 1.01)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.01, 1.03)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.97, 0.99)
                
        elif pattern_type == 'BULLISH_ARRANGEMENT':
            # 创建WMA多头排列：短、中、长期WMA呈多头排列
            base_price = adjusted_data['close'].iloc[0]
            
            for i in range(n):
                # 持续上涨趋势，形成多头排列
                trend_factor = 1 + (0.25 * i / n)  # 逐步上涨25%
                momentum = 1 + (0.05 * (i / n) ** 2)  # 增强趋势动量
                new_price = base_price * trend_factor * momentum
                
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.99, 1.01)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.01, 1.05)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.95, 0.99)
        
        return adjusted_data

    def _create_generic_pattern(self, data: pd.DataFrame, pattern_type: str) -> pd.DataFrame:
        """创建通用形态"""
        adjusted_data = data.copy()
        n = len(adjusted_data)
        base_price = adjusted_data['close'].iloc[0]
        
        # 根据形态名称判断趋势方向
        if any(keyword in pattern_type.upper() for keyword in ['GOLDEN', 'BUY', 'BULL', 'UP', 'SUPPORT']):
            # 上涨形态
            for i in range(n):
                trend_factor = 1 + (0.20 * i / n)  # 逐步上涨20%
                new_price = base_price * trend_factor
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.99, 1.01)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.01, 1.05)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.95, 0.99)
                
        elif any(keyword in pattern_type.upper() for keyword in ['DEATH', 'SELL', 'BEAR', 'DOWN', 'RESISTANCE']):
            # 下跌形态
            for i in range(n):
                trend_factor = 1 - (0.20 * i / n)  # 逐步下跌20%
                new_price = base_price * trend_factor
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.99, 1.01)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.01, 1.03)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.97, 0.99)
        
        return adjusted_data
    
    def _generate_basic_stockinfo_data(self, stock_code: str, history_days: int) -> pd.DataFrame:
        """生成基础stockInfo数据"""
        # 生成日期序列（确保获得足够的工作日）
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=history_days * 2)  # 生成更多天数以确保有足够工作日
        dates = pd.date_range(start=start_date, end=end_date, freq='D')

        # 过滤工作日并确保数量正确
        work_dates = [d for d in dates if d.weekday() < 5]
        if len(work_dates) >= history_days:
            dates = work_dates[-history_days:]  # 取最近的history_days个工作日
        else:
            # 如果工作日不够，补充一些日期
            dates = work_dates
            while len(dates) < history_days:
                dates.append(dates[-1] + timedelta(days=1))
        
        # 生成基础价格数据
        base_price = random.uniform(5.0, 50.0)
        prices = self._generate_realistic_price_series(base_price, len(dates))
        
        # 生成成交量数据
        volumes = self._generate_realistic_volume_series(len(dates))
        
        # 构建基础数据
        data = pd.DataFrame({
            'date': [d.strftime('%Y%m%d') for d in dates],
            'code': [stock_code] * len(dates),
            'name': [f'测试股票_{stock_code}'] * len(dates),
            'open': [p * (1 + random.uniform(-0.02, 0.02)) for p in prices],
            'high': [p * (1 + random.uniform(0.01, 0.05)) for p in prices],
            'low': [p * (1 + random.uniform(-0.05, -0.01)) for p in prices],
            'close': prices,
            'volume': volumes,
            'industry': [random.choice(self.industries)] * len(dates)
        })
        
        # 确保价格逻辑正确
        for i in range(len(data)):
            low = min(data.loc[i, 'open'], data.loc[i, 'close']) * 0.95
            high = max(data.loc[i, 'open'], data.loc[i, 'close']) * 1.05
            data.loc[i, 'low'] = min(data.loc[i, 'low'], low)
            data.loc[i, 'high'] = max(data.loc[i, 'high'], high)
        
        return data

    def _generate_random_base_data(self, stock_code: str, history_days: int) -> pd.DataFrame:
        """生成随机基础数据"""
        # 生成日期序列（确保获得足够的工作日）
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=history_days * 2)  # 生成更多天数
        dates = pd.date_range(start=start_date, end=end_date, freq='D')

        # 过滤工作日并确保数量正确
        work_dates = [d for d in dates if d.weekday() < 5]
        if len(work_dates) >= history_days:
            work_dates = work_dates[-history_days:]  # 取最近的history_days个工作日
        else:
            # 如果工作日不够，补充一些日期
            while len(work_dates) < history_days:
                work_dates.append(work_dates[-1] + timedelta(days=1))

        # 生成随机价格数据（更加随机）
        base_price = random.uniform(3.0, 100.0)
        prices = []
        current_price = base_price

        for _ in range(len(work_dates)):
            # 随机价格变动
            change_rate = random.uniform(-0.1, 0.1)
            current_price = max(1.0, current_price * (1 + change_rate))
            prices.append(current_price)

        # 生成随机成交量
        base_volume = random.randint(500000, 20000000)
        volumes = [base_volume * random.uniform(0.3, 3.0) for _ in range(len(work_dates))]

        # 构建随机数据
        data = pd.DataFrame({
            'date': [d.strftime('%Y%m%d') for d in work_dates],
            'code': [stock_code] * len(work_dates),
            'name': [f'随机股票_{stock_code}'] * len(work_dates),
            'open': [p * random.uniform(0.98, 1.02) for p in prices],
            'high': [p * random.uniform(1.01, 1.08) for p in prices],
            'low': [p * random.uniform(0.92, 0.99) for p in prices],
            'close': prices,
            'volume': volumes,
            'industry': [random.choice(self.industries)] * len(work_dates)
        })

        # 修正价格逻辑
        for i in range(len(data)):
            open_price = data.loc[i, 'open']
            close_price = data.loc[i, 'close']

            # 确保最高价不低于开盘价和收盘价
            data.loc[i, 'high'] = max(data.loc[i, 'high'], open_price, close_price)

            # 确保最低价不高于开盘价和收盘价
            data.loc[i, 'low'] = min(data.loc[i, 'low'], open_price, close_price)

        return data

    def _generate_realistic_price_series(self, base_price: float, length: int) -> List[float]:
        """生成真实的价格序列"""
        prices = [base_price]

        for i in range(1, length):
            # 使用随机游走模型
            change_rate = np.random.normal(0, 0.02)  # 2%的标准差
            new_price = prices[-1] * (1 + change_rate)

            # 确保价格在合理范围内
            new_price = max(1.0, min(1000.0, new_price))
            prices.append(new_price)

        return prices

    def _generate_realistic_volume_series(self, length: int) -> List[float]:
        """生成真实的成交量序列"""
        base_volume = random.randint(1000000, 10000000)
        volumes = []

        for i in range(length):
            # 成交量有一定的随机性和趋势性
            volume_multiplier = np.random.lognormal(0, 0.5)  # 对数正态分布
            volume = base_volume * volume_multiplier

            # 确保成交量在合理范围内
            volume = max(10000, min(100000000, volume))
            volumes.append(volume)

        return volumes

    def _ensure_stockinfo_compatibility(self, data: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """确保数据与stockInfo对象结构完全兼容"""
        try:
            # 创建完整的stockInfo兼容数据
            compatible_data = pd.DataFrame()

            # 基础字段
            compatible_data['date'] = data.get('date', pd.Series([datetime.now().strftime('%Y%m%d')] * len(data)))
            compatible_data['code'] = data.get('code', pd.Series([stock_code] * len(data)))
            compatible_data['name'] = data.get('name', pd.Series([f'股票_{stock_code}'] * len(data)))
            compatible_data['level'] = pd.Series([Period.DAILY] * len(data))

            # 价格字段
            compatible_data['open'] = pd.to_numeric(data.get('open', 10.0), errors='coerce').fillna(10.0)
            compatible_data['high'] = pd.to_numeric(data.get('high', 10.5), errors='coerce').fillna(10.5)
            compatible_data['low'] = pd.to_numeric(data.get('low', 9.5), errors='coerce').fillna(9.5)
            compatible_data['close'] = pd.to_numeric(data.get('close', 10.0), errors='coerce').fillna(10.0)

            # 交易量字段
            compatible_data['volume'] = pd.to_numeric(data.get('volume', 1000000), errors='coerce').fillna(1000000).astype('float64')

            # 计算衍生字段
            compatible_data['turnover_rate'] = self._calculate_turnover_rate(compatible_data['volume'])
            compatible_data['price_change'] = self._calculate_price_change(compatible_data['close'])
            compatible_data['price_range'] = self._calculate_price_range(
                compatible_data['high'], compatible_data['low']
            )

            # 分类字段
            compatible_data['industry'] = data.get('industry', pd.Series(['测试行业'] * len(data)))

            # 时间戳字段
            compatible_data['datetime'] = pd.to_datetime(
                compatible_data['date'], format='%Y%m%d'
            ).dt.strftime('%Y-%m-%d %H:%M:%S')
            compatible_data['seq'] = pd.Series(range(len(data)), dtype='int32')

            # 确保数据类型正确
            compatible_data = self._ensure_correct_dtypes(compatible_data)

            return compatible_data

        except Exception as e:
            logger.error(f"确保stockInfo兼容性失败: {e}")
            raise

    def _calculate_turnover_rate(self, volumes: pd.Series) -> pd.Series:
        """计算换手率"""
        # 简单模拟换手率计算
        base_turnover = 0.05  # 5%基础换手率
        return volumes.apply(lambda v: base_turnover * random.uniform(0.1, 3.0))

    def _calculate_price_change(self, closes: pd.Series) -> pd.Series:
        """计算价格变动"""
        price_changes = [0.0]  # 第一天变动为0

        for i in range(1, len(closes)):
            if closes.iloc[i-1] != 0:
                change = (closes.iloc[i] - closes.iloc[i-1]) / closes.iloc[i-1]
            else:
                change = 0.0
            price_changes.append(change)

        return pd.Series(price_changes)

    def _calculate_price_range(self, highs: pd.Series, lows: pd.Series) -> pd.Series:
        """计算价格区间"""
        return (highs - lows) / lows

    def _ensure_correct_dtypes(self, data: pd.DataFrame) -> pd.DataFrame:
        """确保数据类型正确"""
        # 字符串字段
        string_fields = ['date', 'code', 'name', 'level', 'industry', 'datetime']
        for field in string_fields:
            if field in data.columns:
                data[field] = data[field].astype(str)

        # 浮点数字段
        float_fields = ['open', 'high', 'low', 'close', 'volume', 'turnover_rate', 'price_change', 'price_range']
        for field in float_fields:
            if field in data.columns:
                data[field] = pd.to_numeric(data[field], errors='coerce').fillna(0.0)

        # 整数字段
        int_fields = ['seq']
        for field in int_fields:
            if field in data.columns:
                data[field] = pd.to_numeric(data[field], errors='coerce').fillna(0).astype('int32')

        return data

    def _validate_stockinfo_structure(self, data: pd.DataFrame) -> bool:
        """验证stockInfo结构"""
        try:
            # 检查必需字段
            required_fields = ['date', 'code', 'name', 'open', 'high', 'low', 'close', 'volume']

            for field in required_fields:
                if field not in data.columns:
                    logger.error(f"缺少必需字段: {field}")
                    return False

            # 检查数据完整性
            if len(data) == 0:
                logger.error("数据为空")
                return False

            # 检查价格逻辑
            for i in range(len(data)):
                high = data.loc[i, 'high']
                low = data.loc[i, 'low']
                open_price = data.loc[i, 'open']
                close = data.loc[i, 'close']

                if not (low <= open_price <= high and low <= close <= high):
                    logger.warning(f"第{i}行价格逻辑错误: high={high}, low={low}, open={open_price}, close={close}")
                    # 修正价格逻辑
                    data.loc[i, 'high'] = max(high, open_price, close)
                    data.loc[i, 'low'] = min(low, open_price, close)

            # 检查成交量
            if (data['volume'] <= 0).any():
                logger.warning("存在非正成交量，将修正为正值")
                data['volume'] = data['volume'].abs() + 1

            logger.debug("stockInfo结构验证通过")
            return True

        except Exception as e:
            logger.error(f"stockInfo结构验证失败: {e}")
            return False

    def cleanup(self):
        """清理资源"""
        try:
            if hasattr(self.base_generator, 'cleanup'):
                self.base_generator.cleanup()
            logger.debug("StockInfo兼容数据生成器资源清理完成")
        except Exception as e:
            logger.warning(f"清理资源时出现警告: {e}")

    def get_supported_indicators(self) -> List[str]:
        """获取支持的指标列表"""
        return list(self.indicator_history_requirements.keys())

    def get_indicator_history_requirement(self, indicator_name: str) -> int:
        """获取指标历史数据需求（公共接口）"""
        return self._get_indicator_history_requirement(indicator_name)

    def generate_large_scale_data(self,
                                stock_codes: List[str],
                                indicator_name: str,
                                pattern_type: str,
                                history_days: int = None) -> Dict[str, pd.DataFrame]:
        """
        生成大规模数据（支持4000+股票）

        Args:
            stock_codes: 股票代码列表
            indicator_name: 指标名称
            pattern_type: 形态类型
            history_days: 历史数据天数

        Returns:
            Dict[str, pd.DataFrame]: 股票代码到数据的映射
        """
        try:
            logger.info(f"开始生成大规模数据: {len(stock_codes)} 只股票")

            results = {}

            for i, stock_code in enumerate(stock_codes):
                if i % 100 == 0:
                    logger.info(f"进度: {i}/{len(stock_codes)} ({i/len(stock_codes)*100:.1f}%)")

                try:
                    data = self.generate_stockinfo_compatible_data(
                        indicator_name, pattern_type, stock_code, history_days
                    )
                    results[stock_code] = data

                except Exception as e:
                    logger.warning(f"生成股票 {stock_code} 数据失败: {e}")
                    # 继续处理其他股票
                    continue

            logger.info(f"大规模数据生成完成: {len(results)}/{len(stock_codes)} 成功")
            return results

        except Exception as e:
            logger.error(f"大规模数据生成失败: {e}")
            return {}

    def get_stockinfo_fields(self) -> Dict[str, str]:
        """获取stockInfo字段定义"""
        return self.stockinfo_fields.copy()

    def validate_data_compatibility(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        验证数据兼容性并返回详细报告

        Args:
            data: 待验证的数据

        Returns:
            Dict: 验证报告
        """
        report = {
            'is_compatible': False,
            'missing_fields': [],
            'invalid_data_types': [],
            'data_quality_issues': [],
            'recommendations': []
        }

        try:
            # 检查必需字段
            required_fields = ['date', 'code', 'name', 'open', 'high', 'low', 'close', 'volume']
            missing_fields = [field for field in required_fields if field not in data.columns]
            report['missing_fields'] = missing_fields

            if missing_fields:
                report['recommendations'].append(f"添加缺失字段: {', '.join(missing_fields)}")

            # 检查数据类型
            expected_types = {
                'date': 'object',
                'code': 'object',
                'name': 'object',
                'open': 'float64',
                'high': 'float64',
                'low': 'float64',
                'close': 'float64',
                'volume': 'float64'
            }

            for field, expected_type in expected_types.items():
                if field in data.columns:
                    actual_type = str(data[field].dtype)
                    # 更宽松的类型检查
                    if expected_type == 'float64':
                        if not ('float' in actual_type or 'int' in actual_type):
                            report['invalid_data_types'].append({
                                'field': field,
                                'expected': expected_type,
                                'actual': actual_type
                            })
                    elif expected_type == 'object':
                        if actual_type not in ['object', 'string']:
                            report['invalid_data_types'].append({
                                'field': field,
                                'expected': expected_type,
                                'actual': actual_type
                            })

            # 检查数据质量
            if len(data) == 0:
                report['data_quality_issues'].append("数据为空")

            # 检查价格逻辑
            price_issues = 0
            for i in range(min(len(data), 100)):  # 只检查前100行
                try:
                    high = float(data.iloc[i]['high'])
                    low = float(data.iloc[i]['low'])
                    open_price = float(data.iloc[i]['open'])
                    close = float(data.iloc[i]['close'])

                    if not (low <= open_price <= high and low <= close <= high):
                        price_issues += 1
                except:
                    price_issues += 1

            if price_issues > 0:
                report['data_quality_issues'].append(f"发现 {price_issues} 行价格逻辑错误")

            # 综合评估
            report['is_compatible'] = (
                len(missing_fields) == 0 and
                len(report['invalid_data_types']) == 0 and
                len(report['data_quality_issues']) == 0
            )

            if report['is_compatible']:
                report['recommendations'].append("数据完全兼容stockInfo结构")

            return report

        except Exception as e:
            report['data_quality_issues'].append(f"验证过程出错: {e}")
            return report

    def _create_cross_pattern(self, data: pd.DataFrame, pattern_type: str) -> pd.DataFrame:
        """创建通用的金叉/死叉形态，适用于大多数技术指标"""
        adjusted_data = data.copy()
        n = len(adjusted_data)
        
        if pattern_type == 'GOLDEN_CROSS':
            # 创建上升趋势，利于形成金叉
            cross_point = max(n - 8, n//2)
            base_price = adjusted_data['close'].iloc[0]
            
            for i in range(n):
                if i < cross_point:
                    # 前期下降
                    decline_factor = 1 - (0.1 * i / cross_point)
                else:
                    # 后期上升形成金叉
                    recovery_ratio = (i - cross_point) / (n - cross_point)
                    decline_factor = 0.9 + (0.2 * recovery_ratio)
                
                new_price = base_price * decline_factor * (1 + random.uniform(-0.02, 0.02))
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.99, 1.01)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.00, 1.03)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.97, 1.00)
                
        elif pattern_type == 'DEATH_CROSS':
            # 创建下降趋势，形成死叉
            cross_point = max(n - 8, n//2)
            base_price = adjusted_data['close'].iloc[0]
            
            for i in range(n):
                if i < cross_point:
                    # 前期上升
                    rise_factor = 1 + (0.1 * i / cross_point)
                else:
                    # 后期下降形成死叉
                    decline_ratio = (i - cross_point) / (n - cross_point)
                    rise_factor = 1.1 - (0.2 * decline_ratio)
                
                new_price = base_price * rise_factor * (1 + random.uniform(-0.02, 0.02))
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.99, 1.01)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.00, 1.03)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.97, 1.00)
        
        return adjusted_data

    def _create_ma_pattern(self, data: pd.DataFrame, pattern_type: str) -> pd.DataFrame:
        """创建移动平均线形态（MA、EMA、WMA、DMA）"""
        adjusted_data = data.copy()
        n = len(adjusted_data)
        
        if pattern_type == 'GOLDEN_CROSS':
            # 短期均线从下方向上穿越长期均线
            base_price = adjusted_data['close'].iloc[0]
            cross_point = max(n - 10, n//2)
            
            for i in range(n):
                if i < cross_point:
                    # 创建下跌然后横盘的走势
                    trend_factor = 1 - (0.08 * min(i, cross_point//2) / (cross_point//2))
                else:
                    # 强势上涨，形成金叉
                    recovery_ratio = (i - cross_point) / (n - cross_point)
                    trend_factor = 0.92 + (0.15 * recovery_ratio)  # 反弹15%
                
                new_price = base_price * trend_factor
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.995, 1.005)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.005, 1.025)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.975, 0.995)
                
        elif pattern_type == 'DEATH_CROSS':
            # 短期均线从上方向下穿越长期均线
            base_price = adjusted_data['close'].iloc[0]
            cross_point = max(n - 10, n//2)
            
            for i in range(n):
                if i < cross_point:
                    # 上升走势
                    trend_factor = 1 + (0.08 * i / cross_point)
                else:
                    # 下跌，形成死叉
                    decline_ratio = (i - cross_point) / (n - cross_point)
                    trend_factor = 1.08 - (0.15 * decline_ratio)
                
                new_price = base_price * trend_factor
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.995, 1.005)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.005, 1.025)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.975, 0.995)
        
        return adjusted_data

    def _create_fibonacci_pattern(self, data: pd.DataFrame, pattern_type: str) -> pd.DataFrame:
        """创建斐波那契回调形态"""
        adjusted_data = data.copy()
        n = len(adjusted_data)
        base_price = adjusted_data['close'].iloc[0]
        
        if pattern_type == 'RETRACEMENT_SUPPORT':
            # 创建上升后回调到支撑位的形态
            peak_point = n // 3
            support_point = peak_point + (n - peak_point) // 2
            
            for i in range(n):
                if i < peak_point:
                    # 上升到峰值
                    trend_factor = 1 + (0.2 * i / peak_point)
                elif i < support_point:
                    # 回调到38.2%位置
                    retracement_ratio = (i - peak_point) / (support_point - peak_point)
                    trend_factor = 1.2 - (0.2 * 0.382 * retracement_ratio)
                else:
                    # 从支撑位反弹
                    recovery_ratio = (i - support_point) / (n - support_point)
                    trend_factor = 1.2 * (1 - 0.382) + (0.1 * recovery_ratio)
                
                new_price = base_price * trend_factor
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.99, 1.01)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.01, 1.03)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.97, 0.99)
                
        elif pattern_type == 'RETRACEMENT_RESISTANCE':
            # 创建下跌后反弹到阻力位的形态  
            bottom_point = n // 3
            resistance_point = bottom_point + (n - bottom_point) // 2
            
            for i in range(n):
                if i < bottom_point:
                    # 下跌到谷底
                    trend_factor = 1 - (0.2 * i / bottom_point)
                elif i < resistance_point:
                    # 反弹到61.8%位置
                    recovery_ratio = (i - bottom_point) / (resistance_point - bottom_point)
                    trend_factor = 0.8 + (0.2 * 0.618 * recovery_ratio)
                else:
                    # 在阻力位受阻回落
                    decline_ratio = (i - resistance_point) / (n - resistance_point)
                    trend_factor = 0.8 * (1 + 0.618) - (0.05 * decline_ratio)
                
                new_price = base_price * trend_factor
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.99, 1.01)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.01, 1.03)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.97, 0.99)
        
        return adjusted_data

    def _create_aroon_pattern(self, data: pd.DataFrame, pattern_type: str) -> pd.DataFrame:
        """创建Aroon指标形态"""
        adjusted_data = data.copy()
        n = len(adjusted_data)
        base_price = adjusted_data['close'].iloc[0]
        
        if pattern_type == 'AROON_UP':
            # 创建强势上升趋势，Aroon Up接近100
            for i in range(n):
                # 在最后部分创建新高
                if i > n - 20:
                    trend_factor = 1 + (0.15 * (i - (n - 20)) / 20)
                else:
                    trend_factor = 1 + (0.05 * i / (n - 20))
                
                new_price = base_price * trend_factor
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.99, 1.01)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.01, 1.03)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.97, 0.99)
                
        elif pattern_type == 'AROON_DOWN':
            # 创建弱势下跌趋势，Aroon Down接近100
            for i in range(n):
                # 在最后部分创建新低
                if i > n - 20:
                    trend_factor = 1 - (0.15 * (i - (n - 20)) / 20)
                else:
                    trend_factor = 1 - (0.05 * i / (n - 20))
                
                new_price = base_price * trend_factor
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.99, 1.01)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.01, 1.03)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.97, 0.99)
        
        return adjusted_data
