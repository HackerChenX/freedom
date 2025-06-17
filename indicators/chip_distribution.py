#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
筹码分布指标

基于换手率的筹码分布模型，分析成本分布和持仓情况
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple, Any
import warnings

from indicators.base_indicator import BaseIndicator
from utils.logger import get_logger

# 静默警告
warnings.filterwarnings("ignore", category=RuntimeWarning)

logger = get_logger(__name__)


class ChipDistribution(BaseIndicator):
    """
    筹码分布指标
    
    分析股票各价位买入持仓情况，计算筹码集中度、套牢盘比例等
    """
    
    def __init__(self, periods: list = [5, 10, 20, 60, 120]):
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        """初始化筹码分布指标"""
        super().__init__(name="ChipDistribution", description="筹码分布指标，分析各价位持仓情况")
        self.periods = periods
        self._parameters = {'half_life': 60, 'price_precision': 0.01, 'use_precision_cost': True}
    
    def set_parameters(self, periods: list = None):
        """
        设置指标参数
        
        Args:
            periods: 新的周期列表
        """
        if periods:
            self.periods = periods
        for key, value in self._parameters.items():
            if key in self._parameters:
                self._parameters[key] = value
    
    def calculate_confidence(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """
        计算ChipDistribution指标的置信度

        Args:
            score: 得分序列
            patterns: 检测到的形态DataFrame
            signals: 生成的信号字典

        Returns:
            float: 置信度分数 (0-1)
        """
        if score.empty:
            return 0.5

        # 基础置信度
        confidence = 0.5

        # 1. 基于评分的置信度
        last_score = score.iloc[-1]

        # 极端评分置信度较高
        if last_score > 80 or last_score < 20:
            confidence += 0.25
        # 中性评分置信度中等
        elif 40 <= last_score <= 60:
            confidence += 0.1
        else:
            confidence += 0.15

        # 2. 基于数据质量的置信度
        if hasattr(self, '_result') and self._result is not None:
            # 检查是否有换手率数据
            if 'turnover_rate' in self._result.columns:
                confidence += 0.1

            # 检查筹码集中度数据质量
            if 'chip_concentration' in self._result.columns:
                concentration_values = self._result['chip_concentration'].dropna()
                if len(concentration_values) > 0:
                    # 集中度数据越完整，置信度越高
                    data_completeness = len(concentration_values) / len(self._result)
                    confidence += data_completeness * 0.1

        # 3. 基于形态的置信度
        if not patterns.empty:
            # 检查ChipDistribution形态
            pattern_count = patterns.sum().sum()
            if pattern_count > 0:
                confidence += min(pattern_count * 0.05, 0.15)

        # 4. 基于信号的置信度
        if signals:
            # 检查信号强度
            signal_count = sum(1 for signal in signals.values() if hasattr(signal, 'any') and signal.any())
            if signal_count > 0:
                confidence += min(signal_count * 0.05, 0.1)

        # 5. 基于评分稳定性的置信度
        if len(score) >= 5:
            recent_scores = score.iloc[-5:]
            score_stability = 1.0 - (recent_scores.std() / 50.0)
            confidence += score_stability * 0.05

        # 确保置信度在0-1范围内
        return max(0.0, min(1.0, confidence))
    
    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        获取ChipDistribution相关形态

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 包含形态信息的DataFrame
        """
        # 确保已计算指标
        if not self.has_result():
            self._calculate(data, **kwargs)

        patterns = pd.DataFrame(index=data.index)

        # 如果没有计算结果，返回空DataFrame
        if self._result is None or self._result.empty:
            return patterns

        # 1. 筹码集中度形态
        if 'chip_concentration' in self._result.columns:
            concentration = self._result['chip_concentration']

            # 高度集中形态
            patterns['CHIP_HIGH_CONCENTRATION'] = concentration > 0.7
            # 分散形态
            patterns['CHIP_DISPERSED'] = concentration < 0.3
            # 集中度上升
            patterns['CHIP_CONCENTRATION_RISING'] = concentration.diff() > 0.1
            # 集中度下降
            patterns['CHIP_CONCENTRATION_FALLING'] = concentration.diff() < -0.1

        # 2. 获利盘比例形态
        if 'profit_ratio' in self._result.columns:
            profit_ratio = self._result['profit_ratio']

            # 高获利盘
            patterns['CHIP_HIGH_PROFIT'] = profit_ratio > 0.8
            # 低获利盘
            patterns['CHIP_LOW_PROFIT'] = profit_ratio < 0.2
            # 获利盘快速增加
            patterns['CHIP_PROFIT_SURGE'] = profit_ratio.diff() > 0.2
            # 获利盘快速减少
            patterns['CHIP_PROFIT_DROP'] = profit_ratio.diff() < -0.2

        # 3. 成本偏离形态
        if 'avg_cost' in self._result.columns:
            avg_cost = self._result['avg_cost']
            current_price = data['close']

            # 价格远高于成本
            patterns['PRICE_FAR_ABOVE_COST'] = (current_price / avg_cost) > 1.2
            # 价格远低于成本
            patterns['PRICE_FAR_BELOW_COST'] = (current_price / avg_cost) < 0.8
            # 价格接近成本
            patterns['PRICE_NEAR_COST'] = ((current_price / avg_cost) >= 0.95) & ((current_price / avg_cost) <= 1.05)

        # 4. 解套难度形态
        if 'untrapped_difficulty' in self._result.columns:
            untrapped = self._result['untrapped_difficulty']

            # 解套容易
            patterns['EASY_UNTRAPPED'] = untrapped < 0.9
            # 解套困难
            patterns['HARD_UNTRAPPED'] = untrapped > 1.1

        # 5. 筹码松散度形态
        if 'chip_looseness' in self._result.columns:
            looseness = self._result['chip_looseness']

            # 筹码松散
            patterns['CHIP_LOOSE'] = looseness > 3.0
            # 筹码紧密
            patterns['CHIP_TIGHT'] = looseness < 1.5

        # 6. 综合形态
        if 'chip_concentration' in self._result.columns and 'profit_ratio' in self._result.columns:
            concentration = self._result['chip_concentration']
            profit_ratio = self._result['profit_ratio']

            # 底部吸筹形态：高集中度 + 低获利盘
            patterns['CHIP_BOTTOM_ACCUMULATION'] = (concentration > 0.6) & (profit_ratio < 0.3)
            # 顶部派发形态：低集中度 + 高获利盘
            patterns['CHIP_TOP_DISTRIBUTION'] = (concentration < 0.4) & (profit_ratio > 0.7)
            # 主升浪形态：中等集中度 + 中等获利盘
            patterns['CHIP_MAIN_WAVE'] = (concentration >= 0.4) & (concentration <= 0.6) & (profit_ratio >= 0.3) & (profit_ratio <= 0.7)

        return patterns
    
    def _calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算筹码分布指标

        Args:
            data: 输入数据，包含OHLCV和换手率数据
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 计算结果，包含筹码分布相关指标
        """
        if data.empty:
            return pd.DataFrame(index=data.index)

        # 初始化结果DataFrame
        result = data.copy()

        # 获取参数
        half_life = kwargs.get('half_life', self._parameters.get('half_life', 60))
        price_precision = kwargs.get('price_precision', self._parameters.get('price_precision', 0.01))

        # 如果没有换手率数据，使用成交量估算
        if 'turnover_rate' not in data.columns:
            # 简单估算换手率：假设流通股本为固定值
            avg_volume = data['volume'].mean()
            estimated_shares = avg_volume * 100  # 估算流通股本
            result['turnover_rate'] = (data['volume'] / estimated_shares) * 100

        # 计算简化的筹码分布指标
        try:
            # 1. 计算平均成本（使用成交量加权平均价格的近似）
            result['avg_cost'] = ((data['high'] + data['low'] + data['close']) / 3).rolling(window=20).mean()

            # 2. 计算筹码集中度（基于价格波动率的倒数）
            price_volatility = data['close'].rolling(window=20).std() / data['close'].rolling(window=20).mean()
            result['chip_concentration'] = np.clip(1 / (price_volatility + 0.01), 0, 1)

            # 3. 计算获利盘比例
            result['profit_ratio'] = np.where(
                data['close'] > result['avg_cost'],
                (data['close'] - result['avg_cost']) / result['avg_cost'],
                0
            )
            result['profit_ratio'] = np.clip(result['profit_ratio'], 0, 1)

            # 4. 计算90%筹码区间宽度（基于价格波动）
            price_range = data['high'].rolling(window=20).max() - data['low'].rolling(window=20).min()
            result['chip_width_90pct'] = price_range / data['close']

            # 5. 计算解套难度
            result['untrapped_difficulty'] = data['close'] / result['avg_cost']

            # 6. 计算筹码松散度
            result['chip_looseness'] = 1 / (result['chip_concentration'] + 0.0001)

            # 7. 计算筹码变动率
            result['profit_ratio_change'] = result['profit_ratio'].diff()

            # 8. 计算成本偏离度
            result['cost_deviation'] = (data['close'] - result['avg_cost']) / result['avg_cost']

        except Exception as e:
            logger.error(f"计算筹码分布指标时出错: {e}")
            # 返回基础数据
            result['avg_cost'] = data['close']
            result['chip_concentration'] = 0.5
            result['profit_ratio'] = 0.5
            result['chip_width_90pct'] = 0.1
            result['untrapped_difficulty'] = 1.0
            result['chip_looseness'] = 2.0
            result['profit_ratio_change'] = 0.0
            result['cost_deviation'] = 0.0

        return result

    def register_patterns(self):
        """
        注册ChipDistribution指标的形态到全局形态注册表
        """
        # 注册筹码集中形态
        self.register_pattern_to_registry(
            pattern_id="CHIP_HIGH_CONCENTRATION",
            display_name="筹码高度集中",
            description="筹码集中度较高，通常表明主力控盘程度较强",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=15.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="CHIP_DISPERSED",
            display_name="筹码分散",
            description="筹码分散度较高，通常表明散户持股较多",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-10.0,
            polarity="NEGATIVE"
        )

        # 注册获利盘形态
        self.register_pattern_to_registry(
            pattern_id="CHIP_HIGH_PROFIT",
            display_name="高获利盘",
            description="获利盘比例较高，存在获利回吐压力",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-15.0,
            polarity="NEGATIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="CHIP_LOW_PROFIT",
            display_name="低获利盘",
            description="获利盘比例较低，上涨阻力相对较小",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=10.0,
            polarity="POSITIVE"
        )

        # 注册成本偏离形态
        self.register_pattern_to_registry(
            pattern_id="PRICE_FAR_ABOVE_COST",
            display_name="价格远高于成本",
            description="当前价格远高于平均成本，存在回调风险",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-20.0,
            polarity="NEGATIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="PRICE_FAR_BELOW_COST",
            display_name="价格远低于成本",
            description="当前价格远低于平均成本，具有反弹潜力",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=20.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="PRICE_NEAR_COST",
            display_name="价格接近成本",
            description="当前价格接近平均成本，处于相对均衡状态",
            pattern_type="NEUTRAL",
            default_strength="WEAK",
            score_impact=0.0,
            polarity="NEUTRAL"
        )

        # 注册综合形态
        self.register_pattern_to_registry(
            pattern_id="CHIP_BOTTOM_ACCUMULATION",
            display_name="底部吸筹",
            description="高集中度低获利盘，通常表明底部吸筹阶段",
            pattern_type="BULLISH",
            default_strength="VERY_STRONG",
            score_impact=25.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="CHIP_TOP_DISTRIBUTION",
            display_name="顶部派发",
            description="低集中度高获利盘，通常表明顶部派发阶段",
            pattern_type="BEARISH",
            default_strength="VERY_STRONG",
            score_impact=-25.0,
            polarity="NEGATIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="CHIP_MAIN_WAVE",
            display_name="主升浪",
            description="中等集中度中等获利盘，通常表明主升浪阶段",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=20.0,
            polarity="POSITIVE"
        )

        # 注册解套难度形态
        self.register_pattern_to_registry(
            pattern_id="EASY_UNTRAPPED",
            display_name="解套容易",
            description="解套难度较低，套牢盘压力较小",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=10.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="HARD_UNTRAPPED",
            display_name="解套困难",
            description="解套难度较高，套牢盘压力较大",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-10.0,
            polarity="NEGATIVE"
        )
    
    def _calculate_day_chip_distribution(self, day_data: pd.Series, 
                                       price_grid: np.ndarray, 
                                       price_precision: float) -> np.ndarray:
        """
        计算单日的筹码分布 (优化版)
        
        使用改进的模型计算日内筹码分布，考虑价格水平、交易习惯和市场微观结构
        
        Args:
            day_data: 单日数据
            price_grid: 价格网格
            price_precision: 价格精度
            
        Returns:
            np.ndarray: 单日筹码分布
        """
        # 提取数据
        open_price = day_data["open"]
        high_price = day_data["high"]
        low_price = day_data["low"]
        close_price = day_data["close"]
        turnover_rate = day_data["turnover_rate"] / 100  # 转换为小数
        
        # 初始化分布
        n_prices = len(price_grid)
        day_chip = np.zeros(n_prices)
        
        # 1. 使用改进的筹码分布模型
        # 筹码分布不再是均匀分布，而是基于价格水平的加权分布
        
        # 价格区间
        price_range = high_price - low_price
        if price_range <= 0:
            # 如果最高价等于最低价（罕见情况），使用简单模型
            for i, price in enumerate(price_grid):
                if low_price <= price <= high_price:
                    day_chip[i] = turnover_rate
            return day_chip
        
        # 2. 根据日内价格特征选择合适的分布模型
        # 获取当日K线形态特征
        price_pattern = self._identify_price_pattern(open_price, high_price, low_price, close_price)
        
        # 判断是否有分时数据
        has_intraday_data = False
        if hasattr(day_data, "intraday_prices") and hasattr(day_data, "intraday_volumes"):
            try:
                intraday_prices = day_data["intraday_prices"]
                intraday_volumes = day_data["intraday_volumes"]
                if isinstance(intraday_prices, (list, np.ndarray)) and len(intraday_prices) > 0:
                    has_intraday_data = True
            except:
                pass
        
        if has_intraday_data:
            # 3. 如果有分时数据，使用实际分时成交数据构建分布
            return self._calculate_distribution_from_intraday(
                intraday_prices, intraday_volumes, price_grid, turnover_rate
            )
        else:
            # 4. 没有分时数据，根据K线形态使用不同的模型
            if price_pattern == "doji":  # 十字星
                # 成交集中在开盘价和收盘价附近
                return self._calculate_doji_distribution(
                    open_price, high_price, low_price, close_price, 
                    price_grid, turnover_rate
                )
            elif price_pattern == "trending_up":  # 上涨趋势
                # 成交量在低位和高位都比较大，中间较小
                return self._calculate_trending_up_distribution(
                    open_price, high_price, low_price, close_price, 
                    price_grid, turnover_rate
                )
            elif price_pattern == "trending_down":  # 下跌趋势
                # 成交量在高位和低位都比较大，中间较小
                return self._calculate_trending_down_distribution(
                    open_price, high_price, low_price, close_price, 
                    price_grid, turnover_rate
                )
            else:  # 默认情况，使用TWAP分布
                return self._calculate_twap_distribution(
                    open_price, high_price, low_price, close_price, 
                    price_grid, turnover_rate
                )
    
    def _identify_price_pattern(self, open_price: float, high_price: float, 
                              low_price: float, close_price: float) -> str:
        """
        识别价格模式
        
        Args:
            open_price: 开盘价
            high_price: 最高价
            low_price: 最低价
            close_price: 收盘价
            
        Returns:
            str: 价格模式
        """
        # 计算实体与影线的关系
        body_size = abs(close_price - open_price)
        total_range = high_price - low_price
        body_ratio = body_size / total_range if total_range > 0 else 0
        
        # 判断十字星
        if body_ratio < 0.1:
            return "doji"
        
        # 判断趋势
        if close_price > open_price:  # 收盘价高于开盘价，看涨
            up_shadow = high_price - max(open_price, close_price)
            down_shadow = min(open_price, close_price) - low_price
            
            if up_shadow < 0.3 * body_size and body_ratio > 0.6:
                return "trending_up"  # 上涨趋势
            else:
                return "mixed"
        else:  # 收盘价低于开盘价，看跌
            up_shadow = high_price - max(open_price, close_price)
            down_shadow = min(open_price, close_price) - low_price
            
            if down_shadow < 0.3 * body_size and body_ratio > 0.6:
                return "trending_down"  # 下跌趋势
            else:
                return "mixed"
    
    def _calculate_distribution_from_intraday(self, intraday_prices: List[float], 
                                           intraday_volumes: List[float],
                                           price_grid: np.ndarray,
                                           turnover_rate: float) -> np.ndarray:
        """
        从分时数据计算筹码分布
        
        Args:
            intraday_prices: 分时价格
            intraday_volumes: 分时成交量
            price_grid: 价格网格
            turnover_rate: 换手率
            
        Returns:
            np.ndarray: 分时筹码分布
        """
        n_prices = len(price_grid)
        day_chip = np.zeros(n_prices)
        
        # 确保数据完整
        if len(intraday_prices) != len(intraday_volumes) or len(intraday_prices) == 0:
            return day_chip
        
        # 计算总成交量
        total_volume = sum(intraday_volumes)
        if total_volume <= 0:
            return day_chip
        
        # 创建价格到成交量的映射
        price_volume_map = {}
        for i in range(len(intraday_prices)):
            price = intraday_prices[i]
            volume = intraday_volumes[i]
            
            # 四舍五入到价格精度
            grid_price = round(price / price_precision) * price_precision
            
            if grid_price in price_volume_map:
                price_volume_map[grid_price] += volume
            else:
                price_volume_map[grid_price] = volume
        
        # 映射到价格网格
        for i, price in enumerate(price_grid):
            if price in price_volume_map:
                day_chip[i] = price_volume_map[price] / total_volume * turnover_rate
        
        return day_chip
    
    def _calculate_doji_distribution(self, open_price: float, high_price: float, 
                                  low_price: float, close_price: float,
                                  price_grid: np.ndarray, turnover_rate: float) -> np.ndarray:
        """
        计算十字星形态的筹码分布
        
        Args:
            open_price: 开盘价
            high_price: 最高价
            low_price: 最低价
            close_price: 收盘价
            price_grid: 价格网格
            turnover_rate: 换手率
            
        Returns:
            np.ndarray: 筹码分布
        """
        n_prices = len(price_grid)
        day_chip = np.zeros(n_prices)
        
        # 十字星特征：成交量集中在开盘价和收盘价附近，同时在最高价和最低价处有小幅放量
        
        # 定义权重
        open_weight = 0.3
        close_weight = 0.3
        high_weight = 0.2
        low_weight = 0.2
        
        # 计算高斯分布的标准差（价格区间的10%）
        sigma = (high_price - low_price) * 0.1
        if sigma <= 0:
            sigma = (high_price + low_price) * 0.001  # 防止除零
        
        # 应用高斯分布模型
        for i, price in enumerate(price_grid):
            if low_price <= price <= high_price:
                # 开盘价附近的分布
                open_contribution = open_weight * np.exp(-0.5 * ((price - open_price) / sigma) ** 2)
                
                # 收盘价附近的分布
                close_contribution = close_weight * np.exp(-0.5 * ((price - close_price) / sigma) ** 2)
                
                # 最高价附近的分布
                high_contribution = high_weight * np.exp(-0.5 * ((price - high_price) / sigma) ** 2)
                
                # 最低价附近的分布
                low_contribution = low_weight * np.exp(-0.5 * ((price - low_price) / sigma) ** 2)
                
                # 组合分布
                day_chip[i] = open_contribution + close_contribution + high_contribution + low_contribution
        
        # 归一化
        if np.sum(day_chip) > 0:
            day_chip = day_chip / np.sum(day_chip) * turnover_rate
        
        return day_chip
    
    def _calculate_trending_up_distribution(self, open_price: float, high_price: float, 
                                         low_price: float, close_price: float,
                                         price_grid: np.ndarray, turnover_rate: float) -> np.ndarray:
        """
        计算上涨趋势的筹码分布
        
        Args:
            open_price: 开盘价
            high_price: 最高价
            low_price: 最低价
            close_price: 收盘价
            price_grid: 价格网格
            turnover_rate: 换手率
            
        Returns:
            np.ndarray: 筹码分布
        """
        n_prices = len(price_grid)
        day_chip = np.zeros(n_prices)
        
        # 上涨趋势特征：成交量在低位(开盘价)和高位(收盘价)都比较大，中间较小
        
        # 定义权重
        open_weight = 0.4  # 开盘价附近成交放量
        close_weight = 0.4  # 收盘价附近成交放量
        middle_weight = 0.2  # 中间价位成交较少
        
        # 计算高斯分布的标准差
        sigma = (high_price - low_price) * 0.15
        if sigma <= 0:
            sigma = (high_price + low_price) * 0.001  # 防止除零
        
        # 计算中间价
        middle_price = (open_price + close_price) / 2
        
        # 应用混合分布模型
        for i, price in enumerate(price_grid):
            if low_price <= price <= high_price:
                # 开盘价附近的分布（买入）
                open_contribution = open_weight * np.exp(-0.5 * ((price - open_price) / sigma) ** 2)
                
                # 收盘价附近的分布（卖出）
                close_contribution = close_weight * np.exp(-0.5 * ((price - close_price) / sigma) ** 2)
                
                # 中间价位的低成交量
                middle_contribution = middle_weight * (1 - 0.8 * np.exp(-0.5 * ((price - middle_price) / sigma) ** 2))
                
                # 组合分布
                day_chip[i] = open_contribution + close_contribution + middle_contribution
        
        # 归一化
        if np.sum(day_chip) > 0:
            day_chip = day_chip / np.sum(day_chip) * turnover_rate
        
        return day_chip
    
    def _calculate_trending_down_distribution(self, open_price: float, high_price: float, 
                                           low_price: float, close_price: float,
                                           price_grid: np.ndarray, turnover_rate: float) -> np.ndarray:
        """
        计算下跌趋势的筹码分布
        
        Args:
            open_price: 开盘价
            high_price: 最高价
            low_price: 最低价
            close_price: 收盘价
            price_grid: 价格网格
            turnover_rate: 换手率
            
        Returns:
            np.ndarray: 筹码分布
        """
        n_prices = len(price_grid)
        day_chip = np.zeros(n_prices)
        
        # 下跌趋势特征：成交量在高位(开盘价)和低位(收盘价)都比较大，中间较小
        
        # 定义权重
        open_weight = 0.4  # 开盘价附近成交放量
        close_weight = 0.4  # 收盘价附近成交放量
        middle_weight = 0.2  # 中间价位成交较少
        
        # 计算高斯分布的标准差
        sigma = (high_price - low_price) * 0.15
        if sigma <= 0:
            sigma = (high_price + low_price) * 0.001  # 防止除零
        
        # 计算中间价
        middle_price = (open_price + close_price) / 2
        
        # 应用混合分布模型
        for i, price in enumerate(price_grid):
            if low_price <= price <= high_price:
                # 开盘价附近的分布（卖出）
                open_contribution = open_weight * np.exp(-0.5 * ((price - open_price) / sigma) ** 2)
                
                # 收盘价附近的分布（买入）
                close_contribution = close_weight * np.exp(-0.5 * ((price - close_price) / sigma) ** 2)
                
                # 中间价位的低成交量
                middle_contribution = middle_weight * (1 - 0.8 * np.exp(-0.5 * ((price - middle_price) / sigma) ** 2))
                
                # 组合分布
                day_chip[i] = open_contribution + close_contribution + middle_contribution
        
        # 归一化
        if np.sum(day_chip) > 0:
            day_chip = day_chip / np.sum(day_chip) * turnover_rate
        
        return day_chip
    
    def _calculate_twap_distribution(self, open_price: float, high_price: float, 
                                   low_price: float, close_price: float,
                                   price_grid: np.ndarray, turnover_rate: float) -> np.ndarray:
        """
        计算基于TWAP的筹码分布
        
        Args:
            open_price: 开盘价
            high_price: 最高价
            low_price: 最低价
            close_price: 收盘价
            price_grid: 价格网格
            turnover_rate: 换手率
            
        Returns:
            np.ndarray: 筹码分布
        """
        n_prices = len(price_grid)
        day_chip = np.zeros(n_prices)
        
        # 时间加权平均价格(TWAP)分布
        # 假设价格在交易日内按OHLC的三角分布运动
        
        # 计算TWAP价格（简化模型）
        twap = (open_price + high_price + low_price + close_price) / 4
        
        # 计算分布宽度
        width = high_price - low_price
        if width <= 0:
            width = (high_price + low_price) * 0.01  # 防止宽度为零
        
        # 应用三角分布模型
        for i, price in enumerate(price_grid):
            if low_price <= price <= high_price:
                # 距离TWAP的偏离度
                distance = abs(price - twap) / width
                
                # 三角分布：越接近TWAP，概率越大
                day_chip[i] = max(0, 1 - distance * 2) * turnover_rate
        
        # 归一化
        if np.sum(day_chip) > 0:
            day_chip = day_chip / np.sum(day_chip) * turnover_rate
        
        return day_chip
    
    def _calculate_chip_metrics(self, data: pd.DataFrame, result: pd.DataFrame, 
                              chip_matrix: np.ndarray, price_grid: np.ndarray) -> pd.DataFrame:
        """
        计算筹码分布相关指标
        
        Args:
            data: 原始数据
            result: 结果数据框
            chip_matrix: 筹码分布矩阵
            price_grid: 价格网格
            
        Returns:
            pd.DataFrame: 更新后的结果数据框
        """
        n_days = len(data)
        close_prices = data["close"].values
        
        # 计算筹码集中度
        chip_concentration = np.zeros(n_days)
        
        # 计算获利盘比例
        profit_ratio = np.zeros(n_days)
        
        # 计算90%筹码区间宽度
        chip_width_90pct = np.zeros(n_days)
        
        # 计算套牢盘成本
        avg_cost = np.zeros(n_days)
        
        for i in range(n_days):
            # 当日收盘价
            close = close_prices[i]
            
            # 找到收盘价对应的价格网格索引
            close_idx = np.argmin(np.abs(price_grid - close))
            
            # 计算筹码集中度：90%筹码所在的价格区间占全部价格区间的比例
            sorted_chip = np.sort(chip_matrix[i])[::-1]  # 降序排列
            cumsum_chip = np.cumsum(sorted_chip)
            chip_90pct_idx = np.argmax(cumsum_chip >= 0.9)
            if chip_90pct_idx > 0:
                chip_concentration[i] = chip_90pct_idx / len(price_grid)
            
            # 计算获利盘比例：当前价格以下的筹码比例
            profit_ratio[i] = np.sum(chip_matrix[i][:close_idx+1])
            
            # 计算90%筹码区间宽度
            chip_cumsum = np.cumsum(chip_matrix[i])
            lower_idx = np.argmax(chip_cumsum >= 0.05)
            upper_idx = np.argmax(chip_cumsum >= 0.95)
            if upper_idx > lower_idx:
                chip_width_90pct[i] = (price_grid[upper_idx] - price_grid[lower_idx]) / close
            
            # 计算平均成本
            avg_cost[i] = np.sum(price_grid * chip_matrix[i]) / np.sum(chip_matrix[i]) if np.sum(chip_matrix[i]) > 0 else 0
        
        # 添加到结果
        result["chip_concentration"] = chip_concentration  # 筹码集中度
        result["profit_ratio"] = profit_ratio              # 获利盘比例
        result["chip_width_90pct"] = chip_width_90pct      # 90%筹码区间宽度
        result["avg_cost"] = avg_cost                      # 平均成本
        
        # 计算解套难度：当前价格与平均成本的比值
        result["untrapped_difficulty"] = close_prices / avg_cost
        
        # 计算筹码松散度：筹码集中度的倒数
        result["chip_looseness"] = 1 / (chip_concentration + 0.0001)  # 避免除以零
        
        # 计算筹码变动率：当日获利盘比例与前一日的差值
        result["profit_ratio_change"] = result["profit_ratio"].diff()
        
        return result
    
    def get_distribution_at_price(self, data: pd.DataFrame, 
                                price_point: float, 
                                half_life: int = 60,
                                price_precision: float = 0.01) -> float:
        """
        获取特定价格点的筹码密度
        
        Args:
            data: 输入数据
            price_point: 价格点
            half_life: 半衰期
            price_precision: 价格精度
            
        Returns:
            float: 价格点的筹码密度
        """
        # 计算筹码分布
        result = self._calculate(data, half_life, price_precision)
        
        # 找到最后一天的收盘价附近的筹码密度
        min_price = data["low"].min() * 0.9
        price_idx = int((price_point - min_price) / price_precision)
        
        # 安全检查
        if price_idx < 0:
            return 0.0
        
        # 计算所需的值（这里简化处理，实际应用需要访问chip_matrix）
        # 由于我们没有直接存储chip_matrix，这里使用估算方法
        
        # 获取最近的价格点
        last_close = data["close"].iloc[-1]
        last_avg_cost = result["avg_cost"].iloc[-1]
        
        # 估算密度：基于正态分布假设，价格越接近平均成本，密度越大
        density = np.exp(-0.5 * ((price_point - last_avg_cost) / (0.1 * last_avg_cost))**2)
        
        return density
    
    def _calculate_precision_cost(self, data: pd.DataFrame, 
                             chip_matrix: np.ndarray, 
                             price_grid: np.ndarray,
                             volume_profile_weight: float = 0.3,
                             long_term_weight: float = 0.2) -> np.ndarray:
        """
        高精度成本估算
        
        使用多因素加权计算更精确的成本分布，考虑了交易量分布特征、
        长期持仓偏好和交易习惯等因素
        
        Args:
            data: 原始数据DataFrame
            chip_matrix: 筹码分布矩阵
            price_grid: 价格网格
            volume_profile_weight: 成交量分布权重
            long_term_weight: 长期持仓权重
            
        Returns:
            np.ndarray: 高精度成本估算数组
        """
        n_days = len(data)
        precision_cost = np.zeros(n_days)
        
        # 提取所需数据
        close_prices = data["close"].values
        high_prices = data["high"].values
        low_prices = data["low"].values
        volumes = data["volume"].values
        
        # 1. 计算基础成本估算（价格加权平均）
        for i in range(n_days):
            if np.sum(chip_matrix[i]) > 0:
                precision_cost[i] = np.sum(price_grid * chip_matrix[i]) / np.sum(chip_matrix[i])
            else:
                precision_cost[i] = close_prices[i]  # 默认为当日收盘价
        
        # 2. 应用成交量分布调整
        if "vwap" in data.columns:
            # 如果有VWAP数据，直接使用
            vwap = data["vwap"].values
        else:
            # 否则估算VWAP (Volume Weighted Average Price)
            vwap = (high_prices + low_prices + close_prices) / 3
            
            # 如果有分时数据，可以使用更精确的VWAP计算
            if "intraday_prices" in data.columns and "intraday_volumes" in data.columns:
                for i in range(n_days):
                    try:
                        intraday_prices = data["intraday_prices"].iloc[i]
                        intraday_volumes = data["intraday_volumes"].iloc[i]
                        if len(intraday_prices) == len(intraday_volumes) and len(intraday_prices) > 0:
                            vwap[i] = np.sum(np.array(intraday_prices) * np.array(intraday_volumes)) / np.sum(intraday_volumes)
                    except:
                        # 使用估算的VWAP
                        pass
        
        # 使用VWAP调整成本估算
        precision_cost = precision_cost * (1 - volume_profile_weight) + vwap * volume_profile_weight
        
        # 3. 应用长期持仓偏好调整
        if n_days > 60:  # 需要足够的历史数据
            # 计算60日移动平均线作为长期持仓偏好
            ma60 = np.zeros(n_days)
            for i in range(n_days):
                start_idx = max(0, i - 59)
                if i >= 59:
                    ma60[i] = np.mean(close_prices[start_idx:i+1])
                else:
                    ma60[i] = np.mean(close_prices[:i+1])
            
            # 长期持仓成本趋向于长期均线
            precision_cost = precision_cost * (1 - long_term_weight) + ma60 * long_term_weight
        
        # 4. 应用大资金行为调整
        # 如果有大单数据，可以更精确地估计机构成本
        if "big_order_price" in data.columns and "big_order_volume" in data.columns:
            big_order_adjustment = np.zeros(n_days)
            big_order_weight = np.zeros(n_days)
            
            for i in range(n_days):
                try:
                    big_prices = data["big_order_price"].iloc[i]
                    big_volumes = data["big_order_volume"].iloc[i]
                    
                    if len(big_prices) > 0 and len(big_volumes) > 0:
                        big_order_adjustment[i] = np.sum(np.array(big_prices) * np.array(big_volumes)) / np.sum(big_volumes)
                        
                        # 大单权重与大单占比相关
                        total_volume = volumes[i] if volumes[i] > 0 else 1
                        big_order_ratio = np.sum(big_volumes) / total_volume
                        big_order_weight[i] = min(0.4, big_order_ratio)  # 最高权重40%
                        
                        # 应用大单调整
                        precision_cost[i] = precision_cost[i] * (1 - big_order_weight[i]) + big_order_adjustment[i] * big_order_weight[i]
                except:
                    # 无大单数据或数据格式错误，跳过调整
                    pass
        
        # 5. 应用波动率调整
        # 在高波动时期，成本分布更分散
        if n_days > 20:
            volatility = np.zeros(n_days)
            for i in range(n_days):
                start_idx = max(0, i - 19)
                if i >= 19:
                    price_changes = np.abs(np.diff(close_prices[start_idx:i+1]) / close_prices[start_idx:i])
                    volatility[i] = np.mean(price_changes)
                else:
                    price_changes = np.abs(np.diff(close_prices[:i+1]) / close_prices[:i])
                    volatility[i] = np.mean(price_changes) if len(price_changes) > 0 else 0
            
            # 使用波动率作为权重进行成本分散调整
            # 在高波动时期，成本更接近于近期价格均值而非理论成本
            recent_prices_mean = np.zeros(n_days)
            for i in range(n_days):
                window = min(10, i + 1)  # 最多使用10天数据
                recent_prices_mean[i] = np.mean(close_prices[i+1-window:i+1])
            
            volatility_adj_weight = np.clip(volatility * 10, 0, 0.3)  # 将波动率转换为权重，最高30%
            precision_cost = precision_cost * (1 - volatility_adj_weight) + recent_prices_mean * volatility_adj_weight
        
        return precision_cost 

    def identify_institutional_chips(self, data: pd.DataFrame, 
                                half_life: int = 60, 
                                price_precision: float = 0.01,
                                volume_threshold: float = 2.0,
                                consecutive_days: int = 3) -> pd.DataFrame:
        """
        识别主力筹码分布
        
        优化实现：精细化主力持仓识别，提高主力筹码识别准确率
        
        Args:
            data: 包含OHLCV数据的DataFrame
            half_life: 半衰期，默认60天
            price_precision: 价格精度，默认0.01
            volume_threshold: 成交量阈值，默认为均值的2倍
            consecutive_days: 连续天数阈值，默认3天
            
        Returns:
            pd.DataFrame: 主力筹码分析结果
        """
        # 计算基础筹码分布
        result = self._calculate(data, half_life, price_precision, use_precision_cost=True)
        
        # 初始化结果
        inst_result = pd.DataFrame(index=data.index)
        
        # 获取价格网格
        min_price = data["low"].min() * 0.9
        max_price = data["high"].max() * 1.1
        price_grid = np.arange(min_price, max_price + price_precision, price_precision)
        
        # 1. 识别大单交易日
        # 计算相对成交量
        volume = data["volume"]
        volume_ma20 = volume.rolling(window=20).mean()
        relative_volume = volume / volume_ma20
        
        # 识别放量交易日
        high_volume_days = relative_volume > volume_threshold
        
        # 2. 识别连续放量
        consecutive_high_volume = np.zeros_like(high_volume_days, dtype=bool)
        for i in range(consecutive_days - 1, len(data)):
            if np.all(high_volume_days.iloc[i-(consecutive_days-1):i+1].values):
                consecutive_high_volume[i] = True
        
        # 3. 识别价格区间
        # 获取连续放量区间的价格范围
        inst_price_ranges = []
        
        i = 0
        while i < len(data):
            if consecutive_high_volume[i]:
                # 找到连续放量的起止点
                start_idx = i
                while i < len(data) and (high_volume_days[i] or consecutive_high_volume[i]):
                    i += 1
                end_idx = i - 1
                
                # 获取区间价格范围
                price_range = {
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "low": data["low"].iloc[start_idx:end_idx+1].min(),
                    "high": data["high"].iloc[start_idx:end_idx+1].max(),
                    "volume": data["volume"].iloc[start_idx:end_idx+1].sum(),
                    "days": end_idx - start_idx + 1
                }
                
                inst_price_ranges.append(price_range)
            else:
                i += 1
        
        # 4. 估算主力筹码分布
        n_days = len(data)
        institutional_chip = np.zeros((n_days, len(price_grid)))
        
        # 根据放量区间分配主力筹码
        for price_range in inst_price_ranges:
            start_idx = price_range["start_idx"]
            end_idx = price_range["end_idx"]
            low_price = price_range["low"]
            high_price = price_range["high"]
            
            # 计算该区间的权重（基于成交量和天数）
            weight = price_range["volume"] / data["volume"].iloc[start_idx:end_idx+1].sum()
            
            # 为该区间分配主力筹码
            for i in range(start_idx, min(end_idx + 1, n_days)):
                # 在价格区间内分配筹码
                for j, price in enumerate(price_grid):
                    if low_price <= price <= high_price:
                        # 使用高斯分布模型
                        mid_price = (low_price + high_price) / 2
                        sigma = (high_price - low_price) / 4  # 标准差为价格区间的1/4
                        
                        # 计算筹码分布密度
                        density = np.exp(-0.5 * ((price - mid_price) / sigma) ** 2)
                        
                        # 累加到主力筹码分布
                        institutional_chip[i, j] += density * weight
        
        # 归一化主力筹码分布
        for i in range(n_days):
            if np.sum(institutional_chip[i]) > 0:
                institutional_chip[i] = institutional_chip[i] / np.sum(institutional_chip[i])
        
        # 5. 计算主力筹码指标
        # 主力成本
        inst_cost = np.zeros(n_days)
        for i in range(n_days):
            if np.sum(institutional_chip[i]) > 0:
                inst_cost[i] = np.sum(price_grid * institutional_chip[i]) / np.sum(institutional_chip[i])
            else:
                inst_cost[i] = data["close"].iloc[i]
        
        # 主力获利比例
        inst_profit_ratio = np.zeros(n_days)
        for i in range(n_days):
            close_price = data["close"].iloc[i]
            close_idx = np.argmin(np.abs(price_grid - close_price))
            
            if np.sum(institutional_chip[i]) > 0:
                inst_profit_ratio[i] = np.sum(institutional_chip[i][:close_idx+1])
        
        # 主力筹码集中度
        inst_concentration = np.zeros(n_days)
        for i in range(n_days):
            if np.sum(institutional_chip[i]) > 0:
                sorted_chip = np.sort(institutional_chip[i])[::-1]
                cumsum_chip = np.cumsum(sorted_chip)
                chip_90pct_idx = np.argmax(cumsum_chip >= 0.9)
                if chip_90pct_idx > 0:
                    inst_concentration[i] = chip_90pct_idx / len(price_grid)
        
        # 6. 主力活跃度评分
        inst_activity_score = np.zeros(n_days)
        
        # 计算最近N天的成交量占比
        activity_window = 10
        for i in range(n_days):
            if i < activity_window:
                continue
                
            recent_volume = data["volume"].iloc[i-activity_window+1:i+1].sum()
            total_volume = data["volume"].iloc[:i+1].sum()
            
            if total_volume > 0:
                recent_volume_ratio = recent_volume / total_volume
                inst_activity_score[i] = min(100, recent_volume_ratio * 500)  # 缩放到0-100
        
        # 7. 添加到结果
        inst_result["inst_cost"] = inst_cost                       # 主力成本
        inst_result["inst_profit_ratio"] = inst_profit_ratio       # 主力获利比例
        inst_result["inst_concentration"] = inst_concentration     # 主力筹码集中度
        inst_result["inst_activity_score"] = inst_activity_score   # 主力活跃度评分
        
        # 计算散户套牢比例（市场平均成本与主力成本的差距）
        inst_result["retail_trapped_ratio"] = (result["avg_cost"] - inst_cost) / result["avg_cost"]
        
        # 计算主力控盘比例估计
        inst_result["inst_control_ratio"] = np.clip(inst_concentration * 2, 0, 0.9)  # 最高控盘比例90%
        
        # 计算主力建仓阶段
        inst_result["inst_phase"] = pd.Series("观望期", index=data.index)
        
        # 识别吸筹期
        absorption_condition = (inst_activity_score > 60) & (inst_profit_ratio < 0.4)
        inst_result.loc[absorption_condition, "inst_phase"] = "吸筹期"
        
        # 识别控盘期
        control_condition = (inst_concentration > 0.3) & (inst_profit_ratio < 0.7) & (inst_profit_ratio > 0.4)
        inst_result.loc[control_condition, "inst_phase"] = "控盘期"
        
        # 识别拉升期
        rally_condition = (inst_profit_ratio > 0.7) & (inst_activity_score > 50)
        inst_result.loc[rally_condition, "inst_phase"] = "拉升期"
        
        # 识别出货期
        distribution_condition = (inst_profit_ratio > 0.9) & (inst_concentration < 0.2)
        inst_result.loc[distribution_condition, "inst_phase"] = "出货期"
        
        return inst_result
    
    def predict_trapped_position_release(self, data: pd.DataFrame, 
                                       half_life: int = 60, 
                                       price_precision: float = 0.01) -> pd.DataFrame:
        """
        预测套牢盘解套引发的抛压
        
        基于筹码分布模型，预测价格上涨过程中可能面临的抛压区域
        
        Args:
            data: 输入数据，包含OHLCV和换手率数据
            half_life: 半衰期，用于计算筹码衰减，默认为60天
            price_precision: 价格精度，用于设置价格区间，默认为0.01元
            
        Returns:
            pd.DataFrame: 预测结果，包含可能的抛压区域和强度
        """
        # 该函数的实现...
        pass
        
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算筹码分布指标原始评分
        
        Args:
            data: 输入数据，包含OHLCV和换手率数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 评分(0-100)
        """
        # 确保已计算筹码分布
        if not self.has_result():
            self._calculate(data, **kwargs)
            
        if self._result is None:
            return pd.Series(50.0, index=data.index)
            
        # 初始化评分
        score = pd.Series(50.0, index=data.index)
        
        # 如果有成本集中度，计算集中度评分
        if 'chip_concentration' in self._result.columns:
            # 筹码集中度评分：集中度高得分高
            concentration_score = self._result['chip_concentration'] * 30
            score += concentration_score
            
        # 如果有获利比例，计算获利比例评分
        if 'profit_ratio' in self._result.columns:
            # 获利比例评分：获利比例高得分高
            profit_score = self._result['profit_ratio'] * 40
            score += profit_score
            
        # 如果有成本偏离度，计算成本偏离评分
        if 'cost_deviation' in self._result.columns:
            # 成本偏离评分：偏离越小越好
            deviation_score = (1 - np.abs(self._result['cost_deviation'])) * 20
            score += deviation_score
            
        # 限制评分范围
        return np.clip(score, 0, 100)
        
    def generate_trading_signals(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """
        生成基于筹码分布的交易信号
        
        Args:
            data: 输入数据
            **kwargs: 额外参数
            
        Returns:
            Dict[str, pd.Series]: 包含交易信号的字典
        """
        # 确保已计算筹码分布
        if not self.has_result():
            self._calculate(data, **kwargs)
            
        if self._result is None:
            # 返回空信号
            return {
                'buy_signal': pd.Series(False, index=data.index),
                'sell_signal': pd.Series(False, index=data.index),
                'signal_strength': pd.Series(0, index=data.index)
            }
        
        # 初始化信号
        signals = {}
        signals['buy_signal'] = pd.Series(False, index=data.index)
        signals['sell_signal'] = pd.Series(False, index=data.index)
        signals['signal_strength'] = pd.Series(0, index=data.index)
        
        # 1. 基于筹码集中度和获利盘比例生成信号
        if 'chip_concentration' in self._result.columns and 'profit_ratio' in self._result.columns:
            # 高度集中且获利比例低的筹码分布是买入信号
            concentration = self._result['chip_concentration']
            profit_ratio = self._result['profit_ratio']
            
            # 买入条件：筹码高度集中(>0.7)且获利比例低(<0.3)
            buy_condition = (concentration > 0.7) & (profit_ratio < 0.3)
            signals['buy_signal'] = buy_condition
            
            # 卖出条件：筹码分散(<0.5)且获利比例高(>0.7)
            sell_condition = (concentration < 0.5) & (profit_ratio > 0.7)
            signals['sell_signal'] = sell_condition
            
            # 设置信号强度
            signals['signal_strength'] = pd.Series(0, index=data.index)
            signals['signal_strength'].loc[buy_condition] = 1
            signals['signal_strength'].loc[sell_condition] = -1
            
            # 极端情况增强信号强度
            strong_buy = buy_condition & (concentration > 0.85) & (profit_ratio < 0.15)
            strong_sell = sell_condition & (concentration < 0.3) & (profit_ratio > 0.85)
            signals['signal_strength'].loc[strong_buy] = 2
            signals['signal_strength'].loc[strong_sell] = -2
        
        # 2. 如果有解套难度数据，增加基于解套难度的信号
        if 'untrapped_difficulty' in self._result.columns:
            untrapped = self._result['untrapped_difficulty']
            
            # 解套难度低（<0.95）是额外的买入信号
            easy_untrapped = untrapped < 0.95
            signals['buy_signal'] = signals['buy_signal'] | easy_untrapped
            
            # 解套难度高（>1.05）是额外的卖出信号
            hard_untrapped = untrapped > 1.05
            signals['sell_signal'] = signals['sell_signal'] | hard_untrapped
            
            # 增强信号强度
            signals['signal_strength'].loc[easy_untrapped] += 0.5
            signals['signal_strength'].loc[hard_untrapped] -= 0.5
        
        return signals

    def get_indicator_type(self) -> str:
        """
        获取指标类型

        Returns:
            str: 指标类型
        """
        return "CHIPDISTRIBUTION"

    def set_market_environment(self, environment: str):
        """
        设置市场环境

        Args:
            environment: 市场环境字符串
        """
        valid_environments = ['bull_market', 'bear_market', 'sideways_market', 'volatile_market', 'normal']
        if environment not in valid_environments:
            raise ValueError(f"无效的市场环境: {environment}. 有效值: {valid_environments}")

        self.market_environment = environment

    def get_pattern_info(self, pattern_id: str) -> dict:
        """
        获取指定形态的详细信息
        
        Args:
            pattern_id: 形态ID
            
        Returns:
            dict: 形态详细信息
        """
        # 默认形态信息
        default_pattern = {
            "id": pattern_id,
            "name": pattern_id,
            "description": f"{pattern_id}形态",
            "type": "NEUTRAL",
            "strength": "MEDIUM",
            "score_impact": 0.0
        }
        
        # ChipDistribution指标特定的形态信息映射
        pattern_info_map = {
            # 基础形态
            "超买区域": {
                "id": "超买区域",
                "name": "超买区域",
                "description": "指标进入超买区域，可能面临回调压力",
                "type": "BEARISH",
                "strength": "MEDIUM",
                "score_impact": -10.0
            },
            "超卖区域": {
                "id": "超卖区域", 
                "name": "超卖区域",
                "description": "指标进入超卖区域，可能出现反弹机会",
                "type": "BULLISH",
                "strength": "MEDIUM",
                "score_impact": 10.0
            },
            "中性区域": {
                "id": "中性区域",
                "name": "中性区域", 
                "description": "指标处于中性区域，趋势不明确",
                "type": "NEUTRAL",
                "strength": "WEAK",
                "score_impact": 0.0
            },
            # 趋势形态
            "上升趋势": {
                "id": "上升趋势",
                "name": "上升趋势",
                "description": "指标显示上升趋势，看涨信号",
                "type": "BULLISH", 
                "strength": "STRONG",
                "score_impact": 15.0
            },
            "下降趋势": {
                "id": "下降趋势",
                "name": "下降趋势",
                "description": "指标显示下降趋势，看跌信号",
                "type": "BEARISH",
                "strength": "STRONG", 
                "score_impact": -15.0
            },
            # 信号形态
            "买入信号": {
                "id": "买入信号",
                "name": "买入信号",
                "description": "指标产生买入信号，建议关注",
                "type": "BULLISH",
                "strength": "STRONG",
                "score_impact": 20.0
            },
            "卖出信号": {
                "id": "卖出信号", 
                "name": "卖出信号",
                "description": "指标产生卖出信号，建议谨慎",
                "type": "BEARISH",
                "strength": "STRONG",
                "score_impact": -20.0
            }
        }
        
        return pattern_info_map.get(pattern_id, default_pattern)