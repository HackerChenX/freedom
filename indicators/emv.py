#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
指数平均数指标(EMV)
易市场数值（Ease of Movement Value）
通过将价格变化与成交量因素的比率来衡量价格的变化是否容易，判断行情上涨或下跌的阻力大小。
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, List, Dict, Any
import logging


from indicators.base_indicator import BaseIndicator
from indicators.common import crossover, crossunder
from utils.logger import get_logger
# from indicators.atr import ATR  # 移除ATR依赖

logger = logging.getLogger(__name__)


class EMV(BaseIndicator):
    """
    量能指标 (Ease of Movement Value)
    
    EMV通过将价格变化与成交量关联，衡量价格变动的难易程度。
    当股价上升/下跌时，若成交量较小，说明价格易于上升/下跌，EMV数值较大；
    当股价上升/下跌时，若成交量较大，说明价格不易上升/下跌，EMV数值较小。
    
    参数:
        volume_divisor: 成交量调整因子，默认为10000
        period: 移动平均期数，默认为14
    """
    
    def __init__(self, volume_divisor: float = 10000, period: int = 14):
        self.REQUIRED_COLUMNS = ['high', 'low', 'volume']
        """初始化EMV指标"""
        super().__init__(name="EMV", description="指数平均数指标，评估价格上涨下跌的难易程度")
        self.volume_divisor = volume_divisor
        self.period = period
        self._result = None

    def set_parameters(self, volume_divisor: float = None, period: int = None):
        """
        设置指标参数

        Args:
            volume_divisor: 成交量调整因子
            period: 移动平均期数
        """
        if volume_divisor is not None:
            self.volume_divisor = volume_divisor
        if period is not None:
            self.period = period
    
    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """
        验证DataFrame是否包含计算所需的列
        
        Args:
            df: 数据源DataFrame
            
        Raises:
            ValueError: 如果DataFrame缺少所需的列
        """
        required_columns = ['high', 'low', 'volume']
        for column in required_columns:
            if column not in df.columns:
                raise ValueError(f"DataFrame必须包含'{column}'列")
                
        if df['volume'].isnull().all():
            raise ValueError("所有成交量数据都是缺失的")
    
    def _calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算EMV指标
        
        Args:
            df: 包含high, low, close, volume列的DataFrame
            
        Returns:
            包含EMV和EMV_MA列的DataFrame
        """
        self._validate_dataframe(df)
        
        # 创建副本以避免修改原始数据
        df_copy = df.copy()
        
        # 计算中间距离
        mid_point_move = ((df_copy['high'] + df_copy['low']) / 2) - ((df_copy['high'].shift(1) + df_copy['low'].shift(1)) / 2)
        
        # 计算箱体比率
        box_ratio = (df_copy['volume'] / self.volume_divisor) / (df_copy['high'] - df_copy['low'])
        
        # 计算单日EMV
        emv_one_day = mid_point_move / box_ratio
        
        # 计算n日EMV
        df_copy['EMV'] = emv_one_day.rolling(window=self.period).sum()
        
        # 计算EMV的移动平均
        df_copy['EMV_MA'] = df_copy['EMV'].rolling(window=9).mean()
        
        # 保存结果
        self._result = df_copy[['EMV', 'EMV_MA']]
        
        return self._result
    
    def has_result(self) -> bool:
        """检查是否已计算结果"""
        return self._result is not None and not self._result.empty
    
    def generate_signals(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        生成EMV指标的标准化交易信号
        
        Args:
            data: 输入数据，包含OHLCV数据
            *args: 位置参数
            **kwargs: 关键字参数
                
        Returns:
            pd.DataFrame: 信号结果DataFrame，包含标准化信号
        """
        # 确保已计算EMV指标
        if not self.has_result():
            self.calculate(data)
        
        # 获取EMV相关值
        emv = self._result['EMV']
        emv_ma = self._result['EMV_MA']
        
        # 初始化信号DataFrame
        signals = pd.DataFrame(index=data.index)
        signals['buy_signal'] = False
        signals['sell_signal'] = False
        signals['neutral_signal'] = True  # 默认为中性信号
        signals['trend'] = 0  # 0表示中性
        signals['score'] = 50.0  # 默认评分50分
        signals['signal_type'] = None
        signals['signal_desc'] = None
        signals['confidence'] = 50.0
        signals['risk_level'] = '中'
        signals['position_size'] = 0.0
        signals['stop_loss'] = None
        signals['market_env'] = '中性'
        signals['volume_confirmation'] = False
        
        # 计算简化的ATR用于止损设置
        try:
            high_low = data['high'] - data['low']
            high_close = abs(data['high'] - data['close'].shift(1))
            low_close = abs(data['low'] - data['close'].shift(1))
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr_values = true_range.rolling(window=14).mean()
        except Exception as e:
            logger.warning(f"计算ATR失败: {e}")
            atr_values = pd.Series(0, index=data.index)
        
        # 1. EMV上穿0轴，买入信号
        emv_crossover_zero = crossover(emv, 0)
        signals.loc[emv_crossover_zero, 'buy_signal'] = True
        signals.loc[emv_crossover_zero, 'neutral_signal'] = False
        signals.loc[emv_crossover_zero, 'trend'] = 1
        signals.loc[emv_crossover_zero, 'signal_type'] = 'EMV上穿0轴'
        signals.loc[emv_crossover_zero, 'signal_desc'] = 'EMV上穿0轴，表明价格上涨动能增强'
        signals.loc[emv_crossover_zero, 'confidence'] = 65.0
        signals.loc[emv_crossover_zero, 'position_size'] = 0.3
        signals.loc[emv_crossover_zero, 'risk_level'] = '中'
        
        # 2. EMV下穿0轴，卖出信号
        emv_crossunder_zero = crossunder(emv, 0)
        signals.loc[emv_crossunder_zero, 'sell_signal'] = True
        signals.loc[emv_crossunder_zero, 'neutral_signal'] = False
        signals.loc[emv_crossunder_zero, 'trend'] = -1
        signals.loc[emv_crossunder_zero, 'signal_type'] = 'EMV下穿0轴'
        signals.loc[emv_crossunder_zero, 'signal_desc'] = 'EMV下穿0轴，表明价格下跌动能增强'
        signals.loc[emv_crossunder_zero, 'confidence'] = 65.0
        signals.loc[emv_crossunder_zero, 'position_size'] = 0.3
        signals.loc[emv_crossunder_zero, 'risk_level'] = '中'
        
        # 3. EMV上穿EMV_MA，买入信号
        emv_crossover_ma = crossover(emv, emv_ma)
        signals.loc[emv_crossover_ma, 'buy_signal'] = True
        signals.loc[emv_crossover_ma, 'neutral_signal'] = False
        signals.loc[emv_crossover_ma, 'trend'] = 1
        signals.loc[emv_crossover_ma, 'signal_type'] = 'EMV金叉'
        signals.loc[emv_crossover_ma, 'signal_desc'] = 'EMV上穿其均线，表明短期动能增强'
        signals.loc[emv_crossover_ma, 'confidence'] = 60.0
        signals.loc[emv_crossover_ma, 'position_size'] = 0.2
        signals.loc[emv_crossover_ma, 'risk_level'] = '中'
        
        # 4. EMV下穿EMV_MA，卖出信号
        emv_crossunder_ma = crossunder(emv, emv_ma)
        signals.loc[emv_crossunder_ma, 'sell_signal'] = True
        signals.loc[emv_crossunder_ma, 'neutral_signal'] = False
        signals.loc[emv_crossunder_ma, 'trend'] = -1
        signals.loc[emv_crossunder_ma, 'signal_type'] = 'EMV死叉'
        signals.loc[emv_crossunder_ma, 'signal_desc'] = 'EMV下穿其均线，表明短期动能减弱'
        signals.loc[emv_crossunder_ma, 'confidence'] = 60.0
        signals.loc[emv_crossunder_ma, 'position_size'] = 0.2
        signals.loc[emv_crossunder_ma, 'risk_level'] = '中'
        
        # 5. EMV处于高位/低位
        high_emv = emv > emv.rolling(window=50).max() * 0.8
        signals.loc[high_emv, 'buy_signal'] = True
        signals.loc[high_emv, 'neutral_signal'] = False
        signals.loc[high_emv, 'trend'] = 1
        signals.loc[high_emv, 'signal_type'] = 'EMV高位'
        signals.loc[high_emv, 'signal_desc'] = 'EMV处于近期高位，表明价格上涨容易'
        signals.loc[high_emv, 'confidence'] = 70.0
        signals.loc[high_emv, 'position_size'] = 0.4
        signals.loc[high_emv, 'risk_level'] = '低'
        
        low_emv = emv < emv.rolling(window=50).min() * 1.2
        signals.loc[low_emv, 'sell_signal'] = True
        signals.loc[low_emv, 'neutral_signal'] = False
        signals.loc[low_emv, 'trend'] = -1
        signals.loc[low_emv, 'signal_type'] = 'EMV低位'
        signals.loc[low_emv, 'signal_desc'] = 'EMV处于近期低位，表明价格下跌容易'
        signals.loc[low_emv, 'confidence'] = 70.0
        signals.loc[low_emv, 'position_size'] = 0.4
        signals.loc[low_emv, 'risk_level'] = '低'
        
        # 6. EMV快速变化
        emv_change_rate = emv.pct_change(5)
        rapid_increase = emv_change_rate > 0.2
        signals.loc[rapid_increase, 'buy_signal'] = True
        signals.loc[rapid_increase, 'neutral_signal'] = False
        signals.loc[rapid_increase, 'trend'] = 1
        signals.loc[rapid_increase, 'signal_type'] = 'EMV快速上升'
        signals.loc[rapid_increase, 'signal_desc'] = 'EMV快速上升，表明价格上涨动能突然增强'
        signals.loc[rapid_increase, 'confidence'] = 75.0
        signals.loc[rapid_increase, 'position_size'] = 0.4
        signals.loc[rapid_increase, 'risk_level'] = '中'
        
        rapid_decrease = emv_change_rate < -0.2
        signals.loc[rapid_decrease, 'sell_signal'] = True
        signals.loc[rapid_decrease, 'neutral_signal'] = False
        signals.loc[rapid_decrease, 'trend'] = -1
        signals.loc[rapid_decrease, 'signal_type'] = 'EMV快速下降'
        signals.loc[rapid_decrease, 'signal_desc'] = 'EMV快速下降，表明价格下跌动能突然增强'
        signals.loc[rapid_decrease, 'confidence'] = 75.0
        signals.loc[rapid_decrease, 'position_size'] = 0.4
        signals.loc[rapid_decrease, 'risk_level'] = '中'
        
        # 7. 计算EMV评分
        for i in range(len(signals)):
            if i > 0:  # 跳过第一个数据点
                emv_val = emv.iloc[i] if i < len(emv) else 0
                emv_ma_val = emv_ma.iloc[i] if i < len(emv_ma) else 0
                
                # 基础分数是50
                score = 50.0
                
                # EMV值为正加分，为负减分
                if emv_val > 0:
                    score += min(20, emv_val * 100)  # 最多加20分
                else:
                    score -= min(20, abs(emv_val) * 100)  # 最多减20分
                
                # EMV相对于均线的位置
                if emv_val > emv_ma_val:
                    score += 10  # EMV在均线上方加10分
                else:
                    score -= 10  # EMV在均线下方减10分
                
                # EMV的变化趋势
                if i > 1 and emv.iloc[i] > emv.iloc[i-1]:
                    score += 5  # EMV上升加5分
                elif i > 1 and emv.iloc[i] < emv.iloc[i-1]:
                    score -= 5  # EMV下降减5分
                
                # 限制评分范围在0-100之间
                signals.iloc[i, signals.columns.get_loc('score')] = max(0, min(100, score))
        
        # 设置止损价格
        if 'low' in data.columns and 'high' in data.columns:
            # 买入信号的止损设为最近的低点
            buy_indices = signals[signals['buy_signal']].index
            if not buy_indices.empty:
                for idx in buy_indices:
                    if idx in data.index and idx > data.index[10]:  # 确保有足够的历史数据
                        pos = data.index.get_loc(idx)
                        if pos >= 10:
                            lookback = 5
                            recent_low = data.iloc[pos-lookback:pos]['low'].min()
                            atr_val = atr_values.iloc[pos] if pos < len(atr_values) else 0
                            signals.loc[idx, 'stop_loss'] = recent_low - atr_val
        
            # 卖出信号的止损设为最近的高点
            sell_indices = signals[signals['sell_signal']].index
            if not sell_indices.empty:
                for idx in sell_indices:
                    if idx in data.index and idx > data.index[10]:  # 确保有足够的历史数据
                        pos = data.index.get_loc(idx)
                        if pos >= 10:
                            lookback = 5
                            recent_high = data.iloc[pos-lookback:pos]['high'].max()
                            atr_val = atr_values.iloc[pos] if pos < len(atr_values) else 0
                            signals.loc[idx, 'stop_loss'] = recent_high + atr_val
        
        # 根据EMV值判断市场环境
        signals['market_env'] = '中性'  # 默认中性市场
        
        # EMV持续为正，上升趋势市场
        positive_emv = emv.rolling(window=10).mean() > 0
        signals.loc[positive_emv, 'market_env'] = '强势'
        
        # EMV持续为负，下降趋势市场
        negative_emv = emv.rolling(window=10).mean() < 0
        signals.loc[negative_emv, 'market_env'] = '弱势'
        
        # EMV频繁在0轴附近波动，震荡市场
        emv_around_zero = abs(emv) < emv.std() * 0.5
        signals.loc[emv_around_zero, 'market_env'] = '震荡'
        
        # 设置成交量确认
        if 'volume' in data.columns:
            # 如果有成交量数据，检查成交量是否支持当前信号
            vol = data['volume']
            vol_avg = vol.rolling(window=20).mean()
            
            # 成交量大于20日均量1.5倍为放量
            vol_increase = vol > vol_avg * 1.5
            
            # 买入信号且成交量放大，确认信号
            signals.loc[signals['buy_signal'] & vol_increase, 'volume_confirmation'] = True
            
            # 卖出信号且成交量放大，确认信号
            signals.loc[signals['sell_signal'] & vol_increase, 'volume_confirmation'] = True
        
        return signals
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算EMV原始评分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 原始评分序列（0-100分）
        """
        # 确保已计算EMV指标
        if not self.has_result():
            self.calculate(data)
        
        if self._result is None:
            return pd.Series(50.0, index=data.index)
        
        # 获取EMV相关值
        emv = self._result['EMV']
        emv_ma = self._result['EMV_MA']
        
        # 初始基础分50分
        score = pd.Series(50.0, index=data.index)
        
        # 基于EMV值的评分
        for i in range(len(score)):
            if i >= len(emv):
                continue
                
            emv_val = emv.iloc[i]
            emv_ma_val = emv_ma.iloc[i] if i < len(emv_ma) else 0
            
            # EMV值为正加分，为负减分
            if emv_val > 0:
                score.iloc[i] += min(20, emv_val * 100)  # 最多加20分
            else:
                score.iloc[i] -= min(20, abs(emv_val) * 100)  # 最多减20分
            
            # EMV相对于均线的位置
            if emv_val > emv_ma_val:
                score.iloc[i] += 10  # EMV在均线上方加10分
            else:
                score.iloc[i] -= 10  # EMV在均线下方减10分
            
            # EMV的变化趋势
            if i > 0:
                if emv.iloc[i] > emv.iloc[i-1]:
                    score.iloc[i] += 5  # EMV上升加5分
                else:
                    score.iloc[i] -= 5  # EMV下降减5分
        
        # 限制评分范围在0-100之间
        return np.clip(score, 0, 100)

    def calculate_confidence(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """
        计算EMV指标的置信度

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
        if last_score > 75 or last_score < 25:
            confidence += 0.25
        # 中性评分置信度中等
        elif 40 <= last_score <= 60:
            confidence += 0.1
        else:
            confidence += 0.15

        # 2. 基于形态的置信度
        if isinstance(patterns, pd.DataFrame) and not patterns.empty:
            try:
                # 统计最近几个周期的形态数量
                numeric_cols = patterns.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    recent_data = patterns[numeric_cols].iloc[-5:] if len(patterns) >= 5 else patterns[numeric_cols]
                    recent_patterns = recent_data.sum().sum()
                    if recent_patterns > 0:
                        confidence += min(recent_patterns * 0.05, 0.2)
            except:
                pass

        # 3. 基于EMV值的稳定性
        if hasattr(self, '_result') and self._result is not None and 'EMV' in self._result.columns:
            try:
                emv_values = self._result['EMV'].dropna()
                if len(emv_values) >= 5:
                    recent_emv = emv_values.iloc[-5:]
                    emv_stability = 1.0 - (recent_emv.std() / (abs(recent_emv.mean()) + 0.001))
                    confidence += min(emv_stability * 0.1, 0.15)
            except:
                pass

        # 4. 基于评分稳定性的置信度
        if len(score) >= 5:
            recent_scores = score.iloc[-5:]
            score_stability = 1.0 - (recent_scores.std() / 50.0)
            confidence += score_stability * 0.1

        return min(confidence, 1.0)

    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别EMV技术形态
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[str]: 识别出的形态列表
        """
        patterns = []
        
        # 确保已计算EMV指标
        if not self.has_result():
            self.calculate(data)
        
        if self._result is None:
            return patterns
        
        # 获取EMV相关值
        emv = self._result['EMV']
        emv_ma = self._result['EMV_MA']
        
        # 最后一个有效的索引位置
        last_valid_idx = -1
        while last_valid_idx >= -len(emv) and pd.isna(emv.iloc[last_valid_idx]):
            last_valid_idx -= 1
        
        if last_valid_idx < -len(emv):
            return patterns
        
        # 1. EMV交叉形态
        if last_valid_idx-1 >= -len(emv) and emv.iloc[last_valid_idx-1] < emv_ma.iloc[last_valid_idx-1] and emv.iloc[last_valid_idx] > emv_ma.iloc[last_valid_idx]:
            patterns.append("EMV金叉（EMV上穿均线）")
        
        if last_valid_idx-1 >= -len(emv) and emv.iloc[last_valid_idx-1] > emv_ma.iloc[last_valid_idx-1] and emv.iloc[last_valid_idx] < emv_ma.iloc[last_valid_idx]:
            patterns.append("EMV死叉（EMV下穿均线）")
        
        # 2. EMV零轴交叉
        if last_valid_idx-1 >= -len(emv) and emv.iloc[last_valid_idx-1] < 0 and emv.iloc[last_valid_idx] > 0:
            patterns.append("EMV上穿零轴（由负转正）")
        
        if last_valid_idx-1 >= -len(emv) and emv.iloc[last_valid_idx-1] > 0 and emv.iloc[last_valid_idx] < 0:
            patterns.append("EMV下穿零轴（由正转负）")
        
        # 3. EMV位置形态
        emv_val = emv.iloc[last_valid_idx]
        
        if emv_val > 0:
            patterns.append("EMV为正值（价格上涨容易）")
            
            # 计算50日内的最大EMV值
            if len(emv) >= 50:
                max_emv = emv.iloc[max(-50, -len(emv)):].max()
                if emv_val > max_emv * 0.8:
                    patterns.append("EMV处于高位（强烈上涨动能）")
        else:
            patterns.append("EMV为负值（价格下跌容易）")
            
            # 计算50日内的最小EMV值
            if len(emv) >= 50:
                min_emv = emv.iloc[max(-50, -len(emv)):].min()
                if emv_val < min_emv * 0.8:
                    patterns.append("EMV处于低位（强烈下跌动能）")
        
        # 4. EMV变化形态
        if last_valid_idx-5 >= -len(emv):
            emv_change = (emv.iloc[last_valid_idx] - emv.iloc[last_valid_idx-5]) / abs(emv.iloc[last_valid_idx-5]) if emv.iloc[last_valid_idx-5] != 0 else 0
            
            if emv_change > 0.2:
                patterns.append("EMV快速上升（动能迅速增强）")
            elif emv_change < -0.2:
                patterns.append("EMV快速下降（动能迅速减弱）")
        
        return patterns

    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        获取EMV指标的技术形态

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 包含形态信息的DataFrame
        """
        # 确保已计算EMV
        if not self.has_result():
            self.calculate(data, **kwargs)

        if self._result is None:
            return pd.DataFrame(index=data.index)

        emv = self._result['EMV']
        emv_ma = self._result['EMV_MA']

        patterns_df = pd.DataFrame(index=data.index)

        # 1. 零轴穿越形态
        patterns_df['EMV_CROSS_UP_ZERO'] = crossover(emv, 0)
        patterns_df['EMV_CROSS_DOWN_ZERO'] = crossunder(emv, 0)
        patterns_df['EMV_ABOVE_ZERO'] = emv > 0
        patterns_df['EMV_BELOW_ZERO'] = emv < 0

        # 2. EMV与其移动平均线的关系
        patterns_df['EMV_ABOVE_MA'] = emv > emv_ma
        patterns_df['EMV_BELOW_MA'] = emv < emv_ma
        patterns_df['EMV_CROSS_UP_MA'] = crossover(emv, emv_ma)
        patterns_df['EMV_CROSS_DOWN_MA'] = crossunder(emv, emv_ma)

        # 3. 趋势形态
        patterns_df['EMV_RISING'] = emv > emv.shift(1)
        patterns_df['EMV_FALLING'] = emv < emv.shift(1)

        # 4. 强度形态
        if len(emv) >= 5:
            emv_change = emv.diff(3)
            emv_std = emv.rolling(20).std()

            patterns_df['EMV_STRONG_RISE'] = emv_change > emv_std
            patterns_df['EMV_STRONG_FALL'] = emv_change < -emv_std

        # 5. 极值形态
        if len(emv) >= 20:
            emv_rolling_max = emv.rolling(20).max()
            emv_rolling_min = emv.rolling(20).min()

            patterns_df['EMV_HIGH_EXTREME'] = emv >= emv_rolling_max * 0.9
            patterns_df['EMV_LOW_EXTREME'] = emv <= emv_rolling_min * 0.9

        # 6. 背离形态（简化版）
        if len(emv) >= 10 and 'high' in data.columns:
            price_trend = data['high'].rolling(5).mean().diff(5)
            emv_trend = emv.diff(5)

            patterns_df['EMV_BULLISH_DIVERGENCE'] = (price_trend < 0) & (emv_trend > 0)
            patterns_df['EMV_BEARISH_DIVERGENCE'] = (price_trend > 0) & (emv_trend < 0)

        return patterns_df

    def register_patterns(self):
        """
        注册EMV指标的技术形态
        """
        # 注册EMV零轴穿越形态
        self.register_pattern_to_registry(
            pattern_id="EMV_CROSS_UP_ZERO",
            display_name="EMV上穿零轴",
            description="EMV从负值区域穿越零轴，表示买盘力量增强",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=25.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="EMV_CROSS_DOWN_ZERO",
            display_name="EMV下穿零轴",
            description="EMV从正值区域穿越零轴，表示卖盘力量增强",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-25.0,
            polarity="NEGATIVE"
        )

        # 注册EMV与移动平均线交叉形态
        self.register_pattern_to_registry(
            pattern_id="EMV_CROSS_UP_MA",
            display_name="EMV上穿均线",
            description="EMV上穿其移动平均线，趋势转强",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=20.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="EMV_CROSS_DOWN_MA",
            display_name="EMV下穿均线",
            description="EMV下穿其移动平均线，趋势转弱",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-20.0,
            polarity="NEGATIVE"
        )

        # 注册EMV强度形态
        self.register_pattern_to_registry(
            pattern_id="EMV_STRONG_RISE",
            display_name="EMV强势上升",
            description="EMV大幅上升，买盘力量强劲",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=18.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="EMV_STRONG_FALL",
            display_name="EMV强势下降",
            description="EMV大幅下降，卖盘力量强劲",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-18.0,
            polarity="NEGATIVE"
        )

        # 注册EMV背离形态
        self.register_pattern_to_registry(
            pattern_id="EMV_BULLISH_DIVERGENCE",
            display_name="EMV底背离",
            description="价格下跌但EMV上升，可能反转向上",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=30.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="EMV_BEARISH_DIVERGENCE",
            display_name="EMV顶背离",
            description="价格上涨但EMV下降，可能反转向下",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-30.0,
            polarity="NEGATIVE"
        )

    def get_pattern_info(self, pattern_id: str) -> dict:
        """
        获取指定形态的详细信息

        Args:
            pattern_id: 形态ID

        Returns:
            dict: 形态信息字典
        """
        pattern_info_map = {
            'EMV_CROSS_UP_ZERO': {
                'name': 'EMV上穿零轴',
                'description': 'EMV从负值区域穿越零轴，表示买盘力量增强',
                'strength': 'strong',
                'type': 'bullish'
            },
            'EMV_CROSS_DOWN_ZERO': {
                'name': 'EMV下穿零轴',
                'description': 'EMV从正值区域穿越零轴，表示卖盘力量增强',
                'strength': 'strong',
                'type': 'bearish'
            },
            'EMV_ABOVE_ZERO': {
                'name': 'EMV零轴上方',
                'description': 'EMV位于零轴上方，买盘力量占优',
                'strength': 'medium',
                'type': 'bullish'
            },
            'EMV_BELOW_ZERO': {
                'name': 'EMV零轴下方',
                'description': 'EMV位于零轴下方，卖盘力量占优',
                'strength': 'medium',
                'type': 'bearish'
            },
            'EMV_CROSS_UP_MA': {
                'name': 'EMV上穿均线',
                'description': 'EMV上穿其移动平均线，趋势转强',
                'strength': 'medium',
                'type': 'bullish'
            },
            'EMV_CROSS_DOWN_MA': {
                'name': 'EMV下穿均线',
                'description': 'EMV下穿其移动平均线，趋势转弱',
                'strength': 'medium',
                'type': 'bearish'
            },
            'EMV_ABOVE_MA': {
                'name': 'EMV均线上方',
                'description': 'EMV位于移动平均线上方',
                'strength': 'weak',
                'type': 'bullish'
            },
            'EMV_BELOW_MA': {
                'name': 'EMV均线下方',
                'description': 'EMV位于移动平均线下方',
                'strength': 'weak',
                'type': 'bearish'
            },
            'EMV_STRONG_RISE': {
                'name': 'EMV强势上升',
                'description': 'EMV大幅上升，买盘力量强劲',
                'strength': 'medium',
                'type': 'bullish'
            },
            'EMV_STRONG_FALL': {
                'name': 'EMV强势下降',
                'description': 'EMV大幅下降，卖盘力量强劲',
                'strength': 'medium',
                'type': 'bearish'
            },
            'EMV_BULLISH_DIVERGENCE': {
                'name': 'EMV底背离',
                'description': '价格下跌但EMV上升，可能反转向上',
                'strength': 'strong',
                'type': 'bullish'
            },
            'EMV_BEARISH_DIVERGENCE': {
                'name': 'EMV顶背离',
                'description': '价格上涨但EMV下降，可能反转向下',
                'strength': 'strong',
                'type': 'bearish'
            },
            'EMV_RISING': {
                'name': 'EMV上升',
                'description': 'EMV值上升',
                'strength': 'weak',
                'type': 'bullish'
            },
            'EMV_FALLING': {
                'name': 'EMV下降',
                'description': 'EMV值下降',
                'strength': 'weak',
                'type': 'bearish'
            },
            'EMV_HIGH_EXTREME': {
                'name': 'EMV极高值',
                'description': 'EMV达到近期高点',
                'strength': 'medium',
                'type': 'neutral'
            },
            'EMV_LOW_EXTREME': {
                'name': 'EMV极低值',
                'description': 'EMV达到近期低点',
                'strength': 'medium',
                'type': 'neutral'
            }
        }

        return pattern_info_map.get(pattern_id, {
            'name': pattern_id,
            'description': f'EMV形态: {pattern_id}',
            'strength': 'medium',
            'type': 'neutral'
        })

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算并生成EMV指标信号
        
        Args:
            df: 包含OHLCV数据的DataFrame
            
        Returns:
            pd.DataFrame: 包含EMV指标和信号的DataFrame
        """
        result = self.calculate(df)
        signal_df = self.generate_signals(df, result)
        
        return signal_df

