#!/usr/bin/env python3
"""
ATR_Atr (Average True Range) 平均真实波幅指标

ATR是衡量价格波动性的技术指标，由J. Welles Wilder开发。
它计算一定周期内的平均真实波幅，用于衡量市场的波动性。
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from utils.logger import getLogger

logger = getLogger(__name__)


class AtrAtr(BaseIndicator, PatternSignalMixin):
    """
    ATR_Atr (Average True Range) 平均真实波幅指标
    
    ATR指标用于衡量价格波动性，通过计算真实波幅的移动平均值来反映市场的波动程度。
    ATR值越高，表示价格波动越大；ATR值越低，表示价格波动越小。
    """
    
    def __init__(self, **kwargs):
        """
        初始化ATR指标
        
        Args:
            **kwargs: 指标参数
        """
        super().__init__()
        self.name = "ATR_Atr"
        
        # 设置默认参数
        self._default_parameters = self._get_default_parameters_atr()
        
        # 应用用户参数
        self.set_parameters_Atr(**kwargs)
    
    def _get_default_parameters_atr(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {"period": 14}
    
    def set_parameters_Atr(self, **kwargs):
        """
        设置指标参数
        
        Args:
            **kwargs: 参数字典
        """
        # 验证参数
        from utils.indicator_parameter_validator import IndicatorParameterValidator
        validator = IndicatorParameterValidator()
        
        # 合并默认参数和用户参数
        params = self._default_parameters.copy()
        params.update(kwargs)        # 验证参数
        try:
            from utils.indicator_parameter_validator import IndicatorParameterValidator
            validator = IndicatorParameterValidator()
            
            # 验证参数
            is_valid, errors = validator.validate_indicator_parameters('ATR_Atr', params)
            if not is_valid:
                # 静默处理验证失败，避免过多警告
                pass
                
        except Exception:
            # 如果验证失败，静默处理，保持向后兼容
            pass
        
        # 设置参数
        self.period = params.get('period', 14)
    
    def calculate_Atr(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算ATR指标

        Args:
            data: 包含OHLCV数据的Data_frame

        Returns:
            添加了ATR指标的Data_frame
        """
        result = self._calculate_atr(data, **kwargs)
        self._result = result
        return result

    def _calculate_atr(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        内部计算ATR指标

        Args:
            data: 包含OHLCV数据的Data_frame

        Returns:
            添加了ATR指标的Data_frame
        """
        df = data.copy()

        # 确保数据有足够的长度
        if len(df) < self.period + 1:
            logger.warning(f"数据长度({len(df)})小于所需的回溯周期({self.period + 1})，返回原始数据")
            df[f'ATR_Atr{self.period}'] = np.nan
            return df
            
        # 计算真实波幅(TR)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['close'].shift(1))
        df['tr3'] = abs(df['low'] - df['close'].shift(1))
        df['TR'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)

        # 计算ATR - TR的period周期平均值
        df[f'ATR_Atr{self.period}'] = df['TR'].rolling(window=self.period).mean()

        # 清理中间计算列
        df.drop(['tr1', 'tr2', 'tr3', 'TR'], axis=1, inplace=True)

        # 添加形态识别和信号生成
        df = self.add_pattern_detection(df)
        df = self.add_signal_generation(df)

        # 重写信号生成逻辑（ATR指标特定逻辑）
        df = self._apply_atr_signal_logic(df)

        return df

    def _apply_atr_signal_logic(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用ATR指标特定的信号生成逻辑
        基于ATR值的变化生成信号
        """
        try:
            # 获取ATR值
            atr_col = f'ATR_Atr{self.period}'
            if atr_col not in df.columns:
                # 如果没有ATR值，使用默认信号
                return df

            atr_value = df[atr_col]

            # ATR信号生成逻辑：
            # BUY: ATR值上升（波动性增加，可能有突破）
            # SELL: ATR值下降（波动性减少，可能趋势结束）
            # HOLD: ATR值稳定

            # 计算ATR变化率
            atr_change = atr_value.pct_change()
            atr_rising = atr_change > 0.05  # ATR上升超过5%
            atr_falling = atr_change < -0.05  # ATR下降超过5%

            # 生成信号
            df.loc[:, 'buy_signal'] = atr_rising
            df.loc[:, 'sell_signal'] = atr_falling
            df.loc[:, 'hold_signal'] = ~(atr_rising | atr_falling)

            # 确保信号类型为布尔值
            df['buy_signal'] = df['buy_signal'].astype(bool)
            df['sell_signal'] = df['sell_signal'].astype(bool)
            df['hold_signal'] = df['hold_signal'].astype(bool)

        except Exception as e:
            logger.warning(f"ATR信号生成失败: {e}")
            # 如果出错，使用默认信号
            df.loc[:, 'buy_signal'] = False
            df.loc[:, 'sell_signal'] = False
            df.loc[:, 'hold_signal'] = True

        return df

    def calculate_raw_score_Atr(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算ATR指标的原始评分（0-100分制）
        
        ATR评分逻辑：
        - ATR是波动性指标，高ATR表示高波动性（风险高但机会大）
        - 基于ATR相对于历史水平的位置进行评分
        - ATR上升趋势表示波动性增加，可能有突破机会
        - ATR下降趋势表示波动性减少，可能趋势稳定
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 原始评分序列，取值范围0-100
        """
        if not self.has_result():
            self.calculate_Atr(data, **kwargs)
        
        # 获取ATR指标值
        atr_col = f'ATR_Atr{self.period}'
        if self._result is None or atr_col not in self._result.columns:
            return pd.Series(50.0, index=data.index)

        atr = self._result[atr_col]
        
        # 计算ATR的相对位置（基于历史百分位数）
        atr_rolling_window = min(50, len(atr))  # 使用50周期或数据长度
        
        # 基础评分计算
        # 1. 相对水平分：基于ATR相对历史水平的位置，贡献50分权重
        position_score = pd.Series(50.0, index=data.index)
        
        if atr_rolling_window >= 20:  # 确保有足够数据计算百分位数
            # 计算ATR的历史百分位数
            atr_percentile = atr.rolling(window=atr_rolling_window).apply(
                lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) * 100 if x.max() > x.min() else 50
            )
            
            # 基于百分位数调整评分
            # 低波动性（0-30百分位）：稳定但机会少，得分45-55
            # 中等波动性（30-70百分位）：平衡，得分50
            # 高波动性（70-100百分位）：机会多但风险高，得分55-65
            position_score = 50 + (atr_percentile - 50) * 0.3
        
        # 2. 趋势分：基于ATR变化趋势，贡献30分权重
        atr_change = atr - atr.shift(5)  # 5周期变化
        atr_change_pct = atr_change / atr.shift(5) * 100  # 变化百分比
        
        trend_score = pd.Series(50.0, index=data.index)
        
        # ATR上升趋势：波动性增加，可能有突破机会，适度加分
        # ATR下降趋势：波动性减少，趋势可能稳定，适度减分
        trend_score += np.clip(atr_change_pct * 0.5, -15, 15)
        
        # 3. 价格突破潜力分：基于ATR与价格变化的关系，贡献20分权重
        breakthrough_score = pd.Series(50.0, index=data.index)
        
        if 'close' in self._result.columns:
            close = self._result['close']
            price_change = close - close.shift(3)  # 3周期价格变化
            price_change_pct = price_change / close.shift(3) * 100
            
            # 当价格变化幅度接近ATR时，表示可能有突破
            # ATR相对于价格的比例
            atr_price_ratio = atr / close * 100
            
            # 高ATR比例且价格变化较大时加分
            high_volatility_move = (atr_price_ratio > 2) & (abs(price_change_pct) > 1)
            breakthrough_score[high_volatility_move] += 10
            
            # 低ATR比例且价格变化较小时减分（盘整状态）
            low_volatility_move = (atr_price_ratio < 1) & (abs(price_change_pct) < 0.5)
            breakthrough_score[low_volatility_move] -= 10
        
        # 4. 综合评分（相对水平分50% + 趋势分30% + 突破潜力分20%）
        final_score = position_score * 0.5 + trend_score * 0.3 + breakthrough_score * 0.2
        
        # 限制评分在0-100之间
        return final_score.clip(0, 100)

    def calculate_confidence_Atr(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """计算置信度"""
        return 0.5

    def get_patterns_Atr(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """获取形态"""
        return pd.DataFrame(index=data.index)
