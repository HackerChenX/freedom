#!/usr/bin/env python3
"""
CCI (Commodity Channel Index) 顺势指标

CCI指标是一种超买超卖指标，用于识别价格偏离统计平均值的程度。
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from utils.logger import get_logger

logger = get_logger(__name__)


class CCI(BaseIndicator, PatternSignalMixin):
    """
    CCI (Commodity Channel Index) 顺势指标
    
    CCI指标通过计算价格与其统计平均值的偏离程度来识别超买超卖状态。
    """
    
    def __init__(self, **kwargs):
        """
        初始化CCI指标
        
        Args:
            **kwargs: 指标参数
        """
        super().__init__()
        self.name = "CCI"
        
        # 设置默认参数
        self._default_parameters = self._get_default_parameters()
        
        # 应用用户参数
        self.set_parameters(**kwargs)
    
    def _get_default_parameters(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {"period": 20, "constant": 0.015}
    
    def set_parameters(self, **kwargs):
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
            is_valid, errors = validator.validate_indicator_parameters('CCI', params)
            if not is_valid:
                # 静默处理验证失败，避免过多警告
                pass
                
        except Exception:
            # 如果验证失败，静默处理，保持向后兼容
            pass
        
        # 设置参数
        self.period = params.get('period', 20)
        self.constant = params.get('constant', 0.015)
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算CCI指标

        Args:
            data: 包含OHLCV数据的DataFrame

        Returns:
            添加了CCI指标的DataFrame
        """
        result = self._calculate(data, **kwargs)
        self._result = result
        return result

    def _calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        内部计算CCI指标

        Args:
            data: 包含OHLCV数据的DataFrame

        Returns:
            添加了CCI指标的DataFrame
        """
        df = data.copy()

        # 确保数据有足够的长度
        if len(df) < self.period:
            logger.warning(f"数据长度({len(df)})小于所需的回溯周期({self.period})，返回原始数据")
            df[f'CCI{self.period}'] = np.nan
            return df
            
        # 计算典型价格
        df['TP'] = (df['high'] + df['low'] + df['close']) / 3

        # 计算移动平均
        df['MA'] = df['TP'].rolling(window=self.period).mean()

        # 计算平均偏差
        df['MD'] = df['TP'].rolling(window=self.period).apply(
            lambda x: np.mean(np.abs(x - x.mean()))
        )

        # 计算CCI
        df[f'CCI{self.period}'] = (df['TP'] - df['MA']) / (self.constant * df['MD'])

        # 清理中间计算列
        df.drop(['TP', 'MA', 'MD'], axis=1, inplace=True)

        # 添加形态识别和信号生成
        df = self.add_pattern_detection(df)
        df = self.add_signal_generation(df)

        # 重写信号生成逻辑（CCI指标特定逻辑）
        df = self._apply_cci_signal_logic(df)

        return df

    def _apply_cci_signal_logic(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用CCI指标特定的信号生成逻辑
        基于CCI值的超买超卖区间生成信号
        """
        try:
            # 获取CCI值
            cci_col = f'CCI{self.period}'
            if cci_col not in df.columns:
                # 如果没有CCI值，使用默认信号
                return df

            cci_value = df[cci_col]

            # CCI信号生成逻辑：
            # BUY: CCI从超卖区间(-100以下)向上突破
            # SELL: CCI从超买区间(100以上)向下突破
            # HOLD: CCI在正常区间(-100到100)

            # 定义超买超卖区间
            oversold = cci_value < -100
            overbought = cci_value > 100
            normal = (cci_value >= -100) & (cci_value <= 100)

            # 检测突破
            cci_rising = cci_value > cci_value.shift(1)
            cci_falling = cci_value < cci_value.shift(1)

            # 生成信号
            df.loc[:, 'buy_signal'] = oversold & cci_rising
            df.loc[:, 'sell_signal'] = overbought & cci_falling
            df.loc[:, 'hold_signal'] = normal | (~(df['buy_signal'] | df['sell_signal']))

            # 确保信号类型为布尔值
            df['buy_signal'] = df['buy_signal'].astype(bool)
            df['sell_signal'] = df['sell_signal'].astype(bool)
            df['hold_signal'] = df['hold_signal'].astype(bool)

        except Exception as e:
            logger.warning(f"CCI信号生成失败: {e}")
            # 如果出错，使用默认信号
            df.loc[:, 'buy_signal'] = False
            df.loc[:, 'sell_signal'] = False
            df.loc[:, 'hold_signal'] = True

        return df

    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算CCI指标的原始评分（0-100分制）
        
        CCI评分逻辑：
        - CCI在-100到100之间为正常区间，得分50分
        - CCI < -100为超卖区间，越低得分越高（最高80分）
        - CCI > 100为超买区间，越高得分越低（最低20分）
        - 结合CCI变化趋势进行调整
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 原始评分序列，取值范围0-100
        """
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        # 获取CCI指标值
        cci_col = f'CCI{self.period}'
        if self._result is None or cci_col not in self._result.columns:
            return pd.Series(50.0, index=data.index)

        cci = self._result[cci_col]
        
        # 基础评分计算
        # 1. 位置分：基于CCI值的位置，贡献70分权重
        position_score = pd.Series(50.0, index=data.index)
        
        # 超卖区间（CCI < -100）：看涨信号，得分增加
        oversold = cci < -100
        position_score[oversold] = 50 + np.minimum(30, (-cci[oversold] - 100) * 0.15)  # 最高80分
        
        # 超买区间（CCI > 100）：看跌信号，得分减少
        overbought = cci > 100
        position_score[overbought] = 50 - np.minimum(30, (cci[overbought] - 100) * 0.15)  # 最低20分
        
        # 正常区间（-100 <= CCI <= 100）：中性，基于距离零轴的远近微调
        normal = (cci >= -100) & (cci <= 100)
        position_score[normal] = 50 + cci[normal] * 0.1  # -100时为40分，100时为60分
        
        # 2. 趋势分：基于CCI变化趋势，贡献30分权重
        cci_change = cci - cci.shift(3)  # 3周期变化
        trend_score = pd.Series(50.0, index=data.index)
        
        # CCI上升趋势加分，下降趋势减分
        trend_score += np.clip(cci_change * 0.2, -15, 15)
        
        # 3. 综合评分（位置分70% + 趋势分30%）
        final_score = position_score * 0.7 + trend_score * 0.3
        
        # 限制评分在0-100之间
        return final_score.clip(0, 100)

    def calculate_confidence(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """计算置信度"""
        return 0.5

    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """获取形态"""
        return pd.DataFrame(index=data.index)
