#!/usr/bin/python
from utils.dependency_injection import get_logger
# -*- coding: UTF-8 -*-

"""
量比指标(VOLUME_RATIO)
量比是指当前成交量与前N个周期平均成交量的比值，用于衡量市场交易活跃度的变化。
"""

import numpy as np
import pandas as pd
from typing import Dict, Any

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from utils.dependency_injection import get_logger

logger = get_logger(__name__)


class VolumeRatio(BaseIndicator, PatternSignalMixin):
    """
    量比指标(VOLUME_RATIO)
    
    特点:
    1. 用于衡量市场交易活跃度的变化
    2. 量比>1表示当前成交量高于参考期平均值，市场相对活跃
    3. 量比<1表示当前成交量低于参考期平均值，市场相对冷清
    4. 通常与价格趋势结合使用，判断市场热度变化
    
    计算方法:
    量比 = 当前成交量 / 前N个周期平均成交量
    
    参数:
    - period: 参考周期，默认为14
    """
    
    def __init__(self, **kwargs):
        """
        初始化VOLUME_RATIO指标
        
        Args:
            **kwargs: 指标参数
        """
        super().__init__()
        self.name = "VOLUME_RATIO"
        
        # 设置默认参数
        self._default_parameters = self._get_default_parameters_volumeratio()
        
        # 应用用户参数
        self.set_parameters_Ratio(**kwargs)
    
    def _get_default_parameters_volumeratio(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {"period": 14}
    
    def set_parameters_Ratio(self, **kwargs):
        """
        设置指标参数
        
        Args:
            **kwargs: 参数字典
        """
        # 验证参数
        try:
            from utils.indicator_parameter_validator import IndicatorParameterValidator
            validator = IndicatorParameterValidator()
            
            # 合并默认参数和用户参数
            params = self._default_parameters.copy()
            params.update(kwargs)
            
            # 验证参数
            is_valid, errors = validator.validate_indicator_parameters('VOLUME_RATIO', params)
            if not is_valid:
                # 静默处理验证失败，避免过多警告
                pass
                
        except Exception:
            # 如果验证失败，静默处理，保持向后兼容
            pass
        
        # 设置参数
        self.period = kwargs.get('period', 14)
    
    def calculate_Ratio(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算VOLUME_RATIO指标
        
        Args:
            data: 包含OHLCV数据的Data_frame
            
        Returns:
            添加了VOLUME_RATIO指标的Data_frame
        """
        result = self._calculate_volumeratio(data, **kwargs)
        self._result = result
        return result
    
    def _calculate_volumeratio(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        内部计算VOLUME_RATIO指标
        
        Args:
            data: 包含OHLCV数据的Data_frame
            
        Returns:
            添加了VOLUME_RATIO指标的Data_frame
        """
        df = data.copy()
        
        # 扩展支持的成交量列名格式
        volume_columns = ['volume', 'Volume', 'VOLUME', 'vol', 'Vol', 'VOL', 'turnover', 'Turnover', 'TURNOVER']
        volume = None
        found_column = None
        
        # 按优先级查找成交量列
        for col in volume_columns:
            if col in df.columns:
                volume = df[col]
                found_column = col
                logger.debug(f"VOLUME_RATIO: 找到成交量列 '{col}'")
                break
        
        if volume is None:
            # 详细日志记录可用列
            available_columns = list(df.columns)
            logger.warning(f"VOLUME_RATIO: 未找到成交量列。可用列: {available_columns}")
            logger.warning(f"VOLUME_RATIO: 支持的成交量列名: {volume_columns}")
            
            # 尝试从列名中查找包含'volume'或'vol'的列
            potential_columns = [col for col in available_columns 
                               if any(vol_name.lower() in col.lower() 
                                     for vol_name in ['volume', 'vol', 'turnover'])]
            
            if potential_columns:
                volume = df[potential_columns[0]]
                found_column = potential_columns[0]
                logger.info(f"VOLUME_RATIO: 使用潜在成交量列 '{potential_columns[0]}'")
            else:
                # 如果没有成交量数据，返回默认值
                logger.warning("VOLUME_RATIO: 无成交量数据，使用默认值1.0")
                df['VOLUME_RATIO_VALUE'] = 1.0
                return df
        
        # 验证成交量数据
        if volume.isna().all():
            logger.warning(f"VOLUME_RATIO: 成交量列 '{found_column}' 全部为空值")
            df['VOLUME_RATIO_VALUE'] = 1.0
            return df
        
        # 计算量比
        try:
            volume_avg = volume.rolling(window=self.period).mean()
            volume_ratio = volume / volume_avg
            df['VOLUME_RATIO_VALUE'] = volume_ratio.fillna(1.0)
            
            logger.debug(f"VOLUME_RATIO: 计算完成，使用列 '{found_column}'，周期 {self.period}")
            
        except Exception as e:
            logger.error(f"VOLUME_RATIO: 计算量比失败: {e}")
            df['VOLUME_RATIO_VALUE'] = 1.0
            return df
        
        # 添加形态识别和信号生成
        df = self.add_pattern_detection(df)
        df = self.add_signal_generation(df)

        # 重写信号生成逻辑（VOLUME_RATIO指标特定逻辑）
        df = self._apply_volume_ratio_signal_logic(df)

        return df

    def _apply_volume_ratio_signal_logic(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用VOLUME_RATIO指标特定的信号生成逻辑
        基于量比值的大小生成信号
        """
        try:
            # 获取量比值
            if 'VOLUME_RATIO_VALUE' not in df.columns:
                # 如果没有量比值，使用默认信号
                return df

            volume_ratio = df['VOLUME_RATIO_VALUE']

            # VOLUME_RATIO信号生成逻辑：
            # BUY: 量比大于1.5（成交量放大）
            # SELL: 量比小于0.5（成交量萎缩）
            # HOLD: 量比在0.5-1.5之间（正常成交量）

            high_volume = volume_ratio > 1.5
            low_volume = volume_ratio < 0.5
            normal_volume = (volume_ratio >= 0.5) & (volume_ratio <= 1.5)

            # 生成信号
            df.loc[:, 'buy_signal'] = high_volume
            df.loc[:, 'sell_signal'] = low_volume
            df.loc[:, 'hold_signal'] = normal_volume

            # 确保信号类型为布尔值
            df['buy_signal'] = df['buy_signal'].astype(bool)
            df['sell_signal'] = df['sell_signal'].astype(bool)
            df['hold_signal'] = df['hold_signal'].astype(bool)

        except Exception as e:
            logger.warning(f"VOLUME_RATIO信号生成失败: {e}")
            # 如果出错，使用默认信号
            df.loc[:, 'buy_signal'] = False
            df.loc[:, 'sell_signal'] = False
            df.loc[:, 'hold_signal'] = True

        return df
    
    def calculate_raw_score_Ratio(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算量比指标的原始评分
        
        基于量比的活跃度和稳定性进行评分：
        1. 量比活跃度：量比偏离1的程度
        2. 量比稳定性：量比的波动程度
        3. 量比趋势：量比的变化趋势
        4. 量比分布：量比的分布特征
        """
        if not self.has_result():
            self.calculate_Ratio(data, **kwargs)
        
        if 'VOLUME_RATIO_VALUE' not in self._result.columns:
            return pd.Series(50.0, index=data.index)
        
        volume_ratio = self._result['VOLUME_RATIO_VALUE'].fillna(1.0)
        scores = pd.Series(index=data.index, dtype=float)
        
        for i in range(len(volume_ratio)):
            if i < self.period:
                scores.iloc[i] = 50.0
                continue
            
            # 获取当前窗口数据
            current_ratio = volume_ratio.iloc[i]
            window_ratios = volume_ratio.iloc[max(0, i-self.period+1):i+1]
            
            score = 50.0  # 基础分数
            
            # 1. 量比活跃度评分 (30分)
            # 量比越偏离1，市场越活跃
            activity_deviation = abs(current_ratio - 1.0)
            if activity_deviation >= 2.0:
                activity_score = 30.0  # 极度活跃
            elif activity_deviation >= 1.0:
                activity_score = 20.0 + (activity_deviation - 1.0) * 10.0  # 活跃
            elif activity_deviation >= 0.5:
                activity_score = 10.0 + (activity_deviation - 0.5) * 20.0  # 较活跃
            else:
                activity_score = activity_deviation * 20.0  # 平淡
            
            score += activity_score - 15.0  # 调整基准
            
            # 2. 量比稳定性评分 (20分)
            # 量比波动越小，市场越稳定
            if len(window_ratios) > 1:
                ratio_std = window_ratios.std()
                if ratio_std <= 0.2:
                    stability_score = 20.0  # 非常稳定
                elif ratio_std <= 0.5:
                    stability_score = 15.0 + (0.5 - ratio_std) / 0.3 * 5.0  # 稳定
                elif ratio_std <= 1.0:
                    stability_score = 10.0 + (1.0 - ratio_std) / 0.5 * 5.0  # 较稳定
                else:
                    stability_score = max(0, 10.0 - (ratio_std - 1.0) * 5.0)  # 不稳定
            else:
                stability_score = 10.0
            
            score += stability_score - 10.0  # 调整基准
            
            # 3. 量比趋势评分 (20分)
            # 量比上升趋势给予更高评分
            if len(window_ratios) >= 3:
                recent_ratios = window_ratios.tail(3)
                if recent_ratios.iloc[-1] > recent_ratios.iloc[-2] > recent_ratios.iloc[-3]:
                    trend_score = 20.0  # 持续上升
                elif recent_ratios.iloc[-1] > recent_ratios.iloc[-2]:
                    trend_score = 15.0  # 上升
                elif recent_ratios.iloc[-1] < recent_ratios.iloc[-2] < recent_ratios.iloc[-3]:
                    trend_score = 5.0   # 持续下降
                elif recent_ratios.iloc[-1] < recent_ratios.iloc[-2]:
                    trend_score = 10.0  # 下降
                else:
                    trend_score = 12.5  # 横盘
            else:
                trend_score = 12.5
            
            score += trend_score - 12.5  # 调整基准
            
            # 4. 量比分布评分 (10分)
            # 量比在合理区间内给予更高评分
            if 0.8 <= current_ratio <= 1.2:
                distribution_score = 10.0  # 正常区间
            elif 0.5 <= current_ratio <= 2.0:
                distribution_score = 8.0   # 较正常区间
            elif 0.3 <= current_ratio <= 3.0:
                distribution_score = 5.0   # 偏离区间
            else:
                distribution_score = 2.0   # 极端区间
            
            score += distribution_score - 5.0  # 调整基准
            
            # 确保分数在合理范围内
            score = max(0, min(100, score))
            scores.iloc[i] = score
        
        return scores
    
    def calculate_confidence_Ratio(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """计算置信度"""
        return 0.5
    
    def get_patterns_Ratio(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """获取形态"""
        return pd.DataFrame(index=data.index)
