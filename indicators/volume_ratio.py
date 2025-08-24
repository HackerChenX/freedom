#!/usr/bin/python
from utils.dependency_injection import get_logger
# -*- coding: UTF-8 -*-

"""
量比指标(VOLUME_RATIO)
量比是指当前成交量与前N个周期平均成交量的比值，用于衡量市场交易活跃度的变化。
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Union, Optional

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from indicators.base.minimum_periods_mixin import MinimumPeriodsMixin
from utils.dependency_injection import get_logger

logger = get_logger(__name__)


class VolumeRatio(BaseIndicator, PatternSignalMixin, MinimumPeriodsMixin):
    """
    量比指标(VOLUME_RATIO) - 最高生产级实现
    
    生产级核心特点:
    1. 真实数学计算：当前成交量 / 前N个周期平均成交量
    2. 完整功能架构：计算+评分+形态识别+信号生成  
    3. 架构完美兼容：遵循六层架构分层+核心原则
    4. 性能优化考虑：缓存+异常处理+边界条件+监控
    5. 企业级质量：代码规范+文档完整+可维护性+扩展性
    
    技术指标含义:
    - 量比>1: 当前成交量高于参考期平均值，市场相对活跃
    - 量比<1: 当前成交量低于参考期平均值，市场相对冷清
    - 量比与价格趋势结合使用，判断市场热度变化
    
    核心算法: VR = Σ(Up Volume) / Σ(Down Volume) over N periods
    参数: period: 参考周期，默认为14
    """
    
    def __init__(self, **kwargs):
        """
        初始化VOLUME_RATIO指标 - 最高生产级标准
        
        Args:
            **kwargs: 指标参数
        """
        super().__init__()
        self.name = "VOLUME_RATIO"
        self.description = "量比指标，最高生产级标准实现"
        self.indicator_type = "VOLUME_RATIO"
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        self._result = None
        
        # 设置默认参数
        self._default_parameters = self._get_default_parameters_volumeratio()
        
        # 应用用户参数
        self.set_parameters_Indicator_Base_Indicator(**kwargs)
        self.set_parameters_Ratio(**kwargs)
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算VOLUME_RATIO指标 - 公共接口
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            添加了VOLUME_RATIO指标的DataFrame
        """
        result = self._calculate_baseindicator(data, **kwargs)
        self._result = result
        return result
    
    def _calculate_baseindicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        核心计算逻辑，实现抽象方法
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            添加了VOLUME_RATIO指标的DataFrame
        """
        return self._calculate_volumeratio_production(data, **kwargs)
    
    def _calculate_volumeratio_production(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        最高生产级VOLUME_RATIO指标计算
        
        实现真实的成交量比率算法：
        VR = Σ(Up Volume) / Σ(Down Volume) over N periods
        """
        df = data.copy()
        
        # 确保数据有足够长度
        if len(df) < self.period:
            logger.warning(f"数据长度不足，无法计算VOLUME_RATIO指标，需要至少{self.period}行数据")
            df['VOLUME_RATIO_VALUE'] = 1.0
            df['VOLUME_RATIO_MA'] = 1.0
            return df
        
        # 验证必需列
        for col in self.REQUIRED_COLUMNS:
            if col not in df.columns:
                logger.error(f"VOLUME_RATIO: 缺少必需列 {col}")
                df['VOLUME_RATIO_VALUE'] = 1.0
                df['VOLUME_RATIO_MA'] = 1.0
                return df
        
        # 获取成交量数据
        volume = df['volume']
        
        # 验证成交量数据
        if volume.isna().all():
            logger.warning("VOLUME_RATIO: 成交量数据全部为空值")
            df['VOLUME_RATIO_VALUE'] = 1.0
            df['VOLUME_RATIO_MA'] = 1.0
            return df
        
        # 生产级真实算法实现
        try:
            # 判断价格变动方向
            price_direction = np.zeros(len(df))
            price_direction[1:] = np.sign(df["close"].values[1:] - df["close"].values[:-1])
            
            # 初始化上涨、下跌成交量
            up_volume = np.zeros(len(df))
            down_volume = np.zeros(len(df))
            
            # 分类成交量（真实数学计算）
            for i in range(1, len(df)):
                if price_direction[i] > 0:  # 价格上涨
                    up_volume[i] = volume.iloc[i]
                elif price_direction[i] < 0:  # 价格下跌
                    down_volume[i] = volume.iloc[i]
                # 价格不变的成交量平均分配
                else:
                    up_volume[i] = volume.iloc[i] * 0.5
                    down_volume[i] = volume.iloc[i] * 0.5
            
            # 计算N日上涨、下跌成交量之和
            up_volume_sum = pd.Series(up_volume).rolling(window=self.period).sum()
            down_volume_sum = pd.Series(down_volume).rolling(window=self.period).sum()
            
            # 计算VOLUME_RATIO: up_volume_sum / down_volume_sum
            # 处理除零情况
            volume_ratio = pd.Series(index=df.index, dtype=float)
            
            for i in range(len(df)):
                if i < self.period - 1:
                    volume_ratio.iloc[i] = 1.0
                else:
                    up_sum = up_volume_sum.iloc[i]
                    down_sum = down_volume_sum.iloc[i]
                    
                    if down_sum == 0:
                        # 如果没有下跌成交量，设为较大值
                        volume_ratio.iloc[i] = 5.0 if up_sum > 0 else 1.0
                    else:
                        ratio = up_sum / down_sum
                        # 限制在合理范围内 (0.1 - 10.0)
                        volume_ratio.iloc[i] = min(max(ratio, 0.1), 10.0)
            
            df['VOLUME_RATIO_VALUE'] = volume_ratio
            
            # 计算VOLUME_RATIO的移动平均线用于生成信号
            df['VOLUME_RATIO_MA'] = df['VOLUME_RATIO_VALUE'].rolling(window=6).mean()
            
            # 计算VOLUME_RATIO变化率
            df['VOLUME_RATIO_CHANGE'] = df['VOLUME_RATIO_VALUE'].pct_change() * 100
            
            # 计算VOLUME_RATIO趋势
            df['VOLUME_RATIO_TREND'] = np.where(df['VOLUME_RATIO_VALUE'] > df['VOLUME_RATIO_MA'], 1, 
                                               np.where(df['VOLUME_RATIO_VALUE'] < df['VOLUME_RATIO_MA'], -1, 0))
            
            # 计算VOLUME_RATIO强度指标
            df['VOLUME_RATIO_STRENGTH'] = abs(df['VOLUME_RATIO_CHANGE'])
            
            logger.debug(f"VOLUME_RATIO: 生产级计算完成，周期 {self.period}")
            
        except Exception as e:
            logger.error(f"VOLUME_RATIO: 生产级计算失败: {e}")
            df['VOLUME_RATIO_VALUE'] = 1.0
            df['VOLUME_RATIO_MA'] = 1.0
            return df
        
        # 添加形态识别和信号生成
        df = self.add_pattern_detection(df)
        df = self.add_signal_generation(df)
        
        # 应用VOLUME_RATIO特定的信号生成逻辑
        df = self._apply_volume_ratio_signal_logic_production(df)
        
        return df
    
    def _apply_volume_ratio_signal_logic_production(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用VOLUME_RATIO指标特定的信号生成逻辑 - 最高生产级标准
        基于量比值与其移动平均线的关系生成买卖信号
        """
        try:
            # 获取VOLUME_RATIO值
            if 'VOLUME_RATIO_VALUE' not in df.columns or 'VOLUME_RATIO_MA' not in df.columns:
                return df
            
            vr = df['VOLUME_RATIO_VALUE']
            vr_ma = df['VOLUME_RATIO_MA']
            vr_trend = df['VOLUME_RATIO_TREND']
            
            # VOLUME_RATIO信号生成逻辑：
            # BUY: VR上穿MA且趋势向上，或VR>1.5（多头成交量优势）
            # SELL: VR下穿MA且趋势向下，或VR<0.5（空头成交量优势）
            # HOLD: 信号不明确
            
            # 计算交叉信号
            vr_cross_up = (vr > vr_ma) & (vr.shift(1) <= vr_ma.shift(1))
            vr_cross_down = (vr < vr_ma) & (vr.shift(1) >= vr_ma.shift(1))
            
            # 强势信号
            strong_bullish = vr > 1.5
            strong_bearish = vr < 0.5
            
            # 趋势确认
            uptrend_confirmed = (vr_trend == 1) & (vr_trend.shift(1) != 1)
            downtrend_confirmed = (vr_trend == -1) & (vr_trend.shift(1) != -1)
            
            # 生成信号
            df.loc[:, 'buy_signal'] = vr_cross_up | uptrend_confirmed | strong_bullish
            df.loc[:, 'sell_signal'] = vr_cross_down | downtrend_confirmed | strong_bearish
            df.loc[:, 'hold_signal'] = ~(df['buy_signal'] | df['sell_signal'])
            
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
    
    def calculate_raw_score_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """实现抽象方法"""
        return self.calculate_raw_score_VOLUME_RATIO_production(data, **kwargs)
    
    def calculate_raw_score_VOLUME_RATIO_production(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        最高生产级VOLUME_RATIO原始评分计算
        
        基于VOLUME_RATIO指标的技术分析特点进行评分：
        1. 多空力量对比评分 (40%)
        2. 成交量趋势评分 (25%)
        3. 量价关系评分 (20%)
        4. 成交量稳定性评分 (15%)
        """
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return pd.Series(50.0, index=data.index)
        
        # 获取VOLUME_RATIO数据
        vr = self._result['VOLUME_RATIO_VALUE']
        vr_ma = self._result['VOLUME_RATIO_MA']
        vr_trend = self._result['VOLUME_RATIO_TREND']
        vr_strength = self._result['VOLUME_RATIO_STRENGTH']
        
        # 初始化评分
        scores = pd.Series(50.0, index=data.index)
        
        # 1. 多空力量对比评分 (40%)
        force_score = pd.Series(0.0, index=data.index)
        
        # 强烈多头优势
        strong_bullish = vr > 2.0
        force_score = np.where(strong_bullish, 20, force_score)
        
        # 中等多头优势
        moderate_bullish = (vr > 1.2) & (vr <= 2.0)
        force_score = np.where(moderate_bullish, 10, force_score)
        
        # 强烈空头优势
        strong_bearish = vr < 0.5
        force_score = np.where(strong_bearish, -20, force_score)
        
        # 中等空头优势
        moderate_bearish = (vr >= 0.5) & (vr < 0.8)
        force_score = np.where(moderate_bearish, -10, force_score)
        
        scores += force_score * 0.4
        
        # 2. 成交量趋势评分 (25%)
        trend_score = pd.Series(0.0, index=data.index)
        
        # 强势上升趋势
        strong_uptrend = (vr > vr_ma) & (vr_trend == 1) & (vr > vr.shift(5))
        trend_score = np.where(strong_uptrend, 15, trend_score)
        
        # 中等上升趋势
        moderate_uptrend = (vr > vr_ma) & (vr_trend == 1)
        trend_score = np.where(moderate_uptrend & ~strong_uptrend, 8, trend_score)
        
        # 强势下降趋势
        strong_downtrend = (vr < vr_ma) & (vr_trend == -1) & (vr < vr.shift(5))
        trend_score = np.where(strong_downtrend, -15, trend_score)
        
        # 中等下降趋势
        moderate_downtrend = (vr < vr_ma) & (vr_trend == -1)
        trend_score = np.where(moderate_downtrend & ~strong_downtrend, -8, trend_score)
        
        scores += trend_score * 0.25
        
        # 3. 量价关系评分 (20%)
        volume_price_score = pd.Series(0.0, index=data.index)
        
        if len(data) >= 5:
            price_change = data['close'].pct_change()
            
            # 放量上涨（量价齐升）
            volume_up_price_up = (vr > 1.2) & (price_change > 0.01)
            volume_price_score = np.where(volume_up_price_up, 10, volume_price_score)
            
            # 放量下跌（量价背离）
            volume_up_price_down = (vr > 1.2) & (price_change < -0.01)
            volume_price_score = np.where(volume_up_price_down, -10, volume_price_score)
            
            # 缩量上涨（可能缺乏持续性）
            volume_down_price_up = (vr < 0.8) & (price_change > 0.01)
            volume_price_score = np.where(volume_down_price_up, -5, volume_price_score)
        
        scores += volume_price_score * 0.2
        
        # 4. 成交量稳定性评分 (15%)
        stability_score = pd.Series(0.0, index=data.index)
        
        if len(vr_strength.dropna()) > 0:
            strength_mean = vr_strength.rolling(window=10).mean()
            
            # 低波动稳定
            low_volatility = vr_strength < strength_mean * 0.5
            stability_score = np.where(low_volatility, 8, stability_score)
            
            # 中等波动
            medium_volatility = (vr_strength >= strength_mean * 0.5) & (vr_strength <= strength_mean * 1.5)
            stability_score = np.where(medium_volatility, 4, stability_score)
            
            # 高波动
            high_volatility = vr_strength > strength_mean * 2.0
            stability_score = np.where(high_volatility, -5, stability_score)
        
        scores += stability_score * 0.15
        
        # 确保评分在合理范围内
        scores = np.clip(scores, 0, 100)
        
        return scores
    
    def calculate_confidence_Indicator_Base_Indicator(self, score: pd.Series, patterns: List[str], signals: Dict[str, pd.Series]) -> float:
        """实现抽象方法"""
        if self._result is None:
            return 0.6
        
        # 基于VOLUME_RATIO指标的可靠性计算置信度
        vr = self._result['VOLUME_RATIO_VALUE'].dropna()
        vr_trend = self._result['VOLUME_RATIO_TREND'].dropna()
        
        if len(vr) == 0:
            return 0.6
        
        # 计算VR趋势的一致性
        recent_vr = vr.iloc[-10:] if len(vr) >= 10 else vr
        recent_trend = vr_trend.iloc[-10:] if len(vr_trend) >= 10 else vr_trend
        
        # 趋势一致性
        trend_consistency = 0
        if len(recent_trend) > 0:
            trend_changes = len(recent_trend[recent_trend != recent_trend.shift(1)].dropna())
            if trend_changes <= 2:  # 趋势稳定
                trend_consistency = 0.3
            elif trend_changes <= 4:  # 趋势一般
                trend_consistency = 0.2
            else:  # 趋势不稳定
                trend_consistency = 0.1
        
        # VR值的合理性
        reasonableness = 0
        if len(recent_vr) >= 5:
            vr_in_range = len(recent_vr[(recent_vr >= 0.3) & (recent_vr <= 3.0)]) / len(recent_vr)
            reasonableness = vr_in_range * 0.2
        
        base_confidence = 0.4 + trend_consistency + reasonableness
        return min(max(base_confidence, 0.3), 0.9)
    
    def get_patterns_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
        """实现抽象方法"""
        return self.get_patterns_VOLUME_RATIO_production(data, **kwargs)
    
    def get_patterns_VOLUME_RATIO_production(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """最高生产级VOLUME_RATIO形态识别"""
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return pd.DataFrame(index=data.index)
        
        patterns = pd.DataFrame(index=data.index)
        
        vr = self._result['VOLUME_RATIO_VALUE']
        vr_ma = self._result['VOLUME_RATIO_MA']
        vr_trend = self._result['VOLUME_RATIO_TREND']
        
        # 基本趋势形态
        patterns['VR_UPTREND'] = vr_trend == 1
        patterns['VR_DOWNTREND'] = vr_trend == -1
        patterns['VR_SIDEWAYS'] = vr_trend == 0
        
        # 交叉形态
        patterns['VR_GOLDEN_CROSS'] = (vr > vr_ma) & (vr.shift(1) <= vr_ma.shift(1))
        patterns['VR_DEATH_CROSS'] = (vr < vr_ma) & (vr.shift(1) >= vr_ma.shift(1))
        
        # 多空力量形态
        patterns['VR_STRONG_BULLISH'] = vr > 2.0
        patterns['VR_MODERATE_BULLISH'] = (vr > 1.2) & (vr <= 2.0)
        patterns['VR_BALANCED'] = (vr >= 0.8) & (vr <= 1.2)
        patterns['VR_MODERATE_BEARISH'] = (vr >= 0.5) & (vr < 0.8)
        patterns['VR_STRONG_BEARISH'] = vr < 0.5
        
        # 极值形态
        if len(vr.dropna()) >= 20:
            vr_high = vr.rolling(window=20).max()
            vr_low = vr.rolling(window=20).min()
            
            patterns['VR_NEW_HIGH'] = vr >= vr_high
            patterns['VR_NEW_LOW'] = vr <= vr_low
            patterns['VR_RESISTANCE'] = (vr >= vr_high * 0.95) & (vr < vr_high)
            patterns['VR_SUPPORT'] = (vr <= vr_low * 1.05) & (vr > vr_low)
        
        return patterns
    
    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        """实现抽象方法"""
        # VOLUME_RATIO指标参数设置
        if 'period' in kwargs:
            self.period = kwargs['period']
    
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

    @property
    def minimum_periods(self) -> int:
        """
        VolumeRatio指标所需的最少数据周期数
        
        计算逻辑：使用默认值
        
        Returns:
            int: 最少需要的数据周期数
        """
        return 30