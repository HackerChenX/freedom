#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
趋向指标(DMI)

判断趋势强度与方向
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple

from indicators.base_indicator import BaseIndicator
from indicators.common import crossover, crossunder
from utils.logger import get_logger

logger = get_logger(__name__)


class DMI(BaseIndicator):
    """
    趋向指标(DMI) (DMI)
    
    分类：趋势类指标
    描述：判断趋势强度与方向
    """
    
    def __init__(self, period: int = 14, adx_period: int = 14):
        """
        初始化趋向指标(DMI)指标
        
        Args:
            period: 计算周期，默认为14
            adx_period: ADX计算周期，默认为14
        """
        super().__init__(name="DMI", description="趋向指标，判断趋势强度与方向")
        self.period = period
        self.adx_period = adx_period
    
    def _validate_dataframe(self, df: pd.DataFrame, required_columns: List[str]) -> None:
        """
        验证DataFrame是否包含所需的列
        
        Args:
            df: 要验证的DataFrame
            required_columns: 所需的列名列表
            
        Raises:
            ValueError: 如果缺少必要的列
        """
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"DataFrame缺少必要的列: {', '.join(missing_columns)}")
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算趋向指标(DMI)指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
                必须包含以下列：
                - close: 收盘价
                - high: 最高价
                - low: 最低价
                
        Returns:
            添加了DMI指标列的DataFrame
        """
        if df.empty:
            return df
            
        # 确保数据包含必要的列
        required_columns = ['close', 'high', 'low']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()
        
        # 计算真实波幅TR
        df_copy['high_low'] = df_copy['high'] - df_copy['low']
        df_copy['high_close'] = abs(df_copy['high'] - df_copy['close'].shift(1))
        df_copy['low_close'] = abs(df_copy['low'] - df_copy['close'].shift(1))
        df_copy['TR'] = df_copy[['high_low', 'high_close', 'low_close']].max(axis=1)
        
        # 计算方向线DM
        df_copy['up_move'] = df_copy['high'] - df_copy['high'].shift(1)
        df_copy['down_move'] = df_copy['low'].shift(1) - df_copy['low']
        
        # 计算+DM和-DM
        df_copy['+DM'] = np.where((df_copy['up_move'] > df_copy['down_move']) & (df_copy['up_move'] > 0), 
                                df_copy['up_move'], 0)
        df_copy['-DM'] = np.where((df_copy['down_move'] > df_copy['up_move']) & (df_copy['down_move'] > 0), 
                                df_copy['down_move'], 0)
        
        # 计算平滑后的TR、+DM和-DM
        df_copy['TR_' + str(self.period)] = df_copy['TR'].rolling(window=self.period).sum()
        df_copy['+DM_' + str(self.period)] = df_copy['+DM'].rolling(window=self.period).sum()
        df_copy['-DM_' + str(self.period)] = df_copy['-DM'].rolling(window=self.period).sum()
        
        # 计算+DI和-DI
        df_copy['PDI'] = 100 * df_copy['+DM_' + str(self.period)] / df_copy['TR_' + str(self.period)]
        df_copy['MDI'] = 100 * df_copy['-DM_' + str(self.period)] / df_copy['TR_' + str(self.period)]
        
        # 计算方向指数DX
        df_copy['DX'] = 100 * abs(df_copy['PDI'] - df_copy['MDI']) / (df_copy['PDI'] + df_copy['MDI'])
        
        # 计算平均方向指数ADX
        df_copy['ADX'] = df_copy['DX'].rolling(window=self.adx_period).mean()
        
        # 计算平均方向指数评估ADXR
        df_copy['ADXR'] = (df_copy['ADX'] + df_copy['ADX'].shift(self.adx_period)) / 2
        
        # 删除中间计算列
        df_copy.drop(['high_low', 'high_close', 'low_close', 'TR', 'up_move', 'down_move',
                     '+DM', '-DM', 'TR_' + str(self.period), '+DM_' + str(self.period), 
                     '-DM_' + str(self.period), 'DX'], axis=1, inplace=True)
        
        # 保存结果
        self._result = df_copy
        
        return df_copy
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算DMI原始评分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 原始评分序列（0-100分）
        """
        # 确保已计算DMI
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return pd.Series(50.0, index=data.index)
        
        score = pd.Series(50.0, index=data.index)  # 基础分50分
        
        pdi = self._result['PDI']
        mdi = self._result['MDI']
        adx = self._result['ADX']
        adxr = self._result['ADXR']
        
        # 1. DI线交叉评分
        di_cross_score = self._calculate_di_cross_score(pdi, mdi)
        score += di_cross_score
        
        # 2. ADX趋势强度评分
        adx_strength_score = self._calculate_adx_strength_score(adx)
        score += adx_strength_score
        
        # 3. ADX趋势评分
        adx_trend_score = self._calculate_adx_trend_score(adx)
        score += adx_trend_score
        
        # 4. DI线位置评分
        di_position_score = self._calculate_di_position_score(pdi, mdi)
        score += di_position_score
        
        # 5. ADXR确认评分
        adxr_confirm_score = self._calculate_adxr_confirm_score(adx, adxr)
        score += adxr_confirm_score
        
        # 6. DMI综合形态评分
        pattern_score = self._calculate_dmi_pattern_score(pdi, mdi, adx)
        score += pattern_score
        
        return np.clip(score, 0, 100)
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别DMI技术形态
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[str]: 识别出的形态列表
        """
        patterns = []
        
        # 确保已计算DMI
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return patterns
        
        pdi = self._result['PDI']
        mdi = self._result['MDI']
        adx = self._result['ADX']
        adxr = self._result['ADXR']
        
        # 检查最近的信号
        recent_periods = min(10, len(pdi))
        if recent_periods == 0:
            return patterns
        
        recent_pdi = pdi.tail(recent_periods)
        recent_mdi = mdi.tail(recent_periods)
        recent_adx = adx.tail(recent_periods)
        recent_adxr = adxr.tail(recent_periods)
        
        # 检查是否有有效数据
        if recent_pdi.isna().all() or recent_mdi.isna().all() or recent_adx.isna().all():
            return patterns
        
        # 获取最后一个有效值，增加更好的边界检查
        valid_pdi = recent_pdi.dropna()
        valid_mdi = recent_mdi.dropna()
        valid_adx = recent_adx.dropna()
        
        if len(valid_pdi) == 0 or len(valid_mdi) == 0 or len(valid_adx) == 0:
            return patterns
        
        try:
            current_pdi = valid_pdi.iloc[-1]
            current_mdi = valid_mdi.iloc[-1]
            current_adx = valid_adx.iloc[-1]
            
            # 1. DI线交叉形态
            if len(recent_pdi) >= 2 and len(recent_mdi) >= 2:
                if self.crossover(recent_pdi, recent_mdi).any():
                    patterns.append("DMI多头交叉")
                if self.crossunder(recent_pdi, recent_mdi).any():
                    patterns.append("DMI空头交叉")
            
            # 2. ADX趋势强度形态
            adx_strength = self._classify_adx_strength(current_adx)
            patterns.append(f"ADX{adx_strength}")
            
            # 3. ADX趋势方向形态
            if len(recent_adx) >= 5:
                adx_trend = self._detect_adx_trend(recent_adx)
                if adx_trend:
                    patterns.append(f"ADX{adx_trend}")
            
            # 4. DI线优势形态
            if current_pdi > current_mdi + 5:
                patterns.append("DMI多头优势")
            elif current_mdi > current_pdi + 5:
                patterns.append("DMI空头优势")
            else:
                patterns.append("DMI多空均衡")
            
            # 5. DMI极值形态
            if current_pdi > 40:
                patterns.append("PDI极强")
            elif current_pdi < 10:
                patterns.append("PDI极弱")
            
            if current_mdi > 40:
                patterns.append("MDI极强")
            elif current_mdi < 10:
                patterns.append("MDI极弱")
            
            # 6. DMI背离形态
            if len(data) >= 20 and len(recent_pdi) >= 10 and len(recent_mdi) >= 10:
                divergence_type = self._detect_dmi_divergence_pattern(data['close'], recent_pdi, recent_mdi)
                if divergence_type:
                    patterns.append(f"DMI{divergence_type}")
            
            # 7. ADX钝化形态
            if len(recent_adx) >= 5:
                if self._detect_adx_stagnation(recent_adx, threshold=25, periods=5):
                    patterns.append("ADX高位钝化")
                    
        except (IndexError, KeyError) as e:
            # 如果出现索引错误，返回空列表
            logger.warning(f"DMI形态识别出现索引错误: {e}")
            return []
        
        return patterns
    
    def _calculate_di_cross_score(self, pdi: pd.Series, mdi: pd.Series) -> pd.Series:
        """
        计算DI线交叉评分
        
        Args:
            pdi: +DI序列
            mdi: -DI序列
            
        Returns:
            pd.Series: 交叉评分
        """
        cross_score = pd.Series(0.0, index=pdi.index)
        
        # +DI上穿-DI（多头信号）+25分
        pdi_cross_up_mdi = self.crossover(pdi, mdi)
        cross_score += pdi_cross_up_mdi * 25
        
        # -DI上穿+DI（空头信号）-25分
        mdi_cross_up_pdi = self.crossover(mdi, pdi)
        cross_score -= mdi_cross_up_pdi * 25
        
        # DI线差距评分
        di_diff = pdi - mdi
        
        # 多头优势+10分
        strong_bullish = di_diff > 10
        cross_score += strong_bullish * 10
        
        # 空头优势-10分
        strong_bearish = di_diff < -10
        cross_score -= strong_bearish * 10
        
        return cross_score
    
    def _calculate_adx_strength_score(self, adx: pd.Series) -> pd.Series:
        """
        计算ADX趋势强度评分
        
        Args:
            adx: ADX序列
            
        Returns:
            pd.Series: 强度评分
        """
        strength_score = pd.Series(0.0, index=adx.index)
        
        # ADX > 25（强趋势）+20分
        strong_trend = adx > 25
        strength_score += strong_trend * 20
        
        # ADX > 40（极强趋势）+25分
        very_strong_trend = adx > 40
        strength_score += very_strong_trend * 25
        
        # ADX < 20（弱趋势）-10分
        weak_trend = adx < 20
        strength_score -= weak_trend * 10
        
        # ADX < 15（无趋势）-15分
        no_trend = adx < 15
        strength_score -= no_trend * 15
        
        return strength_score
    
    def _calculate_adx_trend_score(self, adx: pd.Series) -> pd.Series:
        """
        计算ADX趋势评分
        
        Args:
            adx: ADX序列
            
        Returns:
            pd.Series: 趋势评分
        """
        trend_score = pd.Series(0.0, index=adx.index)
        
        if len(adx) < 5:
            return trend_score
        
        # ADX上升趋势+15分
        adx_rising = adx > adx.shift(3)
        trend_score += adx_rising * 15
        
        # ADX下降趋势-10分
        adx_falling = adx < adx.shift(3)
        trend_score -= adx_falling * 10
        
        # ADX加速上升+20分
        if len(adx) >= 6:
            adx_accelerating = (adx.diff(3) > adx.shift(3).diff(3))
            trend_score += adx_accelerating * 20
        
        return trend_score
    
    def _calculate_di_position_score(self, pdi: pd.Series, mdi: pd.Series) -> pd.Series:
        """
        计算DI线位置评分
        
        Args:
            pdi: +DI序列
            mdi: -DI序列
            
        Returns:
            pd.Series: 位置评分
        """
        position_score = pd.Series(0.0, index=pdi.index)
        
        # +DI在上方+10分
        pdi_above = pdi > mdi
        position_score += pdi_above * 10
        
        # -DI在上方-10分
        mdi_above = mdi > pdi
        position_score -= mdi_above * 10
        
        # DI线极值评分
        # +DI > 40+15分
        pdi_extreme_high = pdi > 40
        position_score += pdi_extreme_high * 15
        
        # -DI > 40-15分
        mdi_extreme_high = mdi > 40
        position_score -= mdi_extreme_high * 15
        
        return position_score
    
    def _calculate_adxr_confirm_score(self, adx: pd.Series, adxr: pd.Series) -> pd.Series:
        """
        计算ADXR确认评分
        
        Args:
            adx: ADX序列
            adxr: ADXR序列
            
        Returns:
            pd.Series: 确认评分
        """
        confirm_score = pd.Series(0.0, index=adx.index)
        
        # ADX > ADXR（趋势加强）+10分
        adx_above_adxr = adx > adxr
        confirm_score += adx_above_adxr * 10
        
        # ADX < ADXR（趋势减弱）-5分
        adx_below_adxr = adx < adxr
        confirm_score -= adx_below_adxr * 5
        
        # ADX与ADXR差距评分
        adx_adxr_diff = abs(adx - adxr)
        
        # 差距较大+5分
        large_diff = adx_adxr_diff > 5
        confirm_score += large_diff * 5
        
        return confirm_score
    
    def _calculate_dmi_pattern_score(self, pdi: pd.Series, mdi: pd.Series, adx: pd.Series) -> pd.Series:
        """
        计算DMI综合形态评分
        
        Args:
            pdi: +DI序列
            mdi: -DI序列
            adx: ADX序列
            
        Returns:
            pd.Series: 形态评分
        """
        pattern_score = pd.Series(0.0, index=pdi.index)
        
        # 强势多头形态：+DI > -DI且ADX > 25
        strong_bullish = (pdi > mdi) & (adx > 25)
        pattern_score += strong_bullish * 20
        
        # 强势空头形态：-DI > +DI且ADX > 25
        strong_bearish = (mdi > pdi) & (adx > 25)
        pattern_score -= strong_bearish * 20
        
        # 震荡形态：ADX < 20且DI线接近
        sideways = (adx < 20) & (abs(pdi - mdi) < 5)
        pattern_score -= sideways * 10
        
        return pattern_score
    
    def _classify_adx_strength(self, adx_value: float) -> str:
        """
        分类ADX强度
        
        Args:
            adx_value: ADX值
            
        Returns:
            str: 强度分类
        """
        if adx_value > 50:
            return "极强趋势"
        elif adx_value > 40:
            return "很强趋势"
        elif adx_value > 25:
            return "强趋势"
        elif adx_value > 20:
            return "中等趋势"
        elif adx_value > 15:
            return "弱趋势"
        else:
            return "无趋势"
    
    def _detect_adx_trend(self, adx: pd.Series) -> Optional[str]:
        """
        检测ADX趋势
        
        Args:
            adx: ADX序列
            
        Returns:
            Optional[str]: 趋势类型或None
        """
        if len(adx) < 5:
            return None
        
        # 计算ADX趋势
        recent_adx = adx.tail(5)
        
        # 连续上升
        if all(recent_adx.iloc[i] >= recent_adx.iloc[i-1] for i in range(1, len(recent_adx))):
            return "持续上升"
        
        # 连续下降
        if all(recent_adx.iloc[i] <= recent_adx.iloc[i-1] for i in range(1, len(recent_adx))):
            return "持续下降"
        
        # 整体上升趋势
        if recent_adx.iloc[-1] > recent_adx.iloc[0] * 1.1:
            return "上升趋势"
        
        # 整体下降趋势
        if recent_adx.iloc[-1] < recent_adx.iloc[0] * 0.9:
            return "下降趋势"
        
        return None
    
    def _detect_dmi_divergence_pattern(self, price: pd.Series, pdi: pd.Series, mdi: pd.Series) -> Optional[str]:
        """
        检测DMI背离形态
        
        Args:
            price: 价格序列
            pdi: +DI序列
            mdi: -DI序列
            
        Returns:
            Optional[str]: 背离类型或None
        """
        if len(price) < 20:
            return None
        
        # 寻找最近的峰值和谷值
        recent_price = price.tail(20)
        recent_pdi = pdi.tail(20)
        recent_mdi = mdi.tail(20)
        
        price_extremes = []
        pdi_extremes = []
        mdi_extremes = []
        
        # 简化的极值检测
        for i in range(2, len(recent_price) - 2):
            if (recent_price.iloc[i] > recent_price.iloc[i-1] and 
                recent_price.iloc[i] > recent_price.iloc[i+1]):
                price_extremes.append(recent_price.iloc[i])
                pdi_extremes.append(recent_pdi.iloc[i])
                mdi_extremes.append(recent_mdi.iloc[i])
            elif (recent_price.iloc[i] < recent_price.iloc[i-1] and 
                  recent_price.iloc[i] < recent_price.iloc[i+1]):
                price_extremes.append(recent_price.iloc[i])
                pdi_extremes.append(recent_pdi.iloc[i])
                mdi_extremes.append(recent_mdi.iloc[i])
        
        if len(price_extremes) >= 2:
            price_trend = price_extremes[-1] - price_extremes[-2]
            pdi_trend = pdi_extremes[-1] - pdi_extremes[-2]
            mdi_trend = mdi_extremes[-1] - mdi_extremes[-2]
            
            # 正背离：价格创新低但+DI未创新低
            if price_trend < -0.01 and pdi_trend > 2:
                return "正背离"
            # 负背离：价格创新高但+DI未创新高
            elif price_trend > 0.01 and pdi_trend < -2:
                return "负背离"
        
        return None
    
    def _detect_adx_stagnation(self, adx: pd.Series, threshold: float, periods: int) -> bool:
        """
        检测ADX钝化
        
        Args:
            adx: ADX序列
            threshold: 阈值
            periods: 检测周期数
            
        Returns:
            bool: 是否钝化
        """
        if len(adx) < periods:
            return False
        
        recent_adx = adx.tail(periods)
        
        # 高位钝化：ADX在高位且变化很小
        return (recent_adx > threshold).all() and (recent_adx.max() - recent_adx.min()) < 5
        
    def get_signals(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        生成趋向指标(DMI)指标交易信号
        
        Args:
            df: 包含价格数据和DMI指标的DataFrame
            **kwargs: 额外参数
                
        Returns:
            添加了信号列的DataFrame:
            - dmi_signal: 1=买入信号, -1=卖出信号, 0=无信号
        """
        if df.empty:
            return df
            
        # 检查必要的指标列是否存在
        required_columns = ['PDI', 'MDI', 'ADX']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()
        
        # 初始化信号列
        df_copy['dmi_signal'] = 0
        
        # +DI上穿-DI为买入信号
        df_copy.loc[crossover(df_copy['PDI'], df_copy['MDI']), 'dmi_signal'] = 1
        
        # -DI上穿+DI为卖出信号
        df_copy.loc[crossover(df_copy['MDI'], df_copy['PDI']), 'dmi_signal'] = -1
        
        # 强化信号：ADX > 25表示趋势显著
        df_copy.loc[(df_copy['dmi_signal'] == 1) & (df_copy['ADX'] < 25), 'dmi_signal'] = 0
        df_copy.loc[(df_copy['dmi_signal'] == -1) & (df_copy['ADX'] < 25), 'dmi_signal'] = 0
        
        return df_copy

    def generate_signals(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        生成DMI指标标准化交易信号
        
        Args:
            data: 输入数据，包含OHLCV数据
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            pd.DataFrame: 信号结果DataFrame，包含标准化信号
        """
        # 确保已计算DMI指标
        if not self.has_result():
            self.calculate(data)
        
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
        signals['market_env'] = 'sideways_market'
        signals['volume_confirmation'] = False
        
        # 计算评分
        score = self.calculate_raw_score(data, **kwargs)
        signals['score'] = score
        
        # 检测形态
        patterns = self.identify_patterns(data, **kwargs)
        
        # 获取DMI数据
        pdi = self._result['PDI']
        mdi = self._result['MDI']
        adx = self._result['ADX']
        adxr = self._result['ADXR']
        
        # 1. +DI上穿-DI，买入信号
        di_crossover = crossover(pdi, mdi)
        signals.loc[di_crossover, 'buy_signal'] = True
        signals.loc[di_crossover, 'neutral_signal'] = False
        signals.loc[di_crossover, 'trend'] = 1
        signals.loc[di_crossover, 'signal_type'] = 'DI金叉'
        signals.loc[di_crossover, 'signal_desc'] = '+DI上穿-DI，多头趋势确立'
        signals.loc[di_crossover, 'confidence'] = 70.0
        signals.loc[di_crossover, 'position_size'] = 0.4
        signals.loc[di_crossover, 'risk_level'] = '中'
        
        # 2. -DI上穿+DI，卖出信号
        di_crossunder = crossover(mdi, pdi)
        signals.loc[di_crossunder, 'sell_signal'] = True
        signals.loc[di_crossunder, 'neutral_signal'] = False
        signals.loc[di_crossunder, 'trend'] = -1
        signals.loc[di_crossunder, 'signal_type'] = 'DI死叉'
        signals.loc[di_crossunder, 'signal_desc'] = '-DI上穿+DI，空头趋势确立'
        signals.loc[di_crossunder, 'confidence'] = 70.0
        signals.loc[di_crossunder, 'position_size'] = 0.4
        signals.loc[di_crossunder, 'risk_level'] = '中'
        
        # 3. ADX上升且大于阈值，趋势增强信号
        adx_rising = (adx > adx.shift(1)) & (adx > 25)
        
        # 强多头趋势信号：ADX上升且+DI>-DI
        strong_uptrend = adx_rising & (pdi > mdi)
        signals.loc[strong_uptrend, 'buy_signal'] = True
        signals.loc[strong_uptrend, 'neutral_signal'] = False
        signals.loc[strong_uptrend, 'trend'] = 1
        signals.loc[strong_uptrend, 'signal_type'] = '强多头趋势'
        signals.loc[strong_uptrend, 'signal_desc'] = 'ADX上升且+DI>-DI，多头趋势增强'
        signals.loc[strong_uptrend, 'confidence'] = 80.0
        signals.loc[strong_uptrend, 'position_size'] = 0.5
        signals.loc[strong_uptrend, 'risk_level'] = '低'
        
        # 强空头趋势信号：ADX上升且-DI>+DI
        strong_downtrend = adx_rising & (mdi > pdi)
        signals.loc[strong_downtrend, 'sell_signal'] = True
        signals.loc[strong_downtrend, 'neutral_signal'] = False
        signals.loc[strong_downtrend, 'trend'] = -1
        signals.loc[strong_downtrend, 'signal_type'] = '强空头趋势'
        signals.loc[strong_downtrend, 'signal_desc'] = 'ADX上升且-DI>+DI，空头趋势增强'
        signals.loc[strong_downtrend, 'confidence'] = 80.0
        signals.loc[strong_downtrend, 'position_size'] = 0.5
        signals.loc[strong_downtrend, 'risk_level'] = '低'
        
        # 4. ADX下降，趋势减弱信号
        adx_falling = (adx < adx.shift(1)) & (adx > 20)
        
        # 多头趋势减弱信号：ADX下降且+DI>-DI
        weakening_uptrend = adx_falling & (pdi > mdi)
        signals.loc[weakening_uptrend, 'buy_signal'] = True
        signals.loc[weakening_uptrend, 'neutral_signal'] = False
        signals.loc[weakening_uptrend, 'trend'] = 1
        signals.loc[weakening_uptrend, 'signal_type'] = '减弱多头趋势'
        signals.loc[weakening_uptrend, 'signal_desc'] = 'ADX下降且+DI>-DI，多头趋势减弱'
        signals.loc[weakening_uptrend, 'confidence'] = 60.0
        signals.loc[weakening_uptrend, 'position_size'] = 0.3
        signals.loc[weakening_uptrend, 'risk_level'] = '中'
        
        # 空头趋势减弱信号：ADX下降且-DI>+DI
        weakening_downtrend = adx_falling & (mdi > pdi)
        signals.loc[weakening_downtrend, 'sell_signal'] = True
        signals.loc[weakening_downtrend, 'neutral_signal'] = False
        signals.loc[weakening_downtrend, 'trend'] = -1
        signals.loc[weakening_downtrend, 'signal_type'] = '减弱空头趋势'
        signals.loc[weakening_downtrend, 'signal_desc'] = 'ADX下降且-DI>+DI，空头趋势减弱'
        signals.loc[weakening_downtrend, 'confidence'] = 60.0
        signals.loc[weakening_downtrend, 'position_size'] = 0.3
        signals.loc[weakening_downtrend, 'risk_level'] = '中'
        
        # 5. ADX非常低，无趋势信号
        no_trend = adx < 15
        signals.loc[no_trend, 'neutral_signal'] = True
        signals.loc[no_trend, 'buy_signal'] = False
        signals.loc[no_trend, 'sell_signal'] = False
        signals.loc[no_trend, 'trend'] = 0
        signals.loc[no_trend, 'signal_type'] = '无趋势'
        signals.loc[no_trend, 'signal_desc'] = 'ADX低于15，市场处于无趋势震荡状态'
        signals.loc[no_trend, 'confidence'] = 60.0
        signals.loc[no_trend, 'position_size'] = 0.0
        signals.loc[no_trend, 'risk_level'] = '中'
        
        # 6. ADX非常高，趋势过热信号
        extreme_trend = adx > 50
        
        # 根据DI判断是多头还是空头过热
        extreme_uptrend = extreme_trend & (pdi > mdi)
        signals.loc[extreme_uptrend, 'buy_signal'] = True
        signals.loc[extreme_uptrend, 'neutral_signal'] = False
        signals.loc[extreme_uptrend, 'trend'] = 1
        signals.loc[extreme_uptrend, 'signal_type'] = '极端多头趋势'
        signals.loc[extreme_uptrend, 'signal_desc'] = 'ADX极高且+DI>-DI，多头趋势过热'
        signals.loc[extreme_uptrend, 'confidence'] = 65.0
        signals.loc[extreme_uptrend, 'position_size'] = 0.3
        signals.loc[extreme_uptrend, 'risk_level'] = '高'
        
        extreme_downtrend = extreme_trend & (mdi > pdi)
        signals.loc[extreme_downtrend, 'sell_signal'] = True
        signals.loc[extreme_downtrend, 'neutral_signal'] = False
        signals.loc[extreme_downtrend, 'trend'] = -1
        signals.loc[extreme_downtrend, 'signal_type'] = '极端空头趋势'
        signals.loc[extreme_downtrend, 'signal_desc'] = 'ADX极高且-DI>+DI，空头趋势过热'
        signals.loc[extreme_downtrend, 'confidence'] = 65.0
        signals.loc[extreme_downtrend, 'position_size'] = 0.3
        signals.loc[extreme_downtrend, 'risk_level'] = '高'
        
        # 7. ADXR确认信号
        adxr_confirming_adx = (adxr > adxr.shift(1)) & (adx > adx.shift(1))
        
        # ADXR确认的多头趋势
        adxr_confirmed_uptrend = adxr_confirming_adx & (pdi > mdi)
        signals.loc[adxr_confirmed_uptrend, 'buy_signal'] = True
        signals.loc[adxr_confirmed_uptrend, 'neutral_signal'] = False
        signals.loc[adxr_confirmed_uptrend, 'trend'] = 1
        signals.loc[adxr_confirmed_uptrend, 'signal_type'] = 'ADXR确认多头'
        signals.loc[adxr_confirmed_uptrend, 'signal_desc'] = 'ADXR确认多头趋势持续'
        signals.loc[adxr_confirmed_uptrend, 'confidence'] = 75.0
        signals.loc[adxr_confirmed_uptrend, 'position_size'] = 0.4
        signals.loc[adxr_confirmed_uptrend, 'risk_level'] = '中'
        
        # ADXR确认的空头趋势
        adxr_confirmed_downtrend = adxr_confirming_adx & (mdi > pdi)
        signals.loc[adxr_confirmed_downtrend, 'sell_signal'] = True
        signals.loc[adxr_confirmed_downtrend, 'neutral_signal'] = False
        signals.loc[adxr_confirmed_downtrend, 'trend'] = -1
        signals.loc[adxr_confirmed_downtrend, 'signal_type'] = 'ADXR确认空头'
        signals.loc[adxr_confirmed_downtrend, 'signal_desc'] = 'ADXR确认空头趋势持续'
        signals.loc[adxr_confirmed_downtrend, 'confidence'] = 75.0
        signals.loc[adxr_confirmed_downtrend, 'position_size'] = 0.4
        signals.loc[adxr_confirmed_downtrend, 'risk_level'] = '中'
        
        # 8. 根据形态设置更多信号
        for pattern in patterns:
            pattern_idx = signals.index[-5:]  # 假设形态影响最近5个周期
            
            if '多头趋势' in pattern or '看涨' in pattern:
                signals.loc[pattern_idx, 'buy_signal'] = True
                signals.loc[pattern_idx, 'neutral_signal'] = False
                signals.loc[pattern_idx, 'trend'] = 1
                signals.loc[pattern_idx, 'signal_type'] = 'DMI趋势信号'
                signals.loc[pattern_idx, 'signal_desc'] = pattern
                signals.loc[pattern_idx, 'confidence'] = 70.0
                signals.loc[pattern_idx, 'position_size'] = 0.4
                signals.loc[pattern_idx, 'risk_level'] = '中'
            
            elif '空头趋势' in pattern or '看跌' in pattern:
                signals.loc[pattern_idx, 'sell_signal'] = True
                signals.loc[pattern_idx, 'neutral_signal'] = False
                signals.loc[pattern_idx, 'trend'] = -1
                signals.loc[pattern_idx, 'signal_type'] = 'DMI趋势信号'
                signals.loc[pattern_idx, 'signal_desc'] = pattern
                signals.loc[pattern_idx, 'confidence'] = 70.0
                signals.loc[pattern_idx, 'position_size'] = 0.4
                signals.loc[pattern_idx, 'risk_level'] = '中'
            
            elif '无趋势' in pattern or '横盘' in pattern:
                signals.loc[pattern_idx, 'neutral_signal'] = True
                signals.loc[pattern_idx, 'buy_signal'] = False
                signals.loc[pattern_idx, 'sell_signal'] = False
                signals.loc[pattern_idx, 'trend'] = 0
                signals.loc[pattern_idx, 'signal_type'] = 'DMI震荡信号'
                signals.loc[pattern_idx, 'signal_desc'] = pattern
                signals.loc[pattern_idx, 'confidence'] = 60.0
                signals.loc[pattern_idx, 'position_size'] = 0.0
                signals.loc[pattern_idx, 'risk_level'] = '低'
        
        # 设置止损价格
        if 'low' in data.columns and 'high' in data.columns:
            # 买入信号的止损设为最近的低点
            buy_indices = signals[signals['buy_signal']].index
            if not buy_indices.empty:
                for idx in buy_indices:
                    if idx > data.index[10]:  # 确保有足够的历史数据
                        lookback = 5
                        # 使用最近低点作为止损
                        recent_low = data.loc[idx-lookback:idx, 'low'].min()
                        signals.loc[idx, 'stop_loss'] = recent_low
        
            # 卖出信号的止损设为最近的高点
            sell_indices = signals[signals['sell_signal']].index
            if not sell_indices.empty:
                for idx in sell_indices:
                    if idx > data.index[10]:  # 确保有足够的历史数据
                        lookback = 5
                        # 使用最近高点作为止损
                        recent_high = data.loc[idx-lookback:idx, 'high'].max()
                        signals.loc[idx, 'stop_loss'] = recent_high
        
        # 根据ADX值判断市场环境
        signals['market_env'] = 'sideways_market'  # 默认震荡市场
        
        # ADX高且+DI>-DI，上升趋势市场
        uptrend_market = (adx > 25) & (pdi > mdi)
        signals.loc[uptrend_market, 'market_env'] = 'bull_market'
        
        # ADX高且-DI>+DI，下降趋势市场
        downtrend_market = (adx > 25) & (mdi > pdi)
        signals.loc[downtrend_market, 'market_env'] = 'bear_market'
        
        # ADX低，震荡市场
        strong_sideways = adx < 15
        signals.loc[strong_sideways, 'market_env'] = 'sideways_market'
        
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
    
    def plot(self, df: pd.DataFrame, ax=None, **kwargs):
        """
        绘制趋向指标(DMI)指标图表
        
        Args:
            df: 包含DMI指标的DataFrame
            ax: matplotlib轴对象，如果为None则创建新的
            **kwargs: 额外绘图参数
            
        Returns:
            matplotlib轴对象
        """
        import matplotlib.pyplot as plt
        
        # 检查必要的指标列是否存在
        required_columns = ['PDI', 'MDI', 'ADX', 'ADXR']
        self._validate_dataframe(df, required_columns)
        
        # 创建新的轴对象（如果未提供）
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
            
        # 绘制指标线
        ax.plot(df.index, df['PDI'], label='+DI', color='g')
        ax.plot(df.index, df['MDI'], label='-DI', color='r')
        ax.plot(df.index, df['ADX'], label='ADX', color='b')
        ax.plot(df.index, df['ADXR'], label='ADXR', color='m', linestyle='--')
        
        # 添加参考线
        ax.axhline(y=25, color='k', linestyle='--', alpha=0.3, label='趋势阈值')
        
        ax.set_ylabel('DMI指标')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算指标并返回结果
        
        Args:
            df: 输入DataFrame
            
        Returns:
            包含计算结果的DataFrame
        """
        return self.calculate(df)

