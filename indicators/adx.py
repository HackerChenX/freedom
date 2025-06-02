#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ADX - 平均方向指数

DMI系统的一部分，用于评估趋势的强度，无论方向如何
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple, Any
import warnings

from indicators.base_indicator import BaseIndicator
from indicators.common import crossover, crossunder
from utils.logger import get_logger

# 静默警告
warnings.filterwarnings("ignore", category=RuntimeWarning)

logger = get_logger(__name__)


class ADX(BaseIndicator):
    """
    平均方向指数(ADX)
    
    衡量趋势的强度，而不考虑其方向。ADX的读数越高，趋势越强
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        初始化ADX指标
        
        Args:
            params: 参数字典，可包含：
                - period: ADX计算周期，默认为14
                - strong_trend: 强趋势阈值，默认为25
        """
        super().__init__(name="ADX", description="平均方向指数指标")
        
        # 设置默认参数
        self.params = {
            "period": 14,
            "strong_trend": 25
        }
        
        # 更新自定义参数
        if params:
            self.params.update(params)
        
        # 注册ADX形态
        self._register_adx_patterns()
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算ADX指标
        
        Args:
            data: 包含OHLCV数据的DataFrame
            **kwargs: 额外的参数
            
        Returns:
            添加了ADX指标的DataFrame
        """
        df = data.copy()
        
        # 提取参数
        period = self.params["period"]
        strong_trend = self.params["strong_trend"]
        
        # 确保数据有足够的长度
        if len(df) < period + 1:
            logger.warning(f"数据长度({len(df)})小于所需的回溯周期({period + 1})，返回原始数据")
            df[f'ADX{period}'] = np.nan
            df[f'PDI{period}'] = np.nan
            df[f'MDI{period}'] = np.nan
            df[f'strong_trend_{period}'] = False
            return df
        
        # 计算价格变化
        df['high_change'] = df['high'] - df['high'].shift(1)
        df['low_change'] = df['low'].shift(1) - df['low']
        
        # 计算+DM和-DM
        df['plus_dm'] = np.where(
            (df['high_change'] > df['low_change']) & (df['high_change'] > 0),
            df['high_change'],
            0
        )
        df['minus_dm'] = np.where(
            (df['low_change'] > df['high_change']) & (df['low_change'] > 0),
            df['low_change'],
            0
        )
        
        # 计算真实波幅(TR)
        df['tr1'] = abs(df['high'] - df['low'])
        df['tr2'] = abs(df['high'] - df['close'].shift(1))
        df['tr3'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # 计算平滑的+DM、-DM和TR
        df['smooth_plus_dm'] = df['plus_dm'].rolling(window=period).sum()
        df['smooth_minus_dm'] = df['minus_dm'].rolling(window=period).sum()
        df['smooth_tr'] = df['tr'].rolling(window=period).sum()
        
        # 计算+DI和-DI
        df[f'PDI{period}'] = 100 * df['smooth_plus_dm'] / df['smooth_tr']
        df[f'MDI{period}'] = 100 * df['smooth_minus_dm'] / df['smooth_tr']
        
        # 计算方向指数(DX)
        df['dx'] = 100 * abs(df[f'PDI{period}'] - df[f'MDI{period}']) / (df[f'PDI{period}'] + df[f'MDI{period}'])
        
        # 计算ADX - DX的period周期平均值
        df[f'ADX{period}'] = df['dx'].rolling(window=period).mean()
        
        # 标记强趋势
        df[f'strong_trend_{period}'] = df[f'ADX{period}'] > strong_trend
        
        # 添加趋势方向
        df[f'trend_direction_{period}'] = np.where(df[f'PDI{period}'] > df[f'MDI{period}'], 'up', 'down')
        
        # 清理中间计算列
        df.drop(['high_change', 'low_change', 'plus_dm', 'minus_dm', 
                'tr1', 'tr2', 'tr3', 'tr', 'smooth_plus_dm', 'smooth_minus_dm', 
                'smooth_tr', 'dx'], axis=1, inplace=True)
        
        return df

    def _validate_dataframe(self, df: pd.DataFrame, required_columns: List[str]) -> None:
        """
        验证DataFrame是否包含所需的列
        
        Args:
            df: 输入数据
            required_columns: 所需的列名列表
            
        Raises:
            ValueError: 如果缺少必要的列
        """
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"输入数据缺少必要的列: {', '.join(missing_columns)}")
    
    def get_signals(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        生成ADX指标交易信号
        
        Args:
            df: 包含价格数据和ADX指标的DataFrame
            **kwargs: 额外参数
                
        Returns:
            添加了信号列的DataFrame:
            - adx_signal: 1=买入信号, -1=卖出信号, 0=无信号
        """
        if df.empty:
            return df
            
        # 检查必要的指标列是否存在
        required_columns = ['PDI', 'MDI', 'ADX', 'ADXR']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()
        
        # 初始化信号列
        df_copy['adx_signal'] = 0
        
        # 1. +DI上穿-DI为买入信号
        df_copy.loc[crossover(df_copy['PDI'], df_copy['MDI']), 'adx_signal'] = 1
        
        # 2. -DI上穿+DI为卖出信号
        df_copy.loc[crossover(df_copy['MDI'], df_copy['PDI']), 'adx_signal'] = -1
        
        # 3. 强化信号：ADX > 25表示趋势显著
        df_copy.loc[(df_copy['adx_signal'] == 1) & (df_copy['ADX'] < 25), 'adx_signal'] = 0
        df_copy.loc[(df_copy['adx_signal'] == -1) & (df_copy['ADX'] < 25), 'adx_signal'] = 0
        
        return df_copy 

    def generate_signals(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        生成ADX指标标准化交易信号
        
        Args:
            data: 输入数据，包含OHLCV数据
            *args: 位置参数
            **kwargs: 关键字参数
                
        Returns:
            pd.DataFrame: 信号结果DataFrame，包含标准化信号
        """
        # 确保已计算ADX指标
        if not self.has_result():
            self.calculate(data)
        
        # 获取DMI相关值
        pdi = self._result['PDI']
        mdi = self._result['MDI']
        adx = self._result['ADX']
        adxr = self._result['ADXR']
        
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
        
        # 计算ATR用于止损设置
        try:
            from indicators.atr import ATR
            atr_indicator = ATR()
            atr_data = atr_indicator.calculate(data)
            atr_values = atr_data['atr']
        except Exception as e:
            logger.warning(f"计算ATR失败: {e}")
            atr_values = pd.Series(0, index=data.index)
        
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
        signals.loc[adxr_confirmed_uptrend, 'signal_desc'] = 'ADXR与ADX同步上升，确认多头趋势'
        signals.loc[adxr_confirmed_uptrend, 'confidence'] = 75.0
        signals.loc[adxr_confirmed_uptrend, 'position_size'] = 0.4
        signals.loc[adxr_confirmed_uptrend, 'risk_level'] = '低'
        
        # ADXR确认的空头趋势
        adxr_confirmed_downtrend = adxr_confirming_adx & (mdi > pdi)
        signals.loc[adxr_confirmed_downtrend, 'sell_signal'] = True
        signals.loc[adxr_confirmed_downtrend, 'neutral_signal'] = False
        signals.loc[adxr_confirmed_downtrend, 'trend'] = -1
        signals.loc[adxr_confirmed_downtrend, 'signal_type'] = 'ADXR确认空头'
        signals.loc[adxr_confirmed_downtrend, 'signal_desc'] = 'ADXR与ADX同步上升，确认空头趋势'
        signals.loc[adxr_confirmed_downtrend, 'confidence'] = 75.0
        signals.loc[adxr_confirmed_downtrend, 'position_size'] = 0.4
        signals.loc[adxr_confirmed_downtrend, 'risk_level'] = '低'
        
        # 8. 根据ADX的值给分
        for i in range(len(signals)):
            if i > 0:  # 跳过第一个数据点
                adx_val = adx.iloc[i] if i < len(adx) else 0
                
                # ADX > 45，极强趋势，+15分
                if adx_val > 45:
                    if signals.iloc[i]['trend'] > 0:
                        signals.iloc[i, signals.columns.get_loc('score')] = 85
                    elif signals.iloc[i]['trend'] < 0:
                        signals.iloc[i, signals.columns.get_loc('score')] = 15
                
                # ADX > 25，强趋势，+10分
                elif adx_val > 25:
                    if signals.iloc[i]['trend'] > 0:
                        signals.iloc[i, signals.columns.get_loc('score')] = 70
                    elif signals.iloc[i]['trend'] < 0:
                        signals.iloc[i, signals.columns.get_loc('score')] = 30
                
                # ADX < 15，无趋势，分数接近50
                elif adx_val < 15:
                    signals.iloc[i, signals.columns.get_loc('score')] = 50
        
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
        signals['market_env'] = '中性'  # 默认中性市场
        
        # ADX高且+DI>-DI，上升趋势市场
        uptrend_market = (adx > 25) & (pdi > mdi)
        signals.loc[uptrend_market, 'market_env'] = '强势'
        
        # ADX高且-DI>+DI，下降趋势市场
        downtrend_market = (adx > 25) & (mdi > pdi)
        signals.loc[downtrend_market, 'market_env'] = '弱势'
        
        # ADX低，震荡市场
        strong_sideways = adx < 15
        signals.loc[strong_sideways, 'market_env'] = '震荡'
        
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
        计算ADX原始评分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 原始评分序列（0-100分）
        """
        # 确保已计算ADX
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return pd.Series(50.0, index=data.index)
        
        score = pd.Series(50.0, index=data.index)  # 基础分50分
        
        # 提取参数
        period = self.params["period"]
        strong_trend = self.params["strong_trend"]
        
        # 获取ADX和DI数据
        adx = self._result[f'ADX{period}']
        pdi = self._result[f'PDI{period}']
        mdi = self._result[f'MDI{period}']
        
        # 1. ADX强度评分
        adx_strength_score = (adx / strong_trend) * 25  # 如果ADX=strong_trend，则得25分
        adx_strength_score = adx_strength_score.clip(0, 25)  # 最高25分
        score += adx_strength_score
        
        # 2. 趋势方向评分
        trend_direction_score = (pdi - mdi) / ((pdi + mdi) / 2) * 30  # 方向分，+DI与-DI差距越大，分值越高
        score += trend_direction_score
        
        # 3. ADX动量评分
        adx_momentum = adx - adx.shift(5)  # 与5日前相比
        adx_momentum_score = adx_momentum / 5  # 每天上升1点，得1分
        adx_momentum_score = adx_momentum_score.clip(-15, 15)  # 限制在±15分
        score += adx_momentum_score
        
        # 4. 趋势交叉评分
        pdi_cross_above_mdi = self.crossover(pdi, mdi)
        mdi_cross_above_pdi = self.crossover(mdi, pdi)
        
        score = score.mask(pdi_cross_above_mdi, score + 15)  # +DI上穿-DI，加15分
        score = score.mask(mdi_cross_above_pdi, score - 15)  # -DI上穿+DI，减15分
        
        # 确保评分在0-100范围内
        return score.clip(0, 100)
    
    def get_patterns(self, data: pd.DataFrame, **kwargs) -> List[Dict[str, Any]]:
        """
        获取ADX形态列表
        
        Args:
            data: 输入K线数据
            **kwargs: 其他参数
            
        Returns:
            List[Dict[str, Any]]: 形态识别结果列表
        """
        from indicators.base_indicator import PatternResult, SignalStrength
        from indicators.pattern_registry import PatternRegistry, PatternType
        
        # 确保已计算ADX
        if not self.has_result():
            self.calculate(data)
        
        patterns = []
        
        # 如果没有结果或数据不足，返回空列表
        if self._result is None or len(self._result) < 2:
            return patterns
        
        # 提取参数
        period = self.params["period"]
        strong_trend = self.params["strong_trend"]
        
        # 获取ADX和DI数据
        adx = self._result[f'ADX{period}']
        pdi = self._result[f'PDI{period}']
        mdi = self._result[f'MDI{period}']
        
        # 获取最近的数据点
        current_adx = adx.iloc[-1]
        prev_adx = adx.iloc[-2]
        current_pdi = pdi.iloc[-1]
        current_mdi = mdi.iloc[-1]
        
        # 1. 检测ADX强度趋势
        if current_adx > strong_trend:
            if current_adx > prev_adx:
                # 强趋势且ADX上升
                strength = min(90, 50 + (current_adx - strong_trend) * 2)
                patterns.append(PatternResult(
                    pattern_id="ADX_STRONG_RISING",
                    display_name="ADX强度上升趋势",
                    strength=strength,
                    duration=self._detect_pattern_duration(adx > strong_trend),
                    details={"adx_value": current_adx, "threshold": strong_trend}
                ).to_dict())
            else:
                # 强趋势但ADX下降
                strength = min(80, 50 + (current_adx - strong_trend) * 1.5)
                patterns.append(PatternResult(
                    pattern_id="ADX_STRONG_FALLING",
                    display_name="ADX强度下降趋势",
                    strength=strength,
                    duration=self._detect_pattern_duration(adx > strong_trend),
                    details={"adx_value": current_adx, "threshold": strong_trend}
                ).to_dict())
        else:
            # 弱趋势
            strength = max(20, 40 - (strong_trend - current_adx) * 1.5)
            patterns.append(PatternResult(
                pattern_id="ADX_WEAK_TREND",
                display_name="ADX弱趋势",
                strength=strength,
                duration=self._detect_pattern_duration(adx <= strong_trend),
                details={"adx_value": current_adx, "threshold": strong_trend}
            ).to_dict())
        
        # 2. 检测PDI和MDI的交叉
        cross_window = min(10, len(pdi))
        recent_pdi = pdi.tail(cross_window)
        recent_mdi = mdi.tail(cross_window)
        
        pdi_cross_above = self.crossover(recent_pdi, recent_mdi)
        mdi_cross_above = self.crossover(recent_mdi, recent_pdi)
        
        if pdi_cross_above.any():
            # PDI上穿MDI - 看涨信号
            cross_idx = pdi_cross_above.to_numpy().nonzero()[0]
            if len(cross_idx) > 0:
                last_cross = cross_idx[-1]
                days_since_cross = len(pdi_cross_above) - 1 - last_cross
                strength = max(60, 80 - days_since_cross * 5)
                
                patterns.append(PatternResult(
                    pattern_id="ADX_BULLISH_CROSS",
                    display_name="ADX看涨交叉",
                    strength=strength,
                    duration=days_since_cross + 1,
                    details={"pdi": current_pdi, "mdi": current_mdi, "days_since_cross": days_since_cross}
                ).to_dict())
        
        if mdi_cross_above.any():
            # MDI上穿PDI - 看跌信号
            cross_idx = mdi_cross_above.to_numpy().nonzero()[0]
            if len(cross_idx) > 0:
                last_cross = cross_idx[-1]
                days_since_cross = len(mdi_cross_above) - 1 - last_cross
                strength = max(60, 80 - days_since_cross * 5)
                
                patterns.append(PatternResult(
                    pattern_id="ADX_BEARISH_CROSS",
                    display_name="ADX看跌交叉",
                    strength=strength,
                    duration=days_since_cross + 1,
                    details={"pdi": current_pdi, "mdi": current_mdi, "days_since_cross": days_since_cross}
                ).to_dict())
        
        # 3. 检测趋势方向
        if current_pdi > current_mdi:
            # 上升趋势
            pdi_mdi_diff = current_pdi - current_mdi
            diff_ratio = pdi_mdi_diff / ((current_pdi + current_mdi) / 2)
            strength = min(90, 50 + diff_ratio * 100)
            
            patterns.append(PatternResult(
                pattern_id="ADX_UPTREND",
                display_name="ADX上升趋势",
                strength=strength,
                duration=self._detect_pattern_duration(pdi > mdi),
                details={"pdi": current_pdi, "mdi": current_mdi, "diff_ratio": diff_ratio}
            ).to_dict())
        else:
            # 下降趋势
            mdi_pdi_diff = current_mdi - current_pdi
            diff_ratio = mdi_pdi_diff / ((current_pdi + current_mdi) / 2)
            strength = min(90, 50 + diff_ratio * 100)
            
            patterns.append(PatternResult(
                pattern_id="ADX_DOWNTREND",
                display_name="ADX下降趋势",
                strength=strength,
                duration=self._detect_pattern_duration(mdi > pdi),
                details={"pdi": current_pdi, "mdi": current_mdi, "diff_ratio": diff_ratio}
            ).to_dict())
        
        # 4. 检测ADX趋势反转
        if len(adx) >= 5:
            adx_5days = adx.tail(5)
            if adx_5days.iloc[-1] > adx_5days.iloc[-2] > adx_5days.iloc[-3] and adx_5days.iloc[-3] < adx_5days.iloc[-4] < adx_5days.iloc[-5]:
                # ADX由下降转为上升 - 趋势即将增强
                patterns.append(PatternResult(
                    pattern_id="ADX_TREND_STRENGTHENING",
                    display_name="ADX趋势增强",
                    strength=75,
                    duration=3,
                    details={"adx_value": current_adx, "adx_change": current_adx - adx_5days.iloc[-5]}
                ).to_dict())
            
            if adx_5days.iloc[-1] < adx_5days.iloc[-2] < adx_5days.iloc[-3] and adx_5days.iloc[-3] > adx_5days.iloc[-4] > adx_5days.iloc[-5]:
                # ADX由上升转为下降 - 趋势即将减弱
                patterns.append(PatternResult(
                    pattern_id="ADX_TREND_WEAKENING",
                    display_name="ADX趋势减弱",
                    strength=75,
                    duration=3,
                    details={"adx_value": current_adx, "adx_change": adx_5days.iloc[-5] - current_adx}
                ).to_dict())
        
        # 5. 检测PDI和MDI的极端分离
        if current_pdi > 0 and current_mdi > 0:
            pdi_mdi_ratio = max(current_pdi, current_mdi) / min(current_pdi, current_mdi)
            if pdi_mdi_ratio > 3:
                # 极端趋势，可能即将反转
                if current_pdi > current_mdi:
                    patterns.append(PatternResult(
                        pattern_id="ADX_EXTREME_UPTREND",
                        display_name="ADX极端上升趋势",
                        strength=85,
                        duration=self._detect_pattern_duration(pdi / mdi > 3),
                        details={"pdi_mdi_ratio": pdi_mdi_ratio}
                    ).to_dict())
                else:
                    patterns.append(PatternResult(
                        pattern_id="ADX_EXTREME_DOWNTREND",
                        display_name="ADX极端下降趋势",
                        strength=85,
                        duration=self._detect_pattern_duration(mdi / pdi > 3),
                        details={"pdi_mdi_ratio": pdi_mdi_ratio}
                    ).to_dict())
        
        return patterns

    def _detect_pattern_duration(self, condition_series: pd.Series) -> int:
        """
        检测形态持续的天数
        
        Args:
            condition_series: 条件序列
        
        Returns:
            int: 持续天数
        """
        if len(condition_series) == 0:
            return 0
        
        # 获取连续满足条件的天数
        reverse_cond = condition_series.iloc[::-1]
        duration = 0
        
        for val in reverse_cond:
            if val:
                duration += 1
            else:
                break
        
        return duration

    def _register_adx_patterns(self):
        """
        注册ADX形态
        """
        from indicators.pattern_registry import PatternRegistry, PatternType
        
        # 注册ADX强度趋势形态
        PatternRegistry.register(
            pattern_id="ADX_STRONG_RISING",
            display_name="ADX强度上升趋势",
            description="ADX值高于阈值且继续上升，表示强趋势增强",
            indicator_types=["ADX", "趋势"],
            score_impact=10.0,
            pattern_type="trend",
            signal_type="neutral"
        )
        
        PatternRegistry.register(
            pattern_id="ADX_STRONG_FALLING",
            display_name="ADX强度下降趋势",
            description="ADX值高于阈值但开始下降，表示强趋势可能减弱",
            indicator_types=["ADX", "趋势"],
            score_impact=5.0,
            pattern_type="trend",
            signal_type="neutral"
        )
        
        PatternRegistry.register(
            pattern_id="ADX_WEAK_TREND",
            display_name="ADX弱趋势",
            description="ADX值低于阈值，表示趋势不明显，可能处于震荡市场",
            indicator_types=["ADX", "趋势"],
            score_impact=-5.0,
            pattern_type="trend",
            signal_type="neutral"
        )
        
        # 注册PDI和MDI交叉形态
        PatternRegistry.register(
            pattern_id="ADX_BULLISH_CROSS",
            display_name="ADX看涨交叉",
            description="+DI上穿-DI，表示可能开始上升趋势",
            indicator_types=["ADX", "趋势"],
            score_impact=15.0,
            pattern_type="reversal",
            signal_type="bullish"
        )
        
        PatternRegistry.register(
            pattern_id="ADX_BEARISH_CROSS",
            display_name="ADX看跌交叉",
            description="-DI上穿+DI，表示可能开始下降趋势",
            indicator_types=["ADX", "趋势"],
            score_impact=-15.0,
            pattern_type="reversal",
            signal_type="bearish"
        )
        
        # 注册趋势方向形态
        PatternRegistry.register(
            pattern_id="ADX_UPTREND",
            display_name="ADX上升趋势",
            description="+DI大于-DI，表示处于上升趋势",
            indicator_types=["ADX", "趋势"],
            score_impact=8.0,
            pattern_type="trend",
            signal_type="bullish"
        )
        
        PatternRegistry.register(
            pattern_id="ADX_DOWNTREND",
            display_name="ADX下降趋势",
            description="-DI大于+DI，表示处于下降趋势",
            indicator_types=["ADX", "趋势"],
            score_impact=-8.0,
            pattern_type="trend",
            signal_type="bearish"
        )
        
        # 注册ADX趋势反转形态
        PatternRegistry.register(
            pattern_id="ADX_TREND_STRENGTHENING",
            display_name="ADX趋势增强",
            description="ADX从下降转为上升，表示趋势即将增强",
            indicator_types=["ADX", "趋势"],
            score_impact=12.0,
            pattern_type="continuation",
            signal_type="neutral"
        )
        
        PatternRegistry.register(
            pattern_id="ADX_TREND_WEAKENING",
            display_name="ADX趋势减弱",
            description="ADX从上升转为下降，表示趋势即将减弱",
            indicator_types=["ADX", "趋势"],
            score_impact=-12.0,
            pattern_type="reversal",
            signal_type="neutral"
        )
        
        # 注册极端趋势形态
        PatternRegistry.register(
            pattern_id="ADX_EXTREME_UPTREND",
            display_name="ADX极端上升趋势",
            description="+DI远大于-DI，表示极端上升趋势，可能即将反转",
            indicator_types=["ADX", "趋势"],
            score_impact=5.0,
            pattern_type="exhaustion",
            signal_type="bullish"
        )
        
        PatternRegistry.register(
            pattern_id="ADX_EXTREME_DOWNTREND",
            display_name="ADX极端下降趋势",
            description="-DI远大于+DI，表示极端下降趋势，可能即将反转",
            indicator_types=["ADX", "趋势"],
            score_impact=-5.0,
            pattern_type="exhaustion",
            signal_type="bearish"
        )

    def generate_trading_signals(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """
        生成交易信号
        
        Args:
            data: 输入数据
            **kwargs: 额外参数
            
        Returns:
            Dict[str, pd.Series]: 包含交易信号的字典
        """
        # 确保已计算指标
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        # 初始化信号
        signals = {}
        signals['buy_signal'] = pd.Series(False, index=data.index)
        signals['sell_signal'] = pd.Series(False, index=data.index)
        signals['signal_strength'] = pd.Series(0, index=data.index)
    
        # 在这里实现指标特定的信号生成逻辑
        # 此处提供默认实现
    
        return signals
        
    def plot(self, df: pd.DataFrame, ax=None, **kwargs):
        """
        绘制ADX指标图表
        
        Args:
            df: 包含ADX指标的DataFrame
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
        
        ax.set_ylabel('ADX指标')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        return ax 