#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
市场方向指标(ADX)

ADX指标用于衡量市场趋势的强度，不关注趋势的方向，只关注趋势的强弱程度
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple

from indicators.base_indicator import BaseIndicator
from indicators.common import crossover, crossunder
from utils.logger import get_logger

logger = get_logger(__name__)


class ADX(BaseIndicator):
    """
    市场方向指标(ADX)
    
    分类：趋势类指标
    描述：ADX指标用于衡量市场趋势的强度，不关注趋势的方向，只关注趋势的强弱程度
    """
    
    def __init__(self, period: int = 14, adx_period: int = 14):
        """
        初始化市场方向指标(ADX)
        
        Args:
            period: DMI计算周期，默认为14
            adx_period: ADX计算周期，默认为14
        """
        super().__init__(name="ADX", description="市场方向指标，用于衡量趋势强度")
        self.period = period
        self.adx_period = adx_period 

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
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算市场方向指标(ADX)
        
        Args:
            df: 包含OHLC数据的DataFrame
                必须包含以下列：
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                
        Returns:
            添加了ADX指标列的DataFrame
        """
        if df.empty:
            return df
            
        # 确保数据包含必要的列
        required_columns = ['high', 'low', 'close']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()
        
        # 计算高低价和收盘价的差值
        df_copy['high_low'] = df_copy['high'] - df_copy['low']
        df_copy['high_close'] = abs(df_copy['high'] - df_copy['close'].shift(1))
        df_copy['low_close'] = abs(df_copy['low'] - df_copy['close'].shift(1))
        
        # 计算真实波幅TR
        df_copy['TR'] = df_copy[['high_low', 'high_close', 'low_close']].max(axis=1)
        
        # 计算方向动量+DM和-DM
        df_copy['up_move'] = df_copy['high'] - df_copy['high'].shift(1)
        df_copy['down_move'] = df_copy['low'].shift(1) - df_copy['low']
        
        # +DM：今日最高价比昨日最高价高，且今日最低价比昨日最低价高
        df_copy['+DM'] = np.where(
            (df_copy['up_move'] > df_copy['down_move']) & (df_copy['up_move'] > 0),
            df_copy['up_move'],
            0
        )
        
        # -DM：今日最低价比昨日最低价低，且今日最高价比昨日最高价低
        df_copy['-DM'] = np.where(
            (df_copy['down_move'] > df_copy['up_move']) & (df_copy['down_move'] > 0),
            df_copy['down_move'],
            0
        )
        
        # 计算平滑值
        df_copy['TR_' + str(self.period)] = df_copy['TR'].rolling(window=self.period).sum()
        df_copy['+DM_' + str(self.period)] = df_copy['+DM'].rolling(window=self.period).sum()
        df_copy['-DM_' + str(self.period)] = df_copy['-DM'].rolling(window=self.period).sum()
        
        # 计算方向指标+DI和-DI
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
        self._result = df_copy[['PDI', 'MDI', 'ADX', 'ADXR']]
        
        return df_copy
        
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
        # 确保已计算ADX指标
        if not self.has_result():
            self.calculate(data)
        
        if self._result is None:
            return pd.Series(50.0, index=data.index)
        
        # 获取ADX相关值
        pdi = self._result['PDI']
        mdi = self._result['MDI']
        adx = self._result['ADX']
        
        # 初始基础分50分
        score = pd.Series(50.0, index=data.index)
        
        # 根据ADX值评分
        # ADX越高，趋势越强，加分越多
        score += (adx / 5)  # 最高可加20分
        
        # 根据DI差值给分
        # +DI > -DI，多头趋势，加分
        # -DI > +DI，空头趋势，减分
        di_diff = pdi - mdi
        score += di_diff / 5  # 最高可加减20分
        
        # 限制评分范围在0-100之间
        return np.clip(score, 0, 100)
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别ADX技术形态
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[str]: 识别出的形态列表
        """
        patterns = []
        
        # 确保已计算ADX指标
        if not self.has_result():
            self.calculate(data)
        
        if self._result is None:
            return patterns
        
        # 获取ADX相关值
        pdi = self._result['PDI']
        mdi = self._result['MDI']
        adx = self._result['ADX']
        
        # 最后一个有效的索引位置
        last_valid_idx = -1
        while last_valid_idx >= -len(adx) and pd.isna(adx.iloc[last_valid_idx]):
            last_valid_idx -= 1
        
        if last_valid_idx < -len(adx):
            return patterns
        
        # 1. DI交叉形态
        if last_valid_idx-1 >= -len(pdi) and pdi.iloc[last_valid_idx-1] < mdi.iloc[last_valid_idx-1] and pdi.iloc[last_valid_idx] > mdi.iloc[last_valid_idx]:
            patterns.append("DI金叉（+DI上穿-DI）")
        
        if last_valid_idx-1 >= -len(pdi) and pdi.iloc[last_valid_idx-1] > mdi.iloc[last_valid_idx-1] and pdi.iloc[last_valid_idx] < mdi.iloc[last_valid_idx]:
            patterns.append("DI死叉（-DI上穿+DI）")
        
        # 2. ADX趋势形态
        adx_value = adx.iloc[last_valid_idx]
        
        if adx_value > 50:
            patterns.append("极强趋势（ADX > 50）")
        elif adx_value > 40:
            patterns.append("很强趋势（ADX > 40）")
        elif adx_value > 25:
            patterns.append("强趋势（ADX > 25）")
        elif adx_value > 20:
            patterns.append("中等趋势（ADX > 20）")
        elif adx_value > 15:
            patterns.append("弱趋势（ADX > 15）")
        else:
            patterns.append("无趋势（ADX < 15）")
            
        # 3. ADX变化形态
        if last_valid_idx-1 >= -len(adx) and adx.iloc[last_valid_idx] > adx.iloc[last_valid_idx-1]:
            patterns.append("ADX上升（趋势增强）")
        elif last_valid_idx-1 >= -len(adx) and adx.iloc[last_valid_idx] < adx.iloc[last_valid_idx-1]:
            patterns.append("ADX下降（趋势减弱）")
            
        # 4. 综合趋势形态
        if pdi.iloc[last_valid_idx] > mdi.iloc[last_valid_idx]:
            if adx_value > 25:
                patterns.append("强多头趋势（+DI > -DI且ADX > 25）")
            else:
                patterns.append("弱多头趋势（+DI > -DI但ADX < 25）")
        else:
            if adx_value > 25:
                patterns.append("强空头趋势（-DI > +DI且ADX > 25）")
            else:
                patterns.append("弱空头趋势（-DI > +DI但ADX < 25）")
        
        return patterns
    
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