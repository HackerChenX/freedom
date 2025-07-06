#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
扩展技术指标计算模块

为反向验证框架添加P1重要指标的计算方法
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class ExtendedTechnicalIndicators:
    """扩展技术指标计算器"""
    
    def __init__(self):
        pass
    
    # P1重要指标计算方法
    
    def calculate_sar(self, data: pd.DataFrame, af_start: float = 0.02, af_increment: float = 0.02, af_max: float = 0.2) -> pd.DataFrame:
        """
        计算抛物线SAR指标
        
        Args:
            data: 包含OHLC数据的DataFrame
            af_start: 初始加速因子
            af_increment: 加速因子增量
            af_max: 最大加速因子
            
        Returns:
            包含SAR的DataFrame
        """
        high = data['high']
        low = data['low']
        close = data['close']
        
        # 初始化
        sar = pd.Series(index=data.index, dtype=float)
        af = af_start
        ep = high.iloc[0]  # 极值点
        trend = 1  # 1为上升趋势，-1为下降趋势
        
        sar.iloc[0] = low.iloc[0]
        
        for i in range(1, len(data)):
            if trend == 1:  # 上升趋势
                sar.iloc[i] = sar.iloc[i-1] + af * (ep - sar.iloc[i-1])
                
                # 检查是否转向
                if low.iloc[i] <= sar.iloc[i]:
                    trend = -1
                    sar.iloc[i] = ep
                    ep = low.iloc[i]
                    af = af_start
                else:
                    if high.iloc[i] > ep:
                        ep = high.iloc[i]
                        af = min(af + af_increment, af_max)
                    
                    # SAR不能高于前两天的最低价
                    if i >= 2:
                        sar.iloc[i] = min(sar.iloc[i], low.iloc[i-1], low.iloc[i-2])
                    elif i >= 1:
                        sar.iloc[i] = min(sar.iloc[i], low.iloc[i-1])
            
            else:  # 下降趋势
                sar.iloc[i] = sar.iloc[i-1] + af * (ep - sar.iloc[i-1])
                
                # 检查是否转向
                if high.iloc[i] >= sar.iloc[i]:
                    trend = 1
                    sar.iloc[i] = ep
                    ep = high.iloc[i]
                    af = af_start
                else:
                    if low.iloc[i] < ep:
                        ep = low.iloc[i]
                        af = min(af + af_increment, af_max)
                    
                    # SAR不能低于前两天的最高价
                    if i >= 2:
                        sar.iloc[i] = max(sar.iloc[i], high.iloc[i-1], high.iloc[i-2])
                    elif i >= 1:
                        sar.iloc[i] = max(sar.iloc[i], high.iloc[i-1])
        
        return pd.DataFrame({'SAR': sar})
    
    def calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        计算ADX指标
        
        Args:
            data: 包含OHLC数据的DataFrame
            period: 计算周期
            
        Returns:
            包含ADX、+DI、-DI的DataFrame
        """
        high = data['high']
        low = data['low']
        close = data['close']
        
        # 计算真实波幅TR
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # 计算方向移动
        dm_plus = high - high.shift(1)
        dm_minus = low.shift(1) - low
        
        # 只保留正值
        dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
        dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
        
        # 计算平滑的TR和DM
        atr = tr.rolling(window=period).mean()
        adm_plus = dm_plus.rolling(window=period).mean()
        adm_minus = dm_minus.rolling(window=period).mean()
        
        # 计算DI
        di_plus = 100 * adm_plus / atr
        di_minus = 100 * adm_minus / atr
        
        # 计算DX
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        
        # 计算ADX
        adx = dx.rolling(window=period).mean()
        
        return pd.DataFrame({
            'ADX': adx,
            'DI_PLUS': di_plus,
            'DI_MINUS': di_minus
        })
    
    def calculate_dmi(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        计算DMI指标（与ADX类似但包含更多信息）
        
        Args:
            data: 包含OHLC数据的DataFrame
            period: 计算周期
            
        Returns:
            包含PDI、MDI、ADX、ADXR的DataFrame
        """
        adx_data = self.calculate_adx(data, period)
        
        # 计算ADXR
        adxr = (adx_data['ADX'] + adx_data['ADX'].shift(period)) / 2
        
        return pd.DataFrame({
            'PDI': adx_data['DI_PLUS'],
            'MDI': adx_data['DI_MINUS'],
            'ADX': adx_data['ADX'],
            'ADXR': adxr
        })
    
    def calculate_trix(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        计算TRIX指标
        
        Args:
            data: 包含OHLC数据的DataFrame
            period: 计算周期
            
        Returns:
            包含TRIX、TRIX_SIGNAL的DataFrame
        """
        close = data['close']
        
        # 三重指数平滑
        ema1 = close.ewm(span=period).mean()
        ema2 = ema1.ewm(span=period).mean()
        ema3 = ema2.ewm(span=period).mean()
        
        # 计算TRIX
        trix = (ema3 - ema3.shift(1)) / ema3.shift(1) * 10000
        
        # 计算信号线
        trix_signal = trix.ewm(span=9).mean()
        
        return pd.DataFrame({
            'TRIX': trix,
            'TRIX_SIGNAL': trix_signal
        })
    
    def calculate_roc(self, data: pd.DataFrame, period: int = 12) -> pd.DataFrame:
        """
        计算ROC指标
        
        Args:
            data: 包含OHLC数据的DataFrame
            period: 计算周期
            
        Returns:
            包含ROC的DataFrame
        """
        close = data['close']
        
        # 计算ROC
        roc = (close - close.shift(period)) / close.shift(period) * 100
        
        # 计算ROC的移动平均
        roc_ma = roc.rolling(window=6).mean()
        
        return pd.DataFrame({
            'ROC': roc,
            'ROC_MA': roc_ma
        })
    
    def calculate_cmo(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        计算CMO指标
        
        Args:
            data: 包含OHLC数据的DataFrame
            period: 计算周期
            
        Returns:
            包含CMO的DataFrame
        """
        close = data['close']
        
        # 计算价格变化
        price_change = close - close.shift(1)
        
        # 分离上涨和下跌
        gains = price_change.where(price_change > 0, 0)
        losses = -price_change.where(price_change < 0, 0)
        
        # 计算期间内的总涨跌
        sum_gains = gains.rolling(window=period).sum()
        sum_losses = losses.rolling(window=period).sum()
        
        # 计算CMO
        cmo = 100 * (sum_gains - sum_losses) / (sum_gains + sum_losses)
        
        return pd.DataFrame({'CMO': cmo})
    
    def calculate_dma(self, data: pd.DataFrame, short_period: int = 10, long_period: int = 50, signal_period: int = 10) -> pd.DataFrame:
        """
        计算DMA指标
        
        Args:
            data: 包含OHLC数据的DataFrame
            short_period: 短期周期
            long_period: 长期周期
            signal_period: 信号线周期
            
        Returns:
            包含DMA、AMA的DataFrame
        """
        close = data['close']
        
        # 计算短期和长期移动平均
        ma_short = close.rolling(window=short_period).mean()
        ma_long = close.rolling(window=long_period).mean()
        
        # 计算DMA
        dma = ma_short - ma_long
        
        # 计算AMA（DMA的移动平均）
        ama = dma.rolling(window=signal_period).mean()
        
        return pd.DataFrame({
            'DMA': dma,
            'AMA': ama
        })
    
    def calculate_mtm(self, data: pd.DataFrame, period: int = 12) -> pd.DataFrame:
        """
        计算MTM指标
        
        Args:
            data: 包含OHLC数据的DataFrame
            period: 计算周期
            
        Returns:
            包含MTM、MTM_MA的DataFrame
        """
        close = data['close']
        
        # 计算MTM
        mtm = close - close.shift(period)
        
        # 计算MTM的移动平均
        mtm_ma = mtm.rolling(window=6).mean()
        
        return pd.DataFrame({
            'MTM': mtm,
            'MTM_MA': mtm_ma
        })
    
    # 通用计算方法
    def calculate_indicator(self, data: pd.DataFrame, indicator_name: str, **kwargs) -> pd.DataFrame:
        """
        通用指标计算方法
        
        Args:
            data: 包含OHLC数据的DataFrame
            indicator_name: 指标名称
            **kwargs: 指标参数
            
        Returns:
            计算结果DataFrame
        """
        method_name = f"calculate_{indicator_name.lower()}"
        if hasattr(self, method_name):
            method = getattr(self, method_name)
            return method(data, **kwargs)
        else:
            raise ValueError(f"不支持的指标: {indicator_name}")
    
    def calculate_stochrsi(self, data: pd.DataFrame, rsi_period: int = 14, stoch_period: int = 14) -> pd.DataFrame:
        """
        计算StochRSI指标

        Args:
            data: 包含OHLC数据的DataFrame
            rsi_period: RSI计算周期
            stoch_period: Stochastic计算周期

        Returns:
            包含STOCHRSI、STOCHRSI_K、STOCHRSI_D的DataFrame
        """
        close = data['close']

        # 先计算RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=rsi_period).mean()
        avg_loss = loss.rolling(window=rsi_period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # 计算StochRSI
        rsi_min = rsi.rolling(window=stoch_period).min()
        rsi_max = rsi.rolling(window=stoch_period).max()

        stochrsi = (rsi - rsi_min) / (rsi_max - rsi_min) * 100

        # 计算K和D线
        stochrsi_k = stochrsi.rolling(window=3).mean()
        stochrsi_d = stochrsi_k.rolling(window=3).mean()

        return pd.DataFrame({
            'STOCHRSI': stochrsi,
            'STOCHRSI_K': stochrsi_k,
            'STOCHRSI_D': stochrsi_d
        })

    def calculate_psy(self, data: pd.DataFrame, period: int = 12) -> pd.DataFrame:
        """
        计算PSY心理线指标

        Args:
            data: 包含OHLC数据的DataFrame
            period: 计算周期

        Returns:
            包含PSY的DataFrame
        """
        close = data['close']

        # 计算涨跌
        price_change = close.diff()
        up_days = (price_change > 0).astype(int)

        # 计算PSY
        psy = up_days.rolling(window=period).sum() / period * 100

        return pd.DataFrame({'PSY': psy})

    def calculate_wr(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        计算威廉指标WR

        Args:
            data: 包含OHLC数据的DataFrame
            period: 计算周期

        Returns:
            包含WR的DataFrame
        """
        high = data['high']
        low = data['low']
        close = data['close']

        # 计算WR
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()

        wr = (highest_high - close) / (highest_high - lowest_low) * (-100)

        return pd.DataFrame({'WR': wr})

    def calculate_bias(self, data: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        计算乖离率BIAS指标

        Args:
            data: 包含OHLC数据的DataFrame
            period: 计算周期

        Returns:
            包含BIAS的DataFrame
        """
        close = data['close']

        # 计算移动平均
        ma = close.rolling(window=period).mean()

        # 计算乖离率
        bias = (close - ma) / ma * 100

        return pd.DataFrame({'BIAS': bias})

    def calculate_vol(self, data: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        计算成交量指标VOL

        Args:
            data: 包含OHLC和volume数据的DataFrame
            period: 计算周期

        Returns:
            包含VOL、VOL_MA的DataFrame
        """
        volume = data['volume']

        # 计算成交量移动平均
        vol_ma = volume.rolling(window=period).mean()

        # 计算成交量比率
        vol_ratio = volume / vol_ma

        return pd.DataFrame({
            'VOL': volume,
            'VOL_MA': vol_ma,
            'VOL_RATIO': vol_ratio
        })

    def calculate_obv(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算能量潮指标OBV

        Args:
            data: 包含OHLC和volume数据的DataFrame

        Returns:
            包含OBV的DataFrame
        """
        close = data['close']
        volume = data['volume']

        # 计算价格变化
        price_change = close.diff()

        # 计算OBV
        obv = pd.Series(index=data.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]

        for i in range(1, len(data)):
            if price_change.iloc[i] > 0:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif price_change.iloc[i] < 0:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]

        return pd.DataFrame({'OBV': obv})

    def calculate_mfi(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        计算资金流量指标MFI

        Args:
            data: 包含OHLC和volume数据的DataFrame
            period: 计算周期

        Returns:
            包含MFI的DataFrame
        """
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']

        # 计算典型价格
        typical_price = (high + low + close) / 3

        # 计算资金流量
        money_flow = typical_price * volume

        # 计算正负资金流量
        price_change = typical_price.diff()
        positive_flow = money_flow.where(price_change > 0, 0)
        negative_flow = money_flow.where(price_change < 0, 0)

        # 计算资金流量比率
        positive_sum = positive_flow.rolling(window=period).sum()
        negative_sum = negative_flow.rolling(window=period).sum()

        money_ratio = positive_sum / negative_sum
        mfi = 100 - (100 / (1 + money_ratio))

        return pd.DataFrame({'MFI': mfi})

    def calculate_emv(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        计算简易波动指标EMV

        Args:
            data: 包含OHLC和volume数据的DataFrame
            period: 计算周期

        Returns:
            包含EMV的DataFrame
        """
        high = data['high']
        low = data['low']
        volume = data['volume']

        # 计算距离移动
        distance_moved = ((high + low) / 2).diff()

        # 计算高低价差
        high_low = high - low

        # 计算EMV
        emv_raw = distance_moved * volume / high_low
        emv = emv_raw.rolling(window=period).mean()

        return pd.DataFrame({'EMV': emv})

    def calculate_cci(self, data: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        计算顺势指标CCI

        Args:
            data: 包含OHLC数据的DataFrame
            period: 计算周期

        Returns:
            包含CCI的DataFrame
        """
        high = data['high']
        low = data['low']
        close = data['close']

        # 计算典型价格
        typical_price = (high + low + close) / 3

        # 计算移动平均
        sma = typical_price.rolling(window=period).mean()

        # 计算平均绝对偏差
        mad = typical_price.rolling(window=period).apply(
            lambda x: abs(x - x.mean()).mean()
        )

        # 计算CCI
        cci = (typical_price - sma) / (0.015 * mad)

        return pd.DataFrame({'CCI': cci})

    def calculate_momentum(self, data: pd.DataFrame, period: int = 10) -> pd.DataFrame:
        """
        计算动量指标MOMENTUM

        Args:
            data: 包含OHLC数据的DataFrame
            period: 计算周期

        Returns:
            包含MOMENTUM的DataFrame
        """
        close = data['close']

        # 计算动量
        momentum = close - close.shift(period)

        return pd.DataFrame({'MOMENTUM': momentum})

    def calculate_vosc(self, data: pd.DataFrame, short_period: int = 12, long_period: int = 26) -> pd.DataFrame:
        """
        计算成交量震荡指标VOSC

        Args:
            data: 包含volume数据的DataFrame
            short_period: 短期周期
            long_period: 长期周期

        Returns:
            包含VOSC的DataFrame
        """
        volume = data['volume']

        # 计算短期和长期成交量移动平均
        short_ma = volume.rolling(window=short_period).mean()
        long_ma = volume.rolling(window=long_period).mean()

        # 计算VOSC
        vosc = (short_ma - long_ma) / long_ma * 100

        return pd.DataFrame({'VOSC': vosc})

    def calculate_vr(self, data: pd.DataFrame, period: int = 26) -> pd.DataFrame:
        """
        计算成交量比率VR

        Args:
            data: 包含OHLC和volume数据的DataFrame
            period: 计算周期

        Returns:
            包含VR的DataFrame
        """
        close = data['close']
        volume = data['volume']

        # 计算价格变化
        price_change = close.diff()

        # 分类成交量
        up_volume = volume.where(price_change > 0, 0)
        down_volume = volume.where(price_change < 0, 0)
        equal_volume = volume.where(price_change == 0, 0)

        # 计算VR
        up_sum = up_volume.rolling(window=period).sum()
        down_sum = down_volume.rolling(window=period).sum()
        equal_sum = equal_volume.rolling(window=period).sum()

        vr = (up_sum + equal_sum / 2) / (down_sum + equal_sum / 2) * 100

        return pd.DataFrame({'VR': vr})

    def calculate_pvt(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算价量趋势指标PVT

        Args:
            data: 包含OHLC和volume数据的DataFrame

        Returns:
            包含PVT的DataFrame
        """
        close = data['close']
        volume = data['volume']

        # 计算价格变化率
        price_change_rate = close.pct_change()

        # 计算PVT
        pvt = (price_change_rate * volume).cumsum()

        return pd.DataFrame({'PVT': pvt})

    def calculate_chaikin(self, data: pd.DataFrame, period: int = 21) -> pd.DataFrame:
        """
        计算佳庆指标Chaikin

        Args:
            data: 包含OHLC和volume数据的DataFrame
            period: 计算周期

        Returns:
            包含CHAIKIN的DataFrame
        """
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']

        # 计算累积/派发线
        clv = ((close - low) - (high - close)) / (high - low)
        ad_line = (clv * volume).cumsum()

        # 计算Chaikin震荡器
        chaikin = ad_line.rolling(window=3).mean() - ad_line.rolling(window=10).mean()

        return pd.DataFrame({'CHAIKIN': chaikin})

    def calculate_ad(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算累积/派发线AD

        Args:
            data: 包含OHLC和volume数据的DataFrame

        Returns:
            包含AD的DataFrame
        """
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']

        # 计算累积/派发线
        clv = ((close - low) - (high - close)) / (high - low)
        clv = clv.fillna(0)  # 处理high=low的情况

        ad = (clv * volume).cumsum()

        return pd.DataFrame({'AD': ad})

    # P3专业指标实现

    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        计算平均真实波幅ATR

        Args:
            data: 包含OHLC数据的DataFrame
            period: 计算周期

        Returns:
            包含ATR的DataFrame
        """
        high = data['high']
        low = data['low']
        close = data['close']

        # 计算真实波幅TR
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # 计算ATR
        atr = tr.rolling(window=period).mean()

        return pd.DataFrame({'ATR': atr, 'TR': tr})

    def calculate_kc(self, data: pd.DataFrame, period: int = 20, multiplier: float = 2.0) -> pd.DataFrame:
        """
        计算肯特纳通道KC

        Args:
            data: 包含OHLC数据的DataFrame
            period: 计算周期
            multiplier: ATR倍数

        Returns:
            包含KC_UPPER、KC_MIDDLE、KC_LOWER的DataFrame
        """
        high = data['high']
        low = data['low']
        close = data['close']

        # 计算中线（EMA）
        kc_middle = close.ewm(span=period).mean()

        # 计算ATR
        atr_data = self.calculate_atr(data, period)
        atr = atr_data['ATR']

        # 计算上下轨
        kc_upper = kc_middle + (multiplier * atr)
        kc_lower = kc_middle - (multiplier * atr)

        return pd.DataFrame({
            'KC_UPPER': kc_upper,
            'KC_MIDDLE': kc_middle,
            'KC_LOWER': kc_lower
        })

    def calculate_vortex(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        计算涡流指标Vortex

        Args:
            data: 包含OHLC数据的DataFrame
            period: 计算周期

        Returns:
            包含VI_PLUS、VI_MINUS的DataFrame
        """
        high = data['high']
        low = data['low']
        close = data['close']

        # 计算VM+和VM-
        vm_plus = abs(high - low.shift(1))
        vm_minus = abs(low - high.shift(1))

        # 计算TR
        atr_data = self.calculate_atr(data, period)
        tr = atr_data['TR']

        # 计算VI+和VI-
        vi_plus = vm_plus.rolling(window=period).sum() / tr.rolling(window=period).sum()
        vi_minus = vm_minus.rolling(window=period).sum() / tr.rolling(window=period).sum()

        return pd.DataFrame({
            'VI_PLUS': vi_plus,
            'VI_MINUS': vi_minus
        })

    def calculate_aroon(self, data: pd.DataFrame, period: int = 25) -> pd.DataFrame:
        """
        计算阿隆指标Aroon

        Args:
            data: 包含OHLC数据的DataFrame
            period: 计算周期

        Returns:
            包含AROON_UP、AROON_DOWN、AROON_OSC的DataFrame
        """
        high = data['high']
        low = data['low']

        # 计算Aroon Up和Aroon Down
        aroon_up = high.rolling(window=period).apply(
            lambda x: (period - x.argmax()) / period * 100
        )
        aroon_down = low.rolling(window=period).apply(
            lambda x: (period - x.argmin()) / period * 100
        )

        # 计算Aroon震荡器
        aroon_osc = aroon_up - aroon_down

        return pd.DataFrame({
            'AROON_UP': aroon_up,
            'AROON_DOWN': aroon_down,
            'AROON_OSC': aroon_osc
        })

    def calculate_ichimoku(self, data: pd.DataFrame,
                          tenkan_period: int = 9,
                          kijun_period: int = 26,
                          senkou_period: int = 52) -> pd.DataFrame:
        """
        计算一目均衡表Ichimoku

        Args:
            data: 包含OHLC数据的DataFrame
            tenkan_period: 转换线周期
            kijun_period: 基准线周期
            senkou_period: 先行带周期

        Returns:
            包含TENKAN、KIJUN、SENKOU_A、SENKOU_B的DataFrame
        """
        high = data['high']
        low = data['low']

        # 转换线（Tenkan-sen）
        tenkan = (high.rolling(window=tenkan_period).max() +
                 low.rolling(window=tenkan_period).min()) / 2

        # 基准线（Kijun-sen）
        kijun = (high.rolling(window=kijun_period).max() +
                low.rolling(window=kijun_period).min()) / 2

        # 先行带A（Senkou Span A）
        senkou_a = ((tenkan + kijun) / 2).shift(kijun_period)

        # 先行带B（Senkou Span B）
        senkou_b = ((high.rolling(window=senkou_period).max() +
                    low.rolling(window=senkou_period).min()) / 2).shift(kijun_period)

        return pd.DataFrame({
            'TENKAN': tenkan,
            'KIJUN': kijun,
            'SENKOU_A': senkou_a,
            'SENKOU_B': senkou_b
        })

    def calculate_wma(self, data: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        计算加权移动平均WMA

        Args:
            data: 包含OHLC数据的DataFrame
            period: 计算周期

        Returns:
            包含WMA的DataFrame
        """
        close = data['close']

        # 计算权重
        weights = np.arange(1, period + 1)

        # 计算WMA
        wma = close.rolling(window=period).apply(
            lambda x: np.dot(x, weights) / weights.sum()
        )

        return pd.DataFrame({'WMA': wma})

    def calculate_vix(self, data: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        计算波动率指数VIX（简化版本）

        Args:
            data: 包含OHLC数据的DataFrame
            period: 计算周期

        Returns:
            包含VIX的DataFrame
        """
        close = data['close']

        # 计算收益率
        returns = close.pct_change()

        # 计算滚动标准差作为波动率
        volatility = returns.rolling(window=period).std() * np.sqrt(252) * 100

        return pd.DataFrame({'VIX': volatility})

    def calculate_volume_ratio(self, data: pd.DataFrame, short_period: int = 5, long_period: int = 20) -> pd.DataFrame:
        """
        计算成交量比率指标

        Args:
            data: 包含volume数据的DataFrame
            short_period: 短期周期
            long_period: 长期周期

        Returns:
            包含VOLUME_RATIO的DataFrame
        """
        volume = data['volume']

        # 计算短期和长期成交量平均
        short_avg = volume.rolling(window=short_period).mean()
        long_avg = volume.rolling(window=long_period).mean()

        # 计算成交量比率
        volume_ratio = short_avg / long_avg

        return pd.DataFrame({'VOLUME_RATIO': volume_ratio})

    def calculate_enhanced_cci(self, data: pd.DataFrame, period: int = 20, factor: float = 0.015) -> pd.DataFrame:
        """
        计算增强版CCI指标

        Args:
            data: 包含OHLC数据的DataFrame
            period: 计算周期
            factor: 计算因子

        Returns:
            包含ENHANCED_CCI的DataFrame
        """
        high = data['high']
        low = data['low']
        close = data['close']

        # 计算典型价格
        typical_price = (high + low + close) / 3

        # 计算移动平均
        sma = typical_price.rolling(window=period).mean()

        # 计算平均绝对偏差
        mad = typical_price.rolling(window=period).apply(
            lambda x: abs(x - x.mean()).mean()
        )

        # 计算增强版CCI
        enhanced_cci = (typical_price - sma) / (factor * mad)

        # 添加平滑处理
        enhanced_cci_smooth = enhanced_cci.rolling(window=3).mean()

        return pd.DataFrame({
            'ENHANCED_CCI': enhanced_cci,
            'ENHANCED_CCI_SMOOTH': enhanced_cci_smooth
        })

    def calculate_enhanced_dmi(self, data: pd.DataFrame, period: int = 14, smooth_period: int = 3) -> pd.DataFrame:
        """
        计算增强版DMI指标

        Args:
            data: 包含OHLC数据的DataFrame
            period: 计算周期
            smooth_period: 平滑周期

        Returns:
            包含ENHANCED_DI_PLUS、ENHANCED_DI_MINUS、ENHANCED_DX的DataFrame
        """
        high = data['high']
        low = data['low']
        close = data['close']

        # 计算价格变动
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        # 计算+DM和-DM
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)

        # 计算TR
        atr_data = self.calculate_atr(data, period)
        tr = atr_data['TR']

        # 计算+DI和-DI
        plus_di = 100 * (plus_dm.rolling(window=period).sum() / tr.rolling(window=period).sum())
        minus_di = 100 * (minus_dm.rolling(window=period).sum() / tr.rolling(window=period).sum())

        # 计算DX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)

        # 增强版：添加平滑处理
        enhanced_plus_di = plus_di.rolling(window=smooth_period).mean()
        enhanced_minus_di = minus_di.rolling(window=smooth_period).mean()
        enhanced_dx = dx.rolling(window=smooth_period).mean()

        return pd.DataFrame({
            'ENHANCED_DI_PLUS': enhanced_plus_di,
            'ENHANCED_DI_MINUS': enhanced_minus_di,
            'ENHANCED_DX': enhanced_dx
        })

    # P4 ZXM系列指标实现

    def calculate_zxm_daily_macd(self, data: pd.DataFrame, short_period: int = 12,
                                long_period: int = 26, mid_period: int = 9) -> pd.DataFrame:
        """
        计算ZXM日线MACD指标

        Args:
            data: 包含OHLC数据的DataFrame
            short_period: 短期EMA周期
            long_period: 长期EMA周期
            mid_period: DEA周期

        Returns:
            包含ZXM_MACD、ZXM_DIF、ZXM_DEA、ZXM_XG的DataFrame
        """
        close = data['close']

        # 计算EMA
        ema12 = close.ewm(span=short_period, adjust=False).mean()
        ema26 = close.ewm(span=long_period, adjust=False).mean()

        # 计算DIF和DEA
        dif = ema12 - ema26
        dea = dif.ewm(span=mid_period, adjust=False).mean()
        macd = 2 * (dif - dea)

        # ZXM买点信号：MACD < 0.9
        xg = macd < 0.9

        return pd.DataFrame({
            'ZXM_MACD': macd,
            'ZXM_DIF': dif,
            'ZXM_DEA': dea,
            'ZXM_XG': xg
        })

    def calculate_zxm_turnover(self, data: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
        """
        计算ZXM换手率指标

        Args:
            data: 包含turnover_rate数据的DataFrame
            threshold: 换手率阈值

        Returns:
            包含ZXM_TURNOVER、ZXM_XG的DataFrame
        """
        # 如果没有换手率数据，使用成交量估算
        if 'turnover_rate' in data.columns:
            turnover = data['turnover_rate']
        else:
            # 简化估算：基于成交量变化
            volume_ma = data['volume'].rolling(window=20).mean()
            turnover = (data['volume'] / volume_ma) * 0.5  # 简化估算

        # ZXM买点信号：换手率 > 0.7%
        xg = turnover > threshold

        return pd.DataFrame({
            'ZXM_TURNOVER': turnover,
            'ZXM_XG': xg
        })

    def calculate_zxm_volume_shrink(self, data: pd.DataFrame, period: int = 2) -> pd.DataFrame:
        """
        计算ZXM缩量指标

        Args:
            data: 包含volume数据的DataFrame
            period: 均量周期

        Returns:
            包含ZXM_VOL_RATIO、ZXM_XG的DataFrame
        """
        volume = data['volume']

        # 计算均量
        ma_vol = volume.rolling(window=period).mean()

        # 计算量比
        vol_ratio = volume / ma_vol

        # ZXM买点信号：量比 < 0.9
        xg = vol_ratio < 0.9

        return pd.DataFrame({
            'ZXM_VOL_RATIO': vol_ratio,
            'ZXM_XG': xg
        })

    def calculate_zxm_ma_callback(self, data: pd.DataFrame, ma_period: int = 20) -> pd.DataFrame:
        """
        计算ZXM均线回调指标

        Args:
            data: 包含close数据的DataFrame
            ma_period: 均线周期

        Returns:
            包含ZXM_MA、ZXM_CALLBACK、ZXM_XG的DataFrame
        """
        close = data['close']

        # 计算移动平均
        ma = close.rolling(window=ma_period).mean()

        # 计算回调幅度
        callback = (close - ma) / ma * 100

        # ZXM买点信号：价格接近均线（回调幅度在-5%到+2%之间）
        xg = (callback >= -5) & (callback <= 2)

        return pd.DataFrame({
            'ZXM_MA': ma,
            'ZXM_CALLBACK': callback,
            'ZXM_XG': xg
        })

    def calculate_zxm_bs_absorb(self, data: pd.DataFrame, period: int = 9) -> pd.DataFrame:
        """
        计算ZXM主力吸筹指标

        Args:
            data: 包含OHLCV数据的DataFrame
            period: 计算周期

        Returns:
            包含ZXM_ABSORB、ZXM_BUY、ZXM_SELL的DataFrame
        """
        close = data['close']
        volume = data['volume']

        # 计算价格变化率
        price_change = close.pct_change()

        # 计算成交量比率
        volume_ma = volume.rolling(window=period).mean()
        volume_ratio = volume / volume_ma

        # 吸筹信号：价格小幅波动，成交量放大
        absorb_signal = (price_change.abs() < 0.02) & (volume_ratio > 1.2)

        # 买入信号：价格上涨，成交量放大
        buy_signal = (price_change > 0.01) & (volume_ratio > 1.5)

        # 卖出信号：价格下跌，成交量放大
        sell_signal = (price_change < -0.01) & (volume_ratio > 1.5)

        return pd.DataFrame({
            'ZXM_ABSORB': absorb_signal,
            'ZXM_BUY': buy_signal,
            'ZXM_SELL': sell_signal,
            'ZXM_VOLUME_RATIO': volume_ratio
        })

    def calculate_zxm_amplitude_elasticity(self, data: pd.DataFrame, threshold: float = 8.1,
                                          period: int = 120) -> pd.DataFrame:
        """
        计算ZXM振幅弹性指标

        Args:
            data: 包含OHLC数据的DataFrame
            threshold: 振幅阈值
            period: 统计周期

        Returns:
            包含ZXM_AMPLITUDE、ZXM_XG的DataFrame
        """
        high = data['high']
        low = data['low']

        # 计算日振幅
        amplitude = 100 * (high - low) / low

        # 计算振幅超过阈值的情况
        a1 = amplitude > threshold

        # 计算period日内是否有超过1次振幅大于阈值
        xg = a1.rolling(window=period).sum() > 1

        return pd.DataFrame({
            'ZXM_AMPLITUDE': amplitude,
            'ZXM_A1': a1,
            'ZXM_XG': xg
        })

    def calculate_zxm_rise_elasticity(self, data: pd.DataFrame, threshold: float = 1.07,
                                     period: int = 80) -> pd.DataFrame:
        """
        计算ZXM涨幅弹性指标

        Args:
            data: 包含close数据的DataFrame
            threshold: 涨幅阈值
            period: 统计周期

        Returns:
            包含ZXM_RISE_RATIO、ZXM_XG的DataFrame
        """
        close = data['close']

        # 计算日涨幅
        rise_ratio = close / close.shift(1)

        # 计算涨幅超过阈值的情况
        a1 = rise_ratio > threshold

        # 计算period日内是否有涨幅大于阈值
        xg = a1.rolling(window=period).sum() > 0

        return pd.DataFrame({
            'ZXM_RISE_RATIO': rise_ratio,
            'ZXM_A1': a1,
            'ZXM_XG': xg
        })

    def calculate_zxm_elasticity(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算ZXM综合弹性指标

        Args:
            data: 包含OHLC数据的DataFrame

        Returns:
            包含ZXM_ELASTICITY、ZXM_RATIO的DataFrame
        """
        close = data['close']
        low = data['low']

        # 计算弹性比率
        low_20 = low.rolling(window=20).min()
        elasticity_ratio = close / low_20

        # 计算反弹强度
        bounce_strength = (close - low_20) / low_20

        # 弹性买点信号
        buy_signal = (elasticity_ratio > 1.1) & (bounce_strength > 0.05)

        return pd.DataFrame({
            'ZXM_ELASTICITY': elasticity_ratio,
            'ZXM_BOUNCE': bounce_strength,
            'ZXM_BUY_SIGNAL': buy_signal
        })

    def calculate_zxm_bounce_detector(self, data: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        计算ZXM反弹检测指标

        Args:
            data: 包含OHLC数据的DataFrame
            period: 检测周期

        Returns:
            包含ZXM_BOUNCE_SIGNAL的DataFrame
        """
        close = data['close']
        low = data['low']

        # 计算最低价
        low_min = low.rolling(window=period).min()

        # 计算反弹幅度
        bounce_ratio = (close - low_min) / low_min

        # 反弹信号：从低点反弹超过5%
        bounce_signal = bounce_ratio > 0.05

        return pd.DataFrame({
            'ZXM_BOUNCE_RATIO': bounce_ratio,
            'ZXM_BOUNCE_SIGNAL': bounce_signal
        })

    def calculate_zxm_elasticity_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算ZXM弹性评分指标

        Args:
            data: 包含OHLC数据的DataFrame

        Returns:
            包含ZXM_ELASTICITY_SCORE的DataFrame
        """
        # 计算振幅弹性
        amplitude_result = self.calculate_zxm_amplitude_elasticity(data)

        # 计算涨幅弹性
        rise_result = self.calculate_zxm_rise_elasticity(data)

        # 计算综合弹性
        elasticity_result = self.calculate_zxm_elasticity(data)

        # 综合评分
        score = pd.Series(50, index=data.index)  # 基础分50

        # 振幅弹性加分
        score[amplitude_result['ZXM_XG']] += 20

        # 涨幅弹性加分
        score[rise_result['ZXM_XG']] += 20

        # 弹性买点加分
        score[elasticity_result['ZXM_BUY_SIGNAL']] += 30

        return pd.DataFrame({'ZXM_ELASTICITY_SCORE': score})

    def calculate_zxm_buypoint_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算ZXM买点评分指标

        Args:
            data: 包含OHLCV数据的DataFrame

        Returns:
            包含ZXM_BUYPOINT_SCORE的DataFrame
        """
        # 计算各个买点指标
        macd_result = self.calculate_zxm_daily_macd(data)
        turnover_result = self.calculate_zxm_turnover(data)
        volume_result = self.calculate_zxm_volume_shrink(data)
        ma_result = self.calculate_zxm_ma_callback(data)

        # 综合评分
        score = pd.Series(50, index=data.index)  # 基础分50

        # MACD买点加分
        score[macd_result['ZXM_XG']] += 25

        # 换手率买点加分
        score[turnover_result['ZXM_XG']] += 20

        # 缩量买点加分
        score[volume_result['ZXM_XG']] += 15

        # 均线回调买点加分
        score[ma_result['ZXM_XG']] += 20

        return pd.DataFrame({'ZXM_BUYPOINT_SCORE': score})

    def calculate_zxm_stock_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算ZXM股票综合评分指标

        Args:
            data: 包含OHLCV数据的DataFrame

        Returns:
            包含ZXM_STOCK_SCORE的DataFrame
        """
        # 计算弹性评分
        elasticity_score = self.calculate_zxm_elasticity_score(data)

        # 计算买点评分
        buypoint_score = self.calculate_zxm_buypoint_score(data)

        # 综合评分（加权平均）
        stock_score = (elasticity_score['ZXM_ELASTICITY_SCORE'] * 0.4 +
                      buypoint_score['ZXM_BUYPOINT_SCORE'] * 0.6)

        return pd.DataFrame({'ZXM_STOCK_SCORE': stock_score})

    def calculate_zxm_daily_trend_up(self, data: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        计算ZXM日线上升趋势指标

        Args:
            data: 包含close数据的DataFrame
            period: 趋势判断周期

        Returns:
            包含ZXM_DAILY_TREND的DataFrame
        """
        close = data['close']

        # 计算移动平均
        ma = close.rolling(window=period).mean()

        # 计算趋势方向
        ma_slope = ma.diff(5)  # 5日斜率

        # 上升趋势：价格在均线上方且均线上升
        trend_up = (close > ma) & (ma_slope > 0)

        return pd.DataFrame({
            'ZXM_DAILY_MA': ma,
            'ZXM_DAILY_SLOPE': ma_slope,
            'ZXM_DAILY_TREND': trend_up
        })

    def calculate_zxm_weekly_trend_up(self, data: pd.DataFrame, period: int = 10) -> pd.DataFrame:
        """
        计算ZXM周线上升趋势指标

        Args:
            data: 包含close数据的DataFrame
            period: 趋势判断周期

        Returns:
            包含ZXM_WEEKLY_TREND的DataFrame
        """
        close = data['close']

        # 模拟周线数据（每5个交易日为一周）
        weekly_close = close.rolling(window=5).mean()

        # 计算周线移动平均
        weekly_ma = weekly_close.rolling(window=period).mean()

        # 计算趋势方向
        weekly_slope = weekly_ma.diff(2)  # 2周斜率

        # 上升趋势
        trend_up = (weekly_close > weekly_ma) & (weekly_slope > 0)

        return pd.DataFrame({
            'ZXM_WEEKLY_MA': weekly_ma,
            'ZXM_WEEKLY_SLOPE': weekly_slope,
            'ZXM_WEEKLY_TREND': trend_up
        })

    def calculate_zxm_monthly_kdj_trend_up(self, data: pd.DataFrame,
                                          k_period: int = 9, d_period: int = 3) -> pd.DataFrame:
        """
        计算ZXM月线KDJ上升趋势指标

        Args:
            data: 包含OHLC数据的DataFrame
            k_period: K值计算周期
            d_period: D值计算周期

        Returns:
            包含ZXM_MONTHLY_KDJ_TREND的DataFrame
        """
        high = data['high']
        low = data['low']
        close = data['close']

        # 模拟月线数据（每20个交易日为一月）
        monthly_high = high.rolling(window=20).max()
        monthly_low = low.rolling(window=20).min()
        monthly_close = close.rolling(window=20).mean()

        # 计算KDJ
        lowest_low = monthly_low.rolling(window=k_period).min()
        highest_high = monthly_high.rolling(window=k_period).max()

        rsv = (monthly_close - lowest_low) / (highest_high - lowest_low) * 100
        k = rsv.ewm(alpha=1/d_period).mean()
        d = k.ewm(alpha=1/d_period).mean()
        j = 3 * k - 2 * d

        # KDJ上升趋势：K>D且K值上升
        kdj_trend_up = (k > d) & (k > k.shift(1))

        return pd.DataFrame({
            'ZXM_MONTHLY_K': k,
            'ZXM_MONTHLY_D': d,
            'ZXM_MONTHLY_J': j,
            'ZXM_MONTHLY_KDJ_TREND': kdj_trend_up
        })

    # P5系统分析指标实现

    def calculate_system_performance_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算系统性能评分指标

        Args:
            data: 包含OHLCV数据的DataFrame

        Returns:
            包含SYSTEM_PERFORMANCE_SCORE的DataFrame
        """
        close = data['close']
        volume = data['volume']

        # 计算价格表现评分
        price_return = close.pct_change(20)  # 20日收益率
        price_score = np.where(price_return > 0.1, 100,
                              np.where(price_return > 0.05, 80,
                                      np.where(price_return > 0, 60,
                                              np.where(price_return > -0.05, 40, 20))))

        # 计算成交量活跃度评分
        volume_ma = volume.rolling(window=20).mean()
        volume_ratio = volume / volume_ma
        volume_score = np.where(volume_ratio > 2, 100,
                               np.where(volume_ratio > 1.5, 80,
                                       np.where(volume_ratio > 1, 60,
                                               np.where(volume_ratio > 0.5, 40, 20))))

        # 计算波动率评分
        volatility = close.rolling(window=20).std() / close.rolling(window=20).mean()
        volatility_score = np.where(volatility < 0.02, 100,
                                   np.where(volatility < 0.04, 80,
                                           np.where(volatility < 0.06, 60,
                                                   np.where(volatility < 0.08, 40, 20))))

        # 综合评分
        performance_score = (price_score * 0.5 + volume_score * 0.3 + volatility_score * 0.2)

        return pd.DataFrame({'SYSTEM_PERFORMANCE_SCORE': performance_score})

    def calculate_market_sentiment_index(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算市场情绪指数

        Args:
            data: 包含OHLCV数据的DataFrame

        Returns:
            包含MARKET_SENTIMENT_INDEX的DataFrame
        """
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data['volume']

        # 计算价格动量
        momentum = close.pct_change(5)

        # 计算成交量动量
        volume_momentum = volume.pct_change(5)

        # 计算振幅
        amplitude = (high - low) / close

        # 计算情绪指数
        sentiment_raw = (momentum * 50 + volume_momentum * 30 + amplitude * 20)

        # 标准化到0-100区间
        sentiment_index = 50 + sentiment_raw * 100
        sentiment_index = np.clip(sentiment_index, 0, 100)

        return pd.DataFrame({'MARKET_SENTIMENT_INDEX': sentiment_index})

    def calculate_risk_assessment_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算风险评估评分

        Args:
            data: 包含OHLCV数据的DataFrame

        Returns:
            包含RISK_ASSESSMENT_SCORE的DataFrame
        """
        close = data['close']
        high = data['high']
        low = data['low']

        # 计算价格风险（基于波动率）
        price_volatility = close.rolling(window=20).std() / close.rolling(window=20).mean()
        price_risk = np.where(price_volatility > 0.08, 80,
                             np.where(price_volatility > 0.06, 60,
                                     np.where(price_volatility > 0.04, 40,
                                             np.where(price_volatility > 0.02, 20, 10))))

        # 计算下跌风险
        max_drawdown = (close / close.rolling(window=20).max() - 1) * 100
        drawdown_risk = np.where(max_drawdown < -20, 80,
                                np.where(max_drawdown < -15, 60,
                                        np.where(max_drawdown < -10, 40,
                                                np.where(max_drawdown < -5, 20, 10))))

        # 计算流动性风险（基于振幅）
        amplitude = (high - low) / close
        liquidity_risk = np.where(amplitude > 0.1, 60,
                                 np.where(amplitude > 0.08, 40,
                                         np.where(amplitude > 0.06, 20, 10)))

        # 综合风险评分（分数越高风险越大）
        risk_score = (price_risk * 0.4 + drawdown_risk * 0.4 + liquidity_risk * 0.2)

        return pd.DataFrame({'RISK_ASSESSMENT_SCORE': risk_score})

    def calculate_trend_strength_indicator(self, data: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        计算趋势强度指标

        Args:
            data: 包含OHLC数据的DataFrame
            period: 计算周期

        Returns:
            包含TREND_STRENGTH的DataFrame
        """
        close = data['close']

        # 计算移动平均
        ma = close.rolling(window=period).mean()

        # 计算趋势方向
        ma_slope = ma.diff(5)

        # 计算价格与均线的偏离度
        deviation = abs(close - ma) / ma

        # 计算趋势一致性
        price_above_ma = (close > ma).rolling(window=10).sum() / 10

        # 计算趋势强度
        trend_strength = (abs(ma_slope) * 100 + (1 - deviation) * 50 + price_above_ma * 50) / 2
        trend_strength = np.clip(trend_strength, 0, 100)

        return pd.DataFrame({'TREND_STRENGTH': trend_strength})

    def calculate_momentum_oscillator(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        计算动量震荡器

        Args:
            data: 包含close数据的DataFrame
            period: 计算周期

        Returns:
            包含MOMENTUM_OSCILLATOR的DataFrame
        """
        close = data['close']

        # 计算价格动量
        momentum = close - close.shift(period)

        # 计算动量的移动平均
        momentum_ma = momentum.rolling(window=period).mean()

        # 计算动量震荡器
        momentum_std = momentum.rolling(window=period).std()
        oscillator = (momentum - momentum_ma) / momentum_std

        # 标准化到-100到100区间
        oscillator = np.clip(oscillator * 50, -100, 100)

        return pd.DataFrame({'MOMENTUM_OSCILLATOR': oscillator})

    def calculate_volatility_index(self, data: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        计算波动率指数

        Args:
            data: 包含OHLC数据的DataFrame
            period: 计算周期

        Returns:
            包含VOLATILITY_INDEX的DataFrame
        """
        close = data['close']
        high = data['high']
        low = data['low']

        # 计算真实波幅
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # 计算平均真实波幅
        atr = tr.rolling(window=period).mean()

        # 计算收益率波动率
        returns = close.pct_change()
        return_volatility = returns.rolling(window=period).std() * np.sqrt(252)

        # 计算波动率指数
        volatility_index = (atr / close * 100 + return_volatility * 100) / 2

        return pd.DataFrame({'VOLATILITY_INDEX': volatility_index})

    def calculate_liquidity_indicator(self, data: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        计算流动性指标

        Args:
            data: 包含OHLCV数据的DataFrame
            period: 计算周期

        Returns:
            包含LIQUIDITY_INDICATOR的DataFrame
        """
        close = data['close']
        volume = data['volume']
        high = data['high']
        low = data['low']

        # 计算成交量比率
        volume_ma = volume.rolling(window=period).mean()
        volume_ratio = volume / volume_ma

        # 计算价格影响（振幅与成交量的关系）
        amplitude = (high - low) / close
        price_impact = amplitude / (volume_ratio + 0.01)  # 避免除零

        # 计算流动性指标（流动性越好，指标值越高）
        liquidity = 100 / (1 + price_impact * 10)

        return pd.DataFrame({'LIQUIDITY_INDICATOR': liquidity})

    def calculate_market_efficiency_ratio(self, data: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        计算市场效率比率

        Args:
            data: 包含close数据的DataFrame
            period: 计算周期

        Returns:
            包含MARKET_EFFICIENCY_RATIO的DataFrame
        """
        close = data['close']

        # 计算净价格变化
        net_change = abs(close - close.shift(period))

        # 计算总价格变化
        daily_changes = abs(close.diff())
        total_change = daily_changes.rolling(window=period).sum()

        # 计算效率比率
        efficiency_ratio = net_change / (total_change + 0.01)  # 避免除零
        efficiency_ratio = np.clip(efficiency_ratio, 0, 1)

        return pd.DataFrame({'MARKET_EFFICIENCY_RATIO': efficiency_ratio})

    def calculate_composite_momentum_index(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算复合动量指数

        Args:
            data: 包含OHLCV数据的DataFrame

        Returns:
            包含COMPOSITE_MOMENTUM_INDEX的DataFrame
        """
        close = data['close']
        volume = data['volume']

        # 计算价格动量
        price_momentum_5 = close.pct_change(5)
        price_momentum_10 = close.pct_change(10)
        price_momentum_20 = close.pct_change(20)

        # 计算成交量动量
        volume_momentum = volume.pct_change(5)

        # 计算复合动量指数
        composite_momentum = (
            price_momentum_5 * 0.4 +
            price_momentum_10 * 0.3 +
            price_momentum_20 * 0.2 +
            volume_momentum * 0.1
        )

        # 标准化到-100到100区间
        composite_momentum = np.clip(composite_momentum * 1000, -100, 100)

        return pd.DataFrame({'COMPOSITE_MOMENTUM_INDEX': composite_momentum})

    def calculate_adaptive_moving_average(self, data: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        计算自适应移动平均

        Args:
            data: 包含close数据的DataFrame
            period: 计算周期

        Returns:
            包含ADAPTIVE_MA的DataFrame
        """
        close = data['close']

        # 计算效率比率
        efficiency_data = self.calculate_market_efficiency_ratio(data, period)
        efficiency_ratio = efficiency_data['MARKET_EFFICIENCY_RATIO']

        # 计算自适应因子
        fastest_sc = 2.0 / (2 + 1)  # 最快平滑常数
        slowest_sc = 2.0 / (30 + 1)  # 最慢平滑常数

        adaptive_sc = (efficiency_ratio * (fastest_sc - slowest_sc) + slowest_sc) ** 2

        # 计算自适应移动平均
        adaptive_ma = pd.Series(index=close.index, dtype=float)
        adaptive_ma.iloc[0] = close.iloc[0]

        for i in range(1, len(close)):
            adaptive_ma.iloc[i] = (adaptive_sc.iloc[i] * close.iloc[i] +
                                  (1 - adaptive_sc.iloc[i]) * adaptive_ma.iloc[i-1])

        return pd.DataFrame({'ADAPTIVE_MA': adaptive_ma})

    def calculate_system_stability_index(self, data: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        计算系统稳定性指数

        Args:
            data: 包含OHLCV数据的DataFrame
            period: 计算周期

        Returns:
            包含SYSTEM_STABILITY_INDEX的DataFrame
        """
        close = data['close']
        volume = data['volume']

        # 计算价格稳定性
        price_cv = close.rolling(window=period).std() / close.rolling(window=period).mean()
        price_stability = 1 / (1 + price_cv)

        # 计算成交量稳定性
        volume_cv = volume.rolling(window=period).std() / volume.rolling(window=period).mean()
        volume_stability = 1 / (1 + volume_cv)

        # 计算趋势稳定性
        ma = close.rolling(window=period).mean()
        trend_changes = (ma.diff() > 0).astype(int).diff().abs()
        trend_stability = 1 - trend_changes.rolling(window=period).mean()

        # 综合稳定性指数
        stability_index = (price_stability * 0.4 + volume_stability * 0.3 + trend_stability * 0.3) * 100

        return pd.DataFrame({'SYSTEM_STABILITY_INDEX': stability_index})

    def calculate_comprehensive_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算综合评分指标

        Args:
            data: 包含OHLCV数据的DataFrame

        Returns:
            包含COMPREHENSIVE_SCORE的DataFrame
        """
        # 计算各个子指标
        performance = self.calculate_system_performance_score(data)
        sentiment = self.calculate_market_sentiment_index(data)
        risk = self.calculate_risk_assessment_score(data)
        trend_strength = self.calculate_trend_strength_indicator(data)
        momentum = self.calculate_composite_momentum_index(data)
        stability = self.calculate_system_stability_index(data)

        # 计算综合评分
        comprehensive_score = (
            performance['SYSTEM_PERFORMANCE_SCORE'] * 0.25 +
            sentiment['MARKET_SENTIMENT_INDEX'] * 0.15 +
            (100 - risk['RISK_ASSESSMENT_SCORE']) * 0.20 +  # 风险分数取反
            trend_strength['TREND_STRENGTH'] * 0.15 +
            (momentum['COMPOSITE_MOMENTUM_INDEX'] + 100) / 2 * 0.15 +  # 标准化到0-100
            stability['SYSTEM_STABILITY_INDEX'] * 0.10
        )

        return pd.DataFrame({'COMPREHENSIVE_SCORE': comprehensive_score})

    def get_supported_indicators_Indicators(self) -> List[str]:
        """获取支持的指标列表"""
        return [
            # P1重要指标
            'SAR', 'ADX', 'DMI', 'TRIX', 'ROC', 'CMO', 'DMA', 'MTM',
            # P2常用指标
            'STOCHRSI', 'PSY', 'WR', 'BIAS', 'VOL', 'OBV', 'MFI',
            'EMV', 'CCI', 'MOMENTUM', 'VOSC', 'VR', 'PVT', 'CHAIKIN', 'AD',
            # P3专业指标
            'ATR', 'KC', 'VORTEX', 'AROON', 'ICHIMOKU', 'WMA', 'VIX',
            'VOLUME_RATIO', 'ENHANCED_CCI', 'ENHANCED_DMI',
            # P4 ZXM系列指标
            'ZXM_DAILY_MACD', 'ZXM_TURNOVER', 'ZXM_VOLUME_SHRINK', 'ZXM_MA_CALLBACK',
            'ZXM_BS_ABSORB', 'ZXM_AMPLITUDE_ELASTICITY', 'ZXM_RISE_ELASTICITY',
            'ZXM_ELASTICITY', 'ZXM_BOUNCE_DETECTOR', 'ZXM_ELASTICITY_SCORE',
            'ZXM_BUYPOINT_SCORE', 'ZXM_STOCK_SCORE', 'ZXM_DAILY_TREND_UP',
            'ZXM_WEEKLY_TREND_UP', 'ZXM_MONTHLY_KDJ_TREND_UP',
            # P5系统分析指标
            'SYSTEM_PERFORMANCE_SCORE', 'MARKET_SENTIMENT_INDEX', 'RISK_ASSESSMENT_SCORE',
            'TREND_STRENGTH_INDICATOR', 'MOMENTUM_OSCILLATOR', 'VOLATILITY_INDEX',
            'LIQUIDITY_INDICATOR', 'MARKET_EFFICIENCY_RATIO', 'COMPOSITE_MOMENTUM_INDEX',
            'ADAPTIVE_MOVING_AVERAGE', 'SYSTEM_STABILITY_INDEX', 'COMPREHENSIVE_SCORE'
        ]
