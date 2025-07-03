#!/usr/bin/env python3
"""
ICHIMOKU 指标 (一目均衡表)

一目均衡表是日本技术分析师一目山人发明的技术指标，用于判断价格趋势和支撑阻力位。
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from utils.logger import get_logger

logger = get_logger(__name__)


class ICHIMOKU(BaseIndicator, PatternSignalMixin):
    """
    ICHIMOKU 指标 (一目均衡表)
    
    特点:
    1. 由五条线组成：转换线、基准线、先行带A、先行带B、滞后线
    2. 用于判断趋势方向、强度和支撑阻力位
    3. 云图(Kumo)提供动态支撑阻力区域
    4. 多时间框架分析工具
    
    计算方法:
    1. 转换线(Tenkan-sen) = (9日最高价 + 9日最低价) / 2
    2. 基准线(Kijun-sen) = (26日最高价 + 26日最低价) / 2
    3. 先行带A(Senkou Span A) = (转换线 + 基准线) / 2，向前移动26日
    4. 先行带B(Senkou Span B) = (52日最高价 + 52日最低价) / 2，向前移动26日
    5. 滞后线(Chikou Span) = 收盘价，向后移动26日
    
    参数:
    - tenkan_period: 转换线周期，默认为9
    - kijun_period: 基准线周期，默认为26
    - senkou_period: 先行带周期，默认为52
    - chikou_period: 滞后线周期，默认为26
    """
    
    def __init__(self, **kwargs):
        """
        初始化ICHIMOKU指标
        
        Args:
            **kwargs: 指标参数
        """
        super().__init__()
        self.name = "ICHIMOKU"
        
        # 设置默认参数
        self._default_parameters = self._get_default_parameters()
        
        # 应用用户参数
        self.set_parameters(**kwargs)
    
    def _get_default_parameters(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {
            "tenkan_period": 9,
            "kijun_period": 26,
            "senkou_period": 52,
            "chikou_period": 26
        }
    
    def set_parameters(self, **kwargs):
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
            is_valid, errors = validator.validate_indicator_parameters('ICHIMOKU', params)
            if not is_valid:
                # 静默处理验证失败，避免过多警告
                pass
                
        except Exception:
            # 如果验证失败，静默处理，保持向后兼容
            pass
        
        # 设置参数
        self.tenkan_period = kwargs.get('tenkan_period', 9)
        self.kijun_period = kwargs.get('kijun_period', 26)
        self.senkou_period = kwargs.get('senkou_period', 52)
        self.chikou_period = kwargs.get('chikou_period', 26)
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算ICHIMOKU指标
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            添加了ICHIMOKU指标的DataFrame
        """
        result = self._calculate(data, **kwargs)
        self._result = result
        return result
    
    def _calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        内部计算ICHIMOKU指标
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            添加了ICHIMOKU指标的DataFrame
        """
        df = data.copy()
        
        # 获取必要的数据
        high = df['high']
        low = df['low']
        close = df['close']
        
        # 1. 计算转换线(Tenkan-sen)
        tenkan_high = high.rolling(window=self.tenkan_period).max()
        tenkan_low = low.rolling(window=self.tenkan_period).min()
        tenkan_sen = (tenkan_high + tenkan_low) / 2
        
        # 2. 计算基准线(Kijun-sen)
        kijun_high = high.rolling(window=self.kijun_period).max()
        kijun_low = low.rolling(window=self.kijun_period).min()
        kijun_sen = (kijun_high + kijun_low) / 2
        
        # 3. 计算先行带A(Senkou Span A)
        senkou_span_a = (tenkan_sen + kijun_sen) / 2
        
        # 4. 计算先行带B(Senkou Span B)
        senkou_high = high.rolling(window=self.senkou_period).max()
        senkou_low = low.rolling(window=self.senkou_period).min()
        senkou_span_b = (senkou_high + senkou_low) / 2
        
        # 5. 计算滞后线(Chikou Span)
        chikou_span = close.shift(-self.chikou_period)
        
        # 6. 计算云图厚度和位置
        kumo_thickness = abs(senkou_span_a - senkou_span_b)
        kumo_top = np.maximum(senkou_span_a, senkou_span_b)
        kumo_bottom = np.minimum(senkou_span_a, senkou_span_b)
        
        # 7. 计算价格与云图的关系
        price_above_kumo = close > kumo_top
        price_below_kumo = close < kumo_bottom
        price_in_kumo = ~(price_above_kumo | price_below_kumo)
        
        # 8. 计算综合信号强度
        ichimoku_signal = pd.Series(0.0, index=df.index)
        
        # 价格位置评分
        ichimoku_signal += price_above_kumo * 30  # 价格在云图上方
        ichimoku_signal += price_below_kumo * (-30)  # 价格在云图下方
        
        # 转换线和基准线关系
        tenkan_above_kijun = tenkan_sen > kijun_sen
        ichimoku_signal += tenkan_above_kijun * 20  # 转换线在基准线上方
        ichimoku_signal += (~tenkan_above_kijun) * (-20)  # 转换线在基准线下方
        
        # 滞后线位置
        chikou_above_price = chikou_span > close.shift(-self.chikou_period)
        ichimoku_signal += chikou_above_price * 15  # 滞后线在价格上方
        ichimoku_signal += (~chikou_above_price) * (-15)  # 滞后线在价格下方
        
        # 云图颜色（先行带A与先行带B的关系）
        green_kumo = senkou_span_a > senkou_span_b  # 绿云（上升云）
        ichimoku_signal += green_kumo * 10
        ichimoku_signal += (~green_kumo) * (-10)
        
        # 保存计算结果
        df['ICHIMOKU_TENKAN'] = tenkan_sen
        df['ICHIMOKU_KIJUN'] = kijun_sen
        df['ICHIMOKU_SENKOU_A'] = senkou_span_a
        df['ICHIMOKU_SENKOU_B'] = senkou_span_b
        df['ICHIMOKU_CHIKOU'] = chikou_span
        df['ICHIMOKU_KUMO_TOP'] = kumo_top
        df['ICHIMOKU_KUMO_BOTTOM'] = kumo_bottom
        df['ICHIMOKU_KUMO_THICKNESS'] = kumo_thickness
        df['ICHIMOKU_VALUE'] = ichimoku_signal
        
        # 添加形态识别和信号生成
        df = self.add_pattern_detection(df)
        df = self.add_signal_generation(df)

        # 重写信号生成逻辑（ICHIMOKU指标特定逻辑）
        df = self._apply_ichimoku_signal_logic(df)

        return df

    def _apply_ichimoku_signal_logic(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用ICHIMOKU指标特定的信号生成逻辑
        基于一目均衡表的多重确认信号
        """
        try:
            # 获取ICHIMOKU各组件
            if 'ICHIMOKU_VALUE' not in df.columns:
                return df

            close_price = df['close']
            tenkan = df['ICHIMOKU_TENKAN']
            kijun = df['ICHIMOKU_KIJUN']
            kumo_top = df['ICHIMOKU_KUMO_TOP']
            kumo_bottom = df['ICHIMOKU_KUMO_BOTTOM']
            chikou = df['ICHIMOKU_CHIKOU']

            # 一目均衡表信号生成逻辑：
            # 强买入信号：价格>云图 AND 转换线>基准线 AND 滞后线>价格(26日前)
            # 强卖出信号：价格<云图 AND 转换线<基准线 AND 滞后线<价格(26日前)
            
            # 基本条件
            price_above_kumo = close_price > kumo_top
            price_below_kumo = close_price < kumo_bottom
            tenkan_above_kijun = tenkan > kijun
            tenkan_below_kijun = tenkan < kijun
            
            # 滞后线确认（需要考虑移位）
            chikou_confirm_buy = chikou.shift(self.chikou_period) > close_price
            chikou_confirm_sell = chikou.shift(self.chikou_period) < close_price

            # 生成信号
            strong_buy = price_above_kumo & tenkan_above_kijun & chikou_confirm_buy
            strong_sell = price_below_kumo & tenkan_below_kijun & chikou_confirm_sell
            
            # 弱信号
            weak_buy = price_above_kumo | (tenkan_above_kijun & chikou_confirm_buy)
            weak_sell = price_below_kumo | (tenkan_below_kijun & chikou_confirm_sell)
            
            df.loc[:, 'buy_signal'] = strong_buy | weak_buy
            df.loc[:, 'sell_signal'] = strong_sell | weak_sell
            df.loc[:, 'hold_signal'] = ~(df['buy_signal'] | df['sell_signal'])

            # 确保信号类型为布尔值
            df['buy_signal'] = df['buy_signal'].astype(bool)
            df['sell_signal'] = df['sell_signal'].astype(bool)
            df['hold_signal'] = df['hold_signal'].astype(bool)

        except Exception as e:
            logger.warning(f"ICHIMOKU信号生成失败: {e}")
            # 如果出错，使用默认信号
            df.loc[:, 'buy_signal'] = False
            df.loc[:, 'sell_signal'] = False
            df.loc[:, 'hold_signal'] = True

        return df

    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算一目均衡表的原始评分
        
        基于一目均衡表的多重信号确认进行评分：
        1. 价格与云图关系：价格在云图上方/下方/内部的评分
        2. 转换线与基准线关系：金叉死叉的评分
        3. 滞后线确认：滞后线与价格的关系
        4. 云图特征：云图厚度和颜色的评分
        """
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if 'ICHIMOKU_VALUE' not in self._result.columns:
            return pd.Series(50.0, index=data.index)
        
        close = self._result['close']
        tenkan = self._result['ICHIMOKU_TENKAN'].fillna(close)
        kijun = self._result['ICHIMOKU_KIJUN'].fillna(close)
        senkou_a = self._result['ICHIMOKU_SENKOU_A'].fillna(close)
        senkou_b = self._result['ICHIMOKU_SENKOU_B'].fillna(close)
        chikou = self._result['ICHIMOKU_CHIKOU'].fillna(close)
        kumo_top = self._result['ICHIMOKU_KUMO_TOP'].fillna(close)
        kumo_bottom = self._result['ICHIMOKU_KUMO_BOTTOM'].fillna(close)
        kumo_thickness = self._result['ICHIMOKU_KUMO_THICKNESS'].fillna(0)
        
        scores = pd.Series(index=data.index, dtype=float)
        
        for i in range(len(close)):
            if i < max(self.tenkan_period, self.kijun_period, self.senkou_period):
                scores.iloc[i] = 50.0
                continue
            
            score = 50.0  # 基础分数
            
            # 1. 价格与云图关系评分 (30分)
            current_close = close.iloc[i]
            current_kumo_top = kumo_top.iloc[i]
            current_kumo_bottom = kumo_bottom.iloc[i]
            
            if current_close > current_kumo_top:
                # 价格在云图上方 - 看涨
                distance_ratio = (current_close - current_kumo_top) / current_close
                if distance_ratio > 0.05:
                    price_score = 30.0  # 强烈看涨
                elif distance_ratio > 0.02:
                    price_score = 25.0  # 看涨
                else:
                    price_score = 20.0  # 弱看涨
            elif current_close < current_kumo_bottom:
                # 价格在云图下方 - 看跌
                distance_ratio = (current_kumo_bottom - current_close) / current_close
                if distance_ratio > 0.05:
                    price_score = 0.0   # 强烈看跌
                elif distance_ratio > 0.02:
                    price_score = 5.0   # 看跌
                else:
                    price_score = 10.0  # 弱看跌
            else:
                # 价格在云图内部 - 中性
                price_score = 15.0
            
            score += price_score - 15.0  # 调整基准
            
            # 2. 转换线与基准线关系评分 (25分)
            current_tenkan = tenkan.iloc[i]
            current_kijun = kijun.iloc[i]
            
            if current_tenkan > current_kijun:
                # 转换线在基准线上方 - 看涨
                tk_ratio = (current_tenkan - current_kijun) / current_kijun
                if tk_ratio > 0.02:
                    tk_score = 25.0  # 强烈看涨
                else:
                    tk_score = 20.0  # 看涨
            elif current_tenkan < current_kijun:
                # 转换线在基准线下方 - 看跌
                tk_ratio = (current_kijun - current_tenkan) / current_kijun
                if tk_ratio > 0.02:
                    tk_score = 5.0   # 强烈看跌
                else:
                    tk_score = 10.0  # 看跌
            else:
                tk_score = 15.0  # 中性
            
            score += tk_score - 15.0  # 调整基准
            
            # 3. 滞后线确认评分 (25分)
            if i >= self.chikou_period:
                current_chikou = chikou.iloc[i]
                past_close = close.iloc[i - self.chikou_period]
                
                if not pd.isna(current_chikou) and not pd.isna(past_close):
                    if current_chikou > past_close:
                        # 滞后线在过去价格上方 - 看涨确认
                        chikou_ratio = (current_chikou - past_close) / past_close
                        if chikou_ratio > 0.02:
                            chikou_score = 25.0  # 强确认
                        else:
                            chikou_score = 20.0  # 确认
                    elif current_chikou < past_close:
                        # 滞后线在过去价格下方 - 看跌确认
                        chikou_ratio = (past_close - current_chikou) / past_close
                        if chikou_ratio > 0.02:
                            chikou_score = 5.0   # 强看跌确认
                        else:
                            chikou_score = 10.0  # 看跌确认
                    else:
                        chikou_score = 15.0  # 中性
                else:
                    chikou_score = 15.0
            else:
                chikou_score = 15.0
            
            score += chikou_score - 15.0  # 调整基准
            
            # 4. 云图特征评分 (20分)
            current_senkou_a = senkou_a.iloc[i]
            current_senkou_b = senkou_b.iloc[i]
            current_thickness = kumo_thickness.iloc[i]
            
            # 云图颜色（绿云vs红云）
            if current_senkou_a > current_senkou_b:
                # 绿云（上升云）- 看涨
                kumo_color_score = 15.0
            else:
                # 红云（下降云）- 看跌
                kumo_color_score = 5.0
            
            # 云图厚度（厚度越大，支撑阻力越强）
            if current_close > 0:
                thickness_ratio = current_thickness / current_close
                if thickness_ratio > 0.03:
                    kumo_thickness_score = 5.0  # 厚云图，强支撑阻力
                elif thickness_ratio > 0.01:
                    kumo_thickness_score = 3.0  # 中等厚度
                else:
                    kumo_thickness_score = 1.0  # 薄云图
            else:
                kumo_thickness_score = 2.0
            
            kumo_score = kumo_color_score + kumo_thickness_score
            score += kumo_score - 12.0  # 调整基准
            
            # 确保分数在合理范围内
            score = max(0, min(100, score))
            scores.iloc[i] = score
        
        return scores
    
    def calculate_confidence(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """计算置信度"""
        if not self.has_result():
            return 0.5
        
        # 基于一目均衡表多重确认的置信度
        close = self._result['close']
        tenkan = self._result['ICHIMOKU_TENKAN']
        kijun = self._result['ICHIMOKU_KIJUN']
        kumo_top = self._result['ICHIMOKU_KUMO_TOP']
        kumo_bottom = self._result['ICHIMOKU_KUMO_BOTTOM']
        
        # 计算各组件的一致性
        price_kumo_consistency = 0.5
        if len(close) > 0:
            above_kumo = (close > kumo_top).sum()
            below_kumo = (close < kumo_bottom).sum()
            total_valid = len(close.dropna())
            if total_valid > 0:
                price_kumo_consistency = max(above_kumo, below_kumo) / total_valid
        
        # 转换线和基准线的一致性
        tk_consistency = 0.5
        if len(tenkan) > 0 and len(kijun) > 0:
            tk_diff = (tenkan > kijun).sum()
            total_valid = len(tenkan.dropna())
            if total_valid > 0:
                tk_consistency = max(tk_diff, total_valid - tk_diff) / total_valid
        
        # 综合置信度
        confidence = (price_kumo_consistency * 0.6 + tk_consistency * 0.4)
        return min(0.9, max(0.1, confidence))
    
    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """获取形态"""
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        patterns = pd.DataFrame(index=data.index)
        
        if 'ICHIMOKU_VALUE' in self._result.columns:
            close = self._result['close']
            tenkan = self._result['ICHIMOKU_TENKAN']
            kijun = self._result['ICHIMOKU_KIJUN']
            kumo_top = self._result['ICHIMOKU_KUMO_TOP']
            kumo_bottom = self._result['ICHIMOKU_KUMO_BOTTOM']
            
            # 识别关键形态
            patterns['price_above_kumo'] = close > kumo_top
            patterns['price_below_kumo'] = close < kumo_bottom
            patterns['price_in_kumo'] = (close >= kumo_bottom) & (close <= kumo_top)
            patterns['tenkan_kijun_golden_cross'] = (tenkan > kijun) & (tenkan.shift(1) <= kijun.shift(1))
            patterns['tenkan_kijun_death_cross'] = (tenkan < kijun) & (tenkan.shift(1) >= kijun.shift(1))
            patterns['kumo_breakout_up'] = (close > kumo_top) & (close.shift(1) <= kumo_top.shift(1))
            patterns['kumo_breakout_down'] = (close < kumo_bottom) & (close.shift(1) >= kumo_bottom.shift(1))
        
        return patterns
