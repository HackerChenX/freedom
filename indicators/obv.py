#!/usr/bin/env python3
"""
OBV (On-Balance Volume) 能量潮指标

OBV指标通过累计成交量来反映资金流向.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from indicators.base.minimum_periods_mixin import MinimumPeriodsMixin
from utils.logger import get_logger
from utils.container import container

logger = get_logger(__name__)


class OnBalanceVolume(BaseIndicator, PatternSignalMixin, MinimumPeriodsMixin):
    """
    OBV (On-Balance Volume) 能量潮指标
    
    OBV指标通过累计成交量变化来判断资金流向.
    """
    
    REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
    
    def __init__(self, **kwargs):
        """
        初始化OBV指标
        
        Args:
            **kwargs: 指标参数
        """
        super().__init__()
        self.name = "OBV"
        
        # 初始化结果存储
        self._result = None
        
        # 设置默认参数
        self._default_parameters = self._get_default_parameters_obv()
        
        # 应用用户参数
        self.set_parameters_Obv(**kwargs)
    
    def _get_default_parameters_obv(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {"signal_period": 10}
    
    def set_parameters_Obv(self, **kwargs):
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
            is_valid, errors = validator.validate_indicator_parameters('OBV', params)
            if not is_valid:
                # 静默处理验证失败,避免过多警告
                pass
                
        except Exception:
            # 如果验证失败,静默处理,保持向后兼容
            pass
        
        # 设置参数
        self.signal_period = kwargs.get('signal_period', 10)
    
    def calculate_Obv(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算OBV指标
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            添加了OBV指标的DataFrame
        """
        result = self._calculate_obv(data, **kwargs)
        self._result = result
        return result
    
    def _calculate_obv(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        内部计算OBV指标
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            添加了OBV指标的DataFrame
        """
        df = data.copy()
        
        # 确保数据有足够的长度
        if len(df) < 2:
            logger.warning(f"数据长度({len(df)})不足,返回原始数据")
            df['OBV'] = np.nan
            df['obv_ma'] = np.nan
            df['obv_signal'] = np.nan
            return df

        # 计算价格变化
        df['price_change'] = df['close'].diff()

        # 计算OBV (On-Balance Volume)
        obv = [0]  # 初始值为0
        for i in range(1, len(df)):
            if df['price_change'].iloc[i] > 0:
                # 价格上涨,加上成交量
                obv.append(obv[-1] + df['volume'].iloc[i])
            elif df['price_change'].iloc[i] < 0:
                # 价格下跌,减去成交量
                obv.append(obv[-1] - df['volume'].iloc[i])
            else:
                # 价格不变,OBV保持不变
                obv.append(obv[-1])
        
        df['OBV'] = obv
        df['obv'] = obv  # 为了一致性
        
        # 计算OBV移动平均线(信号线)
        df['obv_ma'] = df['OBV'].rolling(window=self.signal_period).mean()
        df[f'OBV_MA{self.signal_period}'] = df['obv_ma']  # 为了向后兼容
        
        # 计算OBV信号线(更短周期的移动平均)
        df['obv_signal'] = df['OBV'].rolling(window=5).mean()
        
        # 计算OBV变化率
        df['obv_change'] = df['OBV'].pct_change() * 100
        
        # 计算OBV波动率
        df['obv_volatility'] = df['obv_change'].rolling(window=10).std()
        
        # 计算OBV相对强度(与历史均值的关系)
        df['obv_strength'] = (df['OBV'] - df['OBV'].rolling(window=20).mean()) / (df['OBV'].rolling(window=20).std() + 1e-8)
        
        # 清理中间计算列
        df.drop(['price_change'], axis=1, inplace=True)
            
        # 添加形态识别和信号生成
        df = self.add_pattern_detection(df)
        df = self.add_signal_generation(df)

        # 重写信号生成逻辑(OBV指标特定逻辑)
        df = self._apply_obv_signal_logic(df)

        return df

    def _apply_obv_signal_logic(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用OBV指标特定的信号生成逻辑
        基于OBV值的变化和量价关系生成信号
        """
        try:
            # 获取OBV值
            if 'OBV' not in df.columns:
                # 如果没有OBV值,使用默认信号
                return df

            obv_value = df['OBV']
            obv_ma = df['obv_ma']
            obv_signal = df['obv_signal']
            close_price = df['close']

            # 计算OBV和价格的变化
            obv_rising = obv_value > obv_value.shift(1)
            obv_falling = obv_value < obv_value.shift(1)
            price_rising = close_price > close_price.shift(1)
            price_falling = close_price < close_price.shift(1)
            
            # OBV与移动平均线的关系
            obv_above_ma = obv_value > obv_ma
            obv_below_ma = obv_value < obv_ma
            
            # OBV突破移动平均线
            obv_breakout_up = obv_above_ma & (obv_value.shift(1) <= obv_ma.shift(1))
            obv_breakdown = obv_below_ma & (obv_value.shift(1) >= obv_ma.shift(1))
            
            # 强势OBV信号
            strong_obv_up = obv_rising & (obv_value > obv_signal)
            strong_obv_down = obv_falling & (obv_value < obv_signal)

            # 生成信号
            df.loc[:, 'buy_signal'] = (obv_rising & price_rising) | obv_breakout_up | strong_obv_up
            df.loc[:, 'sell_signal'] = (obv_falling & price_falling) | obv_breakdown | strong_obv_down
            df.loc[:, 'hold_signal'] = ~(df['buy_signal'] | df['sell_signal'])

            # 确保信号类型为布尔值
            df['buy_signal'] = df['buy_signal'].astype(bool)
            df['sell_signal'] = df['sell_signal'].astype(bool)
            df['hold_signal'] = df['hold_signal'].astype(bool)

        except Exception as e:
            logger.warning(f"OBV信号生成失败: {e}")
            # 如果出错,使用默认信号
            df.loc[:, 'buy_signal'] = False
            df.loc[:, 'sell_signal'] = False
            df.loc[:, 'hold_signal'] = True

        return df
    
    def calculate_raw_score_Obv(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算OBV原始评分
        """
        if not self.has_result():
            self.calculate_Obv(data, **kwargs)
        
        if self._result is None:
            return pd.Series(50.0, index=data.index)
        
        # 获取OBV数据
        obv = self._result['OBV']
        obv_ma = self._result['obv_ma']
        obv_strength = self._result['obv_strength']
        
        # 初始化评分
        scores = pd.Series(50.0, index=data.index)
        
        # 简化的评分逻辑
        obv_change = obv - obv.shift(1)
        
        # OBV上升得分
        scores += np.where(obv_change > 0, 10, 0)
        # OBV下降扣分
        scores += np.where(obv_change < 0, -10, 0)
        
        # OBV与移动平均线的关系
        if len(obv_ma.dropna()) > 0:
            scores += np.where(obv > obv_ma, 5, -5)
        
        # 确保评分在合理范围内
        scores = np.clip(scores, 0, 100)
        
        return scores
    
    def get_patterns_Obv(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """获取OBV相关形态"""
        if not self.has_result():
            self.calculate_Obv(data, **kwargs)
            
        if self._result is None:
            return pd.DataFrame(index=data.index)
            
        patterns = pd.DataFrame(index=data.index)
        
        obv = self._result['OBV']
        obv_ma = self._result['obv_ma']
        
        # 基本形态
        obv_change = obv - obv.shift(1)
        patterns['OBV_RISING'] = obv_change > 0
        patterns['OBV_FALLING'] = obv_change < 0
        patterns['OBV_STABLE'] = abs(obv_change) < (obv.std() * 0.1)
        
        # 与移动平均线的关系
        if len(obv_ma.dropna()) > 0:
            patterns['OBV_ABOVE_MA'] = obv > obv_ma
            patterns['OBV_BELOW_MA'] = obv < obv_ma
            patterns['OBV_BREAKOUT_UP'] = (obv > obv_ma) & (obv.shift(1) <= obv_ma.shift(1))
            patterns['OBV_BREAKDOWN'] = (obv < obv_ma) & (obv.shift(1) >= obv_ma.shift(1))
        
        return patterns

    # ================== 抽象方法实现 ==================
    
    def _calculate_baseindicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """抽象方法实现:调用OBV计算逻辑"""
        return self._calculate_obv(data, **kwargs)
    
    def calculate_raw_score_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """抽象方法实现:计算OBV原始评分"""
        return self.calculate_raw_score_Obv(data, **kwargs)
    
    def get_patterns_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """抽象方法实现:获取OBV形态"""
        return self.get_patterns_Obv(data, **kwargs)
    
    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        """抽象方法实现:设置参数"""
        return self.set_parameters_Obv(**kwargs)
    
    def calculate_confidence_Indicator_Base_Indicator(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """抽象方法实现:计算置信度"""
        return self.calculate_confidence_Obv(score, patterns, signals)
    
    # ================== 兼容性方法 ==================
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """兼容性方法:计算指标"""
        return self.calculate_Obv(data, **kwargs)
    
    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """兼容性方法:获取形态"""
        return self.get_patterns_Obv(data, **kwargs)
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """兼容性方法:计算原始评分"""
        return self.calculate_raw_score_Obv(data, **kwargs)
    
    def calculate_confidence_Obv(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """计算置信度"""
        if self._result is None:
            return 0.5
            
        # 基于OBV指标的明确性计算置信度
        try:
            obv = self._result['OBV'].dropna()
            if len(obv) == 0:
                return 0.5
                
            # 基础置信度
            confidence = 0.5
            
            # 根据OBV强度调整置信度
            if 'obv_strength' in self._result.columns:
                strength = abs(self._result['obv_strength'].iloc[-1])
                if pd.notna(strength):
                    confidence += min(strength * 0.1, 0.3)
            
            return min(max(confidence, 0.0), 1.0)
            
        except Exception as e:
            logger.warning(f"OBV置信度计算失败: {e}")
            return 0.0

    def get_signal(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        【核心抽象方法2】基于OBV (On-Balance Volume) 指标数值生成最新的交易信号
        
        OBV交易信号逻辑：
        - OBV上升 + 价格上升：量价配合买入信号
        - OBV下降 + 价格下降：量价配合卖出信号
        - OBV突破均线：趋势确认信号
        - OBV背离价格：潜在反转信号
        - OBV强度评估：信号强度判断
        
        Args:
            data: 包含OHLCV数据的DataFrame
            **kwargs: 额外参数
            
        Returns:
            Dict[str, Any]: 标准化交易信号格式
        """
        try:
            # 1. 数据验证
            if not self._validate_signal_data(data):
                return self._get_default_signal("数据验证失败")
            
            # 2. 确保已计算指标
            if not self.has_result():
                result = self.calculate_Obv(data, **kwargs)
                if result is not None:
                    self._result = result
            
            if self._result is None or len(self._result) == 0:
                return self._get_default_signal("OBV计算结果为空")

            # 3. 获取最新数据
            latest_close = data['close'].iloc[-1]
            
            # 4. 获取OBV相关值
            if len(self._result) < 2:
                return self._get_default_signal("OBV数据不足")
                
            # 检查必要的列是否存在
            if 'OBV' not in self._result.columns:
                return self._get_default_signal("OBV结果列不存在")
                
            obv_values = self._result['OBV'].dropna()
            if len(obv_values) < 2:
                return self._get_default_signal("OBV有效数据不足")
                
            latest_obv = obv_values.iloc[-1]
            prev_obv = obv_values.iloc[-2]
            
            # 检查是否有NaN值
            if pd.isna(latest_obv) or pd.isna(prev_obv):
                return self._get_default_signal("OBV数据包含NaN值")
            
            # 5. 获取价格数据
            close_values = data['close'].iloc[-2:]
            if len(close_values) < 2:
                return self._get_default_signal("价格数据不足")
                
            latest_price = close_values.iloc[-1]
            prev_price = close_values.iloc[-2]
            
            # 6. OBV信号生成逻辑
            signal_type = "hold"
            strength = 0.0
            confidence = 0.5
            reason = "无明确信号"
            metadata = {}
            
            # 计算OBV和价格变化
            obv_change = latest_obv - prev_obv
            price_change = latest_price - prev_price
            obv_change_pct = (obv_change / abs(prev_obv)) * 100 if prev_obv != 0 else 0
            price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0
            
            # 获取OBV均线和强度信息（如果存在）
            obv_ma = None
            obv_strength = 0
            if 'obv_ma' in self._result.columns:
                obv_ma_values = self._result['obv_ma'].dropna()
                if len(obv_ma_values) > 0:
                    obv_ma = obv_ma_values.iloc[-1]
            
            if 'obv_strength' in self._result.columns:
                obv_strength_values = self._result['obv_strength'].dropna()
                if len(obv_strength_values) > 0:
                    obv_strength = obv_strength_values.iloc[-1]
            
            # 量价配合分析（核心OBV信号）
            if obv_change > 0 and price_change > 0:
                # OBV和价格同时上升 - 强烈买入信号
                signal_type = "buy"
                volume_price_strength = min(abs(obv_change_pct) + abs(price_change_pct), 100) / 100
                strength = min(0.9, 0.8 + volume_price_strength * 0.1)
                confidence = 0.9
                reason = f"OBV量价配合上升({obv_change_pct:.2f}%, {price_change_pct:.2f}%)，强烈买入信号"
                
            elif obv_change < 0 and price_change < 0:
                # OBV和价格同时下降 - 强烈卖出信号
                signal_type = "sell"
                volume_price_strength = min(abs(obv_change_pct) + abs(price_change_pct), 100) / 100
                strength = min(0.9, 0.8 + volume_price_strength * 0.1)
                confidence = 0.9
                reason = f"OBV量价配合下降({obv_change_pct:.2f}%, {price_change_pct:.2f}%)，强烈卖出信号"
                
            elif obv_change > 0 and price_change < 0:
                # OBV上升但价格下降 - 潜在底部，谨慎买入
                signal_type = "buy"
                divergence_strength = min(abs(obv_change_pct) + abs(price_change_pct), 80) / 100
                strength = min(0.7, 0.6 + divergence_strength * 0.1)
                confidence = 0.7
                reason = f"OBV正背离({obv_change_pct:.2f}% vs {price_change_pct:.2f}%)，潜在反转买入信号"
                
            elif obv_change < 0 and price_change > 0:
                # OBV下降但价格上升 - 潜在顶部，谨慎卖出
                signal_type = "sell"
                divergence_strength = min(abs(obv_change_pct) + abs(price_change_pct), 80) / 100
                strength = min(0.7, 0.6 + divergence_strength * 0.1)
                confidence = 0.7
                reason = f"OBV负背离({obv_change_pct:.2f}% vs {price_change_pct:.2f}%)，潜在反转卖出信号"
            
            # OBV均线突破信号
            elif obv_ma is not None:
                obv_ma_prev = None
                if len(self._result) >= 2 and 'obv_ma' in self._result.columns:
                    obv_ma_prev_values = self._result['obv_ma'].iloc[-2:-1]
                    if len(obv_ma_prev_values) > 0 and not pd.isna(obv_ma_prev_values.iloc[0]):
                        obv_ma_prev = obv_ma_prev_values.iloc[0]
                
                if obv_ma_prev is not None:
                    # OBV突破均线向上
                    if prev_obv <= obv_ma_prev and latest_obv > obv_ma:
                        signal_type = "buy"
                        breakout_strength = min(abs(latest_obv - obv_ma) / abs(obv_ma), 0.2) if obv_ma != 0 else 0
                        strength = min(0.8, 0.7 + breakout_strength * 5)
                        confidence = 0.8
                        reason = f"OBV突破均线向上({latest_obv:.0f} > {obv_ma:.0f})，买入信号"
                        
                    # OBV跌破均线向下
                    elif prev_obv >= obv_ma_prev and latest_obv < obv_ma:
                        signal_type = "sell"
                        breakdown_strength = min(abs(obv_ma - latest_obv) / abs(obv_ma), 0.2) if obv_ma != 0 else 0
                        strength = min(0.8, 0.7 + breakdown_strength * 5)
                        confidence = 0.8
                        reason = f"OBV跌破均线向下({latest_obv:.0f} < {obv_ma:.0f})，卖出信号"
            
            # OBV强度调整
            if abs(obv_strength) > 0:
                if obv_strength > 1.5:  # 强势OBV
                    strength = min(strength + 0.1, 1.0)
                    confidence = min(confidence + 0.05, 1.0)
                elif obv_strength < -1.5:  # 弱势OBV
                    if signal_type == "buy":
                        strength = max(strength - 0.1, 0.0)
                        confidence = max(confidence - 0.05, 0.0)
                    elif signal_type == "sell":
                        strength = min(strength + 0.1, 1.0)
                        confidence = min(confidence + 0.05, 1.0)
            
            # 计算OBV特有的元数据
            obv_trend = "上升" if obv_change > 0 else "下降" if obv_change < 0 else "平稳"
            price_trend = "上升" if price_change > 0 else "下降" if price_change < 0 else "平稳"
            volume_price_relationship = "配合" if (obv_change > 0) == (price_change > 0) else "背离" if obv_change != 0 and price_change != 0 else "中性"
            
            metadata = {
                'obv_value': latest_obv,
                'obv_previous': prev_obv,
                'obv_change': obv_change,
                'obv_change_pct': obv_change_pct,
                'obv_trend': obv_trend,
                'price_change': price_change,
                'price_change_pct': price_change_pct,
                'price_trend': price_trend,
                'volume_price_relationship': volume_price_relationship,
                'obv_ma': obv_ma,
                'obv_strength': obv_strength,
                'above_ma': latest_obv > obv_ma if obv_ma is not None else None,
                'signal_period': getattr(self, 'signal_period', 10)
            }
            
            # 7. 标准化输出
            return {
                'signal_type': signal_type,
                'strength': max(0.0, min(1.0, strength)),
                'confidence': max(0.0, min(1.0, confidence)),
                'timestamp': pd.Timestamp.now(),
                'reason': reason,
                'metadata': {
                    'latest_close': latest_close,
                    **metadata
                }
            }

        except Exception as e:
            logger.warning(f"OBV信号生成失败: {e}")
            return self._get_default_signal(f"信号生成失败: {str(e)}")

    def _validate_signal_data(self, data: pd.DataFrame) -> bool:
        """
        验证信号生成所需的数据
        
        Args:
            data: 输入数据DataFrame
            
        Returns:
            bool: 数据是否有效
        """
        if data is None or data.empty:
            return False
            
        required_columns = ['close', 'volume']
        if not all(col in data.columns for col in required_columns):
            return False
            
        # OBV需要足够的数据用于计算
        min_periods = max(getattr(self, 'signal_period', 10), 2)
        if len(data) < min_periods:
            return False
            
        return True

    def _get_default_signal(self, reason: str = "数据不足") -> Dict[str, Any]:
        """
        生成默认信号（持有信号）
        
        Args:
            reason: 生成默认信号的原因
            
        Returns:
            Dict[str, Any]: 默认信号
        """
        return {
            'signal_type': 'hold',
            'strength': 0.0,
            'confidence': 0.0,
            'timestamp': pd.Timestamp.now(),
            'reason': reason,
            'metadata': {}
        }

    def has_result(self) -> bool:
        """检查是否有计算结果"""
        return self._result is not None

    @property
    def minimum_periods(self) -> int:
        """
        返回OBV指标计算所需的最少数据周期数

        Returns:
            int: 最少需要的数据周期数
        """
        return 2  # OBV只需要2个数据点就可以开始计算


# 类别名,供指标注册系统使用
OnBalanceVolumeOBV = OnBalanceVolume
Obv = OnBalanceVolume
OBV = OnBalanceVolume  # 添加OBV别名
