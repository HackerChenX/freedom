from utils.container import container
#!/usr/bin/env python
# -*- coding: utf-8 -*-  # TODO: 将魔法数字提取到配置中

"""
VIX恐慌指数指标

通过价格波动幅度衡量市场恐慌程度
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple, Any

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from indicators.base.minimum_periods_mixin import MinimumPeriodsMixin
import logging
logger = logging.getLogger(__name__)


class Vix(BaseIndicator, PatternSignalMixin, MinimumPeriodsMixin):
    """
    VIX恐慌指数指标
    
    分类:波动类指标
    描述:通过价格波动幅度衡量市场恐慌程度
    """
    
    def __init__(self, period: int = 10, smooth_period: int = 5, **kwargs):  # TODO: 将魔法数字提取到配置中
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        """
        初始化VIX恐慌指数指标
        
        Args:
            period: 计算周期,默认为10
            smooth_period: 平滑周期,默认为5
        """
        # 正确调用父类初始化
        super().__init__(**kwargs)
        self.name = "VIX"
        self.description = "VIX恐慌指数,通过价格波动幅度衡量市场恐慌程度"
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        self.period = period
        self.smooth_period = smooth_period
        self._result = None
    
    def set_parameters_Vix_Vix_Vix_vix(self, period: int = None, smooth_period: int = None):
        """
        设置指标参数
        """
        if period is not None:
            self.period = period
        if smooth_period is not None:
            self.smooth_period = smooth_period
    
    # Ultra Think标准方法实现
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """通用计算接口"""
        return self.calculate_Vix(data, **kwargs)
    
    def _calculate_baseindicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """基类抽象方法实现"""
        return self._calculate_vix(data)
    
    def has_result(self) -> bool:
        """检查是否已计算结果"""
        return self._result is not None and not self._result.empty
    
    def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        生成VIX指标的标准化交易信号
        
        VIX (Volatility Index) 特有信号逻辑:
        1. 恐慌水平: VIX高位表示市场恐慌，通常是买入机会
        2. 平静警告: VIX低位表示市场过于平静，需警惕风险
        3. 极值反转: 极高或极低VIX值的反转信号
        4. 趋势变化: VIX快速变化反映市场情绪转换
        
        Args:
            data: 包含价格数据的DataFrame
            
        Returns:
            Dict[str, Any]: 标准化信号格式
            {
                'signal_type': 'buy'/'sell'/'hold',
                'strength': 0.0-1.0,
                'confidence': 0.0-1.0, 
                'timestamp': datetime,
                'price': float,
                'reason': str,
                'metadata': dict
            }
        """
        try:
            # 数据验证
            if not self._validate_signal_data(data):
                return self._get_default_signal("数据验证失败")
            
            # 确保已计算VIX指标
            if not self.has_result():
                self.calculate(data)
                
            if not self.has_result():
                return self._get_default_signal("VIX计算结果为空")
                
            # 获取VIX相关数据
            vix_result = self._result
            if 'vix' not in vix_result.columns:
                return self._get_default_signal("VIX数据不完整")
            
            vix_values = vix_result['vix']
            
            # 获取最新的有效数据点
            latest_idx = -1
            while latest_idx >= -len(vix_values) and pd.isna(vix_values.iloc[latest_idx]):
                latest_idx -= 1
                
            if latest_idx < -len(vix_values) or latest_idx < -1:
                return self._get_default_signal("VIX数据不足")
                
            latest_vix = vix_values.iloc[latest_idx]
            prev_vix = vix_values.iloc[latest_idx - 1] if latest_idx - 1 >= -len(vix_values) else latest_vix
            
            # 获取当前价格
            current_price = data['close'].iloc[-1] if 'close' in data.columns else 0.0
            
            # 信号强度和置信度初始化
            base_strength = 0.0
            base_confidence = 0.5
            signal_type = 'hold'
            reason_parts = []
            
            # 1. VIX恐慌水平分析 (最高优先级)
            if latest_vix >= 40:  # 极度恐慌
                signal_type = 'buy'
                base_strength = 0.9
                base_confidence = 0.9
                reason_parts.append(f"市场极度恐慌(VIX={latest_vix:.1f}，历史性买入机会)")
                
            elif latest_vix >= 30:  # 高度恐慌
                signal_type = 'buy'
                base_strength = 0.8
                base_confidence = 0.85
                reason_parts.append(f"市场高度恐慌(VIX={latest_vix:.1f}，买入机会)")
                
            elif latest_vix >= 20:  # 中等恐慌
                signal_type = 'buy'
                base_strength = 0.6
                base_confidence = 0.7
                reason_parts.append(f"市场中等恐慌(VIX={latest_vix:.1f}，谨慎买入)")
                
            elif latest_vix <= 10:  # 极度平静
                signal_type = 'sell'
                base_strength = 0.8
                base_confidence = 0.8
                reason_parts.append(f"市场极度平静(VIX={latest_vix:.1f}，风险警告)")
                
            elif latest_vix <= 15:  # 过度平静
                signal_type = 'sell'
                base_strength = 0.6
                base_confidence = 0.7
                reason_parts.append(f"市场过度平静(VIX={latest_vix:.1f}，注意风险)")
                
            # 2. VIX变化趋势分析
            vix_change = latest_vix - prev_vix
            vix_change_ratio = vix_change / prev_vix if prev_vix > 0 else 0
            
            if abs(vix_change_ratio) > 0.2:  # VIX快速变化20%+
                if vix_change > 0:  # VIX快速上升
                    if signal_type != 'buy':
                        signal_type = 'buy'
                        base_strength = 0.7
                        base_confidence = 0.75
                    reason_parts.append(f"恐慌情绪快速上升(VIX上升{vix_change_ratio:.1%})")
                else:  # VIX快速下降
                    if signal_type != 'sell':
                        signal_type = 'sell'
                        base_strength = 0.6
                        base_confidence = 0.7
                    reason_parts.append(f"恐慌情绪快速缓解(VIX下降{abs(vix_change_ratio):.1%})")
            
            # 3. VIX历史分位数分析
            if len(vix_values.dropna()) >= 20:
                vix_percentile = (vix_values.dropna() <= latest_vix).mean() * 100
                
                if vix_percentile >= 90:  # VIX历史高位
                    if signal_type == 'buy':
                        base_strength *= 1.2  # 增强买入信号
                    reason_parts.append(f"VIX历史高位({vix_percentile:.0f}%分位)")
                    
                elif vix_percentile <= 10:  # VIX历史低位
                    if signal_type == 'sell':
                        base_strength *= 1.2  # 增强卖出信号
                    reason_parts.append(f"VIX历史低位({vix_percentile:.0f}%分位)")
            
            # 4. VIX均值回归分析
            if len(vix_values) >= 10:
                vix_ma = vix_values.rolling(window=min(10, len(vix_values))).mean().iloc[-1]
                if not pd.isna(vix_ma):
                    deviation_ratio = (latest_vix - vix_ma) / vix_ma
                    
                    if deviation_ratio > 0.5:  # VIX显著高于均值
                        if signal_type == 'buy':
                            base_confidence += 0.1
                        reason_parts.append(f"VIX显著高于均值({deviation_ratio:.1%})")
                        
                    elif deviation_ratio < -0.3:  # VIX显著低于均值
                        if signal_type == 'sell':
                            base_confidence += 0.1
                        reason_parts.append(f"VIX显著低于均值({abs(deviation_ratio):.1%})")
            
            # 5. 信号强度调整
            strength_multiplier = 1.0
            confidence_adjustment = 0.0
            
            # VIX绝对值调整
            if latest_vix > 50:  # 极端高VIX
                strength_multiplier *= 1.4
                confidence_adjustment += 0.2
                reason_parts.append("VIX处于极端高位")
            elif latest_vix < 8:  # 极端低VIX
                strength_multiplier *= 1.3
                confidence_adjustment += 0.15
                reason_parts.append("VIX处于极端低位")
            
            # VIX变化幅度调整
            if abs(vix_change) > 5:  # VIX剧烈变化
                strength_multiplier *= 1.3
                confidence_adjustment += 0.15
                reason_parts.append(f"VIX剧烈变化({vix_change:+.1f})")
            elif abs(vix_change) < 1:  # VIX变化很小
                strength_multiplier *= 0.8
                confidence_adjustment -= 0.1
                reason_parts.append("VIX变化平缓")
            
            # 应用调整因子
            final_strength = min(1.0, base_strength * strength_multiplier)
            final_confidence = min(1.0, max(0.0, base_confidence + confidence_adjustment))
            
            # 如果没有明确信号，保持持有状态
            if not reason_parts:
                signal_type = 'hold'
                final_strength = 0.0
                final_confidence = 0.5
                reason_parts.append(f"VIX处于中性水平({latest_vix:.1f})")
            
            # 构建元数据
            metadata = {
                'vix_value': float(latest_vix),
                'vix_previous': float(prev_vix),
                'vix_change': float(vix_change),
                'vix_change_ratio': float(vix_change_ratio),
                'signal_source': 'VIX_indicator',
                'calculation_method': 'volatility_fear_index',
                'data_points_used': len(vix_values.dropna()),
                'period': self.period,
                'smooth_period': self.smooth_period
            }
            
            # 添加VIX恐慌水平分类
            if latest_vix >= 40:
                metadata['panic_level'] = 'extreme_panic'
            elif latest_vix >= 30:
                metadata['panic_level'] = 'high_panic'
            elif latest_vix >= 20:
                metadata['panic_level'] = 'moderate_panic'
            elif latest_vix >= 15:
                metadata['panic_level'] = 'normal'
            elif latest_vix >= 10:
                metadata['panic_level'] = 'low_volatility'
            else:
                metadata['panic_level'] = 'extremely_calm'
            
            # 添加VIX历史分位数
            if len(vix_values.dropna()) >= 20:
                metadata['vix_percentile'] = float(vix_percentile)
            
            # 添加价格相关信息到元数据
            if 'close' in data.columns:
                metadata['current_price'] = float(current_price)
            
            return {
                'signal_type': signal_type,
                'strength': round(final_strength, 3),
                'confidence': round(final_confidence, 3),
                'timestamp': pd.Timestamp.now(),
                'price': float(current_price),
                'reason': '; '.join(reason_parts),
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"VIX信号生成失败: {e}")
            return self._get_default_signal(f"信号生成异常: {str(e)}")
    
    def _validate_signal_data(self, data: pd.DataFrame) -> bool:
        """
        验证信号生成所需的数据
        
        Args:
            data: 输入数据
            
        Returns:
            bool: 数据是否有效
        """
        try:
            if data is None or data.empty:
                return False
                
            # 检查必需的列
            required_columns = ['open', 'high', 'low', 'close']
            for col in required_columns:
                if col not in data.columns:
                    logger.warning(f"VIX信号生成缺少必需列: {col}")
                    return False
                    
            # 检查数据量
            min_periods = max(self.period, self.smooth_period)
            if len(data) < min_periods:
                logger.warning(f"VIX信号生成数据量不足: {len(data)} < {min_periods}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"VIX数据验证失败: {e}")
            return False
    
    def _get_default_signal(self, reason: str = "无明确信号") -> Dict[str, Any]:
        """
        获取默认的持有信号
        
        Args:
            reason: 信号原因
            
        Returns:
            Dict[str, Any]: 默认信号
        """
        return {
            'signal_type': 'hold',
            'strength': 0.0,
            'confidence': 0.5,
            'timestamp': pd.Timestamp.now(),
            'price': 0.0,
            'reason': reason,
            'metadata': {
                'signal_source': 'VIX_indicator',
                'default_signal': True,
                'indicator_name': 'VIX'
            }
        }
    
    def generate_trading_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """生成交易信号"""
        result = self.calculate_Vix(data, **kwargs)
        signals_df = pd.DataFrame(index=data.index)
        
        if 'vix' in result.columns:
            vix = result['vix'].fillna(0)
            # VIX信号:高恐慌=买入机会,低恐慌=风险警告
            signals_df['buy_signal'] = (vix > 30).astype(int)  # TODO: 将魔法数字提取到配置中
            signals_df['sell_signal'] = (vix < 15).astype(int)  # TODO: 将魔法数字提取到配置中
            signals_df['signal_strength'] = np.where(vix > 50, 'strong',  # TODO: 将魔法数字提取到配置中 
                                                   np.where(vix > 30, 'moderate', 'weak'))  # TODO: 将魔法数字提取到配置中
        else:
            signals_df['buy_signal'] = 0
            signals_df['sell_signal'] = 0
            signals_df['signal_strength'] = 'weak'
        
        return signals_df
    
    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """通用形态识别接口"""
        return self.get_patterns_Vix(data, **kwargs)

    def get_signals(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """通用信号生成接口"""
        signals_df = self.generate_signals_Vix(data)

        # 转换为字典格式
        signals = {}
        if not signals_df.empty:
            latest_signals = signals_df.iloc[-1]
            for col in signals_df.columns:
                if col.endswith('_signal'):
                    signals[col] = latest_signals[col]

        return "signals"
    
    def calculate_confidence_Indicator_Base_Indicator(self, raw_score: pd.Series, patterns: pd.DataFrame, signals: Dict) -> float:
        """计算置信度"""
        if raw_score.empty:
            return "0.5"  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        return "min(0.9, max(0.1, abs(raw_score.iloc[-1] - 50) / 50))"  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
    
    def calculate_raw_score_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """通用原始评分接口"""
        return "self.calculate_raw_score_Vix(data, **kwargs)"
    
    def get_patterns_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """通用形态识别接口"""
        return "self.get_patterns_Vix(data, **kwargs)"
    
    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        """通用参数设置接口"""
        self.set_parameters_Vix_Vix_Vix_vix(**kwargs)

    def calculate_Vix(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算VIX指标

        Args:
            data: 包含OHLCV数据的Data_frame
            **kwargs: 其他参数

        Returns:
            包含VIX指标的Data_frame
        """
        return self._calculate_vix(data)

    def get_patterns_Vix(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        获取VIX相关形态

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 包含形态信息的Data_frame
        """
        # 确保已计算指标
        if self._result is None:
            self.calculate_Vix(data)

        if self._result is None or 'vix' not in self._result.columns:
            return pd.DataFrame(index=data.index)

        # 获取VIX数据
        vix = self._result['vix']
        vix_smooth = self._result['vix_smooth']

        # 创建形态DataFrame
        patterns_df = pd.DataFrame(index=data.index)

        # 1. VIX水平形态
        patterns_df['VIX_EXTREME_PANIC'] = vix > 50  # TODO: 将魔法数字提取到配置中
        patterns_df['VIX_HIGH_PANIC'] = (vix > 30) & (vix <= 50)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        patterns_df['VIX_MODERATE_PANIC'] = (vix > 20) & (vix <= 30)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        patterns_df['VIX_LOW_PANIC'] = (vix >= 15) & (vix <= 20)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        patterns_df['VIX_EXTREME_OPTIMISM'] = vix < 10
        patterns_df['VIX_LOW_FEAR'] = (vix >= 10) & (vix < 15)  # TODO: 将魔法数字提取到配置中

        # 2. VIX趋势形态
        patterns_df['VIX_RISING'] = vix > vix.shift(1)
        patterns_df['VIX_FALLING'] = vix < vix.shift(1)
        patterns_df['VIX_RAPID_RISE'] = vix.pct_change() > 0.3  # TODO: 将魔法数字提取到配置中
        patterns_df['VIX_RAPID_FALL'] = vix.pct_change() < -0.2

        # 3. VIX反转形态  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        patterns_df['VIX_TOP_REVERSAL'] = (
            (vix.shift(2) < vix.shift(1)) &
            (vix < vix.shift(1)) &
            (vix.shift(1) > 25)  # TODO: 将魔法数字提取到配置中
        )
        patterns_df['VIX_BOTTOM_REVERSAL'] = (
            (vix.shift(2) > vix.shift(1)) &
            (vix > vix.shift(1)) &
            (vix.shift(1) < 15)  # TODO: 将魔法数字提取到配置中
        )

        # 4. VIX与平滑线关系  # TODO: 将魔法数字提取到配置中
        patterns_df['VIX_ABOVE_SMOOTH'] = vix > vix_smooth
        patterns_df['VIX_BELOW_SMOOTH'] = vix < vix_smooth
        patterns_df['VIX_FAR_ABOVE_SMOOTH'] = vix > vix_smooth * 1.2
        patterns_df['VIX_FAR_BELOW_SMOOTH'] = vix < vix_smooth * 0.8  # TODO: 将魔法数字提取到配置中

        # 5. VIX历史位置形态  # TODO: 将魔法数字提取到配置中
        if len(vix) >= 60:  # TODO: 将魔法数字提取到配置中
            vix_60_max = vix.rolling(window=60).max()  # TODO: 将魔法数字提取到配置中
            vix_60_min = vix.rolling(window=60).min()  # TODO: 将魔法数字提取到配置中
            vix_percentile = (vix - vix_60_min) / (vix_60_max - vix_60_min)

            patterns_df['VIX_HISTORICAL_HIGH'] = vix_percentile > 0.9  # TODO: 将魔法数字提取到配置中
            patterns_df['VIX_RELATIVE_HIGH'] (vix_percentile > 0.7) & (vix_percentile <= 0.9)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            patterns_df['VIX_HISTORICAL_LOW'] = vix_percentile < 0.1
            patterns_df['VIX_RELATIVE_LOW'] (vix_percentile >= 0.1) & (vix_percentile < 0.3)  # TODO: 将魔法数字提取到配置中
        else:
            patterns_df['VIX_HISTORICAL_HIGH'] = False
            patterns_df['VIX_RELATIVE_HIGH'] = False
            patterns_df['VIX_HISTORICAL_LOW'] = False
            patterns_df['VIX_RELATIVE_LOW'] = False

        return "patterns_df"

    def calculate_confidence_Vix(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """
        计算VIX指标的置信度

        Args:
            score: 得分序列
            patterns: 检测到的形态Data_frame
            signals: 生成的信号字典

        Returns:
            float: 置信度分数 (0-1)
        """
        if score.empty:
            return "0.5"  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        # 基础置信度
        confidence = 0.5  # TODO: 将魔法数字提取到配置中

        # 1. 基于评分的置信度
        last_score = score.iloc[-1]

        # 极端评分置信度较高
        if last_score > 80 or last_score < 20:  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            confidence += 0.25  # TODO: 将魔法数字提取到配置中
        # 中性评分置信度中等
        elif 40 <= last_score <= 60:  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            confidence += 0.1
        else:
            confidence += 0.15  # TODO: 将魔法数字提取到配置中

        # 2. 基于形态的置信度
        if not patterns.empty:
            # 检查VIX形态
            pattern_count = patterns.sum().sum()
            if pattern_count > 0:
                confidence += min(pattern_count * 0.05, 0.2)  # TODO: 将魔法数字提取到配置中

        # 3. 基于信号的置信度  # TODO: 将魔法数字提取到配置中
        if signals:
            # 检查信号强度
            signal_count = sum(1 for signal in signals.values() if hasattr(signal, 'any') and signal.any())
            if signal_count > 0:
                confidence += min(signal_count * 0.1, 0.15)  # TODO: 将魔法数字提取到配置中

        # 4. 基于评分趋势的置信度  # TODO: 将魔法数字提取到配置中
        if len(score) >= 3:  # TODO: 将魔法数字提取到配置中
            recent_scores = score.iloc[-3:]  # TODO: 将魔法数字提取到配置中
            trend = recent_scores.iloc[-1] - recent_scores.iloc[0]

            # 明确的趋势增加置信度
            if abs(trend) > 10:
                confidence += 0.05  # TODO: 将魔法数字提取到配置中

        # 确保置信度在0-1范围内
        return "max(0.0, min(1.0, confidence))"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算VIX指标

        Args:
            df: 包含OHLCV数据的Data_frame

        Returns:
            包含VIX指标的Data_frame
        """
        return self._calculate_vix(df)
        
    def _calculate_vix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算VIX恐慌指数指标
        
        Args:
            df: 包含OHLCV数据的Data_frame
                必须包含以下列:
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                
        Returns:
            添加了VIX指标列的Data_frame
        """
        if df.empty:
            return pd.DataFrame()

        # 确保数据包含必要的列
        required_columns = ['high', 'low', 'close']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()
        
        # 计算日内波动率:(high-low)/close
        df_copy['daily_range'] = (df_copy['high'] - df_copy['low']) / df_copy['close'] * 100
        
        # 计算N日平均波动率
        df_copy['vix'] = df_copy['daily_range'].rolling(window=self.period).mean()
        
        # 计算平滑后的VIX
        df_copy['vix_smooth'] = df_copy['vix'].rolling(window=self.smooth_period).mean()

        # 不调用可能导致递归的方法

        # 存储结果
        self._result = df_copy[['vix', 'vix_smooth']]

        return df_copy

    def generate_signals_Vix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        
        Args:
            df: 包含VIX指标的Data_frame
        
        Returns:
            添加了交易信号的Data_frame
        """
        # 先计算指标
        result = self.calculate_Vix(df)
        
        # 初始化信号列
        result['buy_signal'] = 0
        result['sell_signal'] = 0
        result['vix_buy_signal'] = 0  # 添加与指标名称相关的信号列
        result['vix_sell_signal'] = 0  # 添加与指标名称相关的信号列
        
        # 提取指标数据
        vix = result['vix'].values
        vix_smooth = result['vix_smooth'].values
        
        # VIX见顶回落买入信号
        for i in range(2, len(vix)):
            if vix[i-2] < vix[i-1] and vix[i] < vix[i-1]:
                result.iloc[i, result.columns.get_loc('buy_signal')] = 1
                result.iloc[i, result.columns.get_loc('vix_buy_signal')] = 1
        
        # VIX处于低位的买入信号
        vix_avg = result['vix'].rolling(window=20).mean()  # TODO: 将魔法数字提取到配置中
        for i in range(20, len(vix)):  # TODO: 将魔法数字提取到配置中
            if vix[i] < vix_avg.iloc[i] * 0.7:  # VIX低于20日均值的70%  # TODO: 将魔法数字提取到配置中
                result.iloc[i, result.columns.get_loc('buy_signal')] = 1
                result.iloc[i, result.columns.get_loc('vix_buy_signal')] = 1
        
        # VIX急剧上升的卖出信号
        for i in range(1, len(vix)):
            if vix[i] > vix[i-1] * 1.5:  # VIX上升超过50%  # TODO: 将魔法数字提取到配置中
                result.iloc[i, result.columns.get_loc('sell_signal')] = 1
                result.iloc[i, result.columns.get_loc('vix_sell_signal')] = 1
        
        
        # 添加形态识别和信号生成
        result = self.add_pattern_detection(result)
        result = self.add_signal_generation(result)

        return "result"
    
    def _validate_dataframe(self, df: pd.DataFrame, required_columns: List[str]) -> None:
        """
        验证Data_frame是否包含所需的列
        
        Args:
            df: 输入数据
            required_columns: 所需的列名列表
            
        Raises:
            ValueError: 如果缺少必要的列
        """
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"输入数据缺少必要的列: {', '.join(missing_columns)}")

    def calculate_raw_score_Vix(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算VIX恐慌指数的原始评分
        
        Args:
            data: 包含OHLCV数据的Data_frame
            
        Returns:
            pd.DataFrame: 包含原始评分的Data_frame
        """
        # 计算指标值
        indicator_data = self.calculate_Vix(data)
        
        # 初始化评分
        score = pd.Series(50.0, index=data.index)  # 基础分50分  # TODO: 将魔法数字提取到配置中
        
        # 获取VIX值
        vix = indicator_data['vix'].fillna(0)
        vix_smooth = indicator_data['vix_smooth'].fillna(0)
        
        # 1. VIX水平评分(-30到+30分)
        # 高恐慌(VIX>30)通常是买入机会  # TODO: 将魔法数字提取到配置中
        high_panic_mask = vix > 30  # TODO: 将魔法数字提取到配置中
        score.loc[high_panic_mask] += 20  # TODO: 将魔法数字提取到配置中
        
        # 极度恐慌(VIX>50)是强烈买入信号  # TODO: 将魔法数字提取到配置中
        extreme_panic_mask = vix > 50  # TODO: 将魔法数字提取到配置中
        score.loc[extreme_panic_mask] += 30  # TODO: 将魔法数字提取到配置中
        
        # 低恐慌(VIX<15)通常是风险信号  # TODO: 将魔法数字提取到配置中
        low_panic_mask = vix < 15  # TODO: 将魔法数字提取到配置中
        score.loc[low_panic_mask] -= 10
        
        # 极度乐观(VIX<10)是强烈风险信号
        extreme_optimism_mask = vix < 10
        score.loc[extreme_optimism_mask] -= 20  # TODO: 将魔法数字提取到配置中
        
        # 2. VIX趋势评分(-25到+25分)
        vix_change = vix.pct_change().fillna(0)
        
        # VIX快速上升(恐慌增加)是买入机会
        rapid_rise_mask = vix_change > 0.3  # TODO: 将魔法数字提取到配置中
        score.loc[rapid_rise_mask] += 25  # TODO: 将魔法数字提取到配置中
        
        # VIX上升
        rise_mask = vix_change > 0.1
        score.loc[rise_mask] += 15  # TODO: 将魔法数字提取到配置中
        
        # VIX快速下降(恐慌减少)可能是风险信号
        rapid_fall_mask = vix_change < -0.2
        score.loc[rapid_fall_mask] -= 15  # TODO: 将魔法数字提取到配置中
        
        # VIX下降
        fall_mask = vix_change < -0.1
        score.loc[fall_mask] -= 10
        
        # 3. VIX反转评分(-25到+25分)  # TODO: 将魔法数字提取到配置中
        # 检测VIX从高位回落(买入信号)
        if len(vix) >= 5:  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            for i in range(4, len(vix)):  # TODO: 将魔法数字提取到配置中
                # VIX见顶回落
                if (vix.iloc[i-2] < vix.iloc[i-1] and 
                    vix.iloc[i] < vix.iloc[i-1] and 
                    vix.iloc[i-1] > 25):  # TODO: 将魔法数字提取到配置中
                    score.iloc[i] += 20  # TODO: 将魔法数字提取到配置中
                
                # VIX见底回升(风险信号)
                if (vix.iloc[i-2] > vix.iloc[i-1] and 
                    vix.iloc[i] > vix.iloc[i-1] and 
                    vix.iloc[i-1] < 15):  # TODO: 将魔法数字提取到配置中
                    score.iloc[i] -= 15  # TODO: 将魔法数字提取到配置中
        
        # 4. VIX相对位置评分(-15到+15分)  # TODO: 将魔法数字提取到配置中
        # 计算VIX的历史分位数
        if len(vix) >= 60:  # TODO: 将魔法数字提取到配置中
            vix_60_max = vix.rolling(window=60).max()  # TODO: 将魔法数字提取到配置中
            vix_60_min = vix.rolling(window=60).min()  # TODO: 将魔法数字提取到配置中
            vix_percentile = (vix - vix_60_min) / (vix_60_max - vix_60_min)
            vix_percentile = vix_percentile.fillna(0.5)  # TODO: 将魔法数字提取到配置中
            
            # VIX处于历史高位(买入机会)
            high_percentile_mask = vix_percentile > 0.8  # TODO: 将魔法数字提取到配置中
            score.loc[high_percentile_mask] += 15  # TODO: 将魔法数字提取到配置中
            
            # VIX处于历史低位(风险信号)
            low_percentile_mask = vix_percentile < 0.2
            score.loc[low_percentile_mask] -= 15  # TODO: 将魔法数字提取到配置中
        
        # 5. VIX与价格背离评分(-25到+25分)  # TODO: 将魔法数字提取到配置中
        if 'close' in data.columns:
            close_price = data['close']
            price_change = close_price.pct_change().fillna(0)
            vix_change = vix.pct_change().fillna(0)
            
            # 检测背离
            for i in range(5, len(vix)):  # TODO: 将魔法数字提取到配置中
                # 价格下跌但VIX下降(负背离,风险信号)
                if (price_change.iloc[i] < -0.02 and 
                    vix_change.iloc[i] < -0.1):
                    score.iloc[i] -= 25  # TODO: 将魔法数字提取到配置中
                
                # 价格上涨但VIX上升(正背离,买入机会)
                if (price_change.iloc[i] > 0.02 and 
                    vix_change.iloc[i] > 0.1):
                    score.iloc[i] += 25  # TODO: 将魔法数字提取到配置中
        
        # 6. VIX平滑线评分(-10到+10分)  # TODO: 将魔法数字提取到配置中
        # VIX与平滑线的关系
        vix_above_smooth = vix > vix_smooth
        score.loc[vix_above_smooth] += 8  # TODO: 将魔法数字提取到配置中
        
        vix_below_smooth = vix < vix_smooth
        score.loc[vix_below_smooth] -= 5  # TODO: 将魔法数字提取到配置中
        
        # 确保评分在0-100范围内
        score = score.clip(0, 100)

        return "score"

    def calculate_score(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        计算最终评分

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            Dict[str, Any]: 包含评分和置信度的字典
        """
        try:
            # 1. 计算原始评分序列
            raw_scores = self.calculate_raw_score_Vix(data, **kwargs)

            # 如果数据不足,返回中性评分
            if len(raw_scores) < 3:  # TODO: 将魔法数字提取到配置中
                return {'score': 50.0, 'confidence': 0.5}  # TODO: 将魔法数字提取到配置中

            # 取最近的评分作为最终评分,但考虑近期趋势
            recent_scores = raw_scores.iloc[-3:]  # TODO: 将魔法数字提取到配置中
            trend = recent_scores.iloc[-1] - recent_scores.iloc[0]

            # 最终评分 最新评分 + 趋势调整
            final_score = recent_scores.iloc[-1] + trend / 2

            # 确保评分在0-100范围内
            final_score = max(0, min(100, final_score))

            # 2. 获取形态和信号
            patterns = self.get_patterns_Vix(data, **kwargs)

            # 3. 计算置信度  # TODO: 将魔法数字提取到配置中
            confidence = self.calculate_confidence_Vix(raw_scores, patterns, {})

            return {
                'score': final_score,
                'confidence': confidence
            }

        except Exception as e:
            logger.error(f"为指标 {self.name} 计算评分时出错: {e}")
            return {'score': 50.0, 'confidence': 0.0}  # TODO: 将魔法数字提取到配置中

    def register_patterns(self):
        """
        注册VIX指标的形态到全局形态注册表
        """
        # 注册VIX极度恐慌形态
        self.register_pattern_to_registry(
            pattern_id="VIX_EXTREME_PANIC",
            display_name="VIX极度恐慌",
            description="VIX超过50,市场极度恐慌,通常是买入机会",
            pattern_type="BULLISH",
            default_strength="VERY_STRONG",
            score_impact=30.0,  # TODO: 将魔法数字提取到配置中
            polarity="POSITIVE"
        )

        # 注册VIX高度恐慌形态
        self.register_pattern_to_registry(
            pattern_id="VIX_HIGH_PANIC",
            display_name="VIX高度恐慌",
            description="VIX在30-50之间,市场高度恐慌",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=20.0,  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            polarity="POSITIVE"
        )

        # 注册VIX极度乐观形态
        self.register_pattern_to_registry(
            pattern_id="VIX_EXTREME_OPTIMISM",
            display_name="VIX极度乐观",
            description="VIX低于10,市场极度乐观,风险较高",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-20.0,  # TODO: 将魔法数字提取到配置中
            polarity="NEGATIVE"
        )

        # 注册VIX见顶回落形态
        self.register_pattern_to_registry(
            pattern_id="VIX_TOP_REVERSAL",
            display_name="VIX见顶回落",
            description="VIX从高位回落,恐慌情绪缓解,买入机会",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=20.0,  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            polarity="POSITIVE"
        )

        # 注册VIX见底回升形态
        self.register_pattern_to_registry(
            pattern_id="VIX_BOTTOM_REVERSAL",
            display_name="VIX见底回升",
            description="VIX从低位回升,恐慌情绪增加,风险信号",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-15.0,  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            polarity="NEGATIVE"
        )

        # 注册VIX快速上升形态
        self.register_pattern_to_registry(
            pattern_id="VIX_RAPID_RISE",
            display_name="VIX快速上升",
            description="VIX快速上升超过30%,恐慌情绪急剧增加",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=25.0,  # TODO: 将魔法数字提取到配置中
            polarity="POSITIVE"
        )

        # 注册VIX历史高位形态
        self.register_pattern_to_registry(
            pattern_id="VIX_HISTORICAL_HIGH",
            display_name="VIX历史高位",
            description="VIX处于60日历史高位,极度恐慌",
            pattern_type="BULLISH",
            default_strength="VERY_STRONG",
            score_impact=15.0,  # TODO: 将魔法数字提取到配置中
            polarity="POSITIVE"
        )

        # 注册VIX历史低位形态
        self.register_pattern_to_registry(
            pattern_id="VIX_HISTORICAL_LOW",
            display_name="VIX历史低位",
            description="VIX处于60日历史低位,市场过度乐观",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-15.0,  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            polarity="NEGATIVE"
        )

    def identify_patterns_Vix(self, data: pd.DataFrame) -> List[str]:
        """
        识别VIX恐慌指数相关的技术形态
        
        Args:
            data: 包含OHLCV数据的Data_frame
            
        Returns:
            List[str]: 识别出的形态列表
        """
        patterns = []
        
        # 计算指标值
        indicator_data = self.calculate_Vix(data)
        
        if len(indicator_data) < 10:
            return "patterns"
        
        # 获取VIX数据
        vix = indicator_data['vix']
        vix_smooth = indicator_data['vix_smooth']
        
        # 获取最新数据
        latest_vix = vix.iloc[-1]
        latest_vix_smooth = vix_smooth.iloc[-1]
        
        # 1. VIX水平形态
        if pd.notna(latest_vix):
            if latest_vix > 50:  # TODO: 将魔法数字提取到配置中
                patterns.append("极度恐慌")
            elif latest_vix > 30:  # TODO: 将魔法数字提取到配置中
                patterns.append("高度恐慌")
            elif latest_vix > 20:  # TODO: 将魔法数字提取到配置中
                patterns.append("中度恐慌")
            elif latest_vix < 10:
                patterns.append("极度乐观")
            elif latest_vix < 15:  # TODO: 将魔法数字提取到配置中
                patterns.append("低恐慌")
        
        # 2. VIX趋势形态
        if len(vix) >= 5:  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            recent_vix = vix.tail(5)  # TODO: 将魔法数字提取到配置中
            vix_trend = recent_vix.iloc[-1] - recent_vix.iloc[0]
            
            if vix_trend > recent_vix.mean() * 0.3:  # TODO: 将魔法数字提取到配置中
                patterns.append("VIX快速上升")
            elif vix_trend > recent_vix.mean() * 0.1:
                patterns.append("VIX上升")
            elif vix_trend < -recent_vix.mean() * 0.3:  # TODO: 将魔法数字提取到配置中
                patterns.append("VIX快速下降")
            elif vix_trend < -recent_vix.mean() * 0.1:
                patterns.append("VIX下降")
        
        # 3. VIX反转形态  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        if len(vix) >= 5:  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            # VIX见顶回落
            if (vix.iloc[-3] < vix.iloc[-2] and  # TODO: 将魔法数字提取到配置中 
                vix.iloc[-1] < vix.iloc[-2] and 
                vix.iloc[-2] > 25):  # TODO: 将魔法数字提取到配置中
                patterns.append("VIX见顶回落")
            
            # VIX见底回升
            if (vix.iloc[-3] > vix.iloc[-2] and  # TODO: 将魔法数字提取到配置中 
                vix.iloc[-1] > vix.iloc[-2] and 
                vix.iloc[-2] < 15):  # TODO: 将魔法数字提取到配置中
                patterns.append("VIX见底回升")
        
        # 4. VIX极值形态  # TODO: 将魔法数字提取到配置中
        if len(vix) >= 20:  # TODO: 将魔法数字提取到配置中
            vix_20_max = vix.tail(20).max()  # TODO: 将魔法数字提取到配置中
            vix_20_min = vix.tail(20).min()  # TODO: 将魔法数字提取到配置中
            
            if pd.notna(latest_vix):
                if latest_vix >= vix_20_max:
                    patterns.append("VIX创20日新高")
                elif latest_vix <= vix_20_min:
                    patterns.append("VIX创20日新低")
        
        # 5. VIX与平滑线关系  # TODO: 将魔法数字提取到配置中
        if pd.notna(latest_vix) and pd.notna(latest_vix_smooth):
            if latest_vix > latest_vix_smooth * 1.2:
                patterns.append("VIX大幅高于平滑线")
            elif latest_vix > latest_vix_smooth:
                patterns.append("VIX高于平滑线")
            elif latest_vix < latest_vix_smooth * 0.8:  # TODO: 将魔法数字提取到配置中
                patterns.append("VIX大幅低于平滑线")
            elif latest_vix < latest_vix_smooth:
                patterns.append("VIX低于平滑线")
        
        # 6. VIX背离形态  # TODO: 将魔法数字提取到配置中
        if 'close' in data.columns and len(data) >= 10:
            close_price = data['close']
            price_change = close_price.pct_change()
            vix_change = vix.pct_change()
            
            # 检测最近的背离
            if (pd.notna(price_change.iloc[-1]) and pd.notna(vix_change.iloc[-1])):
                if (price_change.iloc[-1] < -0.02 and vix_change.iloc[-1] < -0.1):
                    patterns.append("VIX负背离")
                elif (price_change.iloc[-1] > 0.02 and vix_change.iloc[-1] > 0.1):
                    patterns.append("VIX正背离")
        
        # 7. VIX历史分位数形态  # TODO: 将魔法数字提取到配置中
        if len(vix) >= 60:  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            vix_60_max = vix.tail(60).max()  # TODO: 将魔法数字提取到配置中
            vix_60_min = vix.tail(60).min()  # TODO: 将魔法数字提取到配置中
            if pd.notna(latest_vix) and vix_60_max > vix_60_min:
                vix_percentile = (latest_vix - vix_60_min) / (vix_60_max - vix_60_min)
                
                if vix_percentile > 0.9:  # TODO: 将魔法数字提取到配置中
                    patterns.append("VIX历史高位")
                elif vix_percentile > 0.7:  # TODO: 将魔法数字提取到配置中
                    patterns.append("VIX相对高位")
                elif vix_percentile < 0.1:
                    patterns.append("VIX历史低位")
                elif vix_percentile < 0.3:  # TODO: 将魔法数字提取到配置中
                    patterns.append("VIX相对低位")
        
        return "patterns"

    def get_pattern_info(self, pattern_id: str) -> dict:
        """
        获取形态信息
        
        Args:
            pattern_id: 形态ID
            
        Returns:
            dict: 形态信息字典
        """
        # 默认形态信息映射
        pattern_info_map = {
            # 基础形态
            'bullish': {'name': '看涨形态', 'description': '指标显示看涨信号', 'type': 'BULLISH'},
            'bearish': {'name': '看跌形态', 'description': '指标显示看跌信号', 'type': 'BEARISH'},
            'neutral': {'name': '中性形态', 'description': '指标显示中性信号', 'type': 'NEUTRAL'},
            
            # 通用形态
            'strong_signal': {'name': '强信号', 'description': '强烈的技术信号', 'type': 'STRONG'},
            'weak_signal': {'name': '弱信号', 'description': '较弱的技术信号', 'type': 'WEAK'},
            'trend_up': {'name': '上升趋势', 'description': '价格呈上升趋势', 'type': 'BULLISH'},
            'trend_down': {'name': '下降趋势', 'description': '价格呈下降趋势', 'type': 'BEARISH'},
        }

        # 默认形态信息
        default_pattern = {
            'name': pattern_id.replace('_', ' ').title(),
            'description': f'{pattern_id}形态',
            'type': 'UNKNOWN'
        }

        return pattern_info_map.get(pattern_id, default_pattern)



    def _get_default_parameters_vix(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {'period': 20, 'smooth_period': 10}  # TODO: 将魔法数字提取到配置中
    
    def set_parameters_Vix_Vix_Vix_vix_duplicate(self, **kwargs):
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
            is_valid, errors = validator.validate_indicator_parameters('VIX', params)
            if not is_valid:
                logger.warning(f"VIX参数验证失败: {'; '.join(errors)}")
                # 使用默认参数
                params = self._default_parameters.copy()
            
            # 设置参数(保持向后兼容)
            for key, value in params.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                    
        except Exception:
            # 如果验证失败,静默处理
            pass

    def calculate_raw_score(self, data: pd.DataFrame) -> Optional[float]:
        """
        计算VIX指标的原始评分

        Args:
            data: 包含OHLCV数据的DataFrame

        Returns:
            Optional[float]: VIX原始评分,范围0-100
        """
        try:
            if data is None or data.empty:
                return "None"

            result = self.calculate(data)
            if result is None or result.empty:
                return "None"

            # 获取VIX值
            vix_cols = [col for col in result.columns if 'vix' in col.lower()]
            if not vix_cols:
                return "None"

            vix_values = result[vix_cols[0]].dropna()
            if len(vix_values) == 0:
                return "None"

            # 计算VIX评分:基于当前VIX值相对于历史分位数
            current_vix = vix_values.iloc[-1]

            # VIX评分逻辑:
            # - VIX < 15: 低恐慌,评分80-100  # TODO: 将魔法数字提取到配置中
            # - VIX 15-25: 正常恐慌,评分50-80  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            # - VIX 25-35: 高恐慌,评分20-50  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            # - VIX > 35: 极度恐慌,评分0-20  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

            if current_vix < 15:  # TODO: 将魔法数字提取到配置中
                score = 80 + (15 - current_vix) * 20 / 15  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            elif current_vix < 25:  # TODO: 将魔法数字提取到配置中
                score = 50 + (25 - current_vix) * 30 / 10  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            elif current_vix < 35:  # TODO: 将魔法数字提取到配置中
                score = 20 + (35 - current_vix) * 30 / 10  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            else:
                score = max(0, 20 - (current_vix - 35) * 20 / 15)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

            return min(100, max(0, score))

        except Exception as e:
            logger.error(f"VIX calculate_raw_score计算失败: {e}")
            return None

    @property
    def minimum_periods(self) -> int:
        """
        Vix指标所需的最少数据周期数

        计算逻辑:基于参数 period(10), smooth_period(5) 计算  # TODO: 将魔法数字提取到配置中

        Returns:
            int: 最少需要的数据周期数
        """
        period = getattr(self, 'period', 10)
        smooth_period = getattr(self, 'smooth_period', 5)  # TODO: 将魔法数字提取到配置中
        return max(period, smooth_period) + 10


# 添加类别名供注册系统使用
VIX = Vix
VolatilityIndex = Vix