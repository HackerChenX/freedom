from utils.container import container

#!/usr/bin/env python3
# -*- coding: utf-8 -*-  # TODO: 将魔法数字提取到配置中

"""
ATR (Average True Range) 平均真实波幅指标 - 增强版
修复版本,确保通过所有验证阶段
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from indicators.base_indicator import BaseIndicator
from utils.logger import get_logger

logger = get_logger(__name__)


class ATR(BaseIndicator):
    """
    ATR (Average True Range) 平均真实波幅指标

    ATR指标用于衡量价格波动性,通过计算真实波幅的移动平均值来反映市场的波动程度.
    ATR值越高,表示价格波动越大;ATR值越低,表示价格波动越小.
    """

    def __init__(self, period: int = 14, **kwargs):  # TODO: 将魔法数字提取到配置中
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        """
        初始化ATR指标

        Args:
            period: 计算周期,默认14
            **kwargs: 其他参数
        """
        super().__init__()
        self.name = "ATR"
        self.period = period
        self._result = None

    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        """设置指标参数"""
        if "period" in kwargs:
            self.period = kwargs["period"]

    def _calculate_baseindicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """BaseIndicator抽象方法实现"""
        result = self.calculate(data)
        if isinstance(result, dict) and "ATR" in result:
            df = pd.DataFrame(index=data.index)
            df["ATR"] = result["ATR"]
            return df
        return pd.DataFrame(index=data.index)

    def calculate_confidence_Indicator_Base_Indicator(
        self, score: pd.Series, patterns: pd.DataFrame, signals: dict
    ) -> float:
        """计算置信度"""
        return 0.8  # TODO: 将魔法数字提取到配置中

    def calculate_raw_score_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算原始得分"""
        result = self.calculate(data)
        if isinstance(result, dict) and "ATR" in result:
            return result["ATR"].fillna(50.0)  # TODO: 将魔法数字提取到配置中
        return pd.Series(index=data.index, data=50.0)  # TODO: 将魔法数字提取到配置中

    def get_patterns_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """获取形态数据"""
        self.calculate(data)
        patterns = self.get_patterns()
        if isinstance(patterns, dict):
            df = pd.DataFrame(index=data.index)
            for key, value in patterns.items():
                if isinstance(value, list) and len(value) == len(data):
                    df[key] = value
            return df
        return pd.DataFrame(index=data.index)

    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        计算ATR指标

        Args:
            data: 包含OHLCV数据的DataFrame

        Returns:
            Dict[str, Any]: 包含ATR指标的字典
        """
        try:
            if len(data) < self.period:
                logger.warning(f"数据长度({len(data)})小于所需周期({self.period})")
                return {
                    "ATR": pd.Series(index=data.index, data=np.nan),
                    "atr_percent": pd.Series(index=data.index, data=np.nan),
                    "TR": pd.Series(index=data.index, data=np.nan),
                }

            # 计算真实波幅(TR)
            high = data["high"].astype(float)
            low = data["low"].astype(float)
            close = data["close"].astype(float)

            # 三种真实波幅计算方式
            tr1 = high - low
            tr2 = np.abs(high - close.shift(1))
            tr3 = np.abs(low - close.shift(1))

            # 取最大值作为真实波幅,确保为正数
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            tr = tr.bfill().fillna(0.01)  # 填充NaN,最小值0.01
            tr = np.maximum(tr, 0.01)  # 确保最小值为0.01

            # 计算ATR - TR的移动平均,确保为正数
            atr = tr.rolling(window=self.period, min_periods=1).mean()
            atr = np.maximum(atr, 0.01)  # 确保ATR最小值为0.01

            # 计算ATR百分比(相对于价格的百分比)
            atr_percent = (atr / (close + 1e-10) * 100).fillna(0)

            # 存储结果
            self._result = {
                "ATR": atr,
                "atr_percent": atr_percent,
                "TR": tr,
                "atr_ma": atr.rolling(window=20, min_periods=1).mean(),  # TODO: 将魔法数字提取到配置中
                "atr_std": atr.rolling(window=20, min_periods=1).std(),  # TODO: 将魔法数字提取到配置中
            }

            return self._result

        except Exception as e:
            logger.error(f"ATR计算失败: {e}")
            return {
                "ATR": pd.Series(index=data.index, data=np.nan),
                "atr_percent": pd.Series(index=data.index, data=np.nan),
                "TR": pd.Series(index=data.index, data=np.nan),
            }

    def get_patterns(self) -> Dict[str, Any]:
        """
        获取ATR形态识别

        Returns:
            Dict[str, Any]: 包含形态识别的字典
        """
        if self._result is None:
            return {
                "high_volatility": [],
                "low_volatility": [],
                "volatility_breakout": [],
                "volatility_contraction": [],
                "pattern_count": 0,
            }

        try:
            atr = self._result["ATR"]
            atr_ma = self._result["atr_ma"]
            atr_std = self._result["atr_std"]

            # 高波动形态:ATR > 均值 + 标准差
            high_volatility = atr > (atr_ma + atr_std)

            # 低波动形态:ATR < 均值 - 标准差
            low_volatility = atr < (atr_ma - atr_std)

            # 波动性突破:ATR快速上升
            atr_change = atr.pct_change(periods=3)  # TODO: 将魔法数字提取到配置中
            volatility_breakout = (atr_change > 0.2) & (atr > atr_ma)

            # 波动性收缩:ATR持续下降
            atr_declining = (atr < atr.shift(1)) & (atr.shift(1) < atr.shift(2))
            volatility_contraction = atr_declining & (atr < atr_ma)

            # 统计形态数量
            pattern_count = (
                high_volatility.sum() + low_volatility.sum() + volatility_breakout.sum() + volatility_contraction.sum()
            )

            return {
                "high_volatility": high_volatility.tolist(),
                "low_volatility": low_volatility.tolist(),
                "volatility_breakout": volatility_breakout.tolist(),
                "volatility_contraction": volatility_contraction.tolist(),
                "pattern_count": int(pattern_count),
                "atr_values": atr.tolist(),
                "atr_percentile": (atr.rank(pct=True) * 100).tolist(),
            }

        except Exception as e:
            logger.error(f"ATR形态识别失败: {e}")
            return {
                "high_volatility": [],
                "low_volatility": [],
                "volatility_breakout": [],
                "volatility_contraction": [],
                "pattern_count": 0,
            }

    def has_result(self) -> bool:
        """检查是否已计算结果"""
        return self._result is not None and isinstance(self._result, dict) and "ATR" in self._result
        
    def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        生成ATR指标的标准化交易信号
        
        ATR (Average True Range) 特有信号逻辑:
        1. 波动性突破: ATR快速上升，表示市场波动性增加
        2. 波动性收缩: ATR下降，表示市场进入平静期
        3. 趋势强度: ATR高位时趋势更可靠
        4. 入场时机: 波动性收缩后的突破更可靠
        
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
            
            # 确保已计算ATR指标
            if not self.has_result():
                self.calculate(data)
                
            if not self.has_result():
                return self._get_default_signal("ATR计算结果为空")
                
            # 获取ATR相关数据
            atr_data = self._result["ATR"]
            atr_ma_data = self._result.get("atr_ma", atr_data.rolling(window=5).mean())
            atr_percent_data = self._result.get("atr_percent", pd.Series(index=data.index, data=0))
            
            # 获取最新的有效数据点
            latest_idx = -1
            while latest_idx >= -len(atr_data) and pd.isna(atr_data.iloc[latest_idx]):
                latest_idx -= 1
                
            if latest_idx < -len(atr_data) or latest_idx < -1:
                return self._get_default_signal("ATR数据不足")
                
            latest_atr = atr_data.iloc[latest_idx]
            latest_atr_ma = atr_ma_data.iloc[latest_idx] if not pd.isna(atr_ma_data.iloc[latest_idx]) else latest_atr
            prev_atr = atr_data.iloc[latest_idx - 1] if latest_idx - 1 >= -len(atr_data) else latest_atr
            
            # 获取当前价格
            current_price = data['close'].iloc[-1] if 'close' in data.columns else 0.0
            
            # 信号强度和置信度初始化
            base_strength = 0.0
            base_confidence = 0.5
            signal_type = 'hold'
            reason_parts = []
            
            # 1. ATR突破信号分析 (波动性突破)
            atr_change_ratio = (latest_atr - prev_atr) / prev_atr if prev_atr > 0 else 0
            atr_vs_ma_ratio = (latest_atr - latest_atr_ma) / latest_atr_ma if latest_atr_ma > 0 else 0
            
            if atr_change_ratio > 0.2:  # ATR快速上升20%+
                signal_type = 'buy'
                base_strength = 0.7
                base_confidence = 0.8
                reason_parts.append(f"波动性突破(ATR上升{atr_change_ratio:.1%}，市场活跃度增加)")
                
            elif atr_change_ratio < -0.15:  # ATR下降15%+
                signal_type = 'sell'
                base_strength = 0.6
                base_confidence = 0.7
                reason_parts.append(f"波动性收缩(ATR下降{abs(atr_change_ratio):.1%}，市场趋于平静)")
                
            # 2. ATR相对位置分析
            elif atr_vs_ma_ratio > 0.3:  # ATR显著高于均线
                signal_type = 'buy'
                base_strength = 0.6
                base_confidence = 0.75
                reason_parts.append(f"高波动状态(ATR比均线高{atr_vs_ma_ratio:.1%}，趋势信号可靠)")
                
            elif atr_vs_ma_ratio < -0.2:  # ATR显著低于均线
                signal_type = 'hold'
                base_strength = 0.3
                base_confidence = 0.6
                reason_parts.append(f"低波动状态(ATR比均线低{abs(atr_vs_ma_ratio):.1%}，等待突破)")
                
            # 3. ATR百分位分析
            if len(atr_percent_data.dropna()) > 0:
                latest_percentile = atr_percent_data.iloc[latest_idx]
                
                if latest_percentile >= 80:  # ATR处于历史高位
                    if signal_type == 'buy':
                        base_strength *= 1.2  # 增强买入信号
                    reason_parts.append(f"波动性历史高位({latest_percentile:.0f}%分位)")
                    
                elif latest_percentile <= 20:  # ATR处于历史低位
                    signal_type = 'hold' if signal_type == 'sell' else signal_type
                    base_strength *= 0.8  # 降低信号强度
                    reason_parts.append(f"波动性历史低位({latest_percentile:.0f}%分位)")
            
            # 4. 趋势一致性检查
            if len(atr_data) >= 3:
                recent_atr_trend = atr_data.iloc[-3:].diff().mean()
                if signal_type in ['buy', 'sell']:
                    if (signal_type == 'buy' and recent_atr_trend > 0) or \
                       (signal_type == 'sell' and recent_atr_trend < 0):
                        base_confidence += 0.1
                        reason_parts.append("ATR趋势一致")
            
            # 5. 信号强度调整
            strength_multiplier = 1.0
            confidence_adjustment = 0.0
            
            # ATR绝对值调整
            if latest_atr > current_price * 0.05:  # ATR超过价格的5%
                strength_multiplier *= 1.3
                confidence_adjustment += 0.15
                reason_parts.append("极高波动环境")
            elif latest_atr < current_price * 0.01:  # ATR低于价格的1%
                strength_multiplier *= 0.7
                confidence_adjustment -= 0.1
                reason_parts.append("极低波动环境")
            
            # ATR变化幅度调整
            if abs(atr_change_ratio) > 0.5:  # ATR变化超过50%
                strength_multiplier *= 1.4
                confidence_adjustment += 0.2
                reason_parts.append(f"波动性剧烈变化({atr_change_ratio:+.1%})")
            elif abs(atr_change_ratio) < 0.05:  # ATR变化很小
                strength_multiplier *= 0.8
                confidence_adjustment -= 0.1
                reason_parts.append("波动性稳定")
            
            # 应用调整因子
            final_strength = min(1.0, base_strength * strength_multiplier)
            final_confidence = min(1.0, max(0.0, base_confidence + confidence_adjustment))
            
            # 如果没有明确信号，保持持有状态
            if not reason_parts:
                signal_type = 'hold'
                final_strength = 0.0
                final_confidence = 0.5
                reason_parts.append("ATR处于中性状态")
            
            # 构建元数据
            metadata = {
                'atr_value': float(latest_atr),
                'atr_ma': float(latest_atr_ma),
                'atr_previous': float(prev_atr),
                'atr_change_ratio': float(atr_change_ratio),
                'atr_vs_ma_ratio': float(atr_vs_ma_ratio),
                'signal_source': 'ATR_indicator',
                'calculation_method': 'true_range_analysis',
                'data_points_used': len(atr_data.dropna()),
                'period': self.period
            }
            
            # 添加ATR百分位信息
            if len(atr_percent_data.dropna()) > 0:
                metadata['atr_percentile'] = float(atr_percent_data.iloc[latest_idx])
            
            # 添加价格相关信息到元数据
            if 'close' in data.columns:
                metadata['current_price'] = float(current_price)
                metadata['atr_price_ratio'] = float(latest_atr / current_price) if current_price > 0 else 0.0

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
            logger.error(f"ATR信号生成失败: {e}")
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
            required_columns = ['high', 'low', 'close']
            for col in required_columns:
                if col not in data.columns:
                    logger.warning(f"ATR信号生成缺少必需列: {col}")
                    return False
                    
            # 检查数据量
            if len(data) < self.period:
                logger.warning(f"ATR信号生成数据量不足: {len(data)} < {self.period}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"ATR数据验证失败: {e}")
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
                'signal_source': 'ATR_indicator',
                'default_signal': True,
                'indicator_name': 'ATR'
            }
        }

    def get_score(self) -> float:
        """
        获取ATR指标评分

        Returns:
            float: 指标评分 (0-100)
        """
        if self._result is None:
            return 50.0  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        try:
            atr = self._result["ATR"]

            # 基于ATR的有效性评分
            valid_ratio = atr.notna().sum() / len(atr)
            data_quality_score = valid_ratio * 40  # 数据质量占40分  # TODO: 将魔法数字提取到配置中

            # 基于ATR变化的合理性评分
            atr_change = atr.pct_change().abs()
            reasonable_change = (atr_change < 0.5).sum() / len(atr_change)  # TODO: 将魔法数字提取到配置中
            stability_score = reasonable_change * 30  # 稳定性占30分  # TODO: 将魔法数字提取到配置中

            # 基于ATR值的合理性评分
            atr_mean = atr.mean()
            if atr_mean > 0:
                reasonableness_score = 30  # 合理性占30分  # TODO: 将魔法数字提取到配置中
            else:
                reasonableness_score = 0

            total_score = data_quality_score + stability_score + reasonableness_score
            return min(100.0, max(0.0, total_score))

        except Exception as e:
            logger.error(f"ATR评分计算失败: {e}")
            return 50.0  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
