#!/usr/bin/env python3
from utils.dependency_injection import get_logger
"""
INSTITUTIONAL_BEHAVIOR 指标

自动生成的最小化指标实现
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from utils.dependency_injection import get_logger

logger = get_logger(__name__)


class InstitutionalBehavior(BaseIndicator, PatternSignalMixin):
    """
    INSTITUTIONAL_BEHAVIOR 指标
    
    自动生成的最小化实现，支持参数标准化
    """
    
    def __init__(self, **kwargs):
        """
        初始化INSTITUTIONAL_BEHAVIOR指标
        
        Args:
            **kwargs: 指标参数
        """
        super().__init__()
        self.name = "INSTITUTIONAL_BEHAVIOR"
        
        # 设置默认参数
        self._default_parameters = self._get_default_parameters_institutionalbehavior()
        
        # 应用用户参数
        self.set_parameters_Behavior(**kwargs)
    
    def _get_default_parameters_institutionalbehavior(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {"period": 14}
    
    def set_parameters_Behavior(self, **kwargs):
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
            is_valid, errors = validator.validate_indicator_parameters('INSTITUTIONAL_BEHAVIOR', params)
            if not is_valid:
                # 静默处理验证失败，避免过多警告
                pass
                # 使用默认参数
                params = self._default_parameters.copy()
            
            # 设置参数
            self.period = params.get('period', 14)
                    
        except Exception:
            # 如果验证失败，静默处理，保持向后兼容
            self.period = 14
    
    def calculate_Behavior(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算INSTITUTIONAL_BEHAVIOR指标
        
        Args:
            data: 包含OHLCV数据的Data_frame
            
        Returns:
            添加了INSTITUTIONAL_BEHAVIOR指标的Data_frame
        """
        result = self._calculate_institutionalbehavior(data, **kwargs)
        self._result = result
        return result
    
    def _calculate_institutionalbehavior(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        内部计算INSTITUTIONAL_BEHAVIOR指标
        
        Args:
            data: 包含OHLCV数据的Data_frame
            
        Returns:
            添加了INSTITUTIONAL_BEHAVIOR指标的Data_frame
        """
        df = data.copy()
        
        # 最小化实现：返回原数据加上一个简单的计算列
        df[f'INSTITUTIONAL_BEHAVIOR_VALUE'] = df['close'].rolling(window=self.period).mean()
        
        
        # 添加形态识别和信号生成
        df = self.add_pattern_detection(df)
        df = self.add_signal_generation(df)

        # 重写专用信号逻辑：基于评分值的阈值判断
        # 对于state_type指标，使用评分阈值模式
        score_threshold = 50.0  # 默认阈值
        df.loc[:, 'buy_signal'] = df[f'INSTITUTIONAL_BEHAVIOR_VALUE'] >= score_threshold
        df.loc[:, 'sell_signal'] = df[f'INSTITUTIONAL_BEHAVIOR_VALUE'] < score_threshold
        df.loc[:, 'hold_signal'] = df[f'INSTITUTIONAL_BEHAVIOR_VALUE'] < score_threshold

        return df
    
    def calculate_raw_score_Behavior(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算原始评分"""
        if not self.has_result():
            self.calculate_Behavior(data, **kwargs)
        
        # 机构行为评分：基于大单交易和资金流向分析
        df = data.copy()
        
        # 计算机构行为相关指标
        # 1. 大单分析（基于成交量和价格变化）
        volume_ma = df['volume'].rolling(window=20).mean()
        large_volume = df['volume'] > volume_ma * 2  # 大成交量
        
        # 2. 价格稳定性（机构通常不会造成剧烈波动）
        price_change = df['close'].pct_change()
        price_volatility = price_change.rolling(window=10).std()
        stable_price = price_volatility < price_volatility.rolling(window=30).mean()
        
        # 3. 连续性分析（机构操作通常有连续性）
        volume_trend = df['volume'].rolling(window=5).mean() / df['volume'].rolling(window=20).mean()
        continuous_volume = volume_trend > 1.2
        
        # 4. 逆向操作检测（机构逆向思维）
        price_down = df['close'] < df['close'].shift(1)
        volume_up = df['volume'] > df['volume'].shift(1)
        contrarian_signal = price_down & volume_up  # 价跌量增
        
        # 5. 资金流向估算
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        money_flow_ma = money_flow.rolling(window=20).mean()
        strong_inflow = money_flow > money_flow_ma * 1.5
        
        # 复合评分计算
        scores = pd.Series(50.0, index=data.index)  # 基准分
        
        # 大单买入信号 (30%)
        large_buy = large_volume & (df['close'] > df['open'])
        large_sell = large_volume & (df['close'] < df['open'])
        scores += np.where(large_buy, 20, 0)
        scores += np.where(large_sell, -15, 0)
        
        # 价格稳定性 (20%)
        stable_accumulation = stable_price & (df['volume'] > volume_ma)
        scores += np.where(stable_accumulation, 15, 0)  # 稳定吸筹
        
        # 连续操作 (20%)
        continuous_buy = continuous_volume & (df['close'] > df['close'].shift(3))
        scores += np.where(continuous_buy, 12, 0)
        
        # 逆向操作 (15%)
        contrarian_buy = contrarian_signal & (df['close'] > df['close'].rolling(window=5).mean())
        scores += np.where(contrarian_buy, 18, 0)  # 逆向买入强信号
        
        # 资金流向 (15%)
        strong_buy_flow = strong_inflow & (df['close'] > df['open'])
        weak_sell_flow = (money_flow < money_flow_ma * 0.8) & (df['close'] < df['open'])
        scores += np.where(strong_buy_flow, 15, 0)
        scores += np.where(weak_sell_flow, -10, 0)
        
        # 机构建仓模式识别
        # 温和建仓：价格缓慢上涨，成交量适中
        gentle_accumulation = (
            (df['close'] > df['close'].shift(5)) &  # 5日上涨
            (price_volatility < price_volatility.rolling(window=20).mean()) &  # 波动率低
            (df['volume'] > volume_ma * 1.1) &  # 成交量略大
            (df['volume'] < volume_ma * 2.0)    # 但不过大
        )
        scores += np.where(gentle_accumulation, 20, 0)
        
        # 机构拉升模式：突然放量上涨
        institutional_pump = (
            (df['close'] > df['close'].shift(1) * 1.03) &  # 单日涨幅>3%
            (df['volume'] > volume_ma * 2.5) &  # 大幅放量
            (df['close'] == df['high'])  # 收盘价接近最高价
        )
        scores += np.where(institutional_pump, 25, 0)
        
        # 机构护盘：下跌时成交量萎缩
        institutional_support = (
            (df['close'] < df['close'].shift(1)) &  # 价格下跌
            (df['volume'] < volume_ma * 0.8) &  # 成交量萎缩
            (df['low'] > df['low'].rolling(window=10).min() * 1.02)  # 有支撑
        )
        scores += np.where(institutional_support, 10, 0)
        
        # 限制评分范围
        scores = np.clip(scores, 0, 100)
        
        return scores
    
    def calculate_confidence_Behavior(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """计算置信度"""
        return 0.5
    
    def get_patterns_Behavior(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """获取形态"""
        return pd.DataFrame(index=data.index)


# 为了向后兼容，创建别名
INSTITUTIONAL_BEHAVIOR = InstitutionalBehavior
institutional_behavior = InstitutionalBehavior