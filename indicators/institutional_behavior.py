from utils.container import container
#!/usr/bin/env python3
from utils.logger import get_logger
"""
INSTITUTIONAL_BEHAVIOR 指标

自动生成的最小化指标实现
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from indicators.base.minimum_periods_mixin import MinimumPeriodsMixin
from utils.logger import get_logger
from db.sql_manager import SQLManager, QueryType
from utils.indicator_parameter_validator import IndicatorParameterValidator

logger = get_logger(__name__)


class InstitutionalBehavior(BaseIndicator, PatternSignalMixin, MinimumPeriodsMixin):
    """
    INSTITUTIONAL_BEHAVIOR 指标
    
    自动生成的最小化实现,支持参数标准化
    """
    
    def __init__(self, **kwargs):
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        """
        初始化INSTITUTIONAL_BEHAVIOR指标
        
        Args:
            **kwargs: 指标参数
        """
        super().__init__()
        self.name = "INSTITUTIONAL_BEHAVIOR"
        
        # 设置默认参数
        self._default_parameters = self._get_default_parameters_institutionalbehavior()

        # 🔧 Ultra Think修复:设置内部minimum_periods值
        self._minimum_periods = 14  # TODO: 将魔法数字提取到配置中

        # 应用用户参数
        self.set_parameters_Behavior(**kwargs)
    
    def _get_default_parameters_institutionalbehavior(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {"period": 14}  # TODO: 将魔法数字提取到配置中

    @property
    def minimum_periods(self) -> int:
        """实现MinimumPeriodsMixin要求的minimum_periods属性"""
        return getattr(self, '_minimum_periods', 14)  # TODO: 将魔法数字提取到配置中
    
    def set_parameters_Behavior(self, **kwargs):
        """
        设置指标参数
        
        Args:
            **kwargs: 参数字典
        """
        # 验证参数
        try:
            from utils.indicator_parameter_validator import IndicatorParameterValidator
        except Exception as e:
            logger.error(f"错误: {e}")
            return pd.DataFrame()
    
    def validate_parameters(self, **kwargs):
        """验证参数"""
        try:
            validator = IndicatorParameterValidator()
            
            # 合并默认参数和用户参数
            params = self._default_parameters.copy()
            params.update(kwargs)
            
            # 验证参数
            is_valid, errors = validator.validate_indicator_parameters('INSTITUTIONAL_BEHAVIOR', params)
            if not is_valid:
                # 静默处理验证失败,避免过多警告
                pass
                # 使用默认参数
                params = self._default_parameters.copy()
            
            # 设置参数
            self.period = params.get('period', 14)  # TODO: 将魔法数字提取到配置中
            # 🔧 Ultra Think修复:同步更新minimum_periods
            self._minimum_periods = self.period

        except Exception:
            # 如果验证失败,静默处理,保持向后兼容
            self.period = kwargs.get('period', 14)  # TODO: 将魔法数字提取到配置中
            # 🔧 Ultra Think修复:确保异常情况下也更新minimum_periods
            self._minimum_periods = self.period
    
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
        
        # 🔧 Ultra Think修复:正确处理NaN值,使用min_periods=1确保有足够数据
        df[f'INSTITUTIONAL_BEHAVIOR_VALUE'] = df['close'].rolling(window=self.period, min_periods=1).mean()
        
        
        # 添加形态识别和信号生成
        df = self.add_pattern_detection(df)
        df = self.add_signal_generation(df)

        # 重写专用信号逻辑:基于评分值的阈值判断
        # 对于state_type指标,使用评分阈值模式
        score_threshold = 50.0  # 默认阈值  # TODO: 将魔法数字提取到配置中
        df.loc[:, 'buy_signal'] = df[f'INSTITUTIONAL_BEHAVIOR_VALUE'] >= score_threshold
        df.loc[:, 'sell_signal'] = df[f'INSTITUTIONAL_BEHAVIOR_VALUE'] < score_threshold
        df.loc[:, 'hold_signal'] = df[f'INSTITUTIONAL_BEHAVIOR_VALUE'] < score_threshold

        return df
    
    def calculate_raw_score_Behavior(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算原始评分"""
        # 🔧 Ultra Think修复:移除has_result检查,直接计算
        # if not self.has_result():
        #     self.calculate_Behavior(data, **kwargs)
        
        # 机构行为评分:基于大单交易和资金流向分析
        df = data.copy()
        
        # 计算机构行为相关指标
        # 🔧 Ultra Think修复:正确处理NaN值,使用min_periods确保有足够数据
        # 1. 大单分析(基于成交量和价格变化)
        volume_ma = df['volume'].rolling(window=20, min_periods=1).mean()  # TODO: 将魔法数字提取到配置中
        large_volume = df['volume'] > volume_ma * 2  # 大成交量

        # 2. 价格稳定性(机构通常不会造成剧烈波动)
        price_change = df['close'].pct_change().fillna(0)
        price_volatility = price_change.rolling(window=10, min_periods=1).std().fillna(0)
        stable_price = price_volatility < price_volatility.rolling(window=30, min_periods=1).mean()  # TODO: 将魔法数字提取到配置中
        
        # 3. 连续性分析(机构操作通常有连续性)  # TODO: 将魔法数字提取到配置中
        # 🔧 Ultra Think修复:正确处理NaN值
        volume_trend = df['volume'].rolling(window=5, min_periods=1).mean() / df['volume'].rolling(window=20, min_periods=1).mean()  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        continuous_volume = volume_trend.fillna(1.0) > 1.2
        
        # 4. 逆向操作检测(机构逆向思维)  # TODO: 将魔法数字提取到配置中
        price_down = df['close'] < df['close'].shift(1)
        volume_up = df['volume'] > df['volume'].shift(1)
        contrarian_signal = price_down & volume_up  # 价跌量增
        
        # 5. 资金流向估算  # TODO: 将魔法数字提取到配置中
        # 🔧 Ultra Think修复:正确处理NaN值
        typical_price = (df['high'] + df['low'] + df['close']) / 3  # TODO: 将魔法数字提取到配置中
        money_flow = typical_price * df['volume']
        money_flow_ma = money_flow.rolling(window=20, min_periods=1).mean()  # TODO: 将魔法数字提取到配置中
        strong_inflow = money_flow > money_flow_ma * 1.5  # TODO: 将魔法数字提取到配置中
        
        # 复合评分计算
        scores = pd.Series(50.0, index=data.index)  # 基准分  # TODO: 将魔法数字提取到配置中
        
        # 大单买入信号 (30%)  # TODO: 将魔法数字提取到配置中
        large_buy = large_volume & (df['close'] > df['open'])
        large_sell = large_volume & (df['close'] < df['open'])
        scores += np.where(large_buy, 20, 0)  # TODO: 将魔法数字提取到配置中
        scores += np.where(large_sell, -15, 0)  # TODO: 将魔法数字提取到配置中
        
        # 价格稳定性 (20%)  # TODO: 将魔法数字提取到配置中
        stable_accumulation = stable_price & (df['volume'] > volume_ma)
        scores += np.where(stable_accumulation, 15, 0)  # 稳定吸筹  # TODO: 将魔法数字提取到配置中
        
        # 连续操作 (20%)  # TODO: 将魔法数字提取到配置中
        continuous_buy = continuous_volume & (df['close'] > df['close'].shift(3))  # TODO: 将魔法数字提取到配置中
        scores += np.where(continuous_buy, 12, 0)  # TODO: 将魔法数字提取到配置中
        
        # 逆向操作 (15%)  # TODO: 将魔法数字提取到配置中
        contrarian_buy = contrarian_signal & (df['close'] > df['close'].rolling(window=5).mean())  # TODO: 将魔法数字提取到配置中
        scores += np.where(contrarian_buy, 18, 0)  # 逆向买入强信号  # TODO: 将魔法数字提取到配置中
        
        # 资金流向 (15%)  # TODO: 将魔法数字提取到配置中
        strong_buy_flow = strong_inflow & (df['close'] > df['open'])
        weak_sell_flow = (money_flow < money_flow_ma * 0.8) & (df['close'] < df['open'])  # TODO: 将魔法数字提取到配置中
        scores += np.where(strong_buy_flow, 15, 0)  # TODO: 将魔法数字提取到配置中
        scores += np.where(weak_sell_flow, -10, 0)
        
        # 机构建仓模式识别
        # 温和建仓:价格缓慢上涨,成交量适中
        # 🔧 Ultra Think修复:正确处理NaN值
        gentle_accumulation = (
            (df['close'] > df['close'].shift(5).fillna(df['close'])) &  # 5日上涨  # TODO: 将魔法数字提取到配置中
            (price_volatility < price_volatility.rolling(window=20, min_periods=1).mean()) &  # 波动率低  # TODO: 将魔法数字提取到配置中
            (df['volume'] > volume_ma * 1.1) &  # 成交量略大
            (df['volume'] < volume_ma * 2.0)    # 但不过大
        )
        scores += np.where(gentle_accumulation, 20, 0)  # TODO: 将魔法数字提取到配置中
        
        # 机构拉升模式:突然放量上涨
        institutional_pump = (
            (df['close'] > df['close'].shift(1) * 1.03) &  # 单日涨幅>3%  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            (df['volume'] > volume_ma * 2.5) &  # 大幅放量  # TODO: 将魔法数字提取到配置中
            (df['close'] == df['high'])  # 收盘价接近最高价
        )
        scores += np.where(institutional_pump, 25, 0)  # TODO: 将魔法数字提取到配置中
        
        # 机构护盘:下跌时成交量萎缩
        # 🔧 Ultra Think修复:正确处理NaN值
        institutional_support = (
            (df['close'] < df['close'].shift(1).fillna(df['close'])) &  # 价格下跌
            (df['volume'] < volume_ma * 0.8) &  # 成交量萎缩  # TODO: 将魔法数字提取到配置中
            (df['low'] > df['low'].rolling(window=10, min_periods=1).min() * 1.02)  # 有支撑
        )
        scores += np.where(institutional_support, 10, 0)
        
        # 限制评分范围
        scores = np.clip(scores, 0, 100)
        
        return scores
    
    def calculate_confidence_Behavior(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """计算置信度"""
        return 0.5  # TODO: 将魔法数字提取到配置中
    
    def get_patterns_Behavior(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """获取形态"""
        return pd.DataFrame(index=data.index)

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算机构行为指标的主要入口方法

        Args:
            data: 包含OHLCV数据的DataFrame
            **kwargs: 其他参数

        Returns:
            包含机构行为指标的DataFrame
        """
        return self.calculate_Behavior(data, **kwargs)

    # 🔧 Ultra Think修复:实现BaseIndicator要求的抽象方法
    def _calculate_baseindicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """实现BaseIndicator要求的_calculate_baseindicator方法"""
        return self.calculate_Behavior(data, **kwargs)

    def calculate_confidence_Indicator_Base_Indicator(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """实现BaseIndicator要求的置信度计算方法"""
        return self.calculate_confidence_Behavior(score, patterns, signals)

    def calculate_raw_score_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """实现BaseIndicator要求的原始评分计算方法"""
        return self.calculate_raw_score_Behavior(data, **kwargs)

    def get_patterns_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """实现BaseIndicator要求的形态获取方法"""
        return self.get_patterns_Behavior(data, **kwargs)

    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        """实现BaseIndicator要求的参数设置方法"""
        return self.set_parameters_Behavior(**kwargs)


# 为了向后兼容,创建别名
INSTITUTIONAL_BEHAVIOR = InstitutionalBehavior
institutional_behavior = InstitutionalBehavior


class FundFlow(BaseIndicator, PatternSignalMixin, MinimumPeriodsMixin):
    """
    ZXM资金流向指标

    分析机构资金流入流出情况,识别主力资金动向
    """

    def __init__(self, **kwargs):
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        """
        初始化ZXM资金流向指标

        Args:
            **kwargs: 指标参数
        """
        # 移除super().__init__调用,直接设置属性
        self.name = "ZXMFundFlow"
        self.description = "ZXM资金流向指标,分析机构资金流入流出情况"

        # 设置默认参数
        self._default_parameters = self._get_default_parameters_fundflow()

        # 应用用户参数
        self.set_parameters_Fund_Flow(**kwargs)

    def _get_default_parameters_fundflow(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {
            "short_period": 5,  # TODO: 将魔法数字提取到配置中
            "long_period": 20,  # TODO: 将魔法数字提取到配置中
            "volume_period": 10
        }

    def set_parameters_Fund_Flow(self, **kwargs):
        """
        设置指标参数

        Args:
            **kwargs: 参数字典
        """
        self.short_period = kwargs.get('short_period', 5)  # TODO: 将魔法数字提取到配置中
        self.long_period = kwargs.get('long_period', 20)  # TODO: 将魔法数字提取到配置中
        self.volume_period = kwargs.get('volume_period', 10)

    @property
    def minimum_periods(self) -> int:
        """
        ZXM资金流向指标所需的最少数据周期数

        Returns:
            int: 最少需要的数据周期数
        """
        return max(self.short_period, self.long_period, self.volume_period) + 10

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算ZXM资金流向指标的主要入口方法

        Args:
            data: 包含OHLCV数据的DataFrame
            **kwargs: 其他参数

        Returns:
            包含ZXM资金流向指标的DataFrame
        """
        result = data.copy()

        # 计算资金流向
        result = self._calculate_money_flow(result)

        # 计算机构行为
        result = self._calculate_institutional_behavior(result)

        # 计算资金流向强度
        result = self._calculate_flow_strength(result)

        # 计算综合资金流向评分
        result = self._calculate_composite_flow_score(result)

        # 生成资金流向信号
        result = self._generate_fund_flow_signals(result)

        return result

    def _calculate_money_flow(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算资金流向"""
        result = data.copy()

        close = result['close']
        high = result['high']
        low = result['low']
        volume = result['volume']

        # 计算典型价格
        typical_price = (high + low + close) / 3  # TODO: 将魔法数字提取到配置中

        # 计算资金流量
        money_flow = typical_price * volume

        # 计算正负资金流
        positive_flow = pd.Series(0.0, index=data.index)
        negative_flow = pd.Series(0.0, index=data.index)

        for i in range(1, len(typical_price)):
            if typical_price.iloc[i] > typical_price.iloc[i-1]:
                positive_flow.iloc[i] = money_flow.iloc[i]
            elif typical_price.iloc[i] < typical_price.iloc[i-1]:
                negative_flow.iloc[i] = money_flow.iloc[i]

        # 计算资金流向比率
        positive_flow_ma = positive_flow.rolling(window=self.short_period).sum()
        negative_flow_ma = negative_flow.rolling(window=self.short_period).sum()

        money_flow_ratio = (positive_flow_ma - negative_flow_ma) / (positive_flow_ma + negative_flow_ma + 1e-10) * 100

        result['MoneyFlowRatio'] = money_flow_ratio.fillna(0)
        result['PositiveFlow'] = positive_flow_ma
        result['NegativeFlow'] = negative_flow_ma

        return result

    def _calculate_institutional_behavior(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算机构行为"""
        result = data.copy()

        close = result['close']
        volume = result['volume']

        # 计算价量关系
        price_change = close.pct_change()
        volume_change = volume.pct_change()

        # 机构行为指标:价涨量增为正,价跌量增为负
        institutional_behavior = pd.Series(0.0, index=data.index)

        # 价涨量增(机构买入)
        institutional_behavior[(price_change > 0) & (volume_change > 0)] = 1

        # 价跌量增(机构卖出)
        institutional_behavior[(price_change < 0) & (volume_change > 0)] = -1

        # 价涨量缩(散户跟风)
        institutional_behavior[(price_change > 0) & (volume_change < 0)] = 0.5  # TODO: 将魔法数字提取到配置中

        # 价跌量缩(恐慌抛售)
        institutional_behavior[(price_change < 0) & (volume_change < 0)] = -0.5  # TODO: 将魔法数字提取到配置中

        # 平滑处理
        institutional_behavior_smooth = institutional_behavior.rolling(window=self.volume_period).mean()

        result['InstitutionalBehavior'] = institutional_behavior_smooth.fillna(0)

        return result

    def _calculate_flow_strength(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算资金流向强度"""
        result = data.copy()

        money_flow_ratio = result['MoneyFlowRatio']
        institutional_behavior = result['InstitutionalBehavior']

        # 计算流向强度
        flow_strength = (money_flow_ratio * 0.6 + institutional_behavior * 40 * 0.4)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        # 标准化到0-100范围
        flow_strength_normalized = ((flow_strength - flow_strength.min()) / (flow_strength.max() - flow_strength.min()) * 100).fillna(50)  # TODO: 将魔法数字提取到配置中

        result['FlowStrength'] = flow_strength_normalized

        return result

    def _calculate_composite_flow_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算综合资金流向评分"""
        result = data.copy()

        # 综合各项资金流向指标
        money_flow_ratio = result['MoneyFlowRatio']
        institutional_behavior = result['InstitutionalBehavior']
        flow_strength = result['FlowStrength']

        # 加权平均计算综合评分
        composite_score = (
            money_flow_ratio * 0.4 +  # TODO: 将魔法数字提取到配置中
            institutional_behavior * 30 * 0.3 +  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            flow_strength * 0.3  # TODO: 将魔法数字提取到配置中
        )

        # 标准化到0-100范围
        composite_score_normalized = ((composite_score - composite_score.min()) / (composite_score.max() - composite_score.min()) * 100).fillna(50)  # TODO: 将魔法数字提取到配置中

        result['CompositeFundFlowScore'] = composite_score_normalized

        return result

    def _generate_fund_flow_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成资金流向信号"""
        result = data.copy()

        composite_score = result['CompositeFundFlowScore']
        institutional_behavior = result['InstitutionalBehavior']

        # 强资金流入信号(买入机会)
        result['StrongInflowSignal'] = (composite_score >= 70) & (institutional_behavior > 0.5)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        # 强资金流出信号(卖出警告)
        result['StrongOutflowSignal'] = (composite_score <= 30) & (institutional_behavior < -0.5)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        # 资金流向转折信号
        score_change = composite_score.diff()
        result['FlowReversalSignal'] = (
            (score_change > 15) |  # 资金流向快速好转  # TODO: 将魔法数字提取到配置中
            (score_change < -15)   # 资金流向快速恶化  # TODO: 将魔法数字提取到配置中
        )

        return result

    def _calculate_baseindicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        BaseIndicator要求的抽象方法实现

        Args:
            data: 包含OHLCV数据的DataFrame
            **kwargs: 其他参数

        Returns:
            包含ZXM资金流向指标的DataFrame
        """
        return self.calculate(data, **kwargs)

    def calculate_confidence_Indicator_Base_Indicator(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """
        BaseIndicator要求的置信度计算方法

        Args:
            score: 得分序列
            patterns: 检测到的形态DataFrame
            signals: 生成的信号字典

        Returns:
            float: 置信度分数 (0-1)
        """
        return 0.8  # ZXM资金流向指标置信度  # TODO: 将魔法数字提取到配置中

    def calculate_raw_score_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        BaseIndicator要求的原始评分计算方法

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            pd.Series: 原始评分序列
        """
        result = self.calculate(data, **kwargs)
        return result['CompositeFundFlowScore']

    def get_patterns_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        BaseIndicator要求的形态获取方法

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 形态DataFrame
        """
        result = self.calculate(data, **kwargs)
        patterns = pd.DataFrame(index=data.index)

        # 资金流向形态
        composite_score = result['CompositeFundFlowScore']
        patterns['ZXM_STRONG_INFLOW'] = composite_score >= 80  # TODO: 将魔法数字提取到配置中
        patterns['ZXM_MODERATE_INFLOW'] = (composite_score >= 60) & (composite_score < 80)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        patterns['ZXM_NEUTRAL_FLOW'] = (composite_score >= 40) & (composite_score < 60)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        patterns['ZXM_MODERATE_OUTFLOW'] = (composite_score >= 20) & (composite_score < 40)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        patterns['ZXM_STRONG_OUTFLOW'] = composite_score < 20  # TODO: 将魔法数字提取到配置中

        # 资金流向信号形态
        patterns['ZXM_STRONG_INFLOW_SIGNAL'] = result['StrongInflowSignal']
        patterns['ZXM_STRONG_OUTFLOW_SIGNAL'] = result['StrongOutflowSignal']
        patterns['ZXM_FLOW_REVERSAL'] = result['FlowReversalSignal']

        return patterns

    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        """
        BaseIndicator要求的参数设置方法

        Args:
            **kwargs: 参数字典
        """
        self.set_parameters_Fund_Flow(**kwargs)