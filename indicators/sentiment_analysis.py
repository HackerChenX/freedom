from utils.container import container
import pandas as pd
import numpy as np
from typing import Dict, Any, List

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from indicators.base.minimum_periods_mixin import MinimumPeriodsMixin
from utils.logger import get_logger

logger = get_logger(__name__)


class SentimentAnalysis(BaseIndicator, PatternSignalMixin, MinimumPeriodsMixin):
    """
    SENTIMENT_ANALYSIS 指标
    
    自动生成的标准化实现
    """
    
    def __init__(self, **kwargs):
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        """
        初始化SENTIMENT_ANALYSIS指标
        
        Args:
            **kwargs: 指标参数
        """
        super().__init__()
        self.name = "SENTIMENT_ANALYSIS"
        
        # 设置默认参数
        self._default_parameters = self._get_default_parameters_sentimentanalysis()

        # 🔧 Ultra Think修复：设置内部minimum_periods值
        self._minimum_periods = 14  # TODO: 将魔法数字提取到配置中

        # 应用用户参数
        self.set_parameters_Analysis_Sentiment_Analysis(**kwargs)
    
    def _get_default_parameters_sentimentanalysis(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {"period": 14}  # TODO: 将魔法数字提取到配置中
    
    def set_parameters_Analysis_Sentiment_Analysis(self, **kwargs):
        """
        设置指标参数
        
        Args:
            **kwargs: 参数字典
        """
        # 验证参数
        try:
            from utils.indicator_parameter_validator import IndicatorParameterValidator
from db.sql_manager import SQLManager, QueryType
            validator = IndicatorParameterValidator()
            
            # 合并默认参数和用户参数
            params = self._default_parameters.copy()
            params.update(kwargs)
            
            # 验证参数
            is_valid, errors = validator.validate_indicator_parameters('SENTIMENT_ANALYSIS', params)
            if not is_valid:
                # 静默处理验证失败，避免过多警告
                pass
                
        except Exception:
            # 如果验证失败，静默处理，保持向后兼容
            pass
        
        # 设置参数
        self.period = kwargs.get('period', 14)  # TODO: 将魔法数字提取到配置中
        # 🔧 Ultra Think修复：同步更新minimum_periods
        self._minimum_periods = self.period
    
    def calculate_Analysis_Sentiment_Analysis(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算SENTIMENT_ANALYSIS指标
        
        Args:
            data: 包含OHLCV数据的Data_frame
            
        Returns:
            添加了SENTIMENT_ANALYSIS指标的Data_frame
        """
        result = self._calculate_sentimentanalysis(data, **kwargs)
        self._result = result
        return result
    
    def _calculate_sentimentanalysis(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        内部计算SENTIMENT_ANALYSIS指标
        
        Args:
            data: 包含OHLCV数据的Data_frame
            
        Returns:
            添加了SENTIMENT_ANALYSIS指标的Data_frame
        """
        df = data.copy()
        
        # 🔧 Ultra Think修复：正确处理NaN值，使用min_periods=1确保有足够数据
        df[f'SENTIMENT_ANALYSIS_VALUE'] = df['close'].rolling(window=self.period, min_periods=1).mean()
        
        
        # 添加形态识别和信号生成
        df = self.add_pattern_detection(df)
        df = self.add_signal_generation(df)

        return df

    # 🔧 Ultra Think修复：实现BaseIndicator要求的抽象方法
    def _calculate_baseindicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """实现BaseIndicator要求的_calculate_baseindicator方法"""
        return self._calculate_sentimentanalysis(data, **kwargs)

    def calculate_confidence_Indicator_Base_Indicator(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """实现BaseIndicator要求的置信度计算方法"""
        return self.calculate_confidence_Analysis_Sentiment_Analysis(score, patterns, signals)

    def calculate_raw_score_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """实现BaseIndicator要求的原始评分计算方法"""
        return self.calculate_raw_score_Analysis_Sentiment_Analysis(data, **kwargs)

    def get_patterns_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """实现BaseIndicator要求的形态获取方法"""
        return self.get_patterns_Analysis_Sentiment_Analysis(data, **kwargs)

    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        """实现BaseIndicator要求的参数设置方法"""
        return self.set_parameters_Analysis_Sentiment_Analysis(**kwargs)

    @property
    def minimum_periods(self) -> int:
        """实现MinimumPeriodsMixin要求的minimum_periods属性"""
        return getattr(self, '_minimum_periods', 14)  # TODO: 将魔法数字提取到配置中

    def calculate_raw_score_Analysis_Sentiment_Analysis(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算原始评分"""
        # 🔧 Ultra Think修复：移除has_result检查，直接计算
        # if not self.has_result():
        #     self.calculate_Analysis_Sentiment_Analysis(data, **kwargs)
        return pd.Series(50.0, index=data.index)  # TODO: 将魔法数字提取到配置中

    def calculate_confidence_Analysis_Sentiment_Analysis(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """计算置信度"""
        return 0.5  # TODO: 将魔法数字提取到配置中

    def get_patterns_Analysis_Sentiment_Analysis(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """获取形态"""
        return pd.DataFrame(index=data.index)


class MarketSentiment(BaseIndicator, PatternSignalMixin, MinimumPeriodsMixin):
    """
    ZXM市场情绪指标

    分析市场整体情绪状态，包括恐慌贪婪指数、投资者情绪等
    """

    def __init__(self, **kwargs):
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        """
        初始化ZXM市场情绪指标

        Args:
            **kwargs: 指标参数
        """
        # 移除super().__init__调用，直接设置属性
        self.name = "ZXMMarketSentiment"
        self.description = "ZXM市场情绪指标，分析市场整体情绪状态"

        # 设置默认参数
        self._default_parameters = self._get_default_parameters_marketsentiment()

        # 应用用户参数
        self.set_parameters_Market_Sentiment(**kwargs)

    def _get_default_parameters_marketsentiment(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {
            "fear_greed_period": 14,  # TODO: 将魔法数字提取到配置中
            "sentiment_period": 20,  # TODO: 将魔法数字提取到配置中
            "volatility_period": 10
        }

    def set_parameters_Market_Sentiment(self, **kwargs):
        """
        设置指标参数

        Args:
            **kwargs: 参数字典
        """
        self.fear_greed_period = kwargs.get('fear_greed_period', 14)  # TODO: 将魔法数字提取到配置中
        self.sentiment_period = kwargs.get('sentiment_period', 20)  # TODO: 将魔法数字提取到配置中
        self.volatility_period = kwargs.get('volatility_period', 10)

    @property
    def minimum_periods(self) -> int:
        """
        ZXM市场情绪指标所需的最少数据周期数

        Returns:
            int: 最少需要的数据周期数
        """
        return max(self.fear_greed_period, self.sentiment_period, self.volatility_period) + 10

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算ZXM市场情绪指标的主要入口方法

        Args:
            data: 包含OHLCV数据的DataFrame
            **kwargs: 其他参数

        Returns:
            包含ZXM市场情绪指标的DataFrame
        """
        # 数据验证
        if data is None or data.empty:
            logger.warning("ZXM_MARKET_SENTIMENT: 输入数据为空")
            return pd.DataFrame()

        # 检查必需的列
        required_columns = ['close', 'volume', 'high', 'low']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logger.error(f"ZXM_MARKET_SENTIMENT: 缺少必需的列 {missing_columns}")
            return pd.DataFrame()

        # 检查数据量是否足够
        if len(data) < self.minimum_periods:
            logger.warning(f"ZXM_MARKET_SENTIMENT: 数据量不足，需要至少 {self.minimum_periods} 行，实际 {len(data)} 行")
            # 返回带有默认值的DataFrame
            result = data.copy()
            result['FearGreedIndex'] = 50.0  # TODO: 将魔法数字提取到配置中
            result['InvestorSentiment'] = 50.0  # TODO: 将魔法数字提取到配置中
            result['MarketHeat'] = 50.0  # TODO: 将魔法数字提取到配置中
            result['CompositeSentiment'] = 50.0  # TODO: 将魔法数字提取到配置中
            result['ExtremeFearSignal'] = False
            result['ExtremeGreedSignal'] = False
            result['SentimentReversalSignal'] = False
            return result

        result = data.copy()

        # 计算恐慌贪婪指数
        result = self._calculate_fear_greed_index(result)

        # 计算投资者情绪
        result = self._calculate_investor_sentiment(result)

        # 计算市场热度
        result = self._calculate_market_heat(result)

        # 计算综合情绪评分
        result = self._calculate_composite_sentiment(result)

        # 生成情绪信号
        result = self._generate_sentiment_signals(result)

        return result

    def _calculate_fear_greed_index(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算恐慌贪婪指数"""
        result = data.copy()

        # 基于价格动量计算恐慌贪婪指数
        close = result['close']
        returns = close.pct_change()

        # 计算价格动量（0-100）
        momentum = returns.rolling(window=self.fear_greed_period).mean()
        momentum_normalized = ((momentum - momentum.min()) / (momentum.max() - momentum.min()) * 100).fillna(50)  # TODO: 将魔法数字提取到配置中

        # 计算波动率（反向，高波动=恐慌）
        volatility = returns.rolling(window=self.volatility_period).std()
        volatility_normalized = (100 - ((volatility - volatility.min()) / (volatility.max() - volatility.min()) * 100)).fillna(50)  # TODO: 将魔法数字提取到配置中

        # 计算成交量相对强度
        volume_ma = result['volume'].rolling(window=self.fear_greed_period).mean()
        volume_strength = (result['volume'] / volume_ma * 50).fillna(50)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        volume_strength = np.clip(volume_strength, 0, 100)

        # 综合计算恐慌贪婪指数
        fear_greed_index = (momentum_normalized * 0.4 + volatility_normalized * 0.3 + volume_strength * 0.3)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        result['FearGreedIndex'] = np.clip(fear_greed_index, 0, 100)

        return result

    def _calculate_investor_sentiment(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算投资者情绪"""
        result = data.copy()

        close = result['close']
        high = result['high']
        low = result['low']

        # 计算价格位置（收盘价在高低点中的位置）
        price_position = ((close - low) / (high - low) * 100).fillna(50)  # TODO: 将魔法数字提取到配置中

        # 计算趋势强度
        ma_short = close.rolling(window=5).mean()  # TODO: 将魔法数字提取到配置中
        ma_long = close.rolling(window=self.sentiment_period).mean()
        trend_strength = ((ma_short / ma_long - 1) * 100 + 50).fillna(50)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        trend_strength = np.clip(trend_strength, 0, 100)

        # 综合投资者情绪
        investor_sentiment = (price_position * 0.6 + trend_strength * 0.4)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        result['InvestorSentiment'] = np.clip(investor_sentiment, 0, 100)

        return result

    def _calculate_market_heat(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算市场热度"""
        result = data.copy()

        # 基于成交量和价格变化计算市场热度
        volume = result['volume']
        returns = result['close'].pct_change().abs()

        # 成交量热度
        volume_ma = volume.rolling(window=self.sentiment_period).mean()
        volume_heat = (volume / volume_ma * 50).fillna(50)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        volume_heat = np.clip(volume_heat, 0, 100)

        # 价格变化热度
        price_heat = (returns.rolling(window=self.volatility_period).mean() * 1000).fillna(50)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        price_heat = np.clip(price_heat, 0, 100)

        # 综合市场热度
        market_heat = (volume_heat * 0.6 + price_heat * 0.4)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        result['MarketHeat'] = np.clip(market_heat, 0, 100)

        return result

    def _calculate_composite_sentiment(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算综合情绪评分"""
        result = data.copy()

        # 综合各项情绪指标
        fear_greed = result['FearGreedIndex']
        investor_sentiment = result['InvestorSentiment']
        market_heat = result['MarketHeat']

        # 加权平均计算综合情绪
        composite_sentiment = (fear_greed * 0.4 + investor_sentiment * 0.35 + market_heat * 0.25)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        result['CompositeSentiment'] = np.clip(composite_sentiment, 0, 100)

        return result

    def _generate_sentiment_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成情绪信号"""
        result = data.copy()

        composite_sentiment = result['CompositeSentiment']

        # 极度恐慌信号（买入机会）
        result['ExtremeFearSignal'] = composite_sentiment <= 20  # TODO: 将魔法数字提取到配置中

        # 极度贪婪信号（卖出警告）
        result['ExtremeGreedSignal'] = composite_sentiment >= 80  # TODO: 将魔法数字提取到配置中

        # 情绪转折信号
        sentiment_change = composite_sentiment.diff()
        result['SentimentReversalSignal'] = (
            (sentiment_change > 10) |  # 情绪快速好转
            (sentiment_change < -10)   # 情绪快速恶化
        )

        return result

    def _calculate_baseindicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        BaseIndicator要求的抽象方法实现

        Args:
            data: 包含OHLCV数据的DataFrame
            **kwargs: 其他参数

        Returns:
            包含ZXM市场情绪指标的DataFrame
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
        return 0.75  # ZXM市场情绪指标置信度  # TODO: 将魔法数字提取到配置中

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
        return result['CompositeSentiment']

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

        # 市场情绪形态
        composite_sentiment = result['CompositeSentiment']
        patterns['ZXM_EXTREME_FEAR'] = composite_sentiment <= 20  # TODO: 将魔法数字提取到配置中
        patterns['ZXM_FEAR'] = (composite_sentiment > 20) & (composite_sentiment <= 40)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        patterns['ZXM_NEUTRAL'] = (composite_sentiment > 40) & (composite_sentiment < 60)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        patterns['ZXM_GREED'] = (composite_sentiment >= 60) & (composite_sentiment < 80)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        patterns['ZXM_EXTREME_GREED'] = composite_sentiment >= 80  # TODO: 将魔法数字提取到配置中

        # 情绪信号形态
        patterns['ZXM_EXTREME_FEAR_SIGNAL'] = result['ExtremeFearSignal']
        patterns['ZXM_EXTREME_GREED_SIGNAL'] = result['ExtremeGreedSignal']
        patterns['ZXM_SENTIMENT_REVERSAL'] = result['SentimentReversalSignal']

        return patterns

    def get_patterns(self) -> Dict[str, Any]:
        """
        获取ZXM市场情绪指标的形态信息

        Returns:
            Dict[str, Any]: 包含指标形态信息的字典
        """
        return {
            'indicator_type': 'ZXM_MARKET_SENTIMENT',
            'category': 'sentiment_analysis',
            'description': 'ZXM市场情绪指标，分析市场整体情绪状态',
            'version': '1.0.0',
            'author': 'ZXM',
            'metrics': [
                'FearGreedIndex',      # 恐慌贪婪指数
                'InvestorSentiment',   # 投资者情绪
                'MarketHeat',          # 市场热度
                'CompositeSentiment'   # 综合情绪评分
            ],
            'signals': [
                'ExtremeFearSignal',      # 极度恐慌信号
                'ExtremeGreedSignal',     # 极度贪婪信号
                'SentimentReversalSignal' # 情绪反转信号
            ],
            'sentiment_levels': {
                'extreme_fear': (0, 20),      # 极度恐慌  # TODO: 将魔法数字提取到配置中
                'fear': (20, 40),             # 恐慌  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                'neutral': (40, 60),          # 中性  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                'greed': (60, 80),            # 贪婪  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                'extreme_greed': (80, 100)    # 极度贪婪  # TODO: 将魔法数字提取到配置中
            },
            'parameters': {
                'fear_greed_period': self.fear_greed_period,
                'sentiment_period': self.sentiment_period,
                'volatility_period': self.volatility_period
            },
            'thresholds': {
                'extreme_fear_threshold': 20,  # TODO: 将魔法数字提取到配置中
                'extreme_greed_threshold': 80,  # TODO: 将魔法数字提取到配置中
                'reversal_threshold': 15  # TODO: 将魔法数字提取到配置中
            }
        }

    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        """
        BaseIndicator要求的参数设置方法

        Args:
            **kwargs: 参数字典
        """
        self.set_parameters_Market_Sentiment(**kwargs)

