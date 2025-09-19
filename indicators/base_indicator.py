from utils.decorators import performance_monitor, exception_handler
from utils.container import container
from utils.logger import get_logger

"""
技术指标基类模块 - L4核心服务层

基于L1L2L3成功修复经验的严格依赖注入实现
提供所有技术指标的统一基础架构，确保生产级质量标准
"""

import abc
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from datetime import datetime


logger = get_logger(__name__)


class StandardColumnNames:
    """
    项目级统一列名标准

    用途：
    1. 确保所有指标使用统一的列名标准
    2. 便于上游系统调用和数据处理
    3. 提高代码可维护性和可读性

    命名规则：
    - 基础数据列：小写英文
    - 指标列：指标名_具体值，如 macd_dif, rsi_value
    - 信号列：统一使用 buy_signal, sell_signal 等
    """

    # 基础价格数据列（输入数据标准）
    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"
    VOLUME = "volume"
    TURNOVER_RATE = "turnover_rate"

    # MACD指标标准列名
    MACD_DIF = "macd_dif"           # DIF线（快线-慢线）
    MACD_DEA = "macd_dea"           # DEA线（信号线）
    MACD_HISTOGRAM = "macd_histogram"  # MACD柱状图

    # RSI指标标准列名
    RSI_VALUE = "rsi_value"         # RSI主值
    RSI_OVERBOUGHT = "rsi_overbought"  # 超买信号
    RSI_OVERSOLD = "rsi_oversold"   # 超卖信号

    # KDJ指标标准列名
    KDJ_K = "kdj_k"                # K值
    KDJ_D = "kdj_d"                # D值
    KDJ_J = "kdj_j"                # J值

    # 移动平均线标准列名
    MA_5 = "ma_5"                  # 5日移动平均
    MA_10 = "ma_10"                # 10日移动平均
    MA_20 = "ma_20"                # 20日移动平均
    MA_60 = "ma_60"                # 60日移动平均

    # 通用信号列名（所有指标统一使用）
    BUY_SIGNAL = "buy_signal"       # 买入信号
    SELL_SIGNAL = "sell_signal"     # 卖出信号
    HOLD_SIGNAL = "hold_signal"     # 持有信号
    SIGNAL_STRENGTH = "signal_strength"    # 信号强度
    SIGNAL_CONFIDENCE = "signal_confidence"  # 信号置信度

    @classmethod
    def get_indicator_columns(cls, indicator_name: str) -> List[str]:
        """
        获取指定指标的标准列名

        Args:
            indicator_name: 指标名称，如 'macd', 'rsi', 'kdj'

        Returns:
            List[str]: 该指标的标准列名列表
        """
        indicator_columns = {
            'macd': [cls.MACD_DIF, cls.MACD_DEA, cls.MACD_HISTOGRAM],
            'rsi': [cls.RSI_VALUE, cls.RSI_OVERBOUGHT, cls.RSI_OVERSOLD],
            'kdj': [cls.KDJ_K, cls.KDJ_D, cls.KDJ_J],
            'ma': [cls.MA_5, cls.MA_10, cls.MA_20, cls.MA_60]
        }
        return indicator_columns.get(indicator_name.lower(), [])

    @classmethod
    def get_signal_columns(cls) -> List[str]:
        """
        获取标准信号列名

        Returns:
            List[str]: 标准信号列名列表
        """
        return [cls.BUY_SIGNAL, cls.SELL_SIGNAL, cls.HOLD_SIGNAL,
                cls.SIGNAL_STRENGTH, cls.SIGNAL_CONFIDENCE]


class DependencyInjectionError(Exception):
    """依赖注入错误 - BaseIndicator专用异常"""
    pass


class PatternInfo:
    """形态信息类"""

    def __init__(self, name: str, signal_type: str, strength: float = 0.0, duration: int = 1, details: str = ""):
        self.name = name
        self.signal_type = signal_type
        self.strength = strength
        self.duration = duration
        self.details = details
        self.display_name = name

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "name": self.name,
            "signal_type": self.signal_type,
            "display_name": self.display_name,
            "strength": self.strength,
            "duration": self.duration,
            "details": self.details,
        }


class BaseIndicator(abc.ABC):
    """
    BaseIndicator - L4核心服务层技术指标基类

    职责合理性说明:
    - 作为L4层核心服务组件,承担多项相关职责
    - 方法分为以下职责组:
      * 核心抽象方法 (calculate, get_signal, get_patterns)
      * 数据处理方法 (validate_data, preprocess_data, postprocess_result)
      * 扩展点方法 (initialize_indicator, register_patterns)
      * 工具方法 (format_output, get_metadata)
    - 符合L4层组件化架构设计原则
    - 基于L3层成功经验的职责分组模式

    技术指标基类

    所有技术指标类应继承此类,并实现必要的抽象方法
    """

    def __init__(self, name: str = "", period: int = 20, **kwargs):  # TODO: 将魔法数字提取到配置中
        """
        初始化指标

        Args:
            name: 指标名称
            period: 计算周期
            **kwargs: 其他参数
        """
        self.name = name or self.__class__.__name__
        self.period = period
        self.params = kwargs
        self._result = None
        self._patterns = []

        # 严格依赖注入 - 基于L1L2L3成功经验，不允许兜底逻辑
        self.data_access = container.resolve("DataAccessInterface")
        self.cache_service = container.resolve("ICacheService")

        # 验证依赖注入成功 - 确保生产级质量
        if not self.data_access:
            raise DependencyInjectionError("DataAccessInterface服务未注册，请检查依赖注入配置")
        if not self.cache_service:
            raise DependencyInjectionError("ICacheService服务未注册，请检查依赖注入配置")

        # 初始化指标
        self.initialize_indicator()

    def initialize_indicator(self):
        """
        在所有子类参数都设置完毕后执行初始化
        """
        # 自动注册形态
        self.register_patterns()

    def register_patterns(self):
        """
        注册指标形态

        子类可以重写此方法来注册自定义形态
        """
        pass

    def register_pattern_to_registry(self, 
                                   pattern_id: str, 
                                   display_name: str, 
                                   description: str = "",
                                   pattern_type: str = "NEUTRAL",
                                   default_strength: str = "MEDIUM", 
                                   score_impact: float = 0.0,
                                   polarity: str = "NEUTRAL",
                                   detection_function=None,
                                   allow_override: bool = True) -> None:
        """
        注册形态到全局注册表

        Args:
            pattern_id: 形态ID
            display_name: 显示名称
            description: 形态描述
            pattern_type: 形态类型(BULLISH/BEARISH/NEUTRAL/REVERSAL)
            default_strength: 默认强度(STRONG/MEDIUM/WEAK)
            score_impact: 评分影响
            polarity: 极性(POSITIVE/NEGATIVE/NEUTRAL)
            detection_function: 检测函数(可选)
            allow_override: 是否允许覆盖
        """
        try:
            # 导入形态注册表
            from indicators.pattern_registry import get_pattern_registry, PatternTypePatternRegistry, PatternStrengthPatternRegistry, PatternPolarity
            
            # 获取注册表实例
            registry = get_pattern_registry()
            
            # 转换形态类型
            pattern_type_enum = PatternTypePatternRegistry.NEUTRAL
            if pattern_type.upper() == "BULLISH":
                pattern_type_enum = PatternTypePatternRegistry.BULLISH
            elif pattern_type.upper() == "BEARISH":
                pattern_type_enum = PatternTypePatternRegistry.BEARISH
            elif pattern_type.upper() == "REVERSAL":
                pattern_type_enum = PatternTypePatternRegistry.REVERSAL
            
            # 转换强度
            strength_enum = PatternStrengthPatternRegistry.MEDIUM
            if default_strength.upper() == "STRONG":
                strength_enum = PatternStrengthPatternRegistry.STRONG
            elif default_strength.upper() == "WEAK":
                strength_enum = PatternStrengthPatternRegistry.WEAK
                
            # 转换极性
            polarity_enum = PatternPolarity.NEUTRAL
            if polarity.upper() == "POSITIVE":
                polarity_enum = PatternPolarity.POSITIVE
            elif polarity.upper() == "NEGATIVE":
                polarity_enum = PatternPolarity.NEGATIVE
            
            # 注册形态
            registry.register_pattern_registry(
                pattern_id=pattern_id,
                display_name=display_name,
                indicator_id=self.name,
                pattern_type=pattern_type_enum,
                default_strength=strength_enum,
                description=description,
                score_impact=score_impact,
                polarity=polarity_enum,
                detection_function=detection_function,
                allow_override=allow_override
            )
            
            logger.debug(f"形态 {pattern_id} 注册成功: {display_name}")
            
        except Exception as e:
            logger.warning(f"注册形态 {pattern_id} 失败: {e}")
            # 不抛出异常，确保指标能正常实例化

    @abc.abstractmethod
    @performance_monitor(threshold=2.0)
    @exception_handler(reraise=True)
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        【核心抽象方法1】计算技术指标的数值结果

        🎯 使用场景：
        - L5买点分析：获取MACD、RSI、KDJ等指标的具体数值用于买点判断
        - L5策略选股：批量计算多个股票的技术指标数值进行筛选
        - L5回测分析：计算历史时间序列的指标数值用于策略回测
        - L5实时监控：计算最新的指标数值用于实时监控和预警

        📊 方法职责：
        1. 接收标准化的股票OHLCV数据
        2. 执行指标的核心数学计算逻辑（如移动平均、RSI计算等）
        3. 返回包含指标数值的标准化DataFrame
        4. 确保输出列名遵循项目统一命名标准

        📋 输入数据要求：
        - 必须包含的列：'close'（收盘价）
        - 推荐包含的列：'open', 'high', 'low', 'close', 'volume'
        - 数据格式：pandas.DataFrame，索引为日期或时间
        - 数据量要求：至少包含指标计算周期所需的最小数据点数

        📈 输出格式标准：
        - 返回类型：pandas.DataFrame
        - 索引：与输入数据保持一致（通常是日期时间索引）
        - 列名规范：使用指标名前缀，如 'macd_dif', 'rsi_value', 'kdj_k'
        - 数据类型：float64（数值型指标）

        💡 实现示例：
        ```python
        # MACD指标实现示例
        def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
            close = data['close']
            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()

            result = pd.DataFrame(index=data.index)
            result['macd_dif'] = ema12 - ema26
            result['macd_dea'] = result['macd_dif'].ewm(span=9).mean()
            result['macd_histogram'] = result['macd_dif'] - result['macd_dea']

            return result
        ```

        Args:
            data (pd.DataFrame): 标准化股票数据
                - 必需列：'close'
                - 可选列：'open', 'high', 'low', 'volume', 'turnover_rate'
                - 索引：日期时间索引

        Returns:
            pd.DataFrame: 指标计算结果数据框
                - 索引：与输入数据一致
                - 列：指标特定的数值列（如macd_dif, rsi_value等）
                - 数据类型：float64

        Raises:
            ValueError: 输入数据缺少必需列或格式不正确
            InsufficientDataError: 数据量不足以计算指标（少于最小周期要求）
            CalculationError: 指标计算过程中出现数学错误
        """
        pass

    @abc.abstractmethod
    def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        【核心抽象方法2】基于指标数值生成最新的交易信号

        🎯 使用场景：
        - L5买点分析：判断当前是否出现买点信号（如MACD金叉、RSI超卖反弹）
        - L5策略选股：为每只股票生成买入/卖出/持有的投资建议
        - L5实时交易：为交易系统提供实时的交易信号和强度评估
        - L5风险控制：生成止损、止盈等风险控制信号
        - L5组合管理：为投资组合调整提供信号依据

        📊 方法职责：
        1. 分析指标数值的最新状态和变化趋势
        2. 应用指标特定的信号生成规则（如金叉死叉、超买超卖等）
        3. 计算信号的强度和置信度
        4. 返回标准化的交易信号字典

        🔄 与其他方法的关系：
        - calculate() → get_signal()：先计算数值，再生成信号
        - get_signal() vs get_signals()：
          * get_signal()：返回最新单一信号，用于实时决策
          * get_signals()：返回历史信号序列，用于回测分析

        📋 输入数据说明：
        - 通常是 calculate() 的返回结果
        - 也可以是包含指标列的原始股票数据
        - 必须包含足够的历史数据以判断趋势和形态

        📈 输出信号标准：
        ```python
        {
            'signal_type': 'buy',           # 信号类型：'buy', 'sell', 'hold'
            'strength': 0.85,               # 信号强度：0.0-1.0（0=最弱，1=最强）
            'confidence': 0.92,             # 信号置信度：0.0-1.0（0=不确定，1=非常确定）
            'timestamp': datetime.now(),    # 信号生成时间
            'price': 12.34,                # 触发价格（当前价格或建议价格）
            'reason': 'MACD金叉确认',        # 信号生成原因
            'metadata': {                   # 额外的信号信息
                'indicator_values': {...},   # 关键指标数值
                'pattern_detected': '...',   # 检测到的技术形态
                'risk_level': 'medium'       # 风险等级
            }
        }
        ```

        💡 实现示例：
        ```python
        # RSI指标信号生成示例
        def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
            latest_rsi = data['rsi_value'].iloc[-1]
            prev_rsi = data['rsi_value'].iloc[-2]

            if latest_rsi < 30 and prev_rsi >= 30:  # 进入超卖区域
                return {
                    'signal_type': 'buy',
                    'strength': min((30 - latest_rsi) / 10, 1.0),
                    'confidence': 0.8,
                    'reason': 'RSI进入超卖区域',
                    'metadata': {'rsi_value': latest_rsi}
                }
            elif latest_rsi > 70 and prev_rsi <= 70:  # 进入超买区域
                return {
                    'signal_type': 'sell',
                    'strength': min((latest_rsi - 70) / 10, 1.0),
                    'confidence': 0.8,
                    'reason': 'RSI进入超买区域',
                    'metadata': {'rsi_value': latest_rsi}
                }
            else:
                return {
                    'signal_type': 'hold',
                    'strength': 0.0,
                    'confidence': 0.5,
                    'reason': 'RSI处于正常区间',
                    'metadata': {'rsi_value': latest_rsi}
                }
        ```

        Args:
            data (pd.DataFrame): 包含指标计算结果的数据
                - 可以是 calculate() 的返回结果
                - 也可以是包含指标列的原始股票数据
                - 必须包含足够的历史数据点以进行趋势判断

        Returns:
            Dict[str, Any]: 标准化交易信号字典，必须包含以下字段：
                - signal_type (str): 'buy', 'sell', 'hold'
                - strength (float): 信号强度 0.0-1.0
                - confidence (float): 信号置信度 0.0-1.0
                - timestamp (datetime): 信号时间（可选，默认当前时间）
                - price (float): 触发价格（可选）
                - reason (str): 信号原因说明（可选）
                - metadata (dict): 额外信息（可选）

        Raises:
            ValueError: 输入数据格式不正确或缺少必需的指标列
            InsufficientDataError: 数据量不足以生成可靠信号
            SignalGenerationError: 信号生成逻辑执行失败
        """
        pass

    # ==================== 扩展功能方法 ====================

    # ==================== 扩展功能方法 ====================
    def get_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        【扩展方法】生成历史交易信号序列

        🎯 使用场景：
        - L5策略回测：生成完整的历史信号序列用于回测分析
        - L5策略验证：验证策略在历史数据上的表现
        - L5信号统计：统计信号的频率、准确率等指标
        - L5可视化分析：在图表上标注历史买卖点

        📊 方法职责：
        1. 基于指标数值生成完整的历史信号序列
        2. 为每个时间点计算信号强度和置信度
        3. 返回标准化的信号DataFrame用于回测分析

        🔄 与其他方法的关系：
        - calculate() → get_signals()：先计算数值，再生成信号序列
        - get_signal() vs get_signals()：
          * get_signal()：返回最新单一信号，用于实时决策
          * get_signals()：返回历史信号序列，用于回测分析

        📈 输出格式标准：
        ```python
        DataFrame包含以下标准列：
        - buy_signal (bool): 买入信号标记
        - sell_signal (bool): 卖出信号标记
        - hold_signal (bool): 持有信号标记
        - signal_strength (float): 信号强度 0.0-1.0
        - signal_confidence (float): 信号置信度 0.0-1.0
        - signal_reason (str): 信号原因（可选）
        ```

        💡 默认实现说明：
        - 子类可以重写此方法实现特定的历史信号生成逻辑
        - 默认实现基于 get_signal() 生成简单信号序列
        - 建议子类实现更精细的历史信号分析逻辑

        Args:
            data (pd.DataFrame): 包含指标计算结果的数据
                - 通常是 calculate() 的返回结果
                - 必须包含完整的历史时间序列数据

        Returns:
            pd.DataFrame: 历史信号序列DataFrame
                - 索引：与输入数据一致（时间序列）
                - 列：标准信号列（buy_signal, sell_signal等）
                - 每行代表一个时间点的信号状态
        """
        # 默认实现：基于 get_signal() 生成简单信号序列
        if data.empty:
            return pd.DataFrame()

        # 创建标准信号DataFrame
        signals_df = pd.DataFrame(index=data.index)
        signals_df['buy_signal'] = False
        signals_df['sell_signal'] = False
        signals_df['hold_signal'] = True
        signals_df['signal_strength'] = 0.0
        signals_df['signal_confidence'] = 0.0

        # 获取最新信号并应用到最后一行（默认实现）
        try:
            latest_signal = self.get_signal(data)
            if latest_signal and isinstance(latest_signal, dict):
                last_idx = signals_df.index[-1]
                signal_type = latest_signal.get('signal_type', 'hold')

                if signal_type == 'buy':
                    signals_df.loc[last_idx, 'buy_signal'] = True
                    signals_df.loc[last_idx, 'hold_signal'] = False
                elif signal_type == 'sell':
                    signals_df.loc[last_idx, 'sell_signal'] = True
                    signals_df.loc[last_idx, 'hold_signal'] = False

                signals_df.loc[last_idx, 'signal_strength'] = latest_signal.get('strength', 0.0)
                signals_df.loc[last_idx, 'signal_confidence'] = latest_signal.get('confidence', 0.0)
        except Exception as e:
            logger.warning(f"生成历史信号序列时出错: {e}")

        return signals_df

    def get_patterns(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        【扩展方法】检测技术形态和图表模式

        🎯 使用场景：
        - L5买点分析：识别经典的买点形态（如双底、头肩底、上升三角形等）
        - L5策略选股：筛选出现特定技术形态的股票
        - L5形态分析：为技术分析师提供形态识别和强度评估
        - L5风险评估：识别可能的反转形态或持续形态
        - L5教育培训：为学习者展示各种技术形态的实例

        📊 方法职责：
        1. 基于指标数值识别经典的技术分析形态
        2. 计算每个形态的强度分数和置信度
        3. 提供形态的详细描述和预期影响
        4. 返回标准化的形态信息列表

        🔍 形态类型示例：
        - 趋势形态：上升趋势、下降趋势、横盘整理
        - 反转形态：双顶、双底、头肩顶、头肩底
        - 持续形态：三角形、楔形、旗形、矩形
        - 指标形态：金叉、死叉、背离、收敛

        📈 输出格式标准：
        ```python
        [
            {
                'pattern_name': 'MACD金叉',           # 形态名称
                'pattern_type': 'bullish_reversal',   # 形态类型
                'strength': 0.85,                     # 形态强度 0.0-1.0
                'confidence': 0.92,                   # 识别置信度 0.0-1.0
                'signal_type': 'buy',                 # 信号方向
                'duration': 3,                        # 形态持续天数
                'start_date': '2024-01-15',          # 形态开始日期
                'end_date': '2024-01-18',            # 形态结束日期
                'description': 'MACD线上穿信号线，确认上涨趋势',
                'price_target': 12.50,               # 价格目标（可选）
                'risk_level': 'medium',              # 风险等级
                'metadata': {                        # 额外信息
                    'macd_dif': 0.15,
                    'macd_dea': 0.08,
                    'crossover_strength': 0.85
                }
            }
        ]
        ```

        💡 实现示例：
        ```python
        # MACD指标形态检测示例
        def detect_technical_patterns(self, indicator_data: pd.DataFrame) -> List[Dict[str, Any]]:
            patterns = []

            if 'macd_dif' in indicator_data.columns and 'macd_dea' in indicator_data.columns:
                # 检测金叉形态
                dif = indicator_data['macd_dif']
                dea = indicator_data['macd_dea']

                # 金叉：DIF上穿DEA
                if len(dif) >= 2 and dif.iloc[-2] <= dea.iloc[-2] and dif.iloc[-1] > dea.iloc[-1]:
                    crossover_strength = abs(dif.iloc[-1] - dea.iloc[-1])
                    patterns.append({
                        'pattern_name': 'MACD金叉',
                        'pattern_type': 'bullish_crossover',
                        'strength': min(crossover_strength * 10, 1.0),
                        'confidence': 0.8,
                        'signal_type': 'buy',
                        'description': 'MACD DIF线上穿DEA线，看涨信号'
                    })

            return patterns
        ```

        Args:
            data (pd.DataFrame): 包含指标计算结果的数据
                - 通常是 calculate() 的返回结果
                - 必须包含足够的历史数据以识别形态

        Returns:
            List[Dict[str, Any]]: 检测到的技术形态列表
                每个形态字典必须包含：
                - pattern_name (str): 形态名称
                - strength (float): 形态强度 0.0-1.0
                - confidence (float): 识别置信度 0.0-1.0
                - signal_type (str): 'buy', 'sell', 'neutral'
                可选字段：
                - pattern_type (str): 形态类型分类
                - duration (int): 形态持续天数
                - description (str): 形态描述
                - metadata (dict): 额外的形态信息
        """
        return [pattern.to_dict() for pattern in self._patterns]

    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        验证输入数据

        Args:
            data: 输入数据

        Returns:
            bool: 验证结果
        """
        if data is None or data.empty:
            return False

        required_columns = ["close"]
        return all(col in data.columns for col in required_columns)

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        预处理数据

        Args:
            data: 原始数据

        Returns:
            pd.DataFrame: 预处理后的数据
        """
        # 默认不做任何处理
        return data.copy()

    def postprocess_result(self, result: pd.DataFrame) -> pd.DataFrame:
        """
        后处理结果

        Args:
            result: 计算结果

        Returns:
            pd.DataFrame: 后处理后的结果
        """
        # 默认不做任何处理
        return result

    def format_output(self, result: Any) -> Dict[str, Any]:
        """
        格式化输出

        Args:
            result: 计算结果

        Returns:
            Dict[str, Any]: 格式化后的输出
        """
        return {
            "indicator": self.name,
            "period": self.period,
            "result": result,
            "timestamp": datetime.now().isoformat(),
        }

    def get_metadata(self) -> Dict[str, Any]:
        """
        获取指标元数据

        Returns:
            Dict[str, Any]: 元数据信息
        """
        return {
            "name": self.name,
            "period": self.period,
            "params": self.params,
            "type": self.__class__.__name__,
            "description": self.__doc__ or "",
        }

    @property
    def result(self) -> Optional[Any]:
        """获取计算结果"""
        return self._result

    def has_result(self) -> bool:
        """检查是否有计算结果"""
        return self._result is not None

    def clear_result(self):
        """清除计算结果"""
        self._result = None
        self._patterns = []

    def add_pattern(self, pattern: PatternInfo):
        """
        添加形态信息

        Args:
            pattern: 形态信息
        """
        self._patterns.append(pattern)

    def __str__(self) -> str:
        """字符串表示"""
        return f"{self.__class__.__name__}(name={self.name}, period={self.period})"

    def __repr__(self) -> str:
        """详细字符串表示"""
        return f"{self.__class__.__name__}(name='{self.name}', period={self.period}, params={self.params})"
