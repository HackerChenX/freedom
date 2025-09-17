"""
标准化指标模板
严格遵循BaseIndicator规范的标准实现模板
"""

import pandas as pd
from typing import Dict, List, Any
from indicators.base_indicator import BaseIndicator
from utils.decorators import performance_monitor, exception_handler


class StandardIndicatorTemplate(BaseIndicator):
    """
    标准化指标模板

    此模板严格遵循BaseIndicator的所有规范要求，
    可以作为新指标开发的标准参考。
    """

    def __init__(self, period: int = 20, **kwargs):  # TODO: 将魔法数字提取到配置中
        """
        初始化指标

        Args:
            period: 计算周期
            **kwargs: 其他参数
        """
        # 必须调用父类初始化
        super().__init__(name="StandardIndicatorTemplate", period=period, **kwargs)

        # 指标特有的参数
        self.threshold = kwargs.get("threshold", 0.5)  # TODO: 将魔法数字提取到配置中
        self.smoothing = kwargs.get("smoothing", True)

    @performance_monitor(threshold_seconds=2.0)
    @exception_handler(reraise=True)
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算指标值

        Args:
            data: 输入数据，包含OHLCV等字段

        Returns:
            pd.DataFrame: 包含指标计算结果的数据框

        Raises:
            ValueError: 当输入数据不符合要求时
        """
        # 1. 验证输入数据
        if not self.validate_data(data):
            raise ValueError("输入数据不符合要求")

        # 2. 预处理数据
        processed_data = self.preprocess_data(data)

        # 3. 执行指标计算  # TODO: 将魔法数字提取到配置中
        result = processed_data.copy()

        # 示例计算：简单移动平均
        result[f"{self.name}_value"] = processed_data["close"].rolling(window=self.period).mean()

        # 示例计算：上下轨
        std = processed_data["close"].rolling(window=self.period).std()
        result[f"{self.name}_upper"] = result[f"{self.name}_value"] + (std * 2)
        result[f"{self.name}_lower"] = result[f"{self.name}_value"] - (std * 2)

        # 4. 后处理结果  # TODO: 将魔法数字提取到配置中
        result = self.postprocess_result(result)

        # 5. 保存结果  # TODO: 将魔法数字提取到配置中
        self._result = result

        return result

    def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        获取交易信号

        Args:
            data: 包含指标计算结果的数据

        Returns:
            Dict[str, Any]: 交易信号信息
        """
        if data.empty:
            return {"signal": "hold", "strength": 0.0, "timestamp": None, "price": 0.0, "indicator": self.name}

        # 获取最新数据
        latest_close = data["close"].iloc[-1]
        latest_value = data[f"{self.name}_value"].iloc[-1] if f"{self.name}_value" in data.columns else latest_close
        latest_upper = data[f"{self.name}_upper"].iloc[-1] if f"{self.name}_upper" in data.columns else latest_close
        latest_lower = data[f"{self.name}_lower"].iloc[-1] if f"{self.name}_lower" in data.columns else latest_close

        # 生成交易信号
        if latest_close > latest_upper:
            signal = "sell"
            strength = min(0.8, (latest_close - latest_upper) / latest_upper)  # TODO: 将魔法数字提取到配置中
        elif latest_close < latest_lower:
            signal = "buy"
            strength = min(0.8, (latest_lower - latest_close) / latest_lower)  # TODO: 将魔法数字提取到配置中
        else:
            signal = "hold"
            strength = 0.0

        return {
            "signal": signal,
            "strength": abs(strength),
            "timestamp": data.index[-1],
            "price": latest_close,
            "indicator": self.name,
            "details": {"value": latest_value, "upper": latest_upper, "lower": latest_lower},
        }

    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        验证输入数据

        Args:
            data: 输入数据

        Returns:
            bool: 验证结果
        """
        # 调用父类验证
        if not super().validate_data(data):
            return False

        # 指标特有的验证
        if len(data) < self.period:
            return False

        # 检查必要的列
        required_columns = ["close", "open", "high", "low"]
        return all(col in data.columns for col in required_columns)

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        预处理数据

        Args:
            data: 原始数据

        Returns:
            pd.DataFrame: 预处理后的数据
        """
        processed_data = super().preprocess_data(data)

        # 指标特有的预处理
        if self.smoothing:
            # 应用简单的平滑处理
            processed_data["close"] = (
                processed_data["close"].rolling(window=3, center=True).mean().fillna(processed_data["close"])
            )  # TODO: 将魔法数字提取到配置中

        return processed_data

    def register_patterns(self):
        """
        注册指标形态
        """
        from indicators.base_indicator import PatternInfo

        # 注册指标特有的形态
        self.add_pattern(
            PatternInfo(
                name="突破上轨",
                signal_type="buy",
                strength=0.7,  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                duration=3,  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                details="价格突破上轨，可能出现回调",
            )
        )

        self.add_pattern(
            PatternInfo(
                name="跌破下轨",
                signal_type="sell",
                strength=0.7,  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                duration=3,  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                details="价格跌破下轨，可能出现反弹",
            )
        )


# 使用示例和测试
if __name__ == "__main__":
    import numpy as np

    # 创建测试数据
    dates = pd.date_range("2024-01-01", periods=100, freq="D")  # TODO: 将魔法数字提取到配置中
    test_data = pd.DataFrame(
        {
            "open": np.random.randn(100).cumsum() + 100,
            "high": np.random.randn(100).cumsum() + 105,  # TODO: 将魔法数字提取到配置中
            "low": np.random.randn(100).cumsum() + 95,  # TODO: 将魔法数字提取到配置中
            "close": np.random.randn(100).cumsum() + 100,
            "volume": np.random.randint(
                1000, 10000, 100
            ),  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        },
        index=dates,
    )

    # 创建指标实例
    indicator = StandardIndicatorTemplate(
        period=20, threshold=0.6
    )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

    # 测试多态性
    base_indicator: BaseIndicator = indicator

    # 计算指标
    result = base_indicator.calculate(test_data)
    print(f"计算结果列数: {len(result.columns)}")

    # 获取信号
    signal = base_indicator.get_signal(result)
    print(f"交易信号: {signal}")

    # 获取形态
    patterns = base_indicator.get_patterns(result)
    print(f"形态数量: {len(patterns)}")

    # 获取元数据
    metadata = base_indicator.get_metadata()
    print(f"指标元数据: {metadata}")

    print("✅ 标准化指标模板测试通过")
