"""
周期类型枚举兼容模块

注意: 这是一个兼容层，用于向后兼容。
新代码应直接使用 `from enums.period import Period`
"""

# 从统一周期模块导入
from enums.period import Period as PeriodType

# 为了保持完全兼容性，重新导出所有功能
__all__ = ['PeriodType']

# 警告消息
import warnings
warnings.warn(
    "PeriodType已合并到Period中，请使用'from enums.period import Period'代替",
    DeprecationWarning,
    stacklevel=2
) 