"""
震荡指标模块包

包含各种震荡类技术指标的实现
"""

from indicators.oscillator.enhanced_kdj import EnhancedKdj

# 导出的类
__all__ = ["EnhancedKdj"]

# 为了向后兼容，创建别名
EnhancedKDJ = EnhancedKdj

# 版本信息
__version__ = "0.1.0"
