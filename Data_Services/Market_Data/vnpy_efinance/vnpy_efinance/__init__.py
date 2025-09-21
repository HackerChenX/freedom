"""
VnPy EFinance数据源模块

EFinance是一个免费开源的Python库，用于获取股票、基金、期货数据。
支持A股、港股、美股等多个市场的历史和实时数据。

主要特性：
- 完全免费，无需注册
- 支持多种数据类型：股票、基金、期货
- 支持多个时间周期：日线、分钟线等
- 数据质量高，更新及时

GitHub: https://github.com/Micro-sheep/efinance
文档: https://efinance.readthedocs.io/en/latest/
"""

from .efinance_datafeed import EfinanceDatafeed

# VnPy约定：数据源类必须命名为Datafeed
Datafeed = EfinanceDatafeed

__version__ = "1.0.0"
__author__ = "VnPy Freedom"

__all__ = ["EfinanceDatafeed", "Datafeed"]
