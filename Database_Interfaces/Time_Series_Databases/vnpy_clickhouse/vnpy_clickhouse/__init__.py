"""
VnPy ClickHouse数据库接口模块

ClickHouse是一个高性能的列式数据库管理系统，特别适合OLAP场景。
本模块为VnPy提供ClickHouse数据库的接口支持，用于存储和查询大量的金融时序数据。

主要特性:
- 高性能的列式存储
- 优秀的数据压缩率
- 支持SQL查询
- 适合大数据量的时序数据分析
- 支持分布式部署

支持的数据类型:
- K线数据 (BarData)
- Tick数据 (TickData)
- 数据概览 (BarOverview, TickOverview)

作者: VnPy Team
版本: 1.0.0
"""

from .clickhouse_database import ClickHouseDatabase as Database

__version__ = "1.0.0"
__author__ = "VnPy Team"
__email__ = "vn.py@foxmail.com"

__all__ = ["Database"]
