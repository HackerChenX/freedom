# 文件结构和模块组织规范

## 项目目录结构

```
.
├── analysis/            # L5: 分析模块
│   ├── buypoints/       # 买点分析
│   └── market/          # 市场分析
├── api/                 # L4: 外部API接口实现
├── bin/                 # L6: 可执行脚本
├── config/              # L2: 配置管理模块
├── crawler/             # L3: 数据爬取模块
├── data/                # L1: 数据目录
│   └── result/          # 结果数据
├── db/                  # L3: 数据库接口模块
├── doc/                 # 文档目录
├── enums/               # L2: 枚举定义模块
├── formula/             # L5: 公式计算模块
├── indicators/          # L4: 技术指标模块
├── logs/                # 日志目录
├── monitoring/          # L4: 监控模块
├── scripts/             # L6: 脚本工具
├── sql/                 # L1: SQL文件
├── strategy/            # L5: 选股策略模块
├── tests/               # 测试模块
└── utils/               # L2: 工具模块
```

## 文件命名规范

### 1. Python文件命名

**使用小写字母加下划线**:
```
# 正确
query_executor.py
data_manager.py
technical_indicators.py
buy_point_analyzer.py

# 错误
QueryExecutor.py
DataManager.py
TechnicalIndicators.py
BuyPointAnalyzer.py
```

### 2. 模块文件组织

**每个目录必须有__init__.py**:
```
db/
├── __init__.py          # 模块初始化
├── query_executor.py    # 查询执行器
├── sql_manager.py       # SQL管理器
└── clickhouse_db.py     # ClickHouse连接
```

**__init__.py内容规范**:
```python
# db/__init__.py
"""
数据库访问模块

提供统一的数据库访问接口和查询管理功能。
"""

from .query_executor import get_query_executor
from .sql_manager import QueryType, SQLManager
from .clickhouse_db import get_clickhouse_db

__all__ = [
    'get_query_executor',
    'QueryType',
    'SQLManager', 
    'get_clickhouse_db'
]
```

## 核心模块规范

### 1. 数据访问层 (L3)

**db/目录结构**:
```
db/
├── __init__.py
├── query_executor.py    # 统一查询执行器 (必须)
├── sql_manager.py       # SQL查询管理器 (必须)
├── clickhouse_db.py     # ClickHouse数据库连接 (必须)
├── data_manager.py      # 数据管理器
└── connection_pool.py   # 连接池管理
```

**关键文件作用**:
- [db/query_executor.py](mdc:db/query_executor.py) - 所有数据访问的统一入口
- [db/sql_manager.py](mdc:db/sql_manager.py) - SQL查询模板管理
- [db/clickhouse_db.py](mdc:db/clickhouse_db.py) - 底层数据库连接

### 2. 业务逻辑层 (L5)

**analysis/目录结构**:
```
analysis/
├── __init__.py
├── buypoints/           # 买点分析模块
│   ├── __init__.py
│   ├── base_analyzer.py
│   └── technical_analyzer.py
└── market/              # 市场分析模块
    ├── __init__.py
    ├── trend_analyzer.py
    └── volume_analyzer.py
```

**strategy/目录结构**:
```
strategy/
├── __init__.py
├── base_strategy.py     # 策略基类
├── momentum_strategy.py # 动量策略
├── value_strategy.py    # 价值策略
└── factory.py          # 策略工厂
```

### 3. 服务层 (L4)

**indicators/目录结构**:
```
indicators/
├── __init__.py
├── base_indicator.py    # 指标基类
├── technical.py         # 技术指标
├── oscillators.py       # 震荡指标
├── trend.py            # 趋势指标
└── volume.py           # 成交量指标
```

### 4. 基础设施层 (L2)

**utils/目录结构**:
```
utils/
├── __init__.py
├── logger.py           # 日志工具 (必须)
├── date_utils.py       # 日期工具
├── file_utils.py       # 文件工具
├── path_utils.py       # 路径工具 (必须)
├── cache.py            # 缓存工具
└── decorators.py       # 装饰器工具
```

**config/目录结构**:
```
config/
├── __init__.py
├── database_config_manager.py  # 数据库配置管理
├── docker_config.json         # Docker配置
└── default_config.yaml        # 默认配置
```

## 导入规范

### 1. 模块导入顺序

```python
# 1. 标准库导入
import os
import sys
import logging
from typing import Dict, List, Optional
from datetime import datetime

# 2. 第三方库导入
import pandas as pd
import numpy as np
from clickhouse_driver import Client

# 3. 项目内模块导入 (按层级顺序)
# L2层导入
from utils.logger import get_logger
from config import get_config
from enums.trend_types import TrendType

# L3层导入
from db.query_executor import get_query_executor
from db.sql_manager import QueryType

# L4层导入
from indicators.technical import calculate_macd

# L5层导入
from analysis.buypoints import BuyPointAnalyzer
```

### 2. 相对导入规范

**在同一模块内使用相对导入**:
```python
# indicators/technical.py
from .base_indicator import BaseIndicator
from .oscillators import RSI, MACD

# 跨模块使用绝对导入
from db.query_executor import get_query_executor
from utils.logger import get_logger
```

### 3. 禁止的导入方式

```python
# ❌ 禁止通配符导入
from utils import *
from indicators import *

# ❌ 禁止循环导入
# 文件A导入文件B，文件B又导入文件A

# ❌ 禁止跨层级导入违规
# L2层文件导入L3层或更高层的模块
```

## 文件内容组织

### 1. 文件头部规范

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
技术指标计算模块

提供各种技术指标的计算功能，包括移动平均线、RSI、MACD等。

Author: Stock Analysis System
Created: 2024-01-01
Modified: 2024-12-01
"""

import logging
from typing import Dict, List, Optional
import pandas as pd

from utils.logger import get_logger
from db.query_executor import get_query_executor

logger = get_logger(__name__)
```

### 2. 类和函数组织

```python
# 1. 常量定义
DEFAULT_PERIOD = 20
MAX_PERIOD = 250

# 2. 异常类定义
class IndicatorCalculationError(Exception):
    """指标计算异常"""
    pass

# 3. 主要类定义
class TechnicalIndicator:
    """技术指标基类"""
    pass

class MovingAverage(TechnicalIndicator):
    """移动平均线指标"""
    pass

# 4. 工具函数
def validate_data(data: pd.DataFrame) -> bool:
    """验证数据格式"""
    pass

def calculate_sma(data: pd.Series, period: int) -> pd.Series:
    """计算简单移动平均"""
    pass

# 5. 主函数或入口函数
def main():
    """主函数"""
    pass

if __name__ == "__main__":
    main()
```

## 配置文件组织

### 1. 配置文件结构

```
config/
├── __init__.py
├── default_config.yaml      # 默认配置
├── database_config.yaml     # 数据库配置
├── docker_config.json       # Docker配置
├── logging_config.yaml      # 日志配置
└── .env.example            # 环境变量示例
```

### 2. 配置文件内容规范

**default_config.yaml**:
```yaml
# 系统配置
system:
  name: "股票分析系统"
  version: "1.0.0"
  debug: false

# 数据库配置
database:
  host: "localhost"
  port: 9000
  database: "stock"
  user: "default"
  password: ""
  timeout: 30

# 日志配置
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/system.log"
```

## 测试文件组织

### 1. 测试目录结构

```
tests/
├── __init__.py
├── unit/                # 单元测试
│   ├── __init__.py
│   ├── test_query_executor.py
│   ├── test_indicators.py
│   └── test_utils.py
├── integration/         # 集成测试
│   ├── __init__.py
│   ├── test_database.py
│   └── test_api.py
├── fixtures/            # 测试数据
│   ├── sample_data.csv
│   └── test_config.json
└── conftest.py         # pytest配置
```

### 2. 测试文件命名

```python
# 测试文件必须以test_开头
test_query_executor.py
test_technical_indicators.py
test_buy_point_analyzer.py

# 测试类必须以Test开头
class TestQueryExecutor:
    pass

class TestTechnicalIndicators:
    pass

# 测试方法必须以test_开头
def test_calculate_moving_average():
    pass

def test_query_stock_data():
    pass
```

## 文档文件组织

### 1. 文档目录结构

```
doc/
├── 使用指南/            # 用户使用指南
├── 公式集/              # 交易和选股公式
├── 回测文档/            # 回测系统文档
├── 指标文档/            # 技术指标文档
├── 需求文档/            # 系统需求文档
├── 系统设计/            # 系统设计文档
└── API文档/             # API接口文档
```

### 2. 文档命名规范

```
# 使用中文命名，便于理解
系统架构设计.md
数据库设计文档.md
API接口规范.md
技术指标使用指南.md
```

## 日志文件组织

### 1. 日志目录结构

```
logs/
├── system.log          # 系统日志
├── error.log           # 错误日志
├── access.log          # 访问日志
├── performance.log     # 性能日志
└── archive/            # 日志归档
    ├── 2024-01/
    └── 2024-02/
```

### 2. 日志文件命名

```python
# 按日期和类型命名
system_2024-12-01.log
error_2024-12-01.log
access_2024-12-01.log

# 按模块命名
db_query_2024-12-01.log
indicator_calc_2024-12-01.log
strategy_exec_2024-12-01.log
```

## 数据文件组织

### 1. 数据目录结构

```
data/
├── raw/                # 原始数据
├── processed/          # 处理后数据
├── result/             # 分析结果
├── cache/              # 缓存数据
└── backup/             # 备份数据
```

### 2. 数据文件命名

```python
# 按日期和类型命名
stock_data_2024-12-01.csv
indicator_data_2024-12-01.csv
analysis_result_2024-12-01.json

# 按股票代码命名
000001_daily_data.csv
000001_indicator_data.csv
000001_analysis_result.json
```
globs: *.py
description: 文件结构和模块组织规范
---
