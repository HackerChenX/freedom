# 股票分析系统

基于Python的股票回测、技术分析与选股系统，提供完整的股票技术指标计算、回测、形态识别、策略生成和选股功能。

## 项目概述

本系统是一个综合性的股票分析工具，用于股票技术分析、策略回测和选股。系统从ClickHouse数据库获取数据，提供一整套从数据获取、技术指标计算、形态识别、回测分析到选股策略生成和执行的完整功能。

### 主要特点

- **丰富的技术指标**: 支持50+种技术指标，包括趋势、震荡、量能指标及特色ZXM体系指标
- **多周期分析**: 支持日/周/月/小时/分钟等多周期数据分析
- **形态识别**: 内置形态识别系统，识别金叉、死叉、背离等技术形态
- **强大的回测**: 支持多股票、多买点的批量回测和分析
- **策略自动生成**: 基于回测结果自动生成选股策略
- **高性能**: 支持并行计算和数据缓存，优化性能
- **完整性**: 提供从数据获取、分析到策略执行的完整解决方案

## 项目结构

```
.
├── analysis/            # 分析模块
│   ├── buypoints/       # 买点分析
│   └── market/          # 市场分析
├── api/                 # 外部API接口实现
├── bin/                 # 可执行脚本
│   ├── main.py          # 主程序入口
│   ├── stock_select.py  # 选股功能入口
│   └── backtest.py      # 回测功能入口
├── config/              # 配置管理模块
├── data/                # 数据目录
│   ├── reference/       # 参考数据
│   └── result/          # 结果数据
├── db/                  # 数据库接口模块
├── doc/                 # 文档目录
│   ├── 使用指南/         # 系统使用相关指南
│   ├── 公式集/           # 交易和选股公式集合
│   ├── 回测文档/         # 回测系统相关文档
│   ├── 指标文档/         # 技术指标实现和使用文档
│   ├── 需求文档/         # 系统需求规格说明
│   └── 系统设计/         # 系统架构和设计方案
├── enums/               # 枚举定义模块
├── formula/             # 公式计算模块
├── indicators/          # 技术指标模块
│   ├── oscillator/      # 震荡指标
│   ├── pattern/         # 形态指标
│   ├── trend/           # 趋势指标
│   ├── volume/          # 成交量指标
│   └── zxm/             # ZXM体系指标
├── logs/                # 日志目录
│   └── archive/         # 日志存档
├── scripts/             # 脚本工具
│   ├── backtest/        # 回测脚本
│   ├── sql/             # SQL脚本
│   └── utils/           # 工具脚本
├── sql/                 # SQL文件
├── strategy/            # 选股策略模块
├── tests/               # 测试模块
└── utils/               # 工具模块
```

## 核心模块功能

### 1. 技术指标模块 (indicators/)

技术指标模块是系统的基础，提供各类技术指标的计算和分析功能。

#### 主要指标类别:

- **趋势指标**: MA(移动平均线)、MACD、DMI、SAR等
- **震荡指标**: RSI、KDJ、BIAS、WR(威廉指标)等
- **成交量指标**: OBV、VOL、EMV、PVT等
- **形态指标**: 支持各种K线形态和技术形态识别
- **ZXM体系指标**: 自定义指标体系，包含多维度分析

#### 使用示例:

```python
from indicators.factory import IndicatorFactory

# 创建MACD指标实例
macd = IndicatorFactory.create("MACD", fast_period=12, slow_period=26, signal_period=9)

# 计算指标值
result = macd.calculate(data)
```

### 2. 回测系统 (scripts/backtest/)

回测系统是分析交易策略有效性的核心工具，支持多种回测模式和分析功能。

#### 主要功能:

- **合并回测系统**: `consolidated_backtest.py` 提供统一的回测接口
- **多买点回测**: 支持批量回测多个股票在多个日期的表现
- **形态回测**: 支持基于技术形态的回测
- **策略生成**: 基于回测结果自动生成选股策略
- **性能分析**: 提供详细的回测性能统计和分析

#### 使用示例:

```python
from scripts.backtest.consolidated_backtest import ConsolidatedBacktest

# 初始化回测系统
backtest = ConsolidatedBacktest()

# 执行批量回测
results = backtest.batch_analyze(
    input_file="data/stocks_to_test.csv", 
    output_file="data/result/backtest_result.xlsx"
)

# 基于回测结果生成策略
strategy = backtest.generate_strategy(
    results, 
    output_file="data/result/strategy.json"
)
```

### 3. 选股系统 (strategy/)

选股系统基于定义的策略和条件，从股票池中筛选符合条件的股票。

#### 主要组件:

- **策略管理器**: `strategy_manager.py` 管理策略的创建、保存和加载
- **策略解析器**: `strategy_parser.py` 解析策略定义
- **策略执行器**: `strategy_executor.py` 执行选股策略
- **策略优化器**: `strategy_optimizer.py` 优化选股策略参数

#### 使用示例:

```python
from bin.stock_select import run_stock_selection

# 执行选股
selected_stocks = run_stock_selection(
    strategy_id="STRATEGY_A1B2C3D4",
    date="20240520",
    stock_pool="all"
)
```

### 4. 数据库接口 (db/)

数据库接口模块提供对ClickHouse数据库的访问，管理股票数据的读写操作。

#### 主要功能:

- **连接管理**: 管理数据库连接池和资源
- **数据查询**: 提供高效的数据查询接口
- **数据写入**: 支持结果数据存储和导出
- **连接优化**: 实现连接池和性能优化

#### 使用示例:

```python
from db.clickhouse_db import get_clickhouse_db

# 获取数据库连接
db = get_clickhouse_db()

# 查询股票数据
df = db.get_kline_data(
    stock_code="000001", 
    start_date="20240101", 
    end_date="20240520", 
    period="day"
)
```

### 5. 分析模块 (analysis/)

分析模块提供市场分析和买点分析功能，帮助用户理解市场状况和识别潜在交易机会。

#### 主要组件:

- **买点分析**: `buypoints/analyze_buypoints.py` 分析潜在买入机会
- **市场分析**: `market/a_stock_market_analysis.py` 分析整体市场状况
- **多维度分析**: `multi_dimension_analyzer.py` 提供多角度分析

#### 使用示例:

```python
from analysis.buypoints.analyze_buypoints import analyze_buypoints

# 分析买点
results = analyze_buypoints(
    stock_code="000001", 
    date="20240520"
)
```

### 6. 工具模块 (utils/)

工具模块提供各种辅助功能，支持系统的其他部分。

#### 主要工具:

- **装饰器**: `decorators.py` 提供性能监控、缓存等装饰器
- **日期工具**: `date_utils.py` 处理日期转换和计算
- **日志工具**: `logger.py` 提供统一的日志记录
- **路径工具**: `path_utils.py` 管理文件路径
- **周期管理**: `period_manager.py` 管理不同周期的数据

#### 使用示例:

```python
from utils.logger import get_logger
from utils.date_utils import get_previous_trade_date

# 获取日志器
logger = get_logger(__name__)

# 获取前一个交易日
prev_date = get_previous_trade_date("20240520")
```

## 配置管理

系统使用集中式配置管理，支持多种配置方式。

### 配置方法:

1. **默认配置**: 系统内置默认配置
2. **配置文件**: 用户可通过 `config/user_config.json` 覆盖默认配置
3. **环境变量**: 支持通过环境变量覆盖配置项
4. **运行时配置**: 支持在运行时动态修改配置

### 主要配置项:

- **数据库配置**: 数据库连接信息
- **路径配置**: 输出目录、日志目录等
- **日期配置**: 默认起止日期
- **日志配置**: 日志级别、文件大小等
- **安全配置**: 敏感信息加密设置

### 获取配置示例:

```python
from config.config import get_config

# 获取数据库配置
db_config = get_config("db")

# 获取特定配置项
output_dir = get_config("paths.output")
```

## 使用指南

### 1. 环境准备

确保已安装以下依赖:

- Python 3.8+
- ClickHouse数据库
- 必要的Python包 (requirements.txt)

### 2. 数据库配置

1. 创建ClickHouse数据库 `stock`
2. 导入基础数据或使用数据同步工具同步数据
3. 配置数据库连接信息 (config/user_config.json)

### 3. 主要功能使用

#### 选股功能:

```bash
# 基于策略ID选股
python bin/stock_select.py --mode strategy --strategy_id STRATEGY_A1B2C3D4 --date 20240520

# 基于策略文件选股
python bin/stock_select.py --mode file --strategy_file path/to/strategy.json --date 20240520

# 多线程选股
python bin/stock_select.py --mode strategy --strategy_id STRATEGY_A1B2C3D4 --date 20240520 --threads 8
```

#### 回测功能:

```bash
# 批量回测
python bin/backtest.py --mode batch --input_file data/stock_list.csv --output_file data/result/backtest_result.xlsx

# 单股回测
python bin/backtest.py --mode single --stock_code 000001 --buy_date 20240101 --output_file data/result/single_result.xlsx

# 指定指标和周期回测
python bin/backtest.py --mode batch --input_file data/stock_list.csv --indicators MACD,KDJ,RSI --periods daily,weekly
```

#### 市场分析:

```bash
# 市场分析
python bin/main.py --mode market --date 20240520

# 板块分析
python bin/main.py --mode sector --date 20240520

# 技术面分析
python bin/main.py --mode technical --date 20240520
```

#### 数据同步:

```bash
# 同步所有数据
python scripts/utils/data_sync.py --mode all

# 同步特定股票数据
python scripts/utils/data_sync.py --mode stock --code 000001

# 同步行业数据
python scripts/utils/data_sync.py --mode industry
```

### 4. 高级功能

#### 策略自动生成:

```bash
# 基于回测结果生成策略
python bin/convert_backtest_to_strategy.py --input data/result/backtest_result.xlsx --output data/result/strategy.json

# 优化策略
python bin/optimize_strategy.py --strategy_id STRATEGY_A1B2C3D4 --target_success_rate 0.6
```

#### 形态分析:

```bash
# 运行形态分析
python bin/run_pattern_analysis.py --stock_code 000001 --date 20240520

# 形态组合回测
python bin/run_pattern_combination_backtest.py --input_file data/stock_list.csv --patterns MACD_GOLDEN_CROSS,KDJ_OVERSOLD_REBOUND
```

## 技术栈

系统使用以下主要技术和库:

- **Python 3.8+**: 主要开发语言
- **ClickHouse**: 高性能分析型数据库
- **NumPy/Pandas**: 数据处理和分析
- **Matplotlib/Plotly**: 数据可视化
- **multiprocessing**: 并行计算支持
- **cryptography**: 敏感信息加密

## 性能优化

系统采用多种优化策略提升性能:

1. **并行计算**: 支持多线程/多进程并行处理
2. **数据缓存**: 实现多级缓存减少重复计算
3. **查询优化**: 优化ClickHouse查询性能
4. **内存管理**: 优化内存使用，减少内存占用
5. **装饰器优化**: 使用装饰器实现性能监控和优化

## 最新优化改进

近期系统进行了多项优化和改进:

### 1. 架构重构

- 重构了技术指标形态识别功能，将形态识别逻辑迁移至各指标类中
- 实现了周期与指标的分离，解决了不同周期同名指标的混淆问题
- 建立了统一的指标评分系统，为所有指标提供标准化的评分功能

### 2. 功能增强

- 增强了回测系统，支持更精细的参数配置
- 改进了共性特征分析，支持全面特征统计和关联分析
- 优化了策略自动生成功能，提高策略质量

### 3. 性能优化

- 优化了数据库查询性能
- 实现了高效的数据缓存机制
- 增强了并行计算支持
- 减少了重复计算和内存消耗

## 贡献指南

欢迎贡献代码、报告问题或提出改进建议:

1. Fork项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 许可证

MIT 