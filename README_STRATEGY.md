# 买点回测与策略生成系统

本系统用于分析股票买点附近的技术指标共性，并自动生成相应的选股策略。

## 功能特点

1. 支持对多个股票买点进行回测分析
2. 自动识别买点附近的技术指标特征和形态
3. 根据分析结果自动生成对应的选股策略
4. 支持多种买点类型，如回踩反弹、横盘突破等
5. 支持从CSV文件、数据库等多种数据源进行分析
6. 支持将分析结果和生成的策略保存为文件或数据库表

## 系统组件

系统主要包含以下组件：

1. **回测模块**：`scripts/backtest/backtest.py` - 用于回测股票买点和导出买点数据
2. **指标分析模块**：`scripts/backtest/indicator_analysis.py` - 用于分析买点附近的技术指标共性
3. **策略生成器**：`bin/strategy_generator.py` - 用于根据分析结果生成选股策略
4. **策略模块**：`strategy/` - 包含各类选股策略的实现
   - `strategy/rebound_strategy.py` - 回踩反弹买点策略
   - `strategy/breakout_strategy.py` - 横盘突破买点策略

## 使用流程

### 1. 准备买点数据

首先需要准备买点数据，格式为CSV文件，至少包含以下列：
- `code`：股票代码
- `buy_date`：买点日期，格式为YYYYMMDD
- `pattern_type`：买点类型（可选），如"回踩反弹"、"横盘突破"等

示例CSV文件（`data/example_buypoints.csv`）：
```csv
code,buy_date,pattern_type
000001,20240410,回踩反弹
000858,20240408,回踩反弹
600519,20240401,横盘突破
```

### 2. 导出回测买点数据

使用回测模块导出买点数据：

```bash
python bin/backtest.py --mode csv --input data/example_buypoints.csv --date 20240410 --export
```

参数说明：
- `--mode`：回测模式，支持csv、list、db、single等
- `--input`：输入源，根据模式不同代表不同含义
- `--date`：回测日期，格式为YYYYMMDD
- `--export`：导出买点数据
- `--pattern-type`：买点类型，用于导出买点数据（单个股票或股票列表回测时使用）

### 3. 分析技术指标共性

使用策略生成器分析买点附近的技术指标共性：

```bash
python bin/strategy_generator.py --input data/example_buypoints.csv
```

参数说明：
- `--input`：输入文件路径，支持CSV或TXT格式
- `--type`：输入源类型，支持csv、db
- `--output`：分析结果输出文件路径
- `--strategy-type`：策略类型，如"回踩反弹"、"横盘突破"等，不指定则自动判断
- `--save-to-db`：是否将买点数据保存到数据库
- `--use-db`：是否使用数据库中的买点数据进行分析
- `--table-name`：数据库表名

分析结果会保存到指定路径或默认路径（`data/result/日期_买点指标分析.json`）。

### 4. 生成选股策略

策略生成器会自动根据分析结果生成选股策略配置，保存到指定路径或默认路径（`data/result/日期_策略名称_策略.json`）。

## 常见用例

### 案例1：分析回踩反弹买点

1. 准备回踩反弹买点数据：
```csv
code,buy_date,pattern_type
000001,20240410,回踩反弹
000858,20240408,回踩反弹
002415,20240409,回踩反弹
```

2. 分析并生成策略：
```bash
python bin/strategy_generator.py --input data/回踩反弹买点.csv --strategy-type 回踩反弹
```

### 案例2：分析横盘突破买点

1. 准备横盘突破买点数据：
```csv
code,buy_date,pattern_type
300059,20240405,横盘突破
600036,20240403,横盘突破
600519,20240401,横盘突破
```

2. 分析并生成策略：
```bash
python bin/strategy_generator.py --input data/横盘突破买点.csv --strategy-type 横盘突破
```

### 案例3：自动识别买点类型并生成策略

```bash
python bin/strategy_generator.py --input data/混合买点.csv
```

系统会根据技术指标共性自动判断适合的策略类型。

## 策略参数优化

系统会根据分析结果自动优化策略参数。例如：

- 回踩反弹策略：
  - 根据回踩均线的周期自动调整`ma_period`参数
  - 根据KDJ金叉的比例自动调整`kdj_up`参数

- 横盘突破策略：
  - 根据成交量放大的情况自动调整`volume_ratio`参数
  - 根据BOLL指标的使用情况自动调整`boll_use`参数

## 数据库集成

系统支持将买点数据保存到ClickHouse数据库，便于后续分析和查询：

```bash
python bin/strategy_generator.py --input data/example_buypoints.csv --save-to-db --table-name my_buypoints
```

保存后，可以直接使用数据库表作为分析源：

```bash
python bin/strategy_generator.py --input my_buypoints --type db
``` 