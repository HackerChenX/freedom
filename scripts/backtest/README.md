# 统一回测系统使用指南

本文档提供股票技术指标分析与回测系统的使用说明。系统支持38种技术指标，可进行多周期、多形态分析。

## 功能特点

1. **多周期分析**：支持日线、15分钟、30分钟、60分钟、周线、月线等多个周期
2. **全面技术指标**：支持38种技术指标计算和形态识别
3. **交叉形态识别**：识别均线交叉、MACD金叉、KDJ金叉等多种交叉形态
4. **特殊形态检测**：V形反转、量价背离、岛型反转等复杂形态识别
5. **跨周期特征提取**：识别多个周期共同出现的形态特征
6. **自动生成策略**：基于回测结果自动生成策略代码
7. **全量历史数据**：使用股票所有可用的历史数据，确保技术指标计算的准确性

## 系统架构

- `unified_backtest.py`: 统一回测系统核心实现
- `indicator_check.py`: 技术指标检查与修复工具
- `bin/run_backtest.py`: 回测系统执行入口

## 快速开始

### 执行回测分析

```bash
# 基本用法
python bin/run_backtest.py -i data/backtest_sample.txt

# 指定输出文件
python bin/run_backtest.py -i data/backtest_sample.txt -o data/result/my_result.json

# 指定买点类型
python bin/run_backtest.py -i data/backtest_sample.txt -t "MACD金叉"
```

### 检查指标系统

```bash
# 运行指标检查
python bin/run_backtest.py --check
```

## 输入文件格式

输入文件为文本文件，每行包含一个买点，格式为：
```
股票代码,买点日期,买点类型描述(可选)
```

示例：
```
000001,20230601,趋势突破
600519,20230515,MACD金叉
```

注意：
- 日期格式为YYYYMMDD
- 以#开头的行为注释
- 买点类型描述可选，不提供则使用命令行中的-t参数
- **买点日期仅作为数据截止日期**，系统会使用股票的所有历史数据进行分析

## 数据处理说明

系统处理数据的方式如下：
1. 对于每个股票，获取**全部历史数据**（从数据库中查询所有可用数据）
2. 将输入文件中指定的买点日期作为**数据截止日期**
3. 在该截止日期前后分析技术指标特征和形态
4. 这种方式确保技术指标计算具有足够的历史数据，特别是对于需要长周期历史数据的指标（如MACD、布林带等）

## 输出结果

系统会输出两类结果：
1. **分析结果JSON文件**：包含详细的技术指标和形态分析
2. **策略文件**：基于分析结果自动生成的策略代码

### 分析结果结构

```json
[
  {
    "code": "股票代码",
    "name": "股票名称",
    "industry": "所属行业",
    "buy_date": "买点日期",
    "pattern_type": "买点类型",
    "buy_price": 买入价格,
    "periods": {
      "daily": {
        "indicators": {...},
        "patterns": [...]
      },
      "min60": {...},
      ...
    },
    "patterns": ["形态1", "形态2", ...],
    "cross_period_patterns": ["跨周期形态1", ...]
  },
  ...
]
```

## 支持的技术指标

系统支持以下38种技术指标：

1. **趋势指标**：MA, EMA, WMA, MACD, BIAS, BOLL, SAR, DMI, TRIX
2. **动量指标**：RSI, KDJ, WR, CCI, MTM, ROC, STOCHRSI, RSIMA, Momentum
3. **成交量指标**：VOL, OBV, VOSC, MFI, VR, PVT, VOLUME_RATIO
4. **波动指标**：ATR, EMV, VIX, INTRADAY_VOLATILITY
5. **形态识别**：CANDLESTICK_PATTERNS, V_SHAPED_REVERSAL, ISLAND_REVERSAL
6. **特殊分析**：ZXM_WASHPLATE, ZXM_ABSORB, CHIP_DISTRIBUTION
7. **高级工具**：FIBONACCI_TOOLS, ELLIOTT_WAVE, GANN_TOOLS, TIME_CYCLE_ANALYSIS
8. **形态分析**：PLATFORM_BREAKOUT, DIVERGENCE, MULTI_PERIOD_RESONANCE

## 常见问题

### 如何添加自定义指标？

1. 在`indicators`目录下创建指标类
2. 在`indicators/__init__.py`中导入该类
3. 在`enums/indicator_types.py`中添加指标类型
4. 在`indicators/factory.py`中注册指标

### 如何修改指标参数？

可以在`unified_backtest.py`文件中修改各指标的默认参数，例如：

```python
# 修改RSI周期
rsi_indicator = IndicatorFactory.create_indicator("RSI", period=14)  # 默认值为14
```

## 进阶用法

### 自定义分析周期

可以在`UnifiedBacktest._analyze_period`方法中修改分析周期：

```python
periods = {
    'daily': KlinePeriod.DAILY,
    'min15': KlinePeriod.MIN_15,
    # 添加或移除周期
}
```

### 定制策略生成

可以修改`UnifiedBacktest.generate_strategy`方法，根据需要定制生成的策略代码。 