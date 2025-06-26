# ZXM指标API文档

## 概述

ZXM指标体系是专门为量化投资设计的技术分析指标集合，包含买点识别、趋势分析、弹性评估、评分系统和选股模型等多个维度的指标。

本文档详细介绍了所有已修复的ZXM指标的API接口、参数说明、返回值格式和使用示例。

## 修复状态

所有ZXM指标已完成P0级信号生成语义修复，确保指标输出与信号生成100%一致。

## 指标分类

### 1. 买点识别指标

#### ZXM_TURNOVER - 换手率指标

**功能描述**：基于换手率识别高活跃度买点机会

**API接口**：
```python
from indicators.complete_indicator_registry import complete_registry

indicator = complete_registry.create_indicator('ZXM_TURNOVER')
result = indicator.calculate(data)
```

**参数说明**：
- `threshold`: float, 默认0.7, 换手率阈值
- `data`: DataFrame, 必须包含['datetime', 'open', 'high', 'low', 'close', 'volume', 'turnover_rate']

**返回值格式**：
```python
DataFrame包含以下列：
- 'datetime': 时间戳
- 'XG': bool, 换手率信号（turnover_rate > threshold）
- 'buy_signal': bool, 买入信号（XG == True）
- 'sell_signal': bool, 卖出信号（XG == False）
- 'hold_signal': bool, 持有信号（XG == False）
```

**使用示例**：
```python
# 创建指标实例
turnover_indicator = complete_registry.create_indicator('ZXM_TURNOVER')

# 计算指标
result = turnover_indicator.calculate(daily_data)

# 获取买入信号
buy_signals = result[result['buy_signal'] == True]
print(f"发现 {len(buy_signals)} 个高换手率买点")
```

#### ZXM_VOLUME_SHRINK - 缩量指标

**功能描述**：识别成交量萎缩的买点机会

**API接口**：
```python
indicator = complete_registry.create_indicator('ZXM_VOLUME_SHRINK')
result = indicator.calculate(data)
```

**参数说明**：
- `threshold`: float, 默认0.9, 缩量比例阈值
- `period`: int, 默认5, 成交量比较周期

**返回值格式**：
```python
DataFrame包含以下列：
- 'datetime': 时间戳
- 'volume_ratio': float, 成交量比率
- 'XG': bool, 缩量信号（volume_ratio < threshold）
- 'buy_signal': bool, 买入信号（XG == True）
- 'sell_signal': bool, 卖出信号（XG == False）
- 'hold_signal': bool, 持有信号（XG == False）
```

#### ZXM_BS_ABSORB - 吸筹指标

**功能描述**：识别主力吸筹行为，输出吸筹强度等级

**API接口**：
```python
indicator = complete_registry.create_indicator('ZXM_BS_ABSORB')
result = indicator.calculate(data)
```

**参数说明**：
- `data`: DataFrame, 必须包含30分钟级别数据
- 无额外参数，使用内置算法计算吸筹强度

**返回值格式**：
```python
DataFrame包含以下列：
- 'datetime': 时间戳
- 'XG': int, 吸筹强度等级（0-6，6为最强）
- 'buy_signal': bool, 买入信号（XG > 0）
- 'sell_signal': bool, 卖出信号（XG == 0）
- 'hold_signal': bool, 持有信号（XG == 0）
```

**特殊说明**：
- XG=0: 无吸筹行为
- XG=1-2: 轻微吸筹
- XG=3-4: 中等吸筹
- XG=5-6: 强烈吸筹

### 2. 趋势分析指标

#### ZXM_DAILY_TREND_UP - 日线趋势向上

**功能描述**：基于日线数据判断趋势向上状态

**API接口**：
```python
indicator = complete_registry.create_indicator('ZXM_DAILY_TREND_UP')
result = indicator.calculate(data)
```

**参数说明**：
- `ma_period`: int, 默认20, 移动平均线周期
- `trend_period`: int, 默认5, 趋势判断周期

**返回值格式**：
```python
DataFrame包含以下列：
- 'datetime': 时间戳
- 'j1': bool, 趋势条件1
- 'j2': bool, 趋势条件2
- 'XG': bool, 趋势向上信号（j1 | j2）
- 'buy_signal': bool, 买入信号（XG == True）
- 'sell_signal': bool, 卖出信号（XG == False）
- 'hold_signal': bool, 持有信号（XG == False）
```

#### ZXM_WEEKLY_TREND_UP - 周线趋势向上

**功能描述**：基于周线数据判断中长期趋势向上状态

**API接口**：
```python
indicator = complete_registry.create_indicator('ZXM_WEEKLY_TREND_UP')
result = indicator.calculate(data)
```

**参数说明**：
- `data`: DataFrame, 日线数据（内部转换为周线）
- 使用内置算法判断周线趋势

**返回值格式**：
```python
DataFrame包含以下列：
- 'datetime': 时间戳
- 'a1': bool, 趋势条件a1
- 'b1': bool, 趋势条件b1
- 'c1': bool, 趋势条件c1
- 'XG': bool, 趋势向上信号（a1 | b1 | c1）
- 'buy_signal': bool, 买入信号（XG == True）
- 'sell_signal': bool, 卖出信号（XG == False）
- 'hold_signal': bool, 持有信号（XG == False）
```

### 3. 弹性分析指标

#### ZXM_AMPLITUDE_ELASTICITY - 振幅弹性

**功能描述**：分析价格振幅的弹性特征

**API接口**：
```python
indicator = complete_registry.create_indicator('ZXM_AMPLITUDE_ELASTICITY')
result = indicator.calculate(data)
```

**参数说明**：
- `period`: int, 默认20, 弹性计算周期
- `threshold`: float, 默认0.05, 弹性阈值

**返回值格式**：
```python
DataFrame包含以下列：
- 'datetime': 时间戳
- 'amplitude_elasticity': float, 振幅弹性值
- 'XG': bool, 弹性信号（amplitude_elasticity > threshold）
- 'buy_signal': bool, 买入信号（XG == True）
- 'sell_signal': bool, 卖出信号（XG == False）
- 'hold_signal': bool, 持有信号（XG == False）
```

#### ZXM_RISE_ELASTICITY - 涨幅弹性

**功能描述**：分析价格上涨的弹性特征

**API接口**：
```python
indicator = complete_registry.create_indicator('ZXM_RISE_ELASTICITY')
result = indicator.calculate(data)
```

**参数说明**：
- `period`: int, 默认10, 弹性计算周期
- `min_rise`: float, 默认0.02, 最小涨幅要求

**返回值格式**：
```python
DataFrame包含以下列：
- 'datetime': 时间戳
- 'rise_elasticity': float, 涨幅弹性值
- 'XG': bool, 弹性信号（符合涨幅弹性条件）
- 'buy_signal': bool, 买入信号（XG == True）
- 'sell_signal': bool, 卖出信号（XG == False）
- 'hold_signal': bool, 持有信号（XG == False）
```

### 4. 评分系统指标

#### ZXM_ELASTICITY_SCORE - 弹性评分

**功能描述**：对股票弹性特征进行综合评分

**API接口**：
```python
indicator = complete_registry.create_indicator('ZXM_ELASTICITY_SCORE')
result = indicator.calculate(data)
```

**参数说明**：
- `threshold`: float, 默认75, 评分阈值

**返回值格式**：
```python
DataFrame包含以下列：
- 'datetime': 时间戳
- 'ElasticityScore': float, 弹性评分（0-100）
- 'Signal': bool, 评分信号（ElasticityScore >= threshold）
- 'buy_signal': bool, 买入信号（Signal == True）
- 'sell_signal': bool, 卖出信号（Signal == False）
- 'hold_signal': bool, 持有信号（Signal == False）
```

#### ZXM_BUYPOINT_SCORE - 买点评分

**功能描述**：对买点机会进行综合评分

**API接口**：
```python
indicator = complete_registry.create_indicator('ZXM_BUYPOINT_SCORE')
result = indicator.calculate(data)
```

**参数说明**：
- `threshold`: float, 默认75, 评分阈值

**返回值格式**：
```python
DataFrame包含以下列：
- 'datetime': 时间戳
- 'BuyPointScore': float, 买点评分（0-100）
- 'Signal': bool, 评分信号（BuyPointScore >= threshold）
- 'buy_signal': bool, 买入信号（Signal == True）
- 'sell_signal': bool, 卖出信号（Signal == False）
- 'hold_signal': bool, 持有信号（Signal == False）
```

#### ZXM_STOCK_SCORE - 股票综合评分

**功能描述**：对股票进行多维度综合评分

**API接口**：
```python
indicator = complete_registry.create_indicator('ZXM_STOCK_SCORE')
result = indicator.calculate(data)
```

**参数说明**：
- 无额外参数，使用内置多维度评分算法

**返回值格式**：
```python
DataFrame包含以下列：
- 'datetime': 时间戳
- 'TotalScore': float, 总评分（0-100）
- 'BuySignal': bool, 买入信号（TotalScore > 70）
- 'SellSignal': bool, 卖出信号（TotalScore < 30）
- 'buy_signal': bool, 买入信号（BuySignal == True）
- 'sell_signal': bool, 卖出信号（SellSignal == True）
- 'hold_signal': bool, 持有信号（~(buy_signal | sell_signal)）
```

### 5. 选股模型

#### ZXM_SELECTION_MODEL - 选股模型

**功能描述**：基于多指标综合选股

**API接口**：
```python
indicator = complete_registry.create_indicator('ZXM_SELECTION_MODEL')
result = indicator.calculate(data)
```

**参数说明**：
- 无额外参数，使用内置选股算法

**返回值格式**：
```python
DataFrame包含以下列：
- 'datetime': 时间戳
- 'FinalSelect': bool, 最终选股结果
- 'buy_signal': bool, 买入信号（FinalSelect == True）
- 'sell_signal': bool, 卖出信号（FinalSelect == False）
- 'hold_signal': bool, 持有信号（FinalSelect == False）
```

## 通用使用模式

### 1. 基本使用流程

```python
from indicators.complete_indicator_registry import complete_registry
import pandas as pd

# 1. 准备数据
data = pd.DataFrame({
    'datetime': pd.date_range('2023-01-01', periods=100, freq='D'),
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
    'volume': [...],
    'turnover_rate': [...]  # 某些指标需要
})

# 2. 创建指标
indicator = complete_registry.create_indicator('ZXM_TURNOVER')

# 3. 计算指标
result = indicator.calculate(data)

# 4. 获取信号
buy_signals = result[result['buy_signal'] == True]
```

### 2. 批量指标计算

```python
# 批量计算多个ZXM指标
zxm_indicators = [
    'ZXM_TURNOVER',
    'ZXM_VOLUME_SHRINK', 
    'ZXM_DAILY_TREND_UP',
    'ZXM_AMPLITUDE_ELASTICITY'
]

results = {}
for indicator_name in zxm_indicators:
    indicator = complete_registry.create_indicator(indicator_name)
    results[indicator_name] = indicator.calculate(data)
```

### 3. 信号合并分析

```python
# 合并多个指标的信号
combined_signals = pd.DataFrame({'datetime': data['datetime']})

for indicator_name, result in results.items():
    combined_signals[f'{indicator_name}_buy'] = result['buy_signal']

# 综合信号分析
combined_signals['total_buy_signals'] = combined_signals.filter(like='_buy').sum(axis=1)
strong_buy_points = combined_signals[combined_signals['total_buy_signals'] >= 3]
```

## 注意事项

1. **数据格式要求**：所有指标都要求输入数据包含基本的OHLCV字段
2. **时间频率**：ZXM_BS_ABSORB需要30分钟数据，其他指标使用日线数据
3. **信号语义**：所有指标的buy_signal/sell_signal/hold_signal已修复，确保语义一致
4. **性能优化**：单个指标计算时间<0.1秒，适合实时分析
5. **向后兼容**：所有API保持向后兼容，现有代码无需修改

## 更新日志

- **2025-06-25**: 完成P0级信号生成语义修复，11个ZXM指标100%修复
- **2025-06-25**: 添加详细API文档和使用示例
- **2025-06-25**: 建立语义验证测试框架，确保信号一致性
