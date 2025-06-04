# PatternRegistry 使用指南

`PatternRegistry` 是一个形态注册表类，用于管理所有技术形态的唯一标识和相关信息。它实现了单例模式，确保整个应用程序中只有一个形态注册表实例。

## 基本用法

### 导入必要的类

```python
from indicators.pattern_registry import PatternRegistry, PatternType, PatternStrength, PatternInfo
```

### 获取注册表实例

```python
registry = PatternRegistry()  # 返回单例实例
```

### 注册单个形态

```python
registry.register(
    pattern_id="KDJ_GOLDEN_CROSS",
    display_name="KDJ金叉",
    indicator_id="KDJ",
    pattern_type=PatternType.BULLISH,
    default_strength=PatternStrength.STRONG
)
```

### 批量注册形态

```python
patterns = [
    PatternInfo(
        pattern_id="MACD_GOLDEN_CROSS",
        display_name="MACD金叉",
        indicator_id="MACD",
        pattern_type=PatternType.BULLISH,
        score_impact=10
    ),
    PatternInfo(
        pattern_id="MACD_DEATH_CROSS",
        display_name="MACD死叉",
        indicator_id="MACD",
        pattern_type=PatternType.BEARISH,
        score_impact=-10
    )
]

registry.register_patterns_batch(patterns)
```

### 查询形态信息

```python
# 按形态ID查询
pattern_info = registry.get_pattern("KDJ_GOLDEN_CROSS")
print(pattern_info)

# 获取形态显示名称
display_name = PatternRegistry.get_display_name("KDJ_GOLDEN_CROSS")
print(display_name)

# 按指标查询形态
kdj_patterns = registry.get_patterns_by_indicator("KDJ")
print(kdj_patterns)
```

### 计算组合评分影响

```python
patterns = ["MACD_GOLDEN_CROSS", "KDJ_GOLDEN_CROSS"]
score_impact = PatternRegistry.calculate_combined_score_impact(patterns)
print(f"组合评分影响: {score_impact}")
```

### 控制形态注册行为

```python
# 设置是否允许覆盖已存在的形态（全局设置）
PatternRegistry.set_allow_override(True)

# 在注册时指定是否允许覆盖（局部设置）
registry.register(
    pattern_id="KDJ_GOLDEN_CROSS",
    display_name="KDJ金叉更新版",
    indicator_id="KDJ",
    pattern_type=PatternType.BULLISH,
    _allow_override=True
)
```

## 从指标导入形态

如果指标类实现了 `_registered_patterns` 属性，可以直接从指标实例导入形态：

```python
from indicators.kdj import KDJIndicator

kdj = KDJIndicator()
PatternRegistry.import_patterns_from_indicator(kdj)
```

## 自动注册所有指标形态

```python
from indicators.factory import IndicatorFactory

# 自动注册所有指标的形态
PatternRegistry.auto_register_from_indicators(IndicatorFactory())
```

## 形态类型和强度

### 形态类型

```python
PatternType.BULLISH   # 看涨形态
PatternType.BEARISH   # 看跌形态
PatternType.NEUTRAL   # 中性形态
```

### 形态强度

```python
PatternStrength.VERY_WEAK    # 极弱（值为1）
PatternStrength.WEAK         # 弱（值为2）
PatternStrength.MEDIUM       # 中等（值为3）
PatternStrength.STRONG       # 强（值为4）
PatternStrength.VERY_STRONG  # 极强（值为5）
```

## 完整示例

```python
from indicators.pattern_registry import PatternRegistry, PatternType, PatternStrength, PatternInfo

# 获取单例实例
registry = PatternRegistry()

# 注册多个形态
patterns = [
    PatternInfo("BOLL_SQUEEZE", "布林带挤压", "BOLL", PatternType.NEUTRAL, score_impact=0),
    PatternInfo("BOLL_BREAKOUT_UP", "布林带向上突破", "BOLL", PatternType.BULLISH, score_impact=15),
    PatternInfo("BOLL_BREAKOUT_DOWN", "布林带向下突破", "BOLL", PatternType.BEARISH, score_impact=-15)
]

registry.register_patterns_batch(patterns)

# 查询指标的所有形态
boll_patterns = registry.get_patterns_by_indicator("BOLL")
print(f"BOLL指标的所有形态: {boll_patterns}")

# 获取特定形态的详细信息
breakout_pattern = registry.get_pattern("BOLL_BREAKOUT_UP")
print(f"向上突破形态详情: {breakout_pattern}")

# 计算组合评分影响
combined_impact = PatternRegistry.calculate_combined_score_impact(["BOLL_BREAKOUT_UP", "BOLL_SQUEEZE"])
print(f"组合评分影响: {combined_impact}")
``` 