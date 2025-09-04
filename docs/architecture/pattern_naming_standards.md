# 指标形态命名规范文档

## 📋 核心原则

### 形态命名一致性原则 ⚠️
**策略中的形态名称必须与指标注册的形态名称保持完全一致**

- **错误示例**: `MACD_DAILY_GOLDEN_CROSS` (包含了周期信息)
- **正确示例**: `GOLDEN_CROSS` (纯形态名称) + `indicator="MACD"` + `period="daily"`

### 设计理念
- **形态名称**: 描述技术形态的本质特征，不包含指标名和周期信息
- **指标属性**: 指定形态所属的技术指标
- **周期属性**: 指定形态所在的时间周期
- **参数属性**: 指定形态识别的具体参数

## 🎯 标准形态命名体系

### 1. 趋势类形态
```python
# 金叉死叉类
GOLDEN_CROSS          # 金叉形态 (适用于MACD、KDJ、MA等)
DEATH_CROSS           # 死叉形态
CROSS_UP              # 向上穿越
CROSS_DOWN            # 向下穿越

# 突破类
UPPER_BREAK           # 上轨突破 (适用于BOLL、通道等)
LOWER_BREAK           # 下轨突破  
RESISTANCE_BREAK      # 阻力突破
SUPPORT_BREAK         # 支撑突破
```

### 2. 超买超卖类形态
```python
# 超买超卖
OVERBOUGHT            # 超买形态 (适用于RSI、KDJ等)
OVERSOLD              # 超卖形态
NEUTRAL_ZONE          # 中性区域

# 恢复类
OVERSOLD_RECOVERY     # 超卖恢复
OVERBOUGHT_CORRECTION # 超买修正
```

### 3. 背离类形态
```python
# 背离形态
BULLISH_DIVERGENCE    # 牛市背离 (适用于MACD、RSI等)
BEARISH_DIVERGENCE    # 熊市背离
HIDDEN_BULLISH_DIV    # 隐藏牛市背离
HIDDEN_BEARISH_DIV    # 隐藏熊市背离
```

### 4. K线形态类
```python
# 经典K线形态
DOJI                  # 十字星
HAMMER                # 锤子线
SHOOTING_STAR         # 流星线
ENGULFING_BULLISH     # 看涨吞没
ENGULFING_BEARISH     # 看跌吞没
MORNING_STAR          # 启明星
EVENING_STAR          # 黄昏星
```

### 5. 成交量形态类
```python
# 成交量形态
VOLUME_SURGE          # 成交量放大
VOLUME_SHRINK         # 成交量萎缩
VOLUME_BREAKTHROUGH   # 成交量突破
PRICE_VOLUME_CONFIRM  # 量价确认
```

### 6. 波动类形态
```python
# 波动形态
VOLATILITY_EXPANSION  # 波动率扩张
VOLATILITY_CONTRACTION # 波动率收缩
TREND_ACCELERATION    # 趋势加速
TREND_DECELERATION    # 趋势减速
```

## 📊 形态使用规范

### 在指标注册中的定义
```python
class MACDIndicator(BaseIndicator):
    def get_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """返回MACD指标的形态识别结果"""
        patterns = {}
        
        # 金叉形态 - 使用标准形态名称
        patterns['GOLDEN_CROSS'] = self._detect_golden_cross(data)
        
        # 死叉形态
        patterns['DEATH_CROSS'] = self._detect_death_cross(data)
        
        # 牛市背离
        patterns['BULLISH_DIVERGENCE'] = self._detect_bullish_divergence(data)
        
        # 熊市背离  
        patterns['BEARISH_DIVERGENCE'] = self._detect_bearish_divergence(data)
        
        return patterns
```

### 在选股策略中的使用
```python
# 方式1: 函数式语法
GOLDEN_CROSS(indicator="MACD", period="daily")
OVERSOLD_RECOVERY(indicator="RSI", period="daily")

# 方式2: 点语法 (推荐)
GOLDEN_CROSS.MACD.DAILY
OVERSOLD_RECOVERY.RSI.DAILY
UPPER_BREAK.BOLL.WEEKLY

# 方式3: JSON配置
{
  "pattern": "GOLDEN_CROSS",
  "indicator": "MACD", 
  "period": "daily",
  "parameters": {}
}
```

### 多周期形态组合
```python
# 多周期共振 - 相同形态不同周期
GOLDEN_CROSS.MACD.15MIN AND GOLDEN_CROSS.MACD.30MIN AND GOLDEN_CROSS.MACD.DAILY

# 多指标确认 - 相同形态不同指标
OVERSOLD_RECOVERY.RSI.DAILY AND OVERSOLD_RECOVERY.KDJ.DAILY

# 复合形态 - 不同形态组合
GOLDEN_CROSS.MACD.DAILY AND VOLUME_SURGE.VOL.DAILY AND UPPER_BREAK.BOLL.DAILY
```

## 🔧 形态参数化配置

### 参数化形态定义
```python
# 带参数的形态配置
{
  "pattern": "NEUTRAL_ZONE",
  "indicator": "RSI",
  "period": "daily", 
  "parameters": {
    "lower_bound": 30,
    "upper_bound": 70
  }
}

{
  "pattern": "VOLUME_SURGE",
  "indicator": "VOL",
  "period": "daily",
  "parameters": {
    "ratio_threshold": 2.0,
    "ma_period": 5
  }
}

{
  "pattern": "CROSS_UP",
  "indicator": "MA",
  "period": "daily",
  "parameters": {
    "fast_period": 5,
    "slow_period": 20
  }
}
```

### 动态参数形态
```python
# 自适应参数
{
  "pattern": "ADAPTIVE_OVERSOLD",
  "indicator": "RSI", 
  "period": "daily",
  "parameters": {
    "base_threshold": 30,
    "volatility_adjustment": true,
    "market_regime_factor": 0.1
  }
}
```

## 📋 形态注册检查清单

### 指标开发者检查清单
- [ ] 形态名称使用标准命名规范
- [ ] 形态名称不包含指标名称
- [ ] 形态名称不包含周期信息
- [ ] 形态名称描述技术特征本质
- [ ] 形态参数可配置化
- [ ] 形态识别结果标准化

### 策略配置者检查清单
- [ ] 使用指标注册的标准形态名称
- [ ] 明确指定indicator和period属性
- [ ] 参数配置与指标定义一致
- [ ] 多周期组合逻辑清晰
- [ ] 形态权重分配合理

## 🎯 常见错误和修正

### ❌ 错误的形态命名
```python
# 错误: 包含指标名称
"MACD_GOLDEN_CROSS"
"RSI_OVERSOLD" 
"KDJ_DEATH_CROSS"

# 错误: 包含周期信息
"GOLDEN_CROSS_DAILY"
"OVERSOLD_WEEKLY"
"BREAK_15MIN"

# 错误: 过于具体的命名
"MACD_DAILY_DIF_CROSS_UP_DEA"
"RSI_DAILY_BELOW_30_RECOVERY"
```

### ✅ 正确的形态命名
```python
# 正确: 纯形态名称
"GOLDEN_CROSS"
"OVERSOLD" 
"DEATH_CROSS"
"UPPER_BREAK"
"BULLISH_DIVERGENCE"

# 配合属性使用
GOLDEN_CROSS.MACD.DAILY
OVERSOLD.RSI.WEEKLY  
UPPER_BREAK.BOLL.15MIN
```

### 修正示例
```python
# 修正前
strategy_formula = "MACD_DAILY_GOLDEN_CROSS AND RSI_DAILY_OVERSOLD"

# 修正后  
strategy_formula = "GOLDEN_CROSS.MACD.DAILY AND OVERSOLD.RSI.DAILY"

# 或者使用JSON配置
conditions = [
  {
    "pattern": "GOLDEN_CROSS",
    "indicator": "MACD",
    "period": "daily"
  },
  {
    "pattern": "OVERSOLD", 
    "indicator": "RSI",
    "period": "daily"
  }
]
```

## 🎉 规范的价值

### 1. 一致性保证
- 策略配置与指标实现完全对应
- 避免形态名称不匹配的错误
- 提高系统的可维护性

### 2. 灵活性提升
- 同一形态可应用于多个指标
- 同一形态可应用于多个周期
- 参数化配置提高适应性

### 3. 可扩展性
- 新增指标时复用现有形态名称
- 新增形态时遵循统一规范
- 便于形态库的标准化建设

---

**形态命名规范是选股策略配置的基础，必须严格遵循！**

---

**文档类型**: 形态命名规范文档  
**适用范围**: 指标开发、策略配置  
**文档版本**: v1.0  
**创建时间**: 2025-09-04
