# 统一技术标准规范文档

## 📋 核心原则

### 统一性原则 ⚠️
**所有模块必须使用统一的技术标准，避免因命名不同导致功能无法运行**

- **形态名称**: 全系统统一的形态命名标准
- **周期定义**: 统一的时间周期标识和转换规则
- **指标名称**: 标准化的指标名称和参数定义
- **数据格式**: 统一的数据交换格式和接口标准

## 🎯 统一技术标准体系

### 1. 形态名称标准 (Pattern Names)

#### 1.1 标准形态名称枚举
```python
class StandardPatternNames:
    """统一的形态名称标准 - 所有模块必须使用"""
    
    # 趋势类形态
    GOLDEN_CROSS = "GOLDEN_CROSS"           # 金叉形态
    DEATH_CROSS = "DEATH_CROSS"             # 死叉形态
    CROSS_UP = "CROSS_UP"                   # 向上穿越
    CROSS_DOWN = "CROSS_DOWN"               # 向下穿越
    
    # 突破类形态
    UPPER_BREAK = "UPPER_BREAK"             # 上轨突破
    LOWER_BREAK = "LOWER_BREAK"             # 下轨突破
    RESISTANCE_BREAK = "RESISTANCE_BREAK"   # 阻力突破
    SUPPORT_BREAK = "SUPPORT_BREAK"         # 支撑突破
    
    # 超买超卖类形态
    OVERBOUGHT = "OVERBOUGHT"               # 超买形态
    OVERSOLD = "OVERSOLD"                   # 超卖形态
    NEUTRAL_ZONE = "NEUTRAL_ZONE"           # 中性区域
    OVERSOLD_RECOVERY = "OVERSOLD_RECOVERY" # 超卖恢复
    OVERBOUGHT_CORRECTION = "OVERBOUGHT_CORRECTION" # 超买修正
    
    # 背离类形态
    BULLISH_DIVERGENCE = "BULLISH_DIVERGENCE"   # 牛市背离
    BEARISH_DIVERGENCE = "BEARISH_DIVERGENCE"   # 熊市背离
    HIDDEN_BULLISH_DIV = "HIDDEN_BULLISH_DIV"   # 隐藏牛市背离
    HIDDEN_BEARISH_DIV = "HIDDEN_BEARISH_DIV"   # 隐藏熊市背离
    
    # K线形态类
    DOJI = "DOJI"                           # 十字星
    HAMMER = "HAMMER"                       # 锤子线
    SHOOTING_STAR = "SHOOTING_STAR"         # 流星线
    ENGULFING_BULLISH = "ENGULFING_BULLISH" # 看涨吞没
    ENGULFING_BEARISH = "ENGULFING_BEARISH" # 看跌吞没
    MORNING_STAR = "MORNING_STAR"           # 启明星
    EVENING_STAR = "EVENING_STAR"           # 黄昏星
    
    # 成交量形态类
    VOLUME_SURGE = "VOLUME_SURGE"           # 成交量放大
    VOLUME_SHRINK = "VOLUME_SHRINK"         # 成交量萎缩
    VOLUME_BREAKTHROUGH = "VOLUME_BREAKTHROUGH" # 成交量突破
    PRICE_VOLUME_CONFIRM = "PRICE_VOLUME_CONFIRM" # 量价确认
    
    # 波动类形态
    VOLATILITY_EXPANSION = "VOLATILITY_EXPANSION"   # 波动率扩张
    VOLATILITY_CONTRACTION = "VOLATILITY_CONTRACTION" # 波动率收缩
    TREND_ACCELERATION = "TREND_ACCELERATION"       # 趋势加速
    TREND_DECELERATION = "TREND_DECELERATION"       # 趋势减速

    @classmethod
    def get_all_patterns(cls) -> List[str]:
        """获取所有标准形态名称"""
        return [value for key, value in cls.__dict__.items() 
                if not key.startswith('_') and isinstance(value, str)]
    
    @classmethod
    def validate_pattern_name(cls, pattern_name: str) -> bool:
        """验证形态名称是否符合标准"""
        return pattern_name in cls.get_all_patterns()
```

#### 1.2 形态使用规范
```python
# ✅ 正确使用方式
pattern_config = {
    "pattern": StandardPatternNames.GOLDEN_CROSS,
    "indicator": "MACD",
    "period": "daily"
}

# ❌ 错误使用方式 - 禁止
pattern_config = {
    "pattern": "MACD_GOLDEN_CROSS",  # 包含指标名称
    "pattern": "GOLDEN_CROSS_DAILY", # 包含周期信息
    "pattern": "macd_golden_cross"   # 非标准命名
}
```

### 2. 周期标准 (Period Standards)

#### 2.1 标准周期定义
```python
class StandardPeriods:
    """统一的周期标准 - 所有模块必须使用"""
    
    # 分钟级周期
    MIN_15 = "15min"
    MIN_30 = "30min" 
    MIN_60 = "60min"
    
    # 日级周期
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    
    # 周期映射关系
    PERIOD_MAPPING = {
        "15分钟": MIN_15,
        "30分钟": MIN_30,
        "60分钟": MIN_60,
        "日线": DAILY,
        "周线": WEEKLY,
        "月线": MONTHLY,
        # 英文别名
        "15m": MIN_15,
        "30m": MIN_30,
        "1h": MIN_60,
        "1d": DAILY,
        "1w": WEEKLY,
        "1M": MONTHLY
    }
    
    # 周期优先级 (用于多周期分析)
    PERIOD_PRIORITY = {
        MIN_15: 1,
        MIN_30: 2,
        MIN_60: 3,
        DAILY: 4,
        WEEKLY: 5,
        MONTHLY: 6
    }
    
    @classmethod
    def normalize_period(cls, period: str) -> str:
        """标准化周期名称"""
        return cls.PERIOD_MAPPING.get(period, period)
    
    @classmethod
    def get_all_periods(cls) -> List[str]:
        """获取所有标准周期"""
        return [cls.MIN_15, cls.MIN_30, cls.MIN_60, cls.DAILY, cls.WEEKLY, cls.MONTHLY]
    
    @classmethod
    def validate_period(cls, period: str) -> bool:
        """验证周期是否符合标准"""
        normalized = cls.normalize_period(period)
        return normalized in cls.get_all_periods()
```

#### 2.2 周期转换规则
```python
class PeriodConverter:
    """周期数据转换规则"""
    
    @staticmethod
    def can_convert(from_period: str, to_period: str) -> bool:
        """判断是否可以进行周期转换"""
        from_priority = StandardPeriods.PERIOD_PRIORITY.get(from_period, 0)
        to_priority = StandardPeriods.PERIOD_PRIORITY.get(to_period, 0)
        return from_priority < to_priority
    
    @staticmethod
    def get_conversion_ratio(from_period: str, to_period: str) -> int:
        """获取周期转换比例"""
        conversion_map = {
            (StandardPeriods.MIN_15, StandardPeriods.MIN_30): 2,
            (StandardPeriods.MIN_15, StandardPeriods.MIN_60): 4,
            (StandardPeriods.MIN_30, StandardPeriods.MIN_60): 2,
            # 可以根据需要扩展更多转换关系
        }
        return conversion_map.get((from_period, to_period), 1)
```

### 3. 指标名称标准 (Indicator Names)

#### 3.1 标准指标名称枚举
```python
class StandardIndicatorNames:
    """统一的指标名称标准 - 所有模块必须使用"""
    
    # 基础技术指标 (17个)
    MA = "MA"                   # 移动平均线
    EMA = "EMA"                 # 指数移动平均线
    MACD = "MACD"               # 指数平滑移动平均线
    RSI = "RSI"                 # 相对强弱指标
    KDJ = "KDJ"                 # 随机指标
    BOLL = "BOLL"               # 布林带
    CCI = "CCI"                 # 顺势指标
    WR = "WR"                   # 威廉指标
    BIAS = "BIAS"               # 乖离率
    PSY = "PSY"                 # 心理线
    VR = "VR"                   # 成交量比率
    ARBR = "ARBR"               # 人气意愿指标
    DMA = "DMA"                 # 平行线差指标
    MTM = "MTM"                 # 动量指标
    ROC = "ROC"                 # 变动率指标
    OSC = "OSC"                 # 振荡量指标
    UOS = "UOS"                 # 终极指标
    
    # ZXM体系指标 (38个)
    ZXM_DAILY_MACD = "ZXM_DAILY_MACD"
    ZXM_BS_ABSORB = "ZXM_BS_ABSORB"
    ZXM_TREND_FOLLOW = "ZXM_TREND_FOLLOW"
    # ... 其他ZXM指标
    
    # 形态指标 (23个)
    CANDLESTICK = "CANDLESTICK" # K线形态
    VOL = "VOL"                 # 成交量
    PRICE = "PRICE"             # 价格形态
    # ... 其他形态指标
    
    # 评分指标 (4个)
    MACD_SCORE = "MACD_SCORE"
    RSI_SCORE = "RSI_SCORE"
    COMPREHENSIVE_SCORE = "COMPREHENSIVE_SCORE"
    PATTERN_SCORE = "PATTERN_SCORE"
    
    @classmethod
    def get_all_indicators(cls) -> List[str]:
        """获取所有标准指标名称"""
        return [value for key, value in cls.__dict__.items() 
                if not key.startswith('_') and isinstance(value, str)]
    
    @classmethod
    def validate_indicator_name(cls, indicator_name: str) -> bool:
        """验证指标名称是否符合标准"""
        return indicator_name in cls.get_all_indicators()
```

#### 3.2 指标分类标准
```python
class IndicatorCategories:
    """指标分类标准"""
    
    TREND = "TREND"             # 趋势类指标
    MOMENTUM = "MOMENTUM"       # 动量类指标
    VOLUME = "VOLUME"           # 成交量类指标
    VOLATILITY = "VOLATILITY"   # 波动率类指标
    PATTERN = "PATTERN"         # 形态类指标
    COMPOSITE = "COMPOSITE"     # 复合类指标
    
    INDICATOR_CATEGORY_MAP = {
        StandardIndicatorNames.MA: TREND,
        StandardIndicatorNames.EMA: TREND,
        StandardIndicatorNames.MACD: MOMENTUM,
        StandardIndicatorNames.RSI: MOMENTUM,
        StandardIndicatorNames.KDJ: MOMENTUM,
        StandardIndicatorNames.BOLL: VOLATILITY,
        StandardIndicatorNames.VOL: VOLUME,
        StandardIndicatorNames.CANDLESTICK: PATTERN,
        # ... 更多映射关系
    }
```

### 4. 数据格式标准 (Data Format Standards)

#### 4.1 标准数据列名
```python
class StandardDataColumns:
    """统一的数据列名标准"""
    
    # 基础OHLCV数据
    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"
    VOLUME = "volume"
    
    # 扩展数据列
    DATE = "date"
    DATETIME = "datetime"
    CODE = "code"
    NAME = "name"
    TURNOVER_RATE = "turnover_rate"
    PRICE_CHANGE = "price_change"
    PRICE_RANGE = "price_range"
    INDUSTRY = "industry"
    
    # 指标计算结果列命名规范
    @staticmethod
    def get_indicator_column_name(indicator: str, field: str, period: str = None) -> str:
        """生成标准的指标列名"""
        if period:
            return f"{indicator.lower()}_{field}_{period}"
        else:
            return f"{indicator.lower()}_{field}"
    
    # 形态识别结果列命名规范
    @staticmethod
    def get_pattern_column_name(pattern: str, indicator: str = None) -> str:
        """生成标准的形态列名"""
        if indicator:
            return f"{pattern}_{indicator}"
        else:
            return pattern
```

#### 4.2 标准数据类型
```python
class StandardDataTypes:
    """统一的数据类型标准"""
    
    # 基础数据类型映射
    COLUMN_DTYPES = {
        StandardDataColumns.OPEN: 'float64',
        StandardDataColumns.HIGH: 'float64',
        StandardDataColumns.LOW: 'float64',
        StandardDataColumns.CLOSE: 'float64',
        StandardDataColumns.VOLUME: 'int64',
        StandardDataColumns.DATE: 'datetime64[ns]',
        StandardDataColumns.CODE: 'string',
        StandardDataColumns.NAME: 'string',
        StandardDataColumns.TURNOVER_RATE: 'float64',
        StandardDataColumns.PRICE_CHANGE: 'float64',
        StandardDataColumns.PRICE_RANGE: 'float64',
        StandardDataColumns.INDUSTRY: 'string'
    }
    
    @staticmethod
    def validate_dataframe_dtypes(df: pd.DataFrame) -> bool:
        """验证DataFrame的数据类型是否符合标准"""
        for col, expected_dtype in StandardDataTypes.COLUMN_DTYPES.items():
            if col in df.columns:
                if not df[col].dtype.name.startswith(expected_dtype.split('[')[0]):
                    return False
        return True
```

### 5. 接口标准 (Interface Standards)

#### 5.1 模块间接口标准
```python
class ModuleInterface:
    """模块间接口标准"""
    
    @staticmethod
    def calculate_indicator(indicator_name: str, data: pd.DataFrame, 
                          period: str, **params) -> pd.DataFrame:
        """标准指标计算接口"""
        # 验证输入参数
        assert StandardIndicatorNames.validate_indicator_name(indicator_name)
        assert StandardPeriods.validate_period(period)
        
        # 执行计算逻辑
        pass
    
    @staticmethod
    def detect_pattern(pattern_name: str, indicator_name: str, 
                      data: pd.DataFrame, period: str, **params) -> pd.Series:
        """标准形态识别接口"""
        # 验证输入参数
        assert StandardPatternNames.validate_pattern_name(pattern_name)
        assert StandardIndicatorNames.validate_indicator_name(indicator_name)
        assert StandardPeriods.validate_period(period)
        
        # 执行识别逻辑
        pass
```

#### 5.2 API响应格式标准
```python
class StandardAPIResponse:
    """统一的API响应格式"""
    
    @staticmethod
    def success_response(data: Any, message: str = "Success") -> Dict:
        """成功响应格式"""
        return {
            "status": "success",
            "code": 200,
            "message": message,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    def error_response(error_code: int, message: str, details: str = None) -> Dict:
        """错误响应格式"""
        return {
            "status": "error",
            "code": error_code,
            "message": message,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
```

### 6. 配置管理标准 (Configuration Standards)

#### 6.1 统一配置格式
```python
class StandardConfiguration:
    """统一的配置格式标准"""
    
    # 策略配置标准格式
    STRATEGY_CONFIG_SCHEMA = {
        "strategy_id": str,
        "strategy_name": str,
        "description": str,
        "conditions": {
            "pattern_conditions": [
                {
                    "pattern": str,      # 必须使用StandardPatternNames
                    "indicator": str,    # 必须使用StandardIndicatorNames
                    "period": str,       # 必须使用StandardPeriods
                    "parameters": dict,
                    "weight": float
                }
            ]
        },
        "filters": dict,
        "scoring": dict
    }
    
    # 指标配置标准格式
    INDICATOR_CONFIG_SCHEMA = {
        "indicator_name": str,   # 必须使用StandardIndicatorNames
        "parameters": dict,
        "periods": list,         # 必须使用StandardPeriods
        "output_columns": list
    }
```

## 🔧 实施规范

### 1. 开发规范
- **命名检查**: 所有模块开发前必须进行命名标准检查
- **接口验证**: 模块间接口必须符合标准接口规范
- **数据验证**: 数据交换必须通过格式和类型验证
- **配置验证**: 所有配置文件必须符合标准配置格式

### 2. 测试规范
- **标准验证测试**: 每个模块必须包含标准验证测试
- **接口兼容性测试**: 模块间接口的兼容性测试
- **数据一致性测试**: 跨模块数据一致性测试
- **配置有效性测试**: 配置文件的有效性测试

### 3. 维护规范
- **版本控制**: 技术标准的版本控制和变更管理
- **向后兼容**: 标准更新时的向后兼容性保证
- **文档同步**: 标准变更时的文档同步更新
- **培训推广**: 新标准的团队培训和推广

## 📊 验收标准

### 1. 命名一致性验收
- [ ] 所有形态名称使用StandardPatternNames
- [ ] 所有周期使用StandardPeriods
- [ ] 所有指标名称使用StandardIndicatorNames
- [ ] 所有数据列名使用StandardDataColumns

### 2. 接口标准化验收
- [ ] 模块间接口符合ModuleInterface标准
- [ ] API响应格式符合StandardAPIResponse
- [ ] 数据类型符合StandardDataTypes
- [ ] 配置格式符合StandardConfiguration

### 3. 功能兼容性验收
- [ ] 跨模块功能调用正常
- [ ] 数据交换无格式错误
- [ ] 配置文件加载成功
- [ ] 标准验证测试通过

## 🚀 实施指南

### 1. 现有代码修正清单

#### 需要修正的指标形态命名
```python
# RSI指标修正
"RSI_OVERBOUGHT" → StandardPatternNames.OVERBOUGHT
"RSI_OVERSOLD" → StandardPatternNames.OVERSOLD
"RSI_BULLISH_DIVERGENCE" → StandardPatternNames.BULLISH_DIVERGENCE

# KDJ指标修正
"KDJ_GOLDEN_CROSS" → StandardPatternNames.GOLDEN_CROSS
"KDJ_DEATH_CROSS" → StandardPatternNames.DEATH_CROSS
"KDJ_OVERBOUGHT" → StandardPatternNames.OVERBOUGHT
"KDJ_OVERSOLD" → StandardPatternNames.OVERSOLD

# MACD指标 (已符合标准，无需修正)
"GOLDEN_CROSS" ✅
"DEATH_CROSS" ✅
"BULLISH_DIVERGENCE" ✅
```

#### 周期标准化修正
```python
# 统一周期命名
"15分钟" → StandardPeriods.MIN_15
"30分钟" → StandardPeriods.MIN_30
"60分钟" → StandardPeriods.MIN_60
"日线" → StandardPeriods.DAILY
"周线" → StandardPeriods.WEEKLY
"月线" → StandardPeriods.MONTHLY
```

### 2. 标准验证工具

#### 2.1 命名标准验证器
```python
class StandardValidator:
    """技术标准验证器"""

    @staticmethod
    def validate_strategy_config(config: Dict) -> Tuple[bool, List[str]]:
        """验证策略配置是否符合标准"""
        errors = []

        # 验证形态名称
        for condition in config.get('conditions', {}).get('pattern_conditions', []):
            pattern = condition.get('pattern')
            if not StandardPatternNames.validate_pattern_name(pattern):
                errors.append(f"非标准形态名称: {pattern}")

            indicator = condition.get('indicator')
            if not StandardIndicatorNames.validate_indicator_name(indicator):
                errors.append(f"非标准指标名称: {indicator}")

            period = condition.get('period')
            if not StandardPeriods.validate_period(period):
                errors.append(f"非标准周期名称: {period}")

        return len(errors) == 0, errors

    @staticmethod
    def validate_indicator_result(result: pd.DataFrame, indicator: str) -> Tuple[bool, List[str]]:
        """验证指标计算结果是否符合标准"""
        errors = []

        # 验证数据类型
        if not StandardDataTypes.validate_dataframe_dtypes(result):
            errors.append("数据类型不符合标准")

        # 验证列名格式
        for col in result.columns:
            if not col.startswith(indicator.lower()):
                errors.append(f"列名格式不符合标准: {col}")

        return len(errors) == 0, errors
```

#### 2.2 自动修正工具
```python
class StandardCorrector:
    """技术标准自动修正工具"""

    @staticmethod
    def correct_pattern_names(pattern_dict: Dict[str, Any]) -> Dict[str, Any]:
        """自动修正形态名称"""
        correction_map = {
            "RSI_OVERBOUGHT": StandardPatternNames.OVERBOUGHT,
            "RSI_OVERSOLD": StandardPatternNames.OVERSOLD,
            "KDJ_GOLDEN_CROSS": StandardPatternNames.GOLDEN_CROSS,
            "KDJ_DEATH_CROSS": StandardPatternNames.DEATH_CROSS,
            # 更多修正映射...
        }

        corrected = {}
        for key, value in pattern_dict.items():
            corrected_key = correction_map.get(key, key)
            corrected[corrected_key] = value

        return corrected

    @staticmethod
    def correct_period_names(period_list: List[str]) -> List[str]:
        """自动修正周期名称"""
        return [StandardPeriods.normalize_period(period) for period in period_list]
```

### 3. 开发工作流集成

#### 3.1 代码提交前检查
```bash
# Git pre-commit hook 示例
#!/bin/bash
echo "检查技术标准合规性..."

# 运行标准验证
python scripts/validate_standards.py

if [ $? -ne 0 ]; then
    echo "❌ 技术标准验证失败，请修正后再提交"
    exit 1
fi

echo "✅ 技术标准验证通过"
```

#### 3.2 CI/CD集成
```yaml
# GitHub Actions 示例
name: Technical Standards Check
on: [push, pull_request]

jobs:
  standards-check:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Setup Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.8
    - name: Install dependencies
      run: pip install -r requirements.txt
    - name: Run standards validation
      run: python scripts/validate_standards.py --strict
```

### 4. 团队协作规范

#### 4.1 代码审查检查清单
- [ ] 形态名称使用StandardPatternNames
- [ ] 周期名称使用StandardPeriods
- [ ] 指标名称使用StandardIndicatorNames
- [ ] 数据列名使用StandardDataColumns
- [ ] 接口符合ModuleInterface标准
- [ ] 配置格式符合StandardConfiguration

#### 4.2 新功能开发流程
1. **设计阶段**: 确认使用的标准名称和格式
2. **开发阶段**: 使用标准验证工具进行实时检查
3. **测试阶段**: 运行标准兼容性测试
4. **代码审查**: 重点检查标准合规性
5. **集成测试**: 验证跨模块兼容性

---

**文档定位**: 统一技术标准规范
**适用范围**: 所有四大模块
**强制执行**: ⚠️ 避免因命名不同导致功能无法运行
**文档版本**: v1.0
**创建时间**: 2025-09-04
