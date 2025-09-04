# 技术标准快速开始指南

## 🚀 5分钟快速上手

### 1. 立即运行标准检查
```bash
# 检查当前代码的标准合规性
python scripts/validate_technical_standards.py

# 检查特定目录
python scripts/validate_technical_standards.py -d indicators/

# 预览自动修正建议
python scripts/validate_technical_standards.py --fix --dry-run

# 应用自动修正 (谨慎使用)
python scripts/validate_technical_standards.py --fix --no-dry-run
```

### 2. 常见问题快速修正

#### ❌ 错误的形态命名
```python
# 错误写法
patterns_df['RSI_OVERBOUGHT'] = rsi_values > 70
patterns_df['KDJ_GOLDEN_CROSS'] = k_cross_d

# ✅ 正确写法
from docs.standards.unified_technical_standards import StandardPatternNames

patterns_df[StandardPatternNames.OVERBOUGHT] = rsi_values > 70
patterns_df[StandardPatternNames.GOLDEN_CROSS] = k_cross_d
```

#### ❌ 错误的周期命名
```python
# 错误写法
period = "日线"
period = "15分钟"

# ✅ 正确写法
from docs.standards.unified_technical_standards import StandardPeriods

period = StandardPeriods.DAILY
period = StandardPeriods.MIN_15
```

#### ❌ 错误的指标命名
```python
# 错误写法
indicator_name = "macd"
indicator_name = "Rsi"

# ✅ 正确写法
from docs.standards.unified_technical_standards import StandardIndicatorNames

indicator_name = StandardIndicatorNames.MACD
indicator_name = StandardIndicatorNames.RSI
```

### 3. 策略配置标准格式

#### ✅ 标准策略配置示例
```json
{
  "strategy_id": "TECH_BREAKTHROUGH_001",
  "strategy_name": "技术突破策略",
  "conditions": {
    "pattern_conditions": [
      {
        "pattern": "GOLDEN_CROSS",
        "indicator": "MACD",
        "period": "daily",
        "weight": 0.4
      },
      {
        "pattern": "OVERSOLD",
        "indicator": "RSI", 
        "period": "daily",
        "weight": 0.3
      },
      {
        "pattern": "VOLUME_SURGE",
        "indicator": "VOL",
        "period": "daily", 
        "weight": 0.3
      }
    ]
  }
}
```

## 📋 开发检查清单

### 新功能开发前
- [ ] 确认使用的形态名称在StandardPatternNames中
- [ ] 确认使用的周期名称在StandardPeriods中
- [ ] 确认使用的指标名称在StandardIndicatorNames中
- [ ] 确认数据列名符合StandardDataColumns规范

### 代码提交前
- [ ] 运行 `python scripts/validate_technical_standards.py --strict`
- [ ] 修正所有错误和警告
- [ ] 确认所有测试通过
- [ ] 更新相关文档

### 代码审查时
- [ ] 检查命名标准合规性
- [ ] 检查接口标准一致性
- [ ] 检查配置格式正确性
- [ ] 检查跨模块兼容性

## 🔧 常用代码模板

### 1. 标准指标实现模板
```python
from docs.standards.unified_technical_standards import (
    StandardPatternNames, StandardPeriods, StandardIndicatorNames
)

class StandardIndicatorTemplate:
    def __init__(self, name: str):
        # 验证指标名称
        assert name in StandardIndicatorNames.get_all_indicators()
        self.name = name
    
    def get_patterns(self, data: pd.DataFrame, period: str) -> pd.DataFrame:
        # 验证周期
        assert StandardPeriods.validate_period(period)
        
        patterns_df = pd.DataFrame(index=data.index)
        
        # 使用标准形态名称
        patterns_df[StandardPatternNames.GOLDEN_CROSS] = self._detect_golden_cross(data)
        patterns_df[StandardPatternNames.DEATH_CROSS] = self._detect_death_cross(data)
        
        return patterns_df
```

### 2. 标准策略配置模板
```python
from docs.standards.unified_technical_standards import StandardConfiguration

class StandardStrategyTemplate:
    def __init__(self, config: Dict):
        # 验证配置格式
        self._validate_config(config)
        self.config = config
    
    def _validate_config(self, config: Dict):
        """验证策略配置是否符合标准"""
        # 使用StandardConfiguration进行验证
        pass
    
    def execute(self, stock_pool: List[str]) -> List[str]:
        """执行选股策略"""
        selected_stocks = []
        
        for condition in self.config['conditions']['pattern_conditions']:
            pattern = condition['pattern']
            indicator = condition['indicator'] 
            period = condition['period']
            
            # 确保使用标准名称
            assert StandardPatternNames.validate_pattern_name(pattern)
            assert StandardIndicatorNames.validate_indicator_name(indicator)
            assert StandardPeriods.validate_period(period)
            
            # 执行选股逻辑
            pass
        
        return selected_stocks
```

### 3. 标准数据处理模板
```python
from docs.standards.unified_technical_standards import StandardDataColumns, StandardDataTypes

class StandardDataProcessor:
    def process_stock_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """标准化股票数据处理"""
        
        # 标准化列名
        data = raw_data.copy()
        data.columns = [StandardDataColumns.OPEN, StandardDataColumns.HIGH, 
                       StandardDataColumns.LOW, StandardDataColumns.CLOSE, 
                       StandardDataColumns.VOLUME]
        
        # 标准化数据类型
        for col, dtype in StandardDataTypes.COLUMN_DTYPES.items():
            if col in data.columns:
                data[col] = data[col].astype(dtype)
        
        # 验证数据格式
        assert StandardDataTypes.validate_dataframe_dtypes(data)
        
        return data
```

## 🚨 紧急修正指南

### 如果系统出现命名冲突错误

#### 1. 快速诊断
```bash
# 运行诊断脚本
python scripts/validate_technical_standards.py --strict

# 查看详细错误信息
python scripts/validate_technical_standards.py -d . | grep "❌"
```

#### 2. 快速修正
```bash
# 自动修正常见问题
python scripts/validate_technical_standards.py --fix

# 手动修正特定文件
# 根据错误信息手动修正代码
```

#### 3. 验证修正
```bash
# 重新验证
python scripts/validate_technical_standards.py --strict

# 运行相关测试
python -m pytest tests/ -v
```

### 常见错误及解决方案

#### 错误1: 形态名称不匹配
```
❌ 错误: indicators/rsi.py: 非标准形态名称 'RSI_OVERBOUGHT'
✅ 解决: 改为 StandardPatternNames.OVERBOUGHT
```

#### 错误2: 周期名称不统一
```
❌ 错误: strategy/config.json: 非标准周期名称 '日线'
✅ 解决: 改为 "daily"
```

#### 错误3: 指标名称不规范
```
❌ 错误: analysis/analyzer.py: 非标准指标名称 'macd'
✅ 解决: 改为 StandardIndicatorNames.MACD
```

## 📞 技术支持

### 遇到问题时的处理流程
1. **查看错误信息**: 仔细阅读验证脚本的错误输出
2. **查阅标准文档**: 参考 `docs/standards/unified_technical_standards.md`
3. **使用自动修正**: 尝试使用自动修正功能
4. **手动修正**: 根据标准文档手动修正代码
5. **重新验证**: 确保修正后通过所有检查

### 联系方式
- **文档**: `docs/standards/unified_technical_standards.md`
- **脚本**: `scripts/validate_technical_standards.py`
- **示例**: 本文档中的代码模板

## 🎯 最佳实践

### 1. 开发习惯
- 始终使用标准常量而不是硬编码字符串
- 在IDE中配置代码片段，快速插入标准代码
- 定期运行标准验证脚本

### 2. 团队协作
- 代码审查时重点检查标准合规性
- 新成员入职时进行标准培训
- 定期更新和同步技术标准

### 3. 持续改进
- 收集标准使用中的问题和建议
- 定期评估和更新技术标准
- 保持标准文档的及时更新

---

**快速上手**: 5分钟掌握技术标准  
**核心目标**: 避免因命名不同导致功能无法运行  
**工具支持**: 自动验证和修正脚本  
**文档版本**: v1.0  
**创建时间**: 2025-09-04
