# 技术指标开发规范

## 概述

本文档定义了技术分析系统中技术指标开发的标准规范，确保所有指标具有一致的接口、可靠的质量和正确的信号生成语义。

## 核心原则

### 1. 信号生成语义一致性原则

**核心要求**：指标的输出字段与信号生成逻辑必须100%语义一致

**常见指标类型及其信号生成模式**：

#### 计数型指标
- **特征**：输出整数计数值（如XG=0,1,2,3...）
- **信号逻辑**：`buy_signal = (count_value > threshold)`
- **示例**：ZXM_BS_ABSORB的XG字段（0-6的吸筹强度）

```python
# 正确的计数型信号生成
result.loc[:, 'buy_signal'] = result["XG"] > 0
result.loc[:, 'sell_signal'] = result["XG"] == 0
result.loc[:, 'hold_signal'] = result["XG"] == 0
```

#### 状态型指标
- **特征**：输出布尔值或枚举状态（如XG=True/False）
- **信号逻辑**：`buy_signal = (state == target_state)`
- **示例**：ZXM_TURNOVER的XG字段（换手率>阈值）

```python
# 正确的状态型信号生成
result.loc[:, 'buy_signal'] = result["XG"] == True
result.loc[:, 'sell_signal'] = result["XG"] == False
result.loc[:, 'hold_signal'] = result["XG"] == False
```

#### 等级型指标
- **特征**：输出评分或等级值（如Score=0-100）
- **信号逻辑**：`buy_signal = (score >= threshold)`
- **示例**：ZXM_ELASTICITY_SCORE的Signal字段

```python
# 正确的等级型信号生成
result.loc[:, 'buy_signal'] = result["Signal"] == True
result.loc[:, 'sell_signal'] = result["Signal"] == False
result.loc[:, 'hold_signal'] = result["Signal"] == False
```

#### 复合型指标
- **特征**：输出多个信号字段（如BuySignal, SellSignal）
- **信号逻辑**：直接使用专用信号字段
- **示例**：ZXM_STOCK_SCORE的BuySignal/SellSignal字段

```python
# 正确的复合型信号生成
result.loc[:, 'buy_signal'] = result["BuySignal"] == True
result.loc[:, 'sell_signal'] = result["SellSignal"] == True
result.loc[:, 'hold_signal'] = ~(result['buy_signal'] | result['sell_signal'])
```

### 2. 标准化接口原则

所有技术指标必须遵循统一的接口规范：

```python
class StandardIndicator(BaseIndicator):
    def __init__(self, **kwargs):
        super().__init__()
        # 参数初始化
    
    def _calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        # 核心计算逻辑
        result = self._perform_calculation(data)
        
        # 添加形态识别和通用信号生成
        result = self.add_pattern_detection(result)
        result = self.add_signal_generation(result)
        
        # 重写信号生成逻辑（如果需要）
        if self._needs_custom_signal_logic():
            result = self._apply_custom_signal_logic(result)
        
        return result
    
    def _needs_custom_signal_logic(self) -> bool:
        # 判断是否需要自定义信号逻辑
        return True  # 对于计数型、状态型、等级型指标
    
    def _apply_custom_signal_logic(self, result: pd.DataFrame) -> pd.DataFrame:
        # 应用自定义信号逻辑
        # 根据指标类型实现相应的信号生成逻辑
        pass
```

### 3. 数据格式标准化原则

#### 输入数据格式
```python
# 标准输入数据格式
required_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
optional_columns = ['turnover_rate', 'amount']  # 根据指标需求

# 数据类型要求
data_types = {
    'datetime': 'datetime64[ns]',
    'open': 'float64',
    'high': 'float64', 
    'low': 'float64',
    'close': 'float64',
    'volume': 'int64'
}
```

#### 输出数据格式
```python
# 标准输出数据格式
standard_output_columns = [
    'datetime',           # 时间戳
    # 指标特有字段（如XG, Score等）
    'buy_signal',         # bool, 买入信号
    'sell_signal',        # bool, 卖出信号  
    'hold_signal',        # bool, 持有信号
    'pattern_detected',   # bool, 形态识别结果
    'signal_strength'     # float, 信号强度
]
```

## 开发流程

### 1. 需求分析阶段

#### 指标类型识别
- 确定指标属于哪种类型（计数型、状态型、等级型、复合型）
- 明确指标的业务语义和预期输出
- 设计合适的信号生成逻辑

#### 参数设计
```python
# 参数设计示例
class NewIndicator(BaseIndicator):
    def __init__(self, 
                 period: int = 20,           # 计算周期
                 threshold: float = 0.7,     # 阈值参数
                 method: str = 'sma',        # 计算方法
                 **kwargs):
        super().__init__()
        self.period = period
        self.threshold = threshold
        self.method = method
```

### 2. 实现阶段

#### 核心计算逻辑
```python
def _calculate(self, data: pd.DataFrame) -> pd.DataFrame:
    # 1. 数据验证
    self._validate_input_data(data)
    
    # 2. 核心计算
    result = data.copy()
    
    # 具体计算逻辑
    result['indicator_value'] = self._compute_indicator_value(data)
    result['XG'] = self._generate_primary_signal(result)
    
    # 3. 添加标准字段
    result = self.add_pattern_detection(result)
    result = self.add_signal_generation(result)
    
    # 4. 自定义信号逻辑（关键步骤）
    result = self._apply_custom_signal_logic(result)
    
    return result

def _apply_custom_signal_logic(self, result: pd.DataFrame) -> pd.DataFrame:
    # 根据指标类型实现正确的信号生成逻辑
    if self.indicator_type == 'state':
        result.loc[:, 'buy_signal'] = result["XG"] == True
        result.loc[:, 'sell_signal'] = result["XG"] == False
        result.loc[:, 'hold_signal'] = result["XG"] == False
    elif self.indicator_type == 'count':
        result.loc[:, 'buy_signal'] = result["XG"] > 0
        result.loc[:, 'sell_signal'] = result["XG"] == 0
        result.loc[:, 'hold_signal'] = result["XG"] == 0
    # 其他类型...
    
    return result
```

#### 边界条件处理
```python
def _validate_input_data(self, data: pd.DataFrame) -> None:
    # 数据完整性检查
    if data.empty:
        raise ValueError("输入数据不能为空")
    
    # 数据量检查
    if len(data) < self.period:
        logger.warning(f"数据量不足，需要至少{self.period}个数据点")
    
    # NaN值处理
    if data[['open', 'high', 'low', 'close']].isnull().any().any():
        logger.warning("发现NaN值，将进行前向填充")
        data.fillna(method='ffill', inplace=True)

def _handle_edge_cases(self, result: pd.DataFrame) -> pd.DataFrame:
    # 处理无限值
    numeric_columns = result.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        result[col] = result[col].replace([np.inf, -np.inf], np.nan)
    
    # 确保信号字段的数据类型
    signal_columns = ['buy_signal', 'sell_signal', 'hold_signal']
    for col in signal_columns:
        if col in result.columns:
            result[col] = result[col].astype(bool)
    
    return result
```

### 3. 测试阶段

#### 单元测试
```python
class TestNewIndicator(unittest.TestCase):
    def setUp(self):
        self.indicator = NewIndicator()
        self.test_data = self._generate_test_data()
    
    def test_calculation_accuracy(self):
        # 测试计算准确性
        result = self.indicator.calculate(self.test_data)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0)
    
    def test_signal_consistency(self):
        # 测试信号一致性
        result = self.indicator.calculate(self.test_data)
        
        # 验证信号字段存在
        required_signals = ['buy_signal', 'sell_signal', 'hold_signal']
        for signal in required_signals:
            self.assertIn(signal, result.columns)
            self.assertEqual(result[signal].dtype, bool)
    
    def test_boundary_conditions(self):
        # 测试边界条件
        # 数据不足
        insufficient_data = self.test_data.head(5)
        result = self.indicator.calculate(insufficient_data)
        self.assertIsInstance(result, pd.DataFrame)
        
        # NaN值
        nan_data = self.test_data.copy()
        nan_data.loc[10:15, 'close'] = np.nan
        result = self.indicator.calculate(nan_data)
        self.assertFalse(result['buy_signal'].isnull().any())
```

#### 语义验证测试
```python
def test_signal_semantic_consistency(self):
    # 测试信号语义一致性
    result = self.indicator.calculate(self.test_data)
    
    if self.indicator.indicator_type == 'state':
        # 状态型指标：XG=True时应该有buy_signal=True
        true_xg_rows = result[result['XG'] == True]
        if len(true_xg_rows) > 0:
            self.assertTrue(true_xg_rows['buy_signal'].all())
        
        false_xg_rows = result[result['XG'] == False]
        if len(false_xg_rows) > 0:
            self.assertFalse(false_xg_rows['buy_signal'].any())
```

### 4. 集成阶段

#### 指标注册
```python
# 在complete_indicator_registry.py中注册
def register_new_indicator(self):
    try:
        self.registry['NEW_INDICATOR'] = NewIndicator
        logger.info("✅ 成功注册指标: NEW_INDICATOR")
    except Exception as e:
        logger.error(f"❌ 注册指标失败: NEW_INDICATOR - {e}")
```

#### 文档更新
- 更新API文档
- 添加使用示例
- 更新指标列表

## 质量保证

### 1. 自动化检查

使用自动化风险检测工具检查新指标：

```python
from tools.automated_risk_detection import AutomatedRiskDetector

detector = AutomatedRiskDetector()
risk_result = detector.analyzer.analyze_indicator_risk(NewIndicator)

# 检查风险等级
if risk_result.risk_level.value == 'high':
    print("⚠️ 高风险指标，需要修复信号生成逻辑")
    print(f"建议: {risk_result.recommendations}")
```

### 2. 性能要求

- **计算时间**：单个指标计算时间<0.1秒
- **内存使用**：避免内存泄漏，及时释放大型数据结构
- **并发安全**：确保指标实例可以并发使用

### 3. 代码质量

- **代码风格**：遵循PEP 8规范，通过flake8检查
- **文档字符串**：所有公共方法必须有详细的docstring
- **类型注解**：使用类型注解提高代码可读性

```python
def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
    """
    计算技术指标
    
    Args:
        data: 输入数据，必须包含OHLCV字段
        
    Returns:
        包含指标值和信号的DataFrame
        
    Raises:
        ValueError: 当输入数据格式不正确时
    """
    pass
```

## 常见问题和解决方案

### 1. 信号生成语义不一致

**问题**：指标输出与信号生成逻辑不匹配

**解决方案**：
1. 识别指标类型（计数型、状态型、等级型、复合型）
2. 实现对应的自定义信号生成逻辑
3. 添加语义验证测试

### 2. 性能问题

**问题**：指标计算时间过长

**解决方案**：
1. 使用向量化操作替代循环
2. 优化算法复杂度
3. 使用缓存机制

### 3. 边界条件处理

**问题**：异常数据导致计算失败

**解决方案**：
1. 添加数据验证逻辑
2. 实现NaN值和无限值处理
3. 提供合理的默认值

## 最佳实践

1. **先设计后编码**：明确指标类型和信号语义
2. **测试驱动开发**：先写测试用例，再实现功能
3. **渐进式开发**：从简单功能开始，逐步完善
4. **文档同步更新**：代码和文档保持同步
5. **性能优先**：优化关键路径的性能
6. **错误处理**：提供清晰的错误信息和处理建议

## 版本控制

- **版本号格式**：使用语义化版本号（如v1.2.3）
- **变更记录**：详细记录每次修改的内容和影响
- **向后兼容**：确保API变更的向后兼容性

## 相关工具文档

- [ZXM指标API文档](../api/zxm_indicators_api.md)
- [分层测试框架使用指南](../testing/layered_testing_framework_guide.md)
- [自动化风险检测工具使用指南](../tools/automated_risk_detection_guide.md)

---

**文档版本**：v2.0
**最后更新**：2025-06-25
**维护团队**：技术开发团队
