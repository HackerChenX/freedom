# 权威指标重构标准规范

## 🎯 **总体目标**

基于MACD、RSI、KDJ三大核心指标的成功重构经验，制定适用于全部100+技术指标的权威重构标准，确保系统架构的一致性、可维护性和专业性。

## 🚨 **核心原则（不可违反）**

### **1. 稳定性优先原则**
- **禁止重命名现有方法**：get_signal()和get_signals()方法名保持不变
- **100%向后兼容**：现有123个指标无需修改即可正常工作
- **渐进式改进**：只增加功能，不破坏现有功能
- **风险控制**：任何修改都必须经过完整验证

### **2. 简洁性原则**
- **避免过度工程化**：不创建复杂的映射机制
- **单一数据源**：每个指标值只存储一次
- **统一命名格式**：{indicator}_{type}格式
- **最小化复杂度**：优先选择简单直接的解决方案

### **3. 标准化原则**
- **统一接口规范**：所有指标遵循相同的接口标准
- **一致的返回格式**：DataFrame和Dict格式标准化
- **规范的列名标准**：项目级统一列名规范
- **完整的文档要求**：每个指标都有标准化文档

## 📊 **方法职责明确化标准**

### **核心方法职责定义**

#### **calculate()方法**
```python
@abc.abstractmethod
def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
    """
    职责：纯指标计算，返回指标数值
    
    返回格式：
    - 必须返回pandas.DataFrame
    - 列名使用{indicator}_{type}格式
    - 如：macd_dif, macd_dea, macd_histogram
    
    示例：
    MACD → ['macd_dif', 'macd_dea', 'macd_histogram']
    RSI  → ['rsi_value']
    KDJ  → ['kdj_k', 'kdj_d', 'kdj_j']
    """
```

#### **get_signal()方法**
```python
@abc.abstractmethod
def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
    """
    职责：获取最新单一信号，用于实时决策
    
    返回格式：
    {
        'signal_type': str,      # 'buy', 'sell', 'hold'
        'strength': float,       # 信号强度 0.0-1.0
        'confidence': float,     # 信号置信度 0.0-1.0
        'timestamp': datetime,   # 信号时间
        'price': float,         # 触发价格
        'metadata': dict        # 额外信息
    }
    """
```

#### **get_signals()方法**
```python
def get_signals(self, data: pd.DataFrame) -> pd.DataFrame:
    """
    职责：生成历史信号序列，用于回测分析
    
    返回格式：
    DataFrame包含标准列：
    - buy_signal: bool
    - sell_signal: bool
    - hold_signal: bool
    - signal_strength: float (0-1)
    - signal_confidence: float (0-1)
    """
```

### **架构分层原则**

#### **L4核心服务层职责边界**
BaseIndicator作为L4核心服务层组件，严格限制职责范围：
- ✅ **允许**：纯技术指标计算（calculate）
- ✅ **允许**：基础信号生成（get_signal, get_signals）
- ✅ **允许**：技术形态识别（get_patterns）
- ❌ **禁止**：业务逻辑实现（策略选股、买点分析）
- ❌ **禁止**：综合评分计算（应在L5业务层）
- ❌ **禁止**：投资建议生成（应在L5业务层）

#### **业务方法正确实现位置**
```python
# ✅ 正确：在L5业务应用层实现
# strategy/stock_selection_analyzer.py
class StockSelectionAnalyzer:
    def analyze_for_stock_selection(self, indicators: List[BaseIndicator],
                                  data: pd.DataFrame) -> Dict[str, Any]:
        """策略选股分析 - 正确的实现位置"""
        pass

# analysis/buypoint_detector.py
class BuypointDetector:
    def analyze_for_buypoint_detection(self, indicators: List[BaseIndicator],
                                     data: pd.DataFrame) -> Dict[str, Any]:
        """买点分析 - 正确的实现位置"""
        pass
```

## 🔧 **列名标准化规范**

### **统一列名格式**
```
{indicator}_{type}[_{period}]
```

### **标准列名示例**
```python
# MACD指标
"macd_dif"        # DIF线
"macd_dea"        # DEA线
"macd_histogram"  # 柱状图

# RSI指标
"rsi_value"       # RSI主值
"rsi_overbought"  # 超买信号
"rsi_oversold"    # 超卖信号

# KDJ指标
"kdj_k"           # K值
"kdj_d"           # D值
"kdj_j"           # J值

# 通用信号列
"buy_signal"      # 买入信号
"sell_signal"     # 卖出信号
"hold_signal"     # 持有信号
"signal_strength" # 信号强度
"signal_confidence" # 信号置信度
```

### **向后兼容处理**
```python
# 简单列名转换工具
LEGACY_MAPPINGS = {
    'DIF': 'macd_dif',
    'DEA': 'macd_dea',
    'MACD': 'macd_histogram',
    'K': 'kdj_k',
    'D': 'kdj_d',
    'J': 'kdj_j',
    'rsi_14': 'rsi_value'
}
```

## 📝 **代码质量标准**

### **必须包含的装饰器**
```python
@performance_monitor(threshold_seconds=2.0)  # 性能监控
@exception_handler(reraise=True)             # 异常处理
def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
    pass
```

### **必须包含的文档字符串**
```python
def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
    """
    计算{指标名称}指标
    
    Args:
        data: 输入数据，必须包含OHLCV字段
        
    Returns:
        pd.DataFrame: 包含{具体列名}的指标数据
        
    Raises:
        ValueError: 当输入数据不符合要求时
    """
```

### **必须包含的参数验证**
```python
def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
    # 数据验证
    if data.empty:
        raise ValueError("输入数据不能为空")
    
    if 'close' not in data.columns:
        raise ValueError("输入数据必须包含'close'列")
    
    if len(data) < self.period:
        logger.warning(f"数据长度不足，需要至少{self.period}个数据点")
```

## 🛡️ **质量保证流程**

### **重构前检查清单**
- [ ] 检查现有实现，避免重复开发
- [ ] 确认方法命名符合标准
- [ ] 验证返回格式符合规范
- [ ] 确保向后兼容性

### **重构中执行标准**
- [ ] 使用统一的列名格式
- [ ] 添加完整的异常处理
- [ ] 包含性能监控装饰器
- [ ] 编写完整的文档字符串

### **重构后验证要求**
- [ ] 单元测试覆盖率 > 90%
- [ ] 性能测试通过（计算时间 < 2秒）
- [ ] 向后兼容性测试100%通过
- [ ] 专业金融标准验证通过

## 📈 **实施计划**

### **阶段1：核心指标标准化（已完成）**
- ✅ MACD、RSI、KDJ三大核心指标
- ✅ 方法职责明确化
- ✅ 列名标准化
- ✅ 专业业务接口添加

### **阶段2：扩展指标标准化（执行中）**
- [ ] P1级别：MA、EMA、BOLL、ATR等核心指标
- [ ] P2级别：其他常用技术指标
- [ ] P3级别：专业和自定义指标

### **阶段3：系统整合（最后执行）**
- [ ] 清理不符合标准的代码
- [ ] 性能优化验证
- [ ] 完整文档体系建立

## 🚨 **强制执行要求**

### **违规处理**
1. **立即修正**：违反标准的代码必须立即修正
2. **架构审查**：重大修改需要架构审查
3. **测试验证**：所有修改必须通过完整测试

### **成功标准**
- ✅ 方法职责清晰，命名规范
- ✅ 返回格式统一，列名标准化
- ✅ 向后兼容100%，现有代码无需修改
- ✅ 专业业务接口完整，支持策略选股和买点分析
- ✅ 代码质量达到生产级标准

这套权威标准基于三大核心指标的成功重构经验制定，必须严格遵循，确保整个指标系统的一致性和专业性。
