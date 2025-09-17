# 股票分析系统架构修复指导原则

## 🚨 强制执行的架构原则

### 原则1: 数据库强依赖原则（不可违反）
**必须要依赖数据库，如果数据库不可用，就要修复数据层的问题，不可以有兜底逻辑**

#### ❌ 严格禁止的行为
```python
# 禁止：使用模拟数据
if database_unavailable:
    return mock_data  # 绝对禁止

# 禁止：使用缓存数据替代
if db_error:
    return cached_data  # 绝对禁止

# 禁止：使用默认值
if no_data:
    return default_stock_data  # 绝对禁止

# 禁止：跳过数据库检查
def get_stock_data(code):
    try:
        return db.query(code)
    except:
        return None  # 绝对禁止静默失败
```

#### ✅ 正确的架构实现
```python
# 正确：数据库不可用时必须报错
def get_stock_data(code: str) -> pd.DataFrame:
    """获取股票数据 - 严格依赖数据库"""
    try:
        connection = get_database_connection()  # 必须是真实连接
        if not connection.is_healthy():
            raise DatabaseUnavailableError("数据库连接不可用，系统无法继续运行")
        
        result = connection.query_dataframe(sql, params)
        if result.empty:
            raise DataNotFoundError(f"股票 {code} 的数据不存在")
        
        return result
    except Exception as e:
        logger.error(f"数据库查询失败: {e}")
        raise  # 必须重新抛出异常，不允许兜底
```

### 原则2: 禁止硬编码原则（不可违反）
**要明确禁止硬编码，禁止只测试结果核心指标**

#### ❌ 严格禁止的硬编码行为
```python
# 禁止：硬编码指标权重
indicator_weights = {
    'MACD': 0.3,
    'RSI': 0.2,
    'KDJ': 0.25
}  # 绝对禁止

# 禁止：硬编码阈值
if score > 60:  # 绝对禁止硬编码阈值
    return "BUY"

# 禁止：硬编码股票代码
test_stocks = ['000001', '000002']  # 绝对禁止

# 禁止：选择性测试指标
core_indicators = ['MACD', 'RSI', 'KDJ']  # 绝对禁止只测试部分指标
for indicator in core_indicators:  # 必须测试全部128个指标
    test_indicator(indicator)
```

#### ✅ 正确的配置驱动实现
```python
# 正确：所有配置来自配置文件
class ConfigurationManager:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
    
    def get_indicator_weights(self) -> Dict[str, float]:
        """从配置文件获取指标权重"""
        weights = self.config.get('indicator_weights', {})
        if not weights:
            raise ConfigurationError("指标权重配置缺失")
        return weights
    
    def get_threshold(self, threshold_name: str) -> float:
        """从配置文件获取阈值"""
        threshold = self.config.get('thresholds', {}).get(threshold_name)
        if threshold is None:
            raise ConfigurationError(f"阈值 {threshold_name} 配置缺失")
        return threshold

# 正确：测试所有指标
def test_all_indicators():
    """测试所有128个指标 - 不允许选择性测试"""
    registry = get_indicator_registry()
    all_indicators = registry.get_all_indicators()  # 必须获取全部指标
    
    failed_indicators = []
    for indicator_name in all_indicators:
        try:
            test_single_indicator(indicator_name)
        except Exception as e:
            failed_indicators.append((indicator_name, str(e)))
    
    if failed_indicators:
        raise TestFailureError(f"指标测试失败: {failed_indicators}")
```

### 原则3: 全局架构视角原则（不可违反）
**每个问题都要从全局架构视角，给出解决方案**

#### ❌ 严格禁止的局部修复行为
```python
# 禁止：局部绕过问题
def calculate_indicator(data):
    try:
        return real_calculation(data)
    except:
        return simple_fallback(data)  # 绝对禁止局部兜底

# 禁止：临时解决方案
def get_data():
    if production_db_down:
        return test_db_data  # 绝对禁止临时方案

# 禁止：只修复表面问题
def fix_syntax_error():
    # 只修复语法，不考虑架构影响  # 绝对禁止
    pass
```

#### ✅ 正确的全局架构解决方案
```python
# 正确：从架构层面解决问题
class GlobalArchitectureRepair:
    """全局架构修复 - 每个修复都考虑整体影响"""
    
    def repair_database_layer(self):
        """修复数据库层 - 影响L2/L3层和所有上层模块"""
        # 1. 统一容器系统（影响所有层）
        self._unify_container_system()
        
        # 2. 重构数据访问层（影响L3-L6层）
        self._restructure_data_access_layer()
        
        # 3. 建立统一服务注册（影响依赖注入体系）
        self._establish_unified_service_registry()
        
        # 4. 验证整体架构一致性
        self._validate_architecture_consistency()
    
    def repair_indicator_system(self):
        """修复指标系统 - 影响买点分析、策略选股等所有上层功能"""
        # 1. 系统性修复所有128个指标（不允许选择性修复）
        self._repair_all_indicators()
        
        # 2. 建立统一质量标准（影响所有指标实现）
        self._establish_quality_standards()
        
        # 3. 重构指标注册机制（影响指标加载和使用）
        self._restructure_indicator_registry()
        
        # 4. 验证指标系统完整性
        self._validate_indicator_system_integrity()
```

## 🏗️ 架构修复的系统性方法

### 1. 问题识别的全局视角
每个问题都要分析其在整个架构中的位置和影响：

```python
class ArchitectureProblemAnalysis:
    """架构问题分析 - 全局视角"""
    
    def analyze_problem(self, problem: str) -> ArchitectureImpact:
        """分析问题的架构影响"""
        return ArchitectureImpact(
            affected_layers=self._identify_affected_layers(problem),
            dependency_chain=self._trace_dependency_chain(problem),
            global_impact=self._assess_global_impact(problem),
            repair_strategy=self._design_repair_strategy(problem)
        )
    
    def _identify_affected_layers(self, problem: str) -> List[str]:
        """识别受影响的架构层"""
        # 分析问题影响哪些架构层（L1-L6）
        pass
    
    def _trace_dependency_chain(self, problem: str) -> List[str]:
        """追踪依赖链条"""
        # 分析问题如何通过依赖关系传播
        pass
```

### 2. 解决方案的架构一致性
所有解决方案都必须符合六层架构规范：

```python
class ArchitectureCompliantSolution:
    """架构合规的解决方案"""
    
    def validate_solution(self, solution: Solution) -> bool:
        """验证解决方案的架构合规性"""
        checks = [
            self._check_layer_boundaries(solution),
            self._check_dependency_direction(solution),
            self._check_service_registration(solution),
            self._check_configuration_management(solution)
        ]
        return all(checks)
    
    def _check_layer_boundaries(self, solution: Solution) -> bool:
        """检查层边界是否清晰"""
        # L6只能调用L5，L5只能调用L4，以此类推
        pass
    
    def _check_dependency_direction(self, solution: Solution) -> bool:
        """检查依赖方向是否正确"""
        # 上层依赖下层，不允许下层依赖上层
        pass
```

### 3. 修复验证的全面性
每个修复都要进行全面验证：

```python
class ComprehensiveRepairValidation:
    """全面的修复验证"""
    
    def validate_repair(self, repair: Repair) -> ValidationResult:
        """全面验证修复结果"""
        return ValidationResult(
            database_dependency=self._validate_database_dependency(),
            no_hardcoding=self._validate_no_hardcoding(),
            architecture_compliance=self._validate_architecture_compliance(),
            global_consistency=self._validate_global_consistency()
        )
    
    def _validate_database_dependency(self) -> bool:
        """验证数据库强依赖原则"""
        # 确保没有兜底逻辑，数据库不可用时系统必须报错
        pass
    
    def _validate_no_hardcoding(self) -> bool:
        """验证无硬编码原则"""
        # 确保所有配置都来自配置文件，测试覆盖所有指标
        pass
    
    def _validate_architecture_compliance(self) -> bool:
        """验证架构合规性"""
        # 确保遵循六层架构，无跨层调用
        pass
```

## 🎯 具体修复指导

### 数据库层修复指导
1. **容器统一**: 只保留`utils/unified_container.py`，废弃其他容器
2. **连接池单例**: 只保留`get_connection_pool()`一个入口
3. **服务注册**: 建立统一注册机制，每个服务只注册一次
4. **错误处理**: 数据库不可用时必须抛出异常，不允许兜底

### 指标系统修复指导
1. **全量修复**: 必须修复所有128个指标，不允许选择性修复
2. **质量统一**: 建立统一的指标质量标准和测试要求
3. **配置驱动**: 所有指标权重和参数都来自配置文件
4. **性能要求**: 所有指标计算时间必须<2秒

### 业务逻辑修复指导
1. **架构合规**: 严格遵循六层架构，消除所有跨层调用
2. **依赖注入**: 所有依赖都通过容器注入，不允许直接创建
3. **配置管理**: 所有业务参数都来自配置，不允许硬编码
4. **错误传播**: 建立正确的错误传播机制，不允许静默失败

## 📊 修复验证标准

### 强制验证项目
- [ ] 数据库强依赖：系统在数据库不可用时必须报错停止
- [ ] 无硬编码：所有配置、阈值、权重都来自配置文件
- [ ] 全量测试：测试覆盖所有128个指标，不允许选择性测试
- [ ] 架构合规：严格遵循六层架构，无跨层调用
- [ ] 依赖注入：所有服务都通过容器注入，无直接创建
- [ ] 错误处理：所有错误都正确传播，无静默失败

### 质量保证措施
- [ ] 代码审查：所有修复都要进行架构合规性审查
- [ ] 自动化测试：建立覆盖所有功能的自动化测试
- [ ] 性能监控：所有关键功能都要有性能监控
- [ ] 文档更新：所有架构变更都要更新相关文档

## 📝 总结

架构修复必须遵循三个核心原则：
1. **数据库强依赖** - 不允许任何兜底逻辑
2. **禁止硬编码** - 不允许选择性测试
3. **全局架构视角** - 每个修复都要考虑整体影响

只有严格遵循这些原则，才能确保系统的长期稳定性和可维护性。
