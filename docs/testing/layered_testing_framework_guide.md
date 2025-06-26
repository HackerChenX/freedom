# 分层测试框架使用指南

## 概述

分层测试框架是技术分析系统的核心质量保证工具，实现了单元测试→语义测试→集成测试→端到端测试的完整分层体系，确保每层测试的独立性和完整性。

## 框架架构

### 测试层级结构

```
分层测试框架
├── UnitTestLayer (单元测试层)
│   ├── 计算准确性测试
│   ├── 边界条件处理测试
│   └── 异常情况处理测试
├── SemanticTestLayer (语义测试层)
│   ├── 信号一致性测试
│   └── 业务逻辑测试
├── IntegrationTestLayer (集成测试层) [可扩展]
│   └── 组件协作测试
└── EndToEndTestLayer (端到端测试层) [可扩展]
    └── 完整流程测试
```

### 核心组件

#### 1. LayeredTestingFramework - 主框架类

```python
from tests.framework.layered_testing_framework import LayeredTestingFramework

# 创建框架实例
framework = LayeredTestingFramework()

# 运行所有测试层
target_indicators = ['ZXM_TURNOVER', 'ZXM_VOLUME_SHRINK', 'ZXM_DAILY_TREND_UP']
summary = framework.run_all_layers(target_indicators)

# 查看结果
print(f"总体成功: {summary['overall_success']}")
print(f"总体覆盖率: {summary['overall_coverage']:.1f}%")
print(f"总执行时间: {summary['total_execution_time']:.1f}秒")
```

#### 2. UnitTestLayer - 单元测试层

**功能**：验证指标的计算逻辑正确性

**覆盖率要求**：95%+  
**执行时间要求**：<30秒

**测试内容**：
- 计算准确性验证
- 边界条件处理（数据不足、NaN值、极值）
- 异常情况处理

```python
from tests.framework.layered_testing_framework import UnitTestLayer

# 单独运行单元测试层
unit_layer = UnitTestLayer()
results = unit_layer.run_tests(['ZXM_TURNOVER'])

# 查看详细结果
for result in results:
    print(f"{result.test_name}: {'通过' if result.success else '失败'}")
    print(f"执行时间: {result.execution_time:.3f}秒")
```

#### 3. SemanticTestLayer - 语义测试层

**功能**：验证指标的业务逻辑正确性

**覆盖率要求**：90%+  
**执行时间要求**：<60秒

**测试内容**：
- 信号生成语义一致性
- 业务逻辑正确性
- 领域知识验证

```python
from tests.framework.layered_testing_framework import SemanticTestLayer

# 单独运行语义测试层
semantic_layer = SemanticTestLayer()
results = semantic_layer.run_tests(['ZXM_TURNOVER'])

# 验证信号一致性
for result in results:
    if 'signal_consistency' in result.test_name:
        print(f"信号一致性测试: {'通过' if result.success else '失败'}")
```

## 使用方法

### 1. 基本使用

#### 快速开始

```python
from tests.framework.layered_testing_framework import LayeredTestingFramework

# 1. 创建框架实例
framework = LayeredTestingFramework()

# 2. 定义要测试的指标
test_indicators = [
    'ZXM_BS_ABSORB',
    'ZXM_TURNOVER', 
    'ZXM_VOLUME_SHRINK',
    'ZXM_DAILY_TREND_UP',
    'ZXM_AMPLITUDE_ELASTICITY'
]

# 3. 运行分层测试
summary = framework.run_all_layers(test_indicators)

# 4. 查看结果
print("=== 分层测试结果 ===")
print(f"总体成功: {summary['overall_success']}")
print(f"总体覆盖率: {summary['overall_coverage']:.1f}%")
print(f"总执行时间: {summary['total_execution_time']:.1f}秒")
print(f"总测试数量: {summary['total_tests']}")

# 5. 查看各层详情
for layer_name, layer_summary in summary['layer_summaries'].items():
    print(f"\n{layer_name}:")
    print(f"  成功: {layer_summary['success']}")
    print(f"  覆盖率: {layer_summary['coverage']:.1f}%")
    print(f"  测试数量: {layer_summary['test_count']}")
    print(f"  执行时间: {layer_summary['execution_time']:.1f}秒")
```

#### 生成详细报告

```python
# 生成详细的测试报告
detailed_report = framework.generate_detailed_report()

# 保存报告到文件
with open('layered_test_report.md', 'w', encoding='utf-8') as f:
    f.write(detailed_report)

print("详细报告已保存到 layered_test_report.md")
```

### 2. 高级使用

#### 自定义测试层

```python
from tests.framework.layered_testing_framework import BaseTestLayer, LayerTestConfig

class CustomTestLayer(BaseTestLayer):
    def __init__(self):
        config = LayerTestConfig(
            layer_name="Custom Tests",
            required_coverage=85.0,
            max_execution_time=45.0,
            success_criteria={
                "custom_metric": 90.0
            }
        )
        super().__init__(config)
    
    def run_tests(self, target_indicators):
        # 实现自定义测试逻辑
        self.start_time = time.time()
        self.results = []
        
        for indicator_name in target_indicators:
            # 自定义测试实现
            result = self._custom_test(indicator_name)
            self.results.append(result)
        
        self.end_time = time.time()
        return self.results
    
    def _custom_test(self, indicator_name):
        # 具体的自定义测试逻辑
        pass

# 使用自定义测试层
custom_layer = CustomTestLayer()
results = custom_layer.run_tests(['ZXM_TURNOVER'])
```

#### 并行测试执行

```python
import concurrent.futures

def run_layer_test(layer_class, indicators):
    layer = layer_class()
    return layer.run_tests(indicators)

# 并行运行多个测试层
test_indicators = ['ZXM_TURNOVER', 'ZXM_VOLUME_SHRINK']

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    futures = {
        executor.submit(run_layer_test, UnitTestLayer, test_indicators): "Unit",
        executor.submit(run_layer_test, SemanticTestLayer, test_indicators): "Semantic"
    }
    
    for future in concurrent.futures.as_completed(futures):
        layer_name = futures[future]
        try:
            results = future.result()
            print(f"{layer_name} 测试完成: {len(results)} 个测试")
        except Exception as e:
            print(f"{layer_name} 测试失败: {e}")
```

### 3. 集成到CI/CD

#### GitHub Actions集成

```yaml
# .github/workflows/layered-tests.yml
name: Layered Testing Framework

on: [push, pull_request]

jobs:
  layered-tests:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.12
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Run Layered Tests
      run: |
        python -c "
        from tests.framework.layered_testing_framework import LayeredTestingFramework
        
        framework = LayeredTestingFramework()
        indicators = ['ZXM_TURNOVER', 'ZXM_VOLUME_SHRINK', 'ZXM_DAILY_TREND_UP']
        summary = framework.run_all_layers(indicators)
        
        if not summary['overall_success']:
            exit(1)
        
        print(f'测试通过: 覆盖率 {summary[\"overall_coverage\"]:.1f}%')
        "
```

#### Pre-commit Hook集成

```python
# 在tools/continuous_quality_assurance.py中已集成
from tools.continuous_quality_assurance import PreCommitHooks

hooks = PreCommitHooks()
results = hooks.run_all_hooks()

# 分层测试作为性能回归检查的一部分自动运行
```

## 配置选项

### 测试层配置

```python
from tests.framework.layered_testing_framework import LayerTestConfig

# 自定义测试层配置
config = LayerTestConfig(
    layer_name="My Custom Layer",
    required_coverage=90.0,        # 要求的覆盖率
    max_execution_time=60.0,       # 最大执行时间（秒）
    success_criteria={             # 成功标准
        "accuracy": 95.0,
        "performance": 0.1,
        "reliability": 99.0
    }
)
```

### 框架全局配置

```python
# 在框架初始化时配置
framework = LayeredTestingFramework()

# 添加自定义测试层
framework.layers.append(CustomTestLayer())

# 配置日志级别
import logging
logging.getLogger('tests.framework.layered_testing_framework').setLevel(logging.DEBUG)
```

## 性能优化

### 1. 测试数据缓存

```python
class OptimizedTestLayer(BaseTestLayer):
    def __init__(self):
        super().__init__(config)
        self._test_data_cache = {}
    
    def _get_test_data(self, data_type):
        if data_type not in self._test_data_cache:
            self._test_data_cache[data_type] = self._generate_test_data(data_type)
        return self._test_data_cache[data_type]
```

### 2. 并行测试执行

```python
# 在测试层内部使用并行处理
def run_tests(self, target_indicators):
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(self._test_indicator, indicator): indicator 
            for indicator in target_indicators
        }
        
        for future in concurrent.futures.as_completed(futures):
            indicator = futures[future]
            try:
                result = future.result()
                self.results.append(result)
            except Exception as e:
                logger.error(f"测试 {indicator} 失败: {e}")
```

## 故障排除

### 常见问题

#### 1. 测试超时

**问题**：测试执行时间超过配置的最大时间

**解决方案**：
```python
# 增加超时时间
config.max_execution_time = 120.0

# 或者优化测试数据大小
def _generate_test_data(self):
    # 使用更小的数据集
    return pd.DataFrame({...}, index=range(50))  # 减少数据量
```

#### 2. 覆盖率不足

**问题**：测试覆盖率低于要求

**解决方案**：
```python
# 添加更多测试用例
def run_tests(self, target_indicators):
    for indicator_name in target_indicators:
        # 添加多种测试场景
        self.results.append(self._test_normal_case(indicator_name))
        self.results.append(self._test_edge_case(indicator_name))
        self.results.append(self._test_error_case(indicator_name))
```

#### 3. 内存使用过高

**问题**：测试过程中内存使用过高

**解决方案**：
```python
# 及时清理测试数据
def run_tests(self, target_indicators):
    for indicator_name in target_indicators:
        test_data = self._generate_test_data()
        result = self._test_indicator(indicator_name, test_data)
        self.results.append(result)
        
        # 清理内存
        del test_data
        gc.collect()
```

## 最佳实践

1. **渐进式测试**：从简单的单元测试开始，逐步增加复杂度
2. **数据驱动**：使用多样化的测试数据覆盖各种场景
3. **快速反馈**：保持测试执行时间在合理范围内
4. **清晰报告**：提供详细且易于理解的测试报告
5. **持续改进**：根据测试结果持续优化测试用例

## 扩展开发

### 添加新的测试层

```python
# 1. 继承BaseTestLayer
class NewTestLayer(BaseTestLayer):
    def __init__(self):
        config = LayerTestConfig(
            layer_name="New Test Layer",
            required_coverage=85.0,
            max_execution_time=30.0,
            success_criteria={"new_metric": 90.0}
        )
        super().__init__(config)
    
    def run_tests(self, target_indicators):
        # 实现测试逻辑
        pass

# 2. 注册到框架
framework = LayeredTestingFramework()
framework.layers.append(NewTestLayer())
```

### 自定义测试指标

```python
class CustomMetricTestLayer(BaseTestLayer):
    def calculate_layer_coverage(self):
        # 自定义覆盖率计算逻辑
        custom_coverage = self._calculate_custom_coverage()
        return custom_coverage
    
    def validate_layer_success(self):
        # 自定义成功验证逻辑
        return self._custom_validation()
```

---

**文档版本**：v1.0  
**最后更新**：2025-06-25  
**相关文档**：[技术指标开发规范](../development/indicator_development_standards.md)
