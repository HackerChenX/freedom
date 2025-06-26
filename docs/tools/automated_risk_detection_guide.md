# 自动化风险检测工具使用指南

## 概述

自动化风险检测工具是技术分析系统的智能质量保证组件，能够自动识别技术指标的类型、评估风险等级、检测信号生成语义一致性，并提供具体的改进建议。

## 核心功能

### 1. 指标类型自动识别

工具能够自动识别以下指标类型：

- **计数型指标** (COUNT_TYPE)：输出整数计数值
- **等级型指标** (LEVEL_TYPE)：输出离散等级值  
- **状态型指标** (STATE_TYPE)：输出布尔或枚举状态
- **比率型指标** (RATIO_TYPE)：输出比率值
- **连续型指标** (CONTINUOUS_TYPE)：输出连续数值
- **复合型指标** (COMPOSITE_TYPE)：使用多个子指标

### 2. 风险等级评估

- **高风险** (HIGH)：信号生成语义不一致，需要立即修复
- **中风险** (MEDIUM)：存在潜在问题，建议优化
- **低风险** (LOW)：运行正常，无需修复

### 3. 信号一致性评分

提供0-100分的信号一致性评分，评估指标输出与信号生成的匹配度。

## 使用方法

### 1. 基本使用

#### 快速扫描所有指标

```python
from tools.automated_risk_detection import AutomatedRiskDetector

# 创建检测器实例
detector = AutomatedRiskDetector()

# 扫描所有已注册的指标
results = detector.scan_all_indicators()

# 查看扫描结果
print(f"扫描完成，共分析 {len(results)} 个指标")

# 获取高风险指标
high_risk_indicators = detector.get_high_risk_indicators()
print(f"发现 {len(high_risk_indicators)} 个高风险指标:")
for indicator in high_risk_indicators:
    print(f"  - {indicator}")
```

#### 生成风险报告

```python
# 生成详细的风险报告
risk_report = detector.generate_risk_report()

# 保存报告到文件
with open('risk_assessment_report.md', 'w', encoding='utf-8') as f:
    f.write(risk_report)

print("风险报告已保存到 risk_assessment_report.md")

# 查看报告摘要
print("\n=== 风险报告摘要 ===")
print(risk_report[:500] + "...")
```

### 2. 单个指标分析

#### 分析特定指标

```python
from tools.automated_risk_detection import IndicatorRiskAnalyzer
from indicators.complete_indicator_registry import complete_registry

# 创建分析器
analyzer = IndicatorRiskAnalyzer()

# 获取指标类
indicator_instance = complete_registry.create_indicator('ZXM_TURNOVER')
indicator_class = indicator_instance.__class__

# 分析指标风险
result = analyzer.analyze_indicator_risk(indicator_class)

# 查看分析结果
print(f"指标名称: {result.indicator_name}")
print(f"指标类型: {result.indicator_type.value}")
print(f"风险等级: {result.risk_level.value}")
print(f"信号一致性评分: {result.signal_consistency_score:.1f}")
print(f"风险因素: {result.risk_factors}")
print(f"改进建议: {result.recommendations}")
```

#### 查看详细信息

```python
# 查看详细分析信息
details = result.details
print("\n=== 详细信息 ===")
print(f"类名: {details['class_name']}")
print(f"指标类型: {details['indicator_type']}")
print(f"有自定义信号逻辑: {details['has_custom_signal_logic']}")
print(f"输出列: {details['output_columns']}")
print(f"源代码长度: {details['source_code_length']}")
print(f"有布尔输出: {details['has_boolean_output']}")
print(f"有计数输出: {details['has_count_output']}")
print(f"有评分输出: {details['has_score_output']}")
```

### 3. 批量指标分析

#### 分析特定类型的指标

```python
# 分析所有ZXM指标
zxm_indicators = [name for name in detector.detection_results.keys() 
                  if name.startswith('ZXM_')]

print(f"=== ZXM指标风险分析 ===")
for indicator_name in zxm_indicators:
    result = detector.detection_results[indicator_name]
    risk_level = result.risk_level.value
    score = result.signal_consistency_score
    
    status = "✅" if risk_level == "low" else "⚠️" if risk_level == "medium" else "❌"
    print(f"{status} {indicator_name}: {risk_level} 风险, 评分 {score:.1f}")
```

#### 按风险等级分类

```python
# 按风险等级分类指标
risk_categories = {"high": [], "medium": [], "low": []}

for name, result in detector.detection_results.items():
    risk_level = result.risk_level.value
    risk_categories[risk_level].append(name)

# 显示分类结果
for level, indicators in risk_categories.items():
    print(f"\n{level.upper()} 风险指标 ({len(indicators)} 个):")
    for indicator in indicators[:5]:  # 只显示前5个
        print(f"  - {indicator}")
    if len(indicators) > 5:
        print(f"  ... 还有 {len(indicators) - 5} 个")
```

### 4. 高级功能

#### 自定义风险模式

```python
# 扩展风险检测模式
analyzer = IndicatorRiskAnalyzer()

# 添加自定义风险模式
custom_patterns = {
    "custom_type_patterns": {
        "output_patterns": [r"custom_field", r"special_output"],
        "value_patterns": ["custom_type"],
        "signal_risk": "medium",
        "description": "自定义类型指标"
    }
}

analyzer.risk_patterns.update(custom_patterns)
```

#### 批量修复建议

```python
# 获取所有改进建议
recommendations_summary = detector.get_recommendations_summary()

print("=== 批量修复建议 ===")
for indicator_name, recommendations in recommendations_summary.items():
    print(f"\n{indicator_name}:")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
```

## 集成到开发流程

### 1. Pre-commit Hook集成

```python
# 在tools/continuous_quality_assurance.py中已集成
from tools.continuous_quality_assurance import PreCommitHooks

hooks = PreCommitHooks()
results = hooks.run_all_hooks()

# 风险检测作为信号一致性检查的一部分自动运行
```

### 2. CI/CD集成

```yaml
# .github/workflows/risk-detection.yml
name: Automated Risk Detection

on: [push, pull_request]

jobs:
  risk-detection:
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
    
    - name: Run Risk Detection
      run: |
        python -c "
        from tools.automated_risk_detection import AutomatedRiskDetector
        
        detector = AutomatedRiskDetector()
        results = detector.scan_all_indicators()
        
        high_risk = detector.get_high_risk_indicators()
        if high_risk:
            print(f'发现 {len(high_risk)} 个高风险指标: {high_risk}')
            exit(1)
        
        accuracy = detector._calculate_accuracy()
        print(f'风险检测完成，准确率: {accuracy:.1f}%')
        "
```

### 3. 定期监控

```python
# 定期风险监控脚本
import schedule
import time

def daily_risk_scan():
    detector = AutomatedRiskDetector()
    results = detector.scan_all_indicators()
    
    high_risk_indicators = detector.get_high_risk_indicators()
    
    if high_risk_indicators:
        # 发送告警通知
        send_alert(f"发现 {len(high_risk_indicators)} 个高风险指标")
    
    # 生成日报
    report = detector.generate_risk_report()
    save_daily_report(report)

# 每天上午9点执行风险扫描
schedule.every().day.at("09:00").do(daily_risk_scan)

while True:
    schedule.run_pending()
    time.sleep(3600)  # 每小时检查一次
```

## 配置选项

### 1. 风险检测配置

```python
# 自定义风险阈值
analyzer = IndicatorRiskAnalyzer()

# 修改信号一致性规则
analyzer.signal_consistency_rules['count_type']['expected_logic'] = "buy_signal = (count_value >= threshold)"

# 修改风险评估阈值
analyzer.alert_thresholds = {
    'high_risk_score_threshold': 50.0,  # 低于50分为高风险
    'medium_risk_score_threshold': 75.0,  # 50-75分为中风险
}
```

### 2. 输出格式配置

```python
# 自定义报告格式
detector = AutomatedRiskDetector()

# 生成JSON格式报告
import json

json_report = {
    'scan_time': datetime.now().isoformat(),
    'total_indicators': len(detector.detection_results),
    'high_risk_count': len(detector.get_high_risk_indicators()),
    'accuracy': detector._calculate_accuracy(),
    'details': {
        name: {
            'type': result.indicator_type.value,
            'risk_level': result.risk_level.value,
            'score': result.signal_consistency_score,
            'recommendations': result.recommendations
        }
        for name, result in detector.detection_results.items()
    }
}

with open('risk_report.json', 'w') as f:
    json.dump(json_report, f, indent=2, ensure_ascii=False)
```

## 性能优化

### 1. 缓存机制

```python
class CachedRiskDetector(AutomatedRiskDetector):
    def __init__(self):
        super().__init__()
        self._analysis_cache = {}
    
    def analyze_indicator_with_cache(self, indicator_class):
        class_name = indicator_class.__name__
        
        if class_name not in self._analysis_cache:
            result = self.analyzer.analyze_indicator_risk(indicator_class)
            self._analysis_cache[class_name] = result
        
        return self._analysis_cache[class_name]
```

### 2. 并行处理

```python
import concurrent.futures

class ParallelRiskDetector(AutomatedRiskDetector):
    def scan_all_indicators_parallel(self, max_workers=4):
        from indicators.complete_indicator_registry import complete_registry
        
        indicator_names = complete_registry.get_indicator_names()
        
        def analyze_single_indicator(indicator_name):
            try:
                indicator_instance = complete_registry.create_indicator(indicator_name)
                indicator_class = indicator_instance.__class__
                return indicator_name, self.analyzer.analyze_indicator_risk(indicator_class)
            except Exception as e:
                logger.error(f"分析指标 {indicator_name} 时出错: {e}")
                return indicator_name, None
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(analyze_single_indicator, name): name 
                      for name in indicator_names}
            
            for future in concurrent.futures.as_completed(futures):
                indicator_name, result = future.result()
                if result:
                    self.detection_results[indicator_name] = result
        
        return self.detection_results
```

## 故障排除

### 常见问题

#### 1. 扫描超时

**问题**：大量指标扫描时间过长

**解决方案**：
```python
# 使用并行处理
parallel_detector = ParallelRiskDetector()
results = parallel_detector.scan_all_indicators_parallel(max_workers=8)

# 或者分批处理
def batch_scan(detector, indicator_names, batch_size=10):
    results = {}
    for i in range(0, len(indicator_names), batch_size):
        batch = indicator_names[i:i+batch_size]
        batch_results = detector.scan_indicators(batch)
        results.update(batch_results)
    return results
```

#### 2. 内存使用过高

**问题**：分析大量指标时内存使用过高

**解决方案**：
```python
# 及时清理缓存
detector = AutomatedRiskDetector()

def memory_efficient_scan():
    indicator_names = complete_registry.get_indicator_names()
    
    for i, indicator_name in enumerate(indicator_names):
        # 分析单个指标
        result = detector.analyze_single_indicator(indicator_name)
        
        # 处理结果
        process_result(result)
        
        # 每100个指标清理一次内存
        if i % 100 == 0:
            gc.collect()
```

#### 3. 误报问题

**问题**：风险检测出现误报

**解决方案**：
```python
# 调整检测阈值
analyzer.risk_patterns['count_type_patterns']['signal_risk'] = 'medium'  # 降低风险等级

# 添加白名单
whitelist_indicators = ['KNOWN_GOOD_INDICATOR']

def filtered_scan(detector):
    results = detector.scan_all_indicators()
    
    # 过滤白名单指标
    filtered_results = {
        name: result for name, result in results.items()
        if name not in whitelist_indicators or result.risk_level.value != 'high'
    }
    
    return filtered_results
```

## 最佳实践

1. **定期扫描**：建立定期风险扫描机制，及时发现问题
2. **渐进式修复**：优先修复高风险指标，逐步改善系统质量
3. **持续监控**：将风险检测集成到CI/CD流程中
4. **文档记录**：详细记录风险检测结果和修复过程
5. **团队协作**：建立风险评估和修复的团队协作机制

## 扩展开发

### 添加新的检测规则

```python
# 扩展指标类型检测
def add_custom_indicator_type():
    analyzer = IndicatorRiskAnalyzer()
    
    # 添加新的指标类型模式
    new_pattern = {
        "ml_type_patterns": {
            "output_patterns": [r"prediction", r"probability", r"confidence"],
            "value_patterns": ["ml_output"],
            "signal_risk": "medium",
            "description": "机器学习类型指标"
        }
    }
    
    analyzer.risk_patterns.update(new_pattern)
    
    # 添加对应的信号一致性规则
    analyzer.signal_consistency_rules['ml_type'] = {
        "expected_logic": "buy_signal = (prediction > threshold)",
        "common_errors": ["使用通用逻辑处理ML输出"],
        "validation_method": "validate_ml_signal_logic"
    }
```

### 自定义风险评分算法

```python
class CustomRiskAnalyzer(IndicatorRiskAnalyzer):
    def _assess_signal_consistency(self, indicator_class, indicator_type):
        # 自定义信号一致性评分算法
        base_score = super()._assess_signal_consistency(indicator_class, indicator_type)
        
        # 添加自定义评分因子
        custom_factors = self._calculate_custom_factors(indicator_class)
        
        # 综合评分
        final_score = base_score * 0.7 + custom_factors * 0.3
        
        return min(100.0, max(0.0, final_score))
    
    def _calculate_custom_factors(self, indicator_class):
        # 实现自定义评分因子计算
        return 80.0  # 示例值
```

---

**文档版本**：v1.0  
**最后更新**：2025-06-25  
**相关文档**：[技术指标开发规范](../development/indicator_development_standards.md)
