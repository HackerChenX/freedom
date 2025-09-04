
# EMV指标严格标准化5阶段验证报告

## 📊 验证概览

- **指标名称**: EMV
- **验证时间**: 2025-09-03 15:49:23
- **总体评分**: 25.5/100分
- **最低评分**: 0.0/100分
- **最终状态**: FAILED
- **验证结果**: ❌ 未通过
- **执行时间**: 0.02秒

## 🔍 各阶段详细结果


### 阶段1: 算法真实性验证

- **评分**: 0.0/100分
- **状态**: ❌ 未通过
- **详细信息**: {'error': '指标 EMV 不是BaseIndicator的实例'}


### 阶段2: 基础功能验证

- **评分**: 0.0/100分
- **状态**: ❌ 未通过
- **详细信息**: {'error': "'dict' object has no attribute 'empty'"}


### 阶段3: 形态识别验证

- **评分**: 0.0/100分
- **状态**: ❌ 未通过
- **详细信息**: {'pattern_method_available': True, 'signal_method_available': False, 'emv_signal_distribution': {'positive_ratio': 0, 'negative_ratio': 0, 'zero_ratio': 0}}


### 阶段4: 架构合规性验证

- **评分**: 32.5/100分
- **状态**: ❌ 未通过
- **详细信息**: {'base_indicator_inheritance': False, 'implemented_methods': 2, 'required_methods': 5, 'implemented_attributes': 1, 'required_attributes': 2}


### 阶段5: 生产就绪性验证

- **评分**: 95.0/100分
- **状态**: ✅ 通过
- **详细信息**: {'performance_ms': 0.17490386962890625, 'stability_score': 4, 'boundary_handling_score': 3, 'memory_increase_mb': 0}

