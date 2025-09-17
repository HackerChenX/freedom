# 数据库层问题跟踪清单

## 🚨 关键阻塞问题

### 问题1: 容器键重复注册错误
**错误信息**: `Attempted to reuse key: 'STOCK_LIST'`
**影响级别**: 🔴 严重 - 阻止所有数据库操作
**状态**: 🔄 分析中
**根本原因**: 多个容器系统冲突，重复注册相同键

#### 具体位置:
1. `db/service_registry.py:42` - `register_singleton_by_name('clickhouse_db')`
2. `config/container_config.py:55` - `register_singleton_by_name()`
3. `db/container.py:271-292` - 多个服务注册
4. `utils/optimized_dependency_injection.py` - 字符串键注册系统

#### 修复计划:
- [ ] 统一使用`utils/unified_container.py`
- [ ] 移除字符串键注册，改用Type键
- [ ] 创建统一服务注册入口

### 问题2: 连接池重复初始化
**错误表现**: 多个连接池实例被创建
**影响级别**: 🟡 中等 - 影响性能和资源使用
**状态**: 🔄 分析中
**根本原因**: 缺乏全局单例控制

#### 重复创建位置:
1. `db/enhanced_connection_pool.py:1940` - `get_connection_pool()`全局函数
2. `db/managers/connection_manager.py:148` - `_initialize_pool_Connection_Manager()`
3. `db/data_manager_adapter.py:59` - `initialize_connection_pool()`调用
4. `db/unified_data_manager.py:76` - 另一个`initialize_connection_pool()`调用
5. `monitoring/clickhouse_performance_monitor.py:147` - 监控模块初始化
6. `api/main.py:118` - API启动时直接创建

#### 修复计划:
- [ ] 保留唯一入口`get_connection_pool()`
- [ ] 移除其他5个创建点
- [ ] 确保真正的单例模式

## 🔧 架构问题

### 问题3: 多重容器系统冲突
**问题描述**: 系统中存在3个不同的依赖注入容器实现
**影响级别**: 🟡 中等 - 导致服务注册混乱
**状态**: 🔄 分析中

#### 冲突的容器:
1. **`utils/unified_container.py`** - 正确的统一容器实现
   - 使用Type作为键
   - 支持单例和瞬态生命周期
   - 设计清晰

2. **`db/container.py`** - 数据库专用容器（有问题）
   - 方法名被污染（如`register_singleton_Container`）
   - 语法错误和命名冲突
   - 需要废弃

3. **`utils/optimized_dependency_injection.py`** - 优化容器实现
   - 使用字符串作为键
   - 与Type-based容器冲突
   - 需要整合

#### 修复计划:
- [ ] 选择`utils/unified_container.py`作为唯一容器
- [ ] 废弃`db/container.py`
- [ ] 整合`utils/optimized_dependency_injection.py`功能

### 问题4: L2/L3层边界模糊
**问题描述**: 存储访问层和数据服务层职责混乱
**影响级别**: 🟡 中等 - 违反六层架构原则
**状态**: 🔄 分析中

#### 边界违规:
1. `db/enhanced_connection_pool.py` (L2) 包含业务逻辑
2. `db/managers/data_access_manager.py` (L3) 直接创建连接池
3. `indicators/` (L4) 有时直接访问数据库

#### 修复计划:
- [ ] 清理L2层业务逻辑
- [ ] 明确L3层职责
- [ ] 修复跨层调用

### 问题5: 循环依赖
**问题描述**: 模块间存在循环依赖
**影响级别**: 🟡 中等 - 影响模块加载和测试
**状态**: 🔄 分析中

#### 循环依赖链:
```
enhanced_connection_pool.py → database_config_manager.py → container → data_access_manager.py → enhanced_connection_pool.py
```

#### 修复计划:
- [ ] 重构依赖关系
- [ ] 使用依赖注入打破循环
- [ ] 延迟导入关键模块

## 🐛 语法和实现问题

### 问题6: 容器方法名污染
**问题描述**: `db/container.py`中的方法名被污染
**影响级别**: 🟡 中等 - 导致方法调用失败
**状态**: 🔄 待修复

#### 污染的方法:
- `register_singleton_Container()` 应该是 `register_singleton()`
- `resolve_Container_Container_Container_1_container()` 应该是 `resolve()`
- `is_registered_Container_Container_Container_1_container()` 应该是 `is_registered()`

#### 修复计划:
- [ ] 重命名所有污染的方法
- [ ] 更新所有调用点
- [ ] 验证功能正常

### 问题7: 58个指标语法错误
**问题描述**: 大量指标存在语法错误导致注册失败
**影响级别**: 🟡 中等 - 影响指标注册成功率
**状态**: 🔄 部分修复

#### 主要错误类型:
1. **ZXM指标系列**: `zxm_abstract_methods_mixin.py:120` - 变量赋值语法错误
2. **成交量指标**: OBV, VOL, VR, VOSC, PVT, CHAIKIN, FORCE_INDEX
3. **形态识别指标**: V_SHAPED_REVERSAL, RECTANGLE等
4. **增强指标**: ENHANCED_MACD, ELLIOTT_WAVE, MTM, COMPOSITE

#### 修复计划:
- [x] 修复ZXM指标部分语法错误
- [x] 修复OBV和VOL指标语法错误
- [ ] 修复剩余56个指标语法错误

### 问题8: Score指标抽象方法缺失
**问题描述**: Score指标无法实例化，缺少抽象方法实现
**影响级别**: 🟡 中等 - 影响新创建的Score指标
**状态**: 🔄 待修复

#### 受影响的指标:
- MACDScoreIndicator
- RSIScoreIndicator  
- BOLLScoreIndicator
- KDJScoreIndicator

#### 修复计划:
- [ ] 实现缺失的抽象方法
- [ ] 验证Score指标可正常实例化
- [ ] 测试Score指标计算功能

## 📊 数据问题

### 问题9: 数据覆盖率不足
**问题描述**: 只有50%的周期有数据（3/6周期）
**影响级别**: 🟡 中等 - 影响多周期分析
**状态**: 🔄 分析中

#### 缺失的周期:
- 30分钟数据：需要从15分钟聚合
- 60分钟数据：需要从15分钟聚合

#### 修复计划:
- [ ] 验证数据聚合功能
- [ ] 测试30分钟/60分钟数据生成
- [ ] 确保100%数据覆盖率

### 问题10: 数据库字段兼容性
**问题描述**: 查询引用不存在的字段
**影响级别**: 🟢 低 - 已修复
**状态**: ✅ 已完成

#### 已修复:
- [x] 移除price_change字段引用
- [x] 移除price_range字段引用  
- [x] 移除industry字段引用
- [x] 修复96个文件，159处修复

## 🧪 测试问题

### 问题11: 基础功能测试失败
**问题描述**: 6个基础功能测试全部失败（0/6通过）
**影响级别**: 🔴 严重 - 表明系统基础功能不可用
**状态**: 🔄 待修复

#### 失败的测试:
1. 容器初始化测试
2. 数据库连接测试
3. 指标注册测试
4. 数据查询测试
5. 数据聚合测试
6. 股票分析测试

#### 修复计划:
- [ ] 修复容器问题后重新测试
- [ ] 逐个验证基础功能
- [ ] 确保≥80%测试通过率

## 📈 修复进度统计

### 已完成 ✅
- [x] 深度问题分析（100%）
- [x] 数据库字段兼容性修复（100%）
- [x] 废弃入口文件清理（100%）
- [x] 部分语法错误修复（3.4% - 2/58个指标）

### 进行中 🔄  
- [ ] 容器键重复问题修复（0%）
- [ ] 连接池单例化（0%）
- [ ] 架构边界清理（0%）
- [ ] 语法错误修复（3.4%）

### 待开始 ⏳
- [ ] 数据聚合功能验证（0%）
- [ ] 完整股票分析测试（0%）
- [ ] 生产级稳定性验证（0%）

## 🎯 总体修复目标

### 当前状态
- **指标注册成功率**: 54.7% (70/128)
- **基础功能测试通过率**: 0% (0/6)
- **数据覆盖率**: 50% (3/6周期)
- **系统稳定性**: 有崩溃风险

### 目标状态
- **指标注册成功率**: ≥90% (≥115/128)
- **基础功能测试通过率**: ≥80% (≥5/6)
- **数据覆盖率**: 100% (6/6周期)
- **系统稳定性**: 无崩溃，完整执行

### 关键里程碑
1. **第1天**: 解决容器键重复问题，恢复基础功能
2. **第2天**: 完成连接池单例化，提升系统稳定性
3. **第3天**: 修复语法错误，达到90%指标注册成功率
4. **第4天**: 完整验证，确保生产级质量
