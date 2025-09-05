# 架构合规性验证报告

**验证时间**: 2025-09-05 16:58:05
**验证文件数**: 11315
**违规总数**: 30
**错误数**: 30
**警告数**: 0
**合规状态**: ❌ 不合规

## 违规详情

### bin/high_performance_stock_select.py

🔴 **第54行** - FORBIDDEN_IMPORT
   - **问题**: 禁止直接导入数据库实现: from db.clickhouse_db import get_clickhouse_db
   - **建议**: 使用依赖注入或接口导入

### config/service_registration.py

🔴 **第100行** - FORBIDDEN_IMPORT
   - **问题**: 禁止直接导入连接池: from db.enhanced_connection_pool import ClickHouseConnectionPool
   - **建议**: 使用依赖注入或接口导入

🔴 **第128行** - LAYER_VIOLATION
   - **问题**: 违反分层架构: L1_基础设施层 不能导入 L3_数据服务层
   - **建议**: 通过依赖注入或使用 L1_基础设施层 允许的层

🔴 **第129行** - LAYER_VIOLATION
   - **问题**: 违反分层架构: L1_基础设施层 不能导入 L3_数据服务层
   - **建议**: 通过依赖注入或使用 L1_基础设施层 允许的层

🔴 **第142行** - LAYER_VIOLATION
   - **问题**: 违反分层架构: L1_基础设施层 不能导入 L3_数据服务层
   - **建议**: 通过依赖注入或使用 L1_基础设施层 允许的层

🔴 **第151行** - LAYER_VIOLATION
   - **问题**: 违反分层架构: L1_基础设施层 不能导入 L3_数据服务层
   - **建议**: 通过依赖注入或使用 L1_基础设施层 允许的层

🔴 **第152行** - LAYER_VIOLATION
   - **问题**: 违反分层架构: L1_基础设施层 不能导入 L3_数据服务层
   - **建议**: 通过依赖注入或使用 L1_基础设施层 允许的层

🔴 **第174行** - LAYER_VIOLATION
   - **问题**: 违反分层架构: L1_基础设施层 不能导入 L4_核心服务层
   - **建议**: 通过依赖注入或使用 L1_基础设施层 允许的层

🔴 **第186行** - LAYER_VIOLATION
   - **问题**: 违反分层架构: L1_基础设施层 不能导入 L4_核心服务层
   - **建议**: 通过依赖注入或使用 L1_基础设施层 允许的层

🔴 **第193行** - LAYER_VIOLATION
   - **问题**: 违反分层架构: L1_基础设施层 不能导入 L4_核心服务层
   - **建议**: 通过依赖注入或使用 L1_基础设施层 允许的层

🔴 **第214行** - LAYER_VIOLATION
   - **问题**: 违反分层架构: L1_基础设施层 不能导入 L5_业务应用层
   - **建议**: 通过依赖注入或使用 L1_基础设施层 允许的层

🔴 **第227行** - LAYER_VIOLATION
   - **问题**: 违反分层架构: L1_基础设施层 不能导入 L5_业务应用层
   - **建议**: 通过依赖注入或使用 L1_基础设施层 允许的层

🔴 **第255行** - LAYER_VIOLATION
   - **问题**: 违反分层架构: L1_基础设施层 不能导入 L6_用户接口层
   - **建议**: 通过依赖注入或使用 L1_基础设施层 允许的层

### config/service_initializer.py

🔴 **第94行** - LAYER_VIOLATION
   - **问题**: 违反分层架构: L1_基础设施层 不能导入 L3_数据服务层
   - **建议**: 通过依赖注入或使用 L1_基础设施层 允许的层

🔴 **第101行** - LAYER_VIOLATION
   - **问题**: 违反分层架构: L1_基础设施层 不能导入 L3_数据服务层
   - **建议**: 通过依赖注入或使用 L1_基础设施层 允许的层

🔴 **第27行** - LAYER_VIOLATION
   - **问题**: 违反分层架构: L1_基础设施层 不能导入 L3_数据服务层
   - **建议**: 通过依赖注入或使用 L1_基础设施层 允许的层

🔴 **第41行** - LAYER_VIOLATION
   - **问题**: 违反分层架构: L1_基础设施层 不能导入 L3_数据服务层
   - **建议**: 通过依赖注入或使用 L1_基础设施层 允许的层

🔴 **第42行** - LAYER_VIOLATION
   - **问题**: 违反分层架构: L1_基础设施层 不能导入 L4_核心服务层
   - **建议**: 通过依赖注入或使用 L1_基础设施层 允许的层

### utils/dependency_injection.py

🔴 **第49行** - FORBIDDEN_IMPORT
   - **问题**: 禁止直接导入数据库实现: from db.clickhouse_db import ClickHouseDB
   - **建议**: 使用依赖注入或接口导入

### utils/compatibility.py

🔴 **第36行** - LAYER_VIOLATION
   - **问题**: 违反分层架构: L1_基础设施层 不能导入 L3_数据服务层
   - **建议**: 通过依赖注入或使用 L1_基础设施层 允许的层

### db/managers/data_access_manager.py

🔴 **第1299行** - FORBIDDEN_IMPORT
   - **问题**: 禁止直接导入数据库实现: from db.clickhouse_db import get_clickhouse_db
   - **建议**: 使用依赖注入或接口导入

🔴 **第1335行** - FORBIDDEN_IMPORT
   - **问题**: 禁止直接导入数据库实现: from db.clickhouse_db import get_clickhouse_db
   - **建议**: 使用依赖注入或接口导入

🔴 **第1380行** - FORBIDDEN_IMPORT
   - **问题**: 禁止直接导入数据库实现: from db.clickhouse_db import get_clickhouse_db
   - **建议**: 使用依赖注入或接口导入

🔴 **第1482行** - FORBIDDEN_IMPORT
   - **问题**: 禁止直接导入数据库实现: from db.clickhouse_db import get_clickhouse_db
   - **建议**: 使用依赖注入或接口导入

🔴 **第1515行** - FORBIDDEN_IMPORT
   - **问题**: 禁止直接导入数据库实现: from db.clickhouse_db import get_clickhouse_db
   - **建议**: 使用依赖注入或接口导入

### strategy/implementations/diagnose_kdj_strategy.py

🔴 **第54行** - FORBIDDEN_IMPORT
   - **问题**: 禁止直接导入数据库实现: from db.clickhouse_db import get_clickhouse_db
   - **建议**: 使用依赖注入或接口导入

### strategy/implementations/execute_kdj_upward_strategy.py

🔴 **第159行** - FORBIDDEN_IMPORT
   - **问题**: 禁止直接导入数据库实现: from db.clickhouse_db import get_clickhouse_db
   - **建议**: 使用依赖注入或接口导入

🔴 **第250行** - FORBIDDEN_IMPORT
   - **问题**: 禁止直接导入数据库实现: from db.clickhouse_db import get_clickhouse_db
   - **建议**: 使用依赖注入或接口导入

### strategy/implementations/full_zxm_absorb_query_20250512.py

🔴 **第38行** - FORBIDDEN_IMPORT
   - **问题**: 禁止直接导入数据库实现: from db.clickhouse_db import get_clickhouse_db
   - **建议**: 使用依赖注入或接口导入

🔴 **第97行** - FORBIDDEN_IMPORT
   - **问题**: 禁止直接导入数据库实现: from db.clickhouse_db import get_clickhouse_db
   - **建议**: 使用依赖注入或接口导入
