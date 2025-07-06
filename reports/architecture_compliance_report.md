# 架构合规性检查报告

❌ 发现 287 个架构违规问题：

## Layer Violations (8个)

1. **文件**: db/interfaces/data_access_interface.py
   **违规**: L2层文件不能直接导入L5层模块

2. **文件**: enums/query_example.py
   **违规**: L1层文件不能直接导入L2层模块

3. **文件**: utils/dependency_injection.py
   **违规**: L1层文件不能直接导入L2层模块

4. **文件**: utils/dependency_injection.py
   **违规**: L1层文件不能直接导入L2层模块

5. **文件**: utils/dependency_injection.py
   **违规**: L1层文件不能直接导入L2层模块

6. **文件**: utils/scoring_validator.py
   **违规**: L1层文件不能直接导入L2层模块

7. **文件**: utils/period_manager.py
   **违规**: L1层文件不能直接导入L2层模块

8. **文件**: utils/strategy_validator.py
   **违规**: L1层文件不能直接导入L3层模块

## Direct Db Dependencies (15个)

1. **文件**: scripts/utils/fix_layer_violations.py
   **行号**: 265
   **内容**: `r'from\s+db\.clickhouse_db\s+import\s+get_clickhouse_db',`
   **违规**: 禁止直接依赖数据库实现: get_clickhouse_db

2. **文件**: scripts/utils/fix_layer_violations.py
   **行号**: 272
   **内容**: `r'get_clickhouse_db\(\)',`
   **违规**: 禁止直接依赖数据库实现: get_clickhouse_db

3. **文件**: scripts/utils/fix_layer_violations.py
   **行号**: 293
   **内容**: `r'from\s+db\.clickhouse_db\s+import\s+get_clickhouse_db',`
   **违规**: 禁止直接依赖数据库实现: get_clickhouse_db

4. **文件**: scripts/utils/batch_architecture_fix.py
   **行号**: 52
   **内容**: `r"from\s+db\.clickhouse_db\s+import\s+get_clickhouse_db":`
   **违规**: 禁止直接依赖数据库实现: get_clickhouse_db

5. **文件**: scripts/utils/batch_architecture_fix.py
   **行号**: 56
   **内容**: `r"get_clickhouse_db\(\)":`
   **违规**: 禁止直接依赖数据库实现: get_clickhouse_db

6. **文件**: scripts/utils/batch_architecture_fix.py
   **行号**: 58
   **内容**: `r"get_clickhouse_db\(.*?\)":`
   **违规**: 禁止直接依赖数据库实现: get_clickhouse_db

7. **文件**: scripts/utils/refactor_database_dependencies.py
   **行号**: 55
   **内容**: `r'db\s*=\s*get_clickhouse_db\(\)',`
   **违规**: 禁止直接依赖数据库实现: get_clickhouse_db

8. **文件**: scripts/utils/refactor_database_dependencies.py
   **行号**: 56
   **内容**: `r'self\.db\s*=\s*get_clickhouse_db\(\)',`
   **违规**: 禁止直接依赖数据库实现: get_clickhouse_db

9. **文件**: scripts/utils/refactor_database_dependencies.py
   **行号**: 57
   **内容**: `r'cls\.db\s*=\s*get_clickhouse_db\(\)',`
   **违规**: 禁止直接依赖数据库实现: get_clickhouse_db

10. **文件**: scripts/utils/refactor_database_dependencies.py
   **行号**: 58
   **内容**: `r'get_clickhouse_db\(\)'`
   **违规**: 禁止直接依赖数据库实现: get_clickhouse_db

   ... 还有 5 个类似违规

## Code Duplication (125个)

1. **文件**: N/A
   **违规**: 发现重复的类/方法名: __init__

2. **文件**: N/A
   **违规**: 发现重复的类/方法名: main

3. **文件**: N/A
   **违规**: 发现重复的类/方法名: get

4. **文件**: N/A
   **违规**: 发现重复的类/方法名: set

5. **文件**: N/A
   **违规**: 发现重复的类/方法名: advance_Processor_Complex_Logic_Processor_Complex_Logic_Processor

6. **文件**: N/A
   **违规**: 发现重复的类/方法名: _fetch_trading_calendar_from_db_Date_Manager_Date_Manager

7. **文件**: N/A
   **违规**: 发现重复的类/方法名: get_index_data_Api_Stock_Data_Api_Stock_Data_Api

8. **文件**: N/A
   **违规**: 发现重复的类/方法名: get_stock_list_Api_Stock_Data_Api_Stock_Data_Api

9. **文件**: N/A
   **违规**: 发现重复的类/方法名: get_stock_data_Api_Stock_Data_Api_Stock_Data_Api

10. **文件**: N/A
   **违规**: 发现重复的类/方法名: get_market_data_Api_Stock_Data_Api_Stock_Data_Api

   ... 还有 115 个类似违规

## Naming Violations (135个)

1. **文件**: analysis/engines/indicator_validation_framework.py
   **违规**: 类名应使用大驼峰命名法

2. **文件**: analysis/market/multi_dimension_analyzer.py
   **违规**: 类名应使用大驼峰命名法

3. **文件**: db/container.py
   **违规**: 类名应使用大驼峰命名法

4. **文件**: db/data_manager.py
   **违规**: 类名应使用大驼峰命名法

5. **文件**: db/enhanced_data_manager.py
   **违规**: 类名应使用大驼峰命名法

6. **文件**: db/data_manager_adapter.py
   **违规**: 类名应使用大驼峰命名法

7. **文件**: db/data_manager_adapter.py
   **违规**: 类名应使用大驼峰命名法

8. **文件**: db/batch_data_optimizer.py
   **违规**: 类名应使用大驼峰命名法

9. **文件**: db/managers/cache_manager.py
   **违规**: 类名应使用大驼峰命名法

10. **文件**: db/interfaces/data_access_interface.py
   **违规**: 类名应使用大驼峰命名法

   ... 还有 125 个类似违规

## Database Query Violations (4个)

1. **文件**: db/sql_manager.py
   **行号**: 72
   **内容**: `WHERE date = (SELECT MAX(date) FROM stock_info)`
   **违规**: 查询stock_info表必须包含WHERE条件

2. **文件**: db/sql_manager.py
   **行号**: 84
   **内容**: `SELECT *`
   **违规**: 禁止使用SELECT code, name, date, level, open, close, high, low, volume

3. **文件**: db/sql_manager.py
   **行号**: 114
   **内容**: `SELECT *`
   **违规**: 禁止使用SELECT code, name, date, level, open, close, high, low, volume

4. **文件**: db/sql_manager.py
   **行号**: 132
   **内容**: `FROM stock_info si`
   **违规**: 查询stock_info表必须包含WHERE条件
