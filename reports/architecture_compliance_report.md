# 架构合规性检查报告

❌ 发现 581 个架构违规问题：

## Layer Violations (5个)

1. **文件**: enums/query_example.py
   **违规**: L1层文件不能直接导入L4层模块

2. **文件**: utils/scoring_validator.py
   **违规**: L1层文件不能直接导入L4层模块

3. **文件**: scripts/utils/data_sync.py
   **违规**: L1层文件不能直接导入L4层模块

4. **文件**: scripts/utils/data_sync.py
   **违规**: L1层文件不能直接导入L4层模块

5. **文件**: scripts/utils/optimize_performance.py
   **违规**: L1层文件不能直接导入L5层模块

## Direct Db Dependencies (122个)

1. **文件**: analysis/multi_dimension_analyzer.py
   **行号**: 16
   **内容**: `from db.clickhouse_db import get_clickhouse_db, get_default_config`
   **违规**: 禁止直接依赖数据库实现: from db.clickhouse_db import

2. **文件**: analysis/multi_dimension_analyzer.py
   **行号**: 16
   **内容**: `from db.clickhouse_db import get_clickhouse_db, get_default_config`
   **违规**: 禁止直接依赖数据库实现: get_clickhouse_db

3. **文件**: analysis/multi_dimension_analyzer.py
   **行号**: 40
   **内容**: `self.ch_db = get_clickhouse_db(config=config)`
   **违规**: 禁止直接依赖数据库实现: get_clickhouse_db

4. **文件**: analysis/buypoints/analyze_buypoints.py
   **行号**: 6
   **内容**: `from db.clickhouse_db import get_clickhouse_db, get_default_config`
   **违规**: 禁止直接依赖数据库实现: from db.clickhouse_db import

5. **文件**: analysis/buypoints/analyze_buypoints.py
   **行号**: 6
   **内容**: `from db.clickhouse_db import get_clickhouse_db, get_default_config`
   **违规**: 禁止直接依赖数据库实现: get_clickhouse_db

6. **文件**: analysis/buypoints/analyze_buypoints.py
   **行号**: 25
   **内容**: `# 使用get_clickhouse_db函数获取ClickHouseDB实例`
   **违规**: 禁止直接依赖数据库实现: get_clickhouse_db

7. **文件**: analysis/buypoints/analyze_buypoints.py
   **行号**: 26
   **内容**: `self.ch_db = get_clickhouse_db(config=config)`
   **违规**: 禁止直接依赖数据库实现: get_clickhouse_db

8. **文件**: analysis/buypoints/buypoint_dimension_analyzer.py
   **行号**: 17
   **内容**: `from db.clickhouse_db import get_clickhouse_db, get_default_config`
   **违规**: 禁止直接依赖数据库实现: from db.clickhouse_db import

9. **文件**: analysis/buypoints/buypoint_dimension_analyzer.py
   **行号**: 17
   **内容**: `from db.clickhouse_db import get_clickhouse_db, get_default_config`
   **违规**: 禁止直接依赖数据库实现: get_clickhouse_db

10. **文件**: analysis/buypoints/buypoint_dimension_analyzer.py
   **行号**: 39
   **内容**: `self.ch_db = get_clickhouse_db(config=config)`
   **违规**: 禁止直接依赖数据库实现: get_clickhouse_db

   ... 还有 112 个类似违规

## Code Duplication (311个)

1. **文件**: N/A
   **违规**: 发现重复的类/方法名: main

2. **文件**: N/A
   **违规**: 发现重复的类/方法名: run_comprehensive_analysis

3. **文件**: N/A
   **违规**: 发现重复的类/方法名: test_single_indicator

4. **文件**: N/A
   **违规**: 发现重复的类/方法名: test_multiple_indicators

5. **文件**: N/A
   **违规**: 发现重复的类/方法名: list_available_indicators

6. **文件**: N/A
   **违规**: 发现重复的类/方法名: start_monitoring

7. **文件**: N/A
   **违规**: 发现重复的类/方法名: analyze_patterns

8. **文件**: N/A
   **违规**: 发现重复的类/方法名: generate_report

9. **文件**: N/A
   **违规**: 发现重复的类/方法名: ValidationResult

10. **文件**: N/A
   **违规**: 发现重复的类/方法名: validate_all_patterns

   ... 还有 301 个类似违规

## Naming Violations (123个)

1. **文件**: indicators/intraday_volatility.py
   **违规**: 类名应使用大驼峰命名法

2. **文件**: indicators/zxm_absorb.py
   **违规**: 类名应使用大驼峰命名法

3. **文件**: indicators/kdj_score.py
   **违规**: 类名应使用大驼峰命名法

4. **文件**: indicators/island_reversal.py
   **违规**: 类名应使用大驼峰命名法

5. **文件**: indicators/score_manager.py
   **违规**: 类名应使用大驼峰命名法

6. **文件**: indicators/platform_breakout.py
   **违规**: 类名应使用大驼峰命名法

7. **文件**: indicators/boll_score.py
   **违规**: 类名应使用大驼峰命名法

8. **文件**: indicators/fibonacci_tools.py
   **违规**: 类名应使用大驼峰命名法

9. **文件**: indicators/zxm_washplate.py
   **违规**: 类名应使用大驼峰命名法

10. **文件**: indicators/composite_indicator.py
   **违规**: 类名应使用大驼峰命名法

   ... 还有 113 个类似违规

## Import Violations (3个)

1. **文件**: analysis/buypoints/analyze_buypoints.py
   **行号**: 8
   **内容**: `from enums.indicators import *`
   **违规**: 禁止使用通配符导入

2. **文件**: analysis/market/a_stock_market_analysis.py
   **行号**: 10
   **内容**: `from enums.indicators import *`
   **违规**: 禁止使用通配符导入

3. **文件**: formula/stock_formula.py
   **行号**: 2
   **内容**: `from enums.indicators import *`
   **违规**: 禁止使用通配符导入

## Database Query Violations (17个)

1. **文件**: debug_stock_count.py
   **行号**: 60
   **内容**: `result = conn.query_dataframe("SELECT COUNT(*) as total FROM stock_info")`
   **违规**: 查询stock_info表必须包含WHERE条件

2. **文件**: debug_stock_count.py
   **行号**: 64
   **内容**: `result2 = conn.query_dataframe("SELECT COUNT(DISTINCT code) as unique_stocks FROM stock_info")`
   **违规**: 查询stock_info表必须包含WHERE条件

3. **文件**: analysis/strategy_comparison.py
   **行号**: 148
   **内容**: `sql = "SELECT DISTINCT stock_code FROM stock_info"`
   **违规**: 查询stock_info表必须包含WHERE条件

4. **文件**: tests/mocks/db_mock.py
   **行号**: 165
   **内容**: `sql = "SELECT * FROM stock_list"`
   **违规**: 禁止使用SELECT *

5. **文件**: tests/mocks/db_mock.py
   **行号**: 196
   **内容**: `sql = f"SELECT * FROM kline_{period} WHERE code = '{stock_code}'"`
   **违规**: 禁止使用SELECT *

6. **文件**: tests/mocks/db_mock.py
   **行号**: 233
   **内容**: `sql = f"SELECT * FROM indicator_{indicator_name} WHERE code = '{stock_code}'"`
   **违规**: 禁止使用SELECT *

7. **文件**: scripts/check_stock_info_columns.py
   **行号**: 32
   **内容**: `columns_query = "SELECT * FROM stock_info LIMIT 1"`
   **违规**: 禁止使用SELECT *

8. **文件**: scripts/check_stock_info_columns.py
   **行号**: 42
   **内容**: `sample_query = "SELECT * FROM stock_info WHERE code = '000001' ORDER BY date DESC LIMIT 3"`
   **违规**: 禁止使用SELECT *

9. **文件**: scripts/database_optimization.py
   **行号**: 248
   **内容**: `table_settings = self.client.query("SELECT * FROM system.tables WHERE name = 'stock_info' AND database = 'stock'")`
   **违规**: 禁止使用SELECT *

10. **文件**: scripts/database_optimization.py
   **行号**: 261
   **内容**: `date_range = self.client.query("SELECT MIN(date) as min_date, MAX(date) as max_date FROM stock_info")`
   **违规**: 查询stock_info表必须包含WHERE条件

   ... 还有 7 个类似违规
