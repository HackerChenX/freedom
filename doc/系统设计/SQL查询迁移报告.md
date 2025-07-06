# SQL查询迁移报告

**生成时间**: 2025年 7月 6日 星期日 18时05分16秒 CST

## 📊 迁移统计

- **总文件数**: 10
- **总查询数**: 54
- **已迁移**: 0

## 📝 发现的SQL查询

### debug_stock_count.py

**查询 1** (第60行, 类别: stock_data)
```sql
SELECT COUNT(*) as total FROM stock_info WHERE date >=
```

**查询 2** (第64行, 类别: stock_data)
```sql
SELECT COUNT(DISTINCT code) as unique_stocks FROM stock_info WHERE date >=
```

**查询 3** (第60行, 类别: stock_data)
```sql
SELECT COUNT(*) as total FROM stock_info WHERE date >=
```

**查询 4** (第64行, 类别: stock_data)
```sql
SELECT COUNT(DISTINCT code) as unique_stocks FROM stock_info WHERE date >=
```

### analysis/strategy_comparison.py

**查询 1** (第148行, 类别: stock_data)
```sql
SELECT DISTINCT stock_code FROM stock_info WHERE date >=
```

**查询 2** (第148行, 类别: stock_data)
```sql
SELECT DISTINCT stock_code FROM stock_info WHERE date >=
```

### bin/strategy_generator.py

**查询 1** (第242行, 类别: metadata)
```sql
INSERT INTO {table_name} (code, buy_date, pattern_type) VALUES
```

### bin/stock_analysis.py

**查询 1** (第106行, 类别: stock_data)
```sql
SELECT code, name FROM stock_info WHERE industry =
```

**查询 2** (第106行, 类别: stock_data)
```sql
SELECT code, name FROM stock_info WHERE industry =
```

### tests/end_to_end/enhanced_real_data_test.py

**查询 1** (第95行, 类别: stock_data)
```sql
SELECT DISTINCT code, name FROM stock_info WHERE date >= 2020-01-01
```

**查询 2** (第118行, 类别: metadata)
```sql
SELECT OHLCV data for recent period
```

**查询 3** (第148行, 类别: metadata)
```sql
SELECT aggregated statistics
```

**查询 4** (第176行, 类别: metadata)
```sql
SELECT with window functions and calculations
```

**查询 5** (第95行, 类别: stock_data)
```sql
SELECT DISTINCT code, name FROM stock_info WHERE date >= 2020-01-01
```

**查询 6** (第118行, 类别: stock_data)
```sql
SELECT OHLCV data for recent period'
            })
            
            logger.info(f"最近数据查询完成: {recent_records}条记录，耗时: {query_time:.3f}秒")
            
            # 3. 测试聚合查询
            query_start = time.time()
            agg_query = f"""
                SELECT 
                    code,
                    COUNT(*) as record_count,
                    AVG(close) as avg_price,
                    MAX(high) as max_price,
                    MIN(low) as min_price,
                    SUM(volume) as total_volume
                FROM stock_info WHERE 1=1
                WHERE date >=
```

**查询 7** (第148行, 类别: stock_data)
```sql
SELECT aggregated statistics'
            })
            
            logger.info(f"聚合查询完成: {agg_records}条记录，耗时: {query_time:.3f}秒")
            
            # 4. 测试复杂条件查询
            query_start = time.time()
            complex_query = f"""
                SELECT
                    code, date, close, volume,
                    (close - open) / open as daily_change,
                    volume / 1000000 as volume_millions
                FROM stock_info WHERE 1=1
                WHERE date >=
```

**查询 8** (第176行, 类别: stock_data)
```sql
SELECT with window functions and calculations'
            })
            
            logger.info(f"复杂查询完成: {complex_records}条记录，耗时: {query_time:.3f}秒")
            
            # 计算总体统计
            total_time = time.time() - start_time
            total_records = sum(q['records'] for q in performance_results['queries'])
            avg_query_time = np.mean([q['duration'] for q in performance_results['queries']])
            
            performance_results.update({
                'total_time': total_time,
                'total_records': total_records,
                'avg_query_time': avg_query_time,
                'queries_per_second': len(performance_results['queries']) / total_time,
                'records_per_second': total_records / total_time
            })
            
            # 生成优化建议
            performance_results['optimization_suggestions'] = self._analyze_database_performance(performance_results)
            
            logger.info(f"数据库性能测试完成，总耗时: {total_time:.3f}秒，平均查询时间: {avg_query_time:.3f}秒")
            
            return performance_results
            
        except Exception as e:
            logger.error(f"数据库性能测试失败: {e}")
            return {'error': str(e)}
    
    def _analyze_database_performance(self, results: Dict[str, Any]) -> List[str]:
        """分析数据库性能并生成优化建议"""
        suggestions = []
        
        # 分析查询时间
        slow_queries = [q for q in results['queries'] if q['duration'] > 2.0]
        if slow_queries:
            suggestions.append(f"发现 {len(slow_queries)} 个慢查询（>2秒），建议优化查询语句或添加索引")
        
        # 分析平均查询时间
        if results['avg_query_time'] > 1.0:
            suggestions.append("平均查询时间较长，建议考虑以下优化：")
            suggestions.append("- 为常用查询字段（date, code）添加索引")
            suggestions.append("- 考虑按日期分区表")
            suggestions.append("- 优化WHERE条件的顺序")
        
        # 分析数据量
        if results['total_records'] > 100000:
            suggestions.append("查询返回大量数据，建议：")
            suggestions.append("- 使用LIMIT限制返回记录数")
            suggestions.append("- 实施分页查询")
            suggestions.append("- 考虑数据预聚合")
        
        # 分析复杂查询
        complex_queries = [q for q in results['queries'] if q['query_type'] == 'complex_analysis']
        if complex_queries and complex_queries[0]['duration'] > 3.0:
            suggestions.append("复杂分析查询耗时较长，建议：")
            suggestions.append("- 预计算技术指标并存储")
            suggestions.append("- 使用物化视图")
            suggestions.append("- 考虑异步计算")
        
        return suggestions
    
    def test_concurrent_database_access(self, concurrent_count: int = 5) -> Dict[str, Any]:
        """测试并发数据库访问"""
        logger.info(f"开始并发数据库访问测试，并发数: {concurrent_count}")
        
        if not self.client:
            return {'error': 'ClickHouse连接未建立'}
        
        import concurrent.futures
        import threading
        
        results = {
            'concurrent_count': concurrent_count,
            'individual_results': [],
            'total_time': 0,
            'success_count': 0,
            'error_count': 0,
            'avg_response_time': 0,
            'max_response_time': 0,
            'min_response_time': float('inf')
        }
        
        def execute_concurrent_query(query_id: int) -> Dict[str, Any]:
            """执行单个并发查询"""
            thread_start = time.time()
            
            try:
                # 每个线程执行不同的查询以模拟真实场景
                if query_id % 3 == 0:
                    query = f"""
                        SELECT code, close, volume 
                        FROM stock_info WHERE 1=1
                        WHERE date =
```

### scripts/simple_clickhouse_test.py

**查询 1** (第80行, 类别: analysis)
```sql
SELECT COUNT(*) FROM {config[
```

**查询 2** (第49行, 类别: metadata)
```sql
SELECT 1")
        print(f"查询结果: {result}")
        
        # 获取数据库列表
        print("获取数据库列表...")
        databases = client.execute("SHOW DATABASES")
        print("可用的数据库:")
        for db in databases:
            print(f"- {db[0]}")
        
        # 如果指定了数据库，尝试获取表列表
        if 'database' in config:
            print(f"\n获取数据库 '{config['database']}' 的表列表...")
            try:
                tables = client.execute(f"SHOW TABLES FROM {config[
```

**查询 3** (第80行, 类别: analysis)
```sql
SELECT COUNT(*) FROM {config[
```

**查询 4** (第54行, 类别: metadata)
```sql
SHOW DATABASES
```

**查询 5** (第63行, 类别: metadata)
```sql
SHOW TABLES FROM {config[
```

**查询 6** (第72行, 类别: metadata)
```sql
DESCRIBE TABLE {config[
```

### scripts/production_database_test.py

**查询 1** (第195行, 类别: analysis)
```sql
SELECT COUNT(*) as count FROM system LIMIT 1000.databases
```

**查询 2** (第239行, 类别: analysis)
```sql
SELECT COUNT(*) as count FROM {table} LIMIT 1
```

**查询 3** (第273行, 类别: stock_data)
```sql
SELECT code, name FROM stock_info WHERE date >=
```

**查询 4** (第195行, 类别: analysis)
```sql
SELECT COUNT(*) as count FROM system LIMIT 1000.databases
```

**查询 5** (第239行, 类别: analysis)
```sql
SELECT COUNT(*) as count FROM {table} LIMIT 1
```

**查询 6** (第273行, 类别: stock_data)
```sql
SELECT code, name FROM stock_info WHERE date >=
```

### scripts/clickhouse_connection_summary.py

**查询 1** (第32行, 类别: metadata)
```sql
SELECT 1 AS test
```

**查询 2** (第55行, 类别: analysis)
```sql
SELECT COUNT(*) FROM stock LIMIT 1000.{table_name}
```

**查询 3** (第32行, 类别: metadata)
```sql
SELECT 1 AS test")
        print(f"   ✓ 查询成功: {result}")
        
        print("\n3. 获取数据库列表...")
        databases = client.execute("SHOW DATABASES")
        print("   ✓ 数据库列表:")
        for db in databases:
            print(f"     - {db[0]}")
        
        print("\n4. 检查stock数据库...")
        if 'stock' in [db[0] for db in databases]:
            print("   ✓ stock数据库存在")
            
            print("\n5. 获取stock数据库中的表...")
            tables = client.execute("SHOW TABLES FROM stock LIMIT 1000
```

**查询 4** (第55行, 类别: analysis)
```sql
SELECT COUNT(*) FROM stock LIMIT 1000.{table_name}
```

**查询 5** (第36行, 类别: metadata)
```sql
SHOW DATABASES
```

**查询 6** (第46行, 类别: metadata)
```sql
SHOW TABLES FROM stock LIMIT 1000
```

### scripts/database_optimization.py

**查询 1** (第119行, 类别: stock_data)
```sql
SELECT code, date, close FROM stock_info WHERE date >=
```

**查询 2** (第123行, 类别: stock_data)
```sql
SELECT code, COUNT(*) as cnt FROM stock_info WHERE date >=
```

**查询 3** (第127行, 类别: stock_data)
```sql
SELECT code, date, close, volume FROM stock_info WHERE close > 20 AND volume > 1000000 AND date >=
```

**查询 4** (第248行, 类别: metadata)
```sql
SELECT code, name, date, level, open, close, high, low, volume FROM system.tables WHERE name =
```

**查询 5** (第261行, 类别: stock_data)
```sql
SELECT MIN(date) as min_date, MAX(date) as max_date FROM stock_info WHERE date >=
```

**查询 6** (第119行, 类别: stock_data)
```sql
SELECT code, date, close FROM stock_info WHERE date >=
```

**查询 7** (第123行, 类别: stock_data)
```sql
SELECT code, COUNT(*) as cnt FROM stock_info WHERE date >=
```

**查询 8** (第127行, 类别: stock_data)
```sql
SELECT code, date, close, volume FROM stock_info WHERE close > 20 AND volume > 1000000 AND date >=
```

**查询 9** (第248行, 类别: metadata)
```sql
SELECT code, name, date, level, open, close, high, low, volume FROM system.tables WHERE name =
```

**查询 10** (第261行, 类别: stock_data)
```sql
SELECT MIN(date) as min_date, MAX(date) as max_date FROM stock_info WHERE date >=
```

**查询 11** (第163行, 类别: stock_data)
```sql
ALTER TABLE stock_info WHERE 1=1 ADD INDEX idx_date_code (date, code) TYPE minmax GRANULARITY 1
```

**查询 12** (第168行, 类别: stock_data)
```sql
ALTER TABLE stock_info WHERE 1=1 ADD INDEX idx_code_date (code, date) TYPE minmax GRANULARITY 1
```

**查询 13** (第173行, 类别: stock_data)
```sql
ALTER TABLE stock_info WHERE 1=1 ADD INDEX idx_volume (volume) TYPE minmax GRANULARITY 1
```

**查询 14** (第178行, 类别: stock_data)
```sql
ALTER TABLE stock_info WHERE 1=1 ADD INDEX idx_close_price (close) TYPE minmax GRANULARITY 1
```

**查询 15** (第279行, 类别: stock_data)
```sql
ALTER TABLE stock_info WHERE 1=1 PARTITION BY toYYYYMM(date)
```

**查询 16** (第74行, 类别: stock_data)
```sql
SHOW CREATE TABLE stock_info
```

**查询 17** (第66行, 类别: stock_data)
```sql
DESCRIBE stock_info
```

### strategy/enhanced_base_strategy.py

**查询 1** (第129行, 类别: stock_data)
```sql
SELECT MAX(date) as max_date FROM stock_info WHERE date >=
```

**查询 2** (第129行, 类别: stock_data)
```sql
SELECT MAX(date) as max_date FROM stock_info WHERE date >=
```

## 🔧 建议的迁移步骤

1. 将常用查询模板化
2. 使用参数化查询替代字符串拼接
3. 统一查询接口
4. 添加查询缓存机制

---

*本报告由SQL迁移工具自动生成*
