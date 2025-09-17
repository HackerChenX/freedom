# 分层测试标准（强制执行）

## 🚨 核心测试原则

### 强制测试要求
**每修复一层，就测试一层。测试完全通过，日志没有任何报错才能进入下一任务**

### 测试标准
1. **功能完整性**: 该层所有功能必须100%正常工作
2. **集成兼容性**: 与已修复层的集成必须100%正常
3. **性能达标性**: 必须满足预定义的性能要求
4. **日志清洁性**: 日志中不能有任何ERROR或WARNING
5. **标准合规性**: 必须100%符合标准化要求

### 阻断机制（严格执行）
- ❌ **功能测试不通过**: 禁止进入下一层修复
- ❌ **日志有ERROR/WARNING**: 必须解决所有错误和警告
- ❌ **性能不达标**: 必须优化到达标
- ❌ **标准不符合**: 必须重构到符合标准
- ❌ **集成测试失败**: 必须修复集成问题

## 📋 L1基础设施层测试标准

### 测试目标
验证依赖注入、配置管理、日志系统的统一性和稳定性

#### 测试1.1: 依赖注入容器测试
```python
def test_l1_dependency_injection():
    """L1层依赖注入容器测试"""
    
    # 1. 容器唯一性测试
    assert only_one_container_exists()
    
    # 2. 服务注册测试
    container = get_unified_container()
    test_services = ['DataAccessInterface', 'Logger', 'ConfigManager']
    
    for service in test_services:
        # 注册测试
        assert container.can_register(service)
        container.register(service, MockImplementation)
        
        # 解析测试
        instance = container.resolve(service)
        assert instance is not None
        
        # 重复注册检查（应该失败）
        with pytest.raises(DuplicateRegistrationError):
            container.register(service, AnotherImplementation)
    
    # 3. 线程安全测试
    assert test_container_thread_safety()

def test_l1_no_duplicate_containers():
    """确保没有重复容器"""
    container_files = [
        'utils/unified_container.py',  # 唯一标准
        'db/container.py',             # 应该被删除
        'utils/optimized_dependency_injection.py'  # 应该被删除
    ]
    
    existing_files = [f for f in container_files if os.path.exists(f)]
    assert len(existing_files) == 1
    assert existing_files[0] == 'utils/unified_container.py'
```

#### 测试1.2: 配置管理测试
```python
def test_l1_configuration_management():
    """L1层配置管理测试"""
    
    # 1. 配置文件存在性测试
    required_configs = [
        'config/database.yml',
        'config/indicators.yml', 
        'config/strategies.yml',
        'config/thresholds.yml',
        'config/system.yml'
    ]
    
    for config_file in required_configs:
        assert os.path.exists(config_file), f"配置文件 {config_file} 不存在"
    
    # 2. 配置加载测试
    config_manager = ConfigManager()
    assert config_manager.load_all_configs()
    
    # 3. 硬编码检查
    assert no_hardcoded_values_in_code()
    
    # 4. 配置验证测试
    assert config_manager.validate_all_configs()
```

#### 测试1.3: 日志系统测试
```python
def test_l1_logging_system():
    """L1层日志系统测试"""
    
    # 1. 日志统一性测试
    logger = get_unified_logger()
    assert logger is not None
    
    # 2. 日志级别测试
    test_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR']
    for level in test_levels:
        logger.log(level, f"测试 {level} 级别日志")
    
    # 3. 日志格式测试
    assert validate_log_format()
    
    # 4. 无自定义日志实现检查
    assert no_custom_logging_implementations()
```

### L1层通过标准
- [ ] 只存在一个依赖注入容器实现
- [ ] 所有服务注册和解析正常
- [ ] 配置文件完整且可正常加载
- [ ] 无任何硬编码配置
- [ ] 日志系统统一且格式正确
- [ ] 运行日志中无ERROR/WARNING
- [ ] 性能测试：容器操作<1ms

## 📋 L2存储访问层测试标准

### 测试目标
验证数据库连接池和SQL管理的统一性和稳定性

#### 测试2.1: 数据库连接池测试
```python
def test_l2_connection_pool():
    """L2层连接池测试"""
    
    # 1. 连接池唯一性测试
    assert only_one_connection_pool_entry()
    
    # 2. 连接池功能测试
    pool = get_connection_pool()
    assert pool is not None
    
    # 3. 连接获取测试
    with pool.get_connection() as conn:
        assert conn.is_healthy()
        result = conn.execute("SELECT 1")
        assert result is not None
    
    # 4. 连接池配置测试
    config = pool.get_config()
    assert 5 <= config['min_connections'] <= config['max_connections'] <= 20
    assert config['timeout'] == 30
    
    # 5. 并发连接测试
    assert test_concurrent_connections(pool, max_connections=20)

def test_l2_no_duplicate_connection_pools():
    """确保没有重复连接池创建"""
    # 检查代码中是否还有其他连接池创建
    duplicate_patterns = [
        'ClickHouseConnectionPool()',
        'ConnectionPool(',
        'create_connection_pool'
    ]
    
    for pattern in duplicate_patterns:
        files_with_pattern = search_pattern_in_files(pattern)
        # 只允许在enhanced_connection_pool.py中存在
        allowed_files = ['db/enhanced_connection_pool.py']
        for file_path in files_with_pattern:
            assert file_path in allowed_files
```

#### 测试2.2: SQL管理测试
```python
def test_l2_sql_management():
    """L2层SQL管理测试"""
    
    # 1. SQL模板统一性测试
    sql_manager = SQLManager()
    
    # 2. 标准查询模板测试
    stock_query = sql_manager.get_stock_data_query()
    assert "SELECT code, name, date, open, high, low, close, volume, turnover_rate" in stock_query
    assert "FROM stock_info" in stock_query
    assert "WHERE code = %s AND level = %s" in stock_query
    assert "ORDER BY date ASC" in stock_query
    
    # 3. 参数化查询测试
    assert sql_manager.is_parameterized_query(stock_query)
    
    # 4. SQL注入防护测试
    assert test_sql_injection_protection()
    
    # 5. 分散SQL检查
    assert no_scattered_sql_in_business_code()
```

### L2层通过标准
- [ ] 只有一个连接池创建入口
- [ ] 连接池配置正确且功能正常
- [ ] 并发连接测试通过
- [ ] SQL查询统一管理
- [ ] 无SQL注入风险
- [ ] 运行日志中无ERROR/WARNING
- [ ] 性能测试：连接获取<100ms，查询<1s

## 📋 L3数据服务层测试标准

### 测试目标
验证数据访问接口和多周期数据服务的统一性和正确性

#### 测试3.1: 数据访问接口测试
```python
def test_l3_data_access_interface():
    """L3层数据访问接口测试"""
    
    # 1. 接口唯一性测试
    assert only_one_data_access_implementation()
    
    # 2. 数据访问功能测试
    data_access = get_data_access_manager()
    
    # 测试股票数据获取
    stock_data = data_access.get_stock_data('000001', '2024-01-01', '2024-01-31', '日线')
    assert not stock_data.empty
    assert all(col in stock_data.columns for col in ['code', 'name', 'date', 'open', 'high', 'low', 'close', 'volume'])
    
    # 3. 数据格式验证
    assert validate_data_format(stock_data)
    
    # 4. 错误处理测试
    with pytest.raises(DataNotFoundError):
        data_access.get_stock_data('INVALID', '2024-01-01', '2024-01-31', '日线')

def test_l3_no_duplicate_data_access():
    """确保没有重复数据访问实现"""
    data_access_files = find_files_with_pattern('*data_access*')
    allowed_files = [
        'db/interfaces/data_access_interface.py',
        'db/managers/data_access_manager.py'
    ]
    
    for file_path in data_access_files:
        assert file_path in allowed_files
```

#### 测试3.2: 多周期数据服务测试
```python
def test_l3_multi_period_data_service():
    """L3层多周期数据服务测试"""
    
    # 1. 数据聚合功能测试
    service = MultiPeriodDataService()
    
    # 测试15分钟到30分钟聚合
    data_15min = service.get_period_data('000001', '15分钟', 100)
    data_30min = service.aggregate_to_30min(data_15min)
    assert not data_30min.empty
    assert len(data_30min) == len(data_15min) // 2
    
    # 测试15分钟到60分钟聚合
    data_60min = service.aggregate_to_60min(data_15min)
    assert not data_60min.empty
    assert len(data_60min) == len(data_15min) // 4
    
    # 2. 数据完整性验证
    periods = ['15分钟', '30分钟', '60分钟', '日线', '周线', '月线']
    for period in periods:
        data = service.get_period_data('000001', period, 50)
        assert not data.empty, f"{period} 数据为空"
    
    # 3. 聚合算法正确性测试
    assert validate_aggregation_algorithm()
```

### L3层通过标准
- [ ] 只有一个数据访问实现
- [ ] 数据访问功能完全正常
- [ ] 6个周期数据完整可用
- [ ] 数据聚合算法正确
- [ ] 数据格式标准统一
- [ ] 运行日志中无ERROR/WARNING
- [ ] 性能测试：数据查询<1s，聚合<2s

## 📋 L4核心服务层测试标准

### 测试目标
验证指标系统的完整性和技术分析服务的正确性

#### 测试4.1: 指标注册系统测试
```python
def test_l4_indicator_registry():
    """L4层指标注册系统测试"""
    
    # 1. 指标注册完整性测试
    registry = get_indicator_registry()
    all_indicators = registry.get_all_indicators()
    
    # 必须有128个指标
    assert len(all_indicators) == 128
    
    # 2. 指标注册成功率测试
    failed_indicators = []
    for indicator_name in all_indicators:
        try:
            indicator = registry.get_indicator(indicator_name)
            assert indicator is not None
        except Exception as e:
            failed_indicators.append((indicator_name, str(e)))
    
    # 注册成功率必须≥95%
    success_rate = (128 - len(failed_indicators)) / 128
    assert success_rate >= 0.95, f"指标注册成功率 {success_rate:.2%} < 95%"
    
    # 3. 指标基类继承测试
    for indicator_name in all_indicators:
        indicator_class = registry.get_indicator_class(indicator_name)
        assert issubclass(indicator_class, BaseIndicator)

def test_l4_indicator_calculation():
    """L4层指标计算测试"""
    
    # 1. 核心指标计算测试
    test_data = get_test_stock_data('000001', 100)  # 100天数据
    core_indicators = ['MA', 'MACD', 'RSI', 'KDJ', 'BOLL']
    
    for indicator_name in core_indicators:
        indicator = get_indicator(indicator_name)
        
        # 计算测试
        start_time = time.time()
        result = indicator.calculate(test_data)
        calculation_time = time.time() - start_time
        
        # 结果验证
        assert not result.empty
        assert calculation_time < 2.0, f"{indicator_name} 计算时间 {calculation_time:.2f}s > 2s"
        
        # 信号生成测试
        signal = indicator.get_signal(result)
        assert 'signal' in signal
        assert signal['signal'] in ['BUY', 'SELL', 'HOLD']
```

### L4层通过标准
- [ ] 指标注册成功率≥95%
- [ ] 所有128个指标可正常实例化
- [ ] 核心指标计算正确且<2秒
- [ ] 指标基类继承统一
- [ ] 形态识别功能正常
- [ ] 运行日志中无ERROR/WARNING
- [ ] 性能测试：指标计算<2s，批量计算<30s

## 📋 L5业务应用层测试标准

### 测试目标
验证买点分析和策略管理的统一性和准确性

#### 测试5.1: 买点分析系统测试
```python
def test_l5_buypoint_analysis():
    """L5层买点分析系统测试"""
    
    # 1. 买点分析功能测试
    analyzer = get_buypoint_analyzer()
    
    # 测试单股票多周期分析
    result = analyzer.analyze_buypoint('000001', periods=['15分钟', '30分钟', '60分钟', '日线', '周线', '月线'])
    
    assert 'buypoint_score' in result
    assert 'period_analysis' in result
    assert len(result['period_analysis']) == 6  # 6个周期
    
    # 2. 一致性验证测试
    consistency = analyzer.verify_consistency(result)
    assert consistency >= 0.9, f"一致性验证 {consistency:.2%} < 90%"
    
    # 3. 买点评分测试
    score = result['buypoint_score']
    assert 0 <= score <= 100

def test_l5_strategy_management():
    """L5层策略管理测试"""
    
    # 1. 策略注册统一性测试
    manager = get_strategy_manager()
    strategies = manager.get_all_strategies()
    
    # 确保策略注册统一
    assert len(strategies) > 0
    
    # 2. 策略执行测试
    for strategy_name in strategies:
        strategy = manager.get_strategy(strategy_name)
        result = strategy.execute('000001')
        assert 'recommendation' in result
```

### L5层通过标准
- [ ] 买点分析功能完全正常
- [ ] 一致性验证≥90%
- [ ] 策略管理统一且功能正常
- [ ] 选股引擎功能完整
- [ ] 多周期分析协调正常
- [ ] 运行日志中无ERROR/WARNING
- [ ] 性能测试：买点分析<30s，策略执行<10s

## 📋 L6用户接口层测试标准

### 测试目标
验证应用入口和API接口的统一性和完整性

#### 测试6.1: 应用入口测试
```python
def test_l6_application_entry():
    """L6层应用入口测试"""
    
    # 1. 入口唯一性测试
    assert only_one_application_entry()
    
    # 2. 主入口功能测试
    result = subprocess.run(['python', 'bin/main_analyzer.py', '--stock-code', '000001'], 
                          capture_output=True, text=True)
    
    assert result.returncode == 0
    assert "分析完成" in result.stdout
    assert len(result.stderr) == 0  # 无错误输出

def test_l6_api_interface():
    """L6层API接口测试"""
    
    # 1. API服务启动测试
    api_process = start_api_server()
    time.sleep(5)  # 等待启动
    
    # 2. API功能测试
    response = requests.get('http://localhost:8000/health')
    assert response.status_code == 200
    
    # 3. 核心API测试
    response = requests.post('http://localhost:8000/analyze', 
                           json={'stock_code': '000001'})
    assert response.status_code == 200
    
    api_process.terminate()
```

### L6层通过标准
- [ ] 只有一个应用入口
- [ ] 应用入口功能完整
- [ ] API接口正常工作
- [ ] 命令行参数处理正确
- [ ] 用户文档完整
- [ ] 运行日志中无ERROR/WARNING
- [ ] 性能测试：应用启动<10s，API响应<5s

## 🔍 整体系统测试标准

### 最终验证测试
```python
def test_complete_system():
    """完整系统测试"""
    
    # 1. 端到端测试
    test_stocks = ['000001', '300005', '603359']
    
    for stock_code in test_stocks:
        result = run_complete_analysis(stock_code)
        assert result['success'] == True
        assert 'buypoint_analysis' in result
        assert 'strategy_recommendation' in result
    
    # 2. 性能测试
    start_time = time.time()
    run_complete_analysis('000001')
    total_time = time.time() - start_time
    assert total_time < 60, f"完整分析时间 {total_time:.2f}s > 60s"
    
    # 3. 稳定性测试
    for i in range(10):
        result = run_complete_analysis('000001')
        assert result['success'] == True
```

### 最终通过标准
- [ ] 所有6层测试100%通过
- [ ] 端到端测试完全正常
- [ ] 系统运行日志完全清洁（无ERROR/WARNING）
- [ ] 性能指标全部达标
- [ ] 稳定性测试通过
- [ ] 标准化要求100%符合

## 📊 测试报告模板

### 每层测试报告格式
```
## L{X}层测试报告

### 测试执行时间
- 开始时间: YYYY-MM-DD HH:MM:SS
- 结束时间: YYYY-MM-DD HH:MM:SS
- 总耗时: X分钟

### 测试结果
- 功能测试: ✅/❌ (通过率: X%)
- 集成测试: ✅/❌ (通过率: X%)
- 性能测试: ✅/❌ (达标率: X%)
- 日志检查: ✅/❌ (清洁度: X%)
- 标准验证: ✅/❌ (合规率: X%)

### 问题记录
- 问题1: 描述 + 解决方案
- 问题2: 描述 + 解决方案

### 通过/阻断决定
- [ ] 通过，可以进入下一层
- [ ] 阻断，需要解决问题后重新测试
```

## 📝 总结

分层测试标准确保了：
1. **质量保证**: 每层修复后都经过严格测试
2. **问题隔离**: 问题在当前层解决，不传播到上层
3. **进度控制**: 测试不通过就不能进入下一阶段
4. **标准执行**: 强制执行所有标准化要求

只有严格执行这些测试标准，才能确保系统优化的质量和效果。
