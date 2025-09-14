# 真实数据使用规范与硬编码消除标准

## 📋 核心原则

### 🚫 绝对禁止的行为
1. **禁止硬编码指标**: 所有指标必须从指标注册中心动态获取
2. **禁止模拟数据**: 100%使用ClickHouse真实股票数据
3. **禁止兜底逻辑**: 不允许任何硬编码的默认值或兜底处理
4. **禁止重复实现**: 如果功能已存在，必须优化现有实现

### ✅ 强制要求的行为
1. **配置文件驱动**: 所有业务逻辑通过配置文件控制
2. **动态指标获取**: 从CompleteIndicatorRegistry动态获取88+指标
3. **真实数据源**: 仅使用ClickHouse数据库的真实股票数据
4. **现有代码优化**: 在现有文件基础上优化，不新建重复功能

---

## 🎯 真实数据使用标准

### 1. 股票数据获取标准

#### ✅ 正确的数据获取方式
```python
class RealDataAccessStandard:
    """真实数据访问标准"""
    
    def __init__(self):
        self.data_access = get_service("DataAccessInterface")
        
    def get_stock_data(self, stock_code: str, start_date: str, 
                      end_date: str = None) -> pd.DataFrame:
        """获取真实股票数据 - 标准方法"""
        # 1. 参数验证
        if not stock_code or not start_date:
            raise ValueError("股票代码和开始日期不能为空")
            
        # 2. 从ClickHouse获取真实数据
        stock_data = self.data_access.get_stock_info(
            stock_code=stock_code,
            start_date=start_date,
            end_date=end_date or start_date
        )
        
        # 3. 数据验证
        if stock_data.empty:
            raise ValueError(f"无法获取股票 {stock_code} 的真实数据")
            
        return stock_data
```

#### ❌ 禁止的数据获取方式
```python
# 禁止：硬编码数据
def get_fake_data():
    return pd.DataFrame({
        'close': [10, 11, 12],
        'volume': [1000, 1100, 1200]
    })

# 禁止：模拟数据生成
def generate_mock_data():
    return np.random.randn(100)

# 禁止：兜底默认数据
def get_data_with_fallback():
    try:
        return real_data()
    except:
        return default_data()  # 禁止兜底
```

### 2. 指标计算标准

#### ✅ 正确的指标计算方式
```python
class RealIndicatorCalculationStandard:
    """真实指标计算标准"""
    
    def __init__(self):
        self.indicator_registry = get_service("IndicatorRegistry")
        
    def calculate_indicators(self, stock_data: pd.DataFrame, 
                           indicator_names: List[str]) -> Dict:
        """计算真实指标 - 标准方法"""
        results = {}
        
        for indicator_name in indicator_names:
            # 1. 从注册中心获取指标实现
            if not self.indicator_registry.has_indicator(indicator_name):
                raise ValueError(f"指标 {indicator_name} 在注册中心中不存在")
                
            indicator = self.indicator_registry.get_indicator(indicator_name)
            
            # 2. 使用真实数据计算
            result = indicator.calculate(stock_data)
            results[indicator_name] = result
            
        return results
        
    def calculate_all_indicators(self, stock_data: pd.DataFrame) -> Dict:
        """计算所有88+指标"""
        all_indicators = self.indicator_registry.get_all_indicators()
        return self.calculate_indicators(stock_data, list(all_indicators.keys()))
```

#### ❌ 禁止的指标计算方式
```python
# 禁止：硬编码指标计算
def calculate_hardcoded_rsi():
    # 硬编码RSI计算逻辑
    pass

# 禁止：写死指标列表
HARDCODED_INDICATORS = ['RSI', 'MACD', 'KDJ']  # 禁止

# 禁止：兜底指标值
def get_indicator_with_default():
    try:
        return calculate_real_indicator()
    except:
        return 50  # 禁止兜底值
```

### 3. 配置文件驱动标准

#### ✅ 正确的配置驱动方式
```python
class ConfigDrivenStandard:
    """配置驱动标准"""
    
    def load_buypoint_config(self, config_file: str) -> Dict:
        """加载买点配置"""
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"配置文件不存在: {config_file}")
            
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        # 验证配置完整性
        self._validate_buypoint_config(config)
        return config
        
    def load_strategy_config(self, strategy_file: str) -> Dict:
        """加载策略配置"""
        if not os.path.exists(strategy_file):
            raise FileNotFoundError(f"策略文件不存在: {strategy_file}")
            
        with open(strategy_file, 'r', encoding='utf-8') as f:
            strategy = yaml.safe_load(f)
            
        # 验证策略配置
        self._validate_strategy_config(strategy)
        return strategy
        
    def _validate_buypoint_config(self, config: Dict):
        """验证买点配置"""
        required_keys = ['analysis_config']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"配置文件缺少必需字段: {key}")
                
    def _validate_strategy_config(self, strategy: Dict):
        """验证策略配置"""
        required_keys = ['strategy_name', 'conditions', 'scoring']
        for key in required_keys:
            if key not in strategy:
                raise ValueError(f"策略文件缺少必需字段: {key}")
```

#### ❌ 禁止的配置方式
```python
# 禁止：硬编码配置
DEFAULT_CONFIG = {
    'indicators': ['RSI', 'MACD'],  # 硬编码指标列表
    'threshold': 0.5
}

# 禁止：兜底配置
def get_config_with_fallback():
    try:
        return load_real_config()
    except:
        return DEFAULT_CONFIG  # 禁止兜底配置
```

---

## 🔧 现有代码优化标准

### 1. 买点分析优化 (analysis/buypoints/analyze_buypoints.py)

#### 优化要点
```python
# 在现有文件基础上优化
class BuyPointAnalyzer:  # 保持现有类名
    """买点分析器 - 优化版本"""
    
    def __init__(self, data_access: Optional[DataAccessInterface] = None):
        # 保持现有初始化逻辑
        logger.info("初始化买点分析器")
        self.data_access = data_access or get_service(DataAccessInterface)
        
        # 新增：集成指标注册中心
        self.indicator_registry = get_service("IndicatorRegistry")
        logger.info("成功连接到指标注册中心")
        
    def analyze_stock(self, stock_code: str, buy_date: str, 
                     stock_name: str = "", 
                     config_file: str = "config/buypoints_config.json") -> Dict:
        """分析买点 - 优化版本"""
        # 1. 加载真实配置
        config = self._load_real_config(config_file)
        
        # 2. 获取真实股票数据
        stock_data = self._get_real_stock_data(stock_code, buy_date)
        
        # 3. 从注册中心动态获取指标
        indicators = self._get_indicators_from_registry(config)
        
        # 4. 计算真实指标
        indicator_results = self._calculate_real_indicators(stock_data, indicators)
        
        # 5. 分析买点特征
        return self._analyze_buypoint_features(indicator_results, config)
```

### 2. 策略选股优化 (strategy/strategy_executor.py)

#### 优化要点
```python
# 在现有文件基础上优化
class UnifiedStrategyExecutor:  # 保持现有类名
    """统一策略执行器 - 优化版本"""
    
    def __init__(self, max_workers: int = None, cache_enabled: bool = True):
        # 保持现有初始化逻辑
        self.data_access = get_service(DataAccessInterface)
        self.max_workers = max_workers or min(50, os.cpu_count() * 8)
        
        # 新增：集成指标注册中心
        self.indicator_registry = get_service("IndicatorRegistry")
        
    def execute_strategy_from_file(self, strategy_file: str) -> Dict:
        """从YAML文件执行策略 - 优化版本"""
        # 1. 加载真实策略配置
        strategy_config = self._load_real_strategy_config(strategy_file)
        
        # 2. 从配置中解析指标需求
        required_indicators = self._parse_indicators_from_config(strategy_config)
        
        # 3. 从注册中心获取指标实现
        indicator_implementations = self._get_indicators_from_registry(required_indicators)
        
        # 4. 执行策略选股
        return self._execute_strategy_with_real_data(strategy_config, indicator_implementations)
```

### 3. 主系统优化 (bin/main.py)

#### 优化要点
```python
# 在现有文件基础上优化
class MarketAnalysisSystem:  # 保持现有类名
    """市场分析系统 - 优化版本"""
    
    def __init__(self):
        # 保持现有初始化逻辑
        self.data_access = get_service(DataAccessInterface)
        
        # 新增：集成指标注册中心
        self.indicator_registry = get_service("IndicatorRegistry")
        
    def analyze_market(self, date: str = None) -> Dict:
        """市场分析 - 优化版本"""
        # 1. 获取真实市场数据
        market_data = self._get_real_market_data(date)
        
        # 2. 从注册中心获取所有指标
        all_indicators = self.indicator_registry.get_all_indicators()
        
        # 3. 计算市场技术指标
        market_indicators = self._calculate_market_indicators(market_data, all_indicators)
        
        return {
            'date': date,
            'market_indicators': market_indicators,
            'analysis_summary': self._generate_analysis_summary(market_indicators)
        }
```

---

## 📊 质量检查标准

### 1. 代码审查检查清单

#### 数据使用检查
- [ ] 所有数据来源于ClickHouse数据库
- [ ] 无硬编码数据或模拟数据
- [ ] 无兜底默认数据逻辑
- [ ] 数据获取有完整的错误处理

#### 指标使用检查
- [ ] 所有指标从注册中心动态获取
- [ ] 无硬编码指标名称或计算逻辑
- [ ] 指标列表通过配置文件控制
- [ ] 指标计算使用真实股票数据

#### 配置驱动检查
- [ ] 业务逻辑通过配置文件控制
- [ ] 配置文件格式标准化
- [ ] 配置验证机制完整
- [ ] 无硬编码业务参数

#### 代码复用检查
- [ ] 在现有文件基础上优化
- [ ] 无重复功能实现
- [ ] 保持现有API兼容性
- [ ] 优化而非重写

### 2. 自动化检查工具

```python
class QualityChecker:
    """质量检查器"""
    
    def check_hardcoded_indicators(self, file_path: str) -> List[str]:
        """检查硬编码指标"""
        violations = []
        with open(file_path, 'r') as f:
            content = f.read()
            
        # 检查硬编码指标名称
        hardcoded_patterns = [
            r'["\']RSI["\']',
            r'["\']MACD["\']',
            r'["\']KDJ["\']',
            # 更多模式...
        ]
        
        for pattern in hardcoded_patterns:
            if re.search(pattern, content):
                violations.append(f"发现硬编码指标: {pattern}")
                
        return violations
        
    def check_mock_data_usage(self, file_path: str) -> List[str]:
        """检查模拟数据使用"""
        violations = []
        with open(file_path, 'r') as f:
            content = f.read()
            
        # 检查模拟数据模式
        mock_patterns = [
            r'mock_data',
            r'fake_data',
            r'random\.',
            r'np\.random',
            # 更多模式...
        ]
        
        for pattern in mock_patterns:
            if re.search(pattern, content):
                violations.append(f"发现模拟数据使用: {pattern}")
                
        return violations
```

---

## 🎯 实施验收标准

### 功能验收
- [ ] 88+指标100%从注册中心动态获取
- [ ] 所有数据100%来源于ClickHouse
- [ ] 配置文件驱动所有业务逻辑
- [ ] 零硬编码指标或数据

### 性能验收
- [ ] 真实数据处理性能不低于原系统
- [ ] 动态指标获取不影响响应时间
- [ ] 配置文件加载时间 < 100ms

### 质量验收
- [ ] 通过所有质量检查工具验证
- [ ] 代码审查100%通过
- [ ] 集成测试覆盖所有场景
- [ ] 文档完整且准确

---

**文档版本**: v1.0  
**创建时间**: 2025-09-14  
**负责人**: 技术架构师  
**强制执行**: 所有开发人员
