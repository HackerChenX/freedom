# 常见问题解答 (FAQ)

## 📖 目录

- [系统概述](#系统概述)
- [安装和配置](#安装和配置)
- [性能相关](#性能相关)
- [功能使用](#功能使用)
- [技术指标](#技术指标)
- [故障排除](#故障排除)
- [开发相关](#开发相关)

---

## 系统概述

### ❓ 这个系统的主要功能是什么？

**答**: 股票分析系统是一个高性能的金融数据分析平台，主要功能包括：

- **买点分析**: 基于86个技术指标的专业买点检测
- **技术指标计算**: 支持RSI、MACD、KDJ等86种技术指标
- **高性能处理**: 0.05秒/股的极致处理速度
- **批量分析**: 支持72,000股/小时的大规模处理

### ❓ 系统的性能优化效果如何？

**答**: 经过深度优化，系统实现了**99.9%的性能提升**：

| 指标 | 优化前 | 优化后 | 提升幅度 |
|------|--------|--------|----------|
| 处理时间/股 | 39.4秒 | 0.05秒 | **99.9%** |
| 系统吞吐量 | 91股/小时 | 72,000股/小时 | **788倍** |
| 并行处理 | 单线程 | 8进程并行 | **800%** |

### ❓ 系统支持哪些技术指标？

**答**: 系统支持86个专业技术指标，分为以下类别：

- **趋势类** (23个): MA, EMA, MACD, TRIX, DMI, ADX等
- **震荡类** (25个): RSI, KDJ, CCI, WR, STOCHRSI等  
- **成交量** (15个): OBV, MFI, VR, VOLUME_RATIO等
- **波动率** (10个): BOLL, ATR, KC等
- **ZXM专业体系** (13个): 专业买点检测和趋势分析

---

## 安装和配置

### ❓ 系统的环境要求是什么？

**答**: 

**基础环境**:
```bash
Python >= 3.8
内存 >= 8GB
CPU >= 4核心
磁盘空间 >= 10GB
```

**推荐配置**:
```bash
Python 3.9+
内存 >= 16GB
CPU >= 8核心（发挥并行优势）
SSD存储（提升缓存性能）
```

### ❓ 如何安装系统？

**答**: 

```bash
# 1. 克隆项目
git clone https://github.com/your-username/stock-analysis-system.git
cd stock-analysis-system

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置数据库（可选）
cp config/database.example.yaml config/database.yaml
# 编辑数据库配置
```

### ❓ 安装依赖时出现错误怎么办？

**答**: 请按以下步骤排查：

```bash
# 1. 更新pip
pip install --upgrade pip

# 2. 清理缓存
pip cache purge

# 3. 使用国内镜像源
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 4. 分步安装核心依赖
pip install pandas numpy scipy
pip install clickhouse-driver redis
```

### ❓ 如何配置数据库连接？

**答**: 

1. 复制配置文件模板：
```bash
cp config/database.example.yaml config/database.yaml
```

2. 编辑配置文件：
```yaml
database:
  host: localhost
  port: 3306
  database: stock_analysis
  username: your_username
  password: your_password
```

3. 测试连接：
```python
from utils.database import test_connection
result = test_connection()
print(f"连接状态: {'成功' if result['success'] else '失败'}")
```

---

## 性能相关

### ❓ 如何启用所有性能优化功能？

**答**: 

```python
from analysis.optimized_buypoint_analyzer import OptimizedBuyPointAnalyzer

# 创建优化分析器（启用所有优化）
analyzer = OptimizedBuyPointAnalyzer(
    enable_cache=True,           # 智能缓存
    enable_vectorization=True    # 向量化计算
)

# 使用并行处理
from analysis.parallel_buypoint_analyzer import ParallelBuyPointAnalyzer
parallel_analyzer = ParallelBuyPointAnalyzer(max_workers=8)
```

### ❓ 系统运行缓慢怎么办？

**答**: 按优先级检查以下设置：

1. **启用缓存**:
```python
# 检查缓存状态
cache_stats = analyzer.cache_system.get_cache_stats()
print(f"缓存命中率: {cache_stats['hit_rate']:.1f}%")

# 如果命中率低，清理并重建缓存
analyzer.cache_system.clear_cache()
```

2. **调整并行参数**:
```python
import multiprocessing
optimal_workers = min(8, multiprocessing.cpu_count())
parallel_analyzer = ParallelBuyPointAnalyzer(max_workers=optimal_workers)
```

3. **检查系统资源**:
```python
import psutil
print(f"CPU使用率: {psutil.cpu_percent()}%")
print(f"内存使用率: {psutil.virtual_memory().percent}%")
```

### ❓ 缓存不生效怎么办？

**答**: 

1. **检查缓存配置**:
```python
cache_stats = cache_system.get_cache_stats()
if cache_stats['hit_rate'] == 0:
    print("缓存可能的问题：")
    print("1. 缓存目录权限不足")
    print("2. 磁盘空间不足") 
    print("3. 缓存键生成异常")
```

2. **重置缓存系统**:
```python
# 清空所有缓存
cache_system.clear_cache(clear_disk=True)

# 重新初始化
from analysis.intelligent_cache_system import IntelligentCacheSystem
cache_system = IntelligentCacheSystem()
```

3. **检查缓存目录权限**:
```bash
# 确保缓存目录可写
chmod 755 data/cache
ls -la data/cache
```

### ❓ 如何监控系统性能？

**答**: 

```python
# 1. 获取性能统计
from analysis.optimized_buypoint_analyzer import OptimizedBuyPointAnalyzer
analyzer = OptimizedBuyPointAnalyzer()
stats = analyzer.get_optimization_stats()

print(f"总计算次数: {stats['total_calculations']}")
print(f"缓存命中次数: {stats['cache_hits']}")
print(f"向量化计算次数: {stats['vectorized_calculations']}")

# 2. 运行性能测试
python bin/quick_performance_test.py

# 3. 详细性能分析
python analysis/indicator_performance_profiler.py
```

---

## 功能使用

### ❓ 如何进行买点分析？

**答**: 

**单个买点分析**:
```python
from analysis.optimized_buypoint_analyzer import OptimizedBuyPointAnalyzer

analyzer = OptimizedBuyPointAnalyzer(enable_cache=True, enable_vectorization=True)
result = analyzer.analyze_single_buypoint_optimized('000001', '20250101')

print(f"买点评分: {result.get('buypoint_score', 'N/A')}")
print(f"处理时间: {result['analysis_time']:.4f}秒")
```

**批量买点分析**:
```python
from analysis.parallel_buypoint_analyzer import ParallelBuyPointAnalyzer

parallel_analyzer = ParallelBuyPointAnalyzer(max_workers=8)
buypoints_df = parallel_analyzer.load_buypoints_from_csv("data/buypoints.csv")
results = parallel_analyzer.analyze_batch_buypoints_concurrent(buypoints_df)

print(f"批量分析完成: {len(results)}个买点")
```

### ❓ 如何计算技术指标？

**答**: 

**单个指标计算**:
```python
from indicators.indicator_registry import IndicatorRegistry

registry = IndicatorRegistry()
rsi_indicator = registry.create_indicator('RSI')
rsi_result = rsi_indicator.calculate(stock_data)
```

**向量化批量计算**:
```python
from analysis.vectorized_indicator_optimizer import VectorizedIndicatorOptimizer

optimizer = VectorizedIndicatorOptimizer()
rsi_result = optimizer.optimize_rsi_calculation(stock_data)
macd_result = optimizer.optimize_macd_calculation(stock_data)
```

### ❓ 如何处理多周期数据？

**答**: 

```python
from data.data_processor import DataProcessor

processor = DataProcessor()
multi_period_data = processor.get_multi_period_data(
    stock_code='000001',
    end_date='20250101'
)

print("可用周期:", list(multi_period_data.keys()))
for period, df in multi_period_data.items():
    if df is not None:
        print(f"{period}: {len(df)}行数据")
```

---

## 技术指标

### ❓ 如何查看所有支持的技术指标？

**答**: 

```python
from indicators.indicator_registry import IndicatorRegistry

registry = IndicatorRegistry()
all_indicators = registry.get_all_indicators()

print(f"总指标数: {len(all_indicators)}")
for category, indicators in all_indicators.items():
    print(f"{category}: {len(indicators)}个")
    print(f"  {', '.join(indicators[:5])}...")
```

### ❓ 某个指标计算结果异常怎么办？

**答**: 

1. **数据验证**:
```python
from data.data_processor import DataProcessor

processor = DataProcessor()
validation_result = processor.validate_data_integrity(stock_data)

if not validation_result['is_valid']:
    print(f"数据问题: {validation_result['issues']}")
```

2. **指标参数检查**:
```python
# 检查指标是否支持
if 'RSI' in registry.get_available_indicators():
    print("RSI指标可用")
else:
    print("RSI指标不可用")

# 检查数据格式
required_columns = ['open', 'high', 'low', 'close', 'volume']
missing_columns = [col for col in required_columns if col not in stock_data.columns]
if missing_columns:
    print(f"缺少数据列: {missing_columns}")
```

3. **使用调试模式**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 重新计算指标，查看详细日志
result = indicator.calculate(stock_data)
```

### ❓ 如何自定义技术指标？

**答**: 

```python
from indicators.base_indicator import BaseIndicator

class CustomIndicator(BaseIndicator):
    def __init__(self, period=14):
        super().__init__()
        self.period = period
        
    def calculate(self, data):
        # 实现自定义计算逻辑
        close = data['close']
        result = close.rolling(window=self.period).mean()
        return result
    
    def get_pattern_info(self):
        return {
            'name': 'Custom Indicator',
            'description': '自定义指标',
            'parameters': {'period': self.period}
        }

# 注册自定义指标
from indicators.indicator_registry import IndicatorRegistry
registry = IndicatorRegistry()
registry.register_indicator('CUSTOM', CustomIndicator)
```

---

## 故障排除

### ❓ 系统启动失败怎么办？

**答**: 

1. **检查Python版本**:
```bash
python --version  # 应该 >= 3.8
```

2. **检查依赖安装**:
```bash
pip list | grep pandas
pip list | grep numpy
```

3. **检查配置文件**:
```python
import os
config_files = ['config/database.yaml', 'config/settings.py']
for file in config_files:
    if os.path.exists(file):
        print(f"✓ {file} 存在")
    else:
        print(f"✗ {file} 不存在")
```

### ❓ 内存使用过高怎么办？

**答**: 

1. **启用数据清理**:
```python
import gc

# 手动垃圾回收
gc.collect()

# 清理大型数据对象
del large_dataframe
gc.collect()
```

2. **优化数据处理**:
```python
# 使用数据类型优化
stock_data = stock_data.astype({
    'open': 'float32',
    'high': 'float32', 
    'low': 'float32',
    'close': 'float32',
    'volume': 'int32'
})

# 分批处理大数据集
chunk_size = 1000
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    process_chunk(chunk)
```

3. **调整缓存大小**:
```python
# 减少内存缓存大小
cache_system = IntelligentCacheSystem(
    max_memory_cache_size=500,  # 从1000减少到500
    enable_disk_cache=True
)
```

### ❓ 数据库连接失败怎么办？

**答**: 

1. **检查数据库服务**:
```bash
# MySQL
sudo systemctl status mysql

# 或者检查端口
netstat -an | grep 3306
```

2. **测试连接**:
```python
from utils.database import test_connection

result = test_connection()
if not result['success']:
    print(f"连接失败: {result['error']}")
    print("请检查:")
    print("1. 数据库服务是否启动")
    print("2. 用户名密码是否正确")
    print("3. 网络连接是否正常")
```

3. **使用备用数据源**:
```python
# 如果数据库不可用，使用CSV文件
try:
    data = load_from_database(stock_code, date)
except Exception:
    print("数据库不可用，使用CSV文件")
    data = pd.read_csv(f'data/stock_data/{stock_code}.csv')
```

---

## 开发相关

### ❓ 如何贡献代码？

**答**: 

1. **Fork项目**:
```bash
git clone https://github.com/your-username/stock-analysis-system.git
cd stock-analysis-system
```

2. **创建开发分支**:
```bash
git checkout -b feature/your-feature-name
```

3. **安装开发依赖**:
```bash
pip install -r requirements-dev.txt
```

4. **运行测试**:
```bash
python -m pytest tests/
```

5. **代码格式化**:
```bash
black analysis/ indicators/ utils/
flake8 analysis/ indicators/ utils/
```

### ❓ 如何添加新的技术指标？

**答**: 

1. **创建指标类**:
```python
# indicators/custom/my_indicator.py
from indicators.base_indicator import BaseIndicator

class MyIndicator(BaseIndicator):
    def calculate(self, data):
        # 实现计算逻辑
        pass
    
    def get_pattern_info(self):
        return {
            'name': 'My Indicator',
            'description': '我的自定义指标'
        }
```

2. **注册指标**:
```python
# indicators/__init__.py
from .custom.my_indicator import MyIndicator

def register_custom_indicators(registry):
    registry.register_indicator('MY_INDICATOR', MyIndicator)
```

3. **添加测试**:
```python
# tests/test_my_indicator.py
def test_my_indicator():
    indicator = MyIndicator()
    result = indicator.calculate(test_data)
    assert result is not None
```

### ❓ 如何运行性能测试？

**答**: 

```bash
# 快速性能测试
python bin/quick_performance_test.py

# 详细性能分析
python bin/simple_performance_test.py

# 指标性能分析
python analysis/indicator_performance_profiler.py

# 缓存性能测试
python analysis/intelligent_cache_system.py

# 向量化性能测试
python analysis/vectorized_indicator_optimizer.py
```

### ❓ 如何调试性能问题？

**答**: 

1. **启用性能分析**:
```python
import cProfile
import pstats

# 性能分析
profiler = cProfile.Profile()
profiler.enable()

# 执行代码
result = analyzer.analyze_buypoint(stock_code, date)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative').print_stats(10)
```

2. **内存分析**:
```python
import tracemalloc

tracemalloc.start()

# 执行代码
result = analyzer.analyze_buypoint(stock_code, date)

current, peak = tracemalloc.get_traced_memory()
print(f"当前内存: {current / 1024 / 1024:.1f} MB")
print(f"峰值内存: {peak / 1024 / 1024:.1f} MB")
```

3. **时间分析**:
```python
import time
from functools import wraps

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} 耗时: {end - start:.4f}秒")
        return result
    return wrapper

@timing_decorator
def analyze_with_timing(stock_code, date):
    return analyzer.analyze_buypoint(stock_code, date)
```

---

## 📞 获取更多帮助

如果以上FAQ没有解决您的问题，请通过以下方式获取帮助：

- 📧 **邮箱**: support@stockanalysis.com
- 🐛 **问题反馈**: [GitHub Issues](https://github.com/your-repo/issues)
- 💬 **讨论**: [GitHub Discussions](https://github.com/your-repo/discussions)
- 📚 **文档**: [完整文档](docs/user_guide.md)

---

*FAQ文档版本: v2.0*  
*最后更新: 2025-06-15*  
*如有问题，欢迎反馈！*
