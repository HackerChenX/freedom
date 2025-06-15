# 可配置选股模块文档

## 📊 模块概览

可配置选股模块是股票分析系统的高级功能，基于**86个技术指标**和**ZXM买点分析**，提供灵活的多条件筛选和智能选股服务。该模块支持复杂的选股策略配置，能够从数千只股票中快速筛选出符合条件的投资标的。

### 🎯 核心特性

- **多条件筛选**: 支持86个技术指标的任意组合
- **智能评分**: 基于ZXM体系的综合评分机制
- **高性能处理**: 支持大规模股票池筛选（10,000+股票）
- **灵活配置**: YAML/JSON配置文件，支持复杂策略
- **实时筛选**: 支持实时数据筛选和监控
- **策略回测**: 内置策略效果验证功能

### 🏗️ 选股架构

1. **条件解析器**: 解析复杂的筛选条件
2. **指标计算引擎**: 批量计算所需技术指标
3. **筛选执行器**: 执行多条件筛选逻辑
4. **评分排序器**: 对筛选结果进行评分排序
5. **结果输出器**: 格式化输出筛选结果

---

## 🔧 配置系统

### 基础配置结构

```yaml
# stock_selection_config.yaml
selection_strategy:
  name: "ZXM买点选股策略"
  description: "基于ZXM体系的买点选股"
  
  # 基础筛选条件
  basic_filters:
    market_cap:
      min: 50  # 最小市值（亿元）
      max: 1000  # 最大市值（亿元）
    price_range:
      min: 5.0   # 最低价格
      max: 100.0 # 最高价格
    volume:
      min_avg_volume: 1000000  # 最小平均成交量
    
  # 技术指标条件
  technical_conditions:
    trend_indicators:
      - indicator: "RSI"
        condition: "between"
        values: [30, 70]
        weight: 0.2
      
      - indicator: "MACD"
        condition: "golden_cross"
        lookback: 5
        weight: 0.3
    
    volume_indicators:
      - indicator: "OBV"
        condition: "increasing"
        periods: 10
        weight: 0.2
    
    zxm_indicators:
      - indicator: "ZXM_BUYPOINT"
        condition: "greater_than"
        value: 80
        weight: 0.3
  
  # 排序和输出
  sorting:
    primary: "zxm_score"
    secondary: "buypoint_score"
    order: "desc"
  
  output:
    max_results: 50
    include_details: true
    export_format: ["csv", "json"]
```

### 高级策略配置

```yaml
# advanced_strategy.yaml
selection_strategy:
  name: "多因子量化选股"
  
  # 多阶段筛选
  stages:
    - name: "基础筛选"
      conditions:
        - "market_cap > 100"
        - "price > 10"
        - "avg_volume_20d > 5000000"
    
    - name: "技术筛选"
      conditions:
        - "RSI < 70 AND RSI > 30"
        - "MACD_histogram > 0"
        - "MA5 > MA20"
    
    - name: "ZXM筛选"
      conditions:
        - "zxm_buypoint_score > 75"
        - "zxm_trend_score > 70"
  
  # 评分模型
  scoring_model:
    factors:
      technical_score:
        weight: 0.4
        components:
          - indicator: "RSI"
            transform: "normalize"
            weight: 0.3
          - indicator: "MACD"
            transform: "signal_strength"
            weight: 0.4
          - indicator: "KDJ"
            transform: "momentum"
            weight: 0.3
      
      zxm_score:
        weight: 0.6
        components:
          - indicator: "ZXM_BUYPOINT"
            weight: 0.4
          - indicator: "ZXM_TREND"
            weight: 0.3
          - indicator: "ZXM_ELASTICITY"
            weight: 0.3
```

---

## 🚀 核心功能实现

### 1. 配置解析器

```python
from selection.config_parser import SelectionConfigParser
import yaml

class SelectionConfigParser:
    """选股配置解析器"""
    
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self):
        """加载配置文件"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def parse_conditions(self):
        """解析筛选条件"""
        conditions = []
        
        # 解析基础筛选条件
        basic_filters = self.config.get('basic_filters', {})
        for filter_type, params in basic_filters.items():
            condition = self._parse_basic_filter(filter_type, params)
            conditions.append(condition)
        
        # 解析技术指标条件
        tech_conditions = self.config.get('technical_conditions', {})
        for category, indicators in tech_conditions.items():
            for indicator_config in indicators:
                condition = self._parse_technical_condition(indicator_config)
                conditions.append(condition)
        
        return conditions
    
    def _parse_technical_condition(self, config):
        """解析技术指标条件"""
        return {
            'type': 'technical',
            'indicator': config['indicator'],
            'condition': config['condition'],
            'params': config.get('values') or config.get('value'),
            'weight': config.get('weight', 1.0)
        }

# 使用示例
parser = SelectionConfigParser('config/stock_selection_config.yaml')
conditions = parser.parse_conditions()
print(f"解析到 {len(conditions)} 个筛选条件")
```

### 2. 多条件筛选引擎

```python
from selection.multi_condition_filter import MultiConditionFilter
import pandas as pd

class MultiConditionFilter:
    """多条件筛选引擎"""
    
    def __init__(self, conditions):
        self.conditions = conditions
        self.indicator_calculator = IndicatorCalculator()
    
    def filter_stocks(self, stock_pool):
        """
        执行多条件筛选
        
        Args:
            stock_pool: 股票池列表
        
        Returns:
            List: 筛选后的股票列表
        """
        results = []
        
        for stock_code in stock_pool:
            try:
                # 获取股票数据
                stock_data = self._get_stock_data(stock_code)
                
                # 计算所需指标
                indicators = self._calculate_indicators(stock_data)
                
                # 执行筛选条件
                if self._check_conditions(stock_code, stock_data, indicators):
                    score = self._calculate_score(indicators)
                    results.append({
                        'stock_code': stock_code,
                        'score': score,
                        'indicators': indicators
                    })
            
            except Exception as e:
                print(f"筛选 {stock_code} 时出错: {e}")
                continue
        
        return sorted(results, key=lambda x: x['score'], reverse=True)
    
    def _check_conditions(self, stock_code, data, indicators):
        """检查是否满足所有筛选条件"""
        for condition in self.conditions:
            if not self._evaluate_condition(condition, data, indicators):
                return False
        return True
    
    def _evaluate_condition(self, condition, data, indicators):
        """评估单个条件"""
        if condition['type'] == 'basic':
            return self._evaluate_basic_condition(condition, data)
        elif condition['type'] == 'technical':
            return self._evaluate_technical_condition(condition, indicators)
        return True

# 使用示例
conditions = parser.parse_conditions()
filter_engine = MultiConditionFilter(conditions)

stock_pool = ['000001', '000002', '000858', '002415', '600519']
filtered_results = filter_engine.filter_stocks(stock_pool)

print(f"筛选结果: {len(filtered_results)} 只股票")
for result in filtered_results[:10]:
    print(f"{result['stock_code']}: {result['score']:.2f}")
```

### 3. 智能评分系统

```python
from selection.scoring_engine import ScoringEngine

class ScoringEngine:
    """智能评分引擎"""
    
    def __init__(self, scoring_config):
        self.scoring_config = scoring_config
        self.weights = self._parse_weights()
    
    def calculate_composite_score(self, indicators):
        """
        计算综合评分
        
        评分模型:
        综合分 = Σ(因子分 × 权重)
        """
        total_score = 0
        total_weight = 0
        
        for factor_name, factor_config in self.scoring_config.items():
            factor_score = self._calculate_factor_score(
                factor_name, factor_config, indicators
            )
            weight = factor_config.get('weight', 1.0)
            
            total_score += factor_score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0
    
    def _calculate_factor_score(self, factor_name, config, indicators):
        """计算单个因子评分"""
        if factor_name == 'technical_score':
            return self._calculate_technical_score(config, indicators)
        elif factor_name == 'zxm_score':
            return self._calculate_zxm_score(config, indicators)
        elif factor_name == 'momentum_score':
            return self._calculate_momentum_score(config, indicators)
        else:
            return 0
    
    def _calculate_zxm_score(self, config, indicators):
        """计算ZXM综合评分"""
        zxm_components = config.get('components', [])
        total_score = 0
        total_weight = 0
        
        for component in zxm_components:
            indicator_name = component['indicator']
            weight = component.get('weight', 1.0)
            
            if indicator_name in indicators:
                score = indicators[indicator_name]
                total_score += score * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0

# 使用示例
scoring_config = {
    'technical_score': {
        'weight': 0.4,
        'components': [
            {'indicator': 'RSI', 'weight': 0.3},
            {'indicator': 'MACD', 'weight': 0.4},
            {'indicator': 'KDJ', 'weight': 0.3}
        ]
    },
    'zxm_score': {
        'weight': 0.6,
        'components': [
            {'indicator': 'ZXM_BUYPOINT', 'weight': 0.4},
            {'indicator': 'ZXM_TREND', 'weight': 0.3},
            {'indicator': 'ZXM_ELASTICITY', 'weight': 0.3}
        ]
    }
}

scoring_engine = ScoringEngine(scoring_config)
composite_score = scoring_engine.calculate_composite_score(indicators)
print(f"综合评分: {composite_score:.2f}")
```

---

## 📋 API参考

### 核心选股API

#### 1. 基础选股API

```python
def select_stocks(
    stock_pool: List[str],
    config_path: str,
    max_results: int = 50
) -> List[Dict]:
    """
    基础选股功能
    
    Args:
        stock_pool: 股票池
        config_path: 配置文件路径
        max_results: 最大结果数量
    
    Returns:
        List[Dict]: 选股结果
    """

# 使用示例
from selection.stock_selector import StockSelector

selector = StockSelector()
results = selector.select_stocks(
    stock_pool=['000001', '000002', '000858'],
    config_path='config/basic_strategy.yaml',
    max_results=20
)

for result in results:
    print(f"{result['stock_code']}: {result['score']:.2f} - {result['reason']}")
```

#### 2. 高级选股API

```python
def advanced_select_stocks(
    stock_pool: List[str],
    strategy_config: Dict,
    enable_parallel: bool = True,
    enable_cache: bool = True
) -> Dict:
    """
    高级选股功能
    
    Args:
        stock_pool: 股票池
        strategy_config: 策略配置
        enable_parallel: 启用并行处理
        enable_cache: 启用缓存
    
    Returns:
        Dict: 详细选股结果和统计信息
    """

# 使用示例
from selection.advanced_stock_selector import AdvancedStockSelector

advanced_selector = AdvancedStockSelector(
    enable_parallel=True,
    max_workers=8,
    enable_cache=True
)

strategy_config = {
    'name': '多因子选股',
    'conditions': [...],
    'scoring': {...}
}

results = advanced_selector.advanced_select_stocks(
    stock_pool=large_stock_pool,
    strategy_config=strategy_config
)

print(f"筛选结果: {len(results['selected_stocks'])} 只股票")
print(f"处理时间: {results['execution_time']:.2f} 秒")
print(f"筛选效率: {results['selection_rate']:.1%}")
```

#### 3. 实时选股API

```python
def realtime_stock_selection(
    strategy_name: str,
    update_interval: int = 300,
    alert_threshold: float = 80.0
) -> None:
    """
    实时选股监控
    
    Args:
        strategy_name: 策略名称
        update_interval: 更新间隔（秒）
        alert_threshold: 提醒阈值
    """

# 使用示例
from selection.realtime_selector import RealtimeStockSelector

realtime_selector = RealtimeStockSelector()
realtime_selector.start_realtime_selection(
    strategy_name='ZXM买点策略',
    update_interval=300,  # 5分钟更新
    alert_threshold=85.0
)
```

---

## 🎯 使用示例

### 基础选股示例

```python
from selection.stock_selector import StockSelector
import yaml

# 创建选股器
selector = StockSelector()

# 定义股票池（示例：沪深300成分股）
stock_pool = [
    '000001', '000002', '000858', '002415', '002594',
    '600036', '600519', '600887', '000858', '002142'
]

# 基础选股配置
basic_config = {
    'basic_filters': {
        'market_cap': {'min': 100, 'max': 2000},
        'price_range': {'min': 10, 'max': 200}
    },
    'technical_conditions': [
        {
            'indicator': 'RSI',
            'condition': 'between',
            'values': [30, 70],
            'weight': 0.3
        },
        {
            'indicator': 'ZXM_BUYPOINT',
            'condition': 'greater_than',
            'value': 75,
            'weight': 0.7
        }
    ]
}

# 执行选股
print("开始执行选股...")
results = selector.select_stocks_with_config(stock_pool, basic_config)

print(f"\n=== 选股结果 ===")
print(f"筛选股票数: {len(results)}")

for i, result in enumerate(results[:10], 1):
    print(f"{i:2d}. {result['stock_code']} - "
          f"评分: {result['score']:.2f} - "
          f"买点分: {result['indicators']['ZXM_BUYPOINT']:.1f}")
```

### 高级多因子选股示例

```python
from selection.advanced_stock_selector import AdvancedStockSelector

# 创建高级选股器
advanced_selector = AdvancedStockSelector(
    enable_parallel=True,
    max_workers=8,
    enable_cache=True
)

# 多因子选股策略
multi_factor_strategy = {
    'name': '多因子量化选股',
    'stages': [
        {
            'name': '基础筛选',
            'conditions': [
                'market_cap > 50',
                'price > 5',
                'avg_volume_20d > 1000000'
            ]
        },
        {
            'name': '技术筛选',
            'conditions': [
                'RSI < 80 AND RSI > 20',
                'MACD_histogram > 0',
                'MA5 > MA10'
            ]
        },
        {
            'name': 'ZXM筛选',
            'conditions': [
                'zxm_buypoint_score > 70',
                'zxm_trend_score > 65'
            ]
        }
    ],
    'scoring_model': {
        'technical_score': {
            'weight': 0.4,
            'components': [
                {'indicator': 'RSI', 'weight': 0.25},
                {'indicator': 'MACD', 'weight': 0.35},
                {'indicator': 'KDJ', 'weight': 0.25},
                {'indicator': 'BOLL', 'weight': 0.15}
            ]
        },
        'zxm_score': {
            'weight': 0.6,
            'components': [
                {'indicator': 'ZXM_BUYPOINT', 'weight': 0.4},
                {'indicator': 'ZXM_TREND', 'weight': 0.3},
                {'indicator': 'ZXM_ELASTICITY', 'weight': 0.3}
            ]
        }
    }
}

# 大规模股票池
large_stock_pool = get_all_a_shares()  # 假设获取所有A股

print(f"开始多因子选股，股票池大小: {len(large_stock_pool)}")
start_time = time.time()

# 执行高级选股
results = advanced_selector.advanced_select_stocks(
    stock_pool=large_stock_pool,
    strategy_config=multi_factor_strategy
)

end_time = time.time()
execution_time = end_time - start_time

print(f"\n=== 选股统计 ===")
print(f"总处理股票: {results['total_processed']}")
print(f"筛选通过股票: {len(results['selected_stocks'])}")
print(f"筛选通过率: {len(results['selected_stocks'])/results['total_processed']:.2%}")
print(f"总执行时间: {execution_time:.2f} 秒")
print(f"平均处理速度: {results['total_processed']/execution_time:.0f} 股/秒")

print(f"\n=== 性能优化效果 ===")
print(f"并行处理提升: {results['parallel_improvement']:.1f}%")
print(f"缓存命中率: {results['cache_hit_rate']:.1%}")

print(f"\n=== 前10名选股结果 ===")
for i, stock in enumerate(results['selected_stocks'][:10], 1):
    print(f"{i:2d}. {stock['stock_code']} - "
          f"综合评分: {stock['composite_score']:.2f} - "
          f"技术分: {stock['technical_score']:.1f} - "
          f"ZXM分: {stock['zxm_score']:.1f}")
```

### 策略回测示例

```python
from selection.strategy_backtester import StrategyBacktester

# 创建策略回测器
backtester = StrategyBacktester()

# 回测配置
backtest_config = {
    'start_date': '2024-01-01',
    'end_date': '2024-12-31',
    'rebalance_frequency': 'monthly',  # 月度调仓
    'max_positions': 20,               # 最大持仓数
    'strategy': multi_factor_strategy
}

print("开始策略回测...")
backtest_results = backtester.run_backtest(backtest_config)

print(f"\n=== 回测结果 ===")
print(f"总收益率: {backtest_results['total_return']:.2%}")
print(f"年化收益率: {backtest_results['annual_return']:.2%}")
print(f"最大回撤: {backtest_results['max_drawdown']:.2%}")
print(f"夏普比率: {backtest_results['sharpe_ratio']:.2f}")
print(f"胜率: {backtest_results['win_rate']:.1%}")

print(f"\n=== 选股效果分析 ===")
print(f"平均选股数量: {backtest_results['avg_selected_stocks']:.0f}")
print(f"选股成功率: {backtest_results['selection_success_rate']:.1%}")
print(f"平均持有期收益: {backtest_results['avg_holding_return']:.2%}")
```

---

## ❓ 常见问题

### Q1: 如何设计有效的选股策略？

A: 有效选股策略设计原则：
1. 多因子组合：技术指标 + ZXM评分 + 基本面
2. 分层筛选：基础筛选 → 技术筛选 → 精选
3. 动态权重：根据市场环境调整指标权重
4. 回测验证：历史数据验证策略有效性

### Q2: 如何处理大规模股票池筛选？

A: 大规模筛选优化方案：
1. 启用并行处理（8进程并行）
2. 使用智能缓存减少重复计算
3. 分批处理避免内存溢出
4. 预筛选减少计算量

### Q3: 选股结果如何验证？

A: 结果验证方法：
1. 历史回测验证策略有效性
2. 样本外测试检验泛化能力
3. 对比基准指数表现
4. 风险调整后收益分析

---

*可配置选股模块文档版本: v2.0*  
*最后更新: 2025-06-15*  
*支持86个技术指标和ZXM体系*
