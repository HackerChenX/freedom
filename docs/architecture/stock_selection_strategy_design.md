# 选股策略设计文档 - 生产级选股核心引擎

## 🎯 核心定位

### 选股是系统的核心功能
**一切都是为了最终在生产级执行选股做配合**：
- 技术指标分析 → 为选股提供技术条件
- 买点回测分析 → 为选股策略提供历史验证  
- 实时监控系统 → 为选股执行提供实时支撑
- 数据管理系统 → 为选股提供高质量数据基础

### 设计理念
参考**通达信公式选股**的成功经验，构建更加灵活、强大的选股策略配置引擎：
- **公式化表达**: 支持类似通达信的公式化条件表达
- **灵活组合**: 支持复杂的逻辑组合和嵌套条件
- **实时执行**: 支持生产级的实时选股执行
- **性能优化**: 针对大规模股票池的高性能优化

## 🏗️ 选股策略架构设计

### 三层选股架构
```
┌─────────────────────────────────────────────────────────────┐
│                    选股策略执行层                            │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │ 实时选股引擎 │ │ 批量选股引擎 │ │ 回测选股引擎 │          │
│  │ - 实时执行  │ │ - 批量处理  │ │ - 历史验证  │          │
│  │ - 增量更新  │ │ - 全量扫描  │ │ - 策略测试  │          │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                    选股策略配置层                            │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │ 公式解析引擎 │ │ 条件组合引擎 │ │ 参数优化引擎 │          │
│  │ - 公式编译  │ │ - 逻辑组合  │ │ - 参数调优  │          │
│  │ - 语法检查  │ │ - 条件嵌套  │ │ - 效果评估  │          │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                    选股数据支撑层                            │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │ 技术指标数据 │ │ 基本面数据  │ │ 市场数据    │          │
│  │ - 103个指标 │ │ - 财务指标  │ │ - 行情数据  │          │
│  │ - 多周期计算│ │ - 估值指标  │ │ - 资金数据  │          │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

## 📝 选股公式设计 (参考通达信思路)

### 公式语法设计

#### 基础语法结构
```python
# 类通达信公式语法
# 条件表达式 AND/OR 条件表达式 AND/OR ...

# 示例1: 简单技术指标条件
MACD_DAILY > 0 AND RSI_DAILY < 30 AND VOL_DAILY > MA_VOL_5

# 示例2: 多周期组合条件  
MACD_DAILY > 0 AND MACD_WEEKLY > 0 AND KDJ_30MIN_K > KDJ_30MIN_D

# 示例3: 复杂嵌套条件
(MACD_DAILY > 0 AND RSI_DAILY < 30) OR (BOLL_DAILY_UPPER_BREAK AND VOL_RATIO > 2)

# 示例4: 历史回测验证条件
PATTERN_HIT("MACD_DAILY_GOLDEN_CROSS", 5) AND SUCCESS_RATE("KDJ_OVERSOLD", 30) > 0.7
```

#### 支持的条件类型

##### 1. 技术指标形态条件
```python
# 指标形态条件 (形态名称与指标注册的形态名称一致)
GOLDEN_CROSS(indicator="MACD", period="daily")           # MACD金叉形态，日线周期
OVERSOLD_RECOVERY(indicator="RSI", period="daily")       # RSI超卖恢复形态，日线周期
BULLISH_DIVERGENCE(indicator="KDJ", period="weekly")     # KDJ牛市背离形态，周线周期

# 简化语法 (推荐)
GOLDEN_CROSS.MACD.DAILY                                 # MACD金叉，日线
OVERSOLD_RECOVERY.RSI.DAILY                             # RSI超卖恢复，日线
UPPER_BREAK.BOLL.DAILY                                  # 布林上轨突破，日线
DOJI.CANDLESTICK.DAILY                                  # 十字星形态，日线

# 数值条件 (基于指标计算结果)
MACD.VALUE(period="daily") > 0                          # MACD数值大于0，日线
RSI.VALUE(period="daily") BETWEEN 20 AND 80             # RSI数值在20-80之间，日线
```

##### 2. 多周期条件
```python
# 多周期同向确认 (使用形态名称)
GOLDEN_CROSS.MACD.15MIN AND GOLDEN_CROSS.MACD.30MIN AND GOLDEN_CROSS.MACD.DAILY

# 多周期背离
OVERBOUGHT.RSI.DAILY AND OVERSOLD.RSI.WEEKLY

# 多周期形态共振
BULLISH_DIVERGENCE.MACD.DAILY AND TREND_REVERSAL.RSI.DAILY AND VOLUME_SURGE.VOL.DAILY
```

##### 3. 历史验证条件
```python
# 历史成功率验证 (使用注册的形态名称)
SUCCESS_RATE("GOLDEN_CROSS", indicator="MACD", period="daily", days=30) > 0.7

# 历史形态命中
PATTERN_HIT("OVERSOLD_RECOVERY", indicator="KDJ", period="daily", within_days=5)

# 回测验证
BACKTEST_SCORE("BULLISH_DIVERGENCE", indicator="MACD", period="weekly") > 80
```

##### 4. 基本面条件
```python
# 财务指标
PE < 20 AND PB < 2 AND ROE > 15

# 成长性指标  
REVENUE_GROWTH > 0.2 AND PROFIT_GROWTH > 0.15

# 估值指标
PEG < 1 AND PS < 3
```

##### 5. 市场条件
```python
# 成交量条件
VOL_RATIO > 2                     # 量比大于2
TURNOVER_RATE > 5                 # 换手率大于5%

# 价格条件
PRICE_CHANGE > 0.05               # 涨幅大于5%
CLOSE > MA_20                     # 收盘价大于20日均线

# 市场环境
MARKET_TREND == "BULL"            # 牛市环境
INDUSTRY_STRENGTH > 0.8           # 行业强度>0.8
```

### 策略配置示例

#### 策略1: 技术面突破策略
```json
{
  "strategy_id": "TECH_BREAKTHROUGH_001",
  "strategy_name": "技术面突破策略",
  "description": "基于多周期技术指标形态的突破选股",
  "formula": "GOLDEN_CROSS.MACD.DAILY AND GOLDEN_CROSS.MACD.WEEKLY AND NEUTRAL_ZONE.RSI.DAILY AND VOLUME_SURGE.VOL.DAILY",
  "conditions": {
    "pattern_conditions": [
      {
        "pattern": "GOLDEN_CROSS",
        "indicator": "MACD",
        "period": "daily",
        "weight": 0.3
      },
      {
        "pattern": "GOLDEN_CROSS",
        "indicator": "MACD",
        "period": "weekly",
        "weight": 0.3
      },
      {
        "pattern": "NEUTRAL_ZONE",
        "indicator": "RSI",
        "period": "daily",
        "parameters": {"range": [30, 70]},
        "weight": 0.2
      },
      {
        "pattern": "VOLUME_SURGE",
        "indicator": "VOL",
        "period": "daily",
        "parameters": {"ratio_threshold": 1.5},
        "weight": 0.2
      }
    ]
  },
  "filters": {
    "market_cap": {"min": 1000000000},
    "exclude_st": true,
    "exclude_suspended": true
  },
  "scoring": {
    "pattern_score_weight": 0.7,
    "momentum_score_weight": 0.2,
    "volume_score_weight": 0.1
  }
}
```

#### 策略2: 历史回测验证策略
```json
{
  "strategy_id": "BACKTEST_VERIFIED_001",
  "strategy_name": "历史回测验证策略",
  "description": "基于历史形态成功率的选股策略",
  "formula": "SUCCESS_RATE('GOLDEN_CROSS', 'MACD', 'daily', 30) > 0.7 AND PATTERN_HIT('OVERSOLD_RECOVERY', 'KDJ', 'daily', 5) AND BACKTEST_SCORE('BULLISH_DIVERGENCE', 'MACD', 'weekly') > 80",
  "conditions": {
    "historical_validation": [
      {
        "pattern": "GOLDEN_CROSS",
        "indicator": "MACD",
        "period": "daily",
        "success_rate_threshold": 0.7,
        "evaluation_days": 30,
        "weight": 0.4
      },
      {
        "pattern": "OVERSOLD_RECOVERY",
        "indicator": "KDJ",
        "period": "daily",
        "hit_within_days": 5,
        "weight": 0.3
      },
      {
        "pattern": "BULLISH_DIVERGENCE",
        "indicator": "MACD",
        "period": "weekly",
        "backtest_score_threshold": 80,
        "weight": 0.3
      }
    ]
  },
  "validation": {
    "min_sample_size": 100,
    "confidence_level": 0.95,
    "lookback_period": 252
  }
}
```

#### 策略3: 复合条件策略
```json
{
  "strategy_id": "COMPLEX_COMBO_001",
  "strategy_name": "复合条件组合策略", 
  "description": "多维度复合条件的灵活组合",
  "formula": "(MACD_DAILY > 0 AND RSI_DAILY < 30) OR (BOLL_UPPER_BREAK AND VOL_RATIO > 2) AND PE < 20 AND MARKET_TREND == 'BULL'",
  "conditions": {
    "logic_groups": [
      {
        "operator": "OR",
        "groups": [
          {
            "operator": "AND", 
            "conditions": [
              {"indicator": "MACD", "period": "daily", "operator": ">", "value": 0},
              {"indicator": "RSI", "period": "daily", "operator": "<", "value": 30}
            ]
          },
          {
            "operator": "AND",
            "conditions": [
              {"pattern": "BOLL_UPPER_BREAK", "period": "daily"},
              {"indicator": "VOL_RATIO", "operator": ">", "value": 2}
            ]
          }
        ]
      }
    ],
    "global_filters": [
      {"field": "PE", "operator": "<", "value": 20},
      {"field": "MARKET_TREND", "operator": "==", "value": "BULL"}
    ]
  }
}
```

## 🚀 选股执行引擎设计

### 实时选股引擎
```python
class RealTimeStockSelectionEngine:
    """实时选股执行引擎"""
    
    def __init__(self):
        self.formula_parser = FormulaParser()
        self.condition_evaluator = ConditionEvaluator()
        self.data_provider = RealTimeDataProvider()
        
    def execute_selection(self, strategy_config: Dict) -> List[str]:
        """执行实时选股"""
        # 1. 解析策略公式
        parsed_conditions = self.formula_parser.parse(strategy_config['formula'])
        
        # 2. 获取股票池
        stock_pool = self._get_stock_pool(strategy_config.get('filters', {}))
        
        # 3. 并行评估条件
        selected_stocks = []
        for stock_code in stock_pool:
            if self._evaluate_stock(stock_code, parsed_conditions):
                selected_stocks.append(stock_code)
        
        # 4. 评分排序
        scored_stocks = self._score_and_rank(selected_stocks, strategy_config)
        
        return scored_stocks
    
    def _evaluate_stock(self, stock_code: str, conditions: Dict) -> bool:
        """评估单只股票是否符合条件"""
        # 获取股票数据
        stock_data = self.data_provider.get_stock_data(stock_code)
        
        # 评估所有条件
        return self.condition_evaluator.evaluate(stock_data, conditions)
```

### 公式解析引擎
```python
class FormulaParser:
    """选股公式解析引擎"""
    
    def parse(self, formula: str) -> Dict:
        """解析选股公式为可执行的条件树"""
        # 1. 词法分析
        tokens = self._tokenize(formula)
        
        # 2. 语法分析
        ast = self._parse_expression(tokens)
        
        # 3. 语义分析
        conditions = self._build_conditions(ast)
        
        return conditions
    
    def _tokenize(self, formula: str) -> List[Token]:
        """词法分析 - 将公式分解为token"""
        # 支持的操作符和关键字
        operators = ['>', '<', '>=', '<=', '==', '!=', 'BETWEEN', 'CROSS_UP', 'CROSS_DOWN']
        keywords = ['AND', 'OR', 'NOT', 'PATTERN_HIT', 'SUCCESS_RATE']
        
        # 实现词法分析逻辑
        pass
    
    def _parse_expression(self, tokens: List[Token]) -> AST:
        """语法分析 - 构建抽象语法树"""
        # 实现递归下降解析器
        pass
```

### 条件评估引擎
```python
class ConditionEvaluator:
    """条件评估引擎"""
    
    def evaluate(self, stock_data: Dict, conditions: Dict) -> bool:
        """评估股票数据是否满足条件"""
        return self._evaluate_node(stock_data, conditions['root'])
    
    def _evaluate_node(self, stock_data: Dict, node: Dict) -> bool:
        """递归评估条件节点"""
        if node['type'] == 'logical':
            return self._evaluate_logical(stock_data, node)
        elif node['type'] == 'comparison':
            return self._evaluate_comparison(stock_data, node)
        elif node['type'] == 'pattern':
            return self._evaluate_pattern(stock_data, node)
        elif node['type'] == 'function':
            return self._evaluate_function(stock_data, node)
    
    def _evaluate_logical(self, stock_data: Dict, node: Dict) -> bool:
        """评估逻辑条件 (AND/OR/NOT)"""
        operator = node['operator']
        operands = [self._evaluate_node(stock_data, child) for child in node['children']]
        
        if operator == 'AND':
            return all(operands)
        elif operator == 'OR':
            return any(operands)
        elif operator == 'NOT':
            return not operands[0]
    
    def _evaluate_comparison(self, stock_data: Dict, node: Dict) -> bool:
        """评估比较条件"""
        left_value = self._get_indicator_value(stock_data, node['left'])
        right_value = node['right']['value']
        operator = node['operator']
        
        if operator == '>':
            return left_value > right_value
        elif operator == '<':
            return left_value < right_value
        # ... 其他比较操作符
```

## 🎯 生产级优化策略

### 性能优化
1. **并行计算**: 股票池并行评估，提高选股速度
2. **增量更新**: 只计算变化的数据，避免全量重算
3. **缓存机制**: 指标计算结果缓存，减少重复计算
4. **索引优化**: 数据库查询索引优化，提高数据获取速度

### 可靠性保证
1. **容错机制**: 单只股票评估失败不影响整体选股
2. **数据验证**: 输入数据质量检查和异常处理
3. **结果验证**: 选股结果的合理性检查
4. **监控告警**: 选股执行状态监控和异常告警

### 扩展性设计
1. **插件化条件**: 支持自定义条件类型的插件扩展
2. **策略模板**: 提供常用策略模板，快速配置
3. **API接口**: 提供标准API接口，支持外部系统集成
4. **配置管理**: 灵活的策略配置管理和版本控制

## 📋 更多策略配置示例

### 策略4: 多周期共振策略
```json
{
  "strategy_id": "MULTI_PERIOD_RESONANCE_001",
  "strategy_name": "多周期共振选股策略",
  "description": "多个周期技术指标同向共振的选股策略",
  "formula": "MACD_15MIN > 0 AND MACD_30MIN > 0 AND MACD_60MIN > 0 AND MACD_DAILY > 0 AND RSI_DAILY BETWEEN 40 AND 60",
  "conditions": {
    "multi_period_resonance": {
      "indicator": "MACD",
      "periods": ["15min", "30min", "60min", "daily"],
      "condition": "ALL_POSITIVE",
      "confirmation_strength": 0.8
    },
    "momentum_filter": {
      "indicator": "RSI",
      "period": "daily",
      "range": [40, 60],
      "description": "避免超买超卖区域"
    }
  },
  "scoring": {
    "resonance_score": 0.5,
    "momentum_score": 0.3,
    "volume_score": 0.2
  }
}
```

### 策略5: 历史形态验证策略
```json
{
  "strategy_id": "HISTORICAL_PATTERN_001",
  "strategy_name": "历史形态验证策略",
  "description": "基于历史成功形态的验证选股",
  "formula": "PATTERN_HIT('MACD_GOLDEN_CROSS', 3) AND SUCCESS_RATE('BOLL_BREAKTHROUGH', 20) > 0.75 AND RECENT_PERFORMANCE('KDJ_OVERSOLD', 10) > 0.6",
  "conditions": {
    "pattern_verification": [
      {
        "pattern": "MACD_GOLDEN_CROSS",
        "hit_within_days": 3,
        "weight": 0.4
      },
      {
        "pattern": "BOLL_BREAKTHROUGH",
        "success_rate": 0.75,
        "evaluation_period": 20,
        "weight": 0.35
      },
      {
        "pattern": "KDJ_OVERSOLD",
        "recent_performance": 0.6,
        "lookback_days": 10,
        "weight": 0.25
      }
    ]
  },
  "validation": {
    "min_historical_samples": 50,
    "confidence_threshold": 0.8
  }
}
```

### 策略6: 动态阈值策略
```json
{
  "strategy_id": "DYNAMIC_THRESHOLD_001",
  "strategy_name": "动态阈值适应策略",
  "description": "根据市场环境动态调整选股阈值",
  "formula": "ADAPTIVE_RSI(MARKET_VOLATILITY) AND DYNAMIC_MACD(MARKET_TREND) AND VOL_PERCENTILE > ADAPTIVE_VOL_THRESHOLD()",
  "conditions": {
    "adaptive_conditions": [
      {
        "indicator": "RSI",
        "period": "daily",
        "threshold_function": "ADAPTIVE_RSI",
        "parameters": {
          "base_oversold": 30,
          "base_overbought": 70,
          "volatility_adjustment": true,
          "market_regime_factor": 0.2
        }
      },
      {
        "indicator": "MACD",
        "period": "daily",
        "threshold_function": "DYNAMIC_MACD",
        "parameters": {
          "trend_sensitivity": 0.15,
          "momentum_factor": 0.1
        }
      }
    ],
    "market_environment": {
      "volatility_regime": "AUTO_DETECT",
      "trend_regime": "AUTO_DETECT",
      "adjustment_frequency": "DAILY"
    }
  }
}
```

### 策略7: 行业轮动策略
```json
{
  "strategy_id": "SECTOR_ROTATION_001",
  "strategy_name": "行业轮动选股策略",
  "description": "基于行业强度和轮动的选股策略",
  "formula": "INDUSTRY_STRENGTH > 0.8 AND INDUSTRY_MOMENTUM > 0.6 AND RELATIVE_STRENGTH_VS_MARKET > 1.2 AND MACD_DAILY > 0",
  "conditions": {
    "sector_analysis": {
      "industry_strength_threshold": 0.8,
      "industry_momentum_threshold": 0.6,
      "relative_strength_threshold": 1.2,
      "sector_rotation_signal": true
    },
    "individual_stock": {
      "technical_confirmation": ["MACD_DAILY > 0"],
      "relative_performance": "TOP_30_PERCENT_IN_SECTOR"
    }
  },
  "sector_weights": {
    "technology": 0.25,
    "healthcare": 0.20,
    "finance": 0.15,
    "consumer": 0.15,
    "industrial": 0.10,
    "energy": 0.10,
    "materials": 0.05
  }
}
```

### 策略8: 量价配合策略
```json
{
  "strategy_id": "VOLUME_PRICE_001",
  "strategy_name": "量价配合突破策略",
  "description": "价格突破配合成交量放大的选股策略",
  "formula": "PRICE_BREAKTHROUGH AND VOLUME_SURGE AND VOLUME_PATTERN_CONFIRM AND TREND_ACCELERATION",
  "conditions": {
    "price_breakthrough": {
      "breakthrough_type": ["RESISTANCE_BREAK", "MA_BREAK", "BOLL_UPPER_BREAK"],
      "breakthrough_strength": 0.03,
      "confirmation_days": 2
    },
    "volume_analysis": {
      "volume_surge_ratio": 2.0,
      "volume_pattern": "INCREASING_VOLUME",
      "volume_ma_breakthrough": true,
      "relative_volume_percentile": 80
    },
    "trend_confirmation": {
      "trend_direction": "UP",
      "trend_acceleration": true,
      "momentum_confirmation": ["RSI_RISING", "MACD_RISING"]
    }
  }
}
```

## 🔧 策略执行优化

### 实时执行优化
```python
class OptimizedSelectionEngine:
    """优化的选股执行引擎"""

    def __init__(self):
        self.cache_manager = CacheManager()
        self.parallel_executor = ParallelExecutor()
        self.incremental_processor = IncrementalProcessor()

    def execute_optimized_selection(self, strategy: Dict) -> List[str]:
        """优化的选股执行"""
        # 1. 增量数据更新
        updated_stocks = self.incremental_processor.get_updated_stocks()

        # 2. 缓存命中检查
        cached_results = self.cache_manager.get_cached_results(strategy['strategy_id'])

        # 3. 并行处理
        if updated_stocks:
            new_results = self.parallel_executor.process_stocks(updated_stocks, strategy)
            # 合并缓存结果和新结果
            final_results = self._merge_results(cached_results, new_results)
        else:
            final_results = cached_results

        return final_results
```

### 策略性能监控
```python
class StrategyPerformanceMonitor:
    """策略性能监控"""

    def monitor_strategy_execution(self, strategy_id: str, execution_result: Dict):
        """监控策略执行性能"""
        metrics = {
            'execution_time': execution_result['execution_time'],
            'stocks_processed': execution_result['stocks_processed'],
            'stocks_selected': len(execution_result['selected_stocks']),
            'selection_rate': execution_result['selection_rate'],
            'cache_hit_rate': execution_result['cache_hit_rate']
        }

        # 记录性能指标
        self._record_metrics(strategy_id, metrics)

        # 性能告警
        if metrics['execution_time'] > self.performance_thresholds['max_execution_time']:
            self._trigger_performance_alert(strategy_id, metrics)
```

---

**选股是我们的核心功能，一切都是为了最终在生产级执行选股做配合！**

**通过灵活的公式配置和复杂的条件组合，我们能够适应各种投资策略和市场环境的需求。**

---

**文档类型**: 选股策略设计文档
**核心定位**: 生产级选股执行引擎
**文档版本**: v1.0
**创建时间**: 2025-09-04
