# ZXM体系指标回测使用说明

本文档介绍如何使用ZXM体系指标进行股票回测分析。

## ZXM体系指标简介

ZXM体系是一套完整的选股模型，由三部分组成：
1. 趋势识别（7个公式）
2. 弹性识别（2个公式）
3. 买点识别（5个公式）

所有指标已整合到系统中，并支持在回测过程中使用。

## 已实现的ZXM指标

### 趋势识别指标
- ZXM日线上移趋势：判断60日或120日均线是否向上移动
- ZXM周线上移趋势：判断周线10周、20周或30周均线是否向上移动
- ZXM月KDJ·D及K上移趋势：判断月线KDJ指标的D值和K值是否同时向上移动
- ZXM周KDJ·D/DEA上移趋势：判断周线KDJ指标的D值或MACD的DEA值是否有一个向上移动
- ZXM周KDJ·D上移趋势：判断周线KDJ指标的D值是否向上移动
- ZXM月MACD<1.5趋势：判断月线MACD指标是否小于1.5
- ZXM周MACD<2趋势：判断周线MACD指标是否小于2

### 弹性识别指标
- ZXM振幅弹性：判断近120日内是否有日振幅超过8.1%的情况
- ZXM涨幅弹性：判断近80日内是否有日涨幅超过7%的情况

### 买点识别指标
- ZXM日MACD买点：判断日线MACD指标是否小于0.9
- ZXM换手买点：判断日线换手率是否大于0.7%
- ZXM缩量买点：判断成交量是否较2日平均成交量缩减10%以上
- ZXM回踩均线买点：判断收盘价是否回踩至20日、30日、60日或120日均线的4%以内
- ZXM BS吸筹买点：判断60分钟级别是否存在低位吸筹特征

## 回测结果说明

### 指标周期显示
回测结果中的所有指标将明确显示其所属周期，例如：
- 日线_ZXM趋势指标
- 周线_ZXM弹性指标
- 60分钟_ZXM买点指标

这样可以清晰区分不同周期的指标数据，避免混淆。

### 指标得分分类
ZXM指标得分按类别分别显示：
- ZXM趋势得分
- ZXM弹性得分
- ZXM买点得分

每类得分单独计算，不合并为一个总分，便于更精确地分析各方面的表现。

### 共性技术形态选股策略
回测结果报告中新增了"共性技术形态选股策略"部分，该部分会自动根据回测样本中出现频率较高的技术形态，总结出可直接实现的选股策略：

1. **ZXM体系选股策略**：根据回测中表现最佳的ZXM指标组合生成的策略
   - 趋势条件：列出最关键的ZXM趋势指标
   - 弹性条件：列出最关键的ZXM弹性指标
   - 买点条件：列出最关键的ZXM买点指标

2. **通用技术形态选股策略**：根据所有技术形态出现频率生成的策略
   - 必要条件：高频出现（>50%）的技术形态，必须全部满足
   - 重要条件：中频出现（30%-50%）的技术形态，至少满足2个
   - 确认条件：关键技术形态（如金叉、底背离等），至少满足1个

3. **策略代码实现示例**：提供两种选股策略的Python代码实现，可直接复制修改后用于实盘选股

## 回测使用方法

### 使用方式一：单股回测

对单只股票进行ZXM体系回测分析，可以使用以下命令：

```bash
python scripts/backtest/zxm_backtest_example.py --mode single --code 股票代码 --date 买点日期
```

例如：
```bash
python scripts/backtest/zxm_backtest_example.py --mode single --code 600519 --date 20230315
```

### 使用方式二：批量回测

从文件中批量读取股票信息进行ZXM体系回测分析，可以使用以下命令：

```bash
python scripts/backtest/zxm_backtest_example.py --mode batch --input 输入文件路径 --pattern "买点类型描述"
```

例如：
```bash
python scripts/backtest/zxm_backtest_example.py --mode batch --input data/zxm_backtest_samples.txt --pattern "ZXM选股"
```

输入文件格式为每行一个股票信息，包含股票代码和买点日期，以逗号分隔，例如：
```
600519,20230315
000001,20230610
```

### 回测结果

回测结果将以JSON格式和Markdown格式保存在`data/result/回测结果/`目录下，包含以下内容：

1. 各指标的计算结果（明确标注周期）
2. 技术形态统计
3. 共性技术形态选股策略（可直接应用于实盘）
4. 各股票分析结果
5. 结论和建议

## 如何将回测结果转化为实战策略

回测报告中的共性技术形态选股策略可以直接用于实战选股：

1. **复制策略代码**：从报告中复制策略代码示例
2. **修改参数**：根据最新市场环境和自己的交易习惯调整参数
3. **测试验证**：在小范围样本上测试策略有效性
4. **应用实战**：将策略应用到实际选股过程中

可以使用以下代码示例快速实现：

```python
# 载入策略代码
from strategy.zxm_strategy import zxm_select_strategy, technical_select_strategy

# 获取股票池数据
stock_list = get_stock_list()
selected_stocks = []

# 应用选股策略
for stock_code in stock_list:
    stock_data = get_stock_data(stock_code)
    
    # 使用ZXM体系选股策略
    if zxm_select_strategy(stock_data):
        selected_stocks.append(stock_code)
        
    # 或使用通用技术形态选股策略
    elif technical_select_strategy(stock_data):
        selected_stocks.append(stock_code)

# 输出选股结果
print(f"选出股票数量: {len(selected_stocks)}")
print(f"选出股票列表: {selected_stocks}")
```

## 常见问题

### 如何识别最优的ZXM指标组合？

可以通过批量回测，观察哪些ZXM指标在成功买点中出现频率最高，从而确定最优的指标组合。

### 如何处理不同周期的指标冲突？

当不同周期的指标出现冲突信号时，通常优先考虑更高级别的周期信号，如日线信号优先于分钟线信号，周线信号优先于日线信号。

## 注意事项

1. 部分指标（如周线、月线指标）需要相应周期的数据，如果数据库中没有相应周期的数据，这些指标可能无法计算。

2. 为了计算完整的ZXM指标，建议获取足够长的历史数据，至少包含买点日期之前的120个交易日数据。

3. 计算ZXM买点-换手率指标需要流通股本数据，如果数据库中没有该数据，系统会使用成交量的100倍作为模拟值。

4. ZXM综合买点的阈值为60分，当综合得分超过60分时，会标记为ZXM综合买点。

## 代码集成

如需在自己的代码中集成ZXM体系指标分析，可以参考以下示例：

```python
from unified_backtest import UnifiedBacktest

# 创建回测实例
backtest = UnifiedBacktest()

# 分析股票
result = backtest.analyze_stock('000001', '20230315', 'ZXM选股')

# 获取日线周期的ZXM指标结果
if result and 'periods' in result and 'daily' in result['periods']:
    daily_result = result['periods']['daily']
    if 'indicators' in daily_result and 'zxm' in daily_result['indicators']:
        zxm_indicators = daily_result['indicators']['zxm']
        # 获取ZXM综合得分
        if 'score' in zxm_indicators:
            total_score = zxm_indicators['score']['total_score']
            print(f"ZXM综合得分: {total_score}")
``` 