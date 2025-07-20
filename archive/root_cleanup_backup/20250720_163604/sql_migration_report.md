# SQL查询迁移报告

## 概述

- **扫描文件数**: 468
- **发现SQL查询数**: 2575
- **生成时间**: 2025-07-06 18:56:58

## 文件统计

### ./analysis/advanced_performance_analyzer.py
- 查询数量: 2

1. ```sql
import os
import sys
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import json
from datetime import datetime
import gc
# import psutil  # 可选依赖
...
```

2. ```sql
)
        
        analyzer = Optimized_buy_point_analyzer(enable_cache=False, enable_vectorization=True)
        buypoints_df = analyzer.load_buypoints_from_csv(buypoints_csv)
        test_df = buypo...
```

### ./analysis/advanced_vectorized_optimizer.py
- 查询数量: 1

1. ```sql
import os
import sys
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import warnings

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirna...
```

### ./analysis/buypoints/auto_indicator_analyzer.py
- 查询数量: 1

1. ```sql
import os
import sys
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union, Set
from datetime import datetime
import logging

# 添加项目根目录到Python路径
root_dir = ...
```

### ./analysis/buypoints/buypoint_batch_analyzer.py
- 查询数量: 1

1. ```sql
import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
from collections import Counter, defaultdict

...
```

### ./analysis/buypoints/buypoint_dimension_analyzer.py
- 查询数量: 131

1. ```sql
, big_up_big_volume)
        
        # 大阴大量（单日下跌超过3%且放量超过1.5倍）
        big_down_big_volume = 0
        for i in range(1, len(data)):
            if (data.iloc[i-1]['close'] / data.iloc[i]['close'] - ...
```

2. ```sql
]) / max_high > 0.05 and i - max_idx > 3:
                high_pullback.append(i)
        self._update_pattern_count(stats, "高位回调", len(high_pullback))
        
        # 低位反弹
        low_rebound = []...
```

3. ```sql
, len(doji))
        
        # 锤子线
        hammer = kline_data[(kline_data['lower_shadow'] > kline_data['body'] * 2) & 
                          (kline_data['upper_shadow'] < kline_data['body'] * 0....
```

4. ```sql
, 1)
        
        # 先量后价（先出现放量，后出现价格变动）
        volume_lead_price = 0
        for i in range(3, len(data)):
            # 前天放量（大于5日均量的1.5倍）但价格变动小
            if (data.iloc[i-2]['volume'] > data.il...
```

5. ```sql
]) / 2)]  # 收复部分跌幅
        self._update_pattern_count(stats, "启明星", len(morning_star))
        
        # 黄昏星
        evening_star = kline_data[(kline_data[
```

6. ```sql
# MA5上穿MA10
        ma5_cross_above_ma10 = data[(data['ma5'] > data['ma10']) & (data['ma5'].shift(1) <= data['ma10'].shift(1))]
        self._update_pattern_count(stats,
```

7. ```sql
] / total_volume if total_volume > 0 else 0
                
                # 单日占比超过40%，说明成交高度集中
                if today_ratio > 0.4:
                    self._update_pattern_count(stats, "量能高度集中(单日...
```

8. ```sql
, len(flag))
        
        # 杯柄形态
        cup_and_handle = self._detect_cup_and_handle(kline_data)
        self._update_pattern_count(stats,
```

9. ```sql
].rolling(20).mean()
        
        # 双底形态检测
        double_bottom = self._detect_double_bottom(kline_data)
        self._update_pattern_count(stats, "双底", len(double_bottom))
        
        # 双顶形...
```

10. ```sql
, volume_step_down)
        
        # 量能分布（单日成交量在总成交量中的占比）
        if len(data) >= 5:
            for i in range(5, len(data)):
                # 计算5日内成交量总和
                total_volume = data.iloc[i...
```

11. ```sql
, 1)
            else:
                consecutive_down = 0
        
        # 大涨
        big_up = data[data['price_change_pct'] > 0.05]  # 单日涨幅超过5%
        self._update_pattern_count(stats,
```

12. ```sql
, volume_up_price_down)
        
        # 量减价增（背离）
        volume_down_price_up = 0
        for i in range(1, len(data)):
            if (data.iloc[i]['volume'] < data.iloc[i-1]['volume'] * 0.8 and
 ...
```

13. ```sql
]):
                volume_step_up += 1
        self._update_pattern_count(stats, "梯量上升", volume_step_up)
        
        # 梯量下降（连续3日以上量能逐步萎缩）
        volume_step_down = 0
        for i in range(3, l...
```

14. ```sql
] * 1.5):
                big_up_big_volume += 1
        self._update_pattern_count(stats, "大阳大量", big_up_big_volume)
        
        # 大阴大量（单日下跌超过3%且放量超过1.5倍）
        big_down_big_volume = 0
       ...
```

15. ```sql
, volume_price_down)
        
        # 大阳大量（单日上涨超过3%且放量超过1.5倍）
        big_up_big_volume = 0
        for i in range(1, len(data)):
            if (data.iloc[i]['close'] / data.iloc[i-1]['close'] - 1 ...
```

16. ```sql
] > 3.0]
        self._update_pattern_count(stats, "大幅放量(>3倍均量)", len(large_volume_surge))
        
        # 极度放量（成交量大于5日均量的5倍）
        extreme_volume_surge = data[data[
```

17. ```sql
].min()
            
            if (max_price1 - mid_price) / max_price1 < 0.03:
                continue
                
            result.append(i)
            
        return result
    
    def...
```

18. ```sql
])]
        self._update_pattern_count(stats, "看涨吞没", len(bullish_engulfing))
        
        # 看跌吞没
        bearish_engulfing = kline_data[(kline_data[
```

19. ```sql
]) / 2)]  # 回撤部分涨幅
        self._update_pattern_count(stats, "黄昏星", len(evening_star))
        
        # 三只乌鸦
        three_black_crows = []
        for i in range(3, len(kline_data)):
            if...
```

20. ```sql
# 获取必要的前置数据
        kline_data['prev_close'] = kline_data['close'].shift(1)
        kline_data['prev_open'] = kline_data['open'].shift(1)
        kline_data['prev_high'] = kline_data['high'].shift(1)
...
```

21. ```sql
]):
                volume_down_price_up += 1
        self._update_pattern_count(stats, "量减价增", volume_down_price_up)
        
        # 量减价减（同向）
        volume_price_down = 0
        for i in range(1...
```

22. ```sql
].shift(1)) / 2)]
        self._update_pattern_count(stats, "穿刺线", len(piercing))
    
    def _analyze_combined_candle_patterns(self, kline_data: pd.Data_frame, stats: Dict[str, int]) -> None:
      ...
```

23. ```sql
, len(bearish_alignment))
        
        # 均线交叉密集
        ma_crosses = 0
        for i in range(1, len(data)):
            crosses = 0
            if (data.iloc[i]['ma5'] > data.iloc[i]['ma10'] and ...
```

24. ```sql
, len(break_high))
        
        # 跌破前低
        break_low = []
        for i in range(window, len(data)):
            prev_low = data.iloc[i-window:i-1]['low'].min()
            if data.iloc[i]['cl...
```

25. ```sql
, len(high_pullback))
        
        # 低位反弹
        low_rebound = []
        for i in range(window, len(data)):
            # 前window天的最低价
            min_low = data.iloc[i-window:i]['low'].min()
  ...
```

26. ```sql
, len(strong_trend))
            
            # 极强趋势 (ADX > 50)
            very_strong_trend = data[data['adx'] > 50]
            self._update_pattern_count(stats,
```

27. ```sql
, 1)
                if consecutive_down >= 5:  # 连续5日下跌
                    self._update_pattern_count(stats,
```

28. ```sql
] * 0.5)]
        self._update_pattern_count(stats, "锤子线", len(hammer))
        
        # 吊颈线
        hanging_man = kline_data[(kline_data[
```

29. ```sql
])]
        self._update_pattern_count(stats, "吊颈线", len(hanging_man))
        
        # 长腿十字星
        long_legged_doji = kline_data[(kline_data[
```

30. ```sql
, len(ma_converge))
        
        # 均线发散
        ma_diverge = data[((data['ma5'] - data['ma30']).abs() > (data['ma5'].shift(5) - data['ma30'].shift(5)).abs() * 1.2)]
        self._update_pattern_co...
```

31. ```sql
]):
                volume_up_price_down += 1
        self._update_pattern_count(stats, "量增价减", volume_up_price_down)
        
        # 量减价增（背离）
        volume_down_price_up = 0
        for i in rang...
```

32. ```sql
, 1)
            else:
                consecutive_up = 0
                
        # 连续下跌
        consecutive_down = 0
        for i in range(1, len(data)):
            if data.iloc[i]['close'] < data...
```

33. ```sql
# 简化检测逻辑
        result = []
        window = 40
        
        for i in range(window, len(data) - window):
            # 待实现
            pass
            
        return result
    
    def _update...
```

34. ```sql
] < 0.5]
        self._update_pattern_count(stats, "缩量(<0.5倍均量)", len(volume_shrink))
        
        # 极度缩量（成交量小于5日均量的0.3倍）
        extreme_volume_shrink = data[data[
```

35. ```sql
, 1)
            else:
                consecutive_surge = 0
        
        # 连续缩量
        consecutive_shrink = 0
        for i in range(1, len(data)):
            if data.iloc[i]['volume'] < data.i...
```

36. ```sql
]):
                volume_step_down += 1
        self._update_pattern_count(stats, "梯量下降", volume_step_down)
        
        # 量能分布（单日成交量在总成交量中的占比）
        if len(data) >= 5:
            for i in ra...
```

37. ```sql
]):
                three_black_crows.append(i)
        self._update_pattern_count(stats, "三只乌鸦", len(three_black_crows))
        
        # 三白兵
        three_white_soldiers = []
        for i in rang...
```

38. ```sql
, volume_lead_price)
        
        # 先价后量（先出现价格变动，后出现放量）
        price_lead_volume = 0
        for i in range(3, len(data)):
            # 前天价格变动大但量能一般
            if (abs(data.iloc[i-2]['close'] /...
```

39. ```sql
]:
                consecutive_up += 1
                if consecutive_up >= 3:  # 连续3日上涨
                    self._update_pattern_count(stats, "连续3日上涨", 1)
                if consecutive_up >= 5:  # 连...
```

40. ```sql
, len(up_strength))
            self._update_pattern_count(stats,
```

41. ```sql
].shift(1))]
        self._update_pattern_count(stats, "MA5上穿MA10", len(ma5_cross_above_ma10))
        
        # MA5下穿MA10
        ma5_cross_below_ma10 = data[(data[
```

42. ```sql
, len(double_bottom))
        
        # 双顶形态检测
        double_top = self._detect_double_top(kline_data)
        self._update_pattern_count(stats,
```

43. ```sql
, len(head_and_shoulders_top))
        
        # 三角形整理形态
        triangle = self._detect_triangle(kline_data)
        self._update_pattern_count(stats,
```

44. ```sql
].mean() * 0.3]
        self._update_pattern_count(stats, "十字星", len(doji))
        
        # 锤子线
        hammer = kline_data[(kline_data[
```

45. ```sql
# 计算量比（当日成交量/5日均量）
        data['volume_ratio'] = data['volume'] / data['volume_ma5']
        
        # 放量（成交量大于5日均量的2倍）
        volume_surge = data[data['volume_ratio'] > 2.0]
        self._update_p...
```

46. ```sql
] * 2.0):
                small_up_big_volume += 1
        self._update_pattern_count(stats, "小阳大量", small_up_big_volume)
        
        # 小阴大量（单日下跌小于2%但放量超过2倍）
        small_down_big_volume = 0
   ...
```

47. ```sql
: self._generate_feature_combinations(selected_features)
            }
            
            logger.info(f
```

48. ```sql
] > prev_high:
                break_high.append(i)
        self._update_pattern_count(stats, "突破前高", len(break_high))
        
        # 跌破前低
        break_low = []
        for i in range(window, len...
```

49. ```sql
) -> Dict[str, Any]:
        """
        分析买点趋势特征
        
        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期，格式YYYYMMDD
            end_date: 结束日期，格式YYYYMMDD
            perio...
```

50. ```sql
] > 3.0]
        self._update_pattern_count(stats, "量能突变(日环比>3倍)", len(volume_spike))
    
    def _analyze_volume_pattern(self, data: pd.Data_frame, stats: Dict[str, int]) -> None:
        """分析成交量形态...
```

51. ```sql
)
            
            # 计算简单的趋势强度：连续上涨或下跌的幅度
            up_strength = []
            down_strength = []
            
            for i in range(5, len(data)):
                # 过去5天的涨跌幅
        ...
```

52. ```sql
, volume_step_up)
        
        # 梯量下降（连续3日以上量能逐步萎缩）
        volume_step_down = 0
        for i in range(3, len(data)):
            if (data.iloc[i]['volume'] < data.iloc[i-1]['volume'] < 
        ...
```

53. ```sql
] > 25]
            self._update_pattern_count(stats, "强趋势(ADX>25)", len(strong_trend))
            
            # 极强趋势 (ADX > 50)
            very_strong_trend = data[data[
```

54. ```sql
, len(bullish_engulfing))
        
        # 看跌吞没
        bearish_engulfing = kline_data[(kline_data['open'] > kline_data['prev_close']) & 
                                     (kline_data['close'] < ...
```

55. ```sql
) -> Dict[str, Any]:
        """
        分析买点量能特征
        
        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期，格式YYYYMMDD
            end_date: 结束日期，格式YYYYMMDD
            perio...
```

56. ```sql
]):
                volume_price_down += 1
        self._update_pattern_count(stats, "量减价减", volume_price_down)
        
        # 大阳大量（单日上涨超过3%且放量超过1.5倍）
        big_up_big_volume = 0
        for i i...
```

57. ```sql
].shift(5)).abs() * 0.8)]
        self._update_pattern_count(stats, "均线收敛", len(ma_converge))
        
        # 均线发散
        ma_diverge = data[((data[
```

58. ```sql
] - 1) > 0.03):
                volume_lead_price += 1
        self._update_pattern_count(stats, "先量后价", volume_lead_price)
        
        # 先价后量（先出现价格变动，后出现放量）
        price_lead_volume = 0
       ...
```

59. ```sql
, volume_price_up)
        
        # 量增价减（背离）
        volume_up_price_down = 0
        for i in range(1, len(data)):
            if (data.iloc[i]['volume'] > data.iloc[i-1]['volume'] * 1.2 and
      ...
```

60. ```sql
, len(double_top))
        
        # 头肩底形态检测
        head_and_shoulders_bottom = self._detect_head_and_shoulders_bottom(kline_data)
        self._update_pattern_count(stats,
```

61. ```sql
] - 1
                
                if change_5d > 0.1:  # 5天涨幅超过10%
                    up_strength.append(i)
                elif change_5d < -0.1:  # 5天跌幅超过10%
                    down_strength....
```

62. ```sql
]):
                volume_price_up += 1
        self._update_pattern_count(stats, "量增价增", volume_price_up)
        
        # 量增价减（背离）
        volume_up_price_down = 0
        for i in range(1, len(d...
```

63. ```sql
, len(big_down))
        
        # 高位回调
        high_pullback = []
        window = 10
        for i in range(window, len(data)):
            # 前window天的最高价
            max_high = data.iloc[i-window:...
```

64. ```sql
, 1)
            else:
                consecutive_shrink = 0
        
        # 量能突变（今日成交量比昨日成交量增加200%以上）
        volume_change = data.copy()
        volume_change['volume_change_ratio'] = volume_cha...
```

65. ```sql
, 1)
                if consecutive_up >= 5:  # 连续5日上涨
                    self._update_pattern_count(stats,
```

66. ```sql
].shift(1))]
        self._update_pattern_count(stats, "MA10上穿MA20", len(ma10_cross_above_ma20))
        
        # MA10下穿MA20
        ma10_cross_below_ma20 = data[(data[
```

67. ```sql
] * 1.5):
                big_down_big_volume += 1
        self._update_pattern_count(stats, "大阴大量", big_down_big_volume)
        
        # 小阳大量（单日上涨小于2%但放量超过2倍）
        small_up_big_volume = 0
     ...
```

68. ```sql
].shift(1))]
        self._update_pattern_count(stats, "MA10下穿MA20", len(ma10_cross_below_ma20))
        
        # 均线多头排列（MA5 > MA10 > MA20 > MA30）
        bullish_alignment = data[(data[
```

69. ```sql
, len(long_legged_doji))
        
        # 射击之星
        shooting_star = kline_data[(kline_data['upper_shadow'] > kline_data['body'] * 2) & 
                                 (kline_data['lower_shadow'...
```

70. ```sql
, len(triangle))
        
        # 旗形整理
        flag = self._detect_flag(kline_data)
        self._update_pattern_count(stats,
```

71. ```sql
, volume_down_price_up)
        
        # 量减价减（同向）
        volume_price_down = 0
        for i in range(1, len(data)):
            if (data.iloc[i]['volume'] < data.iloc[i-1]['volume'] * 0.8 and
    ...
```

72. ```sql
# 梯量上升（连续3日以上量能逐步放大）
        volume_step_up = 0
        for i in range(3, len(data)):
            if (data.iloc[i]['volume'] > data.iloc[i-1]['volume'] > 
                data.iloc[i-2]['volume'] > da...
```

73. ```sql
, len(head_and_shoulders_bottom))
        
        # 头肩顶形态检测
        head_and_shoulders_top = self._detect_head_and_shoulders_top(kline_data)
        self._update_pattern_count(stats,
```

74. ```sql
] < prev_low:
                break_low.append(i)
        self._update_pattern_count(stats, "跌破前低", len(break_low))
    
    def _analyze_trend_strength(self, data: pd.Data_frame, stats: Dict[str, int...
```

75. ```sql
, len(extreme_volume_shrink))
        
        # 连续放量
        consecutive_surge = 0
        for i in range(1, len(data)):
            if data.iloc[i]['volume'] > data.iloc[i]['volume_ma5'] * 1.5:
    ...
```

76. ```sql
.join(top_feature_texts)}。"
            
            return description
        else:
            return "无法生成买点描述，关键特征不足。" 

    def analyze_feature_correlation_Analyzer(self, features_data: List[Dic...
```

77. ```sql
# 计算ADX (Average Directional Index)
        try:
            import talib
            data['adx'] = talib.ADX(data['high'].values, data['low'].values, data['close'].values, timeperiod=14)
            ...
```

78. ```sql
, len(bullish_alignment))
        
        # 均线空头排列（MA5 < MA10 < MA20 < MA30）
        bearish_alignment = data[(data['ma5'] < data['ma10']) & (data['ma10'] < data['ma20']) & (data['ma20'] < data['ma30...
```

79. ```sql
] < 0.3]
        self._update_pattern_count(stats, "极度缩量(<0.3倍均量)", len(extreme_volume_shrink))
        
        # 连续放量
        consecutive_surge = 0
        for i in range(1, len(data)):
            ...
```

80. ```sql
, len(ma5_cross_below_ma10))
        
        # MA10上穿MA20
        ma10_cross_above_ma20 = data[(data['ma10'] > data['ma20']) & (data['ma10'].shift(1) <= data['ma20'].shift(1))]
        self._update_p...
```

81. ```sql
, len(big_up))
        
        # 大跌
        big_down = data[data['price_change_pct'] < -0.05]  # 单日跌幅超过5%
        self._update_pattern_count(stats,
```

82. ```sql
] - min_low) / min_low > 0.05 and i - min_idx > 3:
                low_rebound.append(i)
        self._update_pattern_count(stats, "低位反弹", len(low_rebound))
        
        # 突破前高
        break_high ...
```

83. ```sql
] > 5.0]
        self._update_pattern_count(stats, "极度放量(>5倍均量)", len(extreme_volume_surge))
        
        # 缩量（成交量小于5日均量的0.5倍）
        volume_shrink = data[data[
```

84. ```sql
]:
                consecutive_down += 1
                if consecutive_down >= 3:  # 连续3日下跌
                    self._update_pattern_count(stats, "连续3日下跌", 1)
                if consecutive_down >= 5...
```

85. ```sql
# 移动平均线
        kline_data['ma5'] = kline_data['close'].rolling(5).mean()
        kline_data['ma10'] = kline_data['close'].rolling(10).mean()
        kline_data['ma20'] = kline_data['close'].rolling(2...
```

86. ```sql
# 计算价格变动百分比
        data['price_change_pct'] = data['close'].pct_change()
        
        # 计算n日涨跌幅
        data['change_5d'] = data['close'] / data['close'].shift(5) - 1 if len(data) >= 5 else np.na...
```

87. ```sql
, len(hanging_man))
        
        # 长腿十字星
        long_legged_doji = kline_data[(kline_data['body'] < kline_data['body'].mean() * 0.3) &
                                    (kline_data['upper_shado...
```

88. ```sql
, len(shooting_star))
        
        # 穿刺线
        piercing = kline_data[(kline_data['close'] > kline_data['open']) & 
                            (kline_data['open'] < kline_data['prev_close']) &
 ...
```

89. ```sql
].shift(5)).abs() * 1.2)]
        self._update_pattern_count(stats, "均线发散", len(ma_diverge))
    
    def _analyze_price_trend(self, data: pd.Data_frame, stats: Dict[str, int]) -> None:
        """分析价...
```

90. ```sql
, len(extreme_volume_surge))
        
        # 缩量（成交量小于5日均量的0.5倍）
        volume_shrink = data[data['volume_ratio'] < 0.5]
        self._update_pattern_count(stats,
```

91. ```sql
, len(very_strong_trend))
            
            # 无趋势 (ADX < 20)
            no_trend = data[data['adx'] < 20]
            self._update_pattern_count(stats,
```

92. ```sql
] > 2.0]
        self._update_pattern_count(stats, "放量(>2倍均量)", len(volume_surge))
        
        # 大幅放量（成交量大于5日均量的3倍）
        large_volume_surge = data[data[
```

93. ```sql
) -> Dict[str, Any]:
        """
        分析指标信号
        
        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期，格式YYYYMMDD
            end_date: 结束日期，格式YYYYMMDD
            indicat...
```

94. ```sql
, ma_crosses)
        
        # 均线收敛
        ma_converge = data[((data['ma5'] - data['ma30']).abs() < (data['ma5'].shift(5) - data['ma30'].shift(5)).abs() * 0.8)]
        self._update_pattern_count(s...
```

95. ```sql
] < -0.05]  # 单日跌幅超过5%
        self._update_pattern_count(stats, "单日大跌(>5%)", len(big_down))
        
        # 高位回调
        high_pullback = []
        window = 10
        for i in range(window, len(d...
```

96. ```sql
, small_up_big_volume)
        
        # 小阴大量（单日下跌小于2%但放量超过2倍）
        small_down_big_volume = 0
        for i in range(1, len(data)):
            if (0 < data.iloc[i-1]['close'] / data.iloc[i]['clos...
```

97. ```sql
] > 50]
            self._update_pattern_count(stats, "极强趋势(ADX>50)", len(very_strong_trend))
            
            # 无趋势 (ADX < 20)
            no_trend = data[data[
```

98. ```sql
, big_down_big_volume)
        
        # 小阳大量（单日上涨小于2%但放量超过2倍）
        small_up_big_volume = 0
        for i in range(1, len(data)):
            if (0 < data.iloc[i]['close'] / data.iloc[i-1]['close'...
```

99. ```sql
] * 0.5)]
        self._update_pattern_count(stats, "射击之星", len(shooting_star))
        
        # 穿刺线
        piercing = kline_data[(kline_data[
```

100. ```sql
])]
        self._update_pattern_count(stats, "均线多头排列", len(bullish_alignment))
        
        # 均线空头排列（MA5 < MA10 < MA20 < MA30）
        bearish_alignment = data[(data[
```

101. ```sql
, len(volume_surge))
        
        # 大幅放量（成交量大于5日均量的3倍）
        large_volume_surge = data[data['volume_ratio'] > 3.0]
        self._update_pattern_count(stats,
```

102. ```sql
] < 20]
            self._update_pattern_count(stats, "无趋势(ADX<20)", len(no_trend))
            
        except ImportError:
            # 如果没有talib，使用简化的趋势判断
            logger.warning("未安装talib库，使用简...
```

103. ```sql
, len(ma5_cross_above_ma10))
        
        # MA5下穿MA10
        ma5_cross_below_ma10 = data[(data['ma5'] < data['ma10']) & (data['ma5'].shift(1) >= data['ma10'].shift(1))]
        self._update_patte...
```

104. ```sql
])]
        self._update_pattern_count(stats, "均线空头排列", len(bearish_alignment))
        
        # 均线交叉密集
        ma_crosses = 0
        for i in range(1, len(data)):
            crosses = 0
         ...
```

105. ```sql
# 计算K线实体和影线
        kline_data['body'] = abs(kline_data['close'] - kline_data['open'])
        kline_data['upper_shadow'] = kline_data['high'] - kline_data[['open', 'close']].max(axis=1)
        kline...
```

106. ```sql
SELECT date, open, high, low, close, volume, amount
                FROM stock_
```

107. ```sql
, len(morning_star))
        
        # 黄昏星
        evening_star = kline_data[(kline_data['prev2_close'] > kline_data['prev2_open']) &  # 第一天上涨
                                (abs(kline_data['prev_cl...
```

108. ```sql
# 量增价增（同向）
        volume_price_up = 0
        for i in range(1, len(data)):
            if (data.iloc[i]['volume'] > data.iloc[i-1]['volume'] * 1.2 and
                data.iloc[i]['close'] > data.il...
```

109. ```sql
, len(low_rebound))
        
        # 突破前高
        break_high = []
        for i in range(window, len(data)):
            prev_high = data.iloc[i-window:i-1]['high'].max()
            if data.iloc[i]...
```

110. ```sql
] * 1.5:
                consecutive_surge += 1
                if consecutive_surge >= 3:  # 连续3日放量
                    self._update_pattern_count(stats, "连续3日放量", 1)
            else:
              ...
```

111. ```sql
, len(three_black_crows))
        
        # 三白兵
        three_white_soldiers = []
        for i in range(3, len(kline_data)):
            if (kline_data.iloc[i-3]['close'] < kline_data.iloc[i-3]['ope...
```

112. ```sql
]):
                crosses += 1
            if crosses >= 2:
                ma_crosses += 1
        self._update_pattern_count(stats, "均线交叉密集", ma_crosses)
        
        # 均线收敛
        ma_converg...
```

113. ```sql
] > 0.05]  # 单日涨幅超过5%
        self._update_pattern_count(stats, "单日大涨(>5%)", len(big_up))
        
        # 大跌
        big_down = data[data[
```

114. ```sql
, len(ma10_cross_above_ma20))
        
        # MA10下穿MA20
        ma10_cross_below_ma20 = data[(data['ma10'] < data['ma20']) & (data['ma10'].shift(1) >= data['ma20'].shift(1))]
        self._update_...
```

115. ```sql
: len(selected_features),
```

116. ```sql
, len(bearish_engulfing))
        
        # 启明星
        morning_star = kline_data[(kline_data['prev2_close'] < kline_data['prev2_open']) &  # 第一天下跌
                                (abs(kline_data['pr...
```

117. ```sql
SELECT date, open, high, low, close, volume, amount
                FROM stock_ LIMIT 1000{period.lower()}
                WHERE stock_code = '{stock_code}'
                  AND date >= '{start_date_...
```

118. ```sql
, len(volume_shrink))
        
        # 极度缩量（成交量小于5日均量的0.3倍）
        extreme_volume_shrink = data[data['volume_ratio'] < 0.3]
        self._update_pattern_count(stats,
```

119. ```sql
])]
        self._update_pattern_count(stats, "看跌吞没", len(bearish_engulfing))
        
        # 启明星
        morning_star = kline_data[(kline_data[
```

120. ```sql
]):
                three_white_soldiers.append(i)
        self._update_pattern_count(stats, "三白兵", len(three_white_soldiers))
    
    def _analyze_complex_patterns(self, kline_data: pd.Data_frame, s...
```

121. ```sql
: sum(f.get(target_metric, 0) for f in features)
                }
            
            # 根据目标指标对特征排序
            sorted_features = sorted(features, 
                                  key=lambda x...
```

122. ```sql
].shift(1))]
        self._update_pattern_count(stats, "MA5下穿MA10", len(ma5_cross_below_ma10))
        
        # MA10上穿MA20
        ma10_cross_above_ma20 = data[(data[
```

123. ```sql
)
            
            # 构建特征共现矩阵
            all_features = set()
            for features in feature_occurrences.values():
                all_features.update(features)
            
            ...
```

124. ```sql
, len(hammer))
        
        # 吊颈线
        hanging_man = kline_data[(kline_data['lower_shadow'] > kline_data['body'] * 2) & 
                               (kline_data['upper_shadow'] < kline_data[...
```

125. ```sql
])]
        self._update_pattern_count(stats, "长腿十字星", len(long_legged_doji))
        
        # 射击之星
        shooting_star = kline_data[(kline_data[
```

126. ```sql
, len(large_volume_surge))
        
        # 极度放量（成交量大于5日均量的5倍）
        extreme_volume_surge = data[data['volume_ratio'] > 5.0]
        self._update_pattern_count(stats,
```

127. ```sql
] * 0.8:
                consecutive_shrink += 1
                if consecutive_shrink >= 3:  # 连续3日缩量
                    self._update_pattern_count(stats, "连续3日缩量", 1)
            else:
            ...
```

128. ```sql
] * 1.5):
                price_lead_volume += 1
        self._update_pattern_count(stats, "先价后量", price_lead_volume)
    
    def _analyze_price_volume_relation(self, data: pd.Data_frame, stats: Dict...
```

129. ```sql
, len(evening_star))
        
        # 三只乌鸦
        three_black_crows = []
        for i in range(3, len(kline_data)):
            if (kline_data.iloc[i-3]['close'] > kline_data.iloc[i-3]['open'] and...
```

130. ```sql
] * 2.0):
                small_down_big_volume += 1
        self._update_pattern_count(stats, "小阴大量", small_down_big_volume)
    
    def _calculate_volume_frequency(self, volume_stats: Dict[str, Dic...
```

131. ```sql
, len(ma10_cross_below_ma20))
        
        # 均线多头排列（MA5 > MA10 > MA20 > MA30）
        bullish_alignment = data[(data['ma5'] > data['ma10']) & (data['ma10'] > data['ma20']) & (data['ma20'] > data['...
```

### ./analysis/buypoints/buypoint_strategy_adapter.py
- 查询数量: 1

1. ```sql
)
                return None
            
            # 转换评分
            score = self._convert_score(buypoint_result)
            
            # 提取技术指标匹配详情
            match_details = self._extract_m...
```

### ./analysis/buypoints/period_data_processor.py
- 查询数量: 1

1. ```sql
import pandas as pd
from typing import Dict, List, Any, Optional
import os
import sys
from datetime import datetime

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.pa...
```

### ./analysis/engines/complex_logic_processor.py
- 查询数量: 2

1. ```sql
] += 1
        
        try:
            # 更新自定义变量
            if variables:
                self.variables.update(variables)
            
            # 生成缓存键
            cache_key = self._generate_ca...
```

2. ```sql
import time
        start_time = time.time()
        self.stats['expressions_evaluated'] += 1
        
        try:
            # 更新自定义变量
            if variables:
                self.variables.updat...
```

### ./analysis/engines/date_manager.py
- 查询数量: 9

1. ```sql
try:
            current_stats = self.statistics.copy()
            
            # 计算缓存命中率
            total_requests = current_stats['cache_hits'] + current_stats['cache_misses']
            cache_hi...
```

2. ```sql
] = self.cache_ttl
        return stats
    
    def clear_cache_Manager(self):
        """清除所有缓存"""
        self.date_cache.clear()
        self.trading_calendar = None
        self.last_cache_update...
```

3. ```sql
# 检查缓存
        if self.trading_calendar and self.last_cache_update:
            if (datetime.datetime.now() - self.last_cache_update).seconds < self.cache_ttl:
                cached_date = self.date_...
```

4. ```sql
# 使用依赖注入架构
        self.container = get_container()
        self.data_access = self.get_service(Data_access_interface)
        
        # 原有的初始化代码保持不变
        self.date_cache = {}
        self.trading...
```

5. ```sql
self.date_cache.clear()
        self.trading_calendar = None
        self.last_cache_update = None
        logger.info(
```

6. ```sql
, DateFormat.ISO_FORMAT),  # 2024-01-01T12:00:00
        ]
        
        for pattern, format_type in patterns:
            if re.match(pattern, date_str):
                try:
                    r...
```

7. ```sql
try:
            start_time = time.time()
            
            # 使用依赖注入的数据访问接口
            calendar_data = self.data_access.get_trading_calendar()
            
            query_time = time.time()...
```

8. ```sql
try:
            self.statistics = {
                'cache_hits': 0,
                'cache_misses': 0,
                'db_queries': 0,
                'calendar_updates': 0,
                'error_...
```

9. ```sql
].iloc[0])
                formatted_date = self.format_date(latest_date, format_type)
                
                # 更新缓存
                self.trading_calendar = calendar_data
                sel...
```

### ./analysis/engines/indicator_validation_framework.py
- 查询数量: 40

1. ```sql
total_selected_stocks
```

2. ```sql
] = ValidationResult.NO_SELECTION.value
            elif result[
```

3. ```sql
* 50)
    
    # 配置验证参数
    config = Indicator_validation_config(
        mode=Validation_mode.QUICK,
        stock_pool_size=500,
        max_selection_ratio=0.05,
        parallel_workers=2,
       ...
```

4. ```sql
,
                    'timestamp': datetime.now().isoformat(),
                    'validation_date': self.config.validation_date,
                    'stock_pool_size': 0,
                    'select...
```

5. ```sql
start_time = time.time()
        
        result = {
            'indicator_name': indicator_name,
            'status': ValidationResult.ERROR.value,
            'validation_date': self.config.valida...
```

6. ```sql
] = ValidationResult.OVER_SELECTION.value
            elif len(selected_stocks) >= self.config.min_selection_count:
                result[
```

7. ```sql
] > self.config.max_selection_ratio:
                result[
```

8. ```sql
)
                        if result.get('selected_count', 0) > 0:
                            f.write(f
```

9. ```sql
] = selected_stocks[:50]  # 只保存前50只股票
            
            # 3. 判断验证结果
            if len(selected_stocks) == 0:
                result[
```

10. ```sql
][status] += 1
            
            if status == Validation_result.SUCCESS.value:
                selection_ratios.append(result.get(
```

11. ```sql
SELECT DISTINCT code 
            FROM stock_info
```

12. ```sql
SELECT DISTINCT code 
                FROM stock_info
```

13. ```sql
][ValidationResult.OVER_SELECTION.value]}\n")
                    f.write(f"  验证错误: {summary[
```

14. ```sql
}
            ]
        
        return conditions
    
    def _execute_strategy_selection(self, strategy_config: Dict[str, Any], stock_pool: List[str]) -> List[str]:
        """执行策略选股"""
        try...
```

15. ```sql
: {
                Validation_result.SUCCESS.value: 0,
                Validation_result.NO_SELECTION.value: 0,
                Validation_result.OVER_SELECTION.value: 0,
                Validation_r...
```

16. ```sql
SELECT DISTINCT code 
                FROM stock_info WHERE 1=1
                WHERE level = '日线' 
                AND volume > 0 AND close > 0
                ORDER BY date DESC, volume DESC
       ...
```

17. ```sql
average_selection_ratio
```

18. ```sql
] = len(selected_stocks) / len(stock_pool) if stock_pool else 0
            result[
```

19. ```sql
过度选择: {summary['validation_results'][ValidationResult.OVER_SELECTION.value]}\n
```

20. ```sql
return result
            
            # 2. 执行策略选股
            selected_stocks = self._execute_strategy_selection(strategy_config, stock_pool)
            
            result['selected_count'] = len(s...
```

21. ```sql
)
            # 返回最基本的条件
            conditions = [
                {
                    'type': 'basic',
                    'field': 'close',
                    'operator': '>',
                  ...
```

22. ```sql
] = "策略生成失败"
                return result
            
            # 2. 执行策略选股
            selected_stocks = self._execute_strategy_selection(strategy_config, stock_pool)
            
            res...
```

23. ```sql
✅ 指标 {indicator} 验证成功: 选出 {result['selected_count']} 只股票
```

24. ```sql
, 0))
                selected_counts.append(result.get(
```

25. ```sql
未选出股票: {summary['validation_results'][ValidationResult.NO_SELECTION.value]}\n
```

26. ```sql
]):
                    logger.error(f"❌ 数据库连接失败: {db_error}")
                    # 抛出异常以便上层捕获并设置为ERROR状态
                    raise ConnectionError(f"数据库连接失败: {db_error}")
                else:
     ...
```

27. ```sql
选股数量: {result['selected_count']}, 选股比例: {result['selection_ratio']:.4f}
```

28. ```sql
summary = {
            'total_indicators': len(results),
            'validation_results': {
                Validation_result.SUCCESS.value: 0,
                Validation_result.NO_SELECTION.value: ...
```

29. ```sql
] = len(selected_stocks)
            result[
```

30. ```sql
import os
import sys
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from concurrent.futures impo...
```

31. ```sql
SELECT DISTINCT code 
            FROM stock_info WHERE 1=1
            WHERE level = '日线' AND date = '{self.config.validation_date}'
            AND volume > 0 
            AND close > 0
            ...
```

32. ```sql
, 0)
        }
        
        # 统计各状态数量
        selection_ratios = []
        selected_counts = []
        
        for result in results:
            status = result.get(
```

33. ```sql
mode: validation_mode = Validation_mode.FULL
    stock_pool_size: int = 1000
    max_selection_ratio: float = 0.1  # 最大选股比例
    min_selection_count: int = 1      # 最小选股数量
    validation_date: str = No...
```

34. ```sql
# 成功：能够选出股票且符合预期
    NO_SELECTION =
```

35. ```sql
] = ValidationResult.NO_SELECTION.value
            
            result[
```

36. ```sql
# 未选出：策略未选出任何股票
    OVER_SELECTION =
```

37. ```sql
]:
            result.extend(sorted(categories[category]))
        
        return result
    
    def _prepare_stock_pool(self) -> List[str]:
        """准备股票池（带缓存，避免重复查询）"""
        try:
            ...
```

38. ```sql
][ValidationResult.NO_SELECTION.value]}\n")
                    f.write(f"  过度选择: {summary[
```

39. ```sql
{i}. {indicator['indicator']}: 选出{indicator['selected_count']}只股票
```

40. ```sql
选中股票: {result['selected_stocks'][:10]}
```

### ./analysis/engines/unified_indicator_engine.py
- 查询数量: 5

1. ```sql
:
                    boll_result = self.calculate_bollinger_bands(data)
                    results.update(boll_result)
                
                elif indicator_upper ==
```

2. ```sql
:
                    kdj_result = self.calculate_kdj(data)
                    results.update(kdj_result)
                
                elif indicator_upper ==
```

3. ```sql
:
                    wvad_result = self.calculate_wvad(data)
                    results.update(wvad_result)
                
                else:
                    logger.warning(f"不支持的指标: {indic...
```

4. ```sql
if not self._validate_data(data):
            return {}
        
        # 默认计算所有指标
        if indicators is None:
            indicators = ['MA', 'EMA', 'MACD', 'KDJ', 'RSI', 'BOLL', 'WVAD']
        ...
```

5. ```sql
:
                    macd_result = self.calculate_macd(data)
                    results.update(macd_result)
                
                elif indicator_upper ==
```

### ./analysis/indicator_performance_profiler.py
- 查询数量: 1

1. ```sql
import os
import sys
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import json
from datetime import datetime

# 添加项目根目录到Python路径
root_dir = os.path.dirna...
```

### ./analysis/integration/unified_analysis_engine.py
- 查询数量: 17

1. ```sql
)
            return self._create_empty_result()
    
    @timing_decorator
    @safe_run
    def analyze_market_selection(self, 
                               strategy_name: str,
                   ...
```

2. ```sql
)
                return self._create_empty_result()
            
            # 执行选股
            selected_stocks = strategy.select_stocks(
                max_count=selection_count,
                da...
```

3. ```sql
)
                return self._create_empty_result()
            
            # 提取股票代码
            stock_codes = [stock['stock_code'] for stock in selected_stocks]
            
            # 执行综合分析
  ...
```

4. ```sql
enable_strategy_selection
```

5. ```sql
市场选股分析
        
        Args:
            strategy_name: 策略名称
            selection_count: 选股数量
            analysis_date: 分析日期
            
        Returns:
            Dict[str, Any]: 分析结果
```

6. ```sql
try:
            results = {}
            
            # 执行买点分析
            if self.analysis_config['enable_buypoint_analysis']:
                results['buypoint'] = self._analyze_buypoints_only(stoc...
```

7. ```sql
] for stock in selected_stocks]
            
            # 执行综合分析
            result = self.analyze_stocks(stock_codes, analysis_date, "comprehensive")
            
            # 添加策略信息
            if...
```

8. ```sql
self.data_adapter = get_unified_data_adapter()
        self.buypoint_analyzer = Buy_point_batch_analyzer()
        self.strategy_factory = Strategy_factory()
        self.max_workers = max_workers
   ...
```

9. ```sql
import sys
import os
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import pandas as pd
import asyncio
from concurrent.futures import Thread_pool_executor, as_comple...
```

10. ```sql
开始市场选股分析，策略: {strategy_name}，目标数量: {selection_count}
```

11. ```sql
self.analysis_config.update(config)
        logger.info(f
```

12. ```sql
return self.performance_stats.copy()
    
    def update_config(self, config: Dict[str, Any]) -> None:
```

13. ```sql
)
            if default_strategy:
                # 对每只股票执行策略评估
                for stock_code in stock_codes:
                    try:
                        # 这里应该调用策略的evaluate_stock方法
           ...
```

14. ```sql
try:
            futures = []
            
            with Thread_pool_executor(max_workers=self.max_workers) as executor:
                # 提交买点分析任务
                if self.analysis_config['enable_b...
```

15. ```sql
)
            
            # 添加策略信息
            if result['success']:
                result['strategy_info'] = {
                    'strategy_name': strategy_name,
                    'selection_cou...
```

16. ```sql
)
            
            logger.info(f"开始市场选股分析，策略: {strategy_name}，目标数量: {selection_count}")
            
            # 获取策略实例
            strategy = self.strategy_factory.create_strategy(strategy_...
```

17. ```sql
]:
                self._save_analysis_results(result, output_dir)
            
            return result
            
        except Exception as e:
            logger.error(f"从CSV文件分析时出错: {e}")
    ...
```

### ./analysis/integration/unified_data_adapter.py
- 查询数量: 8

1. ```sql
)
                return None
            
            # 转换指标结果
            match_details = self._convert_indicator_results(buypoint_result.get('indicator_results', {}))
            
            # 计算综...
```

2. ```sql
SELECT 
                name,
                industry,
                close as price,
                change_pct,
                market_cap,
                pe_ratio,
                pb_ratio,
    ...
```

3. ```sql
SELECT 
                name,
                industry,
                close as price,
                change_pct,
                market_cap,
                pe_ratio,
                pb_ratio,
    ...
```

4. ```sql
, 0), reverse=True)
            
            logger.info(f"合并分析结果完成: {len(final_results)} 只股票")
            return final_results
            
        except Exception as e:
            logger.error(f"...
```

5. ```sql
import sys
import os
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import pandas as pd
import threading

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os....
```

6. ```sql
self.data_manager = get_unified_data_manager()
        
        # 定义标准数据格式规范
        self.standard_format = {
            'required_fields': [
                'stock_code',      # 股票代码
               ...
```

7. ```sql
)
                return None
            
            # 策略选股结果通常已经接近标准格式，主要是补全缺失字段
            standard_data = strategy_result.copy()
            
            # 确保必需字段存在
            required_fields =...
```

8. ```sql
)
                        return None
            
            # 添加来源标识
            standard_data['source_system'] = 'strategy_selection'
            
            # 验证数据格式
            if self._validat...
```

### ./analysis/intelligent_cache_system.py
- 查询数量: 1

1. ```sql
import os
import sys
import time
import hashlib
import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import json
from datetime import datetim...
```

### ./analysis/market/a_stock_market_analysis.py
- 查询数量: 5

1. ```sql
)
            return {}
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=15.0)
    def analyze_stock_selection_opportunities(self, date: Optional[str] = None) -> Di...
```

2. ```sql
] = f"行业表现分化，{top_industry[0]}领涨"
            
            return summary
            
        except Exception as e:
            logger.error(f"生成市场总结失败: {e}")
            return {}
    
    @excepti...
```

3. ```sql
try:
        # 初始化分析器
        analyzer = AStock_market_analyzer()
        
        # 获取市场概况
        market_overview = analyzer.get_market_overview()
        
        # 分析选股机会
        opportunities = a...
```

4. ```sql
) as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
                
            logger.info(f"分析结果已保存到: {output_file}")
            
        except Exception as e...
```

5. ```sql
import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
import logging

# 添加项目根目录到路径
root_d...
```

### ./analysis/multi_dimension_analyzer.py
- 查询数量: 15

1. ```sql
SELECT code as stock_code, name as stock_name, industry
            FROM stock_info
```

2. ```sql
].pct_change()
                    index_data[code] = df.dropna()
            
            # 获取股票数据
            stock_returns = {}
            for stock_code in stock_codes:
                sql = f"""...
```

3. ```sql
]
            index_data = {}
            
            # 获取各指数数据
            for code in index_codes:
                sql = f"""
                SELECT date, close
                FROM index_daily
   ...
```

4. ```sql
SELECT industry, COUNT(*) as count
            FROM stock_info
```

5. ```sql
SELECT date, close
                FROM index_daily
                WHERE code = '{code}' AND date >= '{start_date}' AND date <= '{end_date}'
                ORDER BY date
```

6. ```sql
SELECT DISTINCT code as stock_code
            FROM stock_info
```

7. ```sql
SELECT date, close
                FROM stock_daily
                WHERE stock_code = '{stock_code}' AND date >= '{start_date}' AND date <= '{end_date}'
                ORDER BY date
```

8. ```sql
)"
                
            sql = f"""
            SELECT DISTINCT code as stock_code
            FROM stock_info WHERE 1=1
            WHERE industry =
```

9. ```sql
".join(stock_codes)
            sql = f"""
            SELECT industry, COUNT(*) as count
            FROM stock_info WHERE 1=1
            WHERE stock_code IN (
```

10. ```sql
SELECT industry, COUNT(*) as count
            FROM stock_info WHERE 1=1
            WHERE stock_code IN ('{stock_codes_str}')
            GROUP BY industry
            ORDER BY count DESC
```

11. ```sql
),
                "industry": industry,
                "date": date,
                "periods": periods,
                "multi_period_analysis": multi_period_result,
                "indicator_anal...
```

12. ```sql
SELECT code as stock_code, name as stock_name, industry
            FROM stock_info WHERE 1=1
            WHERE code = '{stock_code}'
            ORDER BY date DESC
            LIMIT 1
```

13. ```sql
SELECT date, close
                FROM stock_daily
```

14. ```sql
SELECT DISTINCT code as stock_code
            FROM stock_info WHERE 1=1
            WHERE industry = '{industry}'{exclude_condition}
            LIMIT 50
```

15. ```sql
SELECT date, close
                FROM index_daily
```

### ./analysis/optimized_batch_analyzer.py
- 查询数量: 1

1. ```sql
import os
import sys
import time
import multiprocessing
import concurrent.futures
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as ...
```

### ./analysis/optimized_buypoint_analyzer.py
- 查询数量: 3

1. ```sql
: self.performance_stats.copy()
            }
            
        except Exception as e:
            logger.error(f"买点分析失败 {stock_code} {buypoint_date}: {e}")
            return None
    
    def get...
```

2. ```sql
import os
import sys
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import json
from datetime import datetime

# 添加项目根目录到Python路径
root_dir = os....
```

3. ```sql
stats = self.performance_stats.copy()
        
        if self.enable_cache:
            cache_stats = self.cache_system.get_cache_stats()
            stats.update(cache_stats)
        
        # 计算优化...
```

### ./analysis/parallel_buypoint_analyzer.py
- 查询数量: 1

1. ```sql
import os
import sys
import time
import multiprocessing
import concurrent.futures
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as ...
```

### ./analysis/performance_analyzer.py
- 查询数量: 1

1. ```sql
import os
import sys
import time
import psutil
import threading
import multiprocessing
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
import nump...
```

### ./analysis/simple_performance_analyzer.py
- 查询数量: 1

1. ```sql
import os
import sys
import time
import gc
import threading
import multiprocessing
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as...
```

### ./analysis/strategy_comparison.py
- 查询数量: 20

1. ```sql
SELECT stock_code, market_cap, pb_ratio, pe_ratio
            FROM stock_info
```

2. ```sql
].tolist(), end_date)
                    
                # 准备策略结果
                strategy_result = {
                    "strategy_id": strategy_id,
                    "strategy_name": strategy_co...
```

3. ```sql
SELECT industry, COUNT(*) as count
            FROM stock_info
```

4. ```sql
SELECT industry, COUNT(*) as count
                FROM stock_info WHERE 1=1
                WHERE stock_code IN ('{stock_codes_str}')
                GROUP BY industry
                ORDER BY count ...
```

5. ```sql
])
                distribution[industry] = count
            
            # 计算行业分布百分比
            total = sum(distribution.values())
            percentages = {k: v / total for k, v in distribution.i...
```

6. ```sql
SELECT DISTINCT stock_code FROM stock_info WHERE date >= '2020-01-01'
```

7. ```sql
)
                    selected = result.get(
```

8. ```sql
| {s_id} | {selected} | {ratio:.2%} | {weighted_return:.2%} | {weighted_positive:.2%} |\n
```

9. ```sql
SELECT DISTINCT stock_code FROM stock_info
```

10. ```sql
, 0):.4f} |\n"
                
                markdown += "\n"
        
        # 添加时间周期表现
        period_results = comparison_result.get("period_results", [])
        if period_results:
           ...
```

11. ```sql
SELECT industry, COUNT(*) as count
            FROM stock_info WHERE 1=1
            WHERE stock_code IN ({placeholders_str})
            GROUP BY industry
            ORDER BY count DESC
```

12. ```sql
".join(safe_codes)
                sql = f"""
                SELECT stock_code, market_cap, pb_ratio, pe_ratio
                FROM stock_info WHERE 1=1
                WHERE stock_code IN (
```

13. ```sql
".join(safe_codes)
                sql = f"""
                SELECT industry, COUNT(*) as count
                FROM stock_info WHERE 1=1
                WHERE stock_code IN (
```

14. ```sql
SELECT stock_code, market_cap, pb_ratio, pe_ratio
            FROM stock_info WHERE 1=1
            WHERE stock_code IN ({placeholders_str})
```

15. ```sql
].tolist())
                    strategy_result["style_features"] = style_features
                
                period_results.append(strategy_result)
                
            # 计算策略之间的重叠度
   ...
```

16. ```sql
] < max_cap)])
                    market_cap_distribution[name] = count
            
            # 计算市值分布百分比
            total = sum(market_cap_distribution.values())
            if total > 0:
      ...
```

17. ```sql
].tolist() if isinstance(result, pd.DataFrame) else []
            }
            
            strategy_results.append(strategy_result)
        
        # 计算组合选股结果
        combined_stocks = self._calcu...
```

18. ```sql
SELECT stock_code, market_cap, pb_ratio, pe_ratio
                FROM stock_info
```

19. ```sql
SELECT industry, COUNT(*) as count
                FROM stock_info
```

20. ```sql
SELECT stock_code, market_cap, pb_ratio, pe_ratio
                FROM stock_info WHERE 1=1
                WHERE stock_code IN ('{stock_codes_str}')
```

### ./analysis/strategy_validator.py
- 查询数量: 25

1. ```sql
选股比例 {period_result['selection_ratio']:.2%}
```

2. ```sql
]
                
                if start_price > 0:
                    return_rate = (end_price - start_price) / start_price
                    returns.append(return_rate)
            
          ...
```

3. ```sql
total_selected_stocks
```

4. ```sql
: result.to_dict(orient='records') if isinstance(result, pd.DataFrame) else []
                }
                
                # 计算选出股票的后续表现
                if isinstance(result, pd.Data_frame) and...
```

5. ```sql
选股比例 {strategy_result['selection_ratio']:.2%}
```

6. ```sql
选股比例 {combo_result['selection_ratio']:.2%}
```

7. ```sql
].tolist(), end_date)
                    period_result.update(performance)
                
                period_results.append(period_result)
                logger.info(f"周期 {start_date} - {end_d...
```

8. ```sql
)
            
            # 如果没有指定股票池，获取默认股票池
            if stock_pool is None:
                # 获取全市场股票
                stock_pool = self.db_manager.get_all_stock_codes()
                
        ...
```

9. ```sql
selection_sensitivity
```

10. ```sql
# 提取关键指标
        selection_ratios = [r[
```

11. ```sql
selection_ratio_stability
```

12. ```sql
].tolist(), end_date)
                    strategy_result.update(performance)
                
                strategy_results.append(strategy_result)
                logger.info(f"策略 {strategy_id} 选...
```

13. ```sql
: len(group)
                }
            
            # 计算参数敏感度
            if len(value_stats) > 1:
                selection_variance = np.var([s[
```

14. ```sql
selection_ratio_trend
```

15. ```sql
参数组合 {param_mapping} 选出 {combo_result['selected_stocks']} 只股票，
```

16. ```sql
, 0) for p in period_results]
        
        # 计算汇总统计
        summary = {
            "avg_selection_ratio": float(np.mean(selection_ratios)),
            "selection_ratio_stability": float(np.std(s...
```

17. ```sql
][param_name]
                    
                    if param_value not in param_groups:
                        param_groups[param_value] = []
                        
                    param_gro...
```

18. ```sql
: float(np.mean(selection_ratios)),
```

19. ```sql
if not period_results:
            return {}
            
        # 提取统计数据
        selection_ratios = [p['selection_ratio'] for p in period_results]
        avg_returns = [p.get('avg_return', 0) for p...
```

20. ```sql
周期 {start_date} - {end_date} 选出 {period_result['selected_stocks']} 只股票，
```

21. ```sql
, 0) for r in period_results]
        
        # 计算趋势
        selection_trend = self._calculate_trend(selection_ratios)
        return_trend = self._calculate_trend(avg_returns)
        positive_trend...
```

22. ```sql
: result['code'].tolist() if isinstance(result, pd.DataFrame) else []
                }
                
                # 更新所有选出的股票集合
                if isinstance(result, pd.Data_frame):
           ...
```

23. ```sql
].tolist(), end_date)
                    combo_result.update(performance)
                
                results.append(combo_result)
                logger.info(f"参数组合 {param_mapping} 选出 {combo_re...
```

24. ```sql
].tolist() if isinstance(result, pd.DataFrame) else []
                }
                
                # 更新所有选出的股票集合
                if isinstance(result, pd.Data_frame):
                    all_se...
```

25. ```sql
: float(np.std(selection_ratios)),
```

### ./analysis/vectorized_indicator_optimizer.py
- 查询数量: 1

1. ```sql
import os
import sys
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import json
from datetime import datetime

# 添加项目根目录到Python路径
root_dir = os....
```

### ./analyze_centralized_mapping_status.py
- 查询数量: 1

1. ```sql
def __init__(self):
        self.registry = Pattern_registry()
        self._initialize_indicators()
        
        # 已知已实现register_patterns()方法的指标
        self.indicators_with_patterns = {
        ...
```

### ./bin/advanced_community_crawler.py
- 查询数量: 2

1. ```sql
for attempt in range(retries):
            try:
                # 更新请求头
                headers = self.get_advanced_headers_Crawler()
                self.session.headers.update(headers)

            ...
```

2. ```sql
] = referer

        return headers

    def safe_request_Crawler(self, url: str, retries: int = 3) -> Dict[str, Any]:
        """安全请求页面"""
        for attempt in range(retries):
            try:
    ...
```

### ./bin/advanced_real_crawler.py
- 查询数量: 2

1. ```sql
for attempt in range(retries):
            try:
                # 更新请求头
                headers = self.get_advanced_headers()
                self.session.headers.update(headers)

                prin...
```

2. ```sql
] = referer

        return headers

    def safe_request_with_retry(self, url: str, retries: int = 3) -> Dict[str, Any]:
        """带重试的安全请求"""
        for attempt in range(retries):
            try:...
```

### ./bin/backtest_integrated.py
- 查询数量: 8

1. ```sql
],
                reverse=True
            )
            
            # 取最常见的模式
            if sorted_patterns:
                pattern_key, pattern_info = sorted_patterns[0]
                selected...
```

2. ```sql
),
                "update_time": datetime.now().strftime(
```

3. ```sql
]
        
        # 如果仍然没有指标，使用默认的ZXM指标
        if not selected_indicators:
            selected_indicators = [
                Indicator_enum.ZXM_TURNOVER,
                Indicator_enum.ZXM_DAILY_M...
```

4. ```sql
] > 50
        ]
        
        # 如果没有合适的指标，尝试使用最常见的模式
        if not selected_indicators and pattern_stats:
            # 按出现次数排序
            sorted_patterns = sorted(
                pattern_stats...
```

5. ```sql
], 
            reverse=True
        )
        
        # 选择成功率超过50%的指标
        selected_indicators = [
            indicator_id for indicator_id, stats in sorted_indicators
            if stats[
```

6. ```sql
) as f:
                f.write(report)
            logger.info(f"回测报告已保存到: {output_file}")
        else:
            print(report)
        
        return {
            "indicator_stats": indicator_s...
```

7. ```sql
)
        
        # 创建策略条件
        conditions = []
        for idx, indicator_id in enumerate(selected_indicators):
            # 获取指标默认参数
            params = {
                IndicatorEnum.ZXM_TUR...
```

8. ```sql
})
            
            # 除了最后一个，每个指标后面都添加OR
            if idx < len(selected_indicators) - 1:
                conditions.append({
```

### ./bin/backtest_strategy_integrate.py
- 查询数量: 5

1. ```sql
选出股票数: {validation_result.get('selected_stocks', 0)}\n
```

2. ```sql
选股比例: {optimized_validation.get('selection_ratio', 0):.2%}\n
```

3. ```sql
选股比例: {validation_result.get('selection_ratio', 0):.2%}\n
```

4. ```sql
)
            orig_ratio = validation_result.get('selection_ratio', 0)
            opt_ratio = optimized_validation.get('selection_ratio', 0)
            ratio_change = (opt_ratio - orig_ratio) / orig...
```

5. ```sql
选出股票数: {optimized_validation.get('selected_stocks', 0)}\n
```

### ./bin/buypoint_batch_analyzer.py
- 查询数量: 1

1. ```sql
import os
import sys
import argparse
from datetime import datetime

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from utils.l...
```

### ./bin/compare_strategies.py
- 查询数量: 5

1. ```sql
) if w.strip()]
        if not weights:
            return None
            
        # 检查权重是否非负
        if any(w < 0 for w in weights):
            raise argparse.ArgumentTypeError("策略权重不能为负")
       ...
```

2. ```sql
比较维度列表，用逗号分隔，可选: selection_ratio,return_performance,stability,risk,style,industry,overlap
```

3. ```sql
, 0)
        selection_ratio = result.get(
```

4. ```sql
)
    
    # 添加子命令
    subparsers = parser.add_subparsers(dest='command', help='子命令')
    
    # 多维度比较子命令
    multi_dim_parser = subparsers.add_parser('multi_dimension', help='多维度策略比较')
    multi_dim_...
```

5. ```sql
组合选出 {combined_stocks_count} 只股票，选股比例 {selection_ratio:.2%}
```

### ./bin/generate_technical_indicators.py
- 查询数量: 15

1. ```sql
+ 
                    updated_content[class_end:]
                )
                
                with open(enum_file,
```

2. ```sql
:
            existing.append(file[:-3])  # 去掉.py后缀
    
    return existing


def update_enum_file(new_indicators):
    """更新技术指标枚举文件"""
    enum_dir = os.path.join(root_dir,
```

3. ```sql
))
                if class_end == -1:
                    class_end = len(updated_content)
                
                # 插入新条目
                updated_content = (
                    updated_con...
```

4. ```sql
]
                    )
                    new_entries.append(template.generate_enum_entry())
            
            if new_entries:
                # 在枚举类的最后添加新条目
                updated_content =...
```

5. ```sql
, last_elif_pos)
            if line_end != -1:
                # 在此位置插入新条目
                updated_method = (
                    method_content[:line_end+1] +
```

6. ```sql
, updated_content.find(
```

7. ```sql
)
        return
    
    method_content = create_method_match.group(0)
    
    # 检查哪些指标尚未添加到工厂方法中
    existing_indicators = re.findall(r'IndicatorType_Generate_Technical_Indicators_Generate_Technica...
```

8. ```sql
)
    else:
        # 已存在枚举文件，追加新指标
        enum_class_match = re.search(r'class IndicatorType_Generate_Technical_Indicators_Generate_Technical_Indicators.*?:(.*?)(?=\n\n|\Z)', existing_content, re.DO...
```

9. ```sql
)
        return
    
    # 更新枚举文件
    update_enum_file(new_indicators)
    
    # 更新工厂文件
    update_factory_file(new_indicators)
    
    # 生成指标模块
    generate_indicator_modules(new_indicators)
    
...
```

10. ```sql
.join(new_entries) + 
                    method_content[line_end:]
                )
                
                # 更新整个文件内容
                updated_content = content.replace(method_content, upda...
```

11. ```sql
) as f:
                    f.write(updated_content)
                
                logger.info(f"已更新工厂文件: {factory_file}")
                return
    
    logger.info("工厂文件中已包含所有指标，无需更新")


def gen...
```

12. ```sql
indicators_dir = get_indicators_dir()
    existing = []
    
    for file in os.listdir(indicators_dir):
        if file.endswith('.py') and not file.startswith('__') and file != 'base_indicator.py' a...
```

13. ```sql
, 
                    1
                )
                
                # 找到类定义的结束位置
                class_end = updated_content.find(
```

14. ```sql
)


def update_factory_file(new_indicators):
```

15. ```sql
) as f:
                    f.write(updated_content)
                
                logger.info(f"已更新枚举文件: {enum_file}")
            else:
                logger.info("枚举文件中已包含所有指标，无需更新")


def upda...
```

### ./bin/get_real_data_enhanced.py
- 查询数量: 1

1. ```sql
def __init___7(self):
        self.concept_extractor = Concept_stock_extractor()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 ...
```

### ./bin/indicator_scoring.py
- 查询数量: 1

1. ```sql
import os
import sys
import argparse
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
import logging

# 添加项目根目录到路径
root_dir = os.pa...
```

### ./bin/optimized_real_crawler.py
- 查询数量: 2

1. ```sql
try:
            headers = self.get_optimized_headers()
            self.session.headers.update(headers)

            print(f
```

2. ```sql
}

    def safe_request(self, url: str) -> Dict[str, Any]:
        """安全请求"""
        try:
            headers = self.get_optimized_headers()
            self.session.headers.update(headers)

        ...
```

### ./bin/performance_test.py
- 查询数量: 1

1. ```sql
import os
import sys
import argparse
import time
import json
from datetime import datetime

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,...
```

### ./bin/quick_performance_test.py
- 查询数量: 1

1. ```sql
import os
import sys
import time
import json
from datetime import datetime

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from...
```

### ./bin/real_crawler.py
- 查询数量: 1

1. ```sql
def __init___4(self):
        self.concept_extractor = Concept_stock_extractor()

        # 设置请求会话
        self.session = requests.Session()
        self.session.headers.update({
            'User-Age...
```

### ./bin/real_crawler_extended.py
- 查询数量: 2

1. ```sql
try:
            # 更新请求头
            self.session.headers.update(self.get_random_headers())

            print(f
```

2. ```sql
}

    def safe_request_Extended(self, url: str, site_config: Dict[str, Any]) -> Dict[str, Any]:
        """安全请求页面内容"""
        try:
            # 更新请求头
            self.session.headers.update(self.ge...
```

### ./bin/real_selenium_crawler.py
- 查询数量: 1

1. ```sql
def __init___6(self):
        self.concept_extractor = Concept_stock_extractor()
        self.driver = None

        # 备用请求会话
        self.session = requests.Session()
        self.session.headers.upd...
```

### ./bin/run_advanced_backtest.py
- 查询数量: 1

1. ```sql
import os
import sys
import argparse
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
import logging

# 添加项目根目录到路径
root_dir ...
```

### ./bin/run_enhanced_backtest.py
- 查询数量: 1

1. ```sql
import sys
import os

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

import argparse
import json
from typing import Dict, Any, ...
```

### ./bin/run_optimized_backtest.py
- 查询数量: 1

1. ```sql
import os
import sys

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

import argparse
import json
import datetime
from typing im...
```

### ./bin/run_production_crawler.py
- 查询数量: 2

1. ```sql
)

                        start_time = time.time()
                        response = spider.get_page(url)

                        if response:
                            article = spider.parse_art...
```

2. ```sql
)
                                )

                                # 合并提取结果
                                article.update(extraction_result)
                                article[
```

### ./bin/run_system_review.py
- 查询数量: 1

1. ```sql
import sys
import os
import subprocess
import time
from datetime import datetime

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)...
```

### ./bin/run_unified_engine_test.py
- 查询数量: 3

1. ```sql
({stats['success_rate']}%) 平均选股: {stats['avg_selected_count']}\n
```

2. ```sql
)
        
        # 临时修改配置为快速测试模式
        original_config = self.config['test_configuration'].copy()
        self.config['test_configuration'].update({
            'test_stock_count': 20,
           ...
```

3. ```sql
import sys
import os
import argparse
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path...
```

### ./bin/simple_performance_test.py
- 查询数量: 1

1. ```sql
import os
import sys
import argparse
import time
import json
from datetime import datetime

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,...
```

### ./bin/stock_analysis.py
- 查询数量: 3

1. ```sql
) as f:
                f.write(report_content)
                
            logger.info(f"分析报告已保存到 {output_file}")
            return output_file
            
        except Exception as e:
         ...
```

2. ```sql
SELECT code, name FROM stock_info WHERE industry = '{industry}'
```

3. ```sql
SELECT code, name FROM stock_info
```

### ./bin/stock_select.py
- 查询数量: 6

1. ```sql
stock_selection_{date_str}.{args.format}
```

2. ```sql
import os
import sys
import argparse
import json
import yaml
import pandas as pd
from datetime import datetime
import time
import threading
from typing import Dict, List, Optional, Any

# 添加项目根目录到路径
r...
```

3. ```sql
# 解析命令行参数
    args = parse_args_Select()
    
    # 设置日志级别
    init_logging(level=args.log_level)
    
    # 加载配置文件
    if args.config:
        config = load_config(args.config)
        # 使用配置文件中的值覆盖命...
```

4. ```sql
)
    #     distribution_img = plot_selection_result_distribution(
    #         result_df=result_df,
    #         title=f
```

5. ```sql
, default=None)
    
    args = parser.parse_args_Select()
    return args


def print_progress(progress: float, message: str):
```

6. ```sql
)
        
        # 限制结果数量
        if args.limit > 0 and len(result_df) > args.limit:
            result_df = result_df.iloc[:args.limit]
        
        # 输出结果
        output_result(result_df, args...
```

### ./bin/stock_watch.py
- 查询数量: 1

1. ```sql
)
        return
    
    # 如果需要保存结果
    if args.save:
        data_manager = get_unified_data_manager()
        success = data_manager.save_selection_result(result)
        if success:
            lo...
```

### ./bin/strategy_generator.py
- 查询数量: 2

1. ```sql
INSERT INTO {table_name} (code, buy_date, pattern_type) VALUES
```

2. ```sql
)
            data.append((code, buy_date, pattern_type))
        
        client.execute(f"INSERT INTO {table_name} (code, buy_date, pattern_type) VALUES", data)
        
        logger.info(f"买点数据已保...
```

### ./bin/update_factory.py
- 查询数量: 19

1. ```sql
.join(new_registrations)
    updated_content = content.replace(registration_section, updated_registrations)
    
    with open(factory_file, 'w', encoding='utf-8') as f:
        f.write(updated_conten...
```

2. ```sql
, content)
    existing_modules = {module: class_name for module, class_name in existing_imports}
    
    new_imports = []
    for class_name, module_name in indicator_modules.items():
        if mod...
```

3. ```sql
)
        return False
    
    # 找到最后一个elif语句
    last_elif_pos = method_content.rfind('elif')
    if last_elif_pos != -1:
        # 找到该elif语句所在行的结束位置
        line_end = method_content.find('\n', las...
```

4. ```sql
) as f:
        f.write(updated_content)
    
    logger.info(f"已添加 {len(new_registrations)} 个指标注册语句到工厂文件")
    return True


def update_create_indicator_method(factory_file: str, indicator_modules: D...
```

5. ```sql
)
        return
    
    # 更新导入语句
    import_updated = update_factory_imports(factory_file, indicator_modules)
    
    # 更新注册语句
    reg_updated = update_factory_registrations(factory_file, indicator...
```

6. ```sql
, last_elif_pos)
        if line_end != -1:
            # 在此位置插入新条目
            updated_method = (
                method_content[:line_end+1] +
```

7. ```sql
updated_content = content.replace(import_section, updated_imports)
    
    with open(factory_file, 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    logger.info(f
```

8. ```sql
def create_indicator_Update_Factory.*?:.*?return.*?$
```

9. ```sql
)
    return True


def update_factory_registrations(factory_file: str, indicator_modules: Dict[str, str]) -> bool:
```

10. ```sql
with open(factory_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 找到create_indicator方法
    create_method_match = re.search(r'def create_indicator_Update_Factory.*?:.*?return.*...
```

11. ```sql
):
            module_names.append(file[:-3])  # 去掉.py后缀
    
    return module_names


def get_indicator_class_names() -> Dict[str, str]:
    """
    获取所有技术指标类名和对应的模块名
    
    Returns:
        Dict[...
```

12. ```sql
)
    
    if not new_registrations:
        logger.info("工厂文件已包含所有指标注册语句，无需更新")
        return False
    
    # 在注册部分末尾添加新的注册语句
    updated_registrations = registration_section + "\n" + "\n".join(new...
```

13. ```sql
) as f:
                f.write(updated_content)
            
            logger.info(f"已添加 {len(new_entries)} 个指标到create_indicator方法")
            return True
    
    logger.info("create_indicator方法...
```

14. ```sql
)
        return False
    
    # 在导入部分末尾添加新的导入语句
    updated_imports = import_section.rstrip() +
```

15. ```sql
result = {}
    
    # 从IndicatorType枚举中获取所有指标类名
    for indicator in Indicator_type:
        if isinstance(indicator.value, str):  # 跳过auto()生成的枚举值
            indicator_name = indicator.name
       ...
```

16. ```sql
)
    return True


def update_create_indicator_method(factory_file: str, indicator_modules: Dict[str, str]) -> bool:
```

17. ```sql
)
        return False
    
    # 在注册部分末尾添加新的注册语句
    updated_registrations = registration_section +
```

18. ```sql
) as f:
        f.write(updated_content)
    
    logger.info(f"已添加 {len(new_imports)} 个指标导入语句到工厂文件")
    return True


def update_factory_registrations(factory_file: str, indicator_modules: Dict[str,...
```

19. ```sql
.join(new_entries) + 
                method_content[line_end:]
            )
            
            # 更新整个文件内容
            updated_content = content.replace(method_content, updated_method)
        ...
```

### ./bin/validate_buypoint_strategy.py
- 查询数量: 9

1. ```sql
original_codes = set(buypoints_group['stock_code'].unique())
        
        try:
            # 执行策略
            selected_stocks = self.strategy_executor.execute_strategy(
                strategy_co...
```

2. ```sql
].unique())
        
        try:
            # 执行策略
            selected_stocks = self.strategy_executor.execute_strategy(
                strategy_config, 
                stock_pool=list(original_c...
```

3. ```sql
return {
            'overall_match_rate': overall_match_rate,
            'total_original_stocks': total_original,
            'total_selected_stocks': total_selected,
            'total_matched_stoc...
```

4. ```sql
total_selected_stocks
```

5. ```sql
)
            return {
                'date': buy_date,
                'original_count': len(original_codes),
                'selected_count': 0,
                'matched_count': 0,
               ...
```

6. ```sql
策略选出数量: {match_analysis['total_selected_stocks']}
```

7. ```sql
] for r in detailed_results)
        total_selected = sum(r[
```

8. ```sql
total_original = sum(r['original_count'] for r in detailed_results)
        total_selected = sum(r['selected_count'] for r in detailed_results)
        total_matched = sum(r['matched_count'] for r in ...
```

9. ```sql
].unique())
            else:
                selected_codes = set()
            
            # 计算匹配结果
            matched_codes = original_codes & selected_codes
            missed_codes = original_c...
```

### ./bin/validate_indicators.py
- 查询数量: 13

1. ```sql
) as f:
        config = json.load(f)
    
    framework = IndicatorValidationFramework()
    framework.config.update(config)
    
    # 获取类别指标
    category_indicators = config.get("test_categories", ...
```

2. ```sql
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from scripts.indicator_validation_frame...
```

3. ```sql
✅ 测试通过 - 选股率: {selection_rate:.1%}
```

4. ```sql
) as f:
        config = json.load(f)
    
    framework = IndicatorValidationFramework()
    framework.config.update(config)
    
    priority_indicators = config.get("priority_indicators", [])
    
...
```

5. ```sql
⚠️ 过度选股 - 选股率: {selection_rate:.1%}
```

6. ```sql
)}")
    
    # 生成摘要
    successful = sum(1 for r in results if r.get("analysis", {}).get("validation_status") == "success")
    print(f"\n📋 {category} 类别验证完成: {successful}/{len(category_indicators)} ...
```

7. ```sql
)
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    framework = IndicatorValidationFramework()
    framework.config.update(config)
    
    priority_indica...
```

8. ```sql
)
            selection_rate = analysis.get(
```

9. ```sql
)
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    framework = IndicatorValidationFramework()
    framework.config.update(config)
    
    # 获取类别指标
    ca...
```

10. ```sql
✅ 验证成功 - 选股率: {selection_rate:.1%}
```

11. ```sql
) as f:
        config = json.load(f)
    
    framework = IndicatorValidationFramework()
    framework.config.update(config)
    
    # 执行完整验证
    output_dir = f"results/full_indicator_validation_{da...
```

12. ```sql
选股率: {analysis.get('selection_rate', 0):.1%}
```

13. ```sql
)
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    framework = IndicatorValidationFramework()
    framework.config.update(config)
    
    # 执行完整验证
    ou...
```

### ./bin/validate_indicators_closed_loop.py
- 查询数量: 5

1. ```sql
📊 验证摘要:
   总指标数量: {summary.get('total_indicators', 0)}
   成功验证: {summary.get('successful_validations', 0)}
   失败验证: {summary.get('failed_validations', 0)}
   有选股结果: {summary.get('indicators_with_selec...
```

2. ```sql
import os
import sys
import argparse
import json
from datetime import datetime

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from s...
```

3. ```sql
indicators_with_selections
```

4. ```sql
status_emoji = {
        'success': '✅',
        'no_selection': '⚠️',
        'over_selection': '📈',
        'failed': '❌',
        'strategy_generation_failed': '🔧'
    }
    
    status = result.ge...
```

5. ```sql
{emoji} 指标: {indicator_name}
   状态: {status}
   选股数量: {result.get('selection_count', 0)}
   选股比例: {result.get('selection_ratio', 0):.2%}
   质量评分: {result.get('quality_score', 0):.2f}
   闭环验证: {'通过' if...
```

### ./bin/validate_strategy.py
- 查询数量: 15

1. ```sql
in result:
            # 提取数据
            dates = []
            selection_ratios = []
            returns = []
            
            for period in result[
```

2. ```sql
])
                    
                    # 绘制参数敏感性图
                    plt.figure(figsize=(10, 6))
                    plt.plot(values, selection_ratios, marker=
```

3. ```sql
)
                        plt.close()
        
        # 策略对比结果可视化
        if 'strategy_results' in result:
            # 提取数据
            names = []
            selection_ratios = []
            retu...
```

4. ```sql
].items():
                        values.append(value)
                        selection_ratios.append(stats[
```

5. ```sql
in result:
            # 提取数据
            names = []
            selection_ratios = []
            returns = []
            
            for strategy in result[
```

6. ```sql
])
            
            # 绘制选股比例对比图
            plt.figure(figsize=(12, 6))
            bars = plt.bar(names, selection_ratios)
            plt.title(
```

7. ```sql
])
                selection_ratios.append(strategy[
```

8. ```sql
)
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(f"{output_prefix}_param_{param}_selection.png")
                    plt.close()
            ...
```

9. ```sql
)
                plt.close()
        
        # 参数敏感性分析结果可视化
        if 'sensitivity' in result and 'parameter_sensitivity' in result['sensitivity']:
            for param, data in result['sensitivit...
```

10. ```sql
try:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_prefix)
        os.makedirs(output_dir, exist_ok=True)
        
        # 多周期验证结果可视化
        if 'periods' in result:
            # 提...
```

11. ```sql
, rotation=0)
            
            plt.tight_layout()
            plt.savefig(f"{output_prefix}_strategy_comparison_selection.png")
            plt.close()
            
            # 如果有收益率数据，绘制收益...
```

12. ```sql
)
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{output_prefix}_selection_ratio.png")
            plt.close()
            
   ...
```

13. ```sql
])
            
            # 绘制选股比例图
            plt.figure(figsize=(10, 6))
            plt.plot(dates, selection_ratios, marker=
```

14. ```sql
])
                selection_ratios.append(period[
```

15. ```sql
in data:
                    # 提取数据
                    values = []
                    selection_ratios = []
                    returns = []
                    
                    for value, stats...
```

### ./bin/verify_complete_strategy.py
- 查询数量: 14

1. ```sql
in evaluation_result:
                                    selected = evaluation_result[
```

2. ```sql
)
                            
                            # 评估股票是否满足策略条件
                            try:
                                evaluation_result = evaluator.evaluate_conditions(strategy['c...
```

3. ```sql
)
                            selected = False
                            verification_results.append({
                                'date': date,
                                'stock_code': sto...
```

4. ```sql
)
                                logger.error(traceback.format_exc())
                                selected = False
                                detail_reason = f
```

5. ```sql
]
                
                # 确保股票代码是字符串
                if isinstance(stock_code, (int, float)):
                    stock_code = str(int(stock_code)).zfill(6)
                
               ...
```

6. ```sql
.join(failing_indicators)}")
                                    
                                    # 准备保存到验证结果中的详情
                                    detail_reason = "满足策略条件" if selected else "不满足...
```

7. ```sql
: detail_reason
                            })
                except Exception as e:
                    logger.error(f"验证股票 {stock_code} 时出错: {e}")
                    logger.error(traceback.format_...
```

8. ```sql
股票 {stock_code} 是否满足策略条件: {selected}
```

9. ```sql
, {})}")
                                            # 单独评估这个条件
                                            try:
                                                single_result = evaluator._evaluate_ind...
```

10. ```sql
)
                    logger.error(traceback.format_exc())
                    selected = False
                    verification_results.append({
                        'date': date,
                ...
```

11. ```sql
.join(failing_indicators)}"
                                else:
                                    # 兼容旧的返回方式
                                    selected = bool(evaluation_result)
                ...
```

12. ```sql
else:
                                    # 兼容旧的返回方式
                                    selected = bool(evaluation_result)
                                    detail_reason =
```

13. ```sql
]
                        missing_columns = [col for col in required_columns if col not in k_data.columns]
                        
                        if missing_columns:
                        ...
```

14. ```sql
)
                        selected = False
                        verification_results.append({
                            'date': date,
                            'stock_code': stock_code,
       ...
```

### ./bin/verify_strategy.py
- 查询数量: 14

1. ```sql
in evaluation_result:
                                    selected = evaluation_result[
```

2. ```sql
)
                            
                            # 评估股票是否满足策略条件
                            try:
                                evaluation_result = evaluator.evaluate_conditions(strategy['c...
```

3. ```sql
)
                            selected = False
                            verification_results.append({
                                'date': date,
                                'stock_code': sto...
```

4. ```sql
)
                                logger.error(traceback.format_exc())
                                selected = False
                                detail_reason = f
```

5. ```sql
]
                
                # 确保股票代码是字符串
                if isinstance(stock_code, (int, float)):
                    stock_code = str(int(stock_code)).zfill(6)
                
               ...
```

6. ```sql
.join(failing_indicators)}")
                                    
                                    # 准备保存到验证结果中的详情
                                    detail_reason = "满足策略条件" if selected else "不满足...
```

7. ```sql
: detail_reason
                            })
                except Exception as e:
                    logger.error(f"验证股票 {stock_code} 时出错: {e}")
                    logger.error(traceback.format_...
```

8. ```sql
股票 {stock_code} 是否满足策略条件: {selected}
```

9. ```sql
, {})}")
                                            # 单独评估这个条件
                                            try:
                                                single_result = evaluator._evaluate_ind...
```

10. ```sql
)
                    logger.error(traceback.format_exc())
                    selected = False
                    verification_results.append({
                        'date': date,
                ...
```

11. ```sql
.join(failing_indicators)}"
                                else:
                                    # 兼容旧的返回方式
                                    selected = bool(evaluation_result)
                ...
```

12. ```sql
else:
                                    # 兼容旧的返回方式
                                    selected = bool(evaluation_result)
                                    detail_reason =
```

13. ```sql
]
                        missing_columns = [col for col in required_columns if col not in k_data.columns]
                        
                        if missing_columns:
                        ...
```

14. ```sql
)
                        selected = False
                        verification_results.append({
                            'date': date,
                            'stock_code': stock_code,
       ...
```

### ./bin/zxm_analysis.py
- 查询数量: 3

1. ```sql
)
            
            if stock_data.empty:
                logger.warning(f"未找到股票 {code} 的数据")
                return pd.DataFrame()
                
            logger.info(f"成功加载 {len(stock_dat...
```

2. ```sql
)
            
            results = {}
            
            # 计算ZXM买点指标
            zxm_buypoint = ZXMBuyPointIndicator()
            buypoint_result = zxm_buypoint.calculate(data)
            
 ...
```

3. ```sql
] = buypoint_result
            
            # 计算ZXM评分指标
            zxm_score = ZXMScoreIndicator()
            score_result = zxm_score.calculate(data)
            
            if isinstance(score_r...
```

### ./config/config.py
- 查询数量: 5

1. ```sql
) as f:
                user_config = json.load(f)
                
                # 递归合并配置
                deep_update(CONFIG, user_config)
                
            # 验证合并后的配置
            errors...
```

2. ```sql
)
    
    return errors

def deep_update(target: Dict[str, Any], source: Dict[str, Any]) -> None:
```

3. ```sql
for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            deep_update(target[key], value)
        else:
            target[k...
```

4. ```sql
global CONFIG
    if os.path.exists(USER_CONFIG_PATH):
        try:
            with open(USER_CONFIG_PATH, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                
   ...
```

5. ```sql
")
    
    return errors

def deep_update(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    """
    递归合并字典
    
    Args:
        target: 目标字典
        source: 源字典
    """
    for key, valu...
```

### ./config/database_config_manager.py
- 查询数量: 2

1. ```sql
# 1. 从默认配置开始
        config = self.DEFAULT_CONFIG.copy()
        
        # 2. 加载.env文件
        self._load_env_file()
        
        # 3. 加载YAML配置文件
        if self.config_file.exists():
           ...
```

2. ```sql
in yaml_config:
                        config.update(yaml_config[
```

### ./config/unified_config.py
- 查询数量: 3

1. ```sql
)
                    return
            
            # 更新配置对象
            self._update_config_from_dict(self._config_data)
            logger.info(f
```

2. ```sql
)):
                    self._config_data = yaml.safe_load(f)
                else:
                    logger.warning(f"不支持的配置文件格式: {self._config_file}")
                    return
            
     ...
```

3. ```sql
)
    
    def _update_config_from_dict(self, config_dict: Dict[str, Any]):
```

### ./crawler/anti_crawler.py
- 查询数量: 2

1. ```sql
session = requests.Session()

        # 设置User-Agent
        session.headers.update({
            'User-Agent': self.ua_rotator.get_random_ua(),
            'Accept': 'text/html,application/xhtml+xml,...
```

2. ```sql
, 6379)
        
        self.proxy_pool = Proxy_pool(redis_host, redis_port)
        self.ua_rotator = User_agent_rotator()
        self.captcha_solver = Captcha_solver()
        if REDIS_AVAILABLE:
...
```

### ./crawler/diagnosis/anti_crawler_analyzer.py
- 查询数量: 1

1. ```sql
def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like...
```

### ./crawler/integration/deployment_manager.py
- 查询数量: 2

1. ```sql
FROM python:3.9-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install -...
```

2. ```sql
dockerfiles = {
            'crawler': '''
FROM python:3.9-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY require...
```

### ./crawler/integration/system_integrator.py
- 查询数量: 5

1. ```sql
try:
            # 数据格式转换
            formatted_data = self._format_article_data(article_data)

            # 实际部署时执行插入
            # self.client.insert('crawler_articles', [formatted_data])

        ...
```

2. ```sql
)

            # 插入到ClickHouse
            if self.clickhouse.insert_article(article_data):
                self.integration_status['total_articles_synced'] += 1
                self.integration_statu...
```

3. ```sql
)
            return False

    def insert_article(self, article_data: Dict[str, Any]) -> bool:
```

4. ```sql
)
        self.client = None

    def connect(self) -> bool:
        """连接到ClickHouse数据库"""
        try:
            # 这里使用简化的连接方式，实际部署时需要安装clickhouse-driver
            logger.info(f"连接到ClickHouse: {...
```

5. ```sql
]}")

            # 插入到ClickHouse
            if self.clickhouse.insert_article(article_data):
                self.integration_status[
```

### ./crawler/monitoring/demo.py
- 查询数量: 1

1. ```sql
)

    # 创建性能监控器
    monitor = Performance_monitor(update_interval=5)

    # 启动监控
    monitor.start()

    # 模拟一些请求
    print(
```

### ./crawler/monitoring/performance_monitor.py
- 查询数量: 7

1. ```sql
: dict(list(self.error_counts.items())[:5])
        }


class Performance_monitor:
    """性能监控器"""

    def __init__(self, update_interval=60):
        self.metrics = Performance_metrics()
        sel...
```

2. ```sql
: 90.0  # 成功率低于90%
        }

        # 告警回调
        self.alert_callbacks = []

    def start_Monitor(self):
        """启动监控"""
        if self.running:
            return

        self.running = True...
```

3. ```sql
self.total_requests += 1

        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
            if error_type:
                self.error_counts...
```

4. ```sql
while self.running:
            try:
                # 更新系统指标
                self.metrics.update_system_metrics()

                # 检查告警条件
                self._check_alerts()

                # 等待下...
```

5. ```sql
def __init__(self, update_interval=60):
        self.metrics = Performance_metrics()
        self.update_interval = update_interval
        self.running = False
        self.monitor_thread = None

   ...
```

6. ```sql
] += 1

    def update_system_metrics(self):
        """更新系统资源指标"""
        try:
            # 简化版本，避免依赖psutil
            import os

            # CPU使用率（简化版）
            try:
                with op...
```

7. ```sql
)
                time.sleep(self.update_interval)

    def _check_alerts(self):
```

### ./crawler/scheduler.py
- 查询数量: 4

1. ```sql
] += 1
        self._update_stats()

    def _update_stats(self):
        """更新统计信息"""
        total = self.stats[
```

2. ```sql
] += 1
        self.response_times.append(response_time)
        self._update_stats()

    def record_task_failed(self, task: Crawler_task):
        """记录任务失败"""
        self.stats[
```

3. ```sql
self.stats['failed_tasks'] += 1
        self._update_stats()

    def _update_stats(self):
```

4. ```sql
self.stats['completed_tasks'] += 1
        self.response_times.append(response_time)
        self._update_stats()

    def record_task_failed(self, task: Crawler_task):
```

### ./crawler/spiders/base_spider.py
- 查询数量: 2

1. ```sql
,
        }

    def init_session(self):
        """初始化会话"""
        if self.anti_crawler:
            self.session = self.anti_crawler.get_session()
        else:
            self.session = requests....
```

2. ```sql
if self.anti_crawler:
            self.session = self.anti_crawler.get_session()
        else:
            self.session = requests.Session()
            self.session.headers.update(self.headers)

    ...
```

### ./crawler/spiders/taoguba_spider.py
- 查询数量: 21

1. ```sql
,  # 股吧页面
            ]

            for base_url in base_urls:
                try:
                    response = self.get_page(base_url)
                    if response and response.status_code == ...
```

2. ```sql
,
            anti_crawler_module=anti_crawler_module
        )

        # 淘股吧特定的请求头
        self.headers.update({
            'Referer': 'https://www.taoguba.com.cn/',
            'X-Requested-With':...
```

3. ```sql
]',
                            'a[title]',  # 有标题的链接
                            '.title a',  # 标题类的链接
                            '.post-title a',  # 帖子标题链接
                        ]

              ...
```

4. ```sql
# 尝试多种选择器来提取内容
            content_selectors = [
                '.article-content', '.topic-content', '.content',
                '.post-content', '.thread-content', '.main-content',
                ...
```

5. ```sql
]):
                    script.decompose()

                # 获取页面文本
                page_text = soup.get_text()
                # 简单清理
                lines = [line.strip() for line in page_text.spli...
```

6. ```sql
]'
            ]

            author = None
            for selector in author_selectors:
                author_elem = soup.select_one(selector)
                if author_elem:
                    au...
```

7. ```sql
]}"

            # 尝试多种选择器来提取内容
            content_selectors = [
```

8. ```sql
)

        for item in article_items:
            try:
                article = {}

                # 标题和链接
                title_elem = item.select_one(
```

9. ```sql
] = content

            # 尝试提取作者信息
            author_selectors = [
```

10. ```sql
]

            title = None
            for selector in title_selectors:
                title_elem = soup.select_one(selector)
                if title_elem:
                    title = self.extract_...
```

11. ```sql
,  # 帖子标题链接
                        ]

                        for selector in selectors:
                            links = soup.select(selector)
                            for link in links:
     ...
```

12. ```sql
# 尝试多种选择器来提取标题
            title_selectors = [
                '.article-title', '.topic-title', 'h1', 'h2',
                '.title', '.post-title', '.thread-title',
                '[class*=
```

13. ```sql
))

                # 作者
                author_elem = item.select_one(
```

14. ```sql
]

            content = None
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    # 移除广告和无关元素
 ...
```

15. ```sql
,
        })

    def get_article_urls(self, page: int = 1) -> List[str]:
        """获取文章URL列表"""
        urls = []

        try:
            # 使用淘股吧的实际URL结构
            # 这里使用一个更通用的方法来获取文章链接
        ...
```

16. ```sql
]

        for pattern in exclude_patterns:
            if pattern in url.lower():
                return False

        return True

    def parse_article_list(self, response) -> List[Dict[str, Any]]...
```

17. ```sql
] = f"taoguba_{hashlib.md5(response.url.encode()).hexdigest()[:8]}"

            # 尝试多种选择器来提取标题
            title_selectors = [
```

18. ```sql
articles = []
        soup = self.parse_html(response.text)

        # 解析文章列表项
        article_items = soup.select('.article-item, .topic-item')

        for item in article_items:
            try:
  ...
```

19. ```sql
]', 'title'
            ]

            title = None
            for selector in title_selectors:
                title_elem = soup.select_one(selector)
                if title_elem:
                 ...
```

20. ```sql
]', 'article', '.article-body'
            ]

            content = None
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if cont...
```

21. ```sql
]

            author = None
            for selector in author_selectors:
                author_elem = soup.select_one(selector)
                if author_elem:
                    author = self.ext...
```

### ./db/batch_data_optimizer.py
- 查询数量: 13

1. ```sql
)
            
            # 批量查询缺失数据
            batch_results = self._batch_query_stocks(missing_codes, start_date, end_date, 
                                                   level, optimized_con...
```

2. ```sql
) -> Dict[str, pd.DataFrame]:
        """
        批量获取股票数据
        
        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            level: K线级别
      ...
```

3. ```sql
SELECT code, name, date, open, high, low, close, volume, turnover_rate,
               price_change, price_range
        FROM stock_info
```

4. ```sql
)
        
        for col in df.select_dtypes(include=[
```

5. ```sql
".join(stock_codes)
            query = f"""
            SELECT code, name, date, open, high, low, close, volume, turnover_rate,
                   price_change, price_range
            FROM stock_inf...
```

6. ```sql
ORDER BY date ASC
        """
        
        return self.data_access.execute_query(query, {})
    
    def _optimize_dataframe_memory(self, df: pd.Data_frame) -> pd.Data_frame:
        """
        优...
```

7. ```sql
SELECT code, name, date, open, high, low, close, volume, turnover_rate,
                   price_change, price_range
            FROM stock_info WHERE 1=1
            WHERE code IN ('{codes_str}')
   ...
```

8. ```sql
SELECT code, name, date, open, high, low, close, volume, turnover_rate,
                   price_change, price_range
            FROM stock_info
```

9. ```sql
] == code].copy()
                if not stock_data.empty:
                    # 重置索引并优化内存使用
                    stock_data = stock_data.reset_index(drop=True)
                    stock_data = self._o...
```

10. ```sql
if df.empty:
            return df
        
        # 优化数值类型
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
 ...
```

11. ```sql
SELECT code, name, date, open, high, low, close, volume, turnover_rate,
               price_change, price_range
        FROM stock_info WHERE 1=1
        WHERE code = '{code}'
        AND level = '{l...
```

12. ```sql
)
        
        # 优化字符串类型
        for col in df.select_dtypes(include=[
```

13. ```sql
)
        
        with Thread_pool_executor(max_workers=config.max_workers) as executor:
            # 提交批次任务
            future_to_batch = {
                executor.submit(self._query_batch, batch,...
```

### ./db/cache_layer.py
- 查询数量: 2

1. ```sql
hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    
    @property
    def hit_rate_Layer(self) -> float:
```

2. ```sql
pass
    
    @abstractmethod
    def delete_Layer_Cache_Layer_Cache_Layer_1_cachelayer(self, key: str) -> bool:
```

### ./db/clickhouse_db.py
- 查询数量: 56

1. ```sql
SELECT 
                date, code, name, open, high, low, close, volume, turnover
            FROM stock_info WHERE 1=1
            WHERE 
                (code = %(symbol)s OR industry = %(symbol)s)...
```

2. ```sql
) and not field_name.isdigit():
                            group_by_fields.add(field_name)

            # 处理多股票查询的情况 - 确保查询中始终保留原始的code字段
            if isinstance(stock_code, list) and len(
        ...
```

3. ```sql
SELECT
            AVG(close) as avg_price
        FROM stock_info
```

4. ```sql
SELECT MIN(date) as min_date
            FROM stock_info WHERE 1=1
            WHERE {" AND ".join(conditions)}
```

5. ```sql
SELECT 
            MAX(date) as max_date
        FROM stock_info
```

6. ```sql
] = industry

            # 构建查询
            query = f"""
            SELECT DISTINCT code, name, industry
            FROM stock_info WHERE 1=1
            WHERE {
```

7. ```sql
]
                })

            logger.info(f"返回 {len(result)} 只示例股票代替指数 {index_code} 的成分股")
            return result

        except Exception as e:
            logger.error(f"获取指数 {index_code} 成分...
```

8. ```sql
save_selection_result方法调用了不存在的表stock_selection_result，当前数据库只有stock_info表
```

9. ```sql
] = db_level

            # 构建查询
            query = "SELECT MAX(date) as max_date FROM stock_info WHERE date >=
```

10. ```sql
.join(clean_query.split())
            
            # 查找SELECT和FROM之间的内容
            select_match = re.search(r
```

11. ```sql
获取指定日期的选股结果（注意：当前数据库中没有stock_selection_result表）

        Args:
            strategy_id: 策略ID
            selection_date: 选股日期

        Returns:
            pd.Data_frame: 选股结果
```

12. ```sql
SELECT DISTINCT code
            FROM stock_info
```

13. ```sql
SELECT {field_str} FROM stock_info WHERE date >= '2020-01-01'
```

14. ```sql
] = level_value

        # 构建插入字段和值
        fields = ", ".join(data.columns)

        # 批量插入（ClickHouse使用ReplacingMergeTree引擎，会自动处理重复数据）
        with self.manager.get_connection_Db(self.config) as con...
```

15. ```sql
SELECT DISTINCT code, name, industry
            FROM stock_info WHERE 1=1
            WHERE {' AND '.join(conditions)}
            ORDER BY code
```

16. ```sql
}

            result.rename(columns=column_map, inplace=True)

            return result

        except Exception as e:
            logger.error(f"从数据库加载行业 {symbol} 数据失败: {e}")
            return pd...
```

17. ```sql
,
                      **kwargs) -> Stock_info:
        """
        获取K线数据（兼容性方法）

        Args:
            stock_code: 股票代码或股票代码列表
            start_date: 开始日期
            end_date: 结束日期
          ...
```

18. ```sql
{result_dir}/selection_{strategy_id}_{selection_date}.csv
```

19. ```sql
获取选股历史记录（注意：当前数据库中没有stock_selection_result表）

        Args:
            strategy_id: 策略ID，None表示所有策略
            start_date: 开始日期，None表示不限制
            end_date: 结束日期，None表示当前日期
            limit: 返回记...
```

20. ```sql
)
                    continue

            # 转换为DataFrame并排序
            result = pd.Data_frame(history_data)
            if not result.empty:
                result = result.sort_values('selection_d...
```

21. ```sql
"""

        result = self.query_Db_Clickhouse_Db_Clickhouse_Db(query)
        if result.empty or pd.isna(result.iloc[0, 0]):
            return datetime.datetime.now()
        return result.iloc[0, 0...
```

22. ```sql
# 构建插入字段和值
        fields = ", ".join(data.columns)

        # 批量插入（ClickHouse使用ReplacingMergeTree引擎，会自动处理重复数据）
        with self.manager.get_connection_Db(self.config) as conn:
            for _, row...
```

23. ```sql
).date()
            except ValueError:
                logger.warning(f"无法将结束日期 {end_date} 转换为日期格式，使用原始字符串")

        try:
            # 构建查询 - 查询两种情况：行业代码匹配或者industry字段匹配
            query = """
   ...
```

24. ```sql
SELECT DISTINCT industry as name, '' as code
            FROM stock_info
```

25. ```sql
SELECT MIN(date) as min_date
            FROM stock_info
```

26. ```sql
INSERT INTO stock_info WHERE 1=1 ({fields})
                VALUES ({values})
```

27. ```sql
)
            return pd.DataFrame(columns=['name', 'code'])

    def save_selection_result_Db(self, result: pd.Data_frame, strategy_id: str,
                              selection_date: Optional[str]...
```

28. ```sql
SELECT 1')
                            logger.debug(f"连接健康检查通过: {key}")
                            continue  # 连接正常，不清理
                        except Exception:
                            logger.de...
```

29. ```sql
SELECT DISTINCT code, name, industry
            FROM stock_info
```

30. ```sql
INSERT INTO stock_info
```

31. ```sql
SELECT 
                date, code, name, open, high, low, close, volume, turnover
            FROM stock_info
```

32. ```sql
SELECT MAX(date) as max_date FROM stock_info WHERE date >= '2020-01-01'
```

33. ```sql
get_selection_history方法调用了不存在的表stock_selection_result，当前数据库只有stock_info表
```

34. ```sql
SELECT MAX(date) as max_date FROM stock_info
```

35. ```sql
] = db_level

            # 构建查询
            query = f"""
            SELECT MIN(date) as min_date
            FROM stock_info WHERE 1=1
            WHERE {" AND ".join(conditions)}
            """

 ...
```

36. ```sql
SELECT DISTINCT code, name, industry
            FROM stock_info WHERE 1=1
            WHERE level = '日线' AND industry = %(industry)s
            ORDER BY code
```

37. ```sql
)
            return False

    def get_selection_history(self, strategy_id: Optional[str] = None,
                              start_date: Optional[str] = None,
                              end_dat...
```

38. ```sql
)
            return pd.DataFrame(columns=['strategy_id', 'selection_date', 'stock_count'])

    def get_selection_result(self, strategy_id: str, selection_date: str) -> pd.Data_frame:
```

39. ```sql
SELECT
            AVG(close) as avg_price
        FROM stock_info WHERE 1=1
        WHERE
            code = %(code)s AND
            date >= %(start_date)s
```

40. ```sql
, clean_query, re.IGNORECASE | re.DOTALL)
            if not select_match:
                return []
            
            select_part = select_match.group(1).strip()
            
            # 如果是...
```

41. ```sql
SELECT code, name FROM stock_info WHERE code = %(stock_code)s AND level = '日线' LIMIT 1
```

42. ```sql
try:
            import re
            # 移除注释和多余空格
            clean_query = re.sub(r'/\*.*?\*/', '', query, flags=re.DOTALL)
            clean_query = re.sub(r'--.*', '', clean_query)
            cle...
```

43. ```sql
with self._lock:
            current_time = time.time()
            keys_to_remove = []

            for key, conn_info in self._connections.items():
                if conn_info['in_use']:
          ...
```

44. ```sql
{result_dir}/selection_*.csv
```

45. ```sql
保存选股结果（注意：当前数据库中没有stock_selection_result表）

        Args:
            result: 选股结果Data_frame
            strategy_id: 策略ID
            selection_date: 选股日期，默认为当前日期

        Returns:
            bool: ...
```

46. ```sql
])

            # 查找匹配的CSV文件
            pattern = f"{result_dir}/selection_*.csv"
            files = glob.glob(pattern)

            history_data = []
            for file_path in files:
           ...
```

47. ```sql
SELECT DISTINCT code
            FROM stock_info WHERE 1=1
            WHERE level = '日线' AND industry = %(industry)s
            ORDER BY code
```

48. ```sql
:
                return []
            
            # 分割字段
            fields = [field.strip() for field in select_part.split(
```

49. ```sql
SELECT 
            MAX(date) as max_date
        FROM stock_info WHERE 1=1
        WHERE
            industry != ''
```

50. ```sql
SELECT DISTINCT industry as name, '' as code
            FROM stock_info WHERE 1=1
            WHERE industry != ''
            ORDER BY industry
```

51. ```sql
get_selection_result方法调用了不存在的表stock_selection_result，当前数据库只有stock_info表
```

52. ```sql
])

    def get_selection_result(self, strategy_id: str, selection_date: str) -> pd.Data_frame:
        """
        获取指定日期的选股结果（注意：当前数据库中没有stock_selection_result表）

        Args:
            strategy_...
```

53. ```sql
SELECT MIN(date) as min_date
            FROM stock_info WHERE 1=1
            WHERE {
```

54. ```sql
SELECT code, name FROM stock_info
```

55. ```sql
])

    def save_selection_result_Db(self, result: pd.Data_frame, strategy_id: str,
                              selection_date: Optional[str] = None) -> bool:
        """
        保存选股结果（注意：当前数据库中没有s...
```

56. ```sql
) if isinstance(max_date, datetime.datetime) else str(max_date)

            return None
        except Exception as e:
            logger.error(f"获取股票最新日期时出错: {e}")
            return None

    def g...
```

### ./db/data_manager.py
- 查询数量: 10

1. ```sql
)
    
    @performance_monitor(threshold=0.5)
    def get_selection_result_Manager(self, strategy_id: str, selection_date: str) -> pd.Data_frame:
```

2. ```sql
try:
            # 使用数据访问接口获取数据
            return self.data_access.get_selection_history_Manager(strategy_id, start_date, end_date, limit)
            
        except Exception as e:
            logg...
```

3. ```sql
保存选股结果
        
        Args:
            result: 选股结果Data_frame
            strategy_id: 策略ID
            selection_date: 选股日期，默认为当前日期
            
        Returns:
            保存成功返回True，否则返回False
 ...
```

4. ```sql
)
                
            if not selection_date:
                raise DataValidationError(
```

5. ```sql
)
            
            # 使用数据访问接口保存数据
            return self.data_access.save_selection_result_Manager(result, strategy_id, selection_date)
            
        except Exception as e:
           ...
```

6. ```sql
)
    
    @performance_monitor(threshold=0.5)
    def get_selection_history_Manager(self, strategy_id: Optional[str] = None, 
                             start_date: Optional[str] = None, 
         ...
```

7. ```sql
,
                         **kwargs) -> Stock_info:
        """
        获取K线数据的兼容性接口（别名）。
        内部直接调用 get_stock_info WHERE 1=1 并返回其结果。
        """
        logger.warning("方法 get_kline_data 已被弃用，请尽快...
```

8. ```sql
)

        # 调用主方法并直接返回StockInfo对象
        return self.get_stock_info_Manager(
            stock_code=stock_code,
            start_date=start_date,
            end_date=end_date,
            level=le...
```

9. ```sql
)
            
            # 使用数据访问接口获取数据
            return self.data_access.get_selection_result_Manager(strategy_id, selection_date)
            
        except Exception as e:
            if isins...
```

10. ```sql
获取指定日期的选股结果
        
        Args:
            strategy_id: 策略ID
            selection_date: 选股日期
            
        Returns:
            选股结果Data_frame
        
        Raises:
            Data_acc...
```

### ./db/data_manager_adapter.py
- 查询数量: 8

1. ```sql
] = industry

                where_clause = " AND ".join(conditions) if conditions else "1=1"

                # 查询不重复的股票代码
                query = f"""
                SELECT DISTINCT code
         ...
```

2. ```sql
SELECT DISTINCT code
                FROM stock_info
```

3. ```sql
)
            
            logger.info(f"选股结果已保存: {strategy_id}, 日期: {selection_date}, 股票数: {len(result)}")
            return True
            
        except Exception as e:
            logger.error...
```

4. ```sql
)
    
    def save_selection_result_Adapter(self, 
                            result: pd.Data_frame, 
                            strategy_id: str, 
                            selection_date: str =...
```

5. ```sql
)
            # 返回一个合理的默认值
            from datetime import datetime, timedelta
            base_date = datetime.strptime(date, '%Y-%m-%d')
            previous_date = base_date - timedelta(days=days ...
```

6. ```sql
)
            
            # 简化实现：记录日志
            if selection_date is None:
                selection_date = datetime.now().strftime('%Y-%m-%d')
            
            logger.info(f
```

7. ```sql
保存选股结果（兼容原有API）
        
        Args:
            result: 选股结果Data_frame
            strategy_id: 策略ID
            selection_date: 选股日期
            
        Returns:
            bool: 保存成功返回True
```

8. ```sql
SELECT DISTINCT code
                FROM stock_info WHERE 1=1
                WHERE {where_clause}
                ORDER BY code
```

### ./db/db_manager.py
- 查询数量: 3

1. ```sql
self._ensure_data_access()
        return self._data_access.execute_dbmanager(sql, params)
    
    def insert_dataframe(self, table_name: str, df, **kwargs):
```

2. ```sql
)
                self._data_access = clickhouse_module.ClickhouseDB()
                self.logger.warning("使用最后的降级ClickhouseDB实现")
            except Exception as fallback_error:
                self...
```

3. ```sql
self._ensure_data_access()
        return self._data_access.insert_dataframe(table_name, df, **kwargs)
    
    def get_stock_data_manager_db_manager(self, stock_code: str, start_date: str, end_date: ...
```

### ./db/enhanced_connection_pool.py
- 查询数量: 2

1. ```sql
:
        """创建新的数据库连接"""
        try:
            client = Client(**self.config)
            
            # 测试连接
            client.execute_1("SELECT 1")
            
            conn_id = f"conn_{in...
```

2. ```sql
].client.execute_1("SELECT 1")
                    conn_info[
```

### ./db/enhanced_data_manager.py
- 查询数量: 5

1. ```sql
{query_hints}
        SELECT {field_str}
        FROM stock_info WHERE 1=1
        WHERE {where_clause}
        ORDER BY {order_by}
```

2. ```sql
的缓存，共 {len(keys_to_remove)} 项")
    
    def get_stats_Manager_Enhanced_Data_Manager(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self.stats_lock:
            stats = self.stats.copy()
 ...
```

3. ```sql
SELECT {field_str}
        FROM stock_info
```

4. ```sql
]:
            query_hints = "/* SETTINGS max_threads = 4 */"
        
        # 构建查询语句
        query = f"""
        {query_hints}
        SELECT {field_str}
        FROM stock_info WHERE 1=1
        ...
```

5. ```sql
with self.stats_lock:
            stats = self.stats.copy()
        
        # 添加缓存统计
        with self.cache_lock:
            stats.update({
                'cache_size': len(self.query_cache),
    ...
```

### ./db/interfaces/cache_interface.py
- 查询数量: 1

1. ```sql
pass
    
    @abstractmethod
    def delete_Interface(self, key: str) -> bool:
```

### ./db/managers/cache_manager.py
- 查询数量: 1

1. ```sql
with self.lock:
            if key in self.cache:
                # 更新现有项
                self.cache.pop(key)
            elif len(self.cache) >= self.max_size:
                # 移除最久未使用的项
           ...
```

### ./db/managers/connection_manager.py
- 查询数量: 3

1. ```sql
health_info = {
            'is_healthy': False,
            'check_time': datetime.now(),
            'connection_test': False,
            'response_time': None,
            'error_message': None
  ...
```

2. ```sql
):
                    raw_connection.__exit__(None, None, None)
    
    def release_connection_Manager(self, connection_id: str) -> None:
        """
        释放数据库连接
        
        Args:
         ...
```

3. ```sql
: None
        }
        
        try:
            start_time = time.time()
            
            # 测试连接
            connection_test = self.connection_manager.test_connection_Manager_Connection_Man...
```

### ./db/managers/data_access_manager.py
- 查询数量: 39

1. ```sql
SELECT 
                COUNT(*) as total_stocks,
                AVG(price_change) as avg_change,
                SUM(CASE WHEN price_change > 0 THEN 1 ELSE 0 END) as rising_count,
                SU...
```

2. ```sql
SELECT COUNT(*) as count
            FROM stock_info
```

3. ```sql
SELECT COUNT(*) as count
            FROM stock_info WHERE 1=1
            WHERE code = %(code)s 
              AND date = %(date)s 
              AND level = %(level)s
```

4. ```sql
: date})
            
            if result.empty:
                return {}
            
            return result.iloc[0].to_dict()
            
        except Exception as e:
            logger.err...
```

5. ```sql
)
            """
            
            return self.query_Manager_Data_Access_Manager(query)
            
        except Exception as e:
            logger.error(f"获取股票基本信息失败: {e}")
            rai...
```

6. ```sql
SELECT DISTINCT code, name, industry
            FROM stock_info WHERE 1=1
            WHERE code IN ('{code_list}')
```

7. ```sql
SELECT AVG(close) as avg_price
            FROM stock_info
```

8. ```sql
SELECT DISTINCT industry
            FROM stock_info
```

9. ```sql
].unique().tolist() if not result_df.empty else []
            
            # 缓存结果
            self.cache_service.set(cache_key, stock_list, ttl=600)  # 10分钟缓存
            
            return stock_li...
```

10. ```sql
".join(codes)
            query = f"""
            SELECT DISTINCT code, name, industry
            FROM stock_info WHERE 1=1
            WHERE code IN (
```

11. ```sql
SELECT DISTINCT stock_code
            FROM index_stocks
```

12. ```sql
AND industry IS NOT NULL
            ORDER BY industry
            """
            
            # 执行查询
            result_df = self.query_Manager_Data_Access_Manager(query)
            
            # ...
```

13. ```sql
SELECT DISTINCT stock_code
            FROM index_stocks 
            WHERE index_code = %(index_code)s
            ORDER BY stock_code
```

14. ```sql
: index_code})
            
            # 尝试从缓存获取
            cached_result = self.cache_service.get(cache_key)
            if cached_result is not None:
                return cached_result
         ...
```

15. ```sql
".join(codes)
            query = f"""
            SELECT code, date, open, high, low, close, volume, amount, 
                   price_change, price_change_pct, name, industry
            FROM stock_...
```

16. ```sql
SELECT code, date, open, high, low, close, volume, amount,
                   price_change, price_change_pct, name, industry
            FROM stock_info
```

17. ```sql
SELECT code, name, date, level, open, high, low, close, volume, 
               turnover_rate, price_change, price_range, industry
        FROM stock_info
```

18. ```sql
SELECT AVG(close) as avg_price
            FROM stock_info WHERE 1=1
            WHERE code = %(code)s AND date >= %(start_date)s
```

19. ```sql
: limit
            })
            
            # 尝试从缓存获取
            cached_result = self.cache_service.get(cache_key)
            if cached_result is not None:
                logger.debug(f"从缓存获取最新...
```

20. ```sql
SELECT 
                industry,
                COUNT(*) as stock_count,
                AVG(price_change) as avg_change,
                AVG(turnover_rate) as avg_turnover
            FROM stock_in...
```

21. ```sql
"
            result = self.query_Manager_Data_Access_Manager(query)
            
            if result.empty or pd.isna(result.iloc[0, 0]):
                return datetime.now()
            
        ...
```

22. ```sql
SELECT DISTINCT industry
            FROM stock_info WHERE 1=1
            WHERE industry != '' AND industry IS NOT NULL
            ORDER BY industry
```

23. ```sql
: date})
            
        except Exception as e:
            logger.error(f"获取行业表现失败: {e}")
            raise DataAccessError(f"获取行业表现失败: {e}")
    
    # 私有辅助方法
    def _generate_cache_key_Data_A...
```

24. ```sql
SELECT 
                COUNT(*) as total_stocks,
                AVG(price_change) as avg_change,
                SUM(CASE WHEN price_change > 0 THEN 1 ELSE 0 END) as rising_count,
                SU...
```

25. ```sql
SELECT DISTINCT code, name, industry
            FROM stock_info
```

26. ```sql
: level
            })
            
            # 尝试从缓存获取
            cached_result = self.cache_service.get(cache_key)
            if cached_result is not None:
                return cached_result
 ...
```

27. ```sql
SELECT DISTINCT code
        FROM stock_info
```

28. ```sql
SELECT MAX(date) as max_date 
            FROM stock_info WHERE 1=1
            WHERE industry != '' AND industry IS NOT NULL
```

29. ```sql
SELECT MAX(date) as max_date FROM stock_info WHERE date >= '2020-01-01'
```

30. ```sql
SELECT MAX(date) as max_date FROM stock_info
```

31. ```sql
SELECT 
                industry,
                COUNT(*) as stock_count,
                AVG(price_change) as avg_change,
                AVG(turnover_rate) as avg_turnover
            FROM stock_in...
```

32. ```sql
SELECT MAX(date) as max_date 
            FROM stock_info
```

33. ```sql
)
            
            query = """
            SELECT AVG(close) as avg_price
            FROM stock_info WHERE 1=1
            WHERE code = %(code)s AND date >= %(start_date)s
            """
   ...
```

34. ```sql
SELECT DISTINCT code
        FROM stock_info WHERE 1=1
        WHERE 1=1
```

35. ```sql
SELECT code, date, open, high, low, close, volume, amount, 
                   price_change, price_change_pct, name, industry
            FROM stock_info WHERE 1=1
            WHERE code IN ('{code_li...
```

36. ```sql
SELECT code, name, date, level, open, high, low, close, volume, 
               turnover_rate, price_change, price_range, industry
        FROM stock_info WHERE 1=1
        WHERE 1=1
```

37. ```sql
SELECT code, date, open, high, low, close, volume, amount, 
                   price_change, price_change_pct, name, industry
            FROM stock_info
```

38. ```sql
SELECT code, date, open, high, low, close, volume, amount,
                   price_change, price_change_pct, name, industry
            FROM stock_info WHERE 1=1
            WHERE code = %(code)s AND...
```

39. ```sql
]:
                    query += f" AND {key} = %({key})s"
                    params[key] = value
        
        # 排序
        query += f" ORDER BY {order_by}"
        
        # 限制条数
        if limi...
```

### ./db/memory_optimizer.py
- 查询数量: 5

1. ```sql
# 优化整数类型
        for col in df.select_dtypes(include=['int64']).columns:
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_min >= -128 and col_max...
```

2. ```sql
)
        
        # 优化浮点类型
        for col in df.select_dtypes(include=[
```

3. ```sql
for col in df.select_dtypes(include=['object']).columns:
            if df[col].dtype == 'object':
                # 检查是否适合转换为分类数据
                unique_ratio = df[col].nunique() / len(df)
          ...
```

4. ```sql
: self.process_memory_mb
        }


class MemoryOptimizer:
    """
    内存优化器
    
    提供智能内存管理功能：
    - 实时内存监控
    - 数据分块处理
    - 自动内存优化
    - Data_frame内存优化
    - 垃圾回收管理
    """
    
    def __init_...
```

5. ```sql
)
        
        return df
    
    def _optimize_string_columns(self, df: pd.Data_frame) -> pd.Data_frame:
        """优化字符串列"""
        for col in df.select_dtypes(include=[
```

### ./db/parallel_processor.py
- 查询数量: 5

1. ```sql
},
            success=True
        )
    
    def _organize_indicator_results(self, results: List[Task_result]) -> Dict[str, Dict[str, Indicator_result]]:
        """整理指标计算结果"""
        organized = {...
```

2. ```sql
: 0.0
        }
        
    def process_indicators_parallel(self, stock_data: Dict[str, pd.Data_frame],
                                  indicators: List[str],
                                  para...
```

3. ```sql
)
        
        # 创建处理任务
        tasks = self._create_indicator_tasks(stock_data, indicators, params)
        
        # 选择处理模式并执行
        if self.config.mode == Processing_mode.THREAD:
           ...
```

4. ```sql
)
        
        return organized
    
    def _update_stats_Parallel_Processor(self, task_count: int, total_time: float):
```

5. ```sql
,
                stock_code=code,
                data=pd.Data_frame(),  # 空Data_frame，由处理函数获取数据
                indicators=[],
                params=func_params
            )
            tasks.appe...
```

### ./db/performance_optimizer.py
- 查询数量: 12

1. ```sql
def __init___32(self, data_access: IData_access, cache_service: ICache_service,
                 config: Optional[Optimization_config] = None):
        self.data_access = data_access
        self.cach...
```

2. ```sql
)
            selection_results = self._execute_selection_strategy(indicator_results, strategy_params)
            
            # 第四步：生成性能报告
            total_time = time.time() - start_time
         ...
```

3. ```sql
selected_stocks = []
        selection_details = {}
        
        for stock_code, indicators in indicator_results.items():
            # 简化的选股逻辑（实际应该根据具体策略实现）
            score = self._calculate_st...
```

4. ```sql
)
        
        return {
            'selected_stocks': selected_stocks,
            'selection_details': selection_details,
            'total_analyzed': len(indicator_results),
            'selec...
```

5. ```sql
, 0.6)
            }
            
            if selection_details[stock_code][
```

6. ```sql
: total_time
            }
    
    def _get_optimized_stock_data(self, stock_codes: List[str],
                                 start_date: str, end_date: str) -> Dict[str, pd.Data_frame]:
        ""...
```

7. ```sql
)
        
        return indicator_results
    
    def _execute_selection_strategy(self, indicator_results: Dict[str, Dict[str, Any]],
                                   strategy_params: Dict[str, A...
```

8. ```sql
: f"{self.stocks_per_second / 2.22:.1f}x"  # 相比原来30分钟的提升
        }


class PerformanceOptimizer:
    """
    性能优化主控制器
    
    整合所有优化组件，提供统一的高性能股票分析接口：
    - 批量数据获取优化
    - 并行指标计算
    - 智能内存管理
    - 性...
```

9. ```sql
for i in range(stock_count)]
            
            # 执行测试
            start_time = time.time()
            try:
                result = self.optimize_stock_selection(
                    stock_cod...
```

10. ```sql
)
            
            return {
                'selection_results': selection_results,
                'performance_report': performance_report.to_dict_Optimizer_Performance_Optimizer(),
        ...
```

11. ```sql
]:
                selected_stocks.append(stock_code)
        
        logger.info(f"选股策略执行完成: 选中 {len(selected_stocks)} 只股票")
        
        return {
```

12. ```sql
, 0.0)
        
        # 获取内存使用情况
        memory_stats = self.memory_optimizer.get_memory_stats()
        
        # 计算并行效率
        processing_stats = self.parallel_processor.get_processing_stats()
 ...
```

### ./db/query_cache.py
- 查询数量: 4

1. ```sql
in query_key:
                # 这里需要更复杂的逻辑来解析查询键并提取相关数据
                return agg_data.head(10)  # 简化实现
        except Exception as e:
            logger.debug(f"从预聚合数据提取失败: {e}")
        
        re...
```

2. ```sql
] += 1
            self._update_query_pattern(key)
            return result
        
        self.stats[
```

3. ```sql
)
        
        return None
    
    def _update_query_pattern(self, key: str):
```

4. ```sql
ttl = ttl or self.default_ttl
        self.stats['total_queries'] += 1
        
        # 1. 检查内存缓存
        result = self._get_from_memory(key, ttl)
        if result is not None:
            self.sta...
```

### ./db/services/cache_service.py
- 查询数量: 6

1. ```sql
]
        
        for key in keys_to_delete:
            if not self.cache_layer.delete_Service(key):
                success = False
        
        logger.info(f
```

2. ```sql
], levels)
    
    def get_strategy_result(self, strategy_name: str, params: Dict[str, Any], 
                          date_param: Union[str, date]) -> Optional[Dict[str, Any]]:
        """获取策略执行结果"...
```

3. ```sql
return self.cache_layer.set_8(key, value, ttl)
    
    def delete_Service(self, key: str) -> bool:
```

4. ```sql
success = True
        
        # 删除市场概览
        keys_to_delete = [
```

5. ```sql
return self.cache_layer.delete_Service(key)
    
    def exists_Service(self, key: str) -> bool:
```

6. ```sql
success = True
        
        # 删除基础信息
        basic_key = self.key_builder.build_stock_basic_key(code)
        if not self.cache_layer.delete_Service(basic_key):
            success = False
       ...
```

### ./db/sql_manager.py
- 查询数量: 22

1. ```sql
SELECT code, date, close,
                       LAG(close, 1) OVER (PARTITION BY code ORDER BY date) as prev_close,
                       (close - LAG(close, 1) OVER (PARTITION BY code ORDER BY date...
```

2. ```sql
SELECT code, name, date, open, high, low, close, volume, 
                       turnover_rate, price_change, price_range, industry
                FROM stock_info 
                WHERE code = %(code...
```

3. ```sql
SELECT code, date, open, high, low, close, volume
                FROM stock_info
```

4. ```sql
SELECT DISTINCT industry, COUNT(*) as stock_count
                FROM stock_info 
                WHERE industry IS NOT NULL 
                AND date = (SELECT MAX(date) FROM stock_info)
           ...
```

5. ```sql
SELECT code, name, date, open, high, low, close, volume, 
                       turnover_rate, price_change, price_range, industry
                FROM stock_info 
                WHERE code IN %(cod...
```

6. ```sql
SELECT config FROM strategy_definitions 
                WHERE strategy_id = %(strategy_id)s 
                LIMIT 1
```

7. ```sql
SELECT code, date, close,
                       LAG(close, 1) OVER (PARTITION BY code ORDER BY date) as prev_close,
                       (close - LAG(close, 1) OVER (PARTITION BY code ORDER BY date...
```

8. ```sql
SELECT * FROM %(table_name)s
                WHERE code = %(code)s 
                AND date BETWEEN %(start_date)s AND %(end_date)s
                ORDER BY date ASC
            """,
            
   ...
```

9. ```sql
SELECT code, name, industry, 
                       MAX(date) as latest_date,
                       COUNT(*) as record_count
                FROM stock_info 
                WHERE code = %(code)s
  ...
```

10. ```sql
SELECT MIN(date) as start_date, MAX(date) as end_date
                FROM stock_info
```

11. ```sql
SELECT MIN(date) as start_date, MAX(date) as end_date
                FROM stock_info 
                WHERE code = %(code)s
                AND level = %(level)s
```

12. ```sql
SELECT COUNT(DISTINCT code) as total_stocks
                FROM stock_info
```

13. ```sql
SELECT * FROM %(table_name)s
                WHERE code = %(code)s 
                AND date BETWEEN %(start_date)s AND %(end_date)s
                ORDER BY date ASC
```

14. ```sql
SELECT DISTINCT code, name, industry
                FROM stock_info
```

15. ```sql
SELECT MAX(date) FROM stock_info
```

16. ```sql
SELECT code, name, date, open, high, low, close, volume, 
                       turnover_rate, price_change, price_range, industry
                FROM stock_info 
                WHERE code = %(code...
```

17. ```sql
SELECT DISTINCT code, name, industry
                FROM stock_info 
                WHERE date = (SELECT MAX(date) FROM stock_info)
                AND level = %(level)s
                ORDER BY cod...
```

18. ```sql
SELECT code, name, date, open, high, low, close, volume, 
                       turnover_rate, price_change, price_range, industry
                FROM stock_info
```

19. ```sql
SELECT code, date, open, high, low, close, volume
                FROM stock_info 
                WHERE code = %(code)s
                AND date = %(date)s
                AND level = %(level)s
     ...
```

20. ```sql
SELECT COUNT(DISTINCT code) as total_stocks
                FROM stock_info 
                WHERE date = (SELECT MAX(date) FROM stock_info)
                AND level = %(level)s
```

21. ```sql
SELECT code, name, industry, 
                       MAX(date) as latest_date,
                       COUNT(*) as record_count
                FROM stock_info
```

22. ```sql
SELECT DISTINCT industry, COUNT(*) as stock_count
                FROM stock_info
```

### ./db/unified_data_manager.py
- 查询数量: 16

1. ```sql
]:
            query_hints = "/* SETTINGS max_threads = 4 */"

        # 构建查询语句
        query = f"""
        {query_hints}
        SELECT {field_str}
        FROM stock_info WHERE 1=1
        WHERE {w...
```

2. ```sql
)

    # ==================== 兼容性API ====================

    @performance_monitor(threshold=0.5)
    def save_selection_result(self, result: pd.Data_frame, strategy_id: str,
                        ...
```

3. ```sql
)

            # 简化实现：记录日志
            if selection_date is None:
                selection_date = datetime.now().strftime('%Y-%m-%d')

            logger.info(f
```

4. ```sql
] = industry

                where_clause = " AND ".join(conditions) if conditions else "1=1"

                # 查询不重复的股票代码
                query = f"""
                SELECT DISTINCT code
         ...
```

5. ```sql
SELECT DISTINCT industry FROM stock_info WHERE industry IS NOT NULL ORDER BY industry
```

6. ```sql
SELECT DISTINCT industry FROM stock_info
```

7. ```sql
保存选股结果（兼容原有API）

        Args:
            result: 选股结果Data_frame
            strategy_id: 策略ID
            selection_date: 选股日期

        Returns:
            bool: 保存成功返回True
```

8. ```sql
的缓存，共 {len(keys_to_remove)} 项")

    # ==================== 兼容性API ====================

    @performance_monitor(threshold=0.5)
    def save_selection_result(self, result: pd.Data_frame, strategy_id:...
```

9. ```sql
{query_hints}
        SELECT {field_str}
        FROM stock_info WHERE 1=1
        WHERE {where_clause}
        ORDER BY {order_by}
```

10. ```sql
]].drop_duplicates()

            return result

        except Exception as e:
            logger.error(f"获取股票基本信息出错: {e}")
            raise DataAccessError(f"获取股票基本信息失败: {e}")

    def get_industry...
```

11. ```sql
,
            failure_threshold=5,
            recovery_timeout=60
        )
        
        # 严格数据库依赖模式：禁用降级服务
        logger.info("严格数据库依赖模式：已禁用所有降级服务和模拟数据支持")
    
    def test_connection_Manager(...
```

12. ```sql
SELECT {field_str}
        FROM stock_info
```

13. ```sql
SELECT 1 as test")
                if result.empty:
                    raise DataAccessError("数据库连接测试返回空结果")
                return True
        except Exception as e:
            logger.error(f"❌ 数据...
```

14. ```sql
with self.stats_lock:
            stats = self.stats.copy()

        # 添加缓存统计
        with self.cache_lock:
            stats.update({
                'cache_size': len(self.query_cache),
            ...
```

15. ```sql
SELECT DISTINCT code
                FROM stock_info WHERE 1=1
                WHERE {where_clause}
                ORDER BY code
```

16. ```sql
)

            logger.info(f"选股结果已保存: {strategy_id}, 日期: {selection_date}, 股票数: {len(result)}")
            return True

        except Exception as e:
            logger.error(f"保存选股结果失败: {e}")
     ...
```

### ./debug_indicator_condition.py
- 查询数量: 1

1. ```sql
import os
import sys

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)

from db.unified_data_manager import Unified_data_manager
from strategy.stra...
```

### ./debug_stock_603359.py
- 查询数量: 6

1. ```sql
SELECT date, open, high, low, close, volume, turnover_rate
            FROM stock_info WHERE 1=1
            WHERE code = '{stock_code}'
            AND date BETWEEN '{start_date}' AND '{end_date}'
  ...
```

2. ```sql
)

        # 直接查询数据库检查数据存在性
        print("2.1 直接查询数据库检查数据")
        with data_manager.connection_pool.get_connection() as conn:
            # 查询日线数据
            daily_query = f"""
            SELECT ...
```

3. ```sql
]}")
                else:
                    print(f"   ❌ 数据库中没有 {target_date} 的日线数据")

            # 查询30分钟数据
            min30_query = f"""
            SELECT date, datetime, open, high, low, clos...
```

4. ```sql
SELECT date, datetime, open, high, low, close, volume
            FROM stock_info
```

5. ```sql
SELECT date, datetime, open, high, low, close, volume
            FROM stock_info WHERE 1=1
            WHERE code = '{stock_code}'
            AND date BETWEEN '{start_date}' AND '{end_date}'
       ...
```

6. ```sql
SELECT date, open, high, low, close, volume, turnover_rate
            FROM stock_info
```

### ./debug_stock_count.py
- 查询数量: 8

1. ```sql
SELECT date, COUNT(DISTINCT code) as stocks_count 
                FROM stock_info
```

2. ```sql
SELECT date, COUNT(DISTINCT code) as stocks_count 
                FROM stock_info WHERE date >= '2020-01-01' 
                GROUP BY date 
                ORDER BY date DESC 
                LIMIT ...
```

3. ```sql
SELECT COUNT(*) as total FROM stock_info WHERE date >= '2020-01-01'
```

4. ```sql
SELECT COUNT(DISTINCT code) as unique_stocks FROM stock_info WHERE date >= '2020-01-01'
```

5. ```sql
SELECT COUNT(*) as total FROM stock_info
```

6. ```sql
SELECT COUNT(DISTINCT code) as unique_stocks FROM stock_info
```

7. ```sql
]}")
            
            # 查询最新日期的股票数量
            result3 = conn.query_dataframe("""
                SELECT date, COUNT(DISTINCT code) as stocks_count 
                FROM stock_info WHERE date...
```

8. ```sql
]}")
            
            # 查询不同股票代码数量
            result2 = conn.query_dataframe("SELECT COUNT(DISTINCT code) as unique_stocks FROM stock_info WHERE date >=
```

### ./enums/indicator_types.py
- 查询数量: 1

1. ```sql
# ZXM买点评分指标
    
    # 综合指标
    ZXM_SELECTION_MODEL =
```

### ./examples/buypoint_strategy_integration_example.py
- 查询数量: 10

1. ```sql
required_fields = [
            'stock_code', 'stock_name', 'industry', 'price',
            'change_pct', 'score', 'match_details', 'selection_date'
        ]
        
        missing_fields = [field...
```

2. ```sql
)}.csv"
                final_selection.to_csv(output_file, index=False, encoding=
```

3. ```sql
, ascending=False)
                
                print("\n🎯 最终选股结果:")
                print("=" * 50)
                
                for i, (_, row) in enumerate(final_selection.iterrows(), 1):
 ...
```

4. ```sql
])}个")
                    print(f"   来源: 买点分析系统")
                    print()
                
                # 保存结果
                output_file = f"selection_results_{datetime.now().strftime(
```

5. ```sql
selection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv
```

6. ```sql
)
                print(f"📄 选股结果已保存: {output_file}")
                
            else:
                print("⚠️ 没有股票符合筛选条件")
                
        except Exception as e:
            print(f"❌ 示例执...
```

7. ```sql
)
            ].copy()
            
            print(f"✅ 筛选出 {len(filtered_df)} 只符合条件的股票")
            
            # 步骤4: 生成最终选股结果
            print("步骤4: 生成最终选股结果...")
            
            if n...
```

8. ```sql
* 60)
        
        try:
            # 创建自定义权重的适配器
            custom_adapter = get_buypoint_strategy_adapter()
            
            # 修改权重配置（更重视吸筹信号）
            custom_adapter.strategy_weight...
```

9. ```sql
)
            
            if not filtered_df.empty:
                # 按评分排序
                final_selection = filtered_df.sort_values('score', ascending=False)
                
                print(
```

10. ```sql
* 50)
                
                for i, (_, row) in enumerate(final_selection.iterrows(), 1):
                    print(f
```

### ./examples/combined_indicators_strategy.py
- 查询数量: 4

1. ```sql
SELECT 
            trade_date as date,
            open,
            high,
            low,
            close,
            volume,
            amount,
            turnover_rate
        FROM stock_dai...
```

2. ```sql
SELECT 
            trade_date as date,
            open,
            high,
            low,
            close,
            volume,
            amount,
            turnover_rate
        FROM stock_dai...
```

3. ```sql
, 0) > 0 else -1 for t in trades_with_profit]
        max_consecutive_losses = 0
        current_consecutive_losses = 0
        
        for flag in profit_flags:
            if flag < 0:
            ...
```

4. ```sql
)]
                    for col in signal_columns:
                        if col in data.columns:
                            indicator_state[col] = data[col].iloc[i]
                    
            ...
```

### ./examples/complete_example.py
- 查询数量: 1

1. ```sql
import sys
import os
import subprocess
import pandas as pd
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(proje...
```

### ./examples/comprehensive_indicator_scoring_demo.py
- 查询数量: 1

1. ```sql
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname...
```

### ./examples/parameter_optimization.py
- 查询数量: 2

1. ```sql
SELECT 
            trade_date as date,
            open,
            high,
            low,
            close,
            volume,
            amount,
            turnover_rate
        FROM stock_dai...
```

2. ```sql
SELECT 
            trade_date as date,
            open,
            high,
            low,
            close,
            volume,
            amount,
            turnover_rate
        FROM stock_dai...
```

### ./examples/pattern_registry_example.py
- 查询数量: 3

1. ```sql
import sys
import os

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from indicators.complete_indicator_registry import complet...
```

2. ```sql
更新后的形态: {updated_pattern}
```

3. ```sql
,
        pattern_type=Pattern_type.NEUTRAL
    )
    
    # 验证更新后的形态
    updated_pattern = registry.get_pattern(
```

### ./examples/test_additional_indicators_scoring.py
- 查询数量: 1

1. ```sql
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path...
```

### ./examples/test_all_indicators_scoring.py
- 查询数量: 1

1. ```sql
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path...
```

### ./examples/test_basic_optimization.py
- 查询数量: 3

1. ```sql
SELECT 
        trade_date,
        open,
        high,
        low,
        close,
        volume,
        amount
    FROM stock_daily_kline
```

2. ```sql
):
    """加载测试数据"""
    print(f"加载测试数据: {stock_code} 从 {start_date} 到 {end_date}")
    
    data_access = get_container().resolve(IData_access)
    sql = f"""
    SELECT 
        trade_date,
        o...
```

3. ```sql
SELECT 
        trade_date,
        open,
        high,
        low,
        close,
        volume,
        amount
    FROM stock_daily_kline
    WHERE 
        ts_code = '{stock_code}' AND
        tr...
```

### ./examples/test_ichimoku_aroon_chaikin_scoring.py
- 查询数量: 1

1. ```sql
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Magic_mock

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname...
```

### ./examples/test_indicator_optimization.py
- 查询数量: 3

1. ```sql
SELECT 
        trade_date,
        open,
        high,
        low,
        close,
        volume,
        amount
    FROM stock_daily_kline
```

2. ```sql
):
    """加载测试数据"""
    print(f"加载测试数据: {stock_code} 从 {start_date} 到 {end_date}")
    
    data_access = get_container().resolve(IData_access)
    sql = f"""
    SELECT 
        trade_date,
        o...
```

3. ```sql
SELECT 
        trade_date,
        open,
        high,
        low,
        close,
        volume,
        amount
    FROM stock_daily_kline
    WHERE 
        ts_code = '{stock_code}' AND
        tr...
```

### ./examples/test_indicator_scoring.py
- 查询数量: 9

1. ```sql
import sys
import os
import pandas as pd
import numpy as np

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from indicators.com...
```

2. ```sql
SELECT date, open, high, low, close, volume
        FROM stock_info WHERE 1=1
        WHERE code = '000001'
        AND level = '日线'
        AND date >= '2024-01-01'
        ORDER BY date
        LIMI...
```

3. ```sql
SELECT date, open, high, low, close, volume
    FROM stock_info WHERE 1=1
    WHERE code = '000001'
    AND level = '日线'
    AND date >= '2024-01-01'
    ORDER BY date
    LIMIT 100
```

4. ```sql
SELECT date, open, high, low, close, volume
    FROM stock_info WHERE 1=1
    WHERE code = '000001'
    AND level = '日线'
    AND date >= '2023-01-01'
    ORDER BY date
    LIMIT 500
```

5. ```sql
: 20}
    ]
    
    try:
        # 注意：complete_registry主要用于指标注册，评分管理器需要单独创建
        score_manager = Indicator_score_manager(indicator_configs)
        logger.info("成功通过注册机制创建评分管理器")
        
        ...
```

6. ```sql
SELECT date, open, high, low, close, volume
        FROM stock_info
```

7. ```sql
].mean():.2f}")
        
    except Exception as e:
        logger.error(f"测试指标注册机制失败: {e}")


def test_pattern_recognition_Scoring():
    """测试形态识别功能"""
    logger.info("开始测试形态识别功能")
    
    # 获取更多测...
```

8. ```sql
SELECT date, open, high, low, close, volume
    FROM stock_info
```

9. ```sql
])}")


def test_comprehensive_scoring_Scoring():
    """测试综合评分系统"""
    logger.info("开始测试综合评分系统")
    
    # 获取测试数据
    data_access = get_container().resolve(IData_access)
    sql = """
    SELECT da...
```

### ./examples/test_ma_ema_sar_scoring.py
- 查询数量: 1

1. ```sql
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path...
```

### ./examples/test_mfi_vr_scoring.py
- 查询数量: 1

1. ```sql
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path...
```

### ./examples/test_momentum_emv_vosc_pvt_scoring.py
- 查询数量: 1

1. ```sql
import sys
import os
import numpy as np
import pandas as pd

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from indicators.com...
```

### ./examples/test_pattern_indicators_scoring.py
- 查询数量: 1

1. ```sql
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path...
```

### ./examples/test_stochrsi_scoring.py
- 查询数量: 1

1. ```sql
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path...
```

### ./examples/test_unified_scoring.py
- 查询数量: 7

1. ```sql
SELECT date, open, high, low, close, volume
    FROM stock_info WHERE 1=1
    WHERE code = '000001'
    AND level = '日线'
    AND date >= '2024-01-01'
    ORDER BY date
    LIMIT 100
```

2. ```sql
SELECT date, open, high, low, close, volume
    FROM stock_info WHERE 1=1
    WHERE code = '000001'
    AND level = '日线'
    AND date >= '2024-01-01'
    ORDER BY date
    LIMIT 50
```

3. ```sql
SELECT date, open, high, low, close, volume
    FROM stock_info WHERE 1=1
    WHERE code = '000001'
    AND level = '日线'
    AND date >= '2023-01-01'
    ORDER BY date
    LIMIT 300
```

4. ```sql
SELECT date, open, high, low, close, volume
    FROM stock_info
```

5. ```sql
import sys
import os
import pandas as pd
import numpy as np

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from indicators.com...
```

6. ```sql
].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)
            logger.info(f"  - 年化波动率: {volatility:.2%}")


def test_scoring_consistency_Scoring():
    """测试评分一致性"""
    lo...
```

7. ```sql
].tail(5)
            logger.info(f"  - 最近5个周期评分: {recent_scores.values}")
            
        except Exception as e:
            logger.error(f"测试 {name} 指标时出错: {e}")
            import traceback
  ...
```

### ./examples/test_wma_bias_mtm_scoring.py
- 查询数量: 1

1. ```sql
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path...
```

### ./examples/test_zxm_score_indicators.py
- 查询数量: 3

1. ```sql
SELECT
        toDate(date) as trade_date,
        open, high, low, close,
        volume, turnover,
        amount
    FROM stock_daily_data
```

2. ```sql
SELECT
        toDate(date) as trade_date,
        open, high, low, close,
        volume, turnover,
        amount
    FROM stock_daily_data
    WHERE code = '{stock_code}'
    AND date <= '{end_date...
```

3. ```sql
].sum()
    logger.info(f"信号生成数量: {signal_count}")
    logger.info(f"信号生成比例: {signal_count / len(result) * 100:.2f}%")
    

def test_zxm_buypoint_score():
    """测试ZXM买点评分指标"""
    logger.info("开始测试Z...
```

### ./examples/use_advanced_indicators.py
- 查询数量: 2

1. ```sql
SELECT 
            trade_date as date,
            open,
            high,
            low,
            close,
            volume,
            amount
        FROM stock_daily_data
```

2. ```sql
SELECT 
            trade_date as date,
            open,
            high,
            low,
            close,
            volume,
            amount
        FROM stock_daily_data
        WHERE 
    ...
```

### ./examples/use_new_indicators.py
- 查询数量: 2

1. ```sql
SELECT 
            trade_date as date,
            open,
            high,
            low,
            close,
            volume,
            amount,
            turnover_rate
        FROM stock_dai...
```

2. ```sql
SELECT 
            trade_date as date,
            open,
            high,
            low,
            close,
            volume,
            amount,
            turnover_rate
        FROM stock_dai...
```

### ./examples/use_zxm_indicators.py
- 查询数量: 9

1. ```sql
, callback_percent=4.0)
    if not zxm_model:
        print("无法创建ZXM_SELECTION_MODEL指标")
        return

    # 计算选股模型
    result = zxm_model.calculate(daily_data, weekly_data, monthly_data)
    
    #...
```

2. ```sql
)
    selection_model = complete_registry.create_indicator(
```

3. ```sql
)


def demo_zxm_selection_model(daily_data, weekly_data, monthly_data):
```

4. ```sql
)
    if selection_model:
        print(f
```

5. ```sql
)}")
    if selection_model:
        print(f"- {selection_model.name}: {getattr(selection_model,
```

6. ```sql
].sum()
    total_days = len(result)
    print(f"回踩均线买点信号出现次数: {signal_count}，占总天数的 {signal_count/total_days*100:.2f}%")


def demo_zxm_selection_model(daily_data, weekly_data, monthly_data):
    """
...
```

7. ```sql
)

    # 创建选股模型实例
    zxm_model = complete_registry.create_indicator('ZXM_SELECTION_MODEL', callback_percent=4.0)
    if not zxm_model:
        print(
```

8. ```sql
)]

    print(f"系统支持的ZXM指标：")
    for i, indicator in enumerate(zxm_indicators, 1):
        print(f"{i}. {indicator}")

    # 演示使用统一注册系统创建ZXM指标
    daily_trend = complete_registry.create_indicator("ZX...
```

9. ```sql
)
    
    # 演示各类指标
    demo_zxm_daily_trend_up(daily_data)
    demo_zxm_amplitude_elasticity(daily_data)
    demo_zxm_ma_callback(daily_data)
    demo_zxm_selection_model(daily_data, weekly_data, mon...
```

### ./fix_table_names.py
- 查询数量: 2

1. ```sql
AND {m.group(1).strip()}")
    ]
    
    files_fixed = 0
    total_fixes = 0
    
    # 收集所有Python文件
    all_files = set()
    for pattern in patterns:
        files = glob.glob(pattern, recursive=Tr...
```

2. ```sql
)
    ]
    
    files_fixed = 0
    total_fixes = 0
    
    # 收集所有Python文件
    all_files = set()
    for pattern in patterns:
        files = glob.glob(pattern, recursive=True)
        all_files.upd...
```

### ./get-pip.py
- 查询数量: 1

1. ```sql
)))

        # Add the zipfile to sys.path so that we can import it
        sys.path.insert(0, pip_zip)

        # Run the bootstrap
        bootstrap(tmpdir=tmpdir)
    finally:
        # Clean up ou...
```

### ./indicators/ad.py
- 查询数量: 2

1. ```sql
}
        
        return pattern_info_map.get(pattern_id, default_pattern)

    def _get_default_parameters_ad(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {"period": 14}
    
    def...
```

2. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

### ./indicators/adapter.py
- 查询数量: 1

1. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

### ./indicators/adx.py
- 查询数量: 4

1. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

2. ```sql
]
        super().__init__(name="ADX", description="平均方向指数指标")
        
        # 设置默认参数
        self.params = {
            "period": 14,
            "strong_trend": 25
        }
        
        # 更...
```

3. ```sql
}
        
        return pattern_info_map.get(pattern_id, default_pattern)

    def _get_default_parameters_adx(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {"period": 14, "strong_tre...
```

4. ```sql
: 25
        }
        
        # 更新自定义参数
        if params:
            self.params.update(params)
        
        # 注册ADX形态
        self._register_adx_patterns()

        # 导入交叉检测函数
        from in...
```

### ./indicators/aroon.py
- 查询数量: 1

1. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

### ./indicators/atr.py
- 查询数量: 1

1. ```sql
# 验证参数
        from utils.indicator_parameter_validator import Indicator_parameter_validator
        validator = Indicator_parameter_validator()
        
        # 合并默认参数和用户参数
        params = self._d...
```

### ./indicators/bias.py
- 查询数量: 4

1. ```sql
: [6, 12, 24]}
    
    def set_parameters_Bias_Bias_Bias_bias_duplicate(self, **kwargs):
        """
        设置指标参数
        
        Args:
            **kwargs: 参数字典
        """
        # 验证参数
      ...
```

2. ```sql
if score.empty:
            return 0.5

        # 基础置信度
        confidence = 0.5

        # 1. 基于评分的置信度
        last_score = score.iloc[-1]

        # 极端评分（超买/超卖）置信度较高
        if last_score > 70 or la...
```

3. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

4. ```sql
]

        # 计算评分
        # BIAS在-10到+10之间为正常范围，对应40-60分
        # BIAS超过+10为超买，对应60-100分
        # BIAS低于-10为超卖，对应0-40分
        scores = pd.Series(50.0, index=data.index)

        # 处理有效值
        val...
```

### ./indicators/boll.py
- 查询数量: 2

1. ```sql
# 合并默认参数和用户参数
        params = self._default_parameters.copy()
        params.update(kwargs)

        # 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_paramete...
```

2. ```sql
# 注册布林带指标形态
        self._register_boll_patterns()

        # 导入交叉检测函数
        from indicators.common import crossover, crossunder
        self.crossover = crossover
        self.crossunder = crossund...
```

### ./indicators/boll_score.py
- 查询数量: 1

1. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

### ./indicators/cci.py
- 查询数量: 1

1. ```sql
# 验证参数
        from utils.indicator_parameter_validator import Indicator_parameter_validator
        validator = Indicator_parameter_validator()
        
        # 合并默认参数和用户参数
        params = self._d...
```

### ./indicators/chaikin.py
- 查询数量: 1

1. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

### ./indicators/chip_distribution.py
- 查询数量: 1

1. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

### ./indicators/cmo.py
- 查询数量: 4

1. ```sql
].shift(5)
            
            # 上升动量加分
            up_momentum_mask = cmo_change > 3
            score[up_momentum_mask] += 5
            
            # 下降动量减分
            down_momentum_mask = c...
```

2. ```sql
# 验证参数
        from utils.indicator_parameter_validator import Indicator_parameter_validator
        validator = Indicator_parameter_validator()

        # 合并默认参数和用户参数
        params = self._default_p...
```

3. ```sql
if score.empty:
            return 0.5

        # 基础置信度
        confidence = 0.5

        # 1. 基于评分的置信度
        last_score = score.iloc[-1]

        # 极端评分置信度较高
        if last_score > 80 or last_scor...
```

4. ```sql
]

        # 设置默认参数
        self._default_parameters = self._get_default_parameters_cmo()

        # 应用用户参数
        self.set_parameters_Cmo(**kwargs)

    def _get_default_parameters_cmo(self) -> Dict...
```

### ./indicators/complete_indicator_registry.py
- 查询数量: 2

1. ```sql
indicators.zxm.selection_model
```

2. ```sql
)

        zxm_indicators = [
            # ZXM Trend (9个)
            ('indicators.zxm.trend_indicators', 'ZXMDailyTrendUp', 'ZXM_DAILY_TREND_UP', 'ZXM日趋势向上'),
            ('indicators.zxm.trend_indi...
```

### ./indicators/composite.py
- 查询数量: 1

1. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

### ./indicators/composite_indicator.py
- 查询数量: 1

1. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

### ./indicators/divergence.py
- 查询数量: 2

1. ```sql
}
        
        return pattern_info_map.get(pattern_id, default_pattern)

    def _get_default_parameters_divergence(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {}
    
    def set...
```

2. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

### ./indicators/dma.py
- 查询数量: 4

1. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

2. ```sql
if score.empty:
            return 0.5

        # 基础置信度
        confidence = 0.5

        # 1. 基于评分的置信度
        last_score = score.iloc[-1]

        # 极端评分置信度较高
        if last_score > 75 or last_scor...
```

3. ```sql
: 10}
    
    def set_parameters_Dma_Dma_Dma_dma_duplicate(self, **kwargs):
        """
        设置指标参数
        
        Args:
            **kwargs: 参数字典
        """
        # 验证参数
        try:
      ...
```

4. ```sql
]
        # 快速上涨
        score[fast_ma_chg > 2] += 5
        # 快速下跌
        score[fast_ma_chg < -2] -= 5
        
        # 确保分数在0-100范围内
        score = score.clip(0, 100)
        
        return sco...
```

### ./indicators/dmi.py
- 查询数量: 4

1. ```sql
] = adx < adx.shift(1)

        return patterns_df

    def calculate_confidence_Dmi(self, score: pd.Series, patterns: pd.Data_frame, signals: dict) -> float:
        """
        计算DMI指标的置信度

        ...
```

2. ```sql
if score.empty:
            return 0.5

        # 基础置信度
        confidence = 0.5

        # 1. 基于评分的置信度
        last_score = score.iloc[-1]

        # 极端评分置信度较高
        if last_score > 80 or last_scor...
```

3. ```sql
]
        """
        初始化趋向指标(DMI)指标

        Args:
            **kwargs: 指标参数，支持period、adx_threshold等
        """
        super().__init__(name="DMI", description="趋向指标，判断趋势强度与方向")

        # 设置默认参数
...
```

4. ```sql
# 验证参数
        from utils.indicator_parameter_validator import Indicator_parameter_validator
        validator = Indicator_parameter_validator()

        # 合并默认参数和用户参数
        params = self._default_p...
```

### ./indicators/elliott_wave.py
- 查询数量: 1

1. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

### ./indicators/ema.py
- 查询数量: 2

1. ```sql
]
        self.register_patterns_Ema()

    def _get_default_parameters_ema(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {"period": 12, "price_field": "close", "alpha": None}
        
...
```

2. ```sql
# 验证参数
        from utils.indicator_parameter_validator import Indicator_parameter_validator
        validator = Indicator_parameter_validator()

        # 合并默认参数和用户参数
        params = self._default_p...
```

### ./indicators/emv.py
- 查询数量: 4

1. ```sql
]
        
        # 初始基础分50分
        score = pd.Series(50.0, index=data.index)
        
        # 基于EMV值的评分
        for i in range(len(score)):
            if i >= len(emv):
                continue
...
```

2. ```sql
if score.empty:
            return 0.5

        # 基础置信度
        confidence = 0.5

        # 1. 基于评分的置信度
        last_score = score.iloc[-1]

        # 极端评分置信度较高
        if last_score > 75 or last_scor...
```

3. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

4. ```sql
: 14}
    
    def set_parameters_Emv_Emv_Emv_emv_duplicate(self, **kwargs):
        """
        设置指标参数
        
        Args:
            **kwargs: 参数字典
        """
        # 验证参数
        try:
      ...
```

### ./indicators/enhanced_factory.py
- 查询数量: 1

1. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

### ./indicators/enhanced_macd.py
- 查询数量: 1

1. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

### ./indicators/enhanced_rsi.py
- 查询数量: 1

1. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

### ./indicators/enhanced_stochrsi.py
- 查询数量: 1

1. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()

            # 合并默认参数和用户参数
       ...
```

### ./indicators/enhanced_wr.py
- 查询数量: 1

1. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()

            # 合并默认参数和用户参数
       ...
```

### ./indicators/factory.py
- 查询数量: 1

1. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

### ./indicators/fibonacci.py
- 查询数量: 1

1. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

### ./indicators/fibonacci_tools.py
- 查询数量: 1

1. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

### ./indicators/formula_indicators.py
- 查询数量: 1

1. ```sql
: 14}
    
    def set_parameters_Indicators_formulaindicators(self, **kwargs):
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            valid...
```

### ./indicators/gann_tools.py
- 查询数量: 1

1. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

### ./indicators/ichimoku.py
- 查询数量: 1

1. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

### ./indicators/indicator_manager.py
- 查询数量: 1

1. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

### ./indicators/institutional_behavior.py
- 查询数量: 1

1. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

### ./indicators/intraday_volatility.py
- 查询数量: 1

1. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

### ./indicators/island_reversal.py
- 查询数量: 1

1. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

### ./indicators/kc.py
- 查询数量: 4

1. ```sql
if score.empty:
            return 0.5

        # 基础置信度
        confidence = 0.5

        # 1. 基于评分的置信度
        last_score = score.iloc[-1]

        # 极端评分置信度较高
        if last_score > 80 or last_scor...
```

2. ```sql
] = in_channel.all() & cross_middle

        return patterns_df

    def register_patterns_Kc(self):
        """
        注册KC指标的技术形态
        """
        # 注册价格突破形态
        self.register_pattern_to_reg...
```

3. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

4. ```sql
]
        # 通道扩大，上升波动
        up_vol_mask = (width_chg > 10) & (close > middle)
        score[up_vol_mask] += 5
        
        # 通道扩大，下降波动
        down_vol_mask = (width_chg > 10) & (close < middle)...
```

### ./indicators/kdj.py
- 查询数量: 2

1. ```sql
: 3}
    
    def set_parameters_Kdj_Kdj_Kdj_kdj_duplicate(self, **kwargs):
        """
        设置指标参数
        
        Args:
            **kwargs: 参数字典
        """
        # 验证参数
        try:
       ...
```

2. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

### ./indicators/kdj_score.py
- 查询数量: 1

1. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

### ./indicators/ma.py
- 查询数量: 4

1. ```sql
if score.empty:
            return 0.5

        # 基础置信度
        confidence = 0.5

        # 1. 基于评分的置信度
        last_score = score.iloc[-1]

        # 极端评分置信度较高
        if last_score > 80 or last_scor...
```

2. ```sql
# 验证参数
        from utils.indicator_parameter_validator import Indicator_parameter_validator
        validator = Indicator_parameter_validator()

        # 合并默认参数和用户参数
        params = self._default_p...
```

3. ```sql
]
        score[close_price > short_ma] += 10
        score[close_price < short_ma] -= 10
        
        if len(sorted_mas) >= 2:
            short_ma_series = sorted_mas[0]
            medium_ma_se...
```

4. ```sql
]
        self.register_patterns_Ma()

    def _get_default_parameters_ma(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {"period": 20, "price_field": "close"}

    def set_parameters_Ma...
```

### ./indicators/macd.py
- 查询数量: 4

1. ```sql
# 覆盖默认参数
        self._parameters.update(kwargs)

        # 核心计算
        macd_df = self._calculate_macd(data, **self._parameters)
        dif = macd_df['macd_line']
        dea = macd_df['macd_signal'...
```

2. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

3. ```sql
})

    def register_patterns_Macd(self):
        """
        注册MACD指标的形态到全局形态注册表
        """
        # 注册MACD金叉形态
        self.register_pattern_to_registry(
            pattern_id="MACD_GOLDEN_CROSS"...
```

4. ```sql
列
            **kwargs: 其他参数，用于覆盖默认参数

        Returns:
            一个包含各种形态布尔值的Data_frame
        """
        # 覆盖默认参数
        self._parameters.update(kwargs)

        # 核心计算
        macd_df = self._...
```

### ./indicators/macd_score.py
- 查询数量: 1

1. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

### ./indicators/market_env.py
- 查询数量: 1

1. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

### ./indicators/mfi.py
- 查询数量: 1

1. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

### ./indicators/momentum.py
- 查询数量: 1

1. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

### ./indicators/mtm.py
- 查询数量: 2

1. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

2. ```sql
: 0.0}

    def get_pattern_info_Mtm(self, pattern_id: str) -> dict:
        """
        获取指定形态的详细信息
        
        Args:
            pattern_id: 形态ID
            
        Returns:
            dict:...
```

### ./indicators/multi_period_resonance.py
- 查询数量: 1

1. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

### ./indicators/obv.py
- 查询数量: 1

1. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

### ./indicators/pattern/advanced_candlestick_patterns.py
- 查询数量: 4

1. ```sql
# 防止数据为空
        if indicator_values is None or len(indicator_values) == 0:
            return pd.Series(index=indicator_values.index if indicator_values is not None else [])
        
        # 初始化信号强...
```

2. ```sql
) and self._result is not None:
            # 检查是否有高级形态数据
            advanced_pattern_columns = [pattern.value for pattern in Advanced_pattern_type]
            available_patterns = [col for col in a...
```

3. ```sql
if score.empty:
            return 0.5

        # 基础置信度
        confidence = 0.5

        # 1. 基于评分的置信度
        last_score = score.iloc[-1]

        # 极端评分置信度较高
        if last_score > 80 or last_scor...
```

4. ```sql
)
        
        # 初始化结果数组
        n = len(data)
        head_shoulders_top = np.zeros(n, dtype=bool)
        head_shoulders_bottom = np.zeros(n, dtype=bool)
        double_top = np.zeros(n, dtype=b...
```

### ./indicators/pattern/candlestick_patterns.py
- 查询数量: 2

1. ```sql
if score.empty:
            return 0.5

        # 基础置信度
        confidence = 0.5

        # 1. 基于评分的置信度
        last_score = score.iloc[-1]

        # 极端评分置信度较高
        if last_score > 80 or last_scor...
```

2. ```sql
) and self._result is not None:
            # 检查是否有形态数据
            pattern_columns = [col for col in self._result.columns
                             if any(pattern.name.lower() in col for pattern i...
```

### ./indicators/pattern/zxm_patterns.py
- 查询数量: 4

1. ```sql
if score.empty:
            return 0.5

        # 基础置信度
        confidence = 0.5

        # 1. 基于评分的置信度
        last_score = score.iloc[-1]

        # 极端评分置信度较高
        if last_score > 80 or last_scor...
```

2. ```sql
]
            available_patterns = [col for col in zxm_pattern_columns if col in self._result.columns]
            if available_patterns:
                # ZXM形态数据越完整，置信度越高
                data_comple...
```

3. ```sql
]
            for col in expected_columns:
                result[col] = False
            return result

        # 提取价格和成交量数据
        open_prices = data["open"].values
        high_prices = data["hig...
```

4. ```sql
].values

        # 计算基础指标
        ma5 = ma(close_prices, 5)
        ma10 = ma(close_prices, 10)
        ma20 = ma(close_prices, 20)
        ma30 = ma(close_prices, 30)
        ma60 = ma(close_prices,...
```

### ./indicators/pattern_detector.py
- 查询数量: 1

1. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

### ./indicators/pattern_manager.py
- 查询数量: 1

1. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

### ./indicators/pattern_recognition.py
- 查询数量: 1

1. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

### ./indicators/platform_breakout.py
- 查询数量: 1

1. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

### ./indicators/psy.py
- 查询数量: 2

1. ```sql
: False}

    def set_parameters_Psy_Psy_Psy_psy(self, **kwargs):
        """
        设置指标参数

        Args:
            **kwargs: 参数字典
        """
        # 验证参数
        try:
            from utils.in...
```

2. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator(silent_mode=True)

            # 合并默...
```

### ./indicators/pvt.py
- 查询数量: 2

1. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

2. ```sql
: 10}
    
    def set_parameters_Pvt_Pvt_Pvt_pvt_duplicate(self, **kwargs):
        """
        设置指标参数
        
        Args:
            **kwargs: 参数字典
        """
        # 验证参数
        try:
      ...
```

### ./indicators/roc.py
- 查询数量: 1

1. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()

            # 合并默认参数和用户参数
       ...
```

### ./indicators/rsi.py
- 查询数量: 2

1. ```sql
: 0.0}

    def register_patterns_Rsi(self):
        """
        注册RSI指标的形态到全局形态注册表
        """
        # 注册RSI超买形态
        self.register_pattern_to_registry(
            pattern_id="RSI_OVERBOUGHT",
...
```

2. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

### ./indicators/rsi_derivatives.py
- 查询数量: 1

1. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

### ./indicators/rsi_score.py
- 查询数量: 1

1. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

### ./indicators/rsima.py
- 查询数量: 2

1. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

2. ```sql
}
        
        return pattern_info_map.get(pattern_id, default_pattern)

    def _get_default_parameters_rsima(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {}
    
    def set_para...
```

### ./indicators/sar.py
- 查询数量: 6

1. ```sql
}
        
        return pattern_info_map.get(pattern_id, default_pattern)


    def _get_default_parameters_sar(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {"acceleration": 0.02, "m...
```

2. ```sql
# 确保已计算指标
        if self._result is None:
            self.calculate_Sar(data)

        if self._result is None or 'sar' not in self._result.columns:
            return pd.Data_frame(index=data.index...
```

3. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

4. ```sql
]
        close_col = None

        for col in close_columns:
            if col in data.columns:
                close_col = col
                break

        if close_col is None:
            # 尝试从...
```

5. ```sql
]
        close_col = None

        for col in close_columns:
            if col in data.columns:
                close_col = col
                break

        if close_col is None:
            # 尝试从...
```

6. ```sql
score = pd.Series(0.0, index=data.index)

        if 'sar' not in self._result.columns:
            return score

        sar = self._result['sar']

        # 支持多种收盘价列名格式，包括中文列名
        close_columns ...
```

### ./indicators/score_manager.py
- 查询数量: 1

1. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

### ./indicators/scoring_framework.py
- 查询数量: 1

1. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

### ./indicators/sentiment_analysis.py
- 查询数量: 1

1. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

### ./indicators/stochrsi.py
- 查询数量: 1

1. ```sql
# 验证参数
        from utils.indicator_parameter_validator import Indicator_parameter_validator
        validator = Indicator_parameter_validator()
        
        # 合并默认参数和用户参数
        params = self._d...
```

### ./indicators/stock_vix.py
- 查询数量: 1

1. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

### ./indicators/synergy.py
- 查询数量: 1

1. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

### ./indicators/technical_indicators.py
- 查询数量: 1

1. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

### ./indicators/time_cycle_analysis.py
- 查询数量: 1

1. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

### ./indicators/trend/enhanced_cci.py
- 查询数量: 2

1. ```sql
conditions = [
            cci > self.EXTREME_OVERBOUGHT,
            (cci > self.OVERBOUGHT) & (cci <= self.EXTREME_OVERBOUGHT),
            (cci > self.NEUTRAL_HIGH) & (cci <= self.OVERBOUGHT),
    ...
```

2. ```sql
]
        
        return pd.Series(np.select(conditions, choices, default=
```

### ./indicators/trend/trend_strength.py
- 查询数量: 2

1. ```sql
]
        """
        初始化趋势强度指标
        
        Args:
            params: 参数字典，可包含：
                - lookback_period: 回溯周期，默认为20
                - min_strength: 最小强度阈值，默认为30
                - strong...
```

2. ```sql
: 70
        }
        
        # 更新自定义参数
        if params:
            self.params.update(params)
    
    def _calculate_trendstrength(self, data: pd.Data_frame, **kwargs) -> pd.Data_frame:
```

### ./indicators/trend_classification.py
- 查询数量: 1

1. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

### ./indicators/trix.py
- 查询数量: 2

1. ```sql
# 验证参数
        from utils.indicator_parameter_validator import Indicator_parameter_validator
        validator = Indicator_parameter_validator()
        
        # 合并默认参数和用户参数
        params = self._d...
```

2. ```sql
: 9}
    
    def set_parameters_Trix(self, **kwargs):
        """
        设置指标参数
        
        Args:
            **kwargs: 参数字典，支持以下参数：
                - period: TRIX计算周期
                - signal_...
```

### ./indicators/unified_ma.py
- 查询数量: 1

1. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

### ./indicators/v_shaped_reversal.py
- 查询数量: 1

1. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

### ./indicators/vix.py
- 查询数量: 2

1. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

2. ```sql
: 10}
    
    def set_parameters_Vix_Vix_Vix_vix_duplicate(self, **kwargs):
        """
        设置指标参数
        
        Args:
            **kwargs: 参数字典
        """
        # 验证参数
        try:
      ...
```

### ./indicators/vol.py
- 查询数量: 2

1. ```sql
: False}
    
    def set_parameters_Vol_Vol_Vol_vol_duplicate(self, **kwargs):
        """
        设置指标参数
        
        Args:
            **kwargs: 参数字典
        """
        # 验证参数
        try:
   ...
```

2. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

### ./indicators/volume_ratio.py
- 查询数量: 1

1. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

### ./indicators/volume_score.py
- 查询数量: 1

1. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

### ./indicators/vortex.py
- 查询数量: 1

1. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

### ./indicators/vosc.py
- 查询数量: 2

1. ```sql
: 26}
    
    def set_parameters_Vosc_Vosc_Vosc_vosc_duplicate(self, **kwargs):
        """
        设置指标参数
        
        Args:
            **kwargs: 参数字典
        """
        # 验证参数
        try:
  ...
```

2. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

### ./indicators/vr.py
- 查询数量: 2

1. ```sql
try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()

            # 合并默认参数和用户参数
            params = s...
```

2. ```sql
: 6}

    def set_parameters_Vr(self, **kwargs):
        """
        设置指标参数

        Args:
            **kwargs: 参数字典
        """
        try:
            from utils.indicator_parameter_validator impo...
```

### ./indicators/wma.py
- 查询数量: 2

1. ```sql
}
        
        return pattern_info_map.get(pattern_id, default_pattern)



    def _get_default_parameters_wma(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {"period": 14, "periods"...
```

2. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

### ./indicators/wr.py
- 查询数量: 1

1. ```sql
# 验证参数
        from utils.indicator_parameter_validator import Indicator_parameter_validator
        validator = Indicator_parameter_validator()
        
        # 合并默认参数和用户参数
        params = self._d...
```

### ./indicators/zxm/diagnostics.py
- 查询数量: 10

1. ```sql
self.lookback_period = kwargs.get('lookback_period', 60)
        self.signal_threshold = kwargs.get('signal_threshold', 70)
        self.require_volume = kwargs.get('require_volume', True)

        # ...
```

2. ```sql
in indicator_name:
                        indicator_result.update(self.analyze_macd_values(indicator_data))
                    elif
```

3. ```sql
in kwargs:
            self.health_weights.update(kwargs[
```

4. ```sql
in indicator_name:
                        indicator_result.update(self.analyze_boll_values(indicator_data))
                    
                    period_result["indicators"][indicator_name] = indi...
```

5. ```sql
] = period_results
        
        # 分析ZXM多周期合成
        composite_analysis = self.analyze_composite(result)
        analysis_result.update(composite_analysis)
        
        # 分析ZXM多周期趋势一致性
       ...
```

6. ```sql
in indicator_name:
                        indicator_result.update(self.analyze_kdj_values(indicator_data))
                    elif
```

7. ```sql
in indicator_name:
                        indicator_result.update(self.analyze_dk_sg_values(indicator_data))
                    elif
```

8. ```sql
in indicator_name:
                        indicator_result.update(self.analyze_momentum_values(indicator_data))
                    elif
```

9. ```sql
: patterns
                    }
                    
                    # 分析ZXM特有的值
                    if 'ZXM_DK_SG' in indicator_name:
                        indicator_result.update(self.analyze...
```

10. ```sql
in kwargs:
            self.opportunity_weights.update(kwargs[
```

### ./indicators/zxm/market_breadth.py
- 查询数量: 2

1. ```sql
] = np.select(conditions, choices, default=
```

2. ```sql
if data.empty or len(data) < 20:
            return pd.Data_frame(index=data.index)

        result = pd.Data_frame(index=data.index)

        # 使用单股票数据模拟市场宽度指标
        try:
            # 1. 基于价格动量的简化...
```

### ./indicators/zxm/selection_model.py
- 查询数量: 27

1. ```sql
] = low_price * 0.97
                
                elif result.loc[i, "WashplateStartSelect"] or result.loc[i, "LowBuyElasticSelect"]:
                    # 洗盘后启动或低吸高弹性 - 使用近期低点作为止损
               ...
```

2. ```sql
, {}))

        # 设置选股模型自身的参数
        self.selection_threshold = kwargs.get(
```

3. ```sql
] = result["FinalSelect"] == True
        result.loc[:,
```

4. ```sql
elif result.loc[i, "WashplateStartSelect"]:
                signals.loc[i,
```

5. ```sql
] = selection_score
            
            # 10. 计算买入优先级
            # 将选中的股票按优先级排序（1-5，1为最高）
            priority = np.zeros(len(data))
            
            for i in range(len(data)):
         ...
```

6. ```sql
)
        
        # 初始化各子指标
        self.trend_detector = Trend_detector()
        self.elasticity_indicator = Elasticity()
        self.amplitude_elasticity_indicator = Amplitude_elasticity()
      ...
```

7. ```sql
]
        """初始化ZXM选股模型"""
        super().__init__(name="SelectionModel", description="ZXM选股模型，整合多个指标的选股系统")
        
        # 初始化各子指标
        self.trend_detector = Trend_detector()
        self.ela...
```

8. ```sql
] = result["FinalSelect"]
        signals.loc[:,
```

9. ```sql
] = True
        
        return signals

    def calculate_confidence_Model(self, score: pd.Series, patterns: List[str], signals: Dict[str, pd.Series]) -> float:
        """
        计算置信度

        Ar...
```

10. ```sql
.join(select_types)
                score_val = score.iloc[i]
                priority = result.loc[i,
```

11. ```sql
].iloc[i]:
                    elasticity_bonus = 7
                else:
                    elasticity_bonus = 0
                
                # 综合计算
                selection_score[i] = min(100,...
```

12. ```sql
elif result.loc[i, "LowBuyElasticSelect"]:
                signals.loc[i,
```

13. ```sql
elif result.loc[i, "PullbackBuySelect"]:
                signals.loc[i,
```

14. ```sql
] = 60  # 基础置信度
        
        # 根据选股得分和优先级调整置信度
        for i in signals.index:
            if result.loc[i, "FinalSelect"]:
                # 根据选股得分调整（最多±20）
                score_adj = (score.ilo...
```

15. ```sql
# 根据选股类型设置信号类型
        for i in signals.index:
            if result.loc[i, "StrongUptrendSelect"]:
                signals.loc[i,
```

16. ```sql
]]:
            try:
                idx = data.index.get_loc(i)
                
                if result.loc[i, "StrongUptrendSelect"]:
                    # 强趋势上涨 - 使用20日均线作为止损
                   ...
```

17. ```sql
# 为每个信号设置详细描述
        for i in signals.index:
            if result.loc[i, "FinalSelect"]:
                # 获取选股类型
                select_types = []
                
                if result.loc[i, ...
```

18. ```sql
: 0.0
        }
        
        # SelectionModel指标特定的形态信息映射
        pattern_info_map = {
            # 基础形态
```

19. ```sql
] = result["FinalSelect"] == False
        result.loc[:,
```

20. ```sql
] = result["FinalSelect"] == False

        return result
    
    def calculate_raw_score_Model(self, data: pd.Data_frame, **kwargs) -> pd.Series:
        """
        计算ZXM选股模型的原始评分
        
        ...
```

21. ```sql
] = data["close"].iloc[idx] * 0.95
                
                elif result.loc[i, "PullbackBuySelect"]:
                    # 回调买点 - 使用回调低点作为止损
                    if idx >= 10:
                 ...
```

22. ```sql
] = ma20 * 0.95
                
                elif result.loc[i, "VolumeBreakoutSelect"]:
                    # 放量突破 - 使用突破点位作为止损
                    # 简单以当前价格的5%作为止损
                    signals.lo...
```

23. ```sql
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple

from indicators.base_indicator import Base_indicator
from indicators.base.pattern_signal_mixin import...
```

24. ```sql
elif result.loc[i, "VolumeBreakoutSelect"]:
                signals.loc[i,
```

25. ```sql
] = pd.Series(0, index=data.index)
        
        
        # 添加形态识别和信号生成
        result = self.add_pattern_detection(result)
        result = self.add_signal_generation(result)

        # 重写buy_sign...
```

26. ```sql
# 设置各子指标的参数
        if hasattr(self, 'trend_detector'):
            self.trend_detector.set_parameters_Model(**kwargs.get('trend_params', {}))
        if hasattr(self, 'elasticity_indicator'):
       ...
```

27. ```sql
] = ~result["FinalSelect"]
        
        # 设置趋势
        if "TrendDirection" in result.columns:
            signals.loc[:,
```

### ./indicators/zxm_absorb.py
- 查询数量: 1

1. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

### ./indicators/zxm_washplate.py
- 查询数量: 1

1. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和...
```

### ./integration/test_integration.py
- 查询数量: 6

1. ```sql
].iloc[-1] < 30)
    
    def test_strategy_config_validation(self):
        """测试策略配置验证"""
        # 创建无效的策略配置（缺少条件）
        invalid_config = {
            "strategy": {
                "id": "INVALI...
```

2. ```sql
])
        
        # 测试导出为JSON
        results_json = json.dumps(selection_results)
        loaded_json = json.loads(results_json)
        
        # 验证JSON
        self.assert_equal(len(loaded_json)...
```

3. ```sql
def test_end_to_end_stock_selection(self):
```

4. ```sql
] = 100 - (100 / (1 + rs))
        
        # 创建测试策略配置
        self.strategy_config = {
            "strategy": {
                "id": "TEST_STRATEGY",
                "name": "测试策略",
               ...
```

5. ```sql
}
                ]
            }
        }


class Stock_selection_integration_test(Integration_test):
```

6. ```sql
: 28.0
            }
        ]
        
        # 测试转换为DataFrame
        results_df = pd.Data_frame(selection_results)
        
        # 验证DataFrame
        self.assert_equal(len(results_df), 2)
    ...
```

### ./migrate_to_unified_data_manager.py
- 查询数量: 1

1. ```sql
)
    updated_files = 0
    for file_path, replacements in migration_plan.items():
        if apply_migration(file_path, replacements):
            updated_files += 1
    
    print(f
```

### ./monitoring/performance_monitor.py
- 查询数量: 3

1. ```sql
: datetime.now().isoformat()
        }


def database_health_check() -> Dict[str, Any]:
    """数据库健康检查"""
    try:
        from db.enhanced_connection_pool import get_connection_pool
        pool = ge...
```

2. ```sql
}

        try:
            start_time = time.time()
            result = self.checks[name]()
            duration = time.time() - start_time

            result.update({
```

3. ```sql
if name not in self.checks:
            return {'status': 'error', 'message': f'检查 {name} 不存在'}

        try:
            start_time = time.time()
            result = self.checks[name]()
            ...
```

### ./run_production_validation.py
- 查询数量: 9

1. ```sql
* 选股数量: {len(volume_shrink_result['selected_stocks'])}
```

2. ```sql
average_selection_rate
```

3. ```sql
- 总选股数: {summary['total_selections']}
```

4. ```sql
)
            for i, stock in enumerate(volume_shrink_result['selected_stocks'][:5]):
                print(f
```

5. ```sql
* 选股比例: {volume_shrink_result.get('selection_rate', 0.0):.2%}
```

6. ```sql
- 平均选股率: {summary['average_selection_rate']:.2%}
```

7. ```sql
import os
import sys
import argparse
from datetime import datetime

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)

from scripts.production_indic...
```

8. ```sql
- 选股率: {result.get('selection_rate', 0.0):.2%}
```

9. ```sql
- 选股数: {len(result['selected_stocks'])}
```

### ./run_risk_detection.py
- 查询数量: 1

1. ```sql
import os
import sys

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)

from tools.automated_risk_detection import Automated_risk_detector, Risk_le...
```

### ./scripts/add_get_pattern_info_methods.py
- 查询数量: 4

1. ```sql
indicators/zxm/selection_model.py
```

2. ```sql
)
    
    # 需要添加方法的指标文件列表
    indicator_files = [
        # ZXM系列指标
        ('indicators/zxm/trend_indicators.py', ['ZXMWeeklyMACD', 'ZXMWeeklyTrendUp', 'ZXMWeeklyKDJDTrendUp', 
                     ...
```

3. ```sql
)
            return False
        
        # 在最后一个方法后插入新方法
        last_match = matches[-1]
        insert_pos = last_match.end(1)
        
        # 生成方法代码
        method_code = get_pattern_info_tem...
```

4. ```sql
matches = list(re.finditer(method_pattern, content, re.DOTALL))
        
        if not matches:
            print(f"  ❌ 无法找到合适的插入位置: {file_path}")
            return False
        
        # 在最后一个方法后...
```

### ./scripts/add_zxm_get_pattern_info.py
- 查询数量: 2

1. ```sql
)
    
    # 需要添加方法的ZXM指标类列表
    zxm_classes = [
        ('indicators/zxm/trend_indicators.py', [
            'ZXMMonthlyMACD', 'ZXMWeeklyTrendUp', 'ZXMWeeklyKDJDTrendUp', 
            'ZXMWeeklyKDJDO...
```

2. ```sql
indicators/zxm/selection_model.py
```

### ./scripts/akshare_to_clickhouse.py
- 查询数量: 4

1. ```sql
SELECT max(date) FROM stock_info
```

2. ```sql
import sys
import os
import logging
import time
import traceback
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
import efinance as ef
...
```

3. ```sql
SELECT max(date) FROM stock_info WHERE 1=1
                    WHERE code = '{stock_code}' AND level = '{level}'
```

4. ```sql
) and db.client:
                    try:
                        # 显式关闭连接
                        db.client.disconnect()
                    except Exception as e:
                        self.logger...
```

### ./scripts/analyze_selection_results.py
- 查询数量: 16

1. ```sql
* 50)
    for indicator, data in sorted(with_selections.items(), key=lambda x: x[1]['selected_count'], reverse=True):
        print(f
```

2. ```sql
\n❌ 无选股结果的指标: {len(without_selections)} 个
```

3. ```sql
)
                
                # 分析每个指标的选股结果
                for result in data['detailed_results']:
                    indicator_name = result['indicator_name']
                    selected_stoc...
```

4. ```sql
import json
import os
import glob
from collections import defaultdict
from typing import Dict, List, Any

def analyze_selection_results():
```

5. ```sql
]
                    selected_stocks = result.get(
```

6. ```sql
, False)
                    }
                    
                    # 显示选股情况
                    if stock_count > 0:
                        print(f"   ✅ {indicator_name}: 选中 {stock_count} 只股票 {se...
```

7. ```sql
, [])
                    stock_count = len(selected_stocks) if selected_stocks else 0
                    execution_time = result.get(
```

8. ```sql
股票代码: {data['selected_stocks']}
```

9. ```sql
有选股的指标: {indicators_with_selection}
```

10. ```sql
选股覆盖率: {overall_selection_rate:.1f}%
```

11. ```sql
] > 0:
            with_selections[indicator] = data
        else:
            without_selections[indicator] = data
    
    print(f"\n🎯 有选股结果的指标: {len(with_selections)} 个")
    print("-" * 50)
    fo...
```

12. ```sql
* 80)
    
    # 按选股能力分类
    with_selections = {}
    without_selections = {}
    
    for indicator, data in all_indicators.items():
        if data['selected_count'] > 0:
            with_selections...
```

13. ```sql
* 50)
    batch_stats = defaultdict(lambda: {'with_selection': 0, 'without_selection': 0, 'total': 0})
    
    for indicator, data in all_indicators.items():
        batch = data['batch']
        bat...
```

14. ```sql
)
    
    # 总体统计
    total_indicators = len(all_indicators)
    indicators_with_selection = len(with_selections)
    overall_selection_rate = (indicators_with_selection / total_indicators * 100) if t...
```

15. ```sql
无选股的指标: {len(without_selections)}
```

16. ```sql
* 50)
    for indicator, data in sorted(without_selections.items()):
        print(f
```

### ./scripts/architecture_compliance_check.py
- 查询数量: 5

1. ```sql
, line, re.IGNORECASE):
                        violations.append(Violation_info(
                            file_path=os.path.relpath(file_path, self.project_root),
                            line_...
```

2. ```sql
violations = []
        
        for file_path in files:
            try:
                if not os.path.exists(file_path):
                    continue
                    
                with open(...
```

3. ```sql
禁止使用SELECT code, name, date, level, open, close, high, low, volume
```

4. ```sql
import os
import sys
import re
import ast
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import logging

# 添加项目根目录到Python路径
root_dir = o...
```

5. ```sql
) as f:
                    lines = f.readlines()
                
                for line_num, line in enumerate(lines, 1):
                    # 检查SELECT code, name, date, level, open, close, high,...
```

### ./scripts/backtest/advanced_backtest.py
- 查询数量: 7

1. ```sql
]),
            reverse=True
        )
        
        # 选择前N个组合
        selected_combinations = sorted_combinations[:max_combinations]
        
        if not selected_combinations:
            logg...
```

2. ```sql
})
            
            # 将该组合条件添加到策略条件列表
            strategy_conditions.extend(pattern_conditions)
            
            # 如果不是最后一个组合，添加OR逻辑操作符
            if combo != selected_combinations[-...
```

3. ```sql
: pattern_id
                }
                
                pattern_conditions.append(condition)
            
            # 如果有多个条件，添加AND逻辑操作符
            if len(pattern_conditions) > 1:
         ...
```

4. ```sql
self.config.update(config)
        self.advanced_results[
```

5. ```sql
)
            
            # 过滤不满足条件的组合
            if success_rate >= min_success_rate and profit_factor >= min_profit_factor and combo_data["total_count"] >= 3:
                filtered_combinations...
```

6. ```sql
: []}}
        
        # 生成策略条件
        strategy_conditions = []
        
        for combo in selected_combinations:
            patterns = combo[
```

7. ```sql
基于{len(selected_combinations)}个高成功率形态组合的选股策略
```

### ./scripts/backtest/archive/advanced_backtest.py
- 查询数量: 7

1. ```sql
]),
            reverse=True
        )
        
        # 选择前N个组合
        selected_combinations = sorted_combinations[:max_combinations]
        
        if not selected_combinations:
            logg...
```

2. ```sql
})
            
            # 将该组合条件添加到策略条件列表
            strategy_conditions.extend(pattern_conditions)
            
            # 如果不是最后一个组合，添加OR逻辑操作符
            if combo != selected_combinations[-...
```

3. ```sql
: pattern_id
                }
                
                pattern_conditions.append(condition)
            
            # 如果有多个条件，添加AND逻辑操作符
            if len(pattern_conditions) > 1:
         ...
```

4. ```sql
self.config.update(config)
        self.advanced_results[
```

5. ```sql
)
            
            # 过滤不满足条件的组合
            if success_rate >= min_success_rate and profit_factor >= min_profit_factor and combo_data["total_count"] >= 3:
                filtered_combinations...
```

6. ```sql
: []}}
        
        # 生成策略条件
        strategy_conditions = []
        
        for combo in selected_combinations:
            patterns = combo[
```

7. ```sql
基于{len(selected_combinations)}个高成功率形态组合的选股策略
```

### ./scripts/backtest/archive/backtest_strategy_integrator.py
- 查询数量: 8

1. ```sql
)
        
        try:
            # 如果选股比例过高或过低，调整参数
            selection_ratio = validation_result.get('selection_ratio', 0)
            
            # 如果选不出股票或选太少，放宽条件
            if selection_ra...
```

2. ```sql
]
        
        # 提取对应的枚举类型
        from enums.kline_period import KlinePeriod
        period_enum = None
        for p in KlinePeriod:
            if p.name == period:
                period_enum ...
```

3. ```sql
选股比例 {validation_result.get('selection_ratio', 0):.2%}
```

4. ```sql
策略验证完成，选出 {validation_result['selected_stocks']} 只股票，
```

5. ```sql
验证结果: 选出 {validation_result.get('selected_stocks', 0)} 只股票，
```

6. ```sql
)}",
                "name": name,
                "description": description,
                "version": "1.0",
                "author": "system",
                "create_time": datetime.now().strft...
```

7. ```sql
)
                strategy_config = self._relax_strategy_conditions(strategy_config)
                
            # 如果选出太多股票，收紧条件
            elif selection_ratio > 0.1:
                logger.info(
```

8. ```sql
选股比例 {validation_result['selection_ratio']:.2%}
```

### ./scripts/backtest/archive/enhanced_backtest.py
- 查询数量: 1

1. ```sql
import sys
import os

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

import argparse
import datetime
import pa...
```

### ./scripts/backtest/archive/indicator_analysis.py
- 查询数量: 3

1. ```sql
) as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            
        logger.info(f"分析结果已保存到 {output_file}")
        return output_file


def analyze_buypoint_indicators(input_so...
```

2. ```sql
SELECT code, buy_date, pattern_type FROM {input_source}
```

3. ```sql
)

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from formula import formula
from enums.kline_period import K...
```

### ./scripts/backtest/archive/optimized_backtest.py
- 查询数量: 11

1. ```sql
import sys
import os

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

import argparse
import datetime
import pa...
```

2. ```sql
)
            
            # 初始化回测结果
            backtest_result = {
                'strategy_name': strategy.get('name', 'unnamed'),
                'start_date': start_date,
                'end_da...
```

3. ```sql
)
                        continue
                    
                    # 遍历股票池
                    for stock_code in stock_pool:
                        if stock_code in selected_stocks:
        ...
```

4. ```sql
)
            import traceback
            logger.error(traceback.format_exc())
            return {
                'error': str(e),
                'strategy_name': strategy.get('name', 'unnamed'),
...
```

5. ```sql
, {})
            
            # 获取日期列表
            start_date_obj = datetime.datetime.strptime(start_date, "%Y%m%d")
            end_date_obj = datetime.datetime.strptime(end_date, "%Y%m%d")
        ...
```

6. ```sql
))
                current_date += datetime.timedelta(days=1)
            
            # 回测每一天
            portfolio = {}  # 当前持仓
            cash = 1000000.0  # 初始资金
            daily_value = []  # 每...
```

7. ```sql
: end_date
            }
    
    def _select_stocks_by_strategy(self, strategy: Dict[str, Any], 
                                 stock_pool: List[str], 
                                 date: str) -...
```

8. ```sql
, False):
                            selected_stocks.append(stock_code)
            
            return selected_stocks
            
        except Exception as e:
            logger.error(f"选择股票时出错:...
```

9. ```sql
try:
            selected_stocks = []
            
            # 获取周期条件
            period_conditions = strategy.get('period_conditions', {})
            
            # 如果策略中直接包含条件，使用这些条件
            ...
```

10. ```sql
].append(date)
                
                # 更新持仓并模拟交易
                if selected_stocks:
                    # 处理买入信号
                    for stock_code in selected_stocks:
                    ...
```

11. ```sql
, False):
                        selected_stocks.append(stock_code)
            else:
                # 使用周期条件
                for period_value, conditions in period_conditions.items():
             ...
```

### ./scripts/backtest/archive/pattern_backtest.py
- 查询数量: 4

1. ```sql
]
        """
        # 初始化指标列表和周期列表
        self.indicators = indicators or ["MACD", "KDJ", "RSI"]
        self.periods = periods or ["DAILY"]
        
        # 创建形态识别分析器
        self.analyzer = Pat...
```

2. ```sql
].iloc[-1]
            
            # 计算涨跌幅
            if buy_price > 0:
                return (future_price - buy_price) / buy_price
            else:
                return 0
            
        ...
```

3. ```sql
)
            return 0
    
    def _update_global_pattern_stats(self, pattern_stats: Dict[str, Dict[str, Any]]) -> None:
```

4. ```sql
][stock_code] = result
        
        # 更新全局形态统计
        self._update_global_pattern_stats(pattern_stats)
        
        return result

    def _get_buy_dates(self, stock_code: str, start_date: st...
```

### ./scripts/backtest/archive/unified_backtest.py
- 查询数量: 4

1. ```sql
def zxm_select_strategy(stock_data):\n
```

2. ```sql
in p]
                
                # 生成选股策略
                f.write("### 推荐选股策略\n\n")
                f.write("根据回测结果中的共性技术形态，建议采用以下选股策略：\n\n")
                
                # 基于ZXM体系的选股策略
    ...
```

3. ```sql
][-1] < 0.04,  # 收盘价回踩至20日均线4%以内\n")
                
                f.write("    ]\n\n")
                f.write("    # 策略判断\n")
                f.write("    has_trend = any(trend_conditions)  # 至少满...
```

4. ```sql
def technical_select_strategy(stock_data):\n
```

### ./scripts/backtest/backtest_runner.py
- 查询数量: 1

1. ```sql
import os
import sys
import logging
import datetime
import argparse
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple

# 获取项目根目录
root_dir = os.path.dirname(os.path.dirname...
```

### ./scripts/backtest/consolidated_backtest.py
- 查询数量: 1

1. ```sql
import os
import sys
import logging
import datetime
import argparse
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set, Union

# 获取项目根目录
root_dir = os.path...
```

### ./scripts/backtest/data_manager.py
- 查询数量: 1

1. ```sql
import os
import sys
import logging
import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple

# 获取项目根目录
root_dir = os.path.dirname(os.path.dirn...
```

### ./scripts/backtest/pattern_analyzer.py
- 查询数量: 2

1. ```sql
import os
import sys
import logging
import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple

# 获取项目根目录
root_dir = os.path.dirname(os.path.dirn...
```

2. ```sql
] >= min_strength:
                    result.update({
```

### ./scripts/backtest/strategy_manager.py
- 查询数量: 2

1. ```sql
import os
import sys
import logging
import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

# 获取项目根目录
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__f...
```

2. ```sql
}
            
            # 执行分析
            result = self._execute_analysis_Strategy_Manager_Strategy_ManagerStrategymanager(stock_code, start_date, end_date)
            
            # 添加基本信息
     ...
```

### ./scripts/batch_indicator_validator.py
- 查询数量: 1

1. ```sql
import os
import sys
import logging
import argparse
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional

# 添加项目根目录到Python路径
root_dir = os.path.di...
```

### ./scripts/batch_test_phase2_trend_indicators.py
- 查询数量: 8

1. ```sql
try:
            result = self.data_access.execute_query('SELECT MAX(date) as max_date FROM stock_info WHERE date >= '2020-01-01' LIMIT 1')
            if not result.empty:
                return str(...
```

2. ```sql
query = f"""
            SELECT DISTINCT code 
            FROM {table_name} 
            WHERE date =
```

3. ```sql
SELECT MAX(date) as max_date FROM stock_info WHERE date >=
```

4. ```sql
SELECT DISTINCT code 
            FROM {table_name} 
            WHERE date = '{self.latest_date}'
              AND volume > 0 
              AND close > 0
            ORDER BY volume DESC 
         ...
```

5. ```sql
SELECT code, date, open, high, low, close, volume
            FROM {table_name}
            WHERE code IN ('{codes_str}')
              AND date <= '{self.latest_date}'
            ORDER BY code, date...
```

6. ```sql
import sys
import os
import time
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
root_dir = os.path.dir...
```

7. ```sql
SELECT MAX(date) as max_date FROM stock_info
```

8. ```sql
".join(stock_codes)
            
            query = f"""
            SELECT code, date, open, high, low, close, volume
            FROM {table_name}
            WHERE code IN (
```

### ./scripts/batch_test_phase2_trend_indicators_fixed.py
- 查询数量: 8

1. ```sql
import sys
import os
import time
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
root_dir = os.path.dir...
```

2. ```sql
try:
            result = self.data_access.execute_query('SELECT MAX(date) as max_date FROM stock_info WHERE date >= '2020-01-01' LIMIT 1')
            if not result.empty:
                return str(...
```

3. ```sql
query = f"""
            SELECT DISTINCT code 
            FROM {table_name} 
            WHERE date =
```

4. ```sql
SELECT MAX(date) as max_date FROM stock_info WHERE date >=
```

5. ```sql
SELECT DISTINCT code 
            FROM {table_name} 
            WHERE date = '{self.latest_date}'
              AND volume > 0 
              AND close > 0
            ORDER BY volume DESC 
         ...
```

6. ```sql
SELECT code, date, open, high, low, close, volume
            FROM {table_name}
            WHERE code IN ('{codes_str}')
              AND date <= '{self.latest_date}'
            ORDER BY code, date...
```

7. ```sql
SELECT MAX(date) as max_date FROM stock_info
```

8. ```sql
".join(stock_codes)
            
            query = f"""
            SELECT code, date, open, high, low, close, volume
            FROM {table_name}
            WHERE code IN (
```

### ./scripts/benchmark_unified_indicator_engine.py
- 查询数量: 1

1. ```sql
import os
import sys
import time
import pandas as pd
import numpy as np

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from analysis.engines.unif...
```

### ./scripts/check_database_data.py
- 查询数量: 11

1. ```sql
]):
                sample_query = f"SELECT code, name, date, level, open, close, high, low, volume FROM {table_name} LIMIT 5"
                try:
                    sample = data_access.query_dataf...
```

2. ```sql
SELECT COUNT(*) FROM {table_name}
```

3. ```sql
import os
import sys

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from utils.dependency_injection import get_service

def check_da...
```

4. ```sql
ORDER BY date DESC LIMIT 3"
                    sample = data_access.query_dataframe(sample_query)
                    print(f"样本数据: {sample}")
                    
            except Exception as e:
...
```

5. ```sql
SELECT code, name, date, level, open, close, high, low, volume FROM {table_name} WHERE code = '000001' ORDER BY date DESC LIMIT 3
```

6. ```sql
LIMIT 1"
            try:
                result = data_access.query_dataframe(check_query)
                print(f"表 {table_name} 中000001的数据量: {result}")
                
                # 如果有数据，查看样本...
```

7. ```sql
SELECT COUNT(*) FROM {table_name} WHERE code = '000001' LIMIT 1
```

8. ```sql
SELECT code, name, date, level, open, close, high, low, volume FROM {table_name} LIMIT 5
```

9. ```sql
SELECT MIN(date) as min_date, MAX(date) as max_date FROM {table_name} LIMIT 1
```

10. ```sql
]
        
        for table_name in possible_tables:
            print(f"\n尝试查询表: {table_name}")
            
            # 检查表是否存在
            check_query = f"SELECT COUNT(*) FROM {table_name} WHERE...
```

11. ```sql
SELECT MIN({date_field}) as min_date, MAX({date_field}) as max_date FROM {table_name} LIMIT 1
```

### ./scripts/check_stock_info_columns.py
- 查询数量: 11

1. ```sql
ORDER BY date DESC LIMIT 3"
        sample_data = data_access.query_dataframe(sample_query)
        print("样本数据:")
        print(sample_data)
        
        # 检查特定股票的数据量
        print("\n=== 数据量检查 =...
```

2. ```sql
) GROUP BY code"
        count_result = data_access.query_dataframe(count_query)
        print("数据量:")
        print(count_result)
        
        # 检查日期范围
        print("\n=== 日期范围检查 ===")
        d...
```

3. ```sql
SELECT 
            code,
            MIN(date) as min_date,
            MAX(date) as max_date,
            COUNT(*) as total_records
        FROM stock_info
```

4. ```sql
SELECT code, COUNT(*) as count FROM stock_info WHERE code IN ('000001', '000002') GROUP BY code
```

5. ```sql
SELECT code, COUNT(*) as count FROM stock_info
```

6. ```sql
SELECT code, name, date, level, open, close, high, low, volume FROM stock_info WHERE date >= '2020-01-01' LIMIT 1
```

7. ```sql
):
            print(list(sample_result.columns))
        else:
            print("无法获取列名")
        
        # 查看样本数据
        print("\n=== 样本数据 ===")
        sample_query = "SELECT code, name, date, l...
```

8. ```sql
SELECT code, name, date, level, open, close, high, low, volume FROM stock_info
```

9. ```sql
SELECT code, name, date, level, open, close, high, low, volume FROM stock_info WHERE code = '000001' ORDER BY date DESC LIMIT 3
```

10. ```sql
import os
import sys

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from utils.dependency_injection import get_service

def ma...
```

11. ```sql
SELECT 
            code,
            MIN(date) as min_date,
            MAX(date) as max_date,
            COUNT(*) as total_records
        FROM stock_info WHERE 1=1
        WHERE code = '000001'
  ...
```

### ./scripts/check_strategy_config.py
- 查询数量: 1

1. ```sql
import os
import sys
import json

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from analysis.engines.indicator_validation_framework...
```

### ./scripts/check_unregistered_indicators.py
- 查询数量: 2

1. ```sql
indicators.zxm.selection_model
```

2. ```sql
indicators = {
        'core': [
            # 基础技术指标 (37个)
            ('indicators.ma', 'MA'),
            ('indicators.ema', 'EMA'),
            ('indicators.wma', 'WMA'),
            ('indicators....
```

### ./scripts/clickhouse_connection_summary.py
- 查询数量: 5

1. ```sql
SELECT COUNT(*) FROM stock LIMIT 1000.{table_name}
```

2. ```sql
SELECT COUNT(*) FROM stock
```

3. ```sql
in [db[0] for db in databases]:
            print("   ✓ stock数据库存在")
            
            print("\n5. 获取stock数据库中的表...")
            tables = client.execute("SHOW TABLES FROM stock LIMIT 1000")
  ...
```

4. ```sql
SELECT 1 AS test")
        print(f"   ✓ 查询成功: {result}")
        
        print("\n3. 获取数据库列表...")
        databases = client.execute("SHOW DATABASES")
        print("   ✓ 数据库列表:")
        for db in d...
```

5. ```sql
}
    
    try:
        print("\n1. 尝试连接到ClickHouse...")
        client = Client(**config)
        print("   ✓ 连接成功！")
        
        print("\n2. 测试基本查询...")
        result = client.execute("SELECT ...
```

### ./scripts/compare_analysis_engines.py
- 查询数量: 1

1. ```sql
import os
import sys
import pandas as pd
from datetime import datetime

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from ana...
```

### ./scripts/complete_validation_workflow.py
- 查询数量: 1

1. ```sql
import sys
import os
import subprocess
import argparse
import json
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, s...
```

### ./scripts/comprehensive_indicator_analysis.py
- 查询数量: 4

1. ```sql
indicators.zxm.selection_model
```

2. ```sql
indicators = {}
        
        # 第一部分：核心技术指标 (37个)
        core_indicators = [
            ('MACD', 'indicators.macd', 'MACD', 'MACD指标'),
            ('KDJ', 'indicators.kdj', 'KDJ', 'KDJ随机指标'),
   ...
```

3. ```sql
] += 1

        # 更新分析结果
        self.analysis_results.update({
```

4. ```sql
)

        # 检查每个指标的可用性和注册状态
        available_count = 0
        registered_count = 0
        category_stats = {}

        for indicator in self.indicators.values():
            # 检查可用性
            in...
```

### ./scripts/comprehensive_indicator_tester.py
- 查询数量: 17

1. ```sql
)
                raise
            
            return Test_result_Tester(
                indicator_name=indicator_name,
                success=False,
                execution_time=execution_time,...
```

2. ```sql
# 基础指标（快速验证）
        basic_indicators = [
            'ma', 'ema', 'wma', 'rsi', 'macd', 'boll', 'kdj', 'vol',
            'momentum', 'mtm', 'roc', 'bias', 'dma'
        ]
        
        # 趋势指标
   ...
```

3. ```sql
)
            
            return Test_result_Tester(
                indicator_name=indicator_name,
                success=True,
                execution_time=execution_time,
                stock_...
```

4. ```sql
, 0.0)
            
            logger.info(f"✅ {indicator_name}: {selected_stocks}/{stock_count} 只股票被选中 ({selection_rate:.1f}%), 耗时 {execution_time:.2f}秒")
            
            return Test_result...
```

5. ```sql
)
            
            # 提取关键指标
            stock_count = result.get('stock_count', 0)
            selected_stocks = result.get('selected_stocks', 0)
            selection_rate = result.get('selec...
```

6. ```sql
)
    
    def run_comprehensive_test_Tester(self, selected_batches: List[str] = None, 
                             concurrent: bool = False, test_date: str = None) -> Dict[str, Any]:
```

7. ```sql
)
        
        overall_start_time = time.time()
        
        # 确定要测试的批次
        batches_to_test = selected_batches if selected_batches else list(self.batches.keys())
        
        total_ind...
```

8. ```sql
)
    
    args = parser.parse_args()
    
    try:
        # 初始化测试器
        tester = Comprehensive_indicator_tester(
            max_stocks=args.max_stocks,
            max_workers=args.max_workers
 ...
```

9. ```sql
- {tr['selected_stocks']}/{tr['stock_count']} 只股票
```

10. ```sql
indicator_name: str
    success: bool
    execution_time: float
    stock_count: int
    selected_stocks: int
    error_message: Optional[str] = None
    performance_metrics: Optional[Dict] = None

@d...
```

11. ```sql
)
            return 0
        
        # 运行全面测试
        result = tester.run_comprehensive_test_Tester(
            selected_batches=args.batches,
            concurrent=args.concurrent,
            t...
```

12. ```sql
, 0)
            selected_stocks = result.get(
```

13. ```sql
.join(failed_indicators)}")
    
    def run_comprehensive_test_Tester(self, selected_batches: List[str] = None, 
                             concurrent: bool = False, test_date: str = None) -> Dict[...
```

14. ```sql
, 0)
            selection_rate = result.get(
```

15. ```sql
: execution_time / max(stock_count, 1)
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
...
```

16. ```sql
*100}")
        
        overall_start_time = time.time()
        
        # 确定要测试的批次
        batches_to_test = selected_batches if selected_batches else list(self.batches.keys())
        
        tot...
```

17. ```sql
)
            if self.stop_on_error:
                raise
        
        overall_time = time.time() - overall_start_time
        
        # 生成综合报告
        comprehensive_result = {
            'test...
```

### ./scripts/comprehensive_unified_engine_test.py
- 查询数量: 22

1. ```sql
categories = {
            'core_indicators': [],
            'enhanced_indicators': [],
            'zxm_indicators': [],
            'composite_indicators': [],
            'pattern_indicators': [],...
```

2. ```sql
selection_count_stats
```

3. ```sql
平均选股数: {sel_stats['selection_count_stats']['avg_selected']}
```

4. ```sql
duration = (
            self.test_stats['end_time'] - self.test_stats['start_time']
        ).total_seconds() if self.test_stats['end_time'] and self.test_stats['start_time'] else 0
        
        ...
```

5. ```sql
) if len(results) > 0 else []
                })
                
                logger.info(f"✅ {indicator_name}: 选出 {len(results)} 只股票，耗时 {execution_time:.2f}秒")
            else:
                #...
```

6. ```sql
({stats['success_rate']}%) 平均选股: {stats['avg_selected_count']}
```

7. ```sql
SELECT DISTINCT code as stock_code
            FROM stock_info
```

8. ```sql
all_results = list(self.test_stats['test_results'].values())
        successful_results = [r for r in all_results if r.get('success', False)]
        
        if not successful_results:
            re...
```

9. ```sql
: {}
        }
        
        # 获取最新交易日期 - 使用数据库中实际存在的日期
        # self.test_date = get_latest_trading_date()
        self.test_date = "2025-05-23"  # 使用数据库中实际存在的最新日期
        logger.info(f"测试日期: {se...
```

10. ```sql
SELECT DISTINCT code as stock_code
            FROM stock_info WHERE 1=1
            WHERE date = '{self.test_date}'
            AND level = '日线'
            AND close > 2.0 
            AND close < 2...
```

11. ```sql
})
                
                logger.warning(f"⚠️ {indicator_name}: 未选出股票，耗时 {execution_time:.2f}秒")
            
        except Exception as e:
            error_msg = f"策略执行失败: {str(e)}"
     ...
```

12. ```sql
strategy_id = strategy['strategy_id']
        indicator_name = strategy_id.replace('test_', '').upper()
        
        test_result = {
            'strategy_id': strategy_id,
            'indicator_...
```

13. ```sql
}
        
        # 选股数量统计
        selection_counts = [r.get(
```

14. ```sql
总选股数: {sel_stats['selection_count_stats']['total_selected']}
```

15. ```sql
零选股测试: {sel_stats['selection_count_stats']['zero_selection_count']}
```

16. ```sql
)
            else:
                # 没有选出股票，但不一定是错误
                test_result.update({
                    'success': True,  # 执行成功，只是没有符合条件的股票
                    'selected_count': 0,
            ...
```

17. ```sql
] > 0 else 0
        )
        
        # 分类统计
        category_stats = self._analyze_by_category()
        
        # 选股效果统计
        selection_stats = self._analyze_selection_effectiveness()
        ...
```

18. ```sql
, False))
                total = len(results)
                # 安全地获取selected_count，如果不存在则默认为0
                avg_selected = sum(r.get(
```

19. ```sql
import sys
import os
import time
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import json
from concurrent.futures i...
```

20. ```sql
: []
        }
        
        try:
            start_time = time.time()
            
            # 执行策略选股
            results = self.executor.execute_strategy_optimized(
                strategy_pla...
```

21. ```sql
selection_effectiveness
```

22. ```sql
test_result.update({
                'success': False,
                'error_message': error_msg,
                'execution_time': time.time() - start_time if 'start_time' in locals() else 0
       ...
```

### ./scripts/database_optimization.py
- 查询数量: 21

1. ```sql
SELECT code, name, date, level, open, close, high, low, volume FROM system
```

2. ```sql
})
            
            # 2. 分析分区策略
            logger.info("分析分区策略...")
            
            # 检查是否需要分区
            date_range = self.client.query("SELECT MIN(date) as min_date, MAX(date) as ...
```

3. ```sql
SELECT code, date, close, volume FROM stock_info WHERE close > 20 AND volume > 1000000 AND date >= '2024-05-01' LIMIT 500
```

4. ```sql
: "SELECT code, COUNT(*) as cnt FROM stock_info WHERE date >=
```

5. ```sql
)
            table_size = self.client.query(f"""
                SELECT 
                    format_readable_size(sum(bytes)) as size,
                    sum(rows) as rows
                FROM syste...
```

6. ```sql
SELECT code, date, close FROM stock_info
```

7. ```sql
] = f"Error checking indexes: {e}"
            
            # 3. 分析数据分布
            logger.info("分析数据分布...")
            data_stats = self.client.query("""
                SELECT 
                    ...
```

8. ```sql
: "SELECT code, date, close, volume FROM stock_info WHERE close > 20 AND volume > 1000000 AND date >=
```

9. ```sql
SELECT MIN(date) as min_date, MAX(date) as max_date FROM stock_info
```

10. ```sql
SELECT 
                    format_readable_size(sum(bytes)) as size,
                    sum(rows) as rows
                FROM system.parts 
                WHERE table = 'stock_info' AND database =...
```

11. ```sql
SELECT code, COUNT(*) as cnt FROM stock_info
```

12. ```sql
SELECT 
                    COUNT(*) as total_records,
                    COUNT(DISTINCT code) as unique_stocks,
                    MIN(date) as earliest_date,
                    MAX(date) as lates...
```

13. ```sql
SELECT 
                    format_readable_size(sum(bytes)) as size,
                    sum(rows) as rows
                FROM system
```

14. ```sql
SELECT code, COUNT(*) as cnt FROM stock_info WHERE date >= '2024-01-01' GROUP BY code LIMIT 100
```

15. ```sql
SELECT 
                    COUNT(*) as total_records,
                    COUNT(DISTINCT code) as unique_stocks,
                    MIN(date) as earliest_date,
                    MAX(date) as lates...
```

16. ```sql
SELECT code, name, date, level, open, close, high, low, volume FROM system.tables WHERE name = 'stock_info' AND database = '{db_name}'
```

17. ```sql
)
            table_settings = self.client.query(f"SELECT code, name, date, level, open, close, high, low, volume FROM system.tables WHERE name =
```

18. ```sql
SELECT MIN(date) as min_date, MAX(date) as max_date FROM stock_info WHERE date >= '2020-01-01'
```

19. ```sql
SELECT code, date, close, volume FROM stock_info
```

20. ```sql
queries = [
            {
                'name': 'simple_select',
                'query':
```

21. ```sql
: "SELECT code, date, close FROM stock_info WHERE date >=
```

### ./scripts/enhanced_batch_indicator_validator.py
- 查询数量: 8

1. ```sql
".join(stock_codes)
            sql = f"""
            SELECT code, name, date, level, open, close, high, low, volume
            FROM {period_config.table_name}
            WHERE code IN (
```

2. ```sql
import sys
import os
import time
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
root_dir = os.p...
```

3. ```sql
SELECT MAX(date) as max_date FROM stock_info WHERE date >=
```

4. ```sql
SELECT code, name, date, level, open, close, high, low, volume
            FROM {period_config.table_name}
            WHERE code IN ('{codes_str}')
              AND date >= '{start_date_str}'
      ...
```

5. ```sql
try:
            result = self.data_access.query('SELECT MAX(date) as max_date FROM stock_info WHERE date >= '2020-01-01' LIMIT 1')
            if not result.empty:
                return str(result.i...
```

6. ```sql
SELECT MAX(date) as max_date FROM stock_info
```

7. ```sql
)
            
            sql = f"""
            SELECT DISTINCT code
            FROM {period_config.table_name}
            WHERE date = %(latest_date)s
              AND name NOT LIKE
```

8. ```sql
SELECT DISTINCT code
            FROM {period_config.table_name}
            WHERE date = %(latest_date)s
              AND name NOT LIKE '%ST%'
              AND volume > 100000
            ORDER BY ...
```

### ./scripts/execute_zxm_absorb_volume_strategy.py
- 查询数量: 1

1. ```sql
import os
import sys
import time
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.di...
```

### ./scripts/expanded_stock_pool_tester.py
- 查询数量: 17

1. ```sql
✅ 成功选出 {selected_count} 只股票: {result.get('selected_stocks', [])}
```

2. ```sql
待测试指标: {len(self.no_selection_indicators)}个
```

3. ```sql
]
    
    def test_expanded_pool(self):
        """使用扩大的股票池测试无选股指标"""
        print("🚀 开始扩大股票池测试")
        print("=" * 80)
        print(f"原股票池: 5只股票")
        print(f"扩大股票池: {len(self.expanded_stock...
```

4. ```sql
仍无选股能力: {len(still_no_selection)} 个
```

5. ```sql
, [])}")
                else:
                    still_no_selection.append(indicator_name)
                    print(f"   ❌ 仍无选股结果")
                    
            except Exception as e:
         ...
```

6. ```sql
: str(e)
                })
        
        # 生成总结报告
        print(f"\n" + "=" * 80)
        print(f"📊 扩大股票池测试结果总结")
        print(f"=" * 80)
        
        improvement_rate = (len(improved_indicat...
```

7. ```sql
import os
import sys
import json
import time
from datetime import datetime
from typing import List, Dict, Any

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))...
```

8. ```sql
测试指标总数: {len(self.no_selection_indicators)}
```

9. ```sql
* 80)
        
        improvement_rate = (len(improved_indicators) / len(self.no_selection_indicators)) * 100
        
        print(f
```

10. ```sql
{indicator}: {result['selected_count']} 只股票
```

11. ```sql
)
        
        return results, improved_indicators, still_no_selection

if __name__ ==
```

12. ```sql
: len(self.no_selection_indicators),
```

13. ```sql
: []
        }
        
        improved_indicators = []
        still_no_selection = []
        
        for i, indicator_name in enumerate(self.no_selection_indicators, 1):
            print(f"\n🔍 测...
```

14. ```sql
].append(indicator_result)
                
                if selected_count > 0:
                    improved_indicators.append(indicator_name)
                    print(f"   ✅ 成功选出 {selected_count}...
```

15. ```sql
\n🔍 测试指标 {i}/{len(self.no_selection_indicators)}: {indicator_name}
```

16. ```sql
)
                still_no_selection.append(indicator_name)
                
                results['results'].append({
                    'indicator_name': indicator_name,
                    'succ...
```

17. ```sql
\n❌ 仍无选股能力的指标 ({len(still_no_selection)}个):
```

### ./scripts/final_system_verification.py
- 查询数量: 1

1. ```sql
)
    
    # 所有应该已注册的指标
    all_indicators = {
        # 第一批：核心指标 (23个)
        'core': [
            'AD', 'ADX', 'AROON', 'ATR', 'EMA', 'KC', 'MA', 'MFI', 'MOMENTUM', 'MTM',
            'OBV', 'PSY'...
```

### ./scripts/fix_ma_indicator_validation.py
- 查询数量: 1

1. ```sql
import os
import sys
import pandas as pd
from datetime import datetime, timedelta

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

fro...
```

### ./scripts/fix_missing_get_pattern_info.py
- 查询数量: 6

1. ```sql
)
            return False
        
        # 插入get_pattern_info方法
        method_lines = get_pattern_info_template_Info().split('\n')
        lines[insert_position:insert_position] = method_lines
   ...
```

2. ```sql
):
                # 找到最后一个非空非注释行
                insert_position = i + 1
                break
        
        if insert_position == -1:
            logger.warning(f
```

3. ```sql
)
        insert_position = -1
        
        # 寻找最后一个方法定义的位置
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i].strip()
            if line and not line.startswith(
```

4. ```sql
)
            return False
        
        # 找到类的结束位置（最后一个方法后）
        lines = content.split('\n')
        insert_position = -1
        
        # 寻找最后一个方法定义的位置
        for i in range(len(lines) - 1,...
```

5. ```sql
)
        lines[insert_position:insert_position] = method_lines
        
        # 写回文件
        with open(file_path,
```

6. ```sql
"):
                # 找到最后一个非空非注释行
                insert_position = i + 1
                break
        
        if insert_position == -1:
            logger.warning(f"无法确定插入位置: {file_path}")
       ...
```

### ./scripts/fix_pandas_warnings.py
- 查询数量: 2

1. ```sql
)
    
    # 需要修复的文件列表
    files_to_fix = [
        'indicators/zxm/trend_indicators.py',
        'indicators/zxm/buy_point_indicators.py',
        'indicators/zxm/elasticity_indicators.py',
        '...
```

2. ```sql
indicators/zxm/selection_model.py
```

### ./scripts/generate_closed_loop_summary.py
- 查询数量: 11

1. ```sql
# 选股数量
        selection_count = result.get('step1_strategy_selection', {}).get('selection_count', 0)
        
        # 验证详情
        verification_result = result.get('step2_indicator_verification', {...
```

2. ```sql
| {indicator_name} | {status} | {consistency_str} | {quality_str} | {selection_count} | {detail_str} |
```

3. ```sql
- 选股比率: {step1.get('selection_ratio', 0):.1%}
```

4. ```sql
, 0)
        detail_str = f"{verified_stocks}/{total_stocks}"
        
        report_lines.append(
            f"| {indicator_name} | {status} | {consistency_str} | {quality_str} | {selection_count} ...
```

5. ```sql
- 选出股票数: {step1.get('selection_count', 0)}
```

6. ```sql
- 选出股票: {', '.join(step1.get('selected_stocks', []))}
```

7. ```sql
summary = {
        'metadata': {
            'report_type': 'true_closed_loop_validation_summary',
            'generation_time': datetime.now().isoformat(),
            'total_indicators_tested': su...
```

8. ```sql
, 0)
        quality_str = f"{quality_score:.2f}"
        
        # 选股数量
        selection_count = result.get(
```

9. ```sql
step1_strategy_selection
```

10. ```sql
import os
import sys
import json
import glob
from datetime import datetime
from typing import Dict, List, Any

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))...
```

11. ```sql
])
            
            # 策略选股结果
            step1 = result.get('step1_strategy_selection', {})
            strategy_config = step1.get('strategy_config', {})
            conditions = strategy_con...
```

### ./scripts/historical_signal_validator.py
- 查询数量: 29

1. ```sql
)
                all_results[indicator_name] = {
                    'indicator_name': indicator_name,
                    'status': 'failed',
                    'error': str(e)
                }
  ...
```

2. ```sql
改善率: {len(improved_indicators)/len(no_selection_indicators)*100:.1f}%
```

3. ```sql
# 无选股的37个指标
        no_selection_indicators = [
            # 第1批-基础指标
            'macd', 'boll', 'kdj',
            # 第2批-趋势指标
            'sar',
            # 第3批-成交量指标
            'emv',
         ...
```

4. ```sql
)
                
                # 获取股票数据（增加历史数据）
                stock_data = self.get_stock_data(stock_code, test_date, days=100 + history_days)
                
                if stock_data.empt...
```

5. ```sql
)
        
        start_time = time.time()
        
        # 获取股票池
        stock_pool = self.get_stock_pool(test_date, max_stocks)
        if not stock_pool:
            return {
                'in...
```

6. ```sql
import os
import sys
import pandas as pd
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirn...
```

7. ```sql
测试指标总数: {len(no_selection_indicators)}
```

8. ```sql
仍无选股能力: {len(still_no_selection)} 个
```

9. ```sql
}
        
        # 获取指标实例
        indicator = self.available_indicators[indicator_name]
        
        # 验证结果
        selected_stocks = []
        failed_stocks = []
        indicator_values = {}
...
```

10. ```sql
)
        
        return {
            'improved_indicators': improved_indicators,
            'still_no_selection': still_no_selection,
            'all_results': all_results
        }

if __name__ ...
```

11. ```sql
: str(e)
                }
                still_no_selection.append(indicator_name)
        
        # 生成总结报告
        print(f"\n📊 历史信号测试总结")
        print("=" * 80)
        print(f"🎯 总体改善情况:")
      ...
```

12. ```sql
]
                    print(f"   ✅ 改善成功！选出 {selected_count} 只股票，总信号数 {total_signals}")
                    improved_indicators.append(indicator_name)
                else:
                    print(f"...
```

13. ```sql
)
            
            try:
                result = self.validate_single_indicator_historical(
                    indicator_name, 
                    history_days=history_days,
                ...
```

14. ```sql
: len(no_selection_indicators),
```

15. ```sql
📋 测试指标数量: {len(no_selection_indicators)}
```

16. ```sql
* 80)
        
        all_results = {}
        improved_indicators = []
        still_no_selection = []
        
        for i, indicator_name in enumerate(no_selection_indicators, 1):
            pr...
```

17. ```sql
os.makedirs(os.path.dirname(result_file), exist_ok=True)
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'test_type': 'historical_signal_validation'...
```

18. ```sql
]
            }
            for code, data in signal_stocks[:top_n]
        ]
    
    def test_no_selection_indicators(self, history_days: int = 30) -> Dict[str, Any]:
        """测试之前无选股能力的37个指标"""
 ...
```

19. ```sql
)
                failed_stocks.append(stock_code)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 计算统计结果
        total_tested = len(stock_po...
```

20. ```sql
]
        
        print(f"🔬 开始历史信号测试")
        print(f"📋 测试指标数量: {len(no_selection_indicators)}")
        print(f"📈 历史检查天数: {history_days}")
        print("=" * 80)
        
        all_results = {}
...
```

21. ```sql
: len(improved_indicators)/len(no_selection_indicators)
                },
```

22. ```sql
]}个信号")
        
        if still_no_selection:
            print(f"\n❌ 仍无选股能力的指标 ({len(still_no_selection)}个):")
            for indicator in still_no_selection:
                print(f"   {indicator...
```

23. ```sql
].iloc[-1]))
                }
                
            except Exception as e:
                self.logger.warning(f"⚠️ 处理股票 {stock_code} 失败: {e}")
                failed_stocks.append(stock_code)...
```

24. ```sql
)
                    still_no_selection.append(indicator_name)
                    
            except Exception as e:
                print(f
```

25. ```sql
{indicator}: {stats['selected_count']}只股票, {stats['total_signals']}个信号
```

26. ```sql
📊 选股结果: {selected_count}/{success_count} = {selection_rate:.2%}
```

27. ```sql
\n❌ 仍无选股能力的指标 ({len(still_no_selection)}个):
```

28. ```sql
signal_stocks = [(code, data) for code, data in indicator_values.items() if data['signal_count'] > 0]
        
        # 按信号数量排序
        signal_stocks.sort(key=lambda x: x[1]['signal_count'], reverse=...
```

29. ```sql
: self._get_top_signals_historical(indicator_values, 10)
            }
        }
        
        self.logger.info(f"✅ 指标 {indicator_name} 历史信号验证完成")
        self.logger.info(f"📊 选股结果: {selected_count...
```

### ./scripts/identify_missing_indicators.py
- 查询数量: 3

1. ```sql
indicators.zxm.selection_model
```

2. ```sql
# 基于之前的工作，这些是已注册的指标
    return {
        # 已存在的基础指标
        'MACD', 'RSI', 'KDJ', 'BIAS', 'CCI', 'EMV', 'ICHIMOKU', 'CMO', 'DMA',
        'Volume', 'BOLL', 'EnhancedKDJ', 'EnhancedMACD', 'EnhancedTRIX...
```

3. ```sql
return {
        # 核心指标 (23个)
        'core': [
            ('indicators.ad', 'AD', 'AD'),
            ('indicators.adx', 'ADX', 'ADX'),
            ('indicators.aroon', 'Aroon', 'AROON'),
           ...
```

### ./scripts/indicator_closed_loop_validator.py
- 查询数量: 35

1. ```sql
] = selected_stocks
            
            # 步骤3: 指标验证分析
            if selected_stocks:
                logger.info(f"🔬 步骤3: 对选出的 {len(selected_stocks)} 只股票进行指标验证")
                indicator_verifi...
```

2. ```sql
)
            return None
    
    def _verify_closed_loop(self, indicator_name: str, selected_stocks: List[str], 
                          indicator_verification: Dict[str, Any]) -> bool:
```

3. ```sql
]
            selection_ratio = selection_count / stock_pool_size
            if selection_ratio > max_selection_ratio:
                logger.info(f"选股比例过高: {selection_ratio:.2%}")
                re...
```

4. ```sql
验证指标闭环一致性
        
        Args:
            indicator_name: 指标名称
            selected_stocks: 选出的股票
            indicator_verification: 指标验证结果
            
        Returns:
            是否通过闭环验证
```

5. ```sql
)
                return macd_dif is not None and macd_dea is not None
            
            else:
                # 通用验证：检查主要指标列是否有有效值
                for col in indicator_data.columns:
          ...
```

6. ```sql
verification_result = {
            'total_stocks': len(selected_stocks),
            'analyzed_stocks': 0,
            'stocks_with_valid_indicator': 0,
            'indicator_consistency_rate': 0.0,...
```

7. ```sql
, 0)
            if 0.01 <= selection_ratio <= 0.3:  # 合理的选股比例
                score += 0.3
            elif selection_ratio > 0:
                score += 0.1
            
            # 指标验证分数
       ...
```

8. ```sql
indicators_with_selections
```

9. ```sql
)
            selected_stocks = self._execute_strategy_selection_Indicator_Closed_Loop_Validator(strategy_config, stock_pool)
            
            result['selection_count'] = len(selected_stocks)
...
```

10. ```sql
] = indicator_verification
                
                # 步骤4: 闭环验证
                logger.info(f"🔄 步骤4: 验证指标闭环一致性")
                closed_loop_verified = self._verify_closed_loop(
              ...
```

11. ```sql
.join(failed_indicators[:5])}")
        
        no_selection_indicators = [name for name, result in self.validation_results.items() if result[
```

12. ```sql
try:
            # 闭环验证标准
            min_selection_count = self.config['validation']['min_selection_count']
            max_selection_ratio = self.config['validation']['max_selection_ratio']
        ...
```

13. ```sql
)
            
            try:
                result = self._validate_single_indicator_closed_loop(
                    indicator_name, stock_pool
                )
                self.validation_r...
```

14. ```sql
}
            ])
        
        return conditions
    
    def _execute_strategy_selection_Indicator_Closed_Loop_Validator(self, strategy_config: Dict[str, Any], stock_pool: List[str]) -> List[str]:...
```

15. ```sql
] = len(selected_stocks) / len(stock_pool) if stock_pool else 0
            result[
```

16. ```sql
)
        
        # 验证结果存储
        self.validation_results = {}
        self.validation_stats = {
            'total_indicators': 0,
            'successful_validations': 0,
            'failed_valid...
```

17. ```sql
: len(selected_stocks),
```

18. ```sql
default_config = {
            'validation': {
                'date': get_latest_trading_date(),
                'lookback_days': 30,
                'stock_pool_size': 100,
                'max_sele...
```

19. ```sql
] = len(selected_stocks)
            result[
```

20. ```sql
try:
            csv_data = []
            for indicator_name, result in self.validation_results.items():
                csv_data.append({
                    'indicator_name': indicator_name,
      ...
```

21. ```sql
start_time = time.time()
        result = {
            'indicator_name': indicator_name,
            'timestamp': datetime.now().isoformat(),
            'status': 'unknown',
            'selection_c...
```

22. ```sql
)
        
        no_selection_indicators = [name for name, result in self.validation_results.items() if result['status'] == 'no_selection']
        if no_selection_indicators:
            recommenda...
```

23. ```sql
对选出的股票进行指标验证分析
        
        Args:
            selected_stocks: 选出的股票列表
            indicator_name: 指标名称
            
        Returns:
            指标验证结果
```

24. ```sql
)
                return False
            
            # 检查2: 选股比例合理
            stock_pool_size = self.config['validation']['stock_pool_size']
            selection_ratio = selection_count / stock_p...
```

25. ```sql
try:
            score = 0.0
            
            # 基础分数：能够执行选股
            if result['status'] == 'success':
                score += 0.4
            
            # 选股效果分数
            selection_r...
```

26. ```sql
import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import time
import traceback

# 添加项...
```

27. ```sql
:
                score += 0.4
            
            # 选股效果分数
            selection_ratio = result.get(
```

28. ```sql
)
                closed_loop_verified = self._verify_closed_loop(
                    indicator_name, selected_stocks, indicator_verification
                )
                result['closed_loop_ver...
```

29. ```sql
.join(no_selection_indicators[:5])}")
        
        return recommendations
    
    def _save_validation_results(self, report: Dict[str, Any]):
        """保存验证结果"""
        try:
            # 确保输出目...
```

30. ```sql
conditions = []
        
        # 根据指标类型生成不同的条件
        if 'RSI' in indicator_name.upper():
            # RSI指标：使用价格条件替代复杂的指标条件
            conditions.extend([
                {
                    '...
```

31. ```sql
].tolist()
                else:
                    logger.warning(f"结果DataFrame中未找到股票代码列，可用列: {result_df.columns.tolist()}")
                    return []
            else:
                return []...
```

32. ```sql
)
            return []
    
    def _perform_indicator_verification(self, selected_stocks: List[str], indicator_name: str) -> Dict[str, Any]:
```

33. ```sql
]
            
            for stock_code in selected_stocks:
                try:
                    # 获取股票数据并计算指标
                    stock_data = self.data_manager.get_stock_data(
                ...
```

34. ```sql
return result
            
            # 步骤2: 使用ClickHouse真实数据执行选股
            logger.info(f"🎯 步骤2: 使用真实数据执行选股")
            selected_stocks = self._execute_strategy_selection_Indicator_Closed_Loop_Va...
```

35. ```sql
)
                indicator_verification = self._perform_indicator_verification(selected_stocks, indicator_name)
                result['indicator_verification'] = indicator_verification
             ...
```

### ./scripts/indicator_debug_analyzer.py
- 查询数量: 1

1. ```sql
import os
import sys
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path....
```

### ./scripts/indicator_debug_validator.py
- 查询数量: 11

1. ```sql
# 映射验证模式
    mode_mapping = {
        'quick': ValidationMode.QUICK,
        'priority': ValidationMode.PRIORITY,
        'category': ValidationMode.CATEGORY,
        'full': ValidationMode.FULL
    }...
```

2. ```sql
• 选股比例: {result['selection_ratio']:.4f}
```

3. ```sql
• 最大选股比例: {config.max_selection_ratio}
```

4. ```sql
)
            for indicator_result in successful_indicators:
                name = indicator_result['indicator_name']
                count = indicator_result['selected_count']
                ratio ...
```

5. ```sql
✅ 成功后早停：指标 {last_result['indicator_name']} 选出了 {last_result['selected_count']} 只股票
```

6. ```sql
, stop_on_success=True, stop_on_error=True):
    """运行调试验证"""
    logger.info("🚀 开始指标调试验证")
    logger.info("=" * 60)
    
    # 创建调试配置
    config = create_debug_config(mode, stop_on_success, stop_on_...
```

7. ```sql
• 选股数量: {result['selected_count']}
```

8. ```sql
)
        
        if result.get('selected_stocks'):
            logger.info(f
```

9. ```sql
: ValidationMode.FULL
    }
    
    return Indicator_validation_config(
        mode=mode_mapping.get(mode, Validation_mode.QUICK),  # 支持动态模式
        stock_pool_size=100,              # 较小的股票池，加快验证速度...
```

10. ```sql
import os
import sys
import json
from datetime import datetime

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from analysis.engines....
```

11. ```sql
• 最小选股数: {config.min_selection_count}
```

### ./scripts/indicator_logic_validator.py
- 查询数量: 20

1. ```sql
)
                return validation_result
            
            validation_result['data_quality'] = True
            validation_result['debug_info']['data_shape'] = stock_data.shape
            va...
```

2. ```sql
]:
                condition_validation = self._validate_selection_condition(
                    indicator_name, stock_code, stock_data
                )
                validation_result.update(cond...
```

3. ```sql
SELECT code as stock_code, close, volume, turnover_rate
            FROM stock_info
```

4. ```sql
checks = {}
        
        try:
            required_columns = ['UPPER', 'MIDDLE', 'LOWER']
            has_columns = all(col in boll_result.columns for col in required_columns)
            checks['...
```

5. ```sql
SELECT date, open, high, low, close, volume, 0 as amount, turnover_rate
            FROM stock_info
```

6. ```sql
] = False
        
        return checks
    
    def _validate_selection_condition(self, indicator_name: str, stock_code: str, 
                                    stock_data: pd.Data_frame) -> Dict[...
```

7. ```sql
:
                checks.update(self._validate_kdj_logic(stock_data, indicator_result))
            elif indicator_name ==
```

8. ```sql
logic_result = {
            'logic_validation': False,
            'logic_checks': {},
            'warnings': []
        }
        
        try:
            # 通用逻辑检查
            checks = {
         ...
```

9. ```sql
:
                checks.update(self._validate_boll_logic(stock_data, indicator_result))
            
            logic_result[
```

10. ```sql
:
                checks.update(self._validate_macd_logic(stock_data, indicator_result))
            elif indicator_name ==
```

11. ```sql
SELECT code as stock_code, close, volume, turnover_rate
            FROM stock_info WHERE 1=1
            WHERE date = '{self.test_date}'
            AND level = '日线'
            AND close > 5.0 AND c...
```

12. ```sql
:
                checks.update(self._validate_rsi_logic(stock_data, indicator_result))
            elif indicator_name ==
```

13. ```sql
)
                    
                    # 3. 验证计算逻辑
                    logic_validation = self._validate_indicator_logic(
                        indicator_name, stock_data, indicator_result
     ...
```

14. ```sql
:
                checks.update(self._validate_ma_logic(stock_data, indicator_result))
            elif indicator_name ==
```

15. ```sql
condition_result = {
            'condition_validation': False,
            'condition_tests': {},
            'selection_reasonable': False
        }
        
        try:
            # 创建测试条件
      ...
```

16. ```sql
: {}
        }
        
        # 获取测试数据 - 使用数据库中实际存在的日期
        # self.test_date = get_latest_trading_date()
        self.test_date = "2025-05-23"  # 使用数据库中实际存在的最新日期
        self.test_stocks = self._...
```

17. ```sql
SELECT date, open, high, low, close, volume, 0 as amount, turnover_rate
            FROM stock_info WHERE 1=1
            WHERE code = '{stock_code}'
            AND level = '日线'
            AND date ...
```

18. ```sql
)
                validation_result['debug_info']['calculation_error'] = str(calc_error)
                if self.debug_mode:
                    validation_result['debug_info']['calculation_traceback'...
```

19. ```sql
)
            
            query = f"""
            SELECT date, open, high, low, close, volume, 0 as amount, turnover_rate
            FROM stock_info WHERE 1=1
            WHERE code =
```

20. ```sql
import sys
import os
import time
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import json

# 添加项...
```

### ./scripts/indicator_multi_pattern_validator.py
- 查询数量: 16

1. ```sql
default_config = {
            'validation': {
                'date': '2024-12-27',
                'min_selection_count': 1,
                'max_selection_ratio': 0.3,
                'stock_pool_s...
```

2. ```sql
)
            return []
    
    def _perform_buypoint_analysis_Indicator_Multi_Pattern_Validator(self, selected_stocks: List[str], 
                                 indicator_name: str, pattern_name:...
```

3. ```sql
]:
                    if col in result_df.columns:
                        stock_code_column = col
                        break
                
                if stock_code_column:
               ...
```

4. ```sql
)
            return False
    
    def _verify_pattern_closed_loop(self, indicator_name: str, pattern_name: str, 
                                  selected_stocks: List[str], buypoint_analysis: Dict...
```

5. ```sql
]
        analyzed_stocks = 0
        stocks_with_buypoints = 0
        total_buypoints = 0
        pattern_confirmed_buypoints = 0
        buypoint_details = []
        
        for stock_code in sel...
```

6. ```sql
:
                    return 40 <= rsi <= 60
            
            # 默认基于信号强度判断
            return signal_strength > 0.5
            
        except Exception as e:
            logger.warning(f"检查形...
```

7. ```sql
}
        }
        
        return strategy_config
    
    def _execute_strategy_selection_Indicator_Multi_Pattern_Validator(self, strategy_config: Dict[str, Any], stock_pool: List[str]) -> List[str...
```

8. ```sql
,
                'description': pattern_config['description'],
                'conditions': pattern_config['conditions'],
                'logic': 'AND'
            }
        }
        
        retu...
```

9. ```sql
]}")
        
        return report
    
    def _validate_single_pattern_Indicator_Multi_Pattern_Validator(self, indicator_name: str, pattern_name: str, 
                               pattern_config...
```

10. ```sql
if not selected_stocks:
            return {
                'analyzed_stocks': 0,
                'stocks_with_buypoints': 0,
                'total_buypoints': 0,
                'pattern_confirmed_...
```

11. ```sql
: datetime.now().isoformat()
            }
            
            logger.info(f"✅ 形态 {indicator_name}.{pattern_name} 验证完成")
            logger.info(f"   选股数量: {len(selected_stocks)}")
            lo...
```

12. ```sql
pattern_stats[pattern_key] = {
                        'selection_count': result.get('selection_count', 0),
                        'selection_ratio': result.get('selection_ratio', 0),
               ...
```

13. ```sql
import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np

# 添加项目根目录到Python路径
root_dir =...
```

14. ```sql
选股数量: {len(selected_stocks)}
```

15. ```sql
try:
            # 检查1: 至少有一个股票被选中
            if len(selected_stocks) == 0:
                return False
            
            # 检查2: 买点分析成功执行
            analyzed_stocks = buypoint_analysis.get('...
```

16. ```sql
start_time = datetime.now()
        
        try:
            # 1. 生成形态策略
            strategy_config = self._generate_pattern_strategy(
                indicator_name, pattern_name, pattern_config
  ...
```

### ./scripts/indicator_optimization_plan.py
- 查询数量: 4

1. ```sql
* 50)
        
        for i, (batch_name, priority_desc) in enumerate(priority_batches, 1):
            if batch_name in self.no_selection_indicators:
                indicators = self.no_selection_i...
```

2. ```sql
* 80)
        
        steps = [
            {
                'step': 1,
                'title': '扩大股票池测试',
                'description': '将测试股票池从5只扩展到50只，验证是否有更多股票符合条件',
                'indicator...
```

3. ```sql
)
        ]
        
        print(f"\n🚀 分阶段优化计划:")
        print("-" * 50)
        
        for i, (batch_name, priority_desc) in enumerate(priority_batches, 1):
            if batch_name in self.no_...
```

4. ```sql
import json
import os
from typing import Dict, List, Any

class IndicatorOptimizationPlan:
    def __init__(self):
        self.no_selection_indicators = {
            # 第1批-基础指标 (3个)
            '第1批...
```

### ./scripts/indicator_validation_demo.py
- 查询数量: 19

1. ```sql
选股数量: {result.get('selected_count', 0)}
```

2. ```sql
*60)
    
    # 配置优先级验证
    config = Indicator_validation_config(
        mode=Validation_mode.PRIORITY,
        stock_pool_size=400,
        max_selection_ratio=0.07,
        parallel_workers=1,
    ...
```

3. ```sql
*60)
    
    # 配置快速验证
    config = Indicator_validation_config(
        mode=Validation_mode.QUICK,
        stock_pool_size=200,
        max_selection_ratio=0.08,
        parallel_workers=1,
        ...
```

4. ```sql
]} "
                  f"(成功率: {success_rate:.1%})")
        
        return results
        
    except Exception as e:
        print(f"分类验证失败: {e}")
        logger.error(f"分类验证失败: {e}")
        retu...
```

5. ```sql
import os
import sys

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from analysis.engines.indicator_validation_framework import (
  ...
```

6. ```sql
选股率: {result.get('selection_ratio', 0):.2%}
```

7. ```sql
(选股率: {result.get('selection_ratio', 0):.2%})
```

8. ```sql
(选股率{indicator['selection_ratio']:.2%})
```

9. ```sql
average_selection_ratio
```

10. ```sql
*60)
    
    # 配置分类验证
    config = Indicator_validation_config(
        mode=Validation_mode.CATEGORY,
        stock_pool_size=300,
        max_selection_ratio=0.06,
        parallel_workers=2,
     ...
```

11. ```sql
平均选股率: {results['summary']['average_selection_ratio']:.2%}
```

12. ```sql
无选股: {results['summary']['validation_results']['no_selection']}
```

13. ```sql
)
            
            if result.get('selected_stocks'):
                print(f
```

14. ```sql
*60)
    
    # 配置验证
    config = Indicator_validation_config(
        stock_pool_size=500,
        max_selection_ratio=0.05,
        save_details=False
    )
    
    # 创建验证框架
    framework = Indicat...
```

15. ```sql
]}")
                
        except Exception as e:
            print(f"验证 {indicator} 时出错: {e}")
            logger.error(f"验证 {indicator} 时出错: {e}")


def demo_category_validation():
    """演示分类验证模...
```

16. ```sql
{i}. {indicator['indicator']}: 选出{indicator['selected_count']}只股票
```

17. ```sql
过度选择: {results['summary']['validation_results']['over_selection']}
```

18. ```sql
选股: {result.get('selected_count', 0)}只
```

19. ```sql
)}")
        
        return results
        
    except Exception as e:
        print(f"快速验证失败: {e}")
        logger.error(f"快速验证失败: {e}")
        return None


def demo_single_indicator_validation()...
```

### ./scripts/indicator_validation_framework.py
- 查询数量: 53

1. ```sql
)
            
            # 执行买点分析
            buypoint_analysis = self._perform_buypoint_analysis(result['selected_stocks'], indicator_name)
            result['buypoint_analysis'] = buypoint_analys...
```

2. ```sql
# {report['indicator_name']} 指标验证报告

## 📊 验证概要

- **指标名称**: {report['indicator_name']}
- **验证时间**: {report['timestamp']}
- **验证结果**: {'✅ 通过' if report['validation_summary']['passed'] else '❌ 失败'}
- **...
```

3. ```sql
report = {
            'indicator_name': indicator_name,
            'timestamp': datetime.now().isoformat(),
            'validation_summary': {
                'passed': result.get('closed_loop_veri...
```

4. ```sql
try:
            csv_data = []
            for indicator_name, result in self.validation_results.items():
                csv_data.append({
                    'indicator_name': indicator_name,
      ...
```

5. ```sql
] = selected_stocks
            
            # 步骤3: 买点分析验证
            if selected_stocks:
                logger.info(f"🔬 步骤3: 对选出的 {len(selected_stocks)} 只股票进行买点分析")
                buypoint_analysi...
```

6. ```sql
, []))
        if selected_count == 0:
            recommendations.append("⚠️ 未选出任何股票，建议放宽策略条件")
        elif selected_count == 1:
            recommendations.append("✅ 选股数量合理，早停功能正常工作")
        
    ...
```

7. ```sql
, 0)
            if 0.01 <= selection_ratio <= 0.3:  # 合理的选股比例
                score += 0.2
            elif selection_ratio > 0:
                score += 0.1
            
            # 买点分析分数
       ...
```

8. ```sql
try:
            score = 0.0
            
            # 基础分数：能够执行选股
            if result['status'] == 'success':
                score += 0.3
            
            # 选股效果分数
            selection_r...
```

9. ```sql
)
        
        selected_count = len(result.get('selected_stocks', []))
        if selected_count == 0:
            recommendations.append(
```

10. ```sql
, 0)
            if correlation_rate < 0.3:  # 至少30%的相关性
                logger.info(f"指标买点相关性过低: {correlation_rate:.2%}")
                return False
            
            logger.info(f"✅ 指标 {ind...
```

11. ```sql
:
                score += 0.3
            
            # 选股效果分数
            selection_ratio = result.get(
```

12. ```sql
验证单个股票的指标闭环一致性（适用于早停场景）
        
        Args:
            indicator_name: 指标名称
            selected_stocks: 选出的股票（通常只有1个）
            buypoint_analysis: 买点分析结果
            
        Returns:
         ...
```

13. ```sql
] = buypoint_analysis
                
                # 步骤4: 闭环验证
                logger.info(f"🔄 步骤4: 验证指标闭环一致性")
                closed_loop_verified = self._verify_closed_loop_Indicator_Validation...
```

14. ```sql
})
        
        return conditions
    
    def _execute_strategy_selection_Indicator_Validation_Framework(self, strategy_config: Dict[str, Any], stock_pool: List[str]) -> List[str]:
        """
  ...
```

15. ```sql
)
            
            try:
                result = self._validate_single_indicator_closed_loop_Indicator_Validation_Framework(
                    indicator_name, stock_pool
                )
  ...
```

16. ```sql
.join(no_selection_indicators[:5])}")
        
        return recommendations
    
    def _save_validation_results_Indicator_Validation_Framework(self, report: Dict[str, Any]):
        """保存验证结果"""
 ...
```

17. ```sql
对选出的股票进行买点分析
        
        Args:
            selected_stocks: 选出的股票列表
            indicator_name: 指标名称
            
        Returns:
            买点分析结果
```

18. ```sql
try:
            # 修改策略配置以使用全部股票池
            modified_config = strategy_config.copy()
            if 'strategy' in modified_config:
                # 在过滤器中添加股票池限制
                if 'filters' not in ...
```

19. ```sql
验证指标闭环一致性
        
        Args:
            indicator_name: 指标名称
            selected_stocks: 选出的股票
            buypoint_analysis: 买点分析结果
            
        Returns:
            是否通过闭环验证
```

20. ```sql
indicators_with_selections
```

21. ```sql
.join(failed_indicators[:5])}")
        
        no_selection_indicators = [name for name, result in self.validation_results.items() if result[
```

22. ```sql
)
                    return selected_stocks
                else:
                    logger.warning(f
```

23. ```sql
]:
                    if col in result_df.columns:
                        stock_code_column = col
                        break
                
                if stock_code_column:
               ...
```

24. ```sql
return result
            
            # 步骤2: 使用ClickHouse真实数据执行选股
            logger.info(f"🎯 步骤2: 使用真实数据执行选股")
            selected_stocks = self._execute_strategy_selection_Indicator_Validation_Fra...
```

25. ```sql
, pd.Series()).iloc[0] if not buypoint_indicator_data.empty else None
                return k_value is not None and (k_value < 20 or (20 <= k_value <= 50))
            
            # 其他指标的通用检查
      ...
```

26. ```sql
try:
            # 闭环验证标准
            min_selection_count = self.config['validation']['min_selection_count']
            max_selection_ratio = self.config['validation']['max_selection_ratio']
        ...
```

27. ```sql
)
            selected_stocks = self._execute_strategy_selection_Indicator_Validation_Framework(strategy_config, stock_pool)
            
            result['selection_count'] = len(selected_stocks)
 ...
```

28. ```sql
)
            return []
    
    def _perform_buypoint_analysis(self, selected_stocks: List[str], indicator_name: str) -> Dict[str, Any]:
```

29. ```sql
)
        
        # 获取股票池
        stock_pool = self._get_stock_pool_Indicator_Validation_Framework()
        
        # 执行闭环验证
        result = self._validate_single_indicator_closed_loop_Indicator_V...
```

30. ```sql
]
            
            for stock_code in selected_stocks:
                try:
                    # 执行简化买点分析（避免DataFrame列数不匹配问题）
                    buypoints = self._analyze_stock_simple(
      ...
```

31. ```sql
import sys
import os
import json
import time
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
...
```

32. ```sql
] = len(selected_stocks) / len(stock_pool) if stock_pool else 0
            result[
```

33. ```sql
analysis_result = {
            'total_stocks': len(selected_stocks),
            'analyzed_stocks': 0,
            'stocks_with_buypoints': 0,
            'total_buypoints': 0,
            'indicator...
```

34. ```sql
: len(selected_stocks),
```

35. ```sql
🔄 开始闭环验证: 指标={indicator_name}, 股票数量={len(selected_stocks)}
```

36. ```sql
]
            selection_ratio = selection_count / stock_pool_size
            if selection_ratio > max_selection_ratio:
                logger.info(f"选股比例过高: {selection_ratio:.2%}")
                re...
```

37. ```sql
, {})
            
            if selected_stocks and buypoint_analysis.get(
```

38. ```sql
)
        
        # 验证结果存储
        self.validation_results = {}
        self.validation_stats = {
            'total_indicators': 0,
            'successful_validations': 0,
            'failed_valid...
```

39. ```sql
consistency = {
            'strategy_buypoint_alignment': False,
            'indicator_signal_consistency': False,
            'data_quality_check': True,
            'overall_consistency_score': 0....
```

40. ```sql
] = len(selected_stocks)
            result[
```

41. ```sql
)
                closed_loop_verified = self._verify_closed_loop_Indicator_Validation_Framework(
                    indicator_name, selected_stocks, buypoint_analysis
                )
             ...
```

42. ```sql
)
        
        no_selection_indicators = [name for name, result in self.validation_results.items() if result['status'] == 'no_selection']
        if no_selection_indicators:
            recommenda...
```

43. ```sql
)
                return False
            
            # 检查2: 选股比例合理
            stock_pool_size = self.config['validation']['stock_pool_size']
            selection_ratio = selection_count / stock_p...
```

44. ```sql
)
        elif selected_count == 1:
            recommendations.append(
```

45. ```sql
: 0.0
        }
        
        try:
            # 检查策略选股与买点分析的一致性
            selected_stocks = result.get(
```

46. ```sql
)
                buypoint_analysis = self._perform_buypoint_analysis(selected_stocks, indicator_name)
                result['buypoint_analysis'] = buypoint_analysis
                
                ...
```

47. ```sql
)
            return False
    
    def _verify_closed_loop_with_single_stock(self, indicator_name: str, selected_stocks: List[str], 
                                            buypoint_analysis: Dic...
```

48. ```sql
)
            return False
    
    def _verify_closed_loop_Indicator_Validation_Framework(self, indicator_name: str, selected_stocks: List[str], 
                          buypoint_analysis: Dict[str...
```

49. ```sql
selected_stocks_count
```

50. ```sql
, False):
            recommendations.append("❌ 闭环验证失败，需要检查指标计算逻辑或策略条件")
        
        selected_count = len(result.get(
```

51. ```sql
start_time = time.time()
        result = {
            'indicator_name': indicator_name,
            'timestamp': datetime.now().isoformat(),
            'status': 'unknown',
            'selection_c...
```

52. ```sql
conditions = []
        
        # 根据指标类型生成特定条件 - 使用更宽松的条件
        indicator_upper = indicator_name.upper()
        
        # 为了确保选出股票，使用基础价格条件而不是复杂的指标条件
        if indicator_upper == 'RSI':
        ...
```

53. ```sql
default_config = {
            'validation': {
                'date': get_latest_trading_date(),
                'lookback_days': 30,
                'stock_pool_size': 100,
                'max_sele...
```

### ./scripts/lightweight_indicator_validator.py
- 查询数量: 1

1. ```sql
import sys
import os
import argparse
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import traceback

# 添加项目根目录到Python路径
root_dir = os.path.dirn...
```

### ./scripts/manage_db_config.py
- 查询数量: 1

1. ```sql
import sys
import os
import argparse
import getpass
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.database...
```

### ./scripts/optimize_strategy.py
- 查询数量: 6

1. ```sql
import sys
import os
import json
import argparse
import pandas as pd
from datetime import datetime
from pathlib import Path
from collections import Counter

# 添加项目根目录到路径
project_root = Path(__file__)....
```

2. ```sql
: True    # 逻辑优化
            }
        
        try:
            # 1. 加载原始策略
            original_strategy = self._load_strategy_Optimize_Strategy(strategy_file)
            
            # 2. 分析条件频率
 ...
```

3. ```sql
)}" for c in conditions]
        analysis["indicator_period_combinations"] = dict(Counter(combinations))
        
        return analysis
    
    def _select_core_conditions(self, strategy: dict, fre...
```

4. ```sql
] = dict(Counter(combinations))
        
        return analysis
    
    def _select_core_conditions(self, strategy: dict, frequency_analysis: dict, config: dict) -> list:
```

5. ```sql
)}"
        optimized_strategy["version"] = "1.0_optimized"
        optimized_strategy["optimization_info"] = {
            "original_conditions": len(original_strategy.get("conditions", [])),
       ...
```

6. ```sql
frequency_based_selection
```

### ./scripts/optimized_indicator_validator.py
- 查询数量: 9

1. ```sql
status = result.get('status', 'unknown')
        selected_count = result.get('selected_count', 0)
        
        print(
```

2. ```sql
选股数量: {selected_count}
```

3. ```sql
)
        selected_count = result.get(
```

4. ```sql
)
        
        if status == 'success' and result.get('selected_stocks'):
            stocks = result['selected_stocks'][:5]  # 只显示前5只
            print(f
```

5. ```sql
import os
import sys
import logging
import argparse
import json
from datetime import datetime
from typing import Dict, List, Any

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path....
```

6. ```sql
, 4378)
        config.max_selection_ratio = validation_config.get(
```

7. ```sql
, 0.1)
        config.min_selection_count = validation_config.get(
```

8. ```sql
) as f:
                config = json.load(f)
            logger.info(f"✅ 加载配置文件: {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"❌ 加载配置文件失败: {...
```

9. ```sql
, 0)
        
        print("\n" + "="*60)
        print(f"📊 指标验证结果: {indicator_name}")
        print("="*60)
        print(f"状态: {status}")
        print(f"选股数量: {selected_count}")
        print(f"执行...
```

### ./scripts/production_database_test.py
- 查询数量: 13

1. ```sql
SELECT code, name FROM stock_info WHERE date >= '2020-01-01' LIMIT 10
```

2. ```sql
SELECT COUNT(*) as count, 
                           MIN(date) as min_date, 
                           MAX(date) as max_date,
                           AVG(close) as avg_close,
                    ...
```

3. ```sql
SELECT COUNT(*) as count FROM {table} LIMIT 1"
                    result = data_manager.execute_query(query)
                    if result is not None and not result.empty:
                        co...
```

4. ```sql
SELECT COUNT(*) as count FROM system LIMIT 1000.databases
```

5. ```sql
SELECT COUNT(*) as count FROM system
```

6. ```sql
SELECT COUNT(*) as count FROM {table} LIMIT 1
```

7. ```sql
]
            
            existing_tables = []
            missing_tables = []
            
            for table in required_tables:
                try:
                    query = f"SELECT COUNT(*...
```

8. ```sql
)
                    
                    data_query = f"""
                    SELECT COUNT(*) as count, 
                           MIN(date) as min_date, 
                           MAX(date) as m...
```

9. ```sql
)
                    
                    # 使用000001作为测试股票
                    test_config = {
                        'validation_date': '2024-01-01',
                        'stock_pool': ['000001'...
```

10. ```sql
: time.time() - start_time
                }
            
            # 测试数据查询
            test_query = "SELECT COUNT(*) as count FROM system LIMIT 1000.databases"
            result = data_manager.ex...
```

11. ```sql
: str(e)
            }
    
    def test_stock_data_quality(self) -> Dict[str, Any]:
        """测试股票数据质量"""
        print("\n📊 测试股票数据质量...")
        
        try:
            data_manager = Unified_da...
```

12. ```sql
SELECT COUNT(*) as count, 
                           MIN(date) as min_date, 
                           MAX(date) as max_date,
                           AVG(close) as avg_close,
                    ...
```

13. ```sql
import sys
import os
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import subprocess

# 添加项目根目录到路径
root_dir = os.path.dirname(os....
```

### ./scripts/production_indicator_tester.py
- 查询数量: 51

1. ```sql
, False):
                    continue
                
                selected_stocks = result.get(
```

2. ```sql
)

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.dependency_injection import get_service
from indicators.zxm.buy_point_indicators import (
...
```

3. ```sql
SELECT date, open, high, low, close, volume, turnover
            FROM stock_info WHERE 1=1
            WHERE code = %(stock_code)s
              AND level = '日线'
              AND date >= %(start_dat...
```

4. ```sql
: has_buy_signal
                    }
                
            except Exception as e:
                logger.debug(f"处理股票 {stock_code} 时出错: {e}")
                failed_stocks.append(stock_code)
...
```

5. ```sql
average_selection_rate
```

6. ```sql
)
                    
                    # 显示部分选中股票
                    selected_stocks = result.get('selected_stocks', [])
                    if selected_stocks:
                        f.write(f
```

7. ```sql
)
            
            try:
                # 获取股票数据
                stock_data = self.get_stock_data_Tester(stock_code, test_date)
                if stock_data.empty or len(stock_data) < 30:
   ...
```

8. ```sql
)
        
        start_time = time.time()
        
        # 获取股票池
        stock_pool = self.get_stock_pool(test_date, max_stocks)
        if not stock_pool:
            return {
                'in...
```

9. ```sql
),
                }
                
                # 添加指标值信息
                if stock_code in indicator_values:
                    values = indicator_values[stock_code]
                    row.upd...
```

10. ```sql
)
    
    def get_stock_pool(self, test_date: str, max_stocks: int = 1000) -> List[str]:
        """
        获取测试股票池
        
        Args:
            test_date: 测试日期
            max_stocks: 最大股票数量
...
```

11. ```sql
, 0))
            selected_counts.append(stats.get(
```

12. ```sql
csv_data = []
        
        # 如果是批量测试结果
        if 'individual_results' in results:
            for indicator_name, result in results['individual_results'].items():
                if not result.ge...
```

13. ```sql
)
            
            query = """
            SELECT date, open, high, low, close, volume, turnover
            FROM stock_info WHERE 1=1
            WHERE code = %(stock_code)s
              AND...
```

14. ```sql
总选股数: {batch_summary.get('total_unique_selections', 0)}\n\n
```

15. ```sql
, {})
            selection_rates.append(stats.get(
```

16. ```sql
) as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        # 生成文本报告
        report_path = output_path / "test_report.txt"
        self._generate_text_report(...
```

17. ```sql
, [])
                if selected_stocks:
                    f.write("选中股票列表:\n")
                    f.write("-" * 40 + "\n")
                    for i, stock_code in enumerate(selected_stocks, 1):
...
```

18. ```sql
SELECT DISTINCT code, name, close, volume
            FROM stock_info
```

19. ```sql
total_unique_selections
```

20. ```sql
, {})
            
            for stock_code in selected_stocks:
                row = {
```

21. ```sql
)
        
        result = {
            'indicator_name': indicator_name,
            'indicator_info': indicator_info,
            'test_date': test_date,
            'execution_time': execution_ti...
```

22. ```sql
)
                failed_stocks.append(stock_code)
                continue
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # 计算统计信息
        tot...
```

23. ```sql
].iloc[-1]
                
                if has_buy_signal:
                    selected_stocks.append(stock_code)
                    
                    # 保存指标值用于分析
                    indicator...
```

24. ```sql
- 选中股票数: {selected_count}
```

25. ```sql
, 0):.2f}秒\n\n")
                
                # 选中股票列表
                selected_stocks = results.get(
```

26. ```sql
SELECT date, open, high, low, close, volume, turnover
            FROM stock_info
```

27. ```sql
)
                
                # 选中股票列表
                selected_stocks = results.get('selected_stocks', [])
                if selected_stocks:
                    f.write(
```

28. ```sql
最差指标: {worst['name']} (选股率: {worst['selection_rate']:.2%}, 选股数: {worst['selected_count']})\n\n
```

29. ```sql
, [])
                    if selected_stocks:
                        f.write(f"选中股票 (前20只): {
```

30. ```sql
)
            selected_stocks = results.get(
```

31. ```sql
)
                        if len(selected_stocks) > 20:
                            f.write(f
```

32. ```sql
import sys
import os
import json
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Set, Any, Optio...
```

33. ```sql
最佳指标: {best['name']} (选股率: {best['selection_rate']:.2%}, 选股数: {best['selected_count']})\n
```

34. ```sql
self._generate_text_report(results, str(report_path))
        
        # 保存选股结果CSV
        csv_path = self._save_selection_csv(results, str(output_path), timestamp)
        
        logger.info(f
```

35. ```sql
平均选股率: {batch_summary.get('average_selection_rate', 0):.2%}\n
```

36. ```sql
- 选股率: {selection_rate:.2%}
```

37. ```sql
successful_results = {k: v for k, v in individual_results.items() 
                            if v.get('success', False)}
        
        if len(successful_results) < 2:
            return {'overlap...
```

38. ```sql
SELECT DISTINCT code, name, close, volume
            FROM stock_info WHERE 1=1
            WHERE date = %(test_date)s
              AND level = '日线'
              AND close > 2.0
              AND cl...
```

39. ```sql
, 0):.2f}秒\n")
                    
                    # 显示部分选中股票
                    selected_stocks = result.get(
```

40. ```sql
.join(selected_stocks[:20])}")
                        if len(selected_stocks) > 20:
                            f.write(f" (共{len(selected_stocks)}只)")
                        f.write("\n")
         ...
```

41. ```sql
),
                    })
                
                csv_data.append(row)
        
        if not csv_data:
            return None
        
        # 保存CSV文件
        df = pd.Data_frame(csv_data...
```

42. ```sql
successful_results = {k: v for k, v in individual_results.items() 
                            if v.get('success', False)}
        
        if not successful_results:
            return {'total_indica...
```

43. ```sql
, 0))
        
        # 最佳和最差指标
        if selection_rates:
            best_indicator = max(successful_results.items(), 
                               key=lambda x: x[1].get(
```

44. ```sql
),
                    }
                    
                    # 添加指标值信息
                    if stock_code in indicator_values:
                        values = indicator_values[stock_code]
       ...
```

45. ```sql
选中股票数: {stats.get('selected_count', 0)}\n
```

46. ```sql
]()
        
        selected_stocks = []
        failed_stocks = []
        indicator_values = {}
        
        logger.info(f"📊 开始处理 {len(stock_pool)} 只股票...")
        
        for i, stock_code i...
```

47. ```sql
, {})
                
                for stock_code in selected_stocks:
                    row = {
```

48. ```sql
)
                    for i, stock_code in enumerate(selected_stocks, 1):
                        f.write(f
```

49. ```sql
)
    
    def _save_selection_csv(self, results: Dict[str, Any], output_dir: str, timestamp: str) -> Optional[str]:
```

50. ```sql
选股率: {stats.get('selection_rate', 0):.2%}\n
```

51. ```sql
: 0}
        
        # 统计信息
        total_indicators = len(individual_results)
        successful_indicators = len(successful_results)
        
        # 选股效果统计
        selection_rates = []
        s...
```

### ./scripts/production_indicator_validator.py
- 查询数量: 32

1. ```sql
选股数量: {stats.get('selected_count')}\n
```

2. ```sql
}
        
        # 获取指标实例
        indicator = self.available_indicators[indicator_name]
        
        # 验证结果
        selected_stocks = []
        failed_stocks = []
        indicator_values = {}
...
```

3. ```sql
filepath = os.path.join(output_dir, filename)
            
            # 保存JSON结果
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False,...
```

4. ```sql
: self._get_top_signals(indicator_values, 10)
            }
        }
        
        logger.info(f"✅ 指标 {indicator_name} 验证完成")
        logger.info(f"📊 选股结果: {selected_count}/{success_count} = {sele...
```

5. ```sql
)
                failed_stocks.append(stock_code)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 计算统计结果
        total_tested = len(stock_po...
```

6. ```sql
平均选股率: {summary.get('avg_selection_rate', 0):.2%}\n
```

7. ```sql
)
                
                # 获取股票数据
                stock_data = self.get_stock_data_Validator(stock_code, test_date)
                
                if stock_data.empty or len(stock_data) < ...
```

8. ```sql
)
            self._generate_text_report_Production_Indicator_Validator(results, report_path)
            
            # 如果有选股结果，保存CSV
            csv_path = self._save_selection_csv_Production_Indica...
```

9. ```sql
)
                            if len(selected_stocks) > 20:
                                f.write(f
```

10. ```sql
SELECT DISTINCT code, name, close, volume
            FROM stock_info
```

11. ```sql
# 收集所有成功的指标的选股结果
        successful_indicators = {}
        for indicator_name, result in individual_results.items():
            if result.get('status') == 'success':
                successful_indic...
```

12. ```sql
)
                        
                        # 选股结果
                        selected_stocks = results.get('selected_stocks', [])
                        if selected_stocks:
                     ...
```

13. ```sql
)
                            for i, stock in enumerate(selected_stocks[:20]):  # 只显示前20只
                                f.write(f
```

14. ```sql
SELECT date, open, high, low, close, volume, turnover
            FROM stock_info
```

15. ```sql
}
        
        # 计算两两重叠度
        overlap_matrix = {}
        indicator_names = list(successful_indicators.keys())
        
        for i, indicator1 in enumerate(indicator_names):
            over...
```

16. ```sql
)
            
            query = f"""
            SELECT date, open, high, low, close, volume, turnover
            FROM stock_info WHERE 1=1
            WHERE code =
```

17. ```sql
}
        
        # 统计成功的指标
        selection_rates = [r[
```

18. ```sql
successful_results = [r for r in individual_results.values() if r.get('status') == 'success']
        failed_results = [r for r in individual_results.values() if r.get('status') == 'failed']
        
...
```

19. ```sql
, [])
                        if selected_stocks:
                            f.write(f"选中股票 ({len(selected_stocks)}只):\n")
                            for i, stock in enumerate(selected_stocks[:20]):...
```

20. ```sql
, 0):.2%}\n\n")
                        
                        # 选股结果
                        selected_stocks = results.get(
```

21. ```sql
)}\n")
                
                f.write("\n" + "="*80 + "\n")
                f.write("报告生成时间: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
                
        except Exceptio...
```

22. ```sql
)
    
    def get_stock_pool_Validator(self, test_date: str, max_stocks: int = 500) -> List[str]:
        """
        获取测试股票池
        
        Args:
            test_date: 测试日期
            max_stocks...
```

23. ```sql
: indicator_name
                        })
            
            if not selected_stocks:
                return None
            
            # 保存CSV
            df = pd.Data_frame(selected_stocks...
```

24. ```sql
try:
            selected_stocks = []
            
            if results.get('validation_type') == 'batch':
                # 批量验证结果
                for indicator_name, result in results.get('individ...
```

25. ```sql
]
                
                if has_buy_signal:
                    selected_stocks.append(stock_code)
                
                # 保存指标值用于分析 - 确保所有值都可以JSON序列化
                indicator_va...
```

26. ```sql
)
        
        start_time = time.time()
        
        # 获取股票池
        stock_pool = self.get_stock_pool_Validator(test_date, max_stocks)
        if not stock_pool:
            return {
         ...
```

27. ```sql
)
    
    def _save_selection_csv_Production_Indicator_Validator(self, results: Dict[str, Any], output_dir: str, timestamp: str) -> Optional[str]:
```

28. ```sql
SELECT DISTINCT code, name, close, volume
            FROM stock_info WHERE 1=1
            WHERE date = '{test_date}'
              AND level = '日线'
              AND close > 2.0
              AND cl...
```

29. ```sql
📊 选股结果: {selected_count}/{success_count} = {selection_rate:.2%}
```

30. ```sql
SELECT date, open, high, low, close, volume, turnover
            FROM stock_info WHERE 1=1
            WHERE code = '{stock_code}'
              AND level = '日线'
              AND date >= '{start_dat...
```

31. ```sql
选股率: {stats.get('selection_rate', 0):.2%}\n
```

32. ```sql
, 0))
                }
                
            except Exception as e:
                logger.warning(f"⚠️ 处理股票 {stock_code} 失败: {e}")
                failed_stocks.append(stock_code)
        
  ...
```

### ./scripts/production_strategy_validator.py
- 查询数量: 2

1. ```sql
import sys
import os
import json
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Set, Any, Optio...
```

2. ```sql
)

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.dependency_injection import get_service
from db.interfaces.data_access_interface import ID...
```

### ./scripts/quick_indicator_verification.py
- 查询数量: 2

1. ```sql
indicators.zxm.selection_model
```

2. ```sql
*60)
    
    # 所有应该可用的指标
    all_indicators = {
        # 核心指标 (23个)
        'core': [
            ('indicators.ad', 'AD', 'AD'),
            ('indicators.adx', 'ADX', 'ADX'),
            ('indicator...
```

### ./scripts/real_data_reverse_validation.py
- 查询数量: 21

1. ```sql
)
    
    def _get_selection_details(self, selected_stocks: set, config: dict) -> dict:
```

2. ```sql
import sys
import os
import json
import argparse
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.pat...
```

3. ```sql
📈 选中股票: {sorted(list(selected_stocks))}
```

4. ```sql
):  # 处理numpy标量
        return obj.item()
    else:
        return obj


class RealDataReverseValidator:
    """真实数据反向验证器"""
    
    def __init__(self):
        """初始化验证器"""
        try:
            ...
```

5. ```sql
]
                stock_column = None
                
                for col in stock_columns:
                    if col in selected_stocks_df.columns:
                        stock_column = col
  ...
```

6. ```sql
)
            
            for stock_code in selected_stocks:
                try:
                    # 获取股票基本信息
                    stock_info WHERE 1=1 = self.data_manager.get_stock_basic_info(stoc...
```

7. ```sql
)
            
            # 提取股票代码
            if selected_stocks_df is not None and len(selected_stocks_df) > 0:
                # 尝试不同的股票代码列名
                stock_columns = ['stock_code', 'code', ...
```

8. ```sql
)
            return {}
    
    def _analyze_matches(self, original_stocks: set, selected_stocks: set) -> dict:
```

9. ```sql
选中股票数: {selection_results['selected_count']}\n
```

10. ```sql
: list(original_stocks)
            })

            # 临时修改策略执行器的股票获取方法
            original_get_filtered_stock_list = self.strategy_executor._get_filtered_stock_list
            def custom_get_filtere...
```

11. ```sql
选中股票数: {match_analysis['total_selected']}\n
```

12. ```sql
]}\n\n")
                
                # 选股结果
                selection_results = results["selection_results"]
                f.write("选股结果:\n")
                f.write(f"  选中股票数: {selection_resul...
```

13. ```sql
if not selected_stocks:
            return {}
        
        try:
            # 获取股票基本信息
            details = {}
            validation_date = config.get(
```

14. ```sql
选中股票: {', '.join(selection_results['selected_stocks'])}\n\n
```

15. ```sql
策略选中股票: {match_analysis['total_selected']} 只
```

16. ```sql
)

            # 修改策略执行计划，只处理原始买点股票
            # 重新加载买点数据来获取原始股票列表
            buypoints_df = self._load_buypoints_Real_Data_Reverse_Validation(buypoints_file)
            original_stocks = self._ext...
```

17. ```sql
# 计算交集和差集
        matched_stocks = original_stocks.intersection(selected_stocks)
        missed_stocks = original_stocks - selected_stocks
        false_positive_stocks = selected_stocks - original_st...
```

18. ```sql
] = self._extract_validation_date(original_buypoints)

            # 4. 使用真实数据执行策略选股
            selected_stocks = self._execute_real_strategy(strategy, config, buypoints_file)
            
          ...
```

19. ```sql
code_str = str(code).strip()
                    # 如果是数字，补齐到6位
                    if code_str.isdigit():
                        return code_str.zfill(6)
                    return code_str

        ...
```

20. ```sql
)
                
                # 选股结果
                selection_results = results[
```

21. ```sql
.join(selection_results[
```

### ./scripts/regenerate_buypoint_report.py
- 查询数量: 1

1. ```sql
import os
import sys
import json
from datetime import datetime

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from utils.logge...
```

### ./scripts/remaining_indicators_registration.py
- 查询数量: 2

1. ```sql
indicators.zxm.selection_model
```

2. ```sql
)
        
        zxm_batch2_indicators = [
            # ZXM Buy Points 剩余 (2个)
            ('indicators.zxm.buy_point_indicators', 'ZXMMACallback', 'ZXM_MA_CALLBACK'),
            ('indicators.zxm....
```

### ./scripts/rerun_buypoint_analysis.py
- 查询数量: 1

1. ```sql
import os
import sys
import json
from datetime import datetime

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from utils.logge...
```

### ./scripts/reverse_validation.py
- 查询数量: 21

1. ```sql
)

            # 提取股票代码
            if selected_stocks_df is not None and len(selected_stocks_df) > 0:
                if 'stock_code' in selected_stocks_df.columns:
                    selected_stock...
```

2. ```sql
)
                return selected_stocks
            else:
                print(
```

3. ```sql
* 60)
    
    results = validator.validate_reverse_selection(args.buypoints, args.strategy, config)
    
    # 保存结果
    timestamp = datetime.now().strftime(
```

4. ```sql
].tolist())
                else:
                    # 如果没有明确的股票代码列，使用第一列
                    selected_stocks = set(selected_stocks_df.iloc[:, 0].tolist())

                print(f"✅ 真实策略选股完成，选出 {len...
```

5. ```sql
)
            return set()

    def _simulate_strategy_selection(self, strategy: dict, original_stocks: set, config: dict) -> set:
```

6. ```sql
in selected_stocks_df.columns:
                    selected_stocks = set(selected_stocks_df[
```

7. ```sql
]}")
    print("-" * 60)
    
    results = validator.validate_reverse_selection(args.buypoints, args.strategy, config)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    res...
```

8. ```sql
# 计算交集和差集
        matched_stocks = original_stocks.intersection(selected_stocks)
        missed_stocks = original_stocks - selected_stocks
        false_positive_stocks = selected_stocks - original_st...
```

9. ```sql
选中股票数: {match_analysis['total_selected']}\n
```

10. ```sql
)
        
        # 分析策略条件的复杂度
        condition_complexity = self._analyze_condition_complexity(conditions)
        
        # 基于策略复杂度模拟选股结果
        selected_stocks = self._simulate_selection_based_...
```

11. ```sql
]
        
        stock_column = None
        for col in possible_columns:
            if col in buypoints.columns:
                stock_column = col
                break
        
        if stock_...
```

12. ```sql
策略选中股票: {match_analysis['total_selected']} 只
```

13. ```sql
: 100                # 样本大小
            }
        
        try:
            # 1. 加载原始买点数据
            original_buypoints = self._load_buypoints_Reverse_Validation(buypoints_file)
            
        ...
```

14. ```sql
)

            # 使用策略执行器进行选股
            selected_stocks_df = self.strategy_executor.execute_strategy_by_id(
                strategy_id=temp_strategy_id,
                strategy_manager=self.strateg...
```

15. ```sql
)
        
        # 模拟选股过程
        selected_stocks = set()
        for stock in stock_list:
            if random.random() < final_probability:
                selected_stocks.add(stock)
        
   ...
```

16. ```sql
])
        
        return complexity
    
    def _simulate_selection_based_on_complexity(self, original_stocks: set, complexity: dict, logic_type: str, config: dict) -> set:
```

17. ```sql
)
        
        return original_stocks

    def _execute_real_strategy_selection(self, strategy: dict, config: dict) -> set:
```

18. ```sql
)
        return selected_stocks
    
    def _analyze_condition_complexity(self, conditions: list) -> dict:
```

19. ```sql
)}"

            print(f"📝 保存临时策略: {temp_strategy_id}")
            self.strategy_manager.save_strategy(temp_strategy_id, strategy)

            # 执行策略选股
            validation_date = config.get("vali...
```

20. ```sql
import sys
import os
import json
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import Counter

# 添加项目根目录到路径
project_roo...
```

21. ```sql
)
        
    def validate_reverse_selection(self, buypoints_file: str, strategy_file: str, config: dict = None) -> dict:
```

### ./scripts/run_all_tests.py
- 查询数量: 1

1. ```sql
\n\nimport os\nimport sys\nimport subprocess\nimport time\nimport json\nfrom datetime import datetime\nfrom typing import Dict, List, Any\n\n# 添加项目根目录到路径\nroot_dir = os.path.dirname(os.path.dirname(os...
```

### ./scripts/run_strategy.py
- 查询数量: 10

1. ```sql
选出股票数量: {summary.get('selected_count', 0)}\n
```

2. ```sql
)
            
            selected_stocks = self.strategy_executor.execute_strategy_by_id(
                strategy_id=temp_strategy_id,
                strategy_manager=self.strategy_manager,
      ...
```

3. ```sql
])
            
            # 4. 过滤和排序结果
            filtered_results = self._filter_results(
                selected_stocks, 
                config.get(
```

4. ```sql
) as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"JSON结果已保存到: {json_file}")
        
        # 导出CSV格式
        if "csv" in formats and results.get("select...
```

5. ```sql
selected_stocks_{timestamp}.csv
```

6. ```sql
import sys
import os
import json
import argparse
import pandas as pd
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0,...
```

7. ```sql
)}"
            self.strategy_manager.save_strategy(temp_strategy_id, strategy)
            
            # 执行策略
            def progress_callback_Strategy_Run_Strategy(progress, message):
            ...
```

8. ```sql
)}\n\n")
            
            if results.get("selected_stocks"):
                f.write("选股结果 (前10名):\n")
                f.write("-" * 40 + "\n")
                for i, stock in enumerate(result...
```

9. ```sql
选出股票: {summary['selected_count']}
```

10. ```sql
))
            
            # 3. 执行策略
            selected_stocks = self._execute_strategy_Run_Strategy(strategy, config[
```

### ./scripts/simple_clickhouse_test.py
- 查询数量: 3

1. ```sql
SELECT COUNT(*) FROM {config['database']}.{first_table}
```

2. ```sql
])
                    print("表结构:")
                    print(structure_df)
                    
                    # 获取第一个表的行数
                    count = client.execute(f"SELECT COUNT(*) FROM {con...
```

3. ```sql
}
]

def test_connection_Test(config):
    """测试与ClickHouse的连接"""
    try:
        print(f"尝试使用配置: {config}")
        # 创建客户端
        print("正在连接到ClickHouse数据库...")
        client = Client(**config)
 ...
```

### ./scripts/simple_early_stop_test.py
- 查询数量: 5

1. ```sql
* 50)
    
    # 创建配置，故意使用会出错的设置
    config = Indicator_validation_config(
        mode=Validation_mode.QUICK,
        stock_pool_size=10,          # 很小的股票池
        max_selection_ratio=0.01,    # 很严格的...
```

2. ```sql
import os
import sys
import time
from datetime import datetime

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from analysis.engines....
```

3. ```sql
📊 选股数量: {result.get('selected_count', 0)}
```

4. ```sql
🎯 选中的股票: {result.get('selected_stocks', [])[:5]}...
```

5. ```sql
* 50)
    
    # 创建配置，使用更宽松的条件确保能选出股票
    config = Indicator_validation_config(
        mode=Validation_mode.QUICK,
        stock_pool_size=50,          # 更小的股票池
        max_selection_ratio=0.5,     #...
```

### ./scripts/simple_strategy_executor.py
- 查询数量: 1

1. ```sql
import os
import sys
import yaml
import pandas as pd
import argparse
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0,...
```

### ./scripts/simple_strategy_validator.py
- 查询数量: 1

1. ```sql
import sys
import os
import json
import argparse
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from collections import Counter

# 添加项目根目录到路径
project_root = Path...
```

### ./scripts/start_unified_engine_test.py
- 查询数量: 5

1. ```sql
, []))}")
    
    print("="*80)


def check_prerequisites():
    """检查前置条件"""
    print("🔍 检查系统前置条件...")
    
    try:
        # 检查数据库连接
        from db.unified_data_manager import get_unified_data_m...
```

2. ```sql
SELECT COUNT(*) as count FROM stock_info WHERE date >= '2020-01-01' LIMIT 1
```

3. ```sql
)
            # 应用自定义参数
            if custom_params:
                # 临时修改配置
                original_config = runner.config['test_configuration'].copy()
                runner.config['test_configur...
```

4. ```sql
SELECT COUNT(*) as count FROM stock_info
```

5. ```sql
import sys
import os
import time
from datetime import datetime

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from utils.logger impo...
```

### ./scripts/test_all_indicators_comprehensive.py
- 查询数量: 3

1. ```sql
import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import json
from datetime import datetime

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.p...
```

2. ```sql
SELECT date, open, high, low, close, volume, turnover
        FROM stock_info WHERE 1=1
        WHERE code = '{code}'
          AND level = '日线'
        ORDER BY date DESC
        LIMIT {limit}
```

3. ```sql
SELECT date, open, high, low, close, volume, turnover
        FROM stock_info
```

### ./scripts/test_condition_evaluation.py
- 查询数量: 1

1. ```sql
import os
import sys

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from db.unified_data_manager import get_unified_data_manag...
```

### ./scripts/test_data_structure.py
- 查询数量: 1

1. ```sql
import os
import sys

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from db.unified_data_manager import get_unified_data_manag...
```

### ./scripts/test_early_stop_functionality.py
- 查询数量: 5

1. ```sql
:
                logger.info("✅ 数据库连接错误被正确检测并触发早停")
            else:
                logger.warning("⚠️ 早停原因可能不是数据库连接错误")
        else:
            logger.warning("❌ 早停功能未触发，可能数据库连接正常或早停配置无效")
     ...
```

2. ```sql
• 最大选股比例: {config.max_selection_ratio}
```

3. ```sql
import os
import sys
import time
from datetime import datetime

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from analysis.engines....
```

4. ```sql
* 60)
    
    # 创建配置，启用成功后早停
    config = Indicator_validation_config(
        mode=Validation_mode.QUICK,
        stock_pool_size=50,
        stop_on_success=True,  # 🔑 启用成功后早停
        stop_on_error...
```

5. ```sql
• 最小选股数: {config.min_selection_count}
```

### ./scripts/test_early_stop_success.py
- 查询数量: 10

1. ```sql
• 选中股票样例: {selected_stocks[:5]}
```

2. ```sql
import os
import sys
import time
from datetime import datetime

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from analysis.engines....
```

3. ```sql
• 选股比例: {result.get('selection_ratio', 0):.4f}
```

4. ```sql
• 最大选股比例: {config.max_selection_ratio}
```

5. ```sql
• 选中股票数: {len(selected_stocks)}
```

6. ```sql
• 选股比例: {len(selected_stocks) / len(stock_pool):.2%}
```

7. ```sql
)
            return
        
        # 执行策略选股，应该选出大部分股票
        selected_stocks = framework._execute_strategy_selection(always_success_strategy, stock_pool)
        
        print(f
```

8. ```sql
* 60)
    
    # 创建宽松配置
    config = Indicator_validation_config(
        mode=Validation_mode.QUICK,
        stock_pool_size=10,           # 更小的股票池
        max_selection_ratio=0.9,      # 90%选股比例
   ...
```

9. ```sql
* 60)
    
    # 创建非常宽松的配置，确保能选出股票
    config = Indicator_validation_config(
        mode=Validation_mode.QUICK,
        stock_pool_size=20,           # 很小的股票池，加快速度
        max_selection_ratio=0.8,   ...
```

10. ```sql
• 选股数量: {result.get('selected_count', 0)}
```

### ./scripts/test_enhanced_indicators.py
- 查询数量: 3

1. ```sql
)
            
        sql = f"""
        SELECT 
            trade_date as date, 
            ts_code as code, 
            open, 
            high, 
            low, 
            close, 
           ...
```

2. ```sql
SELECT 
            trade_date as date, 
            ts_code as code, 
            open, 
            high, 
            low, 
            close, 
            vol as volume,
            amount,
      ...
```

3. ```sql
SELECT 
            trade_date as date, 
            ts_code as code, 
            open, 
            high, 
            low, 
            close, 
            vol as volume,
            amount,
      ...
```

### ./scripts/test_enhanced_indicators_fix.py
- 查询数量: 4

1. ```sql
SELECT date, open, high, low, close, volume
        FROM stock_info
```

2. ```sql
SELECT date, open, high, low, close, volume
        FROM stock_info WHERE 1=1
        WHERE code = '{code}'
          AND level = '日线'
          AND date >= '{start_date}'
          AND date <= '{end_...
```

3. ```sql
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__...
```

4. ```sql
)
        
        # 获取股票数据
        sql = f"""
        SELECT date, open, high, low, close, volume
        FROM stock_info WHERE 1=1
        WHERE code =
```

### ./scripts/test_enhanced_macd.py
- 查询数量: 3

1. ```sql
SELECT 
            trade_date,
            open,
            high,
            low,
            close,
            volume as vol
        FROM stock_kline_day
        WHERE code = '{stock_code}'
     ...
```

2. ```sql
SELECT 
            trade_date,
            open,
            high,
            low,
            close,
            volume as vol
        FROM stock_kline_day
```

3. ```sql
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.inser...
```

### ./scripts/test_enhanced_oscillator.py
- 查询数量: 3

1. ```sql
SELECT 
            date,
            open,
            high,
            low,
            close,
            volume,
            turnover
        FROM stock_info WHERE 1=1
        WHERE code = '{stoc...
```

2. ```sql
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.inser...
```

3. ```sql
SELECT 
            date,
            open,
            high,
            low,
            close,
            volume,
            turnover
        FROM stock_info
```

### ./scripts/test_indicator_data.py
- 查询数量: 1

1. ```sql
import os
import sys

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from db.unified_data_manager import Unified_data_manager
from in...
```

### ./scripts/test_indicators_real_selection.py
- 查询数量: 33

1. ```sql
)
            
            result = self.test_single_indicator_selection(indicator_name)
            test_results.append(result)
            
            if result[
```

2. ```sql
不可选股指标: {results['cannot_select_stocks']}
```

3. ```sql
)
        
        # 测试每个指标
        test_results = []
        successful_indicators = []
        failed_indicators = []
        can_select_indicators = []
        cannot_select_indicators = []
       ...
```

4. ```sql
⚠️  有 {results['cannot_select_stocks']} 个指标无法选股
```

5. ```sql
测试完成: {len(can_select_indicators)}/{len(all_indicators)} 个指标可以选股
```

6. ```sql
选股成功率: {results['selection_success_rate']:.2f}%
```

7. ```sql
可选股指标: {results['can_select_stocks']}
```

8. ```sql
)
            
            # 使用指标验证框架测试
            result = self.framework.validate_single_indicator(indicator_name=indicator_name)
            
            # 检查验证结果
            if result:
          ...
```

9. ```sql
, 0)
                selection_ratio = result.get(
```

10. ```sql
)

def main_testindicatorsrealselection():
```

11. ```sql
)
                selected_count = result.get(
```

12. ```sql
can_select_indicator_list
```

13. ```sql
, 0)
                    }
                }
            else:
                return {
                    "indicator": indicator_name,
                    "status": "error",
                    "can...
```

14. ```sql
def __init__(self):
        self.framework = IndicatorValidationFramework()
        self.results = {}
        
    def test_single_indicator_selection(self, indicator_name: str) -> Dict[str, Any]:
```

15. ```sql
import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__f...
```

16. ```sql
)
                
        # 询问是否继续测试所有指标
        if results['selection_success_rate'] > 80:
            print(f
```

17. ```sql
)
            output_file = f"results/indicator_real_selection_test_{timestamp}.json"
            
        # 确保目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # ...
```

18. ```sql
)
            for indicator in results['cannot_select_indicator_list']:
                print(f
```

19. ```sql
不可选股指标: {results['cannot_select_stocks']}\n
```

20. ```sql
results/indicator_real_selection_test_{timestamp}.json
```

21. ```sql
选股成功率: {results['selection_success_rate']:.2f}%\n\n
```

22. ```sql
\n前10个指标选股成功率较低({results['selection_success_rate']:.1f}%)，建议先修复问题再测试全部指标
```

23. ```sql
cannot_select_indicator_list
```

24. ```sql
try:
        tester = IndicatorRealSelectionTester()
        
        # 先测试前10个指标
        print(
```

25. ```sql
, 0.0)
                
                # 判断是否成功选股
                can_select_stocks = (
                    status ==
```

26. ```sql
selection_success_rate
```

27. ```sql
)
        results = tester.run_comprehensive_test_Selection(test_limit=10)
        
        # 保存结果
        tester.save_results_Selection(results)
        
        # 打印摘要
        print(f
```

28. ```sql
),
            "total_indicators": len(all_indicators),
            "successful_indicators": len(successful_indicators),
            "failed_indicators": len(failed_indicators),
            "can_selec...
```

29. ```sql
and 
                    selected_count > 0 and 
                    selection_ratio > 0
                )
                
                return {
                    "indicator": indicator_name,
  ...
```

30. ```sql
可选股指标: {results['can_select_stocks']}\n
```

31. ```sql
)
            for indicator in results['cannot_select_indicator_list']:
                f.write(f
```

32. ```sql
: False
                }
            }
    
    def run_comprehensive_test_Selection(self, test_limit: int = None) -> Dict[str, Any]:
```

33. ```sql
)
            for indicator in results['can_select_indicator_list']:
                f.write(f
```

### ./scripts/test_new_config.py
- 查询数量: 1

1. ```sql
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_config_manager():
```

### ./scripts/test_obv_indicator.py
- 查询数量: 6

1. ```sql
SELECT 
        date as trade_date,
        code as ts_code,
        open,
        high,
        low,
        close,
        volume,
        turnover as amount
    FROM stock_info WHERE 1=1
    WHERE ...
```

2. ```sql
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path...
```

3. ```sql
]
    
    obv = OBV(signal_period=10)
    
    for stock_code in test_stocks:
        print(f"\n测试股票: {stock_code}")
        
        # 查询股票数据
        query = f"""
        SELECT date, open, high, lo...
```

4. ```sql
SELECT date, open, high, low, close, volume
        FROM stock_info
```

5. ```sql
SELECT 
        date as trade_date,
        code as ts_code,
        open,
        high,
        low,
        close,
        volume,
        turnover as amount
    FROM stock_info
```

6. ```sql
SELECT date, open, high, low, close, volume
        FROM stock_info WHERE 1=1
        WHERE code = '{stock_code}'
          AND level = '日线'
          AND date >= '2025-03-25'
          AND date <= '2...
```

### ./scripts/test_optimized_selection.py
- 查询数量: 6

1. ```sql
import time
import pandas as pd
import os
import sys
from datetime import datetime

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_di...
```

2. ```sql
)
        import traceback
        traceback.print_exc()
        return False


def main_testoptimizedselection():
```

3. ```sql
: 10
            }
        }
        
        # 执行测试
        start_time = time.time()
        
        def progress_callback_Selection_Test_Optimized_Selection(progress, message):
            print(f
```

4. ```sql
]
            start_time = time.time()
            indicators_data = optimizer.calculate_indicators_batch(
                stocks_data=stocks_data,
                indicators=indicators
            )
...
```

5. ```sql
)
        return False


def test_optimized_executor_Selection():
```

6. ```sql
*60)
    
    # 测试批量数据优化器
    batch_test_success = test_batch_data_optimizer()
    
    # 测试优化执行器
    executor_test_success = test_optimized_executor_Selection()
    
    # 总结
    print(
```

### ./scripts/test_optimized_stock_selection.py
- 查询数量: 6

1. ```sql
*80)


def main_testoptimizedstockselection():
```

2. ```sql
self.strategy_manager.save_strategy(strategy_id, self.test_strategy)
            
            # 执行策略
            def progress_callback_Selection_Test_Optimized_Stock_Selection_Test_Optimized_Stock_Sel...
```

3. ```sql
for i in range(len(stock_codes))]
            })
            
        except Exception as e:
            logger.error(f"获取股票样本失败: {e}")
            return pd.Data_frame()
    
    def test_original_ex...
```

4. ```sql
import time
import pandas as pd
import psutil
import os
from datetime import datetime
from typing import Dict, Any

# 添加项目根目录到Python路径
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.abs...
```

5. ```sql
)}")
        
        print("\n" + "="*80)


def main_testoptimizedstockselection():
    """主函数"""
    print("🚀 开始股票选股性能优化测试")
    
    # 创建性能对比器
    comparator = Performance_comparator()
    
    # 运...
```

6. ```sql
),
                progress_callback=progress_callback
            )
            
            # 记录结束状态
            end_time = time.time()
            end_memory = psutil.virtual_memory().percent
     ...
```

### ./scripts/test_real_stock_selection.py
- 查询数量: 22

1. ```sql
: str(e)
            }
    
    def run_comprehensive_test_Selection_Test_Real_Stock_Selection(self) -> Dict[str, Any]:
```

2. ```sql
def __init__(self):
        self.data_access = get_container().resolve(IData_access)
        self.strategy_factory = Strategy_factory()
        self.results = {}
        
    def get_test_stocks_Selec...
```

3. ```sql
)
            
            result = self.test_single_indicator_Selection(indicator_name, test_stocks)
            test_results.append(result)
            
            if result[
```

4. ```sql
]:
                f.write(f"  ❌ {indicator}\n")
                
        logger.info(f"测试报告已保存到: {report_file}")

def main_testrealstockselection():
    """主函数"""
    try:
        tester = Real_stock...
```

5. ```sql
: selected_stocks[:5],  # 只保存前5只
```

6. ```sql
]}")
            return
            
        # 保存结果
        tester.save_results_Selection_Test_Real_Stock_Selection(results)
        
        # 打印摘要
        print(f"\n{
```

7. ```sql
)
        return summary
    
    def save_results_Selection_Test_Real_Stock_Selection(self, results: Dict[str, Any], output_file: str = None):
```

8. ```sql
指标 {indicator_name} 测试完成: 选中 {len(selected_stocks)} 只股票
```

9. ```sql
results/real_stock_selection_test_{timestamp}.json
```

10. ```sql
)

def main_testrealstockselection():
```

11. ```sql
import os
import sys
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
import pandas as pd

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path....
```

12. ```sql
: 1.0
                    }
                ]
            }
            
            strategy = self.strategy_factory.create_strategy(strategy_config)
            
            # 对每只股票进行选股测试
          ...
```

13. ```sql
if len(selected_stocks) > 0 else
```

14. ```sql
)
        
        # 获取测试股票
        test_stocks = self.get_test_stocks_Selection(limit=100)
        if not test_stocks:
            logger.error(
```

15. ```sql
,
                        limit=100
                    )
                    
                    if not stock_data.is_collection or len(stock_data) < 50:
                        continue
           ...
```

16. ```sql
)
            return
            
        # 保存结果
        tester.save_results_Selection_Test_Real_Stock_Selection(results)
        
        # 打印摘要
        print(f
```

17. ```sql
try:
        tester = Real_stock_selection_tester()
        
        # 运行综合测试
        results = tester.run_comprehensive_test_Selection_Test_Real_Stock_Selection()
        
        if
```

18. ```sql
: len(selected_stocks) / len(test_stocks) * 100 if test_stocks else 0,
```

19. ```sql
)
            return []
    
    def test_single_indicator_Selection(self, indicator_name: str, test_stocks: List[str]) -> Dict[str, Any]:
```

20. ```sql
,
                    limit=100
                )
                if stock_data.is_collection and len(stock_data) >= 50:
                    valid_stocks.append(code)
                    if len(valid_...
```

21. ```sql
),
            "total_indicators": len(all_indicators),
            "successful_indicators": len(successful_indicators),
            "failed_indicators": len(failed_indicators),
            "success_r...
```

22. ```sql
)
            output_file = f"results/real_stock_selection_test_{timestamp}.json"
            
        # 确保目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 保存JS...
```

### ./scripts/test_simple_data_flow.py
- 查询数量: 1

1. ```sql
import os
import sys

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from db.unified_data_manager import get_unified_data_manag...
```

### ./scripts/test_true_closed_loop_validation.py
- 查询数量: 24

1. ```sql
: []
        }
        
        for stock_code in selected_stocks:
            try:
                # 获取股票数据
                stock_data = self.data_manager.get_stock_data(
                    stock_co...
```

2. ```sql
: 0.0
        }
        
        try:
            # 步骤1: 使用指标策略进行选股
            logger.info(f"📋 步骤1: 使用{indicator_name}指标策略进行选股")
            strategy_result = self._perform_indicator_strategy_selecti...
```

3. ```sql
)
    
    selection_result = result.get('step1_strategy_selection', {})
    print(f
```

4. ```sql
}
            ]
        
        return strategy_config
    
    def _execute_strategy_selection_Test_True_Closed_Loop_Validation(self, strategy_config: Dict[str, Any], stock_pool: List[str]) -> List[...
```

5. ```sql
)
            strategy_result = self._perform_indicator_strategy_selection(indicator_name)
            result['step1_strategy_selection'] = strategy_result
            
            if strategy_result[...
```

6. ```sql
try:
            selected_stocks = []
            conditions = strategy_config.get('conditions', [])
            
            # 限制测试股票数量，避免超时
            test_stocks = stock_pool[:20]  # 只测试前20只股票
   ...
```

7. ```sql
, {})
    print(f"选股结果: {selection_result.get(
```

8. ```sql
import os
import sys

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

import json
import pandas as pd
import numpy as np
from da...
```

9. ```sql
)
            result['error'] = str(e)
            result['closed_loop_verified'] = False
            return result
    
    def _perform_indicator_strategy_selection(self, indicator_name: str) -> Dic...
```

10. ```sql
] = rsi_values[:len(stock_data)]
                return result_df
            
            # 其他指标的简化实现...
            else:
                logger.warning(f"暂不支持指标: {indicator_id}")
                re...
```

11. ```sql
)
            return None
    
    def _verify_selected_stocks_indicators(self, selected_stocks: List[str], indicator_name: str, 
                                         strategy_conditions: List[Dic...
```

12. ```sql
] = False
            return result
    
    def _perform_indicator_strategy_selection(self, indicator_name: str) -> Dict[str, Any]:
        """
        使用指标策略进行选股
        """
        try:
           ...
```

13. ```sql
verification_result = {
            'total_stocks': len(selected_stocks),
            'verified_stocks': 0,
            'verification_details': []
        }
        
        for stock_code in selected...
```

14. ```sql
try:
            score = 0.0
            
            # 选股成功性 (30%)
            if result['step1_strategy_selection'].get('selection_count', 0) > 0:
                score += 0.3
            
         ...
```

15. ```sql
: len(selected_stocks),
```

16. ```sql
)
                    
                    if stock_data.empty:
                        continue
                    
                    # 检查是否满足条件
                    if self._check_stock_conditions...
```

17. ```sql
)
        
        result = {
            'indicator_name': indicator_name,
            'validation_date': self.validation_date,
            'timestamp': datetime.now().isoformat(),
            'step1...
```

18. ```sql
)
                    continue
            
            return selected_stocks
            
        except Exception as e:
            logger.error(f
```

19. ```sql
)
            return {
                'error': str(e),
                'selected_stocks': [],
                'selection_count': 0,
                'selection_ratio': 0.0
            }
    
    def _...
```

20. ```sql
]:.2f}")
    
    selection_result = result.get(
```

21. ```sql
step1_strategy_selection
```

22. ```sql
try:
            # 生成真正的指标策略（而不是简单的价格条件）
            strategy_config = self._generate_real_indicator_strategy(indicator_name)
            
            # 获取股票池
            stock_pool = self._get_test_s...
```

23. ```sql
🧮 步骤2: 对选出的{strategy_result['selection_count']}只股票重新计算{indicator_name}指标
```

24. ```sql
strategy_config = {
            'strategy_id': f'TRUE_VALIDATION_{indicator_name}',
            'name': f'{indicator_name}指标真实验证策略',
            'description': f'用于真正闭环验证{indicator_name}指标的策略',
      ...
```

### ./scripts/utils/analyze_query_violations.py
- 查询数量: 5

1. ```sql
: get_line_content(content, match.start())
                    })
                
                # 检查没有WHERE条件的查询
                select_pattern = r
```

2. ```sql
) as f:
                    content = f.read()
                    
                # 检查SELECT *
                select_star_matches = list(re.finditer(r
```

3. ```sql
, content, re.IGNORECASE))
                for match in select_star_matches:
                    line_num = content[:match.start()].count(
```

4. ```sql
for match in re.finditer(select_pattern, content, re.IGNORECASE | re.DOTALL):
                    query = match.group(0)
                    if
```

5. ```sql
project_root = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    violations = []
    
    # 检查的目录
    check_dirs = ['utils', 'config', 'db', 'strategy', 'anal...
```

### ./scripts/utils/architecture_validation.py
- 查询数量: 1

1. ```sql
import os
import sys
import time
from pathlib import Path

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from...
```

### ./scripts/utils/batch_architecture_fix.py
- 查询数量: 1

1. ```sql
import os
import sys
import re
import json
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
import subprocess
import logging

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.pa...
```

### ./scripts/utils/batch_sql_migration.py
- 查询数量: 18

1. ```sql
\']UPDATE\s+.*?SET\s+.*?[
```

2. ```sql
]UPDATE\s+.*?SET\s+.*?["\
```

3. ```sql
\']*(?:SELECT|INSERT|UPDATE|DELETE)[^
```

4. ```sql
]DELETE\s+FROM\s+.*?["\
```

5. ```sql
#!/usr/bin/env python3
"""
自动生成的SQL迁移脚本

此脚本用于将分散的SQL查询迁移到统一的SQL管理系统。
请仔细审查每个迁移项，确保迁移的正确性。
"""

import os
import sys

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__fi...
```

6. ```sql
]*(?:SELECT|INSERT|UPDATE|DELETE)[^
```

7. ```sql
import os
import sys
import re
import ast
import logging
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os....
```

8. ```sql
\']DELETE\s+FROM\s+.*?[
```

9. ```sql
import os
import sys

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from db.sql_manager import get_sql_manager, QueryType
from...
```

10. ```sql
]SELECT\s+.*?FROM\s+.*?["\
```

11. ```sql
\']INSERT\s+INTO\s+.*?[
```

12. ```sql
"([^"]*(?:SELECT|INSERT|UPDATE|DELETE)[^"]*?)"
```

13. ```sql
SELECT\s+.*?\s+FROM\s+\w+
```

14. ```sql
]*(?:SELECT|INSERT|UPDATE|DELETE)[^"\
```

15. ```sql
def __init__(self):
        self.sql_manager = get_sql_manager()
        self.sql_patterns = [
            # 基本SQL模式
            r'SELECT\s+.*?\s+FROM\s+\w+',
            r'INSERT\s+INTO\s+\w+',
     ...
```

16. ```sql
]INSERT\s+INTO\s+.*?["\
```

17. ```sql
'([^']*(?:SELECT|INSERT|UPDATE|DELETE)[^']*?)'
```

18. ```sql
sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP']
        text_upper = text.upper()
        
        for keyword in sql_keywords:
            if keyword in text_upper ...
```

### ./scripts/utils/detailed_compliance_check.py
- 查询数量: 8

1. ```sql
Add WHERE clause to UPDATE statements
```

2. ```sql
Add WHERE clause to DELETE statements
```

3. ```sql
suggestions = {
            r'SELECT\s+\*\s+FROM': 'Use specific column names instead of SELECT code, name, price',
            r'stock_info\s+(?!WHERE)': 'Add WHERE clause to stock_info WHERE 1=1 que...
```

4. ```sql
import os
import re
import ast
import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple
import sys

# 添加项目根目录到路径
root_dir = os.path.d...
```

5. ```sql
def __init__(self):
        self.root_dir = Path(root_dir)
        self.violations = {
            'layer_violations': [],
            'db_dependencies': [],
            'naming_violations': [],
     ...
```

6. ```sql
DELETE\s+FROM\s+\w+\s*$
```

7. ```sql
UPDATE\s+\w+\s+SET.*(?!WHERE)
```

8. ```sql
Use specific column names instead of SELECT code, name, price
```

### ./scripts/utils/final_cleanup.py
- 查询数量: 5

1. ```sql
# 跳过模板文件和SQL管理文件
    if 'sql_manager.py' in file_path or 'template' in file_path.lower():
        return content
    
    lines = content.split('\n')
    new_lines = []
    
    for line in lines:
   ...
```

2. ```sql
):
                # 替换SELECT *为具体列名
                line = re.sub(r
```

3. ```sql
SELECT\s+.*?\s+FROM\s+\w+
```

4. ```sql
SELECT code, date, value
```

5. ```sql
)
    new_lines = []
    
    for line in lines:
        original_line = line
        
        # 修复真正的SELECT *查询
        if re.search(r
```

### ./scripts/utils/final_compliance_fix.py
- 查询数量: 4

1. ```sql
SELECT code, name, price
```

2. ```sql
# 修复SELECT code, name, price
    def fix_select_star(match):
        fixes['query_select_star'] += 1
        return match.group(0).replace('SELECT code, name, price', 'SELECT code, name, price')
    
...
```

3. ```sql
, fix_function_name, content)
    
    return content

def fix_query_violations(content: str, fixes: dict) -> str:
    """修复查询违规"""
    
    # 修复SELECT code, name, price
    def fix_select_star(match)...
```

4. ```sql
SELECT\s+.*?\s+FROM\s+\w+(?:\s+[^;]*)?
```

### ./scripts/utils/final_query_fix.py
- 查询数量: 27

1. ```sql
SELECT code, date, strategy_name, score FROM strategy_results
```

2. ```sql
SELECT ts_code, symbol, name, area, industry FROM stock_basic
```

3. ```sql
SELECT code, date, indicator_name, value FROM indicators
```

4. ```sql
SELECT\s+\*\s+FROM\s+stock_daily
```

5. ```sql
SELECT\s+\*\s+FROM\s+(?:daily_)?kline
```

6. ```sql
# 匹配SELECT ... FROM table_name的模式
    pattern = r'SELECT\s+[^;]*?\s+FROM\s+(\w+)(?:\s+[^;]*?)?(?=\s*[;\n]|$)'
    content = re.sub(pattern, add_where_condition, content, flags=re.IGNORECASE | re.DOTAL...
```

7. ```sql
SELECT code, date, buypoint_type, score FROM buypoint_results
```

8. ```sql
SELECT *查询
                content = fix_select_star_queries(content)
                
                # 修复没有WHERE条件的查询
                content = fix_queries_without_where(content)
                
  ...
```

9. ```sql
SELECT code, date, value FROM \1
```

10. ```sql
# 常见的SELECT *替换模式
    replacements = [
        # 股票基础信息查询
        (r'SELECT\s+\*\s+FROM\s+stock_info', 'SELECT code, name, industry, market FROM stock_info'), LIMIT 1000
        (r'SELECT\s+\*\s+FROM\...
```

11. ```sql
SELECT code, date, value FROM \1'),
    ]
    
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
    
    return content

def f...
```

12. ```sql
修复SELECT code, date, value查询
```

13. ```sql
SELECT ts_code, trade_date, open, high, low, close, vol FROM stock_daily
```

14. ```sql
LIMIT 1000"
        else:
            return query + " LIMIT 1000"
    
    # 匹配SELECT ... FROM table_name的模式
    pattern = r
```

15. ```sql
) as f:
                        f.write(content)
                    fixes += 1
                    print(f"修复文件: {py_file}")
                    
            except Exception as e:
                pr...
```

16. ```sql
) as f:
                    content = f.read()
                    
                original_content = content
                
                # 修复SELECT *查询
                content = fix_select_star...
```

17. ```sql
SELECT\s+\*\s+FROM\s+technical_indicators?
```

18. ```sql
SELECT code, name, industry, market FROM stock_info
```

19. ```sql
project_root = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    fixes = 0
    
    # 检查的目录
    check_dirs = ['utils', 'config', 'db', 'strategy', 'analysis',...
```

20. ```sql
SELECT code, date, macd, kdj_k, rsi FROM technical_indicators
```

21. ```sql
)
                
    return fixes

def fix_select_star_queries(content: str) -> str:
```

22. ```sql
SELECT\s+\*\s+FROM\s+stock_basic
```

23. ```sql
SELECT\s+\*\s+FROM\s+indicators?
```

24. ```sql
SELECT code, date, open, high, low, close, volume FROM kline
```

25. ```sql
SELECT\s+\*\s+FROM\s+strategy_results?
```

26. ```sql
SELECT\s+\*\s+FROM\s+buypoint_results?
```

27. ```sql
SELECT\s+\*\s+FROM\s+(\w+)
```

### ./scripts/utils/fix_code_quality_issues.py
- 查询数量: 6

1. ```sql
]}
- **修复内容**: 
  - 替换SELECT code, name, date, level, open, close, high, low, volume为具体字段列表
  - 为stock_info表查询添加WHERE条件
- **效果**: 提高查询性能，避免全表扫描

### 4. 重复名称修复
- **修复数量**: {self.fix_stats[
```

2. ```sql
SELECT code, name, date, level, open, close, high, low, volume
```

3. ```sql
) as f:
                    content = f.read()
                
                original_content = content
                
                # 修复SELECT code, name, date, level, open, close, high, low, ...
```

4. ```sql
query_fixes = 0
        
        for file_path in files:
            try:
                if not os.path.exists(file_path):
                    continue
                
                with open(file...
```

5. ```sql
# 代码质量改进报告

## 修复统计

- **总文件数**: {self.fix_stats['total_files']}
- **成功修复**: {self.fix_stats['successful_fixes']}
- **失败修复**: {self.fix_stats['failed_fixes']}
- **命名规范修复**: {self.fix_stats['naming_fix...
```

6. ```sql
import os
import re
import sys
import ast
from typing import List, Dict, Tuple, Set
from pathlib import Path

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.absp...
```

### ./scripts/utils/fix_hardcoded_configs.py
- 查询数量: 1

1. ```sql
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
...
```

### ./scripts/utils/fix_indicators.py
- 查询数量: 1

1. ```sql
import os
import sys
import re
import importlib
import inspect
from typing import List, Dict, Set, Optional, Any

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path....
```

### ./scripts/utils/fix_layer_violations.py
- 查询数量: 8

1. ```sql
templates = {
        'stock_data': 'SELECT code, name, date, level, open, close, high, low, volume FROM stock_data WHERE code = {code}',
        'indicator_data': 'SELECT code, name, date, level, ope...
```

2. ```sql
import os
import re
import sys
from typing import List, Dict, Tuple
from pathlib import Path

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))...
```

3. ```sql
SELECT code, name, date, level, open, close, high, low, volume FROM market_overview WHERE date = {date}
```

4. ```sql
SELECT code, name, date, level, open, close, high, low, volume FROM market_overview
```

5. ```sql
SELECT code, name, date, level, open, close, high, low, volume FROM indicators
```

6. ```sql
SELECT code, name, date, level, open, close, high, low, volume FROM stock_data WHERE code = {code}
```

7. ```sql
SELECT code, name, date, level, open, close, high, low, volume FROM indicators WHERE code = {code}
```

8. ```sql
SELECT code, name, date, level, open, close, high, low, volume FROM stock_data
```

### ./scripts/utils/fix_remaining_hardcoded_configs.py
- 查询数量: 7

1. ```sql
):
                    # 在主函数开始处添加配置管理器
                    func_indent = len(line) - len(line.lstrip())
                    config_line = " " * (func_indent + 4) + "config_manager = UnifiedConfigMana...
```

2. ```sql
* (init_indent + 4))):
                                    lines.insert(k, config_line)
                                    break
                            break
                elif line.strip().st...
```

3. ```sql
]
            
            # 在导入区域末尾添加新导入
            for imp in reversed(new_imports):
                lines.insert(import_end_idx, imp)
                import_end_idx += 1
            
            #...
```

4. ```sql
# 查找__init__方法的结束位置
                            for k in range(j + 1, len(lines)):
                                if (lines[k].strip() and 
                                    not lines[k].startswith...
```

5. ```sql
import os
import re
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspa...
```

6. ```sql
lines.insert(i + 1, config_line)
                break
        
        return '\n'.join(lines)
    
    def scan_and_fix_files(self) -> None:
```

7. ```sql
in line for line in lines)
        
        if not has_config_import:
            # 添加必要的导入
            new_imports = [
                "from config.unified_config import UnifiedConfigManager"
       ...
```

### ./scripts/utils/fix_sql_queries.py
- 查询数量: 22

1. ```sql
, re.IGNORECASE | re.DOTALL),
            # INSERT语句
            re.compile(r
```

2. ```sql
'''[\s\S]*?SELECT[\s\S]*?FROM[\s\S]*?'''
```

3. ```sql
SELECT\s+.*?FROM\s+stock_info\s+WHERE\s+code\s*=.*?ORDER\s+BY\s+date\s+DESC\s+LIMIT
```

4. ```sql
SELECT\s+code,\s*name,\s*industry.*?FROM\s+stock_info\s+WHERE\s+code\s*=.*?GROUP\s+BY
```

5. ```sql
, re.IGNORECASE | re.DOTALL),
            # UPDATE语句
            re.compile(r
```

6. ```sql
]SELECT\s+.*?FROM\s+\w+.*?["\
```

7. ```sql
\']', re.IGNORECASE | re.DOTALL),
            # INSERT语句
            re.compile(r'[
```

8. ```sql
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
...
```

9. ```sql
return [
            # SELECT语句
            re.compile(r'[
```

10. ```sql
SELECT\s+DISTINCT\s+code.*?FROM\s+stock_info\s+WHERE\s+date\s*=.*?ORDER\s+BY\s+code
```

11. ```sql
SELECT\s+DISTINCT\s+industry.*?FROM\s+stock_info.*?GROUP\s+BY\s+industry
```

12. ```sql
[\s\S]*?SELECT[\s\S]*?FROM[\s\S]*?
```

13. ```sql
]DELETE\s+FROM\s+\w+.*?["\
```

14. ```sql
, re.IGNORECASE | re.DOTALL),
            # DELETE语句
            re.compile(r
```

15. ```sql
\']', re.IGNORECASE | re.DOTALL),
            # UPDATE语句
            re.compile(r'[
```

16. ```sql
]INSERT\s+INTO\s+\w+.*?["\
```

17. ```sql
"""[\s\S]*?SELECT[\s\S]*?FROM[\s\S]*?"""
```

18. ```sql
]UPDATE\s+\w+\s+SET.*?["\
```

19. ```sql
SELECT\s+.*?FROM\s+stock_info\s+WHERE\s+code\s*=.*?AND\s+date\s+BETWEEN.*?ORDER\s+BY\s+date
```

20. ```sql
\']', re.IGNORECASE | re.DOTALL),
            # DELETE语句
            re.compile(r'[
```

21. ```sql
SELECT\s+COUNT\(DISTINCT\s+code\).*?FROM\s+stock_info
```

22. ```sql
return {
            # 股票数据查询
            r'SELECT\s+.*?FROM\s+stock_info\s+WHERE\s+code\s*=.*?AND\s+date\s+BETWEEN.*?ORDER\s+BY\s+date': 
                'executor.get_stock_data(code, start_date, en...
```

### ./scripts/utils/fix_tests.py
- 查询数量: 1

1. ```sql
import os
import re
import sys

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from utils.logger import get_lo...
```

### ./scripts/utils/fix_tests_alt.py
- 查询数量: 1

1. ```sql
import os
import sys

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

# 需要修复的文件列表
TEST_FILES = [
```

### ./scripts/utils/fix_tests_direct.py
- 查询数量: 1

1. ```sql
import os
import sys

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from utils.logger import get_logger

# 获取...
```

### ./scripts/utils/implement_sql_management.py
- 查询数量: 29

1. ```sql
SELECT DISTINCT code 
FROM stock_info
```

2. ```sql
]',
            # DELETE查询
            r'[\'
```

3. ```sql
,
            # INSERT查询
            r
```

4. ```sql
] = """
SELECT code, date, open, high, low, close, volume
FROM stock_info 
WHERE date = (SELECT MAX(date) FROM stock_info WHERE code = %(code)s)
AND code = %(code)s
"""
        
        templates[
```

5. ```sql
SELECT code, date, open, high, low, close, volume
FROM stock_info 
WHERE date = (SELECT MAX(date) FROM stock_info WHERE code = %(code)s)
AND code = %(code)s
```

6. ```sql
"](SELECT\s+.*?(?:\n.*?)*?FROM\s+.*?(?:\n.*?)*?)[\
```

7. ```sql
] = """
SELECT code, date, open, high, low, close, volume
FROM stock_info 
WHERE code IN %(codes)s 
AND date >= %(start_date)s 
AND date <= %(end_date)s
ORDER BY code, date
"""
        
        templa...
```

8. ```sql
def __init__(self):
        self.sql_patterns = [
            # 基本SELECT查询
            r'[\'
```

9. ```sql
] = """
SELECT MIN(date) as min_date, MAX(date) as max_date
FROM stock_info
WHERE code = %(code)s
"""
        
        return templates

class SQLMigrationTool:
    """SQL迁移工具"""
    
    def __init__...
```

10. ```sql
,
            # UPDATE查询
            r
```

11. ```sql
SELECT code, date, open, high, low, close, volume
FROM stock_info 
WHERE code IN %(codes)s 
AND date >= %(start_date)s 
AND date <= %(end_date)s
ORDER BY code, date
```

12. ```sql
SELECT MIN(date) as min_date, MAX(date) as max_date
FROM stock_info
WHERE code = %(code)s
```

13. ```sql
SELECT code, COUNT(*) as record_count
FROM stock_info 
WHERE date >= %(start_date)s
GROUP BY code
ORDER BY record_count DESC
```

14. ```sql
,
            # DELETE查询
            r
```

15. ```sql
]',
            # INSERT查询
            r'[\'
```

16. ```sql
SELECT code, date, open, high, low, close, volume
FROM stock_info 
WHERE code = %(code)s 
AND date >= %(start_date)s 
AND date <= %(end_date)s
ORDER BY date
```

17. ```sql
] = """
SELECT code, date, open, high, low, close, volume
FROM stock_info 
WHERE code = %(code)s 
AND date >= %(start_date)s 
AND date <= %(end_date)s
ORDER BY date
"""
        
        templates[
```

18. ```sql
,
            # 复杂SELECT查询（多行）
            r
```

19. ```sql
SELECT MAX(date) FROM stock_info
```

20. ```sql
)
logger = logging.getLogger(__name__)

class SQLQueryExtractor:
    """SQL查询提取器"""
    
    def __init__(self):
        self.sql_patterns = [
            # 基本SELECT查询
            r
```

21. ```sql
SELECT DISTINCT code 
FROM stock_info 
WHERE date >= %(min_date)s
ORDER BY code
```

22. ```sql
]',
            # UPDATE查询
            r'[\'
```

23. ```sql
]',
            # 复杂SELECT查询（多行）
            r'[\'
```

24. ```sql
SELECT MIN(date) as min_date, MAX(date) as max_date
FROM stock_info
```

25. ```sql
SELECT code, COUNT(*) as record_count
FROM stock_info
```

26. ```sql
SELECT code, date, open, high, low, close, volume
FROM stock_info
```

27. ```sql
import os
import sys
import re
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path....
```

28. ```sql
] = """
SELECT code, COUNT(*) as record_count
FROM stock_info 
WHERE date >= %(start_date)s
GROUP BY code
ORDER BY record_count DESC
"""
        
        templates[
```

29. ```sql
] = """
SELECT DISTINCT code 
FROM stock_info 
WHERE date >= %(min_date)s
ORDER BY code
"""
        
        templates[
```

### ./scripts/utils/massive_compliance_fix.py
- 查询数量: 12

1. ```sql
SELECT\s+\*\s+FROM\s+stock_info\b
```

2. ```sql
SELECT code, name, industry FROM stock_info WHERE 1=1
```

3. ```sql
SELECT code, name, industry FROM stock_info
```

4. ```sql
UPDATE\s+(\w+)\s+SET\s+([^W]+)(?!WHERE)
```

5. ```sql
UPDATE \1 SET \2 WHERE 1=1
```

6. ```sql
SELECT code, name, price FROM \1 WHERE 1=1
```

7. ```sql
DELETE\s+FROM\s+(\w+)\s*$
```

8. ```sql
def __init__(self):
        self.root_dir = Path(root_dir)
        self.fixes_applied = {
            'naming_violations': 0,
            'code_duplications': 0,
            'layer_violations': 0,
   ...
```

9. ```sql
import os
import re
import ast
import json
from pathlib import Path
from collections import defaultdict, Counter
import sys

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os....
```

10. ```sql
DELETE FROM \1 WHERE 1=1
```

11. ```sql
SELECT code, name, price FROM \1 WHERE 1=1',
            r'\bstock_info\s+(?!WHERE)(?!LIMIT)': 'stock_info WHERE 1=1 ',
            r'DELETE\s+FROM\s+(\w+)\s*$': r'DELETE FROM \1 WHERE 1=1',
         ...
```

12. ```sql
SELECT\s+\*\s+FROM\s+(\w+)\b
```

### ./scripts/utils/optimize_global_singletons.py
- 查询数量: 8

1. ```sql
class_name = module_info['class']
        
        comment = f'''
# 注意：{class_name}现在通过依赖注入容器管理
# 可以在应用启动时预注册：
# container.register_singleton({class_name}, {class_name})
'''
        
        # 在文件末尾添加...
```

2. ```sql
# 在文件末尾添加注释
        content += comment
        
        return content
    
    def update_container_configuration(self):
        """更新容器配置"""
        logger.info("更新容器配置...")
        
        contain...
```

3. ```sql
+ 
                          content[insert_pos:])
            else:
                # 在文件开头添加
                content =
```

4. ```sql
: 0
        }
    
    def optimize_all_singletons(self) -> Dict[str, int]:
        """优化所有全局单例"""
        logger.info("开始优化全局单例模块...")
        
        for module_info in self.singleton_modules:
    ...
```

5. ```sql
)
        
        for module_info in self.singleton_modules:
            self.optimize_singleton_module(module_info)
        
        # 更新容器配置
        self.update_container_configuration()
        
 ...
```

6. ```sql
matches = list(re.finditer(import_pattern, content))
            if matches:
                # 在最后一个导入后添加
                last_import = matches[-1]
                insert_pos = last_import.end()
     ...
```

7. ```sql
if 'from utils.dependency_injection import get_service_Optimize_Global_Singletons_Optimize_Global_Singletons' not in content:
            # 在现有导入后添加容器导入
            import_pattern = r'(from *? import ...
```

8. ```sql
import os
import sys
import re
from typing import List, Dict, Tuple

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_...
```

### ./scripts/utils/optimize_performance.py
- 查询数量: 1

1. ```sql
OPTIMIZE TABLE stock_selection_result FINAL
```

### ./scripts/utils/precise_query_fix.py
- 查询数量: 16

1. ```sql
fixes = {'select_star': 0, 'no_where': 0}
    
    # 1. 修复SELECT *查询
    content, select_star_fixes = fix_select_star_queries(content, file_path)
    fixes['select_star'] = select_star_fixes
    
    ...
```

2. ```sql
SELECT *查询
    content, select_star_fixes = fix_select_star_queries(content, file_path)
    fixes['select_star'] = select_star_fixes
    
    # 2. 修复没有WHERE条件的查询
    content, no_where_fixes = fix_no_w...
```

3. ```sql
LIMIT 1000"
        else:
            # 其他表，添加通用限制
            condition = " LIMIT 1000"
        
        fixes += 1
        return query + condition
    
    content = re.sub(select_pattern, add_wher...
```

4. ```sql
- SELECT code, date, value 修复: {fixes['select_star_fixed']}
```

5. ```sql
project_root = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    fixes = {
        'select_star_fixed': 0,
        'no_where_fixed': 0,
        'files_modifie...
```

6. ```sql
修复SELECT code, date, value查询
```

7. ```sql
fixes = 0
    
    # 查找SELECT ... FROM table但没有WHERE/LIMIT/ORDER BY的查询
    select_pattern = r'SELECT\s+[^;]*?\s+FROM\s+(\w+)(?:\s+[^;]*?)?(?=\s*[;\n]|$)'
    
    def add_where_condition(match):
     ...
```

8. ```sql
def replace_select_star(match):
        nonlocal fixes
        table_name = match.group(1)
        
        # 根据表名提供合适的列名
        column_mapping = {
```

9. ```sql
] = select_star_fixes
    
    # 2. 修复没有WHERE条件的查询
    content, no_where_fixes = fix_no_where_queries(content, file_path)
    fixes[
```

10. ```sql
fixes += 1
        return query + condition
    
    content = re.sub(select_pattern, add_where_condition, content, flags=re.IGNORECASE | re.DOTALL)
    
    return content, fixes

def main():
```

11. ```sql
: 0}
    
    # 1. 修复SELECT *查询
    content, select_star_fixes = fix_select_star_queries(content, file_path)
    fixes[
```

12. ```sql
)
    
    total_fixes = fixes['select_star_fixed'] + fixes['no_where_fixed']
    print(f
```

13. ```sql
SELECT {columns} FROM {table_name}
```

14. ```sql
fixes = 0
    
    # 查找所有SELECT *模式
    select_star_pattern = r'SELECT\s+\*\s+FROM\s+(\w+)'
    
    def replace_select_star(match):
        nonlocal fixes
        table_name = match.group(1)
        ...
```

15. ```sql
] = no_where_fixes
    
    return content, fixes

def fix_select_star_queries(content: str, file_path: str) -> Tuple[str, int]:
    """修复SELECT code, date, value查询"""
    fixes = 0
    
    # 查找所有SEL...
```

16. ```sql
SELECT\s+[^;]*?\s+FROM\s+(\w+)(?:\s+[^;]*?)?(?=\s*[;\n]|$)
```

### ./scripts/utils/priority_compliance_fix.py
- 查询数量: 8

1. ```sql
import os
import re
import ast
import json
from pathlib import Path
from collections import defaultdict
import sys

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.absp...
```

2. ```sql
SELECT code, name, industry FROM stock_info WHERE 1=1
```

3. ```sql
SELECT code, name, industry FROM stock_info
```

4. ```sql
def __init__(self):
        self.root_dir = Path(root_dir)
        self.fixes_applied = {
            'db_dependencies': 0,
            'layer_violations': 0,
            'query_violations': 0,
      ...
```

5. ```sql
SELECT code, name, price FROM \1 WHERE 1=1
```

6. ```sql
SELECT code, name, price FROM \1 WHERE 1=1',
            r'stock_info\s+(?!WHERE)': 'stock_info WHERE 1=1 ',
        }
        
        # 关键命名修复 - 只修复最常见的违规
        self.critical_naming_fixes = {
    ...
```

7. ```sql
SELECT\s+\*\s+FROM\s+stock_info
```

8. ```sql
SELECT\s+\*\s+FROM\s+(\w+)
```

### ./scripts/utils/refactor_database_dependencies.py
- 查询数量: 3

1. ```sql
)
        
        if new_imports:
            # 在导入区域末尾插入新导入
            for import_stmt in reversed(new_imports):
                lines.insert(import_end_idx + 1, import_stmt)
        
        retur...
```

2. ```sql
import os
import re
import sys
from typing import List, Dict, Set, Tuple
from pathlib import Path

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file_...
```

3. ```sql
lines = content.split('\n')
        
        # 查找导入区域
        import_end_idx = 0
        for i, line in enumerate(lines):
            if (line.strip().startswith('import ') or 
                line.st...
```

### ./scripts/utils/refactor_sql_statements.py
- 查询数量: 5

1. ```sql
SELECT code, name, date, level, open, close, high, low, volume FROM table
```

2. ```sql
]}

## 重构内容

### 1. strategy/strategy_manager.py
- **问题**: SQL语句硬编码在业务代码中
- **修复**: 使用SQL管理器获取SQL模板
- **效果**: 提高SQL的可维护性

### 2. bin/stock_analysis.py
- **问题**: 动态SQL拼接存在安全风险
- **修复**: 使用参数化SQL模板
- **...
```

3. ```sql
# SQL语句重构报告

## 重构统计

- **总文件数**: {stats['total_files']}
- **成功重构**: {stats['successful_refactors']}
- **失败重构**: {stats['failed_refactors']}
- **SQL模板提取**: {stats['sql_extracted']}
- **SQL语句替换**: {sta...
```

4. ```sql
SELECT code, name, date, level, open, close, high, low, volume FROM table WHERE id =
```

5. ```sql
import os
import sys

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from utils.logger import get_logger
from ...
```

### ./scripts/utils/simple_compliance_check.py
- 查询数量: 4

1. ```sql
, content, re.IGNORECASE):
        violations += 1
        
    # 检查没有WHERE条件的查询
    select_pattern = r
```

2. ```sql
violations = 0
    
    # 检查SELECT code, name, price
    if re.search(r'SELECT\s+\*', content, re.IGNORECASE):
        violations += 1
        
    # 检查没有WHERE条件的查询
    select_pattern = r'SELECT\s+.*?...
```

3. ```sql
, content):
        violations += 1
        
    return violations

def check_query_violations(content: str) -> int:
    """检查查询违规"""
    violations = 0
    
    # 检查SELECT code, name, price
    if re...
```

4. ```sql
for match in re.finditer(select_pattern, content, re.IGNORECASE | re.DOTALL):
        query = match.group(0)
        if
```

### ./scripts/utils/simple_hardcoded_config_fix.py
- 查询数量: 3

1. ```sql
import os
import re
import sys
import logging

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

# 配置日志
logging.b...
```

2. ```sql
)
            continue
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
      ...
```

3. ```sql
):
                            import_idx = i
                            break
                    
                    lines.insert(import_idx,
```

### ./scripts/utils/simple_sql_migration.py
- 查询数量: 6

1. ```sql
def __init__(self):
        self.sql_patterns = [
            r'SELECT\s+.*?\s+FROM\s+\w+',
            r'INSERT\s+INTO\s+\w+',
            r'UPDATE\s+\w+\s+SET',
            r'DELETE\s+FROM\s+\w+',
 ...
```

2. ```sql
]*(?:SELECT|INSERT|UPDATE|DELETE)[^
```

3. ```sql
"([^"]*(?:SELECT|INSERT|UPDATE|DELETE)[^"]*?)"
```

4. ```sql
SELECT\s+.*?\s+FROM\s+\w+
```

5. ```sql
text_upper = text.upper()
        sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE']
        
        for keyword in sql_keywords:
            if keyword in text_upper and ('FROM' in text_upper o...
```

6. ```sql
'([^']*(?:SELECT|INSERT|UPDATE|DELETE)[^']*?)'
```

### ./scripts/utils/simplified_integration_test.py
- 查询数量: 3

1. ```sql
import os
import sys
import time
import logging
from pathlib import Path

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ...
```

2. ```sql
in dev_config
            
            return True
        except Exception as e:
            logger.error(f"缓存配置测试失败: {e}")
            return False
    
    def _test_cache_operations(self) -> bool:...
```

3. ```sql
: time.time()}
            
            # 设置
            cache.set(test_key, test_value)
            
            # 获取
            retrieved = cache.get(test_key)
            assert retrieved == test_...
```

### ./scripts/utils/smart_compliance_check.py
- 查询数量: 12

1. ```sql
"):
            continue
            
        # 检查SELECT *
        if re.search(r
```

2. ```sql
violations = 0
    
    # 查找SELECT *模式，但排除注释和字符串
    lines = content.split('\n')
    for line in lines:
        stripped = line.strip()
        
        # 跳过注释行
        if stripped.startswith('#') or ...
```

3. ```sql
SELECT *查询（排除注释和字符串中的）
    violations += check_real_select_star(content)
    
    # 2. 检查真正缺少WHERE条件的查询
    violations += check_real_no_where(content, file_path)
    
    return violations

def check_...
```

4. ```sql
, content):
        violations += 1
        
    return violations

def check_query_violations_smart(content: str, file_path: str) -> int:
    """智能检查查询违规"""
    violations = 0
    
    # 1. 检查真正的SELE...
```

5. ```sql
"):
            continue
            
        # 检查SELECT FROM模式
        if re.search(r
```

6. ```sql
检查真正的SELECT code, date, value查询
```

7. ```sql
):
            continue
            
        # 检查SELECT *
        if re.search(r'SELECT\s+\*', stripped, re.IGNORECASE):
            # 进一步检查是否在注释中
            if not ('#' in stripped and stripped.inde...
```

8. ```sql
violations = 0
    
    # 跳过模板文件和SQL管理文件
    if 'sql_manager.py' in file_path or 'template' in file_path.lower():
        return 0
    
    # 查找SELECT ... FROM table但没有WHERE/LIMIT/ORDER BY的查询
    line...
```

9. ```sql
):
            continue
            
        # 检查SELECT FROM模式
        if re.search(r'SELECT\s+.*?\s+FROM\s+\w+', stripped, re.IGNORECASE):
            # 检查是否有WHERE、LIMIT等条件
            if not any(key...
```

10. ```sql
') or stripped.startswith("'''"):
            continue
            
        # 检查SELECT FROM模式
        if re.search(r'SELECT\s+.*?\s+FROM\s+\w+', stripped, re.IGNORECASE):
            # 检查是否有WHERE、LIMI...
```

11. ```sql
in file_path.lower():
        return 0
    
    # 查找SELECT ... FROM table但没有WHERE/LIMIT/ORDER BY的查询
    lines = content.split(
```

12. ```sql
violations = 0
    
    # 1. 检查真正的SELECT *查询（排除注释和字符串中的）
    violations += check_real_select_star(content)
    
    # 2. 检查真正缺少WHERE条件的查询
    violations += check_real_no_where(content, file_path)
    ...
```

### ./scripts/utils/system_integration_test.py
- 查询数量: 6

1. ```sql
optimize_stock_selection
```

2. ```sql
]
            logger.info(f"系统集成测试完成，成功率: {success_rate:.2%}")
            
            return success_rate >= 0.8  # 80%以上通过率认为成功
            
        except Exception as e:
            logger.error(...
```

3. ```sql
try:
            from db.performance_optimizer import Performance_optimizer
            
            optimizer = Performance_optimizer()
            
            # 验证基本方法存在
            if not hasattr(...
```

4. ```sql
import os
import sys
import time
import traceback
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(o...
```

5. ```sql
try:
            from strategy.base_strategy import Base_strategy
            
            # 验证基类存在基本方法
            if not hasattr(BaseStrategy, 'select'):
                return False
            
  ...
```

6. ```sql
: time.time()}
            
            cache.set(test_key, test_value)
            retrieved_value = cache.get(test_key)
            
            if retrieved_value != test_value:
                ret...
```

### ./scripts/utils/test_performance_optimization.py
- 查询数量: 10

1. ```sql
)
        start_time = time.time()
        result_no_cache = optimizer_without_cache.optimize_stock_selection(**test_params)
        time_no_cache = time.time() - start_time
        
        # 测试首次有缓存...
```

2. ```sql
)
    
    try:
        # 设置测试环境
        data_access, cache_service = setup_test_environment()
        
        # 创建性能优化器
        config = Optimization_config(
            batch_size=100,
            ...
```

3. ```sql
)
        start_time = time.time()
        result_second_cache = optimizer_with_cache.optimize_stock_selection(**test_params)
        time_second_cache = time.time() - start_time
        
        # 计算...
```

4. ```sql
: str(e)
        }


def test_memory_optimization():
    """测试内存优化"""
    logger.info("开始内存优化测试")
    
    try:
        # 设置测试环境
        data_access, cache_service = setup_test_environment()
        
...
```

5. ```sql
import os
import sys
import time
import json
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.pa...
```

6. ```sql
)
    
    try:
        # 设置测试环境
        data_access, cache_service = setup_test_environment()
        
        # 创建性能优化器
        config = Optimization_config(
            batch_size=50,
            m...
```

7. ```sql
)
        start_time = time.time()
        result_first_cache = optimizer_with_cache.optimize_stock_selection(**test_params)
        time_first_cache = time.time() - start_time
        
        # 测试二次...
```

8. ```sql
)
            
            test_codes = get_test_stock_codes(scale)
            
            start_time = time.time()
            try:
                result = optimizer.optimize_stock_selection(
    ...
```

9. ```sql
: 0.5}
        }
        
        # 测试无缓存性能
        logger.info("测试无缓存性能")
        start_time = time.time()
        result_no_cache = optimizer_without_cache.optimize_stock_selection(**test_params)
  ...
```

10. ```sql
: str(e)
        }


def test_scalability():
    """测试可扩展性"""
    logger.info("开始可扩展性测试")
    
    try:
        # 设置测试环境
        data_access, cache_service = setup_test_environment()
        
        ...
```

### ./scripts/utils/test_unified_cache.py
- 查询数量: 4

1. ```sql
)
    
    # 验证删除后不存在
    cached_data_after_delete = cache_layer.get(key)
    print(f
```

2. ```sql
)
    
    # 测试删除缓存
    deleted = cache_layer.delete(key)
    print(f
```

3. ```sql
)
    
    assert cached_data_after_delete is None,
```

4. ```sql
import os
import sys
import time
import json
from typing import Dict, Any

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,...
```

### ./scripts/utils/ultimate_compliance_fix.py
- 查询数量: 1

1. ```sql
def __init__(self):
        self.root_dir = Path(root_dir)
        self.fixes_applied = {
            'naming_violations': 0,
            'layer_violations': 0,
            'code_duplications': 0,
   ...
```

### ./scripts/validate_buypoint_strategy.py
- 查询数量: 33

1. ```sql
])
            
            # 4. 分析选股结果
            selection_analysis = self._analyze_selection_results(selected_stocks, strategy)
            
            # 5. 回测验证（如果启用）
            backtest_result...
```

2. ```sql
)
    
    def _analyze_selection_results(self, selected_stocks: pd.DataFrame, strategy: dict) -> dict:
```

3. ```sql
, {}))} 个行业"
        
        return analysis
    
    def _run_backtest(self, selected_stocks: pd.DataFrame, start_date: str, days: int) -> dict:
        """运行回测"""
        if selected_stocks is None...
```

4. ```sql
if selection_rate < 0.005:
                assessment[
```

5. ```sql
)
            
            stock_codes = selected_stocks['stock_code'].tolist()
            
            # 获取回测期间的价格数据
            returns = []
            for stock_code in stock_codes[:20]:  # 限制回测股...
```

6. ```sql
]}")
        print(f"选出股票: {selection[
```

7. ```sql
) as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 打印结果摘要
    print("\n" + "=" * 60)
    print("📊 验证结果摘要")
    print("=" * 60)
    
    if results["validation_status"] == "...
```

8. ```sql
]}")
        print(f"选股率: {selection[
```

9. ```sql
])
            
            # 3. 执行策略选股
            selected_stocks = self._execute_strategy(strategy, stock_pool, validation_config[
```

10. ```sql
:
        selection = results[
```

11. ```sql
in selected_stocks.columns:
            industry_dist = selected_stocks[
```

12. ```sql
]
            analysis["score_distribution"] = {
                "mean": float(scores.mean()),
                "median": float(scores.median()),
                "std": float(scores.std()),
           ...
```

13. ```sql
]
                        stock_return = (end_price - start_price) / start_price
                        returns.append(stock_return)
                        
                except Exception as e:
  ...
```

14. ```sql
] = industry_dist
        
        # 评分分布
        if 'score' in selected_stocks.columns:
            scores = selected_stocks['score']
            analysis[
```

15. ```sql
in selected_stocks.columns:
            scores = selected_stocks[
```

16. ```sql
策略验证完成，选出 {len(selected_stocks) if selected_stocks is not None else 0} 只股票
```

17. ```sql
] for stock in stock_pool]
            
            # 执行策略
            def progress_callback_Strategy(progress, message):
                if progress % 0.1 < 0.01:  # 每10%打印一次
                    logg...
```

18. ```sql
import sys
import os
import json
import argparse
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.pat...
```

19. ```sql
选出股票: {selection['selected_count']}
```

20. ```sql
策略选出 {len(selected_stocks)} 只股票，涵盖 {len(analysis.get('industry_distribution', {}))} 个行业
```

21. ```sql
if selected_stocks is None or len(selected_stocks) == 0:
            return {
```

22. ```sql
: []
            }
        
        analysis = {}
        
        # 行业分布
        if 'industry' in selected_stocks.columns:
            industry_dist = selected_stocks['industry'].value_counts().to_di...
```

23. ```sql
: float(scores.max())
            }
        
        # 前10名股票
        top_stocks = selected_stocks.head(10).to_dict('records')
        analysis[
```

24. ```sql
: selection_analysis,
```

25. ```sql
]}")
        print(f"股票池大小: {selection[
```

26. ```sql
, False):
                backtest_results = self._run_backtest(
                    selected_stocks, 
                    validation_config[
```

27. ```sql
) if selected_stocks is not None else []
                },
                "analysis": selection_analysis,
                "backtest": backtest_results,
                "validation_status": "success"...
```

28. ```sql
)
        analysis["top_stocks"] = top_stocks
        
        # 策略条件分析
        conditions = strategy.get("conditions", [])
        condition_summary = {
            "total_conditions": len(conditions...
```

29. ```sql
)
            
            selected_stocks = self.strategy_executor.execute_strategy_by_id(
                strategy_id=temp_strategy_id,
                strategy_manager=self.strategy_manager,
      ...
```

30. ```sql
: []
        }
        
        # 选股率评分
        selection_rate = results[
```

31. ```sql
elif 0.005 <= selection_rate <= 0.2:
            assessment[
```

32. ```sql
选股率: {selection['selection_rate']:.2%}
```

33. ```sql
股票池大小: {selection['total_pool_size']}
```

### ./scripts/verify_indicator_registration.py
- 查询数量: 2

1. ```sql
: missing_in_category
            }
            
            print(f"{category} 类别: {available_in_category} 可用, {len(missing_in_category)} 未注册\n")
        
        # 更新总体结果
        self.verification_r...
```

2. ```sql
)
        
        # 更新总体结果
        self.verification_results.update({
            'total_available': total_available,
            'total_registered': len(self.registered_indicators),
            'tot...
```

### ./simple_architecture_check.py
- 查询数量: 2

1. ```sql
)
        
        sql_patterns = [
            r'SELECT\s+.*\s+FROM\s+',
            r'INSERT\s+INTO\s+',
            r'UPDATE\s+.*\s+SET\s+',
            r'DELETE\s+FROM\s+'
        ]
        
     ...
```

2. ```sql
SELECT\s+.*\s+FROM\s+
```

### ./strategy/base_strategy.py
- 查询数量: 3

1. ```sql
return self._error is not None
    
    @abc.abstractmethod
    def select_Strategy_Base_Strategy(self, universe: List[str], *args, **kwargs) -> pd.Data_frame:
```

2. ```sql
self._parameters.update(params)
    
    def has_result_Strategy(self) -> bool:
```

3. ```sql
try:
            self._result = self.select_Strategy_Base_Strategy(universe, *args, **kwargs)
            self._error = None
            return self._result
        except Exception as e:
            ...
```

### ./strategy/batch_data_optimizer.py
- 查询数量: 3

1. ```sql
])
                results.update(macd_results)
            
            return results
            
        except Exception as e:
            logger.error(f"计算股票 {stock_code} 指标时出错: {e}")
          ...
```

2. ```sql
if stock_data.empty:
            return None
        
        try:
            results = {}
            
            # 获取最新的数据点
            latest = stock_data.iloc[-1]
            results['latest_pri...
```

3. ```sql
)
        
        # 分批查询，避免单次查询过多数据
        for i in range(0, len(stock_codes), self.batch_size):
            batch_codes = stock_codes[i:i + self.batch_size]
            batch_results = self._query_...
```

### ./strategy/batch_optimizer.py
- 查询数量: 10

1. ```sql
)")
            
            if conditions:
                where_clause = " OR ".join(conditions)
                return f"""
                SELECT DISTINCT code
                FROM stock_info WHER...
```

2. ```sql
SELECT DISTINCT code
                FROM stock_info
```

3. ```sql
SELECT DISTINCT code
                FROM stock_info WHERE 1=1
                WHERE level = '日线' AND ({where_clause})
```

4. ```sql
try:
            start_time = time.time()
            
            # 计算开始日期
            start_date = self.data_manager.get_previous_trade_date(end_date, days_back)
            
            # 分批处理，避免单次...
```

5. ```sql
SELECT DISTINCT code, industry
            FROM stock_info WHERE 1=1
            WHERE code IN ('{codes_str}')
            AND level = '日线'
```

6. ```sql
".join(stock_codes)
            query = f"""
            SELECT DISTINCT code, industry
            FROM stock_info WHERE 1=1
            WHERE code IN (
```

7. ```sql
SELECT code, trade_date, open, high, low, close, volume, amount, pct_chg
            FROM stock_info
```

8. ```sql
SELECT code, trade_date, open, high, low, close, volume, amount, pct_chg
            FROM stock_info WHERE 1=1
            WHERE code IN ('{codes_str}')
            AND level = '日线'
            AND tr...
```

9. ```sql
".join(stock_codes)
            query = f"""
            SELECT code, trade_date, open, high, low, close, volume, amount, pct_chg
            FROM stock_info WHERE 1=1
            WHERE code IN (
```

10. ```sql
SELECT DISTINCT code, industry
            FROM stock_info
```

### ./strategy/breakout_strategy.py
- 查询数量: 2

1. ```sql
: 5             # 最低股价要求
        }
    
    def select_Strategy(self, universe: List[str], *args, **kwargs) -> pd.Data_frame:
        """
        执行横盘突破选股策略
        
        Args:
            universe...
```

2. ```sql
super().__init___67(name, description)
        
        # 设置默认参数
        self._parameters = {
            'consolidation_days': 10,  # 横盘整理的最短天数
            'price_range_pct': 0.05,   # 横盘整理期间的价格波动范围百...
```

### ./strategy/dual_ma_strategy.py
- 查询数量: 8

1. ```sql
}")
            
            except Exception as e:
                logger.error(f"处理股票 {code} 时出错: {e}")
        
        # 转换为DataFrame
        result_df = pd.Data_frame(selected_stocks)
        
  ...
```

2. ```sql
]
        
        # 自动获取最近日期
        latest_date = self.data_manager.get_latest_trading_date()
        if end_date is None:
            end_date = latest_date
        
        if start_date is None:
...
```

3. ```sql
].iloc[-(lookback_days+5):-lookback_days].mean()
                            
                            if current_volume < avg_volume * volume_ratio:
                                has_breakout = ...
```

4. ```sql
: None         # 结束日期，默认为None表示使用最近交易日期
        }
        
        # 初始化数据管理器
        self.data_manager = get_unified_data_manager()
    
    def select_Strategy_Dual_Ma_Strategy(self, universe: List[...
```

5. ```sql
)
        
        # 转换为DataFrame
        result_df = pd.Data_frame(selected_stocks)
        
        if len(result_df) > 0:
            # 添加排序
            result_df = result_df.sort_values(by='breako...
```

6. ```sql
)
                
                # 获取股票日线数据
                df = self.data_manager.get_daily_data(code, start_date, end_date)
                
                # 跳过没有数据的股票
                if df is No...
```

7. ```sql
)
        
        # 设置默认参数
        self._parameters = {
            'short_period': 5,       # 短期均线周期
            'long_period': 10,       # 长期均线周期
            'volume_ratio': 1.5,     # 成交量放大倍数
    ...
```

8. ```sql
# 更新参数
        if kwargs:
            self._parameters.update(kwargs)
        
        # 提取参数
        short_period = self._parameters['short_period']
        long_period = self._parameters['long_perio...
```

### ./strategy/enhanced_base_strategy.py
- 查询数量: 12

1. ```sql
self._parameters.update({
            'start_date': None,  # 开始日期，None表示自动计算
            'end_date': None,    # 结束日期，None表示使用最新数据
            'period': self.default_period,  # 数据周期
            'lookba...
```

2. ```sql
SELECT DISTINCT code 
            FROM stock_info WHERE 1=1
            WHERE 1=1
```

3. ```sql
SELECT MAX(date) as max_date FROM stock_info WHERE date >=
```

4. ```sql
weight: float = 1.0  # 权重
    
    def get_unique_key(self) -> str:
        """获取唯一标识键（周期+指标）"""
        return f"{self.period}_{self.indicator_name}_{hash(str(self.parameters_Enhanced_Base_Strategy))...
```

5. ```sql
, 1000) if filters else 1000
            sql += f" LIMIT {max_results}"
            
            # 执行查询
            result_Enhanced_Base_Strategy = self.data_access.query(sql, params)
            
   ...
```

6. ```sql
SELECT MAX(date) as max_date FROM stock_info
```

7. ```sql
)
    
    @abc.abstractmethod
    def select_Enhanced_Base_Strategy(self, universe: Optional[List[str]] = None, *args, **kwargs) -> pd.Data_frame:
```

8. ```sql
try:
            # 从stock_info表查询最新日期
            sql = 'SELECT MAX(date) as max_date FROM stock_info WHERE date >= '2020-01-01' LIMIT 1'
            result_Enhanced_Base_Strategy = self.data_access.q...
```

9. ```sql
SELECT DISTINCT code 
            FROM stock_info
```

10. ```sql
)
            
            # 清除之前的结果和错误
            self._result = None
            self._error = None
            
            # 执行选股逻辑
            result_Enhanced_Base_Strategy = self.select_Enhance...
```

11. ```sql
)
            
            logger.debug(f"获取股票 {stock_code} 数据成功，共 {len(df)} 条记录")
            return df
            
        except Exception as e:
            logger.error_Enhanced_Base_Strategy(f"获...
```

12. ```sql
self._parameters.update(params)
        logger.debug(f
```

### ./strategy/institutional_strategy.py
- 查询数量: 7

1. ```sql
)
        
        return selected_stocks
    
    def _score_behavior_pattern(self, pattern: str) -> float:
```

2. ```sql
] # 关注的行为模式
        }
        
        # 更新默认参数
        if params:
            default_params.update(params)
        
        super().__init___83(
            name=
```

3. ```sql
]}天)(+{abs_score:.1f})")
                    except Exception as e:
                        logger.error(f"预测吸筹完成时间出错: {e}")
                
                # 记录评分和理由
                stock_scores[cod...
```

4. ```sql
,
            params=default_params
        )
        
        self.institutional_behavior = Institutional_behavior()
    
    def select_Strategy_Institutional_Strategy(self, data_dict: Dict[str, pd....
```

5. ```sql
selected_stocks = []
        
        # 策略评分
        stock_scores = {}
        
        for code, data in data_dict.items():
            try:
                # 检查数据长度
                if len(data) < se...
```

6. ```sql
] >= 10:  # 分数阈值
                selected_stocks.append(code)
                logger.info(f
```

7. ```sql
])}")
        
        return selected_stocks
    
    def _score_behavior_pattern(self, pattern: str) -> float:
        """
        对行为模式进行评分
        
        Args:
            pattern: 行为模式
        ...
```

### ./strategy/momentum_strategy.py
- 查询数量: 6

1. ```sql
)
        
        # 转换为DataFrame
        result_df = pd.Data_frame(selected_stocks)
        
        logger.info(f
```

2. ```sql
)
                
                # 创建股票公式对象
                f = Stock_formula(code, start=start_date, end=end_date)
                
                # 跳过没有数据的股票
                if f.data_day.history...
```

3. ```sql
# 结束日期
        }
    
    def select_Strategy_Momentum_Strategy(self, universe: List[str], *args, **kwargs) -> pd.Data_frame:
        """
        执行动量选股策略
        
        Args:
            universe: ...
```

4. ```sql
]
        
        # 存储选股结果
        selected_stocks = []
        
        # 遍历股票池，应用选股条件
        total_stocks = len(universe)
        logger.info(f"开始对 {total_stocks} 只股票进行动量策略筛选")
        
        fo...
```

5. ```sql
)
        
        # 设置默认参数
        self._parameters = {
            'min_turnover_rate': 1.5,  # 最小换手率
            'require_elasticity': True,  # 是否要求弹性
            'require_daily_absorption': True, ...
```

6. ```sql
# 更新参数
        if kwargs:
            self._parameters.update(kwargs)
        
        # 提取参数
        min_turnover_rate = self._parameters['min_turnover_rate']
        require_elasticity = self._param...
```

### ./strategy/multi_period_strategy.py
- 查询数量: 4

1. ```sql
# 日线指标条件
        daily_conditions = [
            Indicator_condition(
                indicator_name='MA',
                period='1d',
                parameters={'period': 20},
                cond...
```

2. ```sql
,
                weight=0.4
            )
        ]
        
        # 添加所有条件
        for condition in daily_conditions + weekly_conditions + hourly_conditions:
            self.add_indicator_conditi...
```

3. ```sql
: list(period_data.keys())
            }
            
            # 添加技术指标详情
            result.update(self._calculate_technical_details(daily_data))
            
            return result
           ...
```

4. ```sql
try:
            # 多周期数据收集
            period_data = {}
            period_scores = {}
            
            # 获取各周期数据
            periods_to_analyze = ['1d', '1w', '1h']
            
            f...
```

### ./strategy/optimized_strategy_executor.py
- 查询数量: 6

1. ```sql
)
            
            result_df = self._process_results(results, strategy_plan)
            
            # 9. 更新性能统计
            total_time = time.time() - start_time
            self._update_per...
```

2. ```sql
, 0)
        if max_results > 0 and len(result_df) > max_results:
            result_df = result_df.head(max_results)
        
        return result_df.reset_index(drop=True)
    
    def _update_perf...
```

3. ```sql
indicators = set()
        
        for condition in conditions:
            condition_type = condition.get('type', '')
            
            # 根据条件类型确定需要的指标
            if condition_type == 'indic...
```

4. ```sql
)
        
        # 添加基础指标
        indicators.update([
```

5. ```sql
if not results:
            return pd.Data_frame()
        
        # 转换为DataFrame
        result_df = pd.Data_frame(results)
        
        # 按评分排序
        if 'score' in result_df.columns:
        ...
```

6. ```sql
, [])
            required_indicators = self._extract_required_indicators(conditions)
            
            indicators_data = self.batch_optimizer.calculate_indicators_batch(
                stocks...
```

### ./strategy/rebound_strategy.py
- 查询数量: 2

1. ```sql
: 0.05     # 股价距离均线最小距离要求
        }
    
    def select_Strategy_Rebound_Strategy(self, universe: List[str], *args, **kwargs) -> pd.Data_frame:
        """
        执行回踩反弹选股策略
        
        Args:
  ...
```

2. ```sql
super().__init___74(name, description)
        
        # 设置默认参数
        self._parameters = {
            'ma_period': 5,          # 均线周期
            'touch_threshold': 0.02, # 接触均线阈值，如0.02表示2%以内都算接触
...
```

### ./strategy/result_filter.py
- 查询数量: 2

1. ```sql
))
        
        if "condition_count" in df.columns:
            return df[(df["condition_count"] >= min_count) & (df["condition_count"] <= max_count)]
        elif "satisfied_conditions" in df.col...
```

2. ```sql
] = result[conditions_field].apply(count_conditions)
            
            # 添加指标类型统计
            indicator_counts = result[conditions_field].apply(count_indicator_types)
            
            #...
```

### ./strategy/strategy_combiner.py
- 查询数量: 1

1. ```sql
if not strategy_results:
            return pd.Data_frame()
            
        # 提取所有选中的股票代码
        all_stocks = set()
        for result in strategy_results.values():
            all_stocks.update...
```

### ./strategy/strategy_condition_evaluator.py
- 查询数量: 2

1. ```sql
try:
            # 获取默认参数
            default_params = self.parameter_validator.get_default_parameters(indicator_id)

            # 合并参数
            merged_params = default_params.copy()
            m...
```

2. ```sql
时出错: {e}")
            return False
    
    def _standardize_condition(self, condition: Dict[str, Any]) -> Dict[str, Any]:
        """
        标准化条件格式

        Args:
            condition: 原始条件

    ...
```

### ./strategy/strategy_evaluator.py
- 查询数量: 4

1. ```sql
))
            current_date += timedelta(days=1)
        
        # 每月选股一次
        selection_dates = []
        for date in dates:
            if date[8:10] ==
```

2. ```sql
selection_date = date
                
                # 模拟后续表现
                future_return_5d = np.random.normal(0.02, 0.05)  # 均值2%，标准差5%
                future_return_10d = np.random.normal(0.03,...
```

3. ```sql
:  # 每月1日
                selection_dates.append(date)
        
        # 生成模拟数据
        data = []
        for date in selection_dates:
            # 模拟选出的股票数量
            stocks_count = np.random.ran...
```

4. ```sql
历史数据中缺少selection_date或future_return_10d列
```

### ./strategy/strategy_executor.py
- 查询数量: 1

1. ```sql
)
                final_result = all(evaluation_results)
            
            # 如果不满足条件，直接返回None
            if not final_result:
                return None
            
            # 4. 获取股票的最新价...
```

### ./strategy/strategy_generator.py
- 查询数量: 1

1. ```sql
import os
import sys
import json
from typing import Dict, List, Any, Optional

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

f...
```

### ./strategy/strategy_manager.py
- 查询数量: 21

1. ```sql
)
            return False
    
    @exception_handler(reraise=False, default_return=False)
    def _delete_strategy_from_file(self, strategy_id: str) -> bool:
```

2. ```sql
合并策略配置
        
        Args:
            original: 原始配置
            updates: 更新配置
            
        Returns:
            Dict[str, Any]: 合并后的配置
```

3. ```sql
success = True
        
        # 从数据库删除
        if not self._delete_strategy_from_db(strategy_id):
            success = False
        
        # 从文件删除
        if not self._delete_strategy_from_file(...
```

4. ```sql
try:
            strategies = []
            
            if not os.path.exists(self.strategy_dir):
                return strategies
            
            for filename in os.listdir(self.strategy_...
```

5. ```sql
def _merge_strategy_configs(self, original: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
```

6. ```sql
query = self.sql_manager.build_dynamic_query(
                'list_strategies', 
                conditions=conditions,
                order_by=['updated_at DESC']
            )
            
       ...
```

7. ```sql
)
        
        # 合并配置
        merged_config = self._merge_strategy_configs(existing_strategy, strategy_config)
        
        # 更新时间
        merged_config['strategy']['update_time'] = datetime.n...
```

8. ```sql
] = True
        
        # 保存到数据库
        if not self._save_strategy_to_db(strategy_config):
            raise Exception("保存策略到数据库失败")
        
        # 保存到文件（可选）
        if save_to_file:
          ...
```

9. ```sql
UPDATE strategy_definitions SET
```

10. ```sql
# 从数据库获取策略列表
        db_strategies = self._list_strategies_from_db(filters)
        
        # 从文件获取策略列表
        file_strategies = self._list_strategies_from_files(filters)
        
        # 合并并去重（以数...
```

11. ```sql
) as f:
                    return json.load(f)
                    
            return None
            
        except Exception as e:
            logger.error(f"从文件获取策略失败: {e}")
            return ...
```

12. ```sql
UPDATE strategy_definitions SET is_active = 0, updated_at = NO WHERE 1=1_w()
            WHERE strategy_id = %(strategy_id)s
```

13. ```sql
]] = strategy
        
        return list(strategy_dict.values())
    
    @exception_handler(reraise=False, default_return=False)
    @performance_monitor(threshold_seconds=1.0)
    def delete_strat...
```

14. ```sql
merged = copy.deepcopy(original)
        
        def merge_dict(target, source):
            for key, value in source.items():
                if key in target and isinstance(target[key], dict) and i...
```

15. ```sql
]
            for field in required_fields:
                if field not in strategy:
                    logger.error(f"策略配置缺少必需字段: {field}")
                    return False
            
           ...
```

16. ```sql
)
        return strategy_id
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=1.0)
    def update_strategy(self, strategy_id: str, strategy_config: Dict[str, Any], ...
```

17. ```sql
SELECT strategy_id, config, created_at, updated_at
            FROM strategy_definitions
```

18. ```sql
)
        
        # 生成策略ID（如果没有提供）
        if 'strategy' not in strategy_config or 'id' not in strategy_config['strategy']:
            strategy_id = self._generate_strategy_id()
            if 'stra...
```

19. ```sql
SELECT strategy_id, config, created_at, updated_at
            FROM strategy_definitions 
            WHERE is_active = 1
```

20. ```sql
] = config
                            
                        strategies.append(strategy_info)
                        
                    except (json.JSONDecode_error, IOError) as e:
            ...
```

21. ```sql
)
            return []
    
    @exception_handler(reraise=False, default_return=False)
    def _delete_strategy_from_db(self, strategy_id: str) -> bool:
```

### ./strategy/strategy_optimizer.py
- 查询数量: 7

1. ```sql
].update(params)
    
    def _evaluate_strategy_performance(self, strategy_config: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """
        评估策略性能
        
        Args:
            strateg...
```

2. ```sql
import os
import sys
import json
import copy
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Set, Callable
from itertools import combinations
import ...
```

3. ```sql
try:
            # 使用策略执行器执行策略
            execution_result = self.strategy_executor.execute_strategy(strategy_config)
            
            if not execution_result or 'selected_stocks' not in exec...
```

4. ```sql
if 'strategy' not in config:
            config['strategy'] = {}
        
        if 'parameters' not in config['strategy']:
            config['strategy']['parameters'] = {}
        
        config['...
```

5. ```sql
not in execution_result:
                return None
            
            # 计算性能指标
            selected_stocks = execution_result[
```

6. ```sql
]:
                weighted_sum = sum(perf[metric] * weight 
                                 for perf, weight in zip(individual_performances, weights))
                combined_performance[metric] = ...
```

7. ```sql
try:
            individual_performances = []
            
            # 评估每个策略的性能
            for config in strategy_configs:
                performance = self._evaluate_strategy_performance(config)...
```

### ./strategy/strategy_parser.py
- 查询数量: 1

1. ```sql
)}")
            conditions = self._parse_conditions(strategy["conditions"])
            logger.debug(f"解析后的条件: {conditions}")
            
            # 解析过滤器
            filters = self._parse_filter...
```

### ./system_integration_analyzer.py
- 查询数量: 2

1. ```sql
return {
            'required_fields': ['stock_code', 'stock_name', 'industry', 'price', 'change_pct', 'score'],
            'optional_fields': ['match_details', 'selection_date'],
            'data_...
```

2. ```sql
import sys
import os
import inspect
from typing import Dict, List, Any, Optional, Set
from datetime import datetime

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(...
```

### ./test_aroon_fix.py
- 查询数量: 4

1. ```sql
SELECT date, open, high, low, close, volume
        FROM stock_data
```

2. ```sql
: len(valid_scores[(valid_scores >= 80) & (valid_scores <= 100)])
        }
        
        print(f"\n评分分布:")
        for range_name, count in score_ranges.items():
            percentage = count / l...
```

3. ```sql
SELECT date, open, high, low, close, volume
        FROM stock_data 
        WHERE code = '000001' 
        ORDER BY date DESC 
        LIMIT 100
```

4. ```sql
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_d...
```

### ./test_enhanced_dmi.py
- 查询数量: 4

1. ```sql
SELECT 
                date,
                open,
                high,
                low,
                close,
                volume
            FROM stock_info
```

2. ```sql
SELECT 
                date,
                open,
                high,
                low,
                close,
                volume
            FROM stock_info WHERE 1=1
            WHERE cod...
```

3. ```sql
]
    
    # 创建指标实例
    indicator = Enhanced_dMI(period=14, adx_period=14, adaptive=True)
    
    total_signals = 0
    total_tests = 0
    
    for code in test_codes:
        print(f"\n测试股票: {code}...
```

4. ```sql
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_d...
```

### ./test_enhanced_macd.py
- 查询数量: 4

1. ```sql
SELECT 
                date,
                open,
                high,
                low,
                close,
                volume
            FROM stock_info
```

2. ```sql
SELECT 
                date,
                open,
                high,
                low,
                close,
                volume
            FROM stock_info WHERE 1=1
            WHERE cod...
```

3. ```sql
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_d...
```

4. ```sql
]
    
    # 创建指标实例
    indicator = Enhanced_mACD(fast_period=12, slow_period=26, signal_period=9)
    
    total_signals = 0
    total_tests = 0
    
    for code in test_codes:
        print(f"\n测试股...
```

### ./test_enhanced_trix.py
- 查询数量: 4

1. ```sql
SELECT 
                date,
                open,
                high,
                low,
                close,
                volume
            FROM stock_info
```

2. ```sql
SELECT 
                date,
                open,
                high,
                low,
                close,
                volume
            FROM stock_info WHERE 1=1
            WHERE cod...
```

3. ```sql
]
    
    # 创建指标实例
    indicator = Enhanced_tRIX(n=12, m=9)
    
    total_signals = 0
    total_tests = 0
    
    for code in test_codes:
        print(f"\n测试股票: {code}")
        print("-" * 40)
  ...
```

4. ```sql
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_d...
```

### ./test_indicator_selection.py
- 查询数量: 1

1. ```sql
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_d...
```

### ./test_kdj_actual_method.py
- 查询数量: 1

1. ```sql
import os
import sys
import pandas as pd
import numpy as np

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)

from indicators.kdj import KDJ
from indica...
```

### ./test_kdj_get_pattern.py
- 查询数量: 1

1. ```sql
import os
import sys
import pandas as pd
import numpy as np

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)

from indicators.kdj import KDJ
from indica...
```

### ./test_kdj_registry.py
- 查询数量: 1

1. ```sql
import sys
import os
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)

import pandas as pd
import numpy as np
from indicators.kdj import KDJ
from indicators.pattern_r...
```

### ./test_kdj_score_debug.py
- 查询数量: 1

1. ```sql
import os
import sys
import pandas as pd
import numpy as np

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)

from indicators.kdj import KDJ
from indica...
```

### ./test_momentum_fix.py
- 查询数量: 1

1. ```sql
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_d...
```

### ./test_pattern_id_issue.py
- 查询数量: 1

1. ```sql
import sys
import os
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)

from indicators.kdj import KDJ
from indicators.pattern_registry import Pattern_registry

def te...
```

### ./test_pattern_registration.py
- 查询数量: 1

1. ```sql
import sys
import os
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)

from indicators.kdj import KDJ
from indicators.pattern_registry import Pattern_registry

def te...
```

### ./test_production_db.py
- 查询数量: 4

1. ```sql
import sys
import os
import time
from datetime import datetime

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)

from config.database_config_manager imp...
```

2. ```sql
SELECT COUNT(*) as count FROM system
```

3. ```sql
}")
        
        # 创建数据管理器
        print("\n🔍 创建数据管理器...")
        data_manager = Unified_data_manager()
        
        # 测试连接
        print("🔗 测试数据库连接...")
        start_time = time.time()
    ...
```

4. ```sql
SELECT COUNT(*) as count FROM system.databases
```

### ./test_roc_fix.py
- 查询数量: 1

1. ```sql
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_d...
```

### ./test_three_indicators_fix.py
- 查询数量: 1

1. ```sql
import sys
import os
import numpy as np
import pandas as pd

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)

from indicators.cci import CCI
from ...
```

### ./test_vortex.py
- 查询数量: 3

1. ```sql
SELECT 
            trade_date as date,
            ts_code as code,
            open,
            high,
            low,
            close,
            vol as volume,
            amount as turnover
 ...
```

2. ```sql
SELECT 
            trade_date as date,
            ts_code as code,
            open,
            high,
            low,
            close,
            vol as volume,
            amount as turnover
 ...
```

3. ```sql
import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Any

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)

from indicato...
```

### ./tests/conftest.py
- 查询数量: 1

1. ```sql
import os
import pytest
import tempfile
import shutil
from unittest.mock import patch, Magic_mock
import sys

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.pa...
```

### ./tests/end_to_end/comprehensive_stock_selection_test.py
- 查询数量: 48

1. ```sql
average_selection_rate
```

2. ```sql
)

        test_result = {
            'strategy_id': strategy_id,
            'strategy_name': strategy_name,
            'success': False,
            'execution_time': 0,
            'stocks_proces...
```

3. ```sql
)

        if self.test_stats['avg_selection_rate'] < 0.01:
            recommendations.append(
```

4. ```sql
- **选股率合理性**: {quality.get('selection_rate_reasonableness', 'unknown')}\n\n
```

5. ```sql
in strategy_name:
            # 突破策略：选择5-8%的股票
            selection_rate = 0.06
        elif
```

6. ```sql
)}")
    print()

    try:
        # 创建测试实例
        test_framework = Comprehensive_stock_selection_test()

        # 运行综合测试
        report = test_framework.run_comprehensive_test_Test_Comprehensive_St...
```

7. ```sql
return integration_status

    def save_test_report(self, report: Dict[str, Any], output_dir: str = "test_reports") -> str:
        """保存测试报告"""
        try:
            # 创建输出目录
            os.makedi...
```

8. ```sql
)
    print()

    try:
        # 创建测试实例
        test_framework = Comprehensive_stock_selection_test()

        # 运行综合测试
        report = test_framework.run_comprehensive_test_Test_Comprehensive_Stock...
```

9. ```sql
- **选股率**: {details['selection_rate']}\n
```

10. ```sql
most_selective_strategy
```

11. ```sql
stock_selection_test_report_{timestamp}.json
```

12. ```sql
total_stocks_selected
```

13. ```sql
metrics = {
            'system_stability': 'stable' if self.test_stats['success_rate'] >= 0.8 else 'unstable',
            'avg_execution_time_per_strategy': (
                self.test_stats['total_...
```

14. ```sql
)
            else:
                test_result['success'] = True  # 执行成功但无结果也算成功
                test_result['stocks_processed'] = len(test_data['stock_codes'][:50])
                test_result['stoc...
```

15. ```sql
selection_rate_reasonableness
```

16. ```sql
)
                self.test_stats['failed_strategies'] += 1
                self.test_stats['error_details'].append({
                    'strategy_id': strategy_id,
                    'error': str(e...
```

17. ```sql
return quality_assessment

    def _generate_recommendations_Comprehensive_Stock_Selection_Test(self) -> List[str]:
        """生成改进建议"""
        recommendations = []

        if self.test_stats[
```

18. ```sql
,
                'total_stocks_processed': self.test_stats['total_stocks_processed'],
                'total_stocks_selected': self.test_stats['total_stocks_selected'],
                'average_selec...
```

19. ```sql
,
                'industry': industry,
                'price': round(base_price, 2),
                'change_pct': round(change_pct, 2),
                'score': round(score, 1),
                'ma...
```

20. ```sql
in strategy_name:
            # 均值回归策略：选择8-12%的股票
            selection_rate = 0.10
        elif
```

21. ```sql
},
            'strategy_details': {},
            'performance_analysis': self.test_stats['performance_metrics'],
            'quality_assessment': self._assess_overall_quality(),
            'recomm...
```

22. ```sql
)
            return None

    def _generate_markdown_report_Comprehensive_Stock_Selection_Test(self, report: Dict[str, Any], output_file: str):
```

23. ```sql
)

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from strategy.strategy_executor import Strategy_executor
from...
```

24. ```sql
- 选股率合理性: {quality.get('selection_rate_reasonableness', 'unknown')}
```

25. ```sql
] = str(e)

        return quality

    def run_comprehensive_test_Test_Comprehensive_Stock_Selection_Test(self) -> Dict[str, Any]:
        """运行综合测试"""
        logger.info("=" * 80)
        logger.in...
```

26. ```sql
)


def main_comprehensivestockselectiontest():
```

27. ```sql
in strategy_name:
            # 多因子策略：选择20-25%的股票
            selection_rate = 0.22
        else:
            selection_rate = 0.10

        # 计算选中股票数量
        selected_count = max(1, int(len(stock_co...
```

28. ```sql
: self._generate_recommendations_Comprehensive_Stock_Selection_Test(),
```

29. ```sql
* 80)

        start_time = time.time()

        # 准备测试数据
        test_data = self.prepare_test_data_Test()

        # 创建测试策略
        test_strategies = self.create_test_strategies()

        # 更新统计信息
...
```

30. ```sql
], 1):
                    f.write(f"{i}. {recommendation}\n")

                f.write("\n---\n\n")
                f.write("*本报告由选股系统端到端综合测试框架自动生成*\n")

        except Exception as e:
            lo...
```

31. ```sql
- **选中股票总数**: {summary['total_stocks_selected']}\n
```

32. ```sql
🔍 选中股票总数: {summary['total_stocks_selected']}
```

33. ```sql
📊 平均选股率: {summary['average_selection_rate']}
```

34. ```sql
stock_selection_test_report_{timestamp}.md
```

35. ```sql
import sys
import os
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import warnings
warning...
```

36. ```sql
in strategy_name:
            # 趋势跟踪策略：选择10-15%的股票
            selection_rate = 0.12
        elif
```

37. ```sql
in strategy_name:
            # ZXM策略：选择15-20%的股票
            selection_rate = 0.18
        elif
```

38. ```sql
) as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)

            # 生成Markdown报告
            markdown_file = os.path.join(output_dir, f"stock_selection_test_report_{...
```

39. ```sql
- **平均选股率**: {summary['average_selection_rate']}\n\n
```

40. ```sql
] for r in successful_results]
            if avg_selection_rates:
                avg_rate = sum(avg_selection_rates) / len(avg_selection_rates)
                if 0.05 <= avg_rate <= 0.3:  # 5%-30%的...
```

41. ```sql
)
        elif self.test_stats['avg_selection_rate'] > 0.5:
            recommendations.append(
```

42. ```sql
)

        strategy = strategy_config['strategy']
        stock_codes = test_data['stock_codes'][:20]  # 限制模拟股票数量

        # 模拟选股结果
        results = []

        # 根据策略类型生成不同的选股结果
        strategy_nam...
```

43. ```sql
)
            quality['analysis_error'] = str(e)

        return quality

    def run_comprehensive_test_Test_Comprehensive_Stock_Selection_Test(self) -> Dict[str, Any]:
```

44. ```sql
)
            self.strategy_executor = None
            self.strategy_manager = None
            self.data_manager = None
            self.use_mock_data = True

        # 测试统计
        self.test_stats ...
```

45. ```sql
quality_assessment = {
            'system_reliability': 'excellent' if self.test_stats['success_rate'] >= 0.9 else
                                 'good' if self.test_stats['success_rate'] >= 0.7 el...
```

46. ```sql
]]
        if successful_results:
            avg_selection_rates = [r[
```

47. ```sql
- **选中股票数**: {details['stocks_selected']}\n
```

48. ```sql
,
                'stocks_processed': result['stocks_processed'],
                'stocks_selected': result['stocks_selected'],
                'selection_rate': f
```

### ./tests/end_to_end/enhanced_real_data_test.py
- 查询数量: 32

1. ```sql
SELECT
                    code, date, close, volume,
                    (close - open) / open as daily_change,
                    volume / 1000000 as volume_millions
                FROM stock_info...
```

2. ```sql
AND close > {10 + query_id}
                        LIMIT 100
                    """
                elif query_id % 3 == 1:
                    query = f"""
                        SELECT code, AVG(...
```

3. ```sql
SELECT code, date, open, high, low, close, volume
                FROM stock_info
```

4. ```sql
SELECT 
                    code,
                    COUNT(*) as record_count,
                    AVG(close) as avg_price,
                    MAX(high) as max_price,
                    MIN(low) as...
```

5. ```sql
GROUP BY code
                        LIMIT 50
                    """
                else:
                    query = f"""
                        SELECT COUNT(*) as count
                        F...
```

6. ```sql
SELECT code, date, open, high, low, close, volume
                FROM stock_info WHERE 1=1
                WHERE date >= '2024-06-01'
                AND code IN ({','.join([f"'{code}'" for code, _ i...
```

7. ```sql
)
            query_time = time.time() - query_start
            
            stock_list = [(row[0], row[1]) for row in result.result_rows]
            
            performance_results['queries'].appe...
```

8. ```sql
result = self.client.query(agg_query)
            query_time = time.time() - query_start
            
            agg_records = len(result.result_rows)
            
            performance_results['qu...
```

9. ```sql
SELECT code, close, volume 
                        FROM stock_info WHERE 1=1
                        WHERE date = '2024-06-20' 
                        AND close > {10 + query_id}
                   ...
```

10. ```sql
SELECT
                    code, date, close, volume,
                    (close - open) / open as daily_change,
                    volume / 1000000 as volume_millions
                FROM stock_info...
```

11. ```sql
SELECT with window functions and calculations
```

12. ```sql
SELECT DISTINCT code, name FROM stock_info WHERE date >= 2020-01-01
```

13. ```sql
SELECT OHLCV data for recent period'
            })
            
            logger.info(f"最近数据查询完成: {recent_records}条记录，耗时: {query_time:.3f}秒")
            
            # 3. 测试聚合查询
            query_...
```

14. ```sql
SELECT aggregated statistics'
            })
            
            logger.info(f"聚合查询完成: {agg_records}条记录，耗时: {query_time:.3f}秒")
            
            # 4. 测试复杂条件查询
            query_start = ti...
```

15. ```sql
result = self.client.query(complex_query)
            query_time = time.time() - query_start
            
            complex_records = len(result.result_rows)
            
            performance_res...
```

16. ```sql
SELECT aggregated statistics
```

17. ```sql
SELECT code, AVG(close) as avg_price
                        FROM stock_info
```

18. ```sql
)
        }
        
        def execute_concurrent_query(query_id: int) -> Dict[str, Any]:
            """执行单个并发查询"""
            thread_start = time.time()
            
            try:
            ...
```

19. ```sql
SELECT code, AVG(close) as avg_price
                        FROM stock_info WHERE 1=1
                        WHERE date >= '2024-06-{10 + query_id % 10:02d}'
                        GROUP BY code
  ...
```

20. ```sql
: []
        }
        
        start_time = time.time()
        
        try:
            # 1. 测试基本股票列表查询
            query_start = time.time()
            result = self.client.query(f"""
           ...
```

21. ```sql
SELECT COUNT(*) as count
                        FROM stock_info
```

22. ```sql
SELECT code, date, open, high, low, close, volume
                FROM stock_info WHERE 1=1
                WHERE date >= '2024-06-01'
                AND code IN ({','.join([f
```

23. ```sql
SELECT COUNT(*) as count
                        FROM stock_info WHERE 1=1
                        WHERE volume > {1000000 * (query_id + 1)}
                        AND date >= '2024-06-01'
```

24. ```sql
SELECT DISTINCT code, name 
                FROM stock_info WHERE 1=1
                WHERE date >= '2024-01-01' 
                LIMIT {stock_limit}
```

25. ```sql
SELECT OHLCV data for recent period
```

26. ```sql
SELECT DISTINCT code, name 
                FROM stock_info
```

27. ```sql
SELECT 
                    code,
                    COUNT(*) as record_count,
                    AVG(close) as avg_price,
                    MAX(high) as max_price,
                    MIN(low) as...
```

28. ```sql
SELECT with window functions and calculations'
            })
            
            logger.info(f"复杂查询完成: {complex_records}条记录，耗时: {query_time:.3f}秒")
            
            # 计算总体统计
            ...
```

29. ```sql
SELECT DISTINCT code, name FROM stock_info
```

30. ```sql
]])
            
            performance_results.update({
```

31. ```sql
result = self.client.query(recent_data_query)
            query_time = time.time() - query_start
            
            recent_records = len(result.result_rows)
            
            performance_...
```

32. ```sql
)
            
            # 计算总体统计
            total_time = time.time() - start_time
            total_records = sum(q['records'] for q in performance_results['queries'])
            avg_query_time =...
```

### ./tests/end_to_end/real_data_performance_test.py
- 查询数量: 29

1. ```sql
,
                'industry': industry,
                'price': round(base_price, 2),
                'change_pct': round(change_pct, 2),
                'score': round(score, 1),
                'ma...
```

2. ```sql
)

        # 初始化结果
        test_result = {
            'strategy_id': strategy_id,
            'strategy_name': strategy_name,
            'success': False,
            'execution_time': 0,
          ...
```

3. ```sql
- 总选中股票数: {strategy_perf['total_stocks_selected']}
```

4. ```sql
in strategy_name:
            selection_rate = 0.10  # 10%
        elif
```

5. ```sql
* 80)

        start_time = time.time()

        # 获取真实数据
        test_data = self.get_real_stock_data(limit=1000)

        # 创建性能测试策略
        test_strategies = self.create_performance_test_strategies...
```

6. ```sql
)
            else:
                test_result['success'] = True  # 执行成功但无结果也算成功
                test_result['stocks_processed'] = test_data['total_stocks']
                test_result['stocks_select...
```

7. ```sql
)
                
                return {
                    'data_source': 'real_clickhouse',
                    'stock_codes': selected_stocks.tolist(),
                    'start_date': start_d...
```

8. ```sql
total_stocks_selected
```

9. ```sql
- **总选中股票数**: {concurrent_perf['total_stocks_selected']}\n
```

10. ```sql
]}',
            'duration': query_time,
            'records': selected_count
        })

        # 生成模拟结果（基于真实股票代码）
        import random
        random.seed(42)  # 固定随机种子

        selected_stocks =...
```

11. ```sql
: selected_count
        })

        # 生成模拟结果（基于真实股票代码）
        import random
        random.seed(42)  # 固定随机种子

        selected_stocks = random.sample(stock_codes, selected_count)

        for i, st...
```

12. ```sql
))
            
            result = client.query("""
                SELECT DISTINCT code 
                FROM stock_info WHERE 1=1
                WHERE date >=
```

13. ```sql
in strategy_name:
            selection_rate = 0.04  # 4%
        elif
```

14. ```sql
SELECT DISTINCT code 
                FROM stock_info
```

15. ```sql
)

        # 启动系统级性能监控
        system_monitor = Performance_monitor_Test()
        system_monitor.start_monitoring_Test()

        start_time = time.time()

        try:
            # 使用线程池并发执行策略
    ...
```

16. ```sql
].unique()
                selected_stocks = unique_stocks[:limit]  # 限制股票数量
                
                logger.info(f"成功获取真实股票数据: {len(selected_stocks)} 只股票，查询耗时: {query_time:.2f}秒")
           ...
```

17. ```sql
)

        strategy = strategy_config['strategy']
        stock_codes = test_data['stock_codes']

        # 模拟基于真实数据的选股结果
        results = []

        # 根据策略类型和真实数据特征生成结果
        strategy_name = stra...
```

18. ```sql
: len(selected_stocks),
```

19. ```sql
in strategy_name:
            selection_rate = 0.08  # 8%
        elif
```

20. ```sql
- **选中股票数**: {result['stocks_selected']}\n
```

21. ```sql
- **选股率**: {result['selection_rate']:.2%}\n
```

22. ```sql
SELECT DISTINCT code 
                FROM stock_info WHERE 1=1
                WHERE date >= '2024-01-01' 
                LIMIT 500
```

23. ```sql
- **总选中股票数**: {strategy_perf['total_stocks_selected']}\n\n
```

24. ```sql
# 计算数据库性能指标
        db_query_times = [q['duration'] for q in self.test_stats['database_query_times']]
        db_metrics = {
            'total_queries': len(db_query_times),
            'avg_query_ti...
```

25. ```sql
self.data_manager = get_unified_data_manager()
        self.strategy_executor = Strategy_executor(max_workers=8, cache_enabled=True)
        self.strategy_manager = Strategy_manager()
        self.per...
```

26. ```sql
in strategy_name:
            selection_rate = 0.12  # 12%
        else:
            selection_rate = 0.05

        # 计算选中股票数量
        selected_count = max(1, int(len(stock_codes) * selection_rate))

...
```

27. ```sql
: selected_stocks.tolist(),
```

28. ```sql
in strategy_name:
            selection_rate = 0.06  # 6%
        elif
```

29. ```sql
)
        
        try:
            # 获取最近的交易日期
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
         ...
```

### ./tests/framework/layered_testing_framework.py
- 查询数量: 2

1. ```sql
# 基本的业务逻辑验证
        if result.empty:
            return False
        
        # 检查数值列是否有无限值
        numeric_columns = result.select_dtypes(include=[np.number]).columns
        for col in numeric_colu...
```

2. ```sql
]
        for col in required_columns:
            if col not in result.columns:
                return False
            if result[col].dtype != bool:
                return False
        
        re...
```

### ./tests/helper/indicator_adapter.py
- 查询数量: 6

1. ```sql
# 提取需要的数据
        open_prices = data['open'].values
        high_prices = data['high'].values
        low_prices = data['low'].values
        close_prices = data['close'].values
        volumes = data...
```

2. ```sql
].values
        
        # 调用ZXM指标计算
        result_dict = self.zxm_indicator.calculate_Adapter_Indicator_Adapter_Indicator_Adapter_indicatoradapter(
            open_prices, high_prices, low_prices,...
```

3. ```sql
):
            scores = self.indicator.calculate_raw_score_Adapter_Indicator_Adapter(result)
        else:
            scores = pd.Series(50, index=data.index)  # 默认中性评分
        
        # 如果指定了信号字段，使...
```

4. ```sql
self.indicator = indicator
        self.threshold = threshold
        self.signal_field = signal_field
        
    def select(self, data: pd.Data_frame) -> List[Dict[str, Any]]:
```

5. ```sql
# 计算指标
        result = self.indicator.calculate_Adapter_Indicator_Adapter_Indicator_Adapter_indicatoradapter(data)
        
        # 获取原始评分
        if hasattr(self.indicator, 'calculate_raw_score'):...
```

6. ```sql
}
            
            selected.append(selected_item)
            
        return selected


class Multi_indicator_adapter:
    """多指标组合适配器"""
    
    def __init__(self, indicators: List, weights...
```

### ./tests/integration/business_workflow_test.py
- 查询数量: 13

1. ```sql
)
    
    def test_stock_selection_workflow(self) -> Dict[str, Any]:
```

2. ```sql
)
            
            # 步骤3: 模拟买点分析 - 简化版本
            buypoint_results = []
            if selected_stocks:
                from analysis.buypoints.period_data_processor import Period_data_proce...
```

3. ```sql
: len(selected_stocks),
```

4. ```sql
: {}
        }
        
        logger.info("业务流程集成测试器初始化完成")
    
    def test_stock_selection_workflow(self) -> Dict[str, Any]:
        """测试选股业务流程"""
        logger.info("开始测试选股业务流程...")
        
 ...
```

5. ```sql
assessment = {
            'business_workflows_functional': True,
            'performance_acceptable': True,
            'system_integration_successful': True,
            'production_deployment_read...
```

6. ```sql
* 80)
        
        # 启动性能监控
        self.performance_monitor.start_monitoring()
        
        test_results = {
            'test_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
           ...
```

7. ```sql
)
            test_results['stock_selection_test'] = self.test_stock_selection_workflow()
            
            # 2. 买点分析业务流程测试
            logger.info(
```

8. ```sql
stock_selection_performance
```

9. ```sql
summary = {
            'stock_selection_performance': {},
            'buypoint_analysis_performance': {},
            'integrated_workflow_performance': {}
        }
        
        # 选股性能总结
      ...
```

10. ```sql
self.data_manager = get_unified_data_manager()
        self.performance_monitor = get_performance_monitor()
        
        # 测试结果
        self.test_results = {
            'stock_selection_test': {}...
```

11. ```sql
: {}
        }
        
        try:
            # 1. 端到端流程测试
            start_time = time.time()
            
            # 步骤1: 获取股票列表
            stock_list = self.data_manager.get_stock_list(limi...
```

12. ```sql
)
        
        integrated_results = {
            'end_to_end_test': {},
            'concurrent_operations': {},
            'system_stability': {}
        }
        
        try:
            # 1...
```

13. ```sql
,
                            limit=50
                        )
                        if isinstance(stock_data, pd.Data_frame) and not stock_data.empty:
                            selected_stocks....
```

### ./tests/integration/buypoint_strategy_integration_test.py
- 查询数量: 2

1. ```sql
if not result:
            return False
        
        required_fields = [
            'stock_code', 'stock_name', 'industry', 'price',
            'change_pct', 'score', 'match_details', 'selection...
```

2. ```sql
)
        
        compatibility_results = {
            'required_fields_test': {},
            'data_types_test': {},
            'score_consistency_test': {},
            'indicator_mapping_test': ...
```

### ./tests/integration/test_stock_selection.py
- 查询数量: 8

1. ```sql
: self.mock_rsi_oversold
        }
        
        self.mock_db_conn = Magic_mock()
        self.data_manager = get_unified_data_manager()
        self.data_manager.db_conn = self.mock_db_conn
      ...
```

2. ```sql
TEST_SELECTION_STRATEGY
```

3. ```sql
)
        with open(self.strategy_file, 'w') as f:
            json.dump(self.strategy_config, f)
        
        # 创建测试用股票列表
        self.stock_list = pd.Data_frame({
            'stock_code': ['000...
```

4. ```sql
def set_up_Selection(self):
```

5. ```sql
, return_value=StrategyParser().parse_strategy(self.strategy_config)):
            result = executor.execute_strategy_by_id(
                strategy_id="TEST_SELECTION_STRATEGY",
                stra...
```

6. ```sql
)
    def test_end_to_end_selection_Selection(self, mock_get_kline, mock_get_stocks, mock_create_indicator):
        """测试端到端的选股流程"""
        # 配置模拟对象行为
        mock_get_stocks.return_value = self.sto...
```

7. ```sql
import unittest
import os
import json
import tempfile
from unittest.mock import patch, Magic_mock

import pandas as pd
import numpy as np

from strategy.strategy_parser import Strategy_parser
from str...
```

8. ```sql
self.temp_dir.cleanup()
        
    @patch.object(IndicatorFactory, 'create')
    @patch.object(DataManager, 'get_stock_list')
    @patch.object(DataManager, 'get_kline_data')
    def test_end_to_end...
```

### ./tests/mocks/db_mock.py
- 查询数量: 8

1. ```sql
SELECT code, name, date, level, open, close, high, low, volume FROM kline_{period} WHERE code = '{stock_code}'
```

2. ```sql
Returns:
            K线数据Data_frame
        """
        cache_key = f"kline_{stock_code}_{period}_{start_date}_{end_date}"
        
        if cache_key in self.cache:
            return self.cache[ca...
```

3. ```sql
SELECT code, name, date, level, open, close, high, low, volume FROM stock_list
```

4. ```sql
SELECT code, name, date, level, open, close, high, low, volume FROM indicator_
```

5. ```sql
SELECT code, name, date, level, open, close, high, low, volume FROM kline_
```

6. ```sql
"
        
        sql += " ORDER BY date"
        
        result = self.db.query(sql)
        self.cache[cache_key] = result
        
        return result
    
    def get_indicator_data(self, 
   ...
```

7. ```sql
SELECT code, name, date, level, open, close, high, low, volume FROM indicator_{indicator_name} WHERE code = '{stock_code}'
```

8. ```sql
}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        sql = "SELECT code, name, date, level, open, close, high, low, volume FROM stock_list"
       ...
```

### ./tests/performance/simple_concurrent_test.py
- 查询数量: 3

1. ```sql
SELECT 1")
            logger.info(f"单连接测试成功: {result}")
            return True
            
    except Exception as e:
        logger.error(f"单连接测试失败: {e}")
        return False


def test_concurren...
```

2. ```sql
),
            max_connections=10,
            min_connections=2
        )
        
        with pool.get_connection() as conn:
            result = conn.execute("SELECT 1")
            logger.info(f"...
```

3. ```sql
SELECT COUNT(*) FROM stock_info WHERE date >= '2020-01-01' LIMIT 1
```

### ./tests/performance/test_system_performance.py
- 查询数量: 1

1. ```sql
self.test_indicators = [
            'ZXM_BS_ABSORB',
            'ZXM_TURNOVER', 
            'ZXM_VOLUME_SHRINK',
            'ZXM_DAILY_TREND_UP',
            'ZXM_AMPLITUDE_ELASTICITY',
          ...
```

### ./tests/reverse_validation/comprehensive_expansion_framework.py
- 查询数量: 1

1. ```sql
def __init__(self):
        # 指标优先级分类
        self.priority_groups = {
            'P0': {  # 核心指标（已完成100%）
                'name': '核心指标(已完成)',
                'indicators': ['RSI', 'MACD', 'KDJ', 'B...
```

### ./tests/reverse_validation/comprehensive_indicator_analysis.py
- 查询数量: 1

1. ```sql
def __init__(self):
        self.indicators = {}
        self.priority_classification = {
            'P0': ['RSI', 'MACD', 'KDJ', 'BOLL', 'MA', 'EMA'],  # 核心指标（已完成）
            'P1': ['SAR', 'ADX', '...
```

### ./tests/review/test_indicators_and_backtest.py
- 查询数量: 5

1. ```sql
SELECT stock_code, stock_name FROM stock.stock_info WHERE stock_code IN (SELECT stock_code FROM stock.index_weight WHERE index_code = '000016.SH' LIMIT 10)
```

2. ```sql
)
            
            # 获取上证50股票列表进行测试
            cls.test_stocks_df = cls.data_access.execute_query("SELECT stock_code, stock_name FROM stock.stock_info WHERE stock_code IN (SELECT stock_code F...
```

3. ```sql
),
                    "update_time": datetime.now().strftime(
```

4. ```sql
SELECT stock_code FROM stock
```

5. ```sql
SELECT stock_code, stock_name FROM stock
```

### ./tests/review/test_multi_period_analysis.py
- 查询数量: 4

1. ```sql
SELECT stock_code FROM stock
```

2. ```sql
)
            
            # 获取上证50股票列表进行测试
            cls.test_stocks_df = cls.data_access.execute_query("SELECT stock_code, stock_name FROM stock.stock_info WHERE stock_code IN (SELECT stock_code F...
```

3. ```sql
SELECT stock_code, stock_name FROM stock.stock_info WHERE stock_code IN (SELECT stock_code FROM stock.index_weight WHERE index_code = '000016.SH' LIMIT 3)
```

4. ```sql
SELECT stock_code, stock_name FROM stock
```

### ./tests/review/test_pattern_recognition.py
- 查询数量: 4

1. ```sql
SELECT stock_code FROM stock
```

2. ```sql
)
            
            # 获取上证50股票列表进行测试
            cls.test_stocks_df = cls.data_access.execute_query("SELECT stock_code, stock_name FROM stock.stock_info WHERE stock_code IN (SELECT stock_code F...
```

3. ```sql
SELECT stock_code, stock_name FROM stock.stock_info WHERE stock_code IN (SELECT stock_code FROM stock.index_weight WHERE index_code = '000016.SH' LIMIT 5)
```

4. ```sql
SELECT stock_code, stock_name FROM stock
```

### ./tests/test_comprehensive_stock_selection.py
- 查询数量: 2

1. ```sql
)
        traceback.print_exc()
        return False


def main_testcomprehensivestockselection():
```

2. ```sql
import sys
import os
import json
import tempfile
import traceback
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.p...
```

### ./tests/test_data_factory.py
- 查询数量: 5

1. ```sql
]].sample(10)
        selected_indices = stock_list[stock_list[
```

2. ```sql
)
        
        # 2. 为部分股票创建K线数据
        # 选择10只股票和3个指数
        selected_stocks = stock_list[~stock_list[
```

3. ```sql
]].sample(3)
        selected = pd.concat([selected_stocks, selected_indices])
        
        for idx, row in selected.iterrows():
            code = row[
```

4. ```sql
) as f:
                return json.load(f)
        else:
            raise ValueError(f"不支持的文件格式: {filename}")
    
    @classmethod
    def generate_standard_test_dataset(cls) -> None:
        """
 ...
```

5. ```sql
)
            
            # 为第一只股票创建指标数据
            if idx == selected.index[0]:
                indicators = cls.create_indicator_data(kline_data)
                for name, data in indicators.items...
```

### ./tests/test_date_manager.py
- 查询数量: 1

1. ```sql
import unittest
import datetime
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, Magic_mock
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path...
```

### ./tests/test_indicator_validation_framework.py
- 查询数量: 31

1. ```sql
config = Indicator_validation_config()
        
        self.assert_equal(config.mode, Validation_mode.FULL)
        self.assert_equal(config.stock_pool_size, 1000)
        self.assert_equal(config.ma...
```

2. ```sql
]
        
        selected_stocks = self.framework._execute_strategy_selection(strategy_config, stock_pool)
        
        self.assert_equal(len(selected_stocks), 0)
    
    def test_execute_strat...
```

3. ```sql
import os
import sys
import unittest
import json
import tempfile
from unittest.mock import Mock, patch, Magic_mock
import pandas as pd

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path....
```

4. ```sql
total_selected_stocks
```

5. ```sql
_execute_strategy_selection
```

6. ```sql
# Mock策略执行结果
        mock_result = pd.Data_frame({
            'code': ['000001', '000002', '600000'],
            'score': [0.8, 0.7, 0.6]
        })
        self.mock_strategy_executor.execute_strat...
```

7. ```sql
self.config = Indicator_validation_config(
            mode=Validation_mode.QUICK,
            stock_pool_size=100,
            max_selection_ratio=0.1,
            parallel_workers=1,
            sav...
```

8. ```sql
test_results = [
            {
                'indicator_name': 'MA',
                'status': ValidationResult.SUCCESS.value,
                'selected_count': 5,
                'selection_ratio':...
```

9. ```sql
: ValidationResult.NO_SELECTION.value,
```

10. ```sql
)
        
        strategy_config = {'test': 'config'}
        stock_pool = ['000001', '000002']
        
        selected_stocks = self.framework._execute_strategy_selection(strategy_config, stock_p...
```

11. ```sql
indicators = ['MA', 'EMA', 'RSI']
        stock_pool = ['000001', '000002']
        
        with patch.object(self.framework, '_validate_indicator') as mock_validate:
            mock_validate.side_e...
```

12. ```sql
)
        self.assertEqual(ValidationResult.OVER_SELECTION.value,
```

13. ```sql
with patch.object(self.framework, '_prepare_stock_pool') as mock_prepare:
            mock_prepare.return_value = ['000001', '000002', '600000']
            
            with patch.object(self.framewo...
```

14. ```sql
config = Indicator_validation_config(
            mode=Validation_mode.QUICK,
            stock_pool_size=500,
            max_selection_ratio=0.05,
            parallel_workers=2,
            output_...
```

15. ```sql
stock_pool = ['000001', '000002', '600000'] * 10  # 30只股票
        
        with patch.object(self.framework, '_generate_indicator_strategy') as mock_gen_strategy:
            mock_gen_strategy.return_...
```

16. ```sql
], ValidationResult.NO_SELECTION.value)
                self.assertEqual(result[
```

17. ```sql
)
        
        self.assert_equal(config.mode, Validation_mode.QUICK)
        self.assert_equal(config.stock_pool_size, 500)
        self.assert_equal(config.max_selection_ratio, 0.05)
        self...
```

18. ```sql
][ValidationResult.NO_SELECTION.value], 1)
        self.assertEqual(summary[
```

19. ```sql
stock_pool = ['000001', '000002', '600000'] * 20  # 60只股票
        
        # Mock策略生成
        with patch.object(self.framework, '_generate_indicator_strategy') as mock_gen_strategy:
            mock_g...
```

20. ```sql
], 2)


class Test_validation_config(unittest.Test_case):
    """验证配置测试类"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = Indicator_validation_config()
        
       ...
```

21. ```sql
: ValidationResult.NO_SELECTION.value},
                {
```

22. ```sql
stock_pool = ['000001', '000002', '600000']
        
        with patch.object(self.framework, '_generate_indicator_strategy') as mock_gen_strategy:
            mock_gen_strategy.return_value = {'test...
```

23. ```sql
]
        
        selected_stocks = self.framework._execute_strategy_selection(strategy_config, stock_pool)
        
        self.assert_equal(len(selected_stocks), 3)
        self.assertEqual(select...
```

24. ```sql
strategy = self.framework._generate_indicator_strategy('UNKNOWN_INDICATOR')
        
        self.assert_is_instance(strategy, dict)
        self.assertIn('conditions', strategy)
        
        # 验证...
```

25. ```sql
])
    
    def test_execute_strategy_selection_empty_result(self):
        """测试策略选股执行返回空结果"""
        self.mock_strategy_executor.execute_strategy.return_value = pd.Data_frame()
        
        str...
```

26. ```sql
self.mock_strategy_executor.execute_strategy.return_value = pd.Data_frame()
        
        strategy_config = {'test': 'config'}
        stock_pool = ['000001', '000002']
        
        selected_st...
```

27. ```sql
], 0)
    
    def test_validate_indicator_over_selection(self):
        """测试指标验证过度选择"""
        stock_pool = [
```

28. ```sql
], 0)
    
    def test_execute_strategy_selection_success(self):
        """测试策略选股执行成功"""
        # Mock策略执行结果
        mock_result = pd.Data_frame({
```

29. ```sql
]
        
        selected_stocks = self.framework._execute_strategy_selection(strategy_config, stock_pool)
        
        self.assert_equal(len(selected_stocks), 0)
    
    def test_validate_indi...
```

30. ```sql
], ValidationResult.OVER_SELECTION.value)
                self.assertEqual(result[
```

31. ```sql
)
        self.assertEqual(ValidationResult.NO_SELECTION.value,
```

### ./tests/test_indicators/test_boll_bandwidth.py
- 查询数量: 4

1. ```sql
], 3.0)
        self.assertEqual(updated_params[
```

2. ```sql
: 5
        }
        boll.set_parameters(new_params)
        
        # 验证参数是否正确设置
        updated_params = boll.parameters
        self.assertEqual(updated_params[
```

3. ```sql
# 创建一个新的BOLL实例
        boll = BOLL()
        
        # 检查默认参数
        params = boll.parameters
        self.assertEqual(params['periods'], 20)
        self.assertEqual(params['std_dev'], 2.0)
       ...
```

4. ```sql
], 10)
        self.assertEqual(updated_params[
```

### ./tests/test_institutional_behavior.py
- 查询数量: 1

1. ```sql
: self.data.iloc[150:]   # 出货期开始
        }
        
        # 执行选股
        selected = self.strategy.select(data_dict)
        
        # 检查选股结果
        self.assert_is_instance(selected, list)
        ...
```

### ./tests/test_integration_comprehensive.py
- 查询数量: 1

1. ```sql
import sys
import os
import json
import tempfile
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any
import unittest

import pandas as pd
import numpy as np

#...
```

### ./tests/test_integration_quick.py
- 查询数量: 1

1. ```sql
import sys
import os
import time
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.unified_data_manager import get_un...
```

### ./tests/test_stock_selection_system.py
- 查询数量: 4

1. ```sql
, test_end_to_end_selection()))
    
    # 输出测试结果汇总
    print(
```

2. ```sql
)
        traceback.print_exc()
        return False


def test_end_to_end_selection():
```

3. ```sql
import sys
import os
import json
import tempfile
import traceback
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.p...
```

4. ```sql
)
        traceback.print_exc()
        return False


def main_teststockselectionsystem():
```

### ./tests/test_unified_analysis_engine_integration.py
- 查询数量: 1

1. ```sql
\n\nimport unittest\nimport sys\nimport os\nfrom unittest.mock import Mock, patch\nimport pandas as pd\nimport datetime\n\n# 添加项目根目录到路径\nroot_dir = os.path.dirname(os.path.dirname(os.path.abspath(__fi...
```

### ./tests/unit/engines/test_shared_condition_evaluator.py
- 查询数量: 2

1. ```sql
: np.array([40, 49, 62, 44, 68, 72, 57, 78, 82, 85]),
        }
        
        # 添加买点分析特有的逻辑字段
        self.test_data.update({
```

2. ```sql
self.evaluator = Shared_condition_evaluator()
        
        # 准备测试数据
        self.test_data = {
            'close': np.array([10.0, 10.5, 11.0, 10.8, 11.2, 11.5, 11.3, 11.8, 12.0, 12.2]),
        ...
```

### ./tests/unit/engines/test_unified_indicator_engine.py
- 查询数量: 1

1. ```sql
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname...
```

### ./tests/unit/test_advanced_candlestick_patterns.py
- 查询数量: 2

1. ```sql
raw_score_df = self.indicator.calculate_raw_score(self.data)
        raw_score = raw_score_df['score'] if 'score' in raw_score_df.columns else pd.Series(50.0, index=self.data.index)
        patterns =...
```

2. ```sql
in raw_score_df.columns else pd.Series(50.0, index=self.data.index)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_scor...
```

### ./tests/unit/test_bias.py
- 查询数量: 2

1. ```sql
] - ma) / ma * 100
                
                # 比较计算结果（允许小的数值误差）
                calculated_bias = result[col_name]
                diff = abs(calculated_bias - expected_bias).dropna()
         ...
```

2. ```sql
raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, pattern...
```

### ./tests/unit/test_candlestick_patterns.py
- 查询数量: 2

1. ```sql
raw_score_df = self.indicator.calculate_raw_score(self.data)
        raw_score = raw_score_df['score'] if 'score' in raw_score_df.columns else pd.Series(50.0, index=self.data.index)
        patterns =...
```

2. ```sql
in raw_score_df.columns else pd.Series(50.0, index=self.data.index)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_scor...
```

### ./tests/unit/test_cci.py
- 查询数量: 2

1. ```sql
raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, pattern...
```

2. ```sql
]
        diff = abs(calculated_cci - expected_cci).dropna()
        self.assertTrue(all(d < 0.001 for d in diff), "CCI计算结果不正确")
    
    def test_cci_score_range(self):
        """测试CCI评分范围"""
      ...
```

### ./tests/unit/test_chaikin.py
- 查询数量: 2

1. ```sql
]
        chaikin_diff = abs(calculated_chaikin - expected_chaikin).dropna()
        self.assertTrue(all(d < 0.001 for d in chaikin_diff), "Chaikin震荡器计算结果不正确")
    
    def test_chaikin_score_range(se...
```

2. ```sql
raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, pattern...
```

### ./tests/unit/test_chip_distribution.py
- 查询数量: 2

1. ```sql
]:
            if col in result.columns:
                values = result[col].dropna()
                
                if len(values) > 0:
                    # 这些值应该在0-1范围内
                    self....
```

2. ```sql
raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, pattern...
```

### ./tests/unit/test_cmo.py
- 查询数量: 2

1. ```sql
raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, pattern...
```

2. ```sql
]
        diff = abs(calculated_cmo - expected_cmo).dropna()
        self.assertTrue(all(d < 0.001 for d in diff), "CMO计算结果不正确")
    
    def test_cmo_score_range(self):
        """测试CMO评分范围"""
      ...
```

### ./tests/unit/test_composite_indicator.py
- 查询数量: 2

1. ```sql
)
        
        # 添加指标
        composite.add_indicator(self.ma_indicator, 0.5)
        composite.add_indicator(self.rsi_indicator, 0.5)
        
        self.assert_equal(len(composite.indicators),...
```

2. ```sql
raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, pattern...
```

### ./tests/unit/test_data_manager.py
- 查询数量: 12

1. ```sql
)
    def test_save_selection_result_empty(self, mock_execute, mock_query):
        """测试保存空选股结果"""
        # 创建空选股结果
        selection_result = pd.Data_frame()
        
        # 调用被测方法
        resul...
```

2. ```sql
# 创建空选股结果
        selection_result = pd.Data_frame()
        
        # 调用被测方法
        result = self.data_manager.save_selection_result(
            result=selection_result,
            strategy_id='T...
```

3. ```sql
)
        
        # 测试数据库错误
        with self.assert_raises(Data_access_error):
            self.data_manager.save_selection_result(
                result=selection_result,
                strategy_...
```

4. ```sql
]]
        })
        
        # 配置模拟对象抛出异常
        mock_execute.side_effect = Exception("数据库连接错误")
        
        # 测试数据库错误
        with self.assert_raises(Data_access_error):
            self.data...
```

5. ```sql
# 创建测试选股结果
        selection_result = pd.Data_frame({
            'stock_code': ['000001', '000002'],
            'stock_name': ['测试1', '测试2'],
            'signal_strength': [0.8, 0.7],
            '...
```

6. ```sql
]]
        })
        
        # 配置模拟对象
        mock_execute.return_value = 2  # 影响的行数
        
        # 调用被测方法
        result = self.data_manager.save_selection_result(
            result=selection_...
```

7. ```sql
,
                selection_date=
```

8. ```sql
)
    def test_save_selection_result_success(self, mock_execute, mock_query):
        """测试成功保存选股结果"""
        # 创建测试选股结果
        selection_result = pd.Data_frame({
```

9. ```sql
# 创建测试选股结果
        selection_result = pd.Data_frame({
            'stock_code': ['000001', '000002'],
            'stock_name': ['测试1', '测试2'],
            'signal_strength': [0.8, 0.7],
            '...
```

10. ```sql
)

        # 测试数据库错误
        with self.assert_raises(Data_access_error):
            self.data_manager.get_stock_info()

        # 验证模拟对象被调用
        mock_get_stock_info.assert_called_once()
    
    @...
```

11. ```sql
)
    def test_save_selection_result_db_error(self, mock_execute, mock_query):
        """测试数据库错误时保存选股结果"""
        # 创建测试选股结果
        selection_result = pd.Data_frame({
```

12. ```sql
,
            selection_date=
```

### ./tests/unit/test_data_service_interfaces.py
- 查询数量: 2

1. ```sql
# 验证接口有必要的方法
        required_methods = [
            'get',
            'set',
            'delete',
            'exists',
            'clear',
            'get_stats'
        ]
        
        for ...
```

2. ```sql
import unittest
from unittest.mock import Mock, patch
import sys
import os

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root...
```

### ./tests/unit/test_dma.py
- 查询数量: 2

1. ```sql
raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, pattern...
```

2. ```sql
]
        ama_diff = abs(calculated_ama - expected_ama).dropna()
        self.assertTrue(all(d < 0.001 for d in ama_diff), "AMA计算结果不正确")
    
    def test_dma_score_range(self):
        """测试DMA评分范围""...
```

### ./tests/unit/test_dmi.py
- 查询数量: 2

1. ```sql
raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, pattern...
```

2. ```sql
].dropna()
        
        self.assertTrue(len(pdi_values) > 0, "PDI值全为NaN")
        self.assertTrue(len(mdi_values) > 0, "MDI值全为NaN")
        self.assertTrue(len(adx_values) > 0, "ADX值全为NaN")
      ...
```

### ./tests/unit/test_elliott_wave.py
- 查询数量: 2

1. ```sql
in raw_score_df.columns else pd.Series(50.0, index=self.data.index)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_scor...
```

2. ```sql
raw_score_df = self.indicator.calculate_raw_score(self.data)
        raw_score = raw_score_df['score'] if 'score' in raw_score_df.columns else pd.Series(50.0, index=self.data.index)
        patterns =...
```

### ./tests/unit/test_emv.py
- 查询数量: 2

1. ```sql
].dropna()
        
        self.assertTrue(len(emv_values) > 0, "EMV值全为NaN")
        self.assertTrue(len(emv_ma_values) > 0, "EMV_MA值全为NaN")
        
        # 验证EMV值都是有限数
        self.assertTrue(all...
```

2. ```sql
raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, pattern...
```

### ./tests/unit/test_enhanced_cci.py
- 查询数量: 2

1. ```sql
].iloc[9]
            if not pd.isna(cci_value):
                # CCI应该是一个合理的数值
                self.assertTrue(-500 <= cci_value <= 500, "CCI值应该在合理范围内")
    
    def test_enhanced_cci_score_range(se...
```

2. ```sql
raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, pattern...
```

### ./tests/unit/test_enhanced_dmi.py
- 查询数量: 2

1. ```sql
raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, pattern...
```

2. ```sql
].iloc[9]
            
            if not pd.isna(plus_di_value) and not pd.isna(minus_di_value) and not pd.isna(adx_value):
                # DMI值应该是合理的数值
                self.assertTrue(0 <= plus_di...
```

### ./tests/unit/test_enhanced_kdj.py
- 查询数量: 2

1. ```sql
raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, pattern...
```

2. ```sql
].iloc[4]
            
            if not pd.isna(k_value) and not pd.isna(d_value) and not pd.isna(j_value):
                # 验证J = 3K - 2D
                expected_j = 3 * k_value - 2 * d_value
   ...
```

### ./tests/unit/test_enhanced_macd.py
- 查询数量: 2

1. ```sql
raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, pattern...
```

2. ```sql
].iloc[9]
            
            if not pd.isna(macd_value) and not pd.isna(signal_value) and not pd.isna(hist_value):
                # 验证柱状体 = MACD - 信号线
                self.assert_almost_equal(h...
```

### ./tests/unit/test_enhanced_mfi.py
- 查询数量: 2

1. ```sql
raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, pattern...
```

2. ```sql
].iloc[4]
            
            if not pd.isna(mfi_value):
                # MFI值应该在合理范围内
                self.assertTrue(0 <= mfi_value <= 100, "MFI值应该在0-100范围内")
    
    def test_enhanced_mfi_sc...
```

### ./tests/unit/test_enhanced_obv.py
- 查询数量: 2

1. ```sql
].iloc[4]
            
            if not pd.isna(obv_value):
                # OBV值应该是有限数值
                self.assertTrue(np.isfinite(obv_value), "OBV值应该是有限数值")
    
    def test_enhanced_obv_score_...
```

2. ```sql
raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, pattern...
```

### ./tests/unit/test_enhanced_trix.py
- 查询数量: 2

1. ```sql
raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, pattern...
```

2. ```sql
].iloc[9]
            
            if not pd.isna(trix_value) and not pd.isna(matrix_value):
                # TRIX值应该是合理的数值
                self.assertTrue(-10 <= trix_value <= 10, "TRIX值应该在合理范围内")
 ...
```

### ./tests/unit/test_gann_tools.py
- 查询数量: 2

1. ```sql
in raw_score_df.columns else pd.Series(50.0, index=self.data.index)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_scor...
```

2. ```sql
raw_score_df = self.indicator.calculate_raw_score(self.data)
        raw_score = raw_score_df['score'] if 'score' in raw_score_df.columns else pd.Series(50.0, index=self.data.index)
        patterns =...
```

### ./tests/unit/test_ichimoku.py
- 查询数量: 2

1. ```sql
]
        diff = abs(calculated_tenkan - expected_tenkan).dropna()
        self.assertTrue(all(d < 0.001 for d in diff), "转换线计算结果不正确")
    
    def test_ichimoku_score_range(self):
        """测试Ichimo...
```

2. ```sql
raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, pattern...
```

### ./tests/unit/test_institutional_behavior.py
- 查询数量: 2

1. ```sql
].dropna()
            if len(profit_values) > 0:
                # 获利盘比例应该在0-1范围内
                self.assert_true(all(0 <= v <= 1 for v in profit_values), 
                               "机构获利盘比例应该在...
```

2. ```sql
raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, pattern...
```

### ./tests/unit/test_kc.py
- 查询数量: 2

1. ```sql
raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, pattern...
```

2. ```sql
].iloc[i])
    
    def test_kc_score_range(self):
        """测试KC评分范围"""
        raw_score = self.indicator.calculate_raw_score(self.data)
        
        # 验证评分在0-100范围内
        valid_scores = raw_...
```

### ./tests/unit/test_mtm.py
- 查询数量: 2

1. ```sql
raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, pattern...
```

2. ```sql
]
        
        # 验证MTM计算
        mtm_diff = abs(calculated_mtm - expected_mtm).dropna()
        self.assertTrue(all(d < 0.001 for d in mtm_diff), "MTM计算结果不正确")
        
        # 验证MTMMA计算
       ...
```

### ./tests/unit/test_psy.py
- 查询数量: 2

1. ```sql
raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, pattern...
```

2. ```sql
]
        
        # 验证PSY计算
        psy_diff = abs(calculated_psy - expected_psy).dropna()
        self.assertTrue(all(d < 0.001 for d in psy_diff), "PSY计算结果不正确")
        
        # 验证PSYMA计算
       ...
```

### ./tests/unit/test_pvt.py
- 查询数量: 2

1. ```sql
raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, pattern...
```

2. ```sql
]
        
        # 验证PVT计算
        pvt_diff = abs(calculated_pvt - expected_pvt).dropna()
        self.assertTrue(all(d < 0.001 for d in pvt_diff), "PVT计算结果不正确")
        
        # 验证信号线计算
        s...
```

### ./tests/unit/test_roc.py
- 查询数量: 2

1. ```sql
raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, pattern...
```

2. ```sql
]
        
        # 验证ROC计算
        roc_diff = abs(calculated_roc - expected_roc).dropna()
        self.assertTrue(all(d < 0.001 for d in roc_diff), "ROC计算结果不正确")
        
        # 验证ROCMA计算
       ...
```

### ./tests/unit/test_sar.py
- 查询数量: 2

1. ```sql
].dropna()
        if len(trend_values) > 0:
            self.assert_true(all(v in [1, -1] for v in trend_values), 
                           "趋势值应该只能是1或-1")
    
    def test_sar_score_range(self):
...
```

2. ```sql
raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, pattern...
```

### ./tests/unit/test_stochrsi.py
- 查询数量: 2

1. ```sql
].dropna()
        
        if len(k_values) > 0:
            # StochRSI值应该在0-100范围内
            self.assertTrue(all(0 <= v <= 100 for v in k_values), "StochRSI K值应在0-100范围内")
        
        if len(...
```

2. ```sql
raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, pattern...
```

### ./tests/unit/test_stock_vix.py
- 查询数量: 2

1. ```sql
raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, pattern...
```

2. ```sql
].dropna()
            
            if len(vix_values) > 0:
                # VIX值应该是正数
                self.assertTrue(all(v > 0 for v in vix_values), "VIX值应该是正数")
                # VIX值应该是有限数值
     ...
```

### ./tests/unit/test_trix.py
- 查询数量: 2

1. ```sql
].dropna()
        if len(trix_values) > 0:
            # TRIX值应该是百分比形式，通常在-10到10之间
            self.assert_true(all(-50 <= v <= 50 for v in trix_values), 
                           "TRIX值应该在合理范围内")
...
```

2. ```sql
raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, pattern...
```

### ./tests/unit/test_unified_ma.py
- 查询数量: 2

1. ```sql
raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, pattern...
```

2. ```sql
].dropna()
                
                if len(ma10_values) > 0:
                    self.assert_true(all(np.isfinite(v) for v in ma10_values), 
                                   f"{ma_type} MA10...
```

### ./tests/unit/test_vix.py
- 查询数量: 2

1. ```sql
raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, pattern...
```

2. ```sql
]
        
        # 比较计算结果（允许小的数值误差）
        diff = abs(calculated_daily_range - expected_daily_range).dropna()
        self.assertTrue(all(d < 0.001 for d in diff), "日内波动率计算不正确")
    
    def test_v...
```

### ./tests/unit/test_vol.py
- 查询数量: 2

1. ```sql
raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, pattern...
```

2. ```sql
]
        
        # 比较计算结果（允许小的数值误差）
        diff = abs(calculated_vol_ratio - expected_vol_ratio).dropna()
        self.assertTrue(all(d < 0.001 for d in diff), "VOL比率计算不正确")
    
    def test_vol_s...
```

### ./tests/unit/test_volume_ratio.py
- 查询数量: 2

1. ```sql
].iloc[5]
            
            self.assert_almost_equal(calculated_vr, expected_vr, places=3, 
                                 msg="Volume Ratio计算不正确")
    
    def test_volume_ratio_score_range(...
```

2. ```sql
raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, pattern...
```

### ./tests/unit/test_vortex.py
- 查询数量: 2

1. ```sql
raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, pattern...
```

2. ```sql
].iloc[2]
            
            if not pd.isna(vi_plus_value) and not pd.isna(vi_minus_value):
                # VI值应该是合理的正数
                self.assertGreater(vi_plus_value, 0, "VI+值应该为正数")
      ...
```

### ./tests/unit/test_vosc.py
- 查询数量: 2

1. ```sql
].iloc[9]
            
            if not pd.isna(calculated_vosc):
                self.assert_almost_equal(calculated_vosc, expected_vosc, places=2, 
                                     msg="VOSC计算...
```

2. ```sql
raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, pattern...
```

### ./tests/unit/test_vr.py
- 查询数量: 2

1. ```sql
].iloc[5]
            if not pd.isna(vr_value):
                # VR应该是一个合理的正数
                self.assertGreater(vr_value, 0, "VR值应该为正数")
                self.assertLess(vr_value, 1000, "VR值应该在合理范围内"...
```

2. ```sql
raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, pattern...
```

### ./tests/unit/test_wma.py
- 查询数量: 2

1. ```sql
raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, pattern...
```

2. ```sql
].iloc[2]
            
            if not pd.isna(calculated_wma):
                self.assert_almost_equal(calculated_wma, expected_wma, places=6, 
                                     msg="WMA计算不正确"...
```

### ./tests/unit/test_wr.py
- 查询数量: 2

1. ```sql
raw_score = self.indicator.calculate_raw_score(self.data)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_score, pattern...
```

2. ```sql
].iloc[2]
            
            if not pd.isna(calculated_wr):
                self.assert_almost_equal(calculated_wr, expected_wr, places=2, 
                                     msg="WR计算不正确")
  ...
```

### ./tests/unit/test_zxm_boundary_conditions.py
- 查询数量: 2

1. ```sql
].dtype, bool, f"{indicator_name}的buy_signal应该是布尔类型")
                    
                    # 验证没有无限值
                    numeric_columns = result.select_dtypes(include=[np.number]).columns
       ...
```

2. ```sql
)
                    
                    # 验证没有无限值
                    numeric_columns = result.select_dtypes(include=[np.number]).columns
                    for col in numeric_columns:
           ...
```

### ./tests/unit/test_zxm_comprehensive.py
- 查询数量: 3

1. ```sql
indicator = Selection_model()

        try:
            # 只测试抽象方法的存在性，不测试复杂的计算逻辑
            # 测试set_parameters方法
            indicator.set_parameters(selection_threshold=80)
            self.assert_e...
```

2. ```sql
)
    
    def test_zxm_selection_model(self):
```

3. ```sql
] = low
    
    def test_all_zxm_trend_indicators(self):
        """测试所有ZXM趋势指标"""
        trend_indicators = [
            ZXMDaily_trend_up(),
            ZXMWeekly_trend_up(),
            ZXMMonth...
```

### ./tests/unit/test_zxm_patterns.py
- 查询数量: 2

1. ```sql
in raw_score_df.columns else pd.Series(50.0, index=self.data.index)
        patterns = self.indicator.get_patterns(self.data)
        
        confidence = self.indicator.calculate_confidence(raw_scor...
```

2. ```sql
raw_score_df = self.indicator.calculate_raw_score(self.data)
        raw_score = raw_score_df['score'] if 'score' in raw_score_df.columns else pd.Series(50.0, index=self.data.index)
        patterns =...
```

### ./tests/unit/test_zxm_signal_semantic_validation.py
- 查询数量: 10

1. ```sql
] == True]
        if len(selected_rows) > 0:
            self.assertTrue(selected_rows[
```

2. ```sql
)

    def test_selection_model_semantic_consistency(self):
```

3. ```sql
indicator = Selection_model()

        # 测试选股场景
        trend_up_data = self.test_scenarios['trend_up']
        result = indicator.calculate(trend_up_data)

        # 验证语义：当FinalSelect为True时，buy_signa...
```

4. ```sql
].all(), "FinalSelect=True时buy_signal应该为True")

        # 验证语义：当FinalSelect为False时，buy_signal应该为False
        not_selected_rows = result[result[
```

5. ```sql
].all(), "总分<30时sell_signal应该为True")

        print("✅ 股票综合评分指标语义验证通过")

    def test_selection_model_semantic_consistency(self):
        """测试ZXM选股模型的语义一致性"""
        indicator = Selection_model()

 ...
```

6. ```sql
)

        # 验证语义：当FinalSelect为False时，buy_signal应该为False
        not_selected_rows = result[result['FinalSelect'] == False]
        if len(not_selected_rows) > 0:
            self.assertFalse(not_sele...
```

7. ```sql
]
        result = indicator.calculate(trend_up_data)

        # 验证语义：当FinalSelect为True时，buy_signal应该为True
        selected_rows = result[result[
```

8. ```sql
].any(), "FinalSelect=False时buy_signal应该为False")

        print("✅ ZXM选股模型语义验证通过")


if __name__ ==
```

9. ```sql
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 导入需要测试的ZXM指标
from indicators.zxm.buy_point_indicators import ZXMTurnover, ZXMVolume_shrink, ZXMBSAbso...
```

10. ```sql
] == False]
        if len(not_selected_rows) > 0:
            self.assertFalse(not_selected_rows[
```

### ./tests/unit/test_zxm_system.py
- 查询数量: 4

1. ```sql
]
        for key in expected_signal_keys:
            self.assertIn(key, signals, f"缺少信号键: {key}")
            self.assert_is_instance(signals[key], pd.Series)
    
    def test_zxm_washplate_initial...
```

2. ```sql
raw_score = self.zxm_absorb.calculate_raw_score(self.data)
        patterns = self.zxm_absorb.get_patterns(self.data)
        
        confidence = self.zxm_absorb.calculate_confidence(raw_score, patt...
```

3. ```sql
raw_score = self.zxm_washplate.calculate_raw_score(self.data)
        patterns = self.zxm_washplate.get_patterns(self.data)
        
        confidence = self.zxm_washplate.calculate_confidence(raw_sc...
```

4. ```sql
]
        for col in core_columns:
            self.assertIn(col, result.columns, f"缺少核心列: {col}")
    
    def test_zxm_absorb_score_range(self):
        """测试ZXMAbsorb评分范围"""
        raw_score = sel...
```

### ./tools/batch_indicator_fix.py
- 查询数量: 9

1. ```sql
try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 查找add_signal_generation调用
            pattern = r'(\s+)# 添加形态识别和信号生成\s*...
```

2. ```sql
.join(lines[max(0, i-5):i]):
                    return_line_idx = i
                    break
            
            if return_line_idx == -1:
                logger.warning(f"未找到return语句: {file_pa...
```

3. ```sql
try:
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检测插入点
            has_pattern, insert_line, insert...
```

4. ```sql
) as f:
                content = f.read()
            
            # 检测插入点
            has_pattern, insert_line, insert_pos = self.detect_signal_generation_pattern(file_path)
            
           ...
```

5. ```sql
)
                insert_line = len(lines) - 1
                return True, insert_line, start_pos
            
            return False, -1, -1
            
        except Exception as e:
           ...
```

6. ```sql
)
                return False
            
            # 插入修复代码
            lines.insert(return_line_idx, fix_code)
            
            # 写回文件
            with open(file_path, 'w', encoding='utf...
```

7. ```sql
# 等级型指标：基于评分阈值
            numeric_cols = df.select_dtypes(include=[
```

8. ```sql
] == False
            else:
                # 使用第一个布尔列作为信号源
                bool_cols = [col for col in df.columns if df[col].dtype == bool]
                if bool_cols:
                    signal_c...
```

9. ```sql
# 计数型指标：基于数值阈值
            numeric_cols = df.select_dtypes(include=[
```

### ./tools/continuous_quality_assurance.py
- 查询数量: 2

1. ```sql
self.alert_config.update({
            'email_enabled': email_enabled,
            'email_recipients': email_recipients or [],
            'smtp_server': smtp_server,
            'smtp_port': smtp_por...
```

2. ```sql
):
        """配置告警设置"""
        self.alert_config.update({
```

### ./tools/indicator_generator.py
- 查询数量: 4

1. ```sql
) as f:
                yaml.dump(schemas, f, default_flow_style=False, allow_unicode=True)
            print(f"  ✓ Schema定义已更新: {self.schema_file}")
            
            # 5. 验证生成的指标
            ...
```

2. ```sql
#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
{description}
"""

import numpy as np
import pandas as pd
from typing import Dict, Any

from indicators.complete_indicator_registry import complete_regis...
```

3. ```sql
# 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator(silent_mode=True)
            
     ...
```

4. ```sql
)
                
                # 导入测试
                sys.path.insert(0, str(self.indicators_dir.parent))
                module_name = f
```

### ./tools/migrate_patterns.py
- 查询数量: 1

1. ```sql
import sys
import os
import re
from typing import Dict, List, Tuple

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from utils....
```

### ./tools/system_cleanup_tool.py
- 查询数量: 10

1. ```sql
)
                
                for file_path in files:
                    full_path = self.project_root / file_path
                    
                    try:
                        if full_p...
```

2. ```sql
,
                'files': obsolete_files['temporary_files'],
                'safe': True
            })
        
        # 过时脚本建议
        if 'obsolete_scripts' in obsolete_files:
            recomme...
```

3. ```sql
selection_results_.*\.csv$
```

4. ```sql
• 删除文件: {actual_results['files_deleted']} 个
```

5. ```sql
)
        
        recommendations = []
        
        # 扫描各种过时文件
        obsolete_files = self.scan_obsolete_files()
        empty_dirs = self.scan_empty_directories()
        large_logs = self.sca...
```

6. ```sql
,
                'files': obsolete_files['backup_files'],
                'safe': True
            })
        
        # 临时文件建议
        if 'temporary_files' in obsolete_files:
            recommendat...
```

7. ```sql
})
        
        # 空目录建议
        if empty_dirs:
            recommendations.append({
                'category': 'empty_directories',
                'priority': 'low',
                'action': 'd...
```

8. ```sql
清理完成，删除了 {cleanup_results['files_deleted']} 个文件，
```

9. ```sql
)
        
        cleanup_results = {
            'dry_run': dry_run,
            'actions_taken': [],
            'files_deleted': 0,
            'space_freed_mb': 0,
            'errors': []
      ...
```

10. ```sql
self.project_root = Path(project_root).resolve()
        self.cleanup_report = {
            'scan_date': datetime.now().isoformat(),
            'categories': {},
            'recommendations': [],
 ...
```

### ./upgrade_strategy_configs.py
- 查询数量: 1

1. ```sql
sample_strategy = {
        'strategy': {
            'id': 'UPGRADED_SAMPLE_STRATEGY',
            'name': '升级后示例策略',
            'description': '展示新标准化格式的示例策略',
            'version': '2.0',
       ...
```

### ./utils/cache.py
- 查询数量: 2

1. ```sql
with self._lock:
            count = 0
            for key in list(self._index.keys()):
                _, timestamp, ttl = self._index[key]
                if ttl is not None and time.time() - timest...
```

2. ```sql
with self._lock:
            self._cache[key] = (value, time.time(), ttl)
    
    def delete(self, key: str) -> bool:
```

### ./utils/file_utils.py
- 查询数量: 2

1. ```sql
def safe_delete(path: str) -> bool:
```

2. ```sql
]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def safe_delete(path: str) -> bool:
    """
    安全删除文件...
```

### ./utils/indicator_name_mapper.py
- 查询数量: 2

1. ```sql
# 策略中的指标名称 -> 注册表中的指标名称
        self.name_mapping = {
            # 核心指标（大部分已匹配）
            'VOL': 'VOL',
            'SAR': 'SAR', 
            'KC': 'KC',
            'MTM': 'MTM',
            'PSY...
```

2. ```sql
)
        
        # 按类别分组
        categories = {
            '核心指标': ['VOL', 'SAR', 'KC', 'MTM', 'PSY', 'PVT', 'TRIX', 'VIX', 'VOSC', 'VR', 'WR', 'MACD', 'BOLL', 'KDJ', 'BIAS', 'DMI', 'EMV', 'CMO', '...
```

### ./utils/parameter_standardizer.py
- 查询数量: 2

1. ```sql
# 获取原始参数
        original_params = condition.get('parameters', {})
        
        # 获取默认参数
        default_params = self.validator.get_default_parameters(indicator_id)
        
        # 合并参数
      ...
```

2. ```sql
, {})
        
        # 获取默认参数
        default_params = self.validator.get_default_parameters(indicator_id)
        
        # 合并参数
        standardized_params = default_params.copy()
        standar...
```

### ./utils/period_manager.py
- 查询数量: 3

1. ```sql
)
                DBManager = db_manager_module.DBManager
                self.db_manager = DBManager.get_instance()
                self.data_access = self.db_manager
            except ImportError:
...
```

2. ```sql
SELECT 
            trade_date as date,
            open,
            high,
            low,
            close,
            volume
        FROM {table_name}
        WHERE ts_code = '{stock_code}'
    ...
```

3. ```sql
SELECT 
            trade_date as date,
            open,
            high,
            low,
            close,
            volume
        FROM {table_name}
        WHERE ts_code = '{stock_code}'
    ...
```

### ./utils/signal_utils.py
- 查询数量: 3

1. ```sql
:
        # 并集，合并所有信号并去重
        combined = set()
        for signals in signal_lists:
            combined.update(signals)
        return sorted(list(combined))
    elif logic ==
```

2. ```sql
if not signal_lists:
        return []
    
    if logic == 'union':
        # 并集，合并所有信号并去重
        combined = set()
        for signals in signal_lists:
            combined.update(signals)
        r...
```

3. ```sql
:
        # 交集，只保留所有列表都有的信号
        if not signal_lists:
            return []
        result = set(signal_lists[0])
        for signals in signal_lists[1:]:
            result.intersection_update(sig...
```

## 分类统计

### Stock Data
- 数量: 219

### Batch Queries
- 数量: 22

### Count Queries
- 数量: 62

### List Queries
- 数量: 74

### Other
- 数量: 2198

## 迁移建议

1. **优先级1**: stock_data 和 batch_queries - 核心业务查询
2. **优先级2**: count_queries 和 list_queries - 统计和列表查询
3. **优先级3**: other - 其他查询

## 迁移步骤

1. 使用 `db/sql_manager.py` 定义标准查询模板
2. 使用 `db/query_executor.py` 替换直接SQL调用
3. 测试验证迁移效果
4. 更新相关文档

