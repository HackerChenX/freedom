"""
ZXM体系买点和吸筹识别模块

实现基于ZXM体系的买点和吸筹形态识别
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Union

from indicators.base_indicator import BaseIndicator
from indicators.common import ma, ema, macd, kdj, ref, highest, lowest, cross, crossover, crossunder
from enums.indicator_types import IndicatorType
from enums.pattern_types import BuyPointType, AbsorptionPatternType, VolumePattern


class ZXMPatternIndicator(BaseIndicator):
    """ZXM体系买点和吸筹形态识别指标"""
    
    def __init__(self):
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        """初始化ZXM体系模式识别指标"""
        super().__init__(indicator_type=IndicatorType.CUSTOM)
        self.name = "ZXMPattern"
        self.description = "基于ZXM体系的买点和吸筹形态识别指标"
    
    def calculate(self, 
                  open_prices: np.ndarray, 
                  high_prices: np.ndarray, 
                  low_prices: np.ndarray, 
                  close_prices: np.ndarray, 
                  volumes: np.ndarray, 
                  *args, **kwargs) -> Dict[str, np.ndarray]:
        """
        计算ZXM体系的买点和吸筹形态指标
        
        Args:
            open_prices: 开盘价数组
            high_prices: 最高价数组
            low_prices: 最低价数组
            close_prices: 收盘价数组
            volumes: 成交量数组
            
        Returns:
            包含各种买点和吸筹形态识别结果的字典
        """
        # 计算基础指标
        ma5 = ma(close_prices, 5)
        ma10 = ma(close_prices, 10)
        ma20 = ma(close_prices, 20)
        ma30 = ma(close_prices, 30)
        ma60 = ma(close_prices, 60)
        
        ema12 = ema(close_prices, 12)
        ema26 = ema(close_prices, 26)
        
        # 计算MACD
        dif, dea, macd_hist = macd(close_prices)
        
        # 计算KDJ
        k, d, j = kdj(close_prices, high_prices, low_prices)
        
        # 成交量相关指标
        vol_ma5 = ma(volumes, 5)
        vol_ma10 = ma(volumes, 10)
        
        # 定义结果字典
        result = {}
        
        # 识别买点形态
        result.update(self._identify_buy_points(
            open_prices, high_prices, low_prices, close_prices, volumes,
            ma5, ma10, ma20, ma30, ma60, dif, dea, macd_hist, k, d, j,
            vol_ma5, vol_ma10
        ))
        
        # 识别吸筹形态
        result.update(self._identify_absorption_patterns(
            open_prices, high_prices, low_prices, close_prices, volumes,
            ma5, ma10, ma20, ma30, ma60, dif, dea, macd_hist, k, d, j,
            vol_ma5, vol_ma10
        ))
        
        return result
        
    def _identify_buy_points(self, 
                             open_prices: np.ndarray, 
                             high_prices: np.ndarray, 
                             low_prices: np.ndarray, 
                             close_prices: np.ndarray, 
                             volumes: np.ndarray,
                             ma5: np.ndarray,
                             ma10: np.ndarray,
                             ma20: np.ndarray,
                             ma30: np.ndarray,
                             ma60: np.ndarray,
                             dif: np.ndarray,
                             dea: np.ndarray,
                             macd_hist: np.ndarray,
                             k: np.ndarray,
                             d: np.ndarray,
                             j: np.ndarray,
                             vol_ma5: np.ndarray,
                             vol_ma10: np.ndarray) -> Dict[str, np.ndarray]:
        """
        识别ZXM体系买点形态
        
        Returns:
            包含各种买点识别结果的字典
        """
        result = {}
        length = len(close_prices)
        
        # 一类买点：主升浪启动
        class_one_buy = np.zeros(length, dtype=bool)
        for i in range(10, length):
            # 前期横盘整理
            is_sideways = np.std(close_prices[i-10:i-1]) / np.mean(close_prices[i-10:i-1]) < 0.03
            # 放量突破
            volume_breakout = volumes[i] > vol_ma5[i] * 1.5
            # MA5上穿MA10
            ma_golden_cross = ma5[i] > ma10[i] and ma5[i-1] <= ma10[i-1]
            # MACD金叉且红柱扩大
            macd_golden_cross = dif[i] > dea[i] and dif[i-1] <= dea[i-1]
            macd_hist_increase = macd_hist[i] > macd_hist[i-1] > 0
            # KDJ三线金叉
            kdj_golden_cross = k[i] > d[i] and j[i] > k[i] and k[i-1] <= d[i-1]
            # 价格突破前期高点
            price_breakout = close_prices[i] > highest(close_prices[:i], 20)[-1]
            
            # 组合条件判断一类买点
            if (is_sideways and volume_breakout and 
                (ma_golden_cross or macd_golden_cross) and 
                (kdj_golden_cross or macd_hist_increase) and
                price_breakout):
                class_one_buy[i] = True
        
        result['class_one_buy'] = class_one_buy
        
        # 二类买点：主升浪调整后
        class_two_buy = np.zeros(length, dtype=bool)
        for i in range(20, length):
            # 前期上涨趋势
            uptrend = ma5[i-10] > ma20[i-10] and ma20[i-10] > ma60[i-10]
            # 回调至MA20/MA30支撑
            ma_support = (low_prices[i] <= ma20[i] * 1.02 and close_prices[i] > ma20[i]) or \
                        (low_prices[i] <= ma30[i] * 1.02 and close_prices[i] > ma30[i])
            # 回调时量能萎缩
            volume_shrink = volumes[i] < vol_ma5[i] * 0.8
            # MACD未跌破零轴
            macd_above_zero = dif[i] > 0
            # KDJ超卖回转
            kdj_oversold_turn = k[i] < 30 and k[i] > k[i-1] and k[i-1] < k[i-2]
            # 回调幅度控制在30%以内
            max_high = highest(high_prices[i-20:i], 20)[0]
            pullback_range = (max_high - low_prices[i]) / max_high < 0.3
            
            # 组合条件判断二类买点
            if (uptrend and ma_support and volume_shrink and 
                macd_above_zero and (kdj_oversold_turn or pullback_range)):
                class_two_buy[i] = True
        
        result['class_two_buy'] = class_two_buy
        
        # 三类买点：超跌反弹
        class_three_buy = np.zeros(length, dtype=bool)
        for i in range(5, length):
            # 连续下跌
            down_trend = all(close_prices[i-j] < close_prices[i-j-1] for j in range(1, 5))
            # 带长下影线的K线
            long_lower_shadow = (low_prices[i] < low_prices[i-1]) and \
                              ((close_prices[i] - low_prices[i]) > (close_prices[i] - open_prices[i]) * 2)
            # RSI超卖区回升(使用KDJ的K值模拟)
            oversold_bounce = k[i] < 20 and k[i] > k[i-1]
            # 成交量见底回升
            volume_bounce = volumes[i] > volumes[i-1] and volumes[i-1] < vol_ma5[i-1]
            # 股价接近前期大底
            near_bottom = low_prices[i] <= lowest(low_prices[:i], 60)[-1] * 1.05
            
            # 组合条件判断三类买点
            if (down_trend and (long_lower_shadow or oversold_bounce) and 
                (volume_bounce or near_bottom)):
                class_three_buy[i] = True
        
        result['class_three_buy'] = class_three_buy
        
        # ZXM特有买点：强势突破回踩型
        breakout_pullback_buy = np.zeros(length, dtype=bool)
        for i in range(10, length):
            if i < length - 1:  # 确保我们可以看到下一天的数据
                # 前期突破
                prev_breakout = close_prices[i-3] > highest(close_prices[i-10:i-3], 7)[-1] and volumes[i-3] > vol_ma5[i-3]
                # 小幅回踩不破颈线位
                neckline = min(close_prices[i-5:i-2])
                pullback_not_break = low_prices[i] >= neckline * 0.98 and close_prices[i] > close_prices[i-1]
                # 再次上攻
                rebound = close_prices[i+1] > close_prices[i] and volumes[i+1] > volumes[i]
                
                if prev_breakout and pullback_not_break and rebound:
                    breakout_pullback_buy[i] = True
        
        result['breakout_pullback_buy'] = breakout_pullback_buy
        
        # ZXM特有买点：连续缩量平台型
        volume_shrink_platform_buy = np.zeros(length, dtype=bool)
        for i in range(5, length):
            if i < length - 1:  # 确保我们可以看到下一天的数据
                # 连续3-5日缩量
                consecutive_shrink = all(volumes[i-j] < vol_ma5[i-j] for j in range(5))
                # 横盘整理
                sideways = np.std(close_prices[i-5:i+1]) / np.mean(close_prices[i-5:i+1]) < 0.02
                # KDJ底部金叉
                kdj_bottom_cross = k[i] > d[i] and k[i-1] <= d[i-1] and k[i] < 30
                # 突破时放量
                breakout_volume = volumes[i+1] > vol_ma5[i] * 1.3
                
                if consecutive_shrink and sideways and kdj_bottom_cross and breakout_volume:
                    volume_shrink_platform_buy[i] = True
        
        result['volume_shrink_platform_buy'] = volume_shrink_platform_buy
        
        # ZXM特有买点：长下影线支撑型
        long_shadow_support_buy = np.zeros(length, dtype=bool)
        for i in range(5, length):
            if i < length - 1:  # 确保我们可以看到下一天的数据
                # 在支撑位附近
                near_support = (low_prices[i] <= ma60[i] * 1.02 and close_prices[i] > ma60[i]) or \
                              (low_prices[i] <= lowest(low_prices[i-20:i], 20)[-1] * 1.02)
                # 带长下影线
                lower_shadow_len = close_prices[i] - low_prices[i]
                body_len = abs(close_prices[i] - open_prices[i])
                long_shadow = lower_shadow_len > body_len * 2
                # 第二天确认
                confirmation = close_prices[i+1] > close_prices[i]
                
                if near_support and long_shadow and confirmation:
                    long_shadow_support_buy[i] = True
        
        result['long_shadow_support_buy'] = long_shadow_support_buy
        
        # ZXM特有买点：均线粘合发散型
        ma_converge_diverge_buy = np.zeros(length, dtype=bool)
        for i in range(20, length):
            if i < length - 1:  # 确保我们可以看到下一天的数据
                # 均线粘合
                ma_converge = abs(ma5[i] - ma10[i]) / ma10[i] < 0.01 and \
                             abs(ma10[i] - ma20[i]) / ma20[i] < 0.01
                # 成交量萎缩
                volume_dry = volumes[i] < min(volumes[i-10:i])
                # 首次放量突破
                first_volume_breakout = volumes[i+1] > vol_ma5[i] * 1.5 and close_prices[i+1] > close_prices[i]
                
                if ma_converge and volume_dry and first_volume_breakout:
                    ma_converge_diverge_buy[i] = True
        
        result['ma_converge_diverge_buy'] = ma_converge_diverge_buy
        
        return result
    
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算ZXM体系买点和吸筹形态指标的原始评分
        
        Args:
            data: 输入数据，包含OHLCV数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 评分结果，0-100分
        """
        # 确保数据包含必需的列
        required_columns = ["open", "high", "low", "close", "volume"]
        for col in required_columns:
            if col not in data.columns:
                return pd.Series(50.0, index=data.index)  # 返回默认中性评分
        
        # 计算指标
        result = self.calculate(
            data["open"].values, 
            data["high"].values, 
            data["low"].values, 
            data["close"].values, 
            data["volume"].values
        )
        
        # 初始化评分为基础分50分（中性）
        score = pd.Series(50.0, index=data.index)
        
        # 买点形态评分
        for i, idx in enumerate(data.index):
            current_score = 50.0
            
            # 一类买点（最强买点）：+40分
            if i < len(result.get('class_one_buy', [])) and result['class_one_buy'][i]:
                current_score += 40
            
            # 二类买点（强势回调买点）：+30分
            elif i < len(result.get('class_two_buy', [])) and result['class_two_buy'][i]:
                current_score += 30
            
            # 三类买点（超跌反弹买点）：+20分
            elif i < len(result.get('class_three_buy', [])) and result['class_three_buy'][i]:
                current_score += 20
            
            # 其他特殊买点：+15-25分
            special_buy_signals = [
                'breakout_pullback_buy',      # 强势突破回踩型：+25分
                'volume_shrink_platform_buy', # 连续缩量平台型：+20分
                'long_shadow_support_buy',    # 长下影线支撑型：+15分
                'ma_converge_diverge_buy'     # 均线粘合发散型：+20分
            ]
            
            for signal_name in special_buy_signals:
                if signal_name in result and i < len(result[signal_name]) and result[signal_name][i]:
                    if signal_name == 'breakout_pullback_buy':
                        current_score += 25
                    elif signal_name in ['volume_shrink_platform_buy', 'ma_converge_diverge_buy']:
                        current_score += 20
                    elif signal_name == 'long_shadow_support_buy':
                        current_score += 15
                    break  # 只取最高的一个特殊买点信号
            
            # 吸筹形态评分（如果有这些指标）
            absorption_signals = [
                'large_scale_absorption',     # 大级别吸筹：+15分
                'stealth_absorption',         # 隐蔽性吸筹：+10分
                'repeated_absorption',        # 反复性吸筹：+12分
                'breakthrough_absorption'     # 突破性吸筹：+18分
            ]
            
            for signal_name in absorption_signals:
                if signal_name in result and i < len(result[signal_name]) and result[signal_name][i]:
                    if signal_name == 'large_scale_absorption':
                        current_score += 15
                    elif signal_name == 'stealth_absorption':
                        current_score += 10
                    elif signal_name == 'repeated_absorption':
                        current_score += 12
                    elif signal_name == 'breakthrough_absorption':
                        current_score += 18
            
            score[idx] = min(100, max(0, current_score))  # 确保评分在0-100范围内
        
        return score
    
def _identify_absorption_patterns(self, 
                                     open_prices: np.ndarray, 
                                     high_prices: np.ndarray, 
                                     low_prices: np.ndarray, 
                                     close_prices: np.ndarray, 
                                     volumes: np.ndarray,
                                     ma5: np.ndarray,
                                     ma10: np.ndarray,
                                     ma20: np.ndarray,
                                     ma30: np.ndarray,
                                     ma60: np.ndarray,
                                     dif: np.ndarray,
                                     dea: np.ndarray,
                                     macd_hist: np.ndarray,
                                     k: np.ndarray,
                                     d: np.ndarray,
                                     j: np.ndarray,
                                     vol_ma5: np.ndarray,
                                     vol_ma10: np.ndarray) -> Dict[str, np.ndarray]:
        """
        识别ZXM体系吸筹形态
        
        Returns:
            包含各种吸筹形态识别结果的字典
        """
        result = {}
        length = len(close_prices)
        
        # 初期吸筹特征：缩量阴线
        volume_decrease = np.zeros(length, dtype=bool)
        for i in range(5, length):
            # 阴线
            is_down = close_prices[i] < open_prices[i]
            # 缩量
            is_volume_shrink = volumes[i] < vol_ma5[i] * 0.8
            # 实体小
            small_body = abs(close_prices[i] - open_prices[i]) / close_prices[i] < 0.02
            
            if is_down and is_volume_shrink and small_body:
                volume_decrease[i] = True
        
        result['volume_decrease'] = volume_decrease
        
        # 初期吸筹特征：均线下趋势变缓
        decline_slow_down = np.zeros(length, dtype=bool)
        for i in range(20, length):
            # 计算前后10天MA20的斜率
            prev_slope = (ma20[i-10] - ma20[i-20]) / 10
            curr_slope = (ma20[i] - ma20[i-10]) / 10
            
            # 斜率变缓但仍为负
            slope_changing = curr_slope < 0 and curr_slope > prev_slope * 0.5
            
            if slope_changing:
                decline_slow_down[i] = True
        
        result['decline_slow_down'] = decline_slow_down
        
        # 初期吸筹特征：股价下跌幅度递减
        decline_reduce = np.zeros(length, dtype=bool)
        for i in range(15, length):
            # 计算前后几波下跌的幅度
            if i >= 30:
                prev_decline = (highest(close_prices[i-30:i-15], 15)[0] - 
                               lowest(close_prices[i-30:i-15], 15)[0]) / highest(close_prices[i-30:i-15], 15)[0]
                curr_decline = (highest(close_prices[i-15:i], 15)[0] - 
                               lowest(close_prices[i-15:i], 15)[0]) / highest(close_prices[i-15:i], 15)[0]
                
                # 下跌幅度递减
                decline_reducing = curr_decline < prev_decline * 0.7
                
                if decline_reducing:
                    decline_reduce[i] = True
        
        result['decline_reduce'] = decline_reduce
        
        # 中期吸筹特征：关键价位精准支撑
        key_support_hold = np.zeros(length, dtype=bool)
        for i in range(20, length):
            # MA60精准支撑
            ma60_support = low_prices[i] <= ma60[i] * 1.01 and close_prices[i] > ma60[i]
            # 前期低点支撑
            prev_low_support = low_prices[i] <= lowest(low_prices[i-60:i-1], 59)[0] * 1.01 and close_prices[i] > lowest(low_prices[i-60:i-1], 59)[0]
            
            if ma60_support or prev_low_support:
                key_support_hold[i] = True
        
        result['key_support_hold'] = key_support_hold
        
        # 中期吸筹特征：MACD二次背离
        macd_double_diverge = np.zeros(length, dtype=bool)
        for i in range(30, length):
            # 寻找前后两个低点
            if i >= 60:
                # 找到前一个低点
                first_low_idx = np.argmin(close_prices[i-60:i-30]) + i-60
                first_low_price = close_prices[first_low_idx]
                first_low_macd = dif[first_low_idx]
                
                # 找到当前低点
                second_low_idx = np.argmin(close_prices[i-30:i]) + i-30
                second_low_price = close_prices[second_low_idx]
                second_low_macd = dif[second_low_idx]
                
                # 判断价格创新低但MACD未创新低(二次背离)
                if second_low_price < first_low_price and second_low_macd > first_low_macd:
                    macd_double_diverge[i] = True
        
        result['macd_double_diverge'] = macd_double_diverge
        
        # 后期吸筹特征：缩量横盘整理
        volume_shrink_range = np.zeros(length, dtype=bool)
        for i in range(10, length):
            # 横盘整理
            is_sideways = np.std(close_prices[i-10:i+1]) / np.mean(close_prices[i-10:i+1]) < 0.03
            # 成交量持续萎缩
            volume_shrinking = all(volumes[i-j] < vol_ma10[i-j] for j in range(5))
            
            if is_sideways and volume_shrinking:
                volume_shrink_range[i] = True
        
        result['volume_shrink_range'] = volume_shrink_range
        
        # 后期吸筹特征：均线开始粘合
        ma_convergence = np.zeros(length, dtype=bool)
        for i in range(20, length):
            # 均线粘合度计算
            ma5_ma10_diff = abs(ma5[i] - ma10[i]) / ma10[i]
            ma10_ma20_diff = abs(ma10[i] - ma20[i]) / ma20[i]
            
            # 前期发散，现在粘合
            prev_diverge = abs(ma5[i-10] - ma10[i-10]) / ma10[i-10] > 0.03
            now_converge = ma5_ma10_diff < 0.01 and ma10_ma20_diff < 0.015
            
            if prev_diverge and now_converge:
                ma_convergence[i] = True
        
        result['ma_convergence'] = ma_convergence
        
        # 后期吸筹特征：MACD零轴附近徘徊
        macd_zero_hover = np.zeros(length, dtype=bool)
        for i in range(10, length):
            # MACD在零轴附近波动
            near_zero = abs(dif[i]) < 0.1 * np.std(close_prices[i-10:i+1])
            hovering = np.std(dif[i-5:i+1]) < 0.05 * np.std(close_prices[i-10:i+1])
            
            if near_zero and hovering:
                macd_zero_hover[i] = True
        
        result['macd_zero_hover'] = macd_zero_hover
        
        # ZXM特有吸筹形态：长下影线收单阳
        long_lower_shadow = np.zeros(length, dtype=bool)
        for i in range(1, length):
            # 在关键支撑位
            is_support = (low_prices[i] <= ma60[i] * 1.02 and close_prices[i] > ma60[i]) or \
                        (low_prices[i] <= lowest(low_prices[max(0, i-60):i], min(60, i))[0] * 1.02)
            # 长下影线
            lower_shadow = close_prices[i] - low_prices[i]
            body = abs(close_prices[i] - open_prices[i])
            is_long_shadow = lower_shadow > body * 2
            # 收阳线
            is_up = close_prices[i] > open_prices[i]
            
            if is_support and is_long_shadow and is_up:
                long_lower_shadow[i] = True
        
        result['long_lower_shadow'] = long_lower_shadow
        
        # ZXM特有吸筹形态：沿均线回调精准支撑
        ma_precise_support = np.zeros(length, dtype=bool)
        for i in range(20, length):
            # MA20精准支撑
            ma20_support = low_prices[i] <= ma20[i] * 1.01 and close_prices[i] > ma20[i]
            # MA60精准支撑
            ma60_support = low_prices[i] <= ma60[i] * 1.01 and close_prices[i] > ma60[i]
            
            if ma20_support or ma60_support:
                ma_precise_support[i] = True
        
        result['ma_precise_support'] = ma_precise_support
        
        # ZXM特有吸筹形态：连续阴阳小实体交替
        small_alternating = np.zeros(length, dtype=bool)
        for i in range(5, length):
            # 小实体
            small_bodies = all(abs(close_prices[i-j] - open_prices[i-j]) / close_prices[i-j] < 0.02 for j in range(5))
            # 阴阳交替
            alternating = all((close_prices[i-j] > open_prices[i-j]) != (close_prices[i-j-1] > open_prices[i-j-1]) for j in range(4))
            
            if small_bodies and alternating:
                small_alternating[i] = True
        
        result['small_alternating'] = small_alternating
        
        return result


# 测试代码
if __name__ == "__main__":
    # 可以添加简单测试用例
    import numpy as np
    
    # 模拟数据
    length = 100
    open_prices = np.random.normal(100, 10, length)
    high_prices = open_prices + np.random.normal(5, 2, length)
    low_prices = open_prices - np.random.normal(5, 2, length)
    close_prices = open_prices + np.random.normal(0, 3, length)
    volumes = np.random.normal(10000, 3000, length)
    
    # 创建ZXM识别器
    zxm_indicator = ZXMPatternIndicator()
    
    # 计算结果
    result = zxm_indicator.calculate(
        open_prices, high_prices, low_prices, close_prices, volumes
    )
    
    # 打印结果
    for key, value in result.items():
        print(f"{key}: {np.sum(value)} signals") 