import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple, List

from indicators.base_indicator import BaseIndicator
from utils.market_analysis import MarketAnalyzer


class EnhancedDMI(BaseIndicator):
    """
    增强型DMI指标
    
    具有以下增强特性:
    1. 自适应周期设计: 根据市场波动率动态调整周期
    2. ADX强度分级系统: 分析趋势强度，提供更精确的趋势强度量化
    3. DI交叉质量评估: 评估交叉角度、ADX支持度和分离速度
    4. 三线协同分析: 分析ADX、+DI和-DI三线协同关系
    5. 市场环境自适应: 在不同市场环境下动态调整评分标准
    """

    def __init__(self, period: int = 14, adx_period: int = 14, adaptive: bool = True):
        """
        初始化增强型DMI指标
        
        Args:
            period (int): DMI计算周期
            adx_period (int): ADX计算周期
            adaptive (bool): 是否启用自适应周期
        """
        super().__init__()
        self.base_period = period
        self.adx_period = adx_period
        self.adaptive = adaptive
        self.market_environment = "normal"
        
        # 内部参数
        self.period = period  # 实际使用的周期（可能根据波动率调整）
        self._result = None
        
    def set_market_environment(self, environment: str) -> None:
        """
        设置市场环境
        
        Args:
            environment (str): 市场环境类型 ('bull_market', 'bear_market', 'sideways_market', 'volatile_market', 'normal')
        """
        valid_environments = ['bull_market', 'bear_market', 'sideways_market', 'volatile_market', 'normal']
        if environment not in valid_environments:
            raise ValueError(f"无效的市场环境类型: {environment}。有效类型: {valid_environments}")
        
        self.market_environment = environment
    
    def adjust_period_by_volatility(self, data: pd.DataFrame) -> int:
        """
        根据市场波动率调整计算周期
        
        Args:
            data (pd.DataFrame): 价格数据
            
        Returns:
            int: 调整后的周期
        """
        if not self.adaptive:
            return self.base_period
        
        # 计算当前波动率与长期波动率的比值
        close_price = data['close']
        current_volatility = close_price.pct_change().rolling(20).std()
        long_term_volatility = close_price.pct_change().rolling(120).std()
        
        # 确保数据足够
        if current_volatility.isna().all() or long_term_volatility.isna().all():
            return self.base_period
        
        # 填充可能的NaN值
        current_volatility = current_volatility.fillna(0.01)
        long_term_volatility = long_term_volatility.fillna(0.01)
        
        # 计算相对波动率
        relative_volatility = current_volatility / long_term_volatility
        latest_relative_volatility = relative_volatility.iloc[-1]
        
        # 根据相对波动率调整周期
        if latest_relative_volatility > 1.5:  # 高波动环境
            adjusted_period = max(self.base_period - 4, 6)  # 缩短周期，提高灵敏度
        elif latest_relative_volatility < 0.7:  # 低波动环境
            adjusted_period = min(self.base_period + 4, 26)  # 延长周期，减少干扰
        else:  # 正常波动环境
            adjusted_period = self.base_period
            
        return adjusted_period
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算DMI指标
        
        Args:
            data (pd.DataFrame): 包含OHLC数据的DataFrame
            
        Returns:
            pd.DataFrame: 包含DMI指标值的DataFrame
        """
        # 调整周期
        self.period = self.adjust_period_by_volatility(data)
        
        # 提取价格数据
        high = data['high']
        low = data['low']
        close = data['close']
        
        # 确保数据足够
        if len(data) < self.period + 1:
            return pd.DataFrame(index=data.index)
        
        # 计算True Range (TR)
        tr1 = abs(high - low)
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # 计算方向变动 (+DM, -DM)
        plus_dm = high - high.shift(1)
        minus_dm = low.shift(1) - low
        
        # 过滤条件
        plus_dm = np.where((plus_dm > 0) & (plus_dm > minus_dm), plus_dm, 0)
        minus_dm = np.where((minus_dm > 0) & (minus_dm > plus_dm), minus_dm, 0)
        
        # 计算平滑值
        smoothed_tr = self._calculate_smoothed_values(tr, self.period)
        smoothed_plus_dm = self._calculate_smoothed_values(pd.Series(plus_dm), self.period)
        smoothed_minus_dm = self._calculate_smoothed_values(pd.Series(minus_dm), self.period)
        
        # 计算方向指数 (+DI, -DI)
        plus_di = 100 * smoothed_plus_dm / smoothed_tr
        minus_di = 100 * smoothed_minus_dm / smoothed_tr
        
        # 计算方向指数差值 (DX)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        
        # 计算平均方向指数 (ADX)
        adx = self._calculate_smoothed_values(dx, self.adx_period)
        
        # 计算ADXR (ADX的平均)
        adxr = (adx + adx.shift(self.adx_period)) / 2
        
        # 创建结果DataFrame
        result = pd.DataFrame({
            'plus_di': plus_di,
            'minus_di': minus_di,
            'adx': adx,
            'adxr': adxr,
            'dx': dx,
            'tr': smoothed_tr
        }, index=data.index)
        
        self._result = result
        return result
    
    def _calculate_smoothed_values(self, series: pd.Series, period: int) -> pd.Series:
        """
        计算Wilder平滑值
        
        Args:
            series (pd.Series): 输入数据
            period (int): 周期
            
        Returns:
            pd.Series: 平滑后的数据
        """
        # 第一个值使用简单平均
        first_value = series.iloc[:period].mean()
        
        # 使用Wilder平滑公式: smoothed = (prev_smoothed * (period-1) + current) / period
        smoothed = pd.Series(np.nan, index=series.index)
        smoothed.iloc[period-1] = first_value
        
        for i in range(period, len(series)):
            smoothed.iloc[i] = (smoothed.iloc[i-1] * (period-1) + series.iloc[i]) / period
            
        return smoothed
    
    def classify_adx_strength(self, adx_value: float) -> str:
        """
        ADX强度分级
        
        Args:
            adx_value (float): ADX值
            
        Returns:
            str: 趋势强度级别
        """
        if adx_value >= 50:
            return "极强趋势"
        elif adx_value >= 40:
            return "强趋势"
        elif adx_value >= 30:
            return "中等趋势"
        elif adx_value >= 20:
            return "弱趋势"
        else:
            return "无趋势"
            
    def evaluate_di_crossover_quality(self, window: int = 5) -> pd.Series:
        """
        评估DI交叉质量
        
        Args:
            window (int): 评估窗口
            
        Returns:
            pd.Series: 交叉质量得分
        """
        if self._result is None:
            return pd.Series()
            
        plus_di = self._result['plus_di']
        minus_di = self._result['minus_di']
        adx = self._result['adx']
        
        # 检测交叉
        golden_cross = (plus_di > minus_di) & (plus_di.shift(1) <= minus_di.shift(1))
        death_cross = (plus_di < minus_di) & (plus_di.shift(1) >= minus_di.shift(1))
        
        # 计算交叉角度 (使用DI差值变化率作为代理)
        di_diff = plus_di - minus_di
        di_diff_change = di_diff - di_diff.shift(1)
        
        # 交叉质量得分
        crossover_quality = pd.Series(0, index=self._result.index)
        
        # 对于金叉
        for i in range(len(crossover_quality)):
            if i < window:
                continue
                
            if golden_cross.iloc[i]:
                # 1. 计算角度分数 (0-40分)
                angle_score = min(40, abs(di_diff_change.iloc[i]) * 20)
                
                # 2. ADX支持度 (0-30分)
                adx_support = min(30, adx.iloc[i] * 0.6)
                
                # 3. 分离速度 (0-30分)
                separation_speed = 0
                if i + window < len(di_diff):
                    future_diff = di_diff.iloc[i:i+window].max()
                    separation_speed = min(30, (future_diff - di_diff.iloc[i]) * 3)
                
                # 综合得分
                crossover_quality.iloc[i] = angle_score + adx_support + separation_speed
                
            elif death_cross.iloc[i]:
                # 1. 计算角度分数 (0-40分)
                angle_score = min(40, abs(di_diff_change.iloc[i]) * 20)
                
                # 2. ADX支持度 (0-30分)
                adx_support = min(30, adx.iloc[i] * 0.6)
                
                # 3. 分离速度 (0-30分)
                separation_speed = 0
                if i + window < len(di_diff):
                    future_diff = abs(di_diff.iloc[i:i+window].min())
                    separation_speed = min(30, (future_diff - abs(di_diff.iloc[i])) * 3)
                
                # 综合得分 (负分表示看跌信号)
                crossover_quality.iloc[i] = -(angle_score + adx_support + separation_speed)
                
        return crossover_quality
    
    def analyze_three_line_synergy(self) -> pd.DataFrame:
        """
        分析ADX、+DI和-DI三线协同关系
        
        Returns:
            pd.DataFrame: 包含协同分析结果的DataFrame
        """
        if self._result is None:
            return pd.DataFrame()
            
        plus_di = self._result['plus_di']
        minus_di = self._result['minus_di']
        adx = self._result['adx']
        
        # 创建结果DataFrame
        synergy = pd.DataFrame(index=self._result.index)
        
        # 强上升趋势: +DI > -DI, ADX > 25且上升
        synergy['strong_uptrend'] = (
            (plus_di > minus_di) & 
            (adx > 25) & 
            (adx > adx.shift(1))
        )
        
        # 强下降趋势: -DI > +DI, ADX > 25且上升
        synergy['strong_downtrend'] = (
            (minus_di > plus_di) & 
            (adx > 25) & 
            (adx > adx.shift(1))
        )
        
        # 趋势减弱: ADX下降
        synergy['weakening_trend'] = adx < adx.shift(1)
        
        # 潜在反转: ADX下降且DI差距收窄
        di_diff = abs(plus_di - minus_di)
        synergy['potential_reversal'] = (
            (adx < adx.shift(1)) & 
            (di_diff < di_diff.shift(1))
        )
        
        # 无趋势市场: ADX < 20且持平或下降
        synergy['no_trend'] = (
            (adx < 20) & 
            (adx <= adx.shift(1))
        )
        
        # 趋势初现: ADX < 20但上升，DI出现交叉
        synergy['emerging_trend'] = (
            (adx < 20) & 
            (adx > adx.shift(1)) &
            ((plus_di > minus_di) & (plus_di.shift(1) <= minus_di.shift(1)) | 
             (plus_di < minus_di) & (plus_di.shift(1) >= minus_di.shift(1)))
        )
        
        return synergy
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        
        Args:
            data (pd.DataFrame): 价格数据
            
        Returns:
            pd.DataFrame: 包含交易信号的DataFrame
        """
        if self._result is None:
            self.calculate(data)
            
        if self._result is None:
            return pd.DataFrame()
            
        # 获取DMI指标数据
        plus_di = self._result['plus_di']
        minus_di = self._result['minus_di']
        adx = self._result['adx']
        
        # 计算交叉
        golden_cross = (plus_di > minus_di) & (plus_di.shift(1) <= minus_di.shift(1))
        death_cross = (plus_di < minus_di) & (plus_di.shift(1) >= minus_di.shift(1))
        
        # 计算交叉质量
        crossover_quality = self.evaluate_di_crossover_quality()
        
        # 分析三线协同关系
        synergy = self.analyze_three_line_synergy()
        
        # 创建信号DataFrame
        signals = pd.DataFrame(index=data.index)
        signals['buy_signal'] = golden_cross
        signals['sell_signal'] = death_cross
        signals['strong_uptrend'] = synergy['strong_uptrend']
        signals['strong_downtrend'] = synergy['strong_downtrend']
        signals['potential_reversal'] = synergy['potential_reversal']
        signals['trend_strength'] = adx
        signals['score'] = self.calculate_score(data)
        signals['signal_type'] = pd.Series('', index=signals.index)
        signals['signal_desc'] = pd.Series('', index=signals.index)
        signals['confidence'] = pd.Series(0, index=signals.index)
        
        # 设置信号类型和描述
        signals.loc[golden_cross, 'signal_type'] = 'DMI金叉'
        signals.loc[golden_cross, 'signal_desc'] = '+DI上穿-DI，买入信号'
        signals.loc[death_cross, 'signal_type'] = 'DMI死叉'
        signals.loc[death_cross, 'signal_desc'] = '+DI下穿-DI，卖出信号'
        signals.loc[synergy['strong_uptrend'], 'signal_type'] = '强上升趋势'
        signals.loc[synergy['strong_uptrend'], 'signal_desc'] = '+DI > -DI, ADX > 25且上升，强上升趋势'
        signals.loc[synergy['strong_downtrend'], 'signal_type'] = '强下降趋势'
        signals.loc[synergy['strong_downtrend'], 'signal_desc'] = '-DI > +DI, ADX > 25且上升，强下降趋势'
        
        # 设置信号置信度
        signals.loc[golden_cross, 'confidence'] = crossover_quality[golden_cross].abs()
        signals.loc[death_cross, 'confidence'] = crossover_quality[death_cross].abs()
        signals.loc[synergy['strong_uptrend'], 'confidence'] = 80
        signals.loc[synergy['strong_downtrend'], 'confidence'] = 80
        
        return signals
    
    def calculate_score(self, data: pd.DataFrame = None) -> pd.Series:
        """
        计算DMI综合评分 (0-100)
        
        Args:
            data (pd.DataFrame, optional): 价格数据，如果未提供则使用上次计算结果
            
        Returns:
            pd.Series: 评分 (0-100，50为中性)
        """
        if self._result is None and data is not None:
            self.calculate(data)
            
        if self._result is None:
            return pd.Series()
            
        # 获取DMI指标数据
        plus_di = self._result['plus_di']
        minus_di = self._result['minus_di']
        adx = self._result['adx']
        
        # 基础分数为50（中性）
        score = pd.Series(50, index=self._result.index)
        
        # 1. DI位置关系评分 (±15分)
        di_diff = plus_di - minus_di
        di_ratio = di_diff / ((plus_di + minus_di) / 2)  # 归一化差异
        di_position_score = di_ratio * 15
        score += di_position_score
        
        # 2. ADX强度评分 (±15分)
        # 高ADX值表示趋势强，与DI方向结合判断加减分
        adx_strength = adx / 50  # 归一化ADX (50视为满分)
        adx_strength = adx_strength.clip(0, 1)
        adx_score = adx_strength * 15
        adx_score = np.where(di_diff > 0, adx_score, -adx_score)  # 根据DI方向确定加减分
        score += adx_score
        
        # 3. DI交叉质量评分 (±20分)
        crossover_quality = self.evaluate_di_crossover_quality()
        # 将交叉质量评分（可能超过100）归一化到±20分范围
        normalized_quality = crossover_quality / 100 * 20
        score += normalized_quality
        
        # 4. 趋势强度变化评分 (±10分)
        adx_change = adx - adx.shift(1)
        adx_change_score = np.where(
            di_diff > 0,  # 多头趋势
            np.where(adx_change > 0, adx_change, adx_change * 0.5) * 10,  # ADX上升加分，下降少量减分
            np.where(adx_change > 0, -adx_change, -adx_change * 0.5) * 10  # 空头趋势，ADX上升减分，下降少量加分
        )
        score += adx_change_score
        
        # 5. 根据市场环境调整评分
        if self.market_environment == "bull_market":
            # 牛市中增强多头信号，降低空头信号权重
            bull_adjustment = np.where(score > 50, (score - 50) * 0.2, (score - 50) * 0.1)
            score += bull_adjustment
        elif self.market_environment == "bear_market":
            # 熊市中增强空头信号，降低多头信号权重
            bear_adjustment = np.where(score < 50, (50 - score) * 0.2, (50 - score) * 0.1)
            score -= bear_adjustment
        elif self.market_environment == "volatile_market":
            # 高波动市场中，需要更强的信号才能确认
            vol_adjustment = np.abs(score - 50) * 0.3
            score = np.where(score > 50, 50 + vol_adjustment, 50 - vol_adjustment)
        
        # 限制得分范围在0-100之间
        score = score.clip(0, 100)
        
        return score
    
    def identify_patterns(self) -> pd.DataFrame:
        """
        识别DMI形态
        
        Returns:
            pd.DataFrame: 包含形态识别结果的DataFrame
        """
        if self._result is None:
            return pd.DataFrame()
            
        plus_di = self._result['plus_di']
        minus_di = self._result['minus_di']
        adx = self._result['adx']
        
        patterns = pd.DataFrame(index=self._result.index)
        
        # 1. 趋势开始形态：ADX从低位开始上升，且DI出现交叉
        patterns['trend_start'] = (
            (adx < 20) & 
            (adx > adx.shift(1)) & 
            (adx.shift(1) > adx.shift(2)) &
            ((plus_di > minus_di) & (plus_di.shift(2) <= minus_di.shift(2)) | 
             (plus_di < minus_di) & (plus_di.shift(2) >= minus_di.shift(2)))
        )
        
        # 2. 趋势加速形态：ADX快速上升，且DI差距扩大
        di_diff = abs(plus_di - minus_di)
        patterns['trend_acceleration'] = (
            (adx > 25) & 
            (adx > adx.shift(1) * 1.05) &  # ADX加速上升
            (di_diff > di_diff.shift(1) * 1.05)  # DI差距扩大
        )
        
        # 3. 趋势衰竭形态：ADX从高位开始下降，但DI差距仍大
        patterns['trend_exhaustion'] = (
            (adx > 35) & 
            (adx < adx.shift(1)) & 
            (adx.shift(1) < adx.shift(2)) &  # ADX连续下降
            (di_diff > 15)  # DI差距仍大
        )
        
        # 4. 趋势反转前兆：ADX下降，DI差距收窄
        patterns['trend_reversal_warning'] = (
            (adx > 25) & 
            (adx < adx.shift(1)) & 
            (di_diff < di_diff.shift(1)) &
            (di_diff.shift(1) < di_diff.shift(2))  # DI差距连续收窄
        )
        
        # 5. 无趋势区间形态：ADX低位徘徊
        patterns['no_trend_zone'] = (
            (adx < 15) & 
            (abs(adx - adx.shift(3)) < 3)  # ADX变化小
        )
        
        # 6. 强势趋势形态：ADX高位保持强势，DI差距大
        patterns['strong_trend'] = (
            (adx > 40) & 
            (adx > adx.shift(5)) &  # 长期上升
            (di_diff > 20)  # DI差距大
        )
        
        # 7. DI交叉但ADX下降：虚假交叉信号
        patterns['false_cross'] = (
            ((plus_di > minus_di) & (plus_di.shift(1) <= minus_di.shift(1)) | 
             (plus_di < minus_di) & (plus_di.shift(1) >= minus_di.shift(1))) &
            (adx < adx.shift(1)) &
            (adx < 20)
        )
        
        # 8. ADX回调后再次上升：趋势延续确认
        patterns['trend_continuation'] = (
            (adx > 25) &
            (adx > adx.shift(1)) &
            (adx.shift(1) < adx.shift(2)) &
            (adx.shift(2) < adx.shift(3)) &
            (adx.shift(3) > adx.shift(4))  # ADX先下降再上升
        )
        
        return patterns 