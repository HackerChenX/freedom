"""
移动平均线指标模块

实现各种移动平均线(MA)指标计算，支持自适应周期优化和筹码分布融合
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Any, Tuple
from indicators.pattern_recognition import SignalStrength, PatternResult

from indicators.base_indicator import BaseIndicator
from indicators.common import ma, ema, wma, sma
from indicators.chip_distribution import ChipDistribution
from utils.logger import get_logger

logger = get_logger(__name__)


class MA(BaseIndicator):
    """
    移动平均线指标类
    
    计算各种类型的移动平均线，支持自适应周期和筹码加权
    """
    
    # MA类型枚举
    MA_TYPE_SIMPLE = 'simple'  # 简单移动平均
    MA_TYPE_EMA = 'ema'        # 指数移动平均
    MA_TYPE_WMA = 'wma'        # 加权移动平均
    MA_TYPE_SMA = 'sma'        # 平滑移动平均
    MA_TYPE_ADAPTIVE = 'adaptive'  # 自适应移动平均
    MA_TYPE_CHIP_WEIGHTED = 'chip_weighted'  # 筹码加权移动平均
    
    def __init__(self, name: str = "MA", description: str = "移动平均线指标", periods: List[int] = None):
        """
        初始化移动平均线指标
        
        Args:
            name: 指标名称
            description: 指标描述
            periods: 周期列表，默认为[5, 10, 20, 30, 60]
        """
        super().__init__(name, description)
        
        # 设置默认参数
        self._parameters = {
            'periods': periods or [5, 10, 20, 30, 60],  # 周期列表
            'ma_type': self.MA_TYPE_SIMPLE,  # MA类型
            'weight': 1,                    # SMA的权重参数
            'adaptive': False,              # 是否启用自适应周期
            'volatility_window': 60,        # 波动率计算窗口
            'min_period_factor': 0.5,       # 最小周期因子(减少基础周期的倍数)
            'max_period_factor': 2.0,       # 最大周期因子(增加基础周期的倍数)
            'chip_weighted': False,         # 是否启用筹码加权
            'chip_half_life': 60,           # 筹码半衰期
            'price_precision': 0.01,        # 价格精度
            'chip_weight': 0.5              # 筹码权重
        }
        
        # 初始化筹码分布计算器
        self._chip_distribution = ChipDistribution()
        
        # 注册MA形态
        self._register_ma_patterns()
    
    @property
    def parameters(self) -> Dict[str, Any]:
        """获取参数"""
        return self._parameters.copy()
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """
        设置参数
        
        Args:
            params: 参数字典
        """
        for key, value in params.items():
            if key in self._parameters:
                self._parameters[key] = value
    
    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算移动平均线指标
        
        Args:
            data: 输入数据，必须包含'close'列
            args: 位置参数
            kwargs: 关键字参数，可包含periods、ma_type和weight
            
        Returns:
            pd.DataFrame: 包含各周期MA的DataFrame
        """
        # 确保数据包含close列
        self.ensure_columns(data, ['close'])
        
        # 获取参数
        periods = kwargs.get('periods', self._parameters['periods'])
        ma_type = kwargs.get('ma_type', self._parameters['ma_type'])
        weight = kwargs.get('weight', self._parameters['weight'])
        adaptive = kwargs.get('adaptive', self._parameters['adaptive'])
        chip_weighted = kwargs.get('chip_weighted', self._parameters['chip_weighted'])
        
        # 如果periods是单个值，转换为列表
        if not isinstance(periods, list):
            periods = [periods]
        
        # 初始化结果DataFrame
        result = pd.DataFrame(index=data.index)
        
        # 处理不同类型的MA计算
        if ma_type == self.MA_TYPE_ADAPTIVE or adaptive:
            # 自适应周期MA计算
            for period in periods:
                adaptive_period = self.adjust_period_by_volatility(data['close'], period)
                ma_values = self._calculate_ma_with_period(data['close'], adaptive_period, ma_type, weight)
                result[f'MA{period}'] = ma_values
                # 保存自适应周期供分析使用
                result[f'MA{period}_adaptive_period'] = adaptive_period
        
        elif ma_type == self.MA_TYPE_CHIP_WEIGHTED or chip_weighted:
            # 筹码加权MA计算
            chip_data = self.calculate_chip_weighted_ma(data)
            for period in periods:
                # 基础MA计算
                if ma_type == self.MA_TYPE_EMA:
                    base_ma = ema(data['close'], period)
                elif ma_type == self.MA_TYPE_WMA:
                    base_ma = wma(data['close'], period)
                elif ma_type == self.MA_TYPE_SMA:
                    base_ma = sma(data['close'], period, weight)
                else:
                    base_ma = ma(data['close'], period)
                
                # 筹码加权
                # 使用动态筹码权重
                if 'chip_dynamic_weight' in chip_data.columns:
                    chip_weight = chip_data['chip_dynamic_weight']
                else:
                    chip_weight = self._parameters['chip_weight']
                
                # 使用主力成本均线或平均成本线
                if 'institutional_cost' in chip_data.columns:
                    # 结合普通筹码成本和主力筹码成本
                    avg_cost = chip_data['avg_cost']
                    inst_cost = chip_data['institutional_cost']
                    
                    # 根据机构筹码所占比例调整权重
                    if 'institutional_chips' in chip_data.columns:
                        inst_ratio = chip_data['institutional_chips'].rolling(window=10).mean()
                        # 机构筹码比例越高，越重视机构成本
                        inst_cost_weight = np.clip(inst_ratio * 2, 0.3, 0.7)
                        chip_ma = avg_cost * (1 - inst_cost_weight) + inst_cost * inst_cost_weight
                    else:
                        # 默认机构筹码权重0.5
                        chip_ma = (avg_cost + inst_cost) / 2
                else:
                    chip_ma = chip_data['avg_cost']
                
                # 考虑解套盘压力调整权重
                if 'trapped_pressure' in chip_data.columns:
                    # 解套盘压力越大，筹码权重越高
                    trapped_adjustment = 1.0 + np.clip(chip_data['trapped_pressure'] * 0.4, 0, 0.2)
                    chip_weight = chip_weight * trapped_adjustment
                
                # 加权组合：基础MA和筹码成本线的加权平均
                weighted_ma = base_ma * (1 - chip_weight) + chip_ma * chip_weight
                result[f'MA{period}'] = weighted_ma
                
                # 保存筹码相关数据供分析使用
                result[f'chip_avg_cost'] = chip_data['avg_cost']
                result[f'chip_concentration'] = chip_data['chip_concentration']
                
                # 保存新增的筹码指标
                if 'institutional_cost' in chip_data.columns:
                    result[f'institutional_cost'] = chip_data['institutional_cost']
                if 'trapped_pressure' in chip_data.columns:
                    result[f'trapped_pressure'] = chip_data['trapped_pressure']
        
        else:
            # 标准MA计算
            for period in periods:
                ma_values = self._calculate_ma_with_period(data['close'], period, ma_type, weight)
                result[f'MA{period}'] = ma_values
        
        # 保存结果
        self._result = result
        
        return result
    
    def _calculate_ma_with_period(self, series: pd.Series, period: Union[int, pd.Series], 
                                ma_type: str, weight: float) -> pd.Series:
        """
        使用指定周期计算MA
        
        Args:
            series: 输入序列
            period: 周期(可以是固定整数或自适应的Series)
            ma_type: MA类型
            weight: 权重参数
            
        Returns:
            pd.Series: MA序列
        """
        if isinstance(period, int):
            # 固定周期计算
            if ma_type == self.MA_TYPE_EMA:
                return ema(series, period)
            elif ma_type == self.MA_TYPE_WMA:
                return wma(series, period)
            elif ma_type == self.MA_TYPE_SMA:
                return sma(series, period, weight)
            else:
                return ma(series, period)
        else:
            # 自适应周期计算（周期为Series）
            result = pd.Series(np.nan, index=series.index)
            
            # 为每个点计算对应周期的MA
            for i in range(len(series)):
                if pd.isna(period.iloc[i]) or period.iloc[i] < 2:
                    continue
                
                # 获取当前点的周期
                current_period = int(period.iloc[i])
                
                # 确保有足够数据计算
                if i < current_period:
                    continue
                
                # 获取用于计算的数据段
                data_slice = series.iloc[max(0, i-current_period+1):i+1]
                
                # 根据MA类型计算
                if ma_type == self.MA_TYPE_EMA:
                    # 指数移动平均需要特殊处理
                    alpha = 2.0 / (current_period + 1)
                    if i > 0 and not pd.isna(result.iloc[i-1]):
                        result.iloc[i] = series.iloc[i] * alpha + result.iloc[i-1] * (1 - alpha)
                    else:
                        result.iloc[i] = data_slice.mean()
                elif ma_type == self.MA_TYPE_WMA:
                    # 加权移动平均
                    weights = np.arange(1, len(data_slice) + 1)
                    result.iloc[i] = (data_slice * weights).sum() / weights.sum()
                elif ma_type == self.MA_TYPE_SMA:
                    # 平滑移动平均
                    if i > 0 and not pd.isna(result.iloc[i-1]):
                        result.iloc[i] = (series.iloc[i] * weight + 
                                         result.iloc[i-1] * (current_period - weight)) / current_period
                    else:
                        result.iloc[i] = data_slice.mean()
                else:
                    # 简单移动平均
                    result.iloc[i] = data_slice.mean()
            
            return result
    
    def adjust_period_by_volatility(self, close: pd.Series, base_period: int) -> pd.Series:
        """
        根据市场波动率动态调整MA周期
        
        高波动市场减小周期提高敏感度，低波动市场增加周期减少噪音。
        同时考虑趋势强度和价格突破，进一步优化自适应周期。
        
        Args:
            close: 收盘价序列
            base_period: 基础周期
            
        Returns:
            pd.Series: 自适应周期序列
        """
        # 获取参数
        volatility_window = self._parameters['volatility_window']
        min_period_factor = self._parameters['min_period_factor']
        max_period_factor = self._parameters['max_period_factor']
        
        # 初始化自适应周期为基础周期
        adaptive_period = pd.Series(base_period, index=close.index)
        
        # 计算滚动波动率（使用对数收益率的标准差）
        returns = np.log(close / close.shift(1)).dropna()
        rolling_volatility = returns.rolling(window=volatility_window).std()
        
        # 计算趋势强度（用短期与长期方向的一致性来表示）
        short_trend = close.rolling(window=10).mean().diff()
        long_trend = close.rolling(window=30).mean().diff()
        trend_alignment = (short_trend * long_trend > 0).astype(int).rolling(window=20).mean()
        
        # 检测价格突破（用短期突破长期均线表示）
        short_ma = close.rolling(window=5).mean()
        long_ma = close.rolling(window=20).mean()
        breakout = ((short_ma > long_ma) & (short_ma.shift(5) < long_ma.shift(5))) | \
                   ((short_ma < long_ma) & (short_ma.shift(5) > long_ma.shift(5)))
        recent_breakout = breakout.rolling(window=10).sum() > 0
        
        if len(rolling_volatility.dropna()) > 0:
            # 计算波动率的历史分位数
            volatility_rank = rolling_volatility.rolling(window=volatility_window*2).rank(pct=True)
            
            # 根据波动率分位数动态调整周期
            # 高波动（分位数高）= 更短的周期，提高敏感度
            # 低波动（分位数低）= 更长的周期，减少噪音
            for i in range(volatility_window, len(close)):
                if pd.notna(volatility_rank.iloc[i]):
                    # 基本波动率调整因子
                    rank = volatility_rank.iloc[i]
                    if rank > 0.8:  # 高波动
                        period_factor = min_period_factor
                    elif rank < 0.2:  # 低波动
                        period_factor = max_period_factor
                    else:  # 正常波动
                        # 线性映射0.2-0.8到max_factor-min_factor
                        period_factor = max_period_factor - (rank - 0.2) * (max_period_factor - min_period_factor) / 0.6
                    
                    # 趋势强度调整
                    # 强趋势下稍微增加周期，减少干扰
                    if pd.notna(trend_alignment.iloc[i]) and trend_alignment.iloc[i] > 0.7:
                        period_factor *= 1.1  # 增加10%
                    
                    # 突破调整
                    # 近期发生突破时减小周期，提高敏感度
                    if pd.notna(recent_breakout.iloc[i]) and recent_breakout.iloc[i]:
                        period_factor *= 0.9  # 减少10%
                    
                    # 市场状态整合调整
                    # 检测市场状态：震荡、趋势、高波动等
                    if rank > 0.7 and (pd.notna(trend_alignment.iloc[i]) and trend_alignment.iloc[i] < 0.3):
                        # 高波动震荡市场，增加周期减少假信号
                        period_factor *= 1.2
                    elif rank < 0.3 and (pd.notna(trend_alignment.iloc[i]) and trend_alignment.iloc[i] > 0.7):
                        # 低波动强趋势，减小周期跟踪更紧密
                        period_factor *= 0.85
                    
                    # 应用调整因子到基础周期
                    adaptive_period.iloc[i] = max(2, round(base_period * period_factor))
        
        return adaptive_period
    
    def calculate_chip_weighted_ma(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        结合筹码分布计算加权MA
        
        利用筹码分布数据加权调整移动平均线，考虑筹码集中度、
        成本分布、主力筹码持仓和解套盘压力等多维度数据。
        
        Args:
            data: 输入数据，包含OHLCV数据
            
        Returns:
            pd.DataFrame: 包含筹码分布相关指标的DataFrame
        """
        # 获取参数
        chip_half_life = self._parameters['chip_half_life']
        price_precision = self._parameters['price_precision']
        
        # 确保数据包含计算筹码分布所需的列
        required_columns = ["open", "high", "low", "close", "volume"]
        self.ensure_columns(data, required_columns)
        
        # 计算基本筹码分布
        try:
            chip_data = self._chip_distribution.calculate(
                data, 
                half_life=chip_half_life, 
                price_precision=price_precision
            )
            
            # 增强筹码分布指标
            result = chip_data.copy()
            
            # 计算主力筹码成本带
            # 将筹码集中度高的区域识别为主力筹码带
            if 'chip_density' in chip_data.columns:
                # 筹码密度高的区域为主力筹码区
                high_density_threshold = chip_data['chip_density'].rolling(window=20).quantile(0.8)
                result['institutional_chips'] = chip_data['chip_density'] > high_density_threshold
                
                # 计算主力成本均线
                price_range = np.arange(
                    data['low'].min() * 0.9,
                    data['high'].max() * 1.1,
                    price_precision
                )
                
                # 计算主力筹码加权成本
                inst_cost = pd.Series(0.0, index=data.index)
                for i in range(len(data)):
                    if pd.notna(chip_data['chip_distribution'].iloc[i]):
                        chip_dist = chip_data['chip_distribution'].iloc[i]
                        density = np.array([chip_dist.get(price, 0) for price in price_range])
                        
                        # 仅考虑高密度区域
                        high_density_mask = density > np.percentile(density[density > 0], 70)
                        if np.any(high_density_mask):
                            inst_prices = price_range[high_density_mask]
                            inst_density = density[high_density_mask]
                            inst_cost.iloc[i] = np.average(inst_prices, weights=inst_density)
                        else:
                            inst_cost.iloc[i] = data['close'].iloc[i]
                
                result['institutional_cost'] = inst_cost
            
            # 计算解套盘压力
            if 'win_ratio' in chip_data.columns:
                # 套牢盘比例 = 1 - 盈利比例
                result['trapped_ratio'] = 1 - chip_data['win_ratio']
                
                # 解套盘压力 = 套牢盘比例 * 筹码集中度
                if 'chip_concentration' in chip_data.columns:
                    result['trapped_pressure'] = result['trapped_ratio'] * chip_data['chip_concentration']
            
            # 计算动态筹码权重
            # 根据市场状态动态调整筹码权重
            base_weight = self._parameters['chip_weight']
            dynamic_weight = pd.Series(base_weight, index=data.index)
            
            # 筹码集中度高时增加权重
            if 'chip_concentration' in chip_data.columns:
                concentration_factor = chip_data['chip_concentration'] / 0.5  # 标准化到1.0为基准
                dynamic_weight *= np.clip(concentration_factor, 0.8, 1.5)
            
            # 多头行情（大部分筹码盈利）时减少权重，空头行情（大部分套牢）时增加权重
            if 'win_ratio' in chip_data.columns:
                win_ratio = chip_data['win_ratio']
                # 当大部分筹码都浮盈时，市场强势，减少筹码权重
                bull_factor = np.where(win_ratio > 0.7, 0.8, 1.0)
                # 当大部分筹码都套牢时，市场弱势，增加筹码权重
                bear_factor = np.where(win_ratio < 0.3, 1.2, 1.0)
                dynamic_weight *= np.array(bull_factor) * np.array(bear_factor)
            
            # 存储动态权重
            result['chip_dynamic_weight'] = dynamic_weight
            
            return result
        except Exception as e:
            logger.error(f"计算筹码分布时发生错误: {e}")
            # 返回空数据框，使用默认MA
            empty_chip_data = pd.DataFrame(index=data.index)
            empty_chip_data['avg_cost'] = data['close']  # 默认使用收盘价
            empty_chip_data['chip_concentration'] = 0.5  # 默认中等集中度
            empty_chip_data['chip_dynamic_weight'] = self._parameters['chip_weight']  # 默认权重
            return empty_chip_data
    
    def is_uptrend(self, period: int) -> pd.Series:
        """
        判断指定周期的均线是否上升趋势
        
        Args:
            period: MA周期
            
        Returns:
            pd.Series: 趋势信号，True表示上升趋势
        """
        if not self.has_result():
            raise ValueError("必须先调用compute方法计算指标")
            
        ma_col = f'MA{period}'
        if ma_col not in self._result.columns:
            raise ValueError(f"结果中不存在{ma_col}列")
            
        return self._result[ma_col] > self._result[ma_col].shift(1)
    
    def is_golden_cross(self, short_period: int, long_period: int) -> pd.Series:
        """
        判断是否形成金叉（短期均线上穿长期均线）
        
        Args:
            short_period: 短期均线周期
            long_period: 长期均线周期
            
        Returns:
            pd.Series: 金叉信号，True表示形成金叉
        """
        if not self.has_result():
            raise ValueError("必须先调用compute方法计算指标")
            
        short_ma = f'MA{short_period}'
        long_ma = f'MA{long_period}'
        
        if short_ma not in self._result.columns or long_ma not in self._result.columns:
            raise ValueError(f"结果中不存在{short_ma}或{long_ma}列")
            
        return self.crossover(self._result[short_ma], self._result[long_ma])
    
    def is_death_cross(self, short_period: int, long_period: int) -> pd.Series:
        """
        判断是否形成死叉（短期均线下穿长期均线）
        
        Args:
            short_period: 短期均线周期
            long_period: 长期均线周期
            
        Returns:
            pd.Series: 死叉信号，True表示形成死叉
        """
        if not self.has_result():
            raise ValueError("必须先调用compute方法计算指标")
            
        short_ma = f'MA{short_period}'
        long_ma = f'MA{long_period}'
        
        if short_ma not in self._result.columns or long_ma not in self._result.columns:
            raise ValueError(f"结果中不存在{short_ma}或{long_ma}列")
            
        return self.crossunder(self._result[short_ma], self._result[long_ma])
    
    def is_multi_uptrend(self, periods: List[int] = None) -> pd.Series:
        """
        判断是否多条均线多头排列（短期均线在上，长期均线在下）
        
        Args:
            periods: 均线周期列表，按从短到长排序
            
        Returns:
            pd.Series: 多头排列信号，True表示形成多头排列
        """
        if not self.has_result():
            raise ValueError("必须先调用compute方法计算指标")
            
        if periods is None:
            # 使用默认周期列表，但要确保是有序的
            periods = sorted(self._parameters['periods'])
            
        # 确保所有需要的列都存在
        for period in periods:
            if f'MA{period}' not in self._result.columns:
                raise ValueError(f"结果中不存在MA{period}列")
        
        # 检查是否形成多头排列
        result = pd.Series(True, index=self._result.index)
        for i in range(len(periods) - 1):
            short_ma = f'MA{periods[i]}'
            long_ma = f'MA{periods[i+1]}'
            result &= (self._result[short_ma] > self._result[long_ma])
            
        return result
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算MA原始评分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 原始评分序列（0-100分）
        """
        # 确保已计算MA
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        periods = kwargs.get('periods', self._parameters['periods'])
        if not isinstance(periods, list):
            periods = [periods]
        
        # 基础评分为50分
        score = pd.Series(50.0, index=data.index)
        
        # 1. 计算价格与MA的位置关系评分（20分）
        price_ma_score = self._calculate_price_ma_score(data['close'], periods)
        
        # 2. 计算MA交叉评分（20分）
        ma_cross_score = self._calculate_ma_cross_score(periods)
        
        # 3. 计算MA趋势评分（20分）
        ma_trend_score = self._calculate_ma_trend_score(periods)
        
        # 4. 计算MA排列评分（20分）
        ma_arrangement_score = self._calculate_ma_arrangement_score(periods)
        
        # 5. 计算价格穿透强度评分（20分）
        price_penetration_score = self._calculate_price_penetration_score(data['close'], periods)
        
        # 合并各部分评分
        score += price_ma_score
        score += ma_cross_score
        score += ma_trend_score
        score += ma_arrangement_score
        score += price_penetration_score
        
        # 6. 自适应周期评分调整
        if self._parameters['adaptive'] and any(f'MA{p}_adaptive_period' in self._result.columns for p in periods):
            # 自适应周期系数：评估自适应周期调整的有效性
            adaptive_adjustment = pd.Series(0.0, index=data.index)
            
            for period in periods:
                adaptive_period_col = f'MA{period}_adaptive_period'
                if adaptive_period_col in self._result.columns:
                    adaptive_period = self._result[adaptive_period_col]
                    
                    # 计算自适应周期与基础周期的比例
                    period_ratio = adaptive_period / period
                    
                    # 根据当前价格相对MA位置，判断自适应周期调整是否有效
                    price_above_ma = data['close'] > self._result[f'MA{period}']
                    
                    # 价格在MA上方且周期缩短 = 有效调整（跟踪更紧密）
                    effective_adjustment_bull = price_above_ma & (period_ratio < 1.0)
                    
                    # 价格在MA下方且周期延长 = 有效调整（减少噪音）
                    effective_adjustment_bear = (~price_above_ma) & (period_ratio > 1.0)
                    
                    # 任一调整有效则加分
                    effective_adjustment = effective_adjustment_bull | effective_adjustment_bear
                    
                    # 根据有效调整情况加分
                    adaptive_adjustment += effective_adjustment * 5
            
            # 限制总调整在-10到+10分之间
            adaptive_adjustment = np.clip(adaptive_adjustment, -10, 10)
            score += adaptive_adjustment
        
        # 7. 筹码分布指标评分调整
        if self._parameters['chip_weighted'] and 'chip_avg_cost' in self._result.columns:
            # 筹码分布评分
            chip_adjustment = pd.Series(0.0, index=data.index)
            
            # 价格相对筹码成本的位置评分
            if 'chip_avg_cost' in self._result.columns:
                avg_cost = self._result['chip_avg_cost']
                # 价格高于筹码成本线 = 多头格局（加分）
                price_above_cost = data['close'] > avg_cost
                # 价格低于筹码成本线 = 空头格局（减分）
                price_below_cost = data['close'] < avg_cost
                
                # 根据价格与成本的距离计算分数
                distance_factor = abs((data['close'] - avg_cost) / avg_cost)
                chip_adjustment += price_above_cost * (5 + distance_factor * 5)
                chip_adjustment -= price_below_cost * (5 + distance_factor * 5)
            
            # 主力成本线评分
            if 'institutional_cost' in self._result.columns:
                inst_cost = self._result['institutional_cost']
                # 价格接近主力成本线 = 可能的支撑/阻力
                near_inst_cost = abs((data['close'] - inst_cost) / inst_cost) < 0.02
                # 价格高于主力成本线 = 主力浮盈（看多）
                above_inst_cost = data['close'] > inst_cost * 1.02
                # 价格低于主力成本线 = 主力套牢（看空）
                below_inst_cost = data['close'] < inst_cost * 0.98
                
                # 根据价格与主力成本的关系评分
                chip_adjustment += near_inst_cost * 5  # 接近主力成本线加分
                chip_adjustment += above_inst_cost * 8  # 主力浮盈加分更多
                chip_adjustment -= below_inst_cost * 8  # 主力套牢减分更多
            
            # 解套盘压力评分
            if 'trapped_pressure' in self._result.columns:
                trapped_pressure = self._result['trapped_pressure']
                # 套牢盘压力大 = 上方阻力大（减分）
                high_pressure = trapped_pressure > 0.5
                # 套牢盘压力小 = 上方阻力小（加分）
                low_pressure = trapped_pressure < 0.2
                
                # 根据套牢盘压力评分
                chip_adjustment -= high_pressure * 10
                chip_adjustment += low_pressure * 5
            
            # 筹码集中度评分
            if 'chip_concentration' in self._result.columns:
                concentration = self._result['chip_concentration']
                # 筹码集中度高 = 突破或回踩更确定
                high_concentration = concentration > 0.7
                
                # 计算筹码集中对价格突破的影响
                price_breakout = (data['close'] > data['close'].shift(5) * 1.05) & high_concentration
                price_breakdown = (data['close'] < data['close'].shift(5) * 0.95) & high_concentration
                
                # 突破加分，跌破减分，且筹码集中时信号更可靠
                chip_adjustment += price_breakout * 10
                chip_adjustment -= price_breakdown * 10
            
            # 限制筹码调整分数在-20到+20之间
            chip_adjustment = np.clip(chip_adjustment, -20, 20)
            score += chip_adjustment
        
        return np.clip(score, 0, 100)
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别移动平均线形态
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[str]: 识别出的形态列表
        """
        patterns = []
        
        # 确保已计算MA
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        # 获取周期
        periods = kwargs.get('periods', self._parameters['periods'])
        if not isinstance(periods, list):
            periods = [periods]
        
        # 确保周期是升序排列的
        periods = sorted(periods)
        
        # 1. 识别MA交叉形态
        patterns.extend(self._detect_ma_cross_patterns(periods))
        
        # 2. 识别MA排列形态
        patterns.extend(self._detect_ma_arrangement_patterns(periods))
        
        # 3. 识别价格与MA关系形态
        patterns.extend(self._detect_price_ma_patterns(data['close'], periods))
        
        # 4. 识别MA趋势形态
        patterns.extend(self._detect_ma_trend_patterns(periods))
        
        # 5. 识别支撑阻力形态
        patterns.extend(self._detect_support_resistance_patterns(data['close'], periods))
        
        # 6. 识别自适应周期相关形态
        if self._parameters['adaptive'] and any(f'MA{p}_adaptive_period' in self._result.columns for p in periods):
            adaptive_patterns = []
            
            # 对于每个周期检测自适应相关形态
            for period in periods:
                adaptive_period_col = f'MA{period}_adaptive_period'
                if adaptive_period_col in self._result.columns:
                    adaptive_period = self._result[adaptive_period_col]
                    
                    if len(adaptive_period) < 10:
                        continue
                    
                    # 检测自适应周期显著减少（市场活跃）
                    recent_adaptive = adaptive_period.iloc[-10:]
                    base_adaptive = adaptive_period.iloc[-20:-10]
                    
                    if recent_adaptive.mean() < base_adaptive.mean() * 0.8:
                        adaptive_patterns.append(f"MA{period}周期缩短-市场活跃")
                    
                    # 检测自适应周期显著增加（市场平静）
                    elif recent_adaptive.mean() > base_adaptive.mean() * 1.2:
                        adaptive_patterns.append(f"MA{period}周期延长-市场平静")
                    
                    # 检测自适应周期快速变化（市场转折）
                    recent_change = (adaptive_period.iloc[-1] - adaptive_period.iloc[-5]) / adaptive_period.iloc[-5]
                    if abs(recent_change) > 0.3:
                        direction = "缩短" if recent_change < 0 else "延长"
                        adaptive_patterns.append(f"MA{period}周期快速{direction}-市场转折点")
            
            # 如果有过多形态，保留最重要的2个
            if len(adaptive_patterns) > 2:
                adaptive_patterns = adaptive_patterns[:2]
            
            patterns.extend(adaptive_patterns)
        
        # 7. 识别筹码分布相关形态
        if self._parameters['chip_weighted'] and 'chip_avg_cost' in self._result.columns:
            chip_patterns = []
            
            # 筹码成本形态
            if 'chip_avg_cost' in self._result.columns:
                avg_cost = self._result['chip_avg_cost']
                price = data['close']
                
                # 价格站上/跌破筹码成本线
                if len(price) >= 5:
                    cross_above = (price.iloc[-1] > avg_cost.iloc[-1]) and (price.iloc[-5] < avg_cost.iloc[-5])
                    cross_below = (price.iloc[-1] < avg_cost.iloc[-1]) and (price.iloc[-5] > avg_cost.iloc[-5])
                    
                    if cross_above:
                        chip_patterns.append("站上筹码成本线-看多信号")
                    elif cross_below:
                        chip_patterns.append("跌破筹码成本线-看空信号")
                
                # 价格远离/接近筹码成本线
                distance_ratio = abs(price.iloc[-1] - avg_cost.iloc[-1]) / avg_cost.iloc[-1]
                if distance_ratio > 0.1:
                    direction = "上方" if price.iloc[-1] > avg_cost.iloc[-1] else "下方"
                    chip_patterns.append(f"价格远离成本线{direction}-筹码极度分化")
                elif distance_ratio < 0.01:
                    chip_patterns.append("价格贴近成本线-多空临界点")
            
            # 主力成本形态
            if 'institutional_cost' in self._result.columns:
                inst_cost = self._result['institutional_cost']
                price = data['close']
                
                # 价格接近主力成本
                if abs(price.iloc[-1] - inst_cost.iloc[-1]) / inst_cost.iloc[-1] < 0.02:
                    chip_patterns.append("接近主力成本线-潜在支撑阻力位")
                
                # 价格突破主力成本
                if len(price) >= 5:
                    breakthrough = (price.iloc[-1] > inst_cost.iloc[-1] * 1.03) and (price.iloc[-5] < inst_cost.iloc[-1])
                    breakdown = (price.iloc[-1] < inst_cost.iloc[-1] * 0.97) and (price.iloc[-5] > inst_cost.iloc[-1])
                    
                    if breakthrough:
                        chip_patterns.append("突破主力成本-主力获利")
                    elif breakdown:
                        chip_patterns.append("跌破主力成本-主力套牢")
            
            # 解套盘压力形态
            if 'trapped_pressure' in self._result.columns:
                trapped_pressure = self._result['trapped_pressure']
                
                if len(trapped_pressure) >= 10:
                    # 套牢盘压力增加
                    if trapped_pressure.iloc[-1] > trapped_pressure.iloc[-10] * 1.3:
                        chip_patterns.append("套牢盘压力增加-上行阻力增强")
                    
                    # 套牢盘压力减轻
                    elif trapped_pressure.iloc[-1] < trapped_pressure.iloc[-10] * 0.7:
                        chip_patterns.append("套牢盘压力减轻-上行阻力减弱")
            
            # 筹码集中度形态
            if 'chip_concentration' in self._result.columns:
                concentration = self._result['chip_concentration']
                
                if len(concentration) >= 10:
                    # 筹码高度集中
                    if concentration.iloc[-1] > 0.8:
                        chip_patterns.append("筹码高度集中-突破阻力明确")
                    
                    # 筹码分散
                    elif concentration.iloc[-1] < 0.3:
                        chip_patterns.append("筹码高度分散-震荡行情")
                    
                    # 筹码集中度变化
                    if concentration.iloc[-1] > concentration.iloc[-10] * 1.3:
                        chip_patterns.append("筹码趋于集中-洗盘完成信号")
                    elif concentration.iloc[-1] < concentration.iloc[-10] * 0.7:
                        chip_patterns.append("筹码趋于分散-派发信号")
            
            # 如果有过多形态，保留最重要的3个
            if len(chip_patterns) > 3:
                chip_patterns = chip_patterns[:3]
            
            patterns.extend(chip_patterns)
        
        return patterns
    
    def _calculate_price_ma_score(self, close_price: pd.Series, periods: List[int]) -> pd.Series:
        """
        计算价格与MA的位置关系评分
        
        Args:
            close_price: 收盘价序列
            periods: MA周期列表
            
        Returns:
            pd.Series: 价格位置评分（-20到20分）
        """
        price_score = pd.Series(0.0, index=close_price.index)
        
        for period in periods:
            ma_col = f'MA{period}'
            if ma_col in self._result.columns:
                ma_values = self._result[ma_col]
                
                # 价格在均线上方+5分
                above_ma = close_price > ma_values
                price_score += above_ma * 5
                
                # 价格在均线下方-5分
                below_ma = close_price < ma_values
                price_score -= below_ma * 5
                
                # 价格距离均线的相对位置评分
                price_distance = (close_price - ma_values) / ma_values * 100
                
                # 距离适中（1-3%）额外加分
                moderate_distance = (abs(price_distance) >= 1) & (abs(price_distance) <= 3)
                price_score += moderate_distance * 3
        
        return price_score / len(periods)  # 平均化
    
    def _calculate_ma_cross_score(self, periods: List[int]) -> pd.Series:
        """
        计算MA交叉评分
        
        Args:
            periods: MA周期列表
            
        Returns:
            pd.Series: MA交叉评分（-20到20分）
        """
        cross_score = pd.Series(0.0, index=self._result.index)
        
        # 需要至少两个周期才能计算交叉
        if len(periods) < 2:
            return cross_score
        
        sorted_periods = sorted(periods)
        
        for i in range(len(sorted_periods) - 1):
            short_period = sorted_periods[i]
            long_period = sorted_periods[i + 1]
            
            short_ma = f'MA{short_period}'
            long_ma = f'MA{long_period}'
            
            if short_ma in self._result.columns and long_ma in self._result.columns:
                # 金叉（短期均线上穿长期均线）+20分
                golden_cross = self.crossover(self._result[short_ma], self._result[long_ma])
                
                # 死叉（短期均线下穿长期均线）-20分
                death_cross = self.crossunder(self._result[short_ma], self._result[long_ma])
                
                # 优化点：交叉角度评估
                if len(self._result) >= 5:
                    # 计算短期均线和长期均线的斜率
                    short_ma_slope = (self._result[short_ma] - self._result[short_ma].shift(5)) / self._result[short_ma].shift(5) * 100
                    long_ma_slope = (self._result[long_ma] - self._result[long_ma].shift(5)) / self._result[long_ma].shift(5) * 100
                    
                    # 计算交叉角度系数（斜率差越大，角度越大）
                    cross_angle = np.abs(short_ma_slope - long_ma_slope)
                    angle_coef = np.clip(1 + cross_angle * 0.05, 0.5, 1.5)  # 角度系数范围：0.5-1.5
                    
                    # 根据交叉角度调整得分
                    cross_score += golden_cross * 20 * np.where(golden_cross, angle_coef, 1.0)
                    cross_score -= death_cross * 20 * np.where(death_cross, angle_coef, 1.0)
                else:
                    # 无足够数据计算角度时，使用默认得分
                    cross_score += golden_cross * 20
                    cross_score -= death_cross * 20
        
        return cross_score
    
    def _calculate_ma_trend_score(self, periods: List[int]) -> pd.Series:
        """
        计算MA趋势评分
        
        Args:
            periods: MA周期列表
            
        Returns:
            pd.Series: MA趋势评分（-20到20分）
        """
        trend_score = pd.Series(0.0, index=self._result.index)
        
        for period in periods:
            ma_col = f'MA{period}'
            if ma_col in self._result.columns:
                ma_values = self._result[ma_col]
                
                # 计算均线斜率得分 - 优化点：引入斜率权重
                if len(ma_values) >= 5:
                    # 计算5周期斜率
                    ma_slope = (ma_values - ma_values.shift(5)) / ma_values.shift(5) * 100
                    
                    # 斜率得分：斜率越大，加分越多
                    slope_score = np.clip(ma_slope * 2, -10, 10)  # 最大±10分
                    trend_score += slope_score
                
                # 均线上升趋势+8分
                ma_rising = ma_values > ma_values.shift(1)
                trend_score += ma_rising * 8
                
                # 均线下降趋势-8分
                ma_falling = ma_values < ma_values.shift(1)
                trend_score -= ma_falling * 8
                
                # 均线加速上升+12分
                if len(ma_values) >= 3:
                    ma_accelerating = (ma_values.diff() > ma_values.shift(1).diff())
                    trend_score += ma_accelerating * 12
                
        return trend_score / len(periods)  # 平均化
    
    def _calculate_ma_arrangement_score(self, periods: List[int]) -> pd.Series:
        """
        计算MA排列评分
        
        Args:
            periods: MA周期列表
            
        Returns:
            pd.Series: MA排列评分（-20到20分）
        """
        arrangement_score = pd.Series(0.0, index=self._result.index)
        
        if len(periods) < 3:
            return arrangement_score
        
        sorted_periods = sorted(periods)
        
        # 检查多头排列（短期均线在上，长期均线在下）
        bullish_arrangement = pd.Series(True, index=self._result.index)
        bearish_arrangement = pd.Series(True, index=self._result.index)
        
        for i in range(len(sorted_periods) - 1):
            short_ma = f'MA{sorted_periods[i]}'
            long_ma = f'MA{sorted_periods[i + 1]}'
            
            if short_ma in self._result.columns and long_ma in self._result.columns:
                # 多头排列：短期均线 > 长期均线
                bullish_arrangement &= (self._result[short_ma] > self._result[long_ma])
                
                # 空头排列：短期均线 < 长期均线
                bearish_arrangement &= (self._result[short_ma] < self._result[long_ma])
        
        # 多头排列+25分
        arrangement_score += bullish_arrangement * 25
        
        # 空头排列-25分
        arrangement_score -= bearish_arrangement * 25
        
        return arrangement_score
    
    def _calculate_price_penetration_score(self, close_price: pd.Series, periods: List[int]) -> pd.Series:
        """
        计算价格穿透强度评分
        
        Args:
            close_price: 收盘价序列
            periods: MA周期列表
            
        Returns:
            pd.Series: 价格穿透评分（-20到20分）
        """
        penetration_score = pd.Series(0.0, index=close_price.index)
        
        for period in periods:
            ma_col = f'MA{period}'
            if ma_col in self._result.columns:
                ma_values = self._result[ma_col]
                
                # 价格上穿均线+15分
                price_cross_up = self.crossover(close_price, ma_values)
                penetration_score += price_cross_up * 15
                
                # 价格下穿均线-15分
                price_cross_down = self.crossunder(close_price, ma_values)
                penetration_score -= price_cross_down * 15
        
        return penetration_score / len(periods)  # 平均化
    
    def _detect_ma_cross_patterns(self, periods: List[int]) -> List[str]:
        """
        检测MA交叉形态
        
        Args:
            periods: MA周期列表
            
        Returns:
            List[str]: 检测到的形态列表
        """
        patterns = []
        
        if len(periods) < 2:
            return patterns
        
        sorted_periods = sorted(periods)
        
        for i in range(len(sorted_periods) - 1):
            short_period = sorted_periods[i]
            long_period = sorted_periods[i + 1]
            
            short_ma = f'MA{short_period}'
            long_ma = f'MA{long_period}'
            
            if short_ma in self._result.columns and long_ma in self._result.columns:
                # 检查最近的交叉
                recent_periods = min(5, len(self._result))
                recent_short = self._result[short_ma].tail(recent_periods)
                recent_long = self._result[long_ma].tail(recent_periods)
                
                if self.crossover(recent_short, recent_long).any():
                    patterns.append(f"MA{short_period}上穿MA{long_period}")
                
                if self.crossunder(recent_short, recent_long).any():
                    patterns.append(f"MA{short_period}下穿MA{long_period}")
        
        return patterns
    
    def _detect_ma_arrangement_patterns(self, periods: List[int]) -> List[str]:
        """
        检测MA排列形态
        
        Args:
            periods: MA周期列表
            
        Returns:
            List[str]: 检测到的形态列表
        """
        patterns = []
        
        if len(periods) < 3:
            return patterns
        
        sorted_periods = sorted(periods)
        
        # 检查当前排列状态
        if len(self._result) > 0:
            current_bullish = True
            current_bearish = True
            
            for i in range(len(sorted_periods) - 1):
                short_ma = f'MA{sorted_periods[i]}'
                long_ma = f'MA{sorted_periods[i + 1]}'
                
                if short_ma in self._result.columns and long_ma in self._result.columns:
                    current_short = self._result[short_ma].iloc[-1]
                    current_long = self._result[long_ma].iloc[-1]
                    
                    if pd.isna(current_short) or pd.isna(current_long):
                        continue
                    
                    if current_short <= current_long:
                        current_bullish = False
                    if current_short >= current_long:
                        current_bearish = False
            
            if current_bullish:
                patterns.append("均线多头排列")
            elif current_bearish:
                patterns.append("均线空头排列")
            else:
                patterns.append("均线交织状态")
        
        return patterns
    
    def _detect_price_ma_patterns(self, close_price: pd.Series, periods: List[int]) -> List[str]:
        """
        检测价格与MA关系形态
        
        Args:
            close_price: 收盘价序列
            periods: MA周期列表
            
        Returns:
            List[str]: 检测到的形态列表
        """
        patterns = []
        
        if len(close_price) == 0:
            return patterns
        
        current_price = close_price.iloc[-1]
        above_count = 0
        below_count = 0
        
        for period in periods:
            ma_col = f'MA{period}'
            if ma_col in self._result.columns:
                current_ma = self._result[ma_col].iloc[-1]
                
                if pd.isna(current_ma):
                    continue
                
                if current_price > current_ma:
                    above_count += 1
                elif current_price < current_ma:
                    below_count += 1
        
        total_ma = above_count + below_count
        if total_ma > 0:
            above_ratio = above_count / total_ma
            
            if above_ratio >= 0.8:
                patterns.append("价格强势上行")
            elif above_ratio >= 0.6:
                patterns.append("价格温和上行")
            elif above_ratio <= 0.2:
                patterns.append("价格强势下行")
            elif above_ratio <= 0.4:
                patterns.append("价格温和下行")
            else:
                patterns.append("价格均线附近震荡")
        
        # 检查价格穿越
        recent_periods = min(5, len(close_price))
        for period in periods:
            ma_col = f'MA{period}'
            if ma_col in self._result.columns:
                recent_price = close_price.tail(recent_periods)
                recent_ma = self._result[ma_col].tail(recent_periods)
                
                if self.crossover(recent_price, recent_ma).any():
                    patterns.append(f"价格上穿MA{period}")
                
                if self.crossunder(recent_price, recent_ma).any():
                    patterns.append(f"价格下穿MA{period}")
        
        return patterns
    
    def _detect_ma_trend_patterns(self, periods: List[int]) -> List[str]:
        """
        检测MA趋势形态
        
        Args:
            periods: MA周期列表
            
        Returns:
            List[str]: 检测到的形态列表
        """
        patterns = []
        
        rising_count = 0
        falling_count = 0
        
        for period in periods:
            ma_col = f'MA{period}'
            if ma_col in self._result.columns and len(self._result) >= 2:
                ma_values = self._result[ma_col]
                current_ma = ma_values.iloc[-1]
                prev_ma = ma_values.iloc[-2]
                
                if pd.isna(current_ma) or pd.isna(prev_ma):
                    continue
                
                if current_ma > prev_ma:
                    rising_count += 1
                elif current_ma < prev_ma:
                    falling_count += 1
        
        total_ma = rising_count + falling_count
        if total_ma > 0:
            rising_ratio = rising_count / total_ma
            
            if rising_ratio >= 0.8:
                patterns.append("均线全面上升")
            elif rising_ratio >= 0.6:
                patterns.append("均线多数上升")
            elif rising_ratio <= 0.2:
                patterns.append("均线全面下降")
            elif rising_ratio <= 0.4:
                patterns.append("均线多数下降")
            else:
                patterns.append("均线方向分化")
        
        return patterns
    
    def _detect_support_resistance_patterns(self, close_price: pd.Series, periods: List[int]) -> List[str]:
        """
        检测支撑阻力形态
        
        Args:
            close_price: 收盘价序列
            periods: MA周期列表
            
        Returns:
            List[str]: 检测到的形态列表
        """
        patterns = []
        
        if len(close_price) < 5:
            return patterns
        
        recent_periods = min(10, len(close_price))
        recent_price = close_price.tail(recent_periods)
        
        for period in periods:
            ma_col = f'MA{period}'
            if ma_col in self._result.columns:
                recent_ma = self._result[ma_col].tail(recent_periods)
                
                # 检查支撑：价格多次接近均线但未跌破
                support_touches = 0
                resistance_touches = 0
                
                for i in range(1, len(recent_price)):
                    price_diff = abs(recent_price.iloc[i] - recent_ma.iloc[i]) / recent_ma.iloc[i]
                    
                    if price_diff < 0.02:  # 2%以内认为是接触
                        if recent_price.iloc[i] >= recent_ma.iloc[i]:
                            if recent_price.iloc[i-1] < recent_ma.iloc[i-1]:
                                support_touches += 1
                        else:
                            if recent_price.iloc[i-1] > recent_ma.iloc[i-1]:
                                resistance_touches += 1
                
                if support_touches >= 2:
                    patterns.append(f"MA{period}形成支撑")
                
                if resistance_touches >= 2:
                    patterns.append(f"MA{period}形成阻力")
        
        return patterns
    
    def get_patterns(self, data: pd.DataFrame, **kwargs) -> List[Dict[str, Any]]:
        """
        获取MA形态列表
        
        Args:
            data: 输入K线数据
            **kwargs: 其他参数
            
        Returns:
            List[Dict[str, Any]]: 形态识别结果列表
        """
        from indicators.base_indicator import PatternResult, SignalStrength
        from indicators.pattern_registry import PatternRegistry, PatternType
        import re
        
        # 确保已计算MA
        if not self.has_result():
            self.calculate(data)
        
        patterns = []
        
        # 如果没有结果或数据不足，返回空列表
        if self._result is None or len(self._result) < 5:
            return patterns
        
        # 获取参数
        periods = self._parameters['periods']
        close_price = data['close']
        
        # 1. MA交叉形态
        cross_patterns = self._detect_ma_cross_patterns(periods)
        for pattern in cross_patterns:
            if "上穿" in pattern:
                # MA金叉
                periods_match = re.findall(r'MA(\d+)', pattern)
                if len(periods_match) >= 2:
                    short_period, long_period = int(periods_match[0]), int(periods_match[1])
                    patterns.append(PatternResult(
                        pattern_id="MA_GOLDEN_CROSS",
                        display_name=f"MA{short_period}上穿MA{long_period}",
                        strength=80,
                        duration=3,
                        details={"short_period": short_period, "long_period": long_period}
                    ).to_dict())
            elif "下穿" in pattern:
                # MA死叉
                periods_match = re.findall(r'MA(\d+)', pattern)
                if len(periods_match) >= 2:
                    short_period, long_period = int(periods_match[0]), int(periods_match[1])
                    patterns.append(PatternResult(
                        pattern_id="MA_DEATH_CROSS",
                        display_name=f"MA{short_period}下穿MA{long_period}",
                        strength=80,
                        duration=3,
                        details={"short_period": short_period, "long_period": long_period}
                    ).to_dict())
        
        # 2. MA排列形态
        arrangement_patterns = self._detect_ma_arrangement_patterns(periods)
        for pattern in arrangement_patterns:
            if "多头排列" in pattern:
                # 多头排列
                patterns.append(PatternResult(
                    pattern_id="MA_BULLISH_ALIGNMENT",
                    display_name="MA多头排列",
                    strength=85,
                    duration=5,
                    details={"periods": periods}
                ).to_dict())
            elif "空头排列" in pattern:
                # 空头排列
                patterns.append(PatternResult(
                    pattern_id="MA_BEARISH_ALIGNMENT",
                    display_name="MA空头排列",
                    strength=85,
                    duration=5,
                    details={"periods": periods}
                ).to_dict())
            elif "交织状态" in pattern:
                # MA交织
                patterns.append(PatternResult(
                    pattern_id="MA_INTERWEAVED",
                    display_name="MA交织状态",
                    strength=60,
                    duration=3,
                    details={"periods": periods}
                ).to_dict())
        
        # 3. 价格与MA关系形态
        price_ma_patterns = self._detect_price_ma_patterns(close_price, periods)
        for pattern in price_ma_patterns:
            if "价格强势突破MA" in pattern:
                patterns.append(PatternResult(
                    pattern_id="PRICE_STRONG_ABOVE_MA",
                    display_name="价格强势突破MA",
                    strength=90,
                    duration=2,
                    details={"price": close_price.iloc[-1]}
                ).to_dict())
            elif "价格温和上行MA" in pattern:
                patterns.append(PatternResult(
                    pattern_id="PRICE_ABOVE_MA",
                    display_name="价格温和上行MA",
                    strength=70,
                    duration=2,
                    details={"price": close_price.iloc[-1]}
                ).to_dict())
            elif "价格强势跌破MA" in pattern:
                patterns.append(PatternResult(
                    pattern_id="PRICE_STRONG_BELOW_MA",
                    display_name="价格强势跌破MA",
                    strength=90,
                    duration=2,
                    details={"price": close_price.iloc[-1]}
                ).to_dict())
            elif "价格温和下行MA" in pattern:
                patterns.append(PatternResult(
                    pattern_id="PRICE_BELOW_MA",
                    display_name="价格温和下行MA",
                    strength=70,
                    duration=2,
                    details={"price": close_price.iloc[-1]}
                ).to_dict())
            elif "价格MA附近震荡" in pattern:
                patterns.append(PatternResult(
                    pattern_id="PRICE_NEAR_MA",
                    display_name="价格MA附近震荡",
                    strength=60,
                    duration=2,
                    details={"price": close_price.iloc[-1]}
                ).to_dict())
            elif "价格上穿MA" in pattern:
                period_match = re.search(r'MA(\d+)', pattern)
                if period_match:
                    period = int(period_match.group(1))
                    patterns.append(PatternResult(
                        pattern_id="PRICE_CROSS_ABOVE_MA",
                        display_name=f"价格上穿MA{period}",
                        strength=85,
                        duration=1,
                        details={"period": period, "price": close_price.iloc[-1]}
                    ).to_dict())
            elif "价格下穿MA" in pattern:
                period_match = re.search(r'MA(\d+)', pattern)
                if period_match:
                    period = int(period_match.group(1))
                    patterns.append(PatternResult(
                        pattern_id="PRICE_CROSS_BELOW_MA",
                        display_name=f"价格下穿MA{period}",
                        strength=85,
                        duration=1,
                        details={"period": period, "price": close_price.iloc[-1]}
                    ).to_dict())
        
        # 4. MA趋势形态
        trend_patterns = self._detect_ma_trend_patterns(periods)
        for pattern in trend_patterns:
            if "MA强势上升" in pattern:
                patterns.append(PatternResult(
                    pattern_id="MA_STRONG_UPTREND",
                    display_name="MA强势上升",
                    strength=90,
                    duration=5,
                    details={"strength": "strong_up"}
                ).to_dict())
            elif "MA温和上升" in pattern:
                patterns.append(PatternResult(
                    pattern_id="MA_MODERATE_UPTREND",
                    display_name="MA温和上升",
                    strength=70,
                    duration=5,
                    details={"strength": "moderate_up"}
                ).to_dict())
            elif "MA强势下降" in pattern:
                patterns.append(PatternResult(
                    pattern_id="MA_STRONG_DOWNTREND",
                    display_name="MA强势下降",
                    strength=90,
                    duration=5,
                    details={"strength": "strong_down"}
                ).to_dict())
            elif "MA温和下降" in pattern:
                patterns.append(PatternResult(
                    pattern_id="MA_MODERATE_DOWNTREND",
                    display_name="MA温和下降",
                    strength=70,
                    duration=5,
                    details={"strength": "moderate_down"}
                ).to_dict())
            elif "MA盘整" in pattern:
                patterns.append(PatternResult(
                    pattern_id="MA_FLAT",
                    display_name="MA盘整",
                    strength=50,
                    duration=5,
                    details={"strength": "flat"}
                ).to_dict())
        
        # 5. MA支撑/阻力形态
        support_resistance_patterns = self._detect_support_resistance_patterns(close_price, periods)
        for pattern in support_resistance_patterns:
            if "MA形成支撑" in pattern:
                period_match = re.search(r'MA(\d+)', pattern)
                if period_match:
                    period = int(period_match.group(1))
                    patterns.append(PatternResult(
                        pattern_id="MA_SUPPORT",
                        display_name=f"MA{period}形成支撑",
                        strength=80,
                        duration=3,
                        details={"period": period, "type": "support"}
                    ).to_dict())
            elif "MA形成阻力" in pattern:
                period_match = re.search(r'MA(\d+)', pattern)
                if period_match:
                    period = int(period_match.group(1))
                    patterns.append(PatternResult(
                        pattern_id="MA_RESISTANCE",
                        display_name=f"MA{period}形成阻力",
                        strength=80,
                        duration=3,
                        details={"period": period, "type": "resistance"}
                    ).to_dict())
        
        # 6. 筹码分布相关形态（如果启用了筹码加权）
        if self._parameters['chip_weighted'] and 'chip_avg_cost' in self._result.columns:
            # 添加筹码分布形态检测逻辑
            chip_patterns = self._detect_chip_distribution_patterns(data)
            patterns.extend(chip_patterns)
        
        return patterns
    
    def _register_ma_patterns(self):
        """
        注册MA形态
        """
        from indicators.pattern_registry import PatternRegistry, PatternType
        
        # 注册MA交叉形态
        PatternRegistry.register(
            pattern_id="MA_GOLDEN_CROSS",
            display_name="MA金叉",
            description="短周期MA上穿长周期MA，看涨信号",
            indicator_types=["MA", "均线"],
            score_impact=15.0,
            pattern_type="reversal",
            signal_type="bullish"
        )
        
        PatternRegistry.register(
            pattern_id="MA_DEATH_CROSS",
            display_name="MA死叉",
            description="短周期MA下穿长周期MA，看跌信号",
            indicator_types=["MA", "均线"],
            score_impact=-15.0,
            pattern_type="reversal",
            signal_type="bearish"
        )
        
        # 注册MA排列形态
        PatternRegistry.register(
            pattern_id="MA_BULLISH_ALIGNMENT",
            display_name="MA多头排列",
            description="短周期MA位于长周期MA上方，呈阶梯状排列，强势上涨信号",
            indicator_types=["MA", "均线"],
            score_impact=20.0,
            pattern_type="trend",
            signal_type="bullish"
        )
        
        PatternRegistry.register(
            pattern_id="MA_BEARISH_ALIGNMENT",
            display_name="MA空头排列",
            description="短周期MA位于长周期MA下方，呈阶梯状排列，强势下跌信号",
            indicator_types=["MA", "均线"],
            score_impact=-20.0,
            pattern_type="trend",
            signal_type="bearish"
        )
        
        PatternRegistry.register(
            pattern_id="MA_INTERWEAVED",
            display_name="MA交织状态",
            description="各周期MA交织在一起，表示市场处于震荡整理中",
            indicator_types=["MA", "均线"],
            score_impact=0.0,
            pattern_type="consolidation",
            signal_type="neutral"
        )
        
        # 注册价格与MA关系形态
        PatternRegistry.register(
            pattern_id="PRICE_STRONG_ABOVE_MA",
            display_name="价格强势突破MA",
            description="价格远高于多数MA，表示强势上涨",
            indicator_types=["MA", "均线"],
            score_impact=18.0,
            pattern_type="momentum",
            signal_type="bullish"
        )
        
        PatternRegistry.register(
            pattern_id="PRICE_ABOVE_MA",
            display_name="价格温和上行MA",
            description="价格位于多数MA上方但距离不远",
            indicator_types=["MA", "均线"],
            score_impact=12.0,
            pattern_type="trend",
            signal_type="bullish"
        )
        
        PatternRegistry.register(
            pattern_id="PRICE_STRONG_BELOW_MA",
            display_name="价格强势跌破MA",
            description="价格远低于多数MA，表示强势下跌",
            indicator_types=["MA", "均线"],
            score_impact=-18.0,
            pattern_type="momentum",
            signal_type="bearish"
        )
        
        PatternRegistry.register(
            pattern_id="PRICE_BELOW_MA",
            display_name="价格温和下行MA",
            description="价格位于多数MA下方但距离不远",
            indicator_types=["MA", "均线"],
            score_impact=-12.0,
            pattern_type="trend",
            signal_type="bearish"
        )
        
        PatternRegistry.register(
            pattern_id="PRICE_NEAR_MA",
            display_name="价格MA附近震荡",
            description="价格在MA附近波动，表示市场处于震荡状态",
            indicator_types=["MA", "均线"],
            score_impact=0.0,
            pattern_type="consolidation",
            signal_type="neutral"
        )
        
        PatternRegistry.register(
            pattern_id="PRICE_CROSS_ABOVE_MA",
            display_name="价格上穿MA",
            description="价格上穿某一周期的MA，可能是买入信号",
            indicator_types=["MA", "均线"],
            score_impact=15.0,
            pattern_type="reversal",
            signal_type="bullish"
        )
        
        PatternRegistry.register(
            pattern_id="PRICE_CROSS_BELOW_MA",
            display_name="价格下穿MA",
            description="价格下穿某一周期的MA，可能是卖出信号",
            indicator_types=["MA", "均线"],
            score_impact=-15.0,
            pattern_type="reversal",
            signal_type="bearish"
        )
        
        # 注册MA趋势形态
        PatternRegistry.register(
            pattern_id="MA_STRONG_UPTREND",
            display_name="MA强势上升",
            description="所有周期的MA都快速上升，表示强势上涨趋势",
            indicator_types=["MA", "均线"],
            score_impact=20.0,
            pattern_type="trend",
            signal_type="bullish"
        )
        
        PatternRegistry.register(
            pattern_id="MA_MODERATE_UPTREND",
            display_name="MA温和上升",
            description="大部分周期的MA平缓上升",
            indicator_types=["MA", "均线"],
            score_impact=10.0,
            pattern_type="trend",
            signal_type="bullish"
        )
        
        PatternRegistry.register(
            pattern_id="MA_STRONG_DOWNTREND",
            display_name="MA强势下降",
            description="所有周期的MA都快速下降，表示强势下跌趋势",
            indicator_types=["MA", "均线"],
            score_impact=-20.0,
            pattern_type="trend",
            signal_type="bearish"
        )
        
        PatternRegistry.register(
            pattern_id="MA_MODERATE_DOWNTREND",
            display_name="MA温和下降",
            description="大部分周期的MA平缓下降",
            indicator_types=["MA", "均线"],
            score_impact=-10.0,
            pattern_type="trend",
            signal_type="bearish"
        )
        
        PatternRegistry.register(
            pattern_id="MA_FLAT",
            display_name="MA盘整",
            description="多数MA水平移动，表示市场处于盘整状态",
            indicator_types=["MA", "均线"],
            score_impact=0.0,
            pattern_type="consolidation",
            signal_type="neutral"
        )
        
        # 注册MA支撑/阻力形态
        PatternRegistry.register(
            pattern_id="MA_SUPPORT",
            display_name="MA支撑",
            description="MA作为价格的支撑位，价格触及后反弹",
            indicator_types=["MA", "均线"],
            score_impact=15.0,
            pattern_type="support",
            signal_type="bullish"
        )
        
        PatternRegistry.register(
            pattern_id="MA_RESISTANCE",
            display_name="MA阻力",
            description="MA作为价格的阻力位，价格触及后回落",
            indicator_types=["MA", "均线"],
            score_impact=-15.0,
            pattern_type="resistance",
            signal_type="bearish"
        )
        
        # 注册筹码分布相关形态
        PatternRegistry.register(
            pattern_id="CHIP_COST_SUPPORT",
            display_name="筹码成本支撑",
            description="价格接近筹码平均成本，形成支撑",
            indicator_types=["MA", "筹码"],
            score_impact=18.0,
            pattern_type="support",
            signal_type="bullish"
        )
        
        PatternRegistry.register(
            pattern_id="CHIP_COST_RESISTANCE",
            display_name="筹码成本阻力",
            description="价格接近筹码平均成本，形成阻力",
            indicator_types=["MA", "筹码"],
            score_impact=-18.0,
            pattern_type="resistance",
            signal_type="bearish"
        )
        
        PatternRegistry.register(
            pattern_id="CHIP_CONCENTRATION_HIGH",
            display_name="筹码高度集中",
            description="筹码分布高度集中，表明持仓者观点一致",
            indicator_types=["MA", "筹码"],
            score_impact=10.0,
            pattern_type="volatility",
            signal_type="neutral"
        )
        
        PatternRegistry.register(
            pattern_id="CHIP_CONCENTRATION_LOW",
            display_name="筹码分散",
            description="筹码分布分散，表明持仓者观点分歧",
            indicator_types=["MA", "筹码"],
            score_impact=-5.0,
            pattern_type="volatility",
            signal_type="neutral"
        )

    def _detect_chip_distribution_patterns(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        检测筹码分布相关形态
        
        Args:
            data: 输入K线数据
            
        Returns:
            List[Dict[str, Any]]: 形态识别结果列表
        """
        from indicators.base_indicator import PatternResult
        
        patterns = []
        
        if not self._parameters['chip_weighted'] or 'chip_avg_cost' not in self._result.columns:
            return patterns
        
        close_price = data['close'].iloc[-1]
        avg_cost = self._result['chip_avg_cost'].iloc[-1]
        
        # 价格与平均成本的关系
        cost_price_ratio = abs(close_price - avg_cost) / avg_cost
        
        # 筹码集中度
        if 'chip_concentration' in self._result.columns:
            concentration = self._result['chip_concentration'].iloc[-1]
            
            # 检测筹码集中度
            if concentration > 0.7:  # 高度集中
                patterns.append(PatternResult(
                    pattern_id="CHIP_CONCENTRATION_HIGH",
                    display_name="筹码高度集中",
                    strength=80,
                    duration=3,
                    details={"concentration": concentration}
                ).to_dict())
            elif concentration < 0.3:  # 分散
                patterns.append(PatternResult(
                    pattern_id="CHIP_CONCENTRATION_LOW",
                    display_name="筹码分散",
                    strength=60,
                    duration=3,
                    details={"concentration": concentration}
                ).to_dict())
        
        # 检测筹码成本支撑/阻力
        if cost_price_ratio < 0.02:  # 价格接近平均成本
            if close_price > avg_cost:
                # 平均成本在下方，可能形成支撑
                patterns.append(PatternResult(
                    pattern_id="CHIP_COST_SUPPORT",
                    display_name="筹码成本支撑",
                    strength=85,
                    duration=3,
                    details={"avg_cost": avg_cost, "price": close_price}
                ).to_dict())
            else:
                # 平均成本在上方，可能形成阻力
                patterns.append(PatternResult(
                    pattern_id="CHIP_COST_RESISTANCE",
                    display_name="筹码成本阻力",
                    strength=85,
                    duration=3,
                    details={"avg_cost": avg_cost, "price": close_price}
                ).to_dict())
        
        return patterns 
    
    def generate_trading_signals(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """
        生成交易信号
        
        Args:
            data: 输入数据
            **kwargs: 额外参数
            
        Returns:
            Dict[str, pd.Series]: 包含交易信号的字典
        """
        # 确保已计算MA
        if not self.has_result():
            self.calculate(data)
        
        # 获取参数
        periods = kwargs.get('periods', self._parameters['periods'])
        
        # 如果periods是单个值，转换为列表
        if not isinstance(periods, list):
            periods = [periods]
        
        # 确保至少有两个周期用于交叉信号生成
        if len(periods) < 2:
            # 如果只有一个周期，添加一个较长的周期
            short_period = periods[0]
            long_period = short_period * 2
            periods = [short_period, long_period]
            
        # 排序周期，确保短周期在前
        periods = sorted(periods)
            
        # 初始化信号
        signals = {}
        signals['buy_signal'] = pd.Series(False, index=data.index)
        signals['sell_signal'] = pd.Series(False, index=data.index)
        signals['signal_strength'] = pd.Series(0, index=data.index)
        
        # 获取短周期和长周期的MA
        short_period = periods[0]
        long_period = periods[1]
        
        if f'MA{short_period}' in self._result.columns and f'MA{long_period}' in self._result.columns:
            short_ma = self._result[f'MA{short_period}']
            long_ma = self._result[f'MA{long_period}']
            
            # 生成金叉信号（买入）
            golden_cross = self.crossover(short_ma, long_ma)
            signals['buy_signal'] = golden_cross
            
            # 生成死叉信号（卖出）
            death_cross = self.crossunder(short_ma, long_ma)
            signals['sell_signal'] = death_cross
            
            # 设置信号强度
            signals['signal_strength'] = pd.Series(0, index=data.index)
            signals['signal_strength'].loc[golden_cross] = 1
            signals['signal_strength'].loc[death_cross] = -1
            
            # 处理价格与MA的关系
            close = data['close']
            
            # 价格站上所有MA线（额外的买入信号）
            above_all_ma = True
            for period in periods:
                if f'MA{period}' in self._result.columns:
                    above_all_ma = above_all_ma & (close > self._result[f'MA{period}'])
            
            # 价格跌破所有MA线（额外的卖出信号）
            below_all_ma = True
            for period in periods:
                if f'MA{period}' in self._result.columns:
                    below_all_ma = below_all_ma & (close < self._result[f'MA{period}'])
            
            # 增强信号强度
            signals['signal_strength'].loc[above_all_ma] = 2
            signals['signal_strength'].loc[below_all_ma] = -2
        
        return signals