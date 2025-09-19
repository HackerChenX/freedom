from utils.container import container
import pandas as pd
import numpy as np
from typing import Dict, Any, List

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from indicators.base.minimum_periods_mixin import MinimumPeriodsMixin
from utils.logger import get_logger

logger = get_logger(__name__)


class Sar(BaseIndicator, PatternSignalMixin, MinimumPeriodsMixin):
    """
    SAR 指标

    生产级真实指标实现
    """

    def __init__(self, **kwargs):
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        """
        初始化SAR指标

        Args:
            **kwargs: 指标参数
        """
        super().__init__()
        self.name = "SAR"

        # 设置默认参数
        self._default_parameters = self._get_default_parameters()
        self._minimum_periods = 14  # TODO: 将魔法数字提取到配置中

        # 应用用户参数
        self.set_parameters(**kwargs)

    def _get_default_parameters(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {"period": 14}  # TODO: 将魔法数字提取到配置中

    def set_parameters(self, **kwargs):
        """设置指标参数"""
        self.period = kwargs.get("period", 14)  # TODO: 将魔法数字提取到配置中
        self._minimum_periods = self.period

    def _calculate_baseindicator(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """计算SAR指标 - 完整的抛物线转向算法实现"""
        df = data.copy()
        
        if len(df) < 2:
            return df
        
        # SAR算法参数
        initial_af = kwargs.get('initial_af', 0.02)  # 初始加速因子
        max_af = kwargs.get('max_af', 0.2)           # 最大加速因子
        af_step = kwargs.get('af_step', 0.02)        # 加速因子步长
        
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        # 初始化SAR数组
        sar = np.zeros(len(df))
        af = np.zeros(len(df))
        trend = np.zeros(len(df))  # 1为上升趋势，-1为下降趋势
        ep = np.zeros(len(df))     # 极值点（Extreme Point）
        
        # 第一个点的初始化
        sar[0] = low[0]
        trend[0] = 1
        af[0] = initial_af
        ep[0] = high[0]
        
        for i in range(1, len(df)):
            prev_sar = sar[i-1]
            prev_trend = trend[i-1]
            prev_af = af[i-1]
            prev_ep = ep[i-1]
            
            # 计算当前SAR
            if prev_trend == 1:  # 上升趋势
                sar[i] = prev_sar + prev_af * (prev_ep - prev_sar)
                
                # 确保SAR不超过前两个周期的最低价
                if i >= 2:
                    sar[i] = min(sar[i], low[i-1], low[i-2])
                else:
                    sar[i] = min(sar[i], low[i-1])
                
                # 检查趋势反转
                if low[i] <= sar[i]:
                    # 趋势反转为下降
                    trend[i] = -1
                    sar[i] = prev_ep  # SAR等于前期极值点
                    af[i] = initial_af
                    ep[i] = low[i]
                else:
                    # 继续上升趋势
                    trend[i] = 1
                    ep[i] = max(prev_ep, high[i])
                    
                    # 更新加速因子
                    if ep[i] > prev_ep:
                        af[i] = min(prev_af + af_step, max_af)
                    else:
                        af[i] = prev_af
                        
            else:  # 下降趋势
                sar[i] = prev_sar + prev_af * (prev_ep - prev_sar)
                
                # 确保SAR不低于前两个周期的最高价
                if i >= 2:
                    sar[i] = max(sar[i], high[i-1], high[i-2])
                else:
                    sar[i] = max(sar[i], high[i-1])
                
                # 检查趋势反转
                if high[i] >= sar[i]:
                    # 趋势反转为上升
                    trend[i] = 1
                    sar[i] = prev_ep  # SAR等于前期极值点
                    af[i] = initial_af
                    ep[i] = high[i]
                else:
                    # 继续下降趋势
                    trend[i] = -1
                    ep[i] = min(prev_ep, low[i])
                    
                    # 更新加速因子
                    if ep[i] < prev_ep:
                        af[i] = min(prev_af + af_step, max_af)
                    else:
                        af[i] = prev_af
        
        # 添加计算结果到DataFrame
        df['sar'] = sar
        df['sar_af'] = af
        df['sar_trend'] = trend
        df['sar_ep'] = ep
        
        # 添加信号列
        df['sar_buy_signal'] = (trend == 1) & (np.roll(trend, 1) == -1)
        df['sar_sell_signal'] = (trend == -1) & (np.roll(trend, 1) == 1)
        
        # 添加形态识别和通用信号生成
        df = self.add_pattern_detection(df)
        df = self.add_signal_generation(df)
        
        return df

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算SAR指标 - 标准接口

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 包含SAR指标的DataFrame
        """
        result = self._calculate_baseindicator(data, **kwargs)
        self._result = result  # 保存结果以供get_signal使用
        return result

    def calculate_confidence_Indicator_Base_Indicator(
        self, score: pd.Series, patterns: List[str], signals: Dict[str, pd.Series]
    ) -> float:
        """计算置信度"""
        return 0.8  # TODO: 将魔法数字提取到配置中

    def calculate_raw_score_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算原始评分"""
        return pd.Series(50.0, index=data.index)  # TODO: 将魔法数字提取到配置中

    def get_patterns_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """获取形态"""
        return pd.DataFrame(index=data.index)

    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        """设置参数"""
        return self.set_parameters(**kwargs)

    @property
    def minimum_periods(self) -> int:
        """最小周期数"""
        return self._minimum_periods

    def get_signal(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        【核心抽象方法2】基于SAR指标数值生成最新的交易信号
        
        SAR交易信号逻辑：
        - 价格上穿SAR：买入信号（趋势反转向上）
        - 价格下穿SAR：卖出信号（趋势反转向下）  
        - 价格持续在SAR上方：持续买入信号
        - 价格持续在SAR下方：持续卖出信号
        
        Args:
            data: 包含OHLCV数据的DataFrame
            **kwargs: 额外参数
            
        Returns:
            Dict[str, Any]: 标准化交易信号格式
        """
        try:
            # 1. 数据验证
            if not self._validate_signal_data(data):
                return self._get_default_signal("数据验证失败")
            
            # 2. 确保已计算指标
            if not self.has_result():
                self.calculate(data, **kwargs)

            if self._result is None or len(self._result) == 0:
                return self._get_default_signal("SAR计算结果为空")

            # 3. 获取最新数据
            latest_close = data['close'].iloc[-1]
            
            # 4. 获取SAR相关值
            if len(self._result) < 2:
                return self._get_default_signal("SAR数据不足")
                
            # 检查必要的列是否存在
            required_columns = ['sar', 'sar_trend']
            if not all(col in self._result.columns for col in required_columns):
                return self._get_default_signal("SAR结果列不完整")
                
            latest_sar = self._result['sar'].iloc[-1]
            latest_trend = self._result['sar_trend'].iloc[-1]
            prev_trend = self._result['sar_trend'].iloc[-2]
            
            # 5. SAR信号生成逻辑
            signal_type = "hold"
            strength = 0.0
            confidence = 0.5
            reason = "无明确信号"
            metadata = {}
            
            # 趋势反转信号（最强信号）
            if latest_trend == 1 and prev_trend == -1:
                # 从下降趋势转为上升趋势 - 强烈买入信号
                signal_type = "buy"
                strength = 0.9
                confidence = 0.9
                reason = "SAR趋势反转向上，强烈买入信号"
                
            elif latest_trend == -1 and prev_trend == 1:
                # 从上升趋势转为下降趋势 - 强烈卖出信号
                signal_type = "sell"
                strength = 0.9
                confidence = 0.9
                reason = "SAR趋势反转向下，强烈卖出信号"
                
            # 趋势持续信号
            elif latest_trend == 1:
                # 持续上升趋势
                signal_type = "buy"
                strength = 0.6
                confidence = 0.7
                reason = "SAR显示持续上升趋势，买入信号"
                
                # 基于价格与SAR的距离调整强度
                if latest_close > latest_sar:
                    price_distance = (latest_close - latest_sar) / latest_sar
                    if price_distance > 0.05:  # 价格明显高于SAR
                        strength = min(0.8, strength + price_distance * 2)
                        confidence = min(0.85, confidence + 0.1)
                        reason = f"价格显著高于SAR({price_distance*100:.1f}%)，持续买入信号"
                        
            elif latest_trend == -1:
                # 持续下降趋势
                signal_type = "sell"
                strength = 0.6
                confidence = 0.7
                reason = "SAR显示持续下降趋势，卖出信号"
                
                # 基于价格与SAR的距离调整强度
                if latest_close < latest_sar:
                    price_distance = (latest_sar - latest_close) / latest_sar
                    if price_distance > 0.05:  # 价格明显低于SAR
                        strength = min(0.8, strength + price_distance * 2)
                        confidence = min(0.85, confidence + 0.1)
                        reason = f"价格显著低于SAR({price_distance*100:.1f}%)，持续卖出信号"
            
            # 计算额外的SAR指标
            price_sar_ratio = latest_close / latest_sar if latest_sar != 0 else 1.0
            sar_af = self._result['sar_af'].iloc[-1] if 'sar_af' in self._result.columns else 0.02
            
            # 设置元数据
            metadata = {
                'sar_value': latest_sar,
                'sar_trend': 'uptrend' if latest_trend == 1 else 'downtrend',
                'sar_af': sar_af,
                'price_sar_ratio': price_sar_ratio,
                'trend_change': latest_trend != prev_trend,
                'trend_strength': 'strong' if abs(latest_trend) == 1 else 'weak'
            }
            
            # 添加极值点信息（如果存在）
            if 'sar_ep' in self._result.columns:
                metadata['sar_ep'] = self._result['sar_ep'].iloc[-1]
            
            # 添加信号确认（如果存在）
            if 'sar_buy_signal' in self._result.columns:
                metadata['sar_buy_signal'] = bool(self._result['sar_buy_signal'].iloc[-1])
            if 'sar_sell_signal' in self._result.columns:
                metadata['sar_sell_signal'] = bool(self._result['sar_sell_signal'].iloc[-1])
            
            # 6. 标准化输出
            return {
                'signal_type': signal_type,
                'strength': max(0.0, min(1.0, strength)),
                'confidence': max(0.0, min(1.0, confidence)),
                'timestamp': pd.Timestamp.now(),
                'reason': reason,
                'metadata': {
                    'latest_close': latest_close,
                    **metadata
                }
            }

        except Exception as e:
            logger.warning(f"SAR信号生成失败: {e}")
            return self._get_default_signal(f"信号生成失败: {str(e)}")

    def _validate_signal_data(self, data: pd.DataFrame) -> bool:
        """
        验证信号生成所需的数据
        
        Args:
            data: 输入数据DataFrame
            
        Returns:
            bool: 数据是否有效
        """
        if data is None or data.empty:
            return False
            
        required_columns = ['high', 'low', 'close']
        if not all(col in data.columns for col in required_columns):
            return False
            
        # SAR需要足够的数据
        min_periods = getattr(self, 'period', 14)
        if len(data) < min_periods:
            return False
            
        return True

    def _get_default_signal(self, reason: str = "数据不足") -> Dict[str, Any]:
        """
        生成默认信号（持有信号）
        
        Args:
            reason: 生成默认信号的原因
            
        Returns:
            Dict[str, Any]: 默认信号
        """
        return {
            'signal_type': 'hold',
            'strength': 0.0,
            'confidence': 0.0,
            'timestamp': pd.Timestamp.now(),
            'reason': reason,
            'metadata': {}
        }

    def has_result(self) -> bool:
        """
        检查是否已有计算结果
        
        Returns:
            bool: 是否已有计算结果
        """
        return (self._result is not None and 
                hasattr(self._result, 'empty') and 
                not self._result.empty)
