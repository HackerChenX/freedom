"""
ZXM核心吸筹公式模块

实现ZXM体系的核心吸筹信号识别算法
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple

from enums.indicator_enum import IndicatorEnum
from indicators.base_indicator import BaseIndicator
from utils.logger import get_logger

logger = get_logger(__name__)


class ZXMAbsorb(BaseIndicator):
    """
    ZXM核心吸筹公式指标
    
    基于KDJ衍生指标的低位区域信号，识别主力在低位吸筹动作
    """
    
    def __init__(self):
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        """初始化ZXM核心吸筹指标"""
        super().__init__(name="ZXM_ABSORB", description="ZXM核心吸筹指标，用于识别主力低位吸筹动作")
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算ZXM核心吸筹指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
                
        Returns:
            包含ZXM核心吸筹指标的DataFrame
        """
        result = self.calculate(df)
        result['absorb_signal'] = result['BUY']
        result['absorb_strength'] = result['XG']
        
        # 修改调试日志，避免百分号
        if len(result) > 0:
            logger.info(f"absorb_strength值: {result['absorb_strength'].iloc[-1]}, 类型: {type(result['absorb_strength'].iloc[-1])}")
            
        return result
    
    def _calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算ZXM核心吸筹指标

        Args:
            data: 输入数据，包含OHLCV数据

        Returns:
            pd.DataFrame: 计算结果，包含核心吸筹信号

        公式说明（基于ZXM体系3.0版权威文档）：
        V11:=3*SMA((CLOSE-LLV(LOW,55))/(HHV(HIGH,55)-LLV(LOW,55))*100,5,1)-2*SMA(SMA((CLOSE-LLV(LOW,55))/(HHV(HIGH,55)-LLV(LOW,55))*100,5,1),3,1);
        V12:=(EMA(V11,3)-REF(EMA(V11,3),1))/REF(EMA(V11,3),1)*100;
        AA:=EMA(V11,3)<=13;
        BB:=EMA(V11,3)<=13 AND V12>13;
        XG:=COUNT(AA OR BB,6);
        BUY:=XG>=3;
        """
        # 确保数据包含必需的列
        required_columns = ["close", "high", "low"]
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"数据必须包含'{col}'列")
        
        # 提取数据
        close = data["close"].values
        high = data["high"].values
        low = data["low"].values
        
        # 初始化结果数据框
        result = data.copy()
        
        # 计算V11 - KDJ衍生指标
        llv_55 = self._llv(low, 55)
        hhv_55 = self._hhv(high, 55)
        
        # 计算RSV变种
        rsv_55 = np.zeros_like(close)
        for i in range(len(close)):
            if hhv_55[i] - llv_55[i] != 0:
                rsv_55[i] = (close[i] - llv_55[i]) / (hhv_55[i] - llv_55[i]) * 100
            else:
                rsv_55[i] = 50  # 防止除以零
        
        # 计算SMA
        sma_rsv_5 = self._sma(rsv_55, 5, 1)
        sma_sma_3 = self._sma(sma_rsv_5, 3, 1)
        
        # 计算V11
        v11 = 3 * sma_rsv_5 - 2 * sma_sma_3
        
        # 计算V11的EMA
        ema_v11_3 = self._ema(v11, 3)
        
        # 计算V12 - 动量指标
        v12 = np.zeros_like(close)
        for i in range(1, len(ema_v11_3)):
            if ema_v11_3[i-1] != 0:
                v12[i] = (ema_v11_3[i] - ema_v11_3[i-1]) / ema_v11_3[i-1] * 100
        
        # 定义AA和BB条件（严格按照文档公式）
        aa = ema_v11_3 <= 13
        bb = (ema_v11_3 <= 13) & (v12 > 13)

        # 计算满足条件的次数
        xg = np.zeros_like(close, dtype=int)
        buy = np.zeros_like(close, dtype=bool)

        # 按照文档公式计算XG和BUY
        for i in range(6, len(close)):
            # 计算XG：近6天内AA或BB条件满足的次数
            xg[i] = np.sum(aa[i-5:i+1] | bb[i-5:i+1])

            # 计算BUY：XG >= 3
            buy[i] = xg[i] >= 3
        
        # 添加计算结果到数据框
        result["V11"] = v11
        result["V12"] = v12
        result["EMA_V11_3"] = ema_v11_3
        result["AA"] = aa
        result["BB"] = bb
        result["XG"] = xg
        result["BUY"] = buy
        
        return result
    
    def _sma(self, series: np.ndarray, n: int, m: int) -> np.ndarray:
        """
        计算简单移动平均
        
        Args:
            series: 输入序列
            n: 周期
            m: 权重
            
        Returns:
            np.ndarray: SMA结果
        """
        result = np.zeros_like(series)
        result[0] = series[0]
        
        for i in range(1, len(series)):
            result[i] = (m * series[i] + (n - m) * result[i-1]) / n
        
        return result
    
    def _ema(self, series: np.ndarray, n: int) -> np.ndarray:
        """
        计算指数移动平均
        
        Args:
            series: 输入序列
            n: 周期
            
        Returns:
            np.ndarray: EMA结果
        """
        alpha = 2 / (n + 1)
        result = np.zeros_like(series)
        result[0] = series[0]
        
        for i in range(1, len(series)):
            result[i] = alpha * series[i] + (1 - alpha) * result[i-1]
        
        return result
    
    def _llv(self, series: np.ndarray, n: int) -> np.ndarray:
        """
        计算n周期内最低值
        
        Args:
            series: 输入序列
            n: 周期
            
        Returns:
            np.ndarray: 最低值序列
        """
        result = np.zeros_like(series)
        
        for i in range(len(series)):
            if i < n:
                result[i] = np.min(series[:i+1])
            else:
                result[i] = np.min(series[i-n+1:i+1])
        
        return result
    
    def _hhv(self, series: np.ndarray, n: int) -> np.ndarray:
        """
        计算n周期内最高值
        
        Args:
            series: 输入序列
            n: 周期
            
        Returns:
            np.ndarray: 最高值序列
        """
        result = np.zeros_like(series)
        
        for i in range(len(series)):
            if i < n:
                result[i] = np.max(series[:i+1])
            else:
                result[i] = np.max(series[i-n+1:i+1])
        
        return result
    
    def generate_signals(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        根据ZXM吸筹指标生成标准化交易信号
        
        Args:
            data: 输入数据，包含OHLCV数据
            *args, **kwargs: 附加参数
            
        Returns:
            pd.DataFrame: 包含标准化信号的DataFrame
        """
        # 计算指标
        indicator_data = self.calculate(data)
        
        # 初始化信号DataFrame
        signals = pd.DataFrame(index=data.index)
        signals['buy_signal'] = False
        signals['sell_signal'] = False
        signals['neutral_signal'] = True
        signals['trend'] = 0
        signals['score'] = 50
        signals['signal_type'] = ''
        signals['signal_desc'] = ''
        signals['confidence'] = 0
        signals['risk_level'] = '中'
        signals['position_size'] = 0.0
        signals['stop_loss'] = 0.0
        signals['market_env'] = '未知'
        signals['volume_confirmation'] = False
        
        # 设置基本信号
        signals.loc[indicator_data['BUY'], 'buy_signal'] = True
        signals.loc[indicator_data['BUY'], 'neutral_signal'] = False
        signals.loc[indicator_data['BUY'], 'trend'] = 1
        
        # 设置信号类型和描述
        signals.loc[indicator_data['BUY'], 'signal_type'] = 'ZXM吸筹信号'
        
        # 根据吸筹强度(XG)设置评分和置信度
        for i in range(len(data)):
            if indicator_data['BUY'].iloc[i]:
                # 吸筹强度为3-6之间，分数为60-80
                xg_value = indicator_data['XG'].iloc[i]
                
                # 设置评分 (XG=3 -> 60分, XG=4 -> 67分, XG=5 -> 73分, XG=6 -> 80分)
                if xg_value >= 3:
                    signals.loc[data.index[i], 'score'] = 60 + min((xg_value - 3) * 7, 20)
                    signals.loc[data.index[i], 'confidence'] = 70 + min((xg_value - 3) * 5, 20)
                    
                    # 描述信号
                    signals.loc[data.index[i], 'signal_desc'] = f"ZXM吸筹信号(强度:{xg_value})"
                    
                    # 设置仓位
                    if xg_value >= 5:  # 强吸筹信号
                        signals.loc[data.index[i], 'position_size'] = 0.3
                        signals.loc[data.index[i], 'risk_level'] = '低'
                    elif xg_value >= 4:  # 中等吸筹信号
                        signals.loc[data.index[i], 'position_size'] = 0.2
                        signals.loc[data.index[i], 'risk_level'] = '中'
                    else:  # 弱吸筹信号
                        signals.loc[data.index[i], 'position_size'] = 0.1
                        signals.loc[data.index[i], 'risk_level'] = '中'
                    
                    # 设置止损位 (当前价格的98%)
                    signals.loc[data.index[i], 'stop_loss'] = data['close'].iloc[i] * 0.98
        
        # 检查是否有成交量数据，用于确认信号
        if 'volume' in data.columns:
            # 计算5日平均成交量
            volume_ma5 = data['volume'].rolling(window=5).mean()
            
            for i in range(5, len(data)):
                # 成交量较前5日平均成交量变化率
                if pd.notna(volume_ma5.iloc[i]) and volume_ma5.iloc[i] > 0:
                    volume_change = data['volume'].iloc[i] / volume_ma5.iloc[i]
                    
                    # 成交量确认信号：成交量萎缩(吸筹过程中常见成交量特征)
                    if indicator_data['BUY'].iloc[i] and volume_change < 0.8:  # 成交量较前5日平均萎缩20%
                        signals.loc[data.index[i], 'volume_confirmation'] = True
                        # 成交量确认增加信号置信度
                        signals.loc[data.index[i], 'confidence'] = min(100, signals.loc[data.index[i], 'confidence'] + 10)
                    # 或者在低位放量，也是吸筹信号
                    elif indicator_data['BUY'].iloc[i] and volume_change > 1.5 and indicator_data['EMA_V11_3'].iloc[i] < 10:
                        signals.loc[data.index[i], 'volume_confirmation'] = True
                        signals.loc[data.index[i], 'confidence'] = min(100, signals.loc[data.index[i], 'confidence'] + 15)
                        signals.loc[data.index[i], 'signal_desc'] += ", 低位放量"
        
        # 分析市场环境
        if len(data) >= 20:
            # 简单的市场趋势判断：20日价格变化率
            for i in range(20, len(data)):
                recent_trend = (data['close'].iloc[i] - data['close'].iloc[i-20]) / data['close'].iloc[i-20]
                
                if recent_trend > 0.05:
                    signals.loc[data.index[i], 'market_env'] = '上升趋势'
                elif recent_trend < -0.05:
                    signals.loc[data.index[i], 'market_env'] = '下降趋势'
                else:
                    signals.loc[data.index[i], 'market_env'] = '横盘整理'
        
        return signals
        
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算ZXM吸筹指标的原始评分
        
        Args:
            data: 输入数据，包含OHLCV数据
            **kwargs: 附加参数
            
        Returns:
            pd.Series: 原始评分序列
        """
        # 计算指标
        indicator_data = self.calculate(data)
        
        # 初始化评分序列，默认分数50
        scores = pd.Series(50.0, index=data.index, dtype=float)
        
        # 根据吸筹信号和强度计算分数
        for i in range(len(data)):
            # 如果出现吸筹信号
            if indicator_data['BUY'].iloc[i]:
                # 获取吸筹强度
                xg_value = indicator_data['XG'].iloc[i]
                
                # 根据吸筹强度计算分数 (3-6的XG值对应60-80分)
                scores.iloc[i] = 60 + min((xg_value - 3) * 7, 20)
            
            # 如果V11指标处于低位但未达到信号阈值
            elif indicator_data['EMA_V11_3'].iloc[i] <= 20:
                # V11在13-20之间，小幅加分
                v11_value = indicator_data['EMA_V11_3'].iloc[i]
                v11_bonus = (20 - v11_value) / 7 * 10  # 最多加10分
                scores.iloc[i] = 50 + v11_bonus
            
            # V11指标处于高位，小幅减分
            elif indicator_data['EMA_V11_3'].iloc[i] >= 80:
                v11_value = indicator_data['EMA_V11_3'].iloc[i]
                v11_penalty = (v11_value - 80) / 20 * 10  # 最多减10分
                scores.iloc[i] = 50 - v11_penalty
        
        return scores
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别ZXM吸筹指标相关的技术形态
        
        Args:
            data: 输入数据，包含OHLCV数据
            **kwargs: 附加参数
            
        Returns:
            List[str]: 识别出的形态列表
        """
        # 计算指标
        indicator_data = self.calculate(data)
        
        patterns = []
        
        # 获取最新的数据点
        if len(indicator_data) == 0:
            return patterns
        
        latest_index = -1
        
        # 吸筹信号
        if indicator_data['BUY'].iloc[latest_index]:
            xg_value = indicator_data['XG'].iloc[latest_index]
            patterns.append(f"ZXM吸筹信号(强度:{xg_value})")
            
            # 区分不同强度的吸筹信号
            if xg_value >= 5:
                patterns.append("强吸筹信号")
            elif xg_value >= 4:
                patterns.append("中等吸筹信号")
            else:
                patterns.append("弱吸筹信号")
        
        # V11指标处于低位
        if indicator_data['EMA_V11_3'].iloc[latest_index] <= 13:
            patterns.append("V11指标低位")
            
            # V12指标上升
            if indicator_data['V12'].iloc[latest_index] > 0:
                patterns.append("V11低位V12上升")
                
                # V12快速上升
                if indicator_data['V12'].iloc[latest_index] > 13:
                    patterns.append("V11低位V12快速上升")
        
        # 检查是否满足持续低位条件(基于AA条件的持续性)
        if len(indicator_data) >= 15:
            recent_aa = indicator_data['AA'].iloc[-15:].sum()
            if recent_aa >= 10:
                patterns.append("持续低位状态")

        # 检查是否满足低位回升条件(基于BB条件的持续性)
        if len(indicator_data) >= 10:
            recent_bb = indicator_data['BB'].iloc[-10:].sum()
            if recent_bb >= 5:
                patterns.append("低位持续回升")
        
        # 成交量特征
        if 'volume' in data.columns and len(data) >= 5:
            volume_ma5 = data['volume'].rolling(window=5).mean().iloc[latest_index]
            latest_volume = data['volume'].iloc[latest_index]
            
            if pd.notna(volume_ma5) and volume_ma5 > 0:
                volume_ratio = latest_volume / volume_ma5
                
                if volume_ratio < 0.8:
                    patterns.append("成交量萎缩")
                elif volume_ratio > 1.5 and indicator_data['EMA_V11_3'].iloc[latest_index] < 13:
                    patterns.append("低位放量")
        
        return patterns

    def validate_buy_point_four_elements(self, data: pd.DataFrame, index: int) -> Dict[str, bool]:
        """
        验证ZXM买点四要素

        Args:
            data: 股价数据
            index: 当前位置索引

        Returns:
            Dict[str, bool]: 四要素验证结果
        """
        if index < 60:  # 需要足够的历史数据
            return {
                'trend_intact': False,
                'volume_shrink': False,
                'pullback_support': False,
                'bs_signal': False
            }

        # 1. 趋势不破：60日和120日均线上移
        ma60 = data['close'].rolling(window=60).mean()
        ma120 = data['close'].rolling(window=120).mean()

        trend_intact = False
        if index >= 120:
            trend_intact = (ma60.iloc[index] >= ma60.iloc[index-1] or
                          ma120.iloc[index] >= ma120.iloc[index-1])

        # 2. 缩量：当日成交量小于前2日或前3日平均量
        volume_shrink = False
        if index >= 3:
            current_vol = data['volume'].iloc[index]
            avg_vol_2 = data['volume'].iloc[index-2:index].mean()
            avg_vol_3 = data['volume'].iloc[index-3:index].mean()
            volume_shrink = (current_vol * 1.1 < avg_vol_2 or
                           current_vol * 1.1 < avg_vol_3)

        # 3. 回踩支撑：价格接近10/20/30/60日均线
        pullback_support = False
        if index >= 60:
            current_price = data['close'].iloc[index]
            ma10 = data['close'].rolling(window=10).mean().iloc[index]
            ma20 = data['close'].rolling(window=20).mean().iloc[index]
            ma30 = data['close'].rolling(window=30).mean().iloc[index]
            ma60_val = ma60.iloc[index]

            # 价格距离均线小于4%视为回踩支撑
            support_distances = [
                abs(current_price - ma10) / ma10,
                abs(current_price - ma20) / ma20,
                abs(current_price - ma30) / ma30,
                abs(current_price - ma60_val) / ma60_val
            ]
            pullback_support = any(dist <= 0.04 for dist in support_distances)

        # 4. BS吸筹信号：当前有BUY信号
        bs_signal = False
        if hasattr(self, '_result') and self._result is not None:
            if 'BUY' in self._result.columns and index < len(self._result):
                bs_signal = bool(self._result['BUY'].iloc[index])

        return {
            'trend_intact': bool(trend_intact),
            'volume_shrink': bool(volume_shrink),
            'pullback_support': bool(pullback_support),
            'bs_signal': bool(bs_signal)
        }

    def set_parameters(self, **kwargs):
        """
        设置指标参数

        Args:
            **kwargs: 参数字典，可包含：
                - v11_threshold: V11低位阈值，默认13
                - v12_threshold: V12上升阈值，默认13
                - xg_threshold: XG吸筹强度阈值，默认3
        """
        self.v11_threshold = kwargs.get('v11_threshold', 13)
        self.v12_threshold = kwargs.get('v12_threshold', 13)
        self.xg_threshold = kwargs.get('xg_threshold', 3)

    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        获取ZXMAbsorb相关形态

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 包含形态信息的DataFrame
        """
        # 确保已计算指标
        if not self.has_result():
            self._calculate(data, **kwargs)

        patterns = pd.DataFrame(index=data.index)

        # 如果没有计算结果，返回空DataFrame
        if self._result is None or self._result.empty:
            return patterns

        # 基于识别的形态创建布尔列
        identified_patterns = self.identify_patterns(data)

        # 初始化所有可能的形态列
        pattern_columns = [
            'ZXM_ABSORB_SIGNAL', 'ZXM_STRONG_ABSORB', 'ZXM_MEDIUM_ABSORB', 'ZXM_WEAK_ABSORB',
            'ZXM_V11_LOW', 'ZXM_V11_LOW_V12_UP', 'ZXM_V11_LOW_V12_FAST_UP',
            'ZXM_CONTINUOUS_LOW', 'ZXM_LOW_RECOVERY', 'ZXM_VOLUME_SHRINK',
            'ZXM_LOW_VOLUME_EXPANSION', 'ZXM_ABSORB_CONFIRMATION'
        ]

        for col in pattern_columns:
            patterns[col] = False

        # 根据识别的形态设置相应的布尔值
        for pattern in identified_patterns:
            if "ZXM吸筹信号" in pattern:
                patterns['ZXM_ABSORB_SIGNAL'] = True
                patterns['ZXM_ABSORB_CONFIRMATION'] = True

            if "强吸筹信号" in pattern:
                patterns['ZXM_STRONG_ABSORB'] = True
            elif "中等吸筹信号" in pattern:
                patterns['ZXM_MEDIUM_ABSORB'] = True
            elif "弱吸筹信号" in pattern:
                patterns['ZXM_WEAK_ABSORB'] = True

            if "V11指标低位" in pattern:
                patterns['ZXM_V11_LOW'] = True

            if "V11低位V12上升" in pattern:
                patterns['ZXM_V11_LOW_V12_UP'] = True

            if "V11低位V12快速上升" in pattern:
                patterns['ZXM_V11_LOW_V12_FAST_UP'] = True

            if "持续低位状态" in pattern:
                patterns['ZXM_CONTINUOUS_LOW'] = True

            if "低位持续回升" in pattern:
                patterns['ZXM_LOW_RECOVERY'] = True

            if "成交量萎缩" in pattern:
                patterns['ZXM_VOLUME_SHRINK'] = True

            if "低位放量" in pattern:
                patterns['ZXM_LOW_VOLUME_EXPANSION'] = True

        return patterns

    def calculate_confidence(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """
        计算ZXMAbsorb指标的置信度

        Args:
            score: 得分序列
            patterns: 检测到的形态DataFrame
            signals: 生成的信号字典

        Returns:
            float: 置信度分数 (0-1)
        """
        if score.empty:
            return 0.5

        # 基础置信度
        confidence = 0.5

        # 1. 基于评分的置信度
        last_score = score.iloc[-1]

        # 极端评分置信度较高
        if last_score > 80:
            confidence += 0.3
        elif last_score > 70:
            confidence += 0.2
        elif last_score < 30:
            confidence += 0.25
        elif last_score < 40:
            confidence += 0.15
        else:
            confidence += 0.1

        # 2. 基于数据质量的置信度
        if hasattr(self, '_result') and self._result is not None:
            # 检查是否有ZXM吸筹数据
            zxm_columns = ['BUY', 'XG', 'EMA_V11_3', 'V12']
            available_columns = [col for col in zxm_columns if col in self._result.columns]
            if available_columns:
                # ZXM数据越完整，置信度越高
                data_completeness = len(available_columns) / len(zxm_columns)
                confidence += data_completeness * 0.1

        # 3. 基于形态的置信度
        if not patterns.empty:
            # 检查ZXM吸筹形态（只计算布尔列）
            bool_columns = patterns.select_dtypes(include=[bool]).columns
            if len(bool_columns) > 0:
                pattern_count = patterns[bool_columns].sum().sum()
                if pattern_count > 0:
                    confidence += min(pattern_count * 0.02, 0.15)

        # 4. 基于信号的置信度
        if signals:
            # 检查信号强度
            signal_count = sum(1 for signal in signals.values() if hasattr(signal, 'any') and signal.any())
            if signal_count > 0:
                confidence += min(signal_count * 0.05, 0.1)

        # 5. 基于数据长度的置信度
        if len(score) >= 60:  # 两个月数据
            confidence += 0.1
        elif len(score) >= 30:  # 一个月数据
            confidence += 0.05

        # 确保置信度在0-1范围内
        return max(0.0, min(1.0, confidence))

    def register_patterns(self):
        """
        注册ZXMAbsorb指标的形态到全局形态注册表
        """
        # 注册吸筹信号形态
        self.register_pattern_to_registry(
            pattern_id="ZXM_ABSORB_SIGNAL",
            display_name="ZXM吸筹信号",
            description="基于KDJ衍生指标的低位吸筹信号",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=30.0
        )

        self.register_pattern_to_registry(
            pattern_id="ZXM_STRONG_ABSORB",
            display_name="ZXM强吸筹信号",
            description="强度≥5的ZXM吸筹信号，主力大量吸筹",
            pattern_type="BULLISH",
            default_strength="VERY_STRONG",
            score_impact=40.0
        )

        self.register_pattern_to_registry(
            pattern_id="ZXM_MEDIUM_ABSORB",
            display_name="ZXM中等吸筹信号",
            description="强度4的ZXM吸筹信号，主力适度吸筹",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=30.0
        )

        self.register_pattern_to_registry(
            pattern_id="ZXM_WEAK_ABSORB",
            display_name="ZXM弱吸筹信号",
            description="强度3的ZXM吸筹信号，主力轻度吸筹",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=20.0
        )

        # 注册V11指标形态
        self.register_pattern_to_registry(
            pattern_id="ZXM_V11_LOW",
            display_name="ZXM V11指标低位",
            description="V11指标处于低位区域，具备吸筹基础",
            pattern_type="NEUTRAL",
            default_strength="MEDIUM",
            score_impact=10.0
        )

        self.register_pattern_to_registry(
            pattern_id="ZXM_V11_LOW_V12_UP",
            display_name="ZXM V11低位V12上升",
            description="V11低位且V12上升，吸筹动能增强",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=15.0
        )

        self.register_pattern_to_registry(
            pattern_id="ZXM_V11_LOW_V12_FAST_UP",
            display_name="ZXM V11低位V12快速上升",
            description="V11低位且V12快速上升，强烈吸筹信号",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=25.0
        )

        # 注册持续性形态
        self.register_pattern_to_registry(
            pattern_id="ZXM_CONTINUOUS_LOW",
            display_name="ZXM持续低位状态",
            description="15天内满足10次低位条件，持续吸筹环境",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=15.0
        )

        self.register_pattern_to_registry(
            pattern_id="ZXM_LOW_RECOVERY",
            display_name="ZXM低位持续回升",
            description="10天内满足5次低位回升条件，吸筹后启动",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=20.0
        )

        # 注册成交量确认形态
        self.register_pattern_to_registry(
            pattern_id="ZXM_VOLUME_SHRINK",
            display_name="ZXM成交量萎缩",
            description="吸筹过程中成交量萎缩，主力控盘",
            pattern_type="NEUTRAL",
            default_strength="MEDIUM",
            score_impact=8.0
        )

        self.register_pattern_to_registry(
            pattern_id="ZXM_LOW_VOLUME_EXPANSION",
            display_name="ZXM低位放量",
            description="低位区域放量，主力积极吸筹",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=18.0
        )

        self.register_pattern_to_registry(
            pattern_id="ZXM_ABSORB_CONFIRMATION",
            display_name="ZXM吸筹确认",
            description="综合确认的ZXM吸筹信号",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=25.0
        )

    def generate_trading_signals(self, data: pd.DataFrame, **kwargs) -> dict:
        """
        生成ZXMAbsorb交易信号

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            dict: 包含买卖信号的字典
        """
        # 确保已计算指标
        if not self.has_result():
            self._calculate(data, **kwargs)

        if self._result is None or self._result.empty:
            return {
                'buy_signal': pd.Series(False, index=data.index),
                'sell_signal': pd.Series(False, index=data.index),
                'signal_strength': pd.Series(0.0, index=data.index)
            }

        # 使用generate_signals方法生成详细信号
        detailed_signals = self.generate_signals(data, **kwargs)

        # 转换为简化的信号格式
        buy_signal = detailed_signals['buy_signal']
        sell_signal = detailed_signals['sell_signal']

        # 计算信号强度
        signal_strength = pd.Series(0.0, index=data.index)

        # 基于吸筹强度计算信号强度
        if 'XG' in self._result.columns:
            xg_values = self._result['XG']

            # 买入信号强度基于XG值
            for i in range(len(data)):
                if buy_signal.iloc[i]:
                    xg_value = xg_values.iloc[i]
                    # XG值3-6对应信号强度0.3-1.0
                    if xg_value >= 3:
                        signal_strength.iloc[i] = min(0.3 + (xg_value - 3) * 0.175, 1.0)

        return {
            'buy_signal': buy_signal,
            'sell_signal': sell_signal,
            'signal_strength': signal_strength
        }

    def get_indicator_type(self) -> str:
        """
        获取指标类型

        Returns:
            str: 指标类型
        """
        return "ZXM_ABSORB"

    def get_pattern_info(self, pattern_id: str) -> dict:
        """
        获取形态信息
        
        Args:
            pattern_id: 形态ID
            
        Returns:
            dict: 形态信息字典
        """
        # 默认形态信息映射
        pattern_info_map = {
            # 基础形态
            'bullish': {'name': '看涨形态', 'description': '指标显示看涨信号', 'type': 'BULLISH'},
            'bearish': {'name': '看跌形态', 'description': '指标显示看跌信号', 'type': 'BEARISH'},
            'neutral': {'name': '中性形态', 'description': '指标显示中性信号', 'type': 'NEUTRAL'},
            
            # 通用形态
            'strong_signal': {'name': '强信号', 'description': '强烈的技术信号', 'type': 'STRONG'},
            'weak_signal': {'name': '弱信号', 'description': '较弱的技术信号', 'type': 'WEAK'},
            'trend_up': {'name': '上升趋势', 'description': '价格呈上升趋势', 'type': 'BULLISH'},
            'trend_down': {'name': '下降趋势', 'description': '价格呈下降趋势', 'type': 'BEARISH'},
        }
        
        # 默认形态信息
        default_pattern = {
            'name': pattern_id.replace('_', ' ').title(),
            'description': f'{pattern_id}形态',
            'type': 'UNKNOWN'
        }
        
        return pattern_info_map.get(pattern_id, default_pattern)
