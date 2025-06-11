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
            
        公式说明：
        V11:=3*SMA((CLOSE-LLV(LOW,55))/(HHV(HIGH,55)-LLV(LOW,55))*100,5,1)-2*SMA(SMA((CLOSE-LLV(LOW,55))/(HHV(HIGH,55)-LLV(LOW,55))*100,5,1),3,1);
        V12:=(EMA(V11,3)-REF(EMA(V11,3),1))/REF(EMA(V11,3),1)*100;
        AA:=EMA(V11,3)<=13;
        BB:=EMA(V11,3)<=13 AND V12>13;
        XC:=COUNT(AA,15)>=10; // 满足低位条件
        XD:=COUNT(BB,10)>=5;  // 满足低位回升条件
        XG:=COUNT(AA OR BB,6); // 近期满足吸筹条件次数
        BUY:=XG>=3; // 满足3次以上视为有效吸筹信号
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["close", "high", "low"])
        
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
        
        # 定义AA和BB条件
        aa = ema_v11_3 <= 13
        bb = (ema_v11_3 <= 13) & (v12 > 13)
        
        # 计算满足条件的次数
        xc = np.zeros_like(close, dtype=bool)
        xd = np.zeros_like(close, dtype=bool)
        xg = np.zeros_like(close, dtype=int)
        buy = np.zeros_like(close, dtype=bool)
        
        for i in range(15, len(close)):
            # 计算XC：近15天内AA条件满足10次以上
            xc[i] = np.sum(aa[i-14:i+1]) >= 10
            
            # 计算XD：近10天内BB条件满足5次以上
            if i >= 10:
                xd[i] = np.sum(bb[i-9:i+1]) >= 5
            
            # 计算XG：近6天内AA或BB条件满足的次数
            if i >= 6:
                xg[i] = np.sum(aa[i-5:i+1] | bb[i-5:i+1])
            
            # 计算BUY：XG >= 3
            buy[i] = xg[i] >= 3
        
        # 添加计算结果到数据框
        result["V11"] = v11
        result["V12"] = v12
        result["EMA_V11_3"] = ema_v11_3
        result["AA"] = aa
        result["BB"] = bb
        result["XC"] = xc
        result["XD"] = xd
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
        scores = pd.Series(50, index=data.index)
        
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
        
        # 检查是否满足XC条件(15天内满足10次低位条件)
        if indicator_data['XC'].iloc[latest_index]:
            patterns.append("持续低位状态")
        
        # 检查是否满足XD条件(10天内满足5次低位回升条件)
        if indicator_data['XD'].iloc[latest_index]:
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