"""
MACD指标分析模块

提供MACD指标的计算和分析功能
"""

from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from utils.technical_utils import calculate_macd
from indicators.base_indicator import BaseIndicator
from utils.signal_utils import crossover, crossunder

class MACD(BaseIndicator):
    """MACD指标"""
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        """
        初始化MACD指标
        
        Args:
            fast_period: 快线周期
            slow_period: 慢线周期
            signal_period: 信号线周期
        """
        super().__init__()
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        
        # 定义MACD形态
        self.patterns = {
            'macd_golden_cross': {
                'name': 'MACD金叉',
                'description': 'DIF从下向上穿越DEA',
                'analyzer': self._analyze_golden_cross
            },
            'macd_death_cross': {
                'name': 'MACD死叉',
                'description': 'DIF从上向下穿越DEA',
                'analyzer': self._analyze_death_cross
            },
            'macd_divergence': {
                'name': 'MACD背离',
                'description': '价格创新高/新低，但MACD未创新高/新低',
                'analyzer': self._analyze_divergence
            }
        }
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算MACD指标
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            包含MACD指标的DataFrame
        """
        df = data.copy()
        
        # 计算EMA
        df['ema_fast'] = df['close'].ewm(span=self.fast_period, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=self.slow_period, adjust=False).mean()
        
        # 计算MACD线
        df['macd'] = df['ema_fast'] - df['ema_slow']
        
        # 计算信号线
        df['signal'] = df['macd'].ewm(span=self.signal_period, adjust=False).mean()
        
        # 计算柱状图
        df['hist'] = df['macd'] - df['signal']
        
        return df
    
    def get_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        获取MACD信号
        
        Args:
            data: 包含MACD指标的DataFrame
            
        Returns:
            包含信号的字典
        """
        signals = {
            'golden_cross': False,  # 金叉
            'death_cross': False,   # 死叉
            'hist_positive': False, # 柱状图为正
            'hist_negative': False, # 柱状图为负
            'hist_increasing': False, # 柱状图增加
            'hist_decreasing': False  # 柱状图减少
        }
        
        if len(data) < 2:
            return signals
            
        # 判断金叉和死叉
        signals['golden_cross'] = crossover(data['macd'], data['signal'])
        signals['death_cross'] = crossunder(data['macd'], data['signal'])
        
        # 判断柱状图状态
        signals['hist_positive'] = data['hist'].iloc[-1] > 0
        signals['hist_negative'] = data['hist'].iloc[-1] < 0
        signals['hist_increasing'] = data['hist'].iloc[-1] > data['hist'].iloc[-2]
        signals['hist_decreasing'] = data['hist'].iloc[-1] < data['hist'].iloc[-2]
        
        return signals
        
    def get_score(self, data: pd.DataFrame) -> float:
        """
        计算MACD得分
        
        Args:
            data: 包含MACD指标的DataFrame
            
        Returns:
            MACD得分 (0-1)
        """
        if len(data) < 2:
            return 0.0
            
        score = 0.0
        signals = self.get_signals(data)
        
        # 根据信号计算得分
        if signals['golden_cross']:
            score += 0.4
        elif signals['death_cross']:
            score -= 0.4
            
        if signals['hist_positive']:
            score += 0.2
        elif signals['hist_negative']:
            score -= 0.2
            
        if signals['hist_increasing']:
            score += 0.2
        elif signals['hist_decreasing']:
            score -= 0.2
            
        # 归一化得分到0-1范围
        return max(0.0, min(1.0, (score + 1.0) / 2.0))
    
    def analyze_pattern(self, pattern_id: str, data: pd.DataFrame) -> List[Dict]:
        """
        分析指定形态
        
        Args:
            pattern_id: 形态ID
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            List[Dict]: 形态识别结果列表
        """
        if pattern_id not in self.patterns:
            raise ValueError(f"不支持的MACD形态: {pattern_id}")
            
        return self.patterns[pattern_id]['analyzer'](data)
    
    def _analyze_golden_cross(self, data: pd.DataFrame) -> List[Dict]:
        """
        分析MACD金叉形态
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            List[Dict]: 金叉形态识别结果列表
        """
        dif, dea, macd = self.calculate(data['close'])
        results = []
        
        # 寻找金叉
        for i in range(1, len(data)):
            if dif.iloc[i-1] < dea.iloc[i-1] and dif.iloc[i] > dea.iloc[i]:
                # 计算形态强度
                strength = min(1.0, abs(dif.iloc[i] - dea.iloc[i]) / abs(dif.iloc[i-1] - dea.iloc[i-1]))
                
                results.append({
                    'date': data.index[i],
                    'pattern': 'macd_golden_cross',
                    'strength': strength,
                    'price': data['close'].iloc[i],
                    'dif': dif.iloc[i],
                    'dea': dea.iloc[i],
                    'macd': macd.iloc[i]
                })
        
        return results
    
    def _analyze_death_cross(self, data: pd.DataFrame) -> List[Dict]:
        """
        分析MACD死叉形态
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            List[Dict]: 死叉形态识别结果列表
        """
        dif, dea, macd = self.calculate(data['close'])
        results = []
        
        # 寻找死叉
        for i in range(1, len(data)):
            if dif.iloc[i-1] > dea.iloc[i-1] and dif.iloc[i] < dea.iloc[i]:
                # 计算形态强度
                strength = min(1.0, abs(dif.iloc[i] - dea.iloc[i]) / abs(dif.iloc[i-1] - dea.iloc[i-1]))
                
                results.append({
                    'date': data.index[i],
                    'pattern': 'macd_death_cross',
                    'strength': strength,
                    'price': data['close'].iloc[i],
                    'dif': dif.iloc[i],
                    'dea': dea.iloc[i],
                    'macd': macd.iloc[i]
                })
        
        return results
    
    def _analyze_divergence(self, data: pd.DataFrame) -> List[Dict]:
        """
        分析MACD背离形态
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            List[Dict]: 背离形态识别结果列表
        """
        dif, dea, macd = self.calculate(data['close'])
        results = []
        
        # 寻找顶背离
        for i in range(2, len(data)):
            # 价格创新高但MACD未创新高
            if (data['high'].iloc[i] > data['high'].iloc[i-1] and 
                data['high'].iloc[i-1] > data['high'].iloc[i-2] and
                macd.iloc[i] < macd.iloc[i-1] and 
                macd.iloc[i-1] < macd.iloc[i-2]):
                
                # 计算形态强度
                price_change = (data['high'].iloc[i] - data['high'].iloc[i-2]) / data['high'].iloc[i-2]
                macd_change = (macd.iloc[i] - macd.iloc[i-2]) / abs(macd.iloc[i-2])
                strength = min(1.0, abs(price_change * macd_change))
                
                results.append({
                    'date': data.index[i],
                    'pattern': 'macd_divergence',
                    'type': 'top',
                    'strength': strength,
                    'price': data['high'].iloc[i],
                    'macd': macd.iloc[i]
                })
            
            # 价格创新低但MACD未创新低
            elif (data['low'].iloc[i] < data['low'].iloc[i-1] and 
                  data['low'].iloc[i-1] < data['low'].iloc[i-2] and
                  macd.iloc[i] > macd.iloc[i-1] and 
                  macd.iloc[i-1] > macd.iloc[i-2]):
                
                # 计算形态强度
                price_change = (data['low'].iloc[i] - data['low'].iloc[i-2]) / data['low'].iloc[i-2]
                macd_change = (macd.iloc[i] - macd.iloc[i-2]) / abs(macd.iloc[i-2])
                strength = min(1.0, abs(price_change * macd_change))
                
                results.append({
                    'date': data.index[i],
                    'pattern': 'macd_divergence',
                    'type': 'bottom',
                    'strength': strength,
                    'price': data['low'].iloc[i],
                    'macd': macd.iloc[i]
                })
        
        return results