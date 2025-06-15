#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
综合技术指标组合器

创建多指标结合的复合技术分析指标
"""

import pandas as pd
import numpy as np
from typing import Union, List, Dict, Optional, Tuple, Any, Callable
import warnings

from indicators.base_indicator import BaseIndicator
from indicators.adapter import CompositeIndicator, register_indicator
from utils.logger import get_logger

# 静默警告
warnings.filterwarnings("ignore", category=RuntimeWarning)

logger = get_logger(__name__)


class TechnicalComposite:
    """
    技术指标组合器
    
    创建和管理复合技术指标，用于多指标协同分析
    """
    
    def __init__(self):
        """初始化技术指标组合器"""
        self.composites = {}
    
    def create_trend_strength_composite(self, name: str = "TrendStrengthComposite") -> BaseIndicator:
        """
        创建趋势强度复合指标
        
        组合多个趋势指标来评估趋势的强度和可持续性
        
        Args:
            name: 复合指标名称
            
        Returns:
            BaseIndicator: 创建的复合指标
        """
        # 指定要组合的指标
        indicators = ["MACD", "RSI", "ADX", "TRENDSTRENGTH"]
        
        # 定义组合函数
        def combine_trend_indicators(results: List[pd.DataFrame], data: pd.DataFrame) -> pd.DataFrame:
            """组合趋势指标结果"""
            # 提取各指标结果
            macd_result = results[0]
            rsi_result = results[1] 
            adx_result = results[2]
            trend_strength_result = results[3]
            
            # 创建组合结果
            combined = pd.DataFrame(index=data.index)
            
            # MACD信号
            if 'histogram' in macd_result.columns:
                combined['macd_signal'] = np.where(macd_result['histogram'] > 0, 1, 
                                        np.where(macd_result['histogram'] < 0, -1, 0))
            else:
                # 尝试其他可能的列名
                hist_col = [col for col in macd_result.columns if 'hist' in col.lower()]
                if hist_col:
                    combined['macd_signal'] = np.where(macd_result[hist_col[0]] > 0, 1, 
                                            np.where(macd_result[hist_col[0]] < 0, -1, 0))
                else:
                    combined['macd_signal'] = 0
            
            # RSI信号
            if 'rsi' in rsi_result.columns:
                combined['rsi_signal'] = np.where(rsi_result['rsi'] > 60, 1, 
                                      np.where(rsi_result['rsi'] < 40, -1, 0))
            else:
                # 使用第一列
                first_col = rsi_result.columns[0]
                combined['rsi_signal'] = np.where(rsi_result[first_col] > 60, 1, 
                                      np.where(rsi_result[first_col] < 40, -1, 0))
            
            # ADX信号
            if 'adx' in adx_result.columns:
                combined['adx_strength'] = adx_result['adx'] / 100
            else:
                # 使用第一列
                first_col = adx_result.columns[0]
                combined['adx_strength'] = adx_result[first_col] / 100
            
            # 趋势强度信号
            if 'trend_strength' in trend_strength_result.columns:
                combined['base_strength'] = trend_strength_result['trend_strength']
            else:
                # 使用第一列
                first_col = trend_strength_result.columns[0]
                combined['base_strength'] = trend_strength_result[first_col]
            
            # 计算综合趋势强度分数 (0-100)
            combined['trend_direction'] = combined['macd_signal'] + combined['rsi_signal']
            combined['trend_direction'] = combined['trend_direction'].clip(-2, 2) / 2  # 归一化到 [-1, 1]
            
            # 计算综合强度分数
            combined['strength'] = (combined['adx_strength'] + combined['base_strength']) / 2
            
            # 最终趋势强度分数
            combined['trend_strength_score'] = combined['trend_direction'] * combined['strength'] * 100
            combined['trend_strength_score'] = combined['trend_strength_score'].clip(-100, 100)
            
            # 趋势类别
            conditions = [
                (combined['trend_strength_score'] > 70),
                (combined['trend_strength_score'] > 40),
                (combined['trend_strength_score'] > 20),
                (combined['trend_strength_score'] > 0),
                (combined['trend_strength_score'] > -20),
                (combined['trend_strength_score'] > -40),
                (combined['trend_strength_score'] > -70),
                (combined['trend_strength_score'] <= -70)
            ]
            
            categories = [
                '强烈上升',
                '明显上升',
                '温和上升',
                '微弱上升',
                '微弱下降',
                '温和下降',
                '明显下降',
                '强烈下降'
            ]
            
            combined['trend_category'] = np.select(conditions, categories, default='无明显趋势')
            
            return combined
        
        # 创建复合指标
        composite = CompositeIndicator(
            name=name,
            indicators=indicators,
            combination_func=combine_trend_indicators,
            description="趋势强度复合指标，结合MACD、RSI、ADX和趋势强度评估整体趋势状态"
        )
        
        # 注册到全局注册表
        register_indicator(composite)
        
        # 保存到本地缓存
        self.composites[name] = composite
        
        return composite
    
    def create_volatility_composite(self, name: str = "VolatilityComposite") -> BaseIndicator:
        """
        创建波动性复合指标
        
        组合多个波动性指标来评估市场波动状况
        
        Args:
            name: 复合指标名称
            
        Returns:
            BaseIndicator: 创建的复合指标
        """
        # 指定要组合的指标
        indicators = ["ATR", "BOLL", "VIX", "INTRADAYVOLATILITY"]
        
        # 定义组合函数
        def combine_volatility_indicators(results: List[pd.DataFrame], data: pd.DataFrame) -> pd.DataFrame:
            """组合波动性指标结果"""
            # 提取各指标结果
            atr_result = results[0]
            boll_result = results[1]
            vix_result = results[2]
            intraday_vol_result = results[3]
            
            # 创建组合结果
            combined = pd.DataFrame(index=data.index)
            
            # ATR百分比
            if 'atr' in atr_result.columns:
                combined['atr_pct'] = atr_result['atr'] / data['close'] * 100
            else:
                # 使用第一列
                first_col = atr_result.columns[0]
                combined['atr_pct'] = atr_result[first_col] / data['close'] * 100
            
            # 布林带宽度
            if all(col in boll_result.columns for col in ['upper', 'lower']):
                combined['boll_width'] = (boll_result['upper'] - boll_result['lower']) / boll_result['middle'] * 100
            else:
                # 查找可能的列名
                upper_col = [col for col in boll_result.columns if 'upper' in col.lower() or 'up' in col.lower()]
                lower_col = [col for col in boll_result.columns if 'lower' in col.lower() or 'down' in col.lower()]
                middle_col = [col for col in boll_result.columns if 'middle' in col.lower() or 'mid' in col.lower()]
                
                if upper_col and lower_col and middle_col:
                    combined['boll_width'] = (boll_result[upper_col[0]] - boll_result[lower_col[0]]) / boll_result[middle_col[0]] * 100
                else:
                    combined['boll_width'] = 0
            
            # VIX信号
            if 'vix' in vix_result.columns:
                combined['vix_signal'] = vix_result['vix'] / 100
            else:
                # 使用第一列
                first_col = vix_result.columns[0]
                combined['vix_signal'] = vix_result[first_col] / 100
            
            # 日内波动率
            if 'volatility' in intraday_vol_result.columns:
                combined['intraday_vol'] = intraday_vol_result['volatility']
            else:
                # 使用第一列
                first_col = intraday_vol_result.columns[0]
                combined['intraday_vol'] = intraday_vol_result[first_col]
            
            # 计算归一化的波动率分数
            # 对所有输入进行Z-score标准化
            for col in ['atr_pct', 'boll_width', 'vix_signal', 'intraday_vol']:
                # 安全处理：如果列全是0或相同值，设置为0
                if combined[col].std() == 0:
                    combined[f'{col}_norm'] = 0
                else:
                    combined[f'{col}_norm'] = (combined[col] - combined[col].mean()) / combined[col].std()
                    # 限制在 [-3, 3] 范围内
                    combined[f'{col}_norm'] = combined[f'{col}_norm'].clip(-3, 3)
                    # 缩放到 [0, 1] 区间
                    combined[f'{col}_norm'] = (combined[f'{col}_norm'] + 3) / 6
            
            # 计算综合波动率分数 (0-100)
            combined['volatility_score'] = (
                combined['atr_pct_norm'] * 0.3 + 
                combined['boll_width_norm'] * 0.3 + 
                combined['vix_signal_norm'] * 0.2 + 
                combined['intraday_vol_norm'] * 0.2
            ) * 100
            
            # 波动性类别
            conditions = [
                (combined['volatility_score'] > 80),
                (combined['volatility_score'] > 60),
                (combined['volatility_score'] > 40),
                (combined['volatility_score'] > 20),
                (combined['volatility_score'] <= 20)
            ]
            
            categories = [
                '极高波动',
                '高波动',
                '中等波动',
                '低波动',
                '极低波动'
            ]
            
            combined['volatility_category'] = np.select(conditions, categories, default='中等波动')
            
            return combined
        
        # 创建复合指标
        composite = CompositeIndicator(
            name=name,
            indicators=indicators,
            combination_func=combine_volatility_indicators,
            description="波动性复合指标，结合ATR、布林带宽度、VIX和日内波动率评估市场波动状况"
        )
        
        # 注册到全局注册表
        register_indicator(composite)
        
        # 保存到本地缓存
        self.composites[name] = composite
        
        return composite
    
    def create_momentum_composite(self, name: str = "MomentumComposite") -> BaseIndicator:
        """
        创建动量复合指标
        
        组合多个动量指标来评估价格动量
        
        Args:
            name: 复合指标名称
            
        Returns:
            BaseIndicator: 创建的复合指标
        """
        # 指定要组合的指标
        indicators = ["RSI", "MOMENTUM", "ROC", "CCI"]
        
        # 定义组合函数
        def combine_momentum_indicators(results: List[pd.DataFrame], data: pd.DataFrame) -> pd.DataFrame:
            """组合动量指标结果"""
            # 提取各指标结果
            rsi_result = results[0]
            momentum_result = results[1]
            roc_result = results[2]
            cci_result = results[3]
            
            # 创建组合结果
            combined = pd.DataFrame(index=data.index)
            
            # RSI信号 (0-100)
            if 'rsi' in rsi_result.columns:
                combined['rsi'] = rsi_result['rsi']
            else:
                # 使用第一列
                first_col = rsi_result.columns[0]
                combined['rsi'] = rsi_result[first_col]
            
            # 动量信号
            if 'momentum' in momentum_result.columns:
                combined['momentum'] = momentum_result['momentum']
            else:
                # 使用第一列
                first_col = momentum_result.columns[0]
                combined['momentum'] = momentum_result[first_col]
            
            # ROC信号 (百分比变化率)
            if 'roc' in roc_result.columns:
                combined['roc'] = roc_result['roc']
            else:
                # 使用第一列
                first_col = roc_result.columns[0]
                combined['roc'] = roc_result[first_col]
            
            # CCI信号
            if 'cci' in cci_result.columns:
                combined['cci'] = cci_result['cci']
            else:
                # 使用第一列
                first_col = cci_result.columns[0]
                combined['cci'] = cci_result[first_col]
            
            # 归一化处理
            # RSI已经是0-100的范围，转化为-1到1
            combined['rsi_norm'] = (combined['rsi'] - 50) / 50
            
            # 动量标准化
            if combined['momentum'].std() != 0:
                combined['momentum_norm'] = (combined['momentum'] - combined['momentum'].mean()) / combined['momentum'].std()
                combined['momentum_norm'] = combined['momentum_norm'].clip(-3, 3) / 3
            else:
                combined['momentum_norm'] = 0
            
            # ROC标准化
            if combined['roc'].std() != 0:
                combined['roc_norm'] = (combined['roc'] - combined['roc'].mean()) / combined['roc'].std()
                combined['roc_norm'] = combined['roc_norm'].clip(-3, 3) / 3
            else:
                combined['roc_norm'] = 0
            
            # CCI标准化 (通常在 -100 到 100 之间，但可能超出)
            combined['cci_norm'] = combined['cci'] / 200
            combined['cci_norm'] = combined['cci_norm'].clip(-1, 1)
            
            # 计算综合动量分数 (-100 到 100)
            combined['momentum_score'] = (
                combined['rsi_norm'] * 0.3 + 
                combined['momentum_norm'] * 0.25 + 
                combined['roc_norm'] * 0.25 + 
                combined['cci_norm'] * 0.2
            ) * 100
            
            # 动量类别
            conditions = [
                (combined['momentum_score'] > 70),
                (combined['momentum_score'] > 40),
                (combined['momentum_score'] > 10),
                (combined['momentum_score'] > -10),
                (combined['momentum_score'] > -40),
                (combined['momentum_score'] > -70),
                (combined['momentum_score'] <= -70)
            ]
            
            categories = [
                '极强上升动量',
                '强上升动量',
                '弱上升动量',
                '中性动量',
                '弱下降动量',
                '强下降动量',
                '极强下降动量'
            ]
            
            combined['momentum_category'] = np.select(conditions, categories, default='中性动量')
            
            return combined
        
        # 创建复合指标
        composite = CompositeIndicator(
            name=name,
            indicators=indicators,
            combination_func=combine_momentum_indicators,
            description="动量复合指标，结合RSI、动量指标、变化率和CCI评估价格动量"
        )
        
        # 注册到全局注册表
        register_indicator(composite)
        
        # 保存到本地缓存
        self.composites[name] = composite
        
        return composite
    
    def create_market_health_composite(self, name: str = "MarketHealthComposite") -> BaseIndicator:
        """
        创建市场健康度复合指标
        
        综合评估市场健康状况，包括趋势、波动性和动量
        
        Args:
            name: 复合指标名称
            
        Returns:
            BaseIndicator: 创建的复合指标
        """
        # 先创建子复合指标
        trend_composite = self.create_trend_strength_composite()
        volatility_composite = self.create_volatility_composite()
        momentum_composite = self.create_momentum_composite()
        
        # 指定要组合的指标
        indicators = [trend_composite, volatility_composite, momentum_composite, "VR"]
        
        # 定义组合函数
        def combine_market_health_indicators(results: List[pd.DataFrame], data: pd.DataFrame) -> pd.DataFrame:
            """组合市场健康度指标结果"""
            # 提取各指标结果
            trend_result = results[0]
            volatility_result = results[1]
            momentum_result = results[2]
            vr_result = results[3]  # 成交量比率
            
            # 创建组合结果
            combined = pd.DataFrame(index=data.index)
            
            # 从子复合指标中提取关键信息
            combined['trend_score'] = trend_result['trend_strength_score']
            combined['trend_category'] = trend_result['trend_category']
            
            combined['volatility_score'] = volatility_result['volatility_score']
            combined['volatility_category'] = volatility_result['volatility_category']
            
            combined['momentum_score'] = momentum_result['momentum_score']
            combined['momentum_category'] = momentum_result['momentum_category']
            
            # 添加成交量健康度
            if 'vr' in vr_result.columns:
                combined['volume_ratio'] = vr_result['vr']
            else:
                # 使用第一列
                first_col = vr_result.columns[0]
                combined['volume_ratio'] = vr_result[first_col]
            
            # 归一化成交量比率 (通常以100为基准)
            combined['volume_health'] = (combined['volume_ratio'] - 100) / 100
            combined['volume_health'] = combined['volume_health'].clip(-1, 1)
            
            # 计算综合市场健康度分数 (0-100)
            # 1. 趋势得分转换为0-100
            combined['trend_score_norm'] = (combined['trend_score'] + 100) / 2
            
            # 2. 动量得分转换为0-100
            combined['momentum_score_norm'] = (combined['momentum_score'] + 100) / 2
            
            # 3. 波动率得分保持不变 (已经是0-100)
            
            # 4. 成交量健康度转换为0-100
            combined['volume_score'] = (combined['volume_health'] + 1) * 50
            
            # 计算加权市场健康度分数
            combined['market_health_score'] = (
                combined['trend_score_norm'] * 0.35 + 
                combined['momentum_score_norm'] * 0.30 + 
                combined['volume_score'] * 0.25 - 
                (combined['volatility_score'] * 0.10)  # 波动率越高，健康度越低
            )
            
            # 处理可能的异常值
            combined['market_health_score'] = combined['market_health_score'].clip(0, 100)
            
            # 市场健康度类别
            conditions = [
                (combined['market_health_score'] > 80),
                (combined['market_health_score'] > 60),
                (combined['market_health_score'] > 40),
                (combined['market_health_score'] > 20),
                (combined['market_health_score'] <= 20)
            ]
            
            categories = [
                '极其健康',
                '健康',
                '中性',
                '不健康',
                '极度不健康'
            ]
            
            combined['market_health_category'] = np.select(conditions, categories, default='中性')
            
            # 市场状态综合评估
            # 结合趋势方向、波动性和健康度
            trend_direction = np.where(combined['trend_score'] > 0, 1, -1)
            vol_level = np.where(combined['volatility_score'] > 60, 2, 
                       np.where(combined['volatility_score'] > 30, 1, 0))
            health_level = np.where(combined['market_health_score'] > 60, 2, 
                          np.where(combined['market_health_score'] > 40, 1, 0))
            
            # 市场状态矩阵 (9种组合)
            market_state_matrix = {
                # 上升趋势
                (1, 0, 2): '稳健上升',
                (1, 0, 1): '温和上升',
                (1, 0, 0): '脆弱上升',
                (1, 1, 2): '动态上升',
                (1, 1, 1): '波动上升',
                (1, 1, 0): '不稳定上升',
                (1, 2, 2): '剧烈上升',
                (1, 2, 1): '过热上升',
                (1, 2, 0): '危险上升',
                # 下降趋势
                (-1, 0, 2): '有序回调',
                (-1, 0, 1): '温和下跌',
                (-1, 0, 0): '持续下跌',
                (-1, 1, 2): '调整中',
                (-1, 1, 1): '波动下跌',
                (-1, 1, 0): '加速下跌',
                (-1, 2, 2): '恐慌抛售',
                (-1, 2, 1): '剧烈波动',
                (-1, 2, 0): '市场崩溃'
            }
            
            # 构建状态键值并查找对应的市场状态
            combined['market_state'] = combined.apply(
                lambda row: market_state_matrix.get(
                    (1 if row['trend_score'] > 0 else -1,
                     2 if row['volatility_score'] > 60 else (1 if row['volatility_score'] > 30 else 0),
                     2 if row['market_health_score'] > 60 else (1 if row['market_health_score'] > 40 else 0)),
                    '未知状态'
                ),
                axis=1
            )
            
            return combined
        
        # 创建复合指标
        composite = CompositeIndicator(
            name=name,
            indicators=indicators,
            combination_func=combine_market_health_indicators,
            description="市场健康度复合指标，综合评估趋势强度、波动性、动量和成交量，全面分析市场状态"
        )
        
        # 注册到全局注册表
        register_indicator(composite)
        
        # 保存到本地缓存
        self.composites[name] = composite
        
        return composite


# 创建全局实例
technical_composite = TechnicalComposite()


# 预创建常用复合指标
def initialize_composites():
    """初始化常用复合指标"""
    technical_composite.create_trend_strength_composite()
    technical_composite.create_volatility_composite()
    technical_composite.create_momentum_composite()
    technical_composite.create_market_health_composite()
    
    logger.info("已初始化常用复合指标")

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



# 通过模块导入时自动初始化
# initialize_composites() 