#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
主力行为模式分析

识别和分析主力资金的行为模式，包括吸筹、控盘、洗盘、出货等
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple, Any
import warnings

from indicators.base_indicator import BaseIndicator
from indicators.chip_distribution import ChipDistribution
from utils.logger import get_logger

# 静默警告
warnings.filterwarnings("ignore", category=RuntimeWarning)

logger = get_logger(__name__)


class InstitutionalBehavior(BaseIndicator):
    """
    主力行为模式分析指标
    
    分析主力资金的行为特征，识别吸筹、控盘、洗盘、出货等模式
    """
    
    def __init__(self):
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        """初始化主力行为分析指标"""
        super().__init__(name="InstitutionalBehavior", description="主力行为模式分析指标")
        self.chip_distribution = ChipDistribution()
        self.volume_quantile = 0.85  # 新增：定义成交量分位数阈值
    
    def set_parameters(self, volume_quantile: float = None, **kwargs):
        """
        设置指标参数
        
        Args:
            volume_quantile: 成交量分位数阈值
        """
        if volume_quantile is not None:
            self.volume_quantile = volume_quantile

    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        获取InstitutionalBehavior相关形态

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

        # 1. 机构阶段形态
        if 'inst_phase' in self._result.columns:
            phase = self._result['inst_phase']

            # 各阶段形态
            patterns['INST_ABSORPTION_PHASE'] = phase == "吸筹期"
            patterns['INST_CONTROL_PHASE'] = phase == "控盘期"
            patterns['INST_RALLY_PHASE'] = phase == "拉升期"
            patterns['INST_DISTRIBUTION_PHASE'] = phase == "出货期"
            patterns['INST_WAITING_PHASE'] = phase == "观望期"

        # 2. 行为模式形态
        if 'behavior_pattern' in self._result.columns:
            behavior = self._result['behavior_pattern']

            # 吸筹形态
            patterns['INST_GENTLE_ABSORPTION'] = behavior == "温和吸筹"
            patterns['INST_STRONG_ABSORPTION'] = behavior == "强势吸筹"
            # 操作形态
            patterns['INST_WASHOUT'] = behavior == "洗盘"
            patterns['INST_RALLY'] = behavior == "拉升"
            patterns['INST_ACCELERATED_RALLY'] = behavior == "加速拉升"
            # 出货形态
            patterns['INST_DISTRIBUTION'] = behavior == "出货"
            patterns['INST_CONCENTRATED_DISTRIBUTION'] = behavior == "集中出货"

        # 3. 阶段转换形态
        if 'phase_change' in self._result.columns:
            phase_change = self._result['phase_change']

            # 重要转换点
            patterns['INST_ABSORPTION_COMPLETE'] = phase_change == "吸筹完成"
            patterns['INST_RALLY_START'] = phase_change == "开始拉升"
            patterns['INST_DISTRIBUTION_START'] = phase_change == "开始出货"
            patterns['INST_NEW_CYCLE_START'] = phase_change == "新一轮开始"

        # 4. 行为强度形态
        if 'behavior_intensity' in self._result.columns:
            intensity = self._result['behavior_intensity']

            # 强度分级
            patterns['INST_WEAK_ACTIVITY'] = intensity < 2
            patterns['INST_MODERATE_ACTIVITY'] = (intensity >= 2) & (intensity < 5)
            patterns['INST_STRONG_ACTIVITY'] = (intensity >= 5) & (intensity < 8)
            patterns['INST_EXTREME_ACTIVITY'] = intensity >= 8

        # 5. 集中度形态
        if 'inst_concentration' in self._result.columns:
            concentration = self._result['inst_concentration']

            # 集中度形态
            patterns['INST_HIGH_CONCENTRATION'] = concentration > 0.7
            patterns['INST_MODERATE_CONCENTRATION'] = (concentration >= 0.4) & (concentration <= 0.7)
            patterns['INST_LOW_CONCENTRATION'] = concentration < 0.4

            # 集中度变化
            conc_change = concentration.diff()
            patterns['INST_CONCENTRATION_RISING'] = conc_change > 0.1
            patterns['INST_CONCENTRATION_FALLING'] = conc_change < -0.1

        # 6. 获利盘形态
        if 'inst_profit_ratio' in self._result.columns:
            profit_ratio = self._result['inst_profit_ratio']

            # 获利盘形态
            patterns['INST_LOW_PROFIT'] = profit_ratio < 0.3
            patterns['INST_MODERATE_PROFIT'] = (profit_ratio >= 0.3) & (profit_ratio <= 0.7)
            patterns['INST_HIGH_PROFIT'] = profit_ratio > 0.7

        return patterns
    
    def _calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算主力行为模式指标
        
        Args:
            data: 输入数据，包含OHLCV数据
            
        Returns:
            pd.DataFrame: 计算结果，包含主力行为模式相关指标
        """
        # 确保数据包含必需的列
        required_columns = ["open", "high", "low", "close", "volume"]
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"数据必须包含'{col}'列")
        
        # 初始化结果数据框
        result = data.copy()
        
        # 计算筹码分布相关指标
        try:
            chip_result = self.chip_distribution.calculate(data)

            # 添加筹码分布的主要指标到结果中（使用简化的映射）
            if 'chip_concentration' in chip_result.columns:
                result['inst_concentration'] = chip_result['chip_concentration']
            if 'profit_ratio' in chip_result.columns:
                result['inst_profit_ratio'] = chip_result['profit_ratio']
            if 'avg_cost' in chip_result.columns:
                result['inst_cost'] = chip_result['avg_cost']

            # 计算活跃度评分（基于成交量和价格波动）
            volume_score = (data['volume'] / data['volume'].rolling(20).mean()).fillna(1.0)
            price_volatility = data['close'].rolling(5).std() / data['close'].rolling(5).mean()
            result['inst_activity_score'] = (volume_score * 10 + price_volatility * 100).fillna(0)

            # 简化的阶段判断
            result['inst_phase'] = self._determine_institutional_phase(data, result)

        except Exception as e:
            logger.error(f"计算筹码分布指标时出错: {e}")
            # 使用默认值
            result['inst_concentration'] = 0.5
            result['inst_profit_ratio'] = 0.5
            result['inst_cost'] = data['close']
            result['inst_activity_score'] = 10.0
            result['inst_phase'] = "观望期"
        
        # 计算行为模式识别指标
        self._calculate_behavior_patterns(data, result)

        # 确保behavior_pattern列存在
        if 'behavior_pattern' not in result.columns:
            result['behavior_pattern'] = "未知"

        # 计算行为模式变化指标
        self._calculate_behavior_transitions(result)

        # 确保phase_change列存在
        if 'phase_change' not in result.columns:
            result['phase_change'] = "无变化"

        # 分析主力行为强度
        self._analyze_behavior_intensity(data, result)

        # 确保behavior_intensity和behavior_description列存在
        if 'behavior_intensity' not in result.columns:
            result['behavior_intensity'] = 0.0
        if 'behavior_description' not in result.columns:
            result['behavior_description'] = ""
        
        return result

    def _determine_institutional_phase(self, data: pd.DataFrame, result: pd.DataFrame) -> pd.Series:
        """
        确定机构投资阶段

        Args:
            data: 原始数据
            result: 结果数据框

        Returns:
            pd.Series: 机构投资阶段
        """
        phase = pd.Series("观望期", index=data.index)

        if len(data) < 20:
            return phase

        # 获取指标
        concentration = result.get('inst_concentration', pd.Series(0.5, index=data.index))
        profit_ratio = result.get('inst_profit_ratio', pd.Series(0.5, index=data.index))
        activity_score = result.get('inst_activity_score', pd.Series(10.0, index=data.index))

        # 计算价格趋势
        price_ma20 = data['close'].rolling(20).mean()
        price_trend = (data['close'] / price_ma20 - 1).fillna(0)

        # 计算成交量趋势
        volume_ma20 = data['volume'].rolling(20).mean()
        volume_trend = (data['volume'] / volume_ma20 - 1).fillna(0)

        for i in range(20, len(data)):
            conc = concentration.iloc[i]
            profit = profit_ratio.iloc[i]
            activity = activity_score.iloc[i]
            p_trend = price_trend.iloc[i]
            v_trend = volume_trend.iloc[i]

            # 吸筹期：高集中度，低获利盘，高活跃度
            if conc > 0.6 and profit < 0.3 and activity > 15:
                phase.iloc[i] = "吸筹期"
            # 控盘期：高集中度，中等获利盘，中等活跃度
            elif conc > 0.5 and 0.3 <= profit <= 0.6 and activity > 10:
                phase.iloc[i] = "控盘期"
            # 拉升期：中等集中度，中高获利盘，高活跃度，价格上涨
            elif conc > 0.4 and profit > 0.5 and activity > 20 and p_trend > 0.05:
                phase.iloc[i] = "拉升期"
            # 出货期：低集中度，高获利盘，高活跃度
            elif conc < 0.4 and profit > 0.7 and activity > 15:
                phase.iloc[i] = "出货期"
            else:
                phase.iloc[i] = "观望期"

        return phase
    
    def _calculate_behavior_patterns(self, data: pd.DataFrame, result: pd.DataFrame) -> None:
        """
        计算主力行为模式识别指标
        
        Args:
            data: 原始数据
            result: 结果数据框
        """
        # 至少需要20天数据进行有效分析
        if len(data) < 20:
            return
        
        # 提取数据
        volume = data["volume"].values
        close = data["close"].values
        high = data["high"].values
        low = data["low"].values
        
        # 计算相对成交量
        rel_volume = np.zeros_like(volume)
        for i in range(20, len(volume)):
            rel_volume[i] = volume[i] / np.mean(volume[i-20:i])
        
        # 初始化行为模式标识
        behavior_pattern = pd.Series("未知", index=data.index)
        
        # 1. 识别吸筹模式
        # 特征：连续温和放量，价格窄幅波动
        for i in range(20, len(data)):
            # 计算价格波动范围
            price_range = (high[i-5:i+1].max() - low[i-5:i+1].min()) / close[i]
            
            # 计算量能特征
            vol_increase = np.mean(volume[i-5:i+1]) / np.mean(volume[i-10:i-5])
            
            # 价格窄幅波动 + 成交量温和增加
            if price_range < 0.05 and 1.2 < vol_increase < 2.0:
                behavior_pattern.iloc[i] = "温和吸筹"
            
            # 价格在低位区域 + 成交量明显放大
            if price_range < 0.08 and vol_increase > 2.0:
                behavior_pattern.iloc[i] = "强势吸筹"
        
        # 2. 识别洗盘模式
        # 特征：价格短期急跌，成交量放大，随后快速收复
        for i in range(20, len(data)):
            if i < 5:
                continue
                
            # 短期急跌
            short_drop = (high[i-3:i].max() - low[i]) / high[i-3:i].max()
            
            # 快速收复
            recovery = (close[i] - low[i]) / (high[i-3:i].max() - low[i]) if (high[i-3:i].max() - low[i]) > 0 else 0
            
            # 判断洗盘
            if short_drop > 0.05 and recovery > 0.5 and rel_volume[i] > 1.5:
                behavior_pattern.iloc[i] = "洗盘"
        
        # 3. 识别拉升模式
        # 特征：价格快速上涨，成交量明显放大
        for i in range(20, len(data)):
            # 计算短期涨幅
            price_increase = (close[i] - close[i-3]) / close[i-3]
            
            # 判断拉升
            if price_increase > 0.05 and rel_volume[i] > 1.8:
                behavior_pattern.iloc[i] = "拉升"
            
            # 判断加速拉升
            if price_increase > 0.08 and rel_volume[i] > 2.5:
                behavior_pattern.iloc[i] = "加速拉升"
        
        # 4. 识别出货模式
        # 特征：价格创新高，成交量异常放大，但收盘价回落
        for i in range(20, len(data)):
            if i < 5:
                continue
                
            # 计算新高特征
            is_new_high = high[i] > np.max(high[i-10:i])
            
            # 计算回落特征
            close_to_high = (close[i] - low[i]) / (high[i] - low[i]) if (high[i] - low[i]) > 0 else 0.5
            
            # 判断出货
            if is_new_high and rel_volume[i] > 2.0 and close_to_high < 0.5:
                behavior_pattern.iloc[i] = "出货"
            
            # 判断集中出货
            if is_new_high and rel_volume[i] > 3.0 and close_to_high < 0.3:
                behavior_pattern.iloc[i] = "集中出货"
        
        # 添加到结果
        result["behavior_pattern"] = behavior_pattern
    
    def _calculate_behavior_transitions(self, result: pd.DataFrame) -> None:
        """
        计算行为模式变化指标
        
        Args:
            result: 结果数据框
        """
        if "inst_phase" not in result.columns or len(result) < 10:
            return
        
        # 初始化模式转换标识
        phase_change = pd.Series("无变化", index=result.index)
        
        # 记录上一个非"观望期"的阶段
        last_active_phase = None
        last_active_idx = -1
        
        # 遍历检测相变
        for i in range(1, len(result)):
            current_phase = result["inst_phase"].iloc[i]
            
            # 跳过观望期
            if current_phase == "观望期":
                continue
            
            # 记录首次出现的活跃阶段
            if last_active_phase is None:
                last_active_phase = current_phase
                last_active_idx = i
                continue
            
            # 检测阶段变化
            if current_phase != last_active_phase:
                # 记录相变
                if last_active_phase == "吸筹期" and current_phase == "控盘期":
                    phase_change.iloc[i] = "吸筹完成"
                elif last_active_phase == "控盘期" and current_phase == "拉升期":
                    phase_change.iloc[i] = "开始拉升"
                elif last_active_phase == "拉升期" and current_phase == "出货期":
                    phase_change.iloc[i] = "开始出货"
                elif last_active_phase == "出货期" and current_phase == "吸筹期":
                    phase_change.iloc[i] = "新一轮开始"
                else:
                    phase_change.iloc[i] = f"{last_active_phase}→{current_phase}"
                
                # 更新最近活跃阶段
                last_active_phase = current_phase
                last_active_idx = i
        
        # 添加到结果
        result["phase_change"] = phase_change
    
    def _analyze_behavior_intensity(self, data: pd.DataFrame, result: pd.DataFrame) -> None:
        """
        分析主力行为强度
        
        Args:
            data: 原始数据
            result: 结果数据框
        """
        if len(data) < 20 or "inst_activity_score" not in result.columns:
            return
            
        # 初始化行为强度评分
        behavior_intensity = pd.Series(0.0, index=result.index)
        
        # 提取相关数据
        activity_score = result["inst_activity_score"].values
        
        if "behavior_pattern" in result.columns:
            behavior_pattern = result["behavior_pattern"].values
        else:
            behavior_pattern = np.array(["未知"] * len(data))
        
        # 初始化行为描述
        behavior_description = pd.Series("", index=result.index)
        
        # 计算行为强度
        for i in range(20, len(data)):
            # 基础强度评分：来自活跃度
            base_intensity = min(activity_score[i] / 20, 5)  # 最高5分
            
            # 行为模式加权
            pattern_weight = 1.0
            if behavior_pattern[i] in ["强势吸筹", "拉升", "加速拉升", "集中出货"]:
                pattern_weight = 2.0
            elif behavior_pattern[i] in ["温和吸筹", "洗盘", "出货"]:
                pattern_weight = 1.5
            
            # 计算最终强度评分
            behavior_intensity.iloc[i] = base_intensity * pattern_weight
            
            # 生成行为描述
            intensity_level = ""
            if behavior_intensity.iloc[i] < 2:
                intensity_level = "弱"
            elif behavior_intensity.iloc[i] < 5:
                intensity_level = "中"
            elif behavior_intensity.iloc[i] < 8:
                intensity_level = "强"
            else:
                intensity_level = "极强"
            
            # 组合行为描述
            if behavior_pattern[i] != "未知":
                behavior_description.iloc[i] = f"{intensity_level}度{behavior_pattern[i]}"
        
        # 添加到结果
        result["behavior_intensity"] = behavior_intensity
        result["behavior_description"] = behavior_description
    
    def classify_institutional_behavior(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        细化主力行为模式分类
        
        识别并细化主力行为模式，返回详细的行为模式描述和特征
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            List[Dict[str, Any]]: 主力行为模式分类结果列表
        """
        # 计算基础指标
        result = self.calculate(data)
        
        # 行为模式分类结果
        behavior_classifications = []
        
        # 至少需要30天数据进行有效分析
        if len(data) < 30:
            return behavior_classifications
        
        # 仅分析最近的数据
        lookback_days = min(60, len(data))
        recent_data = data.iloc[-lookback_days:]
        recent_result = result.iloc[-lookback_days:]
        
        # 1. 识别主导行为模式
        if "inst_phase" in recent_result.columns:
            # 统计各阶段出现次数
            phase_counts = recent_result["inst_phase"].value_counts()
            dominant_phase = phase_counts.idxmax() if not phase_counts.empty else "未知"
            
            # 添加主导行为模式
            behavior_classifications.append({
                "type": "dominant_phase",
                "value": dominant_phase,
                "confidence": float(phase_counts[dominant_phase] / len(recent_result)) if dominant_phase in phase_counts else 0,
                "description": f"主导行为模式: {dominant_phase}"
            })
        
        # 2. 识别近期转折点
        if "phase_change" in recent_result.columns:
            # 找出非"无变化"的转折点
            transitions = recent_result[recent_result["phase_change"] != "无变化"]
            
            if not transitions.empty:
                latest_transition = transitions.iloc[-1]
                transition_date = latest_transition.name
                transition_type = latest_transition["phase_change"]
                
                # 添加最近转折点
                behavior_classifications.append({
                    "type": "recent_transition",
                    "value": transition_type,
                    "date": transition_date,
                    "days_ago": (recent_data.index[-1] - transition_date).days,
                    "description": f"最近转折点: {transition_type} ({transition_date.strftime('%Y-%m-%d')})"
                })
        
        # 3. 识别主力行为特征
        if "behavior_pattern" in recent_result.columns:
            # 统计各行为模式出现次数
            pattern_counts = recent_result["behavior_pattern"].value_counts()
            
            # 记录主要行为特征
            main_patterns = []
            for pattern, count in pattern_counts.items():
                if pattern != "未知" and count >= 3:  # 至少出现3次才考虑
                    freq = count / len(recent_result)
                    main_patterns.append({
                        "pattern": pattern,
                        "count": int(count),
                        "frequency": float(freq),
                        "description": f"{pattern} (出现{count}次, 频率{freq:.1%})"
                    })
            
            # 添加主要行为特征
            if main_patterns:
                behavior_classifications.append({
                    "type": "main_patterns",
                    "value": main_patterns,
                    "description": f"主要行为特征: {', '.join([p['pattern'] for p in main_patterns])}"
                })
        
        # 4. 识别行为强度趋势
        if "behavior_intensity" in recent_result.columns:
            # 计算行为强度趋势
            intensity_values = recent_result["behavior_intensity"].values
            intensity_trend = "平稳"
            
            if len(intensity_values) >= 10:
                recent_avg = np.mean(intensity_values[-5:])
                previous_avg = np.mean(intensity_values[-10:-5])
                
                if recent_avg > previous_avg * 1.3:
                    intensity_trend = "显著增强"
                elif recent_avg > previous_avg * 1.1:
                    intensity_trend = "略有增强"
                elif recent_avg < previous_avg * 0.7:
                    intensity_trend = "显著减弱"
                elif recent_avg < previous_avg * 0.9:
                    intensity_trend = "略有减弱"
            
            # 添加行为强度趋势
            behavior_classifications.append({
                "type": "intensity_trend",
                "value": intensity_trend,
                "recent_avg": float(np.mean(intensity_values[-5:])) if len(intensity_values) >= 5 else 0,
                "description": f"行为强度趋势: {intensity_trend}"
            })
        
        # 5. 量价配合分析
        volume = recent_data["volume"].values
        close = recent_data["close"].values
        
        # 计算成交量变化率和价格变化率
        vol_change = []
        price_change = []
        
        for i in range(5, len(recent_data)):
            vol_ratio = volume[i] / np.mean(volume[i-5:i])
            price_ratio = close[i] / close[i-5] - 1
            vol_change.append(vol_ratio)
            price_change.append(price_ratio)
        
        if vol_change and price_change:
            # 计算量价相关性
            correlation = np.corrcoef(vol_change, price_change)[0, 1]
            
            vol_price_relation = "未知"
            if correlation > 0.6:
                vol_price_relation = "高度正相关"
            elif correlation > 0.3:
                vol_price_relation = "正相关"
            elif correlation > -0.3:
                vol_price_relation = "弱相关"
            elif correlation > -0.6:
                vol_price_relation = "负相关"
            else:
                vol_price_relation = "高度负相关"
            
            # 添加量价配合分析
            behavior_classifications.append({
                "type": "volume_price_relation",
                "value": vol_price_relation,
                "correlation": float(correlation),
                "description": f"量价配合: {vol_price_relation} (相关系数: {correlation:.2f})"
            })
        
        return behavior_classifications
    
    def predict_absorption_completion(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        预测吸筹完成时间窗口
        
        基于主力行为模式和历史数据，预测吸筹完成的时间窗口
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            Dict[str, Any]: 吸筹完成时间预测结果
        """
        # 初始化结果
        prediction = {
            "is_in_absorption": False,
            "completion_days_min": None,
            "completion_days_max": None,
            "completion_date_min": None,
            "completion_date_max": None,
            "confidence": 0.0,
            "volume_needed": None,
            "price_range_min": None,
            "price_range_max": None,
            "description": "无法预测"
        }
        
        # 计算基础指标
        result = self.calculate(data)
        
        # 需要足够的历史数据
        if len(data) < 60:
            prediction["description"] = "历史数据不足，无法进行预测"
            return prediction
        
        # 检查是否处于吸筹阶段
        if "inst_phase" in result.columns:
            last_phase = result["inst_phase"].iloc[-1]
            
            if last_phase != "吸筹期":
                prediction["description"] = f"当前处于{last_phase}，非吸筹阶段"
                return prediction
            
            # 确认处于吸筹阶段
            prediction["is_in_absorption"] = True
            
            # 获取最近的吸筹周期数据
            absorption_period = 0
            for i in range(len(result)-1, -1, -1):
                if result["inst_phase"].iloc[i] == "吸筹期":
                    absorption_period += 1
                else:
                    break
            
            # 如果吸筹时间太短，可能刚开始
            if absorption_period < 5:
                prediction["description"] = "吸筹刚开始，数据不足以做出准确预测"
                prediction["completion_days_min"] = 20
                prediction["completion_days_max"] = 60
                prediction["confidence"] = 0.3
            else:
                # 分析历史吸筹周期
                historical_cycles = self._analyze_historical_absorption_cycles(data, result)
                
                if not historical_cycles:
                    # 无历史参考，使用一般规则估计
                    prediction["description"] = "无历史吸筹周期数据，使用一般规则估计"
                    prediction["completion_days_min"] = max(5, int(20 - absorption_period * 0.5))
                    prediction["completion_days_max"] = max(15, int(60 - absorption_period * 0.7))
                    prediction["confidence"] = 0.4
                else:
                    # 基于历史周期估计
                    avg_duration = np.mean([cycle["duration"] for cycle in historical_cycles])
                    min_duration = np.min([cycle["duration"] for cycle in historical_cycles])
                    max_duration = np.max([cycle["duration"] for cycle in historical_cycles])
                    
                    remaining_min = max(3, int(min_duration - absorption_period))
                    remaining_max = max(10, int(max_duration - absorption_period))
                    
                    prediction["completion_days_min"] = remaining_min
                    prediction["completion_days_max"] = remaining_max
                    prediction["confidence"] = min(0.7, 0.4 + len(historical_cycles) * 0.1)
                    prediction["description"] = f"基于{len(historical_cycles)}个历史吸筹周期估计"
            
            # 估计所需成交量
            if "inst_concentration" in result.columns:
                current_concentration = result["inst_concentration"].iloc[-1]
                target_concentration = 0.3  # 控盘阶段的典型集中度
                
                if current_concentration < target_concentration:
                    # 估算达到目标集中度所需的交易量
                    daily_volume_avg = data["volume"].iloc[-20:].mean()
                    volume_needed = daily_volume_avg * prediction["completion_days_min"] * 1.2
                    prediction["volume_needed"] = float(volume_needed)
            
            # 估计价格范围
            current_price = data["close"].iloc[-1]
            price_range_min = current_price * 0.95
            price_range_max = current_price * 1.1
            
            prediction["price_range_min"] = float(price_range_min)
            prediction["price_range_max"] = float(price_range_max)
            
            # 计算预计完成日期
            last_date = data.index[-1]
            if prediction["completion_days_min"] is not None:
                import datetime
                prediction["completion_date_min"] = (last_date + datetime.timedelta(days=prediction["completion_days_min"])).strftime('%Y-%m-%d')
                prediction["completion_date_max"] = (last_date + datetime.timedelta(days=prediction["completion_days_max"])).strftime('%Y-%m-%d')
        
        return prediction
    
    def generate_trading_signals(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """
        生成交易信号
        
        Args:
            data: 输入数据
            **kwargs: 额外参数
            
        Returns:
            Dict[str, pd.Series]: 包含交易信号的字典
        """
        # 确保已计算指标
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        # 初始化信号
        signals = {}
        signals['buy_signal'] = pd.Series(False, index=data.index)
        signals['sell_signal'] = pd.Series(False, index=data.index)
        signals['signal_strength'] = pd.Series(0, index=data.index)
    
        # 在这里实现指标特定的信号生成逻辑
        # 此处提供默认实现
    
        return signals
        
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算指标原始评分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 评分(0-100)
        """
        # 确保已计算指标
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return pd.Series(50.0, index=data.index)
        
        # 初始化评分
        score = pd.Series(50.0, index=data.index)
    
        # 在这里实现指标特定的评分逻辑
        # 此处提供默认实现
    
        return score
    
    def analyze_historical_absorption_cycles(self, data: pd.DataFrame, result: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        分析历史吸筹周期
        
        Args:
            data: 原始数据
            result: 结果数据框
            
        Returns:
            List[Dict[str, Any]]: 历史吸筹周期列表
        """
        cycles = []
        
        if "inst_phase" not in result.columns:
            return cycles
        
        # 找出所有完整的吸筹周期
        in_absorption = False
        start_idx = -1
        
        for i in range(len(result)):
            phase = result["inst_phase"].iloc[i]
            
            if not in_absorption and phase == "吸筹期":
                # 开始新的吸筹周期
                in_absorption = True
                start_idx = i
            elif in_absorption and phase != "吸筹期":
                # 吸筹周期结束
                if start_idx >= 0 and phase == "控盘期":  # 确保是完整的吸筹→控盘转变
                    duration = i - start_idx
                    
                    # 记录完整的吸筹周期
                    if duration >= 5:  # 忽略太短的周期
                        cycle_data = {
                            "start_idx": start_idx,
                            "end_idx": i - 1,
                            "duration": duration,
                            "start_date": data.index[start_idx],
                            "end_date": data.index[i-1],
                            "avg_volume": data["volume"].iloc[start_idx:i].mean(),
                            "price_change": (data["close"].iloc[i-1] / data["close"].iloc[start_idx]) - 1
                        }
                        cycles.append(cycle_data)
                
                # 重置状态
                in_absorption = False
                start_idx = -1
        
        return cycles

    def calculate_confidence(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """
        计算InstitutionalBehavior指标的置信度

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
        if last_score > 80 or last_score < 20:
            confidence += 0.25
        # 中性评分置信度中等
        elif 40 <= last_score <= 60:
            confidence += 0.1
        else:
            confidence += 0.15

        # 2. 基于数据质量的置信度
        if hasattr(self, '_result') and self._result is not None:
            # 检查是否有机构阶段数据
            if 'inst_phase' in self._result.columns:
                phase_values = self._result['inst_phase'].dropna()
                if len(phase_values) > 0:
                    # 非观望期的比例越高，置信度越高
                    active_ratio = len(phase_values[phase_values != "观望期"]) / len(phase_values)
                    confidence += active_ratio * 0.1

            # 检查行为模式数据质量
            if 'behavior_pattern' in self._result.columns:
                pattern_values = self._result['behavior_pattern'].dropna()
                if len(pattern_values) > 0:
                    # 明确行为模式的比例
                    clear_pattern_ratio = len(pattern_values[pattern_values != "未知"]) / len(pattern_values)
                    confidence += clear_pattern_ratio * 0.1

        # 3. 基于形态的置信度
        if not patterns.empty:
            # 检查InstitutionalBehavior形态
            pattern_count = patterns.sum().sum()
            if pattern_count > 0:
                confidence += min(pattern_count * 0.02, 0.15)

        # 4. 基于信号的置信度
        if signals:
            # 检查信号强度
            signal_count = sum(1 for signal in signals.values() if hasattr(signal, 'any') and signal.any())
            if signal_count > 0:
                confidence += min(signal_count * 0.05, 0.1)

        # 5. 基于数据长度的置信度
        if len(score) >= 60:
            confidence += 0.1  # 长期数据更可靠
        elif len(score) >= 30:
            confidence += 0.05

        # 确保置信度在0-1范围内
        return max(0.0, min(1.0, confidence))

    def register_patterns(self):
        """
        注册InstitutionalBehavior指标的形态到全局形态注册表
        """
        # 注册机构阶段形态
        self.register_pattern_to_registry(
            pattern_id="INST_ABSORPTION_PHASE",
            display_name="机构吸筹期",
            description="机构处于吸筹阶段，通常是买入良机",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=20.0
        )

        self.register_pattern_to_registry(
            pattern_id="INST_CONTROL_PHASE",
            display_name="机构控盘期",
            description="机构处于控盘阶段，价格相对稳定",
            pattern_type="NEUTRAL",
            default_strength="MEDIUM",
            score_impact=5.0
        )

        self.register_pattern_to_registry(
            pattern_id="INST_RALLY_PHASE",
            display_name="机构拉升期",
            description="机构处于拉升阶段，价格快速上涨",
            pattern_type="BULLISH",
            default_strength="VERY_STRONG",
            score_impact=30.0
        )

        self.register_pattern_to_registry(
            pattern_id="INST_DISTRIBUTION_PHASE",
            display_name="机构出货期",
            description="机构处于出货阶段，存在下跌风险",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-25.0
        )

        # 注册行为模式形态
        self.register_pattern_to_registry(
            pattern_id="INST_STRONG_ABSORPTION",
            display_name="机构强势吸筹",
            description="机构强势吸筹，买入信号强烈",
            pattern_type="BULLISH",
            default_strength="VERY_STRONG",
            score_impact=25.0
        )

        self.register_pattern_to_registry(
            pattern_id="INST_WASHOUT",
            display_name="机构洗盘",
            description="机构洗盘操作，短期调整后可能继续上涨",
            pattern_type="NEUTRAL",
            default_strength="MEDIUM",
            score_impact=0.0
        )

        self.register_pattern_to_registry(
            pattern_id="INST_ACCELERATED_RALLY",
            display_name="机构加速拉升",
            description="机构加速拉升，但需警惕顶部风险",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=20.0
        )

        self.register_pattern_to_registry(
            pattern_id="INST_CONCENTRATED_DISTRIBUTION",
            display_name="机构集中出货",
            description="机构集中出货，卖出信号强烈",
            pattern_type="BEARISH",
            default_strength="VERY_STRONG",
            score_impact=-30.0
        )

        # 注册转换形态
        self.register_pattern_to_registry(
            pattern_id="INST_ABSORPTION_COMPLETE",
            display_name="吸筹完成",
            description="机构吸筹完成，即将进入拉升阶段",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=25.0
        )

        self.register_pattern_to_registry(
            pattern_id="INST_RALLY_START",
            display_name="开始拉升",
            description="机构开始拉升，买入时机",
            pattern_type="BULLISH",
            default_strength="VERY_STRONG",
            score_impact=30.0
        )

        self.register_pattern_to_registry(
            pattern_id="INST_DISTRIBUTION_START",
            display_name="开始出货",
            description="机构开始出货，卖出时机",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-25.0
        )

        # 注册活动强度形态
        self.register_pattern_to_registry(
            pattern_id="INST_EXTREME_ACTIVITY",
            display_name="机构极强活动",
            description="机构活动极其活跃，重要变化即将发生",
            pattern_type="NEUTRAL",
            default_strength="STRONG",
            score_impact=15.0
        )

    def generate_trading_signals(self, data: pd.DataFrame, **kwargs) -> dict:
        """
        生成InstitutionalBehavior交易信号

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

        # 初始化信号
        buy_signal = pd.Series(False, index=data.index)
        sell_signal = pd.Series(False, index=data.index)
        signal_strength = pd.Series(0.0, index=data.index)

        # 基于机构阶段生成信号
        if 'inst_phase' in self._result.columns:
            phase = self._result['inst_phase']

            # 买入信号
            buy_signal |= (phase == "吸筹期")
            buy_signal |= (phase == "控盘期")

            # 卖出信号
            sell_signal |= (phase == "出货期")

        # 基于阶段转换生成信号
        if 'phase_change' in self._result.columns:
            phase_change = self._result['phase_change']

            # 强买入信号
            strong_buy = (phase_change == "吸筹完成") | (phase_change == "开始拉升")
            buy_signal |= strong_buy
            signal_strength[strong_buy] = 0.9

            # 强卖出信号
            strong_sell = (phase_change == "开始出货")
            sell_signal |= strong_sell
            signal_strength[strong_sell] = -0.9

        # 基于行为模式调整信号强度
        if 'behavior_pattern' in self._result.columns:
            behavior = self._result['behavior_pattern']

            # 增强买入信号
            signal_strength[(behavior == "强势吸筹") & buy_signal] = 0.8
            signal_strength[(behavior == "温和吸筹") & buy_signal] = 0.6

            # 增强卖出信号
            signal_strength[(behavior == "集中出货") & sell_signal] = -0.8
            signal_strength[(behavior == "出货") & sell_signal] = -0.6

        # 基于行为强度调整信号
        if 'behavior_intensity' in self._result.columns:
            intensity = self._result['behavior_intensity']

            # 高强度时增强信号
            high_intensity = intensity > 5
            signal_strength[high_intensity & buy_signal] *= 1.2
            signal_strength[high_intensity & sell_signal] *= 1.2

        # 标准化信号强度
        signal_strength = signal_strength.clip(-1, 1)

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
        return "INSTITUTIONALBEHAVIOR"