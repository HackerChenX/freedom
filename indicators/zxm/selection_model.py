"""
ZXM选股模型模块

整合多个指标的选股系统
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple

from indicators.base_indicator import BaseIndicator
from utils.logger import get_logger
from utils.decorators import log_calls, error_handling

from indicators.zxm.trend_indicators import TrendDetector
from indicators.zxm.elasticity_indicators import Elasticity, AmplitudeElasticity, BounceDetector
from indicators.zxm.buy_point_indicators import BuyPointDetector
from indicators.zxm.score_indicators import StockScoreCalculator
from indicators.zxm_washplate import ZXMWashPlate

logger = get_logger(__name__)


class SelectionModel(BaseIndicator):
    """
    ZXM选股模型
    
    整合多个指标的选股系统，综合评估股票买入价值
    """
    
    def __init__(self):
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        """初始化ZXM选股模型"""
        super().__init__(name="SelectionModel", description="ZXM选股模型，整合多个指标的选股系统")
        
        # 初始化各子指标
        self.trend_detector = TrendDetector()
        self.elasticity_indicator = Elasticity()
        self.amplitude_elasticity_indicator = AmplitudeElasticity()
        self.bounce_detector = BounceDetector()
        self.score_calculator = StockScoreCalculator()
        self.buy_point_detector = BuyPointDetector()
        self.wash_plate_detector = ZXMWashPlate()
    
    def _calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算ZXM选股模型
        
        Args:
            data: 输入数据，包含OHLCV数据
            
        Returns:
            pd.DataFrame: 计算结果，包含选股得分和信号
        """
        # 确保数据包含必需的列
        required_cols = ["open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"数据缺少必需的列: {missing_cols}")
        
        # 初始化结果数据框
        result = data.copy()
        
        try:
            # 1. 计算趋势指标
            trend_result = self.trend_detector.calculate(data)
            # TrendDetector返回的是TrendState，不是TrendDirection
            result.loc[:, "TrendDirection"] = trend_result["TrendState"]
            
            # 2. 计算弹性指标
            elasticity_result = self.elasticity_indicator.calculate(data)
            result.loc[:, "ElasticityRatio"] = elasticity_result["ElasticityRatio"]
            result.loc[:, "BounceStrength"] = elasticity_result["BounceStrength"]
            result.loc[:, "ElasticityBuySignal"] = elasticity_result["BuySignal"]
            
            # 3. 计算幅度弹性指标
            amplitude_elasticity_result = self.amplitude_elasticity_indicator.calculate(data)
            # AmplitudeElasticity指标返回的是"Amplitude"列，不是"AmplitudeElasticityRatio"
            if "Amplitude" in amplitude_elasticity_result.columns:
                result.loc[:, "AmplitudeElasticityRatio"] = amplitude_elasticity_result["Amplitude"]
            else:
                # 如果没有Amplitude列，使用默认值
                result.loc[:, "AmplitudeElasticityRatio"] = 0.0
            
            # 4. 计算反弹检测指标
            bounce_result = self.bounce_detector.calculate(data)
            result.loc[:, "BounceSignal"] = bounce_result["BounceSignal"]
            result.loc[:, "PullbackBuyPoint"] = bounce_result["PullbackBuyPoint"]
            
            # 5. 计算综合评分
            score_result = self.score_calculator.calculate(data)
            result.loc[:, "TrendScore"] = score_result["TrendScore"]
            result.loc[:, "MomentumScore"] = score_result["MomentumScore"]
            result.loc[:, "VolatilityScore"] = score_result["VolatilityScore"]
            result.loc[:, "VolumeScore"] = score_result["VolumeScore"]

            # ValueScore可能不存在，需要检查
            if "ValueScore" in score_result.columns:
                result.loc[:, "ValueScore"] = score_result["ValueScore"]
            else:
                result.loc[:, "ValueScore"] = 50.0  # 默认中性评分

            result.loc[:, "FinalScore"] = score_result["FinalScore"]
            
            # 6. 计算买点信号
            buypoint_result = self.buy_point_detector.calculate(data)
            result.loc[:, "VolumeRiseBuyPoint"] = buypoint_result["VolumeRiseBuyPoint"]
            result.loc[:, "PullbackStabilizeBuyPoint"] = buypoint_result["PullbackStabilizeBuyPoint"]
            result.loc[:, "BreakoutBuyPoint"] = buypoint_result["BreakoutBuyPoint"]
            result.loc[:, "BottomVolumeBuyPoint"] = buypoint_result["BottomVolumeBuyPoint"]
            result.loc[:, "VolumeShrinkBuyPoint"] = buypoint_result["VolumeShrinkBuyPoint"]
            result.loc[:, "AnyBuyPoint"] = (
                buypoint_result["VolumeRiseBuyPoint"] | 
                buypoint_result["PullbackStabilizeBuyPoint"] | 
                buypoint_result["BreakoutBuyPoint"] | 
                buypoint_result["BottomVolumeBuyPoint"] | 
                buypoint_result["VolumeShrinkBuyPoint"]
            )
            
            # 7. 计算洗盘信号
            try:
                washplate_result = self.wash_plate_detector.calculate(data)

                # 使用正确的列名（WashPlateType枚举值）
                result.loc[:, "ShockWash"] = washplate_result.get("横盘震荡洗盘", pd.Series(False, index=data.index))
                result.loc[:, "PullbackWash"] = washplate_result.get("回调洗盘", pd.Series(False, index=data.index))
                result.loc[:, "FalseBreakWash"] = washplate_result.get("假突破洗盘", pd.Series(False, index=data.index))
                result.loc[:, "TimeWash"] = washplate_result.get("时间洗盘", pd.Series(False, index=data.index))
                result.loc[:, "ContinuousYinWash"] = washplate_result.get("连续阴线洗盘", pd.Series(False, index=data.index))

                result.loc[:, "AnyWashPlate"] = (
                    result["ShockWash"] |
                    result["PullbackWash"] |
                    result["FalseBreakWash"] |
                    result["TimeWash"] |
                    result["ContinuousYinWash"]
                )
            except Exception as e:
                logger.error(f"洗盘指标计算错误: {e}")
                result.loc[:, "ShockWash"] = pd.Series(False, index=data.index)
                result.loc[:, "PullbackWash"] = pd.Series(False, index=data.index)
                result.loc[:, "FalseBreakWash"] = pd.Series(False, index=data.index)
                result.loc[:, "TimeWash"] = pd.Series(False, index=data.index)
                result.loc[:, "ContinuousYinWash"] = pd.Series(False, index=data.index)
                result.loc[:, "AnyWashPlate"] = pd.Series(False, index=data.index)
            
            # 8. 计算整合选股信号
            # a. 强趋势上涨股
            strong_uptrend = (result["TrendDirection"] == 1)
            
            # b. 放量突破股
            volume_breakout = result["BreakoutBuyPoint"] & (result["VolumeScore"] >= 70)
            
            # c. 回调买点股
            pullback_buy = result["PullbackBuyPoint"] | result["PullbackStabilizeBuyPoint"]
            
            # d. 洗盘后启动股
            washplate_start = result["AnyWashPlate"] & (result["TrendScore"] >= 60) & (result["VolumeScore"] >= 60)
            
            # e. 低吸高弹性股
            low_buy_elastic = (result["ElasticityBuySignal"] | result["BottomVolumeBuyPoint"]) & (result["BounceStrength"] >= 50)
            
            # 整合选股结果
            result.loc[:, "StrongUptrendSelect"] = strong_uptrend
            result.loc[:, "VolumeBreakoutSelect"] = volume_breakout
            result.loc[:, "PullbackBuySelect"] = pullback_buy
            result.loc[:, "WashplateStartSelect"] = washplate_start
            result.loc[:, "LowBuyElasticSelect"] = low_buy_elastic
            
            # 综合选股信号 - 任意一种选股条件满足
            result.loc[:, "FinalSelect"] = (
                strong_uptrend | 
                volume_breakout | 
                pullback_buy | 
                washplate_start | 
                low_buy_elastic
            )
            
            # 9. 计算选股综合得分
            # 将各维度得分加权平均
            selection_score = np.zeros(len(data))
            
            for i in range(len(data)):
                # 基础分值 - 使用综合评分系统的结果
                base_score = result["FinalScore"].iloc[i]
                
                # 买点加分
                if result["AnyBuyPoint"].iloc[i]:
                    if result["BreakoutBuyPoint"].iloc[i]:
                        buy_point_bonus = 15  # 突破买点价值最高
                    elif result["PullbackStabilizeBuyPoint"].iloc[i]:
                        buy_point_bonus = 12  # 回调企稳买点次之
                    else:
                        buy_point_bonus = 10  # 其他买点
                else:
                    buy_point_bonus = 0
                
                # 洗盘加分
                if result["AnyWashPlate"].iloc[i]:
                    washplate_bonus = 8
                else:
                    washplate_bonus = 0
                
                # 趋势强度加减分
                if result["TrendDirection"].iloc[i] == 1:
                    trend_bonus = 15
                elif result["TrendDirection"].iloc[i] == -1:
                    trend_bonus = -15
                else:
                    trend_bonus = 0
                
                # 弹性因素加分
                if result["ElasticityBuySignal"].iloc[i]:
                    elasticity_bonus = 10
                elif result["BounceSignal"].iloc[i]:
                    elasticity_bonus = 7
                else:
                    elasticity_bonus = 0
                
                # 综合计算
                selection_score[i] = min(100, base_score + buy_point_bonus + washplate_bonus + trend_bonus + elasticity_bonus)
            
            result.loc[:, "SelectionScore"] = selection_score
            
            # 10. 计算买入优先级
            # 将选中的股票按优先级排序（1-5，1为最高）
            priority = np.zeros(len(data))
            
            for i in range(len(data)):
                if not result["FinalSelect"].iloc[i]:
                    priority[i] = 0  # 未选中
                    continue
                
                # 计算基础优先级
                if result["StrongUptrendSelect"].iloc[i] and result["SelectionScore"].iloc[i] >= 85:
                    priority[i] = 1  # 最高优先级
                elif result["VolumeBreakoutSelect"].iloc[i] or (result["PullbackBuySelect"].iloc[i] and result["TrendScore"].iloc[i] >= 70):
                    priority[i] = 2  # 次高优先级
                elif result["WashplateStartSelect"].iloc[i]:
                    priority[i] = 3  # 中等优先级
                elif result["PullbackBuySelect"].iloc[i]:
                    priority[i] = 4  # 次低优先级
                elif result["LowBuyElasticSelect"].iloc[i]:
                    priority[i] = 5  # 最低优先级
                
                # 调整优先级
                # 如果综合得分特别高，提升优先级
                if result["SelectionScore"].iloc[i] >= 90 and priority[i] > 1:
                    priority[i] -= 1
            
            result.loc[:, "BuyPriority"] = priority
        
        except Exception as e:
            logger.error(f"选股模型计算错误: {e}")
            # 设置默认值，确保结果不为空
            result.loc[:, "FinalSelect"] = pd.Series(False, index=data.index)
            result.loc[:, "SelectionScore"] = pd.Series(50, index=data.index)
            result.loc[:, "BuyPriority"] = pd.Series(0, index=data.index)
        
        return result
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算ZXM选股模型的原始评分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 评分结果，0-100分
        """
        # 计算选股模型
        result = self.calculate(data)
        
        # 直接使用选股得分作为原始评分
        return result["SelectionScore"]
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别ZXM选股模型相关的技术形态
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[str]: 识别的形态列表
        """
        # 计算选股模型
        result = self.calculate(data)
        
        # 只关注最后一个交易日的形态
        patterns = []
        if len(result) > 0:
            last_row = result.iloc[-1]
            
            # 基础选股信号
            if last_row["FinalSelect"]:
                patterns.append("选股系统买入信号")
                
                # 选股优先级
                priority = last_row["BuyPriority"]
                if priority == 1:
                    patterns.append("最高优先级选股")
                elif priority == 2:
                    patterns.append("高优先级选股")
                elif priority == 3:
                    patterns.append("中等优先级选股")
                elif priority == 4:
                    patterns.append("低优先级选股")
                elif priority == 5:
                    patterns.append("最低优先级选股")
                
                # 选股类型
                if last_row["StrongUptrendSelect"]:
                    patterns.append("强趋势上涨股")
                if last_row["VolumeBreakoutSelect"]:
                    patterns.append("放量突破股")
                if last_row["PullbackBuySelect"]:
                    patterns.append("回调买点股")
                if last_row["WashplateStartSelect"]:
                    patterns.append("洗盘后启动股")
                if last_row["LowBuyElasticSelect"]:
                    patterns.append("低吸高弹性股")
            
            # 详细技术形态
            # 趋势状态
            if "TrendDirection" in last_row.index:
                if last_row["TrendDirection"] == 1:
                    patterns.append("超强上升趋势")
                elif last_row["TrendDirection"] == -1:
                    patterns.append("下降趋势")
                else:
                    patterns.append("震荡趋势")
            elif "TrendState" in last_row.index:
                if last_row["TrendState"] == 1:
                    patterns.append("上升趋势")
                elif last_row["TrendState"] == -1:
                    patterns.append("下降趋势")
                else:
                    patterns.append("震荡趋势")
            else:
                patterns.append("趋势未知")
            
            # 买点信号
            if "AnyBuyPoint" in last_row.index and last_row["AnyBuyPoint"]:
                if "VolumeRiseBuyPoint" in last_row.index and last_row["VolumeRiseBuyPoint"]:
                    patterns.append("放量上涨买点")
                if "PullbackStabilizeBuyPoint" in last_row.index and last_row["PullbackStabilizeBuyPoint"]:
                    patterns.append("回调企稳买点")
                if "BreakoutBuyPoint" in last_row.index and last_row["BreakoutBuyPoint"]:
                    patterns.append("突破买点")
                if "BottomVolumeBuyPoint" in last_row.index and last_row["BottomVolumeBuyPoint"]:
                    patterns.append("底部放量买点")
                if "VolumeShrinkBuyPoint" in last_row.index and last_row["VolumeShrinkBuyPoint"]:
                    patterns.append("缩量整理买点")
            
            # 洗盘信号
            if "AnyWashPlate" in last_row.index and last_row["AnyWashPlate"]:
                if "ShockWash" in last_row.index and last_row["ShockWash"]:
                    patterns.append("横盘震荡洗盘")
                if "PullbackWash" in last_row.index and last_row["PullbackWash"]:
                    patterns.append("回调洗盘")
                if "FalseBreakWash" in last_row.index and last_row["FalseBreakWash"]:
                    patterns.append("假突破洗盘")
                if "TimeWash" in last_row.index and last_row["TimeWash"]:
                    patterns.append("时间洗盘")
                if "ContinuousYinWash" in last_row.index and last_row["ContinuousYinWash"]:
                    patterns.append("连续阴线洗盘")
        
        return patterns
    
    def generate_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        生成标准化的信号输出
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.DataFrame: 包含标准化信号的DataFrame
        """
        # 计算选股模型和评分
        result = self.calculate(data)
        score = self.calculate_raw_score(data, **kwargs)
        
        # 初始化信号DataFrame
        signals = pd.DataFrame(index=data.index)
        
        # 设置买卖信号
        signals.loc[:, 'buy_signal'] = result["FinalSelect"]
        signals.loc[:, 'sell_signal'] = False  # 选股模型不产生卖出信号
        signals.loc[:, 'neutral_signal'] = ~result["FinalSelect"]
        
        # 设置趋势
        if "TrendDirection" in result.columns:
            signals.loc[:, 'trend'] = result["TrendDirection"]
        elif "TrendState" in result.columns:
            signals.loc[:, 'trend'] = result["TrendState"]
        else:
            signals.loc[:, 'trend'] = 0  # 默认中性
        
        # 设置评分
        signals.loc[:, 'score'] = score
        
        # 设置信号类型
        signals.loc[:, 'signal_type'] = 'neutral'
        
        # 根据选股类型设置信号类型
        for i in signals.index:
            if result.loc[i, "StrongUptrendSelect"]:
                signals.loc[i, 'signal_type'] = 'strong_uptrend'
            elif result.loc[i, "VolumeBreakoutSelect"]:
                signals.loc[i, 'signal_type'] = 'volume_breakout'
            elif result.loc[i, "PullbackBuySelect"]:
                signals.loc[i, 'signal_type'] = 'pullback_buy'
            elif result.loc[i, "WashplateStartSelect"]:
                signals.loc[i, 'signal_type'] = 'washplate_start'
            elif result.loc[i, "LowBuyElasticSelect"]:
                signals.loc[i, 'signal_type'] = 'low_buy_elastic'
        
        # 设置信号描述
        signals.loc[:, 'signal_desc'] = ''
        
        # 为每个信号设置详细描述
        for i in signals.index:
            if result.loc[i, "FinalSelect"]:
                # 获取选股类型
                select_types = []
                
                if result.loc[i, "StrongUptrendSelect"]:
                    select_types.append("强趋势上涨")
                if result.loc[i, "VolumeBreakoutSelect"]:
                    select_types.append("放量突破")
                if result.loc[i, "PullbackBuySelect"]:
                    select_types.append("回调买点")
                if result.loc[i, "WashplateStartSelect"]:
                    select_types.append("洗盘后启动")
                if result.loc[i, "LowBuyElasticSelect"]:
                    select_types.append("低吸高弹性")
                
                # 组合描述
                type_desc = "、".join(select_types)
                score_val = score.iloc[i]
                priority = result.loc[i, "BuyPriority"]
                
                signals.loc[i, 'signal_desc'] = f"选股信号：{type_desc}，得分{score_val:.1f}，优先级{priority:.0f}"
        
        # 置信度设置
        signals.loc[:, 'confidence'] = 60  # 基础置信度
        
        # 根据选股得分和优先级调整置信度
        for i in signals.index:
            if result.loc[i, "FinalSelect"]:
                # 根据选股得分调整（最多±20）
                score_adj = (score.iloc[i] - 70) / 30 * 20
                
                # 根据优先级调整（最多±10）
                priority = result.loc[i, "BuyPriority"]
                priority_adj = (6 - priority) * 2.5
                
                signals.loc[i, 'confidence'] = min(95, max(50, 60 + score_adj + priority_adj))
        
        # 风险等级
        signals.loc[:, 'risk_level'] = '中'  # 默认中等风险
        
        # 高优先级选股风险较低
        signals.loc[result["BuyPriority"] <= 2, 'risk_level'] = '低'
        
        # 低优先级选股风险较高
        signals.loc[result["BuyPriority"] >= 4, 'risk_level'] = '高'
        
        # 建议仓位
        signals.loc[:, 'position_size'] = 0.0
        
        # 根据优先级和得分设置仓位
        for i in signals.index[signals['buy_signal']]:
            priority = result.loc[i, "BuyPriority"]
            score_val = score.iloc[i]
            
            # 基础仓位 - 根据优先级
            if priority == 1:
                base_position = 0.5
            elif priority == 2:
                base_position = 0.4
            elif priority == 3:
                base_position = 0.3
            elif priority == 4:
                base_position = 0.2
            else:
                base_position = 0.1
            
            # 得分调整 - 高分加仓
            if score_val >= 90:
                score_adj = 0.2
            elif score_val >= 80:
                score_adj = 0.1
            elif score_val >= 70:
                score_adj = 0.05
            else:
                score_adj = 0
            
            signals.loc[i, 'position_size'] = min(0.7, base_position + score_adj)
        
        # 止损位
        signals.loc[:, 'stop_loss'] = 0.0
        
        # 根据选股类型设置不同的止损策略
        for i in signals.index[signals['buy_signal']]:
            try:
                idx = data.index.get_loc(i)
                
                if result.loc[i, "StrongUptrendSelect"]:
                    # 强趋势上涨 - 使用20日均线作为止损
                    ma20 = data["close"].rolling(window=20).mean().iloc[idx]
                    signals.loc[i, 'stop_loss'] = ma20 * 0.95
                
                elif result.loc[i, "VolumeBreakoutSelect"]:
                    # 放量突破 - 使用突破点位作为止损
                    # 简单以当前价格的5%作为止损
                    signals.loc[i, 'stop_loss'] = data["close"].iloc[idx] * 0.95
                
                elif result.loc[i, "PullbackBuySelect"]:
                    # 回调买点 - 使用回调低点作为止损
                    if idx >= 10:
                        low_price = data["low"].iloc[idx-10:idx+1].min()
                        signals.loc[i, 'stop_loss'] = low_price * 0.97
                
                elif result.loc[i, "WashplateStartSelect"] or result.loc[i, "LowBuyElasticSelect"]:
                    # 洗盘后启动或低吸高弹性 - 使用近期低点作为止损
                    if idx >= 20:
                        low_price = data["low"].iloc[idx-20:idx+1].min()
                        signals.loc[i, 'stop_loss'] = low_price * 0.97
                
                else:
                    # 默认止损策略 - 当前价格的7%止损
                    signals.loc[i, 'stop_loss'] = data["close"].iloc[idx] * 0.93
            
            except Exception as e:
                logger.error(f"计算止损位错误: {e}")
                continue
        
        # 市场环境
        signals.loc[:, 'market_env'] = 'normal'
        if "TrendDirection" in result.columns:
            signals.loc[result["TrendDirection"] == 1, 'market_env'] = 'bull_market'
            signals.loc[result["TrendDirection"] == -1, 'market_env'] = 'bear_market'
            signals.loc[result["TrendDirection"] == 0, 'market_env'] = 'sideways_market'
        elif "TrendState" in result.columns:
            signals.loc[result["TrendState"] == 1, 'market_env'] = 'bull_market'
            signals.loc[result["TrendState"] == -1, 'market_env'] = 'bear_market'
            signals.loc[result["TrendState"] == 0, 'market_env'] = 'sideways_market'
        
        # 成交量确认
        signals.loc[:, 'volume_confirmation'] = True
        
        return signals

    def calculate_confidence(self, score: pd.Series, patterns: List[str], signals: Dict[str, pd.Series]) -> float:
        """
        计算置信度

        Args:
            score: 评分序列
            patterns: 形态列表
            signals: 信号字典

        Returns:
            float: 置信度值，0-1之间
        """
        if score.empty:
            return 0.5

        latest_score = score.iloc[-1]

        # 基础置信度基于评分
        base_confidence = min(0.9, max(0.1, latest_score / 100))

        # 根据形态调整置信度
        pattern_boost = 0.0
        if "选股系统买入信号" in patterns:
            pattern_boost += 0.15
        if "最高优先级选股" in patterns:
            pattern_boost += 0.25
        elif "高优先级选股" in patterns:
            pattern_boost += 0.2
        elif "中等优先级选股" in patterns:
            pattern_boost += 0.15

        # 选股类型调整
        if "强趋势上涨股" in patterns:
            pattern_boost += 0.15
        elif "放量突破股" in patterns:
            pattern_boost += 0.15
        elif "洗盘后启动股" in patterns:
            pattern_boost += 0.1

        # 技术形态调整
        if "超强上升趋势" in patterns:
            pattern_boost += 0.1
        elif "强上升趋势" in patterns:
            pattern_boost += 0.08

        # 买点信号调整
        if "突破买点" in patterns:
            pattern_boost += 0.1
        elif "回调企稳买点" in patterns:
            pattern_boost += 0.08

        # 最终置信度
        final_confidence = min(1.0, max(0.0, base_confidence + pattern_boost))
        return final_confidence

    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        获取技术形态

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 包含形态信号的DataFrame
        """
        # 计算选股模型
        result = self.calculate(data)

        # 初始化形态DataFrame
        patterns_df = pd.DataFrame(index=data.index)

        # 基础选股信号形态
        patterns_df.loc[:, "选股系统买入信号"] = result["FinalSelect"]

        # 选股类型形态
        patterns_df.loc[:, "强趋势上涨股"] = result["StrongUptrendSelect"]
        patterns_df.loc[:, "放量突破股"] = result["VolumeBreakoutSelect"]
        patterns_df.loc[:, "回调买点股"] = result["PullbackBuySelect"]
        patterns_df.loc[:, "洗盘后启动股"] = result["WashplateStartSelect"]
        patterns_df.loc[:, "低吸高弹性股"] = result["LowBuyElasticSelect"]

        # 优先级形态
        patterns_df.loc[:, "最高优先级选股"] = result["BuyPriority"] == 1
        patterns_df.loc[:, "高优先级选股"] = result["BuyPriority"] == 2
        patterns_df.loc[:, "中等优先级选股"] = result["BuyPriority"] == 3
        patterns_df.loc[:, "低优先级选股"] = result["BuyPriority"] == 4
        patterns_df.loc[:, "最低优先级选股"] = result["BuyPriority"] == 5

        # 趋势形态
        if "TrendDirection" in result.columns:
            patterns_df.loc[:, "超强上升趋势"] = result["TrendDirection"] == 1
            patterns_df.loc[:, "下降趋势"] = result["TrendDirection"] == -1
            patterns_df.loc[:, "震荡趋势"] = result["TrendDirection"] == 0
        elif "TrendState" in result.columns:
            patterns_df.loc[:, "上升趋势"] = result["TrendState"] == 1
            patterns_df.loc[:, "下降趋势"] = result["TrendState"] == -1
            patterns_df.loc[:, "震荡趋势"] = result["TrendState"] == 0
        else:
            patterns_df.loc[:, "趋势未知"] = True

        # 买点信号形态
        patterns_df.loc[:, "放量上涨买点"] = result["VolumeRiseBuyPoint"]
        patterns_df.loc[:, "回调企稳买点"] = result["PullbackStabilizeBuyPoint"]
        patterns_df.loc[:, "突破买点"] = result["BreakoutBuyPoint"]
        patterns_df.loc[:, "底部放量买点"] = result["BottomVolumeBuyPoint"]
        patterns_df.loc[:, "缩量整理买点"] = result["VolumeShrinkBuyPoint"]

        # 洗盘信号形态
        if "AnyWashPlate" in result.columns:
            patterns_df.loc[:, "横盘震荡洗盘"] = result.get("ShockWash", False)
            patterns_df.loc[:, "回调洗盘"] = result.get("PullbackWash", False)
            patterns_df.loc[:, "假突破洗盘"] = result.get("FalseBreakWash", False)
            patterns_df.loc[:, "时间洗盘"] = result.get("TimeWash", False)
            patterns_df.loc[:, "连续阴线洗盘"] = result.get("ContinuousYinWash", False)

        return patterns_df

    def set_parameters(self, **kwargs):
        """
        设置指标参数

        Args:
            **kwargs: 参数字典，可包含各子指标的参数
        """
        # 设置各子指标的参数
        if hasattr(self, 'trend_detector'):
            self.trend_detector.set_parameters(**kwargs.get('trend_params', {}))
        if hasattr(self, 'elasticity_indicator'):
            self.elasticity_indicator.set_parameters(**kwargs.get('elasticity_params', {}))
        if hasattr(self, 'score_calculator'):
            self.score_calculator.set_parameters(**kwargs.get('score_params', {}))
        if hasattr(self, 'buy_point_detector'):
            self.buy_point_detector.set_parameters(**kwargs.get('buypoint_params', {}))

        # 设置选股模型自身的参数
        self.selection_threshold = kwargs.get('selection_threshold', 70)
        self.priority_threshold = kwargs.get('priority_threshold', 85)

    def get_pattern_info(self, pattern_id: str) -> dict:
        """
        获取指定形态的详细信息
        
        Args:
            pattern_id: 形态ID
            
        Returns:
            dict: 形态详细信息
        """
        # 默认形态信息
        default_pattern = {
            "id": pattern_id,
            "name": pattern_id,
            "description": f"{pattern_id}形态",
            "type": "NEUTRAL",
            "strength": "MEDIUM",
            "score_impact": 0.0
        }
        
        # SelectionModel指标特定的形态信息映射
        pattern_info_map = {
            # 基础形态
            "超买区域": {
                "id": "超买区域",
                "name": "超买区域",
                "description": "指标进入超买区域，可能面临回调压力",
                "type": "BEARISH",
                "strength": "MEDIUM",
                "score_impact": -10.0
            },
            "超卖区域": {
                "id": "超卖区域", 
                "name": "超卖区域",
                "description": "指标进入超卖区域，可能出现反弹机会",
                "type": "BULLISH",
                "strength": "MEDIUM",
                "score_impact": 10.0
            },
            "中性区域": {
                "id": "中性区域",
                "name": "中性区域", 
                "description": "指标处于中性区域，趋势不明确",
                "type": "NEUTRAL",
                "strength": "WEAK",
                "score_impact": 0.0
            },
            # 趋势形态
            "上升趋势": {
                "id": "上升趋势",
                "name": "上升趋势",
                "description": "指标显示上升趋势，看涨信号",
                "type": "BULLISH", 
                "strength": "STRONG",
                "score_impact": 15.0
            },
            "下降趋势": {
                "id": "下降趋势",
                "name": "下降趋势",
                "description": "指标显示下降趋势，看跌信号",
                "type": "BEARISH",
                "strength": "STRONG", 
                "score_impact": -15.0
            },
            # 信号形态
            "买入信号": {
                "id": "买入信号",
                "name": "买入信号",
                "description": "指标产生买入信号，建议关注",
                "type": "BULLISH",
                "strength": "STRONG",
                "score_impact": 20.0
            },
            "卖出信号": {
                "id": "卖出信号",
                "name": "卖出信号",
                "description": "指标产生卖出信号，建议谨慎",
                "type": "BEARISH",
                "strength": "STRONG",
                "score_impact": -20.0
            },
            # 选股类型形态
            "StrongUptrendSelect": {
                "id": "StrongUptrendSelect",
                "name": "强趋势上涨股",
                "description": "符合强趋势上涨选股条件的股票",
                "type": "BULLISH",
                "strength": "STRONG",
                "score_impact": 25.0
            },
            "VolumeBreakoutSelect": {
                "id": "VolumeBreakoutSelect",
                "name": "放量突破股",
                "description": "符合放量突破选股条件的股票",
                "type": "BULLISH",
                "strength": "STRONG",
                "score_impact": 20.0
            },
            "PullbackBuySelect": {
                "id": "PullbackBuySelect",
                "name": "回调买点股",
                "description": "符合回调买点选股条件的股票",
                "type": "BULLISH",
                "strength": "MEDIUM",
                "score_impact": 15.0
            },
            "WashplateStartSelect": {
                "id": "WashplateStartSelect",
                "name": "洗盘后启动股",
                "description": "符合洗盘后启动选股条件的股票",
                "type": "BULLISH",
                "strength": "MEDIUM",
                "score_impact": 18.0
            },
            "LowBuyElasticSelect": {
                "id": "LowBuyElasticSelect",
                "name": "低吸高弹性股",
                "description": "符合低吸高弹性选股条件的股票",
                "type": "BULLISH",
                "strength": "MEDIUM",
                "score_impact": 15.0
            }
        }
        
        return pattern_info_map.get(pattern_id, default_pattern)