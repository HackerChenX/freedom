#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
主力行为模式选股策略

基于主力行为模式分析的选股策略，识别主力资金行为，抓取主力建仓和拉升机会
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple, Any
import warnings

from strategy.base_strategy import BaseStrategy
from indicators.institutional_behavior import InstitutionalBehavior
from utils.logger import get_logger

# 静默警告
warnings.filterwarnings("ignore", category=RuntimeWarning)

logger = get_logger(__name__)


class InstitutionalStrategy(BaseStrategy):
    """主力行为模式选股策略
    
    基于主力行为模式分析，识别主力资金吸筹、控盘、拉升等行为，捕捉主力建仓完成和启动初期的机会
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        初始化主力行为选股策略
        
        Args:
            params: 策略参数字典
        """
        default_params = {
            "lookback_days": 120,              # 回溯分析的天数
            "absorption_threshold": 0.3,       # 吸筹完成的阈值
            "concentration_threshold": 0.25,   # 筹码集中度阈值
            "behavior_intensity_min": 3.0,     # 行为强度最小值
            "volume_ratio_threshold": 1.5,     # 成交量放大阈值
            "enable_chip_analysis": True,      # 是否启用筹码分析
            "enable_absorption_prediction": True, # 是否启用吸筹预测
            "max_absorption_days": 10,         # 最大吸筹剩余天数
            "behavior_pattern_filter": ["温和吸筹", "强势吸筹", "洗盘", "拉升"] # 关注的行为模式
        }
        
        # 更新默认参数
        if params:
            default_params.update(params)
        
        super().__init__(
            name="InstitutionalStrategy",
            description="主力行为模式选股策略",
            params=default_params
        )
        
        self.institutional_behavior = InstitutionalBehavior()
    
    def select(self, data_dict: Dict[str, pd.DataFrame]) -> List[str]:
        """
        执行选股策略
        
        Args:
            data_dict: 股票数据字典，键为股票代码，值为包含OHLCV数据的DataFrame
        
        Returns:
            List[str]: 选出的股票代码列表
        """
        selected_stocks = []
        
        # 策略评分
        stock_scores = {}
        
        for code, data in data_dict.items():
            try:
                # 检查数据长度
                if len(data) < self.params["lookback_days"]:
                    logger.warning(f"股票 {code} 数据不足，跳过分析")
                    continue
                
                # 截取用于分析的数据
                analysis_data = data.iloc[-self.params["lookback_days"]:]
                
                # 计算主力行为指标
                behavior_result = self.institutional_behavior.calculate(analysis_data)
                
                # 获取最新的指标值
                latest_result = behavior_result.iloc[-1]
                
                # 初始化评分
                score = 0
                reasons = []
                
                # 1. 基于主力行为模式评分
                if "behavior_pattern" in behavior_result.columns:
                    latest_pattern = latest_result["behavior_pattern"]
                    if latest_pattern in self.params["behavior_pattern_filter"]:
                        pattern_score = self._score_behavior_pattern(latest_pattern)
                        score += pattern_score
                        reasons.append(f"{latest_pattern}(+{pattern_score})")
                
                # 2. 基于筹码分布评分
                if self.params["enable_chip_analysis"] and "inst_concentration" in behavior_result.columns:
                    # 筹码集中度评分
                    concentration = latest_result["inst_concentration"]
                    if concentration > self.params["concentration_threshold"]:
                        conc_score = min(5, concentration * 10)
                        score += conc_score
                        reasons.append(f"筹码集中度{concentration:.2f}(+{conc_score:.1f})")
                    
                    # 主力获利比例评分
                    if "inst_profit_ratio" in behavior_result.columns:
                        profit_ratio = latest_result["inst_profit_ratio"]
                        if 0.4 <= profit_ratio <= 0.7:
                            # 主力获利比例在黄金区间
                            profit_score = 3
                            score += profit_score
                            reasons.append(f"主力获利{profit_ratio:.2f}(+{profit_score})")
                
                # 3. 基于主力行为强度评分
                if "behavior_intensity" in behavior_result.columns:
                    intensity = latest_result["behavior_intensity"]
                    if intensity > self.params["behavior_intensity_min"]:
                        intensity_score = min(5, intensity - 2)
                        score += intensity_score
                        reasons.append(f"行为强度{intensity:.1f}(+{intensity_score:.1f})")
                
                # 4. 基于主力阶段评分
                if "inst_phase" in behavior_result.columns:
                    phase = latest_result["inst_phase"]
                    phase_score = self._score_inst_phase(phase)
                    if phase_score > 0:
                        score += phase_score
                        reasons.append(f"{phase}(+{phase_score})")
                
                # 5. 基于阶段变化评分
                if "phase_change" in behavior_result.columns:
                    # 寻找最近的阶段变化
                    recent_changes = behavior_result["phase_change"].iloc[-5:]
                    significant_changes = recent_changes[recent_changes != "无变化"]
                    
                    if not significant_changes.empty:
                        latest_change = significant_changes.iloc[-1]
                        change_score = self._score_phase_change(latest_change)
                        if change_score > 0:
                            score += change_score
                            reasons.append(f"{latest_change}(+{change_score})")
                
                # 6. 吸筹预测评分
                if self.params["enable_absorption_prediction"]:
                    try:
                        absorption_pred = self.institutional_behavior.predict_absorption_completion(analysis_data)
                        
                        if absorption_pred["is_in_absorption"]:
                            # 吸筹即将完成
                            if absorption_pred["completion_days_min"] is not None and \
                               absorption_pred["completion_days_min"] <= self.params["max_absorption_days"]:
                                abs_score = max(1, 6 - absorption_pred["completion_days_min"] / 2)
                                score += abs_score
                                reasons.append(f"吸筹即将完成({absorption_pred['completion_days_min']}天)(+{abs_score:.1f})")
                    except Exception as e:
                        logger.error(f"预测吸筹完成时间出错: {e}")
                
                # 记录评分和理由
                stock_scores[code] = {
                    "score": score,
                    "reasons": reasons,
                    "latest_price": data["close"].iloc[-1],
                    "latest_pattern": latest_result.get("behavior_pattern", "未知"),
                    "latest_phase": latest_result.get("inst_phase", "未知")
                }
                
            except Exception as e:
                logger.error(f"分析股票 {code} 时出错: {e}")
        
        # 根据评分选出股票
        sorted_scores = sorted(stock_scores.items(), key=lambda x: x[1]["score"], reverse=True)
        
        # 记录选股结果
        for code, score_info in sorted_scores:
            if score_info["score"] >= 10:  # 分数阈值
                selected_stocks.append(code)
                logger.info(f"选出股票 {code}, 评分: {score_info['score']:.1f}, 理由: {', '.join(score_info['reasons'])}")
        
        return selected_stocks
    
    def _score_behavior_pattern(self, pattern: str) -> float:
        """
        对行为模式进行评分
        
        Args:
            pattern: 行为模式
            
        Returns:
            float: 评分
        """
        pattern_scores = {
            "温和吸筹": 2.0,
            "强势吸筹": 3.0,
            "洗盘": 3.5,
            "拉升": 4.0,
            "加速拉升": 2.5,  # 已经启动的不给太高分
            "出货": -5.0,
            "集中出货": -10.0,
            "未知": 0.0
        }
        
        return pattern_scores.get(pattern, 0.0)
    
    def _score_inst_phase(self, phase: str) -> float:
        """
        对主力阶段进行评分
        
        Args:
            phase: 主力阶段
            
        Returns:
            float: 评分
        """
        phase_scores = {
            "吸筹期": 3.0,
            "控盘期": 5.0,
            "拉升期": 2.0,
            "出货期": -5.0,
            "观望期": 0.0,
            "未知": 0.0
        }
        
        return phase_scores.get(phase, 0.0)
    
    def _score_phase_change(self, change: str) -> float:
        """
        对阶段变化进行评分
        
        Args:
            change: 阶段变化
            
        Returns:
            float: 评分
        """
        change_scores = {
            "吸筹完成": 6.0,
            "开始拉升": 5.0,
            "开始出货": -5.0,
            "新一轮开始": 2.0,
            "无变化": 0.0
        }
        
        # 对于未明确定义的变化，尝试基于包含的关键词评分
        if change not in change_scores:
            if "吸筹" in change and "控盘" in change:
                return 4.0
            elif "控盘" in change and "拉升" in change:
                return 4.0
            elif "出货" in change:
                return -3.0
        
        return change_scores.get(change, 0.0)
    
    def analyze_stock(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        详细分析单只股票的主力行为
        
        Args:
            data: 股票的OHLCV数据
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        result = {}
        
        try:
            # 检查数据长度
            if len(data) < self.params["lookback_days"]:
                return {"error": "数据长度不足"}
            
            # 截取用于分析的数据
            analysis_data = data.iloc[-self.params["lookback_days"]:]
            
            # 计算基础指标
            behavior_result = self.institutional_behavior.calculate(analysis_data)
            
            # 细化行为模式分类
            behavior_classifications = self.institutional_behavior.classify_institutional_behavior(analysis_data)
            
            # 预测吸筹完成时间
            absorption_prediction = self.institutional_behavior.predict_absorption_completion(analysis_data)
            
            # 构建分析结果
            result = {
                "latest_price": data["close"].iloc[-1],
                "latest_volume": data["volume"].iloc[-1],
                "behavior_result": behavior_result.iloc[-1].to_dict(),
                "behavior_classifications": behavior_classifications,
                "absorption_prediction": absorption_prediction,
                "score": 0,
                "investment_suggestion": "",
                "risk_level": ""
            }
            
            # 计算投资建议评分
            score = 0
            reasons = []
            
            # 合并所有评分因素
            if "behavior_pattern" in behavior_result.columns:
                pattern_score = self._score_behavior_pattern(behavior_result["behavior_pattern"].iloc[-1])
                score += pattern_score
                if pattern_score != 0:
                    reasons.append(f"行为模式: {behavior_result['behavior_pattern'].iloc[-1]} ({pattern_score:+.1f})")
            
            if "inst_phase" in behavior_result.columns:
                phase_score = self._score_inst_phase(behavior_result["inst_phase"].iloc[-1])
                score += phase_score
                if phase_score != 0:
                    reasons.append(f"主力阶段: {behavior_result['inst_phase'].iloc[-1]} ({phase_score:+.1f})")
            
            if "phase_change" in behavior_result.columns:
                recent_changes = behavior_result["phase_change"].iloc[-5:]
                significant_changes = recent_changes[recent_changes != "无变化"]
                
                if not significant_changes.empty:
                    latest_change = significant_changes.iloc[-1]
                    change_score = self._score_phase_change(latest_change)
                    score += change_score
                    if change_score != 0:
                        reasons.append(f"阶段变化: {latest_change} ({change_score:+.1f})")
            
            # 添加吸筹预测评分
            if absorption_prediction["is_in_absorption"]:
                if absorption_prediction["completion_days_min"] is not None and \
                   absorption_prediction["completion_days_min"] <= self.params["max_absorption_days"]:
                    abs_score = max(1, 6 - absorption_prediction["completion_days_min"] / 2)
                    score += abs_score
                    reasons.append(f"吸筹预计{absorption_prediction['completion_days_min']}天完成 ({abs_score:+.1f})")
            
            # 添加评分和理由
            result["score"] = score
            result["score_reasons"] = reasons
            
            # 生成投资建议
            if score >= 15:
                result["investment_suggestion"] = "强烈推荐买入"
                result["risk_level"] = "低"
            elif score >= 10:
                result["investment_suggestion"] = "建议买入"
                result["risk_level"] = "中低"
            elif score >= 5:
                result["investment_suggestion"] = "可以考虑"
                result["risk_level"] = "中"
            elif score >= 0:
                result["investment_suggestion"] = "观望"
                result["risk_level"] = "中高"
            else:
                result["investment_suggestion"] = "不建议介入"
                result["risk_level"] = "高"
            
        except Exception as e:
            result["error"] = str(e)
        
        return result 