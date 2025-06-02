#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
形态分析模块

用于分析和识别技术形态，支持多种指标和周期
"""

import os
import sys
import logging
import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Set

# 获取项目根目录
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from utils.logger import get_logger
from utils.decorators import performance_monitor, time_it
from utils.period_data_structure import (
    IndicatorPeriodResult,
    PeriodAnalysisResult,
    MultiPeriodAnalysisResult
)
from indicators.factory import IndicatorFactory
from indicators.pattern_registry import PatternRegistry
from indicators.zxm.diagnostics import ZXMDiagnostics

# 获取日志记录器
logger = get_logger(__name__)


class PatternAnalyzer:
    """
    形态分析器
    
    用于识别和分析各种技术形态，包括ZXM体系形态
    """

    def __init__(self):
        """初始化形态分析器"""
        self.indicator_factory = IndicatorFactory()
        self.pattern_registry = PatternRegistry()
        self.zxm_diagnostics = ZXMDiagnostics()
        
        # 缓存
        self.pattern_cache = {}
        
        logger.info("形态分析器初始化完成")
        
    def analyze_pattern(self, data: pd.DataFrame, pattern_id: str, 
                       min_strength: float = 0.6) -> Dict[str, Any]:
        """
        分析特定形态
        
        Args:
            data: 股票数据
            pattern_id: 形态ID
            min_strength: 最小形态强度
            
        Returns:
            Dict[str, Any]: 形态分析结果
        """
        try:
            # 检查缓存
            cache_key = f"{pattern_id}_{len(data)}"
            if cache_key in self.pattern_cache:
                return self.pattern_cache[cache_key]
            
            # 获取形态定义
            pattern_def = self.pattern_registry.get_pattern(pattern_id)
            if not pattern_def:
                logger.warning(f"未找到形态定义: {pattern_id}")
                return {"detected": False}
            
            # 计算形态
            indicator_id = pattern_def.get("indicator_id")
            indicator = self.indicator_factory.create_indicator(indicator_id)
            
            if not indicator:
                logger.warning(f"未找到指标: {indicator_id}")
                return {"detected": False}
            
            # 计算指标
            indicator_result = indicator.calculate(
                data['open'].values,
                data['high'].values,
                data['low'].values,
                data['close'].values,
                data['volume'].values
            )
            
            # 检查形态检测结果
            pattern_key = f"pattern_{pattern_id}"
            if pattern_key in indicator_result:
                pattern_array = indicator_result[pattern_key]
                detected = np.any(pattern_array[-5:])  # 检查最后5个交易日是否有形态
                
                result = {
                    "detected": detected,
                    "pattern_id": pattern_id,
                    "pattern_name": pattern_def.get("name", ""),
                    "description": pattern_def.get("description", ""),
                    "indicator_id": indicator_id,
                    "strength": float(np.max(pattern_array[-5:])) if detected else 0.0,
                }
                
                # 缓存结果
                self.pattern_cache[cache_key] = result
                return result
            else:
                logger.warning(f"指标结果中未找到形态键: {pattern_key}")
                return {"detected": False}
                
        except Exception as e:
            logger.error(f"分析形态时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"detected": False, "error": str(e)}
    
    def analyze_patterns(self, data: pd.DataFrame, pattern_ids: List[str], 
                        min_strength: float = 0.6) -> List[Dict[str, Any]]:
        """
        分析多个形态
        
        Args:
            data: 股票数据
            pattern_ids: 形态ID列表
            min_strength: 最小形态强度
            
        Returns:
            List[Dict[str, Any]]: 形态分析结果列表
        """
        results = []
        
        for pattern_id in pattern_ids:
            pattern_result = self.analyze_pattern(data, pattern_id, min_strength)
            if pattern_result.get("detected", False) and pattern_result.get("strength", 0) >= min_strength:
                results.append(pattern_result)
                
        return results
    
    def analyze_all_patterns(self, data: pd.DataFrame, 
                            min_strength: float = 0.6) -> List[Dict[str, Any]]:
        """
        分析所有已注册的形态
        
        Args:
            data: 股票数据
            min_strength: 最小形态强度
            
        Returns:
            List[Dict[str, Any]]: 形态分析结果列表
        """
        all_patterns = self.pattern_registry.get_all_patterns()
        pattern_ids = [pattern.get("id") for pattern in all_patterns]
        
        return self.analyze_patterns(data, pattern_ids, min_strength)
    
    def match_pattern_combinations(self, analysis_result: MultiPeriodAnalysisResult,
                                  pattern_combinations: List[Dict[str, Any]],
                                  min_strength: float = 60) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        检测是否符合形态组合条件
        
        Args:
            analysis_result: 分析结果
            pattern_combinations: 形态组合列表
            min_strength: 最小形态强度
            
        Returns:
            Tuple[bool, List]: 是否匹配和匹配的形态列表
        """
        matched_patterns = []

        # 获取分析结果中的所有形态
        for period_key, period_result in analysis_result.periods.items():
            for indicator_name, indicator_result in period_result.indicators.items():
                patterns = indicator_result.get_patterns()

                # 检查每个形态是否匹配组合条件
                for pattern in patterns:
                    pattern_id = pattern.get('pattern_id', '')
                    indicator_id = indicator_name
                    strength = pattern.get('strength', 0) * 100  # 转换为百分比

                    # 检查是否满足最小强度要求
                    if strength < min_strength:
                        continue

                    # 检查是否匹配组合中的任何一个条件
                    for combination in pattern_combinations:
                        comb_pattern_id = combination.get('pattern_id', '')
                        comb_indicator_id = combination.get('indicator_id', '')
                        comb_period = combination.get('period', '')

                        # 匹配条件：形态ID、指标ID和周期都匹配
                        if (comb_pattern_id == pattern_id and 
                            comb_indicator_id in indicator_id and 
                            comb_period == period_key):
                            
                            # 添加到匹配结果
                            matched_patterns.append({
                                'pattern_id': pattern_id,
                                'indicator_id': indicator_id,
                                'period': period_key,
                                'strength': strength,
                                'description': pattern.get('description', '')
                            })

        # 如果有匹配的形态，则返回True
        return len(matched_patterns) > 0, matched_patterns
    
    def analyze_zxm_indicators(self, result: MultiPeriodAnalysisResult, 
                              title: str = "ZXM体系指标分析结果") -> Dict[str, Any]:
        """
        分析ZXM体系指标
        
        Args:
            result: 多周期分析结果
            title: 分析标题
            
        Returns:
            Dict[str, Any]: ZXM体系指标分析结果
        """
        # 使用ZXMDiagnostics分析结果
        return self.zxm_diagnostics.analyze_indicators_result(result, title)
    
    def format_zxm_analysis(self, zxm_analysis: Dict[str, Any]) -> str:
        """
        格式化ZXM分析结果为可读文本
        
        Args:
            zxm_analysis: ZXM分析结果
            
        Returns:
            str: 格式化后的文本
        """
        if "error" in zxm_analysis:
            return f"分析错误: {zxm_analysis['error']}"
        
        output = []
        output.append("\n" + "=" * 80)
        output.append(f"{zxm_analysis.get('title', 'ZXM体系指标分析结果')}")
        output.append("=" * 80)
        
        # 输出各周期指标
        for period_key, period_data in zxm_analysis.get('periods', {}).items():
            output.append(f"\n周期: {period_key}")
            output.append("-" * 80)
            
            for indicator_name, indicator_data in period_data.get('indicators', {}).items():
                output.append(f"\n指标: {indicator_name}")
                output.append(f"得分: {indicator_data.get('score', 0):.2f}%")
                
                # 输出形态
                patterns = indicator_data.get('patterns', [])
                if patterns:
                    output.append("形态:")
                    for pattern in patterns:
                        pattern_id = pattern.get('pattern_id', 'unknown')
                        strength = pattern.get('strength', 0) * 100
                        description = pattern.get('description', '')
                        output.append(f"  - ID: {pattern_id}, 强度: {strength:.2f}%, 描述: {description}")
                
                # 输出ZXM特有的值
                if 'ZXM_DK_SG' in indicator_name:
                    if 'dk_value' in indicator_data:
                        output.append("多空势格值:")
                        output.append(f"  - 多空值: {indicator_data.get('dk_value', 0):.2f} ({indicator_data.get('dk_status', '')})")
                        output.append(f"  - 势格值: {indicator_data.get('sg_value', 0):.2f} ({indicator_data.get('sg_level', '')})")
                
                elif 'ZXM_MACD' in indicator_name:
                    if 'dif' in indicator_data:
                        output.append("MACD值:")
                        output.append(f"  - DIF: {indicator_data.get('dif', 0):.4f}")
                        output.append(f"  - DEA: {indicator_data.get('dea', 0):.4f}")
                        output.append(f"  - MACD: {indicator_data.get('macd', 0):.4f}")
                        output.append(f"  - 状态: {indicator_data.get('status', '')}")
                
                elif 'ZXM_MOMENTUM' in indicator_name:
                    if 'momentum' in indicator_data:
                        output.append("动量值:")
                        output.append(f"  - 动量: {indicator_data.get('momentum', 0):.4f}")
                        output.append(f"  - 信号: {indicator_data.get('signal', 0):.4f}")
                        output.append(f"  - 状态: {indicator_data.get('status', '')}")
                
                elif 'ZXM_KDJ' in indicator_name:
                    if 'k' in indicator_data:
                        output.append("KDJ值:")
                        output.append(f"  - K: {indicator_data.get('k', 0):.2f}")
                        output.append(f"  - D: {indicator_data.get('d', 0):.2f}")
                        output.append(f"  - J: {indicator_data.get('j', 0):.2f}")
                        output.append(f"  - 状态: {indicator_data.get('status', '')}")
                
                elif 'ZXM_BOLL' in indicator_name:
                    if 'upper' in indicator_data:
                        output.append("布林带值:")
                        output.append(f"  - 上轨: {indicator_data.get('upper', 0):.2f}")
                        output.append(f"  - 中轨: {indicator_data.get('middle', 0):.2f}")
                        output.append(f"  - 下轨: {indicator_data.get('lower', 0):.2f}")
                        output.append(f"  - 当前价: {indicator_data.get('price', 0):.2f}")
                        output.append(f"  - 带宽: {indicator_data.get('width', 0):.4f}")
                        output.append(f"  - 价格状态: {indicator_data.get('status', '')}")
                        output.append(f"  - 带宽状态: {indicator_data.get('width_status', '')}")
        
        # 输出综合分析
        output.append("\n多周期合成分析:")
        output.append("-" * 80)
        output.append(f"综合得分: {zxm_analysis.get('composite_score', 0):.2f}%")
        output.append(f"综合评级: {zxm_analysis.get('composite_rating', '')}")
        
        # 输出趋势一致性分析
        trend_consistency = zxm_analysis.get('trend_consistency', {})
        output.append("\n多周期趋势分析:")
        output.append("-" * 50)
        
        # 输出各周期趋势
        for period_key, trend_data in trend_consistency.get('trends', {}).items():
            output.append(f"{period_key}: {trend_data.get('trend', '')} (强度: {trend_data.get('strength', 0):.2f})")
        
        # 输出一致性分析
        output.append(f"\n趋势一致性: {trend_consistency.get('consistency', '')} ({trend_consistency.get('consistency_description', '')})")
        output.append(f"多周期趋势方向: {trend_consistency.get('direction', '')}")
        output.append(f"操作建议: {trend_consistency.get('advice', '')}")
        
        output.append("\n" + "=" * 80)
        
        return "\n".join(output)
    
    def get_pattern_stats(self, analysis_results: List[MultiPeriodAnalysisResult]) -> Dict[str, Dict[str, Any]]:
        """
        获取形态统计信息
        
        Args:
            analysis_results: 分析结果列表
            
        Returns:
            Dict[str, Dict[str, Any]]: 形态统计信息
        """
        pattern_stats = {}
        
        for result in analysis_results:
            for period_key, period_result in result.periods.items():
                for indicator_name, indicator_result in period_result.indicators.items():
                    patterns = indicator_result.get_patterns()
                    
                    for pattern in patterns:
                        pattern_id = pattern.get('pattern_id', '')
                        if not pattern_id:
                            continue
                            
                        # 生成唯一的形态键
                        pattern_key = f"{period_key}_{indicator_name}_{pattern_id}"
                        
                        # 更新统计信息
                        if pattern_key not in pattern_stats:
                            pattern_stats[pattern_key] = {
                                "period": period_key,
                                "indicator": indicator_name,
                                "pattern_id": pattern_id,
                                "description": pattern.get('description', ''),
                                "count": 0,
                                "avg_strength": 0,
                                "occurrences": []
                            }
                        
                        # 添加出现情况
                        pattern_stats[pattern_key]["count"] += 1
                        pattern_stats[pattern_key]["avg_strength"] += pattern.get('strength', 0)
                        pattern_stats[pattern_key]["occurrences"].append({
                            "stock_code": result.stock_code,
                            "date": result.buy_date,
                            "strength": pattern.get('strength', 0)
                        })
        
        # 计算平均强度
        for pattern_key, stats in pattern_stats.items():
            if stats["count"] > 0:
                stats["avg_strength"] = stats["avg_strength"] / stats["count"]
        
        return pattern_stats


# 如果直接运行此模块，进行简单测试
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="形态分析器测试")
    parser.add_argument("--stock", "-s", type=str, required=True, help="股票代码")
    parser.add_argument("--start-date", "-sd", type=str, required=True, help="开始日期")
    parser.add_argument("--end-date", "-ed", type=str, required=True, help="结束日期")
    
    args = parser.parse_args()
    
    from db.db_manager import DBManager
    
    # 获取数据
    db_manager = DBManager()
    data = db_manager.get_stock_info(args.stock, "daily", args.start_date, args.end_date)
    
    # 初始化分析器
    analyzer = PatternAnalyzer()
    
    # 分析所有形态
    patterns = analyzer.analyze_all_patterns(data)
    
    # 打印结果
    print(f"股票 {args.stock} 的形态分析结果:")
    for pattern in patterns:
        print(f"- {pattern['pattern_name']}: 强度 {pattern['strength']:.2f}")
        print(f"  描述: {pattern['description']}")
        print() 