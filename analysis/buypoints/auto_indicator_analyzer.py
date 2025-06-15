#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
自动指标分析器

自动分析多个技术指标并识别形态
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union, Set
from datetime import datetime
import logging

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from utils.logger import get_logger
from indicators.factory import IndicatorFactory
from indicators.scoring_framework import IndicatorScoreManager
from indicators.base_indicator import BaseIndicator

# 尝试导入pattern_recognition_analyzer，如果talib不可用则跳过
try:
    from analysis.pattern_recognition_analyzer import PatternRecognitionAnalyzer
    PATTERN_ANALYZER_AVAILABLE = True
except ImportError as e:
    PATTERN_ANALYZER_AVAILABLE = False
    print(f"警告: 形态识别分析器不可用，跳过K线形态分析: {e}")

logger = get_logger(__name__)

class AutoIndicatorAnalyzer:
    """自动指标分析器"""
    
    def __init__(self):
        """初始化分析器"""
        self.indicator_factory = IndicatorFactory()

        # 确保指标注册表正确初始化
        from indicators.indicator_registry import get_registry
        self.indicator_registry = get_registry()

        self.score_manager = IndicatorScoreManager()

        # 获取所有可用指标
        self.all_indicators = self._get_all_indicators()

        # 仅在可用时初始化形态分析器
        self.pattern_analyzer = None
        if PATTERN_ANALYZER_AVAILABLE:
            try:
                self.pattern_analyzer = PatternRecognitionAnalyzer()
            except Exception as e:
                logger.error(f"初始化形态分析器时出错: {e}")
        
    def _get_all_indicators(self) -> List[str]:
        """
        获取所有可用指标

        Returns:
            List[str]: 所有可用指标列表
        """
        # 获取指标注册表中所有注册的指标
        registry_indicators = self.indicator_registry.get_indicator_names()

        # 获取工厂中所有注册的指标
        factory_indicators = self.indicator_factory.get_all_registered_indicators()

        # 合并两个列表，去重
        all_indicators = list(set(registry_indicators + factory_indicators))

        logger.info(f"已加载 {len(all_indicators)} 个技术指标")
        logger.info(f"  - 注册表指标: {len(registry_indicators)} 个: {registry_indicators}")
        logger.info(f"  - 工厂指标: {len(factory_indicators)} 个: {factory_indicators}")

        return all_indicators
    
    def analyze_all_indicators(self, 
                      stock_data: Dict[str, pd.DataFrame],
                      target_rows: Dict[str, int]) -> Dict[str, List[Dict[str, Any]]]:
        """
        分析所有指标
        
        Args:
            stock_data: 多周期股票数据
            target_rows: 每个周期的目标行索引
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: 按周期组织的指标分析结果
        """
        try:
            results = {}
            
            # 对每个周期的数据分别进行分析
            for period, df in stock_data.items():
                if df is None or df.empty:
                    logger.warning(f"周期 {period} 的数据为空，跳过分析")
                    continue
                    
                # 获取目标行索引
                target_row = target_rows.get(period)
                if target_row is None:
                    logger.warning(f"周期 {period} 没有提供目标行索引，使用最后一行")
                    target_row = len(df) - 1
                    
                if target_row < 0 or target_row >= len(df):
                    logger.warning(f"周期 {period} 的目标行索引 {target_row} 超出范围，跳过分析")
                    continue
                    
                # 分析该周期的所有指标
                period_results = self._analyze_period_indicators(df, target_row)
                
                if period_results:
                    results[period] = period_results
                    
            return results
        except Exception as e:
            logger.error(f"分析所有指标时出错: {e}")
            return {}
    
    def _find_closest_date_row(self, df: pd.DataFrame, target_date: str) -> Optional[int]:
        """
        找到最接近目标日期的数据行索引
        
        Args:
            df: 股票数据
            target_date: 目标日期
            
        Returns:
            Optional[int]: 最接近的行索引，如果找不到则返回None
        """
        try:
            # 确保目标日期格式正确
            if isinstance(target_date, str):
                if len(target_date) == 8:  # YYYYMMDD格式
                    target_date = datetime.strptime(target_date, '%Y%m%d')
                else:
                    target_date = datetime.strptime(target_date, '%Y-%m-%d')
                    
            # 如果DataFrame中的日期是字符串，转换为datetime
            date_col = pd.to_datetime(df['date'])
                
            # 找到不超过目标日期的最大日期
            valid_dates = date_col[date_col <= target_date]
            
            if valid_dates.empty:
                return None
                
            closest_idx = valid_dates.index[-1]  # 取最后一个有效日期的索引
            return closest_idx
            
        except Exception as e:
            logger.error(f"查找最接近日期时出错: {e}")
            return None
    
    def _analyze_period_indicators(self, 
                               df: pd.DataFrame, 
                               target_idx: int) -> List[Dict[str, Any]]:
        """
        分析特定周期下所有指标
        
        Args:
            df: 股票数据
            target_idx: 目标行索引
            
        Returns:
            List[Dict[str, Any]]: 命中的指标列表
        """
        hit_indicators = []
        
        # 分析K线形态，仅在形态分析器可用时执行
        if self.pattern_analyzer is not None:
            pattern_results = self._analyze_patterns(df, target_idx)
            if pattern_results:
                hit_indicators.extend(pattern_results)
        
        # 分析所有技术指标
        for indicator_name in self.all_indicators:
            try:
                # 尝试从注册表创建指标实例
                indicator = self.indicator_registry.create_indicator(indicator_name)

                # 如果注册表创建失败，尝试从工厂创建
                if indicator is None:
                    indicator = self.indicator_factory.create_indicator(indicator_name)

                # 检查指标实例是否有效
                if indicator is None:
                    logger.warning(f"无法创建指标 {indicator_name} 的实例，跳过分析")
                    continue
                    
                # 检查是否有calculate方法
                if not hasattr(indicator, 'calculate') or not callable(getattr(indicator, 'calculate')):
                    logger.warning(f"指标 {indicator_name} 没有calculate方法，跳过分析")
                    continue
                
                # 计算指标值 - 使用数据副本确保原始数据不被修改
                df_copy = df.copy()
                indicator_df = indicator.calculate(df_copy)

                # 检查指标计算结果是否有效
                if indicator_df is None or indicator_df.empty:
                    logger.debug(f"指标 {indicator_name} 计算结果为空，跳过形态分析")
                    continue

                # 识别形态
                indicator_hits = self._analyze_indicator_patterns(
                    indicator,
                    indicator_df,
                    target_idx
                )
                
                # 如果有命中形态，添加到结果中
                if indicator_hits:
                    hit_indicators.extend(indicator_hits)
                    
            except Exception as e:
                err_msg = str(e)
                if "输入数据缺少" in err_msg or "DataFrame必须包含" in err_msg or "输入数据缺少所需的列" in err_msg:
                    logger.warning(f"分析指标 {indicator_name} 时因数据缺失而跳过: {err_msg}")
                else:
                    logger.error(f"分析指标 {indicator_name} 时出错: {e}")
                continue
                
        return hit_indicators
    
    def _analyze_patterns(self, 
                       df: pd.DataFrame, 
                       target_idx: int) -> List[Dict[str, Any]]:
        """
        分析K线形态
        
        Args:
            df: 股票数据
            target_idx: 目标行索引
            
        Returns:
            List[Dict[str, Any]]: 命中的K线形态列表
        """
        try:
            # 使用形态识别分析器
            patterns = self.pattern_analyzer.identify_patterns(df)
            
            # 过滤出目标日期的形态
            target_date = df.iloc[target_idx]['date']
            
            # 找出最接近但不超过目标日期的形态
            hit_patterns = []
            
            for pattern_name, pattern_data in patterns.items():
                if pattern_data.empty:
                    continue
                    
                # 找到目标日期附近的形态
                date_col = pd.to_datetime(pattern_data['date'])
                target_date_dt = pd.to_datetime(target_date)
                
                # 找出目标日期或之前最近的形态
                valid_patterns = pattern_data[date_col <= target_date_dt]
                
                if valid_patterns.empty:
                    continue
                    
                # 获取最接近目标日期的形态
                closest_pattern = valid_patterns.iloc[-1]
                pattern_date = closest_pattern['date']
                
                # 计算日期差距（天数）
                if isinstance(pattern_date, str):
                    pattern_date = pd.to_datetime(pattern_date)
                date_diff = (target_date_dt - pattern_date).days
                
                # 只考虑最近5个交易日内的形态
                if date_diff <= 5:
                    # 计算形态得分
                    score = self.score_manager.score_pattern(
                        pattern_name, 
                        closest_pattern.to_dict()
                    )
                    
                    hit_patterns.append({
                        'type': 'pattern',
                        'name': pattern_name,
                        'score': score,
                        'date_diff': date_diff,
                        'pattern_date': pattern_date.strftime('%Y-%m-%d') if hasattr(pattern_date, 'strftime') else pattern_date,
                        'details': closest_pattern.to_dict()
                    })
            
            return hit_patterns
            
        except Exception as e:
            logger.error(f"分析K线形态时出错: {e}")
            return []
    
    def _analyze_indicator_patterns(self, 
                                   indicator: BaseIndicator,
                                   indicator_df: pd.DataFrame, 
                                   target_idx: int) -> List[Dict[str, Any]]:
        """
        分析具体指标的形态
        
        Args:
            indicator: 指标实例
            indicator_df: 包含指标值的DataFrame
            target_idx: 目标行索引
            
        Returns:
            List[Dict[str, Any]]: 命中的形态列表
        """
        hit_patterns = []
        
        try:
            # 获取指标名称（更安全的方式）
            indicator_name = getattr(indicator, 'name', indicator.__class__.__name__)

            # 尝试获取形态识别结果
            patterns_result = None

            # 方法1：尝试使用get_patterns方法
            if hasattr(indicator, 'get_patterns') and callable(getattr(indicator, 'get_patterns')):
                try:
                    patterns_result = indicator.get_patterns(indicator_df)
                except Exception as e:
                    logger.debug(f"指标 {indicator_name} 的get_patterns方法调用失败: {e}")

            # 方法2：尝试使用identify_patterns方法
            if patterns_result is None and hasattr(indicator, 'identify_patterns') and callable(getattr(indicator, 'identify_patterns')):
                try:
                    patterns_result = indicator.identify_patterns(indicator_df)
                except Exception as e:
                    logger.debug(f"指标 {indicator_name} 的identify_patterns方法调用失败: {e}")

            # 如果都没有获取到结果，直接返回
            if patterns_result is None:
                logger.debug(f"指标 {indicator_name} 没有返回形态识别结果")
                return hit_patterns
            
            # 如果返回的是布尔型DataFrame
            if isinstance(patterns_result, pd.DataFrame) and not patterns_result.empty:
                # 确保索引连续性，避免"Gaps in blk ref_locs"错误
                try:
                    # 检查target_idx是否在有效范围内
                    if target_idx >= len(patterns_result):
                        logger.debug(f"指标 {indicator_name} 的target_idx({target_idx})超出patterns_result长度({len(patterns_result)})，尝试使用最后一行数据")
                        # 使用最后一行数据作为替代
                        if len(patterns_result) > 0:
                            target_idx = len(patterns_result) - 1
                        else:
                            return hit_patterns

                    # 创建数据副本并重置索引以确保连续性
                    patterns_result_safe = patterns_result.copy()
                    patterns_result_safe = patterns_result_safe.reset_index(drop=True)

                    # 确保所有列都是布尔类型，避免数据类型冲突
                    for col in patterns_result_safe.columns:
                        if patterns_result_safe[col].dtype != bool:
                            patterns_result_safe[col] = patterns_result_safe[col].astype(bool)

                    # 安全地提取目标行的形态
                    if target_idx < len(patterns_result_safe):
                        target_patterns = patterns_result_safe.iloc[target_idx].copy()
                        # 筛选出值为True的形态
                        hit_pattern_ids = target_patterns[target_patterns == True].index.tolist()
                    else:
                        hit_pattern_ids = []

                except Exception as e:
                    logger.warning(f"处理指标 {indicator_name} 的形态数据时出错: {e}，尝试备用方法")
                    # 备用方法：使用最安全的方式访问数据
                    try:
                        hit_pattern_ids = []
                        if target_idx < len(patterns_result):
                            # 使用最安全的.iat方法逐列检查，避免pandas内部索引问题
                            for i, col in enumerate(patterns_result.columns):
                                try:
                                    # 使用.iat进行最安全的单元格访问
                                    cell_value = patterns_result.iat[target_idx, i]
                                    if pd.notna(cell_value) and bool(cell_value):
                                        hit_pattern_ids.append(col)
                                except (IndexError, ValueError) as e:
                                    logger.debug(f"指标 {indicator_name} 访问列 {col} 时出错: {e}")
                                    continue
                    except Exception as e2:
                        logger.error(f"指标 {indicator_name} 形态数据处理完全失败: {e2}")
                        return hit_patterns
                
                # 获取命中的形态信息
                for pattern_id in hit_pattern_ids:
                    # 获取形态的详细信息
                    try:
                        pattern_info = indicator.get_pattern_info(pattern_id)

                        # 确保pattern_info是字典类型
                        if isinstance(pattern_info, dict):
                            pattern_name = pattern_info.get('name', pattern_id)
                            description = pattern_info.get('description', '')
                            pattern_type = pattern_info.get('type', 'UNKNOWN')
                        else:
                            # 如果返回的不是字典，使用默认值
                            pattern_name = str(pattern_info) if pattern_info else pattern_id
                            description = ''
                            pattern_type = 'UNKNOWN'
                    except Exception as e:
                        logger.warning(f"获取指标 {indicator_name} 形态 {pattern_id} 信息时出错: {e}")
                        pattern_name = pattern_id
                        description = ''
                        pattern_type = 'UNKNOWN'

                    hit_patterns.append({
                        "type": "indicator",
                        "indicator_name": indicator_name,
                        "pattern_id": pattern_id,
                        "pattern_name": pattern_name,
                        "description": description,
                        "pattern_type": pattern_type
                    })

            # 如果返回的是列表（已弃用的旧格式）
            elif isinstance(patterns_result, list) and patterns_result:
                logging.warning(f"指标 {indicator_name} 正在使用已弃用的列表格式返回形态，影响性能且不规范。请计划将其重构为返回布尔型DataFrame的新标准。")
                
                # 获取目标日期
                target_date = pd.to_datetime(indicator_df.index[target_idx])
                
                # 遍历所有历史形态，只选择在目标日期命中的
                for pattern_dict in patterns_result:
                    pattern_start_date = pattern_dict.get('start_date') or pattern_dict.get('date')
                    pattern_end_date = pattern_dict.get('end_date') or pattern_dict.get('date')

                    if not pattern_start_date:
                        continue
                        
                    # 转换日期为datetime对象
                    start_dt = pd.to_datetime(pattern_start_date)
                    end_dt = pd.to_datetime(pattern_end_date) if pattern_end_date else start_dt

                    # 检查目标日期是否在形态有效期内
                    if start_dt <= target_date <= end_dt:
                        # 确保返回格式统一
                        try:
                            pattern_info = indicator.get_pattern_info(pattern_dict.get('id'))

                            # 确保pattern_info是字典类型
                            if isinstance(pattern_info, dict):
                                pattern_name = pattern_dict.get('name') or pattern_info.get('name')
                                description = pattern_dict.get('description') or pattern_info.get('description')
                                pattern_type = pattern_dict.get('type') or pattern_info.get('type', 'UNKNOWN')
                            else:
                                pattern_name = pattern_dict.get('name') or str(pattern_info) if pattern_info else pattern_dict.get('id')
                                description = pattern_dict.get('description', '')
                                pattern_type = pattern_dict.get('type', 'UNKNOWN')
                        except Exception as e:
                            logger.warning(f"获取指标 {indicator_name} 形态信息时出错: {e}")
                            pattern_name = pattern_dict.get('name', pattern_dict.get('id'))
                            description = pattern_dict.get('description', '')
                            pattern_type = pattern_dict.get('type', 'UNKNOWN')

                        hit_patterns.append({
                            "type": "indicator",
                            "indicator_name": indicator_name,
                            "pattern_id": pattern_dict.get('id'),
                            "pattern_name": pattern_name,
                            "description": description,
                            "pattern_type": pattern_type,
                            "details": pattern_dict # 保留原始信息
                        })

        except Exception as e:
            logger.error(f"分析指标 {indicator_name} 形态时出错: {e}")
            
        return hit_patterns

    def _get_pattern_display_name(self, pattern_id: str) -> str:
        """
        获取形态的显示名称（如果可用）
        """
        # 此处可以添加逻辑，将形态ID映射到更友好的显示名称
        return pattern_id.replace('_', ' ').title()

    def _get_indicator_instance(self, indicator_name: str) -> Optional[BaseIndicator]:
        """
        获取指标实例

        Args:
            indicator_name: 指标名称

        Returns:
            Optional[BaseIndicator]: 指标实例，获取失败则返回None
        """
        try:
            # 尝试从注册表创建指标实例
            indicator = self.indicator_registry.create_indicator(indicator_name)

            # 如果注册表创建失败，尝试从工厂创建
            if indicator is None:
                indicator = self.indicator_factory.create_indicator(indicator_name)

            return indicator
        except Exception as e:
            logger.error(f"创建指标 {indicator_name} 实例时出错: {e}")
            return None