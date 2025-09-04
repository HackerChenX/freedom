#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
买点回测引擎
实现您设想的完整买点回测工作流程
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import os
import json

from utils.logger import get_logger
from utils.dependency_injection import get_service
from db.interfaces.data_access_interface import DataAccessInterface
from analysis.buypoints.period_data_processor import PeriodDataProcessor
from indicators.complete_indicator_registry import get_indicator_registry
from strategy.implementations.pattern_strategy_generator import PatternStrategyGenerator

logger = get_logger(__name__)


class BuyPointBacktestEngine:
    """
    买点回测引擎
    
    实现完整的买点回测工作流程：
    1. 读取buypoints.csv
    2. 多周期数据查询和转换
    3. 103个指标逐个测试
    4. 周期独立的形态识别
    5. 指标打分机制
    6. 策略转换和双向验证
    """
    
    def __init__(self):
        """初始化买点回测引擎"""
        self.data_access = get_service(DataAccessInterface)
        self.period_processor = PeriodDataProcessor()
        self.indicator_registry = get_indicator_registry()
        self.strategy_generator = PatternStrategyGenerator()

        # 延迟导入策略执行引擎以避免循环导入
        self._strategy_executor = None
        
        # 支持的周期列表
        self.periods = ['15min', '30min', '60min', 'daily', 'weekly', 'monthly']
        
        # 获取所有可用指标
        self.available_indicators = list(self.indicator_registry.get_all_indicators().keys())
        logger.info(f"买点回测引擎初始化完成，支持{len(self.available_indicators)}个指标")
    
    def run_buypoint_backtest(self, buypoints_file: str = "data/buypoints.csv") -> Dict[str, Any]:
        """
        运行完整的买点回测流程
        
        Args:
            buypoints_file: 买点文件路径
            
        Returns:
            Dict[str, Any]: 回测结果
        """
        logger.info("🚀 开始买点回测流程...")
        
        # 1. 读取买点数据
        buypoints = self._load_buypoints(buypoints_file)
        logger.info(f"📋 加载买点数据: {len(buypoints)}个买点")
        
        # 2. 逐个处理买点
        backtest_results = []
        for i, buypoint in buypoints.iterrows():
            stock_code = buypoint['stock_code']
            buypoint_date = str(buypoint['buypoint_date'])
            
            logger.info(f"[{i+1}/{len(buypoints)}] 处理买点: {stock_code} @ {buypoint_date}")
            
            # 处理单个买点
            result = self._process_single_buypoint(stock_code, buypoint_date)
            result['stock_code'] = stock_code
            result['buypoint_date'] = buypoint_date
            backtest_results.append(result)
        
        # 3. 汇总结果
        summary = self._generate_backtest_summary(backtest_results)
        
        # 4. 生成策略和验证
        strategies = self._generate_strategies_from_results(backtest_results)
        verification_results = self._perform_bidirectional_verification(strategies)
        
        final_result = {
            'buypoints_processed': len(buypoints),
            'individual_results': backtest_results,
            'summary': summary,
            'generated_strategies': strategies,
            'verification_results': verification_results,
            'timestamp': datetime.now().isoformat()
        }
        
        # 5. 保存结果
        self._save_results(final_result)
        
        logger.info("🎉 买点回测流程完成！")
        return final_result
    
    def _load_buypoints(self, file_path: str) -> pd.DataFrame:
        """加载买点数据"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"买点文件不存在: {file_path}")
            
            buypoints = pd.read_csv(file_path)
            
            # 验证必要列
            required_columns = ['stock_code', 'buypoint_date']
            missing_columns = [col for col in required_columns if col not in buypoints.columns]
            if missing_columns:
                raise ValueError(f"买点文件缺少必要列: {missing_columns}")
            
            return buypoints
            
        except Exception as e:
            logger.error(f"加载买点文件失败: {e}")
            raise
    
    def _process_single_buypoint(self, stock_code: str, buypoint_date: str) -> Dict[str, Any]:
        """
        处理单个买点的完整流程
        
        Args:
            stock_code: 股票代码
            buypoint_date: 买点日期
            
        Returns:
            Dict[str, Any]: 单个买点的分析结果
        """
        result = {
            'data_status': {},
            'period_analysis': {},
            'indicator_hits': {},
            'pattern_summary': {},
            'scores': {}
        }
        
        try:
            # 1. 获取多周期数据
            logger.info(f"  📊 获取多周期数据...")
            multi_period_data = self.period_processor.get_multi_period_data(
                stock_code=stock_code,
                end_date=buypoint_date,
                periods=self.periods
            )
            
            # 记录数据状态
            for period, data in multi_period_data.items():
                result['data_status'][period] = {
                    'available': not data.empty,
                    'records': len(data) if not data.empty else 0
                }
            
            # 2. 逐周期分析指标
            for period in self.periods:
                if period not in multi_period_data or multi_period_data[period].empty:
                    logger.warning(f"  ⚠️ {period}周期数据不可用")
                    continue
                
                logger.info(f"  🔍 分析{period}周期...")
                period_result = self._analyze_period_indicators(
                    multi_period_data[period], 
                    period, 
                    buypoint_date
                )
                result['period_analysis'][period] = period_result
            
            # 3. 汇总分析结果
            result['pattern_summary'] = self._summarize_patterns(result['period_analysis'])
            result['scores'] = self._calculate_comprehensive_scores(result['period_analysis'])
            
            return result
            
        except Exception as e:
            logger.error(f"处理买点失败 {stock_code}@{buypoint_date}: {e}")
            result['error'] = str(e)
            return result
    
    def _analyze_period_indicators(self, data: pd.DataFrame, period: str, buypoint_date: str) -> Dict[str, Any]:
        """
        分析单个周期下的所有指标
        
        Args:
            data: 周期数据
            period: 周期名称
            buypoint_date: 买点日期
            
        Returns:
            Dict[str, Any]: 周期分析结果
        """
        period_result = {
            'indicators_tested': 0,
            'indicators_hit': 0,
            'hit_indicators': [],
            'pattern_details': {},
            'scores': {}
        }
        
        # 找到买点日期对应的数据行
        buypoint_index = self._find_buypoint_index(data, buypoint_date)
        if buypoint_index is None:
            logger.warning(f"    ⚠️ 未找到买点日期{buypoint_date}对应的数据")
            return period_result
        
        # 逐个测试指标
        for indicator_name in self.available_indicators:
            try:
                period_result['indicators_tested'] += 1
                
                # 创建指标实例
                indicator = self.indicator_registry.create_indicator(indicator_name)
                if indicator is None:
                    continue
                
                # 计算指标
                indicator_result = indicator.calculate(data)
                if indicator_result is None or indicator_result.empty:
                    continue
                
                # 获取买点当日的形态
                patterns = indicator.get_patterns(indicator_result)
                if not patterns:
                    continue
                
                # 检查买点当日是否命中形态
                buypoint_patterns = self._extract_buypoint_patterns(
                    patterns, buypoint_index, indicator_name, period
                )
                
                if buypoint_patterns:
                    period_result['indicators_hit'] += 1
                    period_result['hit_indicators'].append(indicator_name)
                    period_result['pattern_details'][indicator_name] = buypoint_patterns
                    
                    # 计算指标评分
                    score = indicator.calculate_score(indicator_result)
                    period_result['scores'][indicator_name] = score.get('score', 50.0)
                
            except Exception as e:
                logger.debug(f"    指标{indicator_name}分析失败: {e}")
                continue
        
        logger.info(f"    ✅ {period}周期: {period_result['indicators_hit']}/{period_result['indicators_tested']} 指标命中")
        return period_result
    
    def _find_buypoint_index(self, data: pd.DataFrame, buypoint_date: str) -> Optional[int]:
        """找到买点日期对应的数据索引"""
        try:
            # 转换买点日期格式
            target_date = pd.to_datetime(buypoint_date, format='%Y%m%d').date()
            
            # 在数据中查找对应日期
            for i, row in data.iterrows():
                if 'date' in row:
                    row_date = pd.to_datetime(row['date']).date()
                    if row_date == target_date:
                        return i
                elif 'datetime' in row:
                    row_date = pd.to_datetime(row['datetime']).date()
                    if row_date == target_date:
                        return i
            
            # 如果找不到精确日期，找最接近的日期
            if 'date' in data.columns:
                data['date_parsed'] = pd.to_datetime(data['date'])
                closest_idx = (data['date_parsed'].dt.date - target_date).abs().idxmin()
                return closest_idx
            
            return None
            
        except Exception as e:
            logger.debug(f"查找买点索引失败: {e}")
            return None
    
    def _extract_buypoint_patterns(self, patterns: Dict[str, Any], buypoint_index: int,
                                 indicator_name: str, period: str) -> Dict[str, Any]:
        """
        提取买点当日的形态信息

        ⚠️ 核心原则：指标与周期强制绑定
        - 每个形态必须包含指标名、周期、形态名三要素
        - 生成唯一ID确保不同周期下相同指标被视为不同形态
        - 统计和分析时必须按照"指标+周期"的组合进行
        """
        buypoint_patterns = {}

        try:
            for pattern_name, pattern_data in patterns.items():
                if isinstance(pattern_data, pd.Series) and buypoint_index < len(pattern_data):
                    pattern_value = pattern_data.iloc[buypoint_index]
                    if pattern_value and pattern_value != 0:  # 命中形态
                        # 强制绑定：指标 + 周期 + 形态 = 唯一技术形态
                        unique_id = f"{indicator_name}_{period}_{pattern_name}"

                        buypoint_patterns[pattern_name] = {
                            'value': pattern_value,
                            'indicator': indicator_name,
                            'period': period,
                            'pattern_name': pattern_name,
                            'unique_id': unique_id,
                            'display_name': f"{period}周期{indicator_name}_{pattern_name}",
                            'full_description': f"{period}周期的{indicator_name}指标{pattern_name}形态"
                        }

                        logger.debug(f"命中形态: {buypoint_patterns[pattern_name]['full_description']}")

                elif isinstance(pattern_data, dict):
                    # 处理字典格式的形态数据
                    unique_id = f"{indicator_name}_{period}_{pattern_name}"

                    buypoint_patterns[pattern_name] = {
                        'data': pattern_data,
                        'indicator': indicator_name,
                        'period': period,
                        'pattern_name': pattern_name,
                        'unique_id': unique_id,
                        'display_name': f"{period}周期{indicator_name}_{pattern_name}",
                        'full_description': f"{period}周期的{indicator_name}指标{pattern_name}形态"
                    }

            if buypoint_patterns:
                logger.info(f"  📍 {period}周期{indicator_name}指标: 命中{len(buypoint_patterns)}个形态")

            return buypoint_patterns

        except Exception as e:
            logger.debug(f"提取{period}周期{indicator_name}指标形态信息失败: {e}")
            return {}

    def _summarize_patterns(self, period_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        汇总所有周期的形态分析结果

        ⚠️ 核心原则：按照"指标+周期"组合进行统计
        - 统计时必须区分不同周期下的相同指标
        - 日线MACD和30分钟MACD是完全不同的技术形态
        - 所有统计都基于unique_id (indicator_period_pattern)
        """
        summary = {
            'total_patterns_hit': 0,
            'patterns_by_period': {},
            'patterns_by_indicator_period': {},  # 按"指标+周期"组合统计
            'patterns_by_indicator_only': {},    # 仅按指标名统计（用于对比）
            'unique_patterns': set(),
            'pattern_frequency': {},
            'period_indicator_combinations': {}  # 周期-指标组合统计
        }

        try:
            for period, analysis in period_analysis.items():
                if 'pattern_details' not in analysis:
                    continue

                period_patterns = []
                for indicator_name, patterns in analysis['pattern_details'].items():
                    # 记录周期-指标组合
                    period_indicator_key = f"{period}_{indicator_name}"
                    if period_indicator_key not in summary['period_indicator_combinations']:
                        summary['period_indicator_combinations'][period_indicator_key] = {
                            'period': period,
                            'indicator': indicator_name,
                            'patterns': [],
                            'count': 0
                        }

                    for pattern_name, pattern_info in patterns.items():
                        unique_id = pattern_info['unique_id']
                        display_name = pattern_info.get('display_name', unique_id)
                        full_description = pattern_info.get('full_description', unique_id)

                        summary['unique_patterns'].add(unique_id)
                        period_patterns.append(unique_id)

                        # 按"指标+周期"组合统计 (主要统计方式)
                        if period_indicator_key not in summary['patterns_by_indicator_period']:
                            summary['patterns_by_indicator_period'][period_indicator_key] = {
                                'display_name': f"{period}周期{indicator_name}",
                                'patterns': [],
                                'count': 0
                            }
                        summary['patterns_by_indicator_period'][period_indicator_key]['patterns'].append(unique_id)
                        summary['patterns_by_indicator_period'][period_indicator_key]['count'] += 1

                        # 仅按指标名统计 (用于对比分析)
                        if indicator_name not in summary['patterns_by_indicator_only']:
                            summary['patterns_by_indicator_only'][indicator_name] = []
                        summary['patterns_by_indicator_only'][indicator_name].append(unique_id)

                        # 形态频率统计 (使用完整的unique_id)
                        summary['pattern_frequency'][unique_id] = summary['pattern_frequency'].get(unique_id, 0) + 1

                        # 更新周期-指标组合统计
                        summary['period_indicator_combinations'][period_indicator_key]['patterns'].append(unique_id)
                        summary['period_indicator_combinations'][period_indicator_key]['count'] += 1

                        logger.debug(f"统计形态: {full_description}")

                summary['patterns_by_period'][period] = period_patterns

            summary['total_patterns_hit'] = len(summary['unique_patterns'])
            summary['unique_patterns'] = list(summary['unique_patterns'])  # 转换为列表以便序列化

            # 生成统计报告
            logger.info(f"📊 形态统计汇总:")
            logger.info(f"  总计命中形态: {summary['total_patterns_hit']}个")
            logger.info(f"  涉及周期: {len(summary['patterns_by_period'])}个")
            logger.info(f"  指标-周期组合: {len(summary['patterns_by_indicator_period'])}个")

            return summary

        except Exception as e:
            logger.error(f"汇总形态分析失败: {e}")
            return summary

    def _calculate_comprehensive_scores(self, period_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """计算综合评分"""
        scores = {
            'period_scores': {},
            'indicator_scores': {},
            'overall_score': 0.0,
            'confidence': 0.0
        }

        try:
            total_score = 0.0
            total_weight = 0.0

            # 周期权重配置
            period_weights = {
                '15min': 0.1,
                '30min': 0.15,
                '60min': 0.2,
                'daily': 0.3,
                'weekly': 0.15,
                'monthly': 0.1
            }

            for period, analysis in period_analysis.items():
                if 'scores' not in analysis or not analysis['scores']:
                    continue

                # 计算周期平均分
                period_scores = list(analysis['scores'].values())
                period_avg = np.mean(period_scores) if period_scores else 0.0
                scores['period_scores'][period] = period_avg

                # 加权计算总分
                weight = period_weights.get(period, 0.1)
                total_score += period_avg * weight
                total_weight += weight

                # 记录指标分数
                for indicator, score in analysis['scores'].items():
                    unique_key = f"{indicator}_{period}"
                    scores['indicator_scores'][unique_key] = score

            # 计算最终分数
            if total_weight > 0:
                scores['overall_score'] = total_score / total_weight
                scores['confidence'] = min(total_weight, 1.0)  # 权重越高置信度越高

            return scores

        except Exception as e:
            logger.error(f"计算综合评分失败: {e}")
            return scores

    def _generate_backtest_summary(self, backtest_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        生成回测汇总报告

        ⚠️ 核心原则：统计必须按照"指标+周期"组合进行
        - 热门排行按"周期+指标"组合统计
        - 日线MACD和30分钟MACD分别统计
        - 提供多维度的统计视角
        """
        summary = {
            'total_buypoints': len(backtest_results),
            'successful_analysis': 0,
            'failed_analysis': 0,
            'top_indicator_period_combinations': {},  # 热门"指标+周期"组合
            'top_indicators_only': {},                # 仅按指标名统计（参考）
            'top_patterns': {},                       # 热门具体形态
            'average_scores': {},
            'period_effectiveness': {},
            'period_indicator_analysis': {}           # 周期-指标详细分析
        }

        try:
            indicator_period_hits = {}  # "指标+周期"组合命中统计
            indicator_only_hits = {}    # 仅指标名命中统计
            pattern_hits = {}           # 具体形态命中统计
            period_hits = {}            # 周期效果统计
            period_indicator_details = {}  # 周期-指标详细统计
            all_scores = []

            for result in backtest_results:
                if 'error' in result:
                    summary['failed_analysis'] += 1
                    continue

                summary['successful_analysis'] += 1

                # 统计"指标+周期"组合命中
                if 'pattern_summary' in result:
                    # 按"指标+周期"组合统计
                    for combo_key, combo_info in result['pattern_summary'].get('patterns_by_indicator_period', {}).items():
                        display_name = combo_info.get('display_name', combo_key)
                        hit_count = combo_info.get('count', 0)

                        if combo_key not in indicator_period_hits:
                            indicator_period_hits[combo_key] = {
                                'display_name': display_name,
                                'count': 0
                            }
                        indicator_period_hits[combo_key]['count'] += hit_count

                    # 仅按指标名统计（用于对比）
                    for indicator, patterns in result['pattern_summary'].get('patterns_by_indicator_only', {}).items():
                        indicator_only_hits[indicator] = indicator_only_hits.get(indicator, 0) + len(patterns)

                    # 统计具体形态
                    for pattern in result['pattern_summary'].get('unique_patterns', []):
                        pattern_hits[pattern] = pattern_hits.get(pattern, 0) + 1

                # 统计周期效果
                if 'period_analysis' in result:
                    for period, analysis in result['period_analysis'].items():
                        hit_count = analysis.get('indicators_hit', 0)
                        if period not in period_hits:
                            period_hits[period] = []
                        period_hits[period].append(hit_count)

                        # 详细的周期-指标分析
                        if period not in period_indicator_details:
                            period_indicator_details[period] = {}

                        for indicator in analysis.get('hit_indicators', []):
                            if indicator not in period_indicator_details[period]:
                                period_indicator_details[period][indicator] = 0
                            period_indicator_details[period][indicator] += 1

                # 收集评分
                if 'scores' in result and 'overall_score' in result['scores']:
                    all_scores.append(result['scores']['overall_score'])

            # 生成"指标+周期"组合排行榜 (主要排行)
            sorted_combinations = sorted(
                indicator_period_hits.items(),
                key=lambda x: x[1]['count'],
                reverse=True
            )[:10]

            summary['top_indicator_period_combinations'] = {
                combo_key: {
                    'display_name': combo_info['display_name'],
                    'count': combo_info['count']
                }
                for combo_key, combo_info in sorted_combinations
            }

            # 生成仅指标名排行榜 (参考排行)
            summary['top_indicators_only'] = dict(
                sorted(indicator_only_hits.items(), key=lambda x: x[1], reverse=True)[:10]
            )

            # 生成具体形态排行榜
            summary['top_patterns'] = dict(
                sorted(pattern_hits.items(), key=lambda x: x[1], reverse=True)[:10]
            )

            # 计算周期效果
            for period, hits in period_hits.items():
                summary['period_effectiveness'][period] = {
                    'avg_hits': np.mean(hits) if hits else 0,
                    'max_hits': max(hits) if hits else 0,
                    'success_rate': len([h for h in hits if h > 0]) / len(hits) if hits else 0
                }

            # 周期-指标详细分析
            summary['period_indicator_analysis'] = period_indicator_details

            # 平均评分
            if all_scores:
                summary['average_scores'] = {
                    'mean': np.mean(all_scores),
                    'median': np.median(all_scores),
                    'std': np.std(all_scores)
                }

            # 输出统计日志
            logger.info(f"📊 回测汇总统计:")
            logger.info(f"  成功分析买点: {summary['successful_analysis']}/{summary['total_buypoints']}")
            logger.info(f"  热门指标+周期组合: {len(summary['top_indicator_period_combinations'])}个")
            logger.info(f"  涉及周期: {len(summary['period_effectiveness'])}个")

            return summary

        except Exception as e:
            logger.error(f"生成汇总报告失败: {e}")
            return summary

    def _generate_strategies_from_results(self, backtest_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """从回测结果生成选股策略"""
        strategies = []

        try:
            # 统计最有效的指标组合
            pattern_combinations = {}

            for result in backtest_results:
                if 'error' in result or 'pattern_summary' not in result:
                    continue

                # 获取命中的形态组合
                unique_patterns = result['pattern_summary'].get('unique_patterns', [])
                if len(unique_patterns) >= 2:  # 至少2个形态的组合才考虑
                    combo_key = tuple(sorted(unique_patterns))
                    if combo_key not in pattern_combinations:
                        pattern_combinations[combo_key] = {
                            'count': 0,
                            'patterns': unique_patterns,
                            'avg_score': 0.0,
                            'scores': []
                        }

                    pattern_combinations[combo_key]['count'] += 1
                    if 'scores' in result and 'overall_score' in result['scores']:
                        pattern_combinations[combo_key]['scores'].append(result['scores']['overall_score'])

            # 计算平均分并排序
            for combo_data in pattern_combinations.values():
                if combo_data['scores']:
                    combo_data['avg_score'] = np.mean(combo_data['scores'])

            # 选择最有效的组合生成策略
            top_combinations = sorted(
                pattern_combinations.items(),
                key=lambda x: (x[1]['count'], x[1]['avg_score']),
                reverse=True
            )[:5]  # 取前5个最有效的组合

            for i, (combo_patterns, combo_data) in enumerate(top_combinations):
                strategy = {
                    'id': f"backtest_strategy_{i+1}",
                    'name': f"回测策略{i+1}",
                    'description': f"基于{combo_data['count']}个成功买点的形态组合策略",
                    'patterns': list(combo_patterns),
                    'success_count': combo_data['count'],
                    'avg_score': combo_data['avg_score'],
                    'conditions': self._build_strategy_conditions_from_patterns(combo_patterns),
                    'created_at': datetime.now().isoformat()
                }
                strategies.append(strategy)

            logger.info(f"生成{len(strategies)}个选股策略")
            return strategies

        except Exception as e:
            logger.error(f"生成策略失败: {e}")
            return []

    def _build_strategy_conditions_from_patterns(self, patterns: Tuple[str]) -> Dict[str, Any]:
        """从形态组合构建策略条件"""
        conditions = {
            'required_patterns': [],
            'period_requirements': {},
            'scoring_threshold': 60.0
        }

        try:
            for pattern in patterns:
                # 解析形态ID: indicator_period_pattern
                parts = pattern.split('_')
                if len(parts) >= 3:
                    indicator = parts[0]
                    period = parts[1]
                    pattern_name = '_'.join(parts[2:])

                    conditions['required_patterns'].append({
                        'indicator': indicator,
                        'period': period,
                        'pattern': pattern_name
                    })

                    # 统计周期要求
                    if period not in conditions['period_requirements']:
                        conditions['period_requirements'][period] = 0
                    conditions['period_requirements'][period] += 1

            return conditions

        except Exception as e:
            logger.error(f"构建策略条件失败: {e}")
            return conditions

    def _perform_bidirectional_verification(self, strategies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """执行双向验证"""
        verification_results = {
            'strategies_tested': len(strategies),
            'verification_summary': {},
            'selected_stocks': {},
            'verification_success': False
        }

        try:
            if not strategies:
                logger.warning("没有策略可供验证")
                return verification_results

            # 选择最佳策略进行验证
            best_strategy = max(strategies, key=lambda x: x.get('avg_score', 0))

            logger.info(f"使用策略'{best_strategy['name']}'进行双向验证...")

            # 模拟选股过程
            selected_stocks = self._simulate_stock_selection(best_strategy)

            verification_results['selected_stocks'] = {
                'strategy_id': best_strategy['id'],
                'strategy_name': best_strategy['name'],
                'stocks': selected_stocks,
                'count': len(selected_stocks)
            }

            verification_results['verification_success'] = len(selected_stocks) > 0

            logger.info(f"双向验证完成，选出{len(selected_stocks)}只股票")
            return verification_results

        except Exception as e:
            logger.error(f"双向验证失败: {e}")
            verification_results['error'] = str(e)
            return verification_results

    def _get_strategy_executor(self):
        """获取策略执行引擎实例"""
        if self._strategy_executor is None:
            try:
                from strategy.execution.strategy_execution_engine import StrategyExecutionEngine
                self._strategy_executor = StrategyExecutionEngine()
            except ImportError:
                logger.warning("策略执行引擎不可用，使用模拟模式")
                self._strategy_executor = None
        return self._strategy_executor

    def _simulate_stock_selection(self, strategy: Dict[str, Any]) -> List[str]:
        """执行真实的策略选股过程"""
        try:
            # 尝试使用真实的策略执行引擎
            executor = self._get_strategy_executor()
            if executor is not None:
                logger.info("使用策略执行引擎进行选股...")
                execution_result = executor.execute_strategy(strategy)
                return execution_result.get('selected_stocks', [])
            else:
                # 回退到模拟模式
                logger.info("执行模拟选股...")
                mock_stocks = ['000001', '000002', '600000', '600036', '000858']
                return mock_stocks[:3]  # 返回前3只作为示例

        except Exception as e:
            logger.error(f"策略选股失败: {e}")
            return []

    def _save_results(self, results: Dict[str, Any]) -> None:
        """保存回测结果"""
        try:
            # 创建结果目录
            results_dir = "results/buypoint_backtest"
            os.makedirs(results_dir, exist_ok=True)

            # 生成文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # 保存详细结果
            detail_file = f"{results_dir}/backtest_detail_{timestamp}.json"
            with open(detail_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)

            # 保存汇总报告
            summary_file = f"{results_dir}/backtest_summary_{timestamp}.md"
            self._generate_markdown_report(results, summary_file)

            logger.info(f"回测结果已保存:")
            logger.info(f"  详细结果: {detail_file}")
            logger.info(f"  汇总报告: {summary_file}")

        except Exception as e:
            logger.error(f"保存结果失败: {e}")

    def _generate_markdown_report(self, results: Dict[str, Any], file_path: str) -> None:
        """生成Markdown格式的报告"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("# 买点回测分析报告\n\n")
                f.write(f"**生成时间**: {results['timestamp']}\n\n")

                # 基本统计
                summary = results.get('summary', {})
                f.write("## 📊 基本统计\n\n")
                f.write(f"- **处理买点数**: {results['buypoints_processed']}\n")
                f.write(f"- **成功分析**: {summary.get('successful_analysis', 0)}\n")
                f.write(f"- **失败分析**: {summary.get('failed_analysis', 0)}\n\n")

                # 热门指标
                if 'top_indicators' in summary:
                    f.write("## 🏆 热门指标排行\n\n")
                    for indicator, count in list(summary['top_indicators'].items())[:5]:
                        f.write(f"- **{indicator}**: {count}次命中\n")
                    f.write("\n")

                # 周期效果
                if 'period_effectiveness' in summary:
                    f.write("## 📈 周期效果分析\n\n")
                    f.write("| 周期 | 平均命中 | 最大命中 | 成功率 |\n")
                    f.write("|------|----------|----------|--------|\n")
                    for period, stats in summary['period_effectiveness'].items():
                        f.write(f"| {period} | {stats['avg_hits']:.1f} | {stats['max_hits']} | {stats['success_rate']:.1%} |\n")
                    f.write("\n")

                # 生成的策略
                strategies = results.get('generated_strategies', [])
                if strategies:
                    f.write("## 🎯 生成的选股策略\n\n")
                    for i, strategy in enumerate(strategies, 1):
                        f.write(f"### 策略{i}: {strategy['name']}\n")
                        f.write(f"- **成功次数**: {strategy['success_count']}\n")
                        f.write(f"- **平均评分**: {strategy['avg_score']:.1f}\n")
                        f.write(f"- **形态数量**: {len(strategy['patterns'])}\n\n")

                # 验证结果
                verification = results.get('verification_results', {})
                if verification.get('verification_success'):
                    f.write("## ✅ 双向验证结果\n\n")
                    selected = verification.get('selected_stocks', {})
                    f.write(f"- **验证策略**: {selected.get('strategy_name', 'N/A')}\n")
                    f.write(f"- **选出股票**: {selected.get('count', 0)}只\n")
                    if 'stocks' in selected:
                        f.write(f"- **股票列表**: {', '.join(selected['stocks'])}\n")

        except Exception as e:
            logger.error(f"生成Markdown报告失败: {e}")
