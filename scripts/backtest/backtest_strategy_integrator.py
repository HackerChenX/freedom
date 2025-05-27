#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from utils.logger import get_logger
from utils.path_utils import get_backtest_result_dir, get_strategies_dir
from strategy.strategy_factory import StrategyFactory
from strategy.strategy_manager import StrategyManager
from scripts.backtest.unified_backtest import UnifiedBacktest

# 获取日志记录器
logger = get_logger(__name__)

class BacktestStrategyIntegrator:
    """
    回测选股集成器 - 将回测结果转换为选股策略并进行验证
    
    主要功能：
    1. 分析回测结果，提取关键特征
    2. 生成对应的选股策略配置
    3. 验证策略有效性
    4. 优化策略参数
    """
    
    def __init__(self):
        """初始化回测选股集成器"""
        logger.info("初始化回测选股集成器")
        
        # 获取路径
        self.backtest_dir = get_backtest_result_dir()
        self.strategy_dir = get_strategies_dir()
        
        # 确保目录存在
        os.makedirs(self.backtest_dir, exist_ok=True)
        os.makedirs(self.strategy_dir, exist_ok=True)
        
        # 初始化策略工厂和管理器
        self.strategy_factory = StrategyFactory()
        self.strategy_manager = StrategyManager()
        
        # 初始化回测系统
        self.backtest = UnifiedBacktest()
        
        logger.info("回测选股集成器初始化完成") 

    def analyze_backtest_result(self, backtest_file: str) -> Dict[str, Any]:
        """
        分析回测结果文件，提取关键特征
        
        Args:
            backtest_file: 回测结果文件路径
            
        Returns:
            Dict: 分析结果
        """
        logger.info(f"分析回测结果文件: {backtest_file}")
        
        try:
            # 检查文件是否存在
            if not os.path.exists(backtest_file):
                logger.error(f"回测结果文件不存在: {backtest_file}")
                return {}
                
            # 根据文件扩展名决定如何读取
            if backtest_file.endswith('.json'):
                with open(backtest_file, 'r', encoding='utf-8') as f:
                    backtest_data = json.load(f)
            elif backtest_file.endswith('.csv'):
                backtest_data = {'stocks_data': pd.read_csv(backtest_file).to_dict(orient='records')}
            else:
                logger.error(f"不支持的回测结果文件格式: {backtest_file}")
                return {}
                
            # 提取关键特征
            if 'stocks_data' not in backtest_data or not backtest_data['stocks_data']:
                logger.error(f"回测结果文件缺少stocks_data字段: {backtest_file}")
                return {}
                
            # 处理回测结果
            return self._extract_features(backtest_data)
            
        except Exception as e:
            logger.error(f"分析回测结果时出错: {e}")
            return {}
            
    def _extract_features(self, backtest_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        从回测数据中提取关键特征
        
        Args:
            backtest_data: 回测数据
            
        Returns:
            Dict: 提取的特征
        """
        # 初始化结果
        result = {
            'stock_count': len(backtest_data['stocks_data']),
            'period_features': {},
            'common_patterns': [],
            'strategy_type': self._determine_strategy_type(backtest_data),
            'parameters': {}
        }
        
        # 统计各周期的特征
        for stock_data in backtest_data['stocks_data']:
            if 'periods' not in stock_data:
                continue
                
            # 统计各周期的特征
            for period, period_data in stock_data['periods'].items():
                if period not in result['period_features']:
                    result['period_features'][period] = {
                        'indicators': {},
                        'patterns': {}
                    }
                    
                # 统计指标
                if 'indicators' in period_data:
                    for indicator, values in period_data['indicators'].items():
                        if indicator not in result['period_features'][period]['indicators']:
                            result['period_features'][period]['indicators'][indicator] = {
                                'count': 0,
                                'values': []
                            }
                        
                        result['period_features'][period]['indicators'][indicator]['count'] += 1
                        result['period_features'][period]['indicators'][indicator]['values'].append(values)
                
                # 统计形态
                if 'patterns' in period_data:
                    for pattern in period_data['patterns']:
                        if pattern not in result['period_features'][period]['patterns']:
                            result['period_features'][period]['patterns'][pattern] = 0
                        
                        result['period_features'][period]['patterns'][pattern] += 1
        
        # 计算共性特征
        result['common_patterns'] = self._calculate_common_patterns(result['period_features'])
        
        # 根据共性特征确定策略参数
        result['parameters'] = self._determine_strategy_parameters(
            result['strategy_type'], result['common_patterns'])
        
        return result
        
    def _determine_strategy_type(self, backtest_data: Dict[str, Any]) -> str:
        """
        根据回测数据确定策略类型
        
        Args:
            backtest_data: 回测数据
            
        Returns:
            str: 策略类型
        """
        # 初始化类型计数
        type_counts = {
            '回踩反弹': 0,
            '横盘突破': 0,
            '趋势跟踪': 0,
            '超跌反弹': 0,
            '量价背离': 0
        }
        
        # 统计各种模式
        for stock_data in backtest_data['stocks_data']:
            if 'pattern_type' in stock_data and stock_data['pattern_type']:
                pattern_type = stock_data['pattern_type']
                for key in type_counts:
                    if key in pattern_type:
                        type_counts[key] += 1
            
            # 如果没有明确的pattern_type，尝试从patterns中推断
            elif 'patterns' in stock_data:
                patterns = stock_data['patterns']
                
                # 判断策略类型
                if any('回踩' in p for p in patterns) or any('支撑' in p for p in patterns):
                    type_counts['回踩反弹'] += 1
                elif any('突破' in p for p in patterns) or any('横盘' in p for p in patterns):
                    type_counts['横盘突破'] += 1
                elif any('趋势' in p for p in patterns) or any('突破' in p for p in patterns):
                    type_counts['趋势跟踪'] += 1
                elif any('超跌' in p for p in patterns) or any('反弹' in p for p in patterns):
                    type_counts['超跌反弹'] += 1
                elif any('背离' in p for p in patterns) or any('量价' in p for p in patterns):
                    type_counts['量价背离'] += 1
        
        # 获取最多的类型
        if not type_counts or max(type_counts.values()) == 0:
            return '通用策略'
            
        return max(type_counts.items(), key=lambda x: x[1])[0]
        
    def _calculate_common_patterns(self, period_features: Dict[str, Dict]) -> List[Dict[str, Any]]:
        """
        计算各周期的共性形态
        
        Args:
            period_features: 各周期的特征统计
            
        Returns:
            List[Dict]: 共性形态列表
        """
        # 提取所有形态
        all_patterns = []
        stock_count = 0
        
        for period, features in period_features.items():
            if 'patterns' not in features:
                continue
                
            # 更新股票数量
            pattern_values = list(features['patterns'].values())
            if pattern_values:
                stock_count = max(stock_count, max(pattern_values))
            
            # 添加形态
            for pattern, count in features['patterns'].items():
                all_patterns.append({
                    'period': period,
                    'pattern': pattern,
                    'count': count,
                    'ratio': count / stock_count if stock_count > 0 else 0
                })
        
        # 按比例排序
        all_patterns.sort(key=lambda x: x['ratio'], reverse=True)
        
        # 只保留比例大于30%的形态
        return [p for p in all_patterns if p['ratio'] >= 0.3]
        
    def _determine_strategy_parameters(self, strategy_type: str, 
                                     common_patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        根据策略类型和共性特征确定策略参数
        
        Args:
            strategy_type: 策略类型
            common_patterns: 共性特征列表
            
        Returns:
            Dict: 策略参数
        """
        # 默认参数
        params = {}
        
        # 根据策略类型设置基本参数
        if strategy_type == '回踩反弹':
            params = {
                'ma_period': 20,
                'bounce_min_pct': 1.0,
                'volume_min_ratio': 1.2,
                'touch_threshold': 0.02,
                'kdj_up': False,
                'rsi_bottom': False
            }
            
            # 根据共性特征调整参数
            for p in common_patterns:
                if '均线' in p['pattern']:
                    if 'MA5' in p['pattern'] or '5日' in p['pattern']:
                        params['ma_period'] = 5
                    elif 'MA10' in p['pattern'] or '10日' in p['pattern']:
                        params['ma_period'] = 10
                    elif 'MA20' in p['pattern'] or '20日' in p['pattern']:
                        params['ma_period'] = 20
                    elif 'MA60' in p['pattern'] or '60日' in p['pattern']:
                        params['ma_period'] = 60
                        
                if 'KDJ' in p['pattern'] and ('金叉' in p['pattern'] or '超卖' in p['pattern']):
                    params['kdj_up'] = True
                    
                if 'RSI' in p['pattern'] and '超卖' in p['pattern']:
                    params['rsi_bottom'] = True
                    
                if '成交量' in p['pattern'] and '放大' in p['pattern']:
                    params['volume_min_ratio'] = 1.5
                    
        elif strategy_type == '横盘突破':
            params = {
                'consolidation_days': 20,
                'price_range_pct': 5.0,
                'breakout_pct': 3.0,
                'volume_ratio': 1.5,
                'macd_up': False,
                'boll_use': False
            }
            
            # 根据共性特征调整参数
            for p in common_patterns:
                if '横盘' in p['pattern'] and '天数' in p['pattern']:
                    # 尝试提取天数
                    import re
                    match = re.search(r'(\d+)天', p['pattern'])
                    if match:
                        params['consolidation_days'] = int(match.group(1))
                        
                if 'MACD' in p['pattern'] and ('金叉' in p['pattern'] or '上穿' in p['pattern']):
                    params['macd_up'] = True
                    
                if 'BOLL' in p['pattern']:
                    params['boll_use'] = True
                    
                if '成交量' in p['pattern'] and '放大' in p['pattern']:
                    params['volume_ratio'] = 2.0
        
        # 其他策略类型的参数设置...
        
        return params 

    def generate_strategy(self, backtest_file: str, output_file: str = None) -> Dict[str, Any]:
        """
        根据回测结果生成选股策略
        
        Args:
            backtest_file: 回测结果文件路径
            output_file: 输出策略文件路径，默认为None表示自动生成
            
        Returns:
            Dict: 生成的策略配置
        """
        logger.info(f"根据回测结果生成选股策略: {backtest_file}")
        
        try:
            # 分析回测结果
            features = self.analyze_backtest_result(backtest_file)
            
            if not features:
                logger.error("分析回测结果失败，无法生成策略")
                return {}
                
            # 生成策略名称
            strategy_name = f"{features['strategy_type']}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # 生成策略描述
            description = f"基于{features['stock_count']}只股票的回测结果自动生成的{features['strategy_type']}策略"
            
            # 构建策略配置
            strategy_config = self._build_strategy_config(
                strategy_name, description, features)
            
            # 保存策略配置
            if output_file is None:
                output_file = os.path.join(
                    self.strategy_dir, f"{strategy_name}.json")
            
            # 确保目录存在
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # 保存到文件
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(strategy_config, f, indent=2, ensure_ascii=False)
                
            logger.info(f"选股策略已生成并保存到: {output_file}")
            
            return strategy_config
            
        except Exception as e:
            logger.error(f"生成选股策略时出错: {e}")
            return {}
            
    def _build_strategy_config(self, name: str, description: str, 
                             features: Dict[str, Any]) -> Dict[str, Any]:
        """
        构建策略配置
        
        Args:
            name: 策略名称
            description: 策略描述
            features: 特征分析结果
            
        Returns:
            Dict: 策略配置
        """
        # 初始化策略配置
        strategy_config = {
            "strategy": {
                "id": f"STRATEGY_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "name": name,
                "description": description,
                "version": "1.0",
                "author": "system",
                "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "conditions": [],
                "filters": {
                    "market": ["主板", "科创板", "创业板"],
                    "industry": [],
                    "market_cap": {
                        "min": 50,
                        "max": 5000
                    }
                }
            }
        }
        
        # 添加条件
        strategy_config["strategy"]["conditions"] = self._build_strategy_conditions(features)
        
        return strategy_config
        
    def _build_strategy_conditions(self, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        构建策略条件
        
        Args:
            features: 特征分析结果
            
        Returns:
            List[Dict]: 策略条件列表
        """
        conditions = []
        
        # 根据策略类型和共性特征构建条件
        strategy_type = features['strategy_type']
        common_patterns = features['common_patterns']
        
        # 处理高频率的共性特征
        high_freq_patterns = [p for p in common_patterns if p['ratio'] >= 0.5]
        medium_freq_patterns = [p for p in common_patterns if 0.3 <= p['ratio'] < 0.5]
        
        # 将共性特征转换为条件
        for pattern in high_freq_patterns:
            condition = self._pattern_to_condition(pattern, features['parameters'])
            if condition:
                conditions.append(condition)
                
                # 如果已有条件，添加AND逻辑
                if len(conditions) > 1:
                    conditions.append({"logic": "AND"})
        
        # 处理中频率特征，至少满足一个
        if medium_freq_patterns:
            # 开始分组
            conditions.append({"group": True})
            
            for i, pattern in enumerate(medium_freq_patterns):
                condition = self._pattern_to_condition(pattern, features['parameters'])
                if condition:
                    conditions.append(condition)
                    
                    # 除了最后一个，其他都添加OR逻辑
                    if i < len(medium_freq_patterns) - 1:
                        conditions.append({"logic": "OR"})
            
            # 结束分组
            conditions.append({"end_group": True})
            
            # 如果有高频特征，与高频特征进行AND连接
            if high_freq_patterns:
                conditions.append({"logic": "AND"})
        
        # 添加回测时间过滤器
        date_condition = {
            "indicator_id": "DATE_RANGE",
            "period": "DAILY",
            "parameters": {
                "min_days": 1,
                "max_days": 10
            }
        }
        conditions.append(date_condition)
        
        return conditions
        
    def _pattern_to_condition(self, pattern: Dict[str, Any], 
                            parameters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        将形态转换为条件配置
        
        Args:
            pattern: 形态信息
            parameters: 策略参数
            
        Returns:
            Optional[Dict]: 条件配置，如果无法转换则返回None
        """
        period = pattern['period']
        pattern_name = pattern['pattern']
        
        # 提取对应的枚举类型
        from enums.kline_period import KlinePeriod
        period_enum = None
        for p in KlinePeriod:
            if p.name == period:
                period_enum = p
                break
                
        if not period_enum:
            logger.warning(f"未找到周期枚举: {period}")
            period_enum = KlinePeriod.DAILY
        
        # 根据形态构建条件
        if "MA" in pattern_name and ("上穿" in pattern_name or "金叉" in pattern_name):
            # 提取周期
            fast_period = 5
            slow_period = 20
            
            if "MA5" in pattern_name:
                fast_period = 5
            elif "MA10" in pattern_name:
                fast_period = 10
            elif "MA20" in pattern_name:
                fast_period = 20
                
            if "MA10" in pattern_name and "上穿" in pattern_name:
                slow_period = 10
            elif "MA20" in pattern_name and "上穿" in pattern_name:
                slow_period = 20
            elif "MA60" in pattern_name and "上穿" in pattern_name:
                slow_period = 60
            
            return {
                "indicator_id": "MA_CROSS",
                "period": period_enum.name,
                "parameters": {
                    "fast_period": fast_period,
                    "slow_period": slow_period
                },
                "signal_type": "CROSS_OVER"
            }
            
        elif "KDJ" in pattern_name and "金叉" in pattern_name:
            return {
                "indicator_id": "KDJ_GOLDEN_CROSS",
                "period": period_enum.name,
                "parameters": {
                    "k_period": 9,
                    "d_period": 3,
                    "j_period": 3
                },
                "signal_type": "CROSS_OVER"
            }
            
        elif "RSI" in pattern_name and "超卖" in pattern_name:
            return {
                "indicator_id": "RSI_OVERSOLD",
                "period": period_enum.name,
                "parameters": {
                    "rsi_period": 6,
                    "oversold_threshold": 30
                },
                "signal_type": "BELOW_THRESHOLD"
            }
            
        elif "MACD" in pattern_name and "金叉" in pattern_name:
            return {
                "indicator_id": "MACD_GOLDEN_CROSS",
                "period": period_enum.name,
                "parameters": {
                    "fast_period": 12,
                    "slow_period": 26,
                    "signal_period": 9
                },
                "signal_type": "CROSS_OVER"
            }
            
        elif "BOLL" in pattern_name and "突破" in pattern_name:
            return {
                "indicator_id": "BOLL_BREAKOUT",
                "period": period_enum.name,
                "parameters": {
                    "boll_period": 20,
                    "std_dev": 2
                },
                "signal_type": "UPPER_BREAKOUT"
            }
            
        elif "成交量" in pattern_name and "放大" in pattern_name:
            return {
                "indicator_id": "VOLUME_SURGE",
                "period": period_enum.name,
                "parameters": {
                    "ma_period": 5,
                    "surge_ratio": parameters.get("volume_ratio", 1.5)
                },
                "signal_type": "ABOVE_THRESHOLD"
            }
            
        elif "回踩" in pattern_name and "均线" in pattern_name:
            ma_period = parameters.get("ma_period", 20)
            return {
                "indicator_id": "MA_SUPPORT",
                "period": period_enum.name,
                "parameters": {
                    "ma_period": ma_period,
                    "touch_threshold": parameters.get("touch_threshold", 0.02)
                },
                "signal_type": "PRICE_TOUCH_MA"
            }
            
        # 对于无法转换的形态，返回None
        logger.warning(f"无法转换形态为条件: {pattern_name}")
        return None
        
    def validate_strategy(self, strategy_config: Dict[str, Any], 
                        start_date: str = None, end_date: str = None,
                        stock_pool: List[str] = None) -> Dict[str, Any]:
        """
        验证生成的选股策略
        
        Args:
            strategy_config: 策略配置
            start_date: 开始日期，格式YYYYMMDD
            end_date: 结束日期，格式YYYYMMDD
            stock_pool: 股票池，默认为None表示使用全市场
            
        Returns:
            Dict: 验证结果
        """
        logger.info("开始验证选股策略")
        
        try:
            # 设置默认日期
            if end_date is None:
                end_date = datetime.now().strftime("%Y%m%d")
                
            if start_date is None:
                # 默认验证最近30天
                start_date = (datetime.strptime(end_date, "%Y%m%d") - 
                            timedelta(days=30)).strftime("%Y%m%d")
            
            # 创建策略对象
            from strategy.strategy_executor import StrategyExecutor
            strategy_executor = StrategyExecutor()
            
            # 如果没有指定股票池，获取默认股票池
            if stock_pool is None:
                # 获取全市场股票
                from db.db_manager import DBManager
                db_manager = DBManager.get_instance()
                stock_pool = db_manager.get_all_stock_codes()
                
                # 限制验证股票数量，避免过长时间
                if len(stock_pool) > 300:
                    import random
                    stock_pool = random.sample(stock_pool, 300)
            
            # 执行策略
            result = strategy_executor.execute_strategy(
                strategy_config["strategy"], stock_pool, start_date, end_date)
            
            # 准备验证结果
            validation_result = {
                "strategy_id": strategy_config["strategy"]["id"],
                "strategy_name": strategy_config["strategy"]["name"],
                "validation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "start_date": start_date,
                "end_date": end_date,
                "total_stocks": len(stock_pool),
                "selected_stocks": len(result) if isinstance(result, pd.DataFrame) else 0,
                "selection_ratio": len(result) / len(stock_pool) if isinstance(result, pd.DataFrame) and len(stock_pool) > 0 else 0,
                "stocks": result.to_dict(orient='records') if isinstance(result, pd.DataFrame) else []
            }
            
            logger.info(f"策略验证完成，选出 {validation_result['selected_stocks']} 只股票，"
                      f"选股比例 {validation_result['selection_ratio']:.2%}")
                      
            return validation_result
            
        except Exception as e:
            logger.error(f"验证选股策略时出错: {e}")
            return {
                "strategy_id": strategy_config["strategy"]["id"] if "strategy" in strategy_config else "",
                "strategy_name": strategy_config["strategy"]["name"] if "strategy" in strategy_config else "",
                "validation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "error": str(e)
            }

    def optimize_strategy(self, strategy_config: Dict[str, Any], 
                        validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据验证结果优化策略参数
        
        Args:
            strategy_config: 策略配置
            validation_result: 验证结果
            
        Returns:
            Dict: 优化后的策略配置
        """
        logger.info("开始优化策略参数")
        
        try:
            # 如果选股比例过高或过低，调整参数
            selection_ratio = validation_result.get('selection_ratio', 0)
            
            # 如果选不出股票或选太少，放宽条件
            if selection_ratio < 0.01:
                logger.info("选股比例过低，放宽条件")
                strategy_config = self._relax_strategy_conditions(strategy_config)
                
            # 如果选出太多股票，收紧条件
            elif selection_ratio > 0.1:
                logger.info("选股比例过高，收紧条件")
                strategy_config = self._tighten_strategy_conditions(strategy_config)
                
            # 更新策略版本和更新时间
            strategy_config["strategy"]["version"] = "1.1"
            strategy_config["strategy"]["update_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            return strategy_config
            
        except Exception as e:
            logger.error(f"优化策略参数时出错: {e}")
            return strategy_config
            
    def _relax_strategy_conditions(self, strategy_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        放宽策略条件
        
        Args:
            strategy_config: 策略配置
            
        Returns:
            Dict: 放宽条件后的策略配置
        """
        # 复制配置
        config = strategy_config.copy()
        
        # 放宽条件的通用方法是减少条件数量或降低阈值
        if "conditions" in config["strategy"]:
            conditions = config["strategy"]["conditions"]
            
            # 筛选出非逻辑操作符的条件
            indicator_conditions = [c for c in conditions if "indicator_id" in c]
            
            # 如果条件太多，保留最重要的几个
            if len(indicator_conditions) > 3:
                # 保留前2个条件
                important_conditions = indicator_conditions[:2]
                
                # 重建条件列表
                new_conditions = []
                for c in important_conditions:
                    new_conditions.append(c)
                    # 如果不是最后一个，添加AND
                    if c != important_conditions[-1]:
                        new_conditions.append({"logic": "AND"})
                        
                config["strategy"]["conditions"] = new_conditions
            
            # 对于保留的条件，放宽参数
            for condition in config["strategy"]["conditions"]:
                if "indicator_id" in condition and "parameters" in condition:
                    # 放宽MA周期
                    if condition["indicator_id"] == "MA_CROSS":
                        if "fast_period" in condition["parameters"]:
                            condition["parameters"]["fast_period"] = max(3, condition["parameters"]["fast_period"] - 2)
                            
                    # 放宽RSI阈值
                    elif condition["indicator_id"] == "RSI_OVERSOLD":
                        if "oversold_threshold" in condition["parameters"]:
                            condition["parameters"]["oversold_threshold"] = min(40, condition["parameters"]["oversold_threshold"] + 10)
                            
                    # 放宽成交量要求
                    elif condition["indicator_id"] == "VOLUME_SURGE":
                        if "surge_ratio" in condition["parameters"]:
                            condition["parameters"]["surge_ratio"] = max(1.2, condition["parameters"]["surge_ratio"] - 0.3)
        
        return config
        
    def _tighten_strategy_conditions(self, strategy_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        收紧策略条件
        
        Args:
            strategy_config: 策略配置
            
        Returns:
            Dict: 收紧条件后的策略配置
        """
        # 复制配置
        config = strategy_config.copy()
        
        # 收紧条件的通用方法是增加条件或提高阈值
        if "conditions" in config["strategy"]:
            for condition in config["strategy"]["conditions"]:
                if "indicator_id" in condition and "parameters" in condition:
                    # 收紧MA周期
                    if condition["indicator_id"] == "MA_CROSS":
                        if "slow_period" in condition["parameters"]:
                            condition["parameters"]["slow_period"] = min(60, condition["parameters"]["slow_period"] + 10)
                            
                    # 收紧RSI阈值
                    elif condition["indicator_id"] == "RSI_OVERSOLD":
                        if "oversold_threshold" in condition["parameters"]:
                            condition["parameters"]["oversold_threshold"] = max(20, condition["parameters"]["oversold_threshold"] - 5)
                            
                    # 收紧成交量要求
                    elif condition["indicator_id"] == "VOLUME_SURGE":
                        if "surge_ratio" in condition["parameters"]:
                            condition["parameters"]["surge_ratio"] = min(3.0, condition["parameters"]["surge_ratio"] + 0.5)
            
            # 如果条件少于3个，考虑添加额外条件
            indicator_conditions = [c for c in config["strategy"]["conditions"] if "indicator_id" in c]
            
            if len(indicator_conditions) < 3:
                # 检查是否已有KDJ条件
                has_kdj = any(c["indicator_id"].startswith("KDJ") for c in indicator_conditions)
                
                # 检查是否已有MACD条件
                has_macd = any(c["indicator_id"].startswith("MACD") for c in indicator_conditions)
                
                # 添加缺失的条件
                new_conditions = config["strategy"]["conditions"].copy()
                
                if not has_kdj:
                    # 添加KDJ条件
                    new_conditions.append({"logic": "AND"})
                    new_conditions.append({
                        "indicator_id": "KDJ_GOLDEN_CROSS",
                        "period": "DAILY",
                        "parameters": {
                            "k_period": 9,
                            "d_period": 3,
                            "j_period": 3
                        },
                        "signal_type": "CROSS_OVER"
                    })
                    
                elif not has_macd:
                    # 添加MACD条件
                    new_conditions.append({"logic": "AND"})
                    new_conditions.append({
                        "indicator_id": "MACD_GOLDEN_CROSS",
                        "period": "DAILY",
                        "parameters": {
                            "fast_period": 12,
                            "slow_period": 26,
                            "signal_period": 9
                        },
                        "signal_type": "CROSS_OVER"
                    })
                
                config["strategy"]["conditions"] = new_conditions
        
        return config

def main():
    """命令行入口函数"""
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="回测选股集成工具")
    parser.add_argument('-i', '--input', required=True, help='回测结果文件路径')
    parser.add_argument('-o', '--output', help='输出策略文件路径，默认自动生成')
    parser.add_argument('-v', '--validate', action='store_true', help='是否验证生成的策略')
    parser.add_argument('-s', '--start_date', help='验证开始日期，格式YYYYMMDD')
    parser.add_argument('-e', '--end_date', help='验证结束日期，格式YYYYMMDD')
    parser.add_argument('-p', '--pool', help='验证股票池文件，每行一个股票代码')
    parser.add_argument('-a', '--optimize', action='store_true', help='是否优化策略参数')
    
    args = parser.parse_args()
    
    # 创建集成器并执行操作
    integrator = BacktestStrategyIntegrator()
    
    # 生成策略
    strategy_config = integrator.generate_strategy(args.input, args.output)
    
    if not strategy_config:
        logger.error("生成策略失败")
        return
    
    # 验证策略
    if args.validate:
        # 准备股票池
        stock_pool = None
        if args.pool:
            try:
                with open(args.pool, 'r', encoding='utf-8') as f:
                    stock_pool = [line.strip() for line in f if line.strip()]
            except Exception as e:
                logger.error(f"读取股票池文件时出错: {e}")
        
        # 执行验证
        validation_result = integrator.validate_strategy(
            strategy_config, args.start_date, args.end_date, stock_pool)
        
        # 显示验证结果
        logger.info(f"验证结果: 选出 {validation_result.get('selected_stocks', 0)} 只股票，"
                  f"选股比例 {validation_result.get('selection_ratio', 0):.2%}")
        
        # 优化策略
        if args.optimize and validation_result:
            optimized_config = integrator.optimize_strategy(strategy_config, validation_result)
            
            # 保存优化后的策略
            if args.output:
                output_file = args.output.replace('.json', '_optimized.json')
            else:
                output_file = os.path.join(
                    integrator.strategy_dir, 
                    f"{strategy_config['strategy']['name']}_optimized.json")
                
            # 保存到文件
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(optimized_config, f, indent=2, ensure_ascii=False)
                
            logger.info(f"优化后的策略已保存到: {output_file}")
    
if __name__ == "__main__":
    main() 