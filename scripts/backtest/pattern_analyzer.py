#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
技术形态分析系统 - 形态分析器

负责分析技术形态
"""

import os
import sys
import logging
import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple

# 获取项目根目录
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from utils.logger import get_logger
from utils.decorators import performance_monitor, time_it
from utils.technical_utils import (
    calculate_ma, calculate_ema, calculate_macd,
    calculate_kdj, calculate_rsi, calculate_bollinger_bands
)
from indicators.macd import MACDIndicator
from indicators.kdj import KDJIndicator

# 获取日志记录器
logger = get_logger(__name__)


class PatternAnalyzer:
    """
    形态分析器
    
    负责分析技术形态
    """

    def __init__(self):
        """初始化形态分析器"""
        # 注册形态分析器
        self.pattern_analyzers = {
            "macd": MACDIndicator(),
            "kdj": KDJIndicator()
        }
        
        # 形态定义
        self.pattern_definitions = {
            "macd_golden_cross": {
                "pattern_id": "macd_golden_cross",
                "pattern_name": "MACD金叉",
                "description": "MACD快线上穿慢线",
                "analyzer": "macd",
                "params": {
                    "fast_period": 12,
                    "slow_period": 26,
                    "signal_period": 9
                }
            },
            "macd_death_cross": {
                "pattern_id": "macd_death_cross",
                "pattern_name": "MACD死叉",
                "description": "MACD快线下穿慢线",
                "analyzer": "macd",
                "params": {
                    "fast_period": 12,
                    "slow_period": 26,
                    "signal_period": 9
                }
            },
            "kdj_golden_cross": {
                "pattern_id": "kdj_golden_cross",
                "pattern_name": "KDJ金叉",
                "description": "K线上穿D线",
                "analyzer": "kdj",
                "params": {
                    "k_period": 9,
                    "d_period": 3,
                    "j_period": 3
                }
            },
            "kdj_death_cross": {
                "pattern_id": "kdj_death_cross",
                "pattern_name": "KDJ死叉",
                "description": "K线下穿D线",
                "analyzer": "kdj",
                "params": {
                    "k_period": 9,
                    "d_period": 3,
                    "j_period": 3
                }
            }
        }
        
        logger.info("形态分析器初始化完成")
    
    def analyze_all_patterns(self, data: pd.DataFrame, min_strength: float = 0.6) -> List[Dict[str, Any]]:
        """
        分析所有形态
        
        Args:
            data: 股票数据
            min_strength: 最小形态强度
            
        Returns:
            List[Dict[str, Any]]: 形态列表
        """
        patterns = []
        
        # 分析每个形态
        for pattern_id, definition in self.pattern_definitions.items():
            # 获取分析器
            analyzer = self.pattern_analyzers.get(definition["analyzer"])
            if not analyzer:
                continue
            
            # 分析形态
            results = analyzer.analyze_pattern(
                pattern_id=pattern_id,
                data=data
            )
            for result in results:
                if result and result["strength"] >= min_strength:
                    result.update({
                        "pattern_id": pattern_id,
                        "pattern_name": definition["pattern_name"],
                        "description": definition["description"]
                    })
                    patterns.append(result)
        
        return patterns
    
    def analyze_patterns(self, data: pd.DataFrame, pattern_ids: List[str], 
                        min_strength: float = 0.6) -> List[Dict[str, Any]]:
        """
        分析指定形态
        
        Args:
            data: 股票数据
            pattern_ids: 形态ID列表
            min_strength: 最小形态强度
            
        Returns:
            List[Dict[str, Any]]: 形态列表
        """
        patterns = []
        
        # 分析每个形态
        for pattern_id in pattern_ids:
            definition = self.pattern_definitions.get(pattern_id)
            if not definition:
                continue
            
            # 获取分析器
            analyzer = self.pattern_analyzers.get(definition["analyzer"])
            if not analyzer:
                continue
            
            # 分析形态
            results = analyzer.analyze_pattern(
                pattern_id=pattern_id,
                data=data
            )
            for result in results:
                if result and result["strength"] >= min_strength:
                    result.update({
                        "pattern_id": pattern_id,
                        "pattern_name": definition["pattern_name"],
                        "description": definition["description"]
                    })
                    patterns.append(result)
        
        return patterns
    
    def register_pattern(self, pattern_id: str, pattern_name: str, description: str,
                        analyzer: str, params: Dict[str, Any]):
        """
        注册形态
        
        Args:
            pattern_id: 形态ID
            pattern_name: 形态名称
            description: 形态描述
            analyzer: 分析器ID
            params: 分析参数
        """
        if pattern_id in self.pattern_definitions:
            logger.warning(f"形态 {pattern_id} 已存在，将被覆盖")
        
        self.pattern_definitions[pattern_id] = {
            "pattern_id": pattern_id,
            "pattern_name": pattern_name,
            "description": description,
            "analyzer": analyzer,
            "params": params
        }
        logger.info(f"注册形态: {pattern_id} - {pattern_name}")
    
    def register_analyzer(self, analyzer_id: str, analyzer: Any):
        """
        注册分析器
        
        Args:
            analyzer_id: 分析器ID
            analyzer: 分析器对象
        """
        if analyzer_id in self.pattern_analyzers:
            logger.warning(f"分析器 {analyzer_id} 已存在，将被覆盖")
        
        self.pattern_analyzers[analyzer_id] = analyzer
        logger.info(f"注册分析器: {analyzer_id}")
    
    def get_pattern_definition(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """
        获取形态定义
        
        Args:
            pattern_id: 形态ID
            
        Returns:
            Optional[Dict[str, Any]]: 形态定义
        """
        return self.pattern_definitions.get(pattern_id)
    
    def get_analyzer(self, analyzer_id: str) -> Optional[Any]:
        """
        获取分析器
        
        Args:
            analyzer_id: 分析器ID
            
        Returns:
            Optional[Any]: 分析器对象
        """
        return self.pattern_analyzers.get(analyzer_id)
    
    def list_patterns(self) -> Dict[str, str]:
        """
        列出所有形态
        
        Returns:
            Dict[str, str]: 形态ID和名称的映射
        """
        return {
            pattern_id: definition["pattern_name"]
            for pattern_id, definition in self.pattern_definitions.items()
        }
    
    def list_analyzers(self) -> Dict[str, str]:
        """
        列出所有分析器
        
        Returns:
            Dict[str, str]: 分析器ID和类型的映射
        """
        return {
            analyzer_id: analyzer.__class__.__name__
            for analyzer_id, analyzer in self.pattern_analyzers.items()
        }


# 测试代码
if __name__ == "__main__":
    # 初始化形态分析器
    analyzer = PatternAnalyzer()
    
    # 获取形态列表
    patterns = analyzer.list_patterns()
    print(f"可用形态: {patterns}")
    
    # 获取分析器列表
    analyzers = analyzer.list_analyzers()
    print(f"可用分析器: {analyzers}")
    
    # 测试形态分析
    data = pd.DataFrame({
        "date": pd.date_range(start="20220101", end="20220131"),
        "open": np.random.randn(31).cumsum() + 100,
        "high": np.random.randn(31).cumsum() + 101,
        "low": np.random.randn(31).cumsum() + 99,
        "close": np.random.randn(31).cumsum() + 100,
        "volume": np.random.randint(1000, 10000, 31)
    })
    
    # 分析所有形态
    results = analyzer.analyze_all_patterns(data)
    print(f"形态分析结果: {results}")
    
    # 分析指定形态
    results = analyzer.analyze_patterns(data, ["macd_golden_cross", "kdj_golden_cross"])
    print(f"指定形态分析结果: {results}") 