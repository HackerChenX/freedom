#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试技术指标形态检测功能
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from indicators.boll import BOLL
from indicators.kdj import KDJ
from indicators.pattern_registry import PatternRegistry, PatternType, PatternStrength

# 导入日志模块
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)


def generate_test_data(days=100):
    """生成测试数据"""
    dates = pd.date_range(end=datetime.now(), periods=days)
    
    # 生成随机价格数据
    np.random.seed(42)  # 设置随机种子，使结果可复现
    
    # 生成带有趋势的随机价格
    close = np.random.normal(0, 1, days).cumsum() + 100
    
    # 生成高低开价格
    high = close + np.random.uniform(0.5, 2, days)
    low = close - np.random.uniform(0.5, 2, days)
    open_price = low + np.random.uniform(0, 1, days) * (high - low)
    
    # 生成成交量数据，与价格有一定相关性
    volume = np.random.normal(500000, 200000, days) + np.abs(close - np.roll(close, 1)) * 10000
    volume[0] = volume[1]  # 修复第一个数据点
    
    # 创建DataFrame
    df = pd.DataFrame({
        'date': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    return df


def test_patterns():
    """测试各指标的形态检测功能"""
    # 生成测试数据
    data = generate_test_data(days=100)
    
    # 清空PatternRegistry
    PatternRegistry.clear_registry()
    
    # 创建各指标实例
    indicators = {
        'BOLL': BOLL(),
        'KDJ': KDJ()
    }
    
    # 计算指标并注册形态
    for name, indicator in indicators.items():
        logger.info(f"计算 {name} 指标...")
        indicator.calculate(data)
        
        # 调用各指标的形态注册方法
        if name == 'BOLL':
            indicator._register_boll_patterns()
        elif name == 'KDJ':
            indicator._register_kdj_patterns()
    
    # 获取PatternRegistry实例
    registry = PatternRegistry()
    
    # 检测最近的数据是否有形态
    latest_data = data.iloc[-30:].copy()  # 取最近的30条数据进行检测
    
    logger.info("\n检测到的形态:")
    for name, indicator in indicators.items():
        # 获取该指标注册的所有形态
        patterns = indicator.get_registered_patterns()
        
        # 检测形态
        detected_patterns = []
        for pattern_id, pattern_info in patterns.items():
            # 获取检测函数
            detection_func = pattern_info.get('detection_func')
            if detection_func and callable(detection_func):
                try:
                    # 调用检测函数
                    is_detected = detection_func(latest_data)
                    if is_detected:
                        detected_patterns.append({
                            'pattern_id': pattern_id,
                            'display_name': pattern_info.get('display_name', ''),
                            'score_impact': pattern_info.get('score_impact', 0)
                        })
                except Exception as e:
                    logger.error(f"检测形态 {pattern_id} 时发生错误: {str(e)}")
        
        # 输出检测结果
        if detected_patterns:
            logger.info(f"\n{name} 指标检测到 {len(detected_patterns)} 个形态:")
            for p in detected_patterns:
                logger.info(f"  - {p['display_name']} (ID: {p['pattern_id']}, 得分影响: {p['score_impact']})")
        else:
            logger.info(f"\n{name} 指标未检测到任何形态")
    
    # 统计各类型形态的数量
    pattern_types = {pt.name: 0 for pt in PatternType}
    all_patterns = registry.get_all_patterns()
    
    for pattern_info in all_patterns.values():
        pattern_type = pattern_info.get('pattern_type')
        if pattern_type:
            pattern_types[pattern_type.name] += 1
    
    logger.info("\n形态类型统计:")
    for type_name, count in pattern_types.items():
        logger.info(f"  - {type_name}: {count} 个形态")
    
    # 打印所有注册的形态
    logger.info("\n所有注册的形态:")
    for pattern_id, pattern_info in all_patterns.items():
        logger.info(f"  - {pattern_id}: {pattern_info.get('display_name')} ({pattern_info.get('indicator_id')})")


if __name__ == "__main__":
    test_patterns() 