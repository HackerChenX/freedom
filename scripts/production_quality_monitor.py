#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
生产级指标质量监控脚本
专为生产环境设计的高质量监控系统
"""

import sys
import os
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any
import traceback

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.dependency_injection import get_logger

logger = get_logger(__name__)


class ProductionQualityMonitor:
    """生产级指标质量监控器"""
    
    def __init__(self):
        self.results = {}
        self.summary = {
            'total_indicators': 0,
            'excellent_indicators': 0,
            'good_indicators': 0,
            'acceptable_indicators': 0,
            'poor_indicators': 0,
            'failed_indicators': 0,
            'execution_time': 0,
            'test_timestamp': datetime.now().isoformat()
        }
        
        # 生产级质量阈值（更严格）
        self.quality_thresholds = {
            'excellent': 95,    # 优秀
            'good': 90,         # 良好
            'acceptable': 85,   # 可接受
            'poor': 75          # 较差
        }
        
        # 五阶段测试权重
        self.stage_weights = {
            'algorithm_correctness': 25,    # 算法正确性（提高权重）
            'numerical_reasonableness': 20, # 数值合理性（提高权重）
            'functional_completeness': 25,  # 功能完整性（提高权重）
            'performance': 15,              # 性能表现（提高权重）
            'stability': 15                 # 稳定性（提高权重）
        }
    
    def get_production_test_data(self, days: int = 252) -> pd.DataFrame:
        """获取生产级测试数据（一年的交易日数据）"""
        try:
            # 尝试从数据库获取真实数据
            from db.enhanced_connection_pool import ClickHouseConnectionPool
            pool = ClickHouseConnectionPool()
            
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=days+50)).strftime('%Y-%m-%d')
            
            query = f"""
            SELECT code, name, date, open, high, low, close, volume, turnover_rate
            FROM stock_info 
            WHERE code = '000001'
            AND level = '日线'
            AND date >= '{start_date}' AND date <= '{end_date}'
            ORDER BY date ASC
            LIMIT {days}
            """
            
            with pool.get_connection() as conn:
                result = conn.query_dataframe(query)
                
            if len(result) >= 100:  # 至少需要100个数据点
                logger.info(f"✅ 获取到 {len(result)} 个真实数据点")
                return result
            else:
                logger.warning("真实数据不足，使用高质量模拟数据")
                return self._generate_production_mock_data(days)
                
        except Exception as e:
            logger.warning(f"数据库连接失败，使用高质量模拟数据: {e}")
            return self._generate_production_mock_data(days)
    
    def _generate_production_mock_data(self, days: int) -> pd.DataFrame:
        """生成生产级高质量模拟数据"""
        np.random.seed(42)  # 固定种子确保可重复性
        
        dates = pd.date_range(start='2023-01-01', periods=days, freq='B')  # 工作日
        
        # 生成更真实的价格数据
        base_price = 10.0
        returns = np.random.normal(0.0005, 0.02, days)  # 年化收益率约12.5%，波动率约32%
        
        prices = [base_price]
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(max(new_price, 0.1))  # 确保价格为正
        
        # 生成OHLC数据
        opens = []
        highs = []
        lows = []
        closes = prices
        volumes = []
        turnover_rates = []
        
        for i, close in enumerate(closes):
            # 开盘价（基于前一日收盘价）
            if i == 0:
                open_price = close
            else:
                gap = np.random.normal(0, 0.005)  # 跳空
                open_price = closes[i-1] * (1 + gap)
            
            # 日内波动
            intraday_vol = abs(np.random.normal(0, 0.015))
            high = max(open_price, close) * (1 + intraday_vol)
            low = min(open_price, close) * (1 - intraday_vol)
            
            # 成交量（对数正态分布）
            volume = int(np.random.lognormal(14, 0.5))  # 约100万股平均
            
            # 换手率
            turnover_rate = np.random.uniform(0.5, 8.0)
            
            opens.append(open_price)
            highs.append(high)
            lows.append(low)
            volumes.append(volume)
            turnover_rates.append(turnover_rate)
        
        data = pd.DataFrame({
            'date': dates[:len(closes)],
            'code': '000001',
            'name': '平安银行',
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes,
            'turnover_rate': turnover_rates
        })
        
        # 确保数据质量
        data = data.dropna()
        data = data[data['volume'] > 0]
        data = data[data['close'] > 0]
        
        logger.info(f"✅ 生成 {len(data)} 个高质量模拟数据点")
        return data
    
    def test_algorithm_correctness(self, indicator_name: str, indicator, data: pd.DataFrame) -> Tuple[float, str]:
        """阶段1：算法正确性测试（生产级标准）"""
        try:
            result = indicator.calculate(data)
            
            if result is None or result.empty:
                return 0, "计算结果为空"
            
            score = 25
            issues = []
            
            # 检查基本数据完整性
            if result.isnull().any().any():
                null_ratio = result.isnull().sum().sum() / (len(result) * len(result.columns))
                if null_ratio > 0.1:
                    score -= 10
                    issues.append(f"NaN值比例过高: {null_ratio:.2%}")
                elif null_ratio > 0.05:
                    score -= 5
                    issues.append(f"存在NaN值: {null_ratio:.2%}")
                else:
                    score -= 2
                    issues.append("少量NaN值")
            
            # 检查数据长度合理性
            expected_length = len(data) - getattr(indicator, 'period', 20) + 1
            if len(result) < expected_length * 0.8:
                score -= 8
                issues.append(f"结果长度过短: {len(result)}/{expected_length}")
            elif len(result) < expected_length * 0.9:
                score -= 4
                issues.append("结果长度略短")
            
            # 检查数值范围合理性（更严格）
            numeric_cols = result.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                values = result[col].dropna()
                if len(values) > 0:
                    # 检查极值
                    if values.abs().max() > 1000:
                        score -= 5
                        issues.append(f"{col}存在极值")
                    elif values.abs().max() > 100:
                        score -= 2
                        issues.append(f"{col}数值较大")
                    
                    # 检查无穷值
                    if np.isinf(values).any():
                        score -= 8
                        issues.append(f"{col}包含无穷值")
            
            # 检查计算一致性
            if hasattr(indicator, 'period'):
                min_data = data.tail(indicator.period + 50)  # 使用足够的数据
                try:
                    partial_result = indicator.calculate(min_data)
                    if partial_result is not None and not partial_result.empty:
                        # 比较最后几个值的一致性
                        common_cols = set(result.columns) & set(partial_result.columns)
                        for col in common_cols:
                            if col in result.columns and col in partial_result.columns:
                                if len(result) > 0 and len(partial_result) > 0:
                                    last_val1 = result[col].iloc[-1] if not pd.isna(result[col].iloc[-1]) else 0
                                    last_val2 = partial_result[col].iloc[-1] if not pd.isna(partial_result[col].iloc[-1]) else 0
                                    if abs(last_val1 - last_val2) > abs(last_val1) * 0.01:  # 1%误差容忍
                                        score -= 3
                                        issues.append(f"{col}计算不一致")
                                        break
                except:
                    pass  # 部分数据测试失败不影响主要评分
            
            return max(score, 0), "; ".join(issues) if issues else "算法正确"
            
        except Exception as e:
            return 0, f"计算异常: {str(e)[:100]}"
    
    def test_numerical_reasonableness(self, indicator_name: str, indicator, data: pd.DataFrame) -> Tuple[float, str]:
        """阶段2：数值合理性测试（生产级标准）"""
        try:
            result = indicator.calculate(data)
            
            if result is None or result.empty:
                return 0, "无计算结果"
            
            score = 20
            issues = []
            
            numeric_cols = result.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                values = result[col].dropna()
                if len(values) == 0:
                    continue
                
                # 统计特征检查
                mean_val = values.mean()
                std_val = values.std()
                min_val = values.min()
                max_val = values.max()
                
                # 检查分布合理性
                if std_val == 0:
                    score -= 3
                    issues.append(f"{col}无变化")
                elif std_val > abs(mean_val) * 10:  # 标准差过大
                    score -= 2
                    issues.append(f"{col}波动过大")
                
                # 检查异常值比例
                if len(values) > 10:
                    q1, q3 = values.quantile([0.25, 0.75])
                    iqr = q3 - q1
                    if iqr > 0:
                        outliers = values[(values < q1 - 1.5*iqr) | (values > q3 + 1.5*iqr)]
                        outlier_ratio = len(outliers) / len(values)
                        if outlier_ratio > 0.1:
                            score -= 3
                            issues.append(f"{col}异常值过多: {outlier_ratio:.2%}")
                        elif outlier_ratio > 0.05:
                            score -= 1
                            issues.append(f"{col}存在异常值: {outlier_ratio:.2%}")
                
                # 检查趋势合理性（对于某些指标）
                if indicator_name in ['RSI', 'KDJ', 'STOCHRSI']:
                    # 这些指标应该在0-100范围内
                    if min_val < -10 or max_val > 110:
                        score -= 4
                        issues.append(f"{col}超出合理范围[0,100]")
                elif indicator_name in ['MACD']:
                    # MACD应该围绕0震荡
                    if abs(mean_val) > abs(std_val) * 2:
                        score -= 2
                        issues.append(f"{col}偏离零轴过远")
            
            return max(score, 0), "; ".join(issues) if issues else "数值合理"
            
        except Exception as e:
            return 0, f"数值检查异常: {str(e)[:100]}"
    
    def test_functional_completeness(self, indicator_name: str, indicator, data: pd.DataFrame) -> Tuple[float, str]:
        """阶段3：功能完整性测试（生产级标准）"""
        try:
            score = 25
            issues = []
            
            # 测试核心方法
            core_methods = ['calculate']
            for method in core_methods:
                if not hasattr(indicator, method):
                    score -= 10
                    issues.append(f"缺少核心方法{method}")
                else:
                    try:
                        result = getattr(indicator, method)(data)
                        if result is None:
                            score -= 8
                            issues.append(f"{method}返回None")
                        elif hasattr(result, 'empty') and result.empty:
                            score -= 6
                            issues.append(f"{method}返回空结果")
                    except Exception as e:
                        score -= 8
                        issues.append(f"{method}执行异常")
            
            # 测试BaseIndicator接口方法
            base_methods = [
                '_calculate_baseindicator',
                'calculate_raw_score_Indicator_Base_Indicator',
                'get_patterns_Indicator_Base_Indicator',
                'calculate_confidence_Indicator_Base_Indicator'
            ]
            
            implemented_methods = 0
            for method in base_methods:
                if hasattr(indicator, method):
                    implemented_methods += 1
                    try:
                        if method == '_calculate_baseindicator':
                            result = indicator._calculate_baseindicator(data)
                        elif method == 'calculate_raw_score_Indicator_Base_Indicator':
                            score_val = indicator.calculate_raw_score_Indicator_Base_Indicator(data)
                            if not isinstance(score_val, (int, float)) or score_val < 0:
                                score -= 2
                                issues.append(f"{method}返回值不合理")
                        elif method == 'get_patterns_Indicator_Base_Indicator':
                            patterns = indicator.get_patterns_Indicator_Base_Indicator(data)
                            if not isinstance(patterns, pd.DataFrame):
                                score -= 2
                                issues.append(f"{method}返回类型错误")
                        elif method == 'calculate_confidence_Indicator_Base_Indicator':
                            conf = indicator.calculate_confidence_Indicator_Base_Indicator(data)
                            if not isinstance(conf, (int, float)) or conf < 0 or conf > 1:
                                score -= 2
                                issues.append(f"{method}置信度不合理")
                    except Exception as e:
                        score -= 1
                        issues.append(f"{method}执行异常")
            
            # 接口完整性评分
            interface_score = (implemented_methods / len(base_methods)) * 5
            score -= (5 - interface_score)
            
            if implemented_methods < len(base_methods):
                issues.append(f"接口实现不完整: {implemented_methods}/{len(base_methods)}")
            
            # 测试参数设置方法
            if hasattr(indicator, 'set_parameters_Indicator_Base_Indicator'):
                try:
                    indicator.set_parameters_Indicator_Base_Indicator(period=20)
                except Exception as e:
                    score -= 1
                    issues.append("参数设置方法异常")
            
            return max(score, 0), "; ".join(issues) if issues else "功能完整"
            
        except Exception as e:
            return 0, f"功能测试异常: {str(e)[:100]}"
