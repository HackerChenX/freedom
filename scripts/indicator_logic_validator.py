#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
指标逻辑验证工具

深入检查每个指标的计算逻辑和选股条件的合理性
提供详细的调试信息和修复建议
"""

import sys
import os
import time
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import json

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from db.unified_data_manager import get_unified_data_manager
from analysis.engines.unified_indicator_engine import UnifiedIndicatorEngine
from analysis.engines.shared_condition_evaluator import SharedConditionEvaluator
from indicators.complete_indicator_registry import complete_registry
from utils.logger import get_logger
from utils.date_utils import get_latest_trading_date
from utils.decorators import performance_monitor

logger = get_logger(__name__)


class IndicatorLogicValidator:
    """指标逻辑验证器"""
    
    def __init__(self, debug_mode: bool = True):
        """
        初始化验证器
        
        Args:
            debug_mode: 是否启用调试模式
        """
        self.debug_mode = debug_mode
        self.data_manager = get_unified_data_manager()
        self.indicator_engine = UnifiedIndicatorEngine(enable_cache=True)
        self.condition_evaluator = SharedConditionEvaluator(self.indicator_engine)
        
        # 验证统计
        self.validation_stats = {
            'total_indicators': 0,
            'valid_indicators': 0,
            'invalid_indicators': 0,
            'calculation_errors': 0,
            'logic_errors': 0,
            'data_errors': 0,
            'validation_results': {}
        }
        
        # 获取测试数据 - 使用数据库中实际存在的日期
        # self.test_date = get_latest_trading_date()
        self.test_date = "2025-05-23"  # 使用数据库中实际存在的最新日期
        self.test_stocks = self._get_test_stocks()
        
        logger.info(f"验证器初始化完成，测试日期: {self.test_date}")
    
    def _get_test_stocks(self) -> List[str]:
        """获取测试股票"""
        try:
            query = f"""
            SELECT code as stock_code, close, volume, turnover_rate
            FROM stock_info 
            WHERE date = '{self.test_date}'
            AND level = '日线'
            AND close > 5.0 AND close < 100.0
            AND volume > 1000000
            AND turnover_rate > 0.5
            ORDER BY volume DESC
            LIMIT 20
            """
            
            result = self.data_manager.query(query)
            if not result.empty:
                return result['stock_code'].tolist()
            else:
                return ['000001', '000002', '000858', '002415', '600000']
                
        except Exception as e:
            logger.error(f"获取测试股票失败: {e}")
            return ['000001', '000002', '000858', '002415', '600000']
    
    def _get_stock_data(self, stock_code: str, days: int = 250) -> pd.DataFrame:
        """获取股票历史数据"""
        try:
            end_date = self.test_date
            start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=days)).strftime('%Y-%m-%d')
            
            query = f"""
            SELECT date, open, high, low, close, volume, 0 as amount, turnover_rate
            FROM stock_info 
            WHERE code = '{stock_code}'
            AND level = '日线'
            AND date >= '{start_date}' AND date <= '{end_date}'
            ORDER BY date ASC
            """
            
            data = self.data_manager.query(query)
            if data.empty:
                logger.warning(f"股票 {stock_code} 无历史数据")
                return pd.DataFrame()
            
            # 数据类型转换
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'amount', 'turnover_rate']
            for col in numeric_columns:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
            
            data['date'] = pd.to_datetime(data['date'])
            data = data.sort_values('date')
            
            return data
            
        except Exception as e:
            logger.error(f"获取股票 {stock_code} 数据失败: {e}")
            return pd.DataFrame()
    
    def validate_single_indicator(self, indicator_name: str, stock_code: str = None) -> Dict[str, Any]:
        """验证单个指标"""
        if stock_code is None:
            stock_code = self.test_stocks[0]
        
        validation_result = {
            'indicator_name': indicator_name,
            'stock_code': stock_code,
            'validation_passed': False,
            'calculation_success': False,
            'logic_validation': False,
            'data_quality': False,
            'error_messages': [],
            'warnings': [],
            'debug_info': {},
            'sample_values': {}
        }
        
        try:
            # 1. 获取测试数据
            stock_data = self._get_stock_data(stock_code)
            if stock_data.empty:
                validation_result['error_messages'].append(f"无法获取股票 {stock_code} 的数据")
                return validation_result
            
            validation_result['data_quality'] = True
            validation_result['debug_info']['data_shape'] = stock_data.shape
            validation_result['debug_info']['date_range'] = {
                'start': str(stock_data['date'].min()),
                'end': str(stock_data['date'].max())
            }
            
            # 2. 尝试计算指标
            try:
                indicator_result = self.indicator_engine.calculate_indicator(
                    indicator_name, stock_data
                )
                
                if indicator_result is not None and not indicator_result.empty:
                    validation_result['calculation_success'] = True
                    validation_result['debug_info']['result_shape'] = indicator_result.shape
                    validation_result['debug_info']['result_columns'] = list(indicator_result.columns)
                    
                    # 获取样本值
                    if len(indicator_result) > 0:
                        sample_data = indicator_result.tail(5)
                        validation_result['sample_values'] = sample_data.to_dict('records')
                    
                    # 3. 验证计算逻辑
                    logic_validation = self._validate_indicator_logic(
                        indicator_name, stock_data, indicator_result
                    )
                    validation_result.update(logic_validation)
                    
                else:
                    validation_result['error_messages'].append("指标计算返回空结果")
                    
            except Exception as calc_error:
                validation_result['error_messages'].append(f"指标计算失败: {str(calc_error)}")
                validation_result['debug_info']['calculation_error'] = str(calc_error)
                if self.debug_mode:
                    validation_result['debug_info']['calculation_traceback'] = traceback.format_exc()
            
            # 4. 验证选股条件
            if validation_result['calculation_success']:
                condition_validation = self._validate_selection_condition(
                    indicator_name, stock_code, stock_data
                )
                validation_result.update(condition_validation)
            
            # 5. 综合评估
            validation_result['validation_passed'] = (
                validation_result['data_quality'] and
                validation_result['calculation_success'] and
                validation_result['logic_validation']
            )
            
        except Exception as e:
            validation_result['error_messages'].append(f"验证过程失败: {str(e)}")
            if self.debug_mode:
                validation_result['debug_info']['validation_error'] = traceback.format_exc()
        
        return validation_result
    
    def _validate_indicator_logic(self, indicator_name: str, stock_data: pd.DataFrame, 
                                indicator_result: pd.DataFrame) -> Dict[str, Any]:
        """验证指标计算逻辑"""
        logic_result = {
            'logic_validation': False,
            'logic_checks': {},
            'warnings': []
        }
        
        try:
            # 通用逻辑检查
            checks = {
                'non_empty_result': len(indicator_result) > 0,
                'no_all_nan': not indicator_result.isnull().all().all(),
                'reasonable_length': len(indicator_result) >= min(20, len(stock_data) * 0.5),
                'date_alignment': True  # 默认通过，具体检查在下面
            }
            
            # 日期对齐检查
            if 'date' in indicator_result.columns and 'date' in stock_data.columns:
                stock_dates = set(stock_data['date'].dt.strftime('%Y-%m-%d'))
                indicator_dates = set(indicator_result['date'].dt.strftime('%Y-%m-%d'))
                common_dates = stock_dates.intersection(indicator_dates)
                checks['date_alignment'] = len(common_dates) > 0
            
            # 特定指标的逻辑检查
            if indicator_name == 'MA':
                checks.update(self._validate_ma_logic(stock_data, indicator_result))
            elif indicator_name == 'RSI':
                checks.update(self._validate_rsi_logic(stock_data, indicator_result))
            elif indicator_name == 'MACD':
                checks.update(self._validate_macd_logic(stock_data, indicator_result))
            elif indicator_name == 'KDJ':
                checks.update(self._validate_kdj_logic(stock_data, indicator_result))
            elif indicator_name == 'BOLL':
                checks.update(self._validate_boll_logic(stock_data, indicator_result))
            
            logic_result['logic_checks'] = checks
            logic_result['logic_validation'] = all(checks.values())
            
            # 生成警告
            for check_name, passed in checks.items():
                if not passed:
                    logic_result['warnings'].append(f"逻辑检查失败: {check_name}")
            
        except Exception as e:
            logic_result['warnings'].append(f"逻辑验证异常: {str(e)}")
        
        return logic_result
    
    def _validate_ma_logic(self, stock_data: pd.DataFrame, ma_result: pd.DataFrame) -> Dict[str, bool]:
        """验证移动平均线逻辑"""
        checks = {}
        
        try:
            # 检查MA值是否在合理范围内
            if 'MA' in ma_result.columns:
                ma_values = ma_result['MA'].dropna()
                close_values = stock_data['close']
                
                if len(ma_values) > 0 and len(close_values) > 0:
                    ma_mean = ma_values.mean()
                    close_mean = close_values.mean()
                    
                    # MA应该接近收盘价的平均值
                    checks['ma_reasonable_range'] = abs(ma_mean - close_mean) / close_mean < 0.5
                    
                    # MA应该是平滑的（相邻值变化不应太大）
                    ma_diff = ma_values.diff().abs()
                    max_diff = ma_diff.max()
                    avg_price = close_values.mean()
                    checks['ma_smoothness'] = max_diff < avg_price * 0.1
                else:
                    checks['ma_reasonable_range'] = False
                    checks['ma_smoothness'] = False
            else:
                checks['ma_reasonable_range'] = False
                checks['ma_smoothness'] = False
                
        except Exception:
            checks['ma_reasonable_range'] = False
            checks['ma_smoothness'] = False
        
        return checks
    
    def _validate_rsi_logic(self, stock_data: pd.DataFrame, rsi_result: pd.DataFrame) -> Dict[str, bool]:
        """验证RSI逻辑"""
        checks = {}
        
        try:
            if 'RSI' in rsi_result.columns:
                rsi_values = rsi_result['RSI'].dropna()
                
                if len(rsi_values) > 0:
                    # RSI应该在0-100范围内
                    checks['rsi_range'] = (rsi_values >= 0).all() and (rsi_values <= 100).all()
                    
                    # RSI应该有合理的分布
                    rsi_mean = rsi_values.mean()
                    checks['rsi_distribution'] = 20 <= rsi_mean <= 80
                else:
                    checks['rsi_range'] = False
                    checks['rsi_distribution'] = False
            else:
                checks['rsi_range'] = False
                checks['rsi_distribution'] = False
                
        except Exception:
            checks['rsi_range'] = False
            checks['rsi_distribution'] = False
        
        return checks
    
    def _validate_macd_logic(self, stock_data: pd.DataFrame, macd_result: pd.DataFrame) -> Dict[str, bool]:
        """验证MACD逻辑"""
        checks = {}
        
        try:
            required_columns = ['DIF', 'DEA', 'MACD']
            has_columns = all(col in macd_result.columns for col in required_columns)
            checks['macd_columns'] = has_columns
            
            if has_columns:
                dif = macd_result['DIF'].dropna()
                dea = macd_result['DEA'].dropna()
                macd = macd_result['MACD'].dropna()
                
                if len(dif) > 0 and len(dea) > 0 and len(macd) > 0:
                    # MACD = 2 * (DIF - DEA)
                    if len(dif) == len(dea):
                        calculated_macd = 2 * (dif - dea)
                        # 允许一定的计算误差
                        macd_aligned = macd.iloc[-len(calculated_macd):]
                        diff = abs(calculated_macd - macd_aligned).mean()
                        checks['macd_formula'] = diff < 0.001
                    else:
                        checks['macd_formula'] = False
                else:
                    checks['macd_formula'] = False
            else:
                checks['macd_formula'] = False
                
        except Exception:
            checks['macd_columns'] = False
            checks['macd_formula'] = False
        
        return checks
    
    def _validate_kdj_logic(self, stock_data: pd.DataFrame, kdj_result: pd.DataFrame) -> Dict[str, bool]:
        """验证KDJ逻辑"""
        checks = {}
        
        try:
            required_columns = ['K', 'D', 'J']
            has_columns = all(col in kdj_result.columns for col in required_columns)
            checks['kdj_columns'] = has_columns
            
            if has_columns:
                k_values = kdj_result['K'].dropna()
                d_values = kdj_result['D'].dropna()
                j_values = kdj_result['J'].dropna()
                
                if len(k_values) > 0 and len(d_values) > 0 and len(j_values) > 0:
                    # K、D值应该在0-100范围内
                    checks['kd_range'] = (
                        (k_values >= 0).all() and (k_values <= 100).all() and
                        (d_values >= 0).all() and (d_values <= 100).all()
                    )
                    
                    # J值可以超出0-100范围，但不应过于极端
                    checks['j_reasonable'] = (j_values >= -50).all() and (j_values <= 150).all()
                else:
                    checks['kd_range'] = False
                    checks['j_reasonable'] = False
            else:
                checks['kd_range'] = False
                checks['j_reasonable'] = False
                
        except Exception:
            checks['kdj_columns'] = False
            checks['kd_range'] = False
            checks['j_reasonable'] = False
        
        return checks
    
    def _validate_boll_logic(self, stock_data: pd.DataFrame, boll_result: pd.DataFrame) -> Dict[str, bool]:
        """验证布林带逻辑"""
        checks = {}
        
        try:
            required_columns = ['UPPER', 'MIDDLE', 'LOWER']
            has_columns = all(col in boll_result.columns for col in required_columns)
            checks['boll_columns'] = has_columns
            
            if has_columns:
                upper = boll_result['UPPER'].dropna()
                middle = boll_result['MIDDLE'].dropna()
                lower = boll_result['LOWER'].dropna()
                
                if len(upper) > 0 and len(middle) > 0 and len(lower) > 0:
                    # 上轨 > 中轨 > 下轨
                    min_len = min(len(upper), len(middle), len(lower))
                    upper_aligned = upper.iloc[-min_len:]
                    middle_aligned = middle.iloc[-min_len:]
                    lower_aligned = lower.iloc[-min_len:]
                    
                    checks['boll_order'] = (
                        (upper_aligned >= middle_aligned).all() and
                        (middle_aligned >= lower_aligned).all()
                    )
                    
                    # 中轨应该接近移动平均线
                    if 'close' in stock_data.columns:
                        close_ma = stock_data['close'].rolling(20).mean().dropna()
                        if len(close_ma) > 0 and len(middle_aligned) > 0:
                            # 对齐数据长度
                            align_len = min(len(close_ma), len(middle_aligned))
                            close_ma_aligned = close_ma.iloc[-align_len:]
                            middle_test = middle_aligned.iloc[-align_len:]
                            
                            diff = abs(close_ma_aligned - middle_test).mean()
                            avg_price = close_ma_aligned.mean()
                            checks['middle_line_accuracy'] = diff < avg_price * 0.01
                        else:
                            checks['middle_line_accuracy'] = False
                    else:
                        checks['middle_line_accuracy'] = False
                else:
                    checks['boll_order'] = False
                    checks['middle_line_accuracy'] = False
            else:
                checks['boll_order'] = False
                checks['middle_line_accuracy'] = False
                
        except Exception:
            checks['boll_columns'] = False
            checks['boll_order'] = False
            checks['middle_line_accuracy'] = False
        
        return checks
    
    def _validate_selection_condition(self, indicator_name: str, stock_code: str, 
                                    stock_data: pd.DataFrame) -> Dict[str, Any]:
        """验证选股条件"""
        condition_result = {
            'condition_validation': False,
            'condition_tests': {},
            'selection_reasonable': False
        }
        
        try:
            # 创建测试条件
            test_conditions = self._create_test_conditions(indicator_name)
            
            for condition_name, condition in test_conditions.items():
                try:
                    # 评估条件
                    result = self.condition_evaluator.evaluate_condition(condition, stock_data)
                    
                    condition_result['condition_tests'][condition_name] = {
                        'success': True,
                        'result_type': type(result).__name__,
                        'result_value': bool(result) if isinstance(result, (bool, np.bool_)) else None
                    }
                    
                except Exception as e:
                    condition_result['condition_tests'][condition_name] = {
                        'success': False,
                        'error': str(e)
                    }
            
            # 检查是否至少有一个条件成功
            successful_conditions = sum(1 for test in condition_result['condition_tests'].values() if test['success'])
            condition_result['condition_validation'] = successful_conditions > 0
            condition_result['selection_reasonable'] = successful_conditions > 0
            
        except Exception as e:
            condition_result['condition_tests']['general_error'] = str(e)
        
        return condition_result
    
    def _create_test_conditions(self, indicator_name: str) -> Dict[str, Dict[str, Any]]:
        """为指标创建测试条件"""
        conditions = {}
        
        # 基础条件模板
        if indicator_name == 'MA':
            conditions['ma_above_close'] = {
                'type': 'indicator',
                'indicator_id': 'MA',
                'period': 20,
                'operator': '>',
                'reference_field': 'close'
            }
            conditions['ma_trend_up'] = {
                'type': 'indicator',
                'indicator_id': 'MA',
                'period': 5,
                'operator': '>',
                'reference_indicator': 'MA',
                'reference_period': 20
            }
        
        elif indicator_name == 'RSI':
            conditions['rsi_oversold'] = {
                'type': 'indicator',
                'indicator_id': 'RSI',
                'period': 14,
                'operator': '<',
                'value': 30
            }
            conditions['rsi_overbought'] = {
                'type': 'indicator',
                'indicator_id': 'RSI',
                'period': 14,
                'operator': '>',
                'value': 70
            }
        
        elif indicator_name == 'MACD':
            conditions['macd_golden_cross'] = {
                'type': 'indicator',
                'indicator_id': 'MACD',
                'field': 'DIF',
                'operator': '>',
                'reference_field': 'DEA'
            }
            conditions['macd_above_zero'] = {
                'type': 'indicator',
                'indicator_id': 'MACD',
                'field': 'DIF',
                'operator': '>',
                'value': 0
            }
        
        elif indicator_name == 'KDJ':
            conditions['kdj_golden_cross'] = {
                'type': 'indicator',
                'indicator_id': 'KDJ',
                'field': 'K',
                'operator': '>',
                'reference_field': 'D'
            }
            conditions['kdj_oversold'] = {
                'type': 'indicator',
                'indicator_id': 'KDJ',
                'field': 'K',
                'operator': '<',
                'value': 20
            }
        
        else:
            # 通用条件
            conditions['indicator_positive'] = {
                'type': 'indicator',
                'indicator_id': indicator_name,
                'operator': '>',
                'value': 0
            }
        
        return conditions
    
    def validate_all_indicators(self, sample_stocks: int = 3) -> Dict[str, Any]:
        """验证所有指标"""
        logger.info("🚀 开始验证所有指标")
        
        # 获取所有指标
        all_indicators = complete_registry.get_indicator_names()
        self.validation_stats['total_indicators'] = len(all_indicators)
        
        # 选择测试股票
        test_stocks = self.test_stocks[:sample_stocks]
        
        logger.info(f"准备验证 {len(all_indicators)} 个指标，使用 {len(test_stocks)} 只测试股票")
        
        for i, indicator_name in enumerate(all_indicators, 1):
            logger.info(f"验证进度: {i}/{len(all_indicators)} - {indicator_name}")
            
            # 对每个测试股票验证指标
            indicator_results = []
            for stock_code in test_stocks:
                result = self.validate_single_indicator(indicator_name, stock_code)
                indicator_results.append(result)
            
            # 汇总结果
            overall_result = self._summarize_indicator_results(indicator_name, indicator_results)
            self.validation_stats['validation_results'][indicator_name] = overall_result
            
            # 更新统计
            if overall_result['overall_validation']:
                self.validation_stats['valid_indicators'] += 1
            else:
                self.validation_stats['invalid_indicators'] += 1
                
                # 分类错误类型
                if any('calculation' in error.lower() for error in overall_result.get('common_errors', [])):
                    self.validation_stats['calculation_errors'] += 1
                if any('logic' in error.lower() for error in overall_result.get('common_errors', [])):
                    self.validation_stats['logic_errors'] += 1
                if any('data' in error.lower() for error in overall_result.get('common_errors', [])):
                    self.validation_stats['data_errors'] += 1
        
        return self._generate_validation_report()
    
    def _summarize_indicator_results(self, indicator_name: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """汇总单个指标的验证结果"""
        summary = {
            'indicator_name': indicator_name,
            'total_tests': len(results),
            'successful_tests': sum(1 for r in results if r['validation_passed']),
            'calculation_success_rate': sum(1 for r in results if r['calculation_success']) / len(results),
            'logic_validation_rate': sum(1 for r in results if r['logic_validation']) / len(results),
            'overall_validation': False,
            'common_errors': [],
            'sample_results': results[:2]  # 保留前两个详细结果作为样本
        }
        
        # 收集常见错误
        all_errors = []
        for result in results:
            all_errors.extend(result.get('error_messages', []))
        
        # 统计错误频率
        error_counts = {}
        for error in all_errors:
            error_counts[error] = error_counts.get(error, 0) + 1
        
        # 获取最常见的错误
        summary['common_errors'] = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # 判断整体验证是否通过
        success_threshold = 0.8  # 80%的测试通过才算验证通过
        summary['overall_validation'] = (summary['successful_tests'] / summary['total_tests']) >= success_threshold
        
        return summary
    
    def _generate_validation_report(self) -> Dict[str, Any]:
        """生成验证报告"""
        success_rate = (
            self.validation_stats['valid_indicators'] / self.validation_stats['total_indicators'] * 100
            if self.validation_stats['total_indicators'] > 0 else 0
        )
        
        # 分析失败原因
        failure_analysis = self._analyze_failures()
        
        # 生成修复建议
        fix_suggestions = self._generate_fix_suggestions()
        
        report = {
            'validation_summary': {
                'total_indicators': self.validation_stats['total_indicators'],
                'valid_indicators': self.validation_stats['valid_indicators'],
                'invalid_indicators': self.validation_stats['invalid_indicators'],
                'success_rate': round(success_rate, 2),
                'calculation_errors': self.validation_stats['calculation_errors'],
                'logic_errors': self.validation_stats['logic_errors'],
                'data_errors': self.validation_stats['data_errors']
            },
            'failure_analysis': failure_analysis,
            'fix_suggestions': fix_suggestions,
            'detailed_results': self.validation_stats['validation_results'],
            'test_configuration': {
                'test_date': self.test_date,
                'test_stocks': self.test_stocks,
                'debug_mode': self.debug_mode
            }
        }
        
        return report
    
    def _analyze_failures(self) -> Dict[str, Any]:
        """分析失败原因"""
        failed_indicators = {
            name: result for name, result in self.validation_stats['validation_results'].items()
            if not result['overall_validation']
        }
        
        analysis = {
            'total_failed': len(failed_indicators),
            'failure_categories': {
                'calculation_failures': [],
                'logic_failures': [],
                'condition_failures': []
            },
            'most_common_errors': {}
        }
        
        # 分类失败原因
        for name, result in failed_indicators.items():
            if result['calculation_success_rate'] < 0.5:
                analysis['failure_categories']['calculation_failures'].append(name)
            elif result['logic_validation_rate'] < 0.5:
                analysis['failure_categories']['logic_failures'].append(name)
            else:
                analysis['failure_categories']['condition_failures'].append(name)
        
        # 统计最常见的错误
        all_errors = {}
        for result in failed_indicators.values():
            for error, count in result.get('common_errors', []):
                all_errors[error] = all_errors.get(error, 0) + count
        
        analysis['most_common_errors'] = sorted(all_errors.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return analysis
    
    def _generate_fix_suggestions(self) -> List[Dict[str, str]]:
        """生成修复建议"""
        suggestions = []
        
        # 基于失败分析生成建议
        failure_analysis = self._analyze_failures()
        
        if failure_analysis['failure_categories']['calculation_failures']:
            suggestions.append({
                'category': 'calculation_failures',
                'suggestion': '检查指标计算公式的实现，确保数学逻辑正确',
                'affected_indicators': failure_analysis['failure_categories']['calculation_failures']
            })
        
        if failure_analysis['failure_categories']['logic_failures']:
            suggestions.append({
                'category': 'logic_failures',
                'suggestion': '验证指标输出值的合理性，检查数值范围和分布',
                'affected_indicators': failure_analysis['failure_categories']['logic_failures']
            })
        
        if failure_analysis['failure_categories']['condition_failures']:
            suggestions.append({
                'category': 'condition_failures',
                'suggestion': '优化选股条件的设置，确保条件逻辑合理',
                'affected_indicators': failure_analysis['failure_categories']['condition_failures']
            })
        
        # 基于常见错误生成建议
        for error, count in failure_analysis['most_common_errors']:
            if 'data' in error.lower():
                suggestions.append({
                    'category': 'data_issue',
                    'suggestion': f'解决数据问题: {error}',
                    'frequency': count
                })
            elif 'calculation' in error.lower():
                suggestions.append({
                    'category': 'calculation_issue',
                    'suggestion': f'修复计算问题: {error}',
                    'frequency': count
                })
        
        return suggestions
    
    def print_validation_summary(self, report: Dict[str, Any]):
        """打印验证摘要"""
        summary = report['validation_summary']
        
        print("\n" + "="*80)
        print("🔍 指标逻辑验证报告")
        print("="*80)
        print(f"总指标数: {summary['total_indicators']}")
        print(f"验证通过: {summary['valid_indicators']}")
        print(f"验证失败: {summary['invalid_indicators']}")
        print(f"成功率: {summary['success_rate']}%")
        print(f"计算错误: {summary['calculation_errors']}")
        print(f"逻辑错误: {summary['logic_errors']}")
        print(f"数据错误: {summary['data_errors']}")
        
        # 失败分析
        if 'failure_analysis' in report:
            failure = report['failure_analysis']
            print(f"\n📋 失败分析:")
            print(f"  计算失败: {len(failure['failure_categories']['calculation_failures'])} 个")
            print(f"  逻辑失败: {len(failure['failure_categories']['logic_failures'])} 个")
            print(f"  条件失败: {len(failure['failure_categories']['condition_failures'])} 个")
            
            if failure['most_common_errors']:
                print(f"\n❌ 最常见错误:")
                for error, count in failure['most_common_errors'][:3]:
                    print(f"  {error} (出现 {count} 次)")
        
        # 修复建议
        if 'fix_suggestions' in report:
            print(f"\n🔧 修复建议:")
            for suggestion in report['fix_suggestions'][:5]:
                print(f"  {suggestion['category']}: {suggestion['suggestion']}")
        
        print("="*80)


def main():
    """主函数"""
    print("🔍 启动指标逻辑验证工具")
    
    # 创建验证器
    validator = IndicatorLogicValidator(debug_mode=True)
    
    try:
        # 运行全面验证
        report = validator.validate_all_indicators(sample_stocks=2)
        
        # 显示验证摘要
        validator.print_validation_summary(report)
        
        # 保存详细报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(root_dir, 'data', 'result', f'indicator_validation_report_{timestamp}.json')
        
        os.makedirs(os.path.dirname(report_file), exist_ok=True)
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"📄 详细报告已保存: {report_file}")
        
        # 检查验证结果
        if report['validation_summary']['invalid_indicators'] > 0:
            print(f"\n⚠️ 发现 {report['validation_summary']['invalid_indicators']} 个问题指标，请检查详细报告")
            return 1
        else:
            print("\n🎉 所有指标验证通过！")
            return 0
            
    except Exception as e:
        logger.error(f"❌ 验证失败: {e}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == '__main__':
    exit(main())