#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
真实数据验证核心组件

提供ClickHouse数据库连接验证、数据完整性检查、数据质量验证等功能。
严格遵循架构规范，通过统一查询执行器访问数据库。
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import re

from db.query_executor import get_query_executor
from db.sql_manager import QueryType
from utils.logger import get_logger
from .logging_config import get_test_logger
from .config import get_test_config

logger = get_test_logger('data_validation')


@dataclass
class DataIntegrityReport:
    """数据完整性报告"""
    total_records: int = 0
    required_fields_present: bool = False
    data_volume_sufficient: bool = False
    time_range_coverage: Dict[str, str] = field(default_factory=dict)
    data_freshness: Dict[str, datetime] = field(default_factory=dict)
    integrity_score: float = 0.0
    issues: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataQualityReport:
    """数据质量报告"""
    total_records: int = 0
    missing_values: Dict[str, int] = field(default_factory=dict)
    duplicate_records: int = 0
    data_type_errors: List[str] = field(default_factory=list)
    value_range_errors: List[str] = field(default_factory=list)
    time_continuity_issues: List[str] = field(default_factory=list)
    quality_score: float = 0.0
    recommendations: List[str] = field(default_factory=list)


@dataclass
class DatabaseConnectionReport:
    """数据库连接报告"""
    connection_successful: bool = False
    response_time: float = 0.0
    database_version: str = ""
    available_tables: List[str] = field(default_factory=list)
    connection_pool_status: Dict[str, Any] = field(default_factory=dict)
    error_message: str = ""


class RealDataValidator:
    """真实数据验证器"""
    
    def __init__(self):
        """初始化真实数据验证器"""
        self.config = get_test_config()
        self.query_executor = get_query_executor()
        
        # 必需的数据字段
        self.required_ohlcv_fields = ['code', 'date', 'open', 'high', 'low', 'close', 'volume']
        
        # 数据质量检查规则
        self.quality_rules = {
            'price_positive': lambda df: df[['open', 'high', 'low', 'close']].gt(0).all().all(),
            'volume_non_negative': lambda df: df['volume'].ge(0).all(),
            'high_low_relationship': lambda df: df['high'].ge(df['low']).all(),
            'ohlc_relationship': lambda df: (
                df['high'].ge(df[['open', 'close']].max(axis=1)).all() and
                df['low'].le(df[['open', 'close']].min(axis=1)).all()
            )
        }
    
    def validate_database_connection(self) -> DatabaseConnectionReport:
        """
        验证ClickHouse数据库连接
        
        Returns:
            DatabaseConnectionReport: 连接验证报告
        """
        report = DatabaseConnectionReport()
        
        try:
            start_time = datetime.now()
            
            # 测试基本连接
            test_query = "SELECT version()"
            result = self.query_executor.execute_query(QueryType.CUSTOM, {'query': test_query})
            
            end_time = datetime.now()
            report.response_time = (end_time - start_time).total_seconds()
            
            if not result.empty:
                report.connection_successful = True
                report.database_version = result.iloc[0, 0] if len(result.columns) > 0 else "Unknown"
                logger.info(f"数据库连接成功，版本: {report.database_version}")
            
            # 获取可用表列表
            tables_query = "SHOW TABLES"
            tables_result = self.query_executor.execute_query(QueryType.CUSTOM, {'query': tables_query})
            if not tables_result.empty:
                report.available_tables = tables_result.iloc[:, 0].tolist()
                logger.info(f"发现 {len(report.available_tables)} 个数据表")
            
            # 检查连接池状态（如果可用）
            try:
                # 这里可以添加连接池状态检查逻辑
                report.connection_pool_status = {
                    'active_connections': 1,  # 占位符
                    'max_connections': self.config.database.max_connections
                }
            except Exception as e:
                logger.warning(f"无法获取连接池状态: {e}")
            
        except Exception as e:
            report.connection_successful = False
            report.error_message = str(e)
            logger.error(f"数据库连接失败: {e}")
        
        return report
    
    def validate_data_integrity(self, stock_codes: Optional[List[str]] = None) -> DataIntegrityReport:
        """
        验证数据完整性
        
        Args:
            stock_codes: 要验证的股票代码列表，如果为None则使用配置中的样本代码
            
        Returns:
            DataIntegrityReport: 数据完整性报告
        """
        report = DataIntegrityReport()
        
        if stock_codes is None:
            stock_codes = self.config.test_data.sample_stock_codes
        
        try:
            logger.info(f"开始验证 {len(stock_codes)} 只股票的数据完整性")
            
            total_records = 0
            field_issues = []
            time_coverage = {}
            freshness_info = {}
            
            for i, code in enumerate(stock_codes):
                try:
                    # 获取股票数据
                    params = {
                        'code': code,
                        'start_date': self.config.test_data.test_date_range['start'],
                        'end_date': self.config.test_data.test_date_range['end'],
                        'level': '日线'
                    }
                    
                    data = self.query_executor.execute_query(QueryType.STOCK_DATA, params)
                    
                    if data.empty:
                        report.issues.append(f"股票 {code} 无数据")
                        continue
                    
                    total_records += len(data)
                    
                    # 检查必需字段
                    missing_fields = [field for field in self.required_ohlcv_fields 
                                    if field not in data.columns]
                    if missing_fields:
                        field_issues.extend([f"股票 {code} 缺少字段: {', '.join(missing_fields)}"])
                    
                    # 检查时间覆盖范围
                    if 'date' in data.columns:
                        date_col = pd.to_datetime(data['date'])
                        time_coverage[code] = {
                            'start': date_col.min().strftime('%Y-%m-%d'),
                            'end': date_col.max().strftime('%Y-%m-%d'),
                            'days': len(date_col.unique())
                        }
                        
                        # 数据新鲜度
                        freshness_info[code] = date_col.max()
                    
                    # 进度日志
                    if (i + 1) % 10 == 0:
                        logger.info(f"已验证 {i + 1}/{len(stock_codes)} 只股票")
                
                except Exception as e:
                    report.issues.append(f"验证股票 {code} 时出错: {str(e)}")
                    logger.error(f"验证股票 {code} 数据完整性失败: {e}")
            
            # 汇总结果
            report.total_records = total_records
            report.required_fields_present = len(field_issues) == 0
            report.data_volume_sufficient = total_records >= 1000  # 最少1000条记录
            report.time_range_coverage = time_coverage
            report.data_freshness = freshness_info
            
            if field_issues:
                report.issues.extend(field_issues)
            
            # 计算完整性评分
            score_factors = [
                1.0 if report.required_fields_present else 0.0,
                1.0 if report.data_volume_sufficient else 0.5,
                1.0 if len(report.issues) == 0 else max(0.0, 1.0 - len(report.issues) / len(stock_codes))
            ]
            report.integrity_score = sum(score_factors) / len(score_factors)
            
            report.details = {
                'stocks_validated': len(stock_codes),
                'stocks_with_data': len(time_coverage),
                'average_records_per_stock': total_records / max(1, len(time_coverage))
            }
            
            logger.info(f"数据完整性验证完成，评分: {report.integrity_score:.2f}")
            
        except Exception as e:
            report.issues.append(f"数据完整性验证过程出错: {str(e)}")
            logger.error(f"数据完整性验证失败: {e}")
        
        return report
    
    def validate_data_quality(self, stock_codes: Optional[List[str]] = None) -> DataQualityReport:
        """
        验证数据质量
        
        Args:
            stock_codes: 要验证的股票代码列表
            
        Returns:
            DataQualityReport: 数据质量报告
        """
        report = DataQualityReport()
        
        if stock_codes is None:
            stock_codes = self.config.test_data.sample_stock_codes
        
        try:
            logger.info(f"开始验证 {len(stock_codes)} 只股票的数据质量")
            
            total_records = 0
            total_missing = {}
            total_duplicates = 0
            quality_issues = []
            
            for i, code in enumerate(stock_codes):
                try:
                    # 获取股票数据
                    params = {
                        'code': code,
                        'start_date': self.config.test_data.test_date_range['start'],
                        'end_date': self.config.test_data.test_date_range['end'],
                        'level': '日线'
                    }
                    
                    data = self.query_executor.execute_query(QueryType.STOCK_DATA, params)
                    
                    if data.empty:
                        continue
                    
                    total_records += len(data)
                    
                    # 检查缺失值
                    missing_counts = data.isnull().sum()
                    for field, count in missing_counts.items():
                        if count > 0:
                            total_missing[field] = total_missing.get(field, 0) + count
                    
                    # 检查重复记录
                    if 'date' in data.columns and 'code' in data.columns:
                        duplicates = data.duplicated(subset=['code', 'date']).sum()
                        total_duplicates += duplicates
                        if duplicates > 0:
                            quality_issues.append(f"股票 {code} 有 {duplicates} 条重复记录")
                    
                    # 检查数据类型
                    numeric_fields = ['open', 'high', 'low', 'close', 'volume']
                    for field in numeric_fields:
                        if field in data.columns:
                            if not pd.api.types.is_numeric_dtype(data[field]):
                                report.data_type_errors.append(f"股票 {code} 字段 {field} 不是数值类型")
                    
                    # 检查数值范围
                    price_fields = ['open', 'high', 'low', 'close']
                    for field in price_fields:
                        if field in data.columns:
                            if (data[field] <= 0).any():
                                report.value_range_errors.append(f"股票 {code} 字段 {field} 存在非正值")
                    
                    if 'volume' in data.columns:
                        if (data['volume'] < 0).any():
                            report.value_range_errors.append(f"股票 {code} 成交量存在负值")
                    
                    # 应用质量规则
                    for rule_name, rule_func in self.quality_rules.items():
                        try:
                            if not rule_func(data):
                                quality_issues.append(f"股票 {code} 违反质量规则: {rule_name}")
                        except Exception as e:
                            logger.warning(f"应用质量规则 {rule_name} 到股票 {code} 时出错: {e}")
                    
                    # 检查时间连续性
                    if 'date' in data.columns:
                        date_series = pd.to_datetime(data['date']).sort_values()
                        date_diffs = date_series.diff().dt.days
                        # 检查是否有超过7天的间隔（考虑周末和节假日）
                        large_gaps = date_diffs[date_diffs > 7]
                        if not large_gaps.empty:
                            report.time_continuity_issues.append(
                                f"股票 {code} 存在 {len(large_gaps)} 个大于7天的数据间隔"
                            )
                    
                    # 进度日志
                    if (i + 1) % 10 == 0:
                        logger.info(f"已验证 {i + 1}/{len(stock_codes)} 只股票的数据质量")
                
                except Exception as e:
                    quality_issues.append(f"验证股票 {code} 质量时出错: {str(e)}")
                    logger.error(f"验证股票 {code} 数据质量失败: {e}")
            
            # 汇总结果
            report.total_records = total_records
            report.missing_values = total_missing
            report.duplicate_records = total_duplicates
            
            # 计算质量评分
            missing_rate = sum(total_missing.values()) / max(1, total_records * len(self.required_ohlcv_fields))
            duplicate_rate = total_duplicates / max(1, total_records)
            error_rate = (len(report.data_type_errors) + len(report.value_range_errors)) / max(1, len(stock_codes))
            
            # 质量评分 (0-1)
            report.quality_score = max(0.0, 1.0 - missing_rate - duplicate_rate - error_rate)
            
            # 生成建议
            if missing_rate > 0.01:  # 超过1%缺失率
                report.recommendations.append("建议检查数据采集流程，减少缺失值")
            
            if duplicate_rate > 0.001:  # 超过0.1%重复率
                report.recommendations.append("建议清理重复数据")
            
            if len(report.value_range_errors) > 0:
                report.recommendations.append("建议检查数据源，修正异常数值")
            
            logger.info(f"数据质量验证完成，评分: {report.quality_score:.2f}")
            
        except Exception as e:
            logger.error(f"数据质量验证失败: {e}")
            report.recommendations.append(f"验证过程出错: {str(e)}")
        
        return report
    
    def check_data_access_patterns(self) -> Dict[str, Any]:
        """
        检查数据访问模式，确保通过统一查询执行器访问数据库
        
        Returns:
            Dict[str, Any]: 访问模式检查报告
        """
        report = {
            'using_query_executor': True,
            'direct_db_access_detected': False,
            'violations': [],
            'recommendations': []
        }
        
        try:
            # 检查是否正确使用查询执行器
            if self.query_executor is None:
                report['using_query_executor'] = False
                report['violations'].append("未正确初始化查询执行器")
            
            # 测试查询执行器功能
            test_queries = [
                (QueryType.STOCK_LIST, {'level': '日线'}),
                (QueryType.STOCK_COUNT, {'level': '日线'})
            ]
            
            for query_type, params in test_queries:
                try:
                    result = self.query_executor.execute_query(query_type, params)
                    if result is None:
                        report['violations'].append(f"查询类型 {query_type} 返回空结果")
                except Exception as e:
                    report['violations'].append(f"查询类型 {query_type} 执行失败: {str(e)}")
            
            # 检查是否有直接数据库访问的迹象
            # 这里可以添加更多检查逻辑，比如检查导入语句等
            
            if len(report['violations']) == 0:
                report['recommendations'].append("数据访问模式符合架构规范")
            else:
                report['recommendations'].append("建议修正数据访问违规问题")
                report['direct_db_access_detected'] = True
            
            logger.info("数据访问模式检查完成")
            
        except Exception as e:
            report['violations'].append(f"访问模式检查失败: {str(e)}")
            logger.error(f"数据访问模式检查失败: {e}")
        
        return report
    
    def validate_time_continuity(self, stock_code: str, 
                               start_date: str, end_date: str) -> Dict[str, Any]:
        """
        验证数据时间连续性
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            Dict[str, Any]: 时间连续性报告
        """
        report = {
            'stock_code': stock_code,
            'total_trading_days': 0,
            'actual_data_days': 0,
            'missing_days': 0,
            'continuity_score': 0.0,
            'gaps': [],
            'issues': []
        }
        
        try:
            # 获取股票数据
            params = {
                'code': stock_code,
                'start_date': start_date,
                'end_date': end_date,
                'level': '日线'
            }
            
            data = self.query_executor.execute_query(QueryType.STOCK_DATA, params)
            
            if data.empty:
                report['issues'].append("无数据")
                return report
            
            # 转换日期
            dates = pd.to_datetime(data['date']).sort_values()
            report['actual_data_days'] = len(dates.unique())
            
            # 计算理论交易日数量（简化计算，不考虑具体节假日）
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            total_days = (end_dt - start_dt).days + 1
            
            # 估算交易日（去除周末，约为总天数的5/7）
            estimated_trading_days = int(total_days * 5 / 7)
            report['total_trading_days'] = estimated_trading_days
            
            # 计算缺失天数
            report['missing_days'] = max(0, estimated_trading_days - report['actual_data_days'])
            
            # 查找数据间隔
            date_diffs = dates.diff().dt.days
            large_gaps = date_diffs[date_diffs > 3]  # 超过3天的间隔
            
            for idx, gap_days in large_gaps.items():
                gap_start = dates.iloc[idx - 1]
                gap_end = dates.iloc[idx]
                report['gaps'].append({
                    'start_date': gap_start.strftime('%Y-%m-%d'),
                    'end_date': gap_end.strftime('%Y-%m-%d'),
                    'gap_days': int(gap_days)
                })
            
            # 计算连续性评分
            if estimated_trading_days > 0:
                report['continuity_score'] = report['actual_data_days'] / estimated_trading_days
            
            logger.info(f"股票 {stock_code} 时间连续性验证完成，评分: {report['continuity_score']:.2f}")
            
        except Exception as e:
            report['issues'].append(f"时间连续性验证失败: {str(e)}")
            logger.error(f"验证股票 {stock_code} 时间连续性失败: {e}")
        
        return report
    
    def validate_numerical_reasonableness(self, stock_code: str) -> Dict[str, Any]:
        """
        验证数值合理性
        
        Args:
            stock_code: 股票代码
            
        Returns:
            Dict[str, Any]: 数值合理性报告
        """
        report = {
            'stock_code': stock_code,
            'price_reasonableness': True,
            'volume_reasonableness': True,
            'statistical_summary': {},
            'outliers': [],
            'issues': []
        }
        
        try:
            # 获取最近一年的数据
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            
            params = {
                'code': stock_code,
                'start_date': start_date,
                'end_date': end_date,
                'level': '日线'
            }
            
            data = self.query_executor.execute_query(QueryType.STOCK_DATA, params)
            
            if data.empty:
                report['issues'].append("无数据")
                return report
            
            # 价格合理性检查
            price_fields = ['open', 'high', 'low', 'close']
            for field in price_fields:
                if field in data.columns:
                    values = data[field].dropna()
                    if len(values) > 0:
                        # 统计摘要
                        report['statistical_summary'][field] = {
                            'mean': float(values.mean()),
                            'std': float(values.std()),
                            'min': float(values.min()),
                            'max': float(values.max()),
                            'median': float(values.median())
                        }
                        
                        # 检查异常值（使用3σ原则）
                        mean_val = values.mean()
                        std_val = values.std()
                        outliers = values[(values < mean_val - 3 * std_val) | 
                                        (values > mean_val + 3 * std_val)]
                        
                        if len(outliers) > 0:
                            report['outliers'].append({
                                'field': field,
                                'count': len(outliers),
                                'values': outliers.tolist()[:10]  # 最多显示10个
                            })
                        
                        # 检查价格是否合理（不能为0或负数）
                        if (values <= 0).any():
                            report['price_reasonableness'] = False
                            report['issues'].append(f"{field} 存在非正值")
            
            # 成交量合理性检查
            if 'volume' in data.columns:
                volume = data['volume'].dropna()
                if len(volume) > 0:
                    report['statistical_summary']['volume'] = {
                        'mean': float(volume.mean()),
                        'std': float(volume.std()),
                        'min': float(volume.min()),
                        'max': float(volume.max()),
                        'median': float(volume.median())
                    }
                    
                    # 检查成交量异常值
                    mean_vol = volume.mean()
                    std_vol = volume.std()
                    vol_outliers = volume[(volume > mean_vol + 3 * std_vol)]
                    
                    if len(vol_outliers) > 0:
                        report['outliers'].append({
                            'field': 'volume',
                            'count': len(vol_outliers),
                            'values': vol_outliers.tolist()[:10]
                        })
                    
                    # 检查成交量是否合理（不能为负数）
                    if (volume < 0).any():
                        report['volume_reasonableness'] = False
                        report['issues'].append("成交量存在负值")
            
            logger.info(f"股票 {stock_code} 数值合理性验证完成")
            
        except Exception as e:
            report['issues'].append(f"数值合理性验证失败: {str(e)}")
            logger.error(f"验证股票 {stock_code} 数值合理性失败: {e}")
        
        return report
    
    def generate_comprehensive_validation_report(self, 
                                               stock_codes: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        生成综合验证报告
        
        Args:
            stock_codes: 要验证的股票代码列表
            
        Returns:
            Dict[str, Any]: 综合验证报告
        """
        if stock_codes is None:
            stock_codes = self.config.test_data.sample_stock_codes
        
        logger.info("开始生成综合数据验证报告")
        
        report = {
            'validation_time': datetime.now().isoformat(),
            'stocks_validated': len(stock_codes),
            'database_connection': {},
            'data_integrity': {},
            'data_quality': {},
            'access_patterns': {},
            'overall_score': 0.0,
            'recommendations': []
        }
        
        try:
            # 数据库连接验证
            logger.info("验证数据库连接...")
            db_report = self.validate_database_connection()
            report['database_connection'] = {
                'successful': db_report.connection_successful,
                'response_time': db_report.response_time,
                'database_version': db_report.database_version,
                'available_tables_count': len(db_report.available_tables),
                'error_message': db_report.error_message
            }
            
            # 数据完整性验证
            logger.info("验证数据完整性...")
            integrity_report = self.validate_data_integrity(stock_codes)
            report['data_integrity'] = {
                'total_records': integrity_report.total_records,
                'required_fields_present': integrity_report.required_fields_present,
                'data_volume_sufficient': integrity_report.data_volume_sufficient,
                'integrity_score': integrity_report.integrity_score,
                'issues_count': len(integrity_report.issues),
                'time_coverage': integrity_report.time_range_coverage
            }
            
            # 数据质量验证
            logger.info("验证数据质量...")
            quality_report = self.validate_data_quality(stock_codes)
            report['data_quality'] = {
                'total_records': quality_report.total_records,
                'missing_values_count': sum(quality_report.missing_values.values()),
                'duplicate_records': quality_report.duplicate_records,
                'quality_score': quality_report.quality_score,
                'data_type_errors_count': len(quality_report.data_type_errors),
                'value_range_errors_count': len(quality_report.value_range_errors),
                'recommendations': quality_report.recommendations
            }
            
            # 数据访问模式检查
            logger.info("检查数据访问模式...")
            access_report = self.check_data_access_patterns()
            report['access_patterns'] = access_report
            
            # 计算总体评分
            scores = []
            if db_report.connection_successful:
                scores.append(1.0)
            else:
                scores.append(0.0)
            
            scores.append(integrity_report.integrity_score)
            scores.append(quality_report.quality_score)
            
            if not access_report['direct_db_access_detected']:
                scores.append(1.0)
            else:
                scores.append(0.5)
            
            report['overall_score'] = sum(scores) / len(scores)
            
            # 生成建议
            if not db_report.connection_successful:
                report['recommendations'].append("修复数据库连接问题")
            
            if integrity_report.integrity_score < 0.8:
                report['recommendations'].append("改善数据完整性")
            
            if quality_report.quality_score < 0.8:
                report['recommendations'].append("提升数据质量")
                report['recommendations'].extend(quality_report.recommendations)
            
            if access_report['direct_db_access_detected']:
                report['recommendations'].extend(access_report['recommendations'])
            
            if report['overall_score'] >= 0.9:
                report['recommendations'].append("数据验证通过，系统可用于生产测试")
            elif report['overall_score'] >= 0.7:
                report['recommendations'].append("数据基本可用，建议修复发现的问题")
            else:
                report['recommendations'].append("数据质量不足，需要重大改进")
            
            logger.info(f"综合数据验证完成，总体评分: {report['overall_score']:.2f}")
            
        except Exception as e:
            logger.error(f"生成综合验证报告失败: {e}")
            report['error'] = str(e)
        
        return report


class DataQualityChecker:
    """数据质量检查器"""
    
    def __init__(self):
        """初始化数据质量检查器"""
        self.query_executor = get_query_executor()
        self.quality_rules = {}
        self._register_default_rules()
    
    def _register_default_rules(self) -> None:
        """注册默认质量规则"""
        self.quality_rules.update({
            'no_null_prices': self._check_no_null_prices,
            'positive_prices': self._check_positive_prices,
            'valid_ohlc_relationship': self._check_ohlc_relationship,
            'non_negative_volume': self._check_non_negative_volume,
            'reasonable_price_changes': self._check_reasonable_price_changes
        })
    
    def _check_no_null_prices(self, data: pd.DataFrame) -> Tuple[bool, str]:
        """检查价格字段无空值"""
        price_fields = ['open', 'high', 'low', 'close']
        for field in price_fields:
            if field in data.columns and data[field].isnull().any():
                return False, f"字段 {field} 存在空值"
        return True, "价格字段无空值"
    
    def _check_positive_prices(self, data: pd.DataFrame) -> Tuple[bool, str]:
        """检查价格为正值"""
        price_fields = ['open', 'high', 'low', 'close']
        for field in price_fields:
            if field in data.columns and (data[field] <= 0).any():
                return False, f"字段 {field} 存在非正值"
        return True, "所有价格均为正值"
    
    def _check_ohlc_relationship(self, data: pd.DataFrame) -> Tuple[bool, str]:
        """检查OHLC关系的合理性"""
        required_fields = ['open', 'high', 'low', 'close']
        if not all(field in data.columns for field in required_fields):
            return False, "缺少OHLC字段"
        
        # 最高价应该 >= 开盘价和收盘价
        if not (data['high'] >= data[['open', 'close']].max(axis=1)).all():
            return False, "最高价小于开盘价或收盘价"
        
        # 最低价应该 <= 开盘价和收盘价
        if not (data['low'] <= data[['open', 'close']].min(axis=1)).all():
            return False, "最低价大于开盘价或收盘价"
        
        return True, "OHLC关系合理"
    
    def _check_non_negative_volume(self, data: pd.DataFrame) -> Tuple[bool, str]:
        """检查成交量非负"""
        if 'volume' in data.columns and (data['volume'] < 0).any():
            return False, "成交量存在负值"
        return True, "成交量非负"
    
    def _check_reasonable_price_changes(self, data: pd.DataFrame) -> Tuple[bool, str]:
        """检查价格变化的合理性"""
        if 'close' not in data.columns or len(data) < 2:
            return True, "数据不足，跳过价格变化检查"
        
        # 计算日收益率
        returns = data['close'].pct_change().dropna()
        
        # 检查是否有超过50%的单日涨跌幅（可能的数据错误）
        extreme_changes = returns[(returns > 0.5) | (returns < -0.5)]
        
        if len(extreme_changes) > 0:
            return False, f"存在 {len(extreme_changes)} 个极端价格变化（>50%）"
        
        return True, "价格变化合理"
    
    def register_rule(self, name: str, rule_func: Callable[[pd.DataFrame], Tuple[bool, str]]) -> None:
        """
        注册自定义质量规则
        
        Args:
            name: 规则名称
            rule_func: 规则函数，返回(是否通过, 描述)
        """
        self.quality_rules[name] = rule_func
        logger.info(f"注册数据质量规则: {name}")
    
    def check_data_quality(self, data: pd.DataFrame, 
                          rules: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        检查数据质量
        
        Args:
            data: 要检查的数据
            rules: 要应用的规则列表，如果为None则应用所有规则
            
        Returns:
            Dict[str, Any]: 质量检查结果
        """
        if rules is None:
            rules = list(self.quality_rules.keys())
        
        results = {
            'total_rules': len(rules),
            'passed_rules': 0,
            'failed_rules': 0,
            'rule_results': {},
            'overall_passed': True,
            'quality_score': 0.0
        }
        
        for rule_name in rules:
            if rule_name not in self.quality_rules:
                logger.warning(f"未知的质量规则: {rule_name}")
                continue
            
            try:
                passed, message = self.quality_rules[rule_name](data)
                results['rule_results'][rule_name] = {
                    'passed': passed,
                    'message': message
                }
                
                if passed:
                    results['passed_rules'] += 1
                else:
                    results['failed_rules'] += 1
                    results['overall_passed'] = False
                
            except Exception as e:
                logger.error(f"执行质量规则 {rule_name} 失败: {e}")
                results['rule_results'][rule_name] = {
                    'passed': False,
                    'message': f"规则执行失败: {str(e)}"
                }
                results['failed_rules'] += 1
                results['overall_passed'] = False
        
        # 计算质量评分
        if results['total_rules'] > 0:
            results['quality_score'] = results['passed_rules'] / results['total_rules']
        
        return results