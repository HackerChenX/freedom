"""
高级数据质量管理器

提供全面的数据质量检查、异常值检测、停牌处理和数据清洗功能。
专门为股票分析系统设计，确保数据的准确性和可靠性。

Author: System
Date: 2025-01-15
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import warnings

from utils.logger import get_logger
from utils.decorators import exception_handler, performance_monitor
from crawler.monitoring.data_quality_checker import DataQualityChecker
from db.sql_manager import SQLManager, QueryType

logger = get_logger(__name__)


class DataQualityLevel(Enum):
    """数据质量等级"""
    EXCELLENT = "优秀"
    GOOD = "良好"
    FAIR = "一般"
    POOR = "较差"
    CRITICAL = "严重问题"


class AnomalyType(Enum):
    """异常类型"""
    PRICE_SPIKE = "价格异常波动"
    VOLUME_ANOMALY = "成交量异常"
    TRADING_HALT = "停牌"
    DATA_MISSING = "数据缺失"
    LOGICAL_ERROR = "逻辑错误"
    STATISTICAL_OUTLIER = "统计异常值"


@dataclass
class DataQualityIssue:
    """数据质量问题"""
    issue_type: AnomalyType
    severity: str
    description: str
    affected_dates: List[str]
    suggested_action: str
    confidence: float
    details: Dict[str, Any]


@dataclass
class StockDataQualityReport:
    """股票数据质量报告"""
    stock_code: str
    quality_level: DataQualityLevel
    quality_score: float
    total_issues: int
    critical_issues: int
    issues: List[DataQualityIssue]
    data_coverage: float
    recommendation: str
    generated_at: str


class AdvancedDataQualityManager:
    """
    高级数据质量管理器
    
    核心功能：
    1. 异常值检测 - 价格、成交量、技术指标异常
    2. 停牌检测 - 识别停牌、复牌、ST等特殊情况
    3. 数据完整性检查 - 缺失值、重复值、时间序列连续性
    4. 逻辑一致性验证 - OHLC关系、成交量合理性
    5. 统计异常检测 - 基于统计模型的离群值检测
    6. 数据清洗建议 - 自动修复和手动处理建议
    """
    
    def __init__(self):
        self.base_checker = DataQualityChecker()
        
        # 异常检测阈值配置
        self.anomaly_thresholds = {
            _limit': 0.2,      # 单日涨跌幅限制20%
            'volume_change_limit': 10.0,     # 成交量变化限制10倍
            'price_spike_std': 3.0,          # 价格异常标准差倍数
            'volume_spike_std': 3.0,         # 成交量异常标准差倍数
            'zero_volume_days': 5,           # 连续零成交量天数
            'missing_data_ratio': 0.1        # 缺失数据比例阈值
        }
        
        # 停牌检测规则
        self.trading_halt_rules = {
            'zero_volume_threshold': 0.001,  # 成交量接近零的阈值
            'price_unchanged_days': 3,       # 价格连续不变天数
            'min_turnover_rate': 0.0001     # 最小换手率
        }
        
        # 统计
        self.quality_stats = {
            'total_checks': 0,
            'issues_found': 0,
            'stocks_with_issues': set(),
            'most_common_issues': {}
        }
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold=5.0)
    def comprehensive_quality_check(
        self, 
        stock_code: str, 
        stock_data: pd.DataFrame,
        check_config: Optional[Dict[str, Any]] = None
    ) -> StockDataQualityReport:
        """
        综合数据质量检查
        
        Args:
            stock_code: 股票代码
            stock_data: 股票数据
            check_config: 检查配置
            
        Returns:
            StockDataQualityReport: 质量报告
        """
        logger.info(f"🔍 开始综合质量检查: {stock_code}")
        
        if stock_data.empty:
            return self._create_empty_data_report(stock_code)
        
        issues = []
        
        # 1. 基础数据完整性检查
        integrity_issues = self._check_data_integrity(stock_data)
        issues.extend(integrity_issues)
        
        # 2. 价格异常检测
        price_issues = self._detect_price_anomalies(stock_data)
        issues.extend(price_issues)
        
        # 3. 成交量异常检测
        volume_issues = self._detect_volume_anomalies(stock_data)
        issues.extend(volume_issues)
        
        # 4. 停牌检测
        halt_issues = self._detect_trading_halts(stock_data)
        issues.extend(halt_issues)
        
        # 5. 逻辑一致性检查
        logic_issues = self._check_logical_consistency(stock_data)
        issues.extend(logic_issues)
        
        # 6. 统计异常检测
        statistical_issues = self._detect_statistical_outliers(stock_data)
        issues.extend(statistical_issues)
        
        # 生成质量报告
        report = self._generate_quality_report(stock_code, stock_data, issues)
        
        # 更新统计
        self._update_quality_stats(stock_code, issues)
        
        logger.info(f"✅ 质量检查完成: {stock_code}, "
                   f"质量等级: {report.quality_level.value}, "
                   f"问题数: {len(issues)}")
        
        return report
    
    def _check_data_integrity(self, stock_data: pd.DataFrame) -> List[DataQualityIssue]:
        """检查数据完整性"""
        issues = []
        
        # 检查必需列
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in stock_data.columns]
        
        if missing_columns:
            issues.append(DataQualityIssue(
                issue_type=AnomalyType.DATA_MISSING,
                severity="critical",
                description=f"缺少必需列: {missing_columns}",
                affected_dates=[],
                suggested_action="补充缺失的数据列",
                confidence=1.0,
                details={'missing_columns': missing_columns}
            ))
        
        # 检查空值
        null_counts = stock_data[required_columns].isnull().sum()
        high_null_columns = null_counts[null_counts > len(stock_data) * self.anomaly_thresholds['missing_data_ratio']]
        
        if not high_null_columns.empty:
            affected_dates = stock_data[stock_data[high_null_columns.index].isnull().any(axis=1)]['date'].tolist()
            issues.append(DataQualityIssue(
                issue_type=AnomalyType.DATA_MISSING,
                severity="error",
                description=f"高空值比例列: {high_null_columns.to_dict()}",
                affected_dates=affected_dates[:10],  # 限制显示数量
                suggested_action="检查数据源，补充或插值缺失数据",
                confidence=0.9,
                details={'null_ratios': high_null_columns.to_dict()}
            ))
        
        # 检查重复数据
        duplicates = stock_data.duplicated(subset=['date'])
        if duplicates.sum() > 0:
            duplicate_dates = stock_data[duplicates]['date'].tolist()
            issues.append(DataQualityIssue(
                issue_type=AnomalyType.DATA_MISSING,
                severity="warning",
                description=f"发现 {duplicates.sum()} 条重复数据",
                affected_dates=duplicate_dates[:10],
                suggested_action="去除重复数据",
                confidence=1.0,
                details={'duplicate_count': duplicates.sum()}
            ))
        
        return issues
    
    def _detect_price_anomalies(self, stock_data: pd.DataFrame) -> List[DataQualityIssue]:
        """检测价格异常"""
        issues = []
        
        if 'close' not in stock_data.columns or len(stock_data) < 5:
            return issues
        
        # 计算价格变化率
        stock_data = stock_data.copy()
        stock_data[] = stock_data['close'].pct_change()
        
        # 检测极端价格变化
        extreme_changes = stock_data[
            abs(stock_data[]) > self.anomaly_thresholds[_limit']
        ]
        
        if not extreme_changes.empty:
            affected_dates = extreme_changes['date'].tolist()
            max_change = extreme_changes[].abs().max()
            
            issues.append(DataQualityIssue(
                issue_type=AnomalyType.PRICE_SPIKE,
                severity="warning" if max_change < 0.5 else "error",
                description=f"检测到 {len(extreme_changes)} 个极端价格变化",
                affected_dates=affected_dates[:10],
                suggested_action="检查是否为除权除息、重大事件或数据错误",
                confidence=0.8,
                details={
                    'max_change': max_change,
                    'extreme_changes': extreme_changes[['date', 'close', ]].to_dict('records')[:5]
                }
            ))
        
        # 统计异常价格检测
        if len(stock_data) >= 20:
            price_mean = stock_data['close'].mean()
            price_std = stock_data['close'].std()
            
            outliers = stock_data[
                abs(stock_data['close'] - price_mean) > self.anomaly_thresholds['price_spike_std'] * price_std
            ]
            
            if not outliers.empty and len(outliers) > 1:
                affected_dates = outliers['date'].tolist()
                issues.append(DataQualityIssue(
                    issue_type=AnomalyType.STATISTICAL_OUTLIER,
                    severity="info",
                    description=f"检测到 {len(outliers)} 个价格统计异常值",
                    affected_dates=affected_dates[:10],
                    suggested_action="检查是否为正常的市场波动或数据异常",
                    confidence=0.6,
                    details={
                        'outlier_count': len(outliers),
                        'price_mean': price_mean,
                        'price_std': price_std
                    }
                ))
        
        return issues
    
    def _detect_volume_anomalies(self, stock_data: pd.DataFrame) -> List[DataQualityIssue]:
        """检测成交量异常"""
        issues = []
        
        if 'volume' not in stock_data.columns or len(stock_data) < 5:
            return issues
        
        stock_data = stock_data.copy()
        
        # 检测零成交量
        zero_volume = stock_data['volume'] <= self.trading_halt_rules['zero_volume_threshold']
        consecutive_zero = self._find_consecutive_periods(zero_volume)
        
        long_zero_periods = [period for period in consecutive_zero 
                           if period['length'] >= self.anomaly_thresholds['zero_volume_days']]
        
        if long_zero_periods:
            all_affected_dates = []
            for period in long_zero_periods:
                period_dates = stock_data.iloc[period['start']:period['end']]['date'].tolist()
                all_affected_dates.extend(period_dates)
            
            issues.append(DataQualityIssue(
                issue_type=AnomalyType.VOLUME_ANOMALY,
                severity="warning",
                description=f"检测到 {len(long_zero_periods)} 个长期零成交量区间",
                affected_dates=all_affected_dates[:15],
                suggested_action="检查是否为停牌期间或数据错误",
                confidence=0.9,
                details={
                    'zero_periods': long_zero_periods,
                    'total_zero_days': sum(p['length'] for p in long_zero_periods)
                }
            ))
        
        # 检测成交量突增
        if len(stock_data) >= 10:
            stock_data['volume_ma'] = stock_data['volume'].rolling(window=10).mean()
            stock_data['volume_ratio'] = stock_data['volume'] / stock_data['volume_ma']
            
            volume_spikes = stock_data[
                stock_data['volume_ratio'] > self.anomaly_thresholds['volume_change_limit']
            ]
            
            if not volume_spikes.empty:
                affected_dates = volume_spikes['date'].tolist()
                max_ratio = volume_spikes['volume_ratio'].max()
                
                issues.append(DataQualityIssue(
                    issue_type=AnomalyType.VOLUME_ANOMALY,
                    severity="info",
                    description=f"检测到 {len(volume_spikes)} 个成交量异常放大",
                    affected_dates=affected_dates[:10],
                    suggested_action="检查是否为重大事件、公告或异常交易",
                    confidence=0.7,
                    details={
                        'max_volume_ratio': max_ratio,
                        'spike_dates': volume_spikes[['date', 'volume', 'volume_ratio']].to_dict('records')[:5]
                    }
                ))
        
        return issues
    
    def _detect_trading_halts(self, stock_data: pd.DataFrame) -> List[DataQualityIssue]:
        """检测停牌情况"""
        issues = []
        
        required_cols = ['volume', 'close']
        if not all(col in stock_data.columns for col in required_cols) or len(stock_data) < 3:
            return issues
        
        stock_data = stock_data.copy()
        
        # 停牌检测：成交量为零且价格不变
        halt_conditions = (
            (stock_data['volume'] <= self.trading_halt_rules['zero_volume_threshold']) &
            (stock_data['close'].diff().abs() < 0.01)
        )
        
        halt_periods = self._find_consecutive_periods(halt_conditions)
        significant_halts = [period for period in halt_periods 
                           if period['length'] >= self.trading_halt_rules['price_unchanged_days']]
        
        if significant_halts:
            all_affected_dates = []
            for period in significant_halts:
                period_dates = stock_data.iloc[period['start']:period['end']]['date'].tolist()
                all_affected_dates.extend(period_dates)
            
            total_halt_days = sum(period['length'] for period in significant_halts)
            
            issues.append(DataQualityIssue(
                issue_type=AnomalyType.TRADING_HALT,
                severity="info",
                description=f"检测到 {len(significant_halts)} 个疑似停牌期间，共 {total_halt_days} 天",
                affected_dates=all_affected_dates[:15],
                suggested_action="确认停牌信息，考虑在分析中排除停牌期间数据",
                confidence=0.8,
                details={
                    'halt_periods': significant_halts,
                    'total_halt_days': total_halt_days
                }
            ))
        
        return issues
    
    def _check_logical_consistency(self, stock_data: pd.DataFrame) -> List[DataQualityIssue]:
        """检查逻辑一致性"""
        issues = []
        
        ohlc_cols = ['open', 'high', 'low', 'close']
        if not all(col in stock_data.columns for col in ohlc_cols):
            return issues
        
        # 检查OHLC逻辑关系
        logical_errors = []
        
        # High应该是最高价
        high_error = stock_data['high'] < stock_data[['open', 'low', 'close']].max(axis=1)
        if high_error.sum() > 0:
            logical_errors.append(f"最高价逻辑错误: {high_error.sum()}条")
        
        # Low应该是最低价
        low_error = stock_data['low'] > stock_data[['open', 'high', 'close']].min(axis=1)
        if low_error.sum() > 0:
            logical_errors.append(f"最低价逻辑错误: {low_error.sum()}条")
        
        # 检查负价格
        negative_prices = (stock_data[ohlc_cols] <= 0).any(axis=1)
        if negative_prices.sum() > 0:
            logical_errors.append(f"负价格或零价格: {negative_prices.sum()}条")
        
        if logical_errors:
            error_indices = (high_error | low_error | negative_prices)
            affected_dates = stock_data[error_indices]['date'].tolist()
            
            issues.append(DataQualityIssue(
                issue_type=AnomalyType.LOGICAL_ERROR,
                severity="error",
                description="OHLC数据逻辑错误: " + "; ".join(logical_errors),
                affected_dates=affected_dates[:10],
                suggested_action="检查并修正价格数据的逻辑错误",
                confidence=1.0,
                details={
                    'error_types': logical_errors,
                    'error_count': error_indices.sum()
                }
            ))
        
        return issues
    
    def _detect_statistical_outliers(self, stock_data: pd.DataFrame) -> List[DataQualityIssue]:
        """检测统计异常值"""
        issues = []
        
        if len(stock_data) < 20:
            return issues
        
        # 使用IQR方法检测异常值
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        available_cols = [col for col in numeric_cols if col in stock_data.columns]
        
        outlier_summary = {}
        
        for col in available_cols:
            series = stock_data[col].dropna()
            if len(series) < 10:
                continue
            
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = (series < lower_bound) | (series > upper_bound)
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                outlier_ratio = outlier_count / len(series)
                if outlier_ratio > 0.05:  # 超过5%的异常值
                    outlier_summary[col] = {
                        'count': outlier_count,
                        'ratio': outlier_ratio,
                        'bounds': (lower_bound, upper_bound)
                    }
        
        if outlier_summary:
            issues.append(DataQualityIssue(
                issue_type=AnomalyType.STATISTICAL_OUTLIER,
                severity="info",
                description=f"统计异常值检测: {len(outlier_summary)}个字段存在较多异常值",
                affected_dates=[],
                suggested_action="检查异常值是否为正常市场波动或数据错误",
                confidence=0.6,
                details=outlier_summary
            ))
        
        return issues
    
    def _find_consecutive_periods(self, condition_series: pd.Series) -> List[Dict[str, int]]:
        """找到连续的时间段"""
        periods = []
        start = None
        
        for i, value in enumerate(condition_series):
            if value and start is None:
                start = i
            elif not value and start is not None:
                periods.append({
                    'start': start,
                    'end': i,
                    'length': i - start
                })
                start = None
        
        # 处理序列末尾的情况
        if start is not None:
            periods.append({
                'start': start,
                'end': len(condition_series),
                'length': len(condition_series) - start
            })
        
        return periods
    
    def _generate_quality_report(
        self, 
        stock_code: str, 
        stock_data: pd.DataFrame, 
        issues: List[DataQualityIssue]
    ) -> StockDataQualityReport:
        """生成质量报告"""
        
        # 计算质量分数
        total_issues = len(issues)
        critical_issues = len([i for i in issues if i.severity == "critical"])
        error_issues = len([i for i in issues if i.severity == "error"])
        warning_issues = len([i for i in issues if i.severity == "warning"])
        
        # 质量分数计算 (100分制)
        score = 100
        score -= critical_issues * 30
        score -= error_issues * 15
        score -= warning_issues * 5
        score = max(0, score)
        
        # 确定质量等级
        if score >= 90:
            quality_level = DataQualityLevel.EXCELLENT
        elif score >= 80:
            quality_level = DataQualityLevel.GOOD
        elif score >= 60:
            quality_level = DataQualityLevel.FAIR
        elif score >= 40:
            quality_level = DataQualityLevel.POOR
        else:
            quality_level = DataQualityLevel.CRITICAL
        
        # 数据覆盖率
        expected_days = len(stock_data)
        actual_days = len(stock_data.dropna())
        data_coverage = actual_days / expected_days if expected_days > 0 else 0
        
        # 生成建议
        recommendation = self._generate_recommendation(quality_level, issues, data_coverage)
        
        return StockDataQualityReport(
            stock_code=stock_code,
            quality_level=quality_level,
            quality_score=score,
            total_issues=total_issues,
            critical_issues=critical_issues,
            issues=issues,
            data_coverage=data_coverage,
            recommendation=recommendation,
            generated_at=datetime.now().isoformat()
        )
    
    def _generate_recommendation(
        self, 
        quality_level: DataQualityLevel, 
        issues: List[DataQualityIssue], 
        data_coverage: float
    ) -> str:
        """生成改进建议"""
        
        if quality_level == DataQualityLevel.CRITICAL:
            return "数据质量严重不足，建议：1)检查数据源；2)修复关键问题；3)重新获取数据"
        
        elif quality_level == DataQualityLevel.POOR:
            critical_types = {issue.issue_type for issue in issues if issue.severity in ["critical", "error"]}
            return f"数据质量较差，优先处理：{', '.join([t.value for t in critical_types])}"
        
        elif quality_level == DataQualityLevel.FAIR:
            if data_coverage < 0.9:
                return "数据覆盖率不足，建议补充缺失数据并处理主要质量问题"
            else:
                return "数据质量一般，建议处理主要质量问题以提升分析准确性"
        
        elif quality_level == DataQualityLevel.GOOD:
            return "数据质量良好，可进行分析，建议关注少量质量问题"
        
        else:  # EXCELLENT
            return "数据质量优秀，可直接用于分析"
    
    def _create_empty_data_report(self, stock_code: str) -> StockDataQualityReport:
        """创建空数据报告"""
        empty_issue = DataQualityIssue(
            issue_type=AnomalyType.DATA_MISSING,
            severity="critical",
            description="数据完全缺失",
            affected_dates=[],
            suggested_action="重新获取股票数据",
            confidence=1.0,
            details={}
        )
        
        return StockDataQualityReport(
            stock_code=stock_code,
            quality_level=DataQualityLevel.CRITICAL,
            quality_score=0,
            total_issues=1,
            critical_issues=1,
            issues=[empty_issue],
            data_coverage=0.0,
            recommendation="数据完全缺失，需要重新获取",
            generated_at=datetime.now().isoformat()
        )
    
    def _update_quality_stats(self, stock_code: str, issues: List[DataQualityIssue]):
        """更新质量统计"""
        self.quality_stats['total_checks'] += 1
        self.quality_stats['issues_found'] += len(issues)
        
        if issues:
            self.quality_stats['stocks_with_issues'].add(stock_code)
        
        # 统计最常见问题
        for issue in issues:
            issue_key = f"{issue.issue_type.value}_{issue.severity}"
            self.quality_stats['most_common_issues'][issue_key] = \
                self.quality_stats['most_common_issues'].get(issue_key, 0) + 1
    
    def batch_quality_check(
        self, 
        stock_data_dict: Dict[str, pd.DataFrame],
        parallel: bool = True
    ) -> Dict[str, StockDataQualityReport]:
        """
        批量数据质量检查
        
        Args:
            stock_data_dict: 股票数据字典
            parallel: 是否并行处理
            
        Returns:
            Dict[str, StockDataQualityReport]: 质量报告字典
        """
        reports = {}
        
        logger.info(f"🔍 开始批量质量检查: {len(stock_data_dict)}只股票")
        
        for stock_code, stock_data in stock_data_dict.items():
            try:
                report = self.comprehensive_quality_check(stock_code, stock_data)
                reports[stock_code] = report
            except Exception as e:
                logger.error(f"质量检查失败 {stock_code}: {e}")
                continue
        
        logger.info(f"✅ 批量质量检查完成: {len(reports)}/{len(stock_data_dict)} 成功")
        
        return reports
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """获取质量统计摘要"""
        return {
            'total_checks': self.quality_stats['total_checks'],
            'total_issues': self.quality_stats['issues_found'],
            'stocks_with_issues': len(self.quality_stats['stocks_with_issues']),
            'issue_rate': len(self.quality_stats['stocks_with_issues']) / max(1, self.quality_stats['total_checks']),
            'most_common_issues': dict(sorted(
                self.quality_stats['most_common_issues'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10])
        }


# 便捷函数
@exception_handler(reraise=False, default_return=None)
def quick_quality_check(stock_code: str, stock_data: pd.DataFrame) -> Optional[StockDataQualityReport]:
    """
    快速质量检查
    
    Args:
        stock_code: 股票代码
        stock_data: 股票数据
        
    Returns:
        Optional[StockDataQualityReport]: 质量报告
    """
    manager = AdvancedDataQualityManager()
    return manager.comprehensive_quality_check(stock_code, stock_data)


if __name__ == "__main__":
    # 测试示例
    import numpy as np
    
    # 创建测试数据
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    test_data = pd.DataFrame({
        'date': dates,
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 105,
        'low': np.random.randn(100).cumsum() + 95,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000000, 10000000, 100)
    })
    
    # 添加一些异常数据进行测试
    test_data.loc[10, 'close'] = test_data.loc[9, 'close'] * 1.5  # 价格异常
    test_data.loc[20:25, 'volume'] = 0  # 停牌期间
    test_data.loc[30, 'high'] = test_data.loc[30, 'low'] - 1  # 逻辑错误
    
    # 执行质量检查
    report = quick_quality_check('000001.SZ', test_data)
    
    if report:
        print(f"质量检查完成:")
        print(f"  股票代码: {report.stock_code}")
        print(f"  质量等级: {report.quality_level.value}")
        print(f"  质量分数: {report.quality_score}")
        print(f"  问题总数: {report.total_issues}")
        print(f"  关键问题: {report.critical_issues}")
        print(f"  数据覆盖率: {report.data_coverage:.1%}")
        print(f"  建议: {report.recommendation}")
        
        print("\n检测到的问题:")
        for i, issue in enumerate(report.issues, 1):
            print(f"  {i}. {issue.issue_type.value} ({issue.severity})")
            print(f"     {issue.description}")
            print(f"     建议: {issue.suggested_action}")
    else:
        print("质量检查失败") 