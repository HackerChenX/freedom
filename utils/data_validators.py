#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据真实性验证装饰器

确保所有数据访问都使用真实数据
"""

from functools import wraps
from typing import Any, Callable
import inspect
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils.logger import get_logger

logger = get_logger(__name__)

def require_real_data(func: Callable) -> Callable:
    """
    装饰器：要求使用真实数据

    检查函数参数和返回值，确保不包含模拟数据
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 检查参数
        func_signature = inspect.signature(func)
        bound_args = func_signature.bind(*args, **kwargs)
        bound_args.apply_defaults()

        for param_name, param_value in bound_args.arguments.items():
            if param_value is not None:
                # 检查类名
                if hasattr(param_value, '__class__'):
                    class_name = param_value.__class__.__name__.lower()
                    if 'mock' in class_name or 'simulate' in class_name or 'fake' in class_name:
                        error_msg = f"数据纯净化违规：禁止使用模拟数据 {param_value.__class__.__name__}"
                        logger.error(error_msg)
                        raise ValueError(error_msg)

                # 检查字符串参数
                if isinstance(param_value, str):
                    forbidden_words = ['mock', 'simulate', 'fake', 'dummy', 'test_data']
                    for word in forbidden_words:
                        if word in param_value.lower():
                            logger.warning(f"警告: 参数 {param_name} 可能包含模拟数据引用: {param_value}")

        # 执行函数
        result = func(*args, **kwargs)

        # 检查返回值
        if result is not None and hasattr(result, '__class__'):
            class_name = result.__class__.__name__.lower()
            if 'mock' in class_name:
                error_msg = f"数据纯净化违规：函数 {func.__name__} 返回了模拟数据"
                logger.error(error_msg)
                raise ValueError(error_msg)

        return result

    return wrapper

def validate_market_data(data: pd.DataFrame) -> bool:
    """
    验证市场数据的真实性和完整性

    Args:
        data: 市场数据（DataFrame）

    Returns:
        bool: 数据是否有效

    Raises:
        ValueError: 数据验证失败时
    """
    if not isinstance(data, pd.DataFrame):
        logger.error("数据必须是pandas DataFrame格式")
        raise ValueError("数据格式错误：不是DataFrame")

    if data.empty:
        logger.error("数据为空")
        raise ValueError("数据纯净化违规：不允许空数据")

    # 检查必要列
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        error_msg = f"缺少必要列: {missing_columns}"
        logger.error(error_msg)
        raise ValueError(f"数据完整性错误：{error_msg}")

    # 检查OHLC逻辑
    if not (data['high'] >= data['low']).all():
        error_msg = "数据逻辑错误: high < low"
        logger.error(error_msg)
        raise ValueError(f"数据真实性错误：{error_msg}")

    if not (data['high'] >= data['close']).all():
        error_msg = "数据逻辑错误: high < close"
        logger.error(error_msg)
        raise ValueError(f"数据真实性错误：{error_msg}")

    if not (data['high'] >= data['open']).all():
        error_msg = "数据逻辑错误: high < open"
        logger.error(error_msg)
        raise ValueError(f"数据真实性错误：{error_msg}")

    if not (data['low'] <= data['close']).all():
        error_msg = "数据逻辑错误: low > close"
        logger.error(error_msg)
        raise ValueError(f"数据真实性错误：{error_msg}")

    if not (data['low'] <= data['open']).all():
        error_msg = "数据逻辑错误: low > open"
        logger.error(error_msg)
        raise ValueError(f"数据真实性错误：{error_msg}")

    # 检查价格合理性（避免异常值）
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        if (data[col] <= 0).any():
            error_msg = f"价格数据异常: {col} 包含非正值"
            logger.error(error_msg)
            raise ValueError(f"数据真实性错误：{error_msg}")

        # 检查价格变化率（单日涨跌幅不超过20%）
        pct_change = data[col].pct_change()
        extreme_changes = abs(pct_change) > 0.2
        if extreme_changes.any():
            suspicious_dates = data.index[extreme_changes].tolist()
            logger.warning(f"价格波动异常 {col}: {suspicious_dates}")

    # 检查成交量
    if (data['volume'] < 0).any():
        error_msg = "成交量数据异常: 包含负值"
        logger.error(error_msg)
        raise ValueError(f"数据真实性错误：{error_msg}")

    # 检查数据源标记
    if hasattr(data, 'attrs') and 'source' in data.attrs:
        forbidden_sources = ['mock', 'simulate', 'fake', 'dummy', 'test']
        source = data.attrs['source'].lower()
        for forbidden in forbidden_sources:
            if forbidden in source:
                error_msg = f"禁止的数据源: {data.attrs['source']}"
                logger.error(error_msg)
                raise ValueError(f"数据纯净化违规：{error_msg}")

    logger.debug(f"市场数据验证通过，记录数: {len(data)}")
    return True

def validate_indicator_data(indicator_name: str, data: Any) -> bool:
    """
    验证指标数据的合理性

    Args:
        indicator_name: 指标名称
        data: 指标数据

    Returns:
        bool: 数据是否有效

    Raises:
        ValueError: 指标数据异常时
    """
    if data is None:
        logger.error(f"指标 {indicator_name} 数据为None")
        raise ValueError(f"数据纯净化违规：指标 {indicator_name} 数据不能为空")

    # RSI应该在0-100之间
    if indicator_name.upper() == 'RSI':
        if isinstance(data, (list, np.ndarray, pd.Series)):
            data_array = np.array(data)
            # 过滤NaN值
            valid_data = data_array[~np.isnan(data_array)]
            if len(valid_data) > 0:
                if (valid_data < 0).any() or (valid_data > 100).any():
                    error_msg = f"RSI数据异常: 值超出[0,100]范围，范围：[{valid_data.min():.2f}, {valid_data.max():.2f}]"
                    logger.error(error_msg)
                    raise ValueError(f"数据真实性错误：{error_msg}")

    # MACD验证
    elif indicator_name.upper() == 'MACD':
        if isinstance(data, dict):
            required_keys = ['dif', 'dea']
            missing_keys = [key for key in required_keys if key not in data]
            if missing_keys:
                error_msg = f"MACD数据不完整，缺少: {missing_keys}"
                logger.error(error_msg)
                raise ValueError(f"数据完整性错误：{error_msg}")

    # KDJ应该在0-100之间
    elif indicator_name.upper() == 'KDJ':
        if isinstance(data, dict):
            for key in ['k', 'd', 'j']:
                if key in data:
                    values = np.array(data[key])
                    valid_values = values[~np.isnan(values)]
                    if len(valid_values) > 0:
                        if (valid_values < 0).any() or (valid_values > 100).any():
                            logger.warning(f"KDJ {key}值超出[0,100]范围，范围：[{valid_values.min():.2f}, {valid_values.max():.2f}]")

    # BOLL带验证
    elif indicator_name.upper() == 'BOLL':
        if isinstance(data, dict):
            required_keys = ['upper', 'mid', 'lower']
            missing_keys = [key for key in required_keys if key not in data]
            if missing_keys:
                error_msg = f"BOLL数据不完整，缺少: {missing_keys}"
                logger.error(error_msg)
                raise ValueError(f"数据完整性错误：{error_msg}")

            # 验证BOLL逻辑：上轨 >= 中轨 >= 下轨
            if all(key in data for key in required_keys):
                upper = np.array(data['upper'])
                mid = np.array(data['mid'])
                lower = np.array(data['lower'])

                # 过滤NaN值
                valid_mask = ~(np.isnan(upper) | np.isnan(mid) | np.isnan(lower))
                if valid_mask.any():
                    upper_valid = upper[valid_mask]
                    mid_valid = mid[valid_mask]
                    lower_valid = lower[valid_mask]

                    if not (upper_valid >= mid_valid).all():
                        error_msg = "BOLL逻辑错误: 上轨 < 中轨"
                        logger.error(error_msg)
                        raise ValueError(f"数据逻辑错误：{error_msg}")

                    if not (mid_valid >= lower_valid).all():
                        error_msg = "BOLL逻辑错误: 中轨 < 下轨"
                        logger.error(error_msg)
                        raise ValueError(f"数据逻辑错误：{error_msg}")

    logger.debug(f"指标 {indicator_name} 验证通过")
    return True

class DataIntegrityChecker:
    """数据完整性检查器"""

    def __init__(self):
        self.checks_passed = 0
        self.checks_failed = 0
        self.issues = []

    def check_data_source(self, source: str) -> bool:
        """检查数据源"""
        forbidden_sources = ['mock', 'simulate', 'fake', 'dummy', 'test']

        if not source:
            self.issues.append("数据源为空")
            self.checks_failed += 1
            return False

        source_lower = source.lower()
        for forbidden in forbidden_sources:
            if forbidden in source_lower:
                error_msg = f"禁止的数据源: {source}"
                self.issues.append(error_msg)
                self.checks_failed += 1
                logger.error(f"数据纯净化违规：{error_msg}")
                return False

        self.checks_passed += 1
        return True

    def check_data_freshness(self, data: pd.DataFrame, max_delay_days: int = 1) -> bool:
        """检查数据时效性"""
        if data.empty:
            self.issues.append("数据为空")
            self.checks_failed += 1
            return False

        try:
            if 'date' in data.columns:
                latest_date = pd.to_datetime(data['date'].max())
                current_date = datetime.now()

                delay_days = (current_date - latest_date).days

                if delay_days > max_delay_days:
                    issue = f"数据延迟 {delay_days} 天，最新日期: {latest_date.strftime('%Y-%m-%d')}"
                    self.issues.append(issue)
                    logger.warning(f"数据时效性警告：{issue}")

            self.checks_passed += 1
            return True

        except Exception as e:
            error_msg = f"数据时效性检查失败: {e}"
            self.issues.append(error_msg)
            self.checks_failed += 1
            logger.error(error_msg)
            return False

    def check_data_completeness(self, data: pd.DataFrame, required_columns: list = None) -> bool:
        """检查数据完整性"""
        if data.empty:
            self.issues.append("数据为空")
            self.checks_failed += 1
            return False

        if required_columns:
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                error_msg = f"缺少必要列: {missing_columns}"
                self.issues.append(error_msg)
                self.checks_failed += 1
                logger.error(f"数据完整性错误：{error_msg}")
                return False

        # 检查数据密度（非空值占比）
        if hasattr(data, 'isnull'):
            null_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
            if null_ratio > 0.5:  # 超过50%的空值
                issue = f"数据稀疏，空值占比: {null_ratio:.2%}"
                self.issues.append(issue)
                logger.warning(f"数据质量警告：{issue}")

        self.checks_passed += 1
        return True

    def generate_report(self) -> dict:
        """生成检查报告"""
        total_checks = self.checks_passed + self.checks_failed
        return {
            'total_checks': total_checks,
            'passed': self.checks_passed,
            'failed': self.checks_failed,
            'pass_rate': self.checks_passed / total_checks if total_checks > 0 else 0,
            'issues': self.issues,
            'summary': f"数据完整性检查: {self.checks_passed}/{total_checks} 通过",
            'status': 'PASS' if self.checks_failed == 0 else 'FAIL',
            'check_time': datetime.now().isoformat()
        }

    def reset(self):
        """重置检查器状态"""
        self.checks_passed = 0
        self.checks_failed = 0
        self.issues = []

# 使用示例和测试代码
if __name__ == "__main__":
    # 测试数据验证功能
    logger.info("数据验证器测试开始...")

    # 创建测试数据
    test_data = pd.DataFrame({
        'open': [10.0, 10.5, 11.0],
        'high': [10.5, 11.0, 11.5],
        'low': [9.5, 10.0, 10.5],
        'close': [10.2, 10.8, 11.2],
        'volume': [1000000, 1200000, 1100000]
    })

    try:
        # 测试市场数据验证
        validate_market_data(test_data)
        logger.info("市场数据验证通过")

        # 测试指标数据验证
        validate_indicator_data('RSI', [30, 45, 60, 75])
        logger.info("RSI指标验证通过")

        validate_indicator_data('MACD', {'dif': [0.1, 0.2], 'dea': [0.05, 0.15]})
        logger.info("MACD指标验证通过")

        # 测试数据完整性检查器
        checker = DataIntegrityChecker()
        checker.check_data_source("ClickHouse")
        checker.check_data_completeness(test_data, ['open', 'high', 'low', 'close', 'volume'])

        report = checker.generate_report()
        logger.info(f"数据完整性检查报告: {report['summary']}")

    except ValueError as e:
        logger.error(f"数据验证失败: {e}")

    logger.info("数据验证器测试完成")