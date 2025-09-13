#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据纯净化改造脚本

移除所有模拟数据，确保系统100%使用真实数据
"""

import os
import re
import shutil
from pathlib import Path
from typing import List, Tuple
import ast
from datetime import datetime

class DataPurificationSystem:
    """数据纯净化系统"""

    def __init__(self, project_root: str = "/Users/hacker/PycharmProjects/freedom"):
        self.project_root = Path(project_root)
        self.archive_dir = self.project_root / "archive" / "mock_data_removal" / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report = []
        self.files_modified = 0
        self.files_removed = 0
        self.mock_references = []

    def run_purification(self):
        """执行数据纯净化"""
        print("=" * 80)
        print("数据纯净化系统 - 移除所有模拟数据")
        print("=" * 80)

        # 1. 创建归档目录
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n归档目录: {self.archive_dir}")

        # 2. 扫描并移除mock文件
        print("\n第1步: 扫描并移除mock相关文件...")
        self.remove_mock_files()

        # 3. 扫描并清理mock引用
        print("\n第2步: 扫描并清理代码中的mock引用...")
        self.clean_mock_references()

        # 4. 强制使用真实数据
        print("\n第3步: 强制使用真实数据...")
        self.enforce_real_data()

        # 5. 创建数据验证装饰器
        print("\n第4步: 创建数据验证机制...")
        self.create_data_validators()

        # 6. 生成报告
        print("\n第5步: 生成清理报告...")
        self.generate_report()

        print("\n" + "=" * 80)
        print("数据纯净化完成!")
        print(f"文件移除: {self.files_removed}")
        print(f"文件修改: {self.files_modified}")
        print(f"Mock引用清理: {len(self.mock_references)}")
        print("=" * 80)

    def remove_mock_files(self):
        """移除所有mock相关文件"""
        mock_patterns = [
            "**/mock*.py",
            "**/test_mocks/**",
            "**/mocks/**",
            "**/simulate*.py",
            "**/dummy*.py",
            "**/fake*.py"
        ]

        files_to_remove = []

        for pattern in mock_patterns:
            files = list(self.project_root.glob(pattern))
            files_to_remove.extend(files)

        # 特定文件列表
        specific_files = [
            "db/interfaces/mock_data_access.py",
            "tests/mocks/db_mock.py",
            "tests/buypoint_analysis/mock_data_interface.py",
            "tests/comprehensive/mock_indicator_calculator.py"
        ]

        for file_path in specific_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                files_to_remove.append(full_path)

        # 移除文件
        for file_path in set(files_to_remove):
            if file_path.is_file() and 'archive' not in str(file_path):
                self._archive_file(file_path)
                self.files_removed += 1
                print(f"  已移除: {file_path.relative_to(self.project_root)}")

    def clean_mock_references(self):
        """清理代码中的mock引用"""
        # 扫描所有Python文件
        py_files = list(self.project_root.glob("**/*.py"))

        for file_path in py_files:
            if 'archive' in str(file_path) or '.venv' in str(file_path) or 'venv' in str(file_path):
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                original_content = content
                modified = False

                # 移除mock相关import
                mock_import_patterns = [
                    r'^from .*mock.* import .*$',
                    r'^import .*mock.*$',
                    r'^from db\.interfaces\.mock_data_access import .*$',
                    r'^from tests\.mocks.* import .*$',
                ]

                for pattern in mock_import_patterns:
                    matches = re.findall(pattern, content, re.MULTILINE)
                    if matches:
                        for match in matches:
                            content = content.replace(match, f"# REMOVED: {match}")
                            self.mock_references.append((file_path, match))
                            modified = True

                # 替换MockDataAccess使用
                if 'MockDataAccess' in content:
                    content = re.sub(
                        r'MockDataAccess\(\)',
                        'DataAccessManager()',
                        content
                    )
                    modified = True

                # 替换create_mock_data_access使用
                if 'create_mock_data_access' in content:
                    content = re.sub(
                        r'create_mock_data_access\(\)',
                        'get_unified_data_manager()',
                        content
                    )
                    modified = True

                # 保存修改
                if modified:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    self.files_modified += 1
                    print(f"  已清理: {file_path.relative_to(self.project_root)}")

            except Exception as e:
                print(f"  警告: 无法处理文件 {file_path}: {e}")

    def enforce_real_data(self):
        """强制使用真实数据"""
        # 更新DataAccessManager
        dam_path = self.project_root / "db/managers/data_access_manager.py"

        if dam_path.exists():
            with open(dam_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 添加真实数据验证
            validation_code = '''
    def _validate_real_data(self, data_source: str):
        """验证数据源是真实数据"""
        if 'mock' in data_source.lower() or 'simulate' in data_source.lower():
            raise ValueError(f"禁止使用模拟数据源: {data_source}")
        return True
'''

            if '_validate_real_data' not in content:
                # 在类定义后添加验证方法
                class_pattern = r'(class DataAccessManager.*?:.*?\n)'
                content = re.sub(
                    class_pattern,
                    r'\1' + validation_code,
                    content,
                    flags=re.DOTALL
                )

                with open(dam_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"  已更新: DataAccessManager with real data validation")

    def create_data_validators(self):
        """创建数据验证装饰器"""
        validator_code = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据真实性验证装饰器

确保所有数据访问都使用真实数据
"""

from functools import wraps
from typing import Any, Callable
import inspect
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
                        error_msg = f"禁止使用模拟数据: {param_value.__class__.__name__}"
                        logger.error(error_msg)
                        raise ValueError(error_msg)

                # 检查字符串参数
                if isinstance(param_value, str):
                    if 'mock' in param_value.lower() or 'simulate' in param_value.lower():
                        logger.warning(f"警告: 参数 {param_name} 可能包含模拟数据引用: {param_value}")

        # 执行函数
        result = func(*args, **kwargs)

        # 检查返回值
        if result is not None and hasattr(result, '__class__'):
            class_name = result.__class__.__name__.lower()
            if 'mock' in class_name:
                error_msg = f"函数 {func.__name__} 返回了模拟数据"
                logger.error(error_msg)
                raise ValueError(error_msg)

        return result

    return wrapper

def validate_market_data(data) -> bool:
    """
    验证市场数据的真实性和完整性

    Args:
        data: 市场数据（DataFrame或dict）

    Returns:
        bool: 数据是否有效
    """
    import pandas as pd

    if isinstance(data, pd.DataFrame):
        # 检查必要列
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in data.columns:
                logger.error(f"缺少必要列: {col}")
                return False

        # 检查OHLC逻辑
        if not (data['high'] >= data['low']).all():
            logger.error("数据逻辑错误: high < low")
            return False

        if not (data['high'] >= data['close']).all():
            logger.error("数据逻辑错误: high < close")
            return False

        if not (data['low'] <= data['close']).all():
            logger.error("数据逻辑错误: low > close")
            return False

        # 检查价格合理性（避免异常值）
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if (data[col] <= 0).any():
                logger.error(f"价格数据异常: {col} 包含非正值")
                return False

            # 检查价格变化率（单日涨跌幅不超过20%）
            pct_change = data[col].pct_change()
            if (abs(pct_change) > 0.2).any():
                suspicious_dates = data.index[abs(pct_change) > 0.2].tolist()
                logger.warning(f"价格波动异常 {col}: {suspicious_dates}")

        # 检查成交量
        if (data['volume'] < 0).any():
            logger.error("成交量数据异常: 包含负值")
            return False

        return True

    return False

def validate_indicator_data(indicator_name: str, data: Any) -> bool:
    """
    验证指标数据的合理性

    Args:
        indicator_name: 指标名称
        data: 指标数据

    Returns:
        bool: 数据是否有效
    """
    import numpy as np

    # RSI应该在0-100之间
    if indicator_name.upper() == 'RSI':
        if isinstance(data, (list, np.ndarray)):
            data = np.array(data)
            if (data < 0).any() or (data > 100).any():
                logger.error(f"RSI数据异常: 值超出[0,100]范围")
                return False

    # MACD验证
    elif indicator_name.upper() == 'MACD':
        if isinstance(data, dict):
            if 'dif' not in data or 'dea' not in data:
                logger.error("MACD数据不完整")
                return False

    # KDJ应该在0-100之间
    elif indicator_name.upper() == 'KDJ':
        if isinstance(data, dict):
            for key in ['k', 'd', 'j']:
                if key in data:
                    values = np.array(data[key])
                    if (values < 0).any() or (values > 100).any():
                        logger.warning(f"KDJ {key}值超出[0,100]范围")

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

        for forbidden in forbidden_sources:
            if forbidden in source.lower():
                self.issues.append(f"禁止的数据源: {source}")
                self.checks_failed += 1
                return False

        self.checks_passed += 1
        return True

    def check_data_freshness(self, data, max_delay_days: int = 1) -> bool:
        """检查数据时效性"""
        import pandas as pd
        from datetime import datetime, timedelta

        if isinstance(data, pd.DataFrame) and 'date' in data.columns:
            latest_date = pd.to_datetime(data['date'].max())
            current_date = datetime.now()

            delay_days = (current_date - latest_date).days

            if delay_days > max_delay_days:
                self.issues.append(f"数据延迟 {delay_days} 天")
                self.checks_failed += 1
                return False

        self.checks_passed += 1
        return True

    def generate_report(self) -> dict:
        """生成检查报告"""
        return {
            'total_checks': self.checks_passed + self.checks_failed,
            'passed': self.checks_passed,
            'failed': self.checks_failed,
            'pass_rate': self.checks_passed / (self.checks_passed + self.checks_failed) if (self.checks_passed + self.checks_failed) > 0 else 0,
            'issues': self.issues
        }
'''

        validator_path = self.project_root / "utils" / "data_validators.py"
        with open(validator_path, 'w', encoding='utf-8') as f:
            f.write(validator_code)
        print(f"  已创建: 数据验证装饰器 utils/data_validators.py")

    def _archive_file(self, file_path: Path):
        """归档文件"""
        relative_path = file_path.relative_to(self.project_root)
        archive_path = self.archive_dir / relative_path

        # 创建目录结构
        archive_path.parent.mkdir(parents=True, exist_ok=True)

        # 移动文件
        shutil.move(str(file_path), str(archive_path))

        # 创建占位文件说明
        placeholder_content = f"""# This file has been removed during data purification
# Original file archived at: {archive_path}
# Removal date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Reason: Ensuring 100% real data usage - no mock/simulated data allowed
"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(placeholder_content)

    def generate_report(self):
        """生成清理报告"""
        report_content = f"""# 数据纯净化报告

## 执行时间
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 执行结果摘要
- 文件移除数量: {self.files_removed}
- 文件修改数量: {self.files_modified}
- Mock引用清理: {len(self.mock_references)}

## 移除的文件
归档目录: {self.archive_dir}

## Mock引用清理详情
"""

        for file_path, reference in self.mock_references:
            report_content += f"\n- {file_path.relative_to(self.project_root)}\n  `{reference}`\n"

        report_content += """

## 新增功能
1. 数据验证装饰器 (utils/data_validators.py)
   - require_real_data: 强制使用真实数据
   - validate_market_data: 市场数据验证
   - validate_indicator_data: 指标数据验证
   - DataIntegrityChecker: 数据完整性检查

2. DataAccessManager增强
   - 添加_validate_real_data方法
   - 自动拒绝模拟数据源

## 后续步骤
1. 运行单元测试确保功能正常
2. 更新所有数据访问代码使用@require_real_data装饰器
3. 配置CI/CD pipeline禁止mock代码提交

## 验证命令
```bash
# 验证是否还有mock引用
grep -r "mock" --include="*.py" . | grep -v archive | grep -v ".venv"

# 验证是否还有simulate引用
grep -r "simulate" --include="*.py" . | grep -v archive | grep -v ".venv"

# 运行测试
pytest tests/ -v
```
"""

        report_path = self.project_root / "docs" / "data_purification_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"\n报告已生成: {report_path}")


def main():
    """主函数"""
    # 确认执行
    print("\n" + "!" * 80)
    print("警告: 此脚本将永久移除所有模拟数据相关代码!")
    print("所有被移除的文件将被归档到 archive/mock_data_removal 目录")
    print("!" * 80)

    response = input("\n确定要继续吗? (yes/no): ")
    if response.lower() != 'yes':
        print("操作已取消")
        return

    # 执行纯净化
    purifier = DataPurificationSystem()
    purifier.run_purification()


if __name__ == "__main__":
    main()