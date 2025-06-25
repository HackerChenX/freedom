"""
数据质量检查模块

检查爬取数据的完整性、准确性和一致性
"""

import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict
from utils.logger import get_logger

logger = get_logger(__name__)


class DataQualityRule:
    """数据质量规则"""

    def __init__(self, name: str, description: str, check_func, severity: str = "warning"):
        self.name = name
        self.description = description
        self.check_func = check_func
        self.severity = severity  # info, warning, error, critical

    def check(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """执行质量检查"""
        try:
            result = self.check_func(data)
            return {
                'rule_name': self.name,
                'description': self.description,
                'severity': self.severity,
                'passed': result.get('passed', False),
                'message': result.get('message', ''),
                'details': result.get('details', {}),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"数据质量检查失败 {self.name}: {e}")
            return {
                'rule_name': self.name,
                'description': self.description,
                'severity': 'error',
                'passed': False,
                'message': f'检查执行失败: {str(e)}',
                'details': {},
                'timestamp': datetime.now().isoformat()
            }


class DataQualityChecker:
    """数据质量检查器"""

    def __init__(self):
        self.rules = {}
        self.check_history = []
        self.max_history = 1000
        self.seen_ids = set()  # 用于重复检查

        # 添加默认质量规则
        self._add_default_rules()

    def _add_default_rules(self):
        """添加默认质量规则"""

        # 必填字段检查
        self.add_rule(DataQualityRule(
            name="required_fields",
            description="检查必填字段是否存在",
            check_func=self._check_required_fields,
            severity="error"
        ))

        # 数据格式检查
        self.add_rule(DataQualityRule(
            name="data_format",
            description="检查数据格式是否正确",
            check_func=self._check_data_format,
            severity="warning"
        ))

        # 内容长度检查
        self.add_rule(DataQualityRule(
            name="content_length",
            description="检查内容长度是否合理",
            check_func=self._check_content_length,
            severity="warning"
        ))

        # 重复数据检查
        self.add_rule(DataQualityRule(
            name="duplicate_check",
            description="检查是否存在重复数据",
            check_func=self._check_duplicates,
            severity="warning"
        ))

        # 股票代码有效性检查
        self.add_rule(DataQualityRule(
            name="stock_code_validity",
            description="检查股票代码格式是否正确",
            check_func=self._check_stock_codes,
            severity="warning"
        ))

    def _check_required_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """检查必填字段"""
        required_fields = ['id', 'title', 'content', 'source', 'url']
        missing_fields = []

        for field in required_fields:
            if field not in data or not data[field]:
                missing_fields.append(field)

        if missing_fields:
            return {
                'passed': False,
                'message': f'缺少必填字段: {", ".join(missing_fields)}',
                'details': {'missing_fields': missing_fields}
            }

        return {
            'passed': True,
            'message': '所有必填字段都存在',
            'details': {}
        }

    def _check_data_format(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """检查数据格式"""
        issues = []

        # 检查URL格式
        if 'url' in data and data['url']:
            url_pattern = r'^https?://.+'
            if not re.match(url_pattern, data['url']):
                issues.append('URL格式不正确')

        # 检查ID格式
        if 'id' in data and data['id']:
            if not isinstance(data['id'], str) or len(data['id']) < 3:
                issues.append('ID格式不正确')

        # 检查数值字段
        numeric_fields = ['view_count', 'like_count', 'comment_count']
        for field in numeric_fields:
            if field in data and data[field] is not None:
                if not isinstance(data[field], (int, float)) or data[field] < 0:
                    issues.append(f'{field}数值格式不正确')

        if issues:
            return {
                'passed': False,
                'message': f'数据格式问题: {"; ".join(issues)}',
                'details': {'format_issues': issues}
            }

        return {
            'passed': True,
            'message': '数据格式正确',
            'details': {}
        }

    def _check_content_length(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """检查内容长度"""
        issues = []

        # 检查标题长度
        if 'title' in data and data['title']:
            title_len = len(data['title'])
            if title_len < 5:
                issues.append('标题过短')
            elif title_len > 200:
                issues.append('标题过长')

        # 检查内容长度
        if 'content' in data and data['content']:
            content_len = len(data['content'])
            if content_len < 10:
                issues.append('内容过短')
            elif content_len > 50000:
                issues.append('内容过长')

        if issues:
            return {
                'passed': False,
                'message': f'内容长度问题: {"; ".join(issues)}',
                'details': {'length_issues': issues}
            }

        return {
            'passed': True,
            'message': '内容长度合理',
            'details': {}
        }

    def _check_duplicates(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """检查重复数据"""
        if 'id' not in data or not data['id']:
            return {
                'passed': False,
                'message': '无法检查重复：缺少ID',
                'details': {}
            }

        data_id = data['id']
        if data_id in self.seen_ids:
            return {
                'passed': False,
                'message': f'发现重复数据: {data_id}',
                'details': {'duplicate_id': data_id}
            }

        self.seen_ids.add(data_id)
        return {
            'passed': True,
            'message': '无重复数据',
            'details': {}
        }

    def _check_stock_codes(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """检查股票代码"""
        issues = []

        if 'stock_codes' in data and data['stock_codes']:
            stock_codes = data['stock_codes']
            if isinstance(stock_codes, list):
                for code in stock_codes:
                    if not self._is_valid_stock_code(code):
                        issues.append(f'无效股票代码: {code}')

        if issues:
            return {
                'passed': False,
                'message': f'股票代码问题: {"; ".join(issues)}',
                'details': {'invalid_codes': issues}
            }

        return {
            'passed': True,
            'message': '股票代码格式正确',
            'details': {}
        }

    def _is_valid_stock_code(self, code: str) -> bool:
        """验证股票代码格式"""
        if not isinstance(code, str) or len(code) != 6:
            return False

        if not code.isdigit():
            return False

        # 检查股票代码前缀
        valid_prefixes = ['60', '688', '00', '002', '300']
        return any(code.startswith(prefix) for prefix in valid_prefixes)

    def add_rule(self, rule: DataQualityRule):
        """添加质量规则"""
        self.rules[rule.name] = rule
        logger.info(f"添加数据质量规则: {rule.name}")

    def remove_rule(self, rule_name: str):
        """移除质量规则"""
        if rule_name in self.rules:
            del self.rules[rule_name]
            logger.info(f"移除数据质量规则: {rule_name}")

    def check_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """检查单条数据质量"""
        results = []
        passed_count = 0
        total_count = len(self.rules)

        for rule_name, rule in self.rules.items():
            result = rule.check(data)
            results.append(result)

            if result['passed']:
                passed_count += 1
            else:
                # 记录失败的检查
                logger.warning(f"数据质量检查失败 [{rule_name}]: {result['message']}")

        # 计算质量分数
        quality_score = (passed_count / total_count * 100) if total_count > 0 else 0

        # 确定质量等级
        if quality_score >= 95:
            quality_level = "优秀"
        elif quality_score >= 85:
            quality_level = "良好"
        elif quality_score >= 70:
            quality_level = "一般"
        elif quality_score >= 50:
            quality_level = "较差"
        else:
            quality_level = "很差"

        check_result = {
            'data_id': data.get('id', 'unknown'),
            'timestamp': datetime.now().isoformat(),
            'quality_score': quality_score,
            'quality_level': quality_level,
            'passed_checks': passed_count,
            'total_checks': total_count,
            'check_results': results,
            'summary': {
                'critical_issues': len([r for r in results if not r['passed'] and r['severity'] == 'critical']),
                'error_issues': len([r for r in results if not r['passed'] and r['severity'] == 'error']),
                'warning_issues': len([r for r in results if not r['passed'] and r['severity'] == 'warning']),
                'info_issues': len([r for r in results if not r['passed'] and r['severity'] == 'info'])
            }
        }

        # 记录检查历史
        self._record_check(check_result)

        return check_result

    def _record_check(self, check_result: Dict[str, Any]):
        """记录检查历史"""
        self.check_history.append(check_result)

        # 限制历史记录数量
        if len(self.check_history) > self.max_history:
            self.check_history = self.check_history[-self.max_history:]

    def get_check_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """获取检查历史"""
        from datetime import datetime, timedelta
        cutoff_time = datetime.now() - timedelta(hours=hours)

        recent_checks = []
        for check in self.check_history:
            check_time = datetime.fromisoformat(check['timestamp'])
            if check_time >= cutoff_time:
                recent_checks.append(check)

        return recent_checks

    def get_quality_stats(self, hours: int = 24) -> Dict[str, Any]:
        """获取质量统计"""
        recent_checks = self.get_check_history(hours)

        if not recent_checks:
            return {'message': '暂无检查数据'}

        total_checks = len(recent_checks)
        avg_score = sum(check['quality_score'] for check in recent_checks) / total_checks

        quality_levels = {}
        for check in recent_checks:
            level = check['quality_level']
            quality_levels[level] = quality_levels.get(level, 0) + 1

        return {
            'total_checks': total_checks,
            'average_quality_score': avg_score,
            'quality_distribution': quality_levels,
            'time_range_hours': hours
        }