#!/usr/bin/env python3
"""
指标质量检查脚本

用于批量验证指标实现质量，检查calculate()和get_signal()方法的正确实现，
生成详细的质量评估报告。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import inspect
import importlib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from indicators.base_indicator import BaseIndicator
from indicators.core.unified_indicator_manager import UnifiedIndicatorManager
from utils.logger import get_logger

logger = get_logger(__name__)

class IndicatorQualityChecker:
    """指标质量检查器"""
    
    def __init__(self):
        self.indicator_manager = UnifiedIndicatorManager()
        self.test_data = self._generate_test_data()
        self.quality_report = {
            'P0_core_indicators': [],
            'P1_trend_indicators': [],
            'P2_other_indicators': [],
            'summary': {},
            'issues': []
        }
        
        # 定义优先级分类
        self.P0_CORE = ['MACD', 'RSI', 'KDJ', 'BOLL']
        self.P1_TREND = ['MA', 'EMA', 'SMA', 'WMA', 'DEMA', 'TEMA', 'TRIMA', 'KAMA']
        self.P2_OTHER = ['ADX', 'CCI', 'ROC', 'STOCH', 'WILLIAMS', 'MFI', 'OBV', 'CHAIKIN']
    
    def _generate_test_data(self) -> pd.DataFrame:
        """生成标准测试数据"""
        days = 100
        dates = pd.date_range(start=datetime.now() - timedelta(days=days), periods=days, freq='D')
        
        # 生成模拟的OHLCV数据
        np.random.seed(42)  # 固定种子确保可重复性
        base_price = 100
        price_changes = np.random.normal(0, 0.02, days)
        prices = base_price * np.exp(np.cumsum(price_changes))
        
        data = pd.DataFrame({
            'date': dates,
            'open': prices * (1 + np.random.normal(0, 0.01, days)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.02, days))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.02, days))),
            'close': prices,
            'volume': np.random.randint(10000, 100000, days),
            'turnover_rate': np.random.uniform(0.01, 0.1, days)
        })
        
        # 确保OHLC关系正确
        data['high'] = data[['open', 'high', 'close']].max(axis=1)
        data['low'] = data[['open', 'low', 'close']].min(axis=1)
        
        return data
    
    def check_indicator_quality(self, indicator_name: str) -> Dict[str, Any]:
        """检查单个指标的质量"""
        logger.info(f"开始检查指标: {indicator_name}")
        
        result = {
            'indicator_name': indicator_name,
            'priority': self._get_priority(indicator_name),
            'can_create': False,
            'has_calculate': False,
            'has_get_signal': False,
            'calculate_works': False,
            'get_signal_works': False,
            'data_validation': False,
            'error_handling': False,
            'signal_format_valid': False,
            'performance_acceptable': False,
            'issues': [],
            'recommendations': []
        }
        
        try:
            # 1. 检查是否能创建实例
            indicator = self.indicator_manager.create_indicator(indicator_name)
            result['can_create'] = True
            
            # 2. 检查是否有必需的抽象方法
            result['has_calculate'] = hasattr(indicator, 'calculate') and callable(getattr(indicator, 'calculate'))
            result['has_get_signal'] = hasattr(indicator, 'get_signal') and callable(getattr(indicator, 'get_signal'))
            
            if not result['has_calculate']:
                result['issues'].append("缺少calculate()方法")
            if not result['has_get_signal']:
                result['issues'].append("缺少get_signal()方法")
            
            # 3. 测试calculate()方法
            if result['has_calculate']:
                try:
                    start_time = datetime.now()
                    values = indicator.calculate(self.test_data)
                    calc_time = (datetime.now() - start_time).total_seconds()
                    
                    result['calculate_works'] = True
                    result['performance_acceptable'] = calc_time < 2.0  # 2秒阈值
                    
                    if not isinstance(values, pd.DataFrame):
                        result['issues'].append("calculate()返回类型不是DataFrame")
                    elif values.empty:
                        result['issues'].append("calculate()返回空DataFrame")
                    else:
                        # 检查数据质量
                        if values.isnull().all().any():
                            result['issues'].append("calculate()返回全为NaN的列")
                        
                        # 4. 测试get_signal()方法
                        if result['has_get_signal']:
                            try:
                                signal = indicator.get_signal(values)
                                result['get_signal_works'] = True
                                
                                # 验证信号格式
                                if isinstance(signal, dict):
                                    required_fields = ['signal_type', 'strength', 'confidence']
                                    missing_fields = [f for f in required_fields if f not in signal]
                                    
                                    if not missing_fields:
                                        result['signal_format_valid'] = True
                                        
                                        # 验证信号值的有效性
                                        if signal['signal_type'] not in ['buy', 'sell', 'hold']:
                                            result['issues'].append(f"无效的signal_type: {signal['signal_type']}")
                                        
                                        if not (0 <= signal.get('strength', -1) <= 1):
                                            result['issues'].append(f"strength超出范围[0,1]: {signal.get('strength')}")
                                        
                                        if not (0 <= signal.get('confidence', -1) <= 1):
                                            result['issues'].append(f"confidence超出范围[0,1]: {signal.get('confidence')}")
                                    else:
                                        result['issues'].append(f"信号缺少必需字段: {missing_fields}")
                                else:
                                    result['issues'].append("get_signal()返回类型不是字典")
                                    
                            except Exception as e:
                                result['issues'].append(f"get_signal()执行失败: {str(e)}")
                        
                except Exception as e:
                    result['issues'].append(f"calculate()执行失败: {str(e)}")
            
            # 5. 测试数据验证
            try:
                # 测试空数据
                empty_data = pd.DataFrame()
                indicator.calculate(empty_data)
                result['issues'].append("未正确验证空数据")
            except Exception:
                result['data_validation'] = True
            
            # 6. 测试错误处理
            try:
                # 测试缺少必需列的数据
                bad_data = pd.DataFrame({'volume': [1, 2, 3]})
                indicator.calculate(bad_data)
                result['issues'].append("未正确验证数据列")
            except Exception:
                result['error_handling'] = True
            
        except Exception as e:
            result['issues'].append(f"创建指标实例失败: {str(e)}")
        
        # 生成建议
        result['recommendations'] = self._generate_recommendations(result)
        
        return result
    
    def _get_priority(self, indicator_name: str) -> str:
        """获取指标优先级"""
        if indicator_name in self.P0_CORE:
            return 'P0'
        elif indicator_name in self.P1_TREND:
            return 'P1'
        elif indicator_name in self.P2_OTHER:
            return 'P2'
        else:
            return 'P3'
    
    def _generate_recommendations(self, result: Dict[str, Any]) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        if not result['can_create']:
            recommendations.append("修复指标实例化问题")
        
        if not result['has_calculate']:
            recommendations.append("实现calculate()抽象方法")
        
        if not result['has_get_signal']:
            recommendations.append("实现get_signal()抽象方法")
        
        if not result['data_validation']:
            recommendations.append("添加输入数据验证逻辑")
        
        if not result['error_handling']:
            recommendations.append("完善异常处理机制")
        
        if not result['signal_format_valid']:
            recommendations.append("标准化信号输出格式")
        
        if not result['performance_acceptable']:
            recommendations.append("优化计算性能")
        
        return recommendations
    
    def run_quality_check(self, indicators_to_check: List[str] = None) -> Dict[str, Any]:
        """运行质量检查"""
        if indicators_to_check is None:
            indicators_to_check = self.P0_CORE + self.P1_TREND[:4] + self.P2_OTHER[:4]
        
        logger.info(f"开始批量质量检查，共{len(indicators_to_check)}个指标")
        
        all_results = []
        
        for indicator_name in indicators_to_check:
            result = self.check_indicator_quality(indicator_name)
            all_results.append(result)
            
            # 按优先级分类
            priority = result['priority']
            if priority == 'P0':
                self.quality_report['P0_core_indicators'].append(result)
            elif priority == 'P1':
                self.quality_report['P1_trend_indicators'].append(result)
            else:
                self.quality_report['P2_other_indicators'].append(result)
        
        # 生成汇总统计
        self.quality_report['summary'] = self._generate_summary(all_results)
        
        return self.quality_report
    
    def _generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成汇总统计"""
        total = len(results)
        
        summary = {
            'total_checked': total,
            'can_create_count': sum(1 for r in results if r['can_create']),
            'has_calculate_count': sum(1 for r in results if r['has_calculate']),
            'has_get_signal_count': sum(1 for r in results if r['has_get_signal']),
            'calculate_works_count': sum(1 for r in results if r['calculate_works']),
            'get_signal_works_count': sum(1 for r in results if r['get_signal_works']),
            'signal_format_valid_count': sum(1 for r in results if r['signal_format_valid']),
            'data_validation_count': sum(1 for r in results if r['data_validation']),
            'error_handling_count': sum(1 for r in results if r['error_handling']),
            'performance_acceptable_count': sum(1 for r in results if r['performance_acceptable']),
        }
        
        # 计算百分比
        count_keys = [key for key in summary.keys() if key.endswith('_count') and key != 'total_checked']
        for key in count_keys:
            percentage_key = key.replace('_count', '_percentage')
            summary[percentage_key] = (summary[key] / total * 100) if total > 0 else 0
        
        return summary
    
    def generate_report(self, output_file: str = None) -> str:
        """生成详细报告"""
        report_lines = []
        
        report_lines.append("# L4层指标质量评估报告")
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # 汇总统计
        summary = self.quality_report['summary']
        report_lines.append("## 📊 **汇总统计**")
        report_lines.append(f"- 检查指标总数: {summary['total_checked']}")
        report_lines.append(f"- 可创建实例: {summary['can_create_count']}/{summary['total_checked']} ({summary['can_create_percentage']:.1f}%)")
        report_lines.append(f"- 有calculate()方法: {summary['has_calculate_count']}/{summary['total_checked']} ({summary['has_calculate_percentage']:.1f}%)")
        report_lines.append(f"- 有get_signal()方法: {summary['has_get_signal_count']}/{summary['total_checked']} ({summary['has_get_signal_percentage']:.1f}%)")
        report_lines.append(f"- calculate()正常工作: {summary['calculate_works_count']}/{summary['total_checked']} ({summary['calculate_works_percentage']:.1f}%)")
        report_lines.append(f"- get_signal()正常工作: {summary['get_signal_works_count']}/{summary['total_checked']} ({summary['get_signal_works_percentage']:.1f}%)")
        report_lines.append(f"- 信号格式标准: {summary['signal_format_valid_count']}/{summary['total_checked']} ({summary['signal_format_valid_percentage']:.1f}%)")
        report_lines.append(f"- 数据验证完善: {summary['data_validation_count']}/{summary['total_checked']} ({summary['data_validation_percentage']:.1f}%)")
        report_lines.append(f"- 错误处理完善: {summary['error_handling_count']}/{summary['total_checked']} ({summary['error_handling_percentage']:.1f}%)")
        report_lines.append(f"- 性能可接受: {summary['performance_acceptable_count']}/{summary['total_checked']} ({summary['performance_acceptable_percentage']:.1f}%)")
        report_lines.append("")
        
        # P0核心指标详情
        if self.quality_report['P0_core_indicators']:
            report_lines.append("## 🔴 **P0核心指标详情**")
            for result in self.quality_report['P0_core_indicators']:
                report_lines.extend(self._format_indicator_result(result))
            report_lines.append("")
        
        # P1趋势指标详情
        if self.quality_report['P1_trend_indicators']:
            report_lines.append("## 🟡 **P1趋势指标详情**")
            for result in self.quality_report['P1_trend_indicators']:
                report_lines.extend(self._format_indicator_result(result))
            report_lines.append("")
        
        # P2其他指标详情
        if self.quality_report['P2_other_indicators']:
            report_lines.append("## 🟢 **P2其他指标详情**")
            for result in self.quality_report['P2_other_indicators']:
                report_lines.extend(self._format_indicator_result(result))
            report_lines.append("")
        
        report_content = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            logger.info(f"质量报告已保存到: {output_file}")
        
        return report_content
    
    def _format_indicator_result(self, result: Dict[str, Any]) -> List[str]:
        """格式化单个指标结果"""
        lines = []
        lines.append(f"### **{result['indicator_name']}** ({result['priority']})")
        
        # 状态检查
        status_items = [
            ("创建实例", result['can_create']),
            ("calculate()方法", result['has_calculate']),
            ("get_signal()方法", result['has_get_signal']),
            ("calculate()工作", result['calculate_works']),
            ("get_signal()工作", result['get_signal_works']),
            ("信号格式", result['signal_format_valid']),
            ("数据验证", result['data_validation']),
            ("错误处理", result['error_handling']),
            ("性能", result['performance_acceptable'])
        ]
        
        lines.append("**状态检查:**")
        for item, status in status_items:
            emoji = "✅" if status else "❌"
            lines.append(f"- {emoji} {item}")
        
        # 问题列表
        if result['issues']:
            lines.append("**发现问题:**")
            for issue in result['issues']:
                lines.append(f"- ⚠️ {issue}")
        
        # 改进建议
        if result['recommendations']:
            lines.append("**改进建议:**")
            for rec in result['recommendations']:
                lines.append(f"- 💡 {rec}")
        
        lines.append("")
        return lines


def main():
    """主函数"""
    print("🔍 L4层指标质量检查开始")
    print("=" * 60)
    
    checker = IndicatorQualityChecker()
    
    # 定义要检查的指标（按优先级）
    indicators_to_check = [
        # P0核心指标
        'MACD', 'RSI', 'KDJ', 'BOLL',
        # P1趋势指标
        'MA', 'EMA', 'SMA', 'WMA',
        # P2其他指标
        'ADX', 'CCI', 'ROC', 'STOCH'
    ]
    
    # 运行质量检查
    quality_report = checker.run_quality_check(indicators_to_check)
    
    # 生成报告
    report_file = "docs/system_optimization_2024/L4_INDICATOR_QUALITY_REPORT.md"
    report_content = checker.generate_report(report_file)
    
    print(f"\n📊 质量检查完成，共检查 {quality_report['summary']['total_checked']} 个指标")
    print(f"📄 详细报告已保存到: {report_file}")
    
    # 显示关键统计
    summary = quality_report['summary']
    print(f"\n🎯 关键指标:")
    print(f"- 可正常工作的指标: {summary['calculate_works_count']}/{summary['total_checked']} ({summary['calculate_works_percentage']:.1f}%)")
    print(f"- 信号格式标准的指标: {summary['signal_format_valid_count']}/{summary['total_checked']} ({summary['signal_format_valid_percentage']:.1f}%)")
    print(f"- 数据验证完善的指标: {summary['data_validation_count']}/{summary['total_checked']} ({summary['data_validation_percentage']:.1f}%)")


if __name__ == "__main__":
    main()
