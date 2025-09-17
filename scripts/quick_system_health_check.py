#!/usr/bin/env python3
"""
快速系统健康检查脚本

用于快速验证系统核心功能是否正常工作
"""

import sys
import time
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.logger import get_logger

logger = get_logger(__name__)


class QuickSystemHealthChecker:
    """快速系统健康检查器"""
    
    def __init__(self):
        self.results = {}
        self.errors = []
    
    def run_all_checks(self) -> bool:
        """运行所有健康检查"""
        logger.info("🚀 开始快速系统健康检查...")
        
        checks = [
            ("指标注册系统", self.check_indicator_registry),
            ("数据访问层", self.check_data_access_layer),
            ("核心指标计算", self.check_core_indicators),
            ("错误处理系统", self.check_error_handling),
            ("性能监控系统", self.check_performance_monitoring)
        ]
        
        passed_checks = 0
        total_checks = len(checks)
        
        for check_name, check_func in checks:
            try:
                logger.info(f"📋 检查: {check_name}")
                start_time = time.time()
                
                result = check_func()
                execution_time = time.time() - start_time
                
                if result:
                    logger.info(f"  ✅ {check_name} 正常 ({execution_time:.2f}s)")
                    passed_checks += 1
                else:
                    logger.error(f"  ❌ {check_name} 异常 ({execution_time:.2f}s)")
                
                self.results[check_name] = {
                    'passed': result,
                    'execution_time': execution_time
                }
                
            except Exception as e:
                logger.error(f"  💥 {check_name} 检查失败: {str(e)[:100]}")
                self.errors.append(f"{check_name}: {str(e)[:100]}")
                self.results[check_name] = {
                    'passed': False,
                    'error': str(e)[:100]
                }
        
        success_rate = (passed_checks / total_checks) * 100
        logger.info(f"📊 健康检查完成: {passed_checks}/{total_checks} 通过 ({success_rate:.1f}%)")
        
        return success_rate >= 80.0
    
    def check_indicator_registry(self) -> bool:
        """检查指标注册系统"""
        try:
            from indicators.complete_indicator_registry import get_indicator_registry

            # 获取全局注册表实例
            registry = get_indicator_registry()

            # 检查注册统计
            stats = registry.get_registration_stats()

            # 基本检查
            assert stats['total_indicators'] > 100, f"指标数量过少: {stats['total_indicators']}"
            assert stats['success_rate'] > 0.9, f"注册成功率过低: {stats['success_rate']:.2%}"

            logger.info(f"  ✓ 指标总数: {stats['total_indicators']}")
            logger.info(f"  ✓ 成功率: {stats['success_rate']:.2%}")

            return True

        except Exception as e:
            logger.error(f"  ✗ 指标注册系统检查失败: {str(e)[:100]}")
            return False
    
    def check_data_access_layer(self) -> bool:
        """检查数据访问层"""
        try:
            from db.managers.data_access_manager import get_production_data_access_layer
            
            # 获取生产级数据访问层
            data_layer = get_production_data_access_layer()
            
            # 执行健康检查
            health_result = data_layer.health_check()
            
            # 检查健康状态
            status = health_result.get('status', 'unknown')
            assert status in ['healthy', 'degraded'], f"数据访问层状态异常: {status}"
            
            logger.info(f"  ✓ 数据访问层状态: {status}")
            
            return True
            
        except Exception as e:
            logger.error(f"  ✗ 数据访问层检查失败: {str(e)[:100]}")
            return False
    
    def check_core_indicators(self) -> bool:
        """检查核心指标计算"""
        try:
            from indicators.complete_indicator_registry import get_indicator
            import pandas as pd
            import numpy as np
            
            # 创建测试数据
            test_data = pd.DataFrame({
                'code': ['000001'] * 30,
                'date': pd.date_range('2024-01-01', periods=30),
                'open': np.random.uniform(10, 15, 30),
                'high': np.random.uniform(15, 20, 30),
                'low': np.random.uniform(8, 12, 30),
                'close': np.random.uniform(10, 15, 30),
                'volume': np.random.uniform(1000000, 5000000, 30)
            })
            
            # 测试核心指标
            core_indicators = ['MACD', 'RSI', 'BOLL', 'KDJ', 'MA']
            successful_indicators = 0
            
            for indicator_name in core_indicators:
                try:
                    indicator = get_indicator(indicator_name)
                    if indicator:
                        result = indicator.calculate(test_data)
                        if isinstance(result, pd.DataFrame) and not result.empty:
                            successful_indicators += 1
                            logger.info(f"    ✓ {indicator_name} 计算正常")
                        else:
                            logger.warning(f"    ⚠ {indicator_name} 计算结果异常")
                    else:
                        logger.warning(f"    ⚠ {indicator_name} 获取失败")
                except Exception as e:
                    logger.warning(f"    ⚠ {indicator_name} 计算失败: {str(e)[:50]}")
            
            success_rate = successful_indicators / len(core_indicators)
            assert success_rate >= 0.6, f"核心指标成功率过低: {success_rate:.2%}"
            
            logger.info(f"  ✓ 核心指标成功率: {success_rate:.2%}")
            
            return True
            
        except Exception as e:
            logger.error(f"  ✗ 核心指标检查失败: {str(e)[:100]}")
            return False
    
    def check_error_handling(self) -> bool:
        """检查错误处理系统"""
        try:
            from utils.exception_handler import exception_handler
            
            # 测试异常处理装饰器
            @exception_handler(reraise=False, default_return="handled")
            def test_function():
                raise ValueError("测试异常")
            
            result = test_function()
            assert result == "handled", "异常处理装饰器工作异常"
            
            logger.info("  ✓ 异常处理装饰器正常")
            
            return True
            
        except Exception as e:
            logger.error(f"  ✗ 错误处理系统检查失败: {str(e)[:100]}")
            return False
    
    def check_performance_monitoring(self) -> bool:
        """检查性能监控系统"""
        try:
            from utils.performance_monitor import performance_monitor
            import time
            
            # 测试性能监控装饰器
            @performance_monitor(threshold_seconds=0.1)
            def test_function():
                time.sleep(0.05)  # 短暂延迟
                return "success"
            
            result = test_function()
            assert result == "success", "性能监控装饰器工作异常"
            
            logger.info("  ✓ 性能监控装饰器正常")
            
            return True
            
        except Exception as e:
            logger.error(f"  ✗ 性能监控系统检查失败: {str(e)[:100]}")
            return False
    
    def generate_summary_report(self) -> str:
        """生成简要报告"""
        report = []
        report.append("=" * 50)
        report.append("📋 快速系统健康检查报告")
        report.append("=" * 50)
        
        passed_count = sum(1 for result in self.results.values() if result.get('passed', False))
        total_count = len(self.results)
        success_rate = (passed_count / total_count * 100) if total_count > 0 else 0
        
        report.append(f"总检查项: {total_count}")
        report.append(f"通过检查: {passed_count}")
        report.append(f"失败检查: {total_count - passed_count}")
        report.append(f"健康度: {success_rate:.1f}%")
        report.append("")
        
        # 详细结果
        for check_name, result in self.results.items():
            status = "✅ 正常" if result.get('passed', False) else "❌ 异常"
            time_info = f"({result.get('execution_time', 0):.2f}s)" if 'execution_time' in result else ""
            report.append(f"  {status} {check_name} {time_info}")
            
            if 'error' in result:
                report.append(f"    错误: {result['error']}")
        
        if self.errors:
            report.append("")
            report.append("❌ 错误详情:")
            for error in self.errors:
                report.append(f"  - {error}")
        
        report.append("=" * 50)
        
        return "\n".join(report)


def main():
    """主函数"""
    checker = QuickSystemHealthChecker()
    
    # 运行所有检查
    success = checker.run_all_checks()
    
    # 生成并显示报告
    report = checker.generate_summary_report()
    print("\n" + report)
    
    # 保存报告
    report_path = project_root / "docs" / "quick_system_health_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"📄 健康检查报告已保存: {report_path}")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
