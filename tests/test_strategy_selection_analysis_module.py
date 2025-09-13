"""
策略选股分析模块全面功能测试

测试新开发的策略选股分析模块的所有核心功能
包括策略配置引擎、选股执行引擎、策略评估系统和主控制器
"""

import sys
import os
import time
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dependency_injection import get_logger
from utils.decorators import performance_monitor, exception_handler
from strategy.enhanced_strategy_config_engine import (
    EnhancedStrategyConfigEngine, StrategyConfig, OperatorType
)
from strategy.enhanced_stock_selection_engine import EnhancedStockSelectionEngine
from strategy.enhanced_strategy_evaluation_system import EnhancedStrategyEvaluationSystem
from strategy.strategy_selection_analysis_controller import StrategySelectionAnalysisController

logger = get_logger(__name__)


class StrategySelectionAnalysisModuleTester:
    """策略选股分析模块全面测试器"""
    
    def __init__(self):
        """初始化测试器"""
        self.logger = logger
        self.test_results = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'test_details': [],
            'start_time': None,
            'end_time': None,
            'execution_time': 0.0
        }
        
        # 初始化测试组件
        self.config_engine = EnhancedStrategyConfigEngine()
        self.selection_engine = EnhancedStockSelectionEngine()
        self.evaluation_system = EnhancedStrategyEvaluationSystem()
        self.controller = StrategySelectionAnalysisController()
        
        self.logger.info("策略选股分析模块测试器初始化完成")
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold=60.0)
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """运行全面测试"""
        self.test_results['start_time'] = datetime.now().isoformat()
        start_time = time.time()
        
        self.logger.info("🚀 开始策略选股分析模块全面测试...")
        self.logger.info("=" * 80)
        
        try:
            # 1. 策略配置引擎测试
            self._test_strategy_config_engine()
            
            # 2. 选股执行引擎测试
            self._test_stock_selection_engine()
            
            # 3. 策略评估系统测试
            self._test_strategy_evaluation_system()
            
            # 4. 主控制器测试
            self._test_main_controller()
            
            # 5. 集成测试
            self._test_integration_scenarios()
            
            # 6. 性能测试
            self._test_performance_scenarios()
            
        except Exception as e:
            self.logger.error(f"测试过程中出现错误: {e}")
            self._record_test_result("全面测试", False, f"测试异常: {str(e)}")
        
        # 计算测试结果
        self.test_results['end_time'] = datetime.now().isoformat()
        self.test_results['execution_time'] = time.time() - start_time
        
        # 生成测试报告
        self._generate_test_report()
        
        self.logger.info("=" * 80)
        self.logger.info("🎉 策略选股分析模块全面测试完成！")
        self.logger.info(f"📊 测试结果: {self.test_results['passed_tests']}/{self.test_results['total_tests']} 通过")
        self.logger.info(f"⏱️ 执行时间: {self.test_results['execution_time']:.2f} 秒")
        
        return self.test_results
    
    def _test_strategy_config_engine(self):
        """测试策略配置引擎"""
        self.logger.info("🔧 测试策略配置引擎...")
        
        # 测试1: 公式解析功能
        try:
            formula = "GOLDEN_CROSS.MACD.DAILY AND OVERSOLD.RSI.DAILY"
            conditions = self.config_engine.parse_formula(formula)
            
            if len(conditions) >= 2:
                self._record_test_result("公式解析功能", True, f"成功解析 {len(conditions)} 个条件")
            else:
                self._record_test_result("公式解析功能", False, f"解析条件数量不足: {len(conditions)}")
        except Exception as e:
            self._record_test_result("公式解析功能", False, f"解析失败: {str(e)}")
        
        # 测试2: 策略配置创建
        try:
            config_data = {
                'name': '测试策略',
                'description': '用于测试的策略配置',
                'rules': [{
                    'name': '主规则',
                    'description': '主要选股规则',
                    'formula': formula,
                    'min_score': 70.0
                }],
                'global_settings': {'min_score': 70.0}
            }
            
            strategy_config = self.config_engine.create_strategy_config(config_data)
            
            if strategy_config and strategy_config.name == '测试策略':
                self._record_test_result("策略配置创建", True, f"成功创建策略: {strategy_config.strategy_id}")
            else:
                self._record_test_result("策略配置创建", False, "策略配置创建失败")
        except Exception as e:
            self._record_test_result("策略配置创建", False, f"创建失败: {str(e)}")
        
        # 测试3: 策略配置验证
        try:
            validation_result = self.config_engine.validate_strategy_config(strategy_config)
            
            if validation_result['is_valid'] and validation_result['score'] > 60:
                self._record_test_result("策略配置验证", True, f"验证通过，得分: {validation_result['score']}")
            else:
                self._record_test_result("策略配置验证", False, f"验证失败: {validation_result.get('errors', [])}")
        except Exception as e:
            self._record_test_result("策略配置验证", False, f"验证异常: {str(e)}")
        
        # 测试4: 配置导出导入
        try:
            exported_config = self.config_engine.export_strategy_config(strategy_config, 'json')
            imported_config = self.config_engine.import_strategy_config(exported_config, 'json')
            
            if imported_config.name == strategy_config.name:
                self._record_test_result("配置导出导入", True, "导出导入功能正常")
            else:
                self._record_test_result("配置导出导入", False, "导出导入数据不一致")
        except Exception as e:
            self._record_test_result("配置导出导入", False, f"导出导入失败: {str(e)}")
    
    def _test_stock_selection_engine(self):
        """测试选股执行引擎"""
        self.logger.info("📊 测试选股执行引擎...")
        
        # 准备测试策略配置
        config_data = {
            'name': '选股测试策略',
            'description': '用于测试选股引擎的策略',
            'rules': [{
                'name': '测试规则',
                'description': '测试选股规则',
                'formula': "GOLDEN_CROSS.MACD.DAILY",
                'min_score': 60.0
            }],
            'global_settings': {'min_score': 60.0, 'max_results': 10}
        }
        
        strategy_config = self.config_engine.create_strategy_config(config_data)
        
        # 测试1: 选股执行功能
        try:
            test_stock_pool = ['000001', '000002', '600519']
            selection_result = self.selection_engine.execute_selection(
                strategy_config=strategy_config,
                stock_pool=test_stock_pool,
                selection_date='2024-09-12'
            )
            
            if 'selected_stocks' in selection_result and 'metrics' in selection_result:
                self._record_test_result("选股执行功能", True, 
                    f"成功执行选股，分析 {len(test_stock_pool)} 只股票")
            else:
                self._record_test_result("选股执行功能", False, "选股结果格式不正确")
        except Exception as e:
            self._record_test_result("选股执行功能", False, f"选股执行失败: {str(e)}")
        
        # 测试2: 性能统计功能
        try:
            performance_stats = self.selection_engine.get_performance_stats()
            
            if 'performance_stats' in performance_stats and 'system_stats' in performance_stats:
                self._record_test_result("性能统计功能", True, "性能统计功能正常")
            else:
                self._record_test_result("性能统计功能", False, "性能统计数据不完整")
        except Exception as e:
            self._record_test_result("性能统计功能", False, f"性能统计失败: {str(e)}")
    
    def _test_strategy_evaluation_system(self):
        """测试策略评估系统"""
        self.logger.info("📈 测试策略评估系统...")
        
        # 准备测试策略配置
        config_data = {
            'name': '评估测试策略',
            'description': '用于测试评估系统的策略',
            'rules': [{
                'name': '评估规则',
                'description': '评估测试规则',
                'formula': "OVERSOLD.RSI.DAILY",
                'min_score': 65.0
            }],
            'global_settings': {'min_score': 65.0}
        }
        
        strategy_config = self.config_engine.create_strategy_config(config_data)
        
        # 测试1: 策略评估功能
        try:
            evaluation_period = ('2024-06-01', '2024-09-01')
            evaluation_result = self.evaluation_system.evaluate_strategy(
                strategy_config=strategy_config,
                evaluation_period=evaluation_period
            )
            
            if (evaluation_result.overall_score >= 0 and 
                evaluation_result.grade in ['A+', 'A', 'B+', 'B', 'C+', 'C', 'D']):
                self._record_test_result("策略评估功能", True, 
                    f"评估完成，得分: {evaluation_result.overall_score:.2f}, 评级: {evaluation_result.grade}")
            else:
                self._record_test_result("策略评估功能", False, "评估结果格式不正确")
        except Exception as e:
            self._record_test_result("策略评估功能", False, f"策略评估失败: {str(e)}")
        
        # 测试2: 评估报告导出
        try:
            report = self.evaluation_system.export_evaluation_report(evaluation_result, 'json')
            
            if report and len(report) > 100:  # 基本的报告长度检查
                self._record_test_result("评估报告导出", True, "报告导出功能正常")
            else:
                self._record_test_result("评估报告导出", False, "报告导出内容不足")
        except Exception as e:
            self._record_test_result("评估报告导出", False, f"报告导出失败: {str(e)}")
    
    def _test_main_controller(self):
        """测试主控制器"""
        self.logger.info("🎛️ 测试主控制器...")
        
        # 测试1: 从公式创建策略
        try:
            formula = "VOLUME_SURGE.VOL.DAILY AND GOLDEN_CROSS.MACD.DAILY"
            result = self.controller.create_strategy_from_formula(
                formula=formula,
                strategy_name="控制器测试策略",
                description="用于测试主控制器的策略",
                min_score=70.0
            )
            
            if result['success'] and result['strategy_id']:
                self._record_test_result("公式创建策略", True, f"成功创建策略: {result['strategy_id']}")
                self.test_strategy_config = self.controller.config_engine.create_strategy_config(result['strategy_config'])
            else:
                self._record_test_result("公式创建策略", False, f"创建失败: {result.get('error', 'Unknown')}")
        except Exception as e:
            self._record_test_result("公式创建策略", False, f"创建异常: {str(e)}")
        
        # 测试2: 完整分析流程
        try:
            if hasattr(self, 'test_strategy_config'):
                complete_result = self.controller.run_complete_analysis(
                    formula=formula,
                    strategy_name="完整分析测试策略",
                    stock_pool=['000001', '000002'],
                    min_score=60.0
                )
                
                if (complete_result['success'] and 
                    'strategy_creation' in complete_result and
                    'stock_selection' in complete_result and
                    'strategy_evaluation' in complete_result):
                    self._record_test_result("完整分析流程", True, "完整分析流程正常")
                else:
                    self._record_test_result("完整分析流程", False, "完整分析流程不完整")
            else:
                self._record_test_result("完整分析流程", False, "缺少测试策略配置")
        except Exception as e:
            self._record_test_result("完整分析流程", False, f"完整分析失败: {str(e)}")
        
        # 测试3: 系统状态获取
        try:
            system_status = self.controller.get_system_status()
            
            if ('system_status' in system_status and 
                system_status['system_status'] == 'running' and
                'performance_stats' in system_status):
                self._record_test_result("系统状态获取", True, "系统状态正常")
            else:
                self._record_test_result("系统状态获取", False, "系统状态异常")
        except Exception as e:
            self._record_test_result("系统状态获取", False, f"状态获取失败: {str(e)}")
    
    def _test_integration_scenarios(self):
        """测试集成场景"""
        self.logger.info("🔗 测试集成场景...")
        
        # 测试1: 多策略并行处理
        try:
            strategies = [
                ("GOLDEN_CROSS.MACD.DAILY", "MACD金叉策略"),
                ("OVERSOLD.RSI.DAILY", "RSI超卖策略"),
                ("VOLUME_SURGE.VOL.DAILY", "放量策略")
            ]
            
            parallel_results = []
            for formula, name in strategies:
                result = self.controller.create_strategy_from_formula(
                    formula=formula,
                    strategy_name=name,
                    min_score=60.0
                )
                parallel_results.append(result['success'])
            
            success_rate = sum(parallel_results) / len(parallel_results)
            if success_rate >= 0.8:  # 80%成功率
                self._record_test_result("多策略并行处理", True, f"成功率: {success_rate:.1%}")
            else:
                self._record_test_result("多策略并行处理", False, f"成功率过低: {success_rate:.1%}")
        except Exception as e:
            self._record_test_result("多策略并行处理", False, f"并行处理失败: {str(e)}")
    
    def _test_performance_scenarios(self):
        """测试性能场景"""
        self.logger.info("⚡ 测试性能场景...")
        
        # 测试1: 大股票池选股性能
        try:
            large_stock_pool = [f"{i:06d}" for i in range(1, 21)]  # 20只股票
            
            start_time = time.time()
            result = self.controller.create_strategy_from_formula(
                formula="GOLDEN_CROSS.MACD.DAILY",
                strategy_name="性能测试策略"
            )
            
            if result['success']:
                strategy_config = self.controller.config_engine.create_strategy_config(result['strategy_config'])
                selection_result = self.selection_engine.execute_selection(
                    strategy_config=strategy_config,
                    stock_pool=large_stock_pool[:5]  # 限制为5只以避免超时
                )
            
            execution_time = time.time() - start_time
            
            if execution_time < 30.0:  # 30秒内完成
                self._record_test_result("大股票池性能", True, f"执行时间: {execution_time:.2f}秒")
            else:
                self._record_test_result("大股票池性能", False, f"执行时间过长: {execution_time:.2f}秒")
        except Exception as e:
            self._record_test_result("大股票池性能", False, f"性能测试失败: {str(e)}")
    
    def _record_test_result(self, test_name: str, passed: bool, details: str = ""):
        """记录测试结果"""
        self.test_results['total_tests'] += 1
        
        if passed:
            self.test_results['passed_tests'] += 1
            status = "✅ PASSED"
        else:
            self.test_results['failed_tests'] += 1
            status = "❌ FAILED"
        
        result = {
            'test_name': test_name,
            'status': status,
            'passed': passed,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        
        self.test_results['test_details'].append(result)
        self.logger.info(f"  {status} - {test_name}: {details}")
    
    def _generate_test_report(self):
        """生成测试报告"""
        try:
            report_data = {
                'test_summary': {
                    'total_tests': self.test_results['total_tests'],
                    'passed_tests': self.test_results['passed_tests'],
                    'failed_tests': self.test_results['failed_tests'],
                    'success_rate': self.test_results['passed_tests'] / self.test_results['total_tests'] if self.test_results['total_tests'] > 0 else 0,
                    'execution_time': self.test_results['execution_time']
                },
                'test_details': self.test_results['test_details'],
                'test_metadata': {
                    'start_time': self.test_results['start_time'],
                    'end_time': self.test_results['end_time'],
                    'test_environment': 'development',
                    'test_version': '1.0.0'
                }
            }
            
            # 保存测试报告
            report_filename = f"results/strategy_selection_analysis_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            os.makedirs('results', exist_ok=True)
            
            with open(report_filename, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"📄 测试报告已生成: {report_filename}")
            
        except Exception as e:
            self.logger.error(f"生成测试报告失败: {e}")


def main():
    """主函数"""
    print("🚀 开始策略选股分析模块全面功能测试...")
    
    tester = StrategySelectionAnalysisModuleTester()
    results = tester.run_comprehensive_tests()
    
    print(f"\n📊 测试完成！")
    print(f"总测试数: {results['total_tests']}")
    print(f"通过测试: {results['passed_tests']}")
    print(f"失败测试: {results['failed_tests']}")
    print(f"成功率: {results['passed_tests']/results['total_tests']*100:.1f}%")
    print(f"执行时间: {results['execution_time']:.2f}秒")
    
    return results['passed_tests'] == results['total_tests']


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
