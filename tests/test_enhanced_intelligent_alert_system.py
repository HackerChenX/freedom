#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
增强智能预警系统测试脚本

测试增强后的智能预警系统功能，包括：
1. 新增预警规则测试
2. 多指标共振分析测试
3. 批量分析功能测试
4. 实时监控功能测试
5. 风险预警功能测试
"""

import sys
import os
import time
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from utils.exception_handler import exception_handler
from utils.performance_monitor import performance_monitor
from monitoring.intelligent_alert_system import (
    IntelligentAlertSystem, 
    SignalType, 
    SignalStrength,
    AlertRule,
    get_intelligent_alert_system
)

logger = get_logger(__name__)


class EnhancedIntelligentAlertSystemTester:
    """增强智能预警系统测试器"""
    
    def __init__(self):
        """初始化测试器"""
        self.alert_system = get_intelligent_alert_system()
        self.test_stocks = ['000001', '000002', '600000', '600036', '000858']
        self.test_results = {}
        
        logger.info("增强智能预警系统测试器初始化完成")
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=5.0)
    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        logger.info("🚀 开始增强智能预警系统全面测试...")
        
        test_results = {
            'test_new_alert_rules': self.test_new_alert_rules(),
            'test_multi_indicator_analysis': self.test_multi_indicator_analysis(),
            'test_batch_analysis': self.test_batch_analysis(),
            'test_signal_management': self.test_signal_management(),
            'test_risk_warning': self.test_risk_warning(),
            'test_system_statistics': self.test_system_statistics()
        }
        
        # 汇总测试结果
        total_tests = len(test_results)
        passed_tests = sum(1 for result in test_results.values() if result.get('success', False))
        
        logger.info(f"📊 测试完成: {passed_tests}/{total_tests} 通过")
        
        if passed_tests == total_tests:
            logger.info("🎉 所有测试通过！")
        else:
            logger.warning(f"⚠️ {total_tests - passed_tests} 个测试失败")
        
        return {
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'success_rate': passed_tests / total_tests,
                'timestamp': datetime.now().isoformat()
            },
            'details': test_results
        }
    
    def test_new_alert_rules(self) -> Dict[str, Any]:
        """测试新增预警规则"""
        logger.info("🔍 测试新增预警规则...")
        
        try:
            # 获取预警规则列表
            rules = self.alert_system.get_alert_rules()
            
            # 检查新增的规则
            expected_rules = [
                'boll_breakout',
                'volume_anomaly', 
                'price_gap',
                'multi_indicator_buy',
                'multi_indicator_sell'
            ]
            
            found_rules = [rule['id'] for rule in rules]
            missing_rules = [rule_id for rule_id in expected_rules if rule_id not in found_rules]
            
            if missing_rules:
                return {
                    'success': False,
                    'message': f"缺少预警规则: {missing_rules}",
                    'total_rules': len(rules),
                    'found_rules': found_rules
                }
            
            logger.info(f"✅ 新增预警规则测试通过，共 {len(rules)} 个规则")
            return {
                'success': True,
                'message': "所有新增预警规则都已正确注册",
                'total_rules': len(rules),
                'new_rules': expected_rules
            }
            
        except Exception as e:
            logger.error(f"新增预警规则测试失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def test_multi_indicator_analysis(self) -> Dict[str, Any]:
        """测试多指标共振分析"""
        logger.info("🔍 测试多指标共振分析...")
        
        try:
            test_stock = self.test_stocks[0]
            
            # 分析股票信号
            signals = self.alert_system.analyze_stock_signals(test_stock, lookback_days=20)
            
            # 查找多指标信号
            multi_signals = [
                signal for signal in signals 
                if 'multi' in signal.id and signal.signal_type in [SignalType.BUY, SignalType.SELL]
            ]
            
            logger.info(f"✅ 多指标共振分析测试通过，发现 {len(multi_signals)} 个多指标信号")
            return {
                'success': True,
                'message': "多指标共振分析功能正常",
                'total_signals': len(signals),
                'multi_signals': len(multi_signals),
                'signal_details': [
                    {
                        'id': signal.id,
                        'type': signal.signal_type.value,
                        'strength': signal.signal_strength.value,
                        'confidence': signal.confidence,
                        'message': signal.message
                    }
                    for signal in multi_signals[:3]  # 只显示前3个
                ]
            }
            
        except Exception as e:
            logger.error(f"多指标共振分析测试失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def test_batch_analysis(self) -> Dict[str, Any]:
        """测试批量分析功能"""
        logger.info("🔍 测试批量分析功能...")
        
        try:
            # 批量分析多只股票
            batch_results = self.alert_system.batch_analyze_stocks(
                self.test_stocks[:3], 
                lookback_days=15
            )
            
            total_signals = sum(len(signals) for signals in batch_results.values())
            analyzed_stocks = len(batch_results)
            
            logger.info(f"✅ 批量分析测试通过，分析 {analyzed_stocks} 只股票，生成 {total_signals} 个信号")
            return {
                'success': True,
                'message': "批量分析功能正常",
                'analyzed_stocks': analyzed_stocks,
                'total_signals': total_signals,
                'stock_results': {
                    stock: len(signals) 
                    for stock, signals in batch_results.items()
                }
            }
            
        except Exception as e:
            logger.error(f"批量分析测试失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def test_signal_management(self) -> Dict[str, Any]:
        """测试信号管理功能"""
        logger.info("🔍 测试信号管理功能...")
        
        try:
            # 获取信号列表
            all_signals = self.alert_system.get_signals(limit=20)
            buy_signals = self.alert_system.get_signals(signal_type=SignalType.BUY, limit=10)
            
            # 获取待处理信号
            pending_signals = self.alert_system.get_pending_signals(priority_threshold=3)
            
            logger.info(f"✅ 信号管理测试通过，总信号: {len(all_signals)}, 买入信号: {len(buy_signals)}, 待处理: {len(pending_signals)}")
            return {
                'success': True,
                'message': "信号管理功能正常",
                'total_signals': len(all_signals),
                'buy_signals': len(buy_signals),
                'pending_signals': len(pending_signals)
            }
            
        except Exception as e:
            logger.error(f"信号管理测试失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def test_risk_warning(self) -> Dict[str, Any]:
        """测试风险预警功能"""
        logger.info("🔍 测试风险预警功能...")
        
        try:
            # 获取风险预警信号
            risk_signals = self.alert_system.get_signals(signal_type=SignalType.RISK_WARNING, limit=10)
            
            # 分析风险信号类型
            risk_types = {}
            for signal_dict in risk_signals:
                signal_id = signal_dict.get('id', '')
                if 'rsi' in signal_id:
                    risk_types['RSI风险'] = risk_types.get('RSI风险', 0) + 1
                elif 'volume' in signal_id:
                    risk_types['成交量风险'] = risk_types.get('成交量风险', 0) + 1
                elif 'gap' in signal_id:
                    risk_types['跳空风险'] = risk_types.get('跳空风险', 0) + 1
                else:
                    risk_types['其他风险'] = risk_types.get('其他风险', 0) + 1
            
            logger.info(f"✅ 风险预警测试通过，发现 {len(risk_signals)} 个风险信号")
            return {
                'success': True,
                'message': "风险预警功能正常",
                'total_risk_signals': len(risk_signals),
                'risk_types': risk_types
            }
            
        except Exception as e:
            logger.error(f"风险预警测试失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def test_system_statistics(self) -> Dict[str, Any]:
        """测试系统统计功能"""
        logger.info("🔍 测试系统统计功能...")
        
        try:
            # 获取系统统计信息
            stats = self.alert_system.get_system_statistics()
            
            required_fields = [
                'total_rules', 'enabled_rules', 'total_signals',
                'processed_signals', 'pending_signals', 
                'type_statistics', 'strength_statistics'
            ]
            
            missing_fields = [field for field in required_fields if field not in stats]
            
            if missing_fields:
                return {
                    'success': False,
                    'message': f"统计信息缺少字段: {missing_fields}",
                    'stats': stats
                }
            
            logger.info(f"✅ 系统统计测试通过，规则: {stats['total_rules']}, 信号: {stats['total_signals']}")
            return {
                'success': True,
                'message': "系统统计功能正常",
                'statistics': stats
            }
            
        except Exception as e:
            logger.error(f"系统统计测试失败: {e}")
            return {'success': False, 'error': str(e)}


def main():
    """主函数"""
    try:
        logger.info("=" * 60)
        logger.info("增强智能预警系统测试开始")
        logger.info("=" * 60)
        
        # 创建测试器并运行测试
        tester = EnhancedIntelligentAlertSystemTester()
        results = tester.run_all_tests()
        
        # 输出测试结果
        logger.info("\n" + "=" * 60)
        logger.info("测试结果汇总:")
        logger.info("=" * 60)
        
        summary = results['summary']
        logger.info(f"总测试数: {summary['total_tests']}")
        logger.info(f"通过测试: {summary['passed_tests']}")
        logger.info(f"成功率: {summary['success_rate']:.1%}")
        
        # 详细结果
        for test_name, test_result in results['details'].items():
            status = "✅ 通过" if test_result.get('success', False) else "❌ 失败"
            message = test_result.get('message', '无详细信息')
            logger.info(f"{test_name}: {status} - {message}")
        
        return results
        
    except Exception as e:
        logger.error(f"测试执行失败: {e}")
        return {'success': False, 'error': str(e)}


if __name__ == "__main__":
    main()
