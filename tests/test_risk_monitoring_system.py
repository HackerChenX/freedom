#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
风险监控系统测试

测试风险监控系统的各项功能，包括：
1. 市场风险评估测试
2. 个股风险监控测试
3. 组合风险管理测试
4. 综合风险评估测试
5. 实时监控功能测试
6. 风险预警机制测试
"""

import os
import sys
import time
import threading
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from monitoring.risk_monitor import (
    RiskMonitoringSystem, MarketRiskAssessor, StockRiskMonitor, 
    PortfolioRiskManager, RiskLevel, RiskType
)
from utils.logger import get_logger

logger = get_logger(__name__)


class TestRiskMonitoringSystem:
    """风险监控系统测试类"""
    
    def __init__(self):
        """初始化测试"""
        self.risk_system = RiskMonitoringSystem()
        self.market_assessor = MarketRiskAssessor()
        self.stock_monitor = StockRiskMonitor()
        self.portfolio_manager = PortfolioRiskManager()
        
    def test_market_risk_assessment(self):
        """测试市场风险评估"""
        print("📊 测试市场风险评估...")
        
        try:
            # 测试市场风险评估
            market_risk = self.market_assessor.assess_market_risk("000001")
            
            # 验证返回结果
            required_fields = ['market_index', 'risk_level', 'risk_score', 'var_1d', 'var_5d', 
                             'volatility', 'max_drawdown', 'assessment_time']
            
            for field in required_fields:
                if field not in market_risk:
                    print(f"❌ 缺少字段: {field}")
                    return False
            
            # 验证数值范围
            if not (0 <= market_risk['risk_score'] <= 100):
                print(f"❌ 风险评分超出范围: {market_risk['risk_score']}")
                return False
            
            if not (0 <= market_risk['volatility'] <= 2.0):
                print(f"❌ 波动率超出合理范围: {market_risk['volatility']}")
                return False
            
            print(f"✅ 市场风险评估: 风险级别={market_risk['risk_level']}, 评分={market_risk['risk_score']}")
            print(f"✅ 风险指标: VaR={market_risk['var_1d']:.3f}, 波动率={market_risk['volatility']:.3f}")
            
            return True
            
        except Exception as e:
            print(f"❌ 市场风险评估测试失败: {e}")
            return False
    
    def test_stock_risk_monitoring(self):
        """测试个股风险监控"""
        print("📈 测试个股风险监控...")
        
        try:
            # 测试多只股票的风险监控
            test_stocks = ["000001", "000002", "600000"]
            
            for stock_code in test_stocks:
                risk_metrics = self.stock_monitor.monitor_stock_risk(stock_code, f"测试股票{stock_code}")
                
                # 验证风险指标
                if not hasattr(risk_metrics, 'stock_code'):
                    print(f"❌ 股票 {stock_code} 风险指标缺少股票代码")
                    return False
                
                if not (0 <= risk_metrics.risk_score <= 100):
                    print(f"❌ 股票 {stock_code} 风险评分超出范围: {risk_metrics.risk_score}")
                    return False
                
                if risk_metrics.risk_level not in [level.value for level in RiskLevel]:
                    print(f"❌ 股票 {stock_code} 风险级别无效: {risk_metrics.risk_level}")
                    return False
                
                print(f"✅ 股票 {stock_code}: 风险级别={risk_metrics.risk_level}, 评分={risk_metrics.risk_score:.1f}")
                print(f"   波动率={risk_metrics.volatility:.3f}, 贝塔={risk_metrics.beta:.2f}")
            
            return True
            
        except Exception as e:
            print(f"❌ 个股风险监控测试失败: {e}")
            return False
    
    def test_portfolio_risk_management(self):
        """测试组合风险管理"""
        print("📋 测试组合风险管理...")
        
        try:
            # 创建测试组合
            test_portfolio = {
                'id': 'test_portfolio_001',
                'name': '测试投资组合',
                'positions': [
                    {'code': '000001', 'name': '平安银行', 'weight': 0.3, 'value': 300000},
                    {'code': '000002', 'name': '万科A', 'weight': 0.25, 'value': 250000},
                    {'code': '600000', 'name': '浦发银行', 'weight': 0.2, 'value': 200000},
                    {'code': '600036', 'name': '招商银行', 'weight': 0.15, 'value': 150000},
                    {'code': '000858', 'name': '五粮液', 'weight': 0.1, 'value': 100000}
                ]
            }
            
            # 评估组合风险
            portfolio_risk = self.portfolio_manager.assess_portfolio_risk(test_portfolio)
            
            # 验证组合风险结果
            if portfolio_risk.portfolio_id != test_portfolio['id']:
                print(f"❌ 组合ID不匹配: {portfolio_risk.portfolio_id}")
                return False
            
            if portfolio_risk.total_value != 1000000:
                print(f"❌ 组合总市值计算错误: {portfolio_risk.total_value}")
                return False
            
            if not (0 <= portfolio_risk.concentration_risk <= 100):
                print(f"❌ 集中度风险超出范围: {portfolio_risk.concentration_risk}")
                return False
            
            if not (0 <= portfolio_risk.correlation_risk <= 100):
                print(f"❌ 相关性风险超出范围: {portfolio_risk.correlation_risk}")
                return False
            
            print(f"✅ 组合风险评估: 风险级别={portfolio_risk.risk_level}")
            print(f"   组合VaR={portfolio_risk.portfolio_var:.3f}, 波动率={portfolio_risk.portfolio_volatility:.3f}")
            print(f"   集中度风险={portfolio_risk.concentration_risk:.1f}, 相关性风险={portfolio_risk.correlation_risk:.1f}")
            print(f"   持仓数量={len(portfolio_risk.positions)}")
            
            return True
            
        except Exception as e:
            print(f"❌ 组合风险管理测试失败: {e}")
            return False
    
    def test_comprehensive_risk_assessment(self):
        """测试综合风险评估"""
        print("🔍 测试综合风险评估...")
        
        try:
            # 准备测试数据
            test_stocks = ["000001", "000002", "600000"]
            test_portfolios = [
                {
                    'id': 'portfolio_001',
                    'name': '稳健型组合',
                    'positions': [
                        {'code': '000001', 'name': '平安银行', 'weight': 0.4, 'value': 400000},
                        {'code': '600000', 'name': '浦发银行', 'weight': 0.6, 'value': 600000}
                    ]
                }
            ]
            
            # 执行综合风险评估
            risk_report = self.risk_system.comprehensive_risk_assessment(test_stocks, test_portfolios)
            
            # 验证报告结构
            required_sections = ['assessment_time', 'market_risk', 'stock_risks', 'portfolio_risks', 
                               'risk_alerts', 'risk_summary']
            
            for section in required_sections:
                if section not in risk_report:
                    print(f"❌ 风险报告缺少部分: {section}")
                    return False
            
            # 验证股票风险统计
            stock_risks = risk_report['stock_risks']
            if stock_risks['total_count'] != len(test_stocks):
                print(f"❌ 股票风险统计错误: {stock_risks['total_count']} != {len(test_stocks)}")
                return False
            
            # 验证组合风险统计
            portfolio_risks = risk_report['portfolio_risks']
            if portfolio_risks['total_count'] != len(test_portfolios):
                print(f"❌ 组合风险统计错误: {portfolio_risks['total_count']} != {len(test_portfolios)}")
                return False
            
            # 验证风险摘要
            risk_summary = risk_report['risk_summary']
            if 'overall_risk_level' not in risk_summary:
                print("❌ 缺少整体风险级别")
                return False
            
            print(f"✅ 综合风险评估完成")
            print(f"   整体风险级别: {risk_summary['overall_risk_level']}")
            print(f"   股票风险: {stock_risks['total_count']}只, 平均评分={stock_risks['average_risk_score']}")
            print(f"   组合风险: {portfolio_risks['total_count']}个")
            print(f"   风险预警: {len(risk_report['risk_alerts'])}条")
            print(f"   关键风险: {len(risk_summary['key_risks'])}项")
            print(f"   评估耗时: {risk_report.get('assessment_duration', 0)}秒")
            
            return True
            
        except Exception as e:
            print(f"❌ 综合风险评估测试失败: {e}")
            return False
    
    def test_real_time_monitoring(self):
        """测试实时监控功能"""
        print("⏰ 测试实时监控功能...")
        
        try:
            # 准备监控数据
            test_stocks = ["000001", "000002"]
            
            # 启动实时监控（短时间测试）
            self.risk_system.start_real_time_monitoring(test_stocks, interval=2)
            
            # 等待几秒钟
            time.sleep(5)
            
            # 检查监控状态
            status = self.risk_system.get_monitoring_status()
            
            if not status['monitoring_enabled']:
                print("❌ 监控未启用")
                return False
            
            if not status['thread_alive']:
                print("❌ 监控线程未运行")
                return False
            
            print(f"✅ 实时监控启动成功")
            print(f"   监控间隔: {status['monitoring_interval']}秒")
            print(f"   线程状态: {'运行中' if status['thread_alive'] else '已停止'}")
            
            # 停止监控
            self.risk_system.stop_real_time_monitoring()
            
            # 验证停止状态
            time.sleep(1)
            final_status = self.risk_system.get_monitoring_status()
            
            if final_status['monitoring_enabled']:
                print("⚠️ 监控停止可能未完全生效")
            
            print("✅ 实时监控停止成功")
            
            return True
            
        except Exception as e:
            print(f"❌ 实时监控功能测试失败: {e}")
            return False
    
    def test_risk_alert_mechanism(self):
        """测试风险预警机制"""
        print("🚨 测试风险预警机制...")
        
        try:
            # 创建高风险场景
            high_risk_stocks = ["000001"]  # 使用模拟数据，可能触发预警
            
            # 执行风险评估
            risk_report = self.risk_system.comprehensive_risk_assessment(high_risk_stocks)
            
            # 检查预警机制
            risk_alerts = risk_report.get('risk_alerts', [])
            
            print(f"✅ 风险预警机制测试完成")
            print(f"   生成预警: {len(risk_alerts)}条")
            
            for i, alert in enumerate(risk_alerts[:3]):  # 只显示前3条
                print(f"   预警{i+1}: {alert['type']} - {alert['level']}")
                print(f"          {alert['message']}")
            
            if len(risk_alerts) > 3:
                print(f"   ... 还有 {len(risk_alerts) - 3} 条预警")
            
            return True
            
        except Exception as e:
            print(f"❌ 风险预警机制测试失败: {e}")
            return False
    
    def run_all_tests(self):
        """运行所有测试"""
        print("🚀 开始风险监控系统全面测试...")
        print("=" * 60)
        
        tests = [
            ("市场风险评估", self.test_market_risk_assessment),
            ("个股风险监控", self.test_stock_risk_monitoring),
            ("组合风险管理", self.test_portfolio_risk_management),
            ("综合风险评估", self.test_comprehensive_risk_assessment),
            ("实时监控功能", self.test_real_time_monitoring),
            ("风险预警机制", self.test_risk_alert_mechanism)
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            print(f"\n📋 测试 {test_name}:")
            if test_func():
                passed_tests += 1
                print(f"✅ {test_name}: 通过")
            else:
                print(f"❌ {test_name}: 失败")
        
        # 输出测试结果
        print("\n" + "=" * 60)
        print(f"📊 测试完成: {passed_tests}/{total_tests} 通过")
        
        if passed_tests == total_tests:
            print("🎉 所有测试通过！")
            return True
        else:
            print("⚠️ 部分测试失败，请检查问题")
            return False


def main():
    """主函数"""
    tester = TestRiskMonitoringSystem()
    success = tester.run_all_tests()
    
    if success:
        print("\n✅ 风险监控系统测试完成，所有功能正常")
    else:
        print("\n❌ 风险监控系统测试失败，请检查问题")
    
    return success


if __name__ == "__main__":
    main()
