#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
实时数据监控功能测试脚本

测试实时数据监控、智能预警系统等功能
"""

import sys
import os
import time
import pandas as pd
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from monitoring.market_monitor import get_market_monitor, StockMonitoringConfig
from monitoring.intelligent_alert_system import get_intelligent_alert_system
from monitoring.alert_manager import get_alert_manager

logger = get_logger(__name__)


class RealTimeMonitoringTester:
    """实时数据监控测试器"""
    
    def __init__(self):
        """初始化测试器"""
        self.market_monitor = get_market_monitor()
        self.intelligent_alert_system = get_intelligent_alert_system()
        self.alert_manager = get_alert_manager()
        
        # 测试股票列表
        self.test_stocks = [
            {"code": "000001", "name": "平安银行"},
            {"code": "000002", "name": "万科A"},
            {"code": "600036", "name": "招商银行"},
            {"code": "600519", "name": "贵州茅台"},
            {"code": "000858", "name": "五粮液"}
        ]
        
        # 测试指标列表
        self.test_indicators = ["RSI", "MACD", "KDJ", "BOLL", "MA"]
        
        logger.info("实时数据监控测试器初始化完成")
    
    def test_market_monitor_basic_functions(self):
        """测试市场监控器基础功能"""
        logger.info("=== 测试市场监控器基础功能 ===")
        
        try:
            # 测试添加监控股票
            logger.info("1. 测试添加监控股票")
            for stock in self.test_stocks[:3]:  # 只测试前3只股票
                config = StockMonitoringConfig(
                    stock_code=stock["code"],
                    stock_name=stock["name"],
                    indicators=self.test_indicators[:3],  # 只使用前3个指标
                    monitoring_interval=30  # 30秒间隔
                )
                
                result = self.market_monitor.real_time_monitor.add_monitoring_stock(config)
                logger.info(f"添加监控股票 {stock['code']}: {'成功' if result else '失败'}")
            
            # 测试启动监控
            logger.info("2. 测试启动监控")
            stock_codes = [stock["code"] for stock in self.test_stocks[:3]]
            start_result = self.market_monitor.start_monitoring(stock_codes, self.test_indicators[:3])
            logger.info(f"启动监控结果: {start_result}")
            
            # 等待一段时间让监控运行
            logger.info("3. 监控运行中，等待30秒...")
            time.sleep(30)
            
            # 测试获取市场状态
            logger.info("4. 测试获取市场状态")
            market_status = self.market_monitor.get_market_status()
            logger.info(f"市场状态: {market_status}")
            
            # 测试获取预警列表
            logger.info("5. 测试获取预警列表")
            alerts = self.market_monitor.get_alerts(limit=10)
            logger.info(f"获取到 {len(alerts)} 个预警")
            for alert in alerts[:3]:  # 只显示前3个
                logger.info(f"预警: {alert['message']} (级别: {alert['level']})")
            
            # 测试获取监控统计
            logger.info("6. 测试获取监控统计")
            stats = self.market_monitor.get_monitoring_statistics()
            logger.info(f"监控统计: {stats}")
            
            # 测试停止监控
            logger.info("7. 测试停止监控")
            stop_result = self.market_monitor.stop_monitoring()
            logger.info(f"停止监控结果: {stop_result}")
            
            logger.info("✅ 市场监控器基础功能测试完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ 市场监控器基础功能测试失败: {e}")
            return False
    
    def test_intelligent_alert_system(self):
        """测试智能预警系统"""
        logger.info("=== 测试智能预警系统 ===")
        
        try:
            # 测试获取预警规则
            logger.info("1. 测试获取预警规则")
            rules = self.intelligent_alert_system.get_alert_rules()
            logger.info(f"获取到 {len(rules)} 个预警规则")
            for rule in rules:
                logger.info(f"规则: {rule['name']} - {rule['description']} (启用: {rule['enabled']})")
            
            # 测试分析股票信号
            logger.info("2. 测试分析股票信号")
            for stock in self.test_stocks[:2]:  # 只测试前2只股票
                logger.info(f"分析股票 {stock['code']} - {stock['name']}")
                signals = self.intelligent_alert_system.analyze_stock_signals(
                    stock_code=stock["code"],
                    stock_name=stock["name"],
                    lookback_days=20
                )
                
                logger.info(f"股票 {stock['code']} 识别到 {len(signals)} 个信号")
                for signal in signals[:2]:  # 只显示前2个信号
                    logger.info(f"信号: {signal.message} (类型: {signal.signal_type.value}, 强度: {signal.signal_strength.value})")
            
            # 测试获取信号列表
            logger.info("3. 测试获取信号列表")
            all_signals = self.intelligent_alert_system.get_signals(limit=10)
            logger.info(f"获取到 {len(all_signals)} 个历史信号")
            
            # 测试获取系统统计
            logger.info("4. 测试获取系统统计")
            stats = self.intelligent_alert_system.get_system_statistics()
            logger.info(f"系统统计: {stats}")
            
            logger.info("✅ 智能预警系统测试完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ 智能预警系统测试失败: {e}")
            return False
    
    def test_alert_manager_integration(self):
        """测试预警管理器集成"""
        logger.info("=== 测试预警管理器集成 ===")
        
        try:
            # 测试设置预警规则
            logger.info("1. 测试设置预警规则")
            alert_configs = [
                {"type": "price_change", "threshold": 0.05},
                {"type": "volume_anomaly", "threshold": 2.0},
                {"type": "rsi_signal", "threshold": 70}
            ]
            
            setup_result = self.alert_manager.setup_alerts(alert_configs)
            logger.info(f"设置预警规则结果: {setup_result}")
            
            # 测试分析股票预警
            logger.info("2. 测试分析股票预警")
            for stock in self.test_stocks[:2]:  # 只测试前2只股票
                logger.info(f"分析股票预警 {stock['code']} - {stock['name']}")
                alert_result = self.alert_manager.analyze_stock_alerts(
                    stock_code=stock["code"],
                    stock_name=stock["name"]
                )
                
                logger.info(f"股票 {stock['code']} 预警分析结果: {alert_result['status']}")
                if alert_result["status"] == "success":
                    logger.info(f"识别到 {alert_result['signals_count']} 个信号")
            
            logger.info("✅ 预警管理器集成测试完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ 预警管理器集成测试失败: {e}")
            return False
    
    def test_performance_and_stability(self):
        """测试性能和稳定性"""
        logger.info("=== 测试性能和稳定性 ===")
        
        try:
            start_time = time.time()
            
            # 批量测试多只股票
            logger.info("1. 批量测试多只股票监控")
            all_stock_codes = [stock["code"] for stock in self.test_stocks]
            
            # 启动监控
            start_result = self.market_monitor.start_monitoring(all_stock_codes, self.test_indicators[:3])
            logger.info(f"批量启动监控: {start_result['status']}")
            
            # 运行一段时间
            logger.info("2. 监控运行60秒...")
            time.sleep(60)
            
            # 检查系统状态
            market_status = self.market_monitor.get_market_status()
            logger.info(f"系统状态: 监控活跃={market_status['monitoring_active']}, 活跃预警={market_status['active_alerts']}")
            
            # 停止监控
            stop_result = self.market_monitor.stop_monitoring()
            logger.info(f"停止监控: {stop_result['status']}")
            
            end_time = time.time()
            total_time = end_time - start_time
            
            logger.info(f"✅ 性能和稳定性测试完成，总耗时: {total_time:.2f}秒")
            return True
            
        except Exception as e:
            logger.error(f"❌ 性能和稳定性测试失败: {e}")
            return False
    
    def run_all_tests(self):
        """运行所有测试"""
        logger.info("🚀 开始实时数据监控功能测试")
        
        test_results = []
        
        # 运行各项测试
        test_results.append(("市场监控器基础功能", self.test_market_monitor_basic_functions()))
        test_results.append(("智能预警系统", self.test_intelligent_alert_system()))
        test_results.append(("预警管理器集成", self.test_alert_manager_integration()))
        test_results.append(("性能和稳定性", self.test_performance_and_stability()))
        
        # 汇总测试结果
        logger.info("📊 测试结果汇总:")
        passed_count = 0
        for test_name, result in test_results:
            status = "✅ 通过" if result else "❌ 失败"
            logger.info(f"  {test_name}: {status}")
            if result:
                passed_count += 1
        
        success_rate = (passed_count / len(test_results)) * 100
        logger.info(f"🎯 测试通过率: {success_rate:.1f}% ({passed_count}/{len(test_results)})")
        
        if success_rate >= 80:
            logger.info("🎉 实时数据监控功能测试整体通过！")
            return True
        else:
            logger.error("💥 实时数据监控功能测试整体失败！")
            return False


def main():
    """主函数"""
    try:
        tester = RealTimeMonitoringTester()
        success = tester.run_all_tests()
        
        if success:
            logger.info("🎊 实时数据监控功能测试全部完成，系统运行正常！")
            return 0
        else:
            logger.error("💔 实时数据监控功能测试失败，需要修复问题！")
            return 1
            
    except Exception as e:
        logger.error(f"💥 测试过程中发生严重错误: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
