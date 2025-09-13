#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
RESTful API系统综合测试

测试所有API端点的功能和性能
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Any

from fastapi.testclient import TestClient
from utils.logger import get_logger

logger = get_logger(__name__)

class RESTfulAPITester:
    """RESTful API系统测试器"""
    
    def __init__(self):
        """初始化测试器"""
        self.client = None
        self.test_results = []
        self.start_time = None
        
    def setup_test_client(self):
        """设置测试客户端"""
        try:
            from api.main import app
            self.client = TestClient(app)
            logger.info("✅ 测试客户端初始化成功")
            return True
        except Exception as e:
            logger.error(f"❌ 测试客户端初始化失败: {e}")
            return False
    
    def test_system_endpoints(self) -> Dict[str, Any]:
        """测试系统端点"""
        logger.info("🔍 测试系统端点...")
        results = {"category": "系统端点", "tests": []}
        
        # 测试健康检查
        try:
            response = self.client.get("/health")
            results["tests"].append({
                "name": "健康检查",
                "status": "PASS" if response.status_code == 200 else "FAIL",
                "status_code": response.status_code,
                "response_time": getattr(response, 'elapsed', 0),
                "details": response.json() if response.status_code == 200 else response.text
            })
        except Exception as e:
            results["tests"].append({
                "name": "健康检查",
                "status": "ERROR",
                "error": str(e)
            })
        
        # 测试系统信息
        try:
            response = self.client.get("/info")
            results["tests"].append({
                "name": "系统信息",
                "status": "PASS" if response.status_code == 200 else "FAIL",
                "status_code": response.status_code,
                "response_time": getattr(response, 'elapsed', 0),
                "details": response.json() if response.status_code == 200 else response.text
            })
        except Exception as e:
            results["tests"].append({
                "name": "系统信息",
                "status": "ERROR",
                "error": str(e)
            })
        
        return results
    
    def test_stock_data_endpoints(self) -> Dict[str, Any]:
        """测试股票数据端点"""
        logger.info("📊 测试股票数据端点...")
        results = {"category": "股票数据", "tests": []}
        
        # 测试股票列表
        try:
            response = self.client.get("/api/v1/stocks")
            results["tests"].append({
                "name": "股票列表",
                "status": "PASS" if response.status_code == 200 else "FAIL",
                "status_code": response.status_code,
                "details": response.json() if response.status_code == 200 else response.text
            })
        except Exception as e:
            results["tests"].append({
                "name": "股票列表",
                "status": "ERROR",
                "error": str(e)
            })
        
        # 测试股票数据查询
        try:
            response = self.client.get(
                "/api/v1/stocks/000001/data",
                params={
                    "start_date": "2024-01-01",
                    "end_date": "2024-12-31"
                }
            )
            results["tests"].append({
                "name": "股票数据查询",
                "status": "PASS" if response.status_code == 200 else "FAIL",
                "status_code": response.status_code,
                "details": response.json() if response.status_code == 200 else response.text
            })
        except Exception as e:
            results["tests"].append({
                "name": "股票数据查询",
                "status": "ERROR",
                "error": str(e)
            })
        
        # 测试最新股票数据
        try:
            response = self.client.get("/api/v1/stocks/000001/latest")
            results["tests"].append({
                "name": "最新股票数据",
                "status": "PASS" if response.status_code == 200 else "FAIL",
                "status_code": response.status_code,
                "details": response.json() if response.status_code == 200 else response.text
            })
        except Exception as e:
            results["tests"].append({
                "name": "最新股票数据",
                "status": "ERROR",
                "error": str(e)
            })
        
        return results
    
    def test_indicator_endpoints(self) -> Dict[str, Any]:
        """测试技术指标端点"""
        logger.info("📈 测试技术指标端点...")
        results = {"category": "技术指标", "tests": []}
        
        # 测试指标列表
        try:
            response = self.client.get("/api/v1/indicators")
            results["tests"].append({
                "name": "指标列表",
                "status": "PASS" if response.status_code == 200 else "FAIL",
                "status_code": response.status_code,
                "details": response.json() if response.status_code == 200 else response.text
            })
        except Exception as e:
            results["tests"].append({
                "name": "指标列表",
                "status": "ERROR",
                "error": str(e)
            })
        
        # 测试指标计算
        try:
            response = self.client.post(
                "/api/v1/indicators/calculate",
                json={
                    "stock_code": "000001",
                    "indicator_name": "MA",
                    "start_date": "2024-01-01",
                    "end_date": "2024-12-31",
                    "parameters": {"period": 20}
                }
            )
            results["tests"].append({
                "name": "指标计算",
                "status": "PASS" if response.status_code == 200 else "FAIL",
                "status_code": response.status_code,
                "details": response.json() if response.status_code == 200 else response.text
            })
        except Exception as e:
            results["tests"].append({
                "name": "指标计算",
                "status": "ERROR",
                "error": str(e)
            })
        
        # 测试指标信息
        try:
            response = self.client.get("/api/v1/indicators/MA")
            results["tests"].append({
                "name": "指标信息",
                "status": "PASS" if response.status_code == 200 else "FAIL",
                "status_code": response.status_code,
                "details": response.json() if response.status_code == 200 else response.text
            })
        except Exception as e:
            results["tests"].append({
                "name": "指标信息",
                "status": "ERROR",
                "error": str(e)
            })
        
        return results
    
    def test_strategy_endpoints(self) -> Dict[str, Any]:
        """测试策略分析端点"""
        logger.info("🎯 测试策略分析端点...")
        results = {"category": "策略分析", "tests": []}
        
        # 测试策略列表
        try:
            response = self.client.get("/api/v1/strategies")
            results["tests"].append({
                "name": "策略列表",
                "status": "PASS" if response.status_code == 200 else "FAIL",
                "status_code": response.status_code,
                "details": response.json() if response.status_code == 200 else response.text
            })
        except Exception as e:
            results["tests"].append({
                "name": "策略列表",
                "status": "ERROR",
                "error": str(e)
            })
        
        # 测试策略分析
        try:
            response = self.client.post(
                "/api/v1/strategies/analyze",
                json={
                    "strategy_name": "趋势跟踪策略",
                    "stock_codes": ["000001", "000002"],
                    "start_date": "2024-01-01",
                    "end_date": "2024-12-31"
                }
            )
            results["tests"].append({
                "name": "策略分析",
                "status": "PASS" if response.status_code == 200 else "FAIL",
                "status_code": response.status_code,
                "details": response.json() if response.status_code == 200 else response.text
            })
        except Exception as e:
            results["tests"].append({
                "name": "策略分析",
                "status": "ERROR",
                "error": str(e)
            })
        
        # 测试策略回测
        try:
            response = self.client.post(
                "/api/v1/strategies/backtest",
                json={
                    "strategy_name": "趋势跟踪策略",
                    "stock_codes": ["000001"],
                    "start_date": "2024-01-01",
                    "end_date": "2024-12-31",
                    "initial_capital": 100000.0
                }
            )
            results["tests"].append({
                "name": "策略回测",
                "status": "PASS" if response.status_code == 200 else "FAIL",
                "status_code": response.status_code,
                "details": response.json() if response.status_code == 200 else response.text
            })
        except Exception as e:
            results["tests"].append({
                "name": "策略回测",
                "status": "ERROR",
                "error": str(e)
            })
        
        return results
    
    def test_risk_endpoints(self) -> Dict[str, Any]:
        """测试风险监控端点"""
        logger.info("⚠️ 测试风险监控端点...")
        results = {"category": "风险监控", "tests": []}
        
        # 测试风险评估
        try:
            response = self.client.post(
                "/api/v1/risk/assess",
                json={
                    "stock_codes": ["000001", "000002"],
                    "assessment_type": "comprehensive",
                    "time_horizon": 30
                }
            )
            results["tests"].append({
                "name": "风险评估",
                "status": "PASS" if response.status_code == 200 else "FAIL",
                "status_code": response.status_code,
                "details": response.json() if response.status_code == 200 else response.text
            })
        except Exception as e:
            results["tests"].append({
                "name": "风险评估",
                "status": "ERROR",
                "error": str(e)
            })
        
        # 测试市场风险
        try:
            response = self.client.get("/api/v1/risk/market")
            results["tests"].append({
                "name": "市场风险",
                "status": "PASS" if response.status_code == 200 else "FAIL",
                "status_code": response.status_code,
                "details": response.json() if response.status_code == 200 else response.text
            })
        except Exception as e:
            results["tests"].append({
                "name": "市场风险",
                "status": "ERROR",
                "error": str(e)
            })
        
        # 测试个股风险
        try:
            response = self.client.get("/api/v1/risk/stock/000001")
            results["tests"].append({
                "name": "个股风险",
                "status": "PASS" if response.status_code == 200 else "FAIL",
                "status_code": response.status_code,
                "details": response.json() if response.status_code == 200 else response.text
            })
        except Exception as e:
            results["tests"].append({
                "name": "个股风险",
                "status": "ERROR",
                "error": str(e)
            })
        
        return results
    
    def test_monitoring_endpoints(self) -> Dict[str, Any]:
        """测试实时监控端点"""
        logger.info("📡 测试实时监控端点...")
        results = {"category": "实时监控", "tests": []}
        
        # 测试监控状态
        try:
            response = self.client.get("/api/v1/monitoring/status")
            results["tests"].append({
                "name": "监控状态",
                "status": "PASS" if response.status_code == 200 else "FAIL",
                "status_code": response.status_code,
                "details": response.json() if response.status_code == 200 else response.text
            })
        except Exception as e:
            results["tests"].append({
                "name": "监控状态",
                "status": "ERROR",
                "error": str(e)
            })
        
        # 测试监控预警
        try:
            response = self.client.get("/api/v1/monitoring/alerts")
            results["tests"].append({
                "name": "监控预警",
                "status": "PASS" if response.status_code == 200 else "FAIL",
                "status_code": response.status_code,
                "details": response.json() if response.status_code == 200 else response.text
            })
        except Exception as e:
            results["tests"].append({
                "name": "监控预警",
                "status": "ERROR",
                "error": str(e)
            })
        
        # 测试预警规则
        try:
            response = self.client.get("/api/v1/monitoring/alerts/rules")
            results["tests"].append({
                "name": "预警规则",
                "status": "PASS" if response.status_code == 200 else "FAIL",
                "status_code": response.status_code,
                "details": response.json() if response.status_code == 200 else response.text
            })
        except Exception as e:
            results["tests"].append({
                "name": "预警规则",
                "status": "ERROR",
                "error": str(e)
            })
        
        return results
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """运行综合测试"""
        logger.info("🚀 开始RESTful API系统综合测试...")
        self.start_time = time.time()
        
        # 设置测试客户端
        if not self.setup_test_client():
            return {"error": "测试客户端初始化失败"}
        
        # 执行所有测试
        test_categories = [
            self.test_system_endpoints,
            self.test_stock_data_endpoints,
            self.test_indicator_endpoints,
            self.test_strategy_endpoints,
            self.test_risk_endpoints,
            self.test_monitoring_endpoints
        ]
        
        for test_func in test_categories:
            try:
                result = test_func()
                self.test_results.append(result)
            except Exception as e:
                logger.error(f"测试类别 {test_func.__name__} 失败: {e}")
                self.test_results.append({
                    "category": test_func.__name__,
                    "error": str(e),
                    "tests": []
                })
        
        # 生成测试报告
        return self.generate_test_report()
    
    def generate_test_report(self) -> Dict[str, Any]:
        """生成测试报告"""
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        error_tests = 0
        
        for category in self.test_results:
            for test in category.get("tests", []):
                total_tests += 1
                if test.get("status") == "PASS":
                    passed_tests += 1
                elif test.get("status") == "FAIL":
                    failed_tests += 1
                elif test.get("status") == "ERROR":
                    error_tests += 1
        
        test_time = time.time() - self.start_time if self.start_time else 0
        
        report = {
            "test_summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "error_tests": error_tests,
                "success_rate": round((passed_tests / total_tests * 100), 2) if total_tests > 0 else 0,
                "test_time": round(test_time, 2)
            },
            "test_results": self.test_results,
            "timestamp": datetime.now().isoformat()
        }
        
        return report

def main():
    """主函数"""
    print("=" * 80)
    print("🚀 RESTful API系统综合测试")
    print("=" * 80)
    
    tester = RESTfulAPITester()
    
    try:
        # 运行综合测试
        report = tester.run_comprehensive_test()
        
        # 打印测试报告
        print("\n📊 测试报告:")
        print("-" * 60)
        
        summary = report["test_summary"]
        print(f"总测试数: {summary['total_tests']}")
        print(f"通过测试: {summary['passed_tests']}")
        print(f"失败测试: {summary['failed_tests']}")
        print(f"错误测试: {summary['error_tests']}")
        print(f"成功率: {summary['success_rate']}%")
        print(f"测试耗时: {summary['test_time']}秒")
        
        # 详细结果
        print("\n📋 详细测试结果:")
        print("-" * 60)
        
        for category in report["test_results"]:
            print(f"\n🔍 {category['category']}:")
            for test in category.get("tests", []):
                status_icon = "✅" if test.get("status") == "PASS" else "❌" if test.get("status") == "FAIL" else "⚠️"
                print(f"  {status_icon} {test['name']}: {test.get('status', 'UNKNOWN')}")
                if test.get("error"):
                    print(f"    错误: {test['error']}")
        
        # 保存详细报告
        report_file = "results/api_test_report.json"
        os.makedirs(os.path.dirname(report_file), exist_ok=True)
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\n📄 详细报告已保存到: {report_file}")
        
        # 判断测试是否成功
        if summary['success_rate'] >= 80:
            print("\n🎉 RESTful API系统测试通过！")
            return True
        else:
            print("\n❌ RESTful API系统测试失败！")
            return False
        
    except Exception as e:
        logger.error(f"测试执行失败: {e}")
        print(f"\n❌ 测试执行失败: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
