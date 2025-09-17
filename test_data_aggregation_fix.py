#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试数据聚合功能
验证30分钟和60分钟数据是否能从15分钟数据正确聚合
"""

import sys
import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enums.period import Period
from db.services.multi_period_data_service import MultiPeriodDataService
from utils.logger import get_logger

logger = get_logger(__name__)

def test_data_aggregation():
    """测试数据聚合功能"""
    print("🔧 开始测试数据聚合功能")
    print("=" * 50)
    
    try:
        # 初始化数据服务
        data_service = MultiPeriodDataService()
        test_stocks = ['300005', '603359']  # 探路者、东珠生态
        target_date = '2024-09-15'
        
        results = {
            "test_type": "DATA_AGGREGATION_TEST",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "stocks_tested": test_stocks,
            "target_date": target_date,
            "test_results": {},
            "summary": {}
        }
        
        # 测试每只股票的数据聚合
        for stock_code in test_stocks:
            print(f"\n📊 测试股票: {stock_code}")
            stock_result = test_stock_aggregation(data_service, stock_code, target_date)
            results["test_results"][stock_code] = stock_result
        
        # 计算总结
        calculate_summary(results)
        
        return results
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return {"error": str(e)}

def test_stock_aggregation(data_service, stock_code: str, target_date: str) -> Dict[str, Any]:
    """测试单只股票的数据聚合"""
    stock_result = {
        "stock_code": stock_code,
        "periods_tested": {},
        "aggregation_tests": {},
        "data_coverage": {},
        "issues_found": []
    }
    
    # 1. 测试基础数据可用性
    print(f"  📋 检查基础数据可用性")
    base_data_result = check_base_data_availability(data_service, stock_code, target_date)
    stock_result["base_data"] = base_data_result
    
    # 2. 测试各周期数据获取
    print(f"  📊 测试各周期数据获取")
    for period in [Period.MIN_15, Period.MIN_30, Period.MIN_60, Period.DAILY, Period.WEEKLY, Period.MONTHLY]:
        period_result = test_period_data(data_service, stock_code, target_date, period)
        stock_result["periods_tested"][period.value] = period_result
    
    # 3. 专门测试聚合功能
    print(f"  🔧 测试数据聚合功能")
    aggregation_result = test_aggregation_logic(data_service, stock_code, target_date)
    stock_result["aggregation_tests"] = aggregation_result
    
    # 4. 计算数据覆盖率
    coverage = calculate_data_coverage(stock_result["periods_tested"])
    stock_result["data_coverage"] = coverage
    
    return stock_result

def check_base_data_availability(data_service, stock_code: str, target_date: str) -> Dict[str, Any]:
    """检查基础数据可用性"""
    base_result = {
        "15min_available": False,
        "15min_count": 0,
        "daily_available": False,
        "daily_count": 0,
        "issues": []
    }
    
    try:
        # 检查15分钟数据
        min15_data = data_service._query_base_period_data(
            stock_code=stock_code,
            target_date=target_date,
            period=Period.MIN_15,
            lookback_days=30
        )
        
        if not min15_data.empty:
            base_result["15min_available"] = True
            base_result["15min_count"] = len(min15_data)
            print(f"    ✅ 15分钟数据: {len(min15_data)}条")
        else:
            base_result["issues"].append("15分钟数据为空")
            print(f"    ❌ 15分钟数据为空")
        
        # 检查日线数据
        daily_data = data_service._query_base_period_data(
            stock_code=stock_code,
            target_date=target_date,
            period=Period.DAILY,
            lookback_days=30
        )
        
        if not daily_data.empty:
            base_result["daily_available"] = True
            base_result["daily_count"] = len(daily_data)
            print(f"    ✅ 日线数据: {len(daily_data)}条")
        else:
            base_result["issues"].append("日线数据为空")
            print(f"    ❌ 日线数据为空")
            
    except Exception as e:
        base_result["issues"].append(f"基础数据检查失败: {e}")
        print(f"    ❌ 基础数据检查失败: {e}")
    
    return base_result

def test_period_data(data_service, stock_code: str, target_date: str, period: Period) -> Dict[str, Any]:
    """测试单个周期的数据获取"""
    period_result = {
        "period": period.value,
        "data_available": False,
        "data_count": 0,
        "aggregated": False,
        "issues": []
    }
    
    try:
        data = data_service.get_single_period_data(
            stock_code=stock_code,
            target_date=target_date,
            period=period,
            lookback_days=30
        )
        
        if not data.empty:
            period_result["data_available"] = True
            period_result["data_count"] = len(data)
            
            # 检查是否是聚合数据（通过level字段判断）
            if 'level' in data.columns and period.value in data['level'].values:
                period_result["aggregated"] = True
            
            print(f"    ✅ {period.value}: {len(data)}条数据")
        else:
            period_result["issues"].append(f"{period.value}数据为空")
            print(f"    ❌ {period.value}: 数据为空")
            
    except Exception as e:
        period_result["issues"].append(f"{period.value}数据获取失败: {e}")
        print(f"    ❌ {period.value}: 获取失败 - {e}")
    
    return period_result

def test_aggregation_logic(data_service, stock_code: str, target_date: str) -> Dict[str, Any]:
    """测试聚合逻辑"""
    aggregation_result = {
        "30min_aggregation": {"success": False, "details": ""},
        "60min_aggregation": {"success": False, "details": ""},
        "aggregation_quality": {}
    }
    
    try:
        # 测试30分钟聚合
        print(f"    🔧 测试30分钟数据聚合")
        min30_data = data_service._aggregate_from_base_period(
            stock_code=stock_code,
            target_date=target_date,
            target_period=Period.MIN_30,
            lookback_days=10
        )
        
        if not min30_data.empty:
            aggregation_result["30min_aggregation"]["success"] = True
            aggregation_result["30min_aggregation"]["details"] = f"成功聚合{len(min30_data)}条30分钟数据"
            print(f"      ✅ 30分钟聚合成功: {len(min30_data)}条")
        else:
            aggregation_result["30min_aggregation"]["details"] = "30分钟聚合失败，结果为空"
            print(f"      ❌ 30分钟聚合失败")
        
        # 测试60分钟聚合
        print(f"    🔧 测试60分钟数据聚合")
        min60_data = data_service._aggregate_from_base_period(
            stock_code=stock_code,
            target_date=target_date,
            target_period=Period.MIN_60,
            lookback_days=10
        )
        
        if not min60_data.empty:
            aggregation_result["60min_aggregation"]["success"] = True
            aggregation_result["60min_aggregation"]["details"] = f"成功聚合{len(min60_data)}条60分钟数据"
            print(f"      ✅ 60分钟聚合成功: {len(min60_data)}条")
        else:
            aggregation_result["60min_aggregation"]["details"] = "60分钟聚合失败，结果为空"
            print(f"      ❌ 60分钟聚合失败")
            
    except Exception as e:
        aggregation_result["error"] = f"聚合测试失败: {e}"
        print(f"    ❌ 聚合测试失败: {e}")
    
    return aggregation_result

def calculate_data_coverage(periods_tested: Dict[str, Any]) -> Dict[str, Any]:
    """计算数据覆盖率"""
    total_periods = len(periods_tested)
    available_periods = sum(1 for result in periods_tested.values() if result["data_available"])
    
    coverage = {
        "total_periods": total_periods,
        "available_periods": available_periods,
        "coverage_rate": available_periods / total_periods if total_periods > 0 else 0,
        "missing_periods": [
            period for period, result in periods_tested.items() 
            if not result["data_available"]
        ]
    }
    
    return coverage

def calculate_summary(results: Dict[str, Any]):
    """计算测试总结"""
    summary = results["summary"]
    
    # 统计各股票的数据覆盖率
    coverage_rates = []
    aggregation_success_rates = []
    
    for stock_code, stock_result in results["test_results"].items():
        if "data_coverage" in stock_result:
            coverage_rates.append(stock_result["data_coverage"]["coverage_rate"])
        
        if "aggregation_tests" in stock_result:
            agg_tests = stock_result["aggregation_tests"]
            success_count = 0
            total_count = 0
            
            for test_name, test_result in agg_tests.items():
                if isinstance(test_result, dict) and "success" in test_result:
                    total_count += 1
                    if test_result["success"]:
                        success_count += 1
            
            if total_count > 0:
                aggregation_success_rates.append(success_count / total_count)
    
    # 计算平均值
    summary["average_coverage_rate"] = sum(coverage_rates) / len(coverage_rates) if coverage_rates else 0
    summary["average_aggregation_success_rate"] = sum(aggregation_success_rates) / len(aggregation_success_rates) if aggregation_success_rates else 0
    
    # 总体评估
    if summary["average_coverage_rate"] >= 0.8:
        summary["coverage_status"] = "良好"
    elif summary["average_coverage_rate"] >= 0.5:
        summary["coverage_status"] = "一般"
    else:
        summary["coverage_status"] = "较差"

def main():
    """主函数"""
    print("🔧 数据聚合功能测试")
    print("=" * 50)
    
    results = test_data_aggregation()
    
    if "error" in results:
        print(f"❌ 测试失败: {results['error']}")
        return False
    
    # 显示测试结果
    print(f"\n📊 测试摘要:")
    print(f"平均数据覆盖率: {results['summary']['average_coverage_rate']:.1%}")
    print(f"平均聚合成功率: {results['summary']['average_aggregation_success_rate']:.1%}")
    print(f"覆盖状态: {results['summary']['coverage_status']}")
    
    # 显示详细结果
    print(f"\n📋 详细结果:")
    for stock_code, stock_result in results["test_results"].items():
        print(f"  📈 {stock_code}:")
        if "data_coverage" in stock_result:
            coverage = stock_result["data_coverage"]
            print(f"    数据覆盖率: {coverage['coverage_rate']:.1%} ({coverage['available_periods']}/{coverage['total_periods']})")
            if coverage["missing_periods"]:
                print(f"    缺失周期: {', '.join(coverage['missing_periods'])}")
        
        if "aggregation_tests" in stock_result:
            agg_tests = stock_result["aggregation_tests"]
            for test_name, test_result in agg_tests.items():
                if isinstance(test_result, dict) and "success" in test_result:
                    status = "✅" if test_result["success"] else "❌"
                    print(f"    {test_name}: {status} {test_result.get('details', '')}")
    
    print(f"\n✅ 数据聚合功能测试完成")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
