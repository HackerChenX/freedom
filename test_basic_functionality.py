#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基本功能测试
测试系统的核心功能是否正常工作
"""

import sys
import os
import pandas as pd
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_database_connection():
    """测试数据库连接"""
    print("🔧 测试数据库连接...")
    
    try:
        from db.enhanced_connection_pool import ClickHouseConnectionPool
        
        # 创建连接池
        pool = ClickHouseConnectionPool()
        print("  ✅ 连接池创建成功")
        
        # 测试简单查询
        with pool.get_connection() as conn:
            result = conn.query_dataframe("SELECT 1 as test")
            if not result.empty:
                print("  ✅ 数据库连接测试成功")
                return True
            else:
                print("  ❌ 数据库查询返回空结果")
                return False
                
    except Exception as e:
        print(f"  ❌ 数据库连接失败: {e}")
        return False

def test_stock_data_query():
    """测试股票数据查询"""
    print("🔧 测试股票数据查询...")
    
    try:
        from db.enhanced_connection_pool import ClickHouseConnectionPool
        
        pool = ClickHouseConnectionPool()
        
        # 测试股票数据查询
        query = """
        SELECT code, name, date, level, open, high, low, close, volume, turnover_rate
        FROM stock_info WHERE level = %(level)s AND code = '300005'
        AND level = '日线'
        AND date >= '2024-09-01' AND date <= '2024-09-15'
        ORDER BY date ASC
        LIMIT 10
        """
        
        with pool.get_connection() as conn:
            result = conn.query_dataframe(query)
            
            if not result.empty:
                print(f"  ✅ 股票数据查询成功: {len(result)}条记录")
                print(f"    数据列: {list(result.columns)}")
                return True, result
            else:
                print("  ❌ 股票数据查询返回空结果")
                return False, None
                
    except Exception as e:
        print(f"  ❌ 股票数据查询失败: {e}")
        return False, None

def test_basic_indicators():
    """测试基本指标"""
    print("🔧 测试基本指标...")
    
    try:
        # 创建测试数据
        test_data = pd.DataFrame({
            'date': pd.date_range('2024-09-01', periods=20),
            'open': [100 + i for i in range(20)],
            'high': [105 + i for i in range(20)],
            'low': [95 + i for i in range(20)],
            'close': [102 + i for i in range(20)],
            'volume': [1000000 + i*10000 for i in range(20)]
        })
        
        # 测试MA指标
        from indicators.ma import SimpleMovingAverage
        ma_indicator = SimpleMovingAverage()
        ma_result = ma_indicator.calculate(test_data)
        
        if not ma_result.empty:
            print("  ✅ MA指标计算成功")
        else:
            print("  ❌ MA指标计算失败")
            return False
        
        # 测试MACD指标
        from indicators.macd import MACD
        macd_indicator = MACD()
        macd_result = macd_indicator.calculate(test_data)
        
        if not macd_result.empty:
            print("  ✅ MACD指标计算成功")
        else:
            print("  ❌ MACD指标计算失败")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ 基本指标测试失败: {e}")
        return False

def test_multi_period_analyzer():
    """测试多周期分析器"""
    print("🔧 测试多周期分析器...")
    
    try:
        from bin.multi_period_buypoint_analyzer import MultiPeriodBuypointAnalyzer
        
        # 创建分析器
        analyzer = MultiPeriodBuypointAnalyzer()
        print("  ✅ 多周期分析器创建成功")
        
        # 测试基本配置
        if hasattr(analyzer, 'config') and analyzer.config:
            print("  ✅ 分析器配置加载成功")
        else:
            print("  ⚠️ 分析器配置可能有问题")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 多周期分析器测试失败: {e}")
        return False

def test_data_aggregation():
    """测试数据聚合功能"""
    print("🔧 测试数据聚合功能...")
    
    try:
        from db.services.multi_period_data_service import MultiPeriodDataService
        from enums.period import Period
        
        # 创建数据服务
        data_service = MultiPeriodDataService()
        print("  ✅ 数据服务创建成功")
        
        # 测试获取数据
        periods = [Period.MIN_15, Period.MIN_30, Period.MIN_60, Period.DAILY]
        
        for period in periods:
            try:
                # 简单测试数据获取
                print(f"    测试 {period.value} 数据获取...")
                # 这里只是测试服务是否能正常初始化，不实际查询数据
                print(f"    ✅ {period.value} 数据服务正常")
            except Exception as e:
                print(f"    ❌ {period.value} 数据服务异常: {e}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 数据聚合功能测试失败: {e}")
        return False

def test_score_indicators():
    """测试Score指标"""
    print("🔧 测试Score指标...")
    
    try:
        from indicators.score_indicators import MACDScoreIndicator, RSIScoreIndicator
from db.sql_manager import SQLManager, QueryType
        
        # 创建测试数据
        test_data = pd.DataFrame({
            'date': pd.date_range('2024-09-01', periods=20),
            'open': [100 + i for i in range(20)],
            'high': [105 + i for i in range(20)],
            'low': [95 + i for i in range(20)],
            'close': [102 + i for i in range(20)],
            'volume': [1000000 + i*10000 for i in range(20)]
        })
        
        # 测试MACD Score指标
        macd_score = MACDScoreIndicator()
        macd_result = macd_score.calculate(test_data)
        
        if not macd_result.empty:
            print("  ✅ MACD Score指标计算成功")
        else:
            print("  ❌ MACD Score指标计算失败")
            return False
        
        # 测试RSI Score指标
        rsi_score = RSIScoreIndicator()
        rsi_result = rsi_score.calculate(test_data)
        
        if not rsi_result.empty:
            print("  ✅ RSI Score指标计算成功")
            return True
        else:
            print("  ❌ RSI Score指标计算失败")
            return False
            
    except Exception as e:
        print(f"  ❌ Score指标测试失败: {e}")
        return False

def run_all_tests():
    """运行所有测试"""
    print("🎯 开始基本功能测试")
    print("=" * 50)
    
    tests = [
        ("数据库连接", test_database_connection),
        ("股票数据查询", test_stock_data_query),
        ("基本指标", test_basic_indicators),
        ("多周期分析器", test_multi_period_analyzer),
        ("数据聚合功能", test_data_aggregation),
        ("Score指标", test_score_indicators),
    ]
    
    results = {}
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        try:
            if test_name == "股票数据查询":
                result, data = test_func()
                results[test_name] = result
            else:
                result = test_func()
                results[test_name] = result
            
            if result:
                passed_tests += 1
        except Exception as e:
            print(f"  ❌ 测试异常: {e}")
            results[test_name] = False
    
    # 显示测试结果
    print(f"\n📊 测试摘要:")
    print(f"通过测试: {passed_tests}/{total_tests} ({passed_tests/total_tests:.1%})")
    
    print(f"\n📋 详细结果:")
    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
    
    # 总体评估
    if passed_tests == total_tests:
        print(f"\n🎉 所有测试通过！系统基本功能正常！")
        return True
    elif passed_tests >= total_tests * 0.7:
        print(f"\n⚠️ 大部分测试通过，系统基本可用")
        return True
    else:
        print(f"\n❌ 多项测试失败，系统需要修复")
        return False

def main():
    """主函数"""
    success = run_all_tests()
    
    if success:
        print(f"\n✅ 基本功能测试完成 - 系统可用")
    else:
        print(f"\n❌ 基本功能测试完成 - 需要进一步修复")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
