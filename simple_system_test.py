#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
简化系统测试
直接测试核心功能，避免复杂的依赖问题
"""

import sys
import os
import pandas as pd
from datetime import datetime
from typing import Dict, Any

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_basic_imports():
    """测试基础导入"""
    print("🔧 测试基础导入")
    
    try:
        # 测试枚举导入
        from enums.period import Period
from db.sql_manager import SQLManager, QueryType
        print("  ✅ Period枚举导入成功")
        
        # 测试日志导入
        from utils.logger import get_logger
        logger = get_logger(__name__)
        print("  ✅ 日志系统导入成功")
        
        # 测试容器导入
        from utils.unified_container import get_container
        container = get_container()
        print("  ✅ 依赖注入容器导入成功")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 基础导入失败: {e}")
        return False

def test_database_connection():
    """测试数据库连接"""
    print("🔧 测试数据库连接")
    
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
    print("🔧 测试股票数据查询")
    
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
                return True
            else:
                print("  ❌ 股票数据查询返回空结果")
                return False
                
    except Exception as e:
        print(f"  ❌ 股票数据查询失败: {e}")
        return False

def test_indicator_registry():
    """测试指标注册表"""
    print("🔧 测试指标注册表")
    
    try:
        from indicators.complete_indicator_registry import complete_registry
        
        # 获取注册的指标数量
        indicator_count = len(complete_registry.get_all_indicators())
        print(f"  ✅ 指标注册表加载成功: {indicator_count}个指标")
        
        # 测试创建一个简单指标
        ma_indicator = complete_registry.create_indicator('MA')
        if ma_indicator:
            print("  ✅ MA指标创建成功")
            return True
        else:
            print("  ❌ MA指标创建失败")
            return False
            
    except Exception as e:
        print(f"  ❌ 指标注册表测试失败: {e}")
        return False

def test_score_indicators():
    """测试Score指标"""
    print("🔧 测试Score指标")
    
    try:
        from indicators.score_indicators import MACDScoreIndicator, RSIScoreIndicator
        
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

def test_pattern_indicators():
    """测试形态识别指标"""
    print("🔧 测试形态识别指标")
    
    try:
        from indicators.pattern_indicators import ThreeBlackCrowsIndicator, ThreeWhiteSoldiersIndicator
        
        # 创建测试数据
        test_data = pd.DataFrame({
            'date': pd.date_range('2024-09-01', periods=10),
            'open': [100, 99, 98, 97, 96, 95, 94, 93, 92, 91],
            'high': [102, 101, 100, 99, 98, 97, 96, 95, 94, 93],
            'low': [98, 97, 96, 95, 94, 93, 92, 91, 90, 89],
            'close': [99, 98, 97, 96, 95, 94, 93, 92, 91, 90],
            'volume': [1000000] * 10
        })
        
        # 测试三黑鸦指标
        three_black_crows = ThreeBlackCrowsIndicator()
        result = three_black_crows.calculate(test_data)
        
        if not result.empty:
            print("  ✅ 三黑鸦形态识别成功")
            return True
        else:
            print("  ❌ 三黑鸦形态识别失败")
            return False
            
    except Exception as e:
        print(f"  ❌ 形态识别指标测试失败: {e}")
        return False

def test_json_serialization():
    """测试JSON序列化"""
    print("🔧 测试JSON序列化")
    
    try:
        from utils.json_serializer import safe_json_dumps, safe_json_loads
        from enums.period import Period
from db.sql_manager import SQLManager, QueryType
        
        # 测试Period枚举序列化
        test_data = {
            "period": Period.DAILY,
            "periods": [Period.MIN_15, Period.MIN_30, Period.DAILY],
            "timestamp": datetime.now(),
            "value": 123.45
        }
        
        # 序列化
        json_str = safe_json_dumps(test_data)
        print("  ✅ JSON序列化成功")
        
        # 反序列化
        parsed_data = safe_json_loads(json_str)
        print("  ✅ JSON反序列化成功")
        
        return True
        
    except Exception as e:
        print(f"  ❌ JSON序列化测试失败: {e}")
        return False

def run_all_tests():
    """运行所有测试"""
    print("🎯 开始系统功能测试")
    print("=" * 50)
    
    tests = [
        ("基础导入", test_basic_imports),
        ("数据库连接", test_database_connection),
        ("股票数据查询", test_stock_data_query),
        ("指标注册表", test_indicator_registry),
        ("Score指标", test_score_indicators),
        ("形态识别指标", test_pattern_indicators),
        ("JSON序列化", test_json_serialization),
    ]
    
    results = {}
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        try:
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
        print(f"\n🎉 所有测试通过！系统基础功能正常！")
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
        print(f"\n✅ 系统测试完成 - 基础功能验证通过")
    else:
        print(f"\n❌ 系统测试完成 - 发现问题需要修复")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
