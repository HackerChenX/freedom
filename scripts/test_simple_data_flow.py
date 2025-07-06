#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
简化的数据流测试
逐步验证每个环节的数据处理
"""

import os
import sys

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from db.unified_data_manager import get_unified_data_manager
import pandas as pd

def test_data_flow():
    print("=== 简化数据流测试 ===")
    
    try:
        # 1. 测试数据管理器初始化
        print("\n1. 初始化数据管理器...")
        data_manager = get_unified_data_manager()
        print("✅ 数据管理器初始化成功")
        
        # 2. 测试股票列表获取
        print("\n2. 获取股票列表...")
        stock_codes = data_manager.get_all_stock_codes(limit=3)
        print(f"✅ 获取到 {len(stock_codes)} 只股票: {stock_codes}")
        
        if not stock_codes:
            print("❌ 没有股票数据，测试结束")
            return
        
        # 3. 测试单个股票数据查询
        test_stock = stock_codes[0]
        print(f"\n3. 查询股票 {test_stock} 的数据...")
        
        # 查询最近的数据，限制数量避免超时
        stock_info WHERE 1=1 = data_manager.get_stock_info(
            stock_code=test_stock,
            level='日线',
            limit=10,
            order_by="date DESC"
        )
        
        print(f"✅ 查询成功，数据类型: {type(stock_info)}")
        print(f"✅ 数据长度: {len(stock_info)}")
        print(f"✅ 是否为集合: {stock_info.is_collection}")
        
        # 4. 测试DataFrame转换
        print("\n4. 测试DataFrame转换...")
        df = stock_info.to_dataframe()
        print(f"✅ DataFrame转换成功，类型: {type(df)}")
        print(f"✅ DataFrame形状: {df.shape}")
        print(f"✅ DataFrame列名: {list(df.columns)}")
        
        # 5. 显示样本数据
        print("\n5. 样本数据:")
        print(df.head(3))
        
        # 6. 验证关键字段
        print("\n6. 验证关键字段...")
        required_fields = ['code', 'date', 'open', 'high', 'low', 'close', 'volume']
        missing_fields = []
        
        for field in required_fields:
            if field not in df.columns:
                missing_fields.append(field)
            else:
                print(f"✅ 字段 {field}: {df[field].dtype}")
        
        if missing_fields:
            print(f"❌ 缺失字段: {missing_fields}")
        else:
            print("✅ 所有关键字段都存在")
        
        # 7. 测试基础指标计算数据准备
        print("\n7. 测试指标计算数据准备...")
        
        # 确保数据有正确的数值类型
        numeric_fields = ['open', 'high', 'low', 'close', 'volume']
        for field in numeric_fields:
            if field in df.columns:
                print(f"   {field}: min={df[field].min():.2f}, max={df[field].max():.2f}, 非空数量={df[field].notna().sum()}")
        
        # 8. 测试简单MA计算
        print("\n8. 测试简单MA计算...")
        if 'close' in df.columns and len(df) >= 5:
            ma5 = df['close'].rolling(window=5).mean()
            print(f"✅ MA5计算成功，最新值: {ma5.iloc[-1]:.2f}")
        else:
            print("❌ 数据不足，无法计算MA5")
        
        print("\n=== 数据流测试完成 ===")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_data_flow() 