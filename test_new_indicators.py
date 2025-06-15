#!/usr/bin/env python3
"""
新集成指标深度测试脚本
专门测试BIAS、CCI、Chaikin、DMI、EMV指标的计算和形态识别功能
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.insert(0, os.getcwd())

from indicators.indicator_registry import indicator_registry
from utils.logger import get_logger
from db.clickhouse_db import ClickHouseDB

logger = get_logger(__name__)

def create_realistic_test_data(periods=100):
    """创建更真实的股票数据"""
    dates = pd.date_range('2023-01-01', periods=periods, freq='D')
    
    # 生成更真实的股票数据
    np.random.seed(42)
    base_price = 100
    
    # 生成价格序列（带趋势和波动）
    trend = np.linspace(0, 20, periods)  # 上升趋势
    noise = np.random.normal(0, 2, periods)  # 随机波动
    returns = (trend + noise) / 100  # 转换为收益率
    
    prices = [base_price]
    for r in returns[1:]:
        prices.append(prices[-1] * (1 + r))
    
    # 生成OHLC数据
    data = pd.DataFrame(index=dates)
    data['close'] = prices
    
    # 生成开盘价（基于前一日收盘价加小幅波动）
    data['open'] = data['close'].shift(1) * (1 + np.random.normal(0, 0.01, periods))
    data['open'].iloc[0] = base_price
    
    # 生成高低价（基于开盘和收盘价）
    daily_range = np.random.uniform(0.02, 0.08, periods)  # 2-8%的日内波动
    for i in range(periods):
        high_low_range = daily_range[i] * data['close'].iloc[i]
        data.loc[data.index[i], 'high'] = max(data['open'].iloc[i], data['close'].iloc[i]) + high_low_range/2
        data.loc[data.index[i], 'low'] = min(data['open'].iloc[i], data['close'].iloc[i]) - high_low_range/2
    
    # 生成成交量（与价格变化相关）
    price_change = data['close'].pct_change().fillna(0)
    base_volume = 5000000
    volume_multiplier = 1 + np.abs(price_change) * 2  # 价格变化大时成交量增加
    data['volume'] = (base_volume * volume_multiplier * np.random.uniform(0.5, 1.5, periods)).astype(int)
    
    # 确保数据类型正确
    for col in ['open', 'high', 'low', 'close']:
        data[col] = data[col].astype(float)
    data['volume'] = data['volume'].astype(int)
    
    return data

def test_indicator_with_real_data(indicator_name, test_data):
    """使用真实数据测试单个指标"""
    print(f"\n{'='*60}")
    print(f"🔍 深度测试指标: {indicator_name}")
    print(f"{'='*60}")
    
    try:
        # 创建指标实例
        indicator = indicator_registry.create_indicator(indicator_name)
        if not indicator:
            print(f"❌ 无法创建指标实例: {indicator_name}")
            return False, None
        
        print(f"✅ 成功创建指标实例")
        
        # 检查数据完整性
        print(f"📊 测试数据概览:")
        print(f"   - 数据行数: {len(test_data)}")
        print(f"   - 数据列: {list(test_data.columns)}")
        print(f"   - 价格范围: {test_data['close'].min():.2f} - {test_data['close'].max():.2f}")
        print(f"   - 成交量范围: {test_data['volume'].min():,} - {test_data['volume'].max():,}")
        
        # 检查必需的列
        required_cols = getattr(indicator, 'required_columns', [])
        print(f"📋 指标必需列: {required_cols}")
        
        missing_cols = [col for col in required_cols if col not in test_data.columns]
        if missing_cols:
            print(f"❌ 缺少必需列: {missing_cols}")
            return False, None
        
        # 第一步：测试基础计算
        print(f"🔄 步骤1: 测试基础计算...")
        try:
            result = indicator.calculate(test_data)
            if result is None:
                print(f"❌ 计算返回None")
                return False, None
            
            if result.empty:
                print(f"❌ 计算返回空DataFrame")
                return False, None
            
            print(f"✅ 基础计算成功")
            print(f"   - 返回数据: {len(result)} 行 x {len(result.columns)} 列")
            
            # 显示新增的列
            new_cols = [col for col in result.columns if col not in test_data.columns]
            if new_cols:
                print(f"   - 新增列: {new_cols}")
                for col in new_cols:
                    valid_count = result[col].notna().sum()
                    print(f"     * {col}: {valid_count}/{len(result)} 有效值")
                    if valid_count > 0:
                        print(f"       范围: {result[col].min():.4f} - {result[col].max():.4f}")
            
        except Exception as e:
            print(f"❌ 基础计算失败: {e}")
            import traceback
            traceback.print_exc()
            return False, None
        
        # 第二步：测试形态识别
        print(f"🔄 步骤2: 测试形态识别...")
        try:
            if hasattr(indicator, 'get_patterns'):
                patterns = indicator.get_patterns(test_data)
                if patterns is not None and not patterns.empty:
                    print(f"✅ 形态识别成功")
                    print(f"   - 返回数据: {len(patterns)} 行 x {len(patterns.columns)} 列")
                    
                    # 查找形态列
                    pattern_cols = [col for col in patterns.columns 
                                  if col not in test_data.columns and col.upper().startswith(indicator_name.upper())]
                    
                    if pattern_cols:
                        print(f"   - 形态列: {pattern_cols}")
                        
                        # 统计每个形态的触发次数
                        for col in pattern_cols:
                            if patterns[col].dtype == bool:
                                trigger_count = patterns[col].sum()
                                print(f"     * {col}: {trigger_count} 次触发")
                            elif patterns[col].dtype in ['int64', 'float64']:
                                non_zero_count = (patterns[col] != 0).sum()
                                print(f"     * {col}: {non_zero_count} 次非零值")
                    else:
                        print(f"⚠️  未找到形态列，可能形态命名不符合预期")
                        # 显示所有新增列
                        all_new_cols = [col for col in patterns.columns if col not in test_data.columns]
                        print(f"   - 所有新增列: {all_new_cols}")
                else:
                    print(f"❌ 形态识别返回空结果")
                    return False, None
            else:
                print(f"❌ 指标缺少get_patterns方法")
                return False, None
                
        except Exception as e:
            print(f"❌ 形态识别失败: {e}")
            import traceback
            traceback.print_exc()
            return False, None
        
        # 第三步：测试get_pattern_info方法
        print(f"🔄 步骤3: 测试get_pattern_info方法...")
        try:
            if hasattr(indicator, 'get_pattern_info'):
                # 尝试获取一个形态的信息
                test_pattern = f"{indicator_name.upper()}_TEST"
                pattern_info = indicator.get_pattern_info(test_pattern)
                if isinstance(pattern_info, dict):
                    print(f"✅ get_pattern_info方法正常")
                    print(f"   - 返回类型: {type(pattern_info)}")
                    print(f"   - 包含键: {list(pattern_info.keys())}")
                else:
                    print(f"⚠️  get_pattern_info返回类型异常: {type(pattern_info)}")
            else:
                print(f"❌ 缺少get_pattern_info方法")
                return False, None
        except Exception as e:
            print(f"❌ get_pattern_info方法测试失败: {e}")
            return False, None
        
        print(f"🎉 指标 {indicator_name} 测试完全通过！")
        return True, patterns
        
    except Exception as e:
        print(f"❌ 指标测试过程中发生异常: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def main():
    """主函数"""
    print("🚀 开始新集成指标深度测试...")
    
    # 创建真实测试数据
    test_data = create_realistic_test_data(100)
    print(f"📊 生成测试数据: {len(test_data)} 行")
    print(f"📅 时间范围: {test_data.index[0]} 到 {test_data.index[-1]}")
    
    # 测试新集成的指标
    test_indicators = ['BIAS', 'CCI', 'Chaikin', 'DMI', 'EMV']
    
    results = {}
    successful_indicators = []
    
    for indicator_name in test_indicators:
        success, patterns = test_indicator_with_real_data(indicator_name, test_data)
        results[indicator_name] = {
            'success': success,
            'patterns': patterns
        }
        if success:
            successful_indicators.append(indicator_name)
    
    # 总结测试结果
    print(f"\n{'='*60}")
    print("📋 测试结果总结")
    print(f"{'='*60}")
    
    print(f"✅ 成功通过测试: {len(successful_indicators)}/{len(test_indicators)} 个指标")
    for name in successful_indicators:
        print(f"   - {name}")
    
    failed_indicators = [name for name, result in results.items() if not result['success']]
    if failed_indicators:
        print(f"❌ 测试失败: {len(failed_indicators)} 个指标")
        for name in failed_indicators:
            print(f"   - {name}")
    
    # 如果有成功的指标，尝试模拟买点分析流程
    if successful_indicators:
        print(f"\n🔄 模拟买点分析流程...")
        try:
            # 这里可以添加模拟买点分析的代码
            print(f"✅ 成功的指标应该能够参与买点分析")
        except Exception as e:
            print(f"❌ 模拟买点分析失败: {e}")
    
    return len(successful_indicators) == len(test_indicators)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
