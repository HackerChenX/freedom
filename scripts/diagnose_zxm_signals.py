#!/usr/bin/env python3
"""
诊断ZXM吸筹信号计算问题

检查000582、000679、000008三只股票的ZXM指标计算逻辑和数据质量
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indicators.zxm_absorb import ZxmAbsorb
from db.unified_data_manager import get_unified_data_manager
from utils.dependency_injection import get_logger

logger = get_logger(__name__)

def diagnose_stock(stock_code: str, target_date: str = '2025-05-12'):
    """
    诊断单只股票的ZXM指标计算
    
    Args:
        stock_code: 股票代码
        target_date: 目标日期
    """
    print(f"\n🔍 开始诊断 {stock_code} 的ZXM指标计算...")
    print("=" * 80)
    
    try:
        # 获取数据管理器
        data_manager = get_unified_data_manager()
        
        # 获取股票数据
        target_dt = datetime.strptime(target_date, '%Y-%m-%d')
        start_date = (target_dt - timedelta(days=150)).strftime('%Y-%m-%d')
        end_date = (target_dt + timedelta(days=30)).strftime('%Y-%m-%d')
        
        print(f"📊 获取数据范围: {start_date} 至 {end_date}")
        
        stock_data = data_manager.get_stock_data(
            code=stock_code,
            start_date=start_date,
            end_date=end_date,
            level='日线'
        )
        
        if stock_data.empty:
            print(f"❌ 未获取到股票 {stock_code} 的数据")
            return
        
        print(f"✅ 获取到 {len(stock_data)} 条数据记录")

        # 确保数据格式正确
        if 'date' in stock_data.columns:
            stock_data['date'] = pd.to_datetime(stock_data['date'])
            stock_data.set_index('date', inplace=True)

        print(f"📅 数据索引类型: {type(stock_data.index)}")
        if hasattr(stock_data.index, 'min'):
            print(f"📅 数据日期范围: {stock_data.index.min()} 至 {stock_data.index.max()}")

        # 检查数据质量
        print(f"\n📋 数据质量检查:")
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col in stock_data.columns:
                null_count = stock_data[col].isnull().sum()
                print(f"   {col}: {len(stock_data) - null_count}/{len(stock_data)} 非空值")
            else:
                print(f"   ❌ 缺少列: {col}")
        
        # 检查价格数据的合理性
        print(f"\n📈 价格数据范围:")
        print(f"   开盘价: {stock_data['open'].min():.2f} - {stock_data['open'].max():.2f}")
        print(f"   最高价: {stock_data['high'].min():.2f} - {stock_data['high'].max():.2f}")
        print(f"   最低价: {stock_data['low'].min():.2f} - {stock_data['low'].max():.2f}")
        print(f"   收盘价: {stock_data['close'].min():.2f} - {stock_data['close'].max():.2f}")
        
        # 创建ZXM指标实例
        zxm_absorb = ZxmAbsorb()
        
        # 计算ZXM指标
        print(f"\n🧮 计算ZXM指标...")
        result = zxm_absorb.calculate(stock_data)
        
        if result.empty:
            print(f"❌ ZXM指标计算失败")
            return
        
        print(f"✅ ZXM指标计算完成，返回 {len(result)} 行数据")
        
        # 检查ZXM指标列
        zxm_columns = [col for col in result.columns if 'ZXM' in col]
        print(f"\n📊 ZXM指标列: {zxm_columns}")
        
        # 分析目标日期前后的数据
        target_window = pd.date_range(
            start=target_dt - timedelta(days=15),
            end=target_dt + timedelta(days=15),
            freq='D'
        )
        
        print(f"\n🎯 目标日期 {target_date} 前后15天的ZXM指标分析:")
        print("-" * 80)
        
        for date in target_window:
            date_str = date.strftime('%Y-%m-%d')

            # 处理不同的索引类型
            if hasattr(result.index, 'strftime'):
                # 日期索引
                matching_rows = result[result.index.strftime('%Y-%m-%d') == date_str]
            else:
                # 数字索引，需要通过date列匹配
                if 'date' in result.columns:
                    matching_rows = result[result['date'].dt.strftime('%Y-%m-%d') == date_str]
                else:
                    # 跳过这个日期
                    continue
            
            if not matching_rows.empty:
                row = matching_rows.iloc[0]
                
                v11 = row.get('ZXM_V11', np.nan)
                v12 = row.get('ZXM_V12', np.nan)
                ema_v11 = row.get('ZXM_EMA_V11', np.nan)
                absorb_signal = row.get('ZXM_ABSORB_SIGNAL', 0)
                buy_signal = row.get('ZXM_BUY_SIGNAL', 0)
                combined_signal = row.get('ZXM_COMBINED_SIGNAL', 0)
                close_price = row.get('close', np.nan)
                
                # 判断是否有信号
                has_signal = absorb_signal == 1 or buy_signal == 1 or combined_signal == 1
                low_position = ema_v11 <= 15
                
                signal_desc = []
                if absorb_signal == 1:
                    signal_desc.append('吸筹')
                if buy_signal == 1:
                    signal_desc.append('买入')
                if combined_signal == 1:
                    signal_desc.append('综合')
                if low_position and not signal_desc:
                    signal_desc.append('低位')
                
                status = "🔔" if has_signal or low_position else "⚪"
                signal_text = "+".join(signal_desc) if signal_desc else "无信号"
                
                print(f"{status} {date_str}: 收盘={close_price:.2f}, V11={v11:.2f}, EMA_V11={ema_v11:.2f}, V12={v12:.2f}, 信号={signal_text}")
        
        # 检查计算中间步骤
        print(f"\n🔬 计算中间步骤检查:")
        
        # 检查LLV和HHV计算
        high = stock_data['high']
        low = stock_data['low']
        close = stock_data['close']
        
        llv_55 = low.rolling(window=55).min()
        hhv_55 = high.rolling(window=55).max()
        
        print(f"   LLV(55)范围: {llv_55.min():.2f} - {llv_55.max():.2f}")
        print(f"   HHV(55)范围: {hhv_55.min():.2f} - {hhv_55.max():.2f}")
        
        # 检查RSV计算
        rsv = (close - llv_55) / (hhv_55 - llv_55) * 100
        rsv_valid = rsv.dropna()
        
        if len(rsv_valid) > 0:
            print(f"   RSV范围: {rsv_valid.min():.2f} - {rsv_valid.max():.2f}")
            print(f"   RSV有效数据: {len(rsv_valid)}/{len(rsv)} 条")
        else:
            print(f"   ❌ RSV计算失败，可能是HHV=LLV导致除零错误")
        
        # 检查目标日期的具体计算
        target_date_str = target_dt.strftime('%Y-%m-%d')

        # 处理不同的索引类型
        if hasattr(result.index, 'strftime'):
            # 日期索引
            target_rows = result[result.index.strftime('%Y-%m-%d') == target_date_str]
        else:
            # 数字索引，需要通过date列匹配
            if 'date' in result.columns:
                target_rows = result[result['date'].dt.strftime('%Y-%m-%d') == target_date_str]
            else:
                target_rows = pd.DataFrame()
        
        if not target_rows.empty:
            target_row = target_rows.iloc[0]
            print(f"\n🎯 目标日期 {target_date_str} 详细分析:")
            print(f"   收盘价: {target_row.get('close', 'N/A')}")
            print(f"   ZXM_V11: {target_row.get('ZXM_V11', 'N/A')}")
            print(f"   ZXM_V12: {target_row.get('ZXM_V12', 'N/A')}")
            print(f"   ZXM_EMA_V11: {target_row.get('ZXM_EMA_V11', 'N/A')}")
            print(f"   吸筹信号: {target_row.get('ZXM_ABSORB_SIGNAL', 'N/A')}")
            print(f"   买入信号: {target_row.get('ZXM_BUY_SIGNAL', 'N/A')}")
            print(f"   综合信号: {target_row.get('ZXM_COMBINED_SIGNAL', 'N/A')}")
            
            # 判断是否应该有信号
            ema_v11 = target_row.get('ZXM_EMA_V11', 100)
            if ema_v11 <= 13:
                print(f"   ⚠️ EMA_V11={ema_v11:.2f} ≤ 13，应该有吸筹信号")
            elif ema_v11 <= 15:
                print(f"   💡 EMA_V11={ema_v11:.2f} ≤ 15，处于低位吸筹区域")
            else:
                print(f"   ✅ EMA_V11={ema_v11:.2f} > 15，无吸筹信号正常")
        else:
            print(f"\n❌ 目标日期 {target_date_str} 无数据")
        
    except Exception as e:
        print(f"❌ 诊断过程发生异常: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    print("🚨 ZXM吸筹信号计算诊断系统")
    print("🎯 检查000582、000679、000008三只股票的计算问题")
    print("=" * 80)
    
    # 需要诊断的股票
    problem_stocks = ['000582', '000679', '000008']
    target_date = '2025-05-12'
    
    for stock_code in problem_stocks:
        diagnose_stock(stock_code, target_date)
        print("\n" + "=" * 80)
    
    print("\n🎯 诊断总结:")
    print("1. 检查数据质量：是否有缺失或异常数据")
    print("2. 检查计算逻辑：V11、V12、EMA_V11计算是否正确")
    print("3. 检查信号条件：EMA_V11 ≤ 13的吸筹条件是否满足")
    print("4. 检查时间窗口：目标日期前后是否有有效数据")

if __name__ == "__main__":
    main()
