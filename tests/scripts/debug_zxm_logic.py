#!/usr/bin/env python3
"""
ZXM指标逻辑调试脚本

深入检查ZXM指标计算逻辑，找出为什么没有选出股票的原因

作者：AI Assistant  
创建时间：2025-01-13
"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# 导入系统模块
from indicators.zxm.buy_point_indicators import ZXMBSAbsorb
from db.unified_data_manager import get_unified_data_manager
from utils.logger import get_logger

logger = get_logger("debug_zxm_logic")

class ZXMLogicDebugger:
    """ZXM指标逻辑调试器"""
    
    def __init__(self):
        self.data_manager = get_unified_data_manager()
        self.indicator = ZXMBSAbsorb()
        
    def debug_stock(self, stock_code: str, target_date: str = "2025-05-12"):
        """
        调试单只股票的ZXM指标计算过程
        
        Args:
            stock_code: 股票代码
            target_date: 目标日期
        """
        print(f"\n{'='*60}")
        print(f"🔍 调试股票: {stock_code} | 目标日期: {target_date}")
        print(f"{'='*60}")
        
        try:
            # 获取足够的历史数据（至少需要55+15=70个交易日的数据）
            start_date = "2025-03-01"
            end_date = target_date
            
            print(f"📊 获取数据: {start_date} 到 {end_date}")
            
            # 获取60分钟数据
            data_60min = self.data_manager.get_period_data(
                stock_code, start_date, end_date, "60min"
            )
            
            if data_60min.empty:
                print(f"❌ 无法获取{stock_code}的60分钟数据")
                return None
                
            print(f"✅ 获取到 {len(data_60min)} 条60分钟数据")
            print(f"📈 数据时间范围: {data_60min['date'].min()} 到 {data_60min['date'].max()}")
            
            # 检查数据质量
            print(f"\n📋 数据质量检查:")
            print(f"   - 开盘价范围: {data_60min['open'].min():.2f} ~ {data_60min['open'].max():.2f}")
            print(f"   - 最高价范围: {data_60min['high'].min():.2f} ~ {data_60min['high'].max():.2f}")
            print(f"   - 最低价范围: {data_60min['low'].min():.2f} ~ {data_60min['low'].max():.2f}")
            print(f"   - 收盘价范围: {data_60min['close'].min():.2f} ~ {data_60min['close'].max():.2f}")
            print(f"   - 成交量范围: {data_60min['volume'].min():,.0f} ~ {data_60min['volume'].max():,.0f}")
            
            # 检查是否有足够的数据计算指标
            if len(data_60min) < 70:
                print(f"⚠️  数据量不足：需要至少70条数据，实际{len(data_60min)}条")
                return None
                
            # 执行ZXM指标计算
            print(f"\n🧮 执行ZXM指标计算...")
            try:
                result = self.indicator._calculate(data_60min)
                print(f"✅ ZXM指标计算成功，结果包含 {len(result)} 行数据")
                
                # 分析计算结果
                self._analyze_zxm_result(result, target_date)
                
                return result
                
            except Exception as calc_error:
                print(f"❌ ZXM指标计算失败: {calc_error}")
                import traceback
                traceback.print_exc()
                return None
                
        except Exception as e:
            print(f"❌ 调试失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _analyze_zxm_result(self, result: pd.DataFrame, target_date: str):
        """分析ZXM计算结果"""
        print(f"\n📊 ZXM指标计算结果分析:")
        
        # 检查目标日期的数据
        target_data = result[result['date'] == target_date]
        if target_data.empty:
            print(f"❌ 目标日期 {target_date} 无数据")
            # 找最接近的日期
            available_dates = result['date'].unique()
            print(f"   可用日期: {sorted(available_dates)[-5:]}")  # 显示最后5个日期
            
            # 使用最后一个交易日
            if len(available_dates) > 0:
                latest_date = max(available_dates)
                target_data = result[result['date'] == latest_date]
                print(f"📅 使用最新日期: {latest_date}")
        
        if not target_data.empty:
            row = target_data.iloc[0]
            
            print(f"\n🎯 目标日期数据详情:")
            print(f"   - 日期: {row['date']}")
            print(f"   - 收盘价: {row['close']:.2f}")
            print(f"   - V11: {row['V11']:.2f}")
            print(f"   - EMA_V11_3: {row['EMA_V11_3']:.2f}")
            print(f"   - V12: {row['V12']:.2f}")
            print(f"   - AA条件: {row['AA']}")
            print(f"   - BB条件: {row['BB']}")
            print(f"   - XG值: {row['XG']}")
            print(f"   - 买入信号: {row['buy_signal']}")
            
            # 检查信号生成逻辑
            print(f"\n🚦 信号生成逻辑检查:")
            print(f"   - XG > 0: {row['XG'] > 0}")
            print(f"   - 应生成买入信号: {row['XG'] > 0}")
            print(f"   - 实际买入信号: {row['buy_signal']}")
            
            if row['XG'] > 0:
                print(f"✅ 该股票满足ZXM买入条件！")
            else:
                print(f"❌ 该股票不满足ZXM买入条件")
                
                # 分析为什么不满足条件
                self._analyze_condition_failure(result, target_data.index[0])
        
        # 统计整体结果
        print(f"\n📈 整体统计:")
        buy_signals = result['buy_signal'].sum()
        xg_positive = (result['XG'] > 0).sum()
        aa_signals = result['AA'].sum()
        bb_signals = result['BB'].sum()
        
        print(f"   - 总数据行数: {len(result)}")
        print(f"   - 买入信号总数: {buy_signals}")
        print(f"   - XG>0的数量: {xg_positive}")
        print(f"   - AA条件满足次数: {aa_signals}")
        print(f"   - BB条件满足次数: {bb_signals}")
        
        if buy_signals > 0:
            print(f"✅ 该股票在历史中有 {buy_signals} 次买入信号")
            # 显示最近的买入信号
            recent_signals = result[result['buy_signal'] == True].tail(3)
            if not recent_signals.empty:
                print(f"📅 最近的买入信号日期:")
                for _, signal_row in recent_signals.iterrows():
                    print(f"     - {signal_row['date']}: XG={signal_row['XG']}")
        else:
            print(f"❌ 该股票在整个时间段内无买入信号")
    
    def _analyze_condition_failure(self, result: pd.DataFrame, target_idx: int):
        """分析条件不满足的原因"""
        print(f"\n🔍 条件不满足原因分析:")
        
        if target_idx >= len(result):
            print(f"❌ 索引超出范围")
            return
            
        # 查看前面的几行数据，分析条件演变
        start_idx = max(0, target_idx - 10)
        end_idx = min(len(result), target_idx + 1)
        
        recent_data = result.iloc[start_idx:end_idx]
        
        print(f"📊 最近10天的条件演变:")
        print(f"{'日期':>12} {'EMA_V11_3':>10} {'V12':>8} {'AA':>5} {'BB':>5} {'XG':>5}")
        print(f"{'-'*50}")
        
        for _, row in recent_data.iterrows():
            print(f"{str(row['date']):>12} {row['EMA_V11_3']:>10.2f} {row['V12']:>8.2f} "
                  f"{str(row['AA']):>5} {str(row['BB']):>5} {row['XG']:>5}")
        
        # 分析具体原因
        target_row = result.iloc[target_idx]
        print(f"\n💡 分析结果:")
        
        if target_row['EMA_V11_3'] > 13:
            print(f"   - EMA_V11_3 = {target_row['EMA_V11_3']:.2f} > 13，不在低位")
        else:
            print(f"   - EMA_V11_3 = {target_row['EMA_V11_3']:.2f} <= 13，处于低位 ✓")
            
        if target_row['V12'] <= 13:
            print(f"   - V12 = {target_row['V12']:.2f} <= 13，动量不足")
        else:
            print(f"   - V12 = {target_row['V12']:.2f} > 13，动量充足 ✓")
        
        if not target_row['AA'] and not target_row['BB']:
            print(f"   - AA和BB条件都不满足")
        elif target_row['AA']:
            print(f"   - AA条件满足 ✓")
        elif target_row['BB']:
            print(f"   - BB条件满足 ✓")
            
        print(f"   - XG = {target_row['XG']}，需要 > 0")

    def batch_debug_sample_stocks(self, target_date: str = "2025-05-12", sample_size: int = 5):
        """
        批量调试采样股票
        
        Args:
            target_date: 目标日期
            sample_size: 采样数量
        """
        print(f"\n🎯 批量调试采样股票 (样本数: {sample_size})")
        
        # 获取所有股票代码
        try:
            # 先获取所有股票代码
            query = """
            SELECT DISTINCT code 
            FROM stock_info 
            WHERE date = '2025-05-12' 
            AND level = '日线'
            ORDER BY code
            LIMIT 100
            """
            
            from db.clickhouse_db import get_clickhouse_db
            db = get_clickhouse_db()
            stock_codes_df = db.query(query)
            
            if stock_codes_df.empty:
                print(f"❌ 无法获取股票代码列表")
                return
                
            stock_codes = stock_codes_df['code'].tolist()
            print(f"📋 获取到 {len(stock_codes)} 只股票代码")
            
            # 随机采样
            import random
            sample_codes = random.sample(stock_codes, min(sample_size, len(stock_codes)))
            
            results = {}
            success_count = 0
            signal_count = 0
            
            for i, stock_code in enumerate(sample_codes):
                print(f"\n[{i+1}/{len(sample_codes)}] 处理股票: {stock_code}")
                
                try:
                    result = self.debug_stock(stock_code, target_date)
                    if result is not None:
                        success_count += 1
                        
                        # 检查是否有信号
                        target_data = result[result['date'] == target_date]
                        if not target_data.empty and target_data.iloc[0]['buy_signal']:
                            signal_count += 1
                            print(f"🎉 {stock_code} 有买入信号！")
                        
                        results[stock_code] = result
                        
                except Exception as e:
                    print(f"❌ {stock_code} 处理失败: {e}")
            
            print(f"\n📊 批量调试总结:")
            print(f"   - 成功处理: {success_count}/{len(sample_codes)}")
            print(f"   - 有买入信号: {signal_count}/{success_count}")
            print(f"   - 信号比例: {signal_count/success_count*100:.1f}%" if success_count > 0 else "   - 信号比例: 0%")
            
            return results
            
        except Exception as e:
            print(f"❌ 批量调试失败: {e}")
            import traceback
            traceback.print_exc()
            return {}

def main():
    """主函数"""
    print("🚀 启动ZXM指标逻辑调试器")
    
    debugger = ZXMLogicDebugger()
    
    # 首先测试几只具体的股票
    test_stocks = ["000001", "000002", "600000", "600036", "000858"]
    
    print(f"\n🎯 测试具体股票:")
    for stock_code in test_stocks:
        try:
            result = debugger.debug_stock(stock_code)
            if result is not None:
                print(f"✅ {stock_code} 调试完成")
            else:
                print(f"❌ {stock_code} 调试失败")
        except Exception as e:
            print(f"❌ {stock_code} 出错: {e}")
    
    # 然后进行批量采样测试
    print(f"\n🔍 进行批量采样测试:")
    batch_results = debugger.batch_debug_sample_stocks(sample_size=10)
    
    print(f"\n✅ ZXM指标逻辑调试完成!")

if __name__ == "__main__":
    main() 