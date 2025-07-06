#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
诊断脚本：分析股票603359在2025-05-12的ZXM选股策略条件
"""

import sys
import os
import pandas as pd
from datetime import datetime, timedelta

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_dir)

from db.unified_data_manager import get_unified_data_manager
from db.query_executor import get_query_executor
from db.sql_manager import QueryType
from indicators.complete_indicator_registry import complete_registry
from strategy.strategy_condition_evaluator import Strategy_condition_evaluator
from strategy.strategy_parser import Strategy_parser
from utils.logger import get_logger, init_logging

# 初始化日志
init_logging(level="INFO")
logger = get_logger(__name__)

def analyze_stock_603359():
    """分析股票603359的详细情况"""
    stock_code = "603359"
    target_date = "2025-05-12"
    
    print(f"=== 诊断股票 {stock_code} 在 {target_date} 的ZXM选股策略条件 ===\n")
    
    try:
        # 1. 获取数据管理器和查询执行器
        data_manager = get_unified_data_manager()
        query_executor = get_query_executor()

        # 2. 检查基础数据
        print("1. 检查基础数据存在性")
        print("-" * 50)

        # 查询该股票的基本信息 - 使用统一查询接口
        try:
            stock_info_df = query_executor.execute_query(
                QueryType.STOCK_INFO,
                {'code': stock_code, 'level': '日线'}
            )
            
            if stock_info_df.empty:
                print(f"❌ 股票 {stock_code} 在数据库中不存在")
                return
            else:
                print(f"✅ 股票基本信息: {stock_info_df.iloc[0]['name']} ({stock_code})")
                if 'industry' in stock_info_df.columns:
                    print(f"   行业: {stock_info_df.iloc[0]['industry']}")
        except Exception as e:
            logger.error(f"查询股票基本信息失败: {e}")
            return
        
        # 3. 检查目标日期及前后的数据
        print(f"\n2. 检查 {target_date} 前后的数据")
        print("-" * 50)

        # 查询前后5天的数据
        start_date = (datetime.strptime(target_date, '%Y-%m-%d') - timedelta(days=10)).strftime('%Y-%m-%d')
        end_date = (datetime.strptime(target_date, '%Y-%m-%d') + timedelta(days=5)).strftime('%Y-%m-%d')

        # 使用统一查询接口检查数据存在性
        print("2.1 使用统一查询接口检查数据")
        try:
            # 查询日线数据
            daily_df = query_executor.execute_query(
                QueryType.STOCK_DATA,
                {
                    'code': stock_code,
                    'start_date': start_date,
                    'end_date': end_date,
                    'level': '日线'
                }
            )

            if daily_df.empty:
                print(f"❌ 数据库中没有 {stock_code} 在 {start_date} 到 {end_date} 的日线数据")
            else:
                print(f"✅ 数据库日线数据: {len(daily_df)} 条记录")
                print(f"   日期范围: {daily_df['date'].min()} 到 {daily_df['date'].max()}")
                print("   前5条记录:")
                for _, row in daily_df.head().iterrows():
                    print(f"     {row['date']}: 开盘={row['open']}, 收盘={row['close']}, 成交量={row['volume']}")

                # 检查目标日期
                target_daily_db = daily_df[daily_df['date'] == target_date]
                if not target_daily_db.empty:
                    row = target_daily_db.iloc[0]
                    print(f"   ✅ {target_date} 日线数据: 开盘={row['open']}, 收盘={row['close']}, 成交量={row['volume']}")
                else:
                    print(f"   ❌ 数据库中没有 {target_date} 的日线数据")

            # 查询30分钟数据
            min30_df = query_executor.execute_query(
                QueryType.STOCK_DATA,
                {
                    'code': stock_code,
                    'start_date': start_date,
                    'end_date': end_date,
                    'level': '30分钟'
                }
            )

            if min30_df.empty:
                print(f"❌ 数据库中没有 {stock_code} 在 {start_date} 到 {end_date} 的30分钟数据")
            else:
                print(f"✅ 数据库30分钟数据: {len(min30_df)} 条记录")
                print(f"   日期范围: {min30_df['date'].min()} 到 {min30_df['date'].max()}")

                # 检查目标日期的30分钟数据
                target_30min_db = min30_df[min30_df['date'] == target_date]
                print(f"   {target_date} 的30分钟数据: {len(target_30min_db)} 条记录")
                if not target_30min_db.empty:
                    print("   前3条30分钟记录:")
                    for _, row in target_30min_db.head(3).iterrows():
                        datetime_col = 'datetime' if 'datetime' in row else 'date'
                        print(f"     {row[datetime_col]}: 开盘={row['open']}, 收盘={row['close']}, 成交量={row['volume']}")

        except Exception as e:
            logger.error(f"使用统一查询接口检查数据失败: {e}")

        # 使用数据管理器获取数据
        print("\n2.2 使用数据管理器获取数据")
        # 日线数据
        daily_data = data_manager.get_stock_data(
            stock_code=stock_code,
            start_date=start_date,
            end_date=end_date,
            period='daily',
            lookback_days=90  # 获取足够的历史数据
        )

        if daily_data.empty:
            print(f"❌ 数据管理器没有返回 {stock_code} 的日线数据")
        else:
            print(f"✅ 数据管理器日线数据: {len(daily_data)} 条记录")
            print(f"   列名: {list(daily_data.columns)}")

            # 检查目标日期的数据
            if 'date' in daily_data.columns:
                target_daily = daily_data[daily_data['date'] == target_date]
                if not target_daily.empty:
                    row = target_daily.iloc[0]
                    print(f"   ✅ {target_date} 数据: 开盘={row.get('open', 'N/A')}, 收盘={row.get('close', 'N/A')}, 成交量={row.get('volume', 'N/A')}")
                else:
                    print(f"   ❌ 数据管理器中没有 {target_date} 的日线数据")

        # 30分钟数据
        print("\n2.3 测试30分钟数据获取和计算")
        min30_data = data_manager.get_stock_data(
            stock_code=stock_code,
            start_date=start_date,
            end_date=end_date,
            period='30min',
            lookback_days=90  # 获取足够的历史数据
        )

        if min30_data.empty:
            print(f"❌ 数据管理器没有返回 {stock_code} 的30分钟数据")
        else:
            print(f"✅ 数据管理器30分钟数据: {len(min30_data)} 条记录")
            print(f"   列名: {list(min30_data.columns)}")

            # 检查目标日期的30分钟数据
            if 'date' in min30_data.columns:
                target_30min = min30_data[min30_data['date'] == target_date]
                print(f"   {target_date} 的30分钟数据: {len(target_30min)} 条记录")
        
        # 4. 测试ZXM指标计算
        print(f"\n3. 测试ZXM指标计算")
        print("-" * 50)
        
        # 测试ZXM_BS_ABSORB指标
        print("3.1 测试ZXM_BS_ABSORB指标（30分钟）")
        try:
            zxm_absorb = complete_registry.create_indicator('ZXM_BS_ABSORB')
            if zxm_absorb and not min30_data.empty:
                absorb_result = zxm_absorb.calculate(min30_data)
                print(f"   ZXM_BS_ABSORB计算成功，结果长度: {len(absorb_result)}")
                
                if not absorb_result.empty:
                    # 查看最近几天的结果
                    recent_absorb = absorb_result.tail(10)
                    print("   最近10个交易时段的ZXM_BS_ABSORB结果:")
                    for idx, row in recent_absorb.iterrows():
                        print(f"     {row.get('date', idx)}: {row.get('signal', 'N/A')}")
                    
                    # 检查目标日期的信号
                    target_absorb = absorb_result[absorb_result.get('date', pd.Series()) == target_date]
                    if not target_absorb.empty:
                        signals = target_absorb['signal'].tolist() if 'signal' in target_absorb.columns else []
                        print(f"   {target_date} 的ZXM_BS_ABSORB信号: {signals}")
                        buy_signals = [s for s in signals if s == 'BUY']
                        print(f"   BUY信号数量: {len(buy_signals)}")
                    else:
                        print(f"   ❌ {target_date} 没有ZXM_BS_ABSORB信号")
                else:
                    print("   ❌ ZXM_BS_ABSORB计算结果为空")
            else:
                print("   ❌ ZXM_BS_ABSORB指标创建失败或30分钟数据为空")
        except Exception as e:
            logger.error(f"ZXM_BS_ABSORB指标计算失败: {e}")
        
        # 测试ZXM_VOLUME_SHRINK指标
        print("\n3.2 测试ZXM_VOLUME_SHRINK指标（日线）")
        try:
            zxm_volume = complete_registry.create_indicator('ZXM_VOLUME_SHRINK')
            if zxm_volume and not daily_data.empty:
                volume_result = zxm_volume.calculate(daily_data)
                print(f"   ZXM_VOLUME_SHRINK计算成功，结果长度: {len(volume_result)}")
                
                if not volume_result.empty:
                    # 查看最近几天的结果
                    recent_volume = volume_result.tail(10)
                    print("   最近10个交易日的ZXM_VOLUME_SHRINK结果:")
                    for idx, row in recent_volume.iterrows():
                        print(f"     {row.get('date', idx)}: {row.get('signal', 'N/A')}")
                    
                    # 检查目标日期的信号
                    target_volume = volume_result[volume_result.get('date', pd.Series()) == target_date]
                    if not target_volume.empty:
                        signals = target_volume['signal'].tolist() if 'signal' in target_volume.columns else []
                        print(f"   {target_date} 的ZXM_VOLUME_SHRINK信号: {signals}")
                        buy_signals = [s for s in signals if s == 'BUY']
                        print(f"   BUY信号数量: {len(buy_signals)}")
                    else:
                        print(f"   ❌ {target_date} 没有ZXM_VOLUME_SHRINK信号")
                else:
                    print("   ❌ ZXM_VOLUME_SHRINK计算结果为空")
            else:
                print("   ❌ ZXM_VOLUME_SHRINK指标创建失败或日线数据为空")
        except Exception as e:
            print(f"   ❌ ZXM_VOLUME_SHRINK计算出错: {e}")
        
        # 5. 测试策略条件评估
        print(f"\n4. 测试策略条件评估")
        print("-" * 50)
        
        # 加载策略配置
        parser = Strategy_parser()
        strategy_plan = parser.parse_from_file('config/strategies/zxm_absorb_volume_shrink_strategy.yaml')
        
        # 创建条件评估器
        evaluator = Strategy_condition_evaluator()
        
        # 准备股票数据
        stock_data = {
            'code': stock_code,
            'name': stock_info_df.iloc[0]['name'] if not stock_info_df.empty else stock_code,
            'daily_data': daily_data,
            '30min_data': min30_data
        }
        
        print("4.1 评估ZXM_BS_ABSORB条件")
        absorb_condition = {
            'type': 'indicator',
            'indicator_id': 'ZXM_BS_ABSORB',
            'period': '30min',
            'parameters': {},
            'signal_type': 'BUY',
            'required': True,
            'weight': 1.0
        }
        
        try:
            absorb_passed = evaluator.evaluate_condition(absorb_condition, stock_data, target_date)
            print(f"   ZXM_BS_ABSORB条件评估结果: {'✅ 通过' if absorb_passed else '❌ 未通过'}")
        except Exception as e:
            print(f"   ❌ ZXM_BS_ABSORB条件评估出错: {e}")
        
        print("\n4.2 评估ZXM_VOLUME_SHRINK条件")
        volume_condition = {
            'type': 'indicator',
            'indicator_id': 'ZXM_VOLUME_SHRINK',
            'period': 'daily',
            'parameters': {},
            'signal_type': 'BUY',
            'required': True,
            'weight': 1.0
        }
        
        try:
            volume_passed = evaluator.evaluate_condition(volume_condition, stock_data, target_date)
            print(f"   ZXM_VOLUME_SHRINK条件评估结果: {'✅ 通过' if volume_passed else '❌ 未通过'}")
        except Exception as e:
            print(f"   ❌ ZXM_VOLUME_SHRINK条件评估出错: {e}")
        
        print(f"\n=== 诊断完成 ===")
        
    except Exception as e:
        logger.error(f"分析过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_stock_603359()
