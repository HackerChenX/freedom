from db.query_executor import get_query_executor
from db.sql_manager import QueryType
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
新实现指标使用示例

演示如何使用新实现的技术指标进行选股
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from indicators.complete_indicator_registry import complete_registry
from utils.dependency_injection import get_service
from utils.logger import get_logger

logger = get_logger(__name__)


def get_stock_data_Indicators(stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    从ClickHouse获取股票数据
    
    Args:
        stock_code: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        
    Returns:
        pd.DataFrame: 股票数据
    """
    try:
        container = get_container()
        db = container.get_data_access()
        
        query = f"""
        SELECT 
            trade_date as date,
            open,
            high,
            low,
            close,
            volume,
            amount,
            turnover_rate
        FROM stock_daily_prices
        WHERE stock_code = '{stock_code}'
          AND trade_date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY trade_date
        """
        
        data = db.query_dataframe(query)
        
        # 验证数据有效性
        if data.empty:
            logger.warning(f"未找到股票 {stock_code} 在 {start_date} 至 {end_date} 的数据")
            return pd.DataFrame()
        
        # 转换日期为索引并确保是正确的日期格式
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)
        
        # 确保数据是按日期排序的
        data.sort_index(inplace=True)
        
        # 检查数据中是否存在缺失值
        if data.isnull().any().any():
            logger.warning(f"数据中存在缺失值，将使用前值填充")
            data.fillna(method='ffill', inplace=True)
        
        return data
        
    except Exception as e:
        logger.error(f"获取股票数据时发生错误: {e}")
        return pd.DataFrame()


def mainUsenewindicators():
    """主函数"""
    # 设置测试参数
    stock_code = "000001.SZ"  # 平安银行
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    try:
        print(f"获取股票 {stock_code} 从 {start_date} 到 {end_date} 的数据...")
        data = get_stock_data_Indicators(stock_code, start_date, end_date)
        
        if data.empty:
            print("未找到股票数据，请检查股票代码或日期范围！")
            return
        
        print(f"获取到 {len(data)} 条数据记录")
        
        # 检查数据量是否足够进行分析
        if len(data) < 60:
            print("数据量不足以进行可靠的技术分析，需要至少60个交易日数据")
            return
        
        # 执行指标测试，增加错误处理
        try:
            print("\n1. 测试量比指标 (VolumeRatio)")
            volume_ratio = complete_registry.create_indicator('VOLUME_RATIO', reference_period=5)
            if volume_ratio:
                vr_result = volume_ratio.calculate(data)
                print(f"量比计算结果示例: \n{vr_result[['volume_ratio']].tail(3)}")
            else:
                print("无法创建VOLUME_RATIO指标")
        except Exception as e:
            print(f"量比指标计算失败: {e}")

        try:
            print("\n2. 测试平台突破指标 (PlatformBreakout)")
            platform_breakout = complete_registry.create_indicator('PLATFORM_BREAKOUT', platform_period=20, threshold=0.03)
            if platform_breakout:
                pb_result = platform_breakout.calculate(data)
                print(f"平台突破识别结果: {pb_result['platform_breakout'].sum()} 次突破")
                if pb_result['platform_breakout'].sum() > 0:
                    breakout_dates = pb_result.index[pb_result['platform_breakout']].tolist()
                    print(f"突破日期: {breakout_dates[:3]}{'...' if len(breakout_dates) > 3 else ''}")
            else:
                print("无法创建PLATFORM_BREAKOUT指标")
        except Exception as e:
            print(f"平台突破指标计算失败: {e}")

        try:
            print("\n3. 测试能量潮指标 (OBV)")
            obv = complete_registry.create_indicator('OBV')
            if obv:
                obv_result = obv.calculate(data)
                print(f"OBV计算结果示例: \n{obv_result[['obv']].tail(3)}")
            else:
                print("无法创建OBV指标")
        except Exception as e:
            print(f"OBV指标计算失败: {e}")

        try:
            print("\n4. 测试成交量变异率 (VOSC)")
            vosc = complete_registry.create_indicator('VOSC', short_period=12, long_period=26)
            if vosc:
                vosc_result = vosc.calculate(data)
                print(f"VOSC计算结果示例: \n{vosc_result[['vosc']].tail(3)}")
            else:
                print("无法创建VOSC指标")
        except Exception as e:
            print(f"VOSC指标计算失败: {e}")

        try:
            print("\n5. 测试资金流向指标 (MFI)")
            mfi = complete_registry.create_indicator('MFI', period=14)
            if mfi:
                mfi_result = mfi.calculate(data)
                print(f"MFI计算结果示例: \n{mfi_result[['mfi']].tail(3)}")
            else:
                print("无法创建MFI指标")
        except Exception as e:
            print(f"MFI指标计算失败: {e}")

        try:
            print("\n6. 测试成交量指标 (VR)")
            vr_indicator = complete_registry.create_indicator('VR', period=26)
            if vr_indicator:
                vr_indicator_result = vr_indicator.calculate(data)
                print(f"VR计算结果示例: \n{vr_indicator_result[['vr']].tail(3)}")
            else:
                print("无法创建VR指标")
        except Exception as e:
            print(f"VR指标计算失败: {e}")

        try:
            print("\n7. 测试价量趋势指标 (PVT)")
            pvt = complete_registry.create_indicator('PVT')
            if pvt:
                pvt_result = pvt.calculate(data)
                print(f"PVT计算结果示例: \n{pvt_result[['pvt']].tail(3)}")
            else:
                print("无法创建PVT指标")
        except Exception as e:
            print(f"PVT指标计算失败: {e}")
        
        try:
            print("\n8. 测试指数平均数指标 (EMV)")
            emv = complete_registry.create_indicator('EMV', period=14, ma_period=9)
            if emv:
                emv_result = emv.calculate(data)
                print(f"EMV计算结果示例: \n{emv_result[['emv', 'emv_ma']].tail(3)}")

                # 获取EMV信号
                if hasattr(emv, 'get_signals'):
                    emv_signals = emv.get_signals(data)
                    emv_buy_signals = emv_signals.index[emv_signals['emv_combined_signal'] == 1].tolist()
                    emv_sell_signals = emv_signals.index[emv_signals['emv_combined_signal'] == -1].tolist()

                    print(f"EMV综合信号：{len(emv_buy_signals)} 个买入信号，{len(emv_sell_signals)} 个卖出信号")
                    if emv_buy_signals:
                        print(f"最近的EMV买入信号日期: {emv_buy_signals[-3:] if len(emv_buy_signals) > 3 else emv_buy_signals}")

                    # 获取市场效率评估
                    if hasattr(emv, 'get_market_efficiency'):
                        market_efficiency = emv.get_market_efficiency(data)
                        latest_efficiency = market_efficiency['market_efficiency'].iloc[-1] if not market_efficiency.empty else None
                        print(f"最新市场效率: {latest_efficiency:.2f if latest_efficiency is not None else 'N/A'}")
            else:
                print("无法创建EMV指标")
        except Exception as e:
            print(f"EMV指标计算失败: {e}")

        try:
            print("\n9. 测试日内波动率指标 (IntradayVolatility)")
            intraday_vol = complete_registry.create_indicator('INTRADAY_VOLATILITY', smooth_period=5)
            if intraday_vol:
                iv_result = intraday_vol.calculate(data)
                print(f"日内波动率计算结果示例: \n{iv_result[['volatility', 'volatility_ma']].tail(3)}")
            else:
                print("无法创建INTRADAY_VOLATILITY指标")
        except Exception as e:
            print(f"日内波动率指标计算失败: {e}")

        try:
            print("\n10. 测试V形反转指标 (VShapedReversal)")
            v_reversal = complete_registry.create_indicator('V_SHAPED_REVERSAL', decline_period=5, rebound_period=5)
            if v_reversal:
                v_result = v_reversal.calculate(data)
                reversal_count = v_result['v_reversal'].sum()
                print(f"V形反转识别结果: {reversal_count} 次反转")
                if reversal_count > 0:
                    reversal_dates = v_result.index[v_result['v_reversal']].tolist()
                    print(f"反转日期: {reversal_dates[:3]}{'...' if len(reversal_dates) > 3 else ''}")
            else:
                print("无法创建V_SHAPED_REVERSAL指标")
        except Exception as e:
            print(f"V形反转指标计算失败: {e}")

        try:
            print("\n11. 测试岛型反转指标 (IslandReversal)")
            island_reversal = complete_registry.create_indicator('ISLAND_REVERSAL', gap_threshold=0.01, island_max_days=5)
            if island_reversal:
                ir_result = island_reversal.calculate(data)
                top_islands = ir_result['top_island_reversal'].sum()
                bottom_islands = ir_result['bottom_island_reversal'].sum()
                print(f"岛型反转识别结果: {top_islands} 次顶部岛型反转, {bottom_islands} 次底部岛型反转")
            else:
                print("无法创建ISLAND_REVERSAL指标")
        except Exception as e:
            print(f"岛型反转指标计算失败: {e}")

        try:
            print("\n12. 测试时间周期分析指标 (TimeCycleAnalysis)")
            cycle_analysis = complete_registry.create_indicator('TIME_CYCLE_ANALYSIS', min_cycle_days=10, max_cycle_days=120)
            if cycle_analysis:
                ca_result = cycle_analysis.calculate(data)
                print("时间周期分析完成")
                if hasattr(ca_result, 'future_turning_points') and ca_result.future_turning_points:
                    print(f"未来潜在转折点: {len(ca_result.future_turning_points)} 个")
                    for i, point in enumerate(ca_result.future_turning_points[:3]):
                        print(f"  {i+1}. {point['date'].strftime('%Y-%m-%d')} - {point['type']} (周期 {point['cycle']})")
            else:
                print("无法创建TIME_CYCLE_ANALYSIS指标")
        except Exception as e:
            print(f"时间周期分析失败: {e}")
        
        # 使用统一注册系统创建指标
        print("\n使用统一注册系统创建指标示例:")

        try:
            factory_vr = complete_registry.create_indicator("VOLUME_RATIO", reference_period=5)
            if factory_vr:
                factory_result = factory_vr.calculate(data)
                print(f"通过统一注册系统创建的量比指标结果: \n{factory_result[['volume_ratio']].tail(3)}")
            else:
                print("无法通过统一注册系统创建VOLUME_RATIO指标")
        except Exception as e:
            print(f"量比指标统一注册系统创建失败: {e}")
        
        # 综合选股策略示例
        print("\n综合选股策略示例:")
        
        try:
            # 结合多个指标进行选股
            combined_signals = pd.DataFrame(index=data.index)
            combined_signals['price'] = data['close']
            
            # 重新创建指标实例用于综合策略
            volume_ratio = complete_registry.create_indicator('VOLUME_RATIO', reference_period=5)
            obv = complete_registry.create_indicator('OBV')
            mfi = complete_registry.create_indicator('MFI', period=14)
            platform_breakout = complete_registry.create_indicator('PLATFORM_BREAKOUT', platform_period=20, threshold=0.03)
            v_reversal = complete_registry.create_indicator('V_SHAPED_REVERSAL', decline_period=5, rebound_period=5)

            # 1. 量比大于1.5
            if volume_ratio:
                vr_data = volume_ratio.calculate(data)
                combined_signals['high_volume'] = vr_data['volume_ratio'] > 1.5
            else:
                combined_signals['high_volume'] = False

            # 2. OBV上升趋势
            if obv:
                obv_data = obv.calculate(data)
                obv_trend = pd.Series(0, index=obv_data.index)
                for i in range(5, len(obv_data)):
                    if obv_data['obv'].iloc[i] > obv_data['obv'].iloc[i-5]:
                        obv_trend.iloc[i] = 1
                    else:
                        obv_trend.iloc[i] = -1
                combined_signals['obv_uptrend'] = obv_trend > 0
            else:
                combined_signals['obv_uptrend'] = False

            # 3. 资金流向指标超卖反转
            if mfi:
                mfi_data = mfi.calculate(data)
                mfi_signal = pd.Series(False, index=mfi_data.index)
                for i in range(1, len(mfi_data)):
                    if mfi_data['mfi'].iloc[i-1] < 20 and mfi_data['mfi'].iloc[i] >= 20:
                        mfi_signal.iloc[i] = True
                combined_signals['mfi_oversold_reversal'] = mfi_signal
            else:
                combined_signals['mfi_oversold_reversal'] = False

            # 4. 平台突破信号
            if platform_breakout:
                pb_data = platform_breakout.calculate(data)
                combined_signals['platform_breakout'] = pb_data['platform_breakout']
            else:
                combined_signals['platform_breakout'] = False

            # 5. V形反转信号
            if v_reversal:
                v_data = v_reversal.calculate(data)
                combined_signals['v_reversal'] = v_data['v_reversal']
            else:
                combined_signals['v_reversal'] = False
            
            # 组合选股条件
            combined_signals['buy_signal'] = (
                combined_signals['high_volume'] & 
                combined_signals['obv_uptrend'] & 
                (combined_signals['mfi_oversold_reversal'] | 
                 combined_signals['platform_breakout'] | 
                 combined_signals['v_reversal'])
            )
            
            buy_dates = combined_signals.index[combined_signals['buy_signal']].tolist()
            print(f"综合选股结果: 找到 {len(buy_dates)} 个买入信号")
            if buy_dates:
                print(f"买入信号日期: {buy_dates[:5]}{'...' if len(buy_dates) > 5 else ''}")
        except Exception as e:
            print(f"综合选股策略执行失败: {e}")
        
        print("\n分析完成！")
            
    except Exception as e:
        print(f"程序执行过程中发生错误: {e}")


if __name__ == "__main__":
    try:
        mainUsenewindicators()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序发生未处理的异常: {e}") 