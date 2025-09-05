#!/usr/bin/env python3
"""
查找2025年5月12日出现ZXM吸筹信号的个股

基于修复后的ZXM_ABSORB指标，使用ClickHouse真实数据查找符合ZXM体系吸筹条件的个股
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

def get_stock_data(stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取股票数据 - 使用ClickHouse真实数据

    Args:
        stock_code: 股票代码
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        包含OHLCV数据的DataFrame
    """
    try:
        # 获取统一数据管理器
        data_manager = get_unified_data_manager()

        # 使用数据层方法获取真实股票数据
        df = data_manager.get_stock_data(
            code=stock_code,
            start_date=start_date,
            end_date=end_date,
            level='日线'
        )

        if df.empty:
            logger.warning(f"未获取到股票 {stock_code} 的数据")
            return pd.DataFrame()

        # 确保数据格式正确
        if 'date' in df.columns:
            df.set_index('date', inplace=True)

        # 确保必要的列存在
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"缺少必要列: {col}")
                return pd.DataFrame()

        # 数据类型转换
        for col in required_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 移除包含NaN的行
        df = df.dropna(subset=required_columns)

        logger.debug(f"获取股票 {stock_code} 数据: {len(df)} 条记录")
        return df

    except Exception as e:
        logger.error(f"获取股票 {stock_code} 数据失败: {e}")
        return pd.DataFrame()

def get_stock_list() -> list:
    """
    获取股票列表 - 从ClickHouse数据库获取真实股票代码

    Returns:
        股票代码列表
    """
    try:
        # 获取统一数据管理器
        data_manager = get_unified_data_manager()

        # 查询数据库中存在的股票代码 - 扩大范围
        query = """
        SELECT DISTINCT code
        FROM stock_info
        WHERE level = '日线'
        AND date >= '2024-01-01'
        ORDER BY code
        LIMIT 200
        """

        with data_manager.connection_pool.get_connection() as connection:
            cursor = connection.execute(query)
            results = list(cursor) if hasattr(cursor, '__iter__') else []

            if results:
                stock_codes = [row[0] for row in results]
                logger.info(f"从数据库获取到 {len(stock_codes)} 只股票")
                return stock_codes
            else:
                logger.warning("数据库中未找到股票数据，使用默认股票列表")
                # 如果数据库中没有数据，使用一些常见的股票代码作为备选
                return [
                    '000001.SZ', '000002.SZ', '000858.SZ', '002415.SZ', '002594.SZ',
                    '300059.SZ', '300750.SZ', '600036.SH', '600519.SH', '600887.SH',
                    '601318.SH', '601888.SH', '603259.SH', '603501.SH', '688981.SH'
                ]

    except Exception as e:
        logger.error(f"获取股票列表失败: {e}")
        # 返回默认股票列表
        return [
            '000001.SZ', '000002.SZ', '000858.SZ', '002415.SZ', '002594.SZ',
            '300059.SZ', '300750.SZ', '600036.SH', '600519.SH', '600887.SH'
        ]

def find_absorb_signals(target_date: str = '2025-05-12') -> list:
    """
    查找指定日期出现ZXM吸筹信号的个股 - 严格按照ZXM体系标准

    ZXM吸筹条件：
    1. 明确信号：ZXM_ABSORB_SIGNAL=1 或 ZXM_BUY_SIGNAL=1 或 ZXM_COMBINED_SIGNAL=1
    2. 标准条件：EMA_V11 ≤ 13 (ZXM体系标准吸筹阈值)

    Args:
        target_date: 目标日期

    Returns:
        出现吸筹信号的个股列表
    """
    print(f"🔍 开始查找 {target_date} 当天出现ZXM吸筹信号的个股...")
    print("🎯 严格按照ZXM体系标准：EMA_V11 ≤ 13 或明确信号")
    print("=" * 80)
    
    # 创建ZXM_ABSORB指标实例
    zxm_absorb = ZxmAbsorb()
    
    # 获取股票列表
    stock_list = get_stock_list()
    
    # 计算数据范围 - 需要足够的历史数据计算ZXM指标
    target_dt = datetime.strptime(target_date, '%Y-%m-%d')
    start_date = (target_dt - timedelta(days=150)).strftime('%Y-%m-%d')  # 150天历史数据
    end_date = target_date  # 只检查目标日期
    
    absorb_signals = []
    
    print(f"📊 分析股票池: {len(stock_list)}只股票")
    print(f"📅 数据范围: {start_date} 至 {end_date}")
    print(f"🎯 目标日期: {target_date}")
    print()
    
    for i, stock_code in enumerate(stock_list):
        try:
            print(f"[{i+1:2d}/{len(stock_list)}] 分析 {stock_code}...", end=' ')
            
            # 获取股票数据
            stock_data = get_stock_data(stock_code, start_date, end_date)
            
            if len(stock_data) < 60:  # 确保有足够数据
                print("❌ 数据不足")
                continue
            
            # 计算ZXM_ABSORB指标
            result = zxm_absorb.calculate(stock_data)

            # 严格检查目标日期的ZXM吸筹信号
            target_date_str = target_dt.strftime('%Y-%m-%d')

            # 检查目标日期是否在结果中
            matching_rows = result[result.index.strftime('%Y-%m-%d') == target_date_str]

            if not matching_rows.empty:
                row = matching_rows.iloc[0]

                # 检查ZXM吸筹信号 - 严格按照ZXM体系标准
                absorb_signal = row.get('ZXM_ABSORB_SIGNAL', 0)
                buy_signal = row.get('ZXM_BUY_SIGNAL', 0)
                combined_signal = row.get('ZXM_COMBINED_SIGNAL', 0)
                v11_value = row.get('ZXM_V11', 0)
                ema_v11 = row.get('ZXM_EMA_V11', 0)

                # 严格的ZXM吸筹条件：必须有明确信号 OR EMA_V11 <= 13
                has_explicit_signal = absorb_signal == 1 or buy_signal == 1 or combined_signal == 1
                strict_absorb_condition = ema_v11 <= 13  # ZXM体系标准吸筹条件

                if has_explicit_signal or strict_absorb_condition:
                    # 获取股票名称
                    stock_name = get_stock_name(stock_code)

                    signal_info = {
                        'code': stock_code,
                        'name': stock_name,
                        'signal_date': target_date_str,
                        'absorb_signal': absorb_signal,
                        'buy_signal': buy_signal,
                        'combined_signal': combined_signal,
                        'strict_absorb': strict_absorb_condition,
                        'v11_value': round(v11_value, 2),
                        'ema_v11': round(ema_v11, 2),
                        'close_price': round(row['close'], 2)
                    }

                    absorb_signals.append(signal_info)

                    # 确定信号类型
                    signal_type = []
                    if absorb_signal == 1:
                        signal_type.append('吸筹信号')
                    if buy_signal == 1:
                        signal_type.append('买入信号')
                    if combined_signal == 1:
                        signal_type.append('综合信号')
                    if strict_absorb_condition and not signal_type:
                        signal_type.append('ZXM标准吸筹')

                    print(f"✅ 发现{'+'.join(signal_type)}! V11={v11_value:.2f}, EMA_V11={ema_v11:.2f}")
                else:
                    print(f"⚪ 无信号 (V11={v11_value:.2f}, EMA_V11={ema_v11:.2f})")
            else:
                print("❌ 目标日期无数据")
                
        except Exception as e:
            print(f"❌ 计算失败: {e}")
            continue
    
    return absorb_signals

def get_stock_name(stock_code: str) -> str:
    """
    获取股票名称 - 从ClickHouse数据库获取真实股票名称

    Args:
        stock_code: 股票代码

    Returns:
        股票名称
    """
    try:
        # 获取统一数据管理器
        data_manager = get_unified_data_manager()

        # 查询股票名称
        query = f"""
        SELECT DISTINCT name
        FROM stock_info
        WHERE code = '{stock_code}'
        LIMIT 1
        """

        with data_manager.connection_pool.get_connection() as connection:
            cursor = connection.execute(query)
            results = list(cursor) if hasattr(cursor, '__iter__') else []

            if results and results[0][0]:
                return results[0][0]
            else:
                # 如果数据库中没有名称，返回股票代码
                return stock_code

    except Exception as e:
        logger.error(f"获取股票名称失败: {e}")
        return stock_code

def main():
    """主函数"""
    print("🚨 ZXM吸筹信号检测系统")
    print("🎯 基于ZXM体系教程修复后的真实算法")
    print("📊 使用ClickHouse数据库真实股票数据")
    print("=" * 80)
    
    # 查找2025年5月12日的ZXM吸筹信号
    target_date = '2025-05-12'
    absorb_signals = find_absorb_signals(target_date)
    
    print("\n" + "=" * 80)
    print(f"📊 {target_date} ZXM吸筹信号检测结果 (严格标准)")
    print("🎯 ZXM体系标准：EMA_V11 ≤ 13 或明确信号")
    print("=" * 80)
    
    if absorb_signals:
        print(f"🎉 发现 {len(absorb_signals)} 只个股出现ZXM吸筹信号:")
        print()
        
        for i, signal in enumerate(absorb_signals, 1):
            print(f"{i}. {signal['name']} ({signal['code']})")
            print(f"   📅 信号日期: {signal['signal_date']}")
            print(f"   💰 收盘价: {signal['close_price']} 元")
            print(f"   📊 ZXM_V11: {signal['v11_value']}")
            print(f"   📈 EMA_V11: {signal['ema_v11']}")
            print(f"   🔔 信号类型: ", end='')

            signals = []
            if signal['absorb_signal'] == 1:
                signals.append('吸筹信号')
            if signal['buy_signal'] == 1:
                signals.append('买入信号')
            if signal['combined_signal'] == 1:
                signals.append('综合信号')
            if signal['strict_absorb'] and not signals:
                signals.append('ZXM标准吸筹')

            print(' + '.join(signals) if signals else 'ZXM标准吸筹')
            print()
        
        print("🎯 ZXM体系分析建议:")
        print("   ✅ 以上个股符合ZXM体系吸筹条件")
        print("   ✅ 建议结合趋势、弹性、基本面进一步筛选")
        print("   ✅ 等待买点四要素确认后择机介入")
        
    else:
        print("❌ 未发现符合条件的ZXM吸筹信号")
        print("💡 建议:")
        print("   - 扩大股票池范围")
        print("   - 调整信号参数")
        print("   - 检查其他时间周期")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
