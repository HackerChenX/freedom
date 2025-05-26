#!/usr/bin/python
# -*- coding: UTF-8 -*-

import datetime
import pandas as pd

# 直接从indicators模块导入所需的函数
from indicators import LLV, HHV, SMA, EMA, REF
from db.clickhouse_db import get_clickhouse_db
from kline_period import KlinePeriod


def main():
    """
    演示如何使用ClickHouseDB类查询不同级别的数据
    """
    
    # 连接到ClickHouse，使用get_clickhouse_db获取连接
    db = get_clickhouse_db()
    
    # 获取今天的日期
    today = datetime.date.today()
    
    # 设置查询范围（过去90天）
    end_date = today.strftime('%Y-%m-%d')
    start_date = (today - datetime.timedelta(days=365*3)).strftime('%Y-%m-%d')
    
    print(f"查询范围: {start_date} 至 {end_date}")
    
    # 1. 查询15分钟级别数据（从数据库直接获取）
    print("\n======== 查询15分钟级别数据（从数据库直接获取） ========")
    stock_code = '603359'  # 平安银行
    data_15min_list = db.get_stock_info(
        code=stock_code,
        start=start_date,
        end=end_date,
        level=str(KlinePeriod.MIN_15)
    )
    
    # 将列表转换为DataFrame
    columns = ['code', 'name', 'date', 'level', 'open', 'close', 'high', 'low', 'volume', 
               'turnover_rate', 'price_change', 'price_range', 'industry']
    data_15min = pd.DataFrame(data_15min_list, columns=columns)
    
    # 展示结果
    if not data_15min.empty:
        print(f"获取到 {len(data_15min)} 条15分钟级别数据")
        print(data_15min.head())
    else:
        print(f"未找到股票 {stock_code} 的15分钟级别数据")
    
    # 2. 查询30分钟级别数据（由15分钟数据计算得出）
    print("\n======== 查询30分钟级别数据（由15分钟数据计算得出） ========")
    data_30min_list = db.get_stock_info(
        code=stock_code,
        start=start_date,
        end=end_date,
        level=str(KlinePeriod.MIN_30)
    )
    
    # 将列表转换为DataFrame
    data_30min = pd.DataFrame(data_30min_list, columns=columns)

    查找最近吸筹日期(data_30min, 30)
    
    # 展示结果
    if not data_30min.empty:
        print(f"计算得到 {len(data_30min)} 条30分钟级别数据")
        print(data_30min.head())
    else:
        print(f"未找到足够的15分钟数据来计算30分钟级别数据")
    
    # 3. 查询日线数据（由15分钟数据计算得出）
    print("\n======== 查询日线数据 ========")
    data_daily_list = db.get_stock_info(
        code=stock_code,
        start=start_date,
        end=end_date,
        level=str(KlinePeriod.DAILY)
    )
    
    # 将列表转换为DataFrame
    data_daily = pd.DataFrame(data_daily_list, columns=columns)
    
    # 展示结果
    if not data_daily.empty:
        print(f"计算得到 {len(data_daily)} 条日线数据")
        print(data_daily.head())
    else:
        print(f"未找到足够的15分钟数据来计算日线数据")

    查找最近吸筹日期(data_daily, 30)

    # 4. 查询周线数据（由15分钟数据计算得出）
    print("\n======== 查询周线数据 ========")
    data_weekly_list = db.get_stock_info(
        code=stock_code,
        start=start_date,
        end=end_date,
        level=str(KlinePeriod.WEEKLY)
    )
    
    # 将列表转换为DataFrame
    data_weekly = pd.DataFrame(data_weekly_list, columns=columns)
    
    # 展示结果
    if not data_weekly.empty:
        print(f"计算得到 {len(data_weekly)} 条周线数据")
        print(data_weekly.head())
    else:
        print(f"未找到足够的15分钟数据来计算周线数据")
    
    # 5. 查询月线数据（由15分钟数据计算得出）
    print("\n======== 查询月线数据 ========")
    data_monthly_list = db.get_stock_info(
        code=stock_code,
        start=start_date,
        end=end_date,
        level=str(KlinePeriod.MONTHLY)
    )
    
    # 将列表转换为DataFrame
    data_monthly = pd.DataFrame(data_monthly_list, columns=columns)
    
    # 展示结果
    if not data_monthly.empty:
        print(f"计算得到 {len(data_monthly)} 条月线数据")
        print(data_monthly.head())
    else:
        print(f"未找到足够的15分钟数据来计算月线数据")
    
    # 6. 查询行业数据
    print("\n======== 查询行业数据 ========")
    industry_name = '银行'
    industry_data = db.get_industry_info(
        symbol=industry_name,
        start=start_date,
        end=end_date
    )
    
    # 展示结果
    if not industry_data.empty:
        print(f"获取到 {len(industry_data)} 条行业数据")
        print(industry_data.head())
    else:
        print(f"未找到行业 {industry_name} 的数据")
    
    # 7. 获取最高价格
    print("\n======== 获取股票平均价格 ========")
    high_price = db.get_avg_price(
        code=stock_code,
        start=start_date
    )
    print(f"股票 {stock_code} 从 {start_date} 起的最高价格: {high_price}")
    
    # 8. 获取股票数据的最大日期
    print("\n======== 获取股票数据的最大日期 ========")
    max_date = db.get_stock_max_date(stock_code)
    print(f"股票数据的最大日期: {max_date}")


def 查找最近吸筹日期(stock_data, n=30):
    """查找最近一次满足吸筹条件的日期

    参数:
        stock_data: 股票数据DataFrame
        n: 吸筹周期

    返回:
        最近的吸筹日期，如果没有找到则返回None
    """
    if stock_data.empty:
        return None

    # 确保DataFrame中有必要的列
    if 'close' not in stock_data.columns or 'low' not in stock_data.columns or 'high' not in stock_data.columns:
        print(f"数据缺少必要的列，当前列: {stock_data.columns.tolist()}")
        return None

    # 获取需要的数据列
    C = stock_data['close'].values
    L = stock_data['low'].values
    H = stock_data['high'].values

    llv = LLV(L, 55)
    hhv = HHV(H, 55)
    v11 = 3 * SMA((C - llv) / (hhv - llv) * 100, 5, 1) - 2 * SMA(SMA((C - llv) / (hhv - llv) * 100, 5, 1), 3, 1)
    v12 = (EMA(v11, 3) - REF(EMA(v11, 3), 1)) / REF(EMA(v11, 3), 1) * 100
    ema_v11 = EMA(v11, 3)

    # 创建一个包含日期和吸筹指标的DataFrame
    df = pd.DataFrame({
        'date': stock_data['date'].values,
        'close': C,
        'ema_v11': ema_v11,
        'v12': v12
    })

    # 计算每一天是否满足吸筹条件
    吸筹结果 = []
    所有日期 = []

    for i in range(n, len(df)):
        # 计算前n天内是否有满足条件的天数
        condition1 = any([df['ema_v11'].iloc[j] < 13 for j in range(i - n + 1, i + 1)])
        condition2 = any([df['v12'].iloc[j] > 13 for j in range(i - n + 1, i + 1)])

        满足吸筹 = condition1 or (condition1 and condition2)
        吸筹结果.append(满足吸筹)
        所有日期.append(df['date'].iloc[i])

    # 创建一个包含日期和吸筹结果的DataFrame
    结果df = pd.DataFrame({
        'date': 所有日期,
        'xc': 吸筹结果
    })

    # 找到最近一次满足吸筹条件的日期
    满足条件 = 结果df[结果df['xc'] == True]

    if len(满足条件) > 0:
        最近日期 = 满足条件.iloc[-1]['date']
        print(f"找到最近吸筹日期: {最近日期}")
        return 最近日期
    else:
        print("在查询时间范围内未找到满足吸筹条件的日期")
        return None


def 计算吸筹(data, n=10, index=None):
    """计算吸筹指标

    参数:
        data: 股票数据
        n: 吸筹周期
        index: 如果指定，则计算该索引处的吸筹情况；如果为None，则计算最后一天

    返回:
        是否满足吸筹条件
    """
    if len(data.close) == 0:
        return False

    C = data.close
    L = data.low
    H = data.high

    llv = LLV(L, 55)
    hhv = HHV(H, 55)
    v11 = 3 * SMA((C - llv) / (hhv - llv) * 100, 5, 1) - 2 * SMA(SMA((C - llv) / (hhv - llv) * 100, 5, 1), 3, 1)
    v12 = (EMA(v11, 3) - REF(EMA(v11, 3), 1)) / REF(EMA(v11, 3), 1) * 100
    ema_v11 = EMA(v11, 3)

    # 如果指定了index，则计算该点的吸筹情况
    if index is not None:
        # 确保index是有效的
        if index < n or index >= len(data.close):
            return False

        # 日线吸筹条件
        condition1 = any([ema_v11[i] < 13 for i in range(index - n + 1, index + 1)])
        condition2 = any([v12[i] > 13 for i in range(index - n + 1, index + 1)])

        return condition1 or (condition1 and condition2)
    else:
        # 计算最后一天的吸筹情况
        condition1 = countListAnyMatch(ema_v11, n, lambda x: x < 13)
        condition2 = countListAnyMatch(v12, n, lambda x: x > 13)

        return condition1 or (condition1 and condition2)


def countListAnyMatch(lst, n, func):
    """判断列表最后n个元素是否包含满足func条件的元素"""
    n = n if n < len(lst) else len(lst)
    return any([func(lst[i]) for i in range(len(lst) - n, len(lst))])


if __name__ == "__main__":
    main() 