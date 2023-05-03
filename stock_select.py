import csv
import datetime
import os
import shutil
import time

import pandas as pd

import db
import formula
import multiprocessing as mp

global_date = "20240316"


def stock_select():
    stock_code_name = pd.read_csv("stock_code_name.csv", dtype=str)
    start_time = time.time()
    pool = mp.Pool()  # 创建线程池
    for index, row in stock_code_name.iterrows():
        pool.apply_async(选股, args=(row["code"],))  # 并发执行每个任务
    pool.close()  # 关闭线程池
    pool.join()  # 等待所有线程执行结束
    end_time = time.time()
    print("耗时", end_time - start_time)
    # task("300981")


def stock_select_by_industry():
    start_time = time.time()
    pool = mp.Pool()  # 创建线程池
    industries = formula.主线()
    for industry in industries:
        codes = db.get_industry_stock(industry)
        # 遍历codes
        for code in iter(codes):
            pool.apply_async(选股, args=(code[0],))  # 并发执行每个任务
    pool.close()  # 关闭线程池
    pool.join()  # 等待所有线程执行结束
    end_time = time.time()
    print("耗时", end_time - start_time)


def stock_select_list():
    code_lst = ["603121", "603660", "603778", "605268", "605488", "605589", "688021", "688169", "688395", "688510",
                "000600", "001323", "002582", "300644", "300991", "301005", "301062", "301093", "301227"]
    start_time = time.time()
    pool = mp.Pool()  # 创建线程池
    for code in code_lst:
        pool.apply_async(选股, args=(code, ""))
    pool.close()  # 关闭线程池
    pool.join()  # 等待所有线程执行结束
    end_time = time.time()
    print("耗时", end_time - start_time)


def stock_select_single():
    start_time = time.time()
    选股("300113", True)
    end_time = time.time()
    print("耗时", end_time - start_time)


def 回测_single():
    lst = (("301029", "20230316", 1),)
    回测_汇总(lst)


def 回测_batch():
    lst = (("605136", "20230316", 1), ("300959", "20230316", 1), ("300280", "20230316", 1), ("002292", "20230307", 1),
           ("300884", "20230316", 1))
    回测_汇总(lst)


def 回测_db():
    回测_汇总(db.get_stock_sample())


def 回测_csv():
    # 读取txt文件
    with open("选股结果.txt", "r", encoding="utf-8") as f:
        lst = []
        for line in f.readlines():
            lst.append(line.strip().split(" "))
    # lst转换为"code", "name", "行业", "涨幅", "指标数"的dataframe
    df = pd.DataFrame(lst, columns=["code", "name", "行业", "涨幅", "指标数"])
    # 按照涨幅从小到大排序
    df = df.sort_values(by="涨幅", ascending=False)
    # 按照涨幅是否小于0分为两组
    df1 = df[df["涨幅"].astype(float) < 0.05]
    df2 = df[df["涨幅"].astype(float) >= 0.1]
    # 输出涨幅大于0.1的数量，占总数的比例，保留两位小数
    print("涨幅大于0.1的数量", len(df2), "占比", round(len(df2) / len(df), 2))
    print("涨幅小于0.05的数量", len(df1), "占比", round(len(df1) / len(df), 2))
    # 输出两组数据大小
    print("踩雷", len(df1))
    print("选股", len(df2))
    # df1按照涨幅从小到大排序
    df1 = df1.sort_values(by="涨幅", ascending=True)
    # df2按照涨幅从大到小排序
    df2 = df2.sort_values(by="涨幅", ascending=False)
    # df2和df1长度保持一直
    # df2 = df2[:len(df1)]
    # 两组数据转换为list，字段为code,date,type，其中df2的type=1，df1的type=3，date=20230316
    lst1 = df1[["code", "name"]].values.tolist()
    lst2 = df2[["code", "name"]].values.tolist()
    lst1 = [(x[0], global_date, 3) for x in lst1]
    lst2 = [(x[0], global_date, 1) for x in lst2]
    lst = lst1 + lst2
    回测_汇总(lst)


def 回测_汇总(lst):
    买入_dict = {}
    选股_dict = {}
    卖出_dict = {}
    踩雷_dict = {}
    for row in lst:
        dict1 = 回测_task(row[0], end=row[1])
        if row[2] == 1:
            # 如果dict1中的value=True，则all_dict对应Key的value加1
            for key in dict1.keys():
                if dict1[key]:
                    买入_dict[key] = 买入_dict.get(key, 0) + 1
        elif row[2] == 2:
            for key in dict1.keys():
                if dict1[key]:
                    选股_dict[key] = 选股_dict.get(key, 0) + 1
        elif row[2] == 3:
            for key in dict1.keys():
                if dict1[key]:
                    卖出_dict[key] = 卖出_dict.get(key, 0) + 1
        elif row[2] == 4:
            for key in dict1.keys():
                if dict1[key]:
                    踩雷_dict[key] = 踩雷_dict.get(key, 0) + 1
    # 选股和买入相同key的value相加
    选股买入_dict = 选股_dict.copy()
    for key in 买入_dict.keys():
        选股买入_dict[key] = 选股买入_dict.get(key, 0) + 买入_dict[key]
    # 买入和卖出相同key的value相减
    买入卖出_dict = 买入_dict.copy()
    for key in 卖出_dict.keys():
        买入卖出_dict[key] = 买入卖出_dict.get(key, 0) - 卖出_dict[key]
    # dict按照value从大到小排序
    买入_dict = sorted(买入_dict.items(), key=lambda x: x[1], reverse=True)
    选股_dict = sorted(选股_dict.items(), key=lambda x: x[1], reverse=True)
    卖出_dict = sorted(卖出_dict.items(), key=lambda x: x[1], reverse=True)
    踩雷_dict = sorted(踩雷_dict.items(), key=lambda x: x[1], reverse=True)
    选股买入_dict = sorted(选股买入_dict.items(), key=lambda x: x[1], reverse=True)
    print("买入", 买入_dict)
    print("选股", 选股_dict)
    print("卖出", 卖出_dict)
    print("踩雷", 踩雷_dict)
    print("选股买入", 选股买入_dict)
    print("买入卖出", 买入卖出_dict)


def 回测_task(code, end="2024001"):
    fm = formula.Formula(code, end=end)
    dict1 = {}
    dict1["弹性"] = fm.弹性()
    dict1["日均线上移"] = fm.日均线上移()
    dict1["周均线上移"] = fm.周均线上移()
    dict1["日-K上移"] = fm.K上移("日")
    dict1["日-D上移"] = fm.D上移("日")
    dict1["日-J上移"] = fm.J上移("日")
    dict1["日-DEA上移"] = fm.DEA上移("日")
    dict1["日-DIFF上移"] = fm.DIFF上移("日")
    dict1["日-MACD上移"] = fm.MACD上移("日")
    dict1["日-MACD小于"] = fm.MACD小于("日", 0.9)
    dict1["周-J上移"] = fm.J上移("周")
    dict1["周-K上移"] = fm.K上移("周")
    dict1["周-D上移"] = fm.D上移("周")
    dict1["周-DEA上移"] = fm.DEA上移("周")
    dict1["周-DIFF上移"] = fm.DIFF上移("周")
    dict1["周-MACD上移"] = fm.MACD上移("周")
    dict1["周-MACD小于"] = fm.MACD小于("周", 1.5)
    dict1["月-J上移"] = fm.J上移("月")
    dict1["月-K上移"] = fm.K上移("月")
    dict1["月-D上移"] = fm.D上移("月")
    dict1["月-DEA上移"] = fm.DEA上移("月")
    dict1["月-DIFF上移"] = fm.DIFF上移("月")
    dict1["月-MACD上移"] = fm.MACD上移("月")
    dict1["日-换手率大于"] = fm.换手率大于(0.6)
    dict1["日-缩量"] = fm.缩量()
    dict1["日-回踩均线"] = fm.回踩均线("日")
    dict1["月-吸筹"] = fm.吸筹("月", 10)
    dict1["周-吸筹"] = fm.吸筹("周", 20)
    dict1["日-吸筹"] = fm.吸筹("日", 20)
    dict1["15-吸筹"] = fm.吸筹("15", 20)
    dict1["30-吸筹"] = fm.吸筹("30", 20)
    dict1["60-吸筹"] = fm.吸筹("60", 20)
    dict1["大于boll中轨"] = fm.大于boll中轨()
    print(code, dict1, code, 涨幅(code, end, fm.dataDay.close[-1]))
    return dict1


def 选股(code, log=False):
    date = global_date
    fm = formula.Formula(code, end=date)
    result = 筛选器(fm, log)
    if result[0]:
        涨幅_value = 涨幅(fm.get_code(), date, fm.dataDay.close[-1])
        print(fm.get_desc(), "符合条件", 涨幅_value, result[1].count(True), result[1])
        # 输出到txt文件
        with open("选股结果.txt", "a", newline="") as f:
            f.write(fm.get_desc() + " " + str(涨幅_value) + " " + str(result[1].count(True)) + "\n")
    # 复制文件
    shutil.copyfile("选股结果.txt", "/Users/hacker/Documents/财富自由/选股结果.txt")


def 筛选器(fm, log=False):
    return 买入_基本(fm, log)


def stock_simple():
    simples = db.get_stock_sample()
    simples = pd.DataFrame(simples, columns=['code', 'date', 'type'])
    # 根据type分组
    grouped = simples.groupby('type')
    # 遍历每一组，根据code和date调用回测_task
    for name, group in grouped:
        print(name)
        for index, row in group.iterrows():
            回测_task(row["code"], row["date"])


def 买入_基本(fm, log=False):
    基本 = fm.弹性() and fm.换手率大于(0.6) and fm.缩量()
    均线 = fm.日均线上移() and fm.周均线上移() and fm.回踩均线("日")
    macd = fm.MACD小于("日", 0.9) and fm.DIFF上移("月") and fm.MACD上移("月") and fm.DIFF上移("周") and fm.DEA上移(
        "周") and fm.MACD上移("周")
    月kdj = fm.K和D和J上移("月")
    周kdj = fm.K和D上移("周")
    月吸筹 = (fm.吸筹("月", 15))
    周吸筹 = fm.吸筹("周", 20)
    日kdj = fm.macd_kdj任一指标上移("日")
    日吸筹 = fm.吸筹("日", 20) or fm.吸筹("60", 20) or fm.吸筹("30", 20) or fm.吸筹("15", 20)
    boll = fm.大于boll中轨()
    吸筹 = False
    if 月吸筹 and 周吸筹:
        吸筹 = True
    elif 日吸筹:
        吸筹 = True
    result = [基本, 均线, macd, 月kdj, 周kdj, 月吸筹, 日吸筹, 周吸筹, 日kdj]
    if log:
        print(fm.get_desc(), "弹性", fm.弹性(), "换手率大于", fm.换手率大于(0.6), "缩量", fm.缩量())
        print(fm.get_desc(), "日均线上移", fm.日均线上移(), "周均线上移", fm.周均线上移(), "回踩均线",
              fm.回踩均线("日"))
        print(fm.get_desc(), "日-MACD小于", fm.MACD小于("日", 0.9), "月-DIFF上移", fm.DIFF上移("月"), "月-MACD上移",
              fm.MACD上移("月"),
              "周-DIFF上移", fm.DIFF上移("周"), "周-DEA上移", fm.DEA上移("周"))
        print(fm.get_desc(), "月-K和D和J上移", fm.K和D和J上移("月"), "周-K和D上移", fm.K和D上移("周"))
        print(fm.get_desc(), "月-吸筹", fm.吸筹("月", 15), "周-吸筹", fm.吸筹("周", 20), "日-吸筹", fm.吸筹("日", 20),
              "60-吸筹", fm.吸筹("60", 20), "30-吸筹", fm.吸筹("30", 20), "15-吸筹", fm.吸筹("15", 20))
        print(fm.get_desc(), "日-KDJ上移", fm.macd_kdj任一指标上移("日"))
        print(fm.get_desc(), "大于boll中轨", fm.大于boll中轨())
    return 基本 and 均线 and macd and (月kdj or 周kdj) and 吸筹 and 日kdj and boll, result


def 涨幅(code, start, close):
    max_price = db.get_avg_price(code, start)
    # 计算涨幅
    if max_price is None:
        return 0
    else:
        return round((max_price - close) / close, 2)


def 同步数据():
    max_date = db.get_max_date()
    # 加一天
    max_date = max_date + datetime.timedelta(days=1)
    max_date = max_date.strftime("%Y%m%d")
    stock_code_name = pd.read_csv("stock_code_name.csv", dtype=str)
    start_time = time.time()
    pool = mp.Pool()  # 创建线程池
    for index, row in stock_code_name.iterrows():
        code = row["code"]
        pool.apply_async(同步数据_task, args=(code, max_date))
    pool.close()  # 关闭线程池
    pool.join()  # 等待所有线程执行结束
    end_time = time.time()
    print("耗时", end_time - start_time)


def 同步数据_task(code, max_date):
    formula.Formula(code, start=max_date, sync=True)


if __name__ == '__main__':
    # stock_select()
    # stock_select_list()
    # stock_select_single()
    # 回测_single()
    # 回测_batch()
    # 回测_db()
    # 回测_csv()
    # formula.主线()
    stock_select_by_industry()
    # 同步数据()
