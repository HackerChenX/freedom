#!/usr/bin/python
# -*- coding: UTF-8 -*-

import pandas as pd
import pymysql

# db = pymysql.connect(host="localhost", user="root", password="cxy223826", database="stock")
db = pymysql.connect(host="82.157.161.208", user="root", password="jpress", database="stock")
cursor = db.cursor()


def save_stock_info(stock_info: pd.DataFrame, level):
    # 循环stock_info，插入
    for index, row in stock_info.iterrows():
        insert_sql = """
        INSERT INTO stock_info(code, name, date, level, open, close, high, low, volume, turnover_rate, 
        price_change, price_range) 
        VALUES ('%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s')
        """ % (row["股票代码"], row["股票名称"].replace(" ", ""), row["日期"], level, row["开盘"],
               row["收盘"], row["最高"], row["最低"], row["成交量"],
               row["换手率"], row["涨跌幅"], row["振幅"])
        try:
            cursor.execute(insert_sql)
            db.commit()
        except:
            # 如果发生错误则回滚
            db.rollback()


def get_stock_info(code, level, start="20170101", end="20240101"):
    sql = """
    SELECT * FROM stock_info WHERE code = '%s' AND level = '%s' AND date BETWEEN "%s" and "%s" order by date
    """ % (code, level, start, end)
    cursor.execute(sql)
    return cursor.fetchall()


def get_stock_sample():
    sql = """
    SELECT * FROM stock_sample
    """
    cursor.execute(sql)
    return cursor.fetchall()


def get_avg_price(code, start):
    sql = """
    SELECT
        max(high) AS avg_high_price
    FROM stock_info WHERE
        `code` = '%s' AND date >'%s' and date<= DATE_SUB( '%s', INTERVAL -60 DAY ) AND level = '日'
	""" % (code, start, start)
    cursor.execute(sql)
    # 获取单个值
    return cursor.fetchone()[0]


def update_industry(codes, industry):
    # codes转换为逗号分割的字符串
    codes = ",".join(codes).replace(",暂无成份股数据", "")
    sql = """
    UPDATE stock_info SET industry = '%s' WHERE code in (%s)
    """ % (industry, codes)
    cursor.execute(sql)
    db.commit()


def get_industry_stock(industry):
    sql = """
    SELECT distinct code FROM stock_info WHERE industry = '%s'
    """ % industry
    cursor.execute(sql)
    return cursor.fetchall()


def get_max_date():
    sql = """
    SELECT max(date) FROM stock_info
    """
    cursor.execute(sql)
    return cursor.fetchone()[0]
