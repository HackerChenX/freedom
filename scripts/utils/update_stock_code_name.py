#!/usr/bin/python
# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
import akshare as ak
import re
import os

def update_stock_code_name():
    """
    更新全市场股票代码和名称，排除ST、科创板和北交所股票
    """
    print("开始获取全市场股票代码和名称...")
    
    try:
        # 使用akshare获取A股所有股票代码和名称
        stock_info = ak.stock_info_a_code_name()
        
        if stock_info.empty:
            print("无法获取股票信息，请检查网络连接或ak接口")
            return False
        
        print(f"原始股票数量: {len(stock_info)}")
        
        # 排除ST股票：检查股票名称中是否包含"ST"
        non_st_stocks = stock_info[~stock_info['name'].str.contains('ST', case=False)]
        print(f"排除ST股票后数量: {len(non_st_stocks)}")
        
        # 排除科创板股票：股票代码以"688"开头
        # 使用更精确的条件：确保code是以688开头的6位数字
        non_st_kc_stocks = non_st_stocks[~(non_st_stocks['code'].str.match('^688\d{3}$'))]
        print(f"排除科创板股票后数量: {len(non_st_kc_stocks)}")
        
        # 排除北交所股票：股票代码以"83"、"87"、"43"或"92"开头
        final_stocks = non_st_kc_stocks[~(
            non_st_kc_stocks['code'].str.match('^83\d{4}$') | 
            non_st_kc_stocks['code'].str.match('^87\d{4}$') |
            non_st_kc_stocks['code'].str.match('^43\d{4}$') |
            non_st_kc_stocks['code'].str.match('^92\d{4}$')   # 增加92开头的北交所股票
        )]
        print(f"排除北交所股票后数量: {len(final_stocks)}")
        
        # 重置索引
        final_stocks = final_stocks.reset_index()
        
        # 保存到CSV文件
        final_stocks.to_csv("stock_code_name.csv", index=False)
        print(f"股票代码和名称已更新并保存到 stock_code_name.csv")
        
        # 输出一些股票分类统计
        print("\n股票市场分布:")
        # 上海主板 (60开头)
        sh_main = stock_info[stock_info['code'].str.match('^60\d{4}$')]
        print(f"上海主板: {len(sh_main)}只")
        
        # 深圳主板 (00开头)
        sz_main = stock_info[stock_info['code'].str.match('^00\d{4}$')]
        print(f"深圳主板: {len(sz_main)}只")
        
        # 创业板 (30开头)
        chinext = stock_info[stock_info['code'].str.match('^30\d{4}$')]
        print(f"创业板: {len(chinext)}只")
        
        # 科创板 (688开头)
        sci_tech = stock_info[stock_info['code'].str.match('^688\d{3}$')]
        print(f"科创板: {len(sci_tech)}只")
        
        # 北交所 (83、87、43、92开头)
        beijing = stock_info[
            stock_info['code'].str.match('^83\d{4}$') | 
            stock_info['code'].str.match('^87\d{4}$') |
            stock_info['code'].str.match('^43\d{4}$') |
            stock_info['code'].str.match('^92\d{4}$')
        ]
        print(f"北交所: {len(beijing)}只")
        
        # ST股票
        st_stocks = stock_info[stock_info['name'].str.contains('ST', case=False)]
        print(f"ST股票: {len(st_stocks)}只")
        
        return True
    
    except Exception as e:
        print(f"更新股票代码和名称时发生错误: {e}")
        return False

if __name__ == "__main__":
    update_stock_code_name() 