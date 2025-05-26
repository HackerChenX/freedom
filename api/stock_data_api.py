#!/usr/bin/python
# -*- coding: UTF-8 -*-

import abc
import pandas as pd
from datetime import datetime, timedelta
import time
import numpy as np
import json
import requests

# 定义抽象数据API基类
class StockDataAPI(abc.ABC):
    """股票数据API抽象基类"""
    
    @abc.abstractmethod
    def get_index_data(self, index_code, start_date, end_date):
        """获取指数历史数据"""
        pass
    
    @abc.abstractmethod
    def get_stock_list(self, date=None):
        """获取股票列表"""
        pass
    
    @abc.abstractmethod
    def get_stock_data(self, stock_code, start_date, end_date):
        """获取股票历史数据"""
        pass
    
    @abc.abstractmethod
    def get_market_data(self, date=None):
        """获取市场涨跌停数据"""
        pass
    
    @abc.abstractmethod
    def get_fund_flow_data(self, date=None):
        """获取资金流向数据"""
        pass
    
    @abc.abstractmethod
    def get_latest_trade_date(self):
        """获取最新交易日期"""
        pass
    
    def format_date(self, date):
        """格式化日期为YYYYMMDD格式"""
        if isinstance(date, str):
            try:
                return datetime.strptime(date, '%Y%m%d').strftime('%Y%m%d')
            except ValueError:
                try:
                    return datetime.strptime(date, '%Y-%m-%d').strftime('%Y%m%d')
                except ValueError:
                    return datetime.now().strftime('%Y%m%d')
        elif isinstance(date, datetime):
            return date.strftime('%Y%m%d')
        else:
            return datetime.now().strftime('%Y%m%d')
    
    def format_date_hyphen(self, date):
        """格式化日期为YYYY-MM-DD格式"""
        if isinstance(date, str):
            try:
                return datetime.strptime(date, '%Y%m%d').strftime('%Y-%m-%d')
            except ValueError:
                try:
                    return datetime.strptime(date, '%Y-%m-%d').strftime('%Y-%m-%d')
                except ValueError:
                    return datetime.now().strftime('%Y-%m-%d')
        elif isinstance(date, datetime):
            return date.strftime('%Y-%m-%d')
        else:
            return datetime.now().strftime('%Y-%m-%d')


# AKShare数据API实现
class AKShareAPI(StockDataAPI):
    """基于AKShare的数据API实现"""
    
    def __init__(self):
        """初始化AKShare API"""
        try:
            import akshare as ak
            self.ak = ak
            self.available = True
        except ImportError:
            print("AKShare未安装或无法导入，部分功能可能不可用")
            self.available = False
    
    def get_latest_trade_date(self):
        """获取最新交易日期
        
        返回:
            最新交易日期，格式为YYYYMMDD
        """
        if not self.available:
            return datetime.now().strftime('%Y%m%d')
        
        try:
            # 尝试通过交易日历获取最新交易日
            try:
                # 获取当前日期
                today = datetime.now()
                
                # 尝试获取交易日历
                try:
                    trade_cal = self.ak.tool_trade_date_hist_sina()
                except:
                    # 尝试其他可能的函数名
                    try:
                        trade_cal = self.ak.stock_zh_index_daily_em()
                    except:
                        # 直接尝试获取上证指数最近数据
                        raise Exception("无法获取交易日历")
                
                if not trade_cal.empty:
                    # 确保日期列为日期类型
                    date_column = 'trade_date' if 'trade_date' in trade_cal.columns else 'date'
                    trade_cal[date_column] = pd.to_datetime(trade_cal[date_column])
                    
                    # 按日期排序
                    trade_cal = trade_cal.sort_values(date_column, ascending=False)
                    
                    # 获取小于等于今天的最近日期
                    recent_dates = trade_cal[trade_cal[date_column] <= pd.to_datetime(today)]
                    
                    # 获取最新交易日
                    if not recent_dates.empty:
                        latest_date = recent_dates.iloc[0][date_column]
                        return latest_date.strftime('%Y%m%d')
            except Exception as calendar_error:
                print(f"通过交易日历获取最新交易日失败: {calendar_error}")
                # 备用方法：通过指数数据判断最新交易日
                try:
                    # 获取上证指数最近数据
                    index_data = self.ak.stock_zh_index_daily(symbol="sh000001")
                    if not index_data.empty:
                        index_data['date'] = pd.to_datetime(index_data['date'])
                        latest_date = index_data['date'].max()
                        return latest_date.strftime('%Y%m%d')
                except Exception as index_error:
                    print(f"通过指数数据获取最新交易日失败: {index_error}")
            
            # 如果上述方法都失败，返回当前日期（工作日判断）
            now = datetime.now()
            # 如果是周末，则返回上一个周五
            if now.weekday() >= 5:  # 5=Saturday, 6=Sunday
                days_to_subtract = now.weekday() - 4  # 回退到周五
                latest_date = now - timedelta(days=days_to_subtract)
            else:
                latest_date = now
            
            return latest_date.strftime('%Y%m%d')
        except Exception as e:
            print(f"AKShare获取最新交易日失败: {e}")
            return datetime.now().strftime('%Y%m%d')
    
    def get_index_data(self, index_code, start_date, end_date):
        """获取指数历史数据"""
        if not self.available:
            return pd.DataFrame()
        
        try:
            # 格式化日期
            start_date = self.format_date(start_date)
            end_date = self.format_date(end_date)
            
            # AKShare的指数代码可能需要转换
            index_map = {
                '000001': 'sh000001',  # 上证指数
                '399001': 'sz399001',  # 深证成指
                '399006': 'sz399006',  # 创业板指
                '000016': 'sh000016',  # 上证50
                '000300': 'sh000300',  # 沪深300
                '000905': 'sh000905',  # 中证500
                '000852': 'sh000852',  # 中证1000
                '880003': 'sh000003'   # 上证B股(暂用代替平均股价指数)
            }
            
            ak_index_code = index_map.get(index_code, index_code)
            
            # 使用ak获取指数数据
            try:
                # 尝试新版接口（无日期参数）
                index_data_orig = self.ak.stock_zh_index_daily(symbol=ak_index_code)
                
                # 如果获取成功，需要手动过滤日期范围
                if not index_data_orig.empty:
                    # 转换日期列为日期类型
                    index_data_orig['date'] = pd.to_datetime(index_data_orig['date'])
                    
                    # 过滤日期范围
                    start_date_dt = pd.to_datetime(self.format_date_hyphen(start_date))
                    end_date_dt = pd.to_datetime(self.format_date_hyphen(end_date))
                    
                    # 创建日期过滤条件
                    date_mask = (
                        (index_data_orig['date'] >= start_date_dt) & 
                        (index_data_orig['date'] <= end_date_dt)
                    )
                    
                    # 应用过滤并创建新的DataFrame（复制而不是视图）
                    filtered_data = index_data_orig.loc[date_mask].copy()
                    
                    # 在复制的DataFrame上进行列重命名操作
                    if not filtered_data.empty:
                        # 重命名列以保持一致性
                        filtered_data.rename(columns={
                            'date': '日期',
                            'open': '开盘',
                            'high': '最高',
                            'low': '最低',
                            'close': '收盘',
                            'volume': '成交量'
                        }, inplace=True)
                    
                    return filtered_data
            except TypeError:
                # 如果报TypeError异常，可能是旧版接口（接受日期参数）
                try:
                    index_data = self.ak.stock_zh_index_daily(
                        symbol=ak_index_code, 
                        start_date=start_date, 
                        end_date=end_date
                    )
                    
                    # 创建一个新的DataFrame（复制而不是视图）
                    if not index_data.empty:
                        index_data = index_data.copy()
                        # 重命名列以保持一致性
                        index_data.rename(columns={
                            'date': '日期',
                            'open': '开盘',
                            'high': '最高',
                            'low': '最低',
                            'close': '收盘',
                            'volume': '成交量'
                        }, inplace=True)
                    
                    return index_data
                except Exception as e:
                    print(f"尝试旧版接口也失败: {e}")
                    # 尝试获取所有历史数据并过滤日期
                    index_data_all = self.ak.stock_zh_index_daily(symbol=ak_index_code)
                    if not index_data_all.empty:
                        # 转换日期列为日期类型
                        index_data_all['date'] = pd.to_datetime(index_data_all['date'])
                        
                        # 过滤日期范围并创建新的DataFrame
                        start_date_dt = pd.to_datetime(self.format_date_hyphen(start_date))
                        end_date_dt = pd.to_datetime(self.format_date_hyphen(end_date))
                        
                        # 创建日期过滤条件
                        date_mask = (
                            (index_data_all['date'] >= start_date_dt) & 
                            (index_data_all['date'] <= end_date_dt)
                        )
                        
                        # 应用过滤并创建新的DataFrame
                        filtered_data = index_data_all.loc[date_mask].copy()
                        
                        # 在复制的DataFrame上进行列重命名操作
                        if not filtered_data.empty:
                            filtered_data.rename(columns={
                                'date': '日期',
                                'open': '开盘',
                                'high': '最高',
                                'low': '最低',
                                'close': '收盘',
                                'volume': '成交量'
                            }, inplace=True)
                        
                        return filtered_data
            
            # 如果上述所有方法都失败，返回空的DataFrame
            return pd.DataFrame()
        except Exception as e:
            print(f"AKShare获取指数数据失败: {e}")
            return pd.DataFrame()
    
    def get_stock_list(self, date=None):
        """获取股票列表"""
        if not self.available:
            return pd.DataFrame()
        
        try:
            # 使用ak获取股票列表
            stock_list = self.ak.stock_info_a_code_name()
            return stock_list
        except Exception as e:
            print(f"AKShare获取股票列表失败: {e}")
            return pd.DataFrame()
    
    def get_stock_data(self, stock_code, start_date, end_date):
        """获取股票历史数据"""
        if not self.available:
            return pd.DataFrame()
        
        try:
            # 格式化日期
            start_date = self.format_date_hyphen(start_date)
            end_date = self.format_date_hyphen(end_date)
            
            # 使用ak获取股票数据
            stock_data = self.ak.stock_zh_a_hist(symbol=stock_code, start_date=start_date, end_date=end_date, adjust="qfq")
            
            # 重命名列以保持一致性
            if not stock_data.empty:
                stock_data.rename(columns={
                    '日期': '日期',
                    '开盘': '开盘',
                    '最高': '最高',
                    '最低': '最低',
                    '收盘': '收盘',
                    '成交量': '成交量',
                    '成交额': '成交额',
                    '振幅': '振幅',
                    '涨跌幅': '涨跌幅',
                    '涨跌额': '涨跌额',
                    '换手率': '换手率'
                }, inplace=True)
            
            return stock_data
        except Exception as e:
            print(f"AKShare获取股票数据失败: {e}")
            return pd.DataFrame()
    
    def get_market_data(self, date=None):
        """获取市场涨跌停数据"""
        if not self.available:
            return pd.DataFrame(), pd.DataFrame()
        
        try:
            # 格式化日期 (不带连字符的格式)
            date_str = self.format_date(date) if date else self.get_latest_trade_date()
            
            # 尝试不同的AK方法获取涨跌家数数据
            stock_data = pd.DataFrame()
            methods_to_try = [
                # lambda: self.ak.stock_zh_a_spot_em(),
                lambda: self.ak.stock_zh_a_spot(),
                lambda: self.ak.stock_zh_a_hist(symbol="000001", period="daily", start_date=self.format_date_hyphen(date_str), end_date=self.format_date_hyphen(date_str)).head(1)  # 只是为了检查API是否可用
            ]
            
            success = False
            for method in methods_to_try:
                try:
                    # 尝试捕获可能出现的JSON解析错误
                    try:
                        stock_data = method()
                        if not stock_data.empty:
                            success = True
                            break
                    except requests.exceptions.HTTPError as e:
                        print(f"HTTP错误: {e}")
                        continue
                    except json.JSONDecodeError as e:
                        print(f"JSON解析错误: {e}. API可能返回了非JSON响应")
                        continue
                    except ValueError as e:
                        if "Can not decode value starting with character '<'" in str(e):
                            print(f"API返回了HTML而不是JSON: {e}")
                        else:
                            print(f"值错误: {e}")
                        continue
                    except Exception as e:
                        print(f"尝试获取股票数据时出现未知错误: {e}")
                        continue
                    
                except Exception as e:
                    print(f"尝试获取股票数据的方法失败: {e}")
                    continue
            
            if not success:
                print("所有获取股票数据的方法都失败，尝试使用备用方法")
                try:
                    # 尝试使用ak.stock_individual_info_em更基本的函数
                    stock_data = self.ak.stock_individual_info_em(symbol="000001")
                    if not stock_data.empty:
                        # 创建一个简单的DataFrame以提供基本信息
                        stock_data = pd.DataFrame({
                            '代码': ['000001'],
                            '名称': ['平安银行'],
                            '涨跌幅': [0.0]
                        })
                        print("使用模拟数据表示市场状态")
                except Exception as e:
                    print(f"备用方法也失败: {e}")
                    stock_data = pd.DataFrame()
            
            # 尝试不同的AK方法获取涨停数据
            limit_up_data = pd.DataFrame()
            limit_methods_to_try = [
                lambda: self.ak.stock_zt_pool_em(date=date_str),
                lambda: self.ak.stock_zt_pool_strong_em(date=date_str),
                lambda: self.ak.stock_zt_pool_zbgc_em(date=date_str),
                lambda: self.ak.stock_zt_pool_dtgc_em(date=date_str),
                lambda: self.ak.stock_zt_pool_previous_em(date=date_str)
            ]
            
            success = False
            for method in limit_methods_to_try:
                try:
                    # 同样，处理可能的JSON解析错误
                    try:
                        limit_up_data = method()
                        if not limit_up_data.empty:
                            # 添加炸板和连板数信息（如果没有）
                            if '炸板' not in limit_up_data.columns:
                                limit_up_data['炸板'] = False
                            if '连板数' not in limit_up_data.columns and '连板天数' not in limit_up_data.columns:
                                limit_up_data['连板数'] = 1
                            success = True
                            break
                    except json.JSONDecodeError as e:
                        print(f"涨停数据JSON解析错误: {e}")
                        continue
                    except ValueError as e:
                        if "Can not decode value starting with character '<'" in str(e):
                            print(f"涨停数据API返回了HTML而不是JSON: {e}")
                        else:
                            print(f"涨停数据值错误: {e}")
                        continue
                    except Exception as e:
                        print(f"获取涨停数据时出现未知错误: {e}")
                        continue
                except Exception as e:
                    print(f"尝试获取涨停数据的方法失败: {e}")
                    continue
            
            if not success and not stock_data.empty:
                print("所有获取涨停数据的方法都失败，创建一个空的涨停数据结构")
                # 创建一个空的涨停数据框架，具有预期的列
                limit_up_data = pd.DataFrame(columns=['代码', '名称', '涨跌幅', '炸板', '连板数'])
            
            return stock_data, limit_up_data
        except Exception as e:
            print(f"AKShare获取市场数据失败: {e}")
            # 考虑到这个错误可能是临时的，提供一个有意义的错误消息
            if "Can not decode value starting with character '<'" in str(e):
                print("检测到HTML响应而不是JSON数据，这可能是由于以下原因：")
                print("1. 网络连接问题或代理设置")
                print("2. 数据源服务器暂时不可用")
                print("3. API接口已变更")
                print("建议稍后重试或检查网络设置")
            return pd.DataFrame(), pd.DataFrame()
    
    def get_fund_flow_data(self, date):
        """获取指定日期的资金流向数据，包括行业资金流和北向资金"""
        if not self.available:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            
        try:
            # 格式化日期
            date_str = self.format_date(date)
            date_hyphen = self.format_date_hyphen(date)
            
            # 获取行业资金流向数据
            try:
                # 尝试使用AKShare获取行业板块资金流向
                sector_fund_flow = self.ak.stock_sector_fund_flow_rank(indicator="今日")
                # 如果成功获取到数据
                if not sector_fund_flow.empty:
                    # 按净额排序
                    if '净额' in sector_fund_flow.columns:
                        sector_fund_flow = sector_fund_flow.sort_values('净额', ascending=False).reset_index(drop=True)
                else:
                    sector_fund_flow = pd.DataFrame()
            except Exception as e:
                print(f"获取行业资金流向数据失败: {e}")
                sector_fund_flow = pd.DataFrame()
            
            # 获取北向资金数据
            try:
                # 尝试直接获取北向资金数据
                north_fund_flow = self.ak.stock_hsgt_fund_flow_summary_em()
                if north_fund_flow.empty:
                    # 如果获取为空，尝试创建简单的数据结构
                    north_fund_data = {
                        '日期': [date_hyphen],
                        '沪股通(亿)': [0],
                        '深股通(亿)': [0],
                        '北向资金(亿)': [0]
                    }
                    north_fund_flow = pd.DataFrame(north_fund_data)
            except Exception as e:
                print(f"获取北向资金数据失败: {e}")
                north_fund_data = {
                    '日期': [date_hyphen],
                    '沪股通(亿)': [0],
                    '深股通(亿)': [0],
                    '北向资金(亿)': [0]
                }
                north_fund_flow = pd.DataFrame(north_fund_data)
            
            # 获取融资融券数据
            try:
                margin_data = self.ak.stock_margin_underlying_info_szse(date=date_hyphen)
                if margin_data.empty:
                    margin_data = pd.DataFrame()
            except Exception as e:
                print(f"获取融资融券数据失败: {e}")
                margin_data = pd.DataFrame()
            
            return sector_fund_flow, north_fund_flow, margin_data
        except Exception as e:
            print(f"AKShare获取资金流向数据失败: {e}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


# BaoStock数据API实现
class BaoStockAPI(StockDataAPI):
    """基于BaoStock的数据API实现"""
    
    def __init__(self):
        """初始化BaoStock API"""
        try:
            import baostock as bs
            self.bs = bs
            self.available = True
            # 登录系统
            self.login()
        except ImportError:
            print("BaoStock未安装或无法导入，部分功能可能不可用")
            self.available = False
    
    def login(self):
        """登录BaoStock系统"""
        if self.available:
            lg = self.bs.login()
            if lg.error_code != '0':
                print(f'BaoStock登录失败: {lg.error_msg}')
                self.available = False
            else:
                print("BaoStock登录成功")
    
    def logout(self):
        """登出BaoStock系统"""
        if hasattr(self, 'bs') and self.available:
            try:
                self.bs.logout()
                print("BaoStock登出成功")
            except Exception as e:
                print(f"BaoStock登出异常: {e}")
    
    def __del__(self):
        """析构函数，确保登出"""
        self.logout()
    
    def get_index_data(self, index_code, start_date, end_date):
        """获取指数历史数据"""
        if not self.available:
            return pd.DataFrame()
        
        try:
            # 格式化日期
            start_date = self.format_date_hyphen(start_date)
            end_date = self.format_date_hyphen(end_date)
            
            # BaoStock的指数代码需要转换
            index_map = {
                '000001': 'sh.000001',  # 上证指数
                '399001': 'sz.399001',  # 深证成指
                '399006': 'sz.399006',  # 创业板指
                '000016': 'sh.000016',  # 上证50
                '000300': 'sh.000300',  # 沪深300
                '000905': 'sh.000905',  # 中证500
                '000852': 'sh.000852',  # 中证1000
                '880003': 'sh.000003'   # 上证B股(暂用代替平均股价指数)
            }
            
            bs_index_code = index_map.get(index_code, 'sh.' + index_code if index_code.startswith('6') else 'sz.' + index_code)
            
            # 使用BaoStock获取指数数据
            try:
                # 使用V0.8.8新增的get_data方法
                rs = self.bs.query_history_k_data_plus(
                    bs_index_code,
                    "date,open,high,low,close,volume,amount",
                    start_date=start_date,
                    end_date=end_date,
                    frequency="d"
                )
                
                # 直接获取DataFrame (需要V0.8.8版本以上)
                try:
                    index_data = rs.get_data()
                except:
                    # 如果不支持get_data()方法，使用传统方式
                    data_list = []
                    while (rs.error_code == '0') & rs.next():
                        data_list.append(rs.get_row_data())
                    index_data = pd.DataFrame(data_list, columns=rs.fields)
            except Exception as e:
                print(f"使用BaoStock get_data()方法失败: {e}, 尝试传统方式")
                # 使用传统方式
                rs = self.bs.query_history_k_data_plus(
                    bs_index_code,
                    "date,open,high,low,close,volume,amount",
                    start_date=start_date,
                    end_date=end_date,
                    frequency="d"
                )
                data_list = []
                while (rs.error_code == '0') & rs.next():
                    data_list.append(rs.get_row_data())
                index_data = pd.DataFrame(data_list, columns=rs.fields)
            
            # 类型转换
            if not index_data.empty:
                for field in ['open', 'high', 'low', 'close', 'volume', 'amount']:
                    if field in index_data.columns:
                        index_data[field] = index_data[field].astype(float)
                
                # 重命名列以保持一致性
                index_data.rename(columns={
                    'date': '日期',
                    'open': '开盘',
                    'high': '最高',
                    'low': '最低',
                    'close': '收盘',
                    'volume': '成交量',
                    'amount': '成交额'
                }, inplace=True)
            
            return index_data
        except Exception as e:
            print(f"BaoStock获取指数数据失败: {e}")
            return pd.DataFrame()
            
    def get_latest_trade_date(self):
        """获取最新交易日期
        
        返回:
            最新交易日期，格式为YYYYMMDD
        """
        if not self.available:
            return datetime.now().strftime('%Y%m%d')
        
        try:
            # 获取当前日期
            today = datetime.now()
            # 获取从当前日期往前30天的交易日历
            start_date = (today - timedelta(days=30)).strftime('%Y-%m-%d')
            end_date = today.strftime('%Y-%m-%d')
            
            # 查询交易日
            rs = self.bs.query_trade_dates(start_date=start_date, end_date=end_date)
            
            # 尝试使用get_data()方法
            try:
                trade_dates = rs.get_data()
            except:
                # 如果不支持get_data()方法，使用传统方式
                data_list = []
                while (rs.error_code == '0') & rs.next():
                    data_list.append(rs.get_row_data())
                trade_dates = pd.DataFrame(data_list, columns=rs.fields)
            
            # 筛选出交易日
            if not trade_dates.empty and 'is_trading_day' in trade_dates.columns:
                trade_dates = trade_dates[trade_dates['is_trading_day'] == '1']
                trade_dates['calendar_date'] = pd.to_datetime(trade_dates['calendar_date'])
                
                # 按日期排序并获取最新交易日
                if not trade_dates.empty:
                    trade_dates = trade_dates.sort_values('calendar_date', ascending=False)
                    latest_date = trade_dates.iloc[0]['calendar_date']
                    return latest_date.strftime('%Y%m%d')
            
            # 如果查询失败，返回当前日期（工作日判断）
            now = datetime.now()
            # 如果是周末，则返回上一个周五
            if now.weekday() >= 5:  # 5=Saturday, 6=Sunday
                days_to_subtract = now.weekday() - 4  # 回退到周五
                latest_date = now - timedelta(days=days_to_subtract)
            else:
                latest_date = now
            
            return latest_date.strftime('%Y%m%d')
        except Exception as e:
            print(f"BaoStock获取最新交易日失败: {e}")
            return datetime.now().strftime('%Y%m%d')
    
    def get_stock_list(self, date=None):
        """获取股票列表"""
        if not self.available:
            return pd.DataFrame()
        
        try:
            # 使用BaoStock获取股票列表
            rs = self.bs.query_all_stock(day=self.format_date_hyphen(date) if date else "")
            
            # 尝试使用get_data()方法
            try:
                stock_list = rs.get_data()
            except:
                # 如果不支持get_data()方法，使用传统方式
                data_list = []
                while (rs.error_code == '0') & rs.next():
                    data_list.append(rs.get_row_data())
                stock_list = pd.DataFrame(data_list, columns=rs.fields)
            
            # 获取股票名称
            if not stock_list.empty:
                # 添加股票名称
                stock_names = []
                for code in stock_list['code']:
                    rs = self.bs.query_stock_basic(code=code)
                    try:
                        basic_info = rs.get_data()
                        if not basic_info.empty:
                            stock_names.append(basic_info.iloc[0]['code_name'])
                        else:
                            stock_names.append("")
                    except:
                        # 传统方式
                        if (rs.error_code == '0') and rs.next():
                            stock_names.append(rs.get_row_data()[1])  # 股票名称在第2列
                        else:
                            stock_names.append("")
                
                stock_list['code_name'] = stock_names
            
            return stock_list
        except Exception as e:
            print(f"BaoStock获取股票列表失败: {e}")
            return pd.DataFrame()
    
    def get_stock_data(self, stock_code, start_date, end_date):
        """获取股票历史数据"""
        if not self.available:
            return pd.DataFrame()
        
        try:
            # 格式化日期
            start_date = self.format_date_hyphen(start_date)
            end_date = self.format_date_hyphen(end_date)
            
            # BaoStock的股票代码需要转换
            bs_stock_code = 'sh.' + stock_code if stock_code.startswith('6') else 'sz.' + stock_code
            
            # 使用BaoStock获取股票数据
            rs = self.bs.query_history_k_data_plus(
                bs_stock_code,
                "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="2"  # 前复权
            )
            
            # 尝试使用get_data()方法
            try:
                stock_data = rs.get_data()
            except:
                # 如果不支持get_data()方法，使用传统方式
                data_list = []
                while (rs.error_code == '0') & rs.next():
                    data_list.append(rs.get_row_data())
                stock_data = pd.DataFrame(data_list, columns=rs.fields)
            
            # 类型转换
            if not stock_data.empty:
                for field in ['open', 'high', 'low', 'close', 'volume', 'amount', 'turn', 'pctChg']:
                    if field in stock_data.columns:
                        stock_data[field] = stock_data[field].astype(float)
                
                # 重命名列以保持一致性
                stock_data.rename(columns={
                    'date': '日期',
                    'open': '开盘',
                    'high': '最高',
                    'low': '最低',
                    'close': '收盘',
                    'volume': '成交量',
                    'amount': '成交额',
                    'turn': '换手率',
                    'pctChg': '涨跌幅'
                }, inplace=True)
            
            return stock_data
        except Exception as e:
            print(f"BaoStock获取股票数据失败: {e}")
            return pd.DataFrame()
    
    def get_market_data(self, date):
        """获取指定日期的市场数据，返回股票涨跌幅数据和涨停数据"""
        if not self.available:
            return pd.DataFrame(), pd.DataFrame()
            
        try:
            # 格式化日期
            date_str = self.format_date(date)
            
            # 获取所有股票代码
            rs = self.bs.query_all_stock(day=date_str)
            if rs.error_code != '0':
                print(f"获取股票代码失败: {rs.error_msg}")
                return pd.DataFrame(), pd.DataFrame()
                
            # 尝试使用get_data获取
            try:
                stock_data = self.bs.get_data(rs)
                if stock_data is not None and not stock_data.empty:
                    # 创建数据的副本以避免SettingWithCopyWarning
                    stock_data = stock_data.copy()
                else:
                    raise ValueError("通过get_data获取股票数据失败")
            except (ValueError, AttributeError) as e:
                print(f"尝试get_data方法失败: {e}，使用传统方法获取")
                # 传统获取方法：逐行读取
                stock_data = pd.DataFrame()
                while (rs.error_code == '0') & rs.next():
                    data_list = rs.get_row_data()
                    stock_data = pd.concat([stock_data, pd.DataFrame([data_list])], ignore_index=True)
            
            # 如果没有获取到数据，则返回空DataFrame
            if stock_data.empty:
                return pd.DataFrame(), pd.DataFrame()
            
            # 给列命名
            if stock_data.shape[1] >= 2:
                stock_data.columns = ['code', 'tradeStatus'] if len(stock_data.columns) == 2 else ['code', 'code_name', 'ipoDate', 'outDate', 'type', 'status']
            
            # 获取股票详细数据
            all_stock_data = pd.DataFrame()
            
            # 批量处理，每次处理100支股票
            batch_size = 100
            stock_codes = stock_data['code'].tolist()
            
            for i in range(0, len(stock_codes), batch_size):
                batch_codes = stock_codes[i:i+batch_size]
                batch_data = pd.DataFrame()
                
                for code in batch_codes:
                    # 查询交易详情
                    rs_k = self.bs.query_history_k_data_plus(
                        code,
                        "date,open,high,low,close,volume,amount,pctChg",
                        start_date=date_str,
                        end_date=date_str,
                        frequency="d"
                    )
                    
                    if rs_k.error_code != '0':
                        print(f"获取股票 {code} 数据失败: {rs_k.error_msg}")
                        continue
                    
                    # 尝试使用get_data获取
                    try:
                        df = self.bs.get_data(rs_k)
                        if df is not None and not df.empty:
                            batch_data = pd.concat([batch_data, df], ignore_index=True)
                    except (ValueError, AttributeError):
                        # 传统获取方法：逐行读取
                        while (rs_k.error_code == '0') & rs_k.next():
                            data_list = rs_k.get_row_data()
                            batch_data = pd.concat([batch_data, pd.DataFrame([data_list])], ignore_index=True)
                
                all_stock_data = pd.concat([all_stock_data, batch_data], ignore_index=True)
            
            # 如果没有获取到数据，则返回空DataFrame
            if all_stock_data.empty:
                return pd.DataFrame(), pd.DataFrame()
            
            # 转换数据类型
            # 创建数据的副本以避免SettingWithCopyWarning
            all_stock_data = all_stock_data.copy()
            
            # 转换数值列
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 'pctChg']
            for col in numeric_cols:
                if col in all_stock_data.columns:
                    all_stock_data[col] = pd.to_numeric(all_stock_data[col], errors='coerce')
            
            # 添加股票名称
            if 'code_name' in stock_data.columns:
                name_dict = dict(zip(stock_data['code'], stock_data['code_name']))
                all_stock_data['code_name'] = all_stock_data['code'].map(name_dict)
            else:
                # 获取股票名称
                all_stock_names = pd.DataFrame()
                for code in stock_codes:
                    rs_info = self.bs.query_stock_basic(code=code)
                    if rs_info.error_code == '0':
                        try:
                            info_df = self.bs.get_data(rs_info)
                            if info_df is not None and not info_df.empty:
                                all_stock_names = pd.concat([all_stock_names, info_df], ignore_index=True)
                        except (ValueError, AttributeError):
                            while (rs_info.error_code == '0') & rs_info.next():
                                info_data = rs_info.get_row_data()
                                all_stock_names = pd.concat([all_stock_names, pd.DataFrame([info_data])], ignore_index=True)
                
                if not all_stock_names.empty and 'code_name' in all_stock_names.columns:
                    name_dict = dict(zip(all_stock_names['code'], all_stock_names['code_name']))
                    all_stock_data['code_name'] = all_stock_data['code'].map(name_dict)
                else:
                    all_stock_data['code_name'] = "未知"
            
            # 识别涨停板股票（涨幅大于9.5%为简化处理）
            if 'pctChg' in all_stock_data.columns:
                # 创建涨停数据的副本
                limit_up_data = all_stock_data[all_stock_data['pctChg'] >= 9.5].copy()
                
                # 标准化返回的列名
                if not limit_up_data.empty:
                    limit_up_data = limit_up_data.rename(columns={
                        'code': '代码',
                        'code_name': '名称',
                        'pctChg': '涨跌幅'
                    })
            else:
                limit_up_data = pd.DataFrame()
            
            # 标准化返回的股票数据列名
            all_stock_data = all_stock_data.rename(columns={
                'code': '代码',
                'code_name': '名称',
                'open': '开盘',
                'high': '最高',
                'low': '最低',
                'close': '收盘',
                'volume': '成交量',
                'amount': '成交额',
                'pctChg': '涨跌幅'
            })
            
            return all_stock_data, limit_up_data
        except Exception as e:
            print(f"BaoStock获取市场数据失败: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    def get_fund_flow_data(self, date):
        """获取指定日期的资金流向数据，包括行业资金流和北向资金
        
        注意：BaoStock不直接提供全面的资金流向数据，这里我们尝试构建相关数据
        """
        if not self.available:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            
        try:
            # 格式化日期
            date_str = self.format_date(date)
            date_hyphen = self.format_date_hyphen(date)
            
            # 获取行业分类数据
            rs_industry = self.bs.query_stock_industry()
            
            # 传统获取方法：逐行读取
            industry_data = pd.DataFrame()
            while (rs_industry.error_code == '0') & rs_industry.next():
                data_list = rs_industry.get_row_data()
                industry_data = pd.concat([industry_data, pd.DataFrame([data_list])], ignore_index=True)
            
            # 如果没有获取到数据，返回空DataFrame
            if industry_data.empty:
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            
            # 创建数据的副本以避免SettingWithCopyWarning
            industry_data = industry_data.copy()
            
            # 提取行业分类，并计算每个行业的股票数量
            if 'industry' in industry_data.columns:
                industry_counts = industry_data['industry'].value_counts().reset_index()
                industry_counts.columns = ['板块名称', '股票数量']
                
                # 创建虚拟的行业资金流向数据
                # 由于BaoStock不直接提供行业资金数据，这里我们基于行业股票数量创建一个模拟数据
                sector_fund_flow = pd.DataFrame()
                sector_fund_flow['板块名称'] = industry_counts['板块名称']
                sector_fund_flow['股票数量'] = industry_counts['股票数量']
                
                # 添加虚拟的资金流入数据（这里只是为了符合API接口，实际应用中应替换为真实数据）
                np.random.seed(int(date_str.replace('-', '')))  # 使用日期作为随机种子，保证同一天生成相同的数据
                
                # 添加虚拟数据列
                n = len(sector_fund_flow)
                sector_fund_flow['流入资金'] = np.random.randn(n) * 10000
                sector_fund_flow['流入量'] = sector_fund_flow['流入资金'] / np.random.randint(100, 200, size=n)
                sector_fund_flow['流出资金'] = np.random.randn(n) * 10000
                sector_fund_flow['流出量'] = sector_fund_flow['流出资金'] / np.random.randint(100, 200, size=n)
                sector_fund_flow['净额'] = sector_fund_flow['流入资金'] - sector_fund_flow['流出资金']
                
                # 计算净占比前确保净额总和不为零
                net_sum = sector_fund_flow['净额'].abs().sum()
                if net_sum > 0:
                    sector_fund_flow['净占比'] = sector_fund_flow['净额'] / net_sum * 100
                else:
                    sector_fund_flow['净占比'] = 0
                
                # 按净额排序
                sector_fund_flow = sector_fund_flow.sort_values('净额', ascending=False).reset_index(drop=True)
                
                # 添加提示信息
                print("注意: BaoStock不直接提供行业资金流数据，此处展示的是基于行业股票数量的模拟数据")
            else:
                sector_fund_flow = pd.DataFrame()
            
            # 尝试获取北向资金数据 - 通过查询沪深300成分股
            rs_hs300 = self.bs.query_hs300_stocks()
            
            # 传统获取方法：逐行读取
            hs300_stocks = pd.DataFrame()
            while (rs_hs300.error_code == '0') & rs_hs300.next():
                data_list = rs_hs300.get_row_data()
                hs300_stocks = pd.concat([hs300_stocks, pd.DataFrame([data_list])], ignore_index=True)
            
            # 如果成功获取沪深300成分股，构建模拟的北向资金数据
            if not hs300_stocks.empty:
                # 随机生成资金流数据
                np.random.seed(int(date_str.replace('-', '')))  # 使用日期作为随机种子
                
                # 创建一个新的DataFrame而不是修改现有的
                north_fund_data = {
                    '日期': [date_hyphen],
                    '沪股通(亿)': [round(np.random.randn() * 10, 2)],
                    '深股通(亿)': [round(np.random.randn() * 8, 2)]
                }
                
                # 计算北向资金总额
                north_fund_data['北向资金(亿)'] = [round(north_fund_data['沪股通(亿)'][0] + north_fund_data['深股通(亿)'][0], 2)]
                
                # 创建DataFrame
                north_fund_flow = pd.DataFrame(north_fund_data)
                
                # 添加提示信息
                print("注意: BaoStock不直接提供北向资金数据，此处展示的是模拟数据")
            else:
                north_fund_flow = pd.DataFrame()
            
            # BaoStock不提供融资融券数据
            margin_data = pd.DataFrame()
            print("注意: BaoStock不提供融资融券数据")
            
            return sector_fund_flow, north_fund_flow, margin_data
        except Exception as e:
            print(f"BaoStock获取资金流向数据失败: {e}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


# 创建一个数据源工厂类
class StockDataFactory:
    """股票数据源工厂类"""
    
    @staticmethod
    def create_api(api_type="akshare"):
        """创建数据API实例
        
        参数:
            api_type: 数据源类型，可选值: "akshare", "baostock", "auto"
        
        返回:
            StockDataAPI实例
        """
        if api_type == "auto":
            # 尝试创建AKShare API
            ak_api = AKShareAPI()
            # 尝试创建BaoStock API
            bs_api = BaoStockAPI()
            
            # 返回可用性较高的API
            if ak_api.available and bs_api.available:
                # 创建混合API
                return DualSourceAPI(ak_api, bs_api)
            elif ak_api.available:
                return ak_api
            elif bs_api.available:
                return bs_api
            else:
                # 如果都不可用，返回一个空实现
                print("警告：所有数据源都不可用！")
                return EmptyAPI()
        
        elif api_type == "akshare":
            return AKShareAPI()
        
        elif api_type == "baostock":
            return BaoStockAPI()
        
        elif api_type == "dual":
            # 创建并返回混合API
            ak_api = AKShareAPI()
            bs_api = BaoStockAPI()
            if ak_api.available or bs_api.available:
                return DualSourceAPI(ak_api, bs_api)
            else:
                print("警告：所有数据源都不可用！")
                return EmptyAPI()
        
        else:
            raise ValueError(f"不支持的数据源类型: {api_type}")


# 新增一个空API实现
class EmptyAPI(StockDataAPI):
    """空API实现，所有方法返回空数据"""
    
    def __init__(self):
        self.available = False
    
    def get_index_data(self, index_code, start_date, end_date):
        print("警告: 无可用数据源，返回空数据")
        return pd.DataFrame()
    
    def get_stock_list(self, date=None):
        print("警告: 无可用数据源，返回空数据")
        return pd.DataFrame()
    
    def get_stock_data(self, stock_code, start_date, end_date):
        print("警告: 无可用数据源，返回空数据")
        return pd.DataFrame()
    
    def get_market_data(self, date=None):
        print("警告: 无可用数据源，返回空数据")
        return pd.DataFrame(), pd.DataFrame()
    
    def get_fund_flow_data(self, date=None):
        print("警告: 无可用数据源，返回空数据")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    def get_latest_trade_date(self):
        print("警告: 无可用数据源，使用当前日期")
        return datetime.now().strftime('%Y%m%d')


# 新增一个双源混合API实现
class DualSourceAPI(StockDataAPI):
    """双数据源API类，可以在主数据源失败时自动切换到备用数据源"""
    def __init__(self, primary_api, fallback_api):
        self.primary_api = primary_api  # 主数据源
        self.fallback_api = fallback_api  # 备用数据源
        self.primary_error_count = 0
        self.available = True
        print(f"已初始化双数据源API：主源-{type(primary_api).__name__}，备用源-{type(fallback_api).__name__}")
    
    def _safe_call(self, primary_method, fallback_method, *args, **kwargs):
        """安全调用方法，如果主源失败则尝试备用源"""
        try:
            try:
                result = primary_method(*args, **kwargs)
                # 检查DataFrame类型结果是否为空
                if isinstance(result, pd.DataFrame) and result.empty:
                    raise ValueError("主数据源返回空DataFrame")
                # 检查多返回值的情况（元组）
                elif isinstance(result, tuple) and all(isinstance(r, pd.DataFrame) and r.empty for r in result):
                    raise ValueError("主数据源返回空DataFrame元组")
                
                # 判断结果是否为DataFrame或其元组，如果是则创建副本以避免SettingWithCopyWarning
                if isinstance(result, pd.DataFrame):
                    return result.copy()
                elif isinstance(result, tuple) and all(isinstance(r, pd.DataFrame) for r in result):
                    return tuple(r.copy() if not r.empty else r for r in result)
                
                return result
            except json.JSONDecodeError as e:
                # 特殊处理JSON解析错误
                print(f"主数据源JSON解析错误: {e}，尝试使用备用数据源")
                self.primary_error_count += 1
                raise  # 重新抛出异常，将由外部catch块处理
            except ValueError as e:
                # 特殊处理HTML响应错误
                if "Can not decode value starting with character '<'" in str(e):
                    print(f"主数据源返回了HTML而不是JSON: {e}，尝试使用备用数据源")
                else:
                    print(f"主数据源值错误: {e}，尝试使用备用数据源")
                self.primary_error_count += 1
                raise  # 重新抛出异常，将由外部catch块处理
            except requests.exceptions.HTTPError as e:
                # 特殊处理HTTP错误
                print(f"主数据源HTTP错误: {e}，尝试使用备用数据源")
                self.primary_error_count += 1
                raise  # 重新抛出异常，将由外部catch块处理
        except Exception as e:
            print(f"主数据源调用失败: {str(e)[:100]}{'...' if len(str(e)) > 100 else ''}，尝试使用备用数据源")
            self.primary_error_count += 1
            
            # 如果错误次数超过阈值，可以考虑自动切换主源
            if self.primary_error_count > 10:
                print(f"警告: 主数据源已累计失败{self.primary_error_count}次，建议考虑切换主源")
            
            try:
                result = fallback_method(*args, **kwargs)
                # 判断结果是否为DataFrame或其元组，如果是则创建副本以避免SettingWithCopyWarning
                if isinstance(result, pd.DataFrame):
                    return result.copy()
                elif isinstance(result, tuple) and all(isinstance(r, pd.DataFrame) for r in result):
                    return tuple(r.copy() if not r.empty else r for r in result)
                
                return result
            except Exception as e2:
                error_msg = str(e2)[:100] + ('...' if len(str(e2)) > 100 else '')
                print(f"备用数据源也调用失败: {error_msg}")
                
                # 提供更详细的错误信息
                if "Can not decode value starting with character '<'" in str(e2):
                    print("检测到HTML响应而不是JSON数据，这可能是由于以下原因：")
                    print("1. 网络连接问题或代理设置")
                    print("2. 数据源服务器暂时不可用")
                    print("3. API接口已变更")
                    print("建议稍后重试或检查网络设置")
                
                # 根据返回类型，返回空的结果
                if fallback_method.__annotations__.get('return') == pd.DataFrame:
                    return pd.DataFrame()
                elif 'tuple' in str(fallback_method.__annotations__.get('return', '')):
                    # 尝试通过观察返回类型推断需要返回几个DataFrame
                    if '3' in str(fallback_method.__annotations__.get('return', '')):
                        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
                    else:
                        return pd.DataFrame(), pd.DataFrame()
                else:
                    return None
    
    def get_index_data(self, index_code, start_date, end_date):
        """获取指数历史数据"""
        return self._safe_call(
            self.primary_api.get_index_data,
            self.fallback_api.get_index_data,
            index_code, start_date, end_date
        )
    
    def get_latest_trade_date(self):
        """获取最新交易日期"""
        return self._safe_call(
            self.primary_api.get_latest_trade_date,
            self.fallback_api.get_latest_trade_date
        )
    
    def get_stock_list(self):
        """获取股票列表"""
        return self._safe_call(
            self.primary_api.get_stock_list,
            self.fallback_api.get_stock_list
        )
    
    def get_stock_data(self, stock_code, start_date, end_date):
        """获取股票历史数据"""
        return self._safe_call(
            self.primary_api.get_stock_data,
            self.fallback_api.get_stock_data,
            stock_code, start_date, end_date
        )
    
    def get_market_data(self, date):
        """获取市场数据"""
        return self._safe_call(
            self.primary_api.get_market_data,
            self.fallback_api.get_market_data,
            date
        )
    
    def get_fund_flow_data(self, date):
        """获取资金流向数据"""
        return self._safe_call(
            self.primary_api.get_fund_flow_data,
            self.fallback_api.get_fund_flow_data,
            date
        ) 