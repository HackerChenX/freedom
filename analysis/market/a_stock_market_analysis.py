#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from api.stock_data_api import StockDataFactory
# 从indicators模块导入所有技术指标函数
from enums.indicators import *
import efinance as ef
from db.clickhouse_db import get_clickhouse_db, get_default_config
from enums.kline_period import KlinePeriod

# 获取默认配置
config = get_default_config()
config['password'] = '223826'
ch_db = get_clickhouse_db(config=config)

# 中国市值前三的股票
A_STOCK_LIST = {"工商银行": "sh601398", "贵州茅台": "sh600519", "中国平安": "sh601318"}
INDUSTRY_LIST = {"食品饮料": "886012", "家用电器": "886032", "医药制造": "886037", "银行": "886015", "保险": "886039",
                 "证券": "886055", "汽车制造": "886033", "有色金属": "886020", "基础化工": "886017",
                 "煤炭行业": "886010", "钢铁行业": "886009", "房地产": "886046", "建筑": "886007", "计算机设备": "886043",
                 "通信行业": "886045", "电力": "886041", "机械": "886021"}

# 沪深300
CHINA_300_MARKET = ["sh000300", "sh510300"]

# 科创板龙头
K_STOCK_LIST = {"晶晨股份": "sh688099", "安集科技": "sh688019", "中微公司": "sh688012", "容百科技": "sh688005", "乐鑫科技": "sh688018",
                "睿创微纳": "sh688002", "心脉医疗": "sh688016", "微芯生物": "sh688321", "交控科技": "sh688015"}

# 创业板龙头
C_STOCK_LIST = {"爱尔眼科": "sz300015", "汇川技术": "sz300124", "乐普医疗": "sz300003", "东方财富": "sz300059", "迈瑞医疗": "sz300760",
                "宁德时代": "sz300750", "智飞生物": "sz300122", "温氏股份": "sz300498"}


class MarketAnalyzer:
    """A股市场分析器，按照指定维度计算市场强度并提供操作建议"""
    
    def __init__(self, date=None, data_source="auto"):
        """初始化市场分析器，默认分析当天市场数据
        
        参数:
            date: 分析日期，格式为YYYYMMDD，默认为最新交易日
            data_source: 数据源，可选值为"akshare", "baostock", "auto"
        """
        # 初始化数据API
        self.api = StockDataFactory.create_api(data_source)
        
        # 检查API是否可用
        if not hasattr(self.api, 'available') or not self.api.available:
            print(f"警告: {data_source} 数据源不可用，尝试使用自动模式")
            self.api = StockDataFactory.create_api("auto")
        
        # 如果没有指定日期，获取最新交易日
        if date is None:
            self.date = self.api.get_latest_trade_date()
            print(f"未指定日期，使用最新交易日: {self.date}")
        else:
            self.date = date
        
        self.date_formatted = datetime.strptime(self.date, '%Y%m%d').strftime('%Y-%m-%d')
        
        # 分析结果存储
        self.index_trend_score = 0
        self.emotion_score = 0
        self.fund_behavior_score = 0
        self.emotion_cycle_score = 0  # 情绪周期得分
        self.market_strength = 0
        
        # 存储获取的数据
        self.sh_index_data = None
        self.gem_index_data = None
        self.avg_price_index_data = None
        self.small_cap_index_data = None
        self.sh50_index_data = None
        
        # 情绪周期数据
        self.up_down_history = None  # 存储历史涨跌家数
        self.emotion_leaders = {}  # 存储情绪标的表现
        self.broken_limit_recovery = {}  # 断板反包情况
    
    def get_index_data(self, index_code, days=30):
        """获取指数历史数据"""
        end_date = self.date
        start_date = (datetime.strptime(end_date, '%Y%m%d') - timedelta(days=days)).strftime('%Y%m%d')
        
        index_data = self.api.get_index_data(index_code, start_date, end_date)
        
        return index_data
    
    def analyze_index_trend(self):
        """分析指数趋势（权重40%）"""
        score = 0
        
        # 获取指数数据
        self.sh_index_data = self.get_index_data('000001')  # 上证指数
        self.gem_index_data = self.get_index_data('399006')  # 创业板指
        self.avg_price_index_data = self.get_index_data('880003')  # 平均股价指数
        self.small_cap_index_data = self.get_index_data('000852')  # 中证1000
        self.sh50_index_data = self.get_index_data('000016')  # 上证50
        
        # 1. 上证指数分析
        if not self.sh_index_data.empty:
            close = self.sh_index_data['收盘'].values
            volume = self.sh_index_data['成交量'].values
            
            # 计算均线
            ma5 = MA(close, 5)
            ma10 = MA(close, 10)
            ma20 = MA(close, 20)
            
            # 均线排列方向
            if not np.isnan(ma5[-1]) and not np.isnan(ma10[-1]) and not np.isnan(ma20[-1]):
                if ma5[-1] > ma10[-1] > ma20[-1]:
                    score += 10  # 多头排列
                elif ma5[-1] < ma10[-1] < ma20[-1]:
                    score -= 10  # 空头排列
            
            # MACD红绿柱变化
            dif, dea, macd = MACD(close)
            if not np.isnan(macd[-1]) and not np.isnan(macd[-2]):
                if macd[-1] > 0 and macd[-1] > macd[-2]:
                    score += 5  # MACD红柱且放大
                elif macd[-1] < 0 and macd[-1] < macd[-2]:
                    score -= 5  # MACD绿柱且放大
            
            # 量能对比
            vol_ma5 = MA(volume, 5)
            if not np.isnan(vol_ma5[-1]) and vol_ma5[-1] > 0:
                vol_ratio = volume[-1] / vol_ma5[-1]
                if vol_ratio > 1.2:
                    score += 5  # 放量
                elif vol_ratio < 0.8:
                    score -= 3  # 缩量
        
        # 2. 创业板指分析
        if not self.gem_index_data.empty:
            close = self.gem_index_data['收盘'].values
            high = self.gem_index_data['最高'].values
            low = self.gem_index_data['最低'].values
            
            # RSI超买超卖
            rsi = RSI(close)
            if not np.isnan(rsi[-1]):
                if rsi[-1] > 70:
                    score -= 3  # 超买
                elif rsi[-1] < 30:
                    score += 3  # 超卖
            
            # 波动率(ATR14)
            atr = ATR(close, high, low, 14)
            ma14 = MA(close, 14)
            if not np.isnan(atr[-1]) and not np.isnan(ma14[-1]) and ma14[-1] > 0:
                atr_ratio = atr[-1] / ma14[-1]
                if atr_ratio > 0.03:  # 波动率大
                    score += 3
            
            # 突破关键压力/支撑位判断
            # 这里简化为与前期高点/低点的关系
            if len(high) > 20 and len(low) > 20:
                recent_high = max(high[-20:-1])
                recent_low = min(low[-20:-1])
                if close[-1] > recent_high:
                    score += 5  # 突破前期高点
                elif close[-1] < recent_low:
                    score -= 5  # 跌破前期低点
        
        # 3. 平均股价指数(880003)与三大指数背离度
        if not self.avg_price_index_data.empty and not self.sh_index_data.empty:
            avg_close = self.avg_price_index_data['收盘'].values
            sh_close = self.sh_index_data['收盘'].values
            
            # 计算背离度(简化为相对变化率之差)
            if len(avg_close) >= 5 and len(sh_close) >= 5:
                avg_change = (avg_close[-1] / avg_close[-5] - 1) * 100
                sh_change = (sh_close[-1] / sh_close[-5] - 1) * 100
                divergence = avg_change - sh_change
                
                if divergence > 2:
                    score += 5  # 平均股价强于大盘
                elif divergence < -2:
                    score -= 5  # 平均股价弱于大盘
        
        # 4. 小盘股指数(中证1000)与上证50剪刀差
        if not self.small_cap_index_data.empty and not self.sh50_index_data.empty:
            small_close = self.small_cap_index_data['收盘'].values
            sh50_close = self.sh50_index_data['收盘'].values
            
            # 计算涨跌幅差值
            if len(small_close) >= 2 and len(sh50_close) >= 2:
                small_change = (small_close[-1] / small_close[-2] - 1) * 100
                sh50_change = (sh50_close[-1] / sh50_close[-2] - 1) * 100
                scissors_diff = small_change - sh50_change
                
                if scissors_diff > 1:
                    score += 5  # 小盘股强于大盘股，市场活跃
                elif scissors_diff < -1:
                    score -= 5  # 小盘股弱于大盘股，市场谨慎
        # 归一化得分到0-100区间
        normalized_score = max(0, min(100, score + 50))
        self.index_trend_score = normalized_score
        return normalized_score
    
    def get_market_data(self):
        """获取全市场涨跌、涨停数据等"""
        return self.api.get_market_data(self.date)
    
    def analyze_emotion_indicators(self, market_data=None):
        """分析情绪指标（权重30%）"""
        score = 0
        
        # 获取市场数据
        if market_data is None:
            stock_data, limit_up_data = self.get_market_data()
        else:
            stock_data, limit_up_data = market_data
        
        if not stock_data.empty:
            # 1. 涨跌比例
            try:
                up_count = len(stock_data[stock_data['涨跌幅'] > 0])
                down_count = len(stock_data[stock_data['涨跌幅'] < 0])
                up_down_ratio = up_count / down_count if down_count > 0 else 10
                
                if up_down_ratio > 1.5:
                    score += 10  # 红盘强势
                elif up_down_ratio < 0.8:
                    score -= 10  # 绿盘强势
            except:
                print("分析涨跌比例失败")
        
        if not limit_up_data.empty:
            try:
                # 2. 涨停梯队
                # 连板股数量
                if '连板数' in limit_up_data.columns:
                    limit_up_data['连板天数'] = limit_up_data['连板数'].astype(int)
                elif '连板天数' in limit_up_data.columns:
                    pass  # 已经有连板天数列
                else:
                    limit_up_data['连板天数'] = 1  # 默认为1
                
                consecutive_boards = limit_up_data[limit_up_data['连板天数'] > 0]
                
                # 三连板及以上数量
                three_consecutive = len(consecutive_boards[consecutive_boards['连板天数'] >= 3])
                if three_consecutive >= 5:
                    score += 10  # 情绪强
                
                # 最高连板数（空间板高度）
                max_consecutive = consecutive_boards['连板天数'].max() if not consecutive_boards.empty else 0
                if max_consecutive >= 5:
                    score += 10  # 周期主升段
            except:
                print("分析涨停梯队失败")
            
            try:
                # 3. 炸板率
                if '炸板' in limit_up_data.columns:
                    limit_up_broken = limit_up_data[limit_up_data['炸板'] == True]
                    if not limit_up_broken.empty and not limit_up_data.empty:
                        broken_rate = len(limit_up_broken) / (len(limit_up_data) + len(limit_up_broken))
                        if broken_rate > 0.4:
                            score -= 15  # 炸板率高，情绪退潮
                        elif broken_rate < 0.2:
                            score += 10  # 炸板率低，情绪良好
            except:
                print("分析炸板率失败")
            
            try:
                # 4. 连板晋级率（简化计算）
                # 假设二进三成功率为50%
                if three_consecutive > 0:
                    score += 5
                
                # 高位断板股次日核按钮比例（简化，数据有限）
                score += 5
            except:
                print("分析连板晋级率失败")
        
        # 归一化得分到0-100区间
        normalized_score = max(0, min(100, score + 50))
        self.emotion_score = normalized_score
        return normalized_score
    
    def get_fund_flow_data(self):
        """获取资金流向数据"""
        return self.api.get_fund_flow_data(self.date)
    
    def analyze_fund_behavior(self):
        """分析资金行为（权重30%）"""
        score = 0
        
        # 获取资金流向数据
        sector_fund_flow, north_fund, margin_data = self.get_fund_flow_data()
        
        # 1. 主力资金流向
        if not sector_fund_flow.empty:
            try:
                # 大单净流入TOP3板块
                if '今日主力净流入-净额' in sector_fund_flow.columns:
                    flow_column = '今日主力净流入-净额'
                elif '主力净流入' in sector_fund_flow.columns:
                    flow_column = '主力净流入'
                else:
                    flow_column = sector_fund_flow.columns[1]  # 假设第二列是资金流入列
                
                sector_fund_flow_sorted = sector_fund_flow.sort_values(by=flow_column, ascending=False)
                top5_inflow = sector_fund_flow_sorted.iloc[0:5]
                
                # 确保将主力净流入值转换为数值类型
                top5_net_flow = 0.0
                for value in top5_inflow[flow_column]:
                    if isinstance(value, str):
                        try:
                            clean_value = value.replace(',', '').replace('亿', '').replace('万', '')
                            numeric_value = float(clean_value)
                            # 如果字符串中包含"亿"，乘以1e8
                            if '亿' in value:
                                numeric_value *= 1e8
                            # 如果字符串中包含"万"，乘以1e4
                            elif '万' in value:
                                numeric_value *= 1e4
                            top5_net_flow += numeric_value
                        except (ValueError, TypeError):
                            pass  # 无法解析，跳过
                    else:
                        # 如果已经是数值类型，直接累加
                        try:
                            if value is not None:
                                top5_net_flow += float(value)
                        except (ValueError, TypeError):
                            pass  # 转换失败，跳过
                
                # 判断主要板块净流入是否为正
                if top5_net_flow > 0:
                    score += 10
                else:
                    score -= 10
                
                # markdown 打印 top5_inflow，板块名、代表个股、净流入额
                markdown = "### 主力资金流向\n\n"
                markdown += "| 板块名 | 代表个股 | 净流入额 |\n"
                markdown += "| ---- | ---- | ---- |\n"
                for index, row in top5_inflow.iterrows():
                    markdown += f"| {row['板块名']} | {row['代表个股']} | {row['净流入额']} |\n"
                print(markdown)
            except Exception as e:
                print("分析主力资金流向失败:", e)
        
        # 归一化得分到0-100区间
        normalized_score = max(0, min(100, score + 50))
        self.fund_behavior_score = normalized_score
        return normalized_score
    
    def analyze_emotion_cycle(self, market_data=None, days=5):
        """分析市场情绪周期（权重20%）
        
        参数：
            market_data: 市场数据，如果为None则重新获取
            days: 分析的历史天数
        """
        score = 0
        
        # 获取市场数据
        if market_data is None:
            stock_data, limit_up_data = self.get_market_data()
        else:
            stock_data, limit_up_data = market_data
        
        # 1. 观察情绪标的表现
        try:
            if not limit_up_data.empty:
                # 获取昨日断板股的今日表现
                if hasattr(self, 'prev_broken_stocks') and self.prev_broken_stocks:
                    broken_recovered = 0
                    for stock_code in self.prev_broken_stocks:
                        if stock_code in stock_data.index:
                            change_pct = stock_data.loc[stock_code, '涨跌幅']
                            if change_pct > 5:  # 定义反包为大于5%的涨幅
                                broken_recovered += 1
                                self.broken_limit_recovery[stock_code] = change_pct
                    
                    # 计算断板反包率
                    if len(self.prev_broken_stocks) > 0:
                        recovery_rate = broken_recovered / len(self.prev_broken_stocks)
                        if recovery_rate > 0.6:
                            score += 10  # 断板高反包率，市场情绪较好
                        elif recovery_rate < 0.3:
                            score -= 5  # 断板低反包率，市场情绪较弱
                
                # 存储当日炸板股，供下次分析使用
                if '炸板' in limit_up_data.columns:
                    broken_stocks = list(limit_up_data[limit_up_data['炸板'] == True].index)
                    self.prev_broken_stocks = broken_stocks
                
                # 分析连板龙头强弱
                if '连板数' in limit_up_data.columns or '连板天数' in limit_up_data.columns:
                    if '连板数' in limit_up_data.columns:
                        limit_up_data['连板天数'] = limit_up_data['连板数'].astype(int)
                    
                    # 找出三连板及以上的股票作为情绪标的
                    emotion_leaders = limit_up_data[limit_up_data['连板天数'] >= 3]
                    if not emotion_leaders.empty:
                        # 记录这些标的，考察其未来表现
                        for idx, row in emotion_leaders.iterrows():
                            if idx not in self.emotion_leaders:
                                self.emotion_leaders[idx] = {
                                    '首次记录日': self.date,
                                    '连板天数': row['连板天数'],
                                    '后续表现': []
                                }
                        score += 5  # 有强势情绪标的
                    
                    # 追踪已有情绪标的的表现
                    for stock_code in list(self.emotion_leaders.keys()):
                        if stock_code in stock_data.index:
                            today_change = stock_data.loc[stock_code, '涨跌幅']
                            self.emotion_leaders[stock_code]['后续表现'].append({
                                '日期': self.date,
                                '涨跌幅': today_change
                            })
                            
                            # 评估情绪标的强度
                            recent_perf = [p['涨跌幅'] for p in self.emotion_leaders[stock_code]['后续表现'][-3:] if p['涨跌幅'] is not None]
                            if recent_perf and len(recent_perf) >= 2:
                                avg_perf = sum(recent_perf) / len(recent_perf)
                                if avg_perf > 5:
                                    score += 5  # 情绪标的持续强势
                                elif avg_perf < -3:
                                    score -= 5  # 情绪标的表现弱势
        except Exception as e:
            print(f"分析情绪标的表现失败: {e}")
        
        # 2. 涨跌幅大比例分析
        try:
            if not stock_data.empty and '涨跌幅' in stock_data.columns:
                # 统计涨幅超过7%和跌幅超过7%的股票数量
                big_up_count = len(stock_data[stock_data['涨跌幅'] > 7])
                big_down_count = len(stock_data[stock_data['涨跌幅'] < -7])
                
                # 计算比例
                if big_down_count > 0:
                    big_ratio = big_up_count / big_down_count
                else:
                    big_ratio = big_up_count if big_up_count > 0 else 0
                
                # 根据比例计算得分
                if big_ratio > 3:
                    score += 15  # 大涨股远多于大跌股，市场强势
                elif big_ratio > 1.5:
                    score += 8  # 大涨股多于大跌股，市场偏强
                elif big_ratio < 0.5:
                    score -= 10  # 大跌股远多于大涨股，市场弱势
                
                # 记录分析结果
                self.big_move_ratio = {
                    '大涨股数量': big_up_count,
                    '大跌股数量': big_down_count,
                    '比例': big_ratio
                }
        except Exception as e:
            print(f"分析涨跌幅大比例失败: {e}")
        
        # 3. 涨跌家数5日线分析
        try:
            # 手动获取历史涨跌家数数据（替代get_market_up_down_history）
            # 这里我们尝试从已获取的数据中计算，如果无法获取历史数据，就使用当天数据
            end_date = self.date
            start_date = (datetime.strptime(end_date, '%Y%m%d') - timedelta(days=days*2)).strftime('%Y%m%d')
            
            # 检查API是否支持get_market_up_down_history方法
            if hasattr(self.api, 'get_market_up_down_history'):
                self.up_down_history = self.api.get_market_up_down_history(start_date, end_date)
            else:
                # 如果API不支持该方法，我们创建一个简单的替代
                # 这里只能用当前数据，历史数据需要实际API支持
                print("警告: API不支持获取历史涨跌家数，使用当日数据替代")
                if self.up_down_history is None:
                    # 创建一个只有当天数据的DataFrame
                    up_count = len(stock_data[stock_data['涨跌幅'] > 0]) if not stock_data.empty else 0
                    down_count = len(stock_data[stock_data['涨跌幅'] < 0]) if not stock_data.empty else 0
                    flat_count = len(stock_data) - up_count - down_count if not stock_data.empty else 0
                    
                    self.up_down_history = pd.DataFrame({
                        '日期': [self.date_formatted],
                        '上涨家数': [up_count],
                        '下跌家数': [down_count],
                        '平盘家数': [flat_count]
                    })
                    self.up_down_history.set_index('日期', inplace=True)
                    
                    # 由于只有一天数据，无法计算均线，直接赋值为当日值
                    self.up_down_history['上涨家数MA5'] = up_count
                    self.up_down_history['下跌家数MA5'] = down_count
                    
                    # 无法进行5日均线比较，但可以记录当天的涨跌比
                    up_down_ratio = up_count / down_count if down_count > 0 else (10 if up_count > 0 else 1)
                    
                    # 根据涨跌比简单评分
                    if up_down_ratio > 1.5:
                        score += 8  # 多头市场
                    elif up_down_ratio < 0.8:
                        score -= 5  # 空头市场
            
            # 如果获取到了历史涨跌家数数据，计算5日均线
            if self.up_down_history is not None and not self.up_down_history.empty and len(self.up_down_history) >= 5:
                # 计算涨跌家数的5日均线
                if '上涨家数MA5' not in self.up_down_history.columns:
                    self.up_down_history['上涨家数MA5'] = self.up_down_history['上涨家数'].rolling(window=5).mean()
                if '下跌家数MA5' not in self.up_down_history.columns:
                    self.up_down_history['下跌家数MA5'] = self.up_down_history['下跌家数'].rolling(window=5).mean()
                
                # 最新一天的均线值
                if len(self.up_down_history) >= 5:
                    last_idx = self.up_down_history.index[-1]
                    up_ma5 = self.up_down_history.loc[last_idx, '上涨家数MA5']
                    down_ma5 = self.up_down_history.loc[last_idx, '下跌家数MA5']
                    
                    # 计算上涨家数5日线趋势（今日值与前5日均值比较）
                    if not np.isnan(up_ma5) and not np.isnan(down_ma5):
                        last_up = self.up_down_history.loc[last_idx, '上涨家数']
                        last_down = self.up_down_history.loc[last_idx, '下跌家数']
                        
                        # 判断上涨家数5日线的方向
                        if last_up > up_ma5 * 1.1:
                            score += 8  # 上涨家数明显高于5日均线，市场情绪转强
                        elif last_up < up_ma5 * 0.9:
                            score -= 5  # 上涨家数明显低于5日均线，市场情绪转弱
                        
                        # 计算上涨/下跌家数比值
                        up_down_ratio = last_up / last_down if last_down > 0 else 10
                        up_down_ma5_ratio = up_ma5 / down_ma5 if down_ma5 > 0 else 10
                        
                        # 比较当日上涨/下跌比与5日均线上涨/下跌比
                        if up_down_ratio > up_down_ma5_ratio * 1.2:
                            score += 7  # 上涨/下跌比明显高于5日均线，市场情绪转强
                        elif up_down_ratio < up_down_ma5_ratio * 0.8:
                            score -= 7  # 上涨/下跌比明显低于5日均线，市场情绪转弱
        except Exception as e:
            print(f"分析涨跌家数5日线失败: {e}")
        
        # 4. 涨跌家数CCI指标分析
        try:
            if self.up_down_history is not None and not self.up_down_history.empty:
                # 如果历史数据不足，使用简化的CCI计算
                if len(self.up_down_history) >= 20:
                    # 计算上涨家数的CCI指标（商品通道指数）
                    up_counts = self.up_down_history['上涨家数'].values
                    down_counts = self.up_down_history['下跌家数'].values
                    
                    # 计算上涨/下跌比值序列
                    up_down_ratios = np.array([u/d if d > 0 else 10 for u, d in zip(up_counts, down_counts)])
                    
                    # 计算CCI
                    typical_price = up_down_ratios
                    sma = np.mean(typical_price[-20:])
                    mean_deviation = np.mean(np.abs(typical_price[-20:] - sma))
                    cci = (typical_price[-1] - sma) / (0.015 * mean_deviation) if mean_deviation > 0 else 0
                    
                    # 根据CCI值评分
                    if cci > 100:
                        score += 5  # CCI超过100，市场强势
                    elif cci < -100:
                        score -= 5  # CCI低于-100，市场弱势
                    
                    # 存储CCI值
                    self.up_down_cci = cci
                else:
                    # 数据不足20天，无法计算标准CCI，使用简化计算
                    last_idx = self.up_down_history.index[-1]
                    last_up = self.up_down_history.loc[last_idx, '上涨家数']
                    last_down = self.up_down_history.loc[last_idx, '下跌家数']
                    
                    # 使用当日涨跌比判断
                    up_down_ratio = last_up / last_down if last_down > 0 else 10
                    
                    # 简单赋值一个伪CCI值
                    if up_down_ratio > 1.5:
                        cci = 50  # 多头市场，但由于数据不足，不给太高值
                    elif up_down_ratio < 0.8:
                        cci = -50  # 空头市场，但由于数据不足，不给太低值
                    else:
                        cci = 0  # 中性
                    
                    self.up_down_cci = cci
                    print(f"警告: 历史涨跌家数数据不足20天，使用简化CCI: {cci}")
        except Exception as e:
            print(f"分析涨跌家数CCI指标失败: {e}")
        
        # 归一化得分到0-100区间
        normalized_score = max(0, min(100, score + 50))
        self.emotion_cycle_score = normalized_score
        return normalized_score
    
    def calculate_market_strength(self):
        """计算市场强度综合得分"""
        # 计算各维度得分
        index_trend = self.analyze_index_trend()
        
        # 提前获取市场数据，避免重复获取
        market_data = self.get_market_data()
        emotion = self.analyze_emotion_indicators(market_data)
        
        # 分析情绪周期
        emotion_cycle = self.analyze_emotion_cycle(market_data)
        
        fund_behavior = self.analyze_fund_behavior()
        
        # 加权计算市场强度（调整权重以包含情绪周期）
        market_strength = index_trend * 0.35 + emotion * 0.25 + emotion_cycle * 0.2 + fund_behavior * 0.2
        self.market_strength = market_strength
        
        return market_strength
    
    def get_operation_advice(self):
        """根据市场强度提供操作建议"""
        strength = self.market_strength
        
        if strength >= 80:
            return "强势市场，适合积极参与，重点关注高景气度行业龙头"
        elif strength >= 65:
            return "偏强市场，可适度进取，关注行业轮动机会"
        elif strength >= 50:
            return "中性市场，选择性参与，精选个股"
        elif strength >= 35:
            return "偏弱市场，谨慎参与，降低仓位，关注防御性板块"
        else:
            return "弱势市场，建议观望或空仓，等待市场企稳信号"


def print_market_indicators(analyzer):
    """打印当日市场指标，包括涨跌比例、涨停跌停数、指数走势，以Markdown格式输出"""
    markdown = f"## {analyzer.date_formatted}\n\n"
    
    # 获取市场数据
    try:
        stock_data, limit_up_data = analyzer.get_market_data()
        
        # 检查是否成功获取数据
        if stock_data.empty:
            markdown += "### ⚠️ 警告：无法获取市场数据\n\n"
            markdown += "可能的原因：\n\n"
            markdown += "1. 网络连接问题\n"
            markdown += "2. 数据源服务器暂时不可用\n"
            markdown += "3. API接口已变更\n\n"
            markdown += "建议：\n\n"
            markdown += "- 检查网络连接\n"
            markdown += "- 稍后重试\n"
            markdown += "- 尝试使用不同的数据源（修改`data_source`参数）\n\n"
        else:
            # 1. 涨跌比例、涨停数、跌停数及情绪周期数据合并展示
            markdown += "### 1. 市场情绪与涨跌统计\n\n"
            try:
                up_count = len(stock_data[stock_data['涨跌幅'] > 0])
                down_count = len(stock_data[stock_data['涨跌幅'] < 0])
                flat_count = len(stock_data) - up_count - down_count
                total_count = len(stock_data)
                up_down_ratio = up_count / down_count if down_count > 0 else 10
                
                # 涨停跌停数量
                limit_up_count = len(stock_data[stock_data['涨跌幅'] >= 9.5])  # 近似值，实际应使用涨停板数据
                limit_down_count = len(stock_data[stock_data['涨跌幅'] <= -9.5])  # 近似值
                
                # 添加大幅涨跌统计
                big_up_count = 0 
                big_down_count = 0
                big_ratio = 0
                
                if hasattr(analyzer, 'big_move_ratio') and analyzer.big_move_ratio:
                    big_up_count = analyzer.big_move_ratio['大涨股数量']
                    big_down_count = analyzer.big_move_ratio['大跌股数量']
                    big_ratio = analyzer.big_move_ratio['比例']
                else:
                    # 如果没有预先计算，现在计算
                    if not stock_data.empty and '涨跌幅' in stock_data.columns:
                        big_up_count = len(stock_data[stock_data['涨跌幅'] > 7])
                        big_down_count = len(stock_data[stock_data['涨跌幅'] < -7])
                        big_ratio = big_up_count / big_down_count if big_down_count > 0 else (big_up_count if big_up_count > 0 else 0)
                
                # 合并表格
                markdown += "| 项目 | 数值 | 占比 | 情绪评价 |\n"
                markdown += "| ---- | ---- | ---- | ---- |\n"
                markdown += f"| 上涨家数 | {up_count} | {up_count/total_count*100:.2f}% | {'偏多' if up_count > down_count else '偏空'} |\n"
                markdown += f"| 下跌家数 | {down_count} | {down_count/total_count*100:.2f}% | - |\n"
                markdown += f"| 平盘家数 | {flat_count} | {flat_count/total_count*100:.2f}% | - |\n"
                markdown += f"| 涨跌比例 | {up_down_ratio:.2f} | - | {'偏强' if up_down_ratio > 1.5 else ('偏弱' if up_down_ratio < 0.8 else '中性')} |\n"
                markdown += f"| 涨幅>7%股票 | {big_up_count} | {big_up_count/total_count*100:.2f}% | - |\n"
                markdown += f"| 跌幅>7%股票 | {big_down_count} | {big_down_count/total_count*100:.2f}% | - |\n"
                markdown += f"| 大涨/大跌比 | {big_ratio:.2f} | - | {'强势' if big_ratio > 3 else ('偏强' if big_ratio > 1.5 else ('偏弱' if big_ratio < 0.8 else '中性'))} |\n"
                markdown += f"| 涨停家数 | {limit_up_count} | {limit_up_count/total_count*100:.2f}% | - |\n"
                markdown += f"| 跌停家数 | {limit_down_count} | {limit_down_count/total_count*100:.2f}% | - |\n"
                
                # 涨跌家数5日线比较
                if analyzer.up_down_history is not None and not analyzer.up_down_history.empty and len(analyzer.up_down_history) >= 5:
                    last_idx = analyzer.up_down_history.index[-1]
                    last_up = analyzer.up_down_history.loc[last_idx, '上涨家数']
                    last_down = analyzer.up_down_history.loc[last_idx, '下跌家数']
                    up_ma5 = analyzer.up_down_history.loc[last_idx, '上涨家数MA5'] if '上涨家数MA5' in analyzer.up_down_history.columns else 0
                    
                    if up_ma5 > 0:
                        up_change_rate = (last_up/up_ma5-1)*100
                        markdown += f"| 上涨家数/5日均值 | {last_up}/{up_ma5:.0f} | {up_change_rate:.2f}% | {'转强' if up_change_rate > 10 else ('转弱' if up_change_rate < -10 else '持平')} |\n"
                
                # CCI指标
                if hasattr(analyzer, 'up_down_cci'):
                    cci_status = '强势' if analyzer.up_down_cci > 100 else ('弱势' if analyzer.up_down_cci < -100 else '中性')
                    markdown += f"| 涨跌比CCI | {analyzer.up_down_cci:.2f} | - | {cci_status} |\n"
                
                # 断板反包情况
                if hasattr(analyzer, 'prev_broken_stocks') and analyzer.prev_broken_stocks and hasattr(analyzer, 'broken_limit_recovery'):
                    recovery_count = len(analyzer.broken_limit_recovery)
                    total_broken = len(analyzer.prev_broken_stocks)
                    if total_broken > 0:
                        recovery_rate = recovery_count / total_broken * 100
                        markdown += f"| 断板反包率 | {recovery_count}/{total_broken} | {recovery_rate:.2f}% | {'活跃' if recovery_rate > 60 else ('低迷' if recovery_rate < 30 else '一般')} |\n"
                
                if not limit_up_data.empty:
                    try:
                        # 涨停梯队分析
                        if '连板数' in limit_up_data.columns:
                            limit_up_data['连板天数'] = limit_up_data['连板数'].astype(int)
                        elif '连板天数' in limit_up_data.columns:
                            pass  # 已经有连板天数列
                        else:
                            limit_up_data['连板天数'] = 1  # 默认为1

                        consecutive_boards = limit_up_data[limit_up_data['连板天数'] > 0]

                        # 连板数量统计
                        one_board = len(consecutive_boards[consecutive_boards['连板天数'] == 1])
                        two_board = len(consecutive_boards[consecutive_boards['连板天数'] == 2])
                        three_plus_board = len(consecutive_boards[consecutive_boards['连板天数'] >= 3])

                        # 最高连板数
                        max_consecutive = consecutive_boards['连板天数'].max() if not consecutive_boards.empty else 0

                        markdown += f"| 首板涨停 | {one_board}家 | - | - |\n"
                        markdown += f"| 二连板 | {two_board}家  | - | - |\n"
                        markdown += f"| 三板及以上 | {three_plus_board}家  | - | {'强势' if three_plus_board >= 5 else ('活跃' if three_plus_board >= 2 else '低迷')} |\n"
                        markdown += f"| 最高连板天数 | {max_consecutive}天 | - | {'主升段' if max_consecutive >= 5 else ('偏强' if max_consecutive >= 3 else '一般')} |\n"

                        # 炸板率分析
                        if '炸板' in limit_up_data.columns:
                            limit_up_broken = limit_up_data[limit_up_data['炸板'] == True]
                            if not limit_up_broken.empty and not limit_up_data.empty:
                                broken_rate = len(limit_up_broken) / (len(limit_up_data) + len(limit_up_broken))
                                markdown += f"| 炸板率 | {len(limit_up_broken)}/{len(limit_up_data)+len(limit_up_broken)} | {broken_rate*100:.2f}% | {'情绪低迷' if broken_rate > 0.4 else ('情绪高涨' if broken_rate < 0.2 else '情绪平稳')} |\n"

                    except Exception as e:
                        markdown += f"计算涨停板数据失败: {e}\n\n"
                else:
                    markdown += "无法获取涨停板数据或当日无涨停股票\n\n"
                
                markdown += "\n"
                
                # 如果有情绪标的，简要列出
                if analyzer.emotion_leaders and len(analyzer.emotion_leaders) > 0:
                    markdown += "**情绪标的**: "
                    leaders_info = []
                    for code, data in analyzer.emotion_leaders.items():
                        # 获取最近表现
                        recent_avg = 0
                        if data['后续表现']:
                            recent_perfs = [p['涨跌幅'] for p in data['后续表现'][-3:] if p['涨跌幅'] is not None]
                            if recent_perfs:
                                recent_avg = sum(recent_perfs) / len(recent_perfs)
                        
                        # 添加标的信息
                        status = "强势" if recent_avg > 5 else ("弱势" if recent_avg < -3 else "中性")
                        leaders_info.append(f"{code}({data['连板天数']}板, {status})")
                    
                    markdown += ", ".join(leaders_info) + "\n\n"
                
            except Exception as e:
                markdown += f"计算涨跌比例失败: {e}\n\n"
    except Exception as e:
        markdown += f"### ⚠️ 获取市场数据时发生错误\n\n"
        markdown += f"错误信息: {str(e)[:200]}\n\n"
        markdown += "请检查网络连接或尝试使用不同的数据源。\n\n"
        # 创建一个空的DataFrame以便后续代码可以继续执行
        stock_data = pd.DataFrame()
        limit_up_data = pd.DataFrame()
    
    # 初始化指数数据
    try:
        analyzer.analyze_index_trend()
        
        # 2. 指数走势分析
        markdown += "### 2. 指数走势分析\n\n"
        
        # 创建合并表格
        markdown += "| 指数名称 | 收盘价 | 涨跌幅 | 指标分析 | 趋势判断 |\n"
        markdown += "| ---- | ---- | ---- | ---- | ---- |\n"
        
        # 上证指数分析
        if analyzer.sh_index_data is not None and not analyzer.sh_index_data.empty:
            close = analyzer.sh_index_data['收盘'].values
            volume = analyzer.sh_index_data['成交量'].values
            
            # 计算均线
            ma5 = MA(close, 5)
            ma10 = MA(close, 10)
            ma20 = MA(close, 20)
            
            # 最新均线值
            last_ma5 = ma5[-1] if not np.isnan(ma5[-1]) else 0
            last_ma10 = ma10[-1] if not np.isnan(ma10[-1]) else 0
            last_ma20 = ma20[-1] if not np.isnan(ma20[-1]) else 0
            
            # 均线排列方向
            if last_ma5 > last_ma10 > last_ma20:
                ma_pattern = "多头排列"
            elif last_ma5 < last_ma10 < last_ma20:
                ma_pattern = "空头排列"
            else:
                ma_pattern = "无明显排列"
            
            # 今日涨跌幅
            if len(close) > 1:
                daily_change = (close[-1] / close[-2] - 1) * 100
            else:
                daily_change = 0
            
            markdown += f"| 上证指数 | {close[-1]:.2f} | {daily_change:.2f}% | MA5={last_ma5:.2f} MA10={last_ma10:.2f} MA20={last_ma20:.2f} | {ma_pattern} |\n"
        else:
            markdown += "| 上证指数 | 数据不可用 | - | - | - |\n"
        
        # 创业板指数分析
        if analyzer.gem_index_data is not None and not analyzer.gem_index_data.empty:
            close = analyzer.gem_index_data['收盘'].values
            high = analyzer.gem_index_data['最高'].values
            low = analyzer.gem_index_data['最低'].values
            
            # RSI指标
            rsi = RSI(close)
            last_rsi = rsi[-1] if not np.isnan(rsi[-1]) else 50
            
            # 今日涨跌幅
            if len(close) > 1:
                daily_change = (close[-1] / close[-2] - 1) * 100
            else:
                daily_change = 0
                
            # 简单趋势判断
            if last_rsi > 70:
                trend = "超买"
            elif last_rsi < 30:
                trend = "超卖"
            else:
                if daily_change > 0:
                    trend = "走强"
                elif daily_change < 0:
                    trend = "走弱"
                else:
                    trend = "震荡"
            
            markdown += f"| 创业板指数 | {close[-1]:.2f} | {daily_change:.2f}% | RSI={last_rsi:.2f} | {trend} |\n"
        else:
            markdown += "| 创业板指数 | 数据不可用 | - | - | - |\n"
        
        # 平均股价指数分析
        if analyzer.avg_price_index_data is not None and not analyzer.avg_price_index_data.empty and analyzer.sh_index_data is not None and not analyzer.sh_index_data.empty:
            avg_close = analyzer.avg_price_index_data['收盘'].values
            sh_close = analyzer.sh_index_data['收盘'].values
            
            # 今日涨跌幅
            if len(avg_close) > 1:
                daily_change = (avg_close[-1] / avg_close[-2] - 1) * 100
            else:
                daily_change = 0
            
            # 计算背离度(简化为相对变化率之差)
            if len(avg_close) >= 5 and len(sh_close) >= 5:
                avg_change = (avg_close[-1] / avg_close[-5] - 1) * 100
                sh_change = (sh_close[-1] / sh_close[-5] - 1) * 100
                divergence = avg_change - sh_change
                
                if divergence > 2:
                    div_status = "强于大盘"
                elif divergence < -2:
                    div_status = "弱于大盘"
                else:
                    div_status = "基本同步"
                
                markdown += f"| 平均股价指数 | {avg_close[-1]:.2f} | {daily_change:.2f}% | 背离度={divergence:.2f}% | {div_status} |\n"
            else:
                markdown += f"| 平均股价指数 | {avg_close[-1]:.2f} | {daily_change:.2f}% | 数据不足 | - |\n"
        else:
            markdown += "| 平均股价指数 | 数据不可用 | - | - | - |\n"
        
        # 小盘股指数分析
        if analyzer.small_cap_index_data is not None and not analyzer.small_cap_index_data.empty and analyzer.sh50_index_data is not None and not analyzer.sh50_index_data.empty:
            small_close = analyzer.small_cap_index_data['收盘'].values
            sh50_close = analyzer.sh50_index_data['收盘'].values
            
            # 今日涨跌幅
            if len(small_close) > 1:
                daily_change = (small_close[-1] / small_close[-2] - 1) * 100
            else:
                daily_change = 0
            
            # 计算剪刀差
            if len(small_close) >= 2 and len(sh50_close) >= 2:
                small_change = (small_close[-1] / small_close[-2] - 1) * 100
                sh50_change = (sh50_close[-1] / sh50_close[-2] - 1) * 100
                scissors_diff = small_change - sh50_change
                
                if scissors_diff > 1:
                    scissors_status = "小盘强于大盘"
                elif scissors_diff < -1:
                    scissors_status = "大盘强于小盘"
                else:
                    scissors_status = "大小盘同步"
                
                markdown += f"| 中证1000(小盘股) | {small_close[-1]:.2f} | {daily_change:.2f}% | 剪刀差={scissors_diff:.2f}% | {scissors_status} |\n"
            else:
                markdown += f"| 中证1000(小盘股) | {small_close[-1]:.2f} | {daily_change:.2f}% | 数据不足 | - |\n"
        else:
            markdown += "| 中证1000(小盘股) | 数据不可用 | - | - | - |\n"
        
        markdown += "\n"
    except Exception as e:
        markdown += f"### ⚠️ 获取指数数据时发生错误\n\n"
        markdown += f"错误信息: {str(e)[:200]}\n\n"
        markdown += "请检查网络连接或尝试使用不同的数据源。\n\n"

    # 3. 市场强度和建议
    try:
        markdown += "### 3. 市场强度和操作建议\n\n"
        
        # 计算市场强度
        market_strength = analyzer.market_strength if analyzer.market_strength > 0 else analyzer.calculate_market_strength()
        
        # 基于分数确定市场状态
        if market_strength >= 80:
            market_status = "积极"
        elif market_strength >= 65:
            market_status = "偏积极"
        elif market_strength >= 50:
            market_status = "中性"
        elif market_strength >= 35:
            market_status = "谨慎"
        else:
            market_status = "空仓"
        
        # 创建市场强度表格
        markdown += "| 分析维度 | 得分 | 状态 |\n"
        markdown += "| ---- | ---- | ---- |\n"
        markdown += f"| 指数趋势 | {analyzer.index_trend_score:.2f}/100 | {get_trend_description(analyzer.index_trend_score)} |\n"
        markdown += f"| 情绪指标 | {analyzer.emotion_score:.2f}/100 | {get_emotion_description(analyzer.emotion_score)} |\n"
        markdown += f"| 情绪周期 | {analyzer.emotion_cycle_score:.2f}/100 | {get_emotion_cycle_description(analyzer.emotion_cycle_score)} |\n"
        markdown += f"| 资金行为 | {analyzer.fund_behavior_score:.2f}/100 | {get_fund_description(analyzer.fund_behavior_score)} |\n"
        markdown += f"| **综合强度** | **{market_strength:.2f}/100** | **{market_status}** |\n\n"
        
        # 操作建议
        markdown += f"**操作建议**: {analyzer.get_operation_advice()}\n\n"
        
        # 添加市场总结分析
        markdown += "### 4. 市场总结分析\n\n"
        markdown += f"根据当日市场指标综合分析，市场强度得分为 **{market_strength:.2f}/100**，处于**{market_status}**状态。\n\n"
    except Exception as e:
        markdown += f"计算市场强度失败: {e}\n\n"
        markdown += "由于数据获取失败，无法提供准确的市场强度评分和操作建议。\n"
        markdown += "建议在数据恢复后重新运行分析。\n\n"
    
    print(markdown)

# 添加辅助函数用于生成分析描述
def get_trend_description(score):
    """根据指数趋势得分生成描述"""
    if score >= 70:
        return "多头排列明显，中短期趋势向上\n"
    elif score >= 50:
        return "偏多格局，整体呈上升趋势\n"
    elif score >= 30:
        return  "震荡格局，无明显趋势\n"
    else:
        return  "空头排列，趋势向下\n"

def get_emotion_description(score):
    """根据情绪指标得分生成描述"""
    if score >= 70:
        return  "市场情绪高涨，短期涨停动能强劲"
    elif score >= 50:
        return  "情绪偏乐观，有一定赚钱效应"
    elif score >= 30:
        return  "情绪平淡，观望情绪较浓"
    else:
        return "情绪低迷，连板稀少，炸板率高"

def get_fund_description(score):
    """根据资金行为得分生成描述"""
    if score >= 70:
        return "主力资金积极进场，北向资金大幅流入"
    elif score >= 50:
        return  "资金面略有改善，有选择性布局迹象"
    elif score >= 30:
        return "资金观望为主，无明显趋势"
    else:
        return  "资金大幅流出，市场承压"

def get_emotion_cycle_description(score):
    """根据情绪周期得分生成描述"""
    if score >= 70:
        return "情绪周期处于高点，标的强势，反包活跃"
    elif score >= 50:
        return "情绪周期向上，标的表现良好"
    elif score >= 30:
        return "情绪周期平稳，标的表现一般"
    else:
        return "情绪周期低迷，断板无反包，市场弱势"

def plot_kdj(stock_code, level=KlinePeriod.DAILY, start_date='20220101', end_date='20230512'):
    """
    绘制KDJ指标
    :param stock_code: 股票代码
    :param level: K线级别
    :param start_date: 开始日期
    :param end_date: 结束日期
    :return:
    """
    # 读取股票数据
    klt = KlinePeriod.get_klt_code(level)
    stock_data = ef.stock.get_quote_history(stock_code, beg=start_date, end=end_date, klt=klt)
    # 计算KDJ指标
    close = stock_data['收盘'].values
    low = stock_data['最低'].values
    high = stock_data['最高'].values
    date = stock_data['日期'].values

    # 计算KDJ指标
    K, D, J = KDJ(close, high, low, 9, 3, 3)
    # 回测20日均线
    ma20 = MA(close, 20)
    fig, ax = plt.subplots(figsize=(15, 8))
    # 绘制K线图
    ax.plot(date, close, label='close', color='r', linewidth=1.5)
    ax.plot(date, ma20, label='MA20', color='b', linewidth=1.5)
    # 绘制KDJ指标
    ax.plot(date, K, label='K', color='g', linewidth=1.5)
    ax.plot(date, D, label='D', color='y', linewidth=1.5)
    ax.plot(date, J, label='J', color='m', linewidth=1.5)
    # 设置x轴刻度，每隔20个显示一个
    ax.set_xticks(range(0, len(date), 20))
    ax.set_xticklabels(date[::20], rotation=45)
    # 设置标题
    ax.set_title('KDJ指标')
    # 设置图例
    ax.legend()
    plt.show()


def plot_macd(stock_code, level=KlinePeriod.DAILY, start_date='20220101', end_date='20230512'):
    """
    绘制MACD指标
    :param stock_code: 股票代码
    :param level: K线级别
    :param start_date: 开始日期
    :param end_date: 结束日期
    :return:
    """
    # 读取股票数据
    klt = KlinePeriod.get_klt_code(level)
    stock_data = ef.stock.get_quote_history(stock_code, beg=start_date, end=end_date, klt=klt)
    # 计算MACD指标
    close = stock_data['收盘'].values
    date = stock_data['日期'].values
    # 计算MACD指标
    diff, dea, macd = MACD(close, 12, 26, 9)
    # 回测20日均线
    ma20 = MA(close, 20)
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(date, close, label='close', color='r', linewidth=1.5)
    ax.plot(date, ma20, label='MA20', color='b', linewidth=1.5)
    # 绘制MACD指标
    ax.plot(date, diff, label='DIFF', color='g', linewidth=1.5)
    ax.plot(date, dea, label='DEA', color='y', linewidth=1.5)
    # 绘制MACD柱状图
    ax.bar(date, macd, label='MACD', width=1)
    # 设置x轴刻度，每隔20个显示一个
    ax.set_xticks(range(0, len(date), 20))
    ax.set_xticklabels(date[::20], rotation=45)
    # 设置标题
    ax.set_title('MACD指标')
    # 设置图例
    ax.legend()
    plt.show()

def main():
    """主函数"""
    # 默认分析当天市场
    analyzer = MarketAnalyzer(data_source="auto")
    
    # 打印主要市场指标
    print_market_indicators(analyzer)
    
    # 生成完整报告
    # report = analyzer.generate_report()
    # print(report)
    
    # 也可以指定日期进行分析
    # analyzer = MarketAnalyzer('20231201', data_source="baostock")
    # report = analyzer.generate_report()
    # print(report)


if __name__ == "__main__":
    main()