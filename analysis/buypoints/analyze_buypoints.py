#!/usr/bin/python
# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
from db.clickhouse_db import get_clickhouse_db, get_default_config
from enums.kline_period import KlinePeriod
from enums.indicators import *
import json
import sys
from datetime import datetime, timedelta

class BuyPointAnalyzer:
    """
    买点分析器 - 分析指定股票在指定日期的技术指标特征
    """
    
    def __init__(self):
        """
        初始化买点分析器
        """
        print("初始化买点分析器")
        # 获取默认配置并指定密码
        config = get_default_config()
        # 使用get_clickhouse_db函数获取ClickHouseDB实例
        self.ch_db = get_clickhouse_db(config=config)
        print("成功连接到数据库")
    
    def analyze_stock(self, stock_code, buy_date, stock_name=""):
        """
        分析指定股票在指定日期的买点特征
        
        Args:
            stock_code: 股票代码
            buy_date: 买点日期 (格式: YYYYMMDD)
            stock_name: 股票名称
        
        Returns:
            dict: 技术指标结果
        """
        # 转换日期格式
        buy_date_obj = datetime.strptime(buy_date, "%Y%m%d")
        
        # 计算前后日期范围
        start_date = (buy_date_obj - timedelta(days=60)).strftime("%Y%m%d")
        end_date = (buy_date_obj + timedelta(days=10)).strftime("%Y%m%d")
        
        print(f"分析 {stock_code} {stock_name} 在 {buy_date} 的买点特征...")
        print(f"获取 {stock_code} 从 {start_date} 到 {end_date} 的数据...")
        
        try:
            # 从数据库获取数据
            stock_data = self.ch_db.get_stock_info(stock_code, KlinePeriod.DAILY.value, start_date, end_date)
            
            if not stock_data or len(stock_data) == 0:
                print(f"警告: 未找到 {stock_code} 的数据")
                return None
            
            # 转换为DataFrame
            df = pd.DataFrame(stock_data, columns=[
                'code', 'name', 'date', 'level', 'open', 'close', 'high', 'low',
                'volume', 'turnover_rate', 'price_change', 'price_range', 'industry'
            ])
            
            # 转换日期列为日期类型
            df['date'] = pd.to_datetime(df['date'])
            
            # 排序数据
            df = df.sort_values('date')
            
            # 查找买点日期的索引
            buy_date_idx = None
            for i, date in enumerate(df['date']):
                if date.strftime("%Y%m%d") == buy_date:
                    buy_date_idx = i
                    break
            
            if buy_date_idx is None:
                print(f"警告: 未找到买点日期 {buy_date} 的数据")
                return None
            
            # 计算企稳反弹买点技术指标
            indicators = self.calculate_buy_point_indicators(df, buy_date_idx)
            
            if indicators:
                print(f"成功计算 {stock_code} {stock_name} 在 {buy_date} 的技术指标")
                # 添加基本信息
                indicators['code'] = stock_code
                indicators['name'] = stock_name if stock_name else df['name'].iloc[0]
                indicators['date'] = buy_date
                indicators['industry'] = df['industry'].iloc[0]
                return indicators
            else:
                print(f"计算 {stock_code} {stock_name} 技术指标失败")
                return None
            
        except Exception as e:
            print(f"分析股票 {stock_code} 时出错: {e}")
            return None
    
    def calculate_buy_point_indicators(self, df, buy_date_idx):
        """
        计算企稳反弹买点的技术指标
        
        Args:
            df: 股票数据DataFrame
            buy_date_idx: 买点日期索引
        
        Returns:
            dict: 技术指标结果
        """
        if buy_date_idx is None or buy_date_idx < 5:  # 需要至少5天的数据来计算指标
            return {}
        
        # 获取数据
        close = df['close'].values
        open_price = df['open'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        # 计算移动平均线
        ma5 = MA(close, 5)
        ma10 = MA(close, 10)
        ma20 = MA(close, 20)
        ma30 = MA(close, 30)
        ma60 = MA(close, 60)
        
        # 成交量移动平均
        vol5 = MA(volume, 5)
        vol10 = MA(volume, 10)
        vol20 = MA(volume, 20)
        
        # 计算MACD
        dif, dea, macd = MACD(close, 12, 26, 9)
        
        # 计算KDJ
        kdj_k, kdj_d, kdj_j = KDJ(close, high, low, 9, 3, 3)
        
        # 计算买点日期的指标
        # 触及均线
        touch_ma10 = abs(low[buy_date_idx] / ma10[buy_date_idx] - 1) < 0.01
        touch_ma20 = abs(low[buy_date_idx] / ma20[buy_date_idx] - 1) < 0.01
        touch_ma30 = abs(low[buy_date_idx] / ma30[buy_date_idx] - 1) < 0.01
        touch_ma60 = abs(low[buy_date_idx] / ma60[buy_date_idx] - 1) < 0.01
        touch_ma = touch_ma10 or touch_ma20 or touch_ma30 or touch_ma60
        
        # 价格企稳
        price_stable = (close[buy_date_idx] > close[buy_date_idx-1] and 
                        low[buy_date_idx] > low[buy_date_idx-1] * 0.995 and
                        (close[buy_date_idx] - low[buy_date_idx]) / (high[buy_date_idx] - low[buy_date_idx]) > 0.5)
        
        # 均线上移
        ma_up = (abs(ma5[buy_date_idx] / ma10[buy_date_idx] - 1) < 0.01 and 
                abs(ma10[buy_date_idx] / ma20[buy_date_idx] - 1) < 0.015 and 
                ma5[buy_date_idx] > ma5[buy_date_idx-1] and 
                ma10[buy_date_idx] > ma10[buy_date_idx-1])
        
        # WVAD指标计算
        l55 = LLV(low, 55)
        h55 = HHV(high, 55)
        diff = (close - l55) / (h55 - l55) * 100
        s1 = SMA(diff, 5, 1)
        s2 = SMA(s1, 3, 1)
        wvad = 3 * s1 - 2 * s2
        wv_ma = MA(wvad, 3)
        wv_chg = np.zeros_like(wv_ma)
        wv_chg[1:] = (wv_ma[1:] - wv_ma[:-1]) / wv_ma[:-1] * 100
        
        # 计算吸筹信号
        abs_signal1 = (wv_ma[buy_date_idx] <= 13 and 
                       np.sum(wv_ma[buy_date_idx-15:buy_date_idx+1] <= 13) > 10)
        abs_signal2 = (wv_ma[buy_date_idx] <= 13 and wv_chg[buy_date_idx] > 13)
        
        # 成交量放大并且收盘在高位
        vol_close_high = (volume[buy_date_idx] > vol5[buy_date_idx] * 1.2 and 
                          close[buy_date_idx] > close[buy_date_idx-1] and
                          (close[buy_date_idx] - low[buy_date_idx]) / (high[buy_date_idx] - low[buy_date_idx]) > 0.7)
        
        money_in = abs_signal1 or abs_signal2 or vol_close_high
        
        # K线形态改善
        xsmall = abs(open_price[buy_date_idx] - close[buy_date_idx]) / close[buy_date_idx] < 0.01
        xstar = (abs(open_price[buy_date_idx] - close[buy_date_idx]) / close[buy_date_idx] < 0.005 and
                (high[buy_date_idx] - max(open_price[buy_date_idx], close[buy_date_idx])) > 0 and
                (min(open_price[buy_date_idx], close[buy_date_idx]) - low[buy_date_idx]) > 0)
        xshadow = ((min(open_price[buy_date_idx], close[buy_date_idx]) - low[buy_date_idx]) / 
                  (high[buy_date_idx] - low[buy_date_idx]) > 0.3 and close[buy_date_idx] >= open_price[buy_date_idx])
        kpattern = xsmall or xstar or xshadow
        
        # 成交量缩量
        vol_shrink = volume[buy_date_idx] < vol5[buy_date_idx] * 0.8 or (
            volume[buy_date_idx] < vol5[buy_date_idx] * 1.2 and volume[buy_date_idx] > vol5[buy_date_idx] * 0.8)
        
        # 回踩均线
        touch_ma_f = touch_ma
        
        # MACD底背离判断
        macd_gold = ((macd[buy_date_idx-1] < 0 and macd[buy_date_idx] > macd[buy_date_idx-1] and 
                     dif[buy_date_idx] > dif[buy_date_idx-1]) or 
                    (macd[buy_date_idx-2] < 0 and macd[buy_date_idx-1] < 0 and macd[buy_date_idx] > 0))
        
        # 其他技术指标变化
        dif_up = dif[buy_date_idx] > dif[buy_date_idx-1]
        dea_up = dea[buy_date_idx] > dea[buy_date_idx-1]
        k_up = kdj_k[buy_date_idx] > kdj_k[buy_date_idx-1]
        d_up = kdj_d[buy_date_idx] > kdj_d[buy_date_idx-1]
        j_up = kdj_j[buy_date_idx] > kdj_j[buy_date_idx-1]
        
        # 汇总技术指标结果
        result = {
            'close': close[buy_date_idx],
            'touch_ma': touch_ma,
            'touch_ma10': touch_ma10,
            'touch_ma20': touch_ma20,
            'touch_ma30': touch_ma30,
            'touch_ma60': touch_ma60,
            'touch_ma_formula': touch_ma_f,
            'price_stable': price_stable,
            'ma_up': ma_up,
            'abs_signal1': abs_signal1,
            'abs_signal2': abs_signal2,
            'vol_close_high': vol_close_high,
            'money_in': money_in,
            'xc': abs_signal1 or abs_signal2,  # 吸筹信号
            'kpattern': kpattern,
            'xsmall': xsmall,
            'xstar': xstar,
            'xshadow': xshadow,
            'vol_shrink': vol_shrink,
            'macd_gold': macd_gold,
            'dif_up': dif_up,
            'dea_up': dea_up,
            'k_up': k_up,
            'd_up': d_up,
            'j_up': j_up,
            'ma5': ma5[buy_date_idx],
            'ma10': ma10[buy_date_idx],
            'ma20': ma20[buy_date_idx],
            'ma30': ma30[buy_date_idx],
            'ma60': ma60[buy_date_idx],
            'vol': volume[buy_date_idx],
            'vol5': vol5[buy_date_idx],
            'vol10': vol10[buy_date_idx],
            'vol20': vol20[buy_date_idx],
            'macd': macd[buy_date_idx],
            'dif': dif[buy_date_idx],
            'dea': dea[buy_date_idx],
            'kdj_k': kdj_k[buy_date_idx],
            'kdj_d': kdj_d[buy_date_idx],
            'kdj_j': kdj_j[buy_date_idx],
            'wv_ma': wv_ma[buy_date_idx],
            'wv_chg': wv_chg[buy_date_idx]
        }
        
        return result
    
    def analyze_multiple_buypoints(self, buypoints_list):
        """
        分析多个股票买点的技术指标特征
        
        Args:
            buypoints_list: 包含多个(股票代码, 买点日期, 股票名称)的列表
        
        Returns:
            DataFrame: 汇总分析结果
        """
        results = []
        
        for item in buypoints_list:
            if len(item) >= 2:
                stock_code = item[0]
                buy_date = item[1]
                stock_name = item[2] if len(item) > 2 else ""
                
                indicators = self.analyze_stock(stock_code, buy_date, stock_name)
                if indicators:
                    results.append(indicators)
        
        if not results:
            print("没有找到任何有效的买点数据，请检查股票代码和日期是否正确。")
            return None, {}
            
        # 转换为DataFrame
        results_df = pd.DataFrame(results)
        
        # 计算各指标的出现频率
        stats = {
            'touch_ma': results_df['touch_ma'].mean() if 'touch_ma' in results_df else 0,
            'price_stable': results_df['price_stable'].mean() if 'price_stable' in results_df else 0,
            'ma_up': results_df['ma_up'].mean() if 'ma_up' in results_df else 0,
            'money_in': results_df['money_in'].mean() if 'money_in' in results_df else 0,
            'kpattern': results_df['kpattern'].mean() if 'kpattern' in results_df else 0,
            'vol_shrink': results_df['vol_shrink'].mean() if 'vol_shrink' in results_df else 0,
            'macd_gold': results_df['macd_gold'].mean() if 'macd_gold' in results_df else 0,
            'dif_up': results_df['dif_up'].mean() if 'dif_up' in results_df else 0,
            'dea_up': results_df['dea_up'].mean() if 'dea_up' in results_df else 0,
            'k_up': results_df['k_up'].mean() if 'k_up' in results_df else 0,
            'd_up': results_df['d_up'].mean() if 'd_up' in results_df else 0,
            'j_up': results_df['j_up'].mean() if 'j_up' in results_df else 0,
            'xc': results_df['xc'].mean() if 'xc' in results_df else 0,
        }
        
        print("\n===== 技术指标出现频率 =====")
        for indicator, freq in stats.items():
            print(f"{indicator}: {freq:.2f} ({int(freq * 100)}%)")
        
        return results_df, stats
    
    def summarize_findings(self, results_df, stats):
        """
        总结买点特征，生成企稳反弹买点总结文档
        """
        # 计算各指标的重要性
        important_indicators = sorted(stats.items(), key=lambda x: x[1], reverse=True)
        
        # 生成总结文档
        summary = "# 企稳反弹买点总结\n\n"
        summary += "## 核心技术特征\n\n"
        
        for indicator, freq in important_indicators:
            if freq >= 0.5:  # 出现频率超过50%的指标
                summary += f"- **{indicator}**: 出现频率 {freq:.2f} ({int(freq * 100)}%)\n"
        
        summary += "\n## 指标详细说明\n\n"
        
        # 添加指标解释
        indicator_explanations = {
            'touch_ma': "价格触及均线：当日最低价触及MA10/MA20/MA30/MA60中的任一均线(偏离小于1%)",
            'price_stable': "价格企稳：收盘价高于前一日收盘价，最低价不低于前一日最低价的99.5%，且收盘价位于当日价格区间的上半部分",
            'ma_up': "均线上移：MA5与MA10接近(偏离小于1%)，MA10与MA20接近(偏离小于1.5%)，且MA5和MA10均上移",
            'money_in': "资金流入信号：WVAD指标低位(≤13)并持续超过10天，或WVAD低位(≤13)且变化率大于13%，或成交量放大并收盘在高位",
            'kpattern': "K线形态改善：小实体、十字星形态，或收盘价高于开盘价且下影线占比超过30%",
            'vol_shrink': "成交量缩量：成交量小于5日均量的80%，或处于5日均量的80%-120%之间",
            'macd_gold': "MACD底背离或金叉：MACD由负转正，或MACD与DIF同时上移",
            'dif_up': "DIF上移：DIF值高于前一日",
            'dea_up': "DEA上移：DEA值高于前一日",
            'k_up': "KDJ的K值上移：K值高于前一日",
            'd_up': "KDJ的D值上移：D值高于前一日",
            'j_up': "KDJ的J值上移：J值高于前一日",
            'xc': "吸筹信号：资金在底部持续流入的信号"
        }
        
        for indicator, explanation in indicator_explanations.items():
            if indicator in [ind for ind, _ in important_indicators]:
                summary += f"### {indicator}\n\n"
                summary += f"{explanation}\n\n"
        
        # 根据分析结果提出公式改进建议
        summary += "## 公式改进建议\n\n"
        
        # 根据重要性调整条件权重
        conditions = []
        for indicator, freq in important_indicators:
            if freq >= 0.8:
                conditions.append(f"- 核心条件（必要）：{indicator_explanations.get(indicator, indicator)}")
            elif freq >= 0.6:
                conditions.append(f"- 重要条件：{indicator_explanations.get(indicator, indicator)}")
            elif freq >= 0.4:
                conditions.append(f"- 辅助条件：{indicator_explanations.get(indicator, indicator)}")
        
        summary += "\n".join(conditions)
        
        # 分析结果表格
        summary += "\n\n## 分析的股票买点\n\n"
        summary += "| 股票代码 | 股票名称 | 买点日期 | 收盘价 |\n"
        summary += "|---------|---------|----------|--------|\n"
        
        for _, row in results_df.iterrows():
            summary += f"| {row['code']} | {row['name']} | {row['date']} | {row['close']:.2f} |\n"
        
        return summary

def improve_formula(stats):
    """根据统计分析结果改进公式"""
    # 读取原始公式
    with open('formula/企稳反弹买点公式.txt', 'r', encoding='utf-8') as f:
        original_formula = f.read()
    
    # 根据统计结果调整条件的重要性和权重
    # 对最重要的指标赋予更高权重
    important_indicators = sorted(stats.items(), key=lambda x: x[1], reverse=True)
    
    # 改进的公式文本
    improved_formula = original_formula
    
    # 添加额外的技术指标计算和改进的条件组合
    # 注意：这里我们保留原始公式，只在末尾添加改进的RESULT计算
    
    # 添加注释说明改进内容
    improved_formula += "\n\n// ==================== 基于数据分析的改进版 ====================\n"
    improved_formula += "// 根据多只股票实际买点的统计分析，调整各条件的权重\n\n"
    
    # 根据统计结果重新组织条件
    core_conditions = []
    important_conditions = []
    auxiliary_conditions = []
    
    # 条件映射关系
    condition_mapping = {
        'touch_ma': 'C1',         # 触及均线
        'price_stable': 'C1',     # 价格企稳
        'ma_up': 'C2',            # 均线上移
        'money_in': 'C2',         # 资金流入
        'kpattern': 'C5',         # K线形态
        'vol_shrink': 'C6',       # 成交量缩量
        'macd_gold': 'C7',        # MACD底背离
        'dif_up': 'DIF>REF(DIF,1)',
        'dea_up': 'DEA>REF(DEA,1)',
        'k_up': 'K>REF(K,1)',
        'd_up': 'D>REF(D,1)',
        'j_up': 'J>REF(J,1)',
        'xc': 'XC'
    }
    
    # 为了添加新的技术指标，补充相关计算
    improved_formula += "// 额外的技术指标\n"
    improved_formula += "K,D,J:=KDJ(CLOSE,HIGH,LOW,9,3,3);\n"
    improved_formula += "XC:=COUNT(WVAD<=13 AND COUNT(WVAD<=13,15)>10,3)>0 OR COUNT(WVAD<=13 AND (WVAD-REF(WVAD,1))/REF(WVAD,1)*100>13,3)>0;\n\n"
    
    # 将条件根据重要性分类
    for indicator, freq in important_indicators:
        if indicator in condition_mapping:
            if freq >= 0.8:
                core_conditions.append(condition_mapping[indicator])
            elif freq >= 0.6:
                important_conditions.append(condition_mapping[indicator])
            elif freq >= 0.4:
                auxiliary_conditions.append(condition_mapping[indicator])
    
    # 去重
    core_conditions = list(set(core_conditions))
    important_conditions = list(set(important_conditions) - set(core_conditions))
    auxiliary_conditions = list(set(auxiliary_conditions) - set(core_conditions) - set(important_conditions))
    
    # 添加改进的结果计算
    improved_formula += "// 根据重要性分类的条件\n"
    if core_conditions:
        improved_formula += "CORE_CONDITIONS := " + " AND ".join(core_conditions) + ";\n"
    else:
        improved_formula += "CORE_CONDITIONS := 1;\n"
    
    if important_conditions:
        improved_formula += "IMPORTANT_CONDITIONS := " + " AND ".join(important_conditions) + ";\n"
    else:
        improved_formula += "IMPORTANT_CONDITIONS := 1;\n"
    
    if auxiliary_conditions:
        improved_formula += "AUXILIARY_CONDITIONS := " + " OR ".join(auxiliary_conditions) + ";\n"
    else:
        improved_formula += "AUXILIARY_CONDITIONS := 1;\n"
    
    improved_formula += "\n// 改进版买点信号\n"
    improved_formula += "IMPROVED_RESULT := CORE_CONDITIONS AND IMPORTANT_CONDITIONS AND AUXILIARY_CONDITIONS;\n"
    improved_formula += "IMPROVED_RESULT;\n"
    
    return improved_formula

def load_buypoints_config(config_file):
    """加载买点配置文件"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载配置文件出错: {e}")
        return []

def main():
    """主函数"""
    # 加载配置
    # 修改配置文件路径为新位置
    buypoints = load_buypoints_config('config/buypoints_config.json')
    
    # 初始化分析器
    analyzer = BuyPointAnalyzer()
    
    # 分析多个买点
    results_df, stats = analyzer.analyze_multiple_buypoints(buypoints)
    
    if results_df is None:
        print("分析失败，未能获取有效数据。")
        return
    
    # 生成总结
    summary = analyzer.summarize_findings(results_df, stats)
    
    # 保存总结到文件
    with open('formula/企稳反弹买点总结.md', 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print("\n总结已保存到 formula/企稳反弹买点总结.md")
    
    # 改进公式并保存
    print("\n开始改进企稳反弹买点公式...")
    improved_formula = improve_formula(stats)
    
    with open('formula/企稳反弹买点公式_改进版.txt', 'w', encoding='utf-8') as f:
        f.write(improved_formula)
    
    print("改进的公式已保存到 formula/企稳反弹买点公式_改进版.txt")

if __name__ == "__main__":
    main() 