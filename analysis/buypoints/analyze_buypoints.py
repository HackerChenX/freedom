#!/usr/bin/python
# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
from db.container import get_container
from db.interfaces.data_access_interface import IDataAccess
from enums.kline_period import KlinePeriod
from enums.indicators import *
import json
import sys
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from utils.logger import get_logger
from utils.decorators import exception_handler, performance_monitor

logger = get_logger(__name__)


class BuyPointAnalyzer:
    """
    买点分析器 - 分析指定股票在指定日期的技术指标特征
    """
    
    def __init__(self, data_access: Optional[IDataAccess] = None):
        """
        初始化买点分析器
        
        Args:
            data_access: 数据访问接口实例，如果为None则从容器获取
        """
        logger.info("初始化买点分析器")
        container = get_container()
        self.data_access = data_access or container.resolve(IDataAccess)
        logger.info("成功连接到数据服务")
    
    @exception_handler(reraise=False, default_return=None)
    @performance_monitor(threshold_seconds=3.0)
    def analyze_stock(self, stock_code: str, buy_date: str, stock_name: str = "") -> Optional[Dict[str, Any]]:
        """
        分析指定股票在指定日期的买点特征
        
        Args:
            stock_code: 股票代码
            buy_date: 买点日期 (格式: YYYYMMDD)
            stock_name: 股票名称
        
        Returns:
            Optional[Dict[str, Any]]: 技术指标结果，失败时返回None
        """
        # 转换日期格式
        buy_date_obj = datetime.strptime(buy_date, "%Y%m%d")
        
        # 计算前后日期范围
        start_date = (buy_date_obj - timedelta(days=60)).strftime("%Y%m%d")
        end_date = (buy_date_obj + timedelta(days=10)).strftime("%Y%m%d")
        
        logger.info(f"分析 {stock_code} {stock_name} 在 {buy_date} 的买点特征...")
        logger.info(f"获取 {stock_code} 从 {start_date} 到 {end_date} 的数据...")
        
        try:
            # 从数据库获取数据
            stock_data = self.data_access.get_stock_info(
                code=stock_code, 
                level=KlinePeriod.DAILY.value, 
                start_date=start_date, 
                end_date=end_date
            )
            
            if not stock_data or len(stock_data) == 0:
                logger.warning(f"未找到 {stock_code} 的数据")
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
                logger.warning(f"未找到买点日期 {buy_date} 的数据")
                return None
            
            # 计算企稳反弹买点技术指标
            indicators = self.calculate_buy_point_indicators(df, buy_date_idx)
            
            if indicators:
                logger.info(f"成功计算 {stock_code} {stock_name} 在 {buy_date} 的技术指标")
                # 添加基本信息
                indicators['code'] = stock_code
                indicators['name'] = stock_name if stock_name else df['name'].iloc[0]
                indicators['date'] = buy_date
                indicators['industry'] = df['industry'].iloc[0]
                return indicators
            else:
                logger.warning(f"计算 {stock_code} {stock_name} 技术指标失败")
                return None
            
        except Exception as e:
            logger.error(f"分析股票 {stock_code} 时出错: {e}")
            return None
    
    @exception_handler(reraise=False, default_return={})
    @performance_monitor(threshold_seconds=2.0)
    def calculate_buy_point_indicators(self, df: pd.DataFrame, buy_date_idx: int) -> Dict[str, Any]:
        """
        计算企稳反弹买点的技术指标
        
        Args:
            df: 股票数据DataFrame
            buy_date_idx: 买点日期索引
        
        Returns:
            Dict[str, Any]: 技术指标结果
        """
        if buy_date_idx is None or buy_date_idx < 5:  # 需要至少5天的数据来计算指标
            logger.warning("数据不足，无法计算技术指标")
            return {}
        
        try:
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
                         (dif[buy_date_idx] > dea[buy_date_idx] and dif[buy_date_idx-1] <= dea[buy_date_idx-1]))
            
            # RSI指标
            rsi = RSI(close, 14)
            rsi_oversold = rsi[buy_date_idx] < 30
            
            # 计算综合评分
            score = 0
            if touch_ma_f:
                score += 15
            if price_stable:
                score += 20
            if ma_up:
                score += 15
            if money_in:
                score += 20
            if kpattern:
                score += 10
            if vol_shrink:
                score += 5
            if macd_gold:
                score += 10
            if rsi_oversold:
                score += 5
                
            # 构建结果字典
            result = {
                'touch_ma': touch_ma_f,
                'price_stable': price_stable,
                'ma_up': ma_up,
                'money_in': money_in,
                'kpattern': kpattern,
                'vol_shrink': vol_shrink,
                'macd_gold': macd_gold,
                'rsi_oversold': rsi_oversold,
                'score': score,
                'ma5': ma5[buy_date_idx],
                'ma10': ma10[buy_date_idx],
                'ma20': ma20[buy_date_idx],
                'ma30': ma30[buy_date_idx],
                'ma60': ma60[buy_date_idx],
                'close': close[buy_date_idx],
                'volume': volume[buy_date_idx],
                'vol5': vol5[buy_date_idx],
                'dif': dif[buy_date_idx],
                'dea': dea[buy_date_idx],
                'macd': macd[buy_date_idx],
                'kdj_k': kdj_k[buy_date_idx],
                'kdj_d': kdj_d[buy_date_idx],
                'kdj_j': kdj_j[buy_date_idx],
                'rsi': rsi[buy_date_idx],
                'wvad': wvad[buy_date_idx]
            }
            
            return result
            
        except Exception as e:
            logger.error(f"计算技术指标失败: {e}")
            return {}
    
    @exception_handler(reraise=False, default_return=[])
    @performance_monitor(threshold_seconds=10.0)
    def analyze_multiple_buypoints(self, buypoints_list: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        批量分析多个买点
        
        Args:
            buypoints_list: 买点列表，每个元素包含code、date、name等字段
            
        Returns:
            List[Dict[str, Any]]: 分析结果列表
        """
        logger.info(f"开始批量分析 {len(buypoints_list)} 个买点...")
        
        results = []
        success_count = 0
        
        for i, buypoint in enumerate(buypoints_list, 1):
            try:
                code = buypoint.get('code', '')
                date = buypoint.get('date', '')
                name = buypoint.get('name', '')
                
                if not code or not date:
                    logger.warning(f"第 {i} 个买点数据不完整，跳过")
                    continue
                
                logger.info(f"分析第 {i}/{len(buypoints_list)} 个买点: {code} {name} {date}")
                
                # 分析单个买点
                result = self.analyze_stock(code, date, name)
                
                if result:
                    results.append(result)
                    success_count += 1
                    logger.info(f"买点 {code} {date} 分析成功，评分: {result.get('score', 0)}")
                else:
                    logger.warning(f"买点 {code} {date} 分析失败")
                    
            except Exception as e:
                logger.error(f"分析第 {i} 个买点时出错: {e}")
                continue
        
        logger.info(f"批量分析完成，成功: {success_count}/{len(buypoints_list)}")
        return results
    
    @exception_handler(reraise=False, default_return={})
    def summarize_findings(self, results_df: pd.DataFrame, stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        总结分析发现
        
        Args:
            results_df: 分析结果DataFrame
            stats: 统计信息
            
        Returns:
            Dict[str, Any]: 总结信息
        """
        try:
            if results_df.empty:
                return {
                    'total_count': 0,
                    'summary': '没有有效的分析结果'
                }
            
            summary = {
                'total_count': len(results_df),
                'avg_score': results_df['score'].mean(),
                'high_score_count': len(results_df[results_df['score'] >= 70]),
                'medium_score_count': len(results_df[(results_df['score'] >= 50) & (results_df['score'] < 70)]),
                'low_score_count': len(results_df[results_df['score'] < 50]),
                'touch_ma_rate': results_df['touch_ma'].mean() * 100,
                'price_stable_rate': results_df['price_stable'].mean() * 100,
                'ma_up_rate': results_df['ma_up'].mean() * 100,
                'money_in_rate': results_df['money_in'].mean() * 100,
                'kpattern_rate': results_df['kpattern'].mean() * 100,
                'vol_shrink_rate': results_df['vol_shrink'].mean() * 100,
                'macd_gold_rate': results_df['macd_gold'].mean() * 100,
                'rsi_oversold_rate': results_df['rsi_oversold'].mean() * 100
            }
            
            # 添加行业分布
            if 'industry' in results_df.columns:
                industry_counts = results_df['industry'].value_counts()
                summary['top_industries'] = industry_counts.head(5).to_dict()
            
            # 添加评分分布
            summary['score_distribution'] = {
                '90-100': len(results_df[results_df['score'] >= 90]),
                '80-89': len(results_df[(results_df['score'] >= 80) & (results_df['score'] < 90)]),
                '70-79': len(results_df[(results_df['score'] >= 70) & (results_df['score'] < 80)]),
                '60-69': len(results_df[(results_df['score'] >= 60) & (results_df['score'] < 70)]),
                '50-59': len(results_df[(results_df['score'] >= 50) & (results_df['score'] < 60)]),
                '0-49': len(results_df[results_df['score'] < 50])
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"总结分析发现失败: {e}")
            return {'error': str(e)}


def improve_formula(stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    根据统计结果改进公式权重
    
    Args:
        stats: 统计信息
        
    Returns:
        Dict[str, Any]: 改进建议
    """
    try:
        improvements = {
            'current_weights': {
                'touch_ma': 15,
                'price_stable': 20,
                'ma_up': 15,
                'money_in': 20,
                'kpattern': 10,
                'vol_shrink': 5,
                'macd_gold': 10,
                'rsi_oversold': 5
            },
            'suggested_adjustments': {},
            'reasoning': []
        }
        
        # 根据各指标的成功率调整权重
        if 'touch_ma_rate' in stats and stats['touch_ma_rate'] > 80:
            improvements['suggested_adjustments']['touch_ma'] = 18
            improvements['reasoning'].append("触及均线指标成功率高，建议增加权重")
        
        if 'price_stable_rate' in stats and stats['price_stable_rate'] > 75:
            improvements['suggested_adjustments']['price_stable'] = 25
            improvements['reasoning'].append("价格企稳指标表现优秀，建议增加权重")
        
        if 'money_in_rate' in stats and stats['money_in_rate'] < 40:
            improvements['suggested_adjustments']['money_in'] = 15
            improvements['reasoning'].append("资金流入指标成功率偏低，建议降低权重")
        
        if 'macd_gold_rate' in stats and stats['macd_gold_rate'] > 60:
            improvements['suggested_adjustments']['macd_gold'] = 15
            improvements['reasoning'].append("MACD金叉指标表现良好，建议增加权重")
        
        return improvements
        
    except Exception as e:
        logger.error(f"改进公式失败: {e}")
        return {'error': str(e)}


def load_buypoints_config(config_file: str) -> List[Dict[str, str]]:
    """
    从配置文件加载买点列表
    
    Args:
        config_file: 配置文件路径
        
    Returns:
        List[Dict[str, str]]: 买点列表
    """
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            if config_file.endswith('.json'):
                data = json.load(f)
            else:
                # 假设是CSV格式
                import csv
                reader = csv.DictReader(f)
                data = list(reader)
        
        return data
        
    except Exception as e:
        logger.error(f"加载买点配置失败: {e}")
        return []


@exception_handler(reraise=False)
def main():
    """
    主函数 - 运行买点分析
    """
    try:
        # 创建买点分析器
        analyzer = BuyPointAnalyzer()
        
        # 示例：分析单个买点
        result = analyzer.analyze_stock("000001", "20231201", "平安银行")
        
        if result:
            print("分析结果:")
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print("分析失败")
            
    except Exception as e:
        logger.error(f"主函数执行失败: {e}")
        print(f"程序执行出错: {e}")


if __name__ == "__main__":
    main() 