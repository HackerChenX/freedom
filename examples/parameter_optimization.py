#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
技术指标参数优化脚本

通过回测不同参数组合找到最优参数设置
"""

import os
import sys
import pandas as pd
import numpy as np
import itertools
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
import concurrent.futures
import time

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from indicators.emv import EMV
from indicators.intraday_volatility import IntradayVolatility
from indicators.v_shaped_reversal import VShapedReversal
from indicators.island_reversal import IslandReversal
from db.clickhouse_db import get_clickhouse_db
from utils.logger import get_logger

logger = get_logger(__name__)


def get_stock_data(stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
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
        db = get_clickhouse_db()
        
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
        
        return data
        
    except Exception as e:
        logger.error(f"获取股票数据时发生错误: {e}")
        return pd.DataFrame()


def backtest_simple_strategy(data: pd.DataFrame, signal_column: str, 
                           initial_capital: float = 100000.0) -> Dict[str, Any]:
    """
    对单一信号进行简单回测
    
    Args:
        data: 包含信号的数据
        signal_column: 信号列名
        initial_capital: 初始资金，默认为10万
        
    Returns:
        Dict[str, Any]: 回测结果
    """
    if data.empty or signal_column not in data.columns:
        return {"error": "无效的数据或信号列名"}
    
    try:
        # 初始化回测变量
        capital = initial_capital
        position = 0
        trade_history = []
        
        # 回测策略
        for i in range(1, len(data)):
            date = data.index[i]
            price = data["close"].iloc[i]
            signal = data[signal_column].iloc[i]
            
            # 买入信号
            if signal == 1 and position == 0:
                # 计算可买入的股数(简化，假设可以买入任意数量)
                shares = int(capital / price)
                if shares > 0:
                    cost = shares * price
                    capital -= cost
                    position = shares
                    
                    trade_history.append({
                        "date": date,
                        "action": "买入",
                        "price": price,
                        "shares": shares,
                        "cost": cost,
                        "capital": capital
                    })
            
            # 卖出信号
            elif signal == -1 and position > 0:
                proceeds = position * price
                capital += proceeds
                
                # 计算此次交易的收益
                buy_price = [t["price"] for t in trade_history if t["action"] == "买入"][-1]
                profit = (price - buy_price) * position
                profit_pct = (price - buy_price) / buy_price * 100
                
                trade_history.append({
                    "date": date,
                    "action": "卖出",
                    "price": price,
                    "shares": position,
                    "proceeds": proceeds,
                    "profit": profit,
                    "profit_pct": profit_pct,
                    "capital": capital
                })
                
                position = 0
        
        # 如果结束时还有持仓，按最后价格平仓
        if position > 0:
            last_date = data.index[-1]
            last_price = data["close"].iloc[-1]
            proceeds = position * last_price
            capital += proceeds
            
            # 计算此次交易的收益
            buy_price = [t["price"] for t in trade_history if t["action"] == "买入"][-1]
            profit = (last_price - buy_price) * position
            profit_pct = (last_price - buy_price) / buy_price * 100
            
            trade_history.append({
                "date": last_date,
                "action": "结束平仓",
                "price": last_price,
                "shares": position,
                "proceeds": proceeds,
                "profit": profit,
                "profit_pct": profit_pct,
                "capital": capital
            })
        
        # 计算回测结果
        total_trades = len([t for t in trade_history if t["action"] == "买入"])
        winning_trades = len([t for t in trade_history if t.get("profit", 0) > 0])
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_profit = capital - initial_capital
        total_return = total_profit / initial_capital * 100
        
        # 计算最大回撤
        peak_capital = initial_capital
        max_drawdown = 0
        
        for trade in trade_history:
            current_capital = trade["capital"] + (trade.get("shares", 0) * trade.get("price", 0))
            peak_capital = max(peak_capital, current_capital)
            drawdown = (peak_capital - current_capital) / peak_capital * 100
            max_drawdown = max(max_drawdown, drawdown)
        
        return {
            "total_profit": total_profit,
            "total_return": total_return,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "max_drawdown": max_drawdown
        }
        
    except Exception as e:
        logger.error(f"回测过程中发生错误: {e}")
        return {"error": str(e)}


def optimize_emv_parameters(data: pd.DataFrame, 
                          periods: List[int] = None, 
                          ma_periods: List[int] = None) -> pd.DataFrame:
    """
    优化EMV指标参数
    
    Args:
        data: 股票数据
        periods: 要测试的EMV周期列表
        ma_periods: 要测试的均线周期列表
        
    Returns:
        pd.DataFrame: 参数优化结果
    """
    if data.empty:
        return pd.DataFrame()
    
    # 设置默认参数范围
    if periods is None:
        periods = [7, 10, 14, 20, 30]
    if ma_periods is None:
        ma_periods = [5, 7, 9, 14, 21]
    
    # 生成参数组合
    param_combinations = list(itertools.product(periods, ma_periods))
    results = []
    
    print(f"开始EMV参数优化，共 {len(param_combinations)} 种组合...")
    
    # 并行处理参数组合
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_to_params = {
            executor.submit(
                _test_emv_parameters, data, period, ma_period
            ): (period, ma_period) for period, ma_period in param_combinations
        }
        
        for future in concurrent.futures.as_completed(future_to_params):
            params = future_to_params[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"处理参数 {params} 时发生错误: {e}")
    
    # 转换为DataFrame并按照总收益率排序
    results_df = pd.DataFrame(results)
    
    if not results_df.empty:
        results_df = results_df.sort_values(by="total_return", ascending=False)
    
    return results_df


def _test_emv_parameters(data: pd.DataFrame, period: int, ma_period: int) -> Dict[str, Any]:
    """
    测试单组EMV参数
    
    Args:
        data: 股票数据
        period: EMV周期
        ma_period: 均线周期
        
    Returns:
        Dict[str, Any]: 测试结果
    """
    try:
        # 创建EMV实例并计算指标
        emv = EMV(period=period, ma_period=ma_period)
        result = emv.get_signals(data)
        
        # 使用综合信号进行回测
        backtest_result = backtest_simple_strategy(result, "emv_combined_signal")
        
        if "error" in backtest_result:
            return {
                "period": period,
                "ma_period": ma_period,
                "error": backtest_result["error"]
            }
        
        # 返回参数和回测结果
        return {
            "period": period,
            "ma_period": ma_period,
            "total_return": backtest_result["total_return"],
            "win_rate": backtest_result["win_rate"],
            "total_trades": backtest_result["total_trades"],
            "max_drawdown": backtest_result["max_drawdown"]
        }
        
    except Exception as e:
        return {
            "period": period,
            "ma_period": ma_period,
            "error": str(e)
        }


def optimize_v_shaped_parameters(data: pd.DataFrame, 
                               decline_periods: List[int] = None, 
                               rebound_periods: List[int] = None,
                               decline_thresholds: List[float] = None,
                               rebound_thresholds: List[float] = None) -> pd.DataFrame:
    """
    优化V形反转指标参数
    
    Args:
        data: 股票数据
        decline_periods: 要测试的下跌周期列表
        rebound_periods: 要测试的反弹周期列表
        decline_thresholds: 要测试的下跌阈值列表
        rebound_thresholds: 要测试的反弹阈值列表
        
    Returns:
        pd.DataFrame: 参数优化结果
    """
    if data.empty:
        return pd.DataFrame()
    
    # 设置默认参数范围
    if decline_periods is None:
        decline_periods = [3, 5, 7, 10]
    if rebound_periods is None:
        rebound_periods = [3, 5, 7, 10]
    if decline_thresholds is None:
        decline_thresholds = [0.03, 0.05, 0.07, 0.10]
    if rebound_thresholds is None:
        rebound_thresholds = [0.03, 0.05, 0.07, 0.10]
    
    # 限制参数组合数量以避免组合爆炸
    if len(decline_periods) * len(rebound_periods) * len(decline_thresholds) * len(rebound_thresholds) > 100:
        print("警告: 参数组合过多，将限制为100组")
        # 选择部分参数组合
        param_combinations = []
        for dp, rp in itertools.product(decline_periods, rebound_periods):
            for dt, rt in itertools.product(decline_thresholds, rebound_thresholds):
                param_combinations.append((dp, rp, dt, rt))
                if len(param_combinations) >= 100:
                    break
            if len(param_combinations) >= 100:
                break
    else:
        # 生成所有参数组合
        param_combinations = list(itertools.product(
            decline_periods, rebound_periods, decline_thresholds, rebound_thresholds
        ))
    
    results = []
    
    print(f"开始V形反转参数优化，共 {len(param_combinations)} 种组合...")
    
    # 并行处理参数组合
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_to_params = {
            executor.submit(
                _test_v_shaped_parameters, data, dp, rp, dt, rt
            ): (dp, rp, dt, rt) for dp, rp, dt, rt in param_combinations
        }
        
        for future in concurrent.futures.as_completed(future_to_params):
            params = future_to_params[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"处理参数 {params} 时发生错误: {e}")
    
    # 转换为DataFrame并按照总收益率排序
    results_df = pd.DataFrame(results)
    
    if not results_df.empty:
        results_df = results_df.sort_values(by="total_return", ascending=False)
    
    return results_df


def _test_v_shaped_parameters(data: pd.DataFrame, decline_period: int, rebound_period: int,
                            decline_threshold: float, rebound_threshold: float) -> Dict[str, Any]:
    """
    测试单组V形反转参数
    
    Args:
        data: 股票数据
        decline_period: 下跌周期
        rebound_period: 反弹周期
        decline_threshold: 下跌阈值
        rebound_threshold: 反弹阈值
        
    Returns:
        Dict[str, Any]: 测试结果
    """
    try:
        # 创建V形反转实例并计算指标
        vr = VShapedReversal(
            decline_period=decline_period, 
            rebound_period=rebound_period,
            decline_threshold=decline_threshold,
            rebound_threshold=rebound_threshold
        )
        result = vr.calculate(data)
        
        # 生成信号
        signals = pd.DataFrame(index=data.index)
        signals["close"] = data["close"]
        
        # 生成简单信号：V形反转为买入信号，持有3天后卖出
        signals["v_signal"] = 0
        v_reversal = result["v_reversal"]
        
        for i in range(len(signals)):
            if v_reversal.iloc[i]:
                signals.iloc[i, signals.columns.get_loc("v_signal")] = 1
                
                # 设置3天后卖出信号
                if i + 3 < len(signals):
                    signals.iloc[i + 3, signals.columns.get_loc("v_signal")] = -1
        
        # 使用信号进行回测
        backtest_result = backtest_simple_strategy(signals, "v_signal")
        
        if "error" in backtest_result:
            return {
                "decline_period": decline_period,
                "rebound_period": rebound_period,
                "decline_threshold": decline_threshold,
                "rebound_threshold": rebound_threshold,
                "error": backtest_result["error"]
            }
        
        # 计算形态检测率和信噪比
        pattern_count = v_reversal.sum()
        pattern_ratio = pattern_count / len(data) * 100
        
        # 返回参数和回测结果
        return {
            "decline_period": decline_period,
            "rebound_period": rebound_period,
            "decline_threshold": decline_threshold,
            "rebound_threshold": rebound_threshold,
            "total_return": backtest_result["total_return"],
            "win_rate": backtest_result["win_rate"],
            "total_trades": backtest_result["total_trades"],
            "max_drawdown": backtest_result["max_drawdown"],
            "pattern_count": pattern_count,
            "pattern_ratio": pattern_ratio
        }
        
    except Exception as e:
        return {
            "decline_period": decline_period,
            "rebound_period": rebound_period,
            "decline_threshold": decline_threshold,
            "rebound_threshold": rebound_threshold,
            "error": str(e)
        }


def optimize_island_reversal_parameters(data: pd.DataFrame, 
                                     gap_thresholds: List[float] = None, 
                                     island_max_days_list: List[int] = None) -> pd.DataFrame:
    """
    优化岛型反转指标参数
    
    Args:
        data: 股票数据
        gap_thresholds: 要测试的跳空阈值列表
        island_max_days_list: 要测试的岛型最大天数列表
        
    Returns:
        pd.DataFrame: 参数优化结果
    """
    if data.empty:
        return pd.DataFrame()
    
    # 设置默认参数范围
    if gap_thresholds is None:
        gap_thresholds = [0.005, 0.01, 0.015, 0.02, 0.03]
    if island_max_days_list is None:
        island_max_days_list = [3, 5, 7, 10]
    
    # 生成参数组合
    param_combinations = list(itertools.product(gap_thresholds, island_max_days_list))
    results = []
    
    print(f"开始岛型反转参数优化，共 {len(param_combinations)} 种组合...")
    
    # 并行处理参数组合
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_to_params = {
            executor.submit(
                _test_island_reversal_parameters, data, gap_threshold, island_max_days
            ): (gap_threshold, island_max_days) for gap_threshold, island_max_days in param_combinations
        }
        
        for future in concurrent.futures.as_completed(future_to_params):
            params = future_to_params[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"处理参数 {params} 时发生错误: {e}")
    
    # 转换为DataFrame并按照总收益率排序
    results_df = pd.DataFrame(results)
    
    if not results_df.empty:
        results_df = results_df.sort_values(by="total_return", ascending=False)
    
    return results_df


def _test_island_reversal_parameters(data: pd.DataFrame, gap_threshold: float, 
                                   island_max_days: int) -> Dict[str, Any]:
    """
    测试单组岛型反转参数
    
    Args:
        data: 股票数据
        gap_threshold: 跳空阈值
        island_max_days: 岛型最大天数
        
    Returns:
        Dict[str, Any]: 测试结果
    """
    try:
        # 创建岛型反转实例并计算指标
        ir = IslandReversal(gap_threshold=gap_threshold, island_max_days=island_max_days)
        result = ir.get_signals(data)
        
        # 使用信号进行回测
        backtest_result = backtest_simple_strategy(result, "island_signal")
        
        if "error" in backtest_result:
            return {
                "gap_threshold": gap_threshold,
                "island_max_days": island_max_days,
                "error": backtest_result["error"]
            }
        
        # 计算形态检测率
        top_island_count = result["top_island_reversal"].sum()
        bottom_island_count = result["bottom_island_reversal"].sum()
        total_island_count = top_island_count + bottom_island_count
        pattern_ratio = total_island_count / len(data) * 100
        
        # 返回参数和回测结果
        return {
            "gap_threshold": gap_threshold,
            "island_max_days": island_max_days,
            "total_return": backtest_result["total_return"],
            "win_rate": backtest_result["win_rate"],
            "total_trades": backtest_result["total_trades"],
            "max_drawdown": backtest_result["max_drawdown"],
            "top_island_count": top_island_count,
            "bottom_island_count": bottom_island_count,
            "pattern_ratio": pattern_ratio
        }
        
    except Exception as e:
        return {
            "gap_threshold": gap_threshold,
            "island_max_days": island_max_days,
            "error": str(e)
        }


def main():
    """主函数"""
    # 设置测试参数
    stock_code = "000001.SZ"  # 平安银行
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365*2)).strftime("%Y-%m-%d")  # 使用2年数据
    
    print(f"使用股票 {stock_code} 从 {start_date} 到 {end_date} 的数据进行参数优化...")
    
    # 获取股票数据
    data = get_stock_data(stock_code, start_date, end_date)
    
    if data.empty:
        print(f"无法获取股票 {stock_code} 的数据，退出程序")
        return
    
    print(f"获取到 {len(data)} 条数据记录")
    
    # 创建结果目录
    results_dir = os.path.join(root_dir, "data", "optimization_results")
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 优化EMV参数
    print("\n开始优化EMV指标参数...")
    start_time = time.time()
    emv_results = optimize_emv_parameters(data)
    end_time = time.time()
    
    if not emv_results.empty:
        # 保存结果
        emv_file = os.path.join(results_dir, f"emv_optimization_{timestamp}.csv")
        emv_results.to_csv(emv_file)
        
        # 显示前5个最佳结果
        print(f"\n完成EMV参数优化，耗时 {end_time - start_time:.2f} 秒")
        print(f"最佳5组参数:")
        print(emv_results.head(5).to_string(index=False))
        print(f"详细结果已保存至: {emv_file}")
    else:
        print("EMV参数优化未产生有效结果")
    
    # 2. 优化V形反转参数
    print("\n开始优化V形反转指标参数...")
    start_time = time.time()
    v_shaped_results = optimize_v_shaped_parameters(data)
    end_time = time.time()
    
    if not v_shaped_results.empty:
        # 保存结果
        v_shaped_file = os.path.join(results_dir, f"v_shaped_optimization_{timestamp}.csv")
        v_shaped_results.to_csv(v_shaped_file)
        
        # 显示前5个最佳结果
        print(f"\n完成V形反转参数优化，耗时 {end_time - start_time:.2f} 秒")
        print(f"最佳5组参数:")
        print(v_shaped_results.head(5).to_string(index=False))
        print(f"详细结果已保存至: {v_shaped_file}")
    else:
        print("V形反转参数优化未产生有效结果")
    
    # 3. 优化岛型反转参数
    print("\n开始优化岛型反转指标参数...")
    start_time = time.time()
    island_reversal_results = optimize_island_reversal_parameters(data)
    end_time = time.time()
    
    if not island_reversal_results.empty:
        # 保存结果
        island_file = os.path.join(results_dir, f"island_reversal_optimization_{timestamp}.csv")
        island_reversal_results.to_csv(island_file)
        
        # 显示前5个最佳结果
        print(f"\n完成岛型反转参数优化，耗时 {end_time - start_time:.2f} 秒")
        print(f"最佳5组参数:")
        print(island_reversal_results.head(5).to_string(index=False))
        print(f"详细结果已保存至: {island_file}")
    else:
        print("岛型反转参数优化未产生有效结果")
    
    print("\n参数优化完成！")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序发生未处理的异常: {e}") 