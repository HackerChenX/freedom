#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
技术指标组合策略示例

演示如何组合使用多个技术指标进行选股
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from indicators.factory import IndicatorFactory
# 导入所需的具体指标
from indicators.emv import EMV
from indicators.intraday_volatility import IntradayVolatility
from indicators.v_shaped_reversal import VShapedReversal
from indicators.island_reversal import IslandReversal
from indicators.time_cycle_analysis import TimeCycleAnalysis
# 量能类指标
from indicators.volume_ratio import VolumeRatio
from indicators.obv import OBV
from indicators.vosc import VOSC
from indicators.mfi import MFI
from indicators.vr import VR
from indicators.pvt import PVT
# 趋势类指标
from indicators.platform_breakout import PlatformBreakout
# 经典指标
from indicators.macd import MACD
from indicators.kdj import KDJ
from indicators.rsi import RSI
from indicators.boll import BOLL
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


def apply_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    应用多个技术指标到数据
    
    Args:
        data: 股票数据
    
    Returns:
        pd.DataFrame: 包含指标结果的数据
    """
    if data.empty:
        return data
    
    # 初始化结果数据框
    result = data.copy()
    
    try:
        # 第一组：新增指标
        # 1. 应用EMV指标
        emv = EMV(period=14, ma_period=9)
        result = emv.get_signals(result)
        
        # 2. 应用日内波动率指标
        iv = IntradayVolatility(smooth_period=5)
        result = iv.get_market_phase(result)
        
        # 3. 应用V形反转指标
        vr = VShapedReversal(decline_period=5, rebound_period=5)
        result = vr.get_reversal_strength(result)
        
        # 4. 应用岛型反转指标
        ir = IslandReversal(gap_threshold=0.01, island_max_days=5)
        result = ir.get_signals(result)
        
        # 5. 应用时间周期分析 (仅当数据足够长时)
        if len(data) >= 252:
            tca = TimeCycleAnalysis(min_cycle_days=10, max_cycle_days=252, n_cycles=3)
            tca_result = tca.get_signals(result)
            result["cycle_signal"] = tca_result["cycle_signal"]
            
            # 获取当前周期阶段
            current_phase = tca.get_current_cycle_phase(result)
            logger.info(f"当前周期阶段: {current_phase}")
        
        # 第二组：量能类指标
        # 6. 量比指标
        vr_indicator = VolumeRatio(period=5)
        result = vr_indicator.get_signals(result)
        
        # 7. 能量潮指标
        obv = OBV()
        result = obv.get_signals(result)
        
        # 8. 成交量变异率
        vosc = VOSC(short_period=12, long_period=26)
        result = vosc.get_signals(result)
        
        # 9. 资金流向指标
        mfi = MFI(period=14)
        result = mfi.get_signals(result)
        
        # 10. 成交量指标
        vr_ind = VR(period=26)
        result = vr_ind.get_signals(result)
        
        # 11. 价量趋势指标
        pvt = PVT()
        result = pvt.get_signals(result)
        
        # 12. 平台突破指标
        pb = PlatformBreakout(period=20, threshold=0.03)
        result = pb.get_signals(result)
        
        # 第三组：经典指标
        # 13. MACD指标
        macd = MACD()
        result = macd.get_signals(result)
        
        # 14. KDJ指标
        kdj = KDJ()
        result = kdj.get_signals(result)
        
        # 15. RSI指标
        rsi = RSI()
        result = rsi.get_signals(result)
        
        # 16. BOLL指标
        boll = BOLL()
        result = boll.get_signals(result)
        
    except Exception as e:
        logger.error(f"应用指标时发生错误: {e}")
        
    return result


def combined_strategy(data: pd.DataFrame) -> pd.DataFrame:
    """
    综合指标策略
    
    Args:
        data: 包含多个指标的数据
        
    Returns:
        pd.DataFrame: 包含综合信号的数据
    """
    if data.empty:
        return data
    
    # 初始化结果数据框
    result = data.copy()
    result["combined_buy_score"] = 0
    
    try:
        # 第一部分：形态类指标得分 (最大12分)
        
        # 1. EMV信号 (0-2分)
        if "emv_combined_signal" in result.columns:
            result["combined_buy_score"] += (result["emv_combined_signal"] == 1).astype(int)
        if "emv" in result.columns and "emv_ma" in result.columns:
            # EMV > 0 且 EMV > EMV_MA: 加1分
            emv_positive = (result["emv"] > 0) & (result["emv"] > result["emv_ma"])
            result.loc[emv_positive, "combined_buy_score"] += 1
        
        # 2. 日内波动率信号 (0-2分)
        if "volatility_state" in result.columns:
            # 低波动 -> 加1分
            result.loc[result["volatility_state"] == "低波动", "combined_buy_score"] += 1
            # 市场阶段为"稳步上涨" -> 加1分
            if "market_phase" in result.columns:
                result.loc[result["market_phase"] == "稳步上涨", "combined_buy_score"] += 1
        
        # 3. V形反转信号 (0-3分)
        if "v_reversal" in result.columns:
            result.loc[result["v_reversal"], "combined_buy_score"] += 2
            # 强烈反转额外加1分
            if "reversal_category" in result.columns:
                result.loc[result["reversal_category"] == "强烈反转", "combined_buy_score"] += 1
        
        # 4. 岛型反转信号 (0-3分)
        if "island_signal" in result.columns:
            # 底部岛型反转信号加3分
            result.loc[result["island_signal"] == 1, "combined_buy_score"] += 3
        
        # 5. 周期信号 (0-2分)
        if "cycle_signal" in result.columns:
            result.loc[result["cycle_signal"] == 1, "combined_buy_score"] += 2
            
        # 第二部分：量能类指标得分 (最大10分)
        
        # 6. 量比指标 (0-2分)
        if "volume_ratio_signal" in result.columns:
            # 量比信号为买入时加1分
            result.loc[result["volume_ratio_signal"] == 1, "combined_buy_score"] += 1
            # 量比大于1.5时加1分
            if "volume_ratio" in result.columns:
                result.loc[result["volume_ratio"] > 1.5, "combined_buy_score"] += 1
        
        # 7. 能量潮OBV (0-2分)
        if "obv_signal" in result.columns:
            result.loc[result["obv_signal"] == 1, "combined_buy_score"] += 1
            # OBV上升趋势额外加1分
            if "obv_trend" in result.columns:
                result.loc[result["obv_trend"] == "上升", "combined_buy_score"] += 1
        
        # 8. 成交量变异率VOSC (0-2分)
        if "vosc_signal" in result.columns:
            result.loc[result["vosc_signal"] == 1, "combined_buy_score"] += 1
            # VOSC从负转正额外加1分
            if "vosc" in result.columns:
                vosc_cross_up = (result["vosc"] > 0) & (result["vosc"].shift(1) <= 0)
                result.loc[vosc_cross_up, "combined_buy_score"] += 1
        
        # 9. MFI资金流向 (0-2分)
        if "mfi_signal" in result.columns:
            result.loc[result["mfi_signal"] == 1, "combined_buy_score"] += 1
            # MFI超卖反转额外加1分
            if "mfi" in result.columns:
                mfi_oversold_reversal = (result["mfi"] < 30) & (result["mfi"] > result["mfi"].shift(1))
                result.loc[mfi_oversold_reversal, "combined_buy_score"] += 1
        
        # 10. VR成交量比率 (0-2分)
        if "vr_signal" in result.columns:
            result.loc[result["vr_signal"] == 1, "combined_buy_score"] += 1
            # VR处于合理区间额外加1分
            if "vr" in result.columns:
                vr_good_range = (result["vr"] > 100) & (result["vr"] < 250)
                result.loc[vr_good_range, "combined_buy_score"] += 1
                
        # 第三部分：趋势类指标得分 (最大8分)
        
        # 11. PVT价量趋势 (0-2分)
        if "pvt_signal" in result.columns:
            result.loc[result["pvt_signal"] == 1, "combined_buy_score"] += 1
            # PVT强劲上升趋势额外加1分
            if "pvt_trend" in result.columns:
                result.loc[result["pvt_trend"] == "强劲上升", "combined_buy_score"] += 1
        
        # 12. 平台突破指标 (0-2分)
        if "platform_breakout_signal" in result.columns:
            result.loc[result["platform_breakout_signal"] == 1, "combined_buy_score"] += 2
            
        # 13. MACD指标 (0-2分)
        if "macd_signal" in result.columns:
            result.loc[result["macd_signal"] == 1, "combined_buy_score"] += 1
            # MACD金叉额外加1分
            if "macd_histogram" in result.columns:
                macd_golden_cross = (result["macd_histogram"] > 0) & (result["macd_histogram"].shift(1) <= 0)
                result.loc[macd_golden_cross, "combined_buy_score"] += 1
        
        # 14. KDJ指标 (0-2分)
        if "kdj_signal" in result.columns:
            result.loc[result["kdj_signal"] == 1, "combined_buy_score"] += 1
            # KDJ金叉额外加1分
            if "k" in result.columns and "d" in result.columns:
                kdj_golden_cross = (result["k"] > result["d"]) & (result["k"].shift(1) <= result["d"].shift(1))
                result.loc[kdj_golden_cross, "combined_buy_score"] += 1
                
        # 第四部分：强弱指标得分 (最大6分)
        
        # 15. RSI指标 (0-3分)
        if "rsi_signal" in result.columns:
            result.loc[result["rsi_signal"] == 1, "combined_buy_score"] += 1
            # RSI超卖反转额外加1分
            if "rsi" in result.columns:
                rsi_oversold_reversal = (result["rsi"] < 30) & (result["rsi"] > result["rsi"].shift(1))
                result.loc[rsi_oversold_reversal, "combined_buy_score"] += 1
                # RSI上升趋势额外加1分
                rsi_uptrend = (result["rsi"] > result["rsi"].shift(5))
                result.loc[rsi_uptrend, "combined_buy_score"] += 1
        
        # 16. BOLL指标 (0-3分)
        if "boll_signal" in result.columns:
            result.loc[result["boll_signal"] == 1, "combined_buy_score"] += 1
            # 价格触及下轨额外加1分
            if "lower" in result.columns:
                boll_touch_lower = (result["close"] <= result["lower"]) & (result["close"].shift(1) > result["lower"].shift(1))
                result.loc[boll_touch_lower, "combined_buy_score"] += 1
                # 布林带收窄后开口额外加1分
                if "bandwidth" in result.columns:
                    boll_expansion = (result["bandwidth"] > result["bandwidth"].shift(1)) & (result["bandwidth"].shift(1) < result["bandwidth"].shift(2))
                    result.loc[boll_expansion, "combined_buy_score"] += 1
                
        # 生成指标类别得分，用于分析各类指标的贡献
        result["form_pattern_score"] = 0  # 形态类得分
        result["volume_score"] = 0        # 量能类得分
        result["trend_score"] = 0         # 趋势类得分
        result["strength_score"] = 0      # 强弱类得分
        
        # 形态类指标 (EMV, 日内波动率, V形反转, 岛型反转, 周期分析)
        for indicator in ["emv_combined_signal", "volatility_state", "v_reversal", "island_signal", "cycle_signal"]:
            if indicator in result.columns:
                if indicator == "volatility_state":
                    result.loc[result[indicator] == "低波动", "form_pattern_score"] += 1
                elif indicator == "v_reversal":
                    result.loc[result[indicator], "form_pattern_score"] += 2
                else:
                    result.loc[result[indicator] == 1, "form_pattern_score"] += 1
        
        # 量能类指标 (量比, OBV, VOSC, MFI, VR)
        for indicator in ["volume_ratio_signal", "obv_signal", "vosc_signal", "mfi_signal", "vr_signal"]:
            if indicator in result.columns:
                result.loc[result[indicator] == 1, "volume_score"] += 1
        
        # 趋势类指标 (PVT, 平台突破, MACD)
        for indicator in ["pvt_signal", "platform_breakout_signal", "macd_signal"]:
            if indicator in result.columns:
                result.loc[result[indicator] == 1, "trend_score"] += 1
                
        # 强弱类指标 (KDJ, RSI, BOLL)
        for indicator in ["kdj_signal", "rsi_signal", "boll_signal"]:
            if indicator in result.columns:
                result.loc[result[indicator] == 1, "strength_score"] += 1
        
        # 归一化类别得分
        max_form_score = 5  # 形态类最大可能得分
        max_volume_score = 5  # 量能类最大可能得分
        max_trend_score = 3  # 趋势类最大可能得分
        max_strength_score = 3  # 强弱类最大可能得分
        
        result["form_pattern_norm"] = result["form_pattern_score"] / max_form_score
        result["volume_norm"] = result["volume_score"] / max_volume_score
        result["trend_norm"] = result["trend_score"] / max_trend_score
        result["strength_norm"] = result["strength_score"] / max_strength_score
        
        # 生成最终买入信号
        result["final_buy_signal"] = False
        
        # 计算平衡得分：要求各类指标都有贡献
        result["balanced_score"] = (
            result["form_pattern_norm"] * 
            result["volume_norm"] * 
            result["trend_norm"] * 
            result["strength_norm"]
        ) * 10  # 缩放到0-10区间
        
        # 买入条件1: 总得分大于等于8分
        high_score = result["combined_buy_score"] >= 8
        
        # 买入条件2: 平衡得分大于等于1分，表示各类指标都有一定贡献
        balanced = result["balanced_score"] >= 1
        
        # 最终买入信号：总得分高或平衡得分足够
        result.loc[high_score | balanced, "final_buy_signal"] = True
        
        # 添加指标组合热力图(为了后续分析指标共性)
        result["indicator_heatmap"] = 0
        # 遍历所有指标信号列
        signal_columns = [col for col in result.columns if col.endswith('_signal')]
        for col in signal_columns:
            if col in result.columns:
                # 将1转换为正热度，-1转换为负热度
                result["indicator_heatmap"] += result[col].fillna(0)
        
    except Exception as e:
        logger.error(f"综合策略计算时发生错误: {e}")
        
    return result


def backtest_strategy(data: pd.DataFrame, initial_capital: float = 100000.0) -> Dict[str, Any]:
    """
    对综合策略进行回测
    
    Args:
        data: 包含综合信号的数据
        initial_capital: 初始资金，默认为10万
        
    Returns:
        Dict[str, Any]: 回测结果
    """
    if data.empty or "final_buy_signal" not in data.columns:
        return {"error": "无效的数据或信号"}
    
    try:
        # 初始化回测变量
        capital = initial_capital
        position = 0
        trade_history = []
        daily_returns = []
        daily_positions = []
        
        # 记录指标状态
        indicator_states_at_buy = []
        
        # 回测策略
        for i in range(1, len(data)):
            date = data.index[i]
            price = data["close"].iloc[i]
            
            # 记录每日持仓和收益率
            if i > 0:
                prev_price = data["close"].iloc[i-1]
                price_return = (price - prev_price) / prev_price
                position_return = price_return * position if position > 0 else 0
                daily_returns.append(position_return)
                daily_positions.append(position * price + capital)
            
            # 买入信号
            if data["final_buy_signal"].iloc[i] and position == 0:
                # 计算可买入的股数(简化，假设可以买入任意数量)
                shares = int(capital / price)
                if shares > 0:
                    cost = shares * price
                    capital -= cost
                    position = shares
                    
                    # 记录买入时的指标状态
                    indicator_state = {}
                    
                    # 记录所有指标的状态
                    signal_columns = [col for col in data.columns if col.endswith('_signal')]
                    for col in signal_columns:
                        if col in data.columns:
                            indicator_state[col] = data[col].iloc[i]
                    
                    # 记录类别得分
                    indicator_state["form_pattern_score"] = data["form_pattern_score"].iloc[i]
                    indicator_state["volume_score"] = data["volume_score"].iloc[i]
                    indicator_state["trend_score"] = data["trend_score"].iloc[i]
                    indicator_state["strength_score"] = data["strength_score"].iloc[i]
                    indicator_state["combined_buy_score"] = data["combined_buy_score"].iloc[i]
                    indicator_state["balanced_score"] = data["balanced_score"].iloc[i]
                    
                    trade_history.append({
                        "date": date,
                        "action": "买入",
                        "price": price,
                        "shares": shares,
                        "cost": cost,
                        "capital": capital,
                        "indicator_state": indicator_state
                    })
                    
                    indicator_states_at_buy.append(indicator_state)
            
            # 卖出信号：持有超过10天或得分降至4分以下
            days_held = 0
            if position > 0:
                buy_date = [t["date"] for t in trade_history if t["action"] == "买入"][-1]
                days_held = (date - buy_date).days
                
                if days_held > 10 or data["combined_buy_score"].iloc[i] <= 4:
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
        
        # 如果没有交易，返回原始资金
        if total_trades == 0:
            return {
                "initial_capital": initial_capital,
                "final_capital": initial_capital,
                "total_profit": 0,
                "total_return": 0,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "max_drawdown": 0,
                "annualized_return": 0,
                "sharpe_ratio": 0,
                "max_consecutive_losses": 0,
                "indicator_states_at_buy": []
            }
        
        # 交易统计
        trades_with_profit = [t for t in trade_history if t.get("action") in ["卖出", "结束平仓"]]
        winning_trades = len([t for t in trades_with_profit if t.get("profit", 0) > 0])
        losing_trades = len([t for t in trades_with_profit if t.get("profit", 0) <= 0])
        win_rate = winning_trades / (winning_trades + losing_trades) if (winning_trades + losing_trades) > 0 else 0
        
        # 收益率统计
        total_profit = capital - initial_capital
        total_return = total_profit / initial_capital * 100
        
        # 计算年化收益率
        start_date = data.index[0]
        end_date = data.index[-1]
        days = (end_date - start_date).days
        annualized_return = (1 + total_return / 100) ** (365 / days) - 1 if days > 0 else 0
        annualized_return *= 100  # 转换为百分比
        
        # 计算最大回撤
        peak_capital = initial_capital
        max_drawdown = 0
        
        for trade in trade_history:
            current_capital = trade["capital"] + (trade.get("shares", 0) * trade.get("price", 0))
            peak_capital = max(peak_capital, current_capital)
            drawdown = (peak_capital - current_capital) / peak_capital * 100
            max_drawdown = max(max_drawdown, drawdown)
        
        # 计算夏普比率
        if len(daily_returns) > 0:
            # 假设无风险利率为4%
            risk_free_rate = 0.04 / 365  # 日化无风险利率
            mean_return = np.mean(daily_returns)
            std_return = np.std(daily_returns)
            sharpe_ratio = (mean_return - risk_free_rate) / std_return * np.sqrt(252) if std_return > 0 else 0
        else:
            sharpe_ratio = 0
        
        # 计算最大连续亏损次数
        profit_flags = [1 if t.get("profit", 0) > 0 else -1 for t in trades_with_profit]
        max_consecutive_losses = 0
        current_consecutive_losses = 0
        
        for flag in profit_flags:
            if flag < 0:
                current_consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, current_consecutive_losses)
            else:
                current_consecutive_losses = 0
        
        # 分析指标共性
        indicator_commonality = {}
        
        # 只分析有交易的情况
        if indicator_states_at_buy:
            # 统计每个指标在交易中的状态
            all_indicators = set()
            for state in indicator_states_at_buy:
                all_indicators.update(state.keys())
            
            signal_indicators = [ind for ind in all_indicators if ind.endswith('_signal')]
            
            for indicator in signal_indicators:
                # 计算该指标为买入信号的比例
                buy_signals = [s.get(indicator, 0) for s in indicator_states_at_buy]
                buy_signal_ratio = sum(1 for sig in buy_signals if sig == 1) / len(buy_signals) if buy_signals else 0
                
                indicator_commonality[indicator] = {
                    "buy_signal_ratio": buy_signal_ratio,
                    "average_value": np.mean(buy_signals) if buy_signals else 0
                }
            
            # 分析类别得分
            for category in ["form_pattern_score", "volume_score", "trend_score", "strength_score", "combined_buy_score", "balanced_score"]:
                category_scores = [s.get(category, 0) for s in indicator_states_at_buy]
                indicator_commonality[category] = {
                    "average_score": np.mean(category_scores) if category_scores else 0,
                    "min_score": min(category_scores) if category_scores else 0,
                    "max_score": max(category_scores) if category_scores else 0
                }
        
        return {
            "initial_capital": initial_capital,
            "final_capital": capital,
            "total_profit": total_profit,
            "total_return": total_return,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "max_drawdown": max_drawdown,
            "annualized_return": annualized_return,
            "sharpe_ratio": sharpe_ratio,
            "max_consecutive_losses": max_consecutive_losses,
            "trade_history": trade_history,
            "indicator_states_at_buy": indicator_states_at_buy,
            "indicator_commonality": indicator_commonality
        }
        
    except Exception as e:
        logger.error(f"回测过程中发生错误: {e}")
        return {"error": str(e)}


def get_stock_list() -> List[str]:
    """
    获取用于测试的股票列表
    
    Returns:
        List[str]: 股票代码列表
    """
    # 简化测试，只使用一个股票
    return [
        "000001.SZ",  # 平安银行
        # "600000.SH",  # 浦发银行
        # "000333.SZ",  # 美的集团
        # "600519.SH",  # 贵州茅台
        # "000651.SZ"   # 格力电器
    ]


def save_commonality_analysis(all_results: List[Dict], all_commonality: Dict, output_dir: str = None):
    """
    保存指标共性分析结果
    
    Args:
        all_results: 所有股票的回测结果
        all_commonality: 所有指标的共性统计
        output_dir: 输出目录，默认为项目根目录下的data/result
    """
    if not all_results or not all_commonality:
        logger.warning("没有结果可保存")
        return
    
    try:
        # 设置默认输出目录
        if output_dir is None:
            output_dir = os.path.join(root_dir, "data", "result")
        
        # 确保目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 保存回测结果摘要
        results_summary = pd.DataFrame([
            {
                "股票代码": r["stock_code"],
                "总收益率(%)": r["total_return"],
                "年化收益率(%)": r["annualized_return"],
                "夏普比率": r["sharpe_ratio"],
                "交易次数": r["total_trades"],
                "胜率(%)": r["win_rate"] * 100,
                "最大回撤(%)": r["max_drawdown"],
                "最大连续亏损": r["max_consecutive_losses"]
            } for r in all_results
        ])
        
        backtest_file = os.path.join(output_dir, f"backtest_results_{timestamp}.csv")
        results_summary.to_csv(backtest_file, index=False, encoding='utf-8-sig')
        logger.info(f"回测结果已保存至: {backtest_file}")
        
        # 2. 保存指标共性分析结果
        # 计算每个指标的平均表现
        commonality_summary = []
        
        for indicator, stats in all_commonality.items():
            if stats["stocks"] > 0:
                if indicator.endswith('_signal'):
                    commonality_summary.append({
                        "指标名称": indicator,
                        "买入信号比例": stats["buy_signal_ratio_sum"] / stats["stocks"],
                        "平均值": stats["average_value_sum"] / stats["stocks"],
                        "有效股票数": stats["stocks"]
                    })
                else:
                    commonality_summary.append({
                        "指标名称": indicator,
                        "平均得分": stats["average_score_sum"] / stats["stocks"],
                        "最低得分": min(stats["min_scores"]) if stats["min_scores"] else 0,
                        "最高得分": max(stats["max_scores"]) if stats["max_scores"] else 0,
                        "有效股票数": stats["stocks"]
                    })
        
        # 将信号类指标和得分类指标分开保存
        signal_indicators = [item for item in commonality_summary if "买入信号比例" in item]
        score_indicators = [item for item in commonality_summary if "平均得分" in item]
        
        if signal_indicators:
            signal_df = pd.DataFrame(signal_indicators)
            signal_df = signal_df.sort_values(by="买入信号比例", ascending=False)
            signal_file = os.path.join(output_dir, f"signal_indicators_commonality_{timestamp}.csv")
            signal_df.to_csv(signal_file, index=False, encoding='utf-8-sig')
            logger.info(f"信号类指标共性分析已保存至: {signal_file}")
        
        if score_indicators:
            score_df = pd.DataFrame(score_indicators)
            score_df = score_df.sort_values(by="平均得分", ascending=False)
            score_file = os.path.join(output_dir, f"score_indicators_commonality_{timestamp}.csv")
            score_df.to_csv(score_file, index=False, encoding='utf-8-sig')
            logger.info(f"得分类指标共性分析已保存至: {score_file}")
        
        # 3. 保存高收益交易的指标状态
        high_profit_trades = []
        
        for result in all_results:
            for state in result.get("indicator_states_at_buy", []):
                if "trade_history" in result:
                    for trade in result["trade_history"]:
                        if trade.get("action") in ["卖出", "结束平仓"] and trade.get("profit_pct", 0) > 0:
                            high_profit_trades.append({
                                "stock_code": result["stock_code"],
                                "profit_pct": trade.get("profit_pct", 0),
                                **{k: v for k, v in state.items() if k.endswith('_signal') or k.endswith('_score')}
                            })
        
        if high_profit_trades:
            # 转换为DataFrame
            trades_df = pd.DataFrame(high_profit_trades)
            # 按收益率排序
            trades_df = trades_df.sort_values(by="profit_pct", ascending=False)
            
            trades_file = os.path.join(output_dir, f"high_profit_trades_{timestamp}.csv")
            trades_df.to_csv(trades_file, index=False, encoding='utf-8-sig')
            logger.info(f"高收益交易的指标状态已保存至: {trades_file}")
        
        # 4. 创建一个HTML报告
        html_report = os.path.join(output_dir, f"indicator_commonality_report_{timestamp}.html")
        
        with open(html_report, 'w', encoding='utf-8') as f:
            f.write('''
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>技术指标共性分析报告</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #f2f2f2; }
                    tr:nth-child(even) { background-color: #f9f9f9; }
                    h1, h2, h3 { color: #333; }
                    .section { margin-bottom: 30px; }
                </style>
            </head>
            <body>
                <h1>技术指标共性分析报告</h1>
                <p>生成时间: ''' + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '''</p>
                
                <div class="section">
                    <h2>回测结果摘要</h2>
                    <table>
                        <tr>
                            <th>股票代码</th>
                            <th>总收益率(%)</th>
                            <th>年化收益率(%)</th>
                            <th>夏普比率</th>
                            <th>交易次数</th>
                            <th>胜率(%)</th>
                            <th>最大回撤(%)</th>
                        </tr>
            ''')
            
            # 添加回测结果表格
            for r in all_results:
                f.write(f'''
                        <tr>
                            <td>{r["stock_code"]}</td>
                            <td>{r["total_return"]:.2f}</td>
                            <td>{r["annualized_return"]:.2f}</td>
                            <td>{r["sharpe_ratio"]:.2f}</td>
                            <td>{r["total_trades"]}</td>
                            <td>{r["win_rate"]*100:.2f}</td>
                            <td>{r["max_drawdown"]:.2f}</td>
                        </tr>
                ''')
            
            f.write('''
                    </table>
                </div>
                
                <div class="section">
                    <h2>指标共性分析</h2>
                    <h3>信号类指标共性分析</h3>
                    <table>
                        <tr>
                            <th>指标名称</th>
                            <th>买入信号比例</th>
                            <th>平均值</th>
                            <th>有效股票数</th>
                        </tr>
            ''')
            
            # 添加信号类指标表格
            if signal_indicators:
                signal_df = pd.DataFrame(signal_indicators)
                signal_df = signal_df.sort_values(by="买入信号比例", ascending=False)
                
                for _, row in signal_df.iterrows():
                    f.write(f'''
                            <tr>
                                <td>{row["指标名称"]}</td>
                                <td>{row["买入信号比例"]:.2f}</td>
                                <td>{row["平均值"]:.2f}</td>
                                <td>{row["有效股票数"]}</td>
                            </tr>
                    ''')
            
            f.write('''
                    </table>
                    
                    <h3>得分类指标共性分析</h3>
                    <table>
                        <tr>
                            <th>指标类别</th>
                            <th>平均得分</th>
                            <th>最低得分</th>
                            <th>最高得分</th>
                            <th>有效股票数</th>
                        </tr>
            ''')
            
            # 添加得分类指标表格
            if score_indicators:
                score_df = pd.DataFrame(score_indicators)
                score_df = score_df.sort_values(by="平均得分", ascending=False)
                
                for _, row in score_df.iterrows():
                    f.write(f'''
                            <tr>
                                <td>{row["指标名称"]}</td>
                                <td>{row["平均得分"]:.2f}</td>
                                <td>{row["最低得分"]:.2f}</td>
                                <td>{row["最高得分"]:.2f}</td>
                                <td>{row["有效股票数"]}</td>
                            </tr>
                    ''')
            
            f.write('''
                    </table>
                </div>
                
                <div class="section">
                    <h2>指标类别贡献分析</h2>
                    <table>
                        <tr>
                            <th>指标类别</th>
                            <th>平均得分</th>
                        </tr>
            ''')
            
            # 添加指标类别贡献分析
            category_scores = {
                "形态类指标 (EMV, V形反转等)": [r.get("indicator_commonality", {}).get("form_pattern_score", {}).get("average_score", 0) for r in all_results if "indicator_commonality" in r],
                "量能类指标 (量比, OBV等)": [r.get("indicator_commonality", {}).get("volume_score", {}).get("average_score", 0) for r in all_results if "indicator_commonality" in r],
                "趋势类指标 (PVT, MACD等)": [r.get("indicator_commonality", {}).get("trend_score", {}).get("average_score", 0) for r in all_results if "indicator_commonality" in r],
                "强弱类指标 (KDJ, RSI等)": [r.get("indicator_commonality", {}).get("strength_score", {}).get("average_score", 0) for r in all_results if "indicator_commonality" in r]
            }
            
            for category, scores in category_scores.items():
                if scores:
                    avg_score = np.mean(scores)
                    f.write(f'''
                            <tr>
                                <td>{category}</td>
                                <td>{avg_score:.2f}</td>
                            </tr>
                    ''')
            
            f.write('''
                    </table>
                </div>
            </body>
            </html>
            ''')
        
        logger.info(f"HTML报告已生成: {html_report}")
        print(f"分析报告已保存至: {output_dir}")
        
    except Exception as e:
        logger.error(f"保存分析结果时发生错误: {e}")


def main():
    """主函数"""
    try:
        print("开始运行综合指标回测系统...")
        
        # 执行主函数
        # 保存所有股票的回测结果
        all_results = []
        # 保存所有股票的指标共性
        all_commonality = {}
        
        # 修改main函数以返回结果
        def run_main():
            global all_results, all_commonality
            
            # 设置测试参数
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            
            print(f"测试周期: {start_date} 至 {end_date}")
            
            # 获取股票列表
            stock_list = get_stock_list()
            print(f"测试股票列表: {stock_list}")
            
            # 保存所有股票的回测结果
            local_results = []
            # 保存所有股票的指标共性
            local_commonality = {}
            
            for stock_code in stock_list:
                try:
                    print(f"\n分析股票 {stock_code}...")
                    # 获取股票数据
                    print(f"获取股票数据...")
                    data = get_stock_data(stock_code, start_date, end_date)
                    
                    if data.empty:
                        print(f"无法获取股票 {stock_code} 的数据，跳过")
                        continue
                        
                    print(f"获取到 {len(data)} 条数据记录")
                    
                    # 应用指标
                    print("计算技术指标...")
                    data_with_indicators = apply_indicators(data)
                    
                    # 应用综合策略
                    print("应用综合策略...")
                    strategy_result = combined_strategy(data_with_indicators)
                    
                    # 计算买入信号数量
                    buy_signals = strategy_result["final_buy_signal"].sum()
                    print(f"检测到 {buy_signals} 个买入信号")
                    
                    if buy_signals > 0:
                        signal_dates = strategy_result.index[strategy_result["final_buy_signal"]].tolist()
                        print(f"买入信号日期: {[d.strftime('%Y-%m-%d') for d in signal_dates[:5]]}" + 
                              ("..." if len(signal_dates) > 5 else ""))
                    
                    # 回测策略
                    print("回测策略...")
                    backtest_result = backtest_strategy(strategy_result)
                    
                    if "error" in backtest_result:
                        print(f"回测失败: {backtest_result['error']}")
                        continue
                    
                    # 打印回测摘要
                    print("\n回测结果摘要:")
                    print(f"初始资金: {backtest_result['initial_capital']:.2f}")
                    print(f"最终资金: {backtest_result['final_capital']:.2f}")
                    print(f"总收益: {backtest_result['total_profit']:.2f} ({backtest_result['total_return']:.2f}%)")
                    print(f"年化收益率: {backtest_result['annualized_return']:.2f}%")
                    print(f"夏普比率: {backtest_result['sharpe_ratio']:.2f}")
                    print(f"交易次数: {backtest_result['total_trades']}")
                    print(f"胜率: {backtest_result['win_rate']*100:.2f}%")
                    print(f"最大连续亏损次数: {backtest_result['max_consecutive_losses']}")
                    print(f"最大回撤: {backtest_result['max_drawdown']:.2f}%")
                    
                    # 保存结果
                    backtest_result["stock_code"] = stock_code
                    local_results.append(backtest_result)
                    
                    # 合并指标共性数据
                    if "indicator_commonality" in backtest_result:
                        for indicator, stats in backtest_result["indicator_commonality"].items():
                            if indicator not in local_commonality:
                                local_commonality[indicator] = {
                                    "stocks": 0,
                                    "buy_signal_ratio_sum": 0,
                                    "average_value_sum": 0,
                                    "average_score_sum": 0,
                                    "min_scores": [],
                                    "max_scores": []
                                }
                            
                            local_commonality[indicator]["stocks"] += 1
                            
                            if "buy_signal_ratio" in stats:
                                local_commonality[indicator]["buy_signal_ratio_sum"] += stats["buy_signal_ratio"]
                            if "average_value" in stats:
                                local_commonality[indicator]["average_value_sum"] += stats["average_value"]
                            if "average_score" in stats:
                                local_commonality[indicator]["average_score_sum"] += stats["average_score"]
                            if "min_score" in stats:
                                local_commonality[indicator]["min_scores"].append(stats["min_score"])
                            if "max_score" in stats:
                                local_commonality[indicator]["max_scores"].append(stats["max_score"])
                    
                except Exception as e:
                    print(f"分析股票 {stock_code} 时发生错误: {e}")
                    import traceback
                    traceback.print_exc()
            
            # 对所有结果进行汇总
            if local_results:
                print("\n\n所有股票回测结果汇总:")
                results_summary = pd.DataFrame([
                    {
                        "股票代码": r["stock_code"],
                        "总收益率(%)": r["total_return"],
                        "年化收益率(%)": r["annualized_return"],
                        "夏普比率": r["sharpe_ratio"],
                        "交易次数": r["total_trades"],
                        "胜率(%)": r["win_rate"] * 100,
                        "最大回撤(%)": r["max_drawdown"],
                        "最大连续亏损": r["max_consecutive_losses"]
                    } for r in local_results
                ])
                
                print(results_summary)
                
                # 计算策略整体表现
                avg_return = results_summary["总收益率(%)"].mean()
                avg_win_rate = results_summary["胜率(%)"].mean()
                avg_drawdown = results_summary["最大回撤(%)"].mean()
                
                print(f"\n策略整体表现:")
                print(f"平均收益率: {avg_return:.2f}%")
                print(f"平均胜率: {avg_win_rate:.2f}%")
                print(f"平均最大回撤: {avg_drawdown:.2f}%")
                
                # 分析指标共性
                if local_commonality:
                    print("\n\n指标共性分析:")
                    
                    # 计算每个指标的平均表现
                    commonality_summary = []
                    
                    for indicator, stats in local_commonality.items():
                        if stats["stocks"] > 0:
                            if indicator.endswith('_signal'):
                                commonality_summary.append({
                                    "指标名称": indicator,
                                    "买入信号比例": stats["buy_signal_ratio_sum"] / stats["stocks"],
                                    "平均值": stats["average_value_sum"] / stats["stocks"],
                                    "有效股票数": stats["stocks"]
                                })
                            else:
                                commonality_summary.append({
                                    "指标名称": indicator,
                                    "平均得分": stats["average_score_sum"] / stats["stocks"],
                                    "最低得分": min(stats["min_scores"]) if stats["min_scores"] else 0,
                                    "最高得分": max(stats["max_scores"]) if stats["max_scores"] else 0,
                                    "有效股票数": stats["stocks"]
                                })
                    
                    # 将信号类指标和得分类指标分开显示
                    signal_indicators = [item for item in commonality_summary if "买入信号比例" in item]
                    score_indicators = [item for item in commonality_summary if "平均得分" in item]
                    
                    if signal_indicators:
                        print("\n信号类指标共性分析 (按买入信号比例排序):")
                        signal_df = pd.DataFrame(signal_indicators)
                        signal_df = signal_df.sort_values(by="买入信号比例", ascending=False)
                        print(signal_df.to_string(index=False))
                    
                    if score_indicators:
                        print("\n得分类指标共性分析 (按平均得分排序):")
                        score_df = pd.DataFrame(score_indicators)
                        score_df = score_df.sort_values(by="平均得分", ascending=False)
                        print(score_df.to_string(index=False))
                    
                    # 输出指标组合的共性分析
                    print("\n指标类别贡献分析:")
                    category_scores = {
                        "形态类指标 (EMV, V形反转等)": [r.get("indicator_commonality", {}).get("form_pattern_score", {}).get("average_score", 0) for r in local_results if "indicator_commonality" in r],
                        "量能类指标 (量比, OBV等)": [r.get("indicator_commonality", {}).get("volume_score", {}).get("average_score", 0) for r in local_results if "indicator_commonality" in r],
                        "趋势类指标 (PVT, MACD等)": [r.get("indicator_commonality", {}).get("trend_score", {}).get("average_score", 0) for r in local_results if "indicator_commonality" in r],
                        "强弱类指标 (KDJ, RSI等)": [r.get("indicator_commonality", {}).get("strength_score", {}).get("average_score", 0) for r in local_results if "indicator_commonality" in r]
                    }
                    
                    for category, scores in category_scores.items():
                        if scores:
                            avg_score = np.mean(scores)
                            print(f"{category}: 平均得分 {avg_score:.2f}")
            else:
                print("没有成功完成任何股票的回测")
            
            # 更新全局结果
            all_results = local_results
            all_commonality = local_commonality
            
            return local_results, local_commonality
        
        # 运行主函数
        print("调用run_main函数...")
        results, commonality = run_main()
        
        print("运行完成，检查结果...")
        # 如果有结果，保存分析
        if results and commonality:
            print("保存分析结果...")
            save_commonality_analysis(results, commonality)
        else:
            print("没有结果可保存")
            
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序发生未处理的异常: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序发生未处理的异常: {e}") 