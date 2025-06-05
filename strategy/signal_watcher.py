"""
观察信号处理器模块

负责识别接近触发条件但尚未满足的股票，提供趋势预测和信号强度梯度评估
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

from indicators.factory import IndicatorFactory
from enums.period import Period
from db.data_manager import DataManager
from utils.logger import get_logger
from utils.decorators import performance_monitor, log_calls, safe_run

logger = get_logger(__name__)


class SignalWatcher:
    """
    观察信号处理器
    
    识别接近触发条件但尚未满足的股票，提供趋势预测和信号强度梯度评估
    """
    
    def __init__(self, data_manager: Optional[DataManager] = None,
                 indicator_factory: Optional[IndicatorFactory] = None):
        """
        初始化观察信号处理器
        
        Args:
            data_manager: 数据管理器实例
            indicator_factory: 指标工厂实例
        """
        self.data_manager = data_manager or DataManager()
        self.indicator_factory = indicator_factory or IndicatorFactory()
        
    @performance_monitor(threshold=5.0)
    @log_calls(level="info")
    def find_watch_signals(self, strategy_plan: Dict[str, Any],
                          stock_pool: List[str],
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None,
                          threshold: float = 0.7,
                          trend_days: int = 3) -> pd.DataFrame:
        """
        寻找观察信号
        
        Args:
            strategy_plan: 策略执行计划
            stock_pool: 股票池
            start_date: 开始日期，默认为近60天
            end_date: 结束日期，默认为当前日期
            threshold: 接近度阈值（0-1之间），表示达到多少比例的条件
            trend_days: 趋势分析天数
            
        Returns:
            观察信号结果DataFrame
        """
        # 设置默认日期
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if start_date is None:
            # 默认取近60天数据（需要更长时间来分析趋势）
            start_date = (datetime.strptime(end_date, "%Y-%m-%d") - 
                        timedelta(days=60)).strftime("%Y-%m-%d")
        
        if not stock_pool:
            logger.warning("股票池为空，无法寻找观察信号")
            return pd.DataFrame()
            
        logger.info(f"将在 {len(stock_pool)} 只股票中寻找观察信号")
        
        # 提取策略条件
        conditions = strategy_plan.get("conditions", [])
        if not conditions:
            logger.warning("策略没有定义条件，无法寻找观察信号")
            return pd.DataFrame()
        
        # 执行观察信号逻辑
        watch_results = []
        total_stocks = len(stock_pool)
        
        for i, stock_code in enumerate(stock_pool):
            if (i + 1) % 50 == 0:
                logger.info(f"处理进度: {i+1}/{total_stocks}")
                
            # 分析该股票的观察信号
            watch_result = self._analyze_stock_watch_signals(
                stock_code, 
                conditions,
                start_date,
                end_date,
                threshold,
                trend_days
            )
            
            if watch_result:
                # 获取股票基本信息
                stock_info = self.data_manager.get_stock_info(stock_code, 'day')
                stock_name = stock_info.name if stock_info else ""
                
                # 添加到结果
                watch_result["stock_code"] = stock_code
                watch_result["stock_name"] = stock_name
                watch_result["strategy_id"] = strategy_plan.get("strategy_id", "")
                watch_result["watch_date"] = end_date
                
                watch_results.append(watch_result)
        
        # 转换为DataFrame
        result_df = pd.DataFrame(watch_results)
        
        # 如果结果为空，返回空DataFrame
        if result_df.empty:
            logger.info("未找到观察信号")
            return result_df
            
        # 按照接近度降序排序
        result_df = result_df.sort_values(by=["proximity", "trend_score"], ascending=[False, False])
        
        # 添加排名
        result_df["rank"] = range(1, len(result_df) + 1)
        
        logger.info(f"共找到 {len(result_df)} 只具有观察信号的股票")
        return result_df
    
    @safe_run(default_return=None)
    def _analyze_stock_watch_signals(self, stock_code: str,
                                   conditions: List[Dict[str, Any]],
                                   start_date: str,
                                   end_date: str,
                                   threshold: float,
                                   trend_days: int) -> Optional[Dict[str, Any]]:
        """
        分析单只股票的观察信号
        
        Args:
            stock_code: 股票代码
            conditions: 条件列表
            start_date: 开始日期
            end_date: 结束日期
            threshold: 接近度阈值
            trend_days: 趋势分析天数
            
        Returns:
            观察信号结果字典或None
        """
        # 分析每个条件的完成度
        condition_results = []
        total_conditions = 0
        satisfied_conditions = 0
        total_proximity = 0.0
        
        for condition in conditions:
            if condition.get("type") == "logic":
                # 逻辑运算符不参与观察信号分析
                continue
                
            total_conditions += 1
            
            # 获取指标对象
            indicator_id = condition.get("indicator_id")
            indicator_params = condition.get("parameters", {})
            
            try:
                indicator = self.indicator_factory.create(indicator_id, **indicator_params)
            except ValueError as e:
                logger.error(f"创建指标 {indicator_id} 参数错误: {e}")
                continue
            except KeyError as e:
                logger.error(f"指标 {indicator_id} 不存在: {e}")
                continue
            except Exception as e:
                logger.error(f"创建指标 {indicator_id} 时出错: {str(e)}", exc_info=True)
                continue
            
            # 获取周期
            period_str = condition.get("period")
            period = Period.from_string(period_str)
            
            # 获取信号类型
            signal_type = condition.get("signal_type", "DEFAULT")
            
            # 获取数据
            data = self.data_manager.get_kline_data(
                stock_code=stock_code,
                period=period,
                start_date=start_date,
                end_date=end_date
            )
            
            if data.empty:
                continue
                
            # 计算指标
            indicator_values = indicator.calculate(data)
            
            # 生成信号
            signals = indicator.generate_signals(indicator_values)
            
            if signals.empty:
                continue
                
            # 分析信号接近度和趋势
            proximity, is_satisfied, trend_direction = self._analyze_signal_proximity(
                signals, signal_type, trend_days
            )
            
            if is_satisfied:
                satisfied_conditions += 1
                
            total_proximity += proximity
            
            # 记录条件结果
            condition_results.append({
                "indicator_id": indicator_id,
                "period": period.name,
                "signal_type": signal_type,
                "proximity": float(proximity),
                "satisfied": bool(is_satisfied),
                "trend": trend_direction
            })
        
        # 如果没有有效条件，返回None
        if total_conditions == 0:
            return None
            
        # 计算整体接近度
        overall_proximity = total_proximity / total_conditions
        
        # 计算条件完成比例
        completion_ratio = satisfied_conditions / total_conditions
        
        # 如果已满足所有条件或接近度低于阈值，不产生观察信号
        if completion_ratio == 1.0 or overall_proximity < threshold:
            return None
            
        # 计算趋势得分
        trend_score = self._calculate_trend_score(condition_results)
        
        # 返回观察信号结果
        return {
            "proximity": overall_proximity,
            "completion_ratio": completion_ratio,
            "trend_score": trend_score,
            "condition_count": total_conditions,
            "satisfied_count": satisfied_conditions,
            "condition_details": json.dumps(condition_results)
        }
    
    def _analyze_signal_proximity(self, signals: pd.DataFrame, 
                                signal_type: str, 
                                trend_days: int) -> Tuple[float, bool, str]:
        """
        分析信号接近度和趋势
        
        Args:
            signals: 信号DataFrame
            signal_type: 信号类型
            trend_days: 趋势分析天数
            
        Returns:
            (proximity, is_satisfied, trend_direction): 
            - proximity: 接近度（0-1之间）
            - is_satisfied: 是否已满足条件
            - trend_direction: 趋势方向（"up", "down", "flat"）
        """
        if signals.empty:
            return 0.0, False, "flat"
            
        # 映射信号类型到信号列
        signal_map = {
            "BUY": "buy_signal",
            "SELL": "sell_signal",
            "WATCH": "watch_signal",
            "CROSS_OVER": "golden_cross",
            "CROSS_UNDER": "dead_cross",
            "OVERBOUGHT": "overbought",
            "OVERSOLD": "oversold"
        }
        
        # 默认信号为BUY
        signal_column = signal_map.get(signal_type, "buy_signal")
        
        # 如果信号列不存在，尝试直接使用信号类型列
        if signal_column not in signals.columns and signal_type in signals.columns:
            signal_column = signal_type
            
        # 如果信号列仍不存在，检查是否有近似列
        if signal_column not in signals.columns:
            for col in signals.columns:
                if signal_column.lower() in col.lower():
                    signal_column = col
                    break
                    
        # 如果仍找不到信号列，返回默认值
        if signal_column not in signals.columns:
            return 0.0, False, "flat"
            
        # 获取最新信号
        latest_signal = signals[signal_column].iloc[-1]
        
        # 检查是否已满足条件
        is_satisfied = bool(latest_signal)
        
        # 如果已满足条件，接近度为1.0
        if is_satisfied:
            return 1.0, True, "flat"
            
        # 分析接近度（使用信号强度或其他指标）
        proximity = 0.0
        
        if "signal_strength" in signals.columns:
            # 使用信号强度作为接近度
            proximity = float(signals["signal_strength"].iloc[-1])
        elif "distance" in signals.columns:
            # 使用距离的倒数作为接近度
            distance = signals["distance"].iloc[-1]
            if distance > 0:
                proximity = 1.0 / (1.0 + distance)
        else:
            # 尝试从信号列的值推断接近度
            try:
                # 如果信号列是布尔值，检查它是否接近变为True
                signal_values = signals[signal_column]
                if signal_values.dtype == bool:
                    # 不能直接计算，使用相关列估计
                    # 这里需要根据具体指标调整
                    proximity = 0.5  # 默认中等接近度
            except Exception:
                proximity = 0.0
        
        # 将接近度限制在0-1之间
        proximity = max(0.0, min(1.0, proximity))
        
        # 分析趋势
        trend_direction = "flat"
        
        if trend_days > 0 and len(signals) >= trend_days + 1:
            # 计算最近几天的信号变化
            if "signal_strength" in signals.columns:
                # 使用信号强度的变化来判断趋势
                recent_strengths = signals["signal_strength"].iloc[-trend_days-1:].values
                if len(recent_strengths) > 1:
                    diff = recent_strengths[-1] - recent_strengths[0]
                    if diff > 0.05:  # 有明显上升
                        trend_direction = "up"
                    elif diff < -0.05:  # 有明显下降
                        trend_direction = "down"
            else:
                # 尝试使用其他列判断趋势
                for potential_col in ["distance", "value", "close"]:
                    if potential_col in signals.columns:
                        recent_values = signals[potential_col].iloc[-trend_days-1:].values
                        if len(recent_values) > 1:
                            diff = recent_values[-1] - recent_values[0]
                            if abs(diff) / (recent_values[0] + 1e-10) > 0.02:  # 变化超过2%
                                trend_direction = "up" if diff > 0 else "down"
                            break
        
        return proximity, is_satisfied, trend_direction
    
    def _calculate_trend_score(self, condition_results: List[Dict[str, Any]]) -> float:
        """
        计算趋势得分
        
        正值表示向上趋势（即接近满足条件），负值表示向下趋势（即远离满足条件）
        
        Args:
            condition_results: 条件结果列表
            
        Returns:
            趋势得分（-1到1之间）
        """
        if not condition_results:
            return 0.0
            
        # 统计趋势方向
        up_count = sum(1 for c in condition_results if c["trend"] == "up")
        down_count = sum(1 for c in condition_results if c["trend"] == "down")
        total_count = len(condition_results)
        
        # 计算趋势得分
        trend_score = (up_count - down_count) / total_count
        
        return trend_score
    
    @log_calls(level="info")
    def get_watch_signal_details(self, stock_code: str, 
                               strategy_plan: Dict[str, Any],
                               start_date: Optional[str] = None,
                               end_date: Optional[str] = None,
                               lookback_days: int = 30) -> Dict[str, Any]:
        """
        获取特定股票的观察信号详细信息
        
        Args:
            stock_code: 股票代码
            strategy_plan: 策略执行计划
            start_date: 开始日期
            end_date: 结束日期
            lookback_days: 回溯天数
            
        Returns:
            观察信号详细信息字典
        """
        # 设置默认日期
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if start_date is None:
            # 默认取近N天数据
            start_date = (datetime.strptime(end_date, "%Y-%m-%d") - 
                        timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        
        # 提取策略条件
        conditions = strategy_plan.get("conditions", [])
        if not conditions:
            return {"error": "策略没有定义条件"}
        
        # 获取股票基本信息
        stock_info = self.data_manager.get_stock_info(stock_code, 'day')
        if not stock_info or not stock_info.name:
            return {"error": f"未找到股票 {stock_code} 的基本信息"}
            
        stock_name = stock_info.name
        
        # 分析每个条件的详细信息
        condition_details = []
        
        for condition in conditions:
            if condition.get("type") == "logic":
                # 逻辑运算符不参与观察信号分析
                continue
                
            # 获取指标对象
            indicator_id = condition.get("indicator_id")
            indicator_params = condition.get("parameters", {})
            
            try:
                indicator = self.indicator_factory.create(indicator_id, **indicator_params)
            except ValueError as e:
                logger.error(f"创建指标 {indicator_id} 参数错误: {e}")
                continue
            
            # 获取周期
            period_str = condition.get("period")
            period = Period.from_string(period_str)
            
            # 获取信号类型
            signal_type = condition.get("signal_type", "DEFAULT")
            
            # 获取数据
            data = self.data_manager.get_kline_data(
                stock_code=stock_code,
                period=period,
                start_date=start_date,
                end_date=end_date
            )
            
            if data.empty:
                continue
                
            # 计算指标
            indicator_values = indicator.calculate(data)
            
            # 生成信号
            signals = indicator.generate_signals(indicator_values)
            
            if signals.empty:
                continue
                
            # 分析信号历史
            signal_history = self._analyze_signal_history(signals, signal_type, lookback_days)
            
            # 提取关键数据用于图表展示
            chart_data = self._extract_chart_data(data, indicator_values, signals, signal_type)
            
            # 记录条件详情
            condition_details.append({
                "indicator_id": indicator_id,
                "period": period.name,
                "signal_type": signal_type,
                "parameters": indicator_params,
                "signal_history": signal_history,
                "chart_data": chart_data
            })
        
        # 返回详细信息
        return {
            "stock_code": stock_code,
            "stock_name": stock_name,
            "strategy_id": strategy_plan.get("strategy_id", ""),
            "start_date": start_date,
            "end_date": end_date,
            "lookback_days": lookback_days,
            "condition_details": condition_details
        }
    
    def _analyze_signal_history(self, signals: pd.DataFrame, 
                             signal_type: str, 
                             lookback_days: int) -> List[Dict[str, Any]]:
        """
        分析信号历史
        
        Args:
            signals: 信号DataFrame
            signal_type: 信号类型
            lookback_days: 回溯天数
            
        Returns:
            信号历史列表
        """
        # 映射信号类型到信号列
        signal_map = {
            "BUY": "buy_signal",
            "SELL": "sell_signal",
            "WATCH": "watch_signal",
            "CROSS_OVER": "golden_cross",
            "CROSS_UNDER": "dead_cross",
            "OVERBOUGHT": "overbought",
            "OVERSOLD": "oversold"
        }
        
        # 默认信号为BUY
        signal_column = signal_map.get(signal_type, "buy_signal")
        
        # 如果信号列不存在，尝试直接使用信号类型列
        if signal_column not in signals.columns and signal_type in signals.columns:
            signal_column = signal_type
            
        # 如果信号列不存在，检查是否有近似列
        if signal_column not in signals.columns:
            for col in signals.columns:
                if signal_column.lower() in col.lower():
                    signal_column = col
                    break
        
        # 如果仍找不到信号列，返回空列表
        if signal_column not in signals.columns:
            return []
            
        # 限制回溯天数
        if len(signals) > lookback_days:
            signals = signals.iloc[-lookback_days:]
            
        # 分析信号历史
        history = []
        
        for i, (idx, row) in enumerate(signals.iterrows()):
            signal_value = bool(row[signal_column]) if signal_column in row else False
            signal_strength = float(row["signal_strength"]) if "signal_strength" in row else 0.0
            
            # 记录信号点
            if "date" in row:
                date_str = row["date"]
                if isinstance(date_str, np.datetime64):
                    date_str = pd.Timestamp(date_str).strftime("%Y-%m-%d")
            else:
                date_str = f"Day {i+1}"
                
            history.append({
                "date": date_str,
                "signal": signal_value,
                "strength": signal_strength
            })
            
        return history
    
    def _extract_chart_data(self, data: pd.DataFrame, 
                          indicator_values: pd.DataFrame,
                          signals: pd.DataFrame,
                          signal_type: str) -> Dict[str, Any]:
        """
        提取用于图表展示的数据
        
        Args:
            data: K线数据
            indicator_values: 指标计算结果
            signals: 信号结果
            signal_type: 信号类型
            
        Returns:
            图表数据字典
        """
        # 映射信号类型到信号列
        signal_map = {
            "BUY": "buy_signal",
            "SELL": "sell_signal",
            "WATCH": "watch_signal",
            "CROSS_OVER": "golden_cross",
            "CROSS_UNDER": "dead_cross",
            "OVERBOUGHT": "overbought",
            "OVERSOLD": "oversold"
        }
        
        # 默认信号为BUY
        signal_column = signal_map.get(signal_type, "buy_signal")
        
        # 如果信号列不存在，尝试直接使用信号类型列
        if signal_column not in signals.columns and signal_type in signals.columns:
            signal_column = signal_type
            
        # 准备数据
        chart_data = {
            "dates": data["date"].astype(str).tolist() if "date" in data.columns else [],
            "prices": {
                "open": data["open"].tolist() if "open" in data.columns else [],
                "high": data["high"].tolist() if "high" in data.columns else [],
                "low": data["low"].tolist() if "low" in data.columns else [],
                "close": data["close"].tolist() if "close" in data.columns else []
            },
            "volumes": data["volume"].tolist() if "volume" in data.columns else [],
            "indicators": {},
            "signals": {}
        }
        
        # 提取指标数据
        for column in indicator_values.columns:
            if column not in data.columns and column != "date":
                chart_data["indicators"][column] = indicator_values[column].tolist()
                
        # 提取信号数据
        for column in signals.columns:
            if column not in data.columns and column != "date":
                chart_data["signals"][column] = signals[column].tolist()
                
        return chart_data 