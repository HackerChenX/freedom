"""
策略条件评估器

用于评估股票数据是否满足策略条件
"""

import pandas as pd
from typing import Dict, List, Any, Optional, Union
from utils.logger import get_logger
from indicators.factory import IndicatorFactory
from enums.period import Period
from db.data_manager import DataManager

logger = get_logger(__name__)

class StrategyConditionEvaluator:
    """策略条件评估器，用于评估单个股票是否满足策略条件"""
    
    def __init__(self):
        """初始化评估器"""
        self.indicator_factory = IndicatorFactory()
        self.data_manager = DataManager()
    
    def evaluate_conditions(self, conditions: List[Dict[str, Any]], data: pd.DataFrame, 
                           stock_code: str) -> Union[bool, Dict[str, Any]]:
        """
        评估股票是否满足策略条件
        
        Args:
            conditions: 策略条件列表
            data: 股票K线数据
            stock_code: 股票代码
            
        Returns:
            Union[bool, Dict[str, Any]]: 如果是布尔值则表示是否满足条件，如果是字典则包含详细评估结果
        """
        logger.debug(f"开始评估股票 {stock_code} 是否满足策略条件")
        
        # 存储每个指标条件的评估结果
        evaluation_details = {}
        
        # 确保数据非空
        if data is None or data.empty:
            logger.warning(f"股票 {stock_code} 的数据为空，无法评估")
            return {"result": False, "details": {"error": "数据为空"}}
            
        # 检查是否有OR逻辑，如果有，采用特殊处理
        has_or_logic = False
        for condition in conditions:
            if isinstance(condition, dict) and (
                (condition.get("type") == "logic" and condition.get("value", "").upper() == "OR") or
                (condition.get("logic", "").upper() == "OR")
            ):
                has_or_logic = True
                break
                
        if has_or_logic:
            logger.info(f"检测到OR逻辑，使用特殊处理方式评估")
            
            # 找出所有指标条件
            indicator_conditions = []
            for condition in conditions:
                if isinstance(condition, dict) and "indicator_id" in condition:
                    indicator_conditions.append(condition)
                elif isinstance(condition, dict) and condition.get("type") == "indicator":
                    indicator_conditions.append(condition)
            
            # 单独评估每个指标条件，收集评估结果
            all_passed = False
            passing_indicators = []
            failing_indicators = []
            
            for i, condition in enumerate(indicator_conditions):
                try:
                    indicator_id = condition.get("indicator_id", f"指标{i+1}")
                    result = self._evaluate_indicator_condition(condition, data, stock_code)
                    logger.info(f"指标条件 {i+1}/{len(indicator_conditions)} [{indicator_id}] 评估结果: {result}")
                    
                    # 记录评估结果
                    evaluation_details[indicator_id] = result
                    
                    if result:
                        logger.info(f"股票 {stock_code} 满足指标条件 {indicator_id}")
                        all_passed = True
                        passing_indicators.append(indicator_id)
                    else:
                        failing_indicators.append(indicator_id)
                except Exception as e:
                    logger.error(f"评估指标条件 {indicator_id} 时出错: {e}")
                    evaluation_details[indicator_id] = f"评估出错: {str(e)}"
                    failing_indicators.append(indicator_id)
            
            if not all_passed:
                logger.info(f"股票 {stock_code} 不满足任何指标条件")
            
            return {
                "result": all_passed, 
                "details": {
                    "passing_indicators": passing_indicators,
                    "failing_indicators": failing_indicators,
                    "indicator_results": evaluation_details
                }
            }
        
        # 标准的条件栈处理方式
        result_stack = []
        indicator_results = {}
        passing_indicators = []
        failing_indicators = []
        
        for i, condition in enumerate(conditions):
            logger.debug(f"处理条件 {i+1}/{len(conditions)}: {condition}")
            
            # 处理不同格式的逻辑操作符
            if isinstance(condition, dict) and condition.get("type") == "logic":
                # 处理 {'type': 'logic', 'value': 'OR'} 格式
                logic_op = condition.get("value", "").upper()
                
                # 如果栈为空，推入默认值
                if len(result_stack) == 0:
                    logger.warning(f"逻辑操作符 {logic_op} 前没有操作数，使用默认值False")
                    result_stack.append(False)
                    
                # 至少需要两个操作数
                if len(result_stack) < 2 and logic_op != "NOT":
                    logger.warning(f"逻辑操作符 {logic_op} 需要至少两个操作数，但栈中只有 {len(result_stack)} 个，使用默认值False")
                    result_stack.append(False)
                
                if logic_op == "AND":
                    # 弹出两个操作数并执行AND操作
                    b = result_stack.pop()
                    a = result_stack.pop()
                    result = a and b
                    logger.debug(f"逻辑运算: {a} AND {b} = {result}")
                    result_stack.append(result)
                elif logic_op == "OR":
                    # 弹出两个操作数并执行OR操作
                    b = result_stack.pop()
                    a = result_stack.pop()
                    result = a or b
                    logger.debug(f"逻辑运算: {a} OR {b} = {result}")
                    result_stack.append(result)
                elif logic_op == "NOT":
                    # 弹出一个操作数并执行NOT操作
                    if result_stack:
                        a = result_stack.pop()
                        result = not a
                        logger.debug(f"逻辑运算: NOT {a} = {result}")
                        result_stack.append(result)
                    else:
                        logger.warning(f"NOT操作符没有操作数，使用默认值False")
                        result_stack.append(False)
            elif "logic" in condition:
                # 处理旧格式的逻辑操作符
                logic_op = condition["logic"].upper()
                
                # 如果栈为空，推入默认值
                if len(result_stack) == 0:
                    logger.warning(f"逻辑操作符 {logic_op} 前没有操作数，使用默认值False")
                    result_stack.append(False)
                    
                # 至少需要两个操作数
                if len(result_stack) < 2 and logic_op != "NOT":
                    logger.warning(f"逻辑操作符 {logic_op} 需要至少两个操作数，但栈中只有 {len(result_stack)} 个，使用默认值False")
                    result_stack.append(False)
                
                if logic_op == "AND":
                    # 弹出两个操作数并执行AND操作
                    b = result_stack.pop()
                    a = result_stack.pop()
                    result = a and b
                    logger.debug(f"逻辑运算: {a} AND {b} = {result}")
                    result_stack.append(result)
                elif logic_op == "OR":
                    # 弹出两个操作数并执行OR操作
                    b = result_stack.pop()
                    a = result_stack.pop()
                    result = a or b
                    logger.debug(f"逻辑运算: {a} OR {b} = {result}")
                    result_stack.append(result)
                elif logic_op == "NOT":
                    # 弹出一个操作数并执行NOT操作
                    if result_stack:
                        a = result_stack.pop()
                        result = not a
                        logger.debug(f"逻辑运算: NOT {a} = {result}")
                        result_stack.append(result)
                    else:
                        logger.warning(f"NOT操作符没有操作数，使用默认值False")
                        result_stack.append(False)
            else:
                # 处理指标条件
                try:
                    indicator_id = condition.get("indicator_id", f"指标{i+1}")
                    indicator_result = self._evaluate_indicator_condition(condition, data, stock_code)
                    logger.debug(f"指标条件 {indicator_id} 评估结果: {indicator_result}")
                    result_stack.append(indicator_result)
                    
                    # 记录每个指标的评估结果
                    indicator_results[indicator_id] = indicator_result
                    if indicator_result:
                        passing_indicators.append(indicator_id)
                    else:
                        failing_indicators.append(indicator_id)
                except Exception as e:
                    logger.error(f"评估指标条件时出错: {e}")
                    result_stack.append(False)
                    indicator_results[indicator_id] = f"评估出错: {str(e)}"
                    failing_indicators.append(indicator_id)
        
        # 最终结果处理
        if not result_stack:
            logger.warning(f"条件评估结束后栈为空，默认返回False")
            return {"result": False, "details": {"error": "评估栈为空"}}
            
        if len(result_stack) > 1:
            logger.warning(f"条件评估结束后栈中有 {len(result_stack)} 个结果，默认进行AND操作")
            final_result = all(result_stack)
        else:
            final_result = result_stack[0]
            
        logger.debug(f"股票 {stock_code} 条件评估最终结果: {final_result}")
        
        return {
            "result": final_result,
            "details": {
                "passing_indicators": passing_indicators,
                "failing_indicators": failing_indicators,
                "indicator_results": indicator_results
            }
        }
    
    def _evaluate_indicator_condition(self, condition: Dict[str, Any], data: pd.DataFrame, 
                                     stock_code: str) -> bool:
        """
        评估指标条件
        
        Args:
            condition: 指标条件
            data: 股票K线数据
            stock_code: 股票代码
            
        Returns:
            是否满足指标条件
        """
        try:
            # 检查条件是否包含必要的字段
            if "indicator_id" not in condition and "type" not in condition:
                logger.error(f"条件缺少必要字段: {condition}")
                return False
                
            # 获取指标ID，兼容不同格式的条件
            indicator_id = None
            if "indicator_id" in condition:
                indicator_id = condition["indicator_id"]
            elif "type" in condition and condition["type"] == "indicator" and "indicator" in condition:
                indicator = condition["indicator"]
                # indicator可能是对象或包含id的字典
                if hasattr(indicator, "id"):
                    indicator_id = indicator.id
                elif isinstance(indicator, dict) and "id" in indicator:
                    indicator_id = indicator["id"]
                else:
                    logger.error(f"无法从condition中提取indicator_id: {condition}")
                    return False
            else:
                logger.error(f"无法从condition中提取indicator_id: {condition}")
                return False
                
            logger.debug(f"解析到指标ID: {indicator_id}")
                
            period = condition.get("period", Period.DAILY.value)
            parameters = condition.get("parameters", {})
            signal_type = condition.get("signal_type", "BUY")
            
            logger.debug(f"评估指标 {indicator_id} 的 {signal_type} 信号，参数: {parameters}")
            
            # 检查数据是否包含必需的列
            required_columns = ['open', 'high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                logger.error(f"数据缺少必需的列: {', '.join(missing_columns)}")
                logger.debug(f"可用列: {data.columns.tolist()}")
                
                # 尝试兼容处理，查看是否是大小写问题
                lower_cols = {col.lower(): col for col in data.columns}
                for req_col in missing_columns[:]:  # 使用副本避免在迭代中修改列表
                    if req_col.lower() in lower_cols:
                        # 添加必要列的别名
                        data[req_col] = data[lower_cols[req_col.lower()]]
                        missing_columns.remove(req_col)
                
                # 再次检查是否还有缺失列
                if missing_columns:
                    logger.error(f"经过兼容处理后，仍然缺少必需的列: {', '.join(missing_columns)}")
                    return False
            
            # 创建指标实例
            try:
                indicator = self.indicator_factory.create(indicator_id, **parameters)
                if indicator is None:
                    logger.error(f"无法创建指标 {indicator_id}，工厂返回None")
                    return False
            except Exception as e:
                logger.error(f"创建指标 {indicator_id} 失败: {str(e)}")
                return False
            
            # 计算指标并生成信号
            try:
                # 确保数据有索引并且索引是日期类型
                if not isinstance(data.index, pd.DatetimeIndex) and 'date' in data.columns:
                    data = data.set_index('date')
                    data.index = pd.to_datetime(data.index)
                
                # 保存一份数据副本，避免原始数据被修改
                data_copy = data.copy()
                
                # 先计算指标
                logger.debug(f"开始计算指标 {indicator_id}")
                try:
                    indicator_values = indicator.calculate(data_copy)
                    logger.debug(f"指标 {indicator_id} 计算成功")
                except Exception as e:
                    logger.error(f"计算指标 {indicator_id} 时出错: {str(e)}")
                    return False
                
                # 然后生成信号
                logger.debug(f"开始生成信号 {indicator_id}")
                try:
                    signals = indicator.generate_signals(data_copy)
                    logger.debug(f"指标 {indicator_id} 信号生成成功，列: {signals.columns.tolist()}")
                except Exception as e:
                    logger.error(f"生成指标 {indicator_id} 信号时出错: {str(e)}")
                    signals = pd.DataFrame(index=data_copy.index)
                    signals['buy_signal'] = False
                    signals['sell_signal'] = False
            except Exception as e:
                logger.error(f"计算指标 {indicator_id} 或生成信号时出错: {str(e)}")
                return False
            
            # 检查最后一天是否有信号
            if signals is None or signals.empty:
                logger.warning(f"指标 {indicator_id} 没有生成信号")
                return False
            
            # 获取最后一个交易日的信号
            last_date = signals.index[-1]
            logger.debug(f"最后交易日: {last_date}")
            
            # 基于信号类型和指标ID决定如何评估条件
            if signal_type == "BUY":
                # 优先检查通用买入信号
                if 'buy_signal' in signals.columns and signals.loc[last_date, 'buy_signal']:
                    return True
                
                # 检查金叉信号
                if 'golden_cross' in signals.columns and signals.loc[last_date, 'golden_cross']:
                    return True
                
                # 检查多头趋势
                if 'bull_trend' in signals.columns and signals.loc[last_date, 'bull_trend']:
                    return True
                
                # 对于KDJ指标，检查K线是否上穿D线或K>D
                if indicator_id == "KDJ":
                    # 检查指标数据中是否有K和D列
                    if hasattr(indicator, '_result') and isinstance(indicator._result, pd.DataFrame):
                        if "K" in indicator._result.columns and "D" in indicator._result.columns:
                            # 检查K > D
                            try:
                                return indicator._result.loc[last_date, "K"] > indicator._result.loc[last_date, "D"]
                            except:
                                logger.warning(f"无法比较KDJ的K和D值")
                                return False
                    return False
                
                # 对于MACD指标，检查DIF是否上穿DEA或DIF>DEA
                elif indicator_id == "MACD":
                    # 检查指标数据中是否有DIF和DEA列
                    if hasattr(indicator, '_result') and isinstance(indicator._result, pd.DataFrame):
                        if "DIF" in indicator._result.columns and "DEA" in indicator._result.columns:
                            # 检查DIF > DEA
                            try:
                                return indicator._result.loc[last_date, "DIF"] > indicator._result.loc[last_date, "DEA"]
                            except:
                                logger.warning(f"无法比较MACD的DIF和DEA值")
                                return False
                    return False
                
                # 对于MA指标，检查短期均线是否在长期均线之上
                elif indicator_id == "MA":
                    if hasattr(indicator, '_result') and isinstance(indicator._result, pd.DataFrame):
                        # 尝试找到MA列
                        ma_columns = [col for col in indicator._result.columns if col.startswith('MA')]
                        if len(ma_columns) >= 2:
                            # 假设较小的数字是短期均线
                            ma_periods = [int(col[2:]) for col in ma_columns if col[2:].isdigit()]
                            if ma_periods:
                                ma_short = f"MA{min(ma_periods)}"
                                ma_long = f"MA{max(ma_periods)}"
                                try:
                                    return indicator._result.loc[last_date, ma_short] > indicator._result.loc[last_date, ma_long]
                                except:
                                    logger.warning(f"无法比较MA的{ma_short}和{ma_long}值")
                                    return False
                    return False
                
                # 默认返回False
                logger.warning(f"无法为指标 {indicator_id} 评估买入信号")
                return False
            
            elif signal_type == "SELL":
                # 优先检查通用卖出信号
                if 'sell_signal' in signals.columns and signals.loc[last_date, 'sell_signal']:
                    return True
                
                # 检查死叉信号
                if 'dead_cross' in signals.columns and signals.loc[last_date, 'dead_cross']:
                    return True
                
                # 检查空头趋势
                if 'bear_trend' in signals.columns and signals.loc[last_date, 'bear_trend']:
                    return True
                
                # 对于特定指标的处理类似于买入信号，但条件相反
                if indicator_id == "KDJ":
                    if hasattr(indicator, '_result') and isinstance(indicator._result, pd.DataFrame):
                        if "K" in indicator._result.columns and "D" in indicator._result.columns:
                            try:
                                return indicator._result.loc[last_date, "K"] < indicator._result.loc[last_date, "D"]
                            except:
                                logger.warning(f"无法比较KDJ的K和D值")
                                return False
                    return False
                
                elif indicator_id == "MACD":
                    if hasattr(indicator, '_result') and isinstance(indicator._result, pd.DataFrame):
                        if "DIF" in indicator._result.columns and "DEA" in indicator._result.columns:
                            try:
                                return indicator._result.loc[last_date, "DIF"] < indicator._result.loc[last_date, "DEA"]
                            except:
                                logger.warning(f"无法比较MACD的DIF和DEA值")
                                return False
                    return False
                
                # 默认返回False
                logger.warning(f"无法为指标 {indicator_id} 评估卖出信号")
                return False
            
            else:
                logger.warning(f"不支持的信号类型: {signal_type}")
                return False
        
        except Exception as e:
            logger.error(f"评估指标条件时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False 