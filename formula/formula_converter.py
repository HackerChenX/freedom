"""
公式转换器模块

提供将通达信风格公式转换为选股策略配置的功能
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import re
import json
import uuid
from datetime import datetime

from utils.logger import get_logger
from enums.period import Period
from indicators.indicator_factory import IndicatorFactory
from formula.stock_formula import StockFormula

logger = get_logger(__name__)


class FormulaConverter:
    """
    公式转换器，将通达信风格公式转换为选股策略配置
    """
    
    def __init__(self, indicator_factory: Optional[IndicatorFactory] = None):
        """
        初始化公式转换器
        
        Args:
            indicator_factory: 指标工厂实例，用于创建和管理指标
        """
        self.indicator_factory = indicator_factory or IndicatorFactory()
        
        # 通达信函数与系统指标的映射
        self.function_map = {
            "MA": "MovingAverage",
            "EMA": "ExponentialMovingAverage",
            "SMA": "SimpleMovingAverage",
            "MACD": "MACD",
            "KDJ": "KDJ",
            "BOLL": "BollingerBands",
            "RSI": "RSI",
            "VOL": "Volume",
            "CROSS": "CrossOver",
            "REF": "Reference",
            "HHV": "HighestHigh",
            "LLV": "LowestLow",
            "COUNT": "Count",
            "SUM": "Sum",
            "ABS": "Absolute",
            "MAX": "Maximum",
            "MIN": "Minimum",
            "IF": "Condition"
        }
        
        # 通达信操作符与系统操作符的映射
        self.operator_map = {
            "AND": "AND",
            "OR": "OR",
            "NOT": "NOT",
            ">": ">",
            "<": "<",
            ">=": ">=",
            "<=": "<=",
            "=": "==",
            "<>": "!="
        }
        
        # 周期映射
        self.period_map = {
            "日线": Period.DAILY,
            "周线": Period.WEEKLY,
            "月线": Period.MONTHLY,
            "60分钟": Period.MIN_60,
            "30分钟": Period.MIN_30,
            "15分钟": Period.MIN_15,
            "5分钟": Period.MIN_5
        }
        
    def convert(self, formula_text: str, strategy_name: str = "", 
              strategy_desc: str = "", author: str = "system") -> Dict[str, Any]:
        """
        将通达信公式转换为选股策略配置
        
        Args:
            formula_text: 通达信公式文本
            strategy_name: 策略名称，如果为空则从公式注释中提取
            strategy_desc: 策略描述，如果为空则从公式注释中提取
            author: 策略作者
            
        Returns:
            Dict[str, Any]: 策略配置字典
        """
        # 预处理公式文本
        formula_text = self._preprocess_formula(formula_text)
        
        # 从注释中提取策略名称和描述
        if not strategy_name or not strategy_desc:
            extracted_name, extracted_desc = self._extract_info_from_comments(formula_text)
            if not strategy_name:
                strategy_name = extracted_name
            if not strategy_desc:
                strategy_desc = extracted_desc
        
        # 解析公式中的变量定义和买入条件
        variables, buy_condition = self._parse_formula(formula_text)
        
        # 转换买入条件为策略条件
        conditions = self._convert_conditions(buy_condition, variables)
        
        # 创建策略ID
        strategy_id = f"FORMULA_{uuid.uuid4().hex[:8]}".upper()
        
        # 构建策略配置
        strategy_config = {
            "strategy": {
                "id": strategy_id,
                "name": strategy_name,
                "description": strategy_desc,
                "author": author,
                "version": "1.0",
                "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "conditions": conditions,
                "filters": {},
                "sort": [
                    {
                        "field": "signal_strength",
                        "direction": "DESC"
                    }
                ],
                "source_formula": formula_text
            }
        }
        
        return strategy_config
    
    def _preprocess_formula(self, formula_text: str) -> str:
        """
        预处理公式文本
        
        Args:
            formula_text: 原始公式文本
            
        Returns:
            str: 预处理后的公式文本
        """
        # 移除注释中的#和{}标记
        formula_text = re.sub(r'#.*?(\n|$)', '\n', formula_text)  # 移除 # 开头的单行注释
        formula_text = re.sub(r'{.*?}', '', formula_text, flags=re.DOTALL)  # 移除 {} 包围的注释
        
        # 替换通达信的赋值符号
        formula_text = formula_text.replace(':=', '=')
        
        # 统一换行符
        formula_text = formula_text.replace('\r\n', '\n').replace('\r', '\n')
        
        # 移除多余的空行和首尾空白
        formula_text = '\n'.join(line.strip() for line in formula_text.split('\n') if line.strip())
        
        return formula_text
    
    def _extract_info_from_comments(self, formula_text: str) -> Tuple[str, str]:
        """
        从公式注释中提取策略名称和描述
        
        Args:
            formula_text: 原始公式文本
            
        Returns:
            Tuple[str, str]: (策略名称, 策略描述)
        """
        name = "通达信公式策略"
        desc = "从通达信公式转换的选股策略"
        
        # 查找注释
        comment_pattern = r'(?:^|\n)(?:#|{)(.*?)(?:}|\n)'
        comments = re.findall(comment_pattern, formula_text, re.DOTALL)
        
        if comments:
            # 尝试从第一个注释中提取名称
            first_comment = comments[0].strip()
            if first_comment:
                name = first_comment.split('\n')[0].strip()
                
                # 如果注释多行，其余部分作为描述
                if '\n' in first_comment:
                    desc = '\n'.join(first_comment.split('\n')[1:]).strip()
        
        return name, desc
    
    def _parse_formula(self, formula_text: str) -> Tuple[Dict[str, str], str]:
        """
        解析公式中的变量定义和买入条件
        
        Args:
            formula_text: 预处理后的公式文本
            
        Returns:
            Tuple[Dict[str, str], str]: (变量定义字典, 买入条件)
        """
        variables = {}
        buy_condition = ""
        
        # 按行解析
        lines = formula_text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 检查是否是变量定义
            if '=' in line and not any(op in line.split('=')[0] for op in ['>', '<', '>=', '<=', '<>']):
                var_name, var_expr = line.split('=', 1)
                variables[var_name.strip()] = var_expr.strip().rstrip(';')
            # 如果不包含赋值，假设是买入条件
            elif '=' not in line:
                buy_condition = line.rstrip(';')
        
        # 如果没有找到明确的买入条件，尝试寻找最后定义的变量
        if not buy_condition and variables:
            last_var = list(variables.keys())[-1]
            if last_var.upper() in ['BUY', 'SIGNAL', 'RESULT']:
                buy_condition = last_var
        
        return variables, buy_condition
    
    def _convert_conditions(self, buy_condition: str, variables: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        将买入条件转换为策略条件
        
        Args:
            buy_condition: 买入条件表达式
            variables: 变量定义字典
            
        Returns:
            List[Dict[str, Any]]: 策略条件列表
        """
        conditions = []
        
        # 如果买入条件是一个变量引用，展开它
        if buy_condition in variables:
            buy_condition = variables[buy_condition]
        
        # 将AND、OR连接的条件拆分为独立条件
        if ' AND ' in buy_condition.upper():
            sub_conditions = buy_condition.split(' AND ')
            # 添加第一个条件
            conditions.extend(self._parse_single_condition(sub_conditions[0], variables))
            
            # 添加AND逻辑操作符
            conditions.append({
                "logic": "AND"
            })
            
            # 添加其余条件
            for sub_cond in sub_conditions[1:]:
                conditions.extend(self._parse_single_condition(sub_cond, variables))
                conditions.append({
                    "logic": "AND"
                })
            
            # 移除最后一个多余的AND
            if conditions and conditions[-1]["logic"] == "AND":
                conditions.pop()
                
        elif ' OR ' in buy_condition.upper():
            sub_conditions = buy_condition.split(' OR ')
            # 添加第一个条件
            conditions.extend(self._parse_single_condition(sub_conditions[0], variables))
            
            # 添加OR逻辑操作符
            conditions.append({
                "logic": "OR"
            })
            
            # 添加其余条件
            for sub_cond in sub_conditions[1:]:
                conditions.extend(self._parse_single_condition(sub_cond, variables))
                conditions.append({
                    "logic": "OR"
                })
            
            # 移除最后一个多余的OR
            if conditions and conditions[-1]["logic"] == "OR":
                conditions.pop()
        else:
            # 单一条件
            conditions.extend(self._parse_single_condition(buy_condition, variables))
        
        return conditions
    
    def _parse_single_condition(self, condition: str, variables: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        解析单个条件表达式
        
        Args:
            condition: 条件表达式
            variables: 变量定义字典
            
        Returns:
            List[Dict[str, Any]]: 解析后的条件列表
        """
        # 如果条件是变量引用，展开它
        condition = condition.strip()
        if condition in variables:
            condition = variables[condition]
        
        # 检查条件中的函数调用
        result = []
        
        # 检查交叉条件 CROSS(A,B)
        cross_match = re.search(r'CROSS\s*\(\s*(.+?)\s*,\s*(.+?)\s*\)', condition, re.IGNORECASE)
        if cross_match:
            fast_line = cross_match.group(1).strip()
            slow_line = cross_match.group(2).strip()
            
            # 解析快线和慢线
            fast_period = self._determine_period(fast_line, variables)
            
            # 创建交叉指标条件
            result.append({
                "indicator_id": "CROSS_OVER",
                "period": fast_period.name,
                "parameters": {
                    "fast_line": self._translate_line(fast_line, variables),
                    "slow_line": self._translate_line(slow_line, variables)
                }
            })
            return result
        
        # 检查KDJ相关条件
        kdj_match = re.search(r'(K|D|J)\s*([<>=]+)\s*(\d+(?:\.\d+)?)', condition, re.IGNORECASE)
        if kdj_match:
            line = kdj_match.group(1).upper()
            operator = kdj_match.group(2)
            value = float(kdj_match.group(3))
            
            period = self._determine_period(line, variables)
            
            result.append({
                "indicator_id": "KDJ_CONDITION",
                "period": period.name,
                "parameters": {
                    "line": line,
                    "operator": operator,
                    "value": value
                }
            })
            return result
        
        # 检查MACD相关条件
        macd_match = re.search(r'(MACD|DIF|DEA)\s*([<>=]+)\s*(\d+(?:\.\d+)?)', condition, re.IGNORECASE)
        if macd_match:
            line = macd_match.group(1).upper()
            operator = macd_match.group(2)
            value = float(macd_match.group(3))
            
            period = self._determine_period(line, variables)
            
            result.append({
                "indicator_id": "MACD_CONDITION",
                "period": period.name,
                "parameters": {
                    "line": line,
                    "operator": operator,
                    "value": value
                }
            })
            return result
        
        # 检查均线相关条件
        ma_match = re.search(r'(MA\d+|EMA\d+)\s*([<>=]+)\s*(.+)', condition, re.IGNORECASE)
        if ma_match:
            ma_type = ma_match.group(1).upper()
            operator = ma_match.group(2)
            compare_value = ma_match.group(3).strip()
            
            period = self._determine_period(ma_type, variables)
            ma_period = re.search(r'\d+', ma_type).group(0)
            
            result.append({
                "indicator_id": "MA_CONDITION",
                "period": period.name,
                "parameters": {
                    "ma_type": "MA" if ma_type.startswith("MA") else "EMA",
                    "ma_period": int(ma_period),
                    "operator": operator,
                    "compare_value": self._translate_line(compare_value, variables)
                }
            })
            return result
        
        # 如果无法识别为特定模式，创建通用条件
        result.append({
            "indicator_id": "GENERIC_CONDITION",
            "period": Period.DAILY.name,
            "parameters": {
                "condition": condition
            }
        })
        
        return result
    
    def _determine_period(self, expression: str, variables: Dict[str, str]) -> Period:
        """
        根据表达式确定周期
        
        Args:
            expression: 表达式
            variables: 变量定义字典
            
        Returns:
            Period: 周期枚举
        """
        # 默认使用日线周期
        period = Period.DAILY
        
        # 检查表达式中是否包含周期信息
        period_keywords = {
            "日线": Period.DAILY,
            "周线": Period.WEEKLY,
            "月线": Period.MONTHLY,
            "60分钟": Period.MIN_60,
            "30分钟": Period.MIN_30,
            "15分钟": Period.MIN_15,
            "5分钟": Period.MIN_5
        }
        
        for keyword, p in period_keywords.items():
            if keyword in expression:
                return p
        
        # 如果表达式是变量，查找变量定义中的周期信息
        if expression in variables:
            var_def = variables[expression]
            for keyword, p in period_keywords.items():
                if keyword in var_def:
                    return p
        
        return period
    
    def _translate_line(self, line: str, variables: Dict[str, str]) -> str:
        """
        翻译行表达式
        
        Args:
            line: 行表达式
            variables: 变量定义字典
            
        Returns:
            str: 翻译后的表达式
        """
        # 如果是变量引用，展开它
        if line in variables:
            return self._translate_line(variables[line], variables)
        
        # 替换通达信特有的价格引用
        line = line.replace('CLOSE', 'C').replace('OPEN', 'O').replace('HIGH', 'H').replace('LOW', 'L')
        
        # 替换通达信函数名
        for tdx_func, sys_func in self.function_map.items():
            pattern = fr'\b{tdx_func}\b'
            line = re.sub(pattern, sys_func, line)
        
        return line
    
    def generate_strategy_from_formula(self, formula_text: str, strategy_name: str = "", 
                                     strategy_desc: str = "", author: str = "system") -> Dict[str, Any]:
        """
        从通达信公式生成选股策略
        
        Args:
            formula_text: 通达信公式文本
            strategy_name: 策略名称
            strategy_desc: 策略描述
            author: 策略作者
            
        Returns:
            Dict[str, Any]: 策略配置字典
        """
        try:
            strategy_config = self.convert(formula_text, strategy_name, strategy_desc, author)
            return strategy_config
        except Exception as e:
            logger.error(f"从公式生成策略时出错: {e}")
            raise ValueError(f"公式转换失败: {e}")


class FormulaEditor:
    """
    公式编辑器，提供公式编辑和测试功能
    """
    
    def __init__(self, converter: Optional[FormulaConverter] = None):
        """
        初始化公式编辑器
        
        Args:
            converter: 公式转换器实例
        """
        self.converter = converter or FormulaConverter()
    
    def validate_formula(self, formula_text: str) -> Tuple[bool, str]:
        """
        验证公式语法
        
        Args:
            formula_text: 公式文本
            
        Returns:
            Tuple[bool, str]: (是否有效, 错误信息)
        """
        try:
            # 预处理公式
            formula_text = self.converter._preprocess_formula(formula_text)
            
            # 解析变量和条件
            variables, buy_condition = self.converter._parse_formula(formula_text)
            
            # 检查是否存在买入条件
            if not buy_condition:
                return False, "无法识别买入条件，请确保公式中包含明确的买入信号"
            
            # 验证变量定义的合法性
            for var_name, var_expr in variables.items():
                # 检查变量名是否合法
                if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', var_name):
                    return False, f"变量名 '{var_name}' 不合法，只能包含字母、数字和下划线，且不能以数字开头"
                
                # 检查引用的变量是否已定义
                for referenced_var in re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', var_expr):
                    if (referenced_var not in variables 
                        and referenced_var not in ['CLOSE', 'OPEN', 'HIGH', 'LOW', 'VOL', 'AMOUNT']
                        and referenced_var not in self.converter.function_map.keys()):
                        return False, f"变量 '{var_name}' 引用了未定义的变量 '{referenced_var}'"
            
            return True, "公式语法正确"
        except Exception as e:
            return False, f"公式验证失败: {str(e)}"
    
    def test_formula(self, formula_text: str, stock_code: str, 
                   start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        测试公式
        
        Args:
            formula_text: 公式文本
            stock_code: 股票代码
            start_date: 开始日期，默认为近30天
            end_date: 结束日期，默认为当前日期
            
        Returns:
            Dict[str, Any]: 测试结果
        """
        try:
            # 验证公式
            is_valid, error_msg = self.validate_formula(formula_text)
            if not is_valid:
                return {"success": False, "message": error_msg}
            
            # 创建StockFormula实例
            formula = StockFormula(stock_code, start=start_date, end=end_date)
            
            # 预处理公式
            formula_text = self.converter._preprocess_formula(formula_text)
            
            # 解析变量和条件
            variables, buy_condition = self.converter._parse_formula(formula_text)
            
            # 计算变量值
            variable_values = {}
            for var_name, var_expr in variables.items():
                # TODO: 实际计算变量值
                variable_values[var_name] = None
            
            # 评估买入条件
            # TODO: 实际评估买入条件
            
            # 转换为策略
            strategy_config = self.converter.convert(formula_text)
            
            return {
                "success": True,
                "message": "公式测试完成",
                "variables": variable_values,
                "buy_condition": buy_condition,
                "strategy_config": strategy_config
            }
        except Exception as e:
            logger.error(f"测试公式时出错: {e}")
            return {"success": False, "message": f"公式测试失败: {str(e)}"} 