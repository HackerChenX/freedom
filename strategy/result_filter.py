"""
结果筛选与排序模块

提供选股结果的多维度排序和筛选功能
"""

from typing import Dict, List, Any, Optional, Union, Callable
import pandas as pd
import re
import json
from datetime import datetime

from utils.logger import get_logger
from utils.decorators import performance_monitor, log_calls, safe_run

logger = get_logger(__name__)


class ResultFilter:
    """
    结果筛选与排序类
    
    提供选股结果的多维度排序和筛选功能
    """
    
    def __init__(self):
        """
        初始化结果筛选与排序类
        """
        # 注册筛选器函数
        self.filter_functions = {
            "score": self._filter_by_score,
            "rank": self._filter_by_rank,
            "industry": self._filter_by_industry,
            "market_cap": self._filter_by_market_cap,
            "strategy_count": self._filter_by_strategy_count,
            "condition_count": self._filter_by_condition_count,
            "date": self._filter_by_date,
            "regex": self._filter_by_regex
        }
        
        # 注册排序函数
        self.sort_functions = {
            "score": lambda df: df.sort_values(by="score", ascending=False),
            "rank": lambda df: df.sort_values(by="rank"),
            "market_cap": lambda df: df.sort_values(by="market_cap", ascending=False),
            "industry": lambda df: df.sort_values(by="industry"),
            "stock_code": lambda df: df.sort_values(by="stock_code"),
            "condition_count": self._sort_by_condition_count,
            "strategy_count": lambda df: df.sort_values(by="strategy_count", ascending=False),
            "combined_score": lambda df: df.sort_values(by="combined_score", ascending=False)
        }
    
    @performance_monitor()
    @log_calls(level="info")
    def filter_results(self, results: pd.DataFrame, 
                     filter_configs: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        根据配置筛选结果
        
        Args:
            results: 选股结果DataFrame
            filter_configs: 筛选配置列表，每个配置是一个字典，包含type, field, value等字段
            
        Returns:
            筛选后的DataFrame
        """
        if results.empty or not filter_configs:
            return results
            
        filtered_results = results.copy()
        
        for filter_config in filter_configs:
            filter_type = filter_config.get("type")
            
            if filter_type in self.filter_functions:
                # 使用对应的筛选函数
                filtered_results = self.filter_functions[filter_type](filtered_results, filter_config)
            else:
                logger.warning(f"未知的筛选类型: {filter_type}")
                
        # 重新计算排名
        if not filtered_results.empty and "rank" in filtered_results.columns:
            filtered_results["rank"] = range(1, len(filtered_results) + 1)
            
        return filtered_results
    
    @performance_monitor()
    @log_calls(level="info")
    def sort_results(self, results: pd.DataFrame, 
                   sort_configs: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        根据配置排序结果
        
        Args:
            results: 选股结果DataFrame
            sort_configs: 排序配置列表，每个配置是一个字典，包含field, direction等字段
            
        Returns:
            排序后的DataFrame
        """
        if results.empty or not sort_configs:
            return results
            
        # 复制结果，避免修改原始数据
        sorted_results = results.copy()
        
        # 构建排序参数
        sort_columns = []
        sort_ascending = []
        
        for sort_config in sort_configs:
            field = sort_config.get("field")
            direction = sort_config.get("direction", "DESC")
            
            # 检查是否有特殊排序函数
            if field in self.sort_functions:
                # 使用特殊排序函数
                sorted_results = self.sort_functions[field](sorted_results)
            elif field in sorted_results.columns:
                # 使用标准排序
                sort_columns.append(field)
                sort_ascending.append(direction.upper() == "ASC")
            else:
                logger.warning(f"排序字段不存在: {field}")
                
        # 如果有标准排序字段，进行排序
        if sort_columns:
            sorted_results = sorted_results.sort_values(by=sort_columns, ascending=sort_ascending)
            
        # 重新计算排名
        if "rank" in sorted_results.columns:
            sorted_results["rank"] = range(1, len(sorted_results) + 1)
            
        return sorted_results
    
    @performance_monitor()
    @log_calls(level="info")
    def group_results(self, results: pd.DataFrame, 
                    group_by: str, 
                    sort_field: Optional[str] = None,
                    top_n: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """
        将结果按照指定字段分组
        
        Args:
            results: 选股结果DataFrame
            group_by: 分组字段
            sort_field: 排序字段，默认为None
            top_n: 每组保留的最大记录数，默认为None表示保留所有
            
        Returns:
            分组后的结果字典，键为分组值，值为对应的DataFrame
        """
        if results.empty or group_by not in results.columns:
            return {}
            
        # 按指定字段分组
        grouped = dict(tuple(results.groupby(group_by)))
        
        # 如果需要排序和限制数量
        if sort_field is not None:
            for key in grouped:
                if sort_field in grouped[key].columns:
                    # 排序
                    grouped[key] = grouped[key].sort_values(by=sort_field, ascending=False)
                    
                    # 限制数量
                    if top_n is not None and len(grouped[key]) > top_n:
                        grouped[key] = grouped[key].head(top_n)
                        
        return grouped
    
    @performance_monitor()
    @log_calls(level="info")
    def enrich_results(self, results: pd.DataFrame, 
                     enrich_configs: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        增强结果，添加更多信息
        
        Args:
            results: 选股结果DataFrame
            enrich_configs: 增强配置列表
            
        Returns:
            增强后的DataFrame
        """
        if results.empty or not enrich_configs:
            return results
            
        enriched_results = results.copy()
        
        for config in enrich_configs:
            enrich_type = config.get("type")
            
            if enrich_type == "extract_json":
                # 从JSON字段中提取特定信息
                json_field = config.get("json_field")
                target_field = config.get("target_field")
                json_path = config.get("json_path")
                
                if json_field in enriched_results.columns:
                    enriched_results = self._extract_from_json(
                        enriched_results, json_field, target_field, json_path
                    )
            
            elif enrich_type == "calculate":
                # 计算新字段
                formula = config.get("formula")
                target_field = config.get("target_field")
                
                if formula and target_field:
                    enriched_results = self._calculate_field(
                        enriched_results, formula, target_field
                    )
                    
            elif enrich_type == "combine":
                # 合并字段
                source_fields = config.get("source_fields", [])
                target_field = config.get("target_field")
                separator = config.get("separator", " ")
                
                if source_fields and target_field:
                    enriched_results = self._combine_fields(
                        enriched_results, source_fields, target_field, separator
                    )
                    
            elif enrich_type == "condition_stats":
                # 分析条件满足情况
                conditions_field = config.get("conditions_field", "satisfied_conditions")
                
                if conditions_field in enriched_results.columns:
                    enriched_results = self._analyze_conditions(
                        enriched_results, conditions_field
                    )
                    
        return enriched_results
    
    # 筛选函数
    
    def _filter_by_score(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """按分数筛选"""
        min_score = config.get("min", 0)
        max_score = config.get("max", float('inf'))
        
        if "score" in df.columns:
            return df[(df["score"] >= min_score) & (df["score"] <= max_score)]
        return df
    
    def _filter_by_rank(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """按排名筛选"""
        min_rank = config.get("min", 1)
        max_rank = config.get("max", float('inf'))
        
        if "rank" in df.columns:
            return df[(df["rank"] >= min_rank) & (df["rank"] <= max_rank)]
        return df
    
    def _filter_by_industry(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """按行业筛选"""
        industries = config.get("values", [])
        exclude = config.get("exclude", False)
        
        if not industries or "industry" not in df.columns:
            return df
            
        if exclude:
            return df[~df["industry"].isin(industries)]
        else:
            return df[df["industry"].isin(industries)]
    
    def _filter_by_market_cap(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """按市值筛选"""
        min_cap = config.get("min", 0)
        max_cap = config.get("max", float('inf'))
        
        if "market_cap" in df.columns:
            return df[(df["market_cap"] >= min_cap) & (df["market_cap"] <= max_cap)]
        return df
    
    def _filter_by_strategy_count(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """按策略数量筛选"""
        min_count = config.get("min", 0)
        max_count = config.get("max", float('inf'))
        
        if "strategy_count" in df.columns:
            return df[(df["strategy_count"] >= min_count) & (df["strategy_count"] <= max_count)]
        return df
    
    def _filter_by_condition_count(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """按条件数量筛选"""
        min_count = config.get("min", 0)
        max_count = config.get("max", float('inf'))
        
        if "condition_count" in df.columns:
            return df[(df["condition_count"] >= min_count) & (df["condition_count"] <= max_count)]
        elif "satisfied_conditions" in df.columns:
            # 尝试从JSON字段解析
            try:
                df = df.copy()
                df["temp_condition_count"] = df["satisfied_conditions"].apply(
                    lambda x: len(json.loads(x).get("conditions", [])) if isinstance(x, str) else 0
                )
                result = df[(df["temp_condition_count"] >= min_count) & (df["temp_condition_count"] <= max_count)]
                result = result.drop("temp_condition_count", axis=1)
                return result
            except Exception as e:
                logger.error(f"解析条件数量时出错: {e}")
        
        return df
    
    def _filter_by_date(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """按日期筛选"""
        start_date = config.get("start")
        end_date = config.get("end")
        date_field = config.get("field", "selection_date")
        
        if date_field not in df.columns:
            return df
            
        result = df.copy()
        
        if start_date:
            result = result[result[date_field] >= start_date]
            
        if end_date:
            result = result[result[date_field] <= end_date]
            
        return result
    
    def _filter_by_regex(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """按正则表达式筛选"""
        field = config.get("field")
        pattern = config.get("pattern")
        
        if not field or not pattern or field not in df.columns:
            return df
            
        try:
            regex = re.compile(pattern)
            return df[df[field].astype(str).str.match(regex)]
        except Exception as e:
            logger.error(f"正则表达式筛选出错: {e}")
            return df
    
    # 排序函数
    
    def _sort_by_condition_count(self, df: pd.DataFrame) -> pd.DataFrame:
        """按条件数量排序"""
        if "condition_count" in df.columns:
            return df.sort_values(by="condition_count", ascending=False)
        elif "satisfied_conditions" in df.columns:
            # 尝试从JSON字段解析
            try:
                df = df.copy()
                df["temp_condition_count"] = df["satisfied_conditions"].apply(
                    lambda x: len(json.loads(x).get("conditions", [])) if isinstance(x, str) else 0
                )
                result = df.sort_values(by="temp_condition_count", ascending=False)
                result = result.drop("temp_condition_count", axis=1)
                return result
            except Exception as e:
                logger.error(f"解析条件数量时出错: {e}")
        
        return df
    
    # 数据处理函数
    
    def _extract_from_json(self, df: pd.DataFrame, 
                         json_field: str, 
                         target_field: str, 
                         json_path: str) -> pd.DataFrame:
        """从JSON字段提取信息"""
        if json_field not in df.columns:
            return df
            
        result = df.copy()
        
        try:
            # 解析JSON并提取指定路径的值
            def extract_value(json_str):
                if not isinstance(json_str, str):
                    return None
                    
                try:
                    data = json.loads(json_str)
                    # 解析路径
                    path_parts = json_path.split('.')
                    value = data
                    
                    for part in path_parts:
                        if isinstance(value, dict) and part in value:
                            value = value[part]
                        elif isinstance(value, list) and part.isdigit():
                            idx = int(part)
                            if 0 <= idx < len(value):
                                value = value[idx]
                            else:
                                return None
                        else:
                            return None
                            
                    return value
                except Exception:
                    return None
            
            result[target_field] = result[json_field].apply(extract_value)
            return result
        except Exception as e:
            logger.error(f"从JSON提取数据时出错: {e}")
            return df
    
    def _calculate_field(self, df: pd.DataFrame, 
                       formula: str, 
                       target_field: str) -> pd.DataFrame:
        """计算新字段"""
        try:
            # 创建一个安全的局部命名空间
            locals_dict = {"df": df}
            
            # 使用eval计算公式
            result = df.copy()
            result[target_field] = eval(formula, {"__builtins__": {}}, locals_dict)
            return result
        except Exception as e:
            logger.error(f"计算字段时出错: {e}")
            return df
    
    def _combine_fields(self, df: pd.DataFrame, 
                      source_fields: List[str], 
                      target_field: str, 
                      separator: str) -> pd.DataFrame:
        """合并字段"""
        # 检查源字段是否存在
        valid_fields = [field for field in source_fields if field in df.columns]
        
        if not valid_fields:
            return df
            
        try:
            result = df.copy()
            
            # 合并字段
            def combine_values(row):
                values = [str(row[field]) for field in valid_fields if pd.notna(row[field])]
                return separator.join(values)
            
            result[target_field] = result.apply(combine_values, axis=1)
            return result
        except Exception as e:
            logger.error(f"合并字段时出错: {e}")
            return df
    
    def _analyze_conditions(self, df: pd.DataFrame, 
                          conditions_field: str) -> pd.DataFrame:
        """分析条件满足情况"""
        if conditions_field not in df.columns:
            return df
            
        result = df.copy()
        
        try:
            # 提取条件数量
            def count_conditions(json_str):
                if not isinstance(json_str, str):
                    return 0
                    
                try:
                    data = json.loads(json_str)
                    conditions = data.get("conditions", [])
                    return len(conditions)
                except Exception:
                    return 0
            
            # 计算各类指标的数量
            def count_indicator_types(json_str):
                if not isinstance(json_str, str):
                    return {}
                    
                try:
                    data = json.loads(json_str)
                    conditions = data.get("conditions", [])
                    
                    indicator_counts = {}
                    for condition in conditions:
                        indicator_id = condition.get("indicator_id", "")
                        if indicator_id:
                            indicator_counts[indicator_id] = indicator_counts.get(indicator_id, 0) + 1
                            
                    return indicator_counts
                except Exception:
                    return {}
            
            # 添加条件数量字段
            result["condition_count"] = result[conditions_field].apply(count_conditions)
            
            # 添加指标类型统计
            indicator_counts = result[conditions_field].apply(count_indicator_types)
            
            # 找出所有出现的指标类型
            all_indicators = set()
            for counts in indicator_counts:
                all_indicators.update(counts.keys())
                
            # 为每种指标类型创建计数列
            for indicator in all_indicators:
                result[f"indicator_{indicator}_count"] = indicator_counts.apply(
                    lambda x: x.get(indicator, 0)
                )
                
            return result
        except Exception as e:
            logger.error(f"分析条件时出错: {e}")
            return df 