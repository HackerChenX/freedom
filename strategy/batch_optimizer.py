#!/usr/bin/env python3
"""
批处理性能优化器

减少数据库查询次数，提升大规模股票筛选效率
"""

import time
from typing import Dict, List, Any, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

from db.data_manager import DataManager
from utils.logger import get_logger
from utils.decorators import performance_monitor

logger = get_logger(__name__)


class BatchOptimizer:
    """批处理性能优化器"""
    
    def __init__(self, data_manager: DataManager):
        """
        初始化批处理优化器
        
        Args:
            data_manager: 数据管理器实例
        """
        self.data_manager = data_manager
        self.batch_cache = {}
        self.preloaded_data = {}
        
        logger.info("批处理优化器初始化完成")
    
    @performance_monitor(threshold=2.0)
    def batch_preload_stock_data(self, stock_codes: List[str], 
                                end_date: str, 
                                days_back: int = 120) -> Dict[str, pd.DataFrame]:
        """
        批量预加载股票数据，减少单独查询
        
        Args:
            stock_codes: 股票代码列表
            end_date: 结束日期
            days_back: 向前查询天数
            
        Returns:
            Dict[str, pd.DataFrame]: 股票代码到数据的映射
        """
        try:
            start_time = time.time()
            
            # 计算开始日期
            start_date = self.data_manager.get_previous_trade_date(end_date, days_back)
            
            # 分批处理，避免单次查询过多
            batch_size = 50  # 每批50只股票
            all_data = {}
            
            for i in range(0, len(stock_codes), batch_size):
                batch_codes = stock_codes[i:i + batch_size]
                
                # 构建批量查询SQL
                batch_data = self._batch_query_stock_data(
                    batch_codes, start_date, end_date
                )
                
                all_data.update(batch_data)
                
                # 避免过于频繁的查询
                if i + batch_size < len(stock_codes):
                    time.sleep(0.1)  # 100ms间隔
            
            end_time = time.time()
            logger.info(f"批量预加载 {len(stock_codes)} 只股票数据完成，耗时 {end_time - start_time:.2f} 秒")
            
            return all_data
            
        except Exception as e:
            logger.error(f"批量预加载股票数据失败: {e}")
            return {}
    
    def _batch_query_stock_data(self, stock_codes: List[str], 
                               start_date: str, 
                               end_date: str) -> Dict[str, pd.DataFrame]:
        """
        批量查询股票数据
        
        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            Dict[str, pd.DataFrame]: 股票数据映射
        """
        try:
            # 构建批量查询SQL
            codes_str = "', '".join(stock_codes)
            query = f"""
            SELECT code, trade_date, open, high, low, close, volume, amount, pct_chg
            FROM stock_data 
            WHERE code IN ('{codes_str}')
            AND level = '日线'
            AND trade_date >= '{start_date}'
            AND trade_date <= '{end_date}'
            ORDER BY code, trade_date
            """
            
            # 执行查询
            result = self.data_manager.db.query(query)
            
            if result.empty:
                return {}
            
            # 按股票代码分组
            stock_data_dict = {}
            for stock_code in stock_codes:
                stock_data = result[result['code'] == stock_code].copy()
                if not stock_data.empty:
                    # 设置日期索引
                    stock_data['trade_date'] = pd.to_datetime(stock_data['trade_date'])
                    stock_data.set_index('trade_date', inplace=True)
                    stock_data.drop('code', axis=1, inplace=True)
                    stock_data_dict[stock_code] = stock_data
            
            return stock_data_dict
            
        except Exception as e:
            logger.error(f"批量查询股票数据失败: {e}")
            return {}
    
    @performance_monitor(threshold=1.0)
    def batch_preload_industry_data(self, stock_codes: List[str]) -> Dict[str, str]:
        """
        批量预加载行业数据
        
        Args:
            stock_codes: 股票代码列表
            
        Returns:
            Dict[str, str]: 股票代码到行业的映射
        """
        try:
            # 构建批量查询SQL
            codes_str = "', '".join(stock_codes)
            query = f"""
            SELECT DISTINCT code, industry
            FROM stock_info 
            WHERE code IN ('{codes_str}')
            AND level = '日线'
            """
            
            result = self.data_manager.db.query(query)
            
            if result.empty:
                return {}
            
            # 转换为字典
            industry_dict = {}
            for _, row in result.iterrows():
                stock_code = row.get('code') or row.get('col_0', '')
                industry = row.get('industry') or row.get('col_1', '未知')
                if stock_code:
                    industry_dict[stock_code] = industry
            
            logger.info(f"批量预加载 {len(industry_dict)} 只股票行业数据完成")
            return industry_dict
            
        except Exception as e:
            logger.error(f"批量预加载行业数据失败: {e}")
            return {}
    
    def optimize_stock_filtering(self, stock_list: pd.DataFrame, 
                               filters: Dict[str, Any]) -> pd.DataFrame:
        """
        优化股票过滤逻辑，减少数据库查询
        
        Args:
            stock_list: 原始股票列表
            filters: 过滤条件
            
        Returns:
            pd.DataFrame: 过滤后的股票列表
        """
        try:
            if stock_list.empty:
                return stock_list
            
            # 如果没有过滤条件，直接返回
            if not filters:
                return stock_list
            
            filtered_list = stock_list.copy()
            
            # 市场过滤（在数据库层面进行）
            if 'market' in filters and filters['market']:
                markets = filters['market']
                if isinstance(markets, str):
                    markets = [markets]
                
                # 构建市场过滤查询
                market_filter_query = self._build_market_filter_query(markets)
                if market_filter_query:
                    market_stocks = self.data_manager.db.query(market_filter_query)
                    if not market_stocks.empty:
                        valid_codes = set(market_stocks.iloc[:, 0].tolist())
                        # 处理列名兼容性
                        stock_code_col = 'stock_code' if 'stock_code' in filtered_list.columns else 'col_0'
                        filtered_list = filtered_list[
                            filtered_list[stock_code_col].isin(valid_codes)
                        ]
            
            # 行业过滤
            if 'industry' in filters and filters['industry']:
                industries = filters['industry']
                if isinstance(industries, str):
                    industries = [industries]
                
                # 批量获取行业数据
                stock_codes = filtered_list.iloc[:, 0].tolist()
                industry_dict = self.batch_preload_industry_data(stock_codes)
                
                # 过滤行业
                valid_codes = [
                    code for code, industry in industry_dict.items()
                    if industry in industries
                ]
                
                stock_code_col = 'stock_code' if 'stock_code' in filtered_list.columns else 'col_0'
                filtered_list = filtered_list[
                    filtered_list[stock_code_col].isin(valid_codes)
                ]
            
            logger.info(f"股票过滤完成: {len(stock_list)} -> {len(filtered_list)}")
            return filtered_list
            
        except Exception as e:
            logger.error(f"优化股票过滤失败: {e}")
            return stock_list
    
    def _build_market_filter_query(self, markets: List[str]) -> str:
        """
        构建市场过滤查询
        
        Args:
            markets: 市场列表
            
        Returns:
            str: SQL查询语句
        """
        try:
            # 市场映射
            market_mapping = {
                '主板': ['SH', 'SZ'],
                '创业板': ['SZ'],
                '科创板': ['SH'],
                '北交所': ['BJ']
            }
            
            # 构建代码前缀条件
            conditions = []
            for market in markets:
                if market in market_mapping:
                    prefixes = market_mapping[market]
                    for prefix in prefixes:
                        if market == '主板':
                            if prefix == 'SH':
                                conditions.append("(code LIKE '60%%')")
                            elif prefix == 'SZ':
                                conditions.append("(code LIKE '00%%' AND code NOT LIKE '002%%' AND code NOT LIKE '003%%')")
                        elif market == '创业板':
                            conditions.append("(code LIKE '30%%')")
                        elif market == '科创板':
                            conditions.append("(code LIKE '688%%')")
                        elif market == '北交所':
                            conditions.append("(code LIKE '8%%' OR code LIKE '4%%')")
            
            if conditions:
                where_clause = " OR ".join(conditions)
                return f"""
                SELECT DISTINCT code
                FROM stock_info
                WHERE level = '日线' AND ({where_clause})
                """
            
            return ""
            
        except Exception as e:
            logger.error(f"构建市场过滤查询失败: {e}")
            return ""
    
    def get_optimized_batch_size(self, total_stocks: int, 
                                available_memory_mb: int = 1024) -> int:
        """
        根据系统资源动态计算最优批次大小
        
        Args:
            total_stocks: 总股票数量
            available_memory_mb: 可用内存（MB）
            
        Returns:
            int: 最优批次大小
        """
        try:
            # 基础批次大小
            base_batch_size = 50
            
            # 根据总股票数量调整
            if total_stocks > 2000:
                base_batch_size = 100
            elif total_stocks > 1000:
                base_batch_size = 75
            elif total_stocks < 500:
                base_batch_size = 30
            
            # 根据可用内存调整
            memory_factor = min(2.0, available_memory_mb / 512)  # 512MB为基准
            optimized_size = int(base_batch_size * memory_factor)
            
            # 确保在合理范围内
            optimized_size = max(20, min(150, optimized_size))
            
            logger.debug(f"优化批次大小: {optimized_size} (总股票: {total_stocks}, 内存: {available_memory_mb}MB)")
            return optimized_size
            
        except Exception as e:
            logger.error(f"计算最优批次大小失败: {e}")
            return 50  # 返回默认值
    
    def clear_batch_cache(self):
        """清理批处理缓存"""
        try:
            cache_size = len(self.batch_cache)
            self.batch_cache.clear()
            self.preloaded_data.clear()
            
            logger.info(f"批处理缓存已清理，释放 {cache_size} 项缓存")
            
        except Exception as e:
            logger.error(f"清理批处理缓存失败: {e}")


def create_batch_optimizer(data_manager: DataManager) -> BatchOptimizer:
    """创建批处理优化器实例"""
    return BatchOptimizer(data_manager)
