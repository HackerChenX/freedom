#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模拟数据访问实现 - 用于测试架构合规性

提供模拟的数据访问实现，用于验证架构设计的正确性，
不依赖外部数据库服务。
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from db.interfaces.data_access_interface import DataAccessInterface, DataAccessError
from utils.logger import get_logger

logger = get_logger(__name__)

class MockDataAccess(DataAccessInterface):
    """模拟数据访问实现 - 严格遵循架构规范"""
    
    def __init__(self):
        """初始化模拟数据访问"""
        self.mock_stock_codes = [
            '000001', '000002', '000858', '002119', '002415',
            '300308', '600036', '600519', '603259', '603650'
        ]
        logger.info("✅ 模拟数据访问初始化完成")
    
    def get_stock_data_data_access_interface(self, code: str, start_date: str, end_date: str, 
                      columns: Optional[List[str]] = None) -> pd.DataFrame:
        """获取模拟股票数据"""
        try:
            logger.debug(f"生成股票{code}的模拟数据: {start_date} 到 {end_date}")
            
            # 生成日期范围
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            dates = pd.date_range(start=start, end=end, freq='D')
            
            # 过滤工作日（简单模拟交易日）
            trading_dates = [d for d in dates if d.weekday() < 5]
            
            if len(trading_dates) == 0:
                return pd.DataFrame()
            
            # 生成模拟价格数据
            np.random.seed(hash(code) % 2**32)  # 使用股票代码作为种子，确保数据一致性
            
            base_price = 10.0 + (hash(code) % 100)  # 基础价格
            price_data = []
            
            current_price = base_price
            for date in trading_dates:
                # 模拟价格波动
                change_pct = np.random.normal(0, 0.02)  # 2%的日波动率
                current_price *= (1 + change_pct)
                
                # 生成OHLC数据
                high = current_price * (1 + abs(np.random.normal(0, 0.01)))
                low = current_price * (1 - abs(np.random.normal(0, 0.01)))
                open_price = current_price * (1 + np.random.normal(0, 0.005))
                close_price = current_price
                volume = int(np.random.lognormal(15, 1))  # 模拟成交量
                
                price_data.append({
                    'date': date,
                    'open': round(open_price, 2),
                    'high': round(high, 2),
                    'low': round(low, 2),
                    'close': round(close_price, 2),
                    'volume': volume
                })
            
            df = pd.DataFrame(price_data)
            
            # 如果指定了列，只返回指定列
            if columns:
                available_columns = [col for col in columns if col in df.columns]
                df = df[available_columns]
            
            logger.info(f"✅ 生成股票{code}模拟数据: {len(df)}条记录")
            return df
            
        except Exception as e:
            logger.error(f"生成股票{code}模拟数据失败: {e}")
            raise DataAccessError(f"获取股票{code}数据失败: {e}")
    
    def get_stocks_data_batch_data_access_interface(self, codes: List[str], start_date: str, end_date: str,
                             columns: Optional[List[str]] = None) -> pd.DataFrame:
        """批量获取模拟股票数据"""
        try:
            logger.debug(f"批量生成{len(codes)}支股票的模拟数据")
            
            all_data = []
            for code in codes:
                stock_df = self.get_stock_data_data_access_interface(code, start_date, end_date, columns)
                if not stock_df.empty:
                    stock_df['code'] = code
                    all_data.append(stock_df)
            
            if all_data:
                result_df = pd.concat(all_data, ignore_index=True)
                # 重新排列列顺序，将code放在前面
                if 'code' in result_df.columns:
                    cols = ['code'] + [col for col in result_df.columns if col != 'code']
                    result_df = result_df[cols]
                
                logger.info(f"✅ 批量生成模拟数据: {len(result_df)}条记录")
                return result_df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"批量生成模拟数据失败: {e}")
            raise DataAccessError(f"批量获取股票数据失败: {e}")
    
    def get_indicator_data_data_access_interface(self, code: str, indicator: str, start_date: str, end_date: str,
                          params: Optional[Dict] = None) -> pd.DataFrame:
        """获取模拟指标数据"""
        try:
            # 暂时返回空DataFrame，指标数据通常是实时计算的
            logger.debug(f"模拟指标{indicator}数据查询（返回空）")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"获取指标{indicator}数据失败: {e}")
            raise DataAccessError(f"获取指标{indicator}数据失败: {e}")
    
    def get_stock_list_data_access_interface(self, market: Optional[str] = None) -> List[str]:
        """获取模拟股票列表"""
        try:
            logger.debug("返回模拟股票列表")
            return self.mock_stock_codes.copy()
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            raise DataAccessError(f"获取股票列表失败: {e}")
    
    def get__list_data_access_interface(self) -> List[str]:
        """获取模拟行业列表"""
        try:
            mock_industries = ['银行', '保险', '证券', '房地产', '钢铁', '煤炭', '有色金属', '化工', '石油石化', '电力']
            logger.debug("返回模拟行业列表")
            return mock_industries
        except Exception as e:
            logger.error(f"获取行业列表失败: {e}")
            raise DataAccessError(f"获取行业列表失败: {e}")
    
    def execute_query_data_access_interface(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """执行模拟查询"""
        try:
            logger.debug(f"模拟执行查询: {query[:100]}...")
            
            # 简单的查询模拟
            if 'stock_info' in query.lower() and 'count' in query.lower():
                # 模拟COUNT查询
                return pd.DataFrame({'count': [len(self.mock_stock_codes)]})
            elif 'stock_info' in query.lower():
                # 模拟股票数据查询
                if 'distinct code' in query.lower():
                    return pd.DataFrame({'code': self.mock_stock_codes})
                else:
                    # 返回第一支股票的数据作为示例
                    return self.get_stock_data_data_access_interface(
                        self.mock_stock_codes[0], 
                        (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                        datetime.now().strftime('%Y-%m-%d')
                    )
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"执行查询失败: {e}")
            raise DataAccessError(f"执行查询失败: {e}")
    
    def check_data_exists_data_access_interface(self, table: str, conditions: Dict) -> bool:
        """检查模拟数据是否存在"""
        try:
            logger.debug(f"模拟检查数据存在性: {table}, {conditions}")
            
            # 简单模拟：如果是已知股票代码，返回True
            if 'code' in conditions:
                return conditions['code'] in self.mock_stock_codes
            else:
                return True  # 默认返回存在
                
        except Exception as e:
            logger.error(f"检查数据存在性失败: {e}")
            return False
    
    def get_latest_data_data_access_interface(self, table: str, code: str, columns: Optional[List[str]] = None) -> Optional[Dict]:
        """获取模拟最新数据"""
        try:
            logger.debug(f"获取股票{code}最新模拟数据")
            
            if code in self.mock_stock_codes:
                # 生成最新数据
                latest_df = self.get_stock_data_data_access_interface(
                    code,
                    (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
                    datetime.now().strftime('%Y-%m-%d')
                )
                
                if not latest_df.empty:
                    latest_row = latest_df.iloc[-1]
                    if columns:
                        return {col: latest_row.get(col) for col in columns if col in latest_row}
                    else:
                        return latest_row.to_dict()
                else:
                    return None
            else:
                return None
                
        except Exception as e:
            logger.error(f"获取最新数据失败: {e}")
            return None

def create_mock_data_access() -> MockDataAccess:
    """
    创建模拟数据访问实例
    
    Returns:
        模拟数据访问实例
    """
    try:
        mock_access = MockDataAccess()
        logger.info("✅ 模拟数据访问创建成功")
        return mock_access
    except Exception as e:
        logger.error(f"❌ 创建模拟数据访问失败: {e}")
        raise DataAccessError(f"创建模拟数据访问失败: {e}")
