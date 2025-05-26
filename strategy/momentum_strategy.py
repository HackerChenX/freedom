"""
动量策略模块

实现基于动量指标的选股策略
"""

import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple

from strategy.base_strategy import BaseStrategy
from formula.stock_formula import StockFormula
from utils.logger import get_logger
from enums.kline_period import KlinePeriod

logger = get_logger(__name__)


class MomentumStrategy(BaseStrategy):
    """
    动量策略
    
    基于股票动量特性的选股策略，结合吸筹、弹性和换手率等因素
    """
    
    def __init__(self):
        """初始化动量策略"""
        super().__init__(name="动量策略", description="基于股票动量特性的选股策略，结合吸筹、弹性和换手率等因素")
        
        # 设置默认参数
        self._parameters = {
            'min_turnover_rate': 1.5,  # 最小换手率
            'require_elasticity': True,  # 是否要求弹性
            'require_daily_absorption': True,  # 是否要求日线吸筹
            'require_weekly_absorption': True,  # 是否要求周线吸筹
            'start_date': '20000101',  # 起始日期
            'end_date': '20241231'  # 结束日期
        }
    
    def select(self, universe: List[str], *args, **kwargs) -> pd.DataFrame:
        """
        执行动量选股策略
        
        Args:
            universe: 股票代码列表，表示选股范围
            args: 位置参数
            kwargs: 关键字参数，可以覆盖默认参数
            
        Returns:
            pd.DataFrame: 选股结果，包含股票代码、名称、行业等信息
        """
        # 更新参数
        if kwargs:
            self._parameters.update(kwargs)
        
        # 提取参数
        min_turnover_rate = self._parameters['min_turnover_rate']
        require_elasticity = self._parameters['require_elasticity']
        require_daily_absorption = self._parameters['require_daily_absorption']
        require_weekly_absorption = self._parameters['require_weekly_absorption']
        start_date = self._parameters['start_date']
        end_date = self._parameters['end_date']
        
        # 存储选股结果
        selected_stocks = []
        
        # 遍历股票池，应用选股条件
        total_stocks = len(universe)
        logger.info(f"开始对 {total_stocks} 只股票进行动量策略筛选")
        
        for i, code in enumerate(universe):
            try:
                # 打印进度
                if (i + 1) % 100 == 0:
                    logger.info(f"已处理 {i + 1}/{total_stocks} 只股票")
                
                # 创建股票公式对象
                f = StockFormula(code, start=start_date, end=end_date)
                
                # 跳过没有数据的股票
                if f.dataDay.history is None or len(f.dataDay.history) == 0:
                    continue
                
                # 应用选股条件
                conditions_met = True
                
                # 换手率条件
                if min_turnover_rate > 0:
                    if not f.换手率大于(min_turnover_rate):
                        conditions_met = False
                
                # 弹性条件
                if require_elasticity and conditions_met:
                    if not f.弹性():
                        conditions_met = False
                
                # 日线吸筹条件
                if require_daily_absorption and conditions_met:
                    if not f.吸筹(KlinePeriod.DAILY):
                        conditions_met = False
                
                # 周线吸筹条件
                if require_weekly_absorption and conditions_met:
                    if not f.吸筹(KlinePeriod.WEEKLY):
                        conditions_met = False
                
                # 如果满足所有条件，添加到选股结果
                if conditions_met:
                    selected_stocks.append({
                        'code': f.stock_code,
                        'name': f.name,
                        'industry': f.industry
                    })
                    logger.info(f"选出股票: {f.get_desc()}")
            
            except Exception as e:
                logger.error(f"处理股票 {code} 时出错: {e}")
        
        # 转换为DataFrame
        result_df = pd.DataFrame(selected_stocks)
        
        logger.info(f"动量策略选股完成，共选出 {len(result_df)} 只股票")
        
        return result_df 