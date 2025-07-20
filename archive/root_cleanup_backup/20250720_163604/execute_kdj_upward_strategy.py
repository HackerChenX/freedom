#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KDJ均上移策略执行器

专门执行KDJ指标K、D、J三线均呈上升趋势的选股策略

Author: AI Assistant
Date: 2025-07-20
"""

import pandas as pd
import numpy as np
import yaml
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class KDJUpwardStrategyExecutor:
    """KDJ均上移策略执行器"""
    
    def __init__(self, config_file: str = "config/strategies/kdj_all_lines_upward_strategy.yaml"):
        """初始化策略执行器"""
        self.config = self._load_config(config_file)
        self.target_date = self.config['strategy']['target_date']
        self.results = []
        
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """加载策略配置"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"✅ 成功加载策略配置: {config_file}")
            return config
        except Exception as e:
            logger.error(f"❌ 加载策略配置失败: {e}")
            raise
    
    def calculate_kdj(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算真实的KDJ指标"""
        try:
            if len(data) < 14:  # 确保有足够的数据
                return {'k': pd.Series(), 'd': pd.Series(), 'j': pd.Series()}
            
            # 获取配置参数
            k_period = self.config['strategy']['indicator']['parameters']['k_period']
            d_period = self.config['strategy']['indicator']['parameters']['d_period']
            j_multiplier = self.config['strategy']['indicator']['parameters']['j_multiplier']
            
            # 确保数据类型正确
            high = pd.to_numeric(data['high'], errors='coerce')
            low = pd.to_numeric(data['low'], errors='coerce')
            close = pd.to_numeric(data['close'], errors='coerce')
            
            # 计算RSV (Raw Stochastic Value)
            lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
            highest_high = high.rolling(window=k_period, min_periods=k_period).max()
            
            rsv = (close - lowest_low) / (highest_high - lowest_low) * 100
            
            # 计算K值 (使用指数移动平均)
            k = rsv.ewm(alpha=1/d_period, adjust=False).mean()
            
            # 计算D值 (K值的指数移动平均)
            d = k.ewm(alpha=1/d_period, adjust=False).mean()
            
            # 计算J值
            j = j_multiplier * k - 2 * d
            
            return {'k': k, 'd': d, 'j': j}
            
        except Exception as e:
            logger.error(f"计算KDJ指标失败: {e}")
            return {'k': pd.Series(), 'd': pd.Series(), 'j': pd.Series()}
    
    def check_kdj_upward_condition(self, kdj_data: Dict[str, pd.Series], target_idx: int) -> Dict[str, Any]:
        """检查KDJ均上移条件"""
        try:
            if target_idx < 1:  # 需要至少2个数据点来比较
                return {'meets_condition': False, 'reason': 'insufficient_data'}
            
            k_series = kdj_data['k']
            d_series = kdj_data['d']
            j_series = kdj_data['j']
            
            # 获取当前值和前一日值
            k_current = k_series.iloc[target_idx]
            k_previous = k_series.iloc[target_idx - 1]
            
            d_current = d_series.iloc[target_idx]
            d_previous = d_series.iloc[target_idx - 1]
            
            j_current = j_series.iloc[target_idx]
            j_previous = j_series.iloc[target_idx - 1]
            
            # 检查是否有无效值
            if pd.isna(k_current) or pd.isna(k_previous) or \
               pd.isna(d_current) or pd.isna(d_previous) or \
               pd.isna(j_current) or pd.isna(j_previous):
                return {'meets_condition': False, 'reason': 'invalid_data'}
            
            # 检查三线均上升条件
            k_upward = k_current > k_previous
            d_upward = d_current > d_previous
            j_upward = j_current > j_previous
            
            # 检查KDJ值范围过滤
            filters = self.config['strategy']['conditions']['filters']['kdj_range']
            k_in_range = filters['k_min'] <= k_current <= filters['k_max']
            d_in_range = filters['d_min'] <= d_current <= filters['d_max']
            j_in_range = filters['j_min'] <= j_current <= filters['j_max']
            
            # 综合判断
            meets_condition = k_upward and d_upward and j_upward and k_in_range and d_in_range and j_in_range
            
            # 计算上升强度
            k_strength = (k_current - k_previous) / k_previous if k_previous != 0 else 0
            d_strength = (d_current - d_previous) / d_previous if d_previous != 0 else 0
            j_strength = (j_current - j_previous) / abs(j_previous) if j_previous != 0 else 0
            
            # 计算综合评分
            scoring = self.config['strategy']['scoring']
            score = (k_strength * scoring['k_upward_strength'] + 
                    d_strength * scoring['d_upward_strength'] + 
                    j_strength * scoring['j_upward_strength'])
            
            return {
                'meets_condition': meets_condition,
                'k_upward': k_upward,
                'd_upward': d_upward,
                'j_upward': j_upward,
                'k_current': float(k_current),
                'k_previous': float(k_previous),
                'd_current': float(d_current),
                'd_previous': float(d_previous),
                'j_current': float(j_current),
                'j_previous': float(j_previous),
                'k_strength': float(k_strength),
                'd_strength': float(d_strength),
                'j_strength': float(j_strength),
                'score': float(score),
                'in_range': k_in_range and d_in_range and j_in_range
            }
            
        except Exception as e:
            logger.error(f"检查KDJ条件失败: {e}")
            return {'meets_condition': False, 'reason': f'calculation_error: {e}'}
    
    def get_stock_data(self, stock_code: str) -> Optional[pd.DataFrame]:
        """获取股票数据"""
        try:
            from db.clickhouse_db import get_clickhouse_db
            
            # 计算查询日期范围
            target_date = datetime.strptime(self.target_date, '%Y-%m-%d')
            start_date = (target_date - timedelta(days=80)).strftime('%Y-%m-%d')
            end_date = self.target_date
            
            db = get_clickhouse_db()
            query = f"""
            SELECT code, name, date, open, close, high, low, volume, turnover_rate
            FROM stock.stock_info 
            WHERE code = '{stock_code}' 
            AND date >= '{start_date}' 
            AND date <= '{end_date}'
            ORDER BY date ASC
            """
            
            result = db.query(query)
            
            if result is not None and not result.empty:
                # 确保数据类型正确
                numeric_columns = ['open', 'close', 'high', 'low', 'volume']
                for col in numeric_columns:
                    if col in result.columns:
                        result[col] = pd.to_numeric(result[col], errors='coerce')
                
                # 过滤掉无效数据
                result = result.dropna(subset=['close', 'high', 'low'])
                
                if len(result) >= 20:  # 确保有足够的数据
                    return result
            
            return None
            
        except Exception as e:
            logger.debug(f"获取股票 {stock_code} 数据失败: {e}")
            return None
    
    def analyze_single_stock(self, stock_code: str) -> Optional[Dict[str, Any]]:
        """分析单只股票"""
        try:
            # 获取股票数据
            stock_data = self.get_stock_data(stock_code)
            
            if stock_data is None or stock_data.empty:
                return None
            
            # 计算KDJ指标
            kdj_data = self.calculate_kdj(stock_data)
            
            if kdj_data['k'].empty:
                return None
            
            # 找到目标日期的索引
            target_data = stock_data[stock_data['date'] == self.target_date]
            if target_data.empty:
                return None
            
            target_idx = target_data.index[0]
            data_idx = stock_data.index.get_loc(target_idx)
            
            # 检查KDJ均上移条件
            condition_result = self.check_kdj_upward_condition(kdj_data, data_idx)
            
            if condition_result['meets_condition']:
                stock_name = stock_data.iloc[0]['name'] if 'name' in stock_data.columns else f'股票{stock_code}'
                
                result = {
                    'stock_code': stock_code,
                    'stock_name': stock_name,
                    'target_date': self.target_date,
                    'strategy_id': self.config['strategy']['id'],
                    **condition_result
                }
                
                return result
            
            return None
            
        except Exception as e:
            logger.error(f"分析股票 {stock_code} 失败: {e}")
            return None
    
    def execute_strategy(self) -> List[Dict[str, Any]]:
        """执行KDJ均上移策略"""
        logger.info("🚀 开始执行KDJ均上移策略")
        logger.info(f"📅 目标日期: {self.target_date}")
        logger.info(f"🎯 策略ID: {self.config['strategy']['id']}")
        
        # 获取股票池
        try:
            from db.clickhouse_db import get_clickhouse_db

            db = get_clickhouse_db()
            query = """
            SELECT DISTINCT code
            FROM stock.stock_info
            WHERE code IS NOT NULL
            AND name IS NOT NULL
            AND code != ''
            ORDER BY code
            LIMIT 4000
            """

            result = db.query(query)

            if result is not None and not result.empty:
                stock_codes = result['code'].tolist()
                logger.info(f"📊 股票池大小: {len(stock_codes)} 只股票")
            else:
                logger.error("❌ 未获取到股票池")
                return []

        except Exception as e:
            logger.error(f"❌ 获取股票池失败: {e}")
            return []
        
        # 分析股票
        selected_stocks = []
        processed_count = 0
        
        for stock_code in stock_codes:
            try:
                processed_count += 1
                
                # 每处理100只股票显示进度
                if processed_count % 100 == 0:
                    progress = processed_count / len(stock_codes) * 100
                    logger.info(f"📈 进度: {progress:.1f}% ({processed_count}/{len(stock_codes)}) - 已选出 {len(selected_stocks)} 只股票")
                
                # 分析单只股票
                result = self.analyze_single_stock(stock_code)
                
                if result:
                    selected_stocks.append(result)
                    logger.info(f"✅ 选中股票: {result['stock_code']} {result['stock_name']} - 评分: {result['score']:.3f}")
                
            except Exception as e:
                logger.error(f"❌ 处理股票 {stock_code} 失败: {e}")
                continue
        
        # 按评分排序
        selected_stocks.sort(key=lambda x: x['score'], reverse=True)
        
        # 限制选股数量
        max_stocks = self.config['strategy']['selection']['max_stocks']
        if len(selected_stocks) > max_stocks:
            selected_stocks = selected_stocks[:max_stocks]
        
        logger.info("=" * 80)
        logger.info(f"🎉 KDJ均上移策略执行完成!")
        logger.info(f"📊 处理股票数: {processed_count}")
        logger.info(f"✅ 选中股票数: {len(selected_stocks)}")
        
        return selected_stocks
    
    def save_results(self, results: List[Dict[str, Any]]):
        """保存结果"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 保存为JSON
            json_file = f"kdj_upward_strategy_results_{timestamp}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'strategy_config': self.config,
                    'execution_time': datetime.now().isoformat(),
                    'target_date': self.target_date,
                    'total_selected': len(results),
                    'results': results
                }, f, ensure_ascii=False, indent=2)
            
            logger.info(f"💾 结果已保存到: {json_file}")
            
            # 保存为CSV
            if results:
                df = pd.DataFrame(results)
                csv_file = f"kdj_upward_strategy_results_{timestamp}.csv"
                df.to_csv(csv_file, index=False, encoding='utf-8')
                logger.info(f"📊 CSV文件已保存到: {csv_file}")
            
        except Exception as e:
            logger.error(f"❌ 保存结果失败: {e}")


def main():
    """主函数"""
    try:
        # 创建策略执行器
        executor = KDJUpwardStrategyExecutor()
        
        # 执行策略
        results = executor.execute_strategy()
        
        # 保存结果
        executor.save_results(results)
        
        # 显示结果摘要
        if results:
            print("\n" + "=" * 80)
            print(f"📈 2025年5月12日KDJ均上移策略选股结果 ({len(results)} 只):")
            print("=" * 80)
            
            for i, stock in enumerate(results[:10], 1):  # 显示前10只
                print(f"{i:2d}. {stock['stock_code']} {stock['stock_name']}")
                print(f"     K: {stock['k_current']:.2f} (↑{stock['k_strength']:.2%})")
                print(f"     D: {stock['d_current']:.2f} (↑{stock['d_strength']:.2%})")
                print(f"     J: {stock['j_current']:.2f} (↑{stock['j_strength']:.2%})")
                print(f"     评分: {stock['score']:.3f}")
                print()
            
            if len(results) > 10:
                print(f"... 还有 {len(results) - 10} 只股票")
        
        else:
            print("\n❌ 未找到符合KDJ均上移条件的股票")
            print("🔧 建议检查:")
            print("   1. 目标日期数据是否存在")
            print("   2. KDJ参数设置是否合理")
            print("   3. 过滤条件是否过于严格")
    
    except Exception as e:
        logger.error(f"❌ 策略执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
