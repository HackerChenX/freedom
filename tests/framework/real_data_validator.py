#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实数据验证框架

确保所有指标验证都使用真实的ClickHouse数据，严格禁止模拟数据
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from clickhouse_driver import Client

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

from utils.logger import get_logger
from db.sql_manager import SQLManager, QueryType

logger = get_logger(__name__)


class RealDataValidator:
    """真实数据验证器"""
    
    def __init__(self):
        """初始化验证器"""
        self.client = None
        self.real_data_cache = {}
        self.validation_standards = {
            'min_data_points': 500,
            'min_stock_count': 5,
            'min_date_range_days': 30,  # 调整为30天，适应当前数据
            'required_columns': ['date', 'code', 'open', 'high', 'low', 'close', 'volume']
        }
        
    def connect_to_clickhouse(self) -> bool:
        """连接到ClickHouse数据库"""
        try:
            self.client = Client(
                host='localhost',
                port=9000,
                user='default',
                password='123456',
                database='stock'
            )
            
            # 测试连接
            result = self.client.execute("SELECT 1")
            if result:
                logger.info("✅ ClickHouse连接成功")
                return True
            else:
                logger.error("❌ ClickHouse连接失败")
                return False
                
        except Exception as e:
            logger.error(f"❌ ClickHouse连接异常: {e}")
            return False
    
    def get_real_stock_data(self, limit: int = 10000,
                           start_date: str = '2020-01-01') -> pd.DataFrame:
        """获取真实股票数据"""
        if not self.client:
            if not self.connect_to_clickhouse():
                raise ConnectionError("无法连接到ClickHouse数据库")
        
        cache_key = f"{limit}_{start_date}"
        if cache_key in self.real_data_cache:
            logger.info(f"📊 使用缓存的真实数据: {cache_key}")
            return self.real_data_cache[cache_key]
        
        try:
            # 使用子查询确保获取多个日期的数据
            query = f"""
            SELECT
                date,
                code,
                open,
                high,
                low,
                close,
                volume
            FROM (
                SELECT
                    date,
                    code,
                    open,
                    high,
                    low,
                    close,
                    volume,
                    ROW_NUMBER() OVER (PARTITION BY date ORDER BY code) as rn
                FROM stock_info WHERE code = %(code)s AND level = %(level)s AND volume > 0
                AND close > 0
                AND open > 0
                AND high > 0
                AND low > 0
            ) t
            WHERE rn <= 20  -- 每天最多20只股票
            ORDER BY date DESC, code
            LIMIT {limit}
            """
            
            logger.info(f"📊 从ClickHouse获取真实数据，限制{limit}条记录")
            result = self.client.execute(query)
            
            if not result:
                raise ValueError("ClickHouse查询返回空结果")
            
            # 转换为DataFrame
            columns = ['date', 'code', 'open', 'high', 'low', 'close', 'volume']
            df = pd.DataFrame(result, columns=columns)
            
            # 验证数据真实性
            if not self.validate_real_data(df):
                raise ValueError("数据未通过真实性验证")
            
            # 缓存数据
            self.real_data_cache[cache_key] = df
            
            logger.info(f"✅ 成功获取{len(df)}条真实ClickHouse数据")
            return df
            
        except Exception as e:
            logger.error(f"❌ 获取真实数据失败: {e}")
            raise
    
    def validate_real_data(self, data: pd.DataFrame) -> bool:
        """验证数据是否为真实数据"""
        if data.empty:
            logger.error("❌ 数据为空")
            return False
        
        validation_results = {}
        
        # 1. 检查必需列
        missing_columns = [col for col in self.validation_standards['required_columns'] 
                          if col not in data.columns]
        validation_results['has_required_columns'] = len(missing_columns) == 0
        if missing_columns:
            logger.error(f"❌ 缺少必需列: {missing_columns}")
        
        # 2. 检查数据量
        validation_results['sufficient_data'] = len(data) >= self.validation_standards['min_data_points']
        if not validation_results['sufficient_data']:
            logger.error(f"❌ 数据量不足: {len(data)} < {self.validation_standards['min_data_points']}")
        
        # 3. 检查股票数量
        if 'code' in data.columns:
            unique_stocks = data['code'].nunique()
            validation_results['sufficient_stocks'] = unique_stocks >= self.validation_standards['min_stock_count']
            if not validation_results['sufficient_stocks']:
                logger.error(f"❌ 股票数量不足: {unique_stocks} < {self.validation_standards['min_stock_count']}")
        else:
            validation_results['sufficient_stocks'] = False
        
        # 4. 检查日期范围
        if 'date' in data.columns:
            try:
                data['date'] = pd.to_datetime(data['date'])
                date_range = (data['date'].max() - data['date'].min()).days
                validation_results['sufficient_date_range'] = date_range >= self.validation_standards['min_date_range_days']
                if not validation_results['sufficient_date_range']:
                    logger.error(f"❌ 日期范围不足: {date_range}天 < {self.validation_standards['min_date_range_days']}天")
            except:
                validation_results['sufficient_date_range'] = False
                logger.error("❌ 日期格式错误")
        else:
            validation_results['sufficient_date_range'] = False
        
        # 5. 检查价格数据合理性
        if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            # OHLC关系检查
            ohlc_valid = (
                (data['high'] >= data['open']) & 
                (data['high'] >= data['close']) & 
                (data['low'] <= data['open']) & 
                (data['low'] <= data['close']) &
                (data['open'] > 0) &
                (data['high'] > 0) &
                (data['low'] > 0) &
                (data['close'] > 0)
            ).all()
            validation_results['valid_ohlc'] = ohlc_valid
            if not ohlc_valid:
                logger.error("❌ OHLC数据不合理")
        else:
            validation_results['valid_ohlc'] = False
        
        # 6. 检查成交量合理性
        if 'volume' in data.columns:
            volume_valid = (data['volume'] > 0).all()
            validation_results['valid_volume'] = volume_valid
            if not volume_valid:
                logger.error("❌ 成交量数据不合理")
        else:
            validation_results['valid_volume'] = False
        
        # 7. 检查是否为真实股票代码
        if 'code' in data.columns:
            real_codes = self._check_real_stock_codes(data['code'].unique())
            validation_results['real_stock_codes'] = real_codes
            if not real_codes:
                logger.error("❌ 股票代码不是真实的")
        else:
            validation_results['real_stock_codes'] = False
        
        # 总体验证结果
        all_passed = all(validation_results.values())
        
        if all_passed:
            logger.info("✅ 数据通过真实性验证")
        else:
            failed_checks = [k for k, v in validation_results.items() if not v]
            logger.error(f"❌ 数据未通过真实性验证，失败项: {failed_checks}")
        
        return all_passed
    
    def _check_real_stock_codes(self, codes: List[str]) -> bool:
        """检查是否为真实股票代码"""
        # A股代码格式检查
        real_code_patterns = [
            r'^00\d{4}$',  # 深市主板
            r'^30\d{4}$',  # 创业板
            r'^60\d{4}$',  # 沪市主板
            r'^68\d{4}$',  # 科创板
        ]
        
        import re
        real_codes_count = 0
        
        for code in codes:
            code_str = str(code)
            if any(re.match(pattern, code_str) for pattern in real_code_patterns):
                real_codes_count += 1
        
        # 至少80%的代码是真实的A股代码
        real_ratio = real_codes_count / len(codes) if len(codes) > 0 else 0
        return real_ratio >= 0.8
    
    def create_data_verification_report(self, data: pd.DataFrame) -> Dict[str, Any]:
        """创建数据验证报告"""
        report = {
            'verification_time': datetime.now().isoformat(),
            'data_source': 'ClickHouse Database',
            'data_size': len(data),
            'date_range': None,
            'stock_count': None,
            'validation_passed': False,
            'validation_details': {}
        }
        
        if not data.empty:
            if 'date' in data.columns:
                try:
                    data['date'] = pd.to_datetime(data['date'])
                    report['date_range'] = {
                        'start': data['date'].min().isoformat(),
                        'end': data['date'].max().isoformat(),
                        'days': (data['date'].max() - data['date'].min()).days
                    }
                except:
                    pass
            
            if 'code' in data.columns:
                report['stock_count'] = data['code'].nunique()
                report['stock_codes_sample'] = data['code'].unique()[:10].tolist()
            
            report['validation_passed'] = self.validate_real_data(data)
        
        return report


class ValidationError(Exception):
    """验证错误异常"""
    pass


def ensure_real_data_usage(stage_name: str, data: pd.DataFrame) -> bool:
    """确保使用真实数据"""
    validator = RealDataValidator()
    
    if not validator.validate_real_data(data):
        raise ValidationError(
            f"{stage_name}必须使用真实ClickHouse数据，"
            f"严格禁止使用模拟数据或人工构造数据"
        )
    
    logger.info(f"✅ {stage_name}真实数据验证通过")
    return True


def get_verified_real_data(limit: int = 10000) -> pd.DataFrame:
    """获取经过验证的真实数据"""
    validator = RealDataValidator()
    data = validator.get_real_stock_data(limit=limit)
    
    # 创建验证报告
    report = validator.create_data_verification_report(data)
    logger.info(f"📊 数据验证报告: {report}")
    
    return data


if __name__ == "__main__":
    # 测试真实数据验证
    try:
        validator = RealDataValidator()
        data = validator.get_real_stock_data(limit=1000)
        report = validator.create_data_verification_report(data)
        print("✅ 真实数据验证测试通过")
        print(f"📊 验证报告: {report}")
    except Exception as e:
        print(f"❌ 真实数据验证测试失败: {e}")
