#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
StockInfo兼容数据生成器

生成与stockInfo对象结构完全兼容的模拟数据，支持各种技术指标形态的数据生成
确保生成的数据能被正式选股脚本正确处理
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional, Union
import random
import math

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(root_dir)

# 首先导入logger
from utils.logger import getLogger
logger = getLogger(__name__)

try:
    from tests.buypoint_analysis.enhanced_test_data_generator import EnhancedTestDataGenerator
    from enums.period import Period
except ImportError as e:
    logger.warning(f"导入部分模块失败: {e}")
    # 如果导入失败，创建简单的占位符
    class EnhancedTestDataGenerator:
        def generate_pattern_data(self, pattern_type, data_points, stock_code):
            return None
    
    class Period:
        DAILY = "daily"


class StockInfoCompatibleDataGenerator:
    """
    StockInfo兼容数据生成器
    
    生成与stockInfo对象结构完全兼容的模拟数据
    支持各种技术指标形态的数据生成
    """
    
    def __init__(self):
        """初始化数据生成器"""
        self.base_generator = EnhancedTestDataGenerator()
        
        # stockInfo字段定义（基于数据库表结构）
        self.stockinfo_fields = {
            # 基础字段
            'date': 'Date',           # 交易日期
            'code': 'String',         # 股票代码
            'name': 'String',         # 股票名称
            'level': 'String',        # K线周期
            
            # 价格字段
            'open': 'Float64',        # 开盘价
            'high': 'Float64',        # 最高价
            'low': 'Float64',         # 最低价
            'close': 'Float64',       # 收盘价
            
            # 交易量字段
            'volume': 'Float64',      # 成交量
            'turnover_rate': 'Float64', # 换手率
            
            # 计算字段
            'price_change': 'Float64',  # 价格变动
            'price_range': 'Float64',   # 价格区间
            
            # 分类字段
            'industry': 'String',     # 行业
            
            # 时间戳字段
            'datetime': 'DateTime',   # 日期时间
            'seq': 'UInt32'          # 序号
        }
        
        # 指标历史数据需求映射
        self.indicator_history_requirements = {
            'MACD': 60,      # MACD需要足够数据计算EMA
            'RSI': 30,       # RSI计算需求
            'KDJ': 20,       # KDJ计算需求
            'BOLL': 40,      # 布林带计算需求
            'DMI': 30,       # DMI计算需求
            'CCI': 25,       # CCI计算需求
            'WR': 20,        # WR计算需求
            'BIAS': 30,      # BIAS计算需求
            'EMA': 50,       # EMA计算需求
            'VOL': 10,       # 成交量指标
            'ADX': 30,       # ADX计算需求
            'DMA': 40,       # DMA计算需求
            'WMA': 50,       # WMA计算需求
            'StochRSI': 35,  # StochRSI计算需求
            'MA': 60,        # MA计算需求
            'OBV': 30,       # OBV计算需求
            'MTM': 25,       # MTM计算需求
            'PVT': 30,       # PVT计算需求
            'MOMENTUM': 25,  # MOMENTUM计算需求
            'FIBONACCI': 60, # FIBONACCI计算需求
            'AROON': 30,     # AROON计算需求
            # 🔧 Ultra Think新增指标历史需求
            'SAR': 40,       # SAR抛物线转向需求
            'TRIX': 45,      # TRIX三重指数平滑需求
            'MFI': 35,       # MFI资金流向指标需求
            'ROC': 30,       # ROC变化率指标需求
            'CMO': 25,       # CMO钱德动量摆动指标需求
            'ATR': 30,       # ATR真实波动范围需求
            'KC': 35,        # KC肯特纳通道需求
            'VIX': 40,       # VIX恐慌指数需求
            'EMV': 30,       # EMV简易波动指标需求
            'CHAIKIN': 30,   # CHAIKIN佳庆指标需求
            'SMA': 50        # SMA简单移动平均需求
        }
        
        # 行业分类
        self.industries = [
            '银行', '证券', '保险', '房地产', '建筑材料', '建筑装饰',
            '钢铁', '有色金属', '煤炭', '石油石化', '化工', '基础化工',
            '机械设备', '电气设备', '国防军工', '汽车', '家用电器',
            '食品饮料', '纺织服装', '轻工制造', '医药生物', '农林牧渔',
            '公用事业', '交通运输', '电子', '计算机', '通信', '传媒',
            '综合', '商业贸易', '休闲服务', '测试行业'
        ]
        
        logger.info("StockInfo兼容数据生成器初始化完成")
    
    def generate_stockinfo_compatible_data(self, 
                                         indicator_name: str,
                                         pattern_type: str,
                                         stock_code: str,
                                         history_days: int = None) -> pd.DataFrame:
        """
        生成StockInfo兼容的数据
        
        Args:
            indicator_name: 指标名称
            pattern_type: 形态类型
            stock_code: 股票代码
            history_days: 历史数据天数
            
        Returns:
            pd.DataFrame: 符合stockInfo结构的数据
        """
        try:
            # 确定历史数据需求
            if history_days is None:
                history_days = self._get_indicator_history_requirement(indicator_name)
            
            logger.debug(f"生成StockInfo兼容数据: {stock_code} ({indicator_name}.{pattern_type}, {history_days}天)")
            
            # 尝试使用增强数据生成器
            base_data = self._generate_base_data_with_pattern(
                indicator_name, pattern_type, stock_code, history_days
            )
            
            if base_data is None or base_data.empty:
                # 如果失败，使用基础数据生成
                base_data = self._generate_basic_stockinfo_data(stock_code, history_days)
            
            # 确保数据结构完全兼容
            stockinfo_data = self._ensure_stockinfo_compatibility(base_data, stock_code)
            
            # 验证数据质量
            if not self._validate_stockinfo_structure(stockinfo_data):
                raise ValueError(f"生成的数据不符合stockInfo结构要求")
            
            logger.debug(f"成功生成StockInfo兼容数据: {len(stockinfo_data)} 行")
            return stockinfo_data
            
        except Exception as e:
            logger.error(f"生成StockInfo兼容数据失败: {e}")
            # 返回基础数据作为备选
            return self._generate_basic_stockinfo_data(stock_code, history_days or 60)
    
    def generate_random_stockinfo_data(self, stock_code: str, history_days: int) -> pd.DataFrame:
        """
        生成随机StockInfo数据
        
        Args:
            stock_code: 股票代码
            history_days: 历史数据天数
            
        Returns:
            pd.DataFrame: 随机的stockInfo数据
        """
        try:
            logger.debug(f"生成随机StockInfo数据: {stock_code} ({history_days}天)")
            
            # 生成随机基础数据
            base_data = self._generate_random_base_data(stock_code, history_days)
            
            # 确保数据结构兼容
            stockinfo_data = self._ensure_stockinfo_compatibility(base_data, stock_code)
            
            return stockinfo_data
            
        except Exception as e:
            logger.error(f"生成随机StockInfo数据失败: {e}")
            return self._generate_basic_stockinfo_data(stock_code, history_days)
    
    def _get_indicator_history_requirement(self, indicator_name: str) -> int:
        """获取指标历史数据需求"""
        return self.indicator_history_requirements.get(indicator_name, 60)
    
    def _generate_base_data_with_pattern(self, 
                                       indicator_name: str,
                                       pattern_type: str,
                                       stock_code: str,
                                       history_days: int) -> Optional[pd.DataFrame]:
        """使用增强数据生成器生成带形态的基础数据"""
        try:
            # 如果是目标股票或测试股票，必须确保包含期望的形态
            if (stock_code.startswith('TARGET') or
                stock_code.startswith('TEST_') or
                stock_code.startswith('DEBUG_') or
                stock_code.startswith('MOCK_') or
                stock_code.startswith('FINAL_') or
                stock_code.startswith('SINGLE_') or
                stock_code.startswith('BATCH_') or
                stock_code.startswith('CONTINUOUS_') or
                stock_code.startswith('CCI_') or
                stock_code.startswith('NEW_') or
                'TEST' in stock_code or
                'DEBUG' in stock_code or
                'CCI' in stock_code):
                return self._generate_target_pattern_data(indicator_name, pattern_type, stock_code, history_days)

            # 非目标股票使用原有逻辑
            if self.base_generator:
                pattern_key = f"{indicator_name}_{pattern_type}"
                data = self.base_generator.generate_pattern_data(
                    pattern_type=pattern_key,
                    data_points=history_days,
                    stock_code=stock_code
                )
                return data
        except Exception as e:
            logger.warning(f"增强数据生成器失败: {e}")
        
        return None

    def _generate_target_pattern_data(self, 
                                    indicator_name: str,
                                    pattern_type: str,
                                    stock_code: str,
                                    history_days: int) -> pd.DataFrame:
        """专门为目标股票生成包含特定形态的数据"""
        try:
            logger.debug(f"为目标股票 {stock_code} 生成 {indicator_name}.{pattern_type} 形态数据")
            
            # 🔧 关键修复：设置随机种子确保数据多样性
            # 使用股票代码、指标、形态和时间戳生成不同的种子
            import time
            timestamp = int(time.time() * 1000000) % 1000000  # 微秒级时间戳
            seed = hash(f"{stock_code}_{indicator_name}_{pattern_type}_{timestamp}") % (2**32)
            random.seed(seed)
            np.random.seed(seed)
            
            # 生成基础数据
            base_data = self._generate_basic_stockinfo_data(stock_code, history_days)
            
            # 根据指标和形态类型调整数据，确保包含期望形态
            adjusted_data = self._adjust_data_for_pattern(base_data, indicator_name, pattern_type)
            
            # 🔧 关键修复：设置expected_pattern属性，用于买点识别验证
            expected_pattern_key = f"{indicator_name}_{pattern_type}"
            adjusted_data.attrs['expected_pattern'] = expected_pattern_key
            adjusted_data.attrs['is_target_stock'] = True
            adjusted_data.attrs['target_indicator'] = indicator_name
            adjusted_data.attrs['target_pattern'] = pattern_type
            
            logger.debug(f"✅ 目标股票数据已设置expected_pattern: {expected_pattern_key}")
            
            return adjusted_data
            
        except Exception as e:
            logger.error(f"生成目标形态数据失败: {e}")
            # 返回基础数据作为备选，但也要设置期望形态
            base_data = self._generate_basic_stockinfo_data(stock_code, history_days)
            expected_pattern_key = f"{indicator_name}_{pattern_type}"
            base_data.attrs['expected_pattern'] = expected_pattern_key
            base_data.attrs['is_target_stock'] = True
            return base_data

    def _adjust_data_for_pattern(self, 
                               data: pd.DataFrame, 
                               indicator_name: str, 
                               pattern_type: str) -> pd.DataFrame:
        """调整数据以确保包含特定形态"""
        try:
            logger.info(f"🚀 _adjust_data_for_pattern 被调用: indicator_name={indicator_name}, pattern_type={pattern_type}")
            
            # 🔧 关键修复：在形态调整前重新设置随机种子，确保形态生成的多样性
            # 基础数据生成消耗了大量随机数，需要重新设置种子
            stock_code = data['code'].iloc[0] if len(data) > 0 else "UNKNOWN"
            import time
            timestamp = int(time.time() * 1000000) % 1000000  # 微秒级时间戳
            seed = hash(f"PATTERN_{stock_code}_{indicator_name}_{pattern_type}_{timestamp}") % (2**32)
            random.seed(seed)
            np.random.seed(seed)
            logger.debug(f"为形态调整重置随机种子: {seed}")
            
            adjusted_data = data.copy()
            
            if indicator_name == 'MACD':
                logger.info(f"📈 处理MACD指标形态: {pattern_type}")
                adjusted_data = self._create_macd_pattern(adjusted_data, pattern_type)
            elif indicator_name == 'RSI':
                logger.info(f"📊 处理RSI指标形态: {pattern_type}")
                adjusted_data = self._create_rsi_pattern(adjusted_data, pattern_type)
            elif indicator_name == 'VOL':
                logger.info(f"🔊 处理VOL指标形态: {pattern_type}")
                adjusted_data = self._create_volume_pattern(adjusted_data, pattern_type)
            elif indicator_name == 'KDJ':
                logger.info(f"📉 处理KDJ指标形态: {pattern_type}")
                adjusted_data = self._create_kdj_pattern(adjusted_data, pattern_type)
            elif indicator_name == 'BOLL':
                logger.info(f"🎯 处理BOLL指标形态: {pattern_type}")
                adjusted_data = self._create_boll_pattern(adjusted_data, pattern_type)
            elif indicator_name == 'CCI':
                logger.info(f"🔄 处理CCI指标形态: {pattern_type}")
                adjusted_data = self._create_cci_pattern(adjusted_data, pattern_type)
            elif indicator_name == 'WMA':
                logger.info(f"📋 处理WMA指标形态: {pattern_type}")
                adjusted_data = self._create_wma_pattern(adjusted_data, pattern_type)
            elif indicator_name == 'DMA':
                logger.info(f"🎪 处理DMA指标形态: {pattern_type}")
                # DMA单独处理，因为有特殊的SUPPORT_RESISTANCE形态
                adjusted_data = self._create_dma_pattern(adjusted_data, pattern_type)
            elif indicator_name in ['MA', 'EMA', 'WMA']:
                logger.info(f"📏 处理移动平均线指标形态: {indicator_name}.{pattern_type}")
                # 移动平均线类指标统一处理
                adjusted_data = self._create_ma_pattern(adjusted_data, pattern_type)
            elif indicator_name == 'ADX':
                logger.info(f"🎯 处理ADX指标形态: {indicator_name}.{pattern_type}")
                # ADX指标专门处理
                adjusted_data = self._create_adx_pattern(adjusted_data, pattern_type)
            elif indicator_name == 'ATR':
                logger.info(f"🎯 处理ATR指标形态: {indicator_name}.{pattern_type}")
                # ATR指标专门处理
                adjusted_data = self._create_atr_pattern(adjusted_data, pattern_type)
            elif indicator_name == 'CMO':
                logger.info(f"🎯 处理CMO指标形态: {indicator_name}.{pattern_type}")
                # CMO指标专门处理
                adjusted_data = self._create_cmo_pattern(adjusted_data, pattern_type)
            elif indicator_name == 'ROC':
                logger.info(f"🎯 处理ROC指标形态: {indicator_name}.{pattern_type}")
                # ROC指标专门处理
                adjusted_data = self._create_roc_pattern(adjusted_data, pattern_type)
            elif indicator_name == 'SAR':
                logger.info(f"🎯 处理SAR指标形态: {indicator_name}.{pattern_type}")
                # SAR抛物线转向指标专门处理
                adjusted_data = self._create_sar_pattern(adjusted_data, pattern_type)
            elif indicator_name == 'TRIX':
                logger.info(f"🎯 处理TRIX指标形态: {indicator_name}.{pattern_type}")
                # TRIX三重指数平滑指标专门处理
                adjusted_data = self._create_trix_pattern(adjusted_data, pattern_type)
            elif indicator_name == 'MFI':
                logger.info(f"🎯 处理MFI指标形态: {indicator_name}.{pattern_type}")
                # MFI资金流向指标专门处理
                adjusted_data = self._create_mfi_pattern(adjusted_data, pattern_type)
            elif indicator_name in ['KC', 'EMV', 'CHAIKIN', 'VIX']:
                logger.info(f"🎯 处理新增技术指标形态: {indicator_name}.{pattern_type}")
                # 新增技术指标统一处理
                adjusted_data = self._create_new_indicator_pattern(adjusted_data, indicator_name, pattern_type)
            elif indicator_name == 'SMA':
                logger.info(f"📊 处理SMA移动平均线形态: {indicator_name}.{pattern_type}")
                # SMA简单移动平均线指标专门处理
                adjusted_data = self._create_ma_pattern(adjusted_data, pattern_type)
            elif indicator_name in ['BIAS', 'STOCHRSI', 'WR', 'OBV', 'MTM', 'PVT', 'MOMENTUM', 'DMI']:
                logger.info(f"🔧 处理其他技术指标形态: {indicator_name}.{pattern_type}")
                # 其他技术指标统一处理GOLDEN_CROSS/DEATH_CROSS形态
                adjusted_data = self._create_cross_pattern(adjusted_data, pattern_type)
            elif indicator_name == 'FIBONACCI':
                logger.info(f"🌀 处理FIBONACCI指标形态: {pattern_type}")
                adjusted_data = self._create_fibonacci_pattern(adjusted_data, pattern_type)
            elif indicator_name == 'AROON':
                logger.info(f"🎳 处理AROON指标形态: {pattern_type}")
                adjusted_data = self._create_aroon_pattern(adjusted_data, pattern_type)
            else:
                logger.warning(f"❓ 未知指标名称，使用通用形态: {indicator_name}.{pattern_type}")
                # 对于其他指标，生成通用的趋势形态
                adjusted_data = self._create_generic_pattern(adjusted_data, pattern_type)
            
            logger.info(f"✅ _adjust_data_for_pattern 完成: {indicator_name}.{pattern_type}")
            return adjusted_data
            
        except Exception as e:
            logger.error(f"调整数据形态失败: {e}")
            return data

    def _create_macd_pattern(self, data: pd.DataFrame, pattern_type: str) -> pd.DataFrame:
        """创建MACD特定形态"""
        adjusted_data = data.copy()
        n = len(adjusted_data)
        
        if pattern_type == 'GOLDEN_CROSS':
            # 创建金叉形态：MACD线从下方向上穿越信号线
            # 在最后10个交易日制造金叉
            golden_cross_point = max(n - 10, n//2)
            
            base_price = adjusted_data['close'].iloc[0]
            
            # 先创建下跌趋势，然后上升形成金叉
            for i in range(n):
                if i < golden_cross_point:
                    # 下跌阶段
                    trend_factor = 1 - (0.15 * i / golden_cross_point)  # 最多下跌15%
                else:
                    # 上升阶段 - 形成金叉
                    recovery_ratio = (i - golden_cross_point) / (n - golden_cross_point)
                    trend_factor = 0.85 + (0.25 * recovery_ratio)  # 反弹25%
                
                new_price = base_price * trend_factor
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.99, 1.01)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.01, 1.05)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.95, 0.99)
                
        elif pattern_type == 'DEATH_CROSS':
            # 创建死叉形态：MACD线从上方向下穿越信号线
            death_cross_point = max(n - 10, n//2)
            
            base_price = adjusted_data['close'].iloc[0]
            
            # 先创建上升趋势，然后下跌形成死叉
            for i in range(n):
                if i < death_cross_point:
                    # 上升阶段
                    trend_factor = 1 + (0.20 * i / death_cross_point)  # 上涨20%
                else:
                    # 下跌阶段 - 形成死叉
                    decline_ratio = (i - death_cross_point) / (n - death_cross_point)
                    trend_factor = 1.20 - (0.30 * decline_ratio)  # 下跌30%
                
                new_price = base_price * trend_factor
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.99, 1.01)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.01, 1.05)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.95, 0.99)
        
        return adjusted_data

    def _create_rsi_pattern(self, data: pd.DataFrame, pattern_type: str) -> pd.DataFrame:
        """创建RSI特定形态 - 修复：确保可靠的RSI极值生成"""
        logger.info(f"🔧 _create_rsi_pattern被调用: pattern_type={pattern_type}, data_length={len(data)}")
        
        adjusted_data = data.copy()
        n = len(adjusted_data)
        
        # 记录原始价格
        original_first_price = adjusted_data['close'].iloc[0]
        original_last_price = adjusted_data['close'].iloc[-1]
        logger.info(f"原始价格: 第一天={original_first_price:.2f}, 最后一天={original_last_price:.2f}")
        
        if pattern_type == 'OVERSOLD':
            # 创建超卖形态：需要RSI最终 < 35（最好 < 30）
            base_price = adjusted_data['close'].iloc[0]
            logger.info(f"OVERSOLD: 基准价格={base_price:.2f}")
            
            # 策略：连续下跌，确保RSI计算结果为超卖
            for i in range(n):
                # 计算下跌进度，越到后面下跌越多
                progress = i / (n - 1) if n > 1 else 0  # 0 到 1 的进度
                
                # 总下跌幅度设为60%，分阶段递增下跌
                if progress < 0.3:
                    # 前30%：小幅下跌（总计10%）
                    decline_ratio = 0.10 * (progress / 0.3)
                elif progress < 0.7:
                    # 中40%：加速下跌（总计再下跌30%）
                    decline_ratio = 0.10 + 0.30 * ((progress - 0.3) / 0.4)
                else:
                    # 后30%：持续下跌到最终（总计再下跌20%）
                    decline_ratio = 0.40 + 0.20 * ((progress - 0.7) / 0.3)
                
                # 计算新价格（每天都比前一天低一点点，形成持续下跌）
                new_price = base_price * (1 - decline_ratio)
                
                # 添加非常小的随机波动（不影响总体趋势）
                daily_noise = random.uniform(0.998, 1.002)  # 减少随机性
                new_price *= daily_noise
                
                # 使用 .iloc 确保正确的索引
                adjusted_data.iloc[i, adjusted_data.columns.get_loc('close')] = new_price
                adjusted_data.iloc[i, adjusted_data.columns.get_loc('open')] = new_price * random.uniform(0.999, 1.001)
                adjusted_data.iloc[i, adjusted_data.columns.get_loc('high')] = new_price * random.uniform(1.001, 1.005)
                adjusted_data.iloc[i, adjusted_data.columns.get_loc('low')] = new_price * random.uniform(0.995, 0.999)
                
        elif pattern_type == 'OVERBOUGHT':
            # 创建超买形态：需要RSI最终 > 65（最好 > 70）
            base_price = adjusted_data['close'].iloc[0]
            logger.info(f"OVERBOUGHT: 基准价格={base_price:.2f}")
            
            # 策略：连续上涨，确保RSI计算结果为超买
            for i in range(n):
                # 计算上涨进度，越到后面涨幅越大
                progress = i / (n - 1) if n > 1 else 0  # 0 到 1 的进度
                
                # 总上涨幅度设为80%，分阶段递增上涨
                if progress < 0.3:
                    # 前30%：小幅上涨（总计15%）
                    rise_ratio = 0.15 * (progress / 0.3)
                elif progress < 0.7:
                    # 中40%：加速上涨（总计再上涨40%）
                    rise_ratio = 0.15 + 0.40 * ((progress - 0.3) / 0.4)
                else:
                    # 后30%：持续上涨到最终（总计再上涨25%）
                    rise_ratio = 0.55 + 0.25 * ((progress - 0.7) / 0.3)
                
                # 计算新价格（每天都比前一天高一点点，形成持续上涨）
                new_price = base_price * (1 + rise_ratio)
                
                # 添加非常小的随机波动（不影响总体趋势）
                daily_noise = random.uniform(0.998, 1.002)  # 减少随机性
                new_price *= daily_noise
                
                # 使用 .iloc 确保正确的索引
                adjusted_data.iloc[i, adjusted_data.columns.get_loc('close')] = new_price
                adjusted_data.iloc[i, adjusted_data.columns.get_loc('open')] = new_price * random.uniform(0.999, 1.001)
                adjusted_data.iloc[i, adjusted_data.columns.get_loc('high')] = new_price * random.uniform(1.001, 1.005)
                adjusted_data.iloc[i, adjusted_data.columns.get_loc('low')] = new_price * random.uniform(0.995, 0.999)

        elif pattern_type == 'GOLDEN_CROSS':
            # 🔧 关键修复：创建RSI金叉形态
            # RSI金叉需要：1) RSI从低位回升 2) RSI短期均线上穿长期均线 3) RSI突破50中线
            base_price = adjusted_data['close'].iloc[0]
            logger.info(f"GOLDEN_CROSS: 基准价格={base_price:.2f}")

            # 策略：先下跌到超卖区域，然后强力反弹，形成RSI金叉
            for i in range(n):
                progress = i / (n - 1) if n > 1 else 0  # 0 到 1 的进度

                if progress < 0.4:
                    # 前40%：下跌到超卖区域（下跌25%）
                    decline_ratio = 0.25 * (progress / 0.4)
                    new_price = base_price * (1 - decline_ratio)
                elif progress < 0.6:
                    # 中20%：筑底阶段（保持低位）
                    new_price = base_price * 0.75  # 保持在最低点
                else:
                    # 后40%：强力反弹，形成RSI金叉（反弹35%）
                    rebound_progress = (progress - 0.6) / 0.4
                    rebound_ratio = 0.35 * rebound_progress
                    new_price = base_price * (0.75 + rebound_ratio)  # 从0.75反弹到1.10

                # 添加小幅随机波动
                daily_noise = random.uniform(0.998, 1.002)
                new_price *= daily_noise

                # 更新价格数据
                adjusted_data.iloc[i, adjusted_data.columns.get_loc('close')] = new_price
                adjusted_data.iloc[i, adjusted_data.columns.get_loc('open')] = new_price * random.uniform(0.999, 1.001)
                adjusted_data.iloc[i, adjusted_data.columns.get_loc('high')] = new_price * random.uniform(1.001, 1.005)
                adjusted_data.iloc[i, adjusted_data.columns.get_loc('low')] = new_price * random.uniform(0.995, 0.999)

        # 记录调整后的价格
        new_first_price = adjusted_data['close'].iloc[0]
        new_last_price = adjusted_data['close'].iloc[-1]
        price_change_pct = ((new_last_price / new_first_price) - 1) * 100
        logger.info(f"调整后价格: 第一天={new_first_price:.2f}, 最后一天={new_last_price:.2f}, 变化={price_change_pct:.2f}%")
        logger.info(f"✅ _create_rsi_pattern完成: pattern_type={pattern_type}")
        
        return adjusted_data

    def _create_volume_pattern(self, data: pd.DataFrame, pattern_type: str) -> pd.DataFrame:
        """创建成交量特定形态"""
        adjusted_data = data.copy()
        n = len(adjusted_data)

        if pattern_type in ['VOLUME_SPIKE', 'VOLUME_SURGE']:
            # 创建成交量突增形态
            spike_point = max(n - 5, n//2)  # 在后半段出现突增
            base_volume = adjusted_data['volume'].mean()

            for i in range(n):
                if i >= spike_point:
                    # 成交量突增2-5倍
                    volume_multiplier = random.uniform(2.5, 5.0)
                    adjusted_data.loc[i, 'volume'] = base_volume * volume_multiplier

                    # 成交量放大通常伴随价格突破
                    price_boost = random.uniform(1.02, 1.08)  # 2-8%的价格上涨
                    adjusted_data.loc[i, 'close'] *= price_boost
                    adjusted_data.loc[i, 'high'] *= price_boost
                    adjusted_data.loc[i, 'open'] *= price_boost
                    adjusted_data.loc[i, 'low'] *= price_boost

        elif pattern_type == 'BREAKOUT_UP':
            # 🔧 新增：创建VOL_BREAKOUT_UP形态（放量上涨）
            logger.info(f"创建VOL_BREAKOUT_UP形态，数据长度: {n}")

            # 策略：前期缩量整理，后期放量上涨
            base_volume = adjusted_data['volume'].mean()
            base_price = adjusted_data['close'].iloc[0]
            breakout_point = max(n - 10, int(n * 0.7))  # 在后30%开始放量上涨

            logger.info(f"BREAKOUT_UP: 基准成交量={base_volume:.0f}, 基准价格={base_price:.2f}, 突破点={breakout_point}")

            for i in range(n):
                progress = i / (n - 1) if n > 1 else 0

                if i < breakout_point:
                    # 前期：缩量整理（成交量减少20-40%）
                    volume_factor = random.uniform(0.6, 0.8)
                    adjusted_data.loc[i, 'volume'] = base_volume * volume_factor

                    # 价格小幅震荡
                    price_oscillation = random.uniform(-0.02, 0.02)
                    new_price = base_price * (1 + price_oscillation)
                else:
                    # 后期：放量上涨
                    breakout_progress = (i - breakout_point) / (n - breakout_point)

                    # 成交量逐步放大（1.5-3倍）
                    volume_multiplier = 1.5 + (1.5 * breakout_progress)
                    adjusted_data.loc[i, 'volume'] = base_volume * volume_multiplier

                    # 价格上涨（总涨幅8-15%）
                    price_gain = 0.08 + (0.07 * breakout_progress)
                    new_price = base_price * (1 + price_gain)

                # 更新价格数据
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.995, 1.005)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.005, 1.015)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.985, 0.995)

            # 记录调整结果
            final_volume_ratio = adjusted_data['volume'].iloc[-1] / base_volume
            final_price_change = (adjusted_data['close'].iloc[-1] / base_price - 1) * 100
            logger.info(f"BREAKOUT_UP完成: 最终量比={final_volume_ratio:.2f}, 价格涨幅={final_price_change:.2f}%")

        return adjusted_data

    def _create_kdj_pattern(self, data: pd.DataFrame, pattern_type: str) -> pd.DataFrame:
        """创建KDJ特定形态"""
        adjusted_data = data.copy()
        n = len(adjusted_data)

        if pattern_type in ['OVERBOUGHT', 'OVERSOLD']:
            return self._create_rsi_pattern(data, pattern_type)
        elif pattern_type == 'GOLDEN_CROSS':
            # 🔧 Ultra Think终极策略：迭代生成确保最终KDJ金叉的数据
            logger.info(f"Ultra Think终极策略: 创建KDJ_GOLDEN_CROSS形态，数据长度: {n}")

            base_price = adjusted_data['close'].iloc[0]
            logger.info(f"KDJ_GOLDEN_CROSS: 基准价格={base_price:.2f}")

            # 迭代策略：多次尝试直到生成真正金叉的数据
            max_attempts = 5
            for attempt in range(max_attempts):
                logger.info(f"尝试第{attempt+1}次生成KDJ金叉数据")

                # 重置数据
                temp_data = adjusted_data.copy()

                # 策略：前期下跌让KDJ进入低位，后期强力反弹确保K>D
                decline_end = max(int(n * 0.65), n - 12)  # 前65%下跌，后35%反弹

                for i in range(n):
                    if i < decline_end:
                        # 前期：持续下跌，让KDJ降到低位
                        decline_progress = i / decline_end
                        # 总跌幅20-30%，确保KDJ进入低位
                        price_decline = 0.20 + (0.10 * decline_progress)
                        new_price = base_price * (1 - price_decline)

                        # 添加小幅波动，但保持下跌趋势
                        daily_noise = random.uniform(-0.015, 0.005)
                        new_price *= (1 + daily_noise)
                    else:
                        # 后期：强力反弹，确保K线上穿D线
                        rebound_progress = (i - decline_end) / (n - decline_end)

                        # 从最低点强力反弹，最后阶段加速上涨
                        min_price = base_price * 0.70  # 最低点
                        if rebound_progress < 0.7:
                            # 前70%温和反弹
                            rebound_gain = 0.10 * (rebound_progress / 0.7)
                        else:
                            # 后30%加速反弹，确保金叉
                            base_rebound = 0.10
                            extra_boost = 0.15 * ((rebound_progress - 0.7) / 0.3)
                            rebound_gain = base_rebound + extra_boost

                        new_price = min_price * (1 + rebound_gain)

                        # 添加上涨波动，最后几个点确保强势
                        if i >= n - 5:  # 最后5个点强势上涨
                            daily_noise = random.uniform(0.015, 0.035)
                        else:
                            daily_noise = random.uniform(0.005, 0.020)
                        new_price *= (1 + daily_noise)

                    # 更新价格数据
                    temp_data.loc[i, 'close'] = new_price
                    temp_data.loc[i, 'open'] = new_price * random.uniform(0.998, 1.002)
                    temp_data.loc[i, 'high'] = new_price * random.uniform(1.005, 1.015)
                    temp_data.loc[i, 'low'] = new_price * random.uniform(0.985, 0.995)

                # 验证是否真正形成金叉
                from indicators.complete_indicator_registry import complete_registry
                temp_kdj = complete_registry.create_indicator('KDJ')
                temp_result = temp_kdj.calculate(temp_data)

                if len(temp_result) > 0 and 'K' in temp_result.columns and 'D' in temp_result.columns:
                    k_values = temp_result['K']
                    d_values = temp_result['D']

                    # 检查最新状态和最近金叉
                    latest_k = k_values.iloc[-1]
                    latest_d = d_values.iloc[-1]

                    # 检查最近10个点的金叉情况
                    recent_crosses = []
                    for j in range(max(1, len(k_values) - 10), len(k_values)):
                        if j >= 1:
                            if k_values.iloc[j-1] <= d_values.iloc[j-1] and k_values.iloc[j] > d_values.iloc[j]:
                                recent_crosses.append(j)

                    # 成功条件：K>D 且 最近有金叉
                    k_above_d = latest_k > latest_d
                    has_recent_cross = len(recent_crosses) > 0

                    if k_above_d and has_recent_cross:
                        # 成功生成金叉！
                        logger.info(f"✅ 第{attempt+1}次尝试成功！K={latest_k:.2f} > D={latest_d:.2f}, 最近金叉:{recent_crosses}")
                        adjusted_data = temp_data
                        break
                    else:
                        # 未成功，记录信息并继续尝试
                        logger.info(f"❌ 第{attempt+1}次尝试失败: K={latest_k:.2f}, D={latest_d:.2f}, K>D:{k_above_d}, 最近金叉:{recent_crosses}")

                        if attempt == max_attempts - 1:
                            # 最后一次尝试，强制调整最后几个点
                            logger.info(f"🔧 最后一次尝试，强制调整最后几个点确保金叉")
                            # 强制让最后几个点快速上涨
                            for j in range(max(0, n-5), n):
                                current_price = temp_data.loc[j, 'close']
                                boosted_price = current_price * random.uniform(1.03, 1.08)  # 强制3-8%上涨
                                temp_data.loc[j, 'close'] = boosted_price
                                temp_data.loc[j, 'high'] = boosted_price * 1.01
                            adjusted_data = temp_data
                else:
                    logger.warning(f"第{attempt+1}次尝试：无法计算KDJ指标")
                    if attempt == max_attempts - 1:
                        # 使用备用方案
                        adjusted_data = temp_data

            # 记录调整结果
            final_price_change = (adjusted_data['close'].iloc[-1] / base_price - 1) * 100
            min_price_change = (adjusted_data['close'].min() / base_price - 1) * 100
            logger.info(f"KDJ_GOLDEN_CROSS完成: 最低跌幅={min_price_change:.2f}%, 最终涨幅={final_price_change:.2f}%")

            return adjusted_data
        else:
            return self._create_macd_pattern(data, pattern_type)

    def _create_boll_pattern(self, data: pd.DataFrame, pattern_type: str) -> pd.DataFrame:
        """创建布林带特定形态 - Ultra Think终极修复版本：确保生成的数据真正突破布林带"""
        adjusted_data = data.copy()
        n = len(adjusted_data)

        if pattern_type in ['UPPER_BREAKOUT', 'BREAKOUT_UP']:
            # 🔧 Ultra Think终极策略：迭代生成确保最终突破的数据
            logger.info(f"Ultra Think终极策略: 创建BOLL上轨突破形态，数据长度: {n}")

            base_price = adjusted_data['close'].iloc[0]
            logger.info(f"BOLL上轨突破: 基准价格={base_price:.2f}")

            # 迭代策略：多次尝试直到生成真正突破的数据
            max_attempts = 5
            for attempt in range(max_attempts):
                logger.info(f"尝试第{attempt+1}次生成突破数据")

                # 重置数据
                temp_data = adjusted_data.copy()

                # 生成渐进式上涨数据
                for i in range(n):
                    if i < n * 0.6:
                        # 前60%：稳定波动
                        oscillation = random.uniform(-0.015, 0.015)
                        price_factor = 1 + oscillation
                    else:
                        # 后40%：逐步上涨，最后强力突破
                        progress = (i - n * 0.6) / (n * 0.4)
                        # 基础上涨 + 最后阶段的强力突破
                        base_increase = 0.05 * progress  # 基础5%上涨
                        if progress > 0.8:  # 最后20%强力突破
                            extra_boost = 0.15 * (progress - 0.8) / 0.2  # 额外15%突破
                        else:
                            extra_boost = 0

                        price_factor = 1 + base_increase + extra_boost
                        # 添加随机波动但保持上涨趋势
                        noise = random.uniform(-0.01, 0.03)
                        price_factor *= (1 + noise)

                    new_price = base_price * price_factor
                    temp_data.loc[i, 'close'] = new_price
                    temp_data.loc[i, 'open'] = new_price * random.uniform(0.998, 1.002)
                    temp_data.loc[i, 'high'] = new_price * random.uniform(1.002, 1.015)
                    temp_data.loc[i, 'low'] = new_price * random.uniform(0.990, 0.998)

                # 验证是否真正突破
                from indicators.complete_indicator_registry import complete_registry
                temp_boll = complete_registry.create_indicator('BOLL')
                temp_result = temp_boll.calculate(temp_data)

                if len(temp_result) > 0 and 'upper' in temp_result.columns:
                    final_close = temp_data['close'].iloc[-1]
                    final_upper = temp_result['upper'].iloc[-1]

                    if final_close > final_upper:
                        # 成功突破！
                        breakthrough_margin = ((final_close - final_upper) / final_upper * 100)
                        logger.info(f"✅ 第{attempt+1}次尝试成功！最终突破幅度: {breakthrough_margin:+.2f}%")
                        adjusted_data = temp_data
                        break
                    else:
                        # 未突破，记录信息并继续尝试
                        gap = ((final_close - final_upper) / final_upper * 100)
                        logger.info(f"❌ 第{attempt+1}次尝试失败，差距: {gap:+.2f}%")

                        if attempt == max_attempts - 1:
                            # 最后一次尝试，强制突破
                            logger.info(f"🔧 最后一次尝试，强制突破")
                            required_price = final_upper * 1.05  # 强制5%突破
                            temp_data.loc[n-1, 'close'] = required_price
                            temp_data.loc[n-1, 'high'] = required_price * 1.01
                            adjusted_data = temp_data
                else:
                    logger.warning(f"第{attempt+1}次尝试：无法计算布林带")
                    if attempt == max_attempts - 1:
                        # 使用备用方案
                        adjusted_data = temp_data

        elif pattern_type in ['LOWER_BREAKOUT', 'BREAKOUT_DOWN']:
            # 🔧 创建下轨突破形态：确保最后几个点保持突破状态
            logger.info(f"创建BOLL下轨突破形态，数据长度: {n}")

            # 突破点设置在最后10个点开始，确保最后几个点都是突破状态
            breakout_start = max(n - 10, int(n * 0.7))
            base_price = adjusted_data['close'].iloc[0]

            logger.info(f"BOLL下轨突破: 基准价格={base_price:.2f}, 突破开始点={breakout_start}")

            for i in range(n):
                if i < breakout_start:
                    # 前期：在布林带中轨附近波动，为突破做准备
                    oscillation = random.uniform(-0.03, 0.03)
                    price_factor = 1 + oscillation
                else:
                    # 后期：持续突破下轨，确保最后几个点都保持突破状态
                    breakout_progress = (i - breakout_start) / (n - breakout_start)
                    # 突破幅度逐渐增强，最终达到8-12%的突破
                    breakout_strength = 0.08 + (0.04 * breakout_progress)
                    price_factor = 1 - breakout_strength

                    # 添加小幅波动但保持突破状态
                    noise = random.uniform(-0.02, 0.01)
                    price_factor *= (1 + noise)

                new_price = base_price * price_factor
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.995, 1.005)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.005, 1.015)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.985, 0.995)
        else:
            # 其他形态使用默认处理
            logger.info(f"BOLL形态 {pattern_type} 使用默认处理")

        # 记录调整结果
        final_price_change = (adjusted_data['close'].iloc[-1] / adjusted_data['close'].iloc[0] - 1) * 100
        logger.info(f"BOLL_{pattern_type}完成: 最终价格变化={final_price_change:+.2f}%")

        return adjusted_data

    def _create_cci_pattern(self, data: pd.DataFrame, pattern_type: str) -> pd.DataFrame:
        """创建CCI特定形态 - 终极修复：确保最终CCI值在正确的极值区间"""
        adjusted_data = data.copy()
        n = len(adjusted_data)
        
        if pattern_type == 'OVERSOLD':
            # 创建CCI超卖形态：需要CCI最终 < -100
            # CCI = (典型价格 - SMA(典型价格,20)) / (0.015 * MeanDeviation)
            base_price = adjusted_data['close'].iloc[0]
            
            # 策略：持续下跌到最后，并操控高低价让典型价格偏离收盘价
            for i in range(n):
                if i < n // 3:
                    # 前1/3：温和下跌
                    decline_factor = 1 - (0.20 * i / (n // 3))  # 下跌20%
                    new_price = base_price * decline_factor
                elif i < 2 * n // 3:
                    # 中1/3：加速下跌
                    decline_factor = 0.80 - (0.30 * (i - n // 3) / (n // 3))  # 继续下跌30%
                    new_price = base_price * decline_factor
                else:
                    # 后1/3：持续下跌到最后，确保CCI < -100
                    decline_factor = 0.50 - (0.25 * (i - 2 * n // 3) / (n // 3))  # 再下跌25%
                    # 最后几天稍微增加下跌确保CCI足够低
                    if i >= n - 5:
                        decline_factor *= random.uniform(0.96, 0.98)
                    new_price = base_price * decline_factor
                
                # 为了形成CCI超卖，需要创造收盘价远低于高低价均值的情况
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(1.00, 1.02)
                
                # 关键：高价和低价设置要让典型价格(H+L+C)/3 明显高于收盘价
                high_multiplier = random.uniform(1.10, 1.20)  # 高价比收盘价高10-20%
                low_multiplier = random.uniform(1.05, 1.12)   # 低价也比收盘价高5-12%
                
                adjusted_data.loc[i, 'high'] = new_price * high_multiplier
                adjusted_data.loc[i, 'low'] = new_price * low_multiplier
            
        elif pattern_type == 'OVERBOUGHT':
            # 创建CCI超买形态：需要CCI最终 > 100
            base_price = adjusted_data['close'].iloc[0]
            
            # 策略：持续上涨到最后，并操控高低价让典型价格偏离收盘价
            for i in range(n):
                if i < n // 3:
                    # 前1/3：温和上涨
                    rise_factor = 1 + (0.25 * i / (n // 3))  # 上涨25%
                    new_price = base_price * rise_factor
                elif i < 2 * n // 3:
                    # 中1/3：加速上涨
                    rise_factor = 1.25 + (0.35 * (i - n // 3) / (n // 3))  # 继续上涨35%
                    new_price = base_price * rise_factor
                else:
                    # 后1/3：持续上涨到最后，确保CCI > 100
                    rise_factor = 1.60 + (0.30 * (i - 2 * n // 3) / (n // 3))  # 再上涨30%
                    # 最后几天稍微增加上涨确保CCI足够高
                    if i >= n - 5:
                        rise_factor *= random.uniform(1.02, 1.04)
                    new_price = base_price * rise_factor
                
                # 为了形成CCI超买，需要创造收盘价远高于高低价均值的情况
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.98, 1.00)
                
                # 关键：高价和低价设置要让典型价格(H+L+C)/3 明显低于收盘价
                high_multiplier = random.uniform(0.95, 1.02)  # 高价接近收盘价
                low_multiplier = random.uniform(0.80, 0.90)   # 低价比收盘价低10-20%
                
                adjusted_data.loc[i, 'high'] = new_price * high_multiplier
                adjusted_data.loc[i, 'low'] = new_price * low_multiplier

        elif pattern_type == 'GOLDEN_CROSS':
            # 🔧 新增：创建CCI金叉形态：从超卖区域回升，穿越零轴或关键阈值
            base_price = adjusted_data['close'].iloc[0]
            cross_point = max(n - 10, n//2)  # 金叉发生在后半段

            for i in range(n):
                if i < cross_point:
                    # 前半段：下跌到超卖区域，为金叉做准备
                    decline_factor = 1 - (0.25 * i / cross_point)  # 下跌25%
                    new_price = base_price * decline_factor

                    # 创造CCI超卖条件：典型价格高于收盘价
                    adjusted_data.loc[i, 'close'] = new_price
                    adjusted_data.loc[i, 'open'] = new_price * random.uniform(1.00, 1.02)
                    adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.08, 1.15)
                    adjusted_data.loc[i, 'low'] = new_price * random.uniform(1.03, 1.08)
                else:
                    # 后半段：强势回升，形成CCI金叉
                    recovery_ratio = (i - cross_point) / (n - cross_point)
                    recovery_factor = 0.75 + (0.40 * recovery_ratio)  # 从75%回升到115%
                    new_price = base_price * recovery_factor

                    # 创造CCI上升条件：典型价格接近或低于收盘价
                    adjusted_data.loc[i, 'close'] = new_price
                    adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.98, 1.00)
                    adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.00, 1.03)
                    adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.95, 0.98)

        elif pattern_type == 'DEATH_CROSS':
            # 🔧 新增：创建CCI死叉形态：从超买区域回落，穿越零轴或关键阈值
            base_price = adjusted_data['close'].iloc[0]
            cross_point = max(n - 10, n//2)  # 死叉发生在后半段

            for i in range(n):
                if i < cross_point:
                    # 前半段：上涨到超买区域，为死叉做准备
                    rise_factor = 1 + (0.30 * i / cross_point)  # 上涨30%
                    new_price = base_price * rise_factor

                    # 创造CCI超买条件：典型价格低于收盘价
                    adjusted_data.loc[i, 'close'] = new_price
                    adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.98, 1.00)
                    adjusted_data.loc[i, 'high'] = new_price * random.uniform(0.98, 1.02)
                    adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.85, 0.92)
                else:
                    # 后半段：快速下跌，形成CCI死叉
                    decline_ratio = (i - cross_point) / (n - cross_point)
                    decline_factor = 1.30 - (0.45 * decline_ratio)  # 从130%下跌到85%
                    new_price = base_price * decline_factor

                    # 创造CCI下降条件：典型价格高于收盘价
                    adjusted_data.loc[i, 'close'] = new_price
                    adjusted_data.loc[i, 'open'] = new_price * random.uniform(1.00, 1.02)
                    adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.05, 1.12)
                    adjusted_data.loc[i, 'low'] = new_price * random.uniform(1.02, 1.05)

        return adjusted_data

    def _create_dma_pattern(self, data: pd.DataFrame, pattern_type: str) -> pd.DataFrame:
        """创建DMA（不同期移动平均）特定形态"""
        adjusted_data = data.copy()
        n = len(adjusted_data)
        base_price = adjusted_data['close'].iloc[0]
        
        if pattern_type == 'GOLDEN_CROSS':
            # DMA金叉：短期DMA上穿长期DMA
            cross_point = max(n - 8, n//2)
            
            for i in range(n):
                if i < cross_point:
                    # 下跌阶段，为金叉做准备
                    decline_factor = 1 - (0.12 * i / cross_point)  # 下跌12%
                else:
                    # 上升阶段，形成DMA金叉
                    recovery_ratio = (i - cross_point) / (n - cross_point)
                    decline_factor = 0.88 + (0.20 * recovery_ratio)  # 反弹20%
                
                new_price = base_price * decline_factor
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.995, 1.005)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.01, 1.03)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.97, 0.995)
                
        elif pattern_type == 'DEATH_CROSS':
            # DMA死叉：短期DMA下穿长期DMA
            cross_point = max(n - 8, n//2)
            
            for i in range(n):
                if i < cross_point:
                    # 上升阶段，为死叉做准备
                    rise_factor = 1 + (0.15 * i / cross_point)  # 上涨15%
                else:
                    # 下跌阶段，形成DMA死叉
                    decline_ratio = (i - cross_point) / (n - cross_point)
                    rise_factor = 1.15 - (0.25 * decline_ratio)  # 下跌25%
                
                new_price = base_price * rise_factor
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.995, 1.005)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.01, 1.03)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.97, 0.995)
                
        elif pattern_type == 'SUPPORT_RESISTANCE':
            # DMA支撑阻力：创建DMA线形成支撑或阻力的形态
            support_level = base_price
            
            for i in range(n):
                # 创建价格在DMA线附近反复测试的走势
                cycle_position = (i % 10) / 10.0  # 10天一个周期
                
                if i < n // 2:
                    # 前半段：价格在支撑线上方震荡
                    base_factor = 1.0 + (0.05 * i / (n // 2))  # 略微上升
                    oscillation = 0.03 * np.sin(cycle_position * 2 * np.pi)  # 振荡
                    price_factor = base_factor + oscillation
                    
                    # 偶尔测试支撑线
                    if i % 15 == 14:  # 每15天测试一次支撑
                        price_factor = base_factor - 0.05  # 短暂跌破
                        
                else:
                    # 后半段：价格在阻力线下方震荡
                    resistance_level = 1.05 + (0.02 * (i - n//2) / (n - n//2))
                    oscillation = 0.025 * np.sin(cycle_position * 2 * np.pi)
                    price_factor = resistance_level - 0.02 + oscillation
                    
                    # 偶尔测试阻力线
                    if i % 12 == 11:  # 每12天测试一次阻力
                        price_factor = resistance_level + 0.02  # 短暂突破后回落
                
                new_price = base_price * price_factor
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.998, 1.002)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.005, 1.02)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.98, 0.998)
        
        return adjusted_data

    def _create_wma_pattern(self, data: pd.DataFrame, pattern_type: str) -> pd.DataFrame:
        """创建WMA（加权移动平均）特定形态"""
        adjusted_data = data.copy()
        n = len(adjusted_data)
        
        if pattern_type == 'GOLDEN_CROSS':
            # 创建WMA金叉形态：短期WMA上穿长期WMA
            cross_point = max(n - 12, n//2)
            base_price = adjusted_data['close'].iloc[0]
            
            for i in range(n):
                if i < cross_point:
                    # 下跌阶段，为金叉做准备
                    decline_factor = 1 - (0.18 * i / cross_point)  # 下跌18%
                    trend_noise = random.uniform(0.98, 1.02)  # 加噪音
                    new_price = base_price * decline_factor * trend_noise
                else:
                    # 上升阶段，形成WMA金叉
                    recovery_ratio = (i - cross_point) / (n - cross_point)
                    # 使用加速上升来形成明显的WMA金叉
                    acceleration_factor = 1 + (recovery_ratio ** 1.5) * 0.3  # 加速上涨30%
                    decline_factor = 0.82 * acceleration_factor
                    new_price = base_price * decline_factor
                
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.99, 1.01)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.01, 1.04)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.96, 0.99)
                
        elif pattern_type == 'DEATH_CROSS':
            # 创建WMA死叉形态：短期WMA下穿长期WMA
            cross_point = max(n - 12, n//2)
            base_price = adjusted_data['close'].iloc[0]
            
            for i in range(n):
                if i < cross_point:
                    # 上涨阶段，为死叉做准备
                    rise_factor = 1 + (0.25 * i / cross_point)  # 上涨25%
                    new_price = base_price * rise_factor
                else:
                    # 下跌阶段，形成WMA死叉
                    decline_ratio = (i - cross_point) / (n - cross_point)
                    # 加速下跌形成明显的WMA死叉
                    acceleration_factor = 1 + (decline_ratio ** 1.5) * 0.4
                    rise_factor = 1.25 - (0.35 * acceleration_factor)  # 下跌35%
                    new_price = base_price * rise_factor
                
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.99, 1.01)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.01, 1.03)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.97, 0.99)
                
        elif pattern_type == 'BULLISH_ARRANGEMENT':
            # 创建WMA多头排列：短、中、长期WMA呈多头排列
            base_price = adjusted_data['close'].iloc[0]
            
            for i in range(n):
                # 持续上涨趋势，形成多头排列
                trend_factor = 1 + (0.25 * i / n)  # 逐步上涨25%
                momentum = 1 + (0.05 * (i / n) ** 2)  # 增强趋势动量
                new_price = base_price * trend_factor * momentum
                
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.99, 1.01)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.01, 1.05)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.95, 0.99)
        
        return adjusted_data

    def _create_generic_pattern(self, data: pd.DataFrame, pattern_type: str) -> pd.DataFrame:
        """创建通用形态"""
        adjusted_data = data.copy()
        n = len(adjusted_data)
        base_price = adjusted_data['close'].iloc[0]
        
        # 根据形态名称判断趋势方向
        if any(keyword in pattern_type.upper() for keyword in ['GOLDEN', 'BUY', 'BULL', 'UP', 'SUPPORT']):
            # 上涨形态
            for i in range(n):
                trend_factor = 1 + (0.20 * i / n)  # 逐步上涨20%
                new_price = base_price * trend_factor
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.99, 1.01)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.01, 1.05)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.95, 0.99)
                
        elif any(keyword in pattern_type.upper() for keyword in ['DEATH', 'SELL', 'BEAR', 'DOWN', 'RESISTANCE']):
            # 下跌形态
            for i in range(n):
                trend_factor = 1 - (0.20 * i / n)  # 逐步下跌20%
                new_price = base_price * trend_factor
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.99, 1.01)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.01, 1.03)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.97, 0.99)
        
        return adjusted_data
    
    def _generate_basic_stockinfo_data(self, stock_code: str, history_days: int) -> pd.DataFrame:
        """生成基础stockInfo数据"""
        # 生成日期序列（确保获得足够的工作日）
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=history_days * 2)  # 生成更多天数以确保有足够工作日
        dates = pd.date_range(start=start_date, end=end_date, freq='D')

        # 过滤工作日并确保数量正确
        work_dates = [d for d in dates if d.weekday() < 5]
        if len(work_dates) >= history_days:
            dates = work_dates[-history_days:]  # 取最近的history_days个工作日
        else:
            # 如果工作日不够，补充一些日期
            dates = work_dates
            while len(dates) < history_days:
                dates.append(dates[-1] + timedelta(days=1))
        
        # 生成基础价格数据
        base_price = random.uniform(5.0, 50.0)
        prices = self._generate_realistic_price_series(base_price, len(dates))
        
        # 生成成交量数据
        volumes = self._generate_realistic_volume_series(len(dates))
        
        # 构建基础数据
        data = pd.DataFrame({
            'date': [d.strftime('%Y%m%d') for d in dates],
            'code': [stock_code] * len(dates),
            'name': [f'测试股票_{stock_code}'] * len(dates),
            'open': [p * (1 + random.uniform(-0.02, 0.02)) for p in prices],
            'high': [p * (1 + random.uniform(0.01, 0.05)) for p in prices],
            'low': [p * (1 + random.uniform(-0.05, -0.01)) for p in prices],
            'close': prices,
            'volume': volumes,
            'industry': [random.choice(self.industries)] * len(dates)
        })
        
        # 确保价格逻辑正确
        for i in range(len(data)):
            low = min(data.loc[i, 'open'], data.loc[i, 'close']) * 0.95
            high = max(data.loc[i, 'open'], data.loc[i, 'close']) * 1.05
            data.loc[i, 'low'] = min(data.loc[i, 'low'], low)
            data.loc[i, 'high'] = max(data.loc[i, 'high'], high)
        
        return data

    def _generate_random_base_data(self, stock_code: str, history_days: int) -> pd.DataFrame:
        """生成随机基础数据"""
        # 生成日期序列（确保获得足够的工作日）
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=history_days * 2)  # 生成更多天数
        dates = pd.date_range(start=start_date, end=end_date, freq='D')

        # 过滤工作日并确保数量正确
        work_dates = [d for d in dates if d.weekday() < 5]
        if len(work_dates) >= history_days:
            work_dates = work_dates[-history_days:]  # 取最近的history_days个工作日
        else:
            # 如果工作日不够，补充一些日期
            while len(work_dates) < history_days:
                work_dates.append(work_dates[-1] + timedelta(days=1))

        # 生成随机价格数据（更加随机）
        base_price = random.uniform(3.0, 100.0)
        prices = []
        current_price = base_price

        for _ in range(len(work_dates)):
            # 随机价格变动
            change_rate = random.uniform(-0.1, 0.1)
            current_price = max(1.0, current_price * (1 + change_rate))
            prices.append(current_price)

        # 生成随机成交量
        base_volume = random.randint(500000, 20000000)
        volumes = [base_volume * random.uniform(0.3, 3.0) for _ in range(len(work_dates))]

        # 构建随机数据
        data = pd.DataFrame({
            'date': [d.strftime('%Y%m%d') for d in work_dates],
            'code': [stock_code] * len(work_dates),
            'name': [f'随机股票_{stock_code}'] * len(work_dates),
            'open': [p * random.uniform(0.98, 1.02) for p in prices],
            'high': [p * random.uniform(1.01, 1.08) for p in prices],
            'low': [p * random.uniform(0.92, 0.99) for p in prices],
            'close': prices,
            'volume': volumes,
            'industry': [random.choice(self.industries)] * len(work_dates)
        })

        # 修正价格逻辑
        for i in range(len(data)):
            open_price = data.loc[i, 'open']
            close_price = data.loc[i, 'close']

            # 确保最高价不低于开盘价和收盘价
            data.loc[i, 'high'] = max(data.loc[i, 'high'], open_price, close_price)

            # 确保最低价不高于开盘价和收盘价
            data.loc[i, 'low'] = min(data.loc[i, 'low'], open_price, close_price)

        return data

    def _generate_realistic_price_series(self, base_price: float, length: int) -> List[float]:
        """生成真实的价格序列"""
        prices = [base_price]

        for i in range(1, length):
            # 使用随机游走模型
            change_rate = np.random.normal(0, 0.02)  # 2%的标准差
            new_price = prices[-1] * (1 + change_rate)

            # 确保价格在合理范围内
            new_price = max(1.0, min(1000.0, new_price))
            prices.append(new_price)

        return prices

    def _generate_realistic_volume_series(self, length: int) -> List[float]:
        """生成真实的成交量序列"""
        base_volume = random.randint(1000000, 10000000)
        volumes = []

        for i in range(length):
            # 成交量有一定的随机性和趋势性
            volume_multiplier = np.random.lognormal(0, 0.5)  # 对数正态分布
            volume = base_volume * volume_multiplier

            # 确保成交量在合理范围内
            volume = max(10000, min(100000000, volume))
            volumes.append(volume)

        return volumes

    def _ensure_stockinfo_compatibility(self, data: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """确保数据与stockInfo对象结构完全兼容"""
        try:
            # 创建完整的stockInfo兼容数据
            compatible_data = pd.DataFrame()

            # 基础字段
            compatible_data['date'] = data.get('date', pd.Series([datetime.now().strftime('%Y%m%d')] * len(data)))
            compatible_data['code'] = data.get('code', pd.Series([stock_code] * len(data)))
            compatible_data['name'] = data.get('name', pd.Series([f'股票_{stock_code}'] * len(data)))
            compatible_data['level'] = pd.Series([Period.DAILY] * len(data))

            # 价格字段
            compatible_data['open'] = pd.to_numeric(data.get('open', 10.0), errors='coerce').fillna(10.0)
            compatible_data['high'] = pd.to_numeric(data.get('high', 10.5), errors='coerce').fillna(10.5)
            compatible_data['low'] = pd.to_numeric(data.get('low', 9.5), errors='coerce').fillna(9.5)
            compatible_data['close'] = pd.to_numeric(data.get('close', 10.0), errors='coerce').fillna(10.0)

            # 交易量字段
            compatible_data['volume'] = pd.to_numeric(data.get('volume', 1000000), errors='coerce').fillna(1000000).astype('float64')

            # 计算衍生字段
            compatible_data['turnover_rate'] = self._calculate_turnover_rate(compatible_data['volume'])
            compatible_data['price_change'] = self._calculate_price_change(compatible_data['close'])
            compatible_data['price_range'] = self._calculate_price_range(
                compatible_data['high'], compatible_data['low']
            )

            # 分类字段
            compatible_data['industry'] = data.get('industry', pd.Series(['测试行业'] * len(data)))

            # 时间戳字段
            compatible_data['datetime'] = pd.to_datetime(
                compatible_data['date'], format='%Y%m%d'
            ).dt.strftime('%Y-%m-%d %H:%M:%S')
            compatible_data['seq'] = pd.Series(range(len(data)), dtype='int32')

            # 确保数据类型正确
            compatible_data = self._ensure_correct_dtypes(compatible_data)

            # 🔧 关键修复：保留原始数据的attrs（包括expected_pattern等重要属性）
            if hasattr(data, 'attrs') and data.attrs:
                compatible_data.attrs.update(data.attrs)
                logger.debug(f"✅ 保留了原始数据的attrs: {list(data.attrs.keys())}")

            return compatible_data

        except Exception as e:
            logger.error(f"确保stockInfo兼容性失败: {e}")
            raise

    def _calculate_turnover_rate(self, volumes: pd.Series) -> pd.Series:
        """计算换手率"""
        # 简单模拟换手率计算
        base_turnover = 0.05  # 5%基础换手率
        return volumes.apply(lambda v: base_turnover * random.uniform(0.1, 3.0))

    def _calculate_price_change(self, closes: pd.Series) -> pd.Series:
        """计算价格变动"""
        price_changes = [0.0]  # 第一天变动为0

        for i in range(1, len(closes)):
            if closes.iloc[i-1] != 0:
                change = (closes.iloc[i] - closes.iloc[i-1]) / closes.iloc[i-1]
            else:
                change = 0.0
            price_changes.append(change)

        return pd.Series(price_changes)

    def _calculate_price_range(self, highs: pd.Series, lows: pd.Series) -> pd.Series:
        """计算价格区间"""
        return (highs - lows) / lows

    def _ensure_correct_dtypes(self, data: pd.DataFrame) -> pd.DataFrame:
        """确保数据类型正确"""
        # 字符串字段
        string_fields = ['date', 'code', 'name', 'level', 'industry', 'datetime']
        for field in string_fields:
            if field in data.columns:
                data[field] = data[field].astype(str)

        # 浮点数字段
        float_fields = ['open', 'high', 'low', 'close', 'volume', 'turnover_rate', 'price_change', 'price_range']
        for field in float_fields:
            if field in data.columns:
                data[field] = pd.to_numeric(data[field], errors='coerce').fillna(0.0)

        # 整数字段
        int_fields = ['seq']
        for field in int_fields:
            if field in data.columns:
                data[field] = pd.to_numeric(data[field], errors='coerce').fillna(0).astype('int32')

        return data

    def _validate_stockinfo_structure(self, data: pd.DataFrame) -> bool:
        """验证stockInfo结构"""
        try:
            # 检查必需字段
            required_fields = ['date', 'code', 'name', 'open', 'high', 'low', 'close', 'volume']

            for field in required_fields:
                if field not in data.columns:
                    logger.error(f"缺少必需字段: {field}")
                    return False

            # 检查数据完整性
            if len(data) == 0:
                logger.error("数据为空")
                return False

            # 检查并修正价格逻辑
            price_fix_count = 0
            for i in range(len(data)):
                high = data.loc[i, 'high']
                low = data.loc[i, 'low']
                open_price = data.loc[i, 'open']
                close = data.loc[i, 'close']

                if not (low <= open_price <= high and low <= close <= high):
                    price_fix_count += 1
                    # 修正价格逻辑：确保高价是最高的，低价是最低的
                    data.loc[i, 'high'] = max(high, open_price, close)
                    data.loc[i, 'low'] = min(low, open_price, close)
                    
            if price_fix_count > 0:
                logger.debug(f"修正了 {price_fix_count} 行价格逻辑")

            # 检查成交量
            if (data['volume'] <= 0).any():
                logger.warning("存在非正成交量，将修正为正值")
                data['volume'] = data['volume'].abs() + 1

            logger.debug("stockInfo结构验证通过")
            return True

        except Exception as e:
            logger.error(f"stockInfo结构验证失败: {e}")
            return False

    def cleanup(self):
        """清理资源"""
        try:
            if hasattr(self.base_generator, 'cleanup'):
                self.base_generator.cleanup()
            logger.debug("StockInfo兼容数据生成器资源清理完成")
        except Exception as e:
            logger.warning(f"清理资源时出现警告: {e}")

    def get_supported_indicators(self) -> List[str]:
        """获取支持的指标列表"""
        return list(self.indicator_history_requirements.keys())

    def get_indicator_history_requirement(self, indicator_name: str) -> int:
        """获取指标历史数据需求（公共接口）"""
        return self._get_indicator_history_requirement(indicator_name)

    def generate_large_scale_data(self,
                                stock_codes: List[str],
                                indicator_name: str,
                                pattern_type: str,
                                history_days: int = None) -> Dict[str, pd.DataFrame]:
        """
        生成大规模数据（支持4000+股票）

        Args:
            stock_codes: 股票代码列表
            indicator_name: 指标名称
            pattern_type: 形态类型
            history_days: 历史数据天数

        Returns:
            Dict[str, pd.DataFrame]: 股票代码到数据的映射
        """
        try:
            logger.info(f"开始生成大规模数据: {len(stock_codes)} 只股票")

            results = {}

            for i, stock_code in enumerate(stock_codes):
                if i % 100 == 0:
                    logger.info(f"进度: {i}/{len(stock_codes)} ({i/len(stock_codes)*100:.1f}%)")

                try:
                    data = self.generate_stockinfo_compatible_data(
                        indicator_name, pattern_type, stock_code, history_days
                    )
                    results[stock_code] = data

                except Exception as e:
                    logger.warning(f"生成股票 {stock_code} 数据失败: {e}")
                    # 继续处理其他股票
                    continue

            logger.info(f"大规模数据生成完成: {len(results)}/{len(stock_codes)} 成功")
            return results

        except Exception as e:
            logger.error(f"大规模数据生成失败: {e}")
            return {}

    def get_stockinfo_fields(self) -> Dict[str, str]:
        """获取stockInfo字段定义"""
        return self.stockinfo_fields.copy()

    def validate_data_compatibility(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        验证数据兼容性并返回详细报告

        Args:
            data: 待验证的数据

        Returns:
            Dict: 验证报告
        """
        report = {
            'is_compatible': False,
            'missing_fields': [],
            'invalid_data_types': [],
            'data_quality_issues': [],
            'recommendations': []
        }

        try:
            # 检查必需字段
            required_fields = ['date', 'code', 'name', 'open', 'high', 'low', 'close', 'volume']
            missing_fields = [field for field in required_fields if field not in data.columns]
            report['missing_fields'] = missing_fields

            if missing_fields:
                report['recommendations'].append(f"添加缺失字段: {', '.join(missing_fields)}")

            # 检查数据类型
            expected_types = {
                'date': 'object',
                'code': 'object',
                'name': 'object',
                'open': 'float64',
                'high': 'float64',
                'low': 'float64',
                'close': 'float64',
                'volume': 'float64'
            }

            for field, expected_type in expected_types.items():
                if field in data.columns:
                    actual_type = str(data[field].dtype)
                    # 更宽松的类型检查
                    if expected_type == 'float64':
                        if not ('float' in actual_type or 'int' in actual_type):
                            report['invalid_data_types'].append({
                                'field': field,
                                'expected': expected_type,
                                'actual': actual_type
                            })
                    elif expected_type == 'object':
                        if actual_type not in ['object', 'string']:
                            report['invalid_data_types'].append({
                                'field': field,
                                'expected': expected_type,
                                'actual': actual_type
                            })

            # 检查数据质量
            if len(data) == 0:
                report['data_quality_issues'].append("数据为空")

            # 检查价格逻辑
            price_issues = 0
            for i in range(min(len(data), 100)):  # 只检查前100行
                try:
                    high = float(data.iloc[i]['high'])
                    low = float(data.iloc[i]['low'])
                    open_price = float(data.iloc[i]['open'])
                    close = float(data.iloc[i]['close'])

                    if not (low <= open_price <= high and low <= close <= high):
                        price_issues += 1
                except:
                    price_issues += 1

            if price_issues > 0:
                report['data_quality_issues'].append(f"发现 {price_issues} 行价格逻辑错误")

            # 综合评估
            report['is_compatible'] = (
                len(missing_fields) == 0 and
                len(report['invalid_data_types']) == 0 and
                len(report['data_quality_issues']) == 0
            )

            if report['is_compatible']:
                report['recommendations'].append("数据完全兼容stockInfo结构")

            return report

        except Exception as e:
            report['data_quality_issues'].append(f"验证过程出错: {e}")
            return report

    def _create_cross_pattern(self, data: pd.DataFrame, pattern_type: str) -> pd.DataFrame:
        """创建通用的金叉/死叉形态，适用于大多数技术指标"""
        adjusted_data = data.copy()
        n = len(adjusted_data)
        
        if pattern_type == 'GOLDEN_CROSS':
            # 创建上升趋势，利于形成金叉
            cross_point = max(n - 8, n//2)
            base_price = adjusted_data['close'].iloc[0]
            
            for i in range(n):
                if i < cross_point:
                    # 前期下降
                    decline_factor = 1 - (0.1 * i / cross_point)
                else:
                    # 后期上升形成金叉
                    recovery_ratio = (i - cross_point) / (n - cross_point)
                    decline_factor = 0.9 + (0.2 * recovery_ratio)
                
                new_price = base_price * decline_factor * (1 + random.uniform(-0.02, 0.02))
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.99, 1.01)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.00, 1.03)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.97, 1.00)
                
        elif pattern_type == 'DEATH_CROSS':
            # 创建下降趋势，形成死叉
            cross_point = max(n - 8, n//2)
            base_price = adjusted_data['close'].iloc[0]
            
            for i in range(n):
                if i < cross_point:
                    # 前期上升
                    rise_factor = 1 + (0.1 * i / cross_point)
                else:
                    # 后期下降形成死叉
                    decline_ratio = (i - cross_point) / (n - cross_point)
                    rise_factor = 1.1 - (0.2 * decline_ratio)
                
                new_price = base_price * rise_factor * (1 + random.uniform(-0.02, 0.02))
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.99, 1.01)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.00, 1.03)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.97, 1.00)
        
        return adjusted_data

    def _create_ma_pattern(self, data: pd.DataFrame, pattern_type: str) -> pd.DataFrame:
        """创建移动平均线形态（MA、EMA、WMA、DMA、SMA）"""
        adjusted_data = data.copy()
        n = len(adjusted_data)
        
        if pattern_type == 'GOLDEN_CROSS':
            # 🎯 Ultra Think优化：让金叉在检测窗口内发生
            logger.info("⚡ 创建SMA金叉形态（检测窗口内优化版）")
            base_price = adjusted_data['close'].iloc[0]
            # 调整金叉时机到数据的75%位置，确保信号在最后10个数据点内
            cross_point = int(n * 0.75)  # 在第37/38个点左右金叉，确保最后10个点能检测到
            
            for i in range(n):
                if i < cross_point:
                    # 创建先下跌然后横盘整理的走势
                    decline_phase = min(i, cross_point//2)
                    if decline_phase < cross_point//2:
                        # 下跌阶段
                        trend_factor = 1 - (0.06 * decline_phase / (cross_point//2))
                    else:
                        # 横盘整理阶段
                        trend_factor = 0.94  # 保持在底部
                else:
                    # 🎯 关键：强势反弹阶段，确保SMA5快速上穿SMA20
                    recovery_ratio = (i - cross_point) / (n - cross_point)
                    # 使用更强势的反弹因子，确保金叉清晰
                    trend_factor = 0.94 + (0.18 * (recovery_ratio ** 0.7))  
                    # 最后阶段额外加速，确保信号明显
                    if i >= n - 10:
                        extra_boost = (i - (n - 10)) / 10 * 0.05
                        trend_factor += extra_boost
                
                new_price = base_price * trend_factor
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.998, 1.002)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.008, 1.025)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.992, 1.002)
                
        elif pattern_type == 'DEATH_CROSS':
            # 🎯 Ultra Think优化：让死叉在检测窗口内发生
            logger.info("⚡ 创建SMA死叉形态（检测窗口内优化版）")
            base_price = adjusted_data['close'].iloc[0]
            # 调整死叉时机到数据的75%位置
            cross_point = int(n * 0.75)
            
            for i in range(n):
                if i < cross_point:
                    # 上升阶段
                    trend_factor = 1 + (0.12 * i / cross_point)
                else:
                    # 🎯 关键：下跌阶段，确保SMA5快速下穿SMA20
                    decline_ratio = (i - cross_point) / (n - cross_point)
                    # 使用更快的下跌速度，确保死叉清晰
                    trend_factor = 1.12 - (0.20 * (decline_ratio ** 0.7))
                    # 最后阶段额外加速下跌
                    if i >= n - 10:
                        extra_decline = (i - (n - 10)) / 10 * 0.03
                        trend_factor -= extra_decline
                
                new_price = base_price * trend_factor
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.998, 1.002)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.002, 1.015)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.985, 0.998)
                
        elif pattern_type == 'SUPPORT_RESISTANCE':
            # 🎯 Ultra Think优化：创建明显的支撑阻力形态
            logger.info("⚡ 创建SMA支撑阻力形态（优化版）")
            base_price = adjusted_data['close'].iloc[0]
            sma20_level = base_price * 1.05  # 设定SMA20作为支撑/阻力位
            
            for i in range(n):
                # 创建价格围绕SMA20波动的形态
                cycle_position = (i % 8) / 8.0  # 8个周期的波动
                if i < n * 0.6:
                    # 前期：价格在SMA之上波动
                    wave_factor = 0.02 * math.sin(cycle_position * 2 * math.pi)
                    trend_factor = 1.05 + wave_factor
                else:
                    # 后期：价格回到SMA附近，形成支撑/阻力测试
                    approaching_factor = (i - n * 0.6) / (n * 0.4)
                    base_trend = 1.05 - (0.04 * approaching_factor)  # 慢慢回落到SMA附近
                    wave_factor = 0.008 * math.sin(cycle_position * 4 * math.pi)  # 小幅波动
                    trend_factor = base_trend + wave_factor
                
                new_price = base_price * trend_factor
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.999, 1.001)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.003, 1.012)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.988, 0.999)
                
        elif pattern_type == 'TREND_FOLLOWING':
            # 🎯 Ultra Think优化：创建多头排列的趋势跟随形态
            logger.info("⚡ 创建SMA趋势跟随形态（多头排列优化版）")
            base_price = adjusted_data['close'].iloc[0]
            
            for i in range(n):
                # 创建持续上升趋势，确保SMA5 > SMA10 > SMA20 > SMA60
                progress = i / n
                # 使用指数增长模式，确保多头排列明显
                trend_factor = 1 + (0.25 * (progress ** 0.8))
                
                # 最后阶段强化趋势，确保价格明显在SMA5之上
                if i >= n * 0.7:
                    late_boost = (i - n * 0.7) / (n * 0.3) * 0.08
                    trend_factor += late_boost
                
                # 添加小幅波动，但保持整体上升趋势
                wave_factor = 0.01 * math.sin((i % 6) / 6.0 * 2 * math.pi)
                trend_factor += wave_factor
                
                new_price = base_price * trend_factor
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.998, 1.002)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.005, 1.020)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.995, 1.002)
        
        # 🎯 Ultra Think完成：确保数据合理性
        # 确保价格数据的完整性（只修改存在的列）
        if 'volume' in adjusted_data.columns:
            adjusted_data['volume'] = adjusted_data['volume'] * random.uniform(0.8, 1.2)
        if 'turnover_rate' in adjusted_data.columns:
            adjusted_data['turnover_rate'] = abs(adjusted_data['turnover_rate'] * random.uniform(0.9, 1.1))
        
        return adjusted_data

    def _create_fibonacci_pattern(self, data: pd.DataFrame, pattern_type: str) -> pd.DataFrame:
        """创建斐波那契回调形态"""
        adjusted_data = data.copy()
        n = len(adjusted_data)
        base_price = adjusted_data['close'].iloc[0]
        
        if pattern_type == 'RETRACEMENT_SUPPORT':
            # 创建上升后回调到支撑位的形态
            peak_point = n // 3
            support_point = peak_point + (n - peak_point) // 2
            
            for i in range(n):
                if i < peak_point:
                    # 上升到峰值
                    trend_factor = 1 + (0.2 * i / peak_point)
                elif i < support_point:
                    # 回调到38.2%位置
                    retracement_ratio = (i - peak_point) / (support_point - peak_point)
                    trend_factor = 1.2 - (0.2 * 0.382 * retracement_ratio)
                else:
                    # 从支撑位反弹
                    recovery_ratio = (i - support_point) / (n - support_point)
                    trend_factor = 1.2 * (1 - 0.382) + (0.1 * recovery_ratio)
                
                new_price = base_price * trend_factor
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.99, 1.01)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.01, 1.03)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.97, 0.99)
                
        elif pattern_type == 'RETRACEMENT_RESISTANCE':
            # 创建下跌后反弹到阻力位的形态  
            bottom_point = n // 3
            resistance_point = bottom_point + (n - bottom_point) // 2
            
            for i in range(n):
                if i < bottom_point:
                    # 下跌到谷底
                    trend_factor = 1 - (0.2 * i / bottom_point)
                elif i < resistance_point:
                    # 反弹到61.8%位置
                    recovery_ratio = (i - bottom_point) / (resistance_point - bottom_point)
                    trend_factor = 0.8 + (0.2 * 0.618 * recovery_ratio)
                else:
                    # 在阻力位受阻回落
                    decline_ratio = (i - resistance_point) / (n - resistance_point)
                    trend_factor = 0.8 * (1 + 0.618) - (0.05 * decline_ratio)
                
                new_price = base_price * trend_factor
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.99, 1.01)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.01, 1.03)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.97, 0.99)
        
        return adjusted_data

    def _create_aroon_pattern(self, data: pd.DataFrame, pattern_type: str) -> pd.DataFrame:
        """创建Aroon指标形态"""
        adjusted_data = data.copy()
        n = len(adjusted_data)
        base_price = adjusted_data['close'].iloc[0]
        
        if pattern_type == 'AROON_UP':
            # 创建强势上升趋势，Aroon Up接近100
            for i in range(n):
                # 在最后部分创建新高
                if i > n - 20:
                    trend_factor = 1 + (0.15 * (i - (n - 20)) / 20)
                else:
                    trend_factor = 1 + (0.05 * i / (n - 20))
                
                new_price = base_price * trend_factor
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.99, 1.01)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.01, 1.03)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.97, 0.99)
                
        elif pattern_type == 'AROON_DOWN':
            # 创建弱势下跌趋势，Aroon Down接近100
            for i in range(n):
                # 在最后部分创建新低
                if i > n - 20:
                    trend_factor = 1 - (0.15 * (i - (n - 20)) / 20)
                else:
                    trend_factor = 1 - (0.05 * i / (n - 20))
                
                new_price = base_price * trend_factor
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.99, 1.01)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.01, 1.03)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.97, 0.99)
        
        return adjusted_data

    def _create_adx_pattern(self, data: pd.DataFrame, pattern_type: str) -> pd.DataFrame:
        """创建ADX指标形态"""
        adjusted_data = data.copy()
        n = len(adjusted_data)

        if pattern_type == 'TREND_STRENGTH':
            # 创建强趋势形态：ADX值逐渐上升到25以上
            logger.info(f"🎯 创建ADX强趋势形态")

            # 前30%数据：横盘整理，ADX较低
            consolidation_end = int(n * 0.3)
            base_price = adjusted_data['close'].iloc[0]

            for i in range(consolidation_end):
                # 横盘整理，小幅波动
                noise = random.uniform(-0.01, 0.01)
                new_price = base_price * (1 + noise)
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.995, 1.005)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.005, 1.015)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.985, 0.995)

            # 中间40%数据：建立趋势，ADX上升
            trend_start = consolidation_end
            trend_end = int(n * 0.7)

            for i in range(trend_start, trend_end):
                # 建立上升趋势
                progress = (i - trend_start) / (trend_end - trend_start)
                trend_factor = 1 + (0.2 * progress)  # 20%的上升
                new_price = base_price * trend_factor

                # 增加波动性以产生更明显的方向性移动
                volatility = 0.02 + (0.01 * progress)
                daily_change = random.uniform(0.005, 0.015)  # 正向变化
                new_price *= (1 + daily_change)

                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.99, 1.005)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.01, 1.025)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.985, 0.995)

                base_price = new_price

            # 最后30%数据：维持强趋势，ADX保持高位
            for i in range(trend_end, n):
                # 维持趋势但减缓上升速度
                daily_change = random.uniform(0.002, 0.008)
                new_price = base_price * (1 + daily_change)

                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.995, 1.005)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.005, 1.02)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.99, 0.998)

                base_price = new_price

        elif pattern_type == 'GOLDEN_CROSS':
            # ADX金叉形态：+DI上穿-DI
            logger.info(f"🎯 创建ADX金叉形态")
            adjusted_data = self._create_cross_pattern(adjusted_data, pattern_type)

        else:
            # 默认处理其他形态
            logger.info(f"🔧 ADX默认形态处理: {pattern_type}")
            adjusted_data = self._create_cross_pattern(adjusted_data, pattern_type)

        return adjusted_data

    def _create_atr_pattern(self, data: pd.DataFrame, pattern_type: str) -> pd.DataFrame:
        """创建ATR指标形态"""
        adjusted_data = data.copy()
        n = len(adjusted_data)

        if pattern_type == 'GOLDEN_CROSS':
            # 创建ATR金叉形态：波动性逐渐增加
            logger.info(f"🎯 创建ATR金叉形态（波动性增加）")

            # 前40%数据：低波动性
            low_vol_end = int(n * 0.4)
            base_price = adjusted_data['close'].iloc[0]

            for i in range(low_vol_end):
                # 低波动性，小幅波动
                noise = random.uniform(-0.005, 0.005)
                new_price = base_price * (1 + noise)
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.998, 1.002)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.002, 1.008)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.992, 0.998)

            # 中间30%数据：波动性逐渐增加
            vol_increase_start = low_vol_end
            vol_increase_end = int(n * 0.7)

            for i in range(vol_increase_start, vol_increase_end):
                # 逐渐增加波动性
                progress = (i - vol_increase_start) / (vol_increase_end - vol_increase_start)
                volatility = 0.005 + (0.02 * progress)  # 从0.5%增加到2.5%

                daily_change = random.uniform(-volatility, volatility)
                new_price = base_price * (1 + daily_change)

                # 增加日内波动范围
                intraday_range = 0.01 + (0.02 * progress)  # 从1%增加到3%
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.995, 1.005)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.005, 1.005 + intraday_range)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.995 - intraday_range, 0.995)

                base_price = new_price

            # 最后30%数据：维持高波动性
            for i in range(vol_increase_end, n):
                # 维持高波动性
                volatility = 0.025  # 2.5%波动性
                daily_change = random.uniform(-volatility, volatility)
                new_price = base_price * (1 + daily_change)

                # 高日内波动范围
                intraday_range = 0.03  # 3%日内波动
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.99, 1.01)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.01, 1.01 + intraday_range)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.99 - intraday_range, 0.99)

                base_price = new_price

        elif pattern_type == 'HIGH_VOLATILITY':
            # 创建高波动性形态
            logger.info(f"🎯 创建ATR高波动性形态")
            base_price = adjusted_data['close'].iloc[0]

            for i in range(n):
                # 持续高波动性
                volatility = 0.03  # 3%波动性
                daily_change = random.uniform(-volatility, volatility)
                new_price = base_price * (1 + daily_change)

                # 高日内波动范围
                intraday_range = 0.04  # 4%日内波动
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.98, 1.02)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.02, 1.02 + intraday_range)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.98 - intraday_range, 0.98)

                base_price = new_price

        elif pattern_type == 'LOW_VOLATILITY':
            # 创建低波动性形态
            logger.info(f"🎯 创建ATR低波动性形态")
            base_price = adjusted_data['close'].iloc[0]

            for i in range(n):
                # 持续低波动性
                volatility = 0.003  # 0.3%波动性
                daily_change = random.uniform(-volatility, volatility)
                new_price = base_price * (1 + daily_change)

                # 低日内波动范围
                intraday_range = 0.005  # 0.5%日内波动
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.999, 1.001)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.001, 1.001 + intraday_range)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.999 - intraday_range, 0.999)

                base_price = new_price

        else:
            # 默认处理其他形态
            logger.info(f"🔧 ATR默认形态处理: {pattern_type}")
            adjusted_data = self._create_cross_pattern(adjusted_data, pattern_type)

        return adjusted_data

    def _create_cmo_pattern(self, data: pd.DataFrame, pattern_type: str) -> pd.DataFrame:
        """创建CMO指标形态"""
        adjusted_data = data.copy()
        n = len(adjusted_data)

        if pattern_type == 'GOLDEN_CROSS':
            # 创建CMO金叉形态：从负值区域上升到正值区域
            logger.info(f"🎯 创建CMO金叉形态（动量转正）")

            # 前40%数据：下跌趋势，CMO为负值
            downtrend_end = int(n * 0.4)
            base_price = adjusted_data['close'].iloc[0]

            for i in range(downtrend_end):
                # 下跌趋势，产生负CMO
                decline_factor = 1 - (0.1 * (i / downtrend_end))  # 10%的下跌
                new_price = base_price * decline_factor

                # 添加小幅波动
                daily_change = random.uniform(-0.015, -0.005)  # 负向变化
                new_price *= (1 + daily_change)

                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.995, 1.005)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.005, 1.015)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.985, 0.995)

                base_price = new_price

            # 中间30%数据：转折点，CMO从负转正
            reversal_start = downtrend_end
            reversal_end = int(n * 0.7)

            for i in range(reversal_start, reversal_end):
                # 逐渐转为上升趋势
                progress = (i - reversal_start) / (reversal_end - reversal_start)
                daily_change = -0.005 + (0.02 * progress)  # 从-0.5%变为+1.5%
                new_price = base_price * (1 + daily_change)

                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.995, 1.005)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.005, 1.02)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.99, 0.998)

                base_price = new_price

            # 最后30%数据：上升趋势，CMO为正值
            for i in range(reversal_end, n):
                # 持续上升趋势
                daily_change = random.uniform(0.005, 0.015)  # 正向变化
                new_price = base_price * (1 + daily_change)

                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.995, 1.005)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.005, 1.02)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.99, 0.998)

                base_price = new_price

        elif pattern_type == 'DEATH_CROSS':
            # 创建CMO死叉形态：从正值区域下降到负值区域
            logger.info(f"🎯 创建CMO死叉形态（动量转负）")

            # 前40%数据：上升趋势，CMO为正值
            uptrend_end = int(n * 0.4)
            base_price = adjusted_data['close'].iloc[0]

            for i in range(uptrend_end):
                # 上升趋势，产生正CMO
                daily_change = random.uniform(0.005, 0.015)  # 正向变化
                new_price = base_price * (1 + daily_change)

                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.995, 1.005)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.005, 1.02)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.99, 0.998)

                base_price = new_price

            # 中间30%数据：转折点，CMO从正转负
            reversal_start = uptrend_end
            reversal_end = int(n * 0.7)

            for i in range(reversal_start, reversal_end):
                # 逐渐转为下降趋势
                progress = (i - reversal_start) / (reversal_end - reversal_start)
                daily_change = 0.01 - (0.025 * progress)  # 从+1%变为-1.5%
                new_price = base_price * (1 + daily_change)

                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.995, 1.005)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.005, 1.015)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.985, 0.995)

                base_price = new_price

            # 最后30%数据：下降趋势，CMO为负值
            for i in range(reversal_end, n):
                # 持续下降趋势
                daily_change = random.uniform(-0.015, -0.005)  # 负向变化
                new_price = base_price * (1 + daily_change)

                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.995, 1.005)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.005, 1.015)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.985, 0.995)

                base_price = new_price

        elif pattern_type == 'OVERBOUGHT':
            # 创建CMO超买形态：价格持续上涨，CMO值超过50
            logger.info(f"📈 创建CMO超买形态（CMO>50）")
            
            # 前50%数据：稳定上升建立基础
            stable_end = int(n * 0.5)
            base_price = adjusted_data['close'].iloc[0]
            
            for i in range(stable_end):
                # 稳定上升趋势
                daily_change = random.uniform(0.003, 0.008)  # 0.3%-0.8%稳定增长
                new_price = base_price * (1 + daily_change)
                
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.995, 1.005)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.005, 1.015)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.985, 0.995)
                
                base_price = new_price
            
            # 后50%数据：加速上涨，产生CMO超买信号
            for i in range(stable_end, n):
                # 加速上涨，确保CMO超过50
                acceleration_factor = 1 + ((i - stable_end) / (n - stable_end)) * 0.5
                daily_change = random.uniform(0.008, 0.020) * acceleration_factor  # 加速上涨
                new_price = base_price * (1 + daily_change)
                
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.995, 1.005)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.005, 1.025)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.99, 0.998)
                
                base_price = new_price

        elif pattern_type == 'OVERSOLD':
            # 创建CMO超卖形态：价格持续下跌，CMO值低于-50
            logger.info(f"📉 创建CMO超卖形态（CMO<-50）")
            
            # 前50%数据：稳定下跌建立基础
            stable_end = int(n * 0.5)
            base_price = adjusted_data['close'].iloc[0]
            
            for i in range(stable_end):
                # 稳定下跌趋势
                daily_change = random.uniform(-0.008, -0.003)  # -0.8%到-0.3%稳定下跌
                new_price = base_price * (1 + daily_change)
                
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.995, 1.005)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.005, 1.015)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.975, 0.995)
                
                base_price = new_price
            
            # 后50%数据：加速下跌，产生CMO超卖信号
            for i in range(stable_end, n):
                # 加速下跌，确保CMO低于-50
                acceleration_factor = 1 + ((i - stable_end) / (n - stable_end)) * 0.5
                daily_change = random.uniform(-0.025, -0.008) * acceleration_factor  # 加速下跌
                new_price = base_price * (1 + daily_change)
                
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.995, 1.005)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.002, 1.01)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.975, 0.99)
                
                base_price = new_price

        elif pattern_type == 'MOMENTUM_SHIFT':
            # 创建CMO动量转换形态：价格趋势发生明显变化
            logger.info(f"⚡ 创建CMO动量转换形态（动量方向改变）")
            
            # 前30%数据：建立初始趋势
            initial_end = int(n * 0.3)
            base_price = adjusted_data['close'].iloc[0]
            
            for i in range(initial_end):
                # 初始上升趋势
                daily_change = random.uniform(0.005, 0.012)  # 正向变化
                new_price = base_price * (1 + daily_change)
                
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.995, 1.005)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.005, 1.015)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.99, 0.998)
                
                base_price = new_price
            
            # 中间40%数据：趋势转换期
            transition_start = initial_end
            transition_end = int(n * 0.7)
            
            for i in range(transition_start, transition_end):
                # 逐渐转换趋势方向
                progress = (i - transition_start) / (transition_end - transition_start)
                # 从正向变化逐渐转为负向变化
                daily_change = 0.01 * (1 - progress) - 0.015 * progress
                new_price = base_price * (1 + daily_change)
                
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.995, 1.005)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.005, 1.015)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.985, 0.995)
                
                base_price = new_price
            
            # 最后30%数据：新的趋势方向
            for i in range(transition_end, n):
                # 下降趋势，产生动量转换信号
                daily_change = random.uniform(-0.015, -0.005)  # 负向变化
                new_price = base_price * (1 + daily_change)
                
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.995, 1.005)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.005, 1.015)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.985, 0.995)
                
                base_price = new_price

        elif pattern_type == 'ZERO_CROSS':
            # 创建CMO零轴穿越形态：CMO从负值穿越到正值
            logger.info(f"🎯 创建CMO零轴穿越形态（穿越零轴）")
            
            # 前40%数据：下跌趋势，CMO为负值
            downtrend_end = int(n * 0.4)
            base_price = adjusted_data['close'].iloc[0]
            
            for i in range(downtrend_end):
                # 下跌趋势，产生负CMO
                daily_change = random.uniform(-0.012, -0.004)  # 负向变化
                new_price = base_price * (1 + daily_change)
                
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.995, 1.005)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.005, 1.015)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.985, 0.995)
                
                base_price = new_price
            
            # 中间20%数据：穿越零轴的关键时期
            cross_start = downtrend_end
            cross_end = int(n * 0.6)
            
            for i in range(cross_start, cross_end):
                # 在零轴附近震荡，然后穿越
                progress = (i - cross_start) / (cross_end - cross_start)
                # 从负向变化逐渐转为正向变化
                daily_change = -0.005 * (1 - progress) + 0.008 * progress
                new_price = base_price * (1 + daily_change)
                
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.995, 1.005)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.005, 1.015)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.99, 0.998)
                
                base_price = new_price
            
            # 最后40%数据：上升趋势，CMO为正值
            for i in range(cross_end, n):
                # 上升趋势，确保CMO为正
                daily_change = random.uniform(0.005, 0.015)  # 正向变化
                new_price = base_price * (1 + daily_change)
                
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.995, 1.005)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.005, 1.02)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.99, 0.998)
                
                base_price = new_price

        else:
            # 默认处理其他形态
            logger.info(f"🔧 CMO默认形态处理: {pattern_type}")
            adjusted_data = self._create_cross_pattern(adjusted_data, pattern_type)

        return adjusted_data

    def _create_roc_pattern(self, data: pd.DataFrame, pattern_type: str) -> pd.DataFrame:
        """创建ROC指标形态
        
        新形态支持：
        - POSITIVE_MOMENTUM: 正动量（ROC为正且增强）
        - NEGATIVE_MOMENTUM: 负动量（ROC为负且减弱）
        - ZERO_CROSS: 零轴穿越（ROC穿越零轴）
        - ACCELERATION: 加速度变化（ROC变化率增大）
        """
        adjusted_data = data.copy()
        n = len(adjusted_data)
        base_price = adjusted_data['close'].iloc[0]

        if pattern_type == 'POSITIVE_MOMENTUM':
            # 🎯 Ultra Think优化：创建正动量形态，确保信号在检测窗口内
            logger.info("⚡ 创建ROC正动量形态（检测窗口内优化版）")
            
            # 前60%：缓慢上升，建立基础
            steady_phase = int(n * 0.6)
            for i in range(steady_phase):
                daily_change = random.uniform(0.002, 0.008)  # 0.2%-0.8%缓慢增长
                new_price = base_price * (1 + daily_change)
                
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.995, 1.005)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.005, 1.015)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.985, 0.995)
                
                base_price = new_price
            
            # 后40%：强势加速上涨，在检测窗口内产生正动量信号
            for i in range(steady_phase, n):
                # 逐渐加速的正动量
                acceleration_factor = (i - steady_phase) / (n - steady_phase)
                daily_change = 0.008 + (0.02 * acceleration_factor)  # 从0.8%加速到2.8%
                new_price = base_price * (1 + daily_change)
                
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.995, 1.005)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.01, 1.035)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.99, 0.998)
                
                base_price = new_price

        elif pattern_type == 'NEGATIVE_MOMENTUM':
            # 🎯 Ultra Think优化：创建负动量形态，确保信号在检测窗口内
            logger.info("⚡ 创建ROC负动量形态（检测窗口内优化版）")
            
            # 前60%：缓慢下降，建立基础
            steady_phase = int(n * 0.6)
            for i in range(steady_phase):
                daily_change = random.uniform(-0.008, -0.002)  # -0.8%到-0.2%缓慢下降
                new_price = base_price * (1 + daily_change)
                
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.995, 1.005)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.005, 1.015)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.985, 0.995)
                
                base_price = new_price
            
            # 后40%：强势加速下跌，在检测窗口内产生负动量信号
            for i in range(steady_phase, n):
                # 逐渐加速的负动量
                acceleration_factor = (i - steady_phase) / (n - steady_phase)
                daily_change = -0.008 - (0.02 * acceleration_factor)  # 从-0.8%加速到-2.8%
                new_price = base_price * (1 + daily_change)
                
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.995, 1.005)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.005, 1.015)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.975, 0.99)
                
                base_price = new_price

        elif pattern_type == 'ZERO_CROSS':
            # 🎯 Ultra Think优化：创建零轴穿越形态，确保信号在检测窗口内
            logger.info("⚡ 创建ROC零轴穿越形态（检测窗口内优化版）")
            
            # 前50%：下跌趋势，产生负ROC
            downtrend_phase = int(n * 0.5)
            for i in range(downtrend_phase):
                daily_change = random.uniform(-0.015, -0.005)  # 持续下跌
                new_price = base_price * (1 + daily_change)
                
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.995, 1.005)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.005, 1.015)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.98, 0.995)
                
                base_price = new_price
            
            # 中间30%：横盘整理，接近零轴
            consolidation_end = int(n * 0.8)
            for i in range(downtrend_phase, consolidation_end):
                daily_change = random.uniform(-0.003, 0.003)  # 小幅波动
                new_price = base_price * (1 + daily_change)
                
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.995, 1.005)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.005, 1.015)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.985, 0.995)
                
                base_price = new_price
            
            # 最后20%：突破上涨，在检测窗口内穿越零轴
            for i in range(consolidation_end, n):
                # 强势向上突破，确保ROC穿越零轴
                daily_change = random.uniform(0.008, 0.02)  # 突破性上涨
                new_price = base_price * (1 + daily_change)
                
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.995, 1.005)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.01, 1.03)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.99, 0.998)
                
                base_price = new_price

        elif pattern_type == 'ACCELERATION':
            # 🎯 Ultra Think优化：创建加速度变化形态，确保信号在检测窗口内
            logger.info("⚡ 创建ROC加速度变化形态（检测窗口内优化版）")
            
            # 前40%：稳定趋势期
            stable_phase = int(n * 0.4)
            for i in range(stable_phase):
                daily_change = random.uniform(0.003, 0.006)  # 稳定上涨
                new_price = base_price * (1 + daily_change)
                
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.995, 1.005)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.005, 1.015)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.985, 0.995)
                
                base_price = new_price
            
            # 中间30%：减速期
            deceleration_end = int(n * 0.7)
            for i in range(stable_phase, deceleration_end):
                daily_change = random.uniform(0.001, 0.003)  # 减速
                new_price = base_price * (1 + daily_change)
                
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.995, 1.005)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.005, 1.015)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.985, 0.995)
                
                base_price = new_price
            
            # 最后30%：急速加速期，在检测窗口内产生加速度变化信号
            for i in range(deceleration_end, n):
                # 指数级加速
                acceleration_factor = (i - deceleration_end) / (n - deceleration_end)
                daily_change = 0.003 + (0.025 * (acceleration_factor ** 2))  # 二次方加速
                new_price = base_price * (1 + daily_change)
                
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.995, 1.005)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.01, 1.04)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.99, 0.998)
                
                base_price = new_price

        else:
            # 兼容处理：如果还有旧形态名称，映射到新形态
            logger.warning(f"⚠️  ROC收到旧形态名称: {pattern_type}，尝试映射到新形态")
            if pattern_type in ['GOLDEN_CROSS', 'MOMENTUM_UP']:
                logger.info("🔄 映射到 POSITIVE_MOMENTUM")
                return self._create_roc_pattern(adjusted_data, 'POSITIVE_MOMENTUM')
            elif pattern_type in ['DEATH_CROSS', 'MOMENTUM_DOWN']:
                logger.info("🔄 映射到 NEGATIVE_MOMENTUM")
                return self._create_roc_pattern(adjusted_data, 'NEGATIVE_MOMENTUM')
            else:
                logger.info(f"🔧 ROC默认形态处理: {pattern_type}")
                adjusted_data = self._create_cross_pattern(adjusted_data, pattern_type)

        return adjusted_data

    def _create_sar_pattern(self, data: pd.DataFrame, pattern_type: str) -> pd.DataFrame:
        """
        创建SAR(抛物线转向)指标形态
        SAR特点：追踪趋势转向点，止损点动态调整
        """
        logger.info(f"🎯 创建SAR指标形态: {pattern_type}")
        adjusted_data = data.copy()
        n = len(adjusted_data)
        base_price = adjusted_data['close'].iloc[0]

        if pattern_type == 'TREND_REVERSAL':
            # 创建趋势反转形态：前期下跌，后期上涨
            logger.info("📈 创建SAR趋势反转形态")
            reversal_point = int(n * 0.6)
            
            # 前60%下跌趋势
            for i in range(reversal_point):
                decline_factor = 1 - (0.15 * i / reversal_point)
                new_price = base_price * decline_factor
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.99, 1.01)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.01, 1.03)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.97, 0.99)
            
            # 后40%上涨趋势
            reversal_base = adjusted_data['close'].iloc[reversal_point-1]
            for i in range(reversal_point, n):
                growth_factor = 1 + (0.20 * (i - reversal_point) / (n - reversal_point))
                new_price = reversal_base * growth_factor
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.99, 1.01)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.01, 1.05)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.98, 1.00)

        elif pattern_type == 'UPTREND_SIGNAL':
            # 创建SAR上涨信号形态
            logger.info("📈 创建SAR上涨信号形态")
            for i in range(n):
                growth_factor = 1 + (0.25 * i / n)
                new_price = base_price * growth_factor
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.995, 1.005)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.01, 1.04)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.98, 1.00)

        elif pattern_type == 'DOWNTREND_SIGNAL':
            # 创建SAR下跌信号形态
            logger.info("📉 创建SAR下跌信号形态")
            for i in range(n):
                decline_factor = 1 - (0.20 * i / n)
                new_price = base_price * decline_factor
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.995, 1.005)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.00, 1.02)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.96, 0.98)

        elif pattern_type == 'STOP_LOSS':
            # 🔧 Ultra Think修复：创建专门的SAR止损信号形态
            logger.info("🛑 创建SAR止损信号形态")
            # 创建先上涨后快速下跌的形态，触发SAR止损点
            uptrend_point = int(n * 0.7)
            
            # 前70%稳步上涨，建立上升趋势
            for i in range(uptrend_point):
                growth_factor = 1 + (0.15 * i / uptrend_point)
                new_price = base_price * growth_factor
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.995, 1.005)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.01, 1.03)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.985, 1.00)
            
            # 后30%快速下跌，触发SAR止损信号
            peak_price = adjusted_data['close'].iloc[uptrend_point-1]
            for i in range(uptrend_point, n):
                # 快速下跌10%，明确触发SAR止损点
                decline_factor = 1 - (0.10 * (i - uptrend_point) / (n - uptrend_point))
                new_price = peak_price * decline_factor
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.99, 1.01)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.005, 1.015)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.98, 0.995)

        else:
            # 默认处理其他SAR形态
            logger.info(f"🔧 SAR默认形态处理: {pattern_type}")
            adjusted_data = self._create_cross_pattern(adjusted_data, pattern_type)

        return adjusted_data

    def _create_trix_pattern(self, data: pd.DataFrame, pattern_type: str) -> pd.DataFrame:
        """
        创建TRIX(三重指数平滑)指标形态
        TRIX特点：消除价格随机波动，识别长期趋势
        """
        logger.info(f"🎯 创建TRIX指标形态: {pattern_type}")
        adjusted_data = data.copy()
        n = len(adjusted_data)
        base_price = adjusted_data['close'].iloc[0]

        if pattern_type == 'MOMENTUM_SHIFT':
            # 🔧 Ultra Think深度修复：强制后期动量转换，确保100%在检测窗口内
            logger.info("⚡ 创建TRIX动量转换形态（强化版）")
            transition_point = int(n * 0.75)  # 75%位置开始强力转换
            
            for i in range(n):
                if i < transition_point:
                    # 前75%：持续温和下跌，为后期转换蓄力
                    decline_rate = 0.03 * (i / transition_point)
                    new_price = base_price * (1 - decline_rate)
                else:
                    # 🎯 后25%：爆发式反弹，强制触发动量转换
                    rebound_progress = (i - transition_point) / (n - transition_point)
                    # 使用更激进的反弹系数，确保TRIX动量转换
                    rebound_factor = 1 + (0.25 * (rebound_progress ** 0.5))  # 开方曲线，早期快速增长
                    transition_base = base_price * (1 - 0.03)  # 下跌后的基准价
                    new_price = transition_base * rebound_factor
                    
                    # 🚀 在最后10个位置加入额外的加速增长
                    if i >= n - 10:
                        extra_boost = 1 + (0.1 * (i - (n - 10)) / 10)
                        new_price *= extra_boost
                
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.998, 1.002)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.005, 1.025)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.995, 1.002)

        elif pattern_type == 'DIVERGENCE':
            # 创建TRIX背离形态
            logger.info("🔄 创建TRIX背离形态")
            peak_point = int(n * 0.7)
            
            # 前70%价格上涨但动量递减
            for i in range(peak_point):
                # 价格仍上涨但涨幅递减
                growth_rate = 0.02 * (1 - i / peak_point)
                new_price = base_price * (1 + growth_rate * i)
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.99, 1.01)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.01, 1.03)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.98, 1.00)
                base_price = new_price
            
            # 后30%价格下跌
            for i in range(peak_point, n):
                decline_rate = 0.015 * (i - peak_point) / (n - peak_point)
                new_price = base_price * (1 - decline_rate)
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.99, 1.01)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.00, 1.02)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.97, 0.99)
                base_price = new_price

        else:
            # 默认处理其他TRIX形态
            logger.info(f"🔧 TRIX默认形态处理: {pattern_type}")
            adjusted_data = self._create_cross_pattern(adjusted_data, pattern_type)

        return adjusted_data

    def _create_mfi_pattern(self, data: pd.DataFrame, pattern_type: str) -> pd.DataFrame:
        """
        创建MFI(资金流向指标)形态
        MFI特点：结合价格和成交量，识别资金流向
        """
        logger.info(f"🎯 创建MFI指标形态: {pattern_type}")
        adjusted_data = data.copy()
        n = len(adjusted_data)
        base_price = adjusted_data['close'].iloc[0]
        base_volume = adjusted_data['volume'].iloc[0]

        if pattern_type == 'MONEY_FLOW_REVERSAL':
            # 创建资金流向反转形态：先到极值，再反转
            logger.info("💰 创建MFI资金流向反转形态")
            extreme_point = int(n * 0.7)  # 前70%达到极值
            reversal_point = int(n * 0.85)  # 85%开始反转
            
            # 前70%：剧烈上涨+成交量枯竭（创造MFI极高值>80）
            for i in range(extreme_point):
                # 价格剧烈上涨（最多30%）
                growth_factor = 1 + (0.30 * i / extreme_point)
                # 成交量急剧枯竭（降至20%）
                volume_factor = 1 - (0.80 * i / extreme_point)
                
                new_price = base_price * growth_factor
                new_volume = max(base_volume * volume_factor, base_volume * 0.1)
                
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'volume'] = new_volume
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.99, 1.01)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.01, 1.04)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.98, 1.00)
            
            # 70%-85%：价格平台整理（维持MFI高位）
            extreme_price = adjusted_data['close'].iloc[extreme_point-1]
            for i in range(extreme_point, reversal_point):
                # 价格小幅波动
                wave_factor = 1 + random.uniform(-0.02, 0.02)
                # 成交量保持低位
                volume_factor = 0.2 + random.uniform(-0.1, 0.1)
                
                new_price = extreme_price * wave_factor
                new_volume = max(base_volume * volume_factor, base_volume * 0.1)
                
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'volume'] = new_volume
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.995, 1.005)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.01, 1.02)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.98, 0.99)
            
            # 85%-100%：价格急跌+成交量放量（触发MFI反转）
            for i in range(reversal_point, n):
                progress = (i - reversal_point) / (n - reversal_point)
                # 价格快速下跌（最多15%）
                decline_factor = 1 - (0.15 * progress)
                # 成交量急剧放大（增至3倍）
                volume_factor = 0.2 + (2.8 * progress)
                
                new_price = extreme_price * decline_factor
                new_volume = base_volume * volume_factor
                
                adjusted_data.loc[i, 'close'] = new_price
                adjusted_data.loc[i, 'volume'] = new_volume
                adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.99, 1.01)
                adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.00, 1.02)
                adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.96, 0.98)

        elif pattern_type in ['OVERBOUGHT', 'OVERSOLD']:
            # 创建MFI超买/超卖形态
            if pattern_type == 'OVERBOUGHT':
                logger.info("📈 创建MFI超买形态")
                # 价格和成交量同步大幅上涨
                for i in range(n):
                    growth_factor = 1 + (0.30 * i / n)
                    volume_factor = 1 + (1.0 * i / n)  # 成交量翻倍
                    
                    new_price = base_price * growth_factor
                    new_volume = base_volume * volume_factor
                    
                    adjusted_data.loc[i, 'close'] = new_price
                    adjusted_data.loc[i, 'volume'] = new_volume
                    adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.995, 1.005)
                    adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.01, 1.05)
                    adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.98, 1.00)
            else:  # OVERSOLD
                logger.info("📉 创建MFI超卖形态")
                # 价格和成交量同步大幅下跌
                for i in range(n):
                    decline_factor = 1 - (0.25 * i / n)
                    volume_factor = 1 + (0.60 * i / n)  # 成交量放大60%
                    
                    new_price = base_price * decline_factor
                    new_volume = base_volume * volume_factor
                    
                    adjusted_data.loc[i, 'close'] = new_price
                    adjusted_data.loc[i, 'volume'] = new_volume
                    adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.995, 1.005)
                    adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.00, 1.02)
                    adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.95, 0.98)

        elif pattern_type == 'DIVERGENCE':
            # 创建MFI背离形态：价格与资金流向走势相反
            logger.info("🔀 创建MFI背离形态")
            
            # 选择背离类型：牛市背离（价格下跌，MFI上涨）或熊市背离（价格上涨，MFI下跌）
            divergence_type = random.choice(['bullish', 'bearish'])
            
            if divergence_type == 'bearish':
                # 熊市背离：价格上涨，但MFI下跌（资金流出）
                logger.info("📈📉 创建熊市背离：价格上涨+MFI下跌")
                divergence_start = int(n * 0.5)  # 从50%开始产生背离
                
                # 前50%：正常上涨，价格和成交量同步增长
                for i in range(divergence_start):
                    growth_factor = 1 + (0.15 * i / divergence_start)
                    volume_factor = 1 + (0.8 * i / divergence_start)
                    
                    new_price = base_price * growth_factor
                    new_volume = base_volume * volume_factor
                    
                    adjusted_data.loc[i, 'close'] = new_price
                    adjusted_data.loc[i, 'volume'] = new_volume
                    adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.995, 1.005)
                    adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.01, 1.03)
                    adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.98, 1.00)
                
                # 后50%：背离阶段，价格继续上涨但成交量萎缩（MFI下跌）
                base_price_div = adjusted_data['close'].iloc[divergence_start-1]
                for i in range(divergence_start, n):
                    progress = (i - divergence_start) / (n - divergence_start)
                    # 价格继续缓慢上涨（总涨幅10%）
                    growth_factor = 1 + (0.10 * progress)
                    # 成交量急剧萎缩（降至30%）- 关键：造成MFI下降
                    volume_factor = 1.8 - (1.5 * progress)  # 从180%降至30%
                    
                    new_price = base_price_div * growth_factor
                    new_volume = max(base_volume * volume_factor, base_volume * 0.2)
                    
                    adjusted_data.loc[i, 'close'] = new_price
                    adjusted_data.loc[i, 'volume'] = new_volume
                    adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.995, 1.005)
                    adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.005, 1.02)
                    adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.99, 1.00)
            
            else:  # bullish divergence
                # 牛市背离：价格下跌，但MFI上涨（资金流入）
                logger.info("📉📈 创建牛市背离：价格下跌+MFI上涨")
                divergence_start = int(n * 0.5)  # 从50%开始产生背离
                
                # 前50%：正常下跌，价格和成交量萎缩
                for i in range(divergence_start):
                    decline_factor = 1 - (0.20 * i / divergence_start)
                    volume_factor = 1 - (0.4 * i / divergence_start)
                    
                    new_price = base_price * decline_factor
                    new_volume = max(base_volume * volume_factor, base_volume * 0.3)
                    
                    adjusted_data.loc[i, 'close'] = new_price
                    adjusted_data.loc[i, 'volume'] = new_volume
                    adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.995, 1.005)
                    adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.01, 1.02)
                    adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.95, 0.98)
                
                # 后50%：背离阶段，价格继续下跌但成交量放大（MFI上涨）
                base_price_div = adjusted_data['close'].iloc[divergence_start-1]
                for i in range(divergence_start, n):
                    progress = (i - divergence_start) / (n - divergence_start)
                    # 价格继续缓慢下跌（总跌幅8%）
                    decline_factor = 1 - (0.08 * progress)
                    # 成交量放大（增至200%）- 关键：造成MFI上升
                    volume_factor = 0.6 + (1.4 * progress)  # 从60%增至200%
                    
                    new_price = base_price_div * decline_factor
                    new_volume = base_volume * volume_factor
                    
                    adjusted_data.loc[i, 'close'] = new_price
                    adjusted_data.loc[i, 'volume'] = new_volume
                    adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.995, 1.005)
                    adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.00, 1.02)
                    adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.94, 0.98)

        else:
            # 默认处理其他MFI形态
            logger.info(f"🔧 MFI默认形态处理: {pattern_type}")
            adjusted_data = self._create_cross_pattern(adjusted_data, pattern_type)

        return adjusted_data

    def _create_new_indicator_pattern(self, data: pd.DataFrame, indicator_name: str, pattern_type: str) -> pd.DataFrame:
        """
        为新增指标(KC, EMV, CHAIKIN, VIX)创建通用形态
        """
        logger.info(f"🎯 创建{indicator_name}指标形态: {pattern_type}")
        adjusted_data = data.copy()
        n = len(adjusted_data)
        base_price = adjusted_data['close'].iloc[0]
        base_volume = adjusted_data['volume'].iloc[0]

        if indicator_name == 'KC':  # 肯特纳通道
            if pattern_type in ['UPPER_BREAKOUT', 'LOWER_BREAKOUT']:
                # KC通道突破形态
                breakout_point = int(n * 0.7)
                if pattern_type == 'UPPER_BREAKOUT':
                    # 上轨突破
                    for i in range(n):
                        if i < breakout_point:
                            growth_factor = 1 + (0.05 * i / breakout_point)
                        else:
                            growth_factor = 1.05 + (0.15 * (i - breakout_point) / (n - breakout_point))
                        
                        new_price = base_price * growth_factor
                        adjusted_data.loc[i, 'close'] = new_price
                        adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.99, 1.01)
                        adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.01, 1.04)
                        adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.98, 1.00)
                else:  # LOWER_BREAKOUT
                    # 下轨突破
                    for i in range(n):
                        if i < breakout_point:
                            decline_factor = 1 - (0.05 * i / breakout_point)
                        else:
                            decline_factor = 0.95 - (0.15 * (i - breakout_point) / (n - breakout_point))
                        
                        new_price = base_price * decline_factor
                        adjusted_data.loc[i, 'close'] = new_price
                        adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.99, 1.01)
                        adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.00, 1.02)
                        adjusted_data.loc[i, 'low'] = new_price * random.uniform(0.96, 0.98)
            else:
                # 其他KC形态使用通用处理
                adjusted_data = self._create_cross_pattern(adjusted_data, pattern_type)

        elif indicator_name == 'VIX':  # 恐慌指数
            if pattern_type == 'FEAR_SPIKE':
                # 恐慌情绪激增：价格大幅下跌，波动性增加
                for i in range(n):
                    decline_factor = 1 - (0.20 * i / n)
                    volatility_factor = 1 + (2.0 * i / n)  # 波动性增加200%
                    
                    new_price = base_price * decline_factor
                    daily_volatility = 0.05 * volatility_factor
                    
                    adjusted_data.loc[i, 'close'] = new_price
                    adjusted_data.loc[i, 'open'] = new_price * random.uniform(0.95, 1.05)
                    adjusted_data.loc[i, 'high'] = new_price * random.uniform(1.00, 1.00 + daily_volatility)
                    adjusted_data.loc[i, 'low'] = new_price * random.uniform(1.00 - daily_volatility, 1.00)
            else:
                # 其他VIX形态使用通用处理
                adjusted_data = self._create_cross_pattern(adjusted_data, pattern_type)

        else:
            # EMV, CHAIKIN等其他指标使用通用形态处理
            adjusted_data = self._create_cross_pattern(adjusted_data, pattern_type)

        return adjusted_data
