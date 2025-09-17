from utils.dependency_injection import get_logger
#!/usr/bin/env python3
"""
严格禁止模拟数据的30分钟吸筹信号+MACD上移选股策略

绝对禁止使用模拟数据！！
只使用真实的ClickHouse 30分钟K线数据
如果没有真实数据，则拒绝执行

根据用户要求：
1. 时间：2025年5月12日
2. 周期：30分钟K线（仅真实数据）
3. 条件1：出现吸筹信号
4. 条件2：MACD上移
5. 绝对禁止：模拟数据
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)

from utils.logger import getLogger
from utils.decorators import performance_monitor, exception_handler

logger = get_logger(__name__)

class StrictRealDataAbsorbMACDStrategy:
    """严格禁止模拟数据的30分钟吸筹信号+MACD上移选股策略"""
    
    def __init__(self):
        """初始化策略"""
        self.strategy_name = "严格禁止模拟数据的30分钟吸筹信号+MACD上移选股策略"
        self.target_date = "2025-05-12"
        self.period = "30分钟"
        self.data_access = None
        
        # 初始化数据服务
        self._initialize_data_services()
        
    def _initialize_data_services(self):
        """初始化数据服务"""
        try:
            from config.service_initializer import initialize_all_services
            container = initialize_all_services()
            
            from utils.dependency_injection import get_service
            from db.interfaces.data_access_interface import DataAccessInterface
            
            self.data_access = get_service(DataAccessInterface)
            logger.info("ClickHouse数据服务初始化成功")
        except Exception as e:
            logger.error(f"初始化数据服务失败: {e}")
            raise

    @performance_monitor(threshold=300.0)  # 5分钟超时
    def execute_strategy(self, stock_pool: Optional[List[str]] = None) -> Dict[str, Any]:
        """执行选股策略"""
        logger.info(f"开始执行{self.strategy_name}")
        logger.info(f"目标日期: {self.target_date}")
        logger.info(f"分析周期: {self.period} (绝对禁止模拟数据)")
        logger.info("🚨 严格模式：只使用真实ClickHouse 30分钟数据，绝对禁止模拟数据")
        
        # 首先验证ClickHouse连接和数据可用性
        if not self._verify_clickhouse_connection():
            error_msg = "❌ ClickHouse数据库连接失败，无法获取真实数据，策略拒绝执行"
            logger.error(error_msg)
            return {
                'strategy_name': self.strategy_name,
                'execution_date': datetime.now().isoformat(),
                'target_date': self.target_date,
                'period': self.period,
                'stock_pool_size': 0,
                'status': 'FAILED',
                'error': error_msg,
                'selected_count': 0,
                'selected_stocks': [],
                'reason': '数据库连接失败，严格禁止使用模拟数据'
            }
        
        # 验证30分钟数据是否存在
        if not self._verify_30min_data_exists():
            error_msg = "❌ ClickHouse数据库中没有30分钟K线数据，策略拒绝执行"
            logger.error(error_msg)
            return {
                'strategy_name': self.strategy_name,
                'execution_date': datetime.now().isoformat(),
                'target_date': self.target_date,
                'period': self.period,
                'stock_pool_size': 0,
                'status': 'FAILED',
                'error': error_msg,
                'selected_count': 0,
                'selected_stocks': [],
                'reason': '没有真实30分钟数据，严格禁止使用模拟数据'
            }
        
        # 使用股票池
        if stock_pool is None:
            stock_pool = self._get_stock_pool()
        
        logger.info(f"股票池大小: {len(stock_pool)}")
        
        selected_stocks = []
        analysis_details = {}
        data_source_stats = {"real_30min": 0, "no_real_data": 0}
        
        for i, stock_code in enumerate(stock_pool, 1):
            logger.info(f"分析股票 {i}/{len(stock_pool)}: {stock_code}")
            
            try:
                # 获取30分钟数据（严格只使用真实数据）
                stock_data = self._get_real_30min_data_only(stock_code, self.target_date)
                
                if stock_data is None or stock_data.empty:
                    data_source_stats["no_real_data"] += 1
                    analysis_details[stock_code] = {
                        'status': 'no_real_data',
                        'message': '没有真实30分钟数据，严格禁止模拟数据',
                        'data_source': 'none'
                    }
                    logger.warning(f"❌ {stock_code} 没有真实30分钟数据，跳过分析")
                    continue
                
                data_source_stats["real_30min"] += 1
                logger.info(f"✅ {stock_code} 使用真实30分钟数据: {len(stock_data)} 条记录")
                
                # 分析吸筹信号
                absorb_result = self._analyze_absorb_signal(stock_data, stock_code)
                
                # 分析MACD上移信号
                macd_result = self._analyze_macd_upward(stock_data, stock_code)
                
                # 严格判断：同时满足两个条件
                meets_criteria = absorb_result['has_absorb_signal'] and macd_result['macd_upward']
                
                if meets_criteria:
                    selected_stocks.append({
                        'stock_code': stock_code,
                        'stock_name': f'股票{stock_code}',
                        'absorb_signal_strength': absorb_result['signal_strength'],
                        'macd_strength': macd_result['macd_strength'],
                        'comprehensive_score': self._calculate_score(absorb_result, macd_result),
                        'signal_time': absorb_result['signal_time'],
                        'selection_reason': self._get_selection_reason(absorb_result, macd_result),
                        'analysis_time': datetime.now().strftime('%H:%M:%S'),
                        'data_source': 'real_30min'
                    })
                    
                    logger.info(f"✅ {stock_code} 符合条件：吸筹信号强度{absorb_result['signal_strength']:.2f}, MACD强度{macd_result['macd_strength']:.2f} (真实数据)")
                
                # 保存详细分析
                analysis_details[stock_code] = {
                    'status': 'analyzed',
                    'absorb_result': absorb_result,
                    'macd_result': macd_result,
                    'meets_criteria': meets_criteria,
                    'data_source': 'real_30min'
                }
                
            except Exception as e:
                logger.error(f"分析股票 {stock_code} 时出错: {e}")
                analysis_details[stock_code] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        # 按综合评分排序
        selected_stocks.sort(key=lambda x: x['comprehensive_score'], reverse=True)
        
        result = {
            'strategy_name': self.strategy_name,
            'execution_date': datetime.now().isoformat(),
            'target_date': self.target_date,
            'period': self.period,
            'stock_pool_size': len(stock_pool),
            'selected_count': len(selected_stocks),
            'selected_stocks': selected_stocks,
            'analysis_details': analysis_details,
            'data_source_statistics': data_source_stats,
            'strategy_criteria': {
                'absorb_signal': '30分钟周期内出现吸筹信号',
                'macd_upward': 'MACD指标上移',
                'target_date': self.target_date,
                'selection_rule': '同时满足吸筹信号AND MACD上移',
                'data_policy': '严格禁止模拟数据，只使用真实ClickHouse 30分钟数据'
            },
            'status': 'SUCCESS'
        }
        
        logger.info(f"策略执行完成，选出 {len(selected_stocks)} 只股票")
        logger.info(f"数据源统计: 真实30分钟数据{data_source_stats['real_30min']}只, 无真实数据{data_source_stats['no_real_data']}只")
        return result

    def _verify_clickhouse_connection(self) -> bool:
        """验证ClickHouse连接"""
        try:
            # 简单连接测试
            test_query = "SELECT 1 as test"
            result = self.data_access.execute_query(test_query)
            
            if result is None or result.empty:
                logger.error("ClickHouse连接测试失败：查询返回空结果")
                return False
            
            logger.info("✅ ClickHouse连接验证成功")
            return True
        except Exception as e:
            logger.error(f"ClickHouse连接验证失败: {e}")
            return False

    def _verify_30min_data_exists(self) -> bool:
        """验证30分钟数据是否存在"""
        try:
            # 检查数据库中是否存在30分钟数据
            query = """
            SELECT COUNT(*) as count 
            FROM stock_info 
            WHERE level = '30分钟' OR level = '30min' OR level = 'MIN_30'
            LIMIT 1
            """
            result = self.data_access.execute_query(query)
            
            if result is None or result.empty:
                logger.error("无法查询30分钟数据存在性")
                return False
            
            count = result.iloc[0]['count'] if 'count' in result.columns else 0
            
            if count > 0:
                logger.info(f"✅ 发现 {count} 条30分钟数据记录")
                return True
            else:
                logger.error("❌ 数据库中没有30分钟K线数据")
                return False
                
        except Exception as e:
            logger.error(f"验证30分钟数据存在性失败: {e}")
            return False

    def _get_stock_pool(self) -> List[str]:
        """获取股票池"""
        # 重点活跃股票池
        stock_pool = [
            # 主要指数成分股
            '000001', '000002', '000858', '000063', '000166', '000725', '000776',
            '600000', '600036', '600519', '600887', '600028', '600030', '600031',
            '600048', '600104', '600309', '600340', '600585', '600690', '600703',
            # 更多活跃股票
            '002415', '002594', '002304', '002230', '300750', '300014', '300015'
        ]
        
        return stock_pool

    def _get_real_30min_data_only(self, stock_code: str, target_date: str) -> Optional[pd.DataFrame]:
        """仅获取真实30分钟K线数据，绝对禁止模拟数据"""
        try:
            # 计算数据范围
            target_date_obj = datetime.strptime(target_date, "%Y-%m-%d")
            start_date = (target_date_obj - timedelta(days=30)).strftime("%Y%m%d")
            end_date = (target_date_obj + timedelta(days=1)).strftime("%Y%m%d")
            
            # 尝试多种30分钟数据的level标识
            level_variants = ['30分钟', '30min', 'MIN_30', '30MIN']
            
            for level in level_variants:
                try:
                    logger.info(f"尝试获取{stock_code}的{level}数据...")
                    
                    # 构建查询
                    query = """
                    SELECT code, name, date, level, open, close, high, low, volume,
                           turnover, price_change, price_range, industry
                    FROM stock_info 
                    WHERE code = %(code)s 
                      AND level = %(level)s 
                      AND date >= %(start_date)s 
                      AND date <= %(end_date)s
                    ORDER BY date ASC
                    """
                    
                    params = {
                        'code': stock_code,
                        'level': level,
                        'start_date': start_date,
                        'end_date': end_date
                    }
                    
                    result_df = self.data_access.execute_query(query, params)
                    
                    if result_df is not None and not result_df.empty:
                        # 验证这是真实的30分钟数据
                        result_df['date'] = pd.to_datetime(result_df['date'])
                        result_df = result_df.sort_values('date')
                        
                        # 检查是否有目标日期的数据
                        target_day_data = result_df[result_df['date'].dt.date == target_date_obj.date()]
                        if len(target_day_data) > 1:  # 真实30分钟数据一天应该有多条记录
                            # 为30分钟数据添加datetime列
                            result_df = self._add_datetime_to_30min_data(result_df, target_date_obj)
                            logger.info(f"✅ {stock_code} 使用真实{level}数据: {len(result_df)} 条记录")
                            return result_df
                        
                except Exception as e:
                    logger.debug(f"获取{stock_code}的{level}数据失败: {e}")
                    continue
            
            # 所有level都尝试失败
            logger.warning(f"❌ {stock_code} 没有任何真实30分钟数据")
            return None
            
        except Exception as e:
            logger.error(f"获取股票 {stock_code} 真实30分钟数据失败: {e}")
            return None

    def _add_datetime_to_30min_data(self, df: pd.DataFrame, target_date_obj: datetime) -> pd.DataFrame:
        """为30分钟数据添加具体的datetime信息"""
        if 'datetime' not in df.columns:
            df = df.copy()
            # 为每个交易日的30分钟数据分配具体时间
            df['datetime'] = df['date']
            
            # 获取目标日期的数据
            target_date_data = df[df['date'].dt.date == target_date_obj.date()].copy()
            if not target_date_data.empty:
                # 为目标日期的数据生成30分钟间隔的时间
                trading_periods = []
                start_time = target_date_obj.replace(hour=9, minute=30)
                
                for i in range(len(target_date_data)):
                    if i < 13:  # 正常交易时段：9:30-15:00 (13个30分钟周期)
                        time_offset = pd.Timedelta(minutes=30*i)
                        current_time = start_time + time_offset
                        trading_periods.append(current_time)
                    else:
                        # 如果有更多数据，继续延续
                        time_offset = pd.Timedelta(minutes=30*i)
                        current_time = start_time + time_offset
                        trading_periods.append(current_time)
                
                # 更新目标日期数据的datetime
                for i, (idx, row) in enumerate(target_date_data.iterrows()):
                    if i < len(trading_periods):
                        df.loc[idx, 'datetime'] = trading_periods[i]
        
        return df

    def _analyze_absorb_signal(self, data: pd.DataFrame, stock_code: str) -> Dict[str, Any]:
        """分析吸筹信号"""
        logger.info(f"分析 {stock_code} 的吸筹信号...")
        
        try:
            # 计算吸筹相关指标
            close = data['close'].values
            volume = data['volume'].values
            high = data['high'].values
            low = data['low'].values
            
            # 1. 成交量分析
            volume_ma5 = pd.Series(volume).rolling(5).mean().values
            volume_ratio = volume / volume_ma5  # 成交量比率
            
            # 2. 价量关系分析
            price_change = np.diff(close, prepend=close[0])
            volume_price_corr = np.corrcoef(price_change[1:], volume[1:])[0, 1] if len(price_change) > 1 else 0
            
            # 3. 计算WVAD指标（量价趋势指标）
            wvad = self._calculate_wvad(close, high, low, volume)
            
            # 4. 检测吸筹信号
            absorb_signals = []
            signal_strength = 0
            
            for i in range(5, len(data)):  # 从第5个数据点开始检测
                current_time = data.iloc[i]['datetime'] if 'datetime' in data.columns else data.iloc[i]['date']
                
                # 吸筹信号条件：
                # 1. 成交量放大
                volume_enlarged = volume_ratio[i] > 1.2  # 成交量比5日均量大20%
                
                # 2. WVAD指标显示资金流入
                wvad_positive = wvad[i] > 0
                
                # 3. 收盘价位置相对较高
                close_high_ratio = (close[i] - low[i]) / (high[i] - low[i]) if high[i] != low[i] else 1
                high_close = close_high_ratio > 0.6
                
                if volume_enlarged and wvad_positive and high_close:
                    signal_strength_current = volume_ratio[i] * 0.4 + wvad[i] * 0.3 + close_high_ratio * 0.3
                    
                    absorb_signals.append({
                        'time': current_time.strftime('%H:%M') if hasattr(current_time, 'strftime') else str(current_time),
                        'signal_strength': signal_strength_current,
                        'volume_ratio': volume_ratio[i],
                        'wvad': wvad[i],
                        'close_position': close_high_ratio
                    })
                    
                    signal_strength = max(signal_strength, signal_strength_current)
            
            has_absorb_signal = len(absorb_signals) > 0
            signal_time = absorb_signals[0]['time'] if absorb_signals else None
            
            logger.info(f"{stock_code} 吸筹信号分析完成：{'有' if has_absorb_signal else '无'}信号，强度: {signal_strength:.2f}")
            
            return {
                'has_absorb_signal': has_absorb_signal,
                'signal_strength': signal_strength,
                'signal_time': signal_time,
                'signal_count': len(absorb_signals),
                'signal_details': absorb_signals,
                'volume_price_correlation': volume_price_corr
            }
            
        except Exception as e:
            logger.error(f"分析 {stock_code} 吸筹信号时出错: {e}")
            return {
                'has_absorb_signal': False,
                'signal_strength': 0,
                'signal_time': None,
                'error': str(e)
            }

    def _analyze_macd_upward(self, data: pd.DataFrame, stock_code: str) -> Dict[str, Any]:
        """分析MACD上移信号"""
        logger.info(f"分析 {stock_code} 的MACD上移信号...")
        
        try:
            close = data['close'].values
            
            # 计算MACD指标
            dif, dea, macd_hist = self._calculate_macd(close)
            
            # 分析MACD上移信号
            macd_upward = False
            macd_strength = 0
            golden_cross = False
            
            if len(dif) >= 2:
                # 检查MACD上移趋势
                dif_rising = dif[-1] > dif[-2]  # DIF线上升
                dea_rising = dea[-1] > dea[-2]  # DEA线上升
                macd_rising = macd_hist[-1] > macd_hist[-2]  # MACD柱状图上升
                
                # 金叉判断
                golden_cross = dif[-1] > dea[-1] and dif[-2] <= dea[-2]
                
                # MACD上移判断
                macd_upward = (dif_rising and dea_rising) or golden_cross or macd_rising
                
                # 计算MACD强度
                if macd_upward:
                    trend_strength = 0
                    if dif_rising: trend_strength += 0.3
                    if dea_rising: trend_strength += 0.3
                    if macd_rising: trend_strength += 0.2
                    if golden_cross: trend_strength += 0.3
                    
                    macd_strength = trend_strength + abs(dif[-1]) * 0.1
            
            logger.info(f"{stock_code} MACD分析完成：{'上移' if macd_upward else '非上移'}，强度: {macd_strength:.2f}")
            
            return {
                'macd_upward': macd_upward,
                'macd_strength': macd_strength,
                'dif_current': float(dif[-1]) if len(dif) > 0 else 0,
                'dea_current': float(dea[-1]) if len(dea) > 0 else 0,
                'macd_current': float(macd_hist[-1]) if len(macd_hist) > 0 else 0,
                'golden_cross': golden_cross
            }
            
        except Exception as e:
            logger.error(f"分析 {stock_code} MACD时出错: {e}")
            return {
                'macd_upward': False,
                'macd_strength': 0,
                'error': str(e)
            }

    def _calculate_wvad(self, close: np.ndarray, high: np.ndarray, low: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """计算WVAD指标（威廉变异离散量）"""
        wvad = np.zeros_like(close)
        
        for i in range(1, len(close)):
            if high[i] != low[i]:
                # WVAD = (收盘价-开盘价)/(最高价-最低价) * 成交量
                # 这里简化使用前一日收盘价作为开盘价
                price_position = (close[i] - close[i-1]) / (high[i] - low[i])
                wvad[i] = price_position * volume[i] / 1000000  # 标准化处理
            else:
                wvad[i] = 0
        
        return wvad

    def _calculate_macd(self, close: np.ndarray, fast=12, slow=26, signal=9) -> tuple:
        """计算MACD指标"""
        try:
            from indicators.common import macd
            return macd(close, fast, slow, signal)
        except:
            # 简化MACD计算
            close_series = pd.Series(close)
            ema12 = close_series.ewm(span=fast).mean()
            ema26 = close_series.ewm(span=slow).mean()
            dif = ema12 - ema26
            dea = dif.ewm(span=signal).mean()
            macd_hist = 2 * (dif - dea)
            
            return dif.values, dea.values, macd_hist.values

    def _calculate_score(self, absorb_result: Dict[str, Any], macd_result: Dict[str, Any]) -> float:
        """计算综合评分"""
        absorb_score = absorb_result.get('signal_strength', 0) * 0.6
        macd_score = macd_result.get('macd_strength', 0) * 0.4
        
        # 额外加分项
        bonus = 0
        if macd_result.get('golden_cross', False):
            bonus += 0.1  # MACD金叉额外加分
        
        if absorb_result.get('signal_count', 0) > 1:
            bonus += 0.05  # 多次吸筹信号加分
        
        return absorb_score + macd_score + bonus

    def _get_selection_reason(self, absorb_result: Dict[str, Any], macd_result: Dict[str, Any]) -> str:
        """获取选股原因"""
        reasons = []
        
        if absorb_result['has_absorb_signal']:
            reasons.append(f"吸筹信号(强度{absorb_result['signal_strength']:.2f})")
        
        if macd_result['macd_upward']:
            reasons.append(f"MACD上移(强度{macd_result['macd_strength']:.2f})")
        
        return " + ".join(reasons)

    def save_results(self, results: Dict[str, Any]) -> str:
        """保存策略结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"strict_real_data_absorb_macd_strategy_results_{timestamp}.json"
        
        try:
            import json
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"策略结果已保存到: {filename}")
            return filename
        except Exception as e:
            logger.error(f"保存结果失败: {e}")
            return ""

    def print_results(self, results: Dict[str, Any]):
        """打印策略结果"""
        print("=" * 80)
        print(f"📊 {results['strategy_name']}")
        print("=" * 80)
        print(f"🎯 目标日期: {results['target_date']}")
        print(f"⏰ 分析周期: {results['period']}")
        print(f"📈 股票池大小: {results['stock_pool_size']}")
        print(f"✅ 选中股票数: {results['selected_count']}")
        print(f"🔧 执行状态: {results.get('status', 'UNKNOWN')}")
        
        # 数据源统计
        stats = results.get('data_source_statistics', {})
        print(f"📊 数据源统计:")
        print(f"   • 真实30分钟数据: {stats.get('real_30min', 0)} 只")
        print(f"   • 无真实数据: {stats.get('no_real_data', 0)} 只")
        print(f"   • 模拟数据: 0 只 (严格禁止)")
        print()
        
        if results.get('status') == 'FAILED':
            print(f"❌ 策略执行失败: {results.get('error', '未知错误')}")
            print(f"💡 失败原因: {results.get('reason', '未知')}")
        elif results['selected_stocks']:
            print("🏆 符合条件的股票:")
            print("-" * 80)
            print(f"{'股票代码':<10} {'股票名称':<15} {'综合评分':<10} {'数据源':<12} {'选股原因':<30} {'信号时间':<10}")
            print("-" * 80)
            
            for stock in results['selected_stocks'][:10]:  # 显示前10只
                print(f"{stock['stock_code']:<10} {stock['stock_name']:<15} "
                      f"{stock['comprehensive_score']:<10.2f} {stock.get('data_source', 'unknown'):<12} "
                      f"{stock['selection_reason']:<30} {stock.get('signal_time', 'N/A'):<10}")
                      
            if len(results['selected_stocks']) > 10:
                print(f"... 还有 {len(results['selected_stocks']) - 10} 只股票")
        else:
            print("😔 未找到符合条件的股票")
            print("💡 可能原因：")
            print("   1. ClickHouse数据库中没有真实30分钟数据")
            print("   2. 目标日期的数据不存在")
            print("   3. 选股条件过于严格")
            print("   4. 严格禁止模拟数据政策限制")
        
        print("\n" + "=" * 80)


def main_strict_real_data_absorb_macd_strategy():
    """主函数"""
    print("🚀 启动严格禁止模拟数据的30分钟吸筹信号+MACD上移选股策略")
    print("📅 目标日期: 2025年5月12日")
    print("⏰ 分析周期: 30分钟K线")
    print("🎯 选股条件: 出现吸筹信号 AND MACD上移")
    print("🚨 严格模式: 绝对禁止模拟数据，只使用真实ClickHouse数据")
    
    try:
        # 创建策略实例
        strategy = StrictRealDataAbsorbMACDStrategy()
        
        # 执行策略
        results = strategy.execute_strategy()
        
        # 打印结果
        strategy.print_results(results)
        
        # 保存结果
        saved_file = strategy.save_results(results)
        if saved_file:
            print(f"\n📄 详细结果已保存到: {saved_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"策略执行失败: {e}")
        print(f"❌ 策略执行失败: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)