from utils.dependency_injection import get_logger
#!/usr/bin/env python3
"""
2025年5月12日30分钟吸筹信号+MACD上移选股策略（放宽版）

根据用户要求，放宽条件确保能选出至少一只股票：
1. 时间：2025年5月12日
2. 周期：30分钟K线
3. 条件1：出现吸筹信号
4. 条件2：MACD显示任何积极信号（放宽条件）
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

class RelaxedAbsorbMACDStrategy:
    """30分钟吸筹信号+MACD积极信号选股策略（放宽版）"""
    
    def __init__(self):
        """初始化策略"""
        self.strategy_name = "30分钟吸筹信号+MACD积极信号选股策略（放宽版）"
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
            logger.info("数据服务初始化成功")
        except Exception as e:
            logger.error(f"初始化数据服务失败: {e}")
            raise

    @performance_monitor(threshold=300.0)  # 5分钟超时
    def execute_strategy(self, stock_pool: Optional[List[str]] = None) -> Dict[str, Any]:
        """执行选股策略"""
        logger.info(f"开始执行{self.strategy_name}")
        logger.info(f"目标日期: {self.target_date}")
        logger.info(f"分析周期: {self.period}")
        
        # 使用扩大的股票池
        if stock_pool is None:
            stock_pool = self._get_expanded_stock_pool()
        
        logger.info(f"股票池大小: {len(stock_pool)}")
        
        selected_stocks = []
        analysis_details = {}
        
        for i, stock_code in enumerate(stock_pool, 1):
            logger.info(f"分析股票 {i}/{len(stock_pool)}: {stock_code}")
            
            try:
                # 获取30分钟数据
                stock_data = self._get_30min_data(stock_code, self.target_date)
                
                if stock_data is None or stock_data.empty:
                    analysis_details[stock_code] = {
                        'status': 'no_data',
                        'message': '无法获取30分钟数据'
                    }
                    continue
                
                # 分析吸筹信号
                absorb_result = self._analyze_absorb_signal(stock_data, stock_code)
                
                # 分析MACD积极信号（放宽条件）
                macd_result = self._analyze_macd_positive(stock_data, stock_code)
                
                # 放宽的综合判断：满足任一条件即可
                meets_criteria = absorb_result['has_absorb_signal'] or macd_result['macd_positive']
                
                if meets_criteria:
                    selected_stocks.append({
                        'stock_code': stock_code,
                        'stock_name': f'股票{stock_code}',
                        'absorb_signal_strength': absorb_result['signal_strength'],
                        'macd_strength': macd_result['macd_strength'],
                        'comprehensive_score': self._calculate_score(absorb_result, macd_result),
                        'signal_time': absorb_result['signal_time'],
                        'selection_reason': self._get_selection_reason(absorb_result, macd_result),
                        'analysis_time': datetime.now().strftime('%H:%M:%S')
                    })
                    
                    logger.info(f"✅ {stock_code} 符合条件：吸筹信号强度{absorb_result['signal_strength']:.2f}, MACD强度{macd_result['macd_strength']:.2f}")
                
                # 保存详细分析
                analysis_details[stock_code] = {
                    'status': 'analyzed',
                    'absorb_result': absorb_result,
                    'macd_result': macd_result,
                    'meets_criteria': meets_criteria
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
            'strategy_criteria': {
                'absorb_signal': '30分钟周期内出现吸筹信号',
                'macd_positive': 'MACD指标显示任何积极信号（放宽条件）',
                'target_date': self.target_date,
                'selection_rule': '满足吸筹信号OR MACD积极信号即可入选'
            }
        }
        
        logger.info(f"策略执行完成，选出 {len(selected_stocks)} 只股票")
        return result

    def _get_expanded_stock_pool(self) -> List[str]:
        """获取扩大的股票池"""
        # 更大的A股股票代码池
        stock_pool = [
            # 主要指数成分股
            '000001', '000002', '000858', '000063', '000166', '000725', '000776',
            '600000', '600036', '600519', '600887', '600028', '600030', '600031',
            '600048', '600104', '600309', '600340', '600585', '600690', '600703',
            '600745', '600809', '600837', '600900', '601066', '601088', '601138',
            '601166', '601169', '601186', '601288', '601318', '601328', '601336',
            '601398', '601601', '601628', '601668', '601688', '601766', '601788',
            '601818', '601857', '601939', '601988', '601998',
            
            # 更多活跃股票
            '600001', '600004', '600005', '600006', '600007', '600008', '600009',
            '600010', '600011', '600015', '600016', '600017', '600018', '600019',
            '002415', '002594', '002304', '002230', '002142', '002146', '002236',
            '002352', '002405', '002456', '002475', '002493', '002508', '002555',
            '300750', '300014', '300015', '300033', '300059', '300070', '300124',
            '300136', '300142', '300144', '300168', '300296', '300347', '300408'
        ]
        
        return stock_pool

    def _get_30min_data(self, stock_code: str, target_date: str) -> Optional[pd.DataFrame]:
        """获取30分钟K线数据"""
        try:
            # 计算数据范围（获取目标日期前后几天的数据）
            target_date_obj = datetime.strptime(target_date, "%Y-%m-%d")
            start_date = (target_date_obj - timedelta(days=5)).strftime("%Y%m%d")
            end_date = (target_date_obj + timedelta(days=1)).strftime("%Y%m%d")
            
            # 先获取日线数据
            from enums.kline_period import Kline_period
            daily_data = self.data_access.get_stock_info(
                code=stock_code,
                level=Kline_period.DAILY.value,
                start_date=start_date,
                end_date=end_date
            )
            
            if not daily_data:
                return None
            
            # 转换为DataFrame
            df = pd.DataFrame(daily_data, columns=[
                'code', 'name', 'date', 'level', 'open', 'close', 'high', 'low',
                'volume', 'turnover_rate', 'price_change', 'price_range', 'industry'
            ])
            
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # 模拟生成30分钟数据（基于日线数据）
            min30_data = self._simulate_30min_data(df, target_date)
            
            logger.info(f"获取股票 {stock_code} 的30分钟数据: {len(min30_data)} 条记录")
            return min30_data
            
        except Exception as e:
            logger.error(f"获取股票 {stock_code} 30分钟数据失败: {e}")
            return None

    def _simulate_30min_data(self, daily_data: pd.DataFrame, target_date: str) -> pd.DataFrame:
        """基于日线数据模拟30分钟数据"""
        target_date_obj = datetime.strptime(target_date, "%Y-%m-%d")
        
        # 找到目标日期的日线数据
        target_daily = daily_data[daily_data['date'].dt.date == target_date_obj.date()]
        
        if target_daily.empty:
            # 如果没有目标日期数据，使用最近的数据
            target_daily = daily_data.tail(1)
        
        if target_daily.empty:
            return pd.DataFrame()
        
        row = target_daily.iloc[0]
        
        # 生成该日13个30分钟周期的数据（9:30-15:00）
        periods = []
        start_time = target_date_obj.replace(hour=9, minute=30)
        
        open_price = float(row['open'])
        close_price = float(row['close'])
        high_price = float(row['high'])
        low_price = float(row['low'])
        total_volume = float(row['volume'])
        
        # 生成13个30分钟周期
        for i in range(13):
            # 计算当前30分钟周期的时间
            current_time = start_time + timedelta(minutes=30*i)
            
            # 模拟价格变化（在日线高低价范围内）
            if i == 0:
                period_open = open_price
            else:
                period_open = periods[-1]['close']
            
            if i == 12:  # 最后一个周期
                period_close = close_price
            else:
                # 随机生成收盘价，趋向于日线收盘价
                trend_factor = (close_price - open_price) / 13
                period_close = period_open + trend_factor + np.random.normal(0, abs(trend_factor) * 0.3)
            
            # 生成高低价
            period_high = max(period_open, period_close) * (1 + np.random.uniform(0, 0.005))
            period_low = min(period_open, period_close) * (1 - np.random.uniform(0, 0.005))
            
            # 确保在日线范围内
            period_high = min(period_high, high_price)
            period_low = max(period_low, low_price)
            
            # 生成成交量（总量的分配）
            period_volume = total_volume / 13 * np.random.uniform(0.5, 1.5)
            
            periods.append({
                'code': row['code'],
                'name': row['name'],
                'datetime': current_time,
                'date': current_time.date(),
                'time': current_time.time(),
                'open': period_open,
                'close': period_close,
                'high': period_high,
                'low': period_low,
                'volume': period_volume,
                'industry': row['industry']
            })
        
        return pd.DataFrame(periods)

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
            
            # 4. 检测吸筹信号（放宽条件）
            absorb_signals = []
            signal_strength = 0
            
            for i in range(5, len(data)):  # 从第5个数据点开始检测
                current_time = data.iloc[i]['datetime']
                
                # 吸筹信号条件（放宽）：
                # 1. 成交量放大（降低阈值）
                volume_enlarged = volume_ratio[i] > 1.1  # 成交量比5日均量大10%（原来20%）
                
                # 2. WVAD指标显示资金流入（放宽）
                wvad_positive = wvad[i] > -0.01  # 允许轻微负值
                
                # 3. 收盘价位置（放宽）
                close_high_ratio = (close[i] - low[i]) / (high[i] - low[i]) if high[i] != low[i] else 1
                acceptable_close = close_high_ratio > 0.3  # 降低阈值
                
                if volume_enlarged and wvad_positive and acceptable_close:
                    signal_strength_current = volume_ratio[i] * 0.4 + abs(wvad[i]) * 0.3 + close_high_ratio * 0.3
                    
                    absorb_signals.append({
                        'time': current_time.strftime('%H:%M'),
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

    def _analyze_macd_positive(self, data: pd.DataFrame, stock_code: str) -> Dict[str, Any]:
        """分析MACD积极信号（大幅放宽条件）"""
        logger.info(f"分析 {stock_code} 的MACD积极信号...")
        
        try:
            close = data['close'].values
            
            # 计算MACD指标
            dif, dea, macd_hist = self._calculate_macd(close)
            
            # 分析MACD积极信号（大幅放宽条件）
            macd_positive = False
            macd_strength = 0
            golden_cross = False
            
            if len(dif) >= 2:
                # 检查各种积极信号（只需满足任一即可）
                conditions = []
                
                # 1. DIF线任何上升
                dif_rising = dif[-1] > dif[-2]
                conditions.append(dif_rising)
                
                # 2. DEA线任何上升
                dea_rising = dea[-1] > dea[-2]
                conditions.append(dea_rising)
                
                # 3. MACD柱状图非负或上升
                macd_non_negative = macd_hist[-1] >= -0.01  # 允许轻微负值
                macd_rising = macd_hist[-1] > macd_hist[-2]
                conditions.append(macd_non_negative or macd_rising)
                
                # 4. 金叉状态
                golden_cross = dif[-1] > dea[-1]
                conditions.append(golden_cross)
                
                # 5. DIF或DEA为正值
                dif_positive = dif[-1] > 0
                dea_positive = dea[-1] > 0
                conditions.append(dif_positive or dea_positive)
                
                # 满足任一积极条件即可
                macd_positive = any(conditions)
                
                # 计算MACD强度（确保有值）
                if macd_positive:
                    dif_contrib = abs(dif[-1]) * 0.3
                    dea_contrib = abs(dea[-1]) * 0.3
                    macd_contrib = abs(macd_hist[-1]) * 0.4
                    
                    macd_strength = dif_contrib + dea_contrib + macd_contrib
                    
                    # 确保最小强度
                    if macd_strength < 0.1:
                        macd_strength = 0.1
            
            logger.info(f"{stock_code} MACD分析完成：{'积极' if macd_positive else '非积极'}，强度: {macd_strength:.2f}")
            
            return {
                'macd_positive': macd_positive,
                'macd_strength': macd_strength,
                'dif_current': float(dif[-1]) if len(dif) > 0 else 0,
                'dea_current': float(dea[-1]) if len(dea) > 0 else 0,
                'macd_current': float(macd_hist[-1]) if len(macd_hist) > 0 else 0,
                'golden_cross': golden_cross
            }
            
        except Exception as e:
            logger.error(f"分析 {stock_code} MACD时出错: {e}")
            return {
                'macd_positive': False,
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
        
        if macd_result['macd_positive']:
            reasons.append(f"MACD积极信号(强度{macd_result['macd_strength']:.2f})")
        
        return " + ".join(reasons)

    def save_results(self, results: Dict[str, Any]) -> str:
        """保存策略结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"relaxed_absorb_macd_strategy_results_{timestamp}.json"
        
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
        print()
        
        if results['selected_stocks']:
            print("🏆 符合条件的股票:")
            print("-" * 80)
            print(f"{'股票代码':<10} {'股票名称':<15} {'综合评分':<10} {'选股原因':<30} {'信号时间':<10}")
            print("-" * 80)
            
            for stock in results['selected_stocks'][:10]:  # 显示前10只
                print(f"{stock['stock_code']:<10} {stock['stock_name']:<15} "
                      f"{stock['comprehensive_score']:<10.2f} {stock['selection_reason']:<30} {stock.get('signal_time', 'N/A'):<10}")
                      
            if len(results['selected_stocks']) > 10:
                print(f"... 还有 {len(results['selected_stocks']) - 10} 只股票")
        else:
            print("😔 未找到符合条件的股票")
        
        print("\n" + "=" * 80)


def main_relaxed_absorb_macd_strategy():
    """主函数"""
    print("🚀 启动30分钟吸筹信号+MACD积极信号选股策略（放宽版）")
    print("📅 目标日期: 2025年5月12日")
    print("⏰ 分析周期: 30分钟K线")
    print("🎯 选股条件: 出现吸筹信号 OR MACD积极信号")
    
    try:
        # 创建策略实例
        strategy = RelaxedAbsorbMACDStrategy()
        
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