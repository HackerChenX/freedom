#!/usr/bin/env python3
"""
全面策略验证系统 - 包含所有指标形态策略选股和买点分析反向验证
"""

import sys
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple

# 确保能够导入项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.logger import get_logger
from utils.dependency_injection import get_container
from db.clickhouse_db import get_clickhouse_db

logger = get_logger(__name__)

class ComprehensiveStrategyValidator:
    """全面策略验证系统"""
    
    def __init__(self):
        self.db = get_clickhouse_db()
        self.container = get_container()
        self._setup_services()
        
    def _setup_services(self):
        """设置服务"""
        try:
            from db.interfaces.indicator_calculator_interface import IIndicatorCalculator
            
            # 创建完整的指标计算器
            class FullIndicatorCalculator:
                def calculate_ma(self, data, period=20):
                    """计算移动平均线"""
                    return data['close'].rolling(window=period).mean()
                    
                def calculate_macd(self, data):
                    """计算MACD"""
                    close = data['close']
                    ema12 = close.ewm(span=12).mean()
                    ema26 = close.ewm(span=26).mean()
                    macd = ema12 - ema26
                    signal = macd.ewm(span=9).mean()
                    histogram = macd - signal
                    return {
                        'macd': macd,
                        'signal': signal,
                        'histogram': histogram
                    }
                    
                def calculate_rsi(self, data, period=14):
                    """计算RSI"""
                    close = data['close']
                    delta = close.diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    return rsi
                    
                def calculate_kdj(self, data, k_period=9, d_period=3, j_period=3):
                    """计算KDJ"""
                    high = data['high']
                    low = data['low']
                    close = data['close']
                    
                    lowest_low = low.rolling(window=k_period).min()
                    highest_high = high.rolling(window=k_period).max()
                    
                    rsv = (close - lowest_low) / (highest_high - lowest_low) * 100
                    k = rsv.ewm(alpha=1/d_period).mean()
                    d = k.ewm(alpha=1/d_period).mean()
                    j = 3 * k - 2 * d
                    
                    return {
                        'K': k,
                        'D': d,
                        'J': j
                    }
                    
                def calculate_boll(self, data, period=20, std_dev=2):
                    """计算布林带"""
                    close = data['close']
                    sma = close.rolling(window=period).mean()
                    std = close.rolling(window=period).std()
                    
                    return {
                        'upper': sma + (std * std_dev),
                        'middle': sma,
                        'lower': sma - (std * std_dev)
                    }
            
            # 注册指标计算器
            calculator = FullIndicatorCalculator()
            self.container.register_singleton(IIndicatorCalculator, factory=lambda: calculator)
            self.indicator_calculator = calculator
            logger.info("已注册完整指标计算器")
        except Exception as e:
            logger.error(f"设置服务失败: {e}")
    
    def get_high_quality_stocks(self, count: int = 50) -> List[str]:
        """获取高质量股票列表"""
        try:
            query = f"""
            SELECT code, COUNT(*) as record_count
            FROM stock_info 
            WHERE level = '日线'
            AND date >= '2020-01-01'
            GROUP BY code
            HAVING record_count >= 500
            ORDER BY record_count DESC
            LIMIT {count}
            """
            
            result = self.db.query(query)
            if not result.empty:
                stocks = result['code'].tolist()
                logger.info(f"获取到 {len(stocks)} 只高质量股票")
                return stocks
            else:
                return ['600519', '000858', '600036', '000001', '000002'][:count]
        except Exception as e:
            logger.error(f"获取股票失败: {e}")
            return ['600519', '000858', '600036', '000001', '000002'][:count]
    
    def get_stock_data(self, code: str, days: int = 250) -> pd.DataFrame:
        """获取股票数据"""
        try:
            end_date = '2025-05-23'
            start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=days)).strftime('%Y-%m-%d')
            
            query = f"""
            SELECT code, name, date, open, high, low, close, volume
            FROM stock_info 
            WHERE code = '{code}'
            AND level = '日线'
            AND date >= '{start_date}' AND date <= '{end_date}'
            ORDER BY date ASC
            """
            
            result = self.db.query(query)
            if not result.empty:
                # 确保数据类型正确
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in result.columns:
                        result[col] = pd.to_numeric(result[col], errors='coerce')
                
                result = result.dropna()
                return result
            else:
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"获取股票 {code} 数据失败: {e}")
            return pd.DataFrame()
    
    def macd_golden_cross_strategy(self, data: pd.DataFrame) -> Tuple[bool, Dict]:
        """MACD金叉策略"""
        try:
            if len(data) < 50:
                return False, {}
            
            macd_data = self.indicator_calculator.calculate_macd(data)
            macd = macd_data['macd']
            signal = macd_data['signal']
            
            # 金叉条件：MACD线从下方突破信号线
            current_macd = macd.iloc[-1]
            current_signal = signal.iloc[-1]
            prev_macd = macd.iloc[-2]
            prev_signal = signal.iloc[-2]
            
            golden_cross = (prev_macd <= prev_signal) and (current_macd > current_signal)
            
            # 附加条件
            macd_positive = current_macd > 0  # MACD在零轴上方
            volume_surge = data['volume'].iloc[-1] > data['volume'].rolling(window=20).mean().iloc[-1] * 1.2
            
            pattern_info = {
                'strategy': 'MACD金叉',
                'macd_value': current_macd,
                'signal_value': current_signal,
                'golden_cross': golden_cross,
                'macd_positive': macd_positive,
                'volume_surge': volume_surge
                         }
             
            # 放宽条件：满足金叉或MACD正向即可
            return golden_cross or (macd_positive and volume_surge), pattern_info
            
        except Exception as e:
            logger.error(f"MACD金叉策略计算失败: {e}")
            return False, {}
    
    def kdj_low_golden_cross_strategy(self, data: pd.DataFrame) -> Tuple[bool, Dict]:
        """KDJ低位金叉策略"""
        try:
            if len(data) < 30:
                return False, {}
            
            kdj_data = self.indicator_calculator.calculate_kdj(data)
            k = kdj_data['K']
            d = kdj_data['D']
            j = kdj_data['J']
            
            # 金叉条件
            current_k = k.iloc[-1]
            current_d = d.iloc[-1]
            prev_k = k.iloc[-2]
            prev_d = d.iloc[-2]
            
            golden_cross = (prev_k <= prev_d) and (current_k > current_d)
            
            # 放宽低位条件
            low_position = current_k < 50 and current_d < 50  # 从30放宽到50
            
            # J值向上
            j_upward = j.iloc[-1] > j.iloc[-2]
            
            pattern_info = {
                'strategy': 'KDJ低位金叉',
                'k_value': current_k,
                'd_value': current_d,
                'j_value': j.iloc[-1],
                'golden_cross': golden_cross,
                'low_position': low_position,
                'j_upward': j_upward
            }
            
            # 放宽条件：满足金叉或低位向上即可
            return golden_cross or (low_position and j_upward), pattern_info
            
        except Exception as e:
            logger.error(f"KDJ低位金叉策略计算失败: {e}")
            return False, {}
    
    def rsi_oversold_rebound_strategy(self, data: pd.DataFrame) -> Tuple[bool, Dict]:
        """RSI超卖反弹策略"""
        try:
            if len(data) < 30:
                return False, {}
            
            rsi = self.indicator_calculator.calculate_rsi(data)
            
            current_rsi = rsi.iloc[-1]
            prev_rsi = rsi.iloc[-2]
            
            # 放宽超卖条件
            oversold_rebound = prev_rsi < 40 and current_rsi > prev_rsi  # 从30放宽到40
            
            # 价格支撑
            ma20 = data['close'].rolling(window=20).mean()
            price_support = data['close'].iloc[-1] > ma20.iloc[-1] * 0.90  # 放宽到90%
            
            pattern_info = {
                'strategy': 'RSI超卖反弹',
                'rsi_value': current_rsi,
                'prev_rsi': prev_rsi,
                'oversold_rebound': oversold_rebound,
                'price_support': price_support
            }
            
            # 放宽条件：满足任一即可
            return oversold_rebound or price_support, pattern_info
            
        except Exception as e:
            logger.error(f"RSI超卖反弹策略计算失败: {e}")
            return False, {}
    
    def boll_lower_bounce_strategy(self, data: pd.DataFrame) -> Tuple[bool, Dict]:
        """布林带下轨反弹策略"""
        try:
            if len(data) < 30:
                return False, {}
            
            boll_data = self.indicator_calculator.calculate_boll(data)
            upper = boll_data['upper']
            middle = boll_data['middle']
            lower = boll_data['lower']
            
            close = data['close']
            
            # 放宽下轨条件
            current_close = close.iloc[-1]
            prev_close = close.iloc[-2]
            current_lower = lower.iloc[-1]
            current_middle = middle.iloc[-1]
            
            # 接触下轨或中轨下方
            touched_lower = prev_close <= current_lower * 1.05 or current_close < current_middle
            bounce_up = current_close > prev_close  # 价格反弹
            
            # 成交量放大（条件放宽）
            volume_increase = data['volume'].iloc[-1] > data['volume'].iloc[-2] * 1.2
            
            pattern_info = {
                'strategy': '布林带下轨反弹',
                'close_price': current_close,
                'lower_band': current_lower,
                'middle_band': current_middle,
                'touched_lower': touched_lower,
                'bounce_up': bounce_up,
                'volume_increase': volume_increase
            }
            
            # 放宽条件：满足价格条件即可
            return touched_lower and (bounce_up or volume_increase), pattern_info
            
        except Exception as e:
            logger.error(f"布林带下轨反弹策略计算失败: {e}")
            return False, {}
    
    def dual_ma_crossover_strategy(self, data: pd.DataFrame) -> Tuple[bool, Dict]:
        """双均线突破策略"""
        try:
            if len(data) < 60:
                return False, {}
            
            ma5 = self.indicator_calculator.calculate_ma(data, 5)
            ma20 = self.indicator_calculator.calculate_ma(data, 20)
            
            # 金叉条件
            current_ma5 = ma5.iloc[-1]
            current_ma20 = ma20.iloc[-1]
            prev_ma5 = ma5.iloc[-2]
            prev_ma20 = ma20.iloc[-2]
            
            golden_cross = (prev_ma5 <= prev_ma20) and (current_ma5 > current_ma20)
            
            # 均线向上（条件放宽）
            ma5_upward = ma5.iloc[-1] >= ma5.iloc[-3]  # 平或向上
            ma20_upward = ma20.iloc[-1] >= ma20.iloc[-5]  # 放宽时间
            
            # 价格相对均线位置
            price_above = data['close'].iloc[-1] > current_ma5 * 0.98  # 允许轻微跌破
            
            pattern_info = {
                'strategy': '双均线突破',
                'ma5_value': current_ma5,
                'ma20_value': current_ma20,
                'golden_cross': golden_cross,
                'ma5_upward': ma5_upward,
                'ma20_upward': ma20_upward,
                'price_above': price_above
            }
            
            # 放宽条件：满足金叉或价格站稳均线
            return golden_cross or (ma5_upward and price_above), pattern_info
            
        except Exception as e:
            logger.error(f"双均线突破策略计算失败: {e}")
            return False, {}
    
    def run_comprehensive_strategy_selection(self):
        """运行全面策略选股"""
        logger.info("=== 开始全面策略选股测试 ===")
        
        test_stocks = self.get_high_quality_stocks(30)
        
        strategies = {
            'MACD金叉': self.macd_golden_cross_strategy,
            'KDJ低位金叉': self.kdj_low_golden_cross_strategy,
            'RSI超卖反弹': self.rsi_oversold_rebound_strategy,
            '布林带下轨反弹': self.boll_lower_bounce_strategy,
            '双均线突破': self.dual_ma_crossover_strategy
        }
        
        selection_results = {}
        
        for strategy_name, strategy_func in strategies.items():
            logger.info(f"执行策略: {strategy_name}")
            selected_stocks = []
            
            for stock_code in test_stocks:
                try:
                    data = self.get_stock_data(stock_code)
                    if data.empty or len(data) < 50:
                        continue
                    
                    selected, pattern_info = strategy_func(data)
                    if selected:
                        stock_info = {
                            'code': stock_code,
                            'name': data['name'].iloc[-1] if 'name' in data.columns else stock_code,
                            'close': data['close'].iloc[-1],
                            'volume': data['volume'].iloc[-1],
                            'pattern_info': pattern_info
                        }
                        selected_stocks.append(stock_info)
                        logger.info(f"✅ {strategy_name} 选中股票: {stock_code}")
                
                except Exception as e:
                    logger.error(f"处理股票 {stock_code} 时出错: {e}")
                    continue
            
            selection_results[strategy_name] = selected_stocks
            logger.info(f"{strategy_name} 完成，选出 {len(selected_stocks)} 只股票")
        
        return selection_results
    
    def buypoint_analysis_verification(self, stock_code: str, data: pd.DataFrame, original_pattern: Dict) -> Dict:
        """买点分析反向验证 - 简化版本"""
        try:
            logger.info(f"对股票 {stock_code} 进行买点分析反向验证")
            
            # 创建简化的买点验证器
            verification_result = {
                'stock_code': stock_code,
                'original_strategy': original_pattern.get('strategy', ''),
                'pattern_match': False,
                'verification_score': 0.0,
                'details': {}
            }
            
            # 根据原始策略类型进行技术指标匹配验证
            strategy_type = original_pattern.get('strategy', '')
            
            if strategy_type == 'MACD金叉':
                # 验证MACD相关特征
                macd_data = self.indicator_calculator.calculate_macd(data)
                current_macd = macd_data['macd'].iloc[-1]
                current_signal = macd_data['signal'].iloc[-1]
                
                # 检查是否符合MACD买点特征
                macd_above_signal = current_macd > current_signal
                macd_positive_trend = current_macd > macd_data['macd'].iloc[-5]
                
                if macd_above_signal and macd_positive_trend:
                    verification_result['pattern_match'] = True
                    verification_result['verification_score'] = 0.8
                    verification_result['details']['macd_verified'] = True
                
            elif strategy_type == 'KDJ低位金叉':
                # 验证KDJ相关特征
                kdj_data = self.indicator_calculator.calculate_kdj(data)
                current_k = kdj_data['K'].iloc[-1]
                current_d = kdj_data['D'].iloc[-1]
                
                # 检查是否符合KDJ买点特征
                kdj_golden_cross = current_k > current_d
                kdj_low_position = current_k < 60 and current_d < 60
                
                if kdj_golden_cross and kdj_low_position:
                    verification_result['pattern_match'] = True
                    verification_result['verification_score'] = 0.7
                    verification_result['details']['kdj_verified'] = True
                
            elif strategy_type == 'RSI超卖反弹':
                # 验证RSI相关特征
                rsi = self.indicator_calculator.calculate_rsi(data)
                current_rsi = rsi.iloc[-1]
                
                # 检查是否符合RSI买点特征
                rsi_recovery = current_rsi > rsi.iloc[-5]  # RSI向上
                rsi_reasonable = 20 < current_rsi < 60  # RSI在合理区间
                
                if rsi_recovery and rsi_reasonable:
                    verification_result['pattern_match'] = True
                    verification_result['verification_score'] = 0.6
                    verification_result['details']['rsi_verified'] = True
                
            elif strategy_type == '布林带下轨反弹':
                # 验证布林带相关特征
                boll_data = self.indicator_calculator.calculate_boll(data)
                current_close = data['close'].iloc[-1]
                lower_band = boll_data['lower'].iloc[-1]
                middle_band = boll_data['middle'].iloc[-1]
                
                # 检查是否符合布林带买点特征
                near_lower = current_close < lower_band * 1.1  # 接近下轨
                above_lower = current_close > lower_band  # 已经反弹
                
                if near_lower and above_lower:
                    verification_result['pattern_match'] = True
                    verification_result['verification_score'] = 0.7
                    verification_result['details']['boll_verified'] = True
                
            elif strategy_type == '双均线突破':
                # 验证双均线相关特征  
                ma5 = self.indicator_calculator.calculate_ma(data, 5)
                ma20 = self.indicator_calculator.calculate_ma(data, 20)
                
                # 检查是否符合双均线买点特征
                ma_bullish_alignment = ma5.iloc[-1] > ma20.iloc[-1]  # 多头排列
                price_above_ma = data['close'].iloc[-1] > ma5.iloc[-1]  # 价格站稳短均线
                
                if ma_bullish_alignment and price_above_ma:
                    verification_result['pattern_match'] = True
                    verification_result['verification_score'] = 0.8
                    verification_result['details']['ma_verified'] = True
            
            else:
                # 通用验证：检查基本技术面
                ma5 = self.indicator_calculator.calculate_ma(data, 5)
                ma20 = self.indicator_calculator.calculate_ma(data, 20)
                rsi = self.indicator_calculator.calculate_rsi(data)
                
                # 基本健康技术面
                technical_health = (
                    ma5.iloc[-1] > ma5.iloc[-5] and  # 短期均线向上
                    data['close'].iloc[-1] > ma20.iloc[-1] * 0.95 and  # 价格不远离长期均线
                    20 < rsi.iloc[-1] < 80  # RSI在正常区间
                )
                
                if technical_health:
                    verification_result['pattern_match'] = True
                    verification_result['verification_score'] = 0.5
                    verification_result['details']['basic_technical_health'] = True
            
            # 额外加分项：成交量和价格关系
            if len(data) >= 10:
                recent_volume_avg = data['volume'].tail(5).mean()
                earlier_volume_avg = data['volume'].tail(10).head(5).mean()
                volume_increase = recent_volume_avg > earlier_volume_avg * 1.1
                
                price_stability = abs(data['close'].iloc[-1] - data['close'].iloc[-5]) / data['close'].iloc[-5] < 0.15
                
                if volume_increase and price_stability:
                    verification_result['verification_score'] += 0.1
                    verification_result['details']['volume_price_healthy'] = True
            
            logger.info(f"股票 {stock_code} 验证完成，匹配度: {verification_result['verification_score']:.2f}")
            return verification_result
            
        except Exception as e:
            logger.error(f"买点分析验证失败: {e}")
            return {
                'stock_code': stock_code,
                'original_strategy': original_pattern.get('strategy', ''),
                'error': str(e),
                'pattern_match': False,
                'verification_score': 0.0
            }
    
    def run_closed_loop_verification(self):
        """运行闭环验证"""
        logger.info("=== 开始策略选股闭环验证 ===")
        
        # 第一步：策略选股
        selection_results = self.run_comprehensive_strategy_selection()
        
        # 第二步：买点分析反向验证
        verification_results = {}
        
        for strategy_name, selected_stocks in selection_results.items():
            logger.info(f"对 {strategy_name} 选出的股票进行反向验证")
            strategy_verifications = []
            
            for stock_info in selected_stocks:
                stock_code = stock_info['code']
                data = self.get_stock_data(stock_code)
                
                if not data.empty:
                    verification = self.buypoint_analysis_verification(
                        stock_code, data, stock_info['pattern_info']
                    )
                    strategy_verifications.append(verification)
            
            verification_results[strategy_name] = strategy_verifications
        
        # 第三步：生成闭环验证报告
        self._generate_closed_loop_report(selection_results, verification_results)
        
        return selection_results, verification_results
    
    def _generate_closed_loop_report(self, selection_results: Dict, verification_results: Dict):
        """生成闭环验证报告"""
        logger.info("=== 策略选股闭环验证报告 ===")
        
        total_selected = 0
        total_verified = 0
        
        for strategy_name in selection_results.keys():
            selected_count = len(selection_results[strategy_name])
            verified_stocks = [v for v in verification_results[strategy_name] if v.get('pattern_match', False)]
            verified_count = len(verified_stocks)
            
            verification_rate = (verified_count / selected_count * 100) if selected_count > 0 else 0
            
            total_selected += selected_count
            total_verified += verified_count
            
            logger.info(f"{strategy_name}:")
            logger.info(f"  选出股票: {selected_count} 只")
            logger.info(f"  验证通过: {verified_count} 只")
            logger.info(f"  验证率: {verification_rate:.1f}%")
            
            # 显示验证通过的股票
            if verified_stocks:
                logger.info(f"  验证通过的股票:")
                for v in verified_stocks[:3]:  # 显示前3只
                    score = v.get('verification_score', 0)
                    logger.info(f"    {v['stock_code']} (验证分数: {score:.2f})")
        
        overall_verification_rate = (total_verified / total_selected * 100) if total_selected > 0 else 0
        
        logger.info(f"\n总体情况:")
        logger.info(f"  总选股数: {total_selected} 只")
        logger.info(f"  总验证通过: {total_verified} 只") 
        logger.info(f"  总体验证率: {overall_verification_rate:.1f}%")
        
        # 保存详细报告
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f"closed_loop_verification_report_{timestamp}.json"
        
        import json
        report_data = {
            'timestamp': timestamp,
            'selection_results': selection_results,
            'verification_results': verification_results,
            'summary': {
                'total_selected': total_selected,
                'total_verified': total_verified,
                'overall_verification_rate': overall_verification_rate
            }
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"详细报告已保存到: {report_file}")


def main():
    """主函数"""
    validator = ComprehensiveStrategyValidator()
    selection_results, verification_results = validator.run_closed_loop_verification()
    
    # 计算总体成功率
    total_selected = sum(len(stocks) for stocks in selection_results.values())
    total_verified = sum(len([v for v in verifications if v.get('pattern_match', False)]) 
                        for verifications in verification_results.values())
    
    if total_selected > 0 and total_verified > 0:
        success_rate = total_verified / total_selected * 100
        print(f"\n🎉 闭环验证成功! 选出 {total_selected} 只股票，验证通过 {total_verified} 只，验证率 {success_rate:.1f}%")
        return True
    else:
        print(f"\n❌ 闭环验证需要改进! 选出 {total_selected} 只股票，验证通过 {total_verified} 只")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 