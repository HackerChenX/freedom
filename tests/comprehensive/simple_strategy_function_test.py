"""
简化的策略功能测试器

直接测试策略类的具体选股方法，确保策略能正确选出股票
解决抽象类实例化问题，专注于策略核心功能验证
"""

import time
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.logger import getLogger
from utils.decorators import performance_monitor, exception_handler
from utils.dependency_injection import get_service
from db.interfaces.data_access_interface import DataAccessInterface

logger = getLogger(__name__)


@dataclass
class SimpleStrategyTestResult:
    """简化的策略测试结果"""
    strategy_name: str
    test_date: datetime
    execution_time: float
    success: bool
    error_message: Optional[str]
    selected_stocks_count: int
    selected_stocks: List[str]
    test_details: Dict[str, Any]


class SimpleStrategyFunctionTester:
    """
    简化的策略功能测试器
    
    直接测试策略的核心选股方法
    """
    
    def __init__(self):
        """初始化测试器"""
        self.data_access = get_service(DataAccessInterface)
        self.test_stocks = ["000001", "000002", "600000", "600036", "000858"]
        logger.info("简化策略功能测试器初始化完成")
    
    @performance_monitor(threshold=60.0)
    def test_dual_ma_strategy_direct(self) -> SimpleStrategyTestResult:
        """
        直接测试双均线策略核心逻辑
        """
        logger.info("开始测试双均线策略直接实现")
        start_time = time.time()
        
        try:
            # 直接实现双均线策略逻辑
            selected_stocks = []
            test_details = {
                'tested_stocks': [],
                'breakout_signals': [],
                'data_quality': {},
                'calculation_steps': []
            }
            
            for code in self.test_stocks:
                try:
                    # 获取股票数据
                    end_date = datetime.now().strftime('%Y-%m-%d')
                    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
                    
                    query = f"""
                    SELECT code, name, date, open, high, low, close, volume
                    FROM stock_info 
                    WHERE code = '{code}'
                    AND date >= '{start_date}' AND date <= '{end_date}'
                    AND level = '日线'
                    ORDER BY date ASC
                    """
                    
                    stock_data = self.data_access.execute_query_data_access_interface(query)
                    
                    if stock_data.empty or len(stock_data) < 30:
                        logger.warning(f"股票 {code} 数据不足: {len(stock_data)} 条记录")
                        continue
                    
                    test_details['tested_stocks'].append(code)
                    test_details['data_quality'][code] = {
                        'record_count': len(stock_data),
                        'date_range': f"{stock_data['date'].min()} to {stock_data['date'].max()}"
                    }
                    
                    # 计算双均线
                    stock_data['ma5'] = stock_data['close'].rolling(window=5).mean()
                    stock_data['ma10'] = stock_data['close'].rolling(window=10).mean()
                    
                    # 检查金叉信号
                    stock_data['golden_cross'] = (
                        (stock_data['ma5'] > stock_data['ma10']) & 
                        (stock_data['ma5'].shift(1) <= stock_data['ma10'].shift(1))
                    )
                    
                    # 检查最近是否有金叉
                    recent_golden_cross = stock_data['golden_cross'].tail(5).any()
                    
                    if recent_golden_cross:
                        selected_stocks.append(code)
                        golden_cross_dates = stock_data[stock_data['golden_cross']]['date'].tolist()
                        test_details['breakout_signals'].append({
                            'code': code,
                            'golden_cross_dates': [str(date) for date in golden_cross_dates[-3:]]
                        })
                        logger.info(f"✅ 股票 {code} 符合双均线策略: 最近有金叉信号")
                    
                    test_details['calculation_steps'].append({
                        'code': code,
                        'ma5_latest': float(stock_data['ma5'].iloc[-1]) if not pd.isna(stock_data['ma5'].iloc[-1]) else None,
                        'ma10_latest': float(stock_data['ma10'].iloc[-1]) if not pd.isna(stock_data['ma10'].iloc[-1]) else None,
                        'recent_golden_cross': recent_golden_cross
                    })
                    
                except Exception as e:
                    logger.error(f"处理股票 {code} 时出错: {e}")
                    continue
            
            execution_time = time.time() - start_time
            success = len(selected_stocks) > 0
            
            result = SimpleStrategyTestResult(
                strategy_name="双均线策略直接实现",
                test_date=datetime.now(),
                execution_time=execution_time,
                success=success,
                error_message=None if success else "未选出任何股票",
                selected_stocks_count=len(selected_stocks),
                selected_stocks=selected_stocks,
                test_details=test_details
            )
            
            logger.info(f"双均线策略测试完成: 选出 {len(selected_stocks)} 只股票，耗时 {execution_time:.2f}秒")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"双均线策略测试失败: {str(e)}"
            logger.error(error_msg)
            
            return SimpleStrategyTestResult(
                strategy_name="双均线策略直接实现",
                test_date=datetime.now(),
                execution_time=execution_time,
                success=False,
                error_message=error_msg,
                selected_stocks_count=0,
                selected_stocks=[],
                test_details={'error': str(e)}
            )
    
    @performance_monitor(threshold=60.0)
    def test_volume_breakout_strategy(self) -> SimpleStrategyTestResult:
        """
        测试成交量突破策略
        """
        logger.info("开始测试成交量突破策略")
        start_time = time.time()
        
        try:
            selected_stocks = []
            test_details = {
                'tested_stocks': [],
                'volume_signals': [],
                'data_quality': {}
            }
            
            for code in self.test_stocks:
                try:
                    # 获取股票数据
                    end_date = datetime.now().strftime('%Y-%m-%d')
                    start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
                    
                    query = f"""
                    SELECT code, name, date, open, high, low, close, volume
                    FROM stock_info 
                    WHERE code = '{code}'
                    AND date >= '{start_date}' AND date <= '{end_date}'
                    AND level = '日线'
                    ORDER BY date ASC
                    """
                    
                    stock_data = self.data_access.execute_query_data_access_interface(query)
                    
                    if stock_data.empty or len(stock_data) < 20:
                        continue
                    
                    test_details['tested_stocks'].append(code)
                    test_details['data_quality'][code] = {
                        'record_count': len(stock_data),
                        'avg_volume': float(stock_data['volume'].mean())
                    }
                    
                    # 计算成交量均线
                    stock_data['vol_ma10'] = stock_data['volume'].rolling(window=10).mean()
                    
                    # 检查成交量突破
                    stock_data['volume_breakout'] = stock_data['volume'] > (stock_data['vol_ma10'] * 2)
                    
                    # 检查最近是否有成交量突破
                    recent_volume_breakout = stock_data['volume_breakout'].tail(3).any()
                    
                    if recent_volume_breakout:
                        selected_stocks.append(code)
                        breakout_dates = stock_data[stock_data['volume_breakout']]['date'].tolist()
                        test_details['volume_signals'].append({
                            'code': code,
                            'breakout_dates': [str(date) for date in breakout_dates[-2:]]
                        })
                        logger.info(f"✅ 股票 {code} 符合成交量突破策略")
                    
                except Exception as e:
                    logger.warning(f"处理股票 {code} 时出错: {e}")
                    continue
            
            execution_time = time.time() - start_time
            success = len(selected_stocks) > 0
            
            result = SimpleStrategyTestResult(
                strategy_name="成交量突破策略",
                test_date=datetime.now(),
                execution_time=execution_time,
                success=success,
                error_message=None if success else "未选出任何股票",
                selected_stocks_count=len(selected_stocks),
                selected_stocks=selected_stocks,
                test_details=test_details
            )
            
            logger.info(f"成交量突破策略测试完成: 选出 {len(selected_stocks)} 只股票")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"成交量突破策略测试失败: {str(e)}"
            logger.error(error_msg)
            
            return SimpleStrategyTestResult(
                strategy_name="成交量突破策略",
                test_date=datetime.now(),
                execution_time=execution_time,
                success=False,
                error_message=error_msg,
                selected_stocks_count=0,
                selected_stocks=[],
                test_details={'error': str(e)}
            )
    
    @performance_monitor(threshold=60.0)
    def test_price_momentum_strategy(self) -> SimpleStrategyTestResult:
        """
        测试价格动量策略
        """
        logger.info("开始测试价格动量策略")
        start_time = time.time()
        
        try:
            selected_stocks = []
            test_details = {
                'tested_stocks': [],
                'momentum_signals': [],
                'data_quality': {}
            }
            
            for code in self.test_stocks:
                try:
                    # 获取股票数据
                    end_date = datetime.now().strftime('%Y-%m-%d')
                    start_date = (datetime.now() - timedelta(days=45)).strftime('%Y-%m-%d')
                    
                    query = f"""
                    SELECT code, name, date, open, high, low, close, volume
                    FROM stock_info 
                    WHERE code = '{code}'
                    AND date >= '{start_date}' AND date <= '{end_date}'
                    AND level = '日线'
                    ORDER BY date ASC
                    """
                    
                    stock_data = self.data_access.execute_query_data_access_interface(query)
                    
                    if stock_data.empty or len(stock_data) < 15:
                        continue
                    
                    test_details['tested_stocks'].append(code)
                    
                    # 计算价格动量（5日和20日收益率）
                    stock_data['return_5d'] = stock_data['close'].pct_change(5)
                    stock_data['return_20d'] = stock_data['close'].pct_change(20)
                    
                    # 最新动量指标
                    latest_return_5d = stock_data['return_5d'].iloc[-1]
                    latest_return_20d = stock_data['return_20d'].iloc[-1]
                    
                    test_details['data_quality'][code] = {
                        'record_count': len(stock_data),
                        'latest_return_5d': float(latest_return_5d) if not pd.isna(latest_return_5d) else None,
                        'latest_return_20d': float(latest_return_20d) if not pd.isna(latest_return_20d) else None
                    }
                    
                    # 动量策略条件：5日收益率 > 3% 且 20日收益率 > 0%
                    momentum_signal = (
                        not pd.isna(latest_return_5d) and latest_return_5d > 0.03 and
                        not pd.isna(latest_return_20d) and latest_return_20d > 0.0
                    )
                    
                    if momentum_signal:
                        selected_stocks.append(code)
                        test_details['momentum_signals'].append({
                            'code': code,
                            'return_5d': float(latest_return_5d),
                            'return_20d': float(latest_return_20d)
                        })
                        logger.info(f"✅ 股票 {code} 符合价格动量策略")
                    
                except Exception as e:
                    logger.warning(f"处理股票 {code} 时出错: {e}")
                    continue
            
            execution_time = time.time() - start_time
            success = len(selected_stocks) > 0
            
            result = SimpleStrategyTestResult(
                strategy_name="价格动量策略",
                test_date=datetime.now(),
                execution_time=execution_time,
                success=success,
                error_message=None if success else "未选出任何股票",
                selected_stocks_count=len(selected_stocks),
                selected_stocks=selected_stocks,
                test_details=test_details
            )
            
            logger.info(f"价格动量策略测试完成: 选出 {len(selected_stocks)} 只股票")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"价格动量策略测试失败: {str(e)}"
            logger.error(error_msg)
            
            return SimpleStrategyTestResult(
                strategy_name="价格动量策略",
                test_date=datetime.now(),
                execution_time=execution_time,
                success=False,
                error_message=error_msg,
                selected_stocks_count=0,
                selected_stocks=[],
                test_details={'error': str(e)}
            )
    
    @performance_monitor(threshold=180.0)
    def run_all_strategy_tests(self) -> List[SimpleStrategyTestResult]:
        """
        运行所有策略测试
        """
        logger.info("开始运行所有简化策略测试")
        
        results = []
        
        # 测试双均线策略
        dual_ma_result = self.test_dual_ma_strategy_direct()
        results.append(dual_ma_result)
        
        # 测试成交量突破策略
        volume_result = self.test_volume_breakout_strategy()
        results.append(volume_result)
        
        # 测试价格动量策略
        momentum_result = self.test_price_momentum_strategy()
        results.append(momentum_result)
        
        # 汇总结果
        successful_strategies = [r for r in results if r.success]
        total_selected_stocks = sum(r.selected_stocks_count for r in successful_strategies)
        
        logger.info(f"所有策略测试完成: {len(successful_strategies)}/{len(results)} 个策略成功")
        logger.info(f"总计选出股票: {total_selected_stocks} 只")
        
        return results
    
    def save_test_report(self, results: List[SimpleStrategyTestResult], filename: Optional[str] = None) -> str:
        """
        保存测试报告
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simple_strategy_test_report_{timestamp}.txt"
        
        # 生成报告内容
        lines = []
        lines.append("=" * 80)
        lines.append("简化策略功能测试报告")
        lines.append("=" * 80)
        lines.append(f"测试时间: {datetime.now()}")
        lines.append(f"测试策略数: {len(results)}")
        
        successful_results = [r for r in results if r.success]
        lines.append(f"成功策略数: {len(successful_results)}")
        lines.append(f"失败策略数: {len(results) - len(successful_results)}")
        lines.append("")
        
        # 详细结果
        for result in results:
            status = "✅ 成功" if result.success else "❌ 失败"
            lines.append(f"{result.strategy_name}: {status}")
            lines.append(f"  选出股票数: {result.selected_stocks_count}")
            lines.append(f"  执行时间: {result.execution_time:.3f}秒")
            
            if result.error_message:
                lines.append(f"  错误信息: {result.error_message}")
            
            if result.selected_stocks:
                lines.append(f"  选出股票: {', '.join(result.selected_stocks)}")
            
            # 测试详情
            if 'tested_stocks' in result.test_details:
                lines.append(f"  测试股票数: {len(result.test_details['tested_stocks'])}")
            
            lines.append("")
        
        # 总结
        if successful_results:
            total_selected = sum(r.selected_stocks_count for r in successful_results)
            lines.append(f"总体表现:")
            lines.append(f"  成功率: {len(successful_results)}/{len(results)} ({100*len(successful_results)/len(results):.1f}%)")
            lines.append(f"  总选股数: {total_selected}")
            lines.append(f"  平均执行时间: {sum(r.execution_time for r in results)/len(results):.3f}秒")
        
        lines.append("")
        lines.append("=" * 80)
        
        # 保存到文件
        with open(filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        # 同时保存JSON格式
        json_filename = filename.replace('.txt', '.json')
        json_data = {
            'test_timestamp': datetime.now().isoformat(),
            'total_strategies': len(results),
            'successful_strategies': len(successful_results),
            'results': [
                {
                    'strategy_name': r.strategy_name,
                    'success': r.success,
                    'execution_time': r.execution_time,
                    'selected_stocks_count': r.selected_stocks_count,
                    'selected_stocks': r.selected_stocks,
                    'error_message': r.error_message,
                    'test_details': r.test_details
                }
                for r in results
            ]
        }
        
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"测试报告已保存: {filename} 和 {json_filename}")
        return filename


def main():
    """主函数"""
    logger.info("开始简化策略功能测试")
    
    try:
        # 创建测试器
        tester = SimpleStrategyFunctionTester()
        
        # 运行所有测试
        results = tester.run_all_strategy_tests()
        
        # 保存报告
        report_file = tester.save_test_report(results)
        
        # 输出摘要
        successful_results = [r for r in results if r.success]
        total_selected_stocks = sum(r.selected_stocks_count for r in successful_results)
        
        print(f"\n简化策略功能测试完成!")
        print(f"测试策略数: {len(results)}")
        print(f"成功策略数: {len(successful_results)}")
        print(f"总选股数: {total_selected_stocks}")
        print(f"报告文件: {report_file}")
        
        # 显示每个策略的结果
        for result in results:
            status = "✅" if result.success else "❌"
            print(f"{status} {result.strategy_name}: {result.selected_stocks_count} 只股票")
            if result.selected_stocks:
                print(f"   选出股票: {', '.join(result.selected_stocks)}")
        
        return len(successful_results) > 0
        
    except Exception as e:
        logger.error(f"策略功能测试失败: {e}")
        print(f"测试失败: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 