#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
基于真实完整数据的4000只股票性能测试

现在我们有了真实的4378只股票数据，进行实际的性能测试
"""

import sys
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
import clickhouse_connect

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from utils.logger import get_logger

logger = get_logger(__name__)


class RealFullDataPerformanceTest:
    """基于真实完整数据的性能测试器"""
    
    def __init__(self):
        self.client = None
        self.test_results = {}
        
    def connect_to_clickhouse(self) -> bool:
        """连接到真实的ClickHouse数据库"""
        try:
            self.client = clickhouse_connect.get_client(
                host='localhost',
                port=8123,
                database='stock',
                user='default',
                password='123456'
            )
            
            # 测试连接并获取数据统计
            result = self.client.query('SELECT COUNT(*) FROM stock_info')
            total_records = result.result_rows[0][0]
            
            result = self.client.query('SELECT COUNT(DISTINCT code) FROM stock_info')
            unique_stocks = result.result_rows[0][0]
            
            logger.info(f"✅ ClickHouse连接成功")
            logger.info(f"📊 总记录数: {total_records:,}")
            logger.info(f"📈 股票数量: {unique_stocks:,}")
            return True
            
        except Exception as e:
            logger.error(f"❌ ClickHouse连接失败: {e}")
            return False
    
    def get_all_stock_codes(self) -> List[str]:
        """获取所有股票代码"""
        try:
            result = self.client.query("""
                SELECT DISTINCT code 
                FROM stock_info 
                WHERE level = '日线'
                ORDER BY code
            """)
            
            stock_codes = [row[0] for row in result.result_rows]
            logger.info(f"📋 获取到 {len(stock_codes)} 只股票代码")
            return stock_codes
            
        except Exception as e:
            logger.error(f"❌ 获取股票代码失败: {e}")
            return []
    
    def query_single_stock(self, code: str) -> Dict[str, Any]:
        """查询单只股票数据"""
        try:
            start_time = time.time()
            
            result = self.client.query(f"""
                SELECT code, name, date, open, high, low, close, volume
                FROM stock_info 
                WHERE code = '{code}' AND level = '日线'
                ORDER BY date DESC
                LIMIT 100
            """)
            
            query_time = time.time() - start_time
            records = len(result.result_rows)
            
            return {
                'code': code,
                'records': records,
                'query_time': query_time,
                'success': True
            }
            
        except Exception as e:
            return {
                'code': code,
                'records': 0,
                'query_time': 0,
                'success': False,
                'error': str(e)
            }
    
    def query_batch_stocks(self, stock_codes: List[str]) -> Dict[str, Any]:
        """批量查询股票数据"""
        try:
            start_time = time.time()
            
            # 批量查询（使用IN语句）
            codes_str = "', '".join(stock_codes)
            result = self.client.query(f"""
                SELECT code, name, date, open, high, low, close, volume
                FROM stock_info 
                WHERE code IN ('{codes_str}') AND level = '日线'
                ORDER BY code, date DESC
            """)
            
            query_time = time.time() - start_time
            records = len(result.result_rows)
            
            return {
                'batch_size': len(stock_codes),
                'records': records,
                'query_time': query_time,
                'success': True
            }
            
        except Exception as e:
            return {
                'batch_size': len(stock_codes),
                'records': 0,
                'query_time': 0,
                'success': False,
                'error': str(e)
            }
    
    def test_serial_performance(self, stock_codes: List[str], test_count: int = 100) -> Dict[str, Any]:
        """测试串行查询性能"""
        print(f"🔄 开始串行查询测试（前{test_count}只股票）...")
        
        start_time = time.time()
        total_records = 0
        successful_queries = 0
        failed_queries = 0
        
        test_codes = stock_codes[:test_count]
        
        for i, code in enumerate(test_codes):
            result = self.query_single_stock(code)
            
            if result['success']:
                total_records += result['records']
                successful_queries += 1
            else:
                failed_queries += 1
            
            if (i + 1) % 20 == 0:
                print(f"  已处理 {i + 1}/{test_count} 只股票...")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 计算所有股票的理论时间
        avg_time_per_stock = total_time / test_count
        theoretical_all_time = avg_time_per_stock * len(stock_codes)
        
        return {
            'method': 'serial',
            'stocks_tested': test_count,
            'total_stocks': len(stock_codes),
            'total_time': total_time,
            'avg_time_per_stock': avg_time_per_stock,
            'theoretical_all_time': theoretical_all_time,
            'total_records': total_records,
            'successful_queries': successful_queries,
            'failed_queries': failed_queries
        }
    
    def test_batch_performance(self, stock_codes: List[str]) -> Dict[str, Any]:
        """测试批量查询性能"""
        print(f"🚀 开始批量查询测试（{len(stock_codes)}只股票）...")
        
        batch_size = 150  # 每批150只股票
        batches = [stock_codes[i:i+batch_size] for i in range(0, len(stock_codes), batch_size)]
        
        start_time = time.time()
        total_records = 0
        successful_batches = 0
        failed_batches = 0
        
        for i, batch in enumerate(batches):
            result = self.query_batch_stocks(batch)
            
            if result['success']:
                total_records += result['records']
                successful_batches += 1
            else:
                failed_batches += 1
            
            if (i + 1) % 10 == 0:
                print(f"  批次 {i + 1}/{len(batches)}: 累计 {total_records:,} 条记录")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        return {
            'method': 'batch',
            'total_stocks': len(stock_codes),
            'total_batches': len(batches),
            'batch_size': batch_size,
            'total_time': total_time,
            'total_records': total_records,
            'successful_batches': successful_batches,
            'failed_batches': failed_batches
        }
    
    def test_parallel_performance(self, stock_codes: List[str]) -> Dict[str, Any]:
        """测试并行查询性能"""
        print(f"⚡ 开始并行查询测试（{len(stock_codes)}只股票）...")
        
        batch_size = 150
        max_workers = 20
        batches = [stock_codes[i:i+batch_size] for i in range(0, len(stock_codes), batch_size)]
        
        start_time = time.time()
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.query_batch_stocks, batch) for batch in batches]
            
            for i, future in enumerate(futures):
                result = future.result()
                results.append(result)
                
                if (i + 1) % 5 == 0:
                    completed_batches = i + 1
                    total_records = sum(r['records'] for r in results if r['success'])
                    print(f"  完成 {completed_batches}/{len(batches)} 批次: 累计 {total_records:,} 条记录")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        total_records = sum(r['records'] for r in results if r['success'])
        successful_batches = sum(1 for r in results if r['success'])
        failed_batches = sum(1 for r in results if not r['success'])
        
        return {
            'method': 'parallel',
            'total_stocks': len(stock_codes),
            'total_batches': len(batches),
            'batch_size': batch_size,
            'max_workers': max_workers,
            'total_time': total_time,
            'total_records': total_records,
            'successful_batches': successful_batches,
            'failed_batches': failed_batches
        }
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """运行综合性能测试"""
        print("="*60)
        print("🎯 基于真实完整数据的性能测试开始")
        print("="*60)
        
        if not self.connect_to_clickhouse():
            return {'error': 'ClickHouse连接失败'}
        
        # 获取所有股票代码
        stock_codes = self.get_all_stock_codes()
        if not stock_codes:
            return {'error': '无法获取股票代码'}
        
        print(f"📊 将测试 {len(stock_codes)} 只真实股票的查询性能")
        
        results = {}
        
        # 1. 串行查询测试（仅测试100只以节省时间）
        results['serial'] = self.test_serial_performance(stock_codes, 100)
        
        # 2. 批量查询测试（测试所有股票）
        results['batch'] = self.test_batch_performance(stock_codes)
        
        # 3. 并行查询测试（测试所有股票）
        results['parallel'] = self.test_parallel_performance(stock_codes)
        
        return results
    
    def print_results(self, results: Dict[str, Any]):
        """打印测试结果"""
        if 'error' in results:
            print(f"❌ 测试失败: {results['error']}")
            return
        
        print("\n" + "="*60)
        print("📈 真实完整数据性能测试结果汇总")
        print("="*60)
        
        for method_name, result in results.items():
            print(f"\n🔍 {method_name.upper()} 方法:")
            
            if method_name == 'serial':
                print(f"  - 测试股票数: {result['stocks_tested']} 只")
                print(f"  - 总股票数: {result['total_stocks']} 只")
                print(f"  - 实际耗时: {result['total_time']:.3f} 秒")
                print(f"  - 平均每股: {result['avg_time_per_stock']:.3f} 秒")
                print(f"  - 全部股票理论时间: {result['theoretical_all_time']:.1f} 秒 ({result['theoretical_all_time']/60:.1f} 分钟)")
            else:
                print(f"  - 总股票数: {result['total_stocks']} 只")
                print(f"  - 总批次数: {result['total_batches']}")
                print(f"  - 批次大小: {result['batch_size']}")
                if 'max_workers' in result:
                    print(f"  - 并发数: {result['max_workers']}")
                print(f"  - 实际耗时: {result['total_time']:.3f} 秒")
                print(f"  - 成功批次: {result['successful_batches']}")
                print(f"  - 失败批次: {result['failed_batches']}")
            
            print(f"  - 查询记录数: {result['total_records']:,}")
        
        # 性能对比
        if 'serial' in results and 'parallel' in results:
            serial_theoretical = results['serial']['theoretical_all_time']
            parallel_actual = results['parallel']['total_time']
            speedup = serial_theoretical / parallel_actual
            
            print(f"\n🚀 性能提升分析:")
            print(f"  - 串行理论时间: {serial_theoretical:.1f} 秒 ({serial_theoretical/60:.1f} 分钟)")
            print(f"  - 并行实际时间: {parallel_actual:.3f} 秒 ({parallel_actual/60:.1f} 分钟)")
            print(f"  - 性能提升倍数: {speedup:.1f}x")
            
            if parallel_actual <= 300:  # 5分钟 = 300秒
                print(f"  ✅ 已达成5分钟目标 (实际: {parallel_actual/60:.1f} 分钟)")
            else:
                print(f"  ❌ 未达成5分钟目标 (实际: {parallel_actual/60:.1f} 分钟)")
                
            print(f"\n📊 真实数据验证:")
            print(f"  - 这是基于 {results['parallel']['total_stocks']} 只真实股票的测试")
            print(f"  - 总计查询了 {results['parallel']['total_records']:,} 条真实记录")
            print(f"  - 完全不含模拟数据，结果真实可信")


def main():
    """主函数"""
    tester = RealFullDataPerformanceTest()
    results = tester.run_comprehensive_test()
    tester.print_results(results)


if __name__ == "__main__":
    main() 