#!/usr/bin/env python3
"""
生产环境数据库连接测试工具

支持多种连接方式：
1. 本地ClickHouse服务器
2. 远程ClickHouse服务器（通过环境变量）
3. Docker容器中的ClickHouse
4. 云端ClickHouse服务

完整的系统验证包括：
- 数据库连接测试
- 数据表结构验证
- 指标计算性能测试
- 选股功能验证
"""

import sys
import os
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import subprocess

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from config.database_config_manager import get_database_config_manager
from db.unified_data_manager import UnifiedDataManager
from utils.logger import get_logger
from analysis.engines.indicator_validation_framework import ProductionIndicatorValidator

logger = get_logger(__name__)


class ProductionDatabaseTester:
    """生产环境数据库测试器"""
    
    def __init__(self):
        self.config_manager = get_database_config_manager()
        self.test_results = []
        self.start_time = None
        self.data_manager = None
        self.validator = None
    
    def print_banner(self):
        """打印测试横幅"""
        print("=" * 80)
        print("🚀 生产环境数据库连接测试工具")
        print("=" * 80)
        print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"项目根目录: {root_dir}")
        print("=" * 80)
    
    def test_connection_methods(self) -> Dict[str, Any]:
        """测试多种连接方式"""
        print("\n🔍 测试数据库连接方式...")
        
        methods = {
            'local_default': self._test_local_default,
            'environment_vars': self._test_environment_vars,
            'docker_container': self._test_docker_container,
            'custom_config': self._test_custom_config
        }
        
        results = {}
        for method_name, test_func in methods.items():
            try:
                print(f"\n📡 测试 {method_name} 连接方式...")
                result = test_func()
                results[method_name] = result
                if result['success']:
                    print(f"✅ {method_name} 连接成功")
                else:
                    print(f"❌ {method_name} 连接失败: {result['error']}")
            except Exception as e:
                results[method_name] = {
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                print(f"❌ {method_name} 连接异常: {e}")
        
        return results
    
    def _test_local_default(self) -> Dict[str, Any]:
        """测试本地默认连接"""
        try:
            config = self.config_manager.get_connection_config()
            if config['host'] == 'localhost' and config['port'] == 9000:
                return self._test_connection_with_config(config)
            else:
                return {'success': False, 'error': '非本地默认配置'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _test_environment_vars(self) -> Dict[str, Any]:
        """测试环境变量连接"""
        env_vars = [
            'CLICKHOUSE_HOST', 'CLICKHOUSE_PORT', 'CLICKHOUSE_DATABASE',
            'CLICKHOUSE_USER', 'CLICKHOUSE_PASSWORD'
        ]
        
        # 检查是否有环境变量设置
        has_env_vars = any(os.environ.get(var) for var in env_vars)
        if not has_env_vars:
            return {'success': False, 'error': '未设置环境变量'}
        
        try:
            # 显示环境变量配置（密码隐藏）
            env_config = {}
            for var in env_vars:
                value = os.environ.get(var)
                if value:
                    if 'PASSWORD' in var:
                        env_config[var] = '***'
                    else:
                        env_config[var] = value
            
            print(f"  环境变量配置: {env_config}")
            
            config = self.config_manager.get_connection_config()
            return self._test_connection_with_config(config)
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _test_docker_container(self) -> Dict[str, Any]:
        """测试Docker容器连接"""
        try:
            # 检查是否有Docker容器运行
            result = subprocess.run(['docker', 'ps', '--filter', 'name=clickhouse', '--format', '{{.Names}}'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                return {'success': False, 'error': 'Docker命令执行失败'}
            
            containers = result.stdout.strip().split('\n')
            if not containers or containers == ['']:
                return {'success': False, 'error': '未找到ClickHouse Docker容器'}
            
            print(f"  找到Docker容器: {containers}")
            
            # 尝试连接Docker容器中的ClickHouse
            docker_config = {
                'host': 'localhost',
                'port': 9000,
                'database': 'default',
                'user': 'default',
                'password': '',
                'compression': False
            }
            
            return self._test_connection_with_config(docker_config)
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'Docker命令超时'}
        except FileNotFoundError:
            return {'success': False, 'error': 'Docker未安装'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _test_custom_config(self) -> Dict[str, Any]:
        """测试自定义配置连接"""
        try:
            # 检查是否有自定义配置文件
            config_file = os.path.join(root_dir, 'config', 'database.yaml')
            if not os.path.exists(config_file):
                return {'success': False, 'error': '未找到自定义配置文件'}
            
            config = self.config_manager.get_connection_config()
            return self._test_connection_with_config(config)
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _test_connection_with_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """使用指定配置测试连接"""
        start_time = time.time()
        
        try:
            # 创建数据管理器实例
            data_manager = UnifiedDataManager()
            
            # 测试基本连接
            connection_test = data_manager.test_connection()
            if not connection_test:
                return {
                    'success': False,
                    'error': '数据库连接失败',
                    'config': {k: v if k != 'password' else '***' for k, v in config.items()},
                    'duration': time.time() - start_time
                }
            
            # 测试数据查询
            test_query = "SELECT COUNT(*) as count FROM system.databases"
            result = data_manager.execute_query(test_query)
            
            if result is None or result.empty:
                return {
                    'success': False,
                    'error': '查询测试失败',
                    'config': {k: v if k != 'password' else '***' for k, v in config.items()},
                    'duration': time.time() - start_time
                }
            
            return {
                'success': True,
                'config': {k: v if k != 'password' else '***' for k, v in config.items()},
                'duration': time.time() - start_time,
                'database_count': int(result.iloc[0]['count']) if not result.empty else 0
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'config': {k: v if k != 'password' else '***' for k, v in config.items()},
                'duration': time.time() - start_time
            }
    
    def test_database_schema(self) -> Dict[str, Any]:
        """测试数据库架构"""
        print("\n🗃️ 测试数据库架构...")
        
        try:
            data_manager = UnifiedDataManager()
            
            # 检查必要的表
            required_tables = [
                'stock_info', 'stock_daily', 'stock_basic',
                'stock_minute', 'stock_weekly', 'stock_monthly'
            ]
            
            existing_tables = []
            missing_tables = []
            
            for table in required_tables:
                try:
                    query = f"SELECT COUNT(*) as count FROM {table} LIMIT 1"
                    result = data_manager.execute_query(query)
                    if result is not None and not result.empty:
                        count = int(result.iloc[0]['count'])
                        existing_tables.append({'table': table, 'count': count})
                        print(f"  ✅ {table}: {count} 条记录")
                    else:
                        missing_tables.append(table)
                        print(f"  ❌ {table}: 表不存在或无数据")
                except Exception as e:
                    missing_tables.append(table)
                    print(f"  ❌ {table}: 查询失败 - {e}")
            
            return {
                'success': len(missing_tables) == 0,
                'existing_tables': existing_tables,
                'missing_tables': missing_tables,
                'total_tables': len(required_tables)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def test_stock_data_quality(self) -> Dict[str, Any]:
        """测试股票数据质量"""
        print("\n📊 测试股票数据质量...")
        
        try:
            data_manager = UnifiedDataManager()
            
            # 获取股票列表
            stocks_query = "SELECT code, name FROM stock_info LIMIT 10"
            stocks_result = data_manager.execute_query(stocks_query)
            
            if stocks_result is None or stocks_result.empty:
                return {
                    'success': False,
                    'error': '无法获取股票列表'
                }
            
            # 测试样本股票数据
            sample_stocks = stocks_result['code'].head(5).tolist()
            stock_data_quality = []
            
            for stock_code in sample_stocks:
                try:
                    # 获取最近30天的数据
                    end_date = datetime.now().strftime('%Y-%m-%d')
                    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
                    
                    data_query = f"""
                    SELECT COUNT(*) as count, 
                           MIN(date) as min_date, 
                           MAX(date) as max_date,
                           AVG(close) as avg_close,
                           AVG(volume) as avg_volume
                    FROM stock_daily 
                    WHERE code = '{stock_code}' 
                    AND date >= '{start_date}'
                    AND date <= '{end_date}'
                    """
                    
                    result = data_manager.execute_query(data_query)
                    if result is not None and not result.empty:
                        row = result.iloc[0]
                        stock_data_quality.append({
                            'code': stock_code,
                            'count': int(row['count']),
                            'min_date': str(row['min_date']),
                            'max_date': str(row['max_date']),
                            'avg_close': float(row['avg_close']) if row['avg_close'] else 0,
                            'avg_volume': float(row['avg_volume']) if row['avg_volume'] else 0
                        })
                        print(f"  ✅ {stock_code}: {int(row['count'])} 条记录")
                    else:
                        print(f"  ❌ {stock_code}: 无数据")
                        
                except Exception as e:
                    print(f"  ❌ {stock_code}: 查询失败 - {e}")
            
            return {
                'success': len(stock_data_quality) > 0,
                'tested_stocks': len(sample_stocks),
                'valid_stocks': len(stock_data_quality),
                'stock_data_quality': stock_data_quality
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def test_indicator_calculation(self) -> Dict[str, Any]:
        """测试指标计算功能"""
        print("\n🧮 测试指标计算功能...")
        
        try:
            validator = ProductionIndicatorValidator()
            
            # 选择几个关键指标进行测试
            test_indicators = [
                'SMA', 'EMA', 'MACD', 'RSI', 'KDJ', 
                'BOLL', 'ATR', 'CCI', 'STOCH', 'ADX'
            ]
            
            indicator_results = []
            
            for indicator_name in test_indicators:
                try:
                    print(f"  🔍 测试 {indicator_name} 指标...")
                    
                    # 使用000001作为测试股票
                    test_config = {
                        'validation_date': '2024-01-01',
                        'stock_pool': ['000001'],
                        'stop_on_error': False,
                        'max_concurrent': 1
                    }
                    
                    start_time = time.time()
                    result = validator.validate_single_indicator(indicator_name, test_config)
                    duration = time.time() - start_time
                    
                    if result and result.get('status') == 'success':
                        indicator_results.append({
                            'indicator': indicator_name,
                            'success': True,
                            'duration': duration,
                            'selected_count': result.get('selected_count', 0)
                        })
                        print(f"    ✅ {indicator_name}: 成功 ({duration:.2f}s)")
                    else:
                        indicator_results.append({
                            'indicator': indicator_name,
                            'success': False,
                            'duration': duration,
                            'error': result.get('error_message', '未知错误')
                        })
                        print(f"    ❌ {indicator_name}: 失败 - {result.get('error_message', '未知错误')}")
                        
                except Exception as e:
                    indicator_results.append({
                        'indicator': indicator_name,
                        'success': False,
                        'error': str(e)
                    })
                    print(f"    ❌ {indicator_name}: 异常 - {e}")
            
            successful_indicators = [r for r in indicator_results if r['success']]
            
            return {
                'success': len(successful_indicators) > 0,
                'tested_indicators': len(test_indicators),
                'successful_indicators': len(successful_indicators),
                'success_rate': len(successful_indicators) / len(test_indicators) * 100,
                'indicator_results': indicator_results
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """运行全面的生产环境测试"""
        self.start_time = time.time()
        self.print_banner()
        
        comprehensive_results = {
            'start_time': datetime.now().isoformat(),
            'test_environment': {
                'python_version': sys.version,
                'platform': sys.platform,
                'working_directory': os.getcwd()
            }
        }
        
        # 1. 测试连接方式
        print("\n" + "="*50)
        print("阶段 1: 数据库连接测试")
        print("="*50)
        connection_results = self.test_connection_methods()
        comprehensive_results['connection_test'] = connection_results
        
        # 检查是否有任何连接成功
        successful_connections = [k for k, v in connection_results.items() if v.get('success')]
        
        if not successful_connections:
            print("\n❌ 所有连接方式都失败了，无法继续测试")
            comprehensive_results['overall_success'] = False
            comprehensive_results['failure_reason'] = '数据库连接失败'
            return comprehensive_results
        
        print(f"\n✅ 找到 {len(successful_connections)} 种有效连接方式: {successful_connections}")
        
        # 2. 测试数据库架构
        print("\n" + "="*50)
        print("阶段 2: 数据库架构测试")
        print("="*50)
        schema_results = self.test_database_schema()
        comprehensive_results['schema_test'] = schema_results
        
        # 3. 测试数据质量
        print("\n" + "="*50)
        print("阶段 3: 数据质量测试")
        print("="*50)
        data_quality_results = self.test_stock_data_quality()
        comprehensive_results['data_quality_test'] = data_quality_results
        
        # 4. 测试指标计算
        print("\n" + "="*50)
        print("阶段 4: 指标计算测试")
        print("="*50)
        indicator_results = self.test_indicator_calculation()
        comprehensive_results['indicator_test'] = indicator_results
        
        # 计算总体结果
        total_duration = time.time() - self.start_time
        comprehensive_results['end_time'] = datetime.now().isoformat()
        comprehensive_results['total_duration'] = total_duration
        
        # 评估整体成功率
        test_scores = [
            len(successful_connections) > 0,  # 连接测试
            schema_results.get('success', False),  # 架构测试
            data_quality_results.get('success', False),  # 数据质量测试
            indicator_results.get('success', False)  # 指标测试
        ]
        
        overall_success = sum(test_scores) >= 3  # 至少3个测试通过
        comprehensive_results['overall_success'] = overall_success
        comprehensive_results['success_score'] = sum(test_scores)
        comprehensive_results['max_score'] = len(test_scores)
        
        # 打印总结
        print("\n" + "="*80)
        print("🎯 生产环境测试总结")
        print("="*80)
        print(f"总耗时: {total_duration:.2f} 秒")
        print(f"成功分数: {sum(test_scores)}/{len(test_scores)}")
        print(f"整体结果: {'✅ 成功' if overall_success else '❌ 失败'}")
        
        if overall_success:
            print("\n🎉 生产环境测试通过！系统可以正常运行。")
        else:
            print("\n⚠️  生产环境测试未完全通过，请检查失败的测试项。")
        
        return comprehensive_results
    
    def save_test_results(self, results: Dict[str, Any]):
        """保存测试结果"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"production_test_{timestamp}.json"
            filepath = os.path.join(root_dir, 'results', filename)
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"\n📄 测试结果已保存到: {filepath}")
            
        except Exception as e:
            print(f"\n❌ 保存测试结果失败: {e}")


def main():
    """主函数"""
    try:
        tester = ProductionDatabaseTester()
        results = tester.run_comprehensive_test()
        tester.save_test_results(results)
        
        # 根据测试结果设置退出码
        if results.get('overall_success'):
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  测试被用户中断")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ 测试过程中发生异常: {e}")
        logger.exception("生产环境测试异常")
        sys.exit(1)


if __name__ == "__main__":
    main() 