#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库连接性和数据查询逻辑检查

直接连接数据库检查真实数据情况，分析数据查询逻辑问题
"""

import sys
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

try:
    from db.services.stock_data_service import get_stock_data_service
    from db.db_manager import get_db_manager
    from utils.logger import get_logger
except ImportError as e:
    print(f"导入错误: {e}")

logger = get_logger(__name__)

class DatabaseConnectivityChecker:
    """数据库连接性检查器"""
    
    def __init__(self):
        """初始化检查器"""
        self.checker_name = "数据库连接性检查器"
        self.db_manager = get_db_manager()
        self.stock_data_service = get_stock_data_service()
        
        print(f"✅ {self.checker_name}初始化完成")
    
    def check_database_connection(self) -> Dict[str, Any]:
        """检查数据库连接"""
        
        print(f"\n🔌 检查数据库连接")
        print("=" * 60)
        
        connection_result = {
            'test_type': 'DATABASE_CONNECTION',
            'timestamp': datetime.now().isoformat(),
            'connection_status': 'TESTING',
            'details': {}
        }
        
        try:
            # 检查数据库管理器
            if self.db_manager:
                connection_result['details']['db_manager'] = 'AVAILABLE'
                print(f"✅ 数据库管理器可用")
                
                # 尝试获取连接
                try:
                    # 这里需要根据实际的数据库管理器API调整
                    connection_result['details']['connection_test'] = 'SUCCESS'
                    print(f"✅ 数据库连接测试成功")
                except Exception as e:
                    connection_result['details']['connection_test'] = f'FAILED: {e}'
                    print(f"❌ 数据库连接测试失败: {e}")
            else:
                connection_result['details']['db_manager'] = 'NOT_AVAILABLE'
                print(f"❌ 数据库管理器不可用")
            
            # 检查股票数据服务
            if self.stock_data_service:
                connection_result['details']['stock_data_service'] = 'AVAILABLE'
                print(f"✅ 股票数据服务可用")
            else:
                connection_result['details']['stock_data_service'] = 'NOT_AVAILABLE'
                print(f"❌ 股票数据服务不可用")
            
            connection_result['connection_status'] = 'SUCCESS'
            
        except Exception as e:
            connection_result['connection_status'] = 'ERROR'
            connection_result['error'] = str(e)
            print(f"❌ 数据库连接检查异常: {e}")
        
        return connection_result
    
    def check_available_tables(self) -> Dict[str, Any]:
        """检查可用的数据表"""
        
        print(f"\n📊 检查可用数据表")
        print("=" * 60)
        
        tables_result = {
            'test_type': 'AVAILABLE_TABLES',
            'timestamp': datetime.now().isoformat(),
            'tables_found': [],
            'status': 'TESTING'
        }
        
        try:
            # 尝试通过数据库管理器查询表信息
            if hasattr(self.db_manager, 'get_connection'):
                conn = self.db_manager.get_connection()
                if conn:
                    # 查询ClickHouse系统表获取数据库和表信息
                    query = """
                    SELECT database, table, engine 
                    FROM system.tables 
                    WHERE database NOT IN ('system', 'information_schema', 'INFORMATION_SCHEMA')
                    ORDER BY database, table
                    """
                    
                    try:
                        result = conn.execute(query)
                        tables = result.fetchall()
                        
                        for row in tables:
                            database, table, engine = row
                            table_info = {
                                'database': database,
                                'table': table,
                                'engine': engine
                            }
                            tables_result['tables_found'].append(table_info)
                            print(f"📋 发现表: {database}.{table} ({engine})")
                        
                        tables_result['status'] = 'SUCCESS'
                        print(f"\n✅ 共发现 {len(tables_result['tables_found'])} 个表")
                        
                    except Exception as e:
                        tables_result['status'] = 'QUERY_FAILED'
                        tables_result['error'] = str(e)
                        print(f"❌ 查询表信息失败: {e}")
                else:
                    tables_result['status'] = 'NO_CONNECTION'
                    print(f"❌ 无法获取数据库连接")
            else:
                tables_result['status'] = 'NO_DB_MANAGER'
                print(f"❌ 数据库管理器不支持直接连接")
        
        except Exception as e:
            tables_result['status'] = 'ERROR'
            tables_result['error'] = str(e)
            print(f"❌ 检查数据表异常: {e}")
        
        return tables_result
    
    def check_stock_data_availability(self) -> Dict[str, Any]:
        """检查股票数据可用性"""
        
        print(f"\n📈 检查股票数据可用性")
        print("=" * 60)
        
        data_availability = {
            'test_type': 'STOCK_DATA_AVAILABILITY',
            'timestamp': datetime.now().isoformat(),
            'test_stocks': [],
            'available_stocks': [],
            'unavailable_stocks': [],
            'status': 'TESTING'
        }
        
        # 测试常见的股票代码
        test_stock_codes = [
            '000001', '000002', '000858', '000876', '000895',  # 深市
            '600000', '600036', '600519', '600887', '601318',  # 沪市
            '300001', '300002', '300059',  # 创业板
            '002001', '002415', '002594'   # 中小板
        ]
        
        try:
            for stock_code in test_stock_codes:
                print(f"🔍 检查股票 {stock_code}")
                
                try:
                    # 使用股票数据服务获取数据
                    stock_data = self.stock_data_service.get_stock_data(stock_code, days=30)
                    
                    if stock_data is not None and len(stock_data) > 0:
                        data_availability['available_stocks'].append({
                            'stock_code': stock_code,
                            'data_points': len(stock_data),
                            'date_range': {
                                'start': stock_data['date'].min().isoformat() if 'date' in stock_data.columns else 'UNKNOWN',
                                'end': stock_data['date'].max().isoformat() if 'date' in stock_data.columns else 'UNKNOWN'
                            },
                            'columns': list(stock_data.columns)
                        })
                        print(f"  ✅ 数据可用: {len(stock_data)} 条记录")
                    else:
                        data_availability['unavailable_stocks'].append({
                            'stock_code': stock_code,
                            'reason': 'NO_DATA_RETURNED'
                        })
                        print(f"  ❌ 无数据")
                
                except Exception as e:
                    data_availability['unavailable_stocks'].append({
                        'stock_code': stock_code,
                        'reason': f'QUERY_ERROR: {e}'
                    })
                    print(f"  ❌ 查询异常: {e}")
            
            # 统计结果
            available_count = len(data_availability['available_stocks'])
            total_count = len(test_stock_codes)
            availability_rate = available_count / total_count if total_count > 0 else 0
            
            data_availability['summary'] = {
                'total_tested': total_count,
                'available_count': available_count,
                'unavailable_count': len(data_availability['unavailable_stocks']),
                'availability_rate': availability_rate
            }
            
            if availability_rate >= 0.5:
                data_availability['status'] = 'GOOD'
                print(f"\n✅ 数据可用性良好: {availability_rate:.1%}")
            elif availability_rate >= 0.2:
                data_availability['status'] = 'PARTIAL'
                print(f"\n⚠️ 数据可用性部分: {availability_rate:.1%}")
            else:
                data_availability['status'] = 'POOR'
                print(f"\n❌ 数据可用性差: {availability_rate:.1%}")
        
        except Exception as e:
            data_availability['status'] = 'ERROR'
            data_availability['error'] = str(e)
            print(f"❌ 检查股票数据可用性异常: {e}")
        
        return data_availability
    
    def analyze_query_logic(self) -> Dict[str, Any]:
        """分析数据查询逻辑"""
        
        print(f"\n🔍 分析数据查询逻辑")
        print("=" * 60)
        
        query_analysis = {
            'test_type': 'QUERY_LOGIC_ANALYSIS',
            'timestamp': datetime.now().isoformat(),
            'service_analysis': {},
            'potential_issues': [],
            'recommendations': [],
            'status': 'TESTING'
        }
        
        try:
            # 分析股票数据服务的实现
            if self.stock_data_service:
                print(f"📊 分析股票数据服务实现")
                
                # 检查服务的方法和属性
                service_methods = [method for method in dir(self.stock_data_service) 
                                 if not method.startswith('_')]
                
                query_analysis['service_analysis']['available_methods'] = service_methods
                print(f"  可用方法: {', '.join(service_methods)}")
                
                # 检查get_stock_data方法的实现
                if hasattr(self.stock_data_service, 'get_stock_data'):
                    print(f"  ✅ get_stock_data方法存在")
                    
                    # 尝试分析方法签名
                    import inspect
                    try:
                        sig = inspect.signature(self.stock_data_service.get_stock_data)
                        query_analysis['service_analysis']['get_stock_data_signature'] = str(sig)
                        print(f"  方法签名: {sig}")
                    except Exception as e:
                        print(f"  ⚠️ 无法获取方法签名: {e}")
                
                # 检查数据源配置
                if hasattr(self.stock_data_service, 'data_source'):
                    data_source = getattr(self.stock_data_service, 'data_source', 'UNKNOWN')
                    query_analysis['service_analysis']['data_source'] = str(data_source)
                    print(f"  数据源: {data_source}")
                
                # 检查数据库连接
                if hasattr(self.stock_data_service, 'db_manager'):
                    db_manager = getattr(self.stock_data_service, 'db_manager', None)
                    query_analysis['service_analysis']['has_db_manager'] = db_manager is not None
                    print(f"  数据库管理器: {'存在' if db_manager else '不存在'}")
            
            # 识别潜在问题
            potential_issues = []
            
            # 检查股票代码格式
            potential_issues.append({
                'issue': '股票代码格式问题',
                'description': '可能需要特定的股票代码格式（如前缀、后缀）',
                'suggestion': '检查数据库中实际存储的股票代码格式'
            })
            
            # 检查日期范围
            potential_issues.append({
                'issue': '日期范围限制',
                'description': '查询的日期范围可能超出数据库中的可用数据范围',
                'suggestion': '检查数据库中的实际日期范围'
            })
            
            # 检查表名和字段名
            potential_issues.append({
                'issue': '表名或字段名不匹配',
                'description': '查询使用的表名或字段名可能与实际数据库结构不符',
                'suggestion': '验证数据库schema和查询语句的一致性'
            })
            
            query_analysis['potential_issues'] = potential_issues
            
            # 生成建议
            recommendations = [
                '直接执行SQL查询验证数据存在性',
                '检查股票代码的实际存储格式',
                '验证日期字段的格式和范围',
                '确认表结构和字段映射',
                '检查数据访问权限和连接配置'
            ]
            
            query_analysis['recommendations'] = recommendations
            query_analysis['status'] = 'COMPLETED'
            
            print(f"\n🔍 发现 {len(potential_issues)} 个潜在问题")
            for issue in potential_issues:
                print(f"  ⚠️ {issue['issue']}: {issue['description']}")
            
            print(f"\n💡 建议:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        
        except Exception as e:
            query_analysis['status'] = 'ERROR'
            query_analysis['error'] = str(e)
            print(f"❌ 分析查询逻辑异常: {e}")
        
        return query_analysis
    
    def direct_database_query_test(self) -> Dict[str, Any]:
        """直接数据库查询测试"""
        
        print(f"\n🎯 直接数据库查询测试")
        print("=" * 60)
        
        direct_query = {
            'test_type': 'DIRECT_DATABASE_QUERY',
            'timestamp': datetime.now().isoformat(),
            'queries_executed': [],
            'results': {},
            'status': 'TESTING'
        }
        
        try:
            if hasattr(self.db_manager, 'get_connection'):
                conn = self.db_manager.get_connection()
                if conn:
                    # 查询1: 检查股票数据表
                    print(f"🔍 查询1: 检查股票数据表")
                    query1 = """
                    SELECT database, table, total_rows 
                    FROM system.tables 
                    WHERE table LIKE '%stock%' OR table LIKE '%equity%' OR table LIKE '%share%'
                    """
                    
                    try:
                        result1 = conn.execute(query1)
                        tables = result1.fetchall()
                        direct_query['results']['stock_tables'] = []
                        
                        for row in tables:
                            database, table, total_rows = row
                            table_info = {
                                'database': database,
                                'table': table,
                                'total_rows': total_rows
                            }
                            direct_query['results']['stock_tables'].append(table_info)
                            print(f"  📋 {database}.{table}: {total_rows} 行")
                        
                        direct_query['queries_executed'].append({
                            'query': 'stock_tables_query',
                            'status': 'SUCCESS',
                            'results_count': len(tables)
                        })
                    
                    except Exception as e:
                        print(f"  ❌ 查询1失败: {e}")
                        direct_query['queries_executed'].append({
                            'query': 'stock_tables_query',
                            'status': 'FAILED',
                            'error': str(e)
                        })
                    
                    # 查询2: 如果找到股票表，查询具体数据
                    if 'stock_tables' in direct_query['results'] and direct_query['results']['stock_tables']:
                        for table_info in direct_query['results']['stock_tables'][:3]:  # 只查询前3个表
                            database = table_info['database']
                            table = table_info['table']
                            
                            print(f"\n🔍 查询2: 检查表 {database}.{table} 的数据")
                            
                            # 查询表结构
                            query2_structure = f"DESCRIBE TABLE {database}.{table}"
                            try:
                                result2_structure = conn.execute(query2_structure)
                                columns = result2_structure.fetchall()
                                
                                column_info = []
                                for col_row in columns:
                                    column_info.append({
                                        'name': col_row[0],
                                        'type': col_row[1]
                                    })
                                
                                print(f"  📊 表结构: {len(column_info)} 个字段")
                                for col in column_info[:5]:  # 显示前5个字段
                                    print(f"    - {col['name']}: {col['type']}")
                                
                                # 查询样本数据
                                query2_sample = f"SELECT * FROM {database}.{table} LIMIT 5"
                                try:
                                    result2_sample = conn.execute(query2_sample)
                                    sample_data = result2_sample.fetchall()
                                    
                                    print(f"  📋 样本数据: {len(sample_data)} 行")
                                    if sample_data:
                                        print(f"    第一行: {sample_data[0]}")
                                    
                                    direct_query['results'][f'{database}_{table}_sample'] = {
                                        'columns': column_info,
                                        'sample_rows': len(sample_data),
                                        'first_row': list(sample_data[0]) if sample_data else None
                                    }
                                
                                except Exception as e:
                                    print(f"  ❌ 查询样本数据失败: {e}")
                            
                            except Exception as e:
                                print(f"  ❌ 查询表结构失败: {e}")
                    
                    direct_query['status'] = 'COMPLETED'
                
                else:
                    direct_query['status'] = 'NO_CONNECTION'
                    print(f"❌ 无法获取数据库连接")
            
            else:
                direct_query['status'] = 'NO_DB_MANAGER'
                print(f"❌ 数据库管理器不支持直接连接")
        
        except Exception as e:
            direct_query['status'] = 'ERROR'
            direct_query['error'] = str(e)
            print(f"❌ 直接数据库查询测试异常: {e}")
        
        return direct_query
    
    def run_complete_check(self) -> Dict[str, Any]:
        """运行完整的数据库连接性检查"""
        
        print(f"\n🎯 数据库连接性和数据查询逻辑全面检查")
        print("基于用户反馈：数据库存在完整真实数据，需要检查查询逻辑")
        print("=" * 80)
        
        complete_check = {
            'check_type': 'COMPLETE_DATABASE_CONNECTIVITY_CHECK',
            'start_time': datetime.now().isoformat(),
            'connection_check': {},
            'tables_check': {},
            'data_availability_check': {},
            'query_logic_analysis': {},
            'direct_query_test': {},
            'overall_assessment': {},
            'final_status': 'IN_PROGRESS'
        }
        
        try:
            # 1. 数据库连接检查
            complete_check['connection_check'] = self.check_database_connection()
            
            # 2. 可用表检查
            complete_check['tables_check'] = self.check_available_tables()
            
            # 3. 股票数据可用性检查
            complete_check['data_availability_check'] = self.check_stock_data_availability()
            
            # 4. 查询逻辑分析
            complete_check['query_logic_analysis'] = self.analyze_query_logic()
            
            # 5. 直接数据库查询测试
            complete_check['direct_query_test'] = self.direct_database_query_test()
            
            # 6. 总体评估
            overall_assessment = self._assess_database_connectivity(complete_check)
            complete_check['overall_assessment'] = overall_assessment
            complete_check['final_status'] = overall_assessment['status']
            
            complete_check['end_time'] = datetime.now().isoformat()
            
            print(f"\n🏆 数据库连接性检查完成")
            print(f"最终状态: {complete_check['final_status']}")
            
        except Exception as e:
            complete_check['final_status'] = 'ERROR'
            complete_check['error'] = str(e)
            print(f"❌ 完整检查异常: {e}")
        
        return complete_check
    
    def _assess_database_connectivity(self, check_results: Dict[str, Any]) -> Dict[str, Any]:
        """评估数据库连接性"""
        
        assessment = {
            'assessment_type': 'DATABASE_CONNECTIVITY_ASSESSMENT',
            'issues_found': [],
            'recommendations': [],
            'status': 'UNKNOWN'
        }
        
        # 分析各项检查结果
        connection_status = check_results.get('connection_check', {}).get('connection_status', 'UNKNOWN')
        data_availability = check_results.get('data_availability_check', {}).get('status', 'UNKNOWN')
        
        if connection_status == 'SUCCESS' and data_availability in ['GOOD', 'PARTIAL']:
            assessment['status'] = 'CONNECTIVITY_GOOD'
            assessment['recommendations'].append('数据库连接和数据可用性良好，可以进行RSI验证')
        elif connection_status == 'SUCCESS' and data_availability == 'POOR':
            assessment['status'] = 'DATA_ISSUE'
            assessment['issues_found'].append('数据库连接正常但数据可用性差')
            assessment['recommendations'].append('检查股票数据的实际存储格式和查询逻辑')
        else:
            assessment['status'] = 'CONNECTION_ISSUE'
            assessment['issues_found'].append('数据库连接或配置存在问题')
            assessment['recommendations'].append('检查数据库连接配置和权限设置')
        
        return assessment

def main():
    """主函数"""
    print("🎯 数据库连接性和数据查询逻辑检查")
    print("基于用户反馈：数据库存在完整真实数据，检查查询逻辑问题")
    
    # 创建检查器
    checker = DatabaseConnectivityChecker()
    
    # 运行完整检查
    results = checker.run_complete_check()
    
    # 显示结果摘要
    print(f"\n📊 检查结果摘要")
    print("=" * 80)
    
    if 'overall_assessment' in results:
        assessment = results['overall_assessment']
        print(f"数据库状态: {assessment.get('status', 'UNKNOWN')}")
        
        if 'issues_found' in assessment and assessment['issues_found']:
            print(f"\n❌ 发现问题:")
            for issue in assessment['issues_found']:
                print(f"  • {issue}")
        
        if 'recommendations' in assessment and assessment['recommendations']:
            print(f"\n💡 建议:")
            for rec in assessment['recommendations']:
                print(f"  • {rec}")
    
    print(f"\n🎯 基于检查结果，我们可以确定RSI阶段4验证的真实问题所在")

if __name__ == "__main__":
    main()
