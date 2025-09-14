#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据库连接测试脚本
用于诊断和修复数据库连接问题
"""

import os
import sys
import time
from pathlib import Path

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

def test_clickhouse_client():
    """测试ClickHouse客户端连接"""
    print("🔍 测试ClickHouse客户端连接...")
    
    try:
        import clickhouse_connect
        
        # 直接连接测试
        client = clickhouse_connect.get_client(
            host='localhost',
            port=9000,
            username='default',
            password='123456',
            database='stock'
        )
        
        # 执行简单查询
        result = client.query('SELECT 1 as test')
        print(f"✅ 直接连接成功: {result.result_rows}")
        
        # 测试数据库查询
        result = client.query('SELECT COUNT(*) as count FROM stock_info LIMIT 1')
        print(f"✅ 数据库查询成功: {result.result_rows}")
        
        client.close()
        return True
        
    except Exception as e:
        print(f"❌ ClickHouse客户端连接失败: {e}")
        return False

def test_enhanced_connection_pool():
    """测试增强连接池"""
    print("\n🔍 测试增强连接池...")
    
    try:
        from db.enhanced_connection_pool import ClickHouseConnectionPool
        
        pool = ClickHouseConnectionPool()
        
        # 测试基本查询
        result = pool.query_dataframe('SELECT 1 as test')
        print(f"✅ 连接池查询成功: {len(result)} 行")
        
        # 测试数据库查询
        result = pool.query_dataframe('SELECT COUNT(*) as count FROM stock_info LIMIT 1')
        print(f"✅ 数据库查询成功: {len(result)} 行")
        
        return True
        
    except Exception as e:
        print(f"❌ 增强连接池测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_buypoint_analyzer():
    """测试买点分析器"""
    print("\n🔍 测试买点分析器...")
    
    try:
        import subprocess
        
        # 运行简单的买点分析
        cmd = [
            'python3', 'bin/buypoint_batch_analyzer.py',
            '--stock-code', '300005',
            '--date', '2025-05-09',
            '--analysis-type', 'simple',
            '--output', 'results/test_connection.json'
        ]
        
        result = subprocess.run(
            cmd,
            cwd=root_dir,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("✅ 买点分析器运行成功")
            return True
        else:
            print(f"❌ 买点分析器运行失败: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ 买点分析器测试失败: {e}")
        return False

def fix_database_connection():
    """修复数据库连接问题"""
    print("\n🔧 尝试修复数据库连接问题...")
    
    # 1. 检查配置文件
    config_file = Path(root_dir) / "config" / "database.yaml"
    if not config_file.exists():
        print("❌ 数据库配置文件不存在")
        return False
    
    # 2. 重启ClickHouse容器
    try:
        import subprocess
        
        print("🔄 重启ClickHouse容器...")
        subprocess.run(['docker', 'restart', 'freedom-clickhouse'], check=True)
        
        # 等待容器启动
        time.sleep(10)
        
        print("✅ ClickHouse容器重启完成")
        return True
        
    except Exception as e:
        print(f"❌ 重启ClickHouse容器失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 开始数据库连接诊断...")
    print("="*60)
    
    # 测试1: ClickHouse客户端连接
    client_ok = test_clickhouse_client()
    
    # 测试2: 增强连接池
    pool_ok = test_enhanced_connection_pool()
    
    # 测试3: 买点分析器
    analyzer_ok = test_buypoint_analyzer()
    
    print("\n" + "="*60)
    print("📊 诊断结果总结:")
    print(f"ClickHouse客户端: {'✅ 正常' if client_ok else '❌ 异常'}")
    print(f"增强连接池: {'✅ 正常' if pool_ok else '❌ 异常'}")
    print(f"买点分析器: {'✅ 正常' if analyzer_ok else '❌ 异常'}")
    
    # 如果有问题，尝试修复
    if not all([client_ok, pool_ok, analyzer_ok]):
        print("\n🔧 检测到问题，尝试修复...")
        fix_ok = fix_database_connection()
        
        if fix_ok:
            print("\n🔄 重新测试...")
            # 重新测试连接池
            pool_ok_retry = test_enhanced_connection_pool()
            analyzer_ok_retry = test_buypoint_analyzer()
            
            print("\n📊 修复后结果:")
            print(f"增强连接池: {'✅ 正常' if pool_ok_retry else '❌ 仍有问题'}")
            print(f"买点分析器: {'✅ 正常' if analyzer_ok_retry else '❌ 仍有问题'}")
            
            return all([client_ok, pool_ok_retry, analyzer_ok_retry])
    
    return all([client_ok, pool_ok, analyzer_ok])

if __name__ == "__main__":
    success = main()
    print("\n" + "="*60)
    if success:
        print("🎉 数据库连接诊断完成 - 所有测试通过！")
        exit(0)
    else:
        print("❌ 数据库连接诊断完成 - 仍有问题需要解决")
        exit(1)
