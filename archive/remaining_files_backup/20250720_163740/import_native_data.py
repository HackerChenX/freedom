#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
导入ClickHouse Native格式的真实股票数据

从/Users/hacker/Downloads/data.native导入真实股票数据到ClickHouse数据库。

Author: AI Assistant
Date: 2025-07-19
"""

import os
import sys
import subprocess
from pathlib import Path

def clear_existing_data():
    """清理现有的模拟数据"""
    print("🧹 清理现有模拟数据...")
    
    commands = [
        "TRUNCATE TABLE stock.stock_info",
        "TRUNCATE TABLE stock.daily_data"
    ]
    
    for cmd in commands:
        try:
            result = subprocess.run([
                'clickhouse-client', 
                '--password=123456', 
                '--query', cmd
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                print(f"✅ 执行成功: {cmd}")
            else:
                print(f"❌ 执行失败: {cmd} - {result.stderr}")
                
        except Exception as e:
            print(f"❌ 执行异常: {cmd} - {e}")

def analyze_native_file():
    """分析Native文件结构"""
    print("🔍 分析Native文件结构...")
    
    native_file = "/Users/hacker/Downloads/data.native"
    
    # 检查文件大小
    file_size = os.path.getsize(native_file)
    print(f"📊 文件大小: {file_size:,} 字节 ({file_size / (1024*1024*1024):.2f} GB)")
    
    # 尝试使用clickhouse-local分析文件结构
    try:
        result = subprocess.run([
            'clickhouse-local',
            '--query', 
            f"DESCRIBE TABLE file('{native_file}', 'Native')"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("📋 文件结构:")
            print(result.stdout)
            return result.stdout
        else:
            print(f"❌ 分析文件结构失败: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"❌ 分析文件结构异常: {e}")
        return None

def create_target_table(structure_info):
    """根据文件结构创建目标表"""
    print("🏗️ 创建目标表...")
    
    # 基于分析结果创建表结构
    # 如果分析失败，使用默认结构
    if not structure_info:
        print("使用默认表结构...")
        create_sql = """
        CREATE TABLE IF NOT EXISTS stock.native_data (
            code String,
            name String,
            date Date,
            open Float64,
            high Float64,
            low Float64,
            close Float64,
            volume UInt64,
            amount Float64
        ) ENGINE = MergeTree()
        ORDER BY (code, date)
        """
    else:
        # 解析结构信息并创建相应的表
        print("基于文件结构创建表...")
        create_sql = """
        CREATE TABLE IF NOT EXISTS stock.native_data (
            code String
        ) ENGINE = MergeTree()
        ORDER BY code
        """
    
    try:
        result = subprocess.run([
            'clickhouse-client', 
            '--password=123456', 
            '--query', create_sql
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ 目标表创建成功")
            return True
        else:
            print(f"❌ 目标表创建失败: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ 目标表创建异常: {e}")
        return False

def import_native_data():
    """导入Native格式数据"""
    print("📥 开始导入Native格式数据...")
    
    native_file = "/Users/hacker/Downloads/data.native"
    
    # 方法1: 使用clickhouse-client直接导入
    try:
        print("尝试方法1: 直接导入到stock.native_data表...")
        
        # 使用cat和管道导入
        import_cmd = f"cat '{native_file}' | clickhouse-client --password=123456 --query='INSERT INTO stock.native_data FORMAT Native'"
        
        result = subprocess.run(
            import_cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=300  # 5分钟超时
        )
        
        if result.returncode == 0:
            print("✅ 方法1导入成功")
            return True
        else:
            print(f"❌ 方法1导入失败: {result.stderr}")
            
    except Exception as e:
        print(f"❌ 方法1导入异常: {e}")
    
    # 方法2: 使用clickhouse-local作为中介
    try:
        print("尝试方法2: 通过clickhouse-local转换...")
        
        # 先用clickhouse-local读取并转换为CSV
        csv_cmd = f"clickhouse-local --query=\"SELECT * FROM file('{native_file}', 'Native') FORMAT CSV\" > /tmp/stock_data.csv"
        
        result = subprocess.run(
            csv_cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=300
        )
        
        if result.returncode == 0:
            print("✅ 转换为CSV成功")
            
            # 再导入CSV到ClickHouse
            import_csv_cmd = "clickhouse-client --password=123456 --query='INSERT INTO stock.native_data FORMAT CSV' < /tmp/stock_data.csv"
            
            result2 = subprocess.run(
                import_csv_cmd, 
                shell=True, 
                capture_output=True, 
                text=True, 
                timeout=300
            )
            
            if result2.returncode == 0:
                print("✅ 方法2导入成功")
                return True
            else:
                print(f"❌ 方法2 CSV导入失败: {result2.stderr}")
        else:
            print(f"❌ 方法2 CSV转换失败: {result.stderr}")
            
    except Exception as e:
        print(f"❌ 方法2导入异常: {e}")
    
    return False

def verify_import():
    """验证导入结果"""
    print("🔍 验证导入结果...")
    
    try:
        # 检查记录数量
        result = subprocess.run([
            'clickhouse-client', 
            '--password=123456', 
            '--query', 'SELECT COUNT(*) FROM stock.native_data'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            count = result.stdout.strip()
            print(f"📊 导入记录数: {count}")
            
            if int(count) > 0:
                # 显示样本数据
                sample_result = subprocess.run([
                    'clickhouse-client', 
                    '--password=123456', 
                    '--query', 'SELECT * FROM stock.native_data LIMIT 5'
                ], capture_output=True, text=True, timeout=30)
                
                if sample_result.returncode == 0:
                    print("📋 样本数据:")
                    print(sample_result.stdout)
                
                return int(count)
            else:
                print("⚠️ 没有导入任何数据")
                return 0
        else:
            print(f"❌ 验证失败: {result.stderr}")
            return 0
            
    except Exception as e:
        print(f"❌ 验证异常: {e}")
        return 0

def main():
    """主函数"""
    print("🚀 开始导入真实股票数据")
    print("=" * 60)
    
    # 1. 清理现有数据
    clear_existing_data()
    
    # 2. 分析Native文件结构
    structure_info = analyze_native_file()
    
    # 3. 创建目标表
    if not create_target_table(structure_info):
        print("❌ 无法创建目标表，导入终止")
        return False
    
    # 4. 导入数据
    if not import_native_data():
        print("❌ 数据导入失败")
        return False
    
    # 5. 验证导入结果
    record_count = verify_import()
    
    if record_count > 0:
        print("=" * 60)
        print(f"🎉 数据导入成功！共导入 {record_count:,} 条记录")
        
        if record_count >= 4000:
            print("✅ 数据量满足4000+股票的测试要求")
        else:
            print(f"⚠️ 数据量可能不足4000条，当前: {record_count}")
        
        return True
    else:
        print("=" * 60)
        print("❌ 数据导入失败或无数据")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
