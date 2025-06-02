#!/usr/bin/env python3
"""
性能优化脚本

用于定期清理缓存、监控系统性能指标和优化数据库
"""

import os
import sys
import time
import argparse
import json
import psutil
import pandas as pd
from datetime import datetime, timedelta
import threading
import logging

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from db.data_manager import DataManager
from strategy.strategy_executor import StrategyExecutor
from utils.logger import get_logger, setup_logger
from utils.path_utils import get_log_dir, get_cache_dir
from utils.exceptions import DataAccessError

logger = get_logger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="性能优化工具")
    
    # 基础参数
    parser.add_argument("--clear-cache", "-c", action="store_true", help="清理所有缓存")
    parser.add_argument("--optimize-db", "-d", action="store_true", help="优化数据库")
    parser.add_argument("--monitor", "-m", action="store_true", help="监控系统性能")
    parser.add_argument("--monitor-time", "-t", type=int, default=60, help="监控时间（秒）")
    parser.add_argument("--log-level", help="日志级别 (DEBUG, INFO, WARNING, ERROR)", default="INFO")
    parser.add_argument("--archive-logs", "-a", action="store_true", help="归档旧日志文件")
    parser.add_argument("--days", type=int, default=30, help="清理多少天前的数据")
    
    args = parser.parse_args()
    
    # 如果没有指定任何操作，显示帮助信息
    if not (args.clear_cache or args.optimize_db or args.monitor or args.archive_logs):
        parser.print_help()
        sys.exit(0)
        
    return args


def clear_cache(days=30):
    """
    清理系统缓存
    
    Args:
        days: 清理多少天前的缓存
    """
    try:
        print("开始清理缓存...")
        
        # 1. 清理数据管理器缓存
        data_manager = DataManager()
        data_manager.clear_cache()
        
        # 2. 清理策略执行器缓存
        strategy_executor = StrategyExecutor()
        strategy_executor.clear_cache()
        
        # 3. 清理文件缓存
        cache_dir = get_cache_dir()
        if os.path.exists(cache_dir):
            cleared_count = 0
            total_size = 0
            
            # 计算截止日期
            cutoff_date = datetime.now() - timedelta(days=days)
            
            for root, dirs, files in os.walk(cache_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    
                    # 获取文件修改时间
                    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    
                    # 检查文件是否过期
                    if file_time < cutoff_date:
                        # 获取文件大小
                        file_size = os.path.getsize(file_path)
                        total_size += file_size
                        
                        # 删除文件
                        os.remove(file_path)
                        cleared_count += 1
                        
            print(f"已清理 {cleared_count} 个缓存文件，释放空间: {total_size / (1024*1024):.2f} MB")
        
        print("缓存清理完成")
        return True
    except Exception as e:
        logger.error(f"清理缓存失败: {e}")
        print(f"清理缓存失败: {e}")
        return False


def optimize_database():
    """优化数据库"""
    try:
        print("开始优化数据库...")
        
        # 获取数据库连接
        data_manager = DataManager()
        db = data_manager.db
        
        # 执行优化操作
        optimize_queries = [
            "OPTIMIZE TABLE stock_info FINAL",
            "OPTIMIZE TABLE stock_daily FINAL",
            "OPTIMIZE TABLE stock_selection_result FINAL"
        ]
        
        for query in optimize_queries:
            print(f"执行: {query}")
            try:
                db.execute(query)
            except Exception as e:
                logger.warning(f"执行优化查询 {query} 失败: {e}")
        
        # 清理过期数据
        cleanup_queries = [
            # 这里可以添加清理过期数据的SQL
        ]
        
        for query in cleanup_queries:
            if query.strip():
                print(f"执行: {query}")
                try:
                    rows_affected = db.execute(query)
                    print(f"受影响的行数: {rows_affected}")
                except Exception as e:
                    logger.warning(f"执行清理查询 {query} 失败: {e}")
        
        print("数据库优化完成")
        return True
    except Exception as e:
        logger.error(f"优化数据库失败: {e}")
        print(f"优化数据库失败: {e}")
        return False


def monitor_system(duration=60, interval=1):
    """
    监控系统性能
    
    Args:
        duration: 监控持续时间（秒）
        interval: 采样间隔（秒）
    """
    try:
        print(f"开始监控系统性能，持续 {duration} 秒...")
        
        # 准备数据存储结构
        timestamps = []
        cpu_usages = []
        memory_usages = []
        disk_io_reads = []
        disk_io_writes = []
        
        # 初始磁盘IO计数
        disk_io_start = psutil.disk_io_counters()
        
        # 监控循环
        start_time = time.time()
        while time.time() - start_time < duration:
            # 记录时间戳
            current_time = datetime.now()
            timestamps.append(current_time)
            
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_usages.append(cpu_percent)
            
            # 内存使用率
            memory = psutil.virtual_memory()
            memory_usages.append(memory.percent)
            
            # 磁盘IO
            disk_io = psutil.disk_io_counters()
            disk_io_reads.append(disk_io.read_bytes)
            disk_io_writes.append(disk_io.write_bytes)
            
            # 输出当前状态
            print(f"\r时间: {current_time.strftime('%H:%M:%S')} "
                  f"CPU: {cpu_percent}% "
                  f"内存: {memory.percent}% "
                  f"磁盘读: {disk_io.read_bytes / (1024*1024):.2f}MB "
                  f"磁盘写: {disk_io.write_bytes / (1024*1024):.2f}MB", 
                  end='')
            
            # 间隔
            time.sleep(interval)
        
        print("\n监控完成")
        
        # 计算IO变化
        for i in range(1, len(disk_io_reads)):
            disk_io_reads[i] = disk_io_reads[i] - disk_io_reads[i-1]
            disk_io_writes[i] = disk_io_writes[i] - disk_io_writes[i-1]
        disk_io_reads[0] = 0
        disk_io_writes[0] = 0
        
        # 创建性能报告
        df = pd.DataFrame({
            'timestamp': timestamps,
            'cpu_percent': cpu_usages,
            'memory_percent': memory_usages,
            'disk_read_bytes': disk_io_reads,
            'disk_write_bytes': disk_io_writes
        })
        
        # 保存报告
        report_path = os.path.join(get_log_dir(), f"performance_report_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv")
        df.to_csv(report_path, index=False)
        
        # 输出摘要
        print("\n性能监控摘要:")
        print(f"CPU使用率:    平均: {df['cpu_percent'].mean():.2f}%  最大: {df['cpu_percent'].max():.2f}%")
        print(f"内存使用率:   平均: {df['memory_percent'].mean():.2f}%  最大: {df['memory_percent'].max():.2f}%")
        print(f"磁盘读取速度: 平均: {df['disk_read_bytes'].mean() / (1024*1024):.2f}MB/s  最大: {df['disk_read_bytes'].max() / (1024*1024):.2f}MB/s")
        print(f"磁盘写入速度: 平均: {df['disk_write_bytes'].mean() / (1024*1024):.2f}MB/s  最大: {df['disk_write_bytes'].max() / (1024*1024):.2f}MB/s")
        print(f"性能报告已保存至: {report_path}")
        
        return True
    except Exception as e:
        logger.error(f"监控系统性能失败: {e}")
        print(f"监控系统性能失败: {e}")
        return False


def archive_logs(days=30):
    """
    归档旧日志文件
    
    Args:
        days: 归档多少天前的日志
    """
    try:
        print("开始归档旧日志文件...")
        
        log_dir = get_log_dir()
        archive_dir = os.path.join(log_dir, 'archive')
        
        # 创建归档目录
        os.makedirs(archive_dir, exist_ok=True)
        
        # 计算截止日期
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # 查找并归档旧日志
        archived_count = 0
        for file in os.listdir(log_dir):
            file_path = os.path.join(log_dir, file)
            
            # 只处理文件，不处理目录
            if os.path.isfile(file_path):
                # 检查是否为日志文件
                if file.endswith('.log'):
                    # 获取文件修改时间
                    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    
                    # 检查文件是否过期
                    if file_time < cutoff_date:
                        # 生成归档文件名
                        archive_file = os.path.join(archive_dir, file)
                        
                        # 如果存在同名文件，添加时间戳
                        if os.path.exists(archive_file):
                            time_suffix = file_time.strftime('%Y%m%d%H%M%S')
                            base_name, ext = os.path.splitext(file)
                            archive_file = os.path.join(archive_dir, f"{base_name}_{time_suffix}{ext}")
                        
                        # 移动文件
                        os.rename(file_path, archive_file)
                        archived_count += 1
        
        print(f"已归档 {archived_count} 个日志文件")
        return True
    except Exception as e:
        logger.error(f"归档日志文件失败: {e}")
        print(f"归档日志文件失败: {e}")
        return False


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置日志级别
    setup_logger(level=args.log_level)
    
    # 执行操作
    if args.clear_cache:
        clear_cache(args.days)
    
    if args.optimize_db:
        optimize_database()
    
    if args.monitor:
        monitor_system(args.monitor_time)
    
    if args.archive_logs:
        archive_logs(args.days)
    
    print("所有操作完成")


if __name__ == "__main__":
    main() 