#!/usr/bin/env python3
"""
检查可能导致终端卡死的进程
"""

import psutil
import os

def check_running_processes():
    """检查正在运行的相关进程"""
    print("🔍 检查正在运行的进程...")
    print("=" * 60)
    
    project_path = "/Users/hacker/PycharmProjects/freedom"
    monitoring_keywords = [
        'monitoring', 'monitor', 'self_healing', 'risk', 'alert',
        'real_time', 'market', 'enterprise', 'intelligent'
    ]
    
    found_processes = []
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_percent']):
        try:
            cmdline = ' '.join(proc.info['cmdline'] or [])
            
            # 检查是否是项目相关的Python进程
            if ('python' in proc.info['name'].lower() and 
                project_path in cmdline):
                
                # 检查是否包含监控关键词
                is_monitoring = any(keyword in cmdline.lower() for keyword in monitoring_keywords)
                
                process_info = {
                    'pid': proc.info['pid'],
                    'name': proc.info['name'],
                    'cmdline': cmdline[:100] + '...' if len(cmdline) > 100 else cmdline,
                    'cpu_percent': proc.info['cpu_percent'],
                    'memory_percent': proc.info['memory_percent'],
                    'is_monitoring': is_monitoring
                }
                
                found_processes.append(process_info)
                
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    if found_processes:
        print(f"发现 {len(found_processes)} 个相关进程:")
        print()
        
        for proc in found_processes:
            status = "🔴 监控进程" if proc['is_monitoring'] else "🟢 普通进程"
            print(f"{status} PID: {proc['pid']}")
            print(f"  名称: {proc['name']}")
            print(f"  命令: {proc['cmdline']}")
            print(f"  CPU: {proc['cpu_percent']:.1f}%")
            print(f"  内存: {proc['memory_percent']:.1f}%")
            print()
            
        # 统计监控进程
        monitoring_count = sum(1 for p in found_processes if p['is_monitoring'])
        if monitoring_count > 0:
            print(f"⚠️ 发现 {monitoring_count} 个监控进程可能导致终端卡死")
            print("建议运行修复脚本: python3 fix_cursor_terminal.py")
        else:
            print("✅ 未发现可疑的监控进程")
            
    else:
        print("✅ 未发现项目相关的Python进程")

def check_system_resources():
    """检查系统资源使用情况"""
    print("\n💻 系统资源使用情况:")
    print("=" * 30)
    
    # CPU使用率
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"CPU使用率: {cpu_percent}%")
    
    # 内存使用率
    memory = psutil.virtual_memory()
    print(f"内存使用率: {memory.percent}%")
    print(f"可用内存: {memory.available / (1024**3):.1f}GB")
    
    # 磁盘使用率
    try:
        disk = psutil.disk_usage('/Users/hacker/PycharmProjects/freedom')
        disk_percent = (disk.used / disk.total) * 100
        print(f"磁盘使用率: {disk_percent:.1f}%")
        print(f"可用空间: {disk.free / (1024**3):.1f}GB")
    except:
        print("无法获取磁盘信息")
    
    # 资源警告
    warnings = []
    if cpu_percent > 80:
        warnings.append("CPU使用率过高")
    if memory.percent > 85:
        warnings.append("内存使用率过高")
    if 'disk_percent' in locals() and disk_percent > 90:
        warnings.append("磁盘空间不足")
    
    if warnings:
        print(f"\n⚠️ 资源警告: {', '.join(warnings)}")
    else:
        print("\n✅ 系统资源使用正常")

def main():
    print("🔍 Cursor终端进程检查工具")
    print("=" * 40)
    
    try:
        check_running_processes()
        check_system_resources()
        
        print("\n📋 建议操作:")
        print("1. 如果发现监控进程，运行: python3 fix_cursor_terminal.py")
        print("2. 如果系统资源紧张，考虑重启系统")
        print("3. 如果问题持续，尝试: bash quick_terminal_reset.sh")
        
    except Exception as e:
        print(f"❌ 检查过程中出现错误: {e}")

if __name__ == "__main__":
    main()

