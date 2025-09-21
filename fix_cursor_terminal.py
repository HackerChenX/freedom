#!/usr/bin/env python3
"""
Cursor终端卡死修复脚本

解决Cursor命令行卡死问题的综合修复方案
"""

import os
import sys
import signal
import psutil
import time
import subprocess
from pathlib import Path

def kill_monitoring_processes():
    """停止所有可能的监控进程"""
    print("🔍 查找并停止监控进程...")
    
    monitoring_keywords = [
        'monitoring', 'monitor', 'real_time', 'self_healing', 
        'risk_monitor', 'market_monitor', 'alert_system'
    ]
    
    killed_count = 0
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline'] or [])
            if any(keyword in cmdline.lower() for keyword in monitoring_keywords):
                if 'python' in cmdline.lower() and 'freedom' in cmdline:
                    print(f"  停止进程: {proc.info['pid']} - {cmdline[:100]}...")
                    proc.terminate()
                    killed_count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    if killed_count > 0:
        print(f"✅ 已停止 {killed_count} 个监控进程")
        time.sleep(2)  # 等待进程完全停止
    else:
        print("ℹ️ 未发现活跃的监控进程")

def kill_python_processes_in_project():
    """停止项目目录下的Python进程"""
    print("🔍 查找并停止项目相关的Python进程...")
    
    project_path = "/Users/hacker/PycharmProjects/freedom"
    killed_count = 0
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cwd']):
        try:
            if proc.info['name'] and 'python' in proc.info['name'].lower():
                cmdline = ' '.join(proc.info['cmdline'] or [])
                cwd = proc.info.get('cwd', '')
                
                if project_path in cmdline or project_path in cwd:
                    # 不要杀死当前脚本
                    if 'fix_cursor_terminal.py' not in cmdline:
                        print(f"  停止Python进程: {proc.info['pid']} - {cmdline[:100]}...")
                        proc.terminate()
                        killed_count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    if killed_count > 0:
        print(f"✅ 已停止 {killed_count} 个Python进程")
        time.sleep(2)
    else:
        print("ℹ️ 未发现需要停止的Python进程")

def clear_python_cache():
    """清理Python缓存"""
    print("🧹 清理Python缓存...")
    
    project_root = Path("/Users/hacker/PycharmProjects/freedom")
    cache_dirs = list(project_root.rglob("__pycache__"))
    
    removed_count = 0
    for cache_dir in cache_dirs:
        try:
            import shutil
            shutil.rmtree(cache_dir)
            removed_count += 1
        except Exception as e:
            print(f"  警告: 无法删除 {cache_dir}: {e}")
    
    print(f"✅ 已清理 {removed_count} 个缓存目录")

def reset_terminal_environment():
    """重置终端环境变量"""
    print("🔄 重置终端环境...")
    
    # 清理可能导致问题的环境变量
    problematic_vars = [
        'PYTHONPATH', 'PYTHON_PATH', 'MONITORING_ACTIVE', 
        'SELF_HEALING_ACTIVE', 'RISK_MONITOR_ACTIVE'
    ]
    
    for var in problematic_vars:
        if var in os.environ:
            del os.environ[var]
            print(f"  清理环境变量: {var}")
    
    print("✅ 终端环境已重置")

def check_disk_space():
    """检查磁盘空间"""
    print("💾 检查磁盘空间...")
    
    try:
        usage = psutil.disk_usage('/Users/hacker/PycharmProjects/freedom')
        free_gb = usage.free / (1024**3)
        total_gb = usage.total / (1024**3)
        used_percent = (usage.used / usage.total) * 100
        
        print(f"  磁盘使用率: {used_percent:.1f}%")
        print(f"  可用空间: {free_gb:.1f}GB / {total_gb:.1f}GB")
        
        if used_percent > 90:
            print("⚠️ 磁盘空间不足，可能影响终端性能")
        else:
            print("✅ 磁盘空间充足")
            
    except Exception as e:
        print(f"❌ 无法检查磁盘空间: {e}")

def check_memory_usage():
    """检查内存使用情况"""
    print("🧠 检查内存使用...")
    
    try:
        memory = psutil.virtual_memory()
        print(f"  内存使用率: {memory.percent}%")
        print(f"  可用内存: {memory.available / (1024**3):.1f}GB")
        
        if memory.percent > 85:
            print("⚠️ 内存使用率过高，可能影响终端性能")
        else:
            print("✅ 内存使用正常")
            
    except Exception as e:
        print(f"❌ 无法检查内存使用: {e}")

def create_terminal_test_script():
    """创建终端测试脚本"""
    print("📝 创建终端测试脚本...")
    
    test_script = """#!/usr/bin/env python3
# 简单的终端测试脚本
import sys
import time

print("🧪 Cursor终端测试")
print("=" * 40)

# 测试基本输出
print("✅ 基本输出正常")

# 测试简单计算
result = 2 + 2
print(f"✅ 计算测试: 2 + 2 = {result}")

# 测试导入
try:
    import pandas as pd
    print("✅ pandas导入正常")
except ImportError as e:
    print(f"❌ pandas导入失败: {e}")

try:
    import numpy as np
    print("✅ numpy导入正常")
except ImportError as e:
    print(f"❌ numpy导入失败: {e}")

print("=" * 40)
print("🎉 终端测试完成！如果你能看到这条消息，说明终端工作正常。")
"""
    
    with open("/Users/hacker/PycharmProjects/freedom/test_terminal.py", "w") as f:
        f.write(test_script)
    
    print("✅ 终端测试脚本已创建: test_terminal.py")

def main():
    """主修复流程"""
    print("🔧 Cursor终端卡死修复工具")
    print("=" * 50)
    
    try:
        # 1. 停止监控进程
        kill_monitoring_processes()
        
        # 2. 停止项目相关Python进程
        kill_python_processes_in_project()
        
        # 3. 清理缓存
        clear_python_cache()
        
        # 4. 重置环境
        reset_terminal_environment()
        
        # 5. 检查系统资源
        check_disk_space()
        check_memory_usage()
        
        # 6. 创建测试脚本
        create_terminal_test_script()
        
        print("\n" + "=" * 50)
        print("🎉 修复完成！")
        print("\n📋 后续步骤:")
        print("1. 重启Cursor应用")
        print("2. 打开新的终端窗口")
        print("3. 运行: python3 test_terminal.py")
        print("4. 如果测试通过，终端应该恢复正常")
        
        print("\n💡 预防措施:")
        print("- 避免同时运行多个监控脚本")
        print("- 定期清理临时文件和缓存")
        print("- 监控系统资源使用情况")
        
    except Exception as e:
        print(f"❌ 修复过程中出现错误: {e}")
        print("请尝试手动重启Cursor应用")

if __name__ == "__main__":
    main()

