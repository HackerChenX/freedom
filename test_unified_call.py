#!/usr/bin/env python3
"""
测试统一监控脚本调用通用验证脚本
"""
import subprocess
import sys
import os

def test_unified_call():
    """测试统一监控脚本的调用方式"""
    print("=== 测试统一监控脚本调用方式 ===")
    
    # 模拟统一监控脚本的调用方式
    script_path = 'scripts/validate_enhanced_indicators.py'
    indicator_name = 'EMA'
    root_dir = '/Users/hacker/PycharmProjects/freedom'
    
    # 构建命令（模拟统一监控脚本的逻辑）
    cmd = [sys.executable, script_path]
    
    # 检查是否是通用验证脚本
    if 'validate_enhanced_indicators.py' in script_path:
        # 通用验证脚本使用位置参数
        cmd.append(indicator_name)
        print(f"✅ 添加位置参数: {indicator_name}")
    
    print(f"执行命令: {' '.join(cmd)}")
    print(f"工作目录: {root_dir}")
    
    # 执行命令
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120,  # 2分钟超时
        cwd=root_dir
    )
    
    print(f"返回码: {result.returncode}")
    print(f"输出长度: {len(result.stdout)}")
    print(f"错误长度: {len(result.stderr)}")
    
    # 检查输出中是否包含正确的指标名称
    if f'🔍 开始{indicator_name}验证' in result.stderr:
        print("✅ 参数传递成功，输出包含正确的指标名称")
    elif '🔍 开始ENHANCED验证' in result.stderr:
        print("❌ 参数传递失败，仍然显示ENHANCED")
    else:
        print("❓ 无法确定参数传递状态")
    
    # 检查是否有分数输出
    if '验证通过，得分' in result.stdout:
        import re
        match = re.search(r'验证通过，得分(\d+(?:\.\d+)?)分', result.stdout)
        if match:
            score = float(match.group(1))
            print(f"✅ 获得分数: {score}分")
        else:
            print("❌ 无法解析分数")
    else:
        print("❌ 没有分数输出")
    
    # 显示部分输出用于调试
    print("\n=== 标准输出（前500字符）===")
    print(result.stdout[:500])
    print("\n=== 标准错误（前500字符）===")
    print(result.stderr[:500])

if __name__ == "__main__":
    test_unified_call()
