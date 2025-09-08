#!/usr/bin/env python3
"""
小批量测试脚本
"""
import subprocess
import sys

def test_small_batch():
    """测试几个使用通用验证脚本的指标"""
    test_indicators = ['EMA', 'CCI', 'WMA', 'SAR', 'ADX']
    
    print("=== 小批量测试 ===")
    
    for indicator in test_indicators:
        print(f"\n--- 测试 {indicator} ---")
        
        # 直接调用验证脚本
        result = subprocess.run([
            sys.executable, 
            'scripts/validate_enhanced_indicators.py', 
            indicator
        ], capture_output=True, text=True, cwd='/Users/hacker/PycharmProjects/freedom')
        
        print(f"返回码: {result.returncode}")
        
        # 查找分数
        if '验证通过，得分' in result.stdout:
            import re
            match = re.search(r'验证通过，得分(\d+(?:\.\d+)?)分', result.stdout)
            if match:
                score = float(match.group(1))
                print(f"✅ {indicator}: {score}分")
            else:
                print(f"❌ {indicator}: 无法解析分数")
        else:
            print(f"❌ {indicator}: 验证失败")

if __name__ == "__main__":
    test_small_batch()
