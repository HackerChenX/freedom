#!/usr/bin/env python3
"""
测试参数传递和解析脚本
"""
import sys
import subprocess
import re

def test_parsing_logic(stdout, stderr, returncode):
    """测试解析逻辑"""
    print("=== 测试解析逻辑 ===")

    # 模拟统一监控脚本的解析逻辑
    result = {
        'status': 'UNKNOWN',
        'score': 0,
        'message': '',
        'error': None,
        'output': stdout
    }

    # 查找总体得分
    score_found = False
    total_score_patterns = [
        r'📊 总体得分:\s*(\d+(?:\.\d+)?)/100',
        r'总体得分:\s*(\d+(?:\.\d+)?)/100',
        r'📊 总体得分[：:]\s*(\d+(?:\.\d+)?)',
        r'总体得分[：:]\s*(\d+(?:\.\d+)?)'
    ]

    for pattern in total_score_patterns:
        # 先在stdout中查找
        match = re.search(pattern, stdout)
        if match:
            result['score'] = float(match.group(1))
            score_found = True
            print(f"✅ 找到总体分数(stdout): {result['score']} (模式: {pattern})")
            break

        # 如果stdout中没有，再在stderr中查找
        match = re.search(pattern, stderr)
        if match:
            result['score'] = float(match.group(1))
            score_found = True
            print(f"✅ 找到总体分数(stderr): {result['score']} (模式: {pattern})")
            break

    # 特殊模式：验证通过，得分XX分
    if not score_found:
        special_pattern = r'验证通过，得分(\d+(?:\.\d+)?)分'
        match = re.search(special_pattern, stdout)
        if match:
            result['score'] = float(match.group(1))
            score_found = True
            print(f"✅ 特殊模式找到分数: {result['score']}")

    if not score_found:
        print("❌ 未找到分数")

    # 查找状态信息
    status_patterns = [
        r'验证状态[：:]\s*(\w+)',
        r'✅ 验证状态[：:]\s*(\w+)',
        r'状态[：:]\s*(\w+)'
    ]

    for pattern in status_patterns:
        # 先在stdout中查找
        match = re.search(pattern, stdout)
        if match:
            status_text = match.group(1)
            if status_text in ['PASSED', 'SUCCESS', '通过']:
                result['status'] = 'PASSED'
            elif status_text in ['FAILED', 'FAILURE', '失败']:
                result['status'] = 'FAILED'
            elif status_text in ['WARNING', '警告']:
                result['status'] = 'WARNING'
            print(f"✅ 找到状态(stdout): {result['status']}")
            break

        # 如果stdout中没有，再在stderr中查找
        match = re.search(pattern, stderr)
        if match:
            status_text = match.group(1)
            if status_text in ['PASSED', 'SUCCESS', '通过']:
                result['status'] = 'PASSED'
            elif status_text in ['FAILED', 'FAILURE', '失败']:
                result['status'] = 'FAILED'
            elif status_text in ['WARNING', '警告']:
                result['status'] = 'WARNING'
            print(f"✅ 找到状态(stderr): {result['status']}")
            break

    # 根据返回码确定状态
    if result['status'] == 'UNKNOWN':
        if returncode == 0:
            result['status'] = 'PASSED'
            print(f"✅ 根据返回码确定状态: {result['status']}")
        else:
            result['status'] = 'FAILED'
            print(f"❌ 根据返回码确定状态: {result['status']}")

    print(f"最终结果: 状态={result['status']}, 分数={result['score']}")
    return result

def test_direct_call():
    """测试直接调用"""
    print("=== 测试直接调用 ===")
    result = subprocess.run([
        sys.executable,
        'scripts/validate_enhanced_indicators.py',
        'EMA'
    ], capture_output=True, text=True, cwd='/Users/hacker/PycharmProjects/freedom')

    print(f"返回码: {result.returncode}")
    print(f"输出长度: {len(result.stdout)}")
    print(f"错误长度: {len(result.stderr)}")

    # 检查是否包含EMA
    if 'EMA' in result.stderr:
        print("✅ 参数传递成功，stderr包含EMA")
    else:
        print("❌ 参数传递失败，stderr不包含EMA")

    # 测试解析逻辑
    parsed = test_parsing_logic(result.stdout, result.stderr, result.returncode)

    return result.returncode == 0

if __name__ == "__main__":
    test_direct_call()
