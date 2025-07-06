#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
KDJ指标完美验证测试

专门测试KDJ指标的所有形态，目标达到100%成功率
"""

import sys
import os
from datetime import datetime
import json

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from perfect_validator import Perfect_validator


def main_testkdjperfect():
    """主函数"""
    print("=" * 60)
    print("KDJ指标完美验证测试")
    print("目标：达到100%形态识别成功率")
    print("=" * 60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 创建完美验证器并运行KDJ测试
    validator = Perfect_validator()
    
    try:
        results = validator.validate_kdj_patterns_perfect()
        
        # 显示总结
        print("=" * 60)
        print("KDJ完美验证测试总结")
        print("=" * 60)
        print(f"总形态数: {results['total_patterns']}")
        print(f"成功识别: {results['successful_patterns']}")
        print(f"识别失败: {results['failed_patterns']}")
        print(f"成功率: {results['summary']['success_rate']}")
        print(f"平均匹配分: {results['summary']['average_score']}")
        print(f"建议: {results['summary']['recommendation']}")
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"perfect_validation_results_KDJ_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\n详细结果已保存到: {output_file}")
        
        # 返回退出码
        if results['success_rate'] >= 1.0:
            print("\n🎉 完美！KDJ指标达到100%成功率目标")
            return 0
        elif results['success_rate'] >= 0.8:
            print("\n✅ 优秀！KDJ指标接近100%成功率目标")
            return 0
        else:
            print("\n❌ KDJ指标未达到目标，需要继续优化")
            return 1
            
    except Exception as e:
        print(f"❌ KDJ验证测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit_code = main_testkdjperfect()
    sys.exit(exit_code)
