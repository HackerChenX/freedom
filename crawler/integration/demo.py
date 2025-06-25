"""
集成测试与部署演示

展示完整的系统集成功能
"""

import sys
import os
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from crawler.integration.system_integrator import SystemIntegrator
from crawler.integration.integration_tester import IntegrationTester
from utils.logger import get_logger

logger = get_logger(__name__)


def demo_system_integration():
    """演示系统集成功能"""
    print("\n=== 系统集成演示 ===")

    try:
        # 创建系统集成器
        integrator = SystemIntegrator()

        # 初始化集成系统
        print("初始化集成系统...")
        if integrator.initialize():
            print("✅ 集成系统初始化成功")
        else:
            print("❌ 集成系统初始化失败")
            return False

        # 测试文章数据同步
        print("\n测试文章数据同步...")
        test_article = {
            'id': 'demo_article_001',
            'title': '【投资机会】新能源汽车产业链迎来重大利好',
            'content': '''
            近期新能源汽车产业链迎来重大政策利好，相关概念股值得关注。
            重点关注：比亚迪(002594)、宁德时代(300750)、赛轮轮胎(601058)等。
            新能源、锂电池、智能汽车等概念持续升温。
            ''',
            'source': '演示数据源',
            'url': 'https://example.com/demo-article',
            'author': '演示作者',
            'stock_codes': ['002594', '300750', '601058'],
            'concepts': ['新能源', '锂电池', '智能汽车']
        }

        if integrator.sync_article_data(test_article):
            print("✅ 文章数据同步成功")
        else:
            print("❌ 文章数据同步失败")

        # 获取集成状态
        print("\n集成状态:")
        status = integrator.get_integration_status()
        for key, value in status.items():
            print(f"- {key}: {value}")

        # 健康检查
        print("\n健康检查:")
        health = integrator.health_check()
        print(f"- 总体状态: {health['overall_status']}")
        print(f"- 组件状态: {health['components']}")
        if health['issues']:
            print(f"- 问题: {health['issues']}")

        return True

    except Exception as e:
        logger.error(f"系统集成演示失败: {e}")
        print(f"❌ 系统集成演示失败: {e}")
        return False


def demo_integration_testing():
    """演示集成测试功能"""
    print("\n=== 集成测试演示 ===")

    try:
        # 创建集成测试器
        tester = IntegrationTester()

        # 运行所有测试
        print("运行集成测试...")
        test_results = tester.run_all_tests()

        # 显示测试结果
        print(f"\n测试结果:")
        print(f"- 总测试数: {test_results['total_tests']}")
        print(f"- 通过测试: {test_results['passed_tests']}")
        print(f"- 失败测试: {test_results['failed_tests']}")
        print(f"- 成功率: {test_results['success_rate']:.1f}%")
        print(f"- 总耗时: {test_results['total_duration']:.2f}秒")
        print(f"- 总体状态: {test_results['overall_status']}")

        # 显示详细结果
        print("\n详细测试结果:")
        for result in test_results['test_results']:
            status_icon = "✅" if result['status'] == 'passed' else "❌"
            print(f"{status_icon} {result['name']}: {result['status'].upper()} ({result['duration']:.2f}秒)")

            if result['error_message']:
                print(f"   错误: {result['error_message']}")

        return test_results['overall_status'] == 'PASSED'

    except Exception as e:
        logger.error(f"集成测试演示失败: {e}")
        print(f"❌ 集成测试演示失败: {e}")
        return False


def main():
    """主函数"""
    print("=== 股市信息爬虫系统集成测试与部署演示 ===")
    print(f"演示时间: {datetime.now()}")

    success_count = 0
    total_demos = 2

    # 演示系统集成
    if demo_system_integration():
        success_count += 1

    # 演示集成测试
    if demo_integration_testing():
        success_count += 1

    # 总结
    print(f"\n=== 演示总结 ===")
    print(f"成功演示: {success_count}/{total_demos}")
    print(f"成功率: {success_count/total_demos*100:.1f}%")

    if success_count == total_demos:
        print("🎉 所有演示成功！系统集成与部署功能正常")
        print("\n系统已准备好投入生产使用")
    else:
        print("⚠️  部分演示失败，请检查相关配置")

    return success_count == total_demos


if __name__ == "__main__":
    main()