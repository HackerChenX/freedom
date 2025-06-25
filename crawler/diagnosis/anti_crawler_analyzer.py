"""
反爬虫机制诊断分析器

分析目标网站的反爬虫策略和技术实现
"""

import requests
import time
import re
from datetime import datetime
from typing import Dict, List, Any
from urllib.parse import urlparse
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.logger import get_logger

logger = get_logger(__name__)


class AntiCrawlerAnalyzer:
    """反爬虫机制分析器"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })

        self.target_sites = {
            'xueqiu': {
                'name': '雪球网',
                'base_url': 'https://xueqiu.com',
                'test_urls': [
                    'https://xueqiu.com',
                    'https://xueqiu.com/today',
                    'https://xueqiu.com/hq'
                ]
            },
            'taoguba': {
                'name': '淘股吧',
                'base_url': 'https://www.taoguba.com.cn',
                'test_urls': [
                    'https://www.taoguba.com.cn',
                    'https://www.taoguba.com.cn/hotTopics'
                ]
            },
            'jiyanshe': {
                'name': '韭研公社',
                'base_url': 'https://www.jiyanshe.com',
                'test_urls': [
                    'https://www.jiyanshe.com'
                ]
            }
        }

    def analyze_site(self, site_key: str) -> Dict[str, Any]:
        """分析单个网站的反爬虫机制"""
        site_config = self.target_sites[site_key]
        analysis_result = {
            'site_name': site_config['name'],
            'site_key': site_key,
            'base_url': site_config['base_url'],
            'analysis_time': datetime.now(),
            'accessibility': {},
            'anti_crawler_features': [],
            'recommendations': []
        }

        print(f"\n🔍 分析 {site_config['name']} ({site_key})...")

        # 测试基本访问性
        for i, test_url in enumerate(site_config['test_urls']):
            print(f"  📡 测试URL {i+1}: {test_url}")

            try:
                response = self.session.get(test_url, timeout=15)
                content = response.text.lower()

                # 检测反爬虫特征
                anti_crawler_indicators = {
                    'cloudflare_protection': 'cloudflare' in content or 'cf-ray' in str(response.headers),
                    'captcha_required': any(keyword in content for keyword in ['captcha', '验证码', 'recaptcha']),
                    'login_required': any(keyword in content for keyword in ['login', '登录', 'signin', '请登录']),
                    'javascript_required': 'javascript' in content and ('enable' in content or 'disabled' in content),
                    'access_denied': response.status_code in [403, 429] or any(keyword in content for keyword in ['access denied', '访问被拒绝', 'forbidden']),
                    'empty_response': len(content) < 500,
                    'bot_detection': any(keyword in content for keyword in ['bot', 'robot', '机器人检测', 'automated'])
                }

                # 检查是否有有效内容
                has_meaningful_content = (
                    len(content) > 1000 and
                    any(keyword in content for keyword in ['股票', '投资', '讨论', '分析', '市场', 'stock'])
                )

                url_analysis = {
                    'url': test_url,
                    'status_code': response.status_code,
                    'content_length': len(response.text),
                    'anti_crawler_indicators': anti_crawler_indicators,
                    'has_meaningful_content': has_meaningful_content,
                    'response_headers': dict(response.headers)
                }

                analysis_result['accessibility'][test_url] = url_analysis

                print(f"    ✅ 状态码: {response.status_code}, 内容长度: {len(response.text)}")

                # 检测到的反爬虫特征
                detected_features = [k for k, v in anti_crawler_indicators.items() if v]
                if detected_features:
                    print(f"    ⚠️  检测到反爬虫特征: {detected_features}")
                    analysis_result['anti_crawler_features'].extend(detected_features)

                if not has_meaningful_content:
                    print(f"    ❌ 内容可能被阻止或需要JavaScript渲染")

            except Exception as e:
                print(f"    ❌ 访问失败: {e}")
                analysis_result['accessibility'][test_url] = {
                    'error': str(e),
                    'accessible': False
                }

        # 去重反爬虫特征
        analysis_result['anti_crawler_features'] = list(set(analysis_result['anti_crawler_features']))

        # 生成建议
        self._generate_site_recommendations(analysis_result)

        return analysis_result

    def _generate_site_recommendations(self, analysis: Dict[str, Any]):
        """为单个网站生成建议"""
        features = analysis['anti_crawler_features']
        recommendations = []

        if 'cloudflare_protection' in features:
            recommendations.append("需要绕过Cloudflare保护")

        if 'captcha_required' in features:
            recommendations.append("需要验证码识别")

        if 'login_required' in features:
            recommendations.append("需要登录验证")

        if 'javascript_required' in features or 'empty_response' in features:
            recommendations.append("需要JavaScript渲染")

        if 'access_denied' in features:
            recommendations.append("存在IP封禁或频率限制")

        if 'bot_detection' in features:
            recommendations.append("存在机器人检测")

        analysis['recommendations'] = recommendations


def main():
    """主函数"""
    print("=== 股市讨论社区反爬虫机制诊断 ===")
    print(f"分析时间: {datetime.now()}")
    print("目标网站: 雪球网、淘股吧、韭研公社")
    print("=" * 60)

    analyzer = AntiCrawlerAnalyzer()

    try:
        all_results = {}

        # 分析每个网站
        for site_key in analyzer.target_sites.keys():
            site_analysis = analyzer.analyze_site(site_key)
            all_results[site_key] = site_analysis

        # 显示总结
        print(f"\n📊 诊断总结:")
        for site_key, analysis in all_results.items():
            print(f"\n{analysis['site_name']}:")
            print(f"  反爬虫特征: {analysis['anti_crawler_features']}")
            print(f"  建议: {analysis['recommendations']}")

        # 保存结果
        import json
        output_file = f"anti_crawler_diagnosis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)

        print(f"\n💾 诊断结果已保存到: {output_file}")

    except Exception as e:
        logger.error(f"诊断过程中发生错误: {e}")
        print(f"❌ 诊断失败: {e}")


if __name__ == "__main__":
    main()