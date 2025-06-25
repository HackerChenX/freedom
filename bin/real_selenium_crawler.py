#!/usr/bin/env python3
"""
真实Selenium爬虫系统

使用Selenium解决JavaScript渲染和登录验证问题
获取真实的股市讨论社区内容
"""

import sys
import os
import json
import time
import random
from datetime import datetime
from typing import Dict, List, Any

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crawler.processors.concept_extractor import ConceptStockExtractor
from utils.logger import get_logger

logger = get_logger(__name__)

# 检查Selenium是否可用
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.action_chains import ActionChains
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
    SELENIUM_AVAILABLE = True
    print("✅ Selenium可用")
except ImportError:
    SELENIUM_AVAILABLE = False
    print("❌ Selenium未安装")

import requests


class RealSeleniumCrawler:
    """真实Selenium爬虫"""

    def __init__(self):
        self.concept_extractor = ConceptStockExtractor()
        self.driver = None

        # 备用请求会话
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })

    def setup_selenium_driver(self) -> bool:
        """设置Selenium WebDriver"""
        if not SELENIUM_AVAILABLE:
            print("❌ Selenium未安装，无法使用浏览器自动化")
            print("💡 安装方法: pip install selenium")
            print("💡 还需要下载ChromeDriver: https://chromedriver.chromium.org/")
            return False

        try:
            print("🔧 正在设置Chrome WebDriver...")

            chrome_options = Options()
            # 反检测设置
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-blink-features=AutomationControlled')
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)

            # 设置窗口大小
            chrome_options.add_argument('--window-size=1920,1080')

            # 尝试创建WebDriver
            self.driver = webdriver.Chrome(options=chrome_options)

            # 反检测脚本
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

            print("✅ Chrome WebDriver设置成功")
            return True

        except Exception as e:
            print(f"❌ Chrome WebDriver设置失败: {e}")
            print("💡 请确保已安装Chrome浏览器和ChromeDriver")
            print("💡 ChromeDriver下载: https://chromedriver.chromium.org/")
            return False