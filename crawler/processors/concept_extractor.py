"""
概念股提取器

从文本中提取股票代码、公司名称、概念信息
"""

import re
import json
from typing import Dict, List, Any, Optional, Set
from utils.logger import get_logger

logger = get_logger(__name__)


class ConceptStockExtractor:
    """概念股提取器"""

    def __init__(self):
        self.stock_dict = self._load_stock_dict()
        self.concept_dict = self._load_concept_dict()

    def _load_stock_dict(self) -> Dict[str, str]:
        """加载股票代码字典"""
        # 这里应该从数据库或文件加载股票代码和名称的映射
        # 暂时返回空字典，实际使用时需要完善
        return {}

    def _load_concept_dict(self) -> Dict[str, List[str]]:
        """加载概念股字典"""
        # 这里应该从数据库或文件加载概念和相关股票的映射
        # 暂时返回空字典，实际使用时需要完善
        return {}

    def extract_stocks(self, text: str) -> Dict[str, Any]:
        """从文本中提取股票信息"""
        if not text:
            return {
                'stock_codes': [],
                'company_names': [],
                'concepts': [],
                'confidence': 0.0
            }

        # 1. 正则匹配股票代码
        stock_codes = self._extract_stock_codes(text)

        # 2. 实体识别公司名称
        company_names = self._extract_company_names(text)

        # 3. 概念关键词匹配
        concepts = self._extract_concepts(text)

        # 4. 置信度计算
        confidence = self._calculate_confidence(stock_codes, company_names, concepts)

        return {
            'stock_codes': list(stock_codes),
            'company_names': list(company_names),
            'concepts': list(concepts),
            'confidence': confidence
        }

    def _extract_stock_codes(self, text: str) -> Set[str]:
        """提取股票代码"""
        stock_codes = set()

        # 匹配6位数字的股票代码
        pattern = r'\b[0-9]{6}\b'
        matches = re.findall(pattern, text)

        for match in matches:
            # 验证是否为有效股票代码
            if self._is_valid_stock_code(match):
                stock_codes.add(match)

        return stock_codes

    def _extract_company_names(self, text: str) -> Set[str]:
        """提取公司名称"""
        company_names = set()

        # 简单的公司名称匹配模式
        patterns = [
            r'[\u4e00-\u9fff]+(?:股份|集团|公司|科技|电子|医药|银行|保险)',
            r'[\u4e00-\u9fff]{2,8}(?:有限公司|股份有限公司)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) >= 2:
                    company_names.add(match)

        return company_names

    def _extract_concepts(self, text: str) -> Set[str]:
        """提取概念关键词"""
        concepts = set()

        # 常见概念关键词
        concept_keywords = [
            '人工智能', 'AI', '5G', '新能源', '锂电池', '光伏', '风电',
            '芯片', '半导体', '新材料', '生物医药', '疫苗', '抗癌',
            '区块链', '数字货币', '元宇宙', 'VR', 'AR',
            '新基建', '充电桩', '特高压', '工业互联网',
            '军工', '航空', '航天', '核电', '环保',
            '消费电子', '智能汽车', '自动驾驶', '物联网'
        ]

        for keyword in concept_keywords:
            if keyword in text:
                concepts.add(keyword)

        return concepts

    def _is_valid_stock_code(self, code: str) -> bool:
        """验证股票代码是否有效"""
        # 简单验证：沪深股票代码规则
        if len(code) != 6:
            return False

        # 沪市：60开头，科创板：688开头
        # 深市：00开头（主板），002开头（中小板），300开头（创业板）
        valid_prefixes = ['60', '688', '00', '002', '300']

        for prefix in valid_prefixes:
            if code.startswith(prefix):
                return True

        return False

    def _calculate_confidence(self, stock_codes: Set[str],
                            company_names: Set[str],
                            concepts: Set[str]) -> float:
        """计算提取置信度"""
        score = 0.0

        # 股票代码权重最高
        if stock_codes:
            score += len(stock_codes) * 0.5

        # 公司名称权重中等
        if company_names:
            score += len(company_names) * 0.3

        # 概念关键词权重较低
        if concepts:
            score += len(concepts) * 0.2

        # 归一化到0-1之间
        return min(score / 3.0, 1.0)