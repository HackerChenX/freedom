#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import os

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import datetime
import logging

from utils.logger import get_logger

# 设置日志输出到控制台
logger = get_logger("json_debug")
logger.handlers = []
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)

def check_value(value, path=""):
    """递归检查值是否包含百分号"""
    if isinstance(value, str):
        if "%" in value:
            logger.info(f"发现百分号! 路径: {path}, 值: {value}")
            return True
    elif isinstance(value, dict):
        for k, v in value.items():
            if check_value(v, f"{path}.{k}" if path else k):
                return True
    elif isinstance(value, list):
        for i, v in enumerate(value):
            if check_value(v, f"{path}[{i}]"):
                return True
    return False

def create_test_data():
    """创建测试数据，模拟实际情况"""
    # 模拟可能产生百分号的情况
    result = {
        "code": "300005",
        "name": "探路者",
        "industry": "纺织服装",
        "buy_date": "20250424",
        "buy_price": 8.21,
        "indicators": {
            "absorb_strength": 3,  # 这个值可能会转换为百分比
            "absorb_strength_str": "3%",  # 显式包含百分号
            "number_values": [1, 2, 3],
            "percent_values": ["10%", "20%", "30%"]
        },
        "nested": {
            "deep": {
                "percent_value": "5%"
            }
        }
    }
    return result

def test_standard_json():
    """测试标准JSON序列化"""
    data = create_test_data()
    logger.info("检查原始数据是否包含百分号:")
    check_value(data)
    
    logger.info("\n=== 使用标准json.dumps测试 ===")
    json_str = json.dumps(data, ensure_ascii=False, indent=2)
    logger.info(f"JSON字符串末尾10个字符: {json_str[-10:]}")
    if "%" in json_str:
        logger.info("JSON字符串中包含百分号")
    else:
        logger.info("JSON字符串中不包含百分号")
    
    # 写入文件测试
    test_file = os.path.join(root_dir, "data", "result", "debug_standard.json")
    os.makedirs(os.path.dirname(test_file), exist_ok=True)
    
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(json_str)
    
    logger.info(f"已写入标准JSON文件: {test_file}")
    
    # 读取文件检查末尾
    with open(test_file, 'rb') as f:
        f.seek(-10, 2)
        last_bytes = f.read()
        if b'%' in last_bytes:
            logger.info(f"文件末尾存在百分号: {last_bytes}")
        else:
            logger.info(f"文件末尾没有百分号: {last_bytes}")

def test_custom_encoder():
    """测试自定义编码器"""
    class CleanEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif pd.isna(obj) or obj is None or obj == np.nan:
                return None
            return super(CleanEncoder, self).default(obj)
        
        def encode(self, obj):
            if isinstance(obj, str):
                return super(CleanEncoder, self).encode(obj.replace('%', ''))
            elif isinstance(obj, dict):
                return super(CleanEncoder, self).encode({k: self._clean_value(v) for k, v in obj.items()})
            elif isinstance(obj, list):
                return super(CleanEncoder, self).encode([self._clean_value(v) for v in obj])
            return super(CleanEncoder, self).encode(obj)
        
        def _clean_value(self, value):
            if isinstance(value, str):
                return value.replace('%', '')
            elif isinstance(value, dict):
                return {k: self._clean_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [self._clean_value(v) for v in value]
            elif isinstance(value, (np.integer, np.floating, np.bool_)):
                return value.item()
            elif isinstance(value, np.ndarray):
                return value.tolist()
            elif pd.isna(value) or value is None or value == np.nan:
                return None
            return value
    
    data = create_test_data()
    logger.info("\n=== 使用自定义编码器测试 ===")
    json_str = json.dumps(data, cls=CleanEncoder, ensure_ascii=False, indent=2)
    logger.info(f"JSON字符串末尾10个字符: {json_str[-10:]}")
    if "%" in json_str:
        logger.info("JSON字符串中包含百分号")
    else:
        logger.info("JSON字符串中不包含百分号")
    
    # 写入文件测试
    test_file = os.path.join(root_dir, "data", "result", "debug_custom.json")
    
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(json_str)
    
    logger.info(f"已写入自定义编码器JSON文件: {test_file}")
    
    # 读取文件检查末尾
    with open(test_file, 'rb') as f:
        f.seek(-10, 2)
        last_bytes = f.read()
        if b'%' in last_bytes:
            logger.info(f"文件末尾存在百分号: {last_bytes}")
        else:
            logger.info(f"文件末尾没有百分号: {last_bytes}")

def test_direct_writing():
    """测试直接写入文件，不使用json.dump"""
    data = create_test_data()
    logger.info("\n=== 使用直接写入文件测试 ===")
    
    # 先转换为JSON字符串
    json_str = json.dumps(data, ensure_ascii=False, indent=2)
    # 移除百分号
    json_str = json_str.replace("%", "")
    
    # 写入文件测试
    test_file = os.path.join(root_dir, "data", "result", "debug_direct.json")
    
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(json_str)
    
    logger.info(f"已直接写入JSON文件: {test_file}")
    
    # 读取文件检查末尾
    with open(test_file, 'rb') as f:
        f.seek(-10, 2)
        last_bytes = f.read()
        if b'%' in last_bytes:
            logger.info(f"文件末尾存在百分号: {last_bytes}")
        else:
            logger.info(f"文件末尾没有百分号: {last_bytes}")

def test_post_process():
    """测试后处理：写入后再用系统命令处理"""
    data = create_test_data()
    logger.info("\n=== 使用后处理测试 ===")
    
    # 写入文件测试
    test_file = os.path.join(root_dir, "data", "result", "debug_post.json")
    temp_file = f"{test_file}.tmp"
    
    with open(temp_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    # 使用系统命令移除百分号
    os.system(f"cat {temp_file} | tr -d '%' > {test_file}")
    os.remove(temp_file)
    
    logger.info(f"已处理并写入JSON文件: {test_file}")
    
    # 读取文件检查末尾
    with open(test_file, 'rb') as f:
        f.seek(-10, 2)
        last_bytes = f.read()
        if b'%' in last_bytes:
            logger.info(f"文件末尾存在百分号: {last_bytes}")
        else:
            logger.info(f"文件末尾没有百分号: {last_bytes}")

def test_final_hack():
    """测试一个严格的解决方案，处理JSON最后一个字符问题"""
    data = create_test_data()
    logger.info("\n=== 使用终极解决方案测试 ===")
    
    # 写入文件测试
    test_file = os.path.join(root_dir, "data", "result", "debug_final.json")
    
    # 序列化为字符串并手动处理
    json_str = json.dumps(data, ensure_ascii=False, indent=2)
    json_str = json_str.replace("%", "")
    
    # 确保最后一个字符不是百分号
    if json_str and json_str[-1] == "%":
        json_str = json_str[:-1] + "}"
    
    # 写入文件
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(json_str)
    
    logger.info(f"已使用终极解决方案写入JSON文件: {test_file}")
    
    # 读取文件检查末尾
    with open(test_file, 'rb') as f:
        f.seek(-10, 2)
        last_bytes = f.read()
        logger.info(f"文件末尾字符: {last_bytes}")

def main():
    logger.info("=== JSON序列化问题调试脚本 ===")
    
    test_standard_json()
    test_custom_encoder()
    test_direct_writing()
    test_post_process()
    test_final_hack()
    
    logger.info("调试完成")

if __name__ == "__main__":
    main() 