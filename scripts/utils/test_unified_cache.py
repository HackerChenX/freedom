#!/usr/bin/env python3
"""
统一缓存层测试脚本

测试统一缓存层的功能，包括多级缓存、性能测试和缓存服务。
"""

import os
import sys
import time
import json
from typing import Dict, Any

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from db.cache_layer import UnifiedCacheLayer, CacheLevel
from db.services.cache_service import Cache_service, Cache_key_builder
from config.cache_config import get_cache_config, CacheProfile
from config.container_config import get_configured_container
from db.interfaces.cache_interface import ICacheService
from utils.logger import get_logger

logger = get_logger(__name__)


def test_basic_cache_operations():
    """测试基本缓存操作"""
    print("\n=== 测试基本缓存操作 ===")
    
    # 创建缓存层
    config = get_cache_config(CacheProfile.TESTING)
    cache_layer = UnifiedCacheLayer(config)
    
    # 测试设置和获取
    test_data = {"name": "测试股票", "code": "000001", "price": 10.5}
    key = "test:stock:000001"
    
    # 设置缓存
    success = cache_layer.set(key, test_data, ttl=300)
    print(f"设置缓存: {success}")
    
    # 获取缓存
    cached_data = cache_layer.get(key)
    print(f"获取缓存: {cached_data}")
    
    # 验证数据一致性
    assert cached_data == test_data, "缓存数据不一致"
    
    # 测试缓存存在性
    exists = cache_layer.exists(key)
    print(f"缓存存在: {exists}")
    
    # 测试删除缓存
    deleted = cache_layer.delete(key)
    print(f"删除缓存: {deleted}")
    
    # 验证删除后不存在
    cached_data_after_delete = cache_layer.get(key)
    print(f"删除后获取: {cached_data_after_delete}")
    
    assert cached_data_after_delete is None, "缓存删除失败"
    
    print("✅ 基本缓存操作测试通过")


def test_multi_level_cache():
    """测试多级缓存"""
    print("\n=== 测试多级缓存 ===")
    
    config = get_cache_config(CacheProfile.DEVELOPMENT)
    cache_layer = UnifiedCacheLayer(config)
    
    # 测试数据
    test_data = {"symbol": "AAPL", "price": 150.0, "volume": 1000000}
    key = "test:multi:level"
    
    # 设置到所有级别
    success = cache_layer.set(key, test_data, ttl=600)
    print(f"多级缓存设置: {success}")
    
    # 从缓存获取（应该从内存缓存获取）
    start_time = time.time()
    cached_data = cache_layer.get(key)
    memory_time = time.time() - start_time
    
    print(f"从缓存获取数据: {cached_data}")
    print(f"获取时间: {memory_time:.6f}秒")
    
    # 清空内存缓存，测试从磁盘缓存获取
    cache_layer.clear([CacheLevel.MEMORY])
    
    start_time = time.time()
    cached_data_from_disk = cache_layer.get(key)
    disk_time = time.time() - start_time
    
    print(f"从磁盘缓存获取: {cached_data_from_disk}")
    print(f"磁盘获取时间: {disk_time:.6f}秒")
    
    # 验证数据一致性
    assert cached_data_from_disk == test_data, "磁盘缓存数据不一致"
    
    print("✅ 多级缓存测试通过")


def test_cache_service():
    """测试缓存服务"""
    print("\n=== 测试缓存服务 ===")
    
    # 使用依赖注入容器，使用类型而不是字符串
    container = get_configured_container()
    cache_service = get_service(Data_access_interface)
    
    # 测试股票基础信息缓存
    stock_code = "000001"
    stock_data = {
        "code": stock_code,
        "name": "平安银行",
        "industry": "银行",
        "market_cap": 1000000000
    }
    
    # 设置股票基础信息
    success = cache_service.set_stock_basic(stock_code, stock_data)
    print(f"设置股票基础信息: {success}")
    
    # 获取股票基础信息
    cached_stock_data = cache_service.get_stock_basic(stock_code)
    print(f"获取股票基础信息: {cached_stock_data}")
    
    assert cached_stock_data == stock_data, "股票基础信息缓存不一致"
    
    # 测试行业列表缓存
    industry_data = [
        {"code": "bank", "name": "银行"},
        {"code": "tech", "name": "科技"},
        {"code": "energy", "name": "能源"}
    ]
    
    success = cache_service.set_industry_list(industry_data)
    print(f"设置行业列表: {success}")
    
    cached_industry_data = cache_service.get_industry_list()
    print(f"获取行业列表: {cached_industry_data}")
    
    assert cached_industry_data == industry_data, "行业列表缓存不一致"
    
    # 测试市场概览缓存
    market_data = {
        "date": "2024-01-15",
        "total_stocks": 4000,
        "up_stocks": 2400,
        "down_stocks": 1600
    }
    
    success = cache_service.set_market_overview(market_data, "2024-01-15")
    print(f"设置市场概览: {success}")
    
    cached_market_data = cache_service.get_market_overview("2024-01-15")
    print(f"获取市场概览: {cached_market_data}")
    
    assert cached_market_data == market_data, "市场概览缓存不一致"
    
    print("✅ 缓存服务测试通过")


def test_cache_performance_Cache():
    """测试缓存性能"""
    print("\n=== 测试缓存性能 ===")
    
    config = get_cache_config(CacheProfile.HIGH_PERFORMANCE)
    cache_layer = UnifiedCacheLayer(config)
    
    # 准备测试数据
    test_count = 1000
    test_data = {
        "timestamp": time.time(),
        "data": list(range(100))  # 模拟一些数据
    }
    
    # 测试写入性能
    print(f"测试写入 {test_count} 条记录...")
    start_time = time.time()
    
    for i in range(test_count):
        key = f"perf:test:{i}"
        cache_layer.set(key, test_data, ttl=3600)
    
    write_time = time.time() - start_time
    write_rate = test_count / write_time
    
    print(f"写入完成: {write_time:.3f}秒, 速率: {write_rate:.0f} 记录/秒")
    
    # 测试读取性能
    print(f"测试读取 {test_count} 条记录...")
    start_time = time.time()
    hit_count = 0
    
    for i in range(test_count):
        key = f"perf:test:{i}"
        result = cache_layer.get(key)
        if result is not None:
            hit_count += 1
    
    read_time = time.time() - start_time
    read_rate = test_count / read_time
    hit_rate = hit_count / test_count * 100
    
    print(f"读取完成: {read_time:.3f}秒, 速率: {read_rate:.0f} 记录/秒")
    print(f"命中率: {hit_rate:.1f}%")
    
    # 获取缓存统计
    stats = cache_layer.get_stats()
    print(f"缓存统计: {json.dumps(stats, indent=2, ensure_ascii=False)}")
    
    print("✅ 缓存性能测试完成")


def test_cache_key_builder():
    """测试缓存键构建器"""
    print("\n=== 测试缓存键构建器 ===")
    
    key_builder = Cache_key_builder()
    
    # 测试股票基础信息键
    stock_key = key_builder.build_stock_basic_key("000001")
    print(f"股票基础信息键: {stock_key}")
    assert stock_key == "stock:basic:000001"
    
    # 测试股票日线数据键
    daily_key = key_builder.build_stock_daily_key("000001", "2024-01-01", "2024-01-31")
    print(f"股票日线数据键: {daily_key}")
    assert daily_key == "stock:daily:000001:2024-01-01:2024-01-31"
    
    # 测试指标键
    indicator_params = {"period": 20, "type": "sma"}
    indicator_key = key_builder.build_indicator_key("MA", "000001", "daily", indicator_params)
    print(f"指标键: {indicator_key}")
    assert "indicator:MA:000001:daily:" in indicator_key
    
    # 测试策略键
    strategy_params = {"threshold": 0.05, "period": 5}
    strategy_key = key_builder.build_strategy_key("momentum", strategy_params, "2024-01-15")
    print(f"策略键: {strategy_key}")
    assert "strategy:momentum:" in strategy_key and "2024-01-15" in strategy_key
    
    print("✅ 缓存键构建器测试通过")


def test_cache_expiration():
    """测试缓存过期"""
    print("\n=== 测试缓存过期 ===")
    
    config = get_cache_config(CacheProfile.TESTING)
    cache_layer = UnifiedCacheLayer(config)
    
    # 设置短期缓存
    key = "test:expiration"
    data = {"message": "这是一个测试数据"}
    ttl = 2  # 2秒过期
    
    success = cache_layer.set(key, data, ttl=ttl)
    print(f"设置短期缓存: {success}, TTL: {ttl}秒")
    
    # 立即获取
    cached_data = cache_layer.get(key)
    print(f"立即获取: {cached_data}")
    assert cached_data == data
    
    # 等待过期
    print("等待缓存过期...")
    time.sleep(ttl + 1)
    
    # 过期后获取
    expired_data = cache_layer.get(key)
    print(f"过期后获取: {expired_data}")
    assert expired_data is None, "缓存应该已过期"
    
    print("✅ 缓存过期测试通过")


def main_testunifiedcache():
    """主函数"""
    print("开始统一缓存层测试")
    print("=" * 50)
    
    try:
        # 运行所有测试
        test_basic_cache_operations()
        test_multi_level_cache()
        test_cache_service()
        test_cache_performance_Cache()
        test_cache_key_builder()
        test_cache_expiration()
        
        print("\n" + "=" * 50)
        print("🎉 所有测试通过！统一缓存层工作正常")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        logger.error(f"缓存测试失败: {e}", exc_info=True)
        return False
    
    return True


if __name__ == "__main__":
    success = main_testunifiedcache()
    sys.exit(0 if success else 1) 