#!/usr/bin/env python3
"""
指标调试验证器

专门用于调试的指标验证脚本，支持：
1. 成功选股后立即停止
2. 遇到错误后立即停止  
3. 详细的调试信息输出
4. 逐个指标验证模式
"""

import os
import sys
import json
from datetime import datetime

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from analysis.engines.indicator_validation_framework import (
    Indicator_validation_framework, 
    Indicator_validation_config, 
    Validation_mode
)
from utils.logger import get_logger

logger = get_logger(__name__)


def create_debug_config(mode='quick', stop_on_success=True, stop_on_error=True):
    """创建调试配置"""
    # 映射验证模式
    mode_mapping = {
        'quick': ValidationMode.QUICK,
        'priority': ValidationMode.PRIORITY,
        'category': ValidationMode.CATEGORY,
        'full': ValidationMode.FULL
    }
    
    return Indicator_validation_config(
        mode=mode_mapping.get(mode, Validation_mode.QUICK),  # 支持动态模式
        stock_pool_size=100,              # 较小的股票池，加快验证速度
        max_selection_ratio=0.2,          # 较宽松的选股比例限制
        min_selection_count=1,            # 最少选出1只股票就算成功
        parallel_workers=1,               # 单线程，确保顺序执行
        stop_on_success=stop_on_success,  # 🔑 成功后立即停止（可配置）
        stop_on_error=stop_on_error,      # 🔑 错误后立即停止（可配置）
        debug_mode=True,                  # 🔑 开启调试模式
        save_details=True,                # 保存详细结果
        output_format="json"              # JSON格式输出
    )


def run_debug_validation(mode='quick', stop_on_success=True, stop_on_error=True):
    """运行调试验证"""
    logger.info("🚀 开始指标调试验证")
    logger.info("=" * 60)
    
    # 创建调试配置
    config = create_debug_config(mode, stop_on_success, stop_on_error)
    
    logger.info("📋 调试配置:")
    logger.info(f"  • 验证模式: {config.mode.value}")
    logger.info(f"  • 股票池大小: {config.stock_pool_size}")
    logger.info(f"  • 成功后停止: {config.stop_on_success}")
    logger.info(f"  • 错误后停止: {config.stop_on_error}")
    logger.info(f"  • 调试模式: {config.debug_mode}")
    logger.info(f"  • 最小选股数: {config.min_selection_count}")
    logger.info(f"  • 最大选股比例: {config.max_selection_ratio}")
    logger.info("-" * 60)
    
    # 创建验证框架
    framework = Indicator_validation_framework(config)
    
    try:
        # 执行验证
        logger.info("🔍 开始验证指标...")
        result = framework.validate_all_indicators()
        
        # 输出结果摘要
        logger.info("=" * 60)
        logger.info("📊 验证结果摘要:")
        
        summary = result['summary']
        logger.info(f"  • 总指标数: {summary['total_indicators']}")
        logger.info(f"  • 已验证数: {summary.get('validated_indicators', 0)}")
        logger.info(f"  • 成功数: {summary.get('successful_validations', 0)}")
        logger.info(f"  • 失败数: {summary.get('failed_validations', 0)}")
        logger.info(f"  • 成功率: {summary.get('success_rate', 0):.2%}")
        logger.info(f"  • 验证耗时: {summary.get('total_duration', 0):.2f}秒")
        
        # 检查是否触发了早停
        if config.stop_on_success or config.stop_on_error:
            actual_validated = len(result['results'])
            total_indicators = summary['total_indicators']
            if actual_validated < total_indicators:
                logger.warning(f"🛑 早停功能已生效！只验证了 {actual_validated}/{total_indicators} 个指标")
                
                # 分析早停原因
                last_result = result['results'][-1] if result['results'] else None
                if last_result:
                    if last_result['status'] == 'success' and config.stop_on_success:
                        logger.info(f"✅ 成功后早停：指标 {last_result['indicator_name']} 选出了 {last_result['selected_count']} 只股票")
                    elif last_result['status'] == 'error' and config.stop_on_error:
                        logger.info(f"❌ 错误后早停：指标 {last_result['indicator_name']} 出现错误")
        
        # 显示成功的指标
        successful_indicators = [
            r for r in result['results'] 
            if r['status'] == 'success'
        ]
        
        if successful_indicators:
            logger.info("✅ 成功的指标:")
            for indicator_result in successful_indicators:
                name = indicator_result['indicator_name']
                count = indicator_result['selected_count']
                ratio = indicator_result['selection_ratio']
                stocks = indicator_result.get('selected_stocks', [])[:5]  # 显示前5只
                logger.info(f"  • {name}: 选出{count}只股票 ({ratio:.2%}) - {stocks}")
        
        # 显示失败的指标
        failed_indicators = [
            r for r in result['results'] 
            if r['status'] != 'success'
        ]
        
        if failed_indicators:
            logger.info("❌ 失败的指标:")
            for indicator_result in failed_indicators:
                name = indicator_result['indicator_name']
                status = indicator_result['status']
                error_msg = indicator_result.get('error_message', '无错误信息')
                logger.info(f"  • {name}: {status} - {error_msg}")
        
        # 保存详细结果（修复JSON序列化问题）
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"results/validation/debug_validation_{timestamp}.json"
            
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # 转换不可序列化的对象
            result_copy = result.copy()
            if 'config' in result_copy:
                config_dict = result_copy['config'].copy()
                if 'mode' in config_dict:
                    config_dict['mode'] = config_dict['mode'].value  # 转换枚举为字符串
                result_copy['config'] = config_dict
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result_copy, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"📄 详细结果已保存到: {output_file}")
        except Exception as save_error:
            logger.warning(f"⚠️ 保存结果文件失败: {save_error}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ 验证过程出错: {e}")
        raise


def run_single_indicator_debug(indicator_name: str):
    """调试单个指标"""
    logger.info(f"🔍 开始调试单个指标: {indicator_name}")
    logger.info("=" * 60)
    
    # 创建调试配置
    config = create_debug_config()
    config.stop_on_success = False  # 单个指标验证不需要早停
    config.stop_on_error = False
    
    # 创建验证框架
    framework = Indicator_validation_framework(config)
    
    try:
        # 验证指标
        result = framework.validate_single_indicator(indicator_name)
        
        # 输出详细结果
        logger.info("📊 验证结果:")
        logger.info(f"  • 指标名称: {result['indicator_name']}")
        logger.info(f"  • 验证状态: {result['status']}")
        logger.info(f"  • 选股数量: {result['selected_count']}")
        logger.info(f"  • 选股比例: {result['selection_ratio']:.4f}")
        logger.info(f"  • 执行时间: {result['execution_time']:.3f}秒")
        
        if result.get('selected_stocks'):
            logger.info(f"  • 选中股票: {result['selected_stocks'][:10]}")
        
        if result.get('error_message'):
            logger.info(f"  • 错误信息: {result['error_message']}")
        
        # 显示策略配置
        if result.get('strategy_config'):
            logger.info("📋 策略配置:")
            strategy = result['strategy_config']
            logger.info(f"  • 策略ID: {strategy.get('strategy_id')}")
            logger.info(f"  • 策略名称: {strategy.get('name')}")
            
            if strategy.get('conditions'):
                logger.info("  • 条件列表:")
                for i, condition in enumerate(strategy['conditions'], 1):
                    logger.info(f"    {i}. {condition.get('description', '无描述')}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ 单个指标验证出错: {e}")
        raise


def main_indicatordebugvalidator():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='指标调试验证器')
    parser.add_argument('--indicator', type=str, help='验证单个指标')
    parser.add_argument('--mode', type=str, choices=['quick', 'priority', 'category', 'full'], 
                        default='quick', help='验证模式')
    parser.add_argument('--no-stop-success', action='store_true', help='成功后不停止')
    parser.add_argument('--no-stop-error', action='store_true', help='错误后不停止')
    
    args = parser.parse_args()
    
    try:
        if args.indicator:
            # 单个指标验证
            run_single_indicator_debug(args.indicator)
        else:
            # 批量验证，根据命令行参数配置早停
            stop_on_success = not args.no_stop_success
            stop_on_error = not args.no_stop_error
            
            if not stop_on_success and not stop_on_error:
                logger.info("⚠️ 注意：已禁用所有早停功能")
            elif not stop_on_success:
                logger.info("⚠️ 注意：已禁用成功后早停功能")
            elif not stop_on_error:
                logger.info("⚠️ 注意：已禁用错误后早停功能")
            
            run_debug_validation(args.mode, stop_on_success, stop_on_error)
            
    except Keyboard_interrupt:
        logger.info("🛑 用户中断验证")
    except Exception as e:
        logger.error(f"❌ 程序执行失败: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main_indicatordebugvalidator()) 