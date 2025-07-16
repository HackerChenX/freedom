#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Freedom股票选择工具

统一的策略选股命令行工具，支持多种执行模式和配置选项
"""

import os
import sys
import argparse
import json
import yaml
import pandas as pd
from datetime import datetime
import time
from typing import Dict, List, Optional, Any
from pathlib import Path

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from strategy.strategy_parser import StrategyParser
from strategy.strategy_executor import StrategyExecutor
from strategy.strategy_manager import StrategyManager
from strategy.signal_watcher import SignalWatcher
from strategy.result_filter import ResultFilter
from indicators.complete_indicator_registry import complete_registry
from utils.path_utils import get_strategy_dir
from db.unified_data_manager import get_unified_data_manager
from utils.logger import get_logger, init_logging
from utils.path_utils import get_result_dir
from utils.exceptions import (
    StrategyExecutionError, 
    StrategyValidationError, 
    DataAccessError
)

# 设置全局变量
import builtins
builtins.complete_registry = complete_registry

logger = get_logger(__name__)


class FreedomSelectError(Exception):
    """Freedom选股工具异常"""
    def __init__(self, message: str, code: str = None, context: dict = None):
        super().__init__(message)
        self.code = code
        self.context = context or {}


class FreedomSelect:
    """
    Freedom选股工具主类
    
    统一的策略选股系统，支持多种执行模式和配置选项
    """
    
    def __init__(self, config_path: str = None):
        """
        初始化选股工具
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.data_manager = None
        self.strategy_executor = None
        self.strategy_manager = None
        self.signal_watcher = None
        self.result_filter = None
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        default_config = {
            "database": {
                "strict_mode": True,
                "connection_timeout": 30
            },
            "execution": {
                "max_threads": None,
                "default_output_format": "csv",
                "max_results": 1000,
                "timeout": 300
            },
            "output": {
                "auto_timestamp": True,
                "include_summary": True,
                "display_count": 10
            },
            "validation": {
                "validate_strategy": True,
                "check_database": True,
                "error_on_empty": False
            }
        }
        
        if not config_path or not os.path.exists(config_path):
            return default_config
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.endswith('.json'):
                    file_config = json.load(f)
                elif config_path.endswith(('.yaml', '.yml')):
                    file_config = yaml.safe_load(f)
                else:
                    logger.warning(f"不支持的配置文件格式: {config_path}")
                    return default_config
                    
            # 合并配置
            return self._merge_config(default_config, file_config)
            
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return default_config
    
    def _merge_config(self, default: dict, override: dict) -> dict:
        """递归合并配置"""
        result = default.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_config(result[key], value)
            else:
                result[key] = value
        return result
    
    def _initialize_components(self, args) -> None:
        """初始化Freedom选股组件"""
        logger.info("🔧 初始化Freedom选股组件...")
        
        # 首先初始化依赖注入服务
        logger.info("🔧 初始化依赖注入服务...")
        try:
            from config.service_initializer import initialize_all_services
            initialize_all_services()
            logger.info("✅ 依赖注入服务初始化成功")
        except Exception as e:
            logger.warning(f"⚠️ 依赖注入服务初始化失败: {e}")
            logger.info("继续使用默认组件初始化...")
        
        # 然后初始化各个组件
        self.strategy_manager = StrategyManager()
        self.strategy_executor = StrategyExecutor(
            max_workers=args.threads or self.config["execution"]["max_threads"]
        )
        self.signal_watcher = SignalWatcher()
        self.result_filter = ResultFilter()
        
        logger.info("✅ 组件初始化完成")
    
    def _check_database_connection(self) -> None:
        """检查数据库连接"""
        logger.info("🔍 检查数据库连接...")
        try:
            self.data_manager = get_unified_data_manager()
            self.data_manager.test_connection()
            logger.info("✅ 数据库连接正常")
        except Exception as e:
            error_msg = f"数据库连接失败: {e}"
            logger.error(f"❌ {error_msg}")
            if self.config["database"]["strict_mode"]:
                raise FreedomSelectError(
                    f"严格模式下数据库连接失败，请检查ClickHouse服务状态",
                    code="DB_CONNECTION_ERROR",
                    context={"original_error": str(e)}
                )
    
    def _is_strategy_file(self, strategy_path: str) -> bool:
        """判断是否为策略文件路径"""
        if not strategy_path:
            return False
        
        if os.path.exists(strategy_path) and os.path.isfile(strategy_path):
            ext = os.path.splitext(strategy_path)[1].lower()
            return ext in ['.json', '.yaml', '.yml']
        
        return False
    
    def _load_strategy_from_file(self, strategy_path: str) -> Dict[str, Any]:
        """从文件加载策略"""
        logger.info(f"📂 从文件加载策略: {strategy_path}")
        
        try:
            parser = StrategyParser()
            strategy_plan = parser.parse_from_file(strategy_path)
            
            # 验证策略
            if self.config["validation"]["validate_strategy"]:
                self._validate_strategy(strategy_plan)
            
            # 输出策略信息
            self._print_strategy_info(strategy_plan)
            
            return strategy_plan
            
        except Exception as e:
            raise FreedomSelectError(
                f"加载策略文件失败: {e}",
                code="STRATEGY_LOAD_ERROR",
                context={"strategy_path": strategy_path}
            )
    
    def _load_strategy_by_id(self, strategy_id: str) -> Dict[str, Any]:
        """通过ID加载策略"""
        logger.info(f"🔍 通过ID加载策略: {strategy_id}")
        
        try:
            strategy_config = self.strategy_manager.get_strategy(strategy_id)
            if not strategy_config:
                raise FreedomSelectError(
                    f"未找到策略: {strategy_id}",
                    code="STRATEGY_NOT_FOUND",
                    context={"strategy_id": strategy_id}
                )
            
            return strategy_config
            
        except Exception as e:
            raise FreedomSelectError(
                f"加载策略失败: {e}",
                code="STRATEGY_LOAD_ERROR",
                context={"strategy_id": strategy_id}
            )
    
    def _validate_strategy(self, strategy_plan: Dict[str, Any]) -> None:
        """验证策略配置"""
        required_fields = ['strategy_id', 'name', 'conditions']
        
        for field in required_fields:
            if field not in strategy_plan:
                raise FreedomSelectError(
                    f"策略配置缺少必要字段: {field}",
                    code="STRATEGY_VALIDATION_ERROR",
                    context={"missing_field": field}
                )
        
        conditions = strategy_plan.get('conditions', [])
        if not conditions:
            raise FreedomSelectError(
                "策略配置缺少有效条件",
                code="STRATEGY_VALIDATION_ERROR",
                context={"conditions_count": len(conditions)}
            )
    
    def _print_strategy_info(self, strategy_plan: Dict[str, Any]) -> None:
        """打印策略信息"""
        print("\n" + "="*60)
        print("📋 策略信息")
        print("="*60)
        print(f"策略ID: {strategy_plan.get('strategy_id', 'N/A')}")
        print(f"策略名称: {strategy_plan.get('name', 'N/A')}")
        print(f"策略描述: {strategy_plan.get('description', 'N/A')}")
        print(f"条件数量: {len(strategy_plan.get('conditions', []))}")
        
        if logger.level <= 20:  # INFO level
            print("\n📝 条件详情:")
            for i, cond in enumerate(strategy_plan.get('conditions', []), 1):
                print(f"  {i}. {cond}")
        
        print("="*60)
    
    def _execute_strategy(self, strategy_plan: Dict[str, Any], args) -> pd.DataFrame:
        """执行策略"""
        logger.info("🚀 开始执行策略...")
        
        progress_callback = self._create_progress_callback(args.verbose)
        
        try:
            result_df = self.strategy_executor.execute_strategy_Executor_Strategy_Executor(
                strategy_plan=strategy_plan,
                end_date=args.date,
                progress_callback=progress_callback
            )
            
            logger.info(f"✅ 策略执行完成，找到 {len(result_df)} 只股票")
            return result_df
            
        except Exception as e:
            raise FreedomSelectError(
                f"策略执行失败: {e}",
                code="STRATEGY_EXECUTION_ERROR",
                context={"strategy_id": strategy_plan.get('strategy_id')}
            )
    
    def _execute_strategy_by_id(self, strategy_id: str, args) -> pd.DataFrame:
        """通过ID执行策略"""
        logger.info("🚀 开始执行策略...")
        
        progress_callback = self._create_progress_callback(args.verbose)
        
        try:
            result_df = self.strategy_executor.execute_strategy_by_id(
                strategy_id=strategy_id,
                strategy_manager=self.strategy_manager,
                end_date=args.date,
                progress_callback=progress_callback
            )
            
            logger.info(f"✅ 策略执行完成，找到 {len(result_df)} 只股票")
            return result_df
            
        except Exception as e:
            raise FreedomSelectError(
                f"策略执行失败: {e}",
                code="STRATEGY_EXECUTION_ERROR",
                context={"strategy_id": strategy_id}
            )
    
    def _create_progress_callback(self, verbose: bool):
        """创建进度回调函数"""
        if not verbose:
            return None
            
        def progress_callback(progress: float, message: str):
            bar_length = 40
            filled_length = int(bar_length * progress)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            percent = progress * 100
            
            sys.stdout.write(f'\r📊 进度: |{bar}| {percent:.1f}% {message}')
            sys.stdout.flush()
            
            if progress >= 1:
                sys.stdout.write('\n')
        
        return progress_callback
    
    def _post_process_results(self, result_df: pd.DataFrame, args) -> pd.DataFrame:
        """后处理结果"""
        if result_df.empty:
            if self.config["validation"]["error_on_empty"]:
                raise FreedomSelectError(
                    "策略执行结果为空",
                    code="EMPTY_RESULTS",
                    context={"strategy": args.strategy}
                )
            else:
                logger.warning("⚠️ 策略执行结果为空")
                return result_df
        
        # 添加观察信号
        if args.watch:
            logger.info("🔍 添加观察信号...")
            result_df = self.signal_watcher.add_watch_signals(result_df)
        
        # 应用过滤器
        result_df = self._apply_filters(result_df, args)
        
        # 排序
        result_df = self._sort_results(result_df, args)
        
        # 限制结果数量
        if args.limit > 0 and len(result_df) > args.limit:
            result_df = result_df.iloc[:args.limit]
            logger.info(f"📊 结果已限制为前 {args.limit} 只股票")
        
        return result_df
    
    def _apply_filters(self, result_df: pd.DataFrame, args) -> pd.DataFrame:
        """应用过滤器"""
        filter_config = {
            'market_cap': {},
            'price': {},
        }
        
        # 市值过滤
        if args.min_cap is not None:
            filter_config['market_cap']['min'] = args.min_cap
        if args.max_cap is not None:
            filter_config['market_cap']['max'] = args.max_cap
        
        # 价格过滤
        if args.min_price is not None:
            filter_config['price']['min'] = args.min_price
        if args.max_price is not None:
            filter_config['price']['max'] = args.max_price
        
        # 行业过滤
        if args.industry:
            filter_config['industry'] = args.industry.split(',')
        
        # 市场过滤
        if args.market:
            filter_config['market'] = args.market.split(',')
        
        # 应用过滤器
        filtered_df = self.result_filter.apply_filters(result_df, filter_config)
        
        if len(filtered_df) != len(result_df):
            logger.info(f"🔍 过滤器已应用，从 {len(result_df)} 只股票筛选出 {len(filtered_df)} 只")
        
        return filtered_df
    
    def _sort_results(self, result_df: pd.DataFrame, args) -> pd.DataFrame:
        """排序结果"""
        sort_field = args.sort_by or self.config["output"].get("sort_field", "score")
        ascending = args.sort_order.upper() != 'DESC'
        
        sorted_df = self.result_filter.sort_results(
            result_df, 
            sort_field=sort_field, 
            ascending=ascending
        )
        
        return sorted_df
    
    def _output_results(self, result_df: pd.DataFrame, args) -> None:
        """输出结果"""
        if result_df.empty:
            print("\n❌ 未找到符合条件的股票")
            return
        
        # 确定输出路径
        output_path = self._get_output_path(args)
        
        # 输出到文件
        self._save_results(result_df, output_path, args.format)
        
        # 显示摘要
        if self.config["output"]["include_summary"]:
            self._print_summary(result_df, args)
    
    def _get_output_path(self, args) -> str:
        """获取输出路径"""
        if args.output:
            return args.output
        
        # 生成默认输出路径
        result_dir = get_result_dir()
        os.makedirs(result_dir, exist_ok=True)
        
        if self.config["output"]["auto_timestamp"]:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"freedom_select_{timestamp}.{args.format}"
        else:
            filename = f"freedom_select.{args.format}"
        
        return os.path.join(result_dir, filename)
    
    def _save_results(self, result_df: pd.DataFrame, output_path: str, format_type: str) -> None:
        """保存结果到文件"""
        try:
            # 确保目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # 根据格式输出
            if format_type.lower() == 'csv':
                result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            elif format_type.lower() == 'excel':
                result_df.to_excel(output_path, index=False, engine='openpyxl')
            elif format_type.lower() == 'json':
                # 处理非原生JSON类型
                result_json = result_df.copy()
                for col in result_json.columns:
                    if result_json[col].dtype == 'object':
                        result_json[col] = result_json[col].apply(
                            lambda x: json.dumps(x, ensure_ascii=False) if not isinstance(x, str) else x
                        )
                result_json.to_json(output_path, orient='records', force_ascii=False, indent=2)
            elif format_type.lower() == 'html':
                result_df.to_html(output_path, index=False, encoding='utf-8')
            else:
                logger.warning(f"不支持的输出格式: {format_type}，使用CSV格式")
                result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            
            logger.info(f"📁 结果已保存至: {output_path}")
            print(f"\n💾 结果已保存至: {output_path}")
            
        except Exception as e:
            raise FreedomSelectError(
                f"保存结果失败: {e}",
                code="OUTPUT_ERROR",
                context={"output_path": output_path, "format": format_type}
            )
    
    def _print_summary(self, result_df: pd.DataFrame, args) -> None:
        """打印结果摘要"""
        print(f"\n📊 共找到 {len(result_df)} 只符合条件的股票")
        
        # 显示前N只股票
        display_count = min(self.config["output"]["display_count"], len(result_df))
        
        if display_count > 0:
            print(f"\n🔝 前 {display_count} 只股票:")
            print("-" * 60)
            
            for i in range(display_count):
                row = result_df.iloc[i]
                stock_code = row.get('stock_code', 'N/A')
                stock_name = row.get('stock_name', 'N/A')
                
                # 尝试获取评分或信号强度
                score = row.get('score', row.get('signal_strength', 0))
                
                print(f"{i+1:2d}. {stock_code} {stock_name} "
                      f"(评分: {score:.2f})")
            
            if len(result_df) > display_count:
                print(f"\n... 及其他 {len(result_df) - display_count} 只股票")
    
    def list_strategies(self) -> None:
        """列出可用策略"""
        print("\n📋 可用策略列表:")
        print("="*60)
        
        try:
            strategies = self.strategy_manager.list_strategies()
            
            if not strategies:
                print("❌ 未找到任何策略")
                return
            
            for i, strategy in enumerate(strategies, 1):
                strategy_id = strategy.get('id', 'N/A')
                name = strategy.get('name', 'N/A')
                description = strategy.get('description', '')
                
                print(f"{i:2d}. {strategy_id}")
                print(f"    名称: {name}")
                if description:
                    print(f"    描述: {description}")
                print()
                
        except Exception as e:
            logger.error(f"获取策略列表失败: {e}")
            print(f"❌ 获取策略列表失败: {e}")
    
    def validate_strategy(self, strategy_path: str) -> None:
        """验证策略文件"""
        print(f"\n🔍 验证策略文件: {strategy_path}")
        print("="*60)
        
        try:
            if not os.path.exists(strategy_path):
                print(f"❌ 策略文件不存在: {strategy_path}")
                return
            
            # 加载策略
            strategy_plan = self._load_strategy_from_file(strategy_path)
            
            # 验证策略
            self._validate_strategy(strategy_plan)
            
            print("✅ 策略验证通过")
            
        except Exception as e:
            print(f"❌ 策略验证失败: {e}")
    
    def run(self, args) -> None:
        """运行选股工具"""
        try:
            # 初始化组件
            self._initialize_components(args)
            
            # 检查特殊命令
            if args.list_strategies:
                self.list_strategies()
                return
            
            if args.validate_strategy:
                self.validate_strategy(args.strategy)
                return
            
            # 加载策略
            if self._is_strategy_file(args.strategy):
                strategy_plan = self._load_strategy_from_file(args.strategy)
                result_df = self._execute_strategy(strategy_plan, args)
            else:
                result_df = self._execute_strategy_by_id(args.strategy, args)
            
            # 后处理结果
            result_df = self._post_process_results(result_df, args)
            
            # 输出结果
            self._output_results(result_df, args)
            
        except FreedomSelectError as e:
            logger.error(f"Freedom选股错误 [{e.code}]: {e}")
            print(f"\n❌ {e}")
            if args.verbose and e.context:
                print(f"   详情: {e.context}")
            sys.exit(1)
        
        except Exception as e:
            logger.error(f"未预期的错误: {e}")
            print(f"\n❌ 执行失败: {e}")
            sys.exit(1)


def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="Freedom股票选择工具 - 统一的策略选股系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 使用策略ID执行选股
  freedom_select --strategy my_strategy
  
  # 使用策略文件执行选股
  freedom_select --strategy /path/to/strategy.json --output results.csv
  
  # 列出可用策略
  freedom_select --list-strategies
  
  # 验证策略文件
  freedom_select --validate-strategy /path/to/strategy.json
  
  # 高级选项
  freedom_select --strategy my_strategy --date 2024-01-01 --limit 50 --verbose
        """
    )
    
    # 主要参数
    parser.add_argument(
        "--strategy", "-s", 
        help="策略ID或策略文件路径", 
        required=False
    )
    
    # 特殊命令
    parser.add_argument(
        "--list-strategies", 
        action="store_true", 
        help="列出可用策略"
    )
    
    parser.add_argument(
        "--validate-strategy", 
        action="store_true", 
        help="验证策略文件"
    )
    
    # 执行参数
    parser.add_argument(
        "--date", "-d", 
        help="选股日期 (YYYY-MM-DD)", 
        default=datetime.now().strftime("%Y-%m-%d")
    )
    
    parser.add_argument(
        "--limit", "-l", 
        type=int, 
        help="限制结果数量", 
        default=0
    )
    
    parser.add_argument(
        "--threads", "-t", 
        type=int, 
        help="最大线程数", 
        default=None
    )
    
    # 输出参数
    parser.add_argument(
        "--output", "-o", 
        help="输出文件路径", 
        default=None
    )
    
    parser.add_argument(
        "--format", "-f", 
        help="输出格式", 
        choices=['csv', 'excel', 'json', 'html'], 
        default="csv"
    )
    
    # 过滤参数
    filter_group = parser.add_argument_group("过滤参数")
    filter_group.add_argument("--min-cap", type=float, help="最小市值 (亿元)")
    filter_group.add_argument("--max-cap", type=float, help="最大市值 (亿元)")
    filter_group.add_argument("--min-price", type=float, help="最小价格")
    filter_group.add_argument("--max-price", type=float, help="最大价格")
    filter_group.add_argument("--industry", help="行业名称 (逗号分隔)")
    filter_group.add_argument("--market", help="市场名称 (逗号分隔)")
    
    # 排序参数
    parser.add_argument("--sort-by", help="排序字段", default="score")
    parser.add_argument("--sort-order", help="排序顺序", choices=['ASC', 'DESC'], default="DESC")
    
    # 其他参数
    parser.add_argument("--watch", "-w", action="store_true", help="包含观察信号")
    parser.add_argument("--config", "-c", help="配置文件路径")
    parser.add_argument("--log-level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default="INFO")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示详细信息")
    
    return parser


def main():
    """主函数"""
    parser = create_parser()
    args = parser.parse_args()
    
    # 检查参数
    if not args.list_strategies and not args.validate_strategy and not args.strategy:
        parser.error("必须指定 --strategy 或使用 --list-strategies / --validate-strategy")
    
    # 设置日志级别
    init_logging(level=args.log_level)
    
    # 创建并运行选股工具
    selector = FreedomSelect(config_path=args.config)
    selector.run(args)


if __name__ == "__main__":
    main()