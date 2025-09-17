"""
系统统一入口设计
整合所有功能模块，提供统一的命令行接口和API入口
"""

import sys
import os
import argparse
import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path

# 添加项目根目录到Python路径
root_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(root_dir))

from utils.logger import get_logger
from utils.dependency_injection import get_container
from utils.decorators import exception_handler, performance_monitor
from db.interfaces.data_access_interface import DataAccessInterface, RealDataValidator
from config.unified_config_manager import UnifiedConfigManager


logger = get_logger(__name__)


class FreedomTradingSystem:
    """
    Freedom交易系统统一入口

    整合股票分析、回测、监控、API等所有功能模块
    提供统一的命令行接口和程序化访问接口
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化交易系统

        Args:
            config_path: 配置文件路径，None使用默认配置
        """
        self.config_manager = UnifiedConfigManager(config_path)
        self.container = get_container()

        # 初始化核心服务
        self._init_core_services()

        # 验证数据源真实性
        self._validate_data_sources()

        logger.info("Freedom交易系统初始化完成")

    def _init_core_services(self):
        """初始化核心服务"""
        try:
            # 数据访问服务
            self.data_access: DataAccessInterface = self.container.resolve('data_access')

            # 指标计算服务
            self.indicator_registry = self.container.resolve('indicator_registry')

            # 策略管理服务
            self.strategy_manager = self.container.resolve('strategy_manager')

            # 监控服务
            self.monitoring_service = self.container.resolve('monitoring_service')

            # API服务
            self.api_service = self.container.resolve('api_service')

            logger.info("核心服务初始化完成")

        except Exception as e:
            logger.error(f"核心服务初始化失败: {e}")
            raise

    def _validate_data_sources(self):
        """验证数据源真实性"""
        try:
            if not RealDataValidator.validate_data_source(self.data_access):
                raise ValueError("检测到模拟数据源，系统仅支持真实数据")

            logger.info("数据源验证通过")

        except Exception as e:
            logger.error(f"数据源验证失败: {e}")
            raise

    @exception_handler(reraise=True)
    @performance_monitor(threshold=10.0)
    def run_analysis(self, mode: str, **kwargs) -> Dict[str, Any]:
        """
        运行股票分析

        Args:
            mode: 分析模式 ('market', 'stock', 'industry', 'technical')
            **kwargs: 分析参数

        Returns:
            分析结果字典

        Examples:
            # 市场分析
            result = system.run_analysis('market', date='2023-12-01')

            # 个股分析
            result = system.run_analysis('stock', code='000001', start_date='2023-11-01', end_date='2023-12-01')

            # 技术指标分析
            result = system.run_analysis('technical', code='000001', indicators=['RSI', 'MACD'])
        """
        logger.info(f"开始执行{mode}分析")

        if mode == 'market':
            return self._run_market_analysis(**kwargs)
        elif mode == 'stock':
            return self._run_stock_analysis(**kwargs)
        elif mode == 'industry':
            return self._run_industry_analysis(**kwargs)
        elif mode == 'technical':
            return self._run_technical_analysis(**kwargs)
        elif mode == 'buypoint':
            return self._run_buypoint_analysis(**kwargs)
        else:
            raise ValueError(f"不支持的分析模式: {mode}")

    @exception_handler(reraise=True)
    @performance_monitor(threshold=30.0)
    def run_backtest(self, strategy: str, **kwargs) -> Dict[str, Any]:
        """
        运行策略回测

        Args:
            strategy: 策略名称或配置
            **kwargs: 回测参数

        Returns:
            回测结果字典

        Examples:
            # ZXM策略回测
            result = system.run_backtest('zxm', period='30d', stocks=['000001', '000002'])

            # 自定义策略回测
            result = system.run_backtest('custom', config='strategy_config.yaml')
        """
        logger.info(f"开始执行{strategy}策略回测")

        return self.strategy_manager.run_backtest(
            strategy_name=strategy,
            data_access=self.data_access,
            **kwargs
        )

    @exception_handler(reraise=True)
    def run_monitoring(self, mode: str = 'realtime', **kwargs) -> Dict[str, Any]:
        """
        运行市场监控

        Args:
            mode: 监控模式 ('realtime', 'batch', 'alert')
            **kwargs: 监控参数

        Returns:
            监控结果字典

        Examples:
            # 实时监控
            result = system.run_monitoring('realtime', symbols=['000001', '000002'])

            # 批量监控
            result = system.run_monitoring('batch', date='2023-12-01')

            # 告警监控
            result = system.run_monitoring('alert', rules=['price_change', 'volume_spike'])
        """
        logger.info(f"开始执行{mode}监控")

        return self.monitoring_service.run_monitoring(
            mode=mode,
            data_access=self.data_access,
            **kwargs
        )

    @exception_handler(reraise=True)
    async def start_api_server(self, host: str = '0.0.0.0', port: int = 8000,
                              **kwargs) -> None:
        """
        启动API服务器

        Args:
            host: 服务器地址
            port: 服务器端口
            **kwargs: 其他服务器配置

        Examples:
            # 启动API服务
            await system.start_api_server(port=8000)
        """
        logger.info(f"启动API服务器 {host}:{port}")

        await self.api_service.start_server(
            host=host,
            port=port,
            trading_system=self,
            **kwargs
        )

    def _run_market_analysis(self, **kwargs) -> Dict[str, Any]:
        """运行市场分析"""
        from analysis.market.a_stock_market_analysis import Market_analyzer

        date = kwargs.get('date', datetime.now().strftime('%Y-%m-%d'))
        analyzer = Market_analyzer(date, self.data_access)
        analyzer.calculate_market_strength()

        return {
            'type': 'market_analysis',
            'date': date,
            'market_strength': analyzer.market_strength,
            'statistics': analyzer.get_market_statistics()
        }

    def _run_stock_analysis(self, **kwargs) -> Dict[str, Any]:
        """运行个股分析"""
        code = kwargs.get('code')
        if not code:
            raise ValueError("个股分析需要指定股票代码")

        start_date = kwargs.get('start_date')
        end_date = kwargs.get('end_date', datetime.now().strftime('%Y-%m-%d'))

        # 获取股票数据
        stock_data = self.data_access.get_stock_data(code, start_date, end_date)

        # 计算技术指标
        indicators = self.indicator_registry.calculate_all_indicators(stock_data)

        return {
            'type': 'stock_analysis',
            'code': code,
            'period': f"{start_date} to {end_date}",
            'data_points': len(stock_data),
            'indicators': indicators
        }

    def _run_industry_analysis(self, **kwargs) -> Dict[str, Any]:
        """运行行业分析"""
        industry = kwargs.get('industry')
        date = kwargs.get('date', datetime.now().strftime('%Y-%m-%d'))

        # 获取行业股票列表
        stocks = self.data_access.get_stock_list(industry=industry)

        # 批量分析
        results = []
        for stock in stocks:
            try:
                result = self._run_stock_analysis(code=stock, end_date=date)
                results.append(result)
            except Exception as e:
                logger.warning(f"分析股票{stock}失败: {e}")

        return {
            'type': 'industry_analysis',
            'industry': industry,
            'date': date,
            'stocks_analyzed': len(results),
            'results': results
        }

    def _run_technical_analysis(self, **kwargs) -> Dict[str, Any]:
        """运行技术指标分析"""
        code = kwargs.get('code')
        indicators = kwargs.get('indicators', [])

        if not code:
            raise ValueError("技术分析需要指定股票代码")

        # 获取股票数据
        stock_data = self.data_access.get_stock_data(code, kwargs.get('start_date'), kwargs.get('end_date'))

        # 计算指定指标
        indicator_results = {}
        for indicator in indicators:
            try:
                result = self.indicator_registry.calculate_indicator(indicator, stock_data)
                indicator_results[indicator] = result
            except Exception as e:
                logger.warning(f"计算指标{indicator}失败: {e}")

        return {
            'type': 'technical_analysis',
            'code': code,
            'indicators': indicator_results
        }

    def _run_buypoint_analysis(self, **kwargs) -> Dict[str, Any]:
        """运行买点分析"""
        from analysis.buypoints.enhanced_buypoint_detector import EnhancedBuypointDetector

        detector = EnhancedBuypointDetector(self.data_access)

        code = kwargs.get('code')
        codes = kwargs.get('codes', [])

        if code:
            codes = [code]
        elif not codes:
            # 获取所有股票列表
            codes = self.data_access.get_stock_list()

        results = detector.batch_detect_buypoints(codes, **kwargs)

        return {
            'type': 'buypoint_analysis',
            'stocks_analyzed': len(codes),
            'buypoints_found': len([r for r in results if r.get('buypoints')]),
            'results': results
        }

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'status': 'running',
            'config': self.config_manager.get_system_config(),
            'services': {
                'data_access': self.data_access.__class__.__name__,
                'indicator_registry': self.indicator_registry.__class__.__name__,
                'strategy_manager': self.strategy_manager.__class__.__name__,
                'monitoring_service': self.monitoring_service.__class__.__name__
            },
            'data_validation': RealDataValidator.validate_data_source(self.data_access)
        }


def create_cli_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description='Freedom Trading System - 统一交易系统入口',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 市场分析
  python main.py analysis market --date 2023-12-01

  # 个股分析
  python main.py analysis stock --code 000001 --start-date 2023-11-01

  # 策略回测
  python main.py backtest zxm --period 30d

  # 启动监控
  python main.py monitor realtime --symbols 000001,000002

  # 启动API服务器
  python main.py api --port 8000

  # 查看系统状态
  python main.py status
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='命令类型')

    # 分析命令
    analysis_parser = subparsers.add_parser('analysis', help='运行分析')
    analysis_parser.add_argument('mode', choices=['market', 'stock', 'industry', 'technical', 'buypoint'])
    analysis_parser.add_argument('--code', help='股票代码')
    analysis_parser.add_argument('--codes', help='股票代码列表，逗号分隔')
    analysis_parser.add_argument('--start-date', help='开始日期')
    analysis_parser.add_argument('--end-date', help='结束日期')
    analysis_parser.add_argument('--date', help='分析日期')
    analysis_parser.add_argument('--industry', help='行业筛选')
    analysis_parser.add_argument('--indicators', help='技术指标列表，逗号分隔')

    # 回测命令
    backtest_parser = subparsers.add_parser('backtest', help='运行策略回测')
    backtest_parser.add_argument('strategy', help='策略名称')
    backtest_parser.add_argument('--period', help='回测周期')
    backtest_parser.add_argument('--stocks', help='股票列表，逗号分隔')
    backtest_parser.add_argument('--config', help='策略配置文件')

    # 监控命令
    monitor_parser = subparsers.add_parser('monitor', help='运行监控')
    monitor_parser.add_argument('mode', choices=['realtime', 'batch', 'alert'])
    monitor_parser.add_argument('--symbols', help='监控股票列表，逗号分隔')
    monitor_parser.add_argument('--date', help='批量监控日期')
    monitor_parser.add_argument('--rules', help='告警规则列表，逗号分隔')

    # API命令
    api_parser = subparsers.add_parser('api', help='启动API服务器')
    api_parser.add_argument('--host', default='0.0.0.0', help='服务器地址')
    api_parser.add_argument('--port', type=int, default=8000, help='服务器端口')

    # 状态命令
    subparsers.add_parser('status', help='查看系统状态')

    return parser


async def main():
    """主入口函数"""
    parser = create_cli_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # 初始化系统
    try:
        system = FreedomTradingSystem()
        logger.info("系统初始化成功")
    except Exception as e:
        logger.error(f"系统初始化失败: {e}")
        return

    try:
        if args.command == 'analysis':
            # 处理分析命令
            kwargs = {}
            if args.code:
                kwargs['code'] = args.code
            if args.codes:
                kwargs['codes'] = args.codes.split(',')
            if args.start_date:
                kwargs['start_date'] = args.start_date
            if args.end_date:
                kwargs['end_date'] = args.end_date
            if args.date:
                kwargs['date'] = args.date
            if args.industry:
                kwargs['industry'] = args.industry
            if args.indicators:
                kwargs['indicators'] = args.indicators.split(',')

            result = system.run_analysis(args.mode, **kwargs)
            print(f"分析完成: {result}")

        elif args.command == 'backtest':
            # 处理回测命令
            kwargs = {}
            if args.period:
                kwargs['period'] = args.period
            if args.stocks:
                kwargs['stocks'] = args.stocks.split(',')
            if args.config:
                kwargs['config'] = args.config

            result = system.run_backtest(args.strategy, **kwargs)
            print(f"回测完成: {result}")

        elif args.command == 'monitor':
            # 处理监控命令
            kwargs = {}
            if args.symbols:
                kwargs['symbols'] = args.symbols.split(',')
            if args.date:
                kwargs['date'] = args.date
            if args.rules:
                kwargs['rules'] = args.rules.split(',')

            result = system.run_monitoring(args.mode, **kwargs)
            print(f"监控完成: {result}")

        elif args.command == 'api':
            # 启动API服务器
            await system.start_api_server(host=args.host, port=args.port)

        elif args.command == 'status':
            # 显示系统状态
            status = system.get_system_status()
            print(f"系统状态: {status}")

    except Exception as e:
        logger.error(f"命令执行失败: {e}")
        raise


if __name__ == '__main__':
    # 运行主程序
    asyncio.run(main())