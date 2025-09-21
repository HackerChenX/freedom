#!/usr/bin/env python
"""
VnPy Freedom 统一启动脚本 (彻底修复版)
自动发现并正确配置所有本地vnpy模块，彻底解决导入问题
"""

import sys
from pathlib import Path

# VnPy Freedom 本地开发环境路径配置
VNPY_FREEDOM_ROOT = Path(__file__).parent
CORE_FRAMEWORK_PATH = VNPY_FREEDOM_ROOT / "Core_Framework" / "vnpy"

# 确保本地vnpy路径在最前面，优先于系统安装的vnpy
if str(CORE_FRAMEWORK_PATH) in sys.path:
    sys.path.remove(str(CORE_FRAMEWORK_PATH))
sys.path.insert(0, str(CORE_FRAMEWORK_PATH))

# 自动发现并添加所有模块路径
def discover_and_add_module_paths():
    """自动发现所有vnpy模块并添加到sys.path - 处理依赖关系"""
    module_paths = []
    
    # 策略应用模块路径
    strategy_base = VNPY_FREEDOM_ROOT / "Strategy_Applications"
    for category in strategy_base.iterdir():
        if category.is_dir() and category.name != "__pycache__":
            for module_dir in category.iterdir():
                if module_dir.is_dir() and module_dir.name.startswith("vnpy_"):
                    module_paths.append(str(module_dir))
    
    # 交易网关模块路径
    gateway_base = VNPY_FREEDOM_ROOT / "Trading_Gateways"
    for category in gateway_base.iterdir():
        if category.is_dir() and category.name != "__pycache__":
            for module_dir in category.iterdir():
                if module_dir.is_dir() and module_dir.name.startswith("vnpy_"):
                    module_paths.append(str(module_dir))
    
    # 数据服务模块路径
    data_base = VNPY_FREEDOM_ROOT / "Data_Services"
    for category in data_base.iterdir():
        if category.is_dir() and category.name != "__pycache__":
            for module_dir in category.iterdir():
                if module_dir.is_dir() and module_dir.name.startswith("vnpy_"):
                    module_paths.append(str(module_dir))
    
    # 数据库接口模块路径
    db_base = VNPY_FREEDOM_ROOT / "Database_Interfaces"
    for category in db_base.iterdir():
        if category.is_dir() and category.name != "__pycache__":
            for module_dir in category.iterdir():
                if module_dir.is_dir() and module_dir.name.startswith("vnpy_"):
                    module_paths.append(str(module_dir))
    
    # 工具模块路径
    util_base = VNPY_FREEDOM_ROOT / "Utility_Tools"
    for module_dir in util_base.iterdir():
        if module_dir.is_dir() and module_dir.name.startswith("vnpy_"):
            module_paths.append(str(module_dir))
    
    # 按依赖关系排序添加路径 - 核心依赖优先
    priority_modules = [
        # 核心策略引擎 - 其他模块的依赖
        "vnpy_ctastrategy",
        "vnpy_portfoliostrategy", 
        "vnpy_spreadtrading",
        # 基础工具模块
        "vnpy_algotrading",
        "vnpy_scripttrader",
        # 数据管理模块
        "vnpy_datamanager",
        "vnpy_datarecorder",
        "vnpy_portfoliomanager",
        # 风险管理模块
        "vnpy_riskmanager",
        "vnpy_paperaccount",
        # 系统服务模块
        "vnpy_chartwizard",
    ]
    
    # 首先添加优先级模块
    for priority_module in priority_modules:
        for path in module_paths:
            if priority_module in path:
                if path not in sys.path:
                    sys.path.insert(0, path)
                break
    
    # 然后添加其他模块
    for path in module_paths:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    return module_paths

# 执行模块路径发现
discovered_paths = discover_and_add_module_paths()

from vnpy.event import EventEngine
from vnpy.trader.engine import MainEngine
from vnpy.trader.ui import MainWindow, create_qapp
from vnpy.trader.setting import SETTINGS
# 网关模块映射 (暂时不可用 - 需要编译扩展)
GATEWAYS = {
    # 数据服务网关 - 已可用 ✅
    "rqdata": ("vnpy_rqdata", "RqdataGateway", "米筐数据服务"),
    # 注意：TuShare是数据源，不是交易网关，已移至数据源配置
}

# 应用模块映射 (已修复可用的模块)
APPLICATIONS = {
    # 策略交易引擎 ✅ 已验证
    "cta_strategy": ("vnpy_ctastrategy", "CtaStrategyApp", "CTA策略引擎"),
    "script_trader": ("vnpy_scripttrader", "ScriptTraderApp", "脚本策略模块"),

    # 交易工具 ✅ 已验证
    "algo_trading": ("vnpy_algotrading", "AlgoTradingApp", "算法交易模块"),
    "spread_trading": ("vnpy_spreadtrading", "SpreadTradingApp", "价差交易模块"),

    # 数据管理 ✅ 已验证
    "data_manager": ("vnpy_datamanager", "DataManagerApp", "数据管理器"),
    "data_recorder": ("vnpy_datarecorder", "DataRecorderApp", "行情记录器"),
    "portfolio_manager": ("vnpy_portfoliomanager", "PortfolioManagerApp", "投资组合管理"),

    # 风险管理 ✅ 已验证
    "risk_manager": ("vnpy_riskmanager", "RiskManagerApp", "风险管理器"),
    "paper_account": ("vnpy_paperaccount", "PaperAccountApp", "本地仿真交易"),

    # 系统服务 ✅ 已验证
    "chart_wizard": ("vnpy_chartwizard", "ChartWizardApp", "实时K线图表"),

    # 特殊处理的模块 🔧
    "data_viewer": ("vnpy_dataviewer", "DataViewerApp", "数据查看器 (自定义版本)"),

    # Python 3.11环境下完全可用的模块 ✅
    "cta_backtester": ("vnpy_ctabacktester", "CtaBacktesterApp", "CTA策略回测"),
    "portfolio_strategy": ("vnpy_portfoliostrategy", "PortfolioStrategyApp", "组合策略引擎"),
    # "portfolio_backtester": ("vnpy_portfoliobacktester", "PortfolioBacktesterApp", "组合策略回测"),  # 暂时禁用 - 缺少UI组件
    "rpc_service": ("vnpy_rpcservice", "RpcServiceApp", "RPC分布式服务"),
    "web_trader": ("vnpy_webtrader", "WebTraderApp", "Web交易界面"),
    "excel_rtd": ("vnpy_excelrtd", "ExcelRtdApp", "Excel RTD服务"),
    "option_master": ("vnpy_optionmaster", "OptionMasterApp", "期权主控台"),
}

def load_module(module_name, class_name):
    """智能动态加载模块 - 解决依赖关系版本"""
    
    # 平台兼容性检查
    import platform
    if platform.system() != "Windows" and module_name == "vnpy_xt":
        print(f"⚠️  {module_name} 仅支持Windows平台，当前系统: {platform.system()}")
        return None
    
    # 预加载核心依赖模块
    dependency_map = {
        "vnpy_ctabacktester.vnpy_ctabacktester": ["vnpy_ctastrategy"],
        "vnpy_portfoliobacktester.vnpy_portfoliobacktester": ["vnpy_portfoliostrategy"],
        "vnpy_datarecorder": ["vnpy_spreadtrading"],  # 可选依赖
    }
    
    # 检查并预加载依赖
    if module_name in dependency_map:
        for dependency in dependency_map[module_name]:
            try:
                __import__(dependency)
            except ImportError:
                pass  # 依赖缺失时继续尝试
    
    # 特殊模块的精确导入路径映射
    special_import_mapping = {
        # 数据服务模块需要特殊处理
        "vnpy_tushare.vnpy_tushare": "vnpy_tushare.vnpy_tushare",
        "vnpy_rqdata.vnpy_rqdata": "vnpy_rqdata.vnpy_rqdata",
        "vnpy_efinance.vnpy_efinance": "vnpy_efinance.vnpy_efinance",
        
        # 自定义数据查看器
        "vnpy_dataviewer": "vnpy_dataviewer.data_viewer_app",
        
        # 内部模块结构的应用  
        "vnpy_ctabacktester.vnpy_ctabacktester": "vnpy_ctabacktester.vnpy_ctabacktester",
        "vnpy_portfoliostrategy.vnpy_portfoliostrategy": "vnpy_portfoliostrategy.vnpy_portfoliostrategy", 
        "vnpy_portfoliobacktester.vnpy_portfoliobacktester": "vnpy_portfoliobacktester.vnpy_portfoliobacktester",
        "vnpy_rpcservice.vnpy_rpcservice.rpc_service.engine": "vnpy_rpcservice.vnpy_rpcservice.rpc_service.engine",
        "vnpy_webtrader.vnpy_webtrader": "vnpy_webtrader.vnpy_webtrader",
        "vnpy_excelrtd.vnpy_excelrtd": "vnpy_excelrtd.vnpy_excelrtd",
        "vnpy_optionmaster.vnpy_optionmaster": "vnpy_optionmaster.vnpy_optionmaster",
    }
    
    # 尝试特殊映射路径
    if module_name in special_import_mapping:
        try:
            import_path = special_import_mapping[module_name]
            module = __import__(import_path, fromlist=[class_name])
            target_class = getattr(module, class_name, None)
            if target_class:
                return target_class
        except ImportError as e:
            # 检查是否是第三方依赖缺失
            if "plotly" in str(e) or "requests" in str(e) or "websocket" in str(e):
                print(f"⚠️  {module_name} 需要额外的依赖包: {e}")
            pass
        except Exception as e:
            pass
    
    # 标准导入路径尝试
    possible_paths = [
        module_name,  # 直接导入
        f"{module_name}.{module_name}",  # 内部模块导入
    ]
    
    # 尝试所有可能的导入路径
    for path in possible_paths:
        try:
            module = __import__(path, fromlist=[class_name])
            target_class = getattr(module, class_name, None)
            if target_class:
                return target_class
        except ImportError as e:
            # 检查是否是第三方依赖缺失
            if "plotly" in str(e) or "requests" in str(e) or "websocket" in str(e):
                print(f"⚠️  {module_name} 需要额外的依赖包: {e}")
        except Exception:
            continue
    
    return None

def show_available_modules():
    """显示可用的模块"""
    print("🚀 VnPy Freedom 可用模块:")
    print(f"\n📍 发现模块路径数量: {len(discovered_paths)}")
    
    success_count = 0
    failed_count = 0
    
    print("\n📱 应用模块 (Applications):")
    for key, (module_name, class_name, description) in APPLICATIONS.items():
        try:
            app_class = load_module(module_name, class_name)
            if app_class:
                print(f"  ✅ {key}: {description}")
                success_count += 1
            else:
                print(f"  ❌ {key}: {description} - 类未找到")
                failed_count += 1
        except Exception as e:
            print(f"  ❌ {key}: {description} - 导入异常: {e}")
            failed_count += 1
    
    print("\n📡 交易网关 (Gateways):")
    gateway_success = 0
    gateway_failed = 0
    for key, (module_name, class_name, description) in GATEWAYS.items():
        gateway_class = load_module(module_name, class_name)
        if gateway_class:
            print(f"  ✅ {key}: {description}")
            gateway_success += 1
        else:
            print(f"  ❌ {key}: {description}")
            gateway_failed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 模块加载统计:")
    print(f"   应用模块: ✅ {success_count} / ❌ {failed_count} (成功率: {success_count/(success_count+failed_count)*100:.1f}%)")
    print(f"   网关模块: ✅ {gateway_success} / ❌ {gateway_failed} (成功率: {gateway_success/(gateway_success+gateway_failed)*100:.1f}%)")
    total_success = success_count + gateway_success
    total_failed = failed_count + gateway_failed
    print(f"   总体成功率: {total_success/(total_success+total_failed)*100:.1f}%")

def launch_full_platform():
    """启动完整平台"""
    print("\n🚀 启动完整VnPy Freedom平台...")
    
    qapp = create_qapp()
    event_engine = EventEngine()
    main_engine = MainEngine(event_engine)
    
    # 添加主要交易网关
    gateways_to_add = ["xtp", "ib", "binance"]
    for gateway_key in gateways_to_add:
        if gateway_key in GATEWAYS:
            module_name, class_name, description = GATEWAYS[gateway_key]
            gateway_class = load_module(module_name, class_name)
            if gateway_class:
                main_engine.add_gateway(gateway_class)
                print(f"✅ 已加载: {description}")

    
    # 添加主要应用模块
    apps_to_add = ["cta_strategy", "algo_trading", "data_manager", "risk_manager", "portfolio_strategy", "portfolio_backtester", "cta_backtester", "chart_wizard", "script_trader", "data_recorder"]
    for app_key in apps_to_add:
        if app_key in APPLICATIONS:
            module_name, class_name, description = APPLICATIONS[app_key]
            app_class = load_module(module_name, class_name)
            if app_class:
                main_engine.add_app(app_class)
                print(f"✅ 已加载: {description}")
    
    main_window = MainWindow(main_engine, event_engine)
    main_window.showMaximized()
    
    print("🎉 VnPy Freedom平台启动成功!")
    qapp.exec()

def launch_verified_platform():
    """启动仅包含已验证模块的平台"""
    print("\n🚀 启动已验证模块VnPy Freedom平台...")
    
    qapp = create_qapp()
    event_engine = EventEngine()
    main_engine = MainEngine(event_engine)
    
    # 仅添加已验证可用的应用模块
    verified_apps = ["cta_strategy", "script_trader", "algo_trading", "spread_trading", 
                     "data_manager", "data_recorder", "portfolio_manager", 
                     "risk_manager", "paper_account", "chart_wizard"]
    
    successful_apps = []
    for app_key in verified_apps:
        if app_key in APPLICATIONS:
            module_name, class_name, description = APPLICATIONS[app_key]
            app_class = load_module(module_name, class_name)
            if app_class:
                try:
                    main_engine.add_app(app_class)
                    successful_apps.append(description)
                except Exception as e:
                    print(f"❌ 加载失败 {description}: {e}")
    
    main_window = MainWindow(main_engine, event_engine)
    main_window.showMaximized()
    qapp.exec()

def launch_stock_trading():
    """启动股票交易平台"""
    print("\n📈 启动股票交易平台...")

    qapp = create_qapp()
    event_engine = EventEngine()
    main_engine = MainEngine(event_engine)

    # 添加XTP证券网关
    xtp_gateway = load_module("vnpy_xtp", "XtpGateway")
    if xtp_gateway:
        main_engine.add_gateway(xtp_gateway)
        print("✅ 已加载: XTP证券交易接口")

    # 添加CTA策略应用
    cta_app = load_module("vnpy_ctastrategy", "CtaStrategyApp")
    if cta_app:
        main_engine.add_app(cta_app)
        print("✅ 已加载: CTA策略引擎")

    # 添加数据记录器
    recorder_app = load_module("vnpy_datarecorder", "DataRecorderApp")
    if recorder_app:
        main_engine.add_app(recorder_app)
        print("✅ 已加载: 行情记录器")

    main_window = MainWindow(main_engine, event_engine)
    main_window.showMaximized()

    print("🎉 股票交易平台启动成功!")
    qapp.exec()

def launch_portfolio_trading():
    """启动组合策略交易"""
    print("\n📊 启动组合策略交易平台...")
    
    qapp = create_qapp()
    event_engine = EventEngine()
    main_engine = MainEngine(event_engine)
    
    # 添加多个网关
    gateways = ["xtp", "ib"]
    for gateway_key in gateways:
        if gateway_key in GATEWAYS:
            module_name, class_name, description = GATEWAYS[gateway_key]
            gateway_class = load_module(module_name, class_name)
            if gateway_class:
                main_engine.add_gateway(gateway_class)
                print(f"✅ 已加载: {description}")
    
    # 添加组合策略应用
    portfolio_app = load_module("vnpy_portfoliostrategy", "PortfolioStrategyApp")
    if portfolio_app:
        main_engine.add_app(portfolio_app)
        print("✅ 已加载: 组合策略引擎")
    
    main_window = MainWindow(main_engine, event_engine)
    main_window.showMaximized()
    
    print("🎉 组合策略交易平台启动成功!")
    qapp.exec()

def launch_algo_trading():
    """启动算法交易"""
    print("\n🤖 启动算法交易平台...")

    qapp = create_qapp()
    event_engine = EventEngine()
    main_engine = MainEngine(event_engine)

    # 添加XTP证券网关
    xtp_gateway = load_module("vnpy_xtp", "XtpGateway")
    if xtp_gateway:
        main_engine.add_gateway(xtp_gateway)
        print("✅ 已加载: XTP证券交易接口")

    # 添加算法交易应用
    algo_app = load_module("vnpy_algotrading", "AlgoTradingApp")
    if algo_app:
        main_engine.add_app(algo_app)
        print("✅ 已加载: 算法交易模块")

    main_window = MainWindow(main_engine, event_engine)
    main_window.showMaximized()

    print("🎉 算法交易平台启动成功!")
    qapp.exec()

def launch_data_management():
    """启动数据管理"""
    print("\n📊 启动数据管理平台...")
    
    qapp = create_qapp()
    event_engine = EventEngine()
    main_engine = MainEngine(event_engine)
    
    # 添加数据网关
    rqdata_gateway = load_module("vnpy_rqdata", "RqdataGateway")
    if rqdata_gateway:
        main_engine.add_gateway(rqdata_gateway)
        print("✅ 已加载: 米筐数据服务")
    
    # 添加数据管理应用
    data_manager = load_module("vnpy_datamanager", "DataManagerApp")
    if data_manager:
        main_engine.add_app(data_manager)
        print("✅ 已加载: 数据管理器")
    
    main_window = MainWindow(main_engine, event_engine)
    main_window.showMaximized()
    
    print("🎉 数据管理平台启动成功!")
    qapp.exec()

def launch_custom():
    """自定义启动配置"""
    print("\n⚙️  自定义配置模式")
    print("请选择要加载的网关和应用...")
    
    # 这里可以添加交互式选择逻辑
    print("🚧 自定义配置功能开发中...")
    print("请使用其他预设模式或直接修改启动脚本")

if __name__ == "__main__":
    try:
        launch_full_platform()
    except KeyboardInterrupt:
        print("\n👋 用户中断，程序退出")
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        import traceback
        traceback.print_exc()
