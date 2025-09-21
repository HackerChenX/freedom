"""
VeighNa Trader启动脚本
按照框架标准方式启动，支持ClickHouse数据库
"""

from vnpy.event import EventEngine
from vnpy.trader.engine import MainEngine
from vnpy.trader.ui import MainWindow, create_qapp

# 导入交易接口（根据实际安装的模块启用）
from vnpy_ctp import CtpGateway
from vnpy_ctptest import CtptestGateway
from vnpy_mini import MiniGateway
from vnpy_femas import FemasGateway
from vnpy_sopt import SoptGateway
from vnpy_uft import UftGateway
from vnpy_esunny import EsunnyGateway
from vnpy_xtp import XtpGateway
from vnpy_tora import ToraStockGateway, ToraOptionGateway
from vnpy_ib import IbGateway
from vnpy_tap import TapGateway
from vnpy_da import DaGateway
from vnpy_rohon import RohonGateway
from vnpy_tts import TtsGateway

# 导入应用模块（根据实际安装的模块启用）
from vnpy_paperaccount import PaperAccountApp
from vnpy_ctastrategy import CtaStrategyApp
from vnpy_ctabacktester import CtaBacktesterApp
from vnpy_spreadtrading import SpreadTradingApp
from vnpy_algotrading import AlgoTradingApp
from vnpy_optionmaster import OptionMasterApp
from vnpy_portfoliostrategy import PortfolioStrategyApp
from vnpy_scripttrader import ScriptTraderApp
from vnpy_chartwizard import ChartWizardApp
from vnpy_rpcservice import RpcServiceApp
from vnpy_excelrtd import ExcelRtdApp
from vnpy_datamanager import DataManagerApp
from vnpy_datarecorder import DataRecorderApp
from vnpy_riskmanager import RiskManagerApp
from vnpy_webtrader import WebTraderApp
from vnpy_portfoliomanager import PortfolioManagerApp


def main():
    """主函数"""
    # 创建Qt应用
    qapp = create_qapp()

    # 创建事件引擎
    event_engine = EventEngine()

    # 创建主引擎
    main_engine = MainEngine(event_engine)

    # 添加交易接口（根据实际安装的模块启用）
    main_engine.add_gateway(CtpGateway)
    main_engine.add_gateway(CtptestGateway)
    main_engine.add_gateway(MiniGateway)
    main_engine.add_gateway(FemasGateway)
    main_engine.add_gateway(SoptGateway)
    main_engine.add_gateway(UftGateway)
    main_engine.add_gateway(EsunnyGateway)
    main_engine.add_gateway(XtpGateway)
    main_engine.add_gateway(ToraStockGateway)
    main_engine.add_gateway(ToraOptionGateway)
    main_engine.add_gateway(IbGateway)
    main_engine.add_gateway(TapGateway)
    main_engine.add_gateway(DaGateway)
    main_engine.add_gateway(RohonGateway)
    main_engine.add_gateway(TtsGateway)

    # 添加应用模块（根据实际安装的模块启用）
    main_engine.add_app(PaperAccountApp)
    main_engine.add_app(CtaStrategyApp)
    main_engine.add_app(CtaBacktesterApp)
    main_engine.add_app(SpreadTradingApp)
    main_engine.add_app(AlgoTradingApp)
    main_engine.add_app(OptionMasterApp)
    main_engine.add_app(PortfolioStrategyApp)
    main_engine.add_app(ScriptTraderApp)
    main_engine.add_app(ChartWizardApp)
    main_engine.add_app(RpcServiceApp)
    main_engine.add_app(ExcelRtdApp)
    main_engine.add_app(DataManagerApp)
    main_engine.add_app(DataRecorderApp)
    main_engine.add_app(RiskManagerApp)
    main_engine.add_app(WebTraderApp)
    main_engine.add_app(PortfolioManagerApp)

    # 创建主窗口
    main_window = MainWindow(main_engine, event_engine)
    main_window.showMaximized()

    # 运行应用
    qapp.exec()


if __name__ == "__main__":
    main()
