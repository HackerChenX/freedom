"""
集成测试与部署模块

提供与现有股票分析系统的集成功能
"""

try:
    from crawler.integration.system_integrator import SystemIntegrator
except ImportError:
    SystemIntegrator = None

try:
    from crawler.integration.deployment_manager import DeploymentManager
except ImportError:
    DeploymentManager = None

try:
    from crawler.integration.integration_tester import IntegrationTester
except ImportError:
    IntegrationTester = None

__all__ = [
    'SystemIntegrator',
    'DeploymentManager',
    'IntegrationTester'
]