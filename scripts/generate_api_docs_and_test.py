#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
API文档生成和测试综合脚本
生成完整的API文档并运行系统集成测试
"""

import os
import sys
import time
import subprocess
import threading
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.logger import get_logger
from utils.exception_handler import exception_handler
from utils.performance_monitor import performance_monitor

logger = get_logger(__name__)

class APIDocumentationAndTestRunner:
    """API文档生成和测试运行器"""
    
    def __init__(self):
        """初始化运行器"""
        self.project_root = project_root
        self.api_server_process = None
        self.server_ready = False
        logger.info("API文档生成和测试运行器初始化完成")
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=10.0)
    def run_complete_workflow(self):
        """运行完整的工作流"""
        logger.info("🚀 开始API文档生成和测试工作流")
        
        try:
            # 1. 生成API文档
            self._generate_api_documentation()
            
            # 2. 启动API服务器
            self._start_api_server()
            
            # 3. 等待服务器就绪
            self._wait_for_server_ready()
            
            # 4. 运行系统集成测试
            integration_result = self._run_integration_tests()
            
            # 5. 运行API文档验证
            documentation_result = self._validate_api_documentation()
            
            # 6. 生成测试报告
            self._generate_test_report(integration_result, documentation_result)
            
            return integration_result and documentation_result
            
        finally:
            # 7. 清理资源
            self._cleanup()
    
    @exception_handler(reraise=True)
    def _generate_api_documentation(self):
        """生成API文档"""
        logger.info("📚 生成API文档")
        
        try:
            # 运行文档生成器
            from docs.api_documentation_generator import APIDocumentationGenerator
            
            generator = APIDocumentationGenerator()
            generator.generate_complete_documentation()
            
            logger.info("✅ API文档生成完成")
            
        except Exception as e:
            logger.error(f"❌ API文档生成失败: {e}")
            raise
    
    @exception_handler(reraise=True)
    def _start_api_server(self):
        """启动API服务器"""
        logger.info("🖥️ 启动API服务器")
        
        try:
            # 启动API服务器进程
            api_script = self.project_root / "api" / "main.py"
            
            self.api_server_process = subprocess.Popen(
                [sys.executable, str(api_script)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.project_root)
            )
            
            logger.info("✅ API服务器启动命令已执行")
            
        except Exception as e:
            logger.error(f"❌ API服务器启动失败: {e}")
            raise
    
    @exception_handler(reraise=True)
    def _wait_for_server_ready(self, timeout: int = 30):
        """等待服务器就绪"""
        logger.info("⏳ 等待API服务器就绪")
        
        import requests
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get("http://localhost:8000/health", timeout=5)
                if response.status_code == 200:
                    self.server_ready = True
                    logger.info("✅ API服务器已就绪")
                    return
                    
            except Exception:
                pass
            
            time.sleep(2)
        
        raise Exception(f"API服务器在{timeout}秒内未就绪")
    
    @exception_handler(reraise=True)
    def _run_integration_tests(self):
        """运行系统集成测试"""
        logger.info("🧪 运行系统集成测试")
        
        try:
            # 运行集成测试
            from tests.test_system_integration import run_system_integration_tests
            
            result = run_system_integration_tests()
            
            if result:
                logger.info("✅ 系统集成测试通过")
            else:
                logger.warning("⚠️ 系统集成测试部分失败")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 系统集成测试失败: {e}")
            return False
    
    @exception_handler(reraise=True)
    def _validate_api_documentation(self):
        """验证API文档"""
        logger.info("📋 验证API文档")
        
        try:
            docs_dir = self.project_root / "docs" / "api"
            
            # 检查文档文件是否存在
            required_files = [
                "api_documentation.md",
                "openapi.json",
                "postman_collection.json",
                "websocket_api.md",
                "integration_guide.md"
            ]
            
            missing_files = []
            for file_name in required_files:
                file_path = docs_dir / file_name
                if not file_path.exists():
                    missing_files.append(file_name)
                else:
                    # 检查文件大小
                    file_size = file_path.stat().st_size
                    if file_size < 100:  # 文件太小可能有问题
                        logger.warning(f"⚠️ 文档文件 {file_name} 可能不完整 ({file_size} bytes)")
            
            if missing_files:
                logger.error(f"❌ 缺少文档文件: {missing_files}")
                return False
            
            logger.info("✅ API文档验证通过")
            return True
            
        except Exception as e:
            logger.error(f"❌ API文档验证失败: {e}")
            return False
    
    @exception_handler(reraise=True)
    def _generate_test_report(self, integration_result: bool, documentation_result: bool):
        """生成测试报告"""
        logger.info("📊 生成测试报告")
        
        try:
            report_content = f"""# API文档和集成测试报告

**生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}

## 测试结果汇总

| 测试项目 | 结果 | 状态 |
|---------|------|------|
| API文档生成 | {'✅ 通过' if documentation_result else '❌ 失败'} | {'正常' if documentation_result else '需要修复'} |
| 系统集成测试 | {'✅ 通过' if integration_result else '❌ 失败'} | {'正常' if integration_result else '需要修复'} |

## 详细信息

### API文档生成
- **状态**: {'成功' if documentation_result else '失败'}
- **生成文件**: 
  - api_documentation.md (完整API文档)
  - openapi.json (OpenAPI规范)
  - postman_collection.json (Postman集合)
  - websocket_api.md (WebSocket文档)
  - integration_guide.md (集成指南)

### 系统集成测试
- **状态**: {'通过' if integration_result else '失败'}
- **测试范围**: 
  - API服务器健康检查
  - RESTful API端点测试
  - WebSocket连接测试
  - 端到端工作流测试

## 总体评估

**整体状态**: {'✅ 系统就绪' if integration_result and documentation_result else '❌ 需要修复'}

### 建议

{'所有测试通过，系统可以投入使用。' if integration_result and documentation_result else '部分测试失败，建议检查失败项目并修复后重新测试。'}

## 文档位置

- **API文档目录**: docs/api/
- **测试报告**: docs/test_reports/
- **集成指南**: docs/api/integration_guide.md

## 下一步

1. 查看详细的API文档
2. 使用Postman集合测试API
3. 参考集成指南进行系统集成
4. 监控系统运行状态

---
**报告生成器**: API文档生成和测试系统  
**版本**: 1.0.0
"""
            
            # 创建报告目录
            report_dir = self.project_root / "docs" / "test_reports"
            report_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存报告
            report_file = report_dir / f"api_docs_and_integration_test_{time.strftime('%Y%m%d_%H%M%S')}.md"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.info(f"✅ 测试报告已生成: {report_file}")
            
        except Exception as e:
            logger.error(f"❌ 测试报告生成失败: {e}")
    
    @exception_handler(reraise=True)
    def _cleanup(self):
        """清理资源"""
        logger.info("🧹 清理资源")
        
        try:
            # 停止API服务器
            if self.api_server_process:
                self.api_server_process.terminate()
                try:
                    self.api_server_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.api_server_process.kill()
                    self.api_server_process.wait()
                
                logger.info("✅ API服务器已停止")
            
        except Exception as e:
            logger.error(f"❌ 资源清理失败: {e}")

def run_api_documentation_and_testing():
    """运行API文档生成和测试"""
    print("🚀 API文档生成和集成测试")
    print("=" * 60)
    
    runner = APIDocumentationAndTestRunner()
    
    try:
        result = runner.run_complete_workflow()
        
        print("\n" + "=" * 60)
        print("🎯 工作流完成结果:")
        print(f"   整体状态: {'✅ 成功' if result else '❌ 失败'}")
        
        if result:
            print("\n📋 后续步骤:")
            print("   1. 查看生成的API文档: docs/api/")
            print("   2. 使用Postman集合测试API")
            print("   3. 参考集成指南进行开发")
            print("   4. 查看测试报告了解详情")
        else:
            print("\n⚠️ 注意事项:")
            print("   1. 检查API服务器是否正常启动")
            print("   2. 确认所有依赖已正确安装")
            print("   3. 查看日志了解具体错误")
            print("   4. 修复问题后重新运行测试")
        
        return result
        
    except Exception as e:
        print(f"\n❌ 工作流执行失败: {e}")
        return False

def main():
    """主函数"""
    print("🧪 API文档生成和集成测试套件")
    print("=" * 60)
    print("📋 工作流程:")
    print("   1. 生成完整的API文档")
    print("   2. 启动API服务器")
    print("   3. 运行系统集成测试")
    print("   4. 验证API文档完整性")
    print("   5. 生成综合测试报告")
    print("=" * 60)
    
    success = run_api_documentation_and_testing()
    
    print("\n" + "=" * 60)
    print(f"🏁 最终结果: {'✅ 成功' if success else '❌ 失败'}")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
