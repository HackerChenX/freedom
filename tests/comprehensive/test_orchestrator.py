#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试执行编排器

负责协调整个综合选股测试的执行流程，包括测试恢复、结果导出和备份机制
支持中断恢复、阶段性保存和完整的测试生命周期管理
"""

import os
import json
import pickle
import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import threading
import queue
import signal

from utils.logger import getLogger
from .stock_selection_tester import ComprehensiveStockSelectionTester as StockSelectionTester, TestResults
from .test_config_manager import TestConfig, TestConfigManager
from .test_result_validator import TestResultValidator, ComprehensiveValidationResult
from .enhanced_report_generator import EnhancedReportGenerator
from .performance_monitor import PerformanceMonitor

logger = getLogger(__name__)


@dataclass
class TestPhase:
    """测试阶段"""
    name: str
    description: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    status: str = "pending"  # pending, running, completed, failed, skipped
    progress: float = 0.0
    error_message: Optional[str] = None
    results: Optional[Any] = None


@dataclass
class TestSession:
    """测试会话"""
    session_id: str
    config: TestConfig
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"  # running, completed, failed, interrupted
    phases: List[TestPhase] = field(default_factory=list)
    results: Optional[TestResults] = None
    validation_result: Optional[ComprehensiveValidationResult] = None
    checkpoint_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class TestOrchestrator:
    """测试执行编排器"""
    
    def __init__(self, 
                 config: TestConfig,
                 workspace_dir: str = "test_workspace",
                 enable_checkpoints: bool = True):
        """
        初始化测试编排器
        
        Args:
            config: 测试配置
            workspace_dir: 工作空间目录
            enable_checkpoints: 是否启用检查点
        """
        self.config = config
        self.workspace_dir = Path(workspace_dir)
        self.enable_checkpoints = enable_checkpoints
        
        # 创建工作空间
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = self.workspace_dir / "checkpoints"
        self.results_dir = self.workspace_dir / "results"
        self.logs_dir = self.workspace_dir / "logs"
        
        for dir_path in [self.checkpoints_dir, self.results_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 初始化组件
        self.tester = StockSelectionTester()
        self.validator = TestResultValidator()
        self.report_generator = EnhancedReportGenerator()
        self.performance_monitor = PerformanceMonitor()
        
        # 会话管理
        self.current_session: Optional[TestSession] = None
        self.progress_callback: Optional[Callable] = None
        self.interrupted = False
        
        # 线程安全
        self.lock = threading.Lock()
        self.progress_queue = queue.Queue()
        
        # 设置信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"测试编排器初始化完成，工作空间: {self.workspace_dir}")
    
    def _signal_handler(self, signum, frame):
        """信号处理器"""
        logger.warning(f"收到中断信号 {signum}")
        self.interrupted = True
        
        if self.current_session:
            self._save_checkpoint("interrupted")
            self.current_session.status = "interrupted"
    
    def execute_comprehensive_test(self, 
                                 progress_callback: Optional[Callable] = None) -> TestSession:
        """
        执行综合测试
        
        Args:
            progress_callback: 进度回调函数
            
        Returns:
            TestSession: 测试会话
        """
        self.progress_callback = progress_callback
        
        # 创建测试会话
        session_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_session = TestSession(
            session_id=session_id,
            config=self.config,
            start_time=datetime.now()
        )
        
        logger.info(f"开始执行综合测试，会话ID: {session_id}")
        
        try:
            # 定义测试阶段
            phases = [
                TestPhase("initialization", "初始化测试环境"),
                TestPhase("indicator_discovery", "发现指标和形态"),
                TestPhase("stock_selection", "执行选股测试"),
                TestPhase("verification", "执行闭环验证"),
                TestPhase("validation", "验证测试结果"),
                TestPhase("reporting", "生成测试报告"),
                TestPhase("cleanup", "清理和归档")
            ]
            
            self.current_session.phases = phases
            
            # 执行各个阶段
            for phase in phases:
                if self.interrupted:
                    break
                
                self._execute_phase(phase)
                
                # 保存检查点
                if self.enable_checkpoints and phase.status == "completed":
                    self._save_checkpoint(phase.name)
            
            # 完成测试
            self._finalize_test_session()
            
        except Exception as e:
            logger.error(f"执行综合测试失败: {e}")
            self.current_session.status = "failed"
            self._save_checkpoint("failed")
            raise
        
        return self.current_session
    
    def _execute_phase(self, phase: TestPhase):
        """
        执行测试阶段
        
        Args:
            phase: 测试阶段对象
        """
        logger.info(f"开始执行阶段: {phase.name} - {phase.description}")
        
        phase.start_time = datetime.now()
        phase.status = "running"
        
        self._report_progress(f"执行阶段: {phase.description}", {
            'current_phase': phase.name,
            'phase_progress': 0.0
        })
        
        try:
            # 检查是否需要中断
            if self.interrupted:
                logger.warning(f"阶段 {phase.name} 被中断")
                phase.status = "interrupted"
                return
            
            # 执行阶段
            if phase.name == "initialization":
                self._phase_initialization(phase)
            elif phase.name == "indicator_discovery":
                self._phase_indicator_discovery(phase)
            elif phase.name == "stock_selection":
                self._phase_stock_selection(phase)
            elif phase.name == "verification":
                self._phase_verification(phase)
            elif phase.name == "validation":
                self._phase_validation(phase)
            elif phase.name == "reporting":
                self._phase_reporting(phase)
            elif phase.name == "cleanup":
                self._phase_cleanup(phase)
            
            phase.status = "completed"
            phase.progress = 1.0
            
            # 保存阶段检查点
            if self.enable_checkpoints:
                self._save_checkpoint(phase.name)
            
        except Exception as e:
            logger.error(f"阶段 {phase.name} 执行失败: {e}")
            phase.status = "failed"
            phase.error_message = str(e)
            
            # 保存失败检查点
            if self.enable_checkpoints:
                self._save_checkpoint(f"{phase.name}_failed")
            
            raise
        
        finally:
            phase.end_time = datetime.now()
            duration = (phase.end_time - phase.start_time).total_seconds()
            logger.info(f"阶段 {phase.name} 完成，耗时: {duration:.1f}秒，状态: {phase.status}")
            
            # 更新会话状态
            self._update_session_status()
    
    def _phase_initialization(self, phase: TestPhase):
        """初始化阶段"""
        # 启动性能监控
        self.performance_monitor.start_monitoring()
        
        # 验证配置
        config_manager = TestConfigManager()
        config_manager.config = self.config
        validation_errors = config_manager.validate_config()
        
        if validation_errors:
            raise Exception(f"配置验证失败: {validation_errors}")
        
        # 初始化测试器
        self.tester.initialize()
        
        phase.results = {
            'config_validated': True,
            'components_initialized': True,
            'monitoring_started': True
        }
        
        self._report_progress("初始化完成", {'phase_progress': 1.0})
    
    def _phase_indicator_discovery(self, phase: TestPhase):
        """指标发现阶段"""
        # 发现指标和形态
        discovery_results = self.tester.discover_indicators_and_patterns(
            indicators_filter=self.config.indicators_to_test,
            patterns_filter=self.config.patterns_to_test
        )
        
        phase.results = discovery_results
        
        self._report_progress(f"发现 {discovery_results['total_indicators']} 个指标，{discovery_results['total_patterns']} 个形态", {
            'phase_progress': 1.0,
            'indicators_discovered': discovery_results['total_indicators'],
            'patterns_discovered': discovery_results['total_patterns']
        })
    
    def _phase_stock_selection(self, phase: TestPhase):
        """选股测试阶段"""
        # 执行选股测试
        selection_results = self.tester.execute_stock_selection(
            start_date=self.config.test_scope.date_range.start_date,
            end_date=self.config.test_scope.date_range.end_date,
            progress_callback=lambda status, data: self._report_progress(
                f"选股测试: {status}", 
                {**data, 'phase_progress': data.get('progress', 0.0)}
            )
        )
        
        phase.results = selection_results
        
        self._report_progress(f"选股完成，共选出 {selection_results.get('total_stocks_selected', 0)} 只股票", {
            'phase_progress': 1.0,
            'stocks_selected': selection_results.get('total_stocks_selected', 0)
        })
    
    def _phase_verification(self, phase: TestPhase):
        """验证阶段"""
        # 执行闭环验证
        verification_results = self.tester.execute_verification(
            progress_callback=lambda status, data: self._report_progress(
                f"闭环验证: {status}",
                {**data, 'phase_progress': data.get('progress', 0.0)}
            )
        )
        
        phase.results = verification_results
        
        self._report_progress(f"验证完成，成功率: {verification_results.get('success_rate', 0):.1%}", {
            'phase_progress': 1.0,
            'verification_success_rate': verification_results.get('success_rate', 0)
        })
    
    def _phase_validation(self, phase: TestPhase):
        """结果验证阶段"""
        # 获取测试结果
        test_results = self.tester.get_test_results()
        self.current_session.results = test_results
        
        # 验证测试结果
        validation_result = self.validator.validate_test_results(test_results)
        self.current_session.validation_result = validation_result
        
        phase.results = {
            'validation_passed': validation_result.overall_validation_passed,
            'validation_score': validation_result.validation_score,
            'recommendations_count': len(validation_result.recommendations)
        }
        
        self._report_progress(f"结果验证完成，验证通过: {'是' if validation_result.overall_validation_passed else '否'}", {
            'phase_progress': 1.0,
            'validation_passed': validation_result.overall_validation_passed,
            'validation_score': validation_result.validation_score
        })
    
    def _phase_reporting(self, phase: TestPhase):
        """报告生成阶段"""
        if not self.current_session.results:
            raise Exception("没有测试结果可用于生成报告")
        
        # 生成各种格式的报告
        report_results = {}
        
        for format_type in self.config.reporting.output_formats:
            try:
                if format_type == 'json':
                    report_path = self.report_generator.generate_json_report(
                        self.current_session.results, 
                        str(self.results_dir)
                    )
                elif format_type == 'csv':
                    report_path = self.report_generator.generate_csv_report(
                        self.current_session.results,
                        str(self.results_dir)
                    )
                elif format_type == 'html':
                    report_path = self.report_generator.generate_html_report(
                        self.current_session.results,
                        str(self.results_dir)
                    )
                
                report_results[format_type] = report_path
                
            except Exception as e:
                logger.error(f"生成 {format_type} 报告失败: {e}")
                report_results[format_type] = f"失败: {str(e)}"
        
        # 保存验证报告
        if self.current_session.validation_result:
            validation_report_path = self.results_dir / "validation_report.json"
            with open(validation_report_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'test_id': self.current_session.validation_result.test_id,
                    'validation_time': self.current_session.validation_result.validation_time.isoformat(),
                    'overall_validation_passed': self.current_session.validation_result.overall_validation_passed,
                    'validation_score': self.current_session.validation_result.validation_score,
                    'summary_statistics': self.current_session.validation_result.summary_statistics,
                    'recommendations': self.current_session.validation_result.recommendations
                }, f, ensure_ascii=False, indent=2)
            
            report_results['validation'] = str(validation_report_path)
        
        phase.results = report_results
        
        self._report_progress(f"报告生成完成，共生成 {len(report_results)} 个报告", {
            'phase_progress': 1.0,
            'reports_generated': len(report_results)
        })
    
    def _phase_cleanup(self, phase: TestPhase):
        """清理阶段"""
        # 停止性能监控
        self.performance_monitor.stop_monitoring()
        
        # 保存会话信息
        session_file = self.results_dir / f"session_{self.current_session.session_id}.json"
        self._save_session_info(session_file)
        
        # 清理临时文件
        cleanup_results = {
            'session_saved': True,
            'monitoring_stopped': True,
            'temp_files_cleaned': 0
        }
        
        phase.results = cleanup_results
        
        self._report_progress("清理完成", {'phase_progress': 1.0})
    
    def _update_session_status(self):
        """更新会话状态"""
        if not self.current_session:
            return
        
        # 确定当前状态
        if self.interrupted:
            self.current_session.status = "interrupted"
        elif any(phase.status == "failed" for phase in self.current_session.phases):
            self.current_session.status = "failed"
        elif all(phase.status == "completed" for phase in self.current_session.phases):
            self.current_session.status = "completed"
        else:
            self.current_session.status = "running"
        
        # 计算总体进度
        completed_phases = sum(1 for phase in self.current_session.phases if phase.status == "completed")
        total_phases = len(self.current_session.phases)
        progress = completed_phases / total_phases if total_phases > 0 else 0
        
        # 更新元数据
        self.current_session.metadata['progress'] = progress
        self.current_session.metadata['last_updated'] = datetime.now().isoformat()
        self.current_session.metadata['active_phase'] = next(
            (phase.name for phase in self.current_session.phases if phase.status == "running"),
            None
        )
    
    def _finalize_test_session(self):
        """完成测试会话"""
        if not self.current_session:
            return
        
        self.current_session.end_time = datetime.now()
        
        # 确定最终状态
        if self.interrupted:
            self.current_session.status = "interrupted"
        elif any(phase.status == "failed" for phase in self.current_session.phases):
            self.current_session.status = "failed"
        else:
            self.current_session.status = "completed"
        
        # 更新元数据
        self.current_session.metadata['completion_time'] = self.current_session.end_time.isoformat()
        self.current_session.metadata['final_status'] = self.current_session.status
        
        # 计算统计信息
        completed_phases = sum(1 for phase in self.current_session.phases if phase.status == "completed")
        failed_phases = sum(1 for phase in self.current_session.phases if phase.status == "failed")
        
        self.current_session.metadata['statistics'] = {
            'total_phases': len(self.current_session.phases),
            'completed_phases': completed_phases,
            'failed_phases': failed_phases,
            'success_rate': completed_phases / len(self.current_session.phases) if len(self.current_session.phases) > 0 else 0
        }
        
        # 保存最终检查点
        self._save_checkpoint("final")
        
        # 保存会话信息
        session_file = self.results_dir / f"session_{self.current_session.session_id}.json"
        self._save_session_info(session_file)
        
        duration = (self.current_session.end_time - self.current_session.start_time).total_seconds()
        logger.info(f"测试会话完成，状态: {self.current_session.status}，耗时: {duration:.1f}秒")
    
    def _save_checkpoint(self, checkpoint_name: str):
        """
        保存检查点
        
        Args:
            checkpoint_name: 检查点名称
        """
        if not self.enable_checkpoints or not self.current_session:
            return
        
        try:
            checkpoint_file = self.checkpoints_dir / f"{self.current_session.session_id}_{checkpoint_name}.pkl"
            
            # 收集当前测试状态
            if self.tester:
                # 保存阶段性结果到会话的检查点数据
                if checkpoint_name == "indicator_discovery" or checkpoint_name == "resume_indicator_discovery":
                    self.current_session.checkpoint_data['indicator_discovery'] = self.tester.get_discovery_results()
                
                elif checkpoint_name == "stock_selection" or checkpoint_name == "resume_stock_selection":
                    self.current_session.checkpoint_data['stock_selection'] = self.tester.get_selection_results()
                
                elif checkpoint_name == "verification" or checkpoint_name == "resume_verification":
                    self.current_session.checkpoint_data['verification'] = self.tester.get_verification_results()
                
                # 如果有测试结果，保存它
                if hasattr(self.tester, 'get_test_results'):
                    test_results = self.tester.get_test_results()
                    if test_results:
                        self.current_session.results = test_results
            
            # 创建检查点数据
            checkpoint_data = {
                'session': self.current_session,
                'timestamp': datetime.now(),
                'checkpoint_name': checkpoint_name,
                'metadata': {
                    'hostname': os.uname().nodename if hasattr(os, 'uname') else 'unknown',
                    'python_version': sys.version,
                    'checkpoint_version': '1.0'
                }
            }
            
            # 保存检查点
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            logger.info(f"检查点已保存: {checkpoint_file}")
            
            # 保存检查点索引
            self._update_checkpoint_index(checkpoint_name, checkpoint_file)
            
        except Exception as e:
            logger.error(f"保存检查点失败: {e}")
    
    def _update_checkpoint_index(self, checkpoint_name: str, checkpoint_file: Path) -> None:
        """
        更新检查点索引
        
        Args:
            checkpoint_name: 检查点名称
            checkpoint_file: 检查点文件路径
        """
        try:
            index_file = self.checkpoints_dir / "checkpoint_index.json"
            
            # 读取现有索引
            if index_file.exists():
                with open(index_file, 'r', encoding='utf-8') as f:
                    index_data = json.load(f)
            else:
                index_data = {}
            
            # 更新会话检查点
            session_id = self.current_session.session_id
            if session_id not in index_data:
                index_data[session_id] = {
                    'session_id': session_id,
                    'start_time': self.current_session.start_time.isoformat(),
                    'checkpoints': {}
                }
            
            # 添加检查点信息
            index_data[session_id]['checkpoints'][checkpoint_name] = {
                'file': checkpoint_file.name,
                'timestamp': datetime.now().isoformat(),
                'size': checkpoint_file.stat().st_size
            }
            
            # 更新最新状态
            index_data[session_id]['latest_checkpoint'] = checkpoint_name
            index_data[session_id]['latest_timestamp'] = datetime.now().isoformat()
            
            # 保存索引
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"更新检查点索引失败: {e}")
    
    def _save_session_info(self, session_file: Path):
        """保存会话信息"""
        try:
            session_info = {
                'session_id': self.current_session.session_id,
                'start_time': self.current_session.start_time.isoformat(),
                'end_time': self.current_session.end_time.isoformat() if self.current_session.end_time else None,
                'status': self.current_session.status,
                'phases': [
                    {
                        'name': phase.name,
                        'description': phase.description,
                        'start_time': phase.start_time.isoformat() if phase.start_time else None,
                        'end_time': phase.end_time.isoformat() if phase.end_time else None,
                        'status': phase.status,
                        'progress': phase.progress,
                        'error_message': phase.error_message
                    }
                    for phase in self.current_session.phases
                ],
                'metadata': self.current_session.metadata
            }
            
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_info, f, ensure_ascii=False, indent=2)
            
            logger.info(f"会话信息已保存: {session_file}")
            
        except Exception as e:
            logger.error(f"保存会话信息失败: {e}")
    
    def _report_progress(self, status: str, progress_data: Dict[str, Any] = None):
        """报告进度"""
        if self.progress_callback:
            try:
                self.progress_callback(status, progress_data or {})
            except Exception as e:
                logger.error(f"进度回调失败: {e}")
    
    def resume_from_checkpoint(self, session_id: str, checkpoint_name: str = None) -> TestSession:
        """从检查点恢复测试"""
        logger.info(f"尝试从检查点恢复测试，会话ID: {session_id}")
        
        # 查找检查点文件
        if checkpoint_name:
            checkpoint_file = self.checkpoints_dir / f"{session_id}_{checkpoint_name}.pkl"
        else:
            # 查找最新的检查点
            checkpoint_files = list(self.checkpoints_dir.glob(f"{session_id}_*.pkl"))
            if not checkpoint_files:
                raise Exception(f"未找到会话 {session_id} 的检查点文件")
            
            checkpoint_file = max(checkpoint_files, key=lambda f: f.stat().st_mtime)
        
        if not checkpoint_file.exists():
            raise Exception(f"检查点文件不存在: {checkpoint_file}")
        
        # 加载检查点
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            self.current_session = checkpoint_data['session']
            logger.info(f"已从检查点恢复: {checkpoint_data['checkpoint_name']}")
            
            # 继续执行剩余阶段
            return self._resume_execution()
            
        except Exception as e:
            logger.error(f"从检查点恢复失败: {e}")
            raise
    
    def _resume_execution(self) -> TestSession:
        """
        恢复执行测试会话
        
        Returns:
            TestSession: 恢复的测试会话
        """
        if not self.current_session:
            raise Exception("没有可恢复的会话")
        
        logger.info(f"恢复执行会话: {self.current_session.session_id}")
        
        # 找到下一个待执行的阶段
        next_phase_index = None
        for i, phase in enumerate(self.current_session.phases):
            if phase.status in ["pending", "failed"]:
                next_phase_index = i
                break
        
        if next_phase_index is None:
            logger.info("所有阶段已完成，无需恢复")
            return self.current_session
        
        # 恢复测试状态
        self._restore_test_state()
        
        # 继续执行剩余阶段
        try:
            for i in range(next_phase_index, len(self.current_session.phases)):
                if self.interrupted:
                    break
                
                phase = self.current_session.phases[i]
                
                # 如果是失败的阶段，重置状态
                if phase.status == "failed":
                    phase.status = "pending"
                    phase.error_message = None
                
                self._execute_phase(phase)
                
                # 保存检查点
                if self.enable_checkpoints and phase.status == "completed":
                    self._save_checkpoint(f"resume_{phase.name}")
                    
                # 报告进度
                self._report_progress(f"恢复执行: 完成阶段 {phase.name}", {
                    'phase': phase.name,
                    'status': phase.status,
                    'progress': (i + 1) / len(self.current_session.phases)
                })
            
            # 完成测试
            self._finalize_test_session()
            
        except Exception as e:
            logger.error(f"恢复执行失败: {e}")
            self.current_session.status = "failed"
            self._save_checkpoint("resume_failed")
            raise
        
        return self.current_session
    
    def _restore_test_state(self) -> None:
        """恢复测试状态"""
        logger.info("恢复测试状态")
        
        # 重新初始化测试器
        self.tester = StockSelectionTester()
        
        # 如果有测试结果，恢复它
        if self.current_session.results:
            self.tester.restore_test_results(self.current_session.results)
            logger.info("已恢复测试结果")
        
        # 恢复检查点数据
        if self.current_session.checkpoint_data:
            # 恢复指标和形态发现结果
            if 'indicator_discovery' in self.current_session.checkpoint_data:
                discovery_results = self.current_session.checkpoint_data['indicator_discovery']
                self.tester.restore_discovery_results(discovery_results)
                logger.info("已恢复指标和形态发现结果")
            
            # 恢复选股结果
            if 'stock_selection' in self.current_session.checkpoint_data:
                selection_results = self.current_session.checkpoint_data['stock_selection']
                self.tester.restore_selection_results(selection_results)
                logger.info("已恢复选股结果")
            
            # 恢复验证结果
            if 'verification' in self.current_session.checkpoint_data:
                verification_results = self.current_session.checkpoint_data['verification']
                self.tester.restore_verification_results(verification_results)
                logger.info("已恢复验证结果")
        
        logger.info("测试状态恢复完成")
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """列出所有测试会话"""
        sessions = []
        
        for session_file in self.results_dir.glob("session_*.json"):
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    session_info = json.load(f)
                sessions.append(session_info)
            except Exception as e:
                logger.error(f"读取会话文件失败 {session_file}: {e}")
        
        return sorted(sessions, key=lambda s: s['start_time'], reverse=True)
    
    def export_results(self, session_id: str, export_path: str, format_type: str = "zip"):
        """
        导出测试结果
        
        Args:
            session_id: 会话ID
            export_path: 导出路径
            format_type: 导出格式 (zip, json, csv)
            
        Returns:
            str: 导出文件路径
        """
        logger.info(f"导出会话 {session_id} 的结果到 {export_path}")
        
        # 查找会话文件
        session_file = self.results_dir / f"session_{session_id}.json"
        if not session_file.exists():
            raise Exception(f"会话文件不存在: {session_file}")
        
        # 创建导出目录
        export_dir = Path(export_path)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        if format_type == "zip":
            import zipfile
            import shutil
            
            # 创建临时目录
            temp_dir = self.workspace_dir / "temp_export"
            temp_dir.mkdir(exist_ok=True)
            
            try:
                # 复制会话文件
                shutil.copy(session_file, temp_dir)
                
                # 复制报告文件
                for report_file in self.results_dir.glob(f"*{session_id}*"):
                    shutil.copy(report_file, temp_dir)
                
                # 复制检查点文件
                for checkpoint_file in self.checkpoints_dir.glob(f"{session_id}_*"):
                    shutil.copy(checkpoint_file, temp_dir)
                
                # 复制日志文件
                for log_file in self.logs_dir.glob(f"*{session_id}*"):
                    shutil.copy(log_file, temp_dir)
                
                # 创建ZIP文件
                zip_file = export_dir / f"{session_id}_results.zip"
                with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for file in temp_dir.glob("*"):
                        zipf.write(file, arcname=file.name)
                
                logger.info(f"测试结果已导出到: {zip_file}")
                return str(zip_file)
                
            finally:
                # 清理临时目录
                shutil.rmtree(temp_dir, ignore_errors=True)
                
        elif format_type == "json":
            # 读取会话信息
            with open(session_file, 'r', encoding='utf-8') as f:
                session_info = json.load(f)
            
            # 查找相关报告
            reports = {}
            for report_file in self.results_dir.glob(f"*{session_id}*"):
                if report_file.name.endswith('.json') and report_file.name != session_file.name:
                    with open(report_file, 'r', encoding='utf-8') as f:
                        reports[report_file.name] = json.load(f)
            
            # 合并导出
            export_data = {
                'session': session_info,
                'reports': reports,
                'export_time': datetime.now().isoformat()
            }
            
            export_file = export_dir / f"{session_id}_results.json"
            with open(export_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"测试结果已导出到: {export_file}")
            return str(export_file)
            
        elif format_type == "csv":
            import csv
            
            # 导出会话信息
            export_file = export_dir / f"{session_id}_session.csv"
            
            with open(session_file, 'r', encoding='utf-8') as f:
                session_info = json.load(f)
            
            with open(export_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['属性', '值'])
                writer.writerow(['会话ID', session_info.get('session_id', '')])
                writer.writerow(['开始时间', session_info.get('start_time', '')])
                writer.writerow(['结束时间', session_info.get('end_time', '')])
                writer.writerow(['状态', session_info.get('status', '')])
                
                writer.writerow([])
                writer.writerow(['阶段', '描述', '状态', '开始时间', '结束时间'])
                
                for phase in session_info.get('phases', []):
                    writer.writerow([
                        phase.get('name', ''),
                        phase.get('description', ''),
                        phase.get('status', ''),
                        phase.get('start_time', ''),
                        phase.get('end_time', '')
                    ])
            
            logger.info(f"测试结果已导出到: {export_file}")
            return str(export_file)
            
        else:
            raise ValueError(f"不支持的导出格式: {format_type}")
            
        return None


def create_orchestrator(config: TestConfig, 
                       workspace_dir: str = "test_workspace") -> TestOrchestrator:
    """
    创建测试编排器
    
    Args:
        config: 测试配置
        workspace_dir: 工作空间目录
        
    Returns:
        TestOrchestrator: 测试编排器实例
    """
    return TestOrchestrator(config, workspace_dir)


def main():
    """测试编排器测试"""
    print("测试编排器功能测试...")
    
    # 创建测试配置
    from .test_config_manager import TestConfigManager
    config_manager = TestConfigManager()
    config = config_manager.get_config()
    
    # 创建编排器
    orchestrator = create_orchestrator(config)
    
    print(f"编排器工作空间: {orchestrator.workspace_dir}")
    print("测试编排器初始化成功")


if __name__ == "__main__":
    main()