"""交易引擎 - FreqTrade 集成和执行管理"""

import subprocess
import sys
import threading
import time
import signal
from pathlib import Path
from typing import Optional, Dict, Any, Callable
import logging

logger = logging.getLogger(__name__)

class TradingEngine:
    """交易引擎"""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.process: Optional[subprocess.Popen] = None
        self.monitor_thread: Optional[threading.Thread] = None
        self.running = False
        self.config_file: Optional[str] = None
        
    def start_trading(self, 
                     config_file: str,
                     dry_run: Optional[bool] = None,
                     log_callback: Optional[Callable[[str], None]] = None) -> bool:
        """
        启动交易
        
        Args:
            config_file: 配置文件路径
            dry_run: 是否为模拟交易（覆盖配置文件设置）
            log_callback: 日志回调函数
            
        Returns:
            是否成功启动
        """
        if self.running:
            logger.warning("交易引擎已在运行中")
            return False
        
        config_path = Path(config_file)
        if not config_path.exists():
            logger.error(f"配置文件不存在: {config_file}")
            return False
        
        self.config_file = config_file
        
        # 构建命令
        cmd = ["uv", "run", "freqtrade", "trade", "--config", config_file]
        
        if dry_run is True:
            cmd.append("--dry-run")
        elif dry_run is False:
            cmd.append("--live")
        
        try:
            logger.info(f"启动 FreqTrade: {' '.join(cmd)}")
            
            self.process = subprocess.Popen(
                cmd,
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            self.running = True
            
            # 启动日志监控线程
            if log_callback:
                self.monitor_thread = threading.Thread(
                    target=self._monitor_logs,
                    args=(log_callback,),
                    daemon=True
                )
                self.monitor_thread.start()
            
            logger.info("交易引擎启动成功")
            return True
            
        except Exception as e:
            logger.error(f"启动交易引擎失败: {e}")
            return False
    
    def stop_trading(self, timeout: int = 30) -> bool:
        """
        停止交易
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            是否成功停止
        """
        if not self.running or not self.process:
            logger.warning("交易引擎未运行")
            return True
        
        try:
            logger.info("正在停止交易引擎...")
            
            # 发送终止信号
            self.process.terminate()
            
            # 等待进程结束
            try:
                self.process.wait(timeout=timeout)
                logger.info("交易引擎已正常停止")
            except subprocess.TimeoutExpired:
                logger.warning("交易引擎停止超时，强制终止")
                self.process.kill()
                self.process.wait()
            
            self.running = False
            self.process = None
            
            # 等待监控线程结束
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=5)
            
            return True
            
        except Exception as e:
            logger.error(f"停止交易引擎失败: {e}")
            return False
    
    def is_running(self) -> bool:
        """检查是否运行中"""
        if not self.running or not self.process:
            return False
        
        # 检查进程是否还活着
        if self.process.poll() is not None:
            self.running = False
            return False
        
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """获取引擎状态"""
        return {
            "running": self.is_running(),
            "config_file": self.config_file,
            "pid": self.process.pid if self.process else None,
            "return_code": self.process.returncode if self.process else None
        }
    
    def _monitor_logs(self, log_callback: Callable[[str], None]):
        """监控日志输出"""
        if not self.process or not self.process.stdout:
            return
        
        try:
            for line in iter(self.process.stdout.readline, ''):
                if not line or not self.running:
                    break
                
                line = line.rstrip()
                if line:
                    log_callback(line)
            
        except Exception as e:
            logger.error(f"日志监控异常: {e}")
        finally:
            logger.debug("日志监控线程结束")
    
    def run_backtest(self, 
                    config_file: str,
                    timerange: Optional[str] = None,
                    strategy_list: Optional[list] = None) -> Dict[str, Any]:
        """
        运行回测
        
        Args:
            config_file: 配置文件路径
            timerange: 时间范围 (例: "20240101-20240201")
            strategy_list: 策略列表
            
        Returns:
            回测结果
        """
        cmd = ["uv", "run", "freqtrade", "backtesting", "--config", config_file]
        
        if timerange:
            cmd.extend(["--timerange", timerange])
        
        if strategy_list:
            cmd.extend(["--strategy-list"] + strategy_list)
        
        try:
            logger.info(f"运行回测: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "回测超时",
                "stdout": "",
                "stderr": "回测运行时间超过5分钟",
                "return_code": -1
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "stdout": "",
                "stderr": str(e),
                "return_code": -1
            }
    
    def download_data(self, 
                     config_file: str,
                     pairs: Optional[list] = None,
                     timeframe: Optional[str] = None,
                     timerange: Optional[str] = None) -> bool:
        """
        下载数据
        
        Args:
            config_file: 配置文件路径
            pairs: 交易对列表
            timeframe: 时间框架
            timerange: 时间范围
            
        Returns:
            是否成功
        """
        cmd = ["uv", "run", "freqtrade", "download-data", "--config", config_file]
        
        if pairs:
            cmd.extend(["--pairs"] + pairs)
        
        if timeframe:
            cmd.extend(["--timeframes", timeframe])
        
        if timerange:
            cmd.extend(["--timerange", timerange])
        
        try:
            logger.info(f"下载数据: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=120  # 2分钟超时
            )
            
            if result.returncode == 0:
                logger.info("数据下载成功")
                return True
            else:
                logger.error(f"数据下载失败: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"下载数据异常: {e}")
            return False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.running:
            self.stop_trading()