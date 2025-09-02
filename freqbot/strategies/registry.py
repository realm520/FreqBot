"""策略注册表 - 管理所有可用策略"""

import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Type, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class StrategyInfo:
    """策略信息"""
    name: str
    description: str
    file_path: str
    class_name: str
    version: str = "1.0.0"
    author: str = ""
    category: str = "general"
    
class StrategyRegistry:
    """策略注册表"""
    
    def __init__(self, strategies_dir: Optional[Path] = None):
        self.strategies_dir = strategies_dir or Path("strategies")
        self._strategies: Dict[str, StrategyInfo] = {}
        self._loaded_classes: Dict[str, Type] = {}
        
    def discover_strategies(self):
        """自动发现策略"""
        if not self.strategies_dir.exists():
            logger.warning(f"策略目录不存在: {self.strategies_dir}")
            return
        
        for py_file in self.strategies_dir.rglob("*.py"):
            if py_file.name.startswith("__"):
                continue
                
            try:
                self._register_strategy_from_file(py_file)
            except Exception as e:
                logger.error(f"注册策略文件失败 {py_file}: {e}")
    
    def _register_strategy_from_file(self, file_path: Path):
        """从文件注册策略"""
        # 构建模块名
        try:
            relative_path = file_path.relative_to(Path.cwd())
        except ValueError:
            # 如果不在当前目录的子路径中，使用绝对路径
            relative_path = file_path
        module_name = str(relative_path.with_suffix('')).replace('/', '.')
        
        try:
            # 动态导入模块
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # 查找策略类
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (hasattr(obj, 'populate_indicators') and 
                    hasattr(obj, 'populate_entry_trend') and
                    hasattr(obj, 'populate_exit_trend')):
                    
                    # 提取策略信息
                    strategy_info = self._extract_strategy_info(obj, file_path, name)
                    self._strategies[strategy_info.name] = strategy_info
                    self._loaded_classes[strategy_info.name] = obj
                    
                    logger.info(f"发现策略: {strategy_info.name} ({strategy_info.description})")
                    
        except Exception as e:
            logger.error(f"导入策略模块失败 {file_path}: {e}")
    
    def _extract_strategy_info(self, strategy_class: Type, file_path: Path, class_name: str) -> StrategyInfo:
        """提取策略信息"""
        # 从类的文档字符串或属性中提取信息
        description = (getattr(strategy_class, '__doc__', '') or 
                      getattr(strategy_class, 'STRATEGY_DESCRIPTION', '')).strip()
        
        version = getattr(strategy_class, 'STRATEGY_VERSION', '1.0.0')
        author = getattr(strategy_class, 'STRATEGY_AUTHOR', '')
        category = getattr(strategy_class, 'STRATEGY_CATEGORY', 'general')
        
        # 策略名称优先使用类名，但可以被覆盖
        name = getattr(strategy_class, 'STRATEGY_NAME', class_name)
        
        return StrategyInfo(
            name=name,
            description=description or f"{class_name} 策略",
            file_path=str(file_path),
            class_name=class_name,
            version=version,
            author=author,
            category=category
        )
    
    def get_strategy_info(self, name: str) -> Optional[StrategyInfo]:
        """获取策略信息"""
        return self._strategies.get(name)
    
    def get_strategy_class(self, name: str) -> Optional[Type]:
        """获取策略类"""
        return self._loaded_classes.get(name)
    
    def list_strategies(self, category: Optional[str] = None) -> List[StrategyInfo]:
        """列出策略"""
        strategies = list(self._strategies.values())
        
        if category:
            strategies = [s for s in strategies if s.category == category]
        
        return sorted(strategies, key=lambda x: x.name)
    
    def list_categories(self) -> List[str]:
        """列出所有策略分类"""
        categories = {s.category for s in self._strategies.values()}
        return sorted(categories)
    
    def register_strategy(self, strategy_info: StrategyInfo, strategy_class: Type):
        """手动注册策略"""
        self._strategies[strategy_info.name] = strategy_info
        self._loaded_classes[strategy_info.name] = strategy_class
        logger.info(f"手动注册策略: {strategy_info.name}")
    
    def validate_strategy(self, name: str) -> bool:
        """验证策略是否有效"""
        if name not in self._loaded_classes:
            return False
        
        strategy_class = self._loaded_classes[name]
        
        # 检查必要方法
        required_methods = ['populate_indicators', 'populate_entry_trend', 'populate_exit_trend']
        for method in required_methods:
            if not hasattr(strategy_class, method):
                logger.error(f"策略 {name} 缺少必要方法: {method}")
                return False
        
        return True