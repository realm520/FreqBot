"""策略加载器 - 动态加载和验证策略"""

import sys
import importlib.util
import inspect
from pathlib import Path
from typing import Type, Optional, Any
import logging

logger = logging.getLogger(__name__)

class StrategyLoader:
    """策略动态加载器"""
    
    def __init__(self, strategies_dir: Optional[Path] = None):
        self.strategies_dir = strategies_dir or Path("strategies")
    
    def load_strategy_class(self, strategy_name: str, file_path: Optional[str] = None) -> Optional[Type]:
        """
        动态加载策略类
        
        Args:
            strategy_name: 策略名称
            file_path: 策略文件路径（可选，会自动查找）
        
        Returns:
            策略类或 None
        """
        if file_path:
            return self._load_from_file(Path(file_path), strategy_name)
        else:
            return self._find_and_load(strategy_name)
    
    def _find_and_load(self, strategy_name: str) -> Optional[Type]:
        """查找并加载策略"""
        # 尝试多种文件名格式
        possible_names = [
            f"{strategy_name}.py",
            f"{strategy_name.lower()}.py",
            f"{''.join(word.capitalize() for word in strategy_name.split('_'))}.py"
        ]
        
        # 在策略目录中递归查找
        for py_file in self.strategies_dir.rglob("*.py"):
            if py_file.name in possible_names:
                strategy_class = self._load_from_file(py_file, strategy_name)
                if strategy_class:
                    return strategy_class
        
        logger.error(f"未找到策略: {strategy_name}")
        return None
    
    def _load_from_file(self, file_path: Path, expected_class_name: str) -> Optional[Type]:
        """从文件加载策略"""
        if not file_path.exists():
            logger.error(f"策略文件不存在: {file_path}")
            return None
        
        try:
            # 构建模块名
            module_name = f"strategy_{file_path.stem}_{id(file_path)}"
            
            # 动态导入
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                logger.error(f"无法创建模块规范: {file_path}")
                return None
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # 查找策略类
            strategy_class = self._find_strategy_class(module, expected_class_name)
            if strategy_class:
                logger.info(f"成功加载策略: {expected_class_name} from {file_path}")
                return strategy_class
            else:
                logger.error(f"在文件中未找到策略类: {expected_class_name} in {file_path}")
                return None
                
        except Exception as e:
            logger.error(f"加载策略文件失败 {file_path}: {e}")
            return None
    
    def _find_strategy_class(self, module: Any, expected_name: str) -> Optional[Type]:
        """在模块中查找策略类"""
        import inspect
        
        # 首先尝试直接按名称查找
        if hasattr(module, expected_name):
            cls = getattr(module, expected_name)
            if inspect.isclass(cls) and self._is_strategy_class(cls):
                return cls
        
        # 查找所有策略类
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if self._is_strategy_class(obj):
                # 如果只有一个策略类，就返回它
                return obj
        
        return None
    
    def _is_strategy_class(self, cls: Type) -> bool:
        """判断是否为策略类"""
        required_methods = ['populate_indicators', 'populate_entry_trend', 'populate_exit_trend']
        return all(hasattr(cls, method) for method in required_methods)
    
    def validate_strategy_file(self, file_path: Path) -> tuple[bool, str]:
        """验证策略文件是否有效"""
        if not file_path.exists():
            return False, "文件不存在"
        
        if not file_path.suffix == '.py':
            return False, "不是 Python 文件"
        
        try:
            # 尝试编译文件
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            compile(code, str(file_path), 'exec')
            
            # 检查是否包含策略类
            temp_module_name = f"temp_validation_{id(file_path)}"
            spec = importlib.util.spec_from_file_location(temp_module_name, file_path)
            if spec is None or spec.loader is None:
                return False, "无法创建模块规范"
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # 查找策略类
            strategy_classes = []
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if self._is_strategy_class(obj):
                    strategy_classes.append(name)
            
            if not strategy_classes:
                return False, "未找到有效的策略类"
            
            return True, f"发现策略类: {', '.join(strategy_classes)}"
            
        except SyntaxError as e:
            return False, f"语法错误: {e}"
        except Exception as e:
            return False, f"验证失败: {e}"
    
    def get_strategy_metadata(self, file_path: Path) -> dict:
        """获取策略元数据"""
        metadata = {
            "name": file_path.stem,
            "file_path": str(file_path),
            "valid": False,
            "classes": [],
            "error": None
        }
        
        try:
            valid, message = self.validate_strategy_file(file_path)
            metadata["valid"] = valid
            
            if valid:
                # 提取类信息
                spec = importlib.util.spec_from_file_location(f"meta_{id(file_path)}", file_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    import inspect
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if self._is_strategy_class(obj):
                            class_info = {
                                "name": name,
                                "description": getattr(obj, '__doc__', '').strip(),
                                "version": getattr(obj, 'STRATEGY_VERSION', '1.0.0'),
                                "author": getattr(obj, 'STRATEGY_AUTHOR', ''),
                            }
                            metadata["classes"].append(class_info)
            else:
                metadata["error"] = message
                
        except Exception as e:
            metadata["error"] = str(e)
        
        return metadata