"""
Simple Registry implementation to replace mmcv.Registry
"""
from typing import Dict, Callable, Optional, Any


class Registry:
    """Simple registry for models and components"""
    
    def __init__(self, name: str):
        self._name = name
        self._module_dict: Dict[str, Callable] = {}
    
    def register_module(self, name: Optional[str] = None, module: Optional[Callable] = None):
        """Register a module. Can be used as decorator or directly."""
        def _register(cls):
            _name = name if name is not None else cls.__name__
            self._module_dict[_name] = cls
            return cls
        
        if module is not None:
            _register(module)
            return module
        return _register
    
    def get(self, name: str) -> Callable:
        """Get registered module by name"""
        if name not in self._module_dict:
            raise KeyError(f"{name} is not in registry {self._name}. Available: {list(self._module_dict.keys())}")
        return self._module_dict[name]
    
    def build(self, cfg: dict, default_args: Optional[dict] = None) -> Any:
        """Build instance from config dict"""
        if not isinstance(cfg, dict):
            raise TypeError(f"cfg must be a dict, but got {type(cfg)}")
        
        if 'type' not in cfg:
            raise KeyError(f"'type' is required in cfg, but got {cfg}")
        
        cfg = cfg.copy()
        obj_type = cfg.pop('type')
        
        if isinstance(obj_type, str):
            obj_cls = self.get(obj_type)
        else:
            obj_cls = obj_type
        
        if default_args is not None:
            cfg.update(default_args)
        
        return obj_cls(**cfg)
