"""
Workflow Context - Manages data flow between workflow steps
"""
from typing import Dict, Any, Optional, List
from datetime import datetime
import copy


class WorkflowContext:
    """
    Manages data flow and state throughout workflow execution.
    Provides isolation and tracking for workflow data.
    """
    
    def __init__(self, initial_data: Optional[Dict[str, Any]] = None):
        """
        Initialize workflow context with optional initial data.
        
        Args:
            initial_data: Initial context data
        """
        self.data = initial_data or {}
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'updates': [],
            'accessed': []
        }
        self.checkpoints = []
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from context.
        
        Args:
            key: Key to retrieve
            default: Default value if key not found
            
        Returns:
            Value from context or default
        """
        self.metadata['accessed'].append({
            'key': key,
            'timestamp': datetime.now().isoformat(),
            'found': key in self.data
        })
        
        return self.data.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set value in context.
        
        Args:
            key: Key to set
            value: Value to store
        """
        old_value = self.data.get(key)
        self.data[key] = value
        
        self.metadata['updates'].append({
            'key': key,
            'timestamp': datetime.now().isoformat(),
            'old_value_type': type(old_value).__name__ if old_value is not None else 'None',
            'new_value_type': type(value).__name__
        })
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update multiple values in context.
        
        Args:
            updates: Dictionary of updates to apply
        """
        for key, value in updates.items():
            self.set(key, value)
    
    def has(self, key: str) -> bool:
        """
        Check if key exists in context.
        
        Args:
            key: Key to check
            
        Returns:
            True if key exists
        """
        return key in self.data
    
    def remove(self, key: str) -> Optional[Any]:
        """
        Remove key from context.
        
        Args:
            key: Key to remove
            
        Returns:
            Removed value or None
        """
        if key in self.data:
            value = self.data.pop(key)
            self.metadata['updates'].append({
                'key': key,
                'timestamp': datetime.now().isoformat(),
                'action': 'removed',
                'value_type': type(value).__name__
            })
            return value
        return None
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get all context data.
        
        Returns:
            Copy of all context data
        """
        return copy.deepcopy(self.data)
    
    def clear(self) -> None:
        """
        Clear all context data.
        """
        self.data.clear()
        self.metadata['updates'].append({
            'action': 'cleared',
            'timestamp': datetime.now().isoformat()
        })
    
    def checkpoint(self, name: str) -> None:
        """
        Create a checkpoint of current context state.
        
        Args:
            name: Name for the checkpoint
        """
        self.checkpoints.append({
            'name': name,
            'timestamp': datetime.now().isoformat(),
            'data': copy.deepcopy(self.data)
        })
    
    def restore_checkpoint(self, name: str) -> bool:
        """
        Restore context to a previous checkpoint.
        
        Args:
            name: Name of checkpoint to restore
            
        Returns:
            True if checkpoint found and restored
        """
        for checkpoint in reversed(self.checkpoints):
            if checkpoint['name'] == name:
                self.data = copy.deepcopy(checkpoint['data'])
                self.metadata['updates'].append({
                    'action': 'restored',
                    'checkpoint': name,
                    'timestamp': datetime.now().isoformat()
                })
                return True
        return False
    
    def list_checkpoints(self) -> List[str]:
        """
        List all checkpoint names.
        
        Returns:
            List of checkpoint names
        """
        return [cp['name'] for cp in self.checkpoints]
    
    def validate_required(self, required_keys: List[str]) -> Dict[str, bool]:
        """
        Validate that required keys exist in context.
        
        Args:
            required_keys: List of required keys
            
        Returns:
            Dictionary of key -> exists status
        """
        return {key: key in self.data for key in required_keys}
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get context metadata.
        
        Returns:
            Context metadata including access patterns
        """
        return copy.deepcopy(self.metadata)
    
    def merge(self, other_context: 'WorkflowContext', overwrite: bool = False) -> None:
        """
        Merge another context into this one.
        
        Args:
            other_context: Context to merge from
            overwrite: Whether to overwrite existing keys
        """
        for key, value in other_context.data.items():
            if overwrite or key not in self.data:
                self.set(key, value)
    
    def filter_keys(self, keys: List[str]) -> Dict[str, Any]:
        """
        Get subset of context with only specified keys.
        
        Args:
            keys: List of keys to include
            
        Returns:
            Dictionary with only specified keys
        """
        return {key: self.data[key] for key in keys if key in self.data}
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in context"""
        return key in self.data
    
    def __len__(self) -> int:
        """Get number of keys in context"""
        return len(self.data)
    
    def __repr__(self) -> str:
        """String representation of context"""
        return f"WorkflowContext(keys={list(self.data.keys())}, checkpoints={len(self.checkpoints)})"