from dataclasses import dataclass
import numpy as np
from typing import Any, Dict, Optional, Tuple


@dataclass
class FluentSpec:
    '''A dataclass to specify the properties of a state or action variable.'''
    shape: Tuple[int, ...]
    prange: str    # real, int, bool, pixel
    values: Optional[Tuple[int | float, int | float]] = None   # (low, high) inclusive

    @property
    def size(self) -> int:
        '''Returns the total number of elements in this variable.'''
        return int(np.prod(self.shape, dtype=np.int64))
    
    def serialize(self) -> Dict[str, Any]:
        '''Converts FluentSpec to plain Python types for safe checkpoint serialization.'''
        return {
            'shape': tuple(self.shape),
            'prange': self.prange,
            'values': self.values,
        }

    @staticmethod
    def deserialize(data: Dict[str, Any]) -> 'FluentSpec':
        '''Builds FluentSpec from serialized data.'''
        return FluentSpec(
            shape=tuple(data['shape']),
            prange=data['prange'],
            values=data.get('values', None),
        )
    

@dataclass
class EnvSpec:
    '''A dataclass to specify the properties of an environment.'''
    state_spec: Dict[str, FluentSpec]
    action_spec: Dict[str, FluentSpec]

    @property
    def all_spec(self) -> Dict[str, FluentSpec]:
        '''Returns a combined dict of all state and action specs.'''
        return {**self.state_spec, **self.action_spec}
    
    def serialize(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        '''Converts EnvSpec to plain Python types for safe checkpoint serialization.'''
        return {
            'state_spec': {k: v.serialize() for k, v in self.state_spec.items()},
            'action_spec': {k: v.serialize() for k, v in self.action_spec.items()},
        }

    @staticmethod
    def deserialize(data: Any) -> 'EnvSpec':
        '''Builds EnvSpec from serialized data; accepts already-instantiated EnvSpec.'''
        if isinstance(data, EnvSpec):
            return data
        return EnvSpec(
            state_spec={k: FluentSpec.deserialize(v) 
                        for k, v in data['state_spec'].items()},
            action_spec={k: FluentSpec.deserialize(v) 
                         for k, v in data['action_spec'].items()},
        )
