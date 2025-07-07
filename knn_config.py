
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Tuple


class DistanceMetric(Enum):
    RMSE = 'rmse'
    DTW = 'dtw'
    FINAL_DIFF = 'final_diff'

class AlignmentMethod(Enum):
    LINEAR = 'linear'
    AFFINE = 'affine'
    RATIO = 'ratio'

@dataclass
class GroupConfig:
    """Configuration for specific event type, sub_event_type, and border combination"""
    early_stage_end: int = 150
    mid_stage_end: int = 230
    
    early_stage_k: int = 5
    mid_stage_k: int = 4
    late_stage_k: int = 3
    
    early_stage_lookback: int = 40
    mid_stage_lookback: int = 30
    late_stage_lookback: int = 15    

    early_stage_metric: DistanceMetric = DistanceMetric.RMSE
    mid_stage_metric: DistanceMetric = DistanceMetric.RMSE
    late_stage_metric: DistanceMetric = DistanceMetric.FINAL_DIFF    

    early_stage_weights: Dict[AlignmentMethod, float] = field(default_factory=lambda: {
        AlignmentMethod.AFFINE: 0.4,
        AlignmentMethod.LINEAR: 0.3,
        AlignmentMethod.RATIO: 0.3
    })
    mid_stage_weights: Dict[AlignmentMethod, float] = field(default_factory=lambda: {
        AlignmentMethod.AFFINE: 0.4,
        AlignmentMethod.LINEAR: 0.3,
        AlignmentMethod.RATIO: 0.3
    })
    late_stage_weights: Dict[AlignmentMethod, float] = field(default_factory=lambda: {
        AlignmentMethod.AFFINE: 0.4,
        AlignmentMethod.LINEAR: 0.3,
        AlignmentMethod.RATIO: 0.3
    })
    
    use_trend_weighting: bool = False
    trend_weight: float = 0.3
    smoothing_window: Optional[int] = None
    outlier_threshold: float = 2.5

    early_stage_use_ensemble: bool = True
    mid_stage_use_ensemble: bool = True
    late_stage_use_ensemble: bool = False

    early_stage_use_smooth_for_neighbors: bool = True
    mid_stage_use_smooth_for_neighbors: bool = True
    late_stage_use_smooth_for_neighbors: bool = False

    early_stage_use_smooth_for_prediction: bool = True
    mid_stage_use_smooth_for_prediction: bool = True
    late_stage_use_smooth_for_prediction: bool = True

    use_error_bounds: bool = False
    error_bound_std_multiplier: float = 2.0

def get_default_group_configs() -> Dict[Tuple[float, Tuple[float], float], GroupConfig]:
    """Get default configurations for all groups"""
    configs = {}

    configs[(3.0, (1.0,), 100.0)] = GroupConfig(
        early_stage_end=150,
        mid_stage_end=230,
        early_stage_k=5,
        mid_stage_k=4,
        late_stage_k=3,
        early_stage_use_smooth_for_neighbors=True,
        mid_stage_use_smooth_for_neighbors=True,
        late_stage_use_smooth_for_neighbors=False,
        early_stage_use_smooth_for_prediction=True,
        mid_stage_use_smooth_for_prediction=True,
        late_stage_use_smooth_for_prediction=True
    )
    
    configs[(3.0, 2.0, 100.0)] = GroupConfig(
        early_stage_end=170,
        mid_stage_end=240,
        early_stage_k=6,
        mid_stage_k=4,
        late_stage_k=3,
        early_stage_lookback=50,
        mid_stage_lookback=35,
        late_stage_lookback=20,
        early_stage_metric=DistanceMetric.RMSE,
        mid_stage_metric=DistanceMetric.RMSE,
        late_stage_metric=DistanceMetric.FINAL_DIFF,
        early_stage_weights={
            AlignmentMethod.AFFINE: 0.7,
            AlignmentMethod.LINEAR: 0.2,
            AlignmentMethod.RATIO: 0.1
        },
        mid_stage_weights={
            AlignmentMethod.AFFINE: 0.5,
            AlignmentMethod.RATIO: 0.3,
            AlignmentMethod.LINEAR: 0.2
        },
        late_stage_weights={
            AlignmentMethod.RATIO: 0.6,
            AlignmentMethod.AFFINE: 0.3,
            AlignmentMethod.LINEAR: 0.1
        },
        early_stage_use_ensemble=True,
        mid_stage_use_ensemble=True,
        late_stage_use_ensemble=False,
        
        early_stage_use_smooth_for_neighbors=True,
        mid_stage_use_smooth_for_neighbors=False,
        late_stage_use_smooth_for_neighbors=False,
        
        early_stage_use_smooth_for_prediction=True,
        mid_stage_use_smooth_for_prediction=True,
        late_stage_use_smooth_for_prediction=True
    )

    configs[(5.0, 1.0, 100.0)] = GroupConfig(
        early_stage_end=120,
        mid_stage_end=250,
        early_stage_k=3,
        mid_stage_k=3,
        late_stage_k=3,
        early_stage_lookback=30,
        mid_stage_lookback=45,
        late_stage_lookback=15,
        early_stage_metric=DistanceMetric.RMSE,
        mid_stage_metric=DistanceMetric.RMSE,
        late_stage_metric=DistanceMetric.FINAL_DIFF,
        early_stage_use_ensemble=True,
        mid_stage_use_ensemble=True,
        late_stage_use_ensemble=False,
        early_stage_use_smooth_for_neighbors=False, 
        mid_stage_use_smooth_for_neighbors=False,
        late_stage_use_smooth_for_neighbors=False,
        early_stage_use_smooth_for_prediction=True,
        mid_stage_use_smooth_for_prediction=True,
        late_stage_use_smooth_for_prediction=True,
        early_stage_weights={
            AlignmentMethod.AFFINE: 0.0,
            AlignmentMethod.LINEAR: 0.0,
            AlignmentMethod.RATIO: 1.0
        },
        mid_stage_weights={
            AlignmentMethod.RATIO: 1.0,
            AlignmentMethod.AFFINE: 0.0,
            AlignmentMethod.LINEAR: 0.0
        },
    )


    configs[(5.0, 1.0, 1000.0)] = GroupConfig(
        early_stage_end=90,
        mid_stage_end=190,
        early_stage_k=3,  
        mid_stage_k=3,    
        late_stage_k=3,
        early_stage_lookback=30,  
        mid_stage_lookback=40,    
        late_stage_lookback=75,
        early_stage_metric=DistanceMetric.RMSE,
        mid_stage_metric=DistanceMetric.RMSE,
        late_stage_metric=DistanceMetric.RMSE,
        early_stage_use_ensemble=True,
        mid_stage_use_ensemble=True,  
        late_stage_use_ensemble=True,
        early_stage_use_smooth_for_neighbors=False,   
        mid_stage_use_smooth_for_neighbors=False,     
        late_stage_use_smooth_for_neighbors=False,
        early_stage_use_smooth_for_prediction=True,
        mid_stage_use_smooth_for_prediction=True,
        late_stage_use_smooth_for_prediction=True,
        early_stage_weights={
            AlignmentMethod.RATIO: 1.0,
            AlignmentMethod.AFFINE: 0.0,
            AlignmentMethod.LINEAR: 0.0,
        },
        mid_stage_weights={
            AlignmentMethod.RATIO: 1.0,
            AlignmentMethod.AFFINE: 0.0,
            AlignmentMethod.LINEAR: 0.0
        },
        late_stage_weights={
            AlignmentMethod.RATIO: 1.0,
            AlignmentMethod.AFFINE: 0.0,
            AlignmentMethod.LINEAR: 0.0
        }
    )
    
    return configs

GROUP_CONFIGS: Dict[Tuple[float, Tuple[float], float], GroupConfig] = get_default_group_configs()

def get_group_config(event_type: float, sub_types: Tuple[float], border: float) -> GroupConfig:
    """Get configuration for specific group"""
    key = (event_type, sub_types, border)
    if key not in GROUP_CONFIGS:
        GROUP_CONFIGS[key] = GroupConfig()
    return GROUP_CONFIGS[key]