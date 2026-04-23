"""Utility functions for the PTM model."""

# Manager classes for refactored training script
from .config_manager import ConfigManager
from .data_utils import DataManager
from .model_utils import ModelManager
# Import training and visualization managers only when needed to avoid dependency issues
# from .training_utils_enhanced import EnhancedTrainingManager
# from .visualization_utils_enhanced import EnhancedVisualizationManager

# Common utility functions
from .common_utils import (
    set_seed,
    custom_collate_fn,
    check_gpu_memory,
    get_device,
    format_time,
    count_parameters,
    save_checkpoint,
    load_checkpoint
)

# Legacy visualization functions (commented out due to missing visualization.py)
# from .visualization import (
#     plot_confusion_matrix,
#     plot_roc_curve,
#     plot_precision_recall_curve,
#     plot_feature_importance,
#     plot_embedding_tsne,
#     plot_training_history,
#     plot_learning_rate,
#     plot_model_comparison,
#     plot_attention_heatmap,
#     plot_residue_importance,
#     plot_calibration_curve
# )

# Legacy metrics functions
from .metrics import (
    calculate_metrics,
    calculate_optimal_threshold,
    generate_threshold_curve,
    calculate_per_class_metrics,
    calculate_expected_calibration_error,
    calculate_metrics_over_time
)

# Legacy callbacks (commented out due to matplotlib dependency issues)
# from .callbacks import (
#     VisualizationCallback,
#     PredictionSamplingCallback
# )