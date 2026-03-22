export interface ColumnInfo {
  name: string;
  dtype: string;
  null_count: number;
  unique_count: number;
  sample_values: unknown[];
  stats?: {
    min: number; max: number; mean: number; std: number; median: number;
  };
}

export interface Dataset {
  id: number;
  name: string;
  filename: string;
  file_size?: number;
  num_rows?: number;
  num_columns?: number;
  columns_info?: ColumnInfo[];
  target_column?: string;
  task_type?: string;
  is_example: boolean;
  preview_data?: Record<string, unknown>[];
  created_at?: string;
}

export interface TrainingJob {
  id: number;
  job_id: string;
  dataset_id: number;
  task_type?: string;
  target_column: string;
  status: "pending" | "running" | "completed" | "failed";
  progress: number;
  error_message?: string;
  config?: Record<string, unknown>;
  created_at: string;
  completed_at?: string;
}

export interface ExperimentRun {
  id: number;
  run_id: string;
  job_id: string;
  dataset_id: number;
  target_column: string;
  task_type?: string;
  status: "pending" | "running" | "completed" | "failed";
  config?: Record<string, unknown>;
  summary_metrics?: {
    primary_metric?: string;
    best_model?: string;
    best_score?: number;
    models_trained?: number;
    failed_models?: number;
  };
  best_model_name?: string;
  best_score?: number;
  error_message?: string;
  started_at: string;
  completed_at?: string;
}

export interface TrainedModel {
  id: number;
  job_id: string;
  model_name: string;
  model_type: string;
  task_type: string;
  is_deployed: boolean;
  accuracy?: number;
  f1_score?: number;
  roc_auc?: number;
  precision?: number;
  recall?: number;
  balanced_accuracy?: number;
  rmse?: number;
  mae?: number;
  r2_score?: number;
  mape?: number;
  explained_variance?: number;
  median_ae?: number;
  metrics?: Record<string, unknown>;
  feature_importance?: Record<string, number>;
  confusion_matrix?: number[][];
  roc_curve_data?: { fpr: number[]; tpr: number[] };
  cv_scores?: number[];
  per_fold?: { fold: number; score: number }[];
  per_class?: Record<string, unknown>;
  confusion_matrix_chart?: {
    labels: string[];
    matrix: number[][];
    normalized_matrix?: number[][];
  };
  roc_curve_chart?: {
    fpr: number[];
    tpr: number[];
    thresholds?: number[];
  };
  training_time?: number;
  created_at: string;
}

export interface MetricsData {
  job_id: string;
  task_type: string;
  status: string;
  models: TrainedModel[];
  best_model?: TrainedModel;
  best_model_id?: number;
  metric_catalog?: string[];
  chart_data?: {
    metric_comparison: {
      labels: string[];
      series: Record<string, Array<number | null>>;
    };
    roc_curves: Array<{
      model_id: number;
      model_name: string;
      curve?: { fpr: number[]; tpr: number[]; thresholds?: number[] };
    }>;
    confusion_matrices: Array<{
      model_id: number;
      model_name: string;
      matrix?: { labels: string[]; matrix: number[][]; normalized_matrix?: number[][] };
    }>;
  };
  dataset_profile?: DatasetProfile;
  leaderboard: {
    rank: number;
    model_id?: number;
    model_name: string;
    primary_metric?: number;
    metric_name: string;
    training_time?: number;
    is_deployed?: boolean;
  }[];
}

export interface PredictionResult {
  model_id: number;
  model_name: string;
  prediction: unknown;
  probability?: Record<string, number>;
  confidence?: number;
}

export interface ExampleDataset {
  key: string;
  name: string;
  description: string;
  target: string;
  task_type: string;
}

export interface DatasetQualityReport {
  dataset_id: number;
  dataset_name: string;
  rows: number;
  columns: number;
  quality_score: number;
  missing_cells: number;
  duplicate_rows: number;
  missing_by_column: Record<string, number>;
  outliers_by_column: Record<string, number>;
  recommendations: string[];
}

export interface CleanPreview {
  dataset_id: number;
  dataset_name: string;
  preview_rows: number;
  rows_after_cleaning: number;
  applied_fixes: string[];
  clean_preview: Record<string, unknown>[];
}

export interface ShapResult {
  metadata: {
    model_id: number;
    model_name: string;
    model_type: string;
    task_type: string;
    job_id: string;
    artifact_hash: string;
    dataset_hash: string;
    cache_key: string;
    used_fallback: boolean;
    explainer: string;
  };
  feature_names: string[];
  expected_value?: number[] | number | null;
  global_importance: { feature: string; mean_abs_contribution: number; rank: number }[];
  local_contributions: { feature: string; value: number; abs_value: number; rank: number }[];
  sample_prediction: {
    sample_index: number;
    prediction: number | string;
    prediction_label?: string;
    confidence?: number;
    class_probabilities?: Record<string, number>;
  };
  chart?: {
    global_bar?: { labels: string[]; values: number[] };
    local_bar?: { labels: string[]; values: number[] };
  };
  model_metadata?: Record<string, unknown>;
}

export interface LimeResult {
  metadata: {
    model_id: number;
    model_name: string;
    model_type: string;
    task_type: string;
    job_id: string;
    artifact_hash: string;
    dataset_hash: string;
    cache_key: string;
    used_fallback: boolean;
    explainer: string;
  };
  feature_names: string[];
  sample_prediction: {
    sample_index: number;
    prediction: number | string;
    prediction_label?: string;
    confidence?: number;
    class_probabilities?: Record<string, number>;
  };
  weights: { feature: string; weight: number; abs_weight: number; direction: "positive" | "negative"; rank: number }[];
  top_positive: { feature: string; weight: number; abs_weight: number; direction: "positive" | "negative"; rank: number }[];
  top_negative: { feature: string; weight: number; abs_weight: number; direction: "positive" | "negative"; rank: number }[];
  class_context: Record<string, unknown>;
  chart?: {
    local_bar?: { labels: string[]; values: number[] };
  };
}

export interface DatasetProfile {
  dataset_id: number;
  dataset_name: string;
  total_rows: number;
  total_columns: number;
  memory_usage_mb: number;
  missing_total: number;
  missing_rate: number;
  duplicate_rows: number;
  columns: Array<{
    name: string;
    dtype: string;
    dtype_group: string;
    missing: number;
    missing_rate: number;
    unique: number;
    unique_rate: number;
    sample_values: string[];
    summary?: Record<string, number | null>;
  }>;
  summary_cards: Array<{ label: string; value: number | string }>;
  target_distribution?: {
    type: "numeric" | "categorical";
    column: string;
    labels: string[];
    counts: number[];
  };
  correlation_heatmap?: {
    labels: string[];
    matrix: number[][];
  };
  missing_by_column: Array<{ column: string; missing: number }>;
  histograms: Array<{ column: string; labels: string[]; counts: number[] }>;
}

export interface APIErrorPayload {
  error?: {
    code: string;
    message: string;
    details?: Record<string, unknown>;
  };
}

export interface ModelMonitoring {
  monitoring: {
    model_id: number;
    model_name: string;
    task_type: string;
    is_deployed: boolean;
    training_time?: number;
    primary_metric?: number;
    cv_mean?: number;
    cv_std?: number;
    stability: "stable" | "moderate" | "unstable" | "unknown";
  };
  drift?: {
    train_dataset_id: number;
    compare_dataset_id: number;
    features_compared: number;
    avg_z_drift: number;
    drift_level: "low" | "medium" | "high";
    top_drift_features: {
      feature: string;
      train_mean: number;
      current_mean: number;
      z_drift: number;
    }[];
  } | null;
}
