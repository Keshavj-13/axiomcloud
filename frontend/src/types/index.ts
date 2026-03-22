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
  created_at: string;
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
  created_at: string;
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
  training_time?: number;
  created_at: string;
}

export interface MetricsData {
  job_id: string;
  task_type: string;
  status: string;
  models: TrainedModel[];
  best_model?: TrainedModel;
  leaderboard: {
    rank: number;
    model_name: string;
    primary_metric?: number;
    metric_name: string;
    training_time?: number;
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
  shap_values?: number[][] | number[][][];
  expected_value?: number[] | number;
  data?: number[][];
  feature_names?: string[];
  global_importance?: { feature: string; importance: number }[];
  error?: string;
}

export interface LimeResult {
  lime_explanations?: {
    instance: number;
    explanation: Array<[string, number]>;
  }[];
  error?: string;
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
