import { LimeResult, MetricsData, ShapResult } from "@/types";

export function buildShapGlobalBars(shap: ShapResult | null) {
  if (!shap?.global_importance?.length) return [];
  return shap.global_importance.map((item) => ({
    feature: item.feature,
    value: item.mean_abs_contribution,
  }));
}

export function buildShapLocalBars(shap: ShapResult | null) {
  if (!shap?.local_contributions?.length) return [];
  return shap.local_contributions.map((item) => ({
    feature: item.feature,
    value: item.value,
  }));
}

export function buildLimeBars(lime: LimeResult | null) {
  if (!lime?.weights?.length) return [];
  return lime.weights.map((item) => ({
    feature: item.feature,
    value: item.weight,
  }));
}

export function buildMetricComparisonBars(metrics: MetricsData | null, metric: string) {
  if (!metrics?.chart_data?.metric_comparison) return [];
  const labels = metrics.chart_data.metric_comparison.labels || [];
  const values = metrics.chart_data.metric_comparison.series?.[metric] || [];
  return labels.map((label, idx) => ({ label, value: values[idx] ?? 0 }));
}
