import { describe, expect, it } from "vitest";

import { buildLimeBars, buildMetricComparisonBars, buildShapGlobalBars } from "@/lib/chartTransforms";


describe("chartTransforms", () => {
  it("builds SHAP global bars", () => {
    const bars = buildShapGlobalBars({
      metadata: {} as never,
      feature_names: [],
      global_importance: [{ feature: "f1", mean_abs_contribution: 0.5, rank: 1 }],
      local_contributions: [],
      sample_prediction: { sample_index: 0, prediction: 1 },
    });

    expect(bars).toEqual([{ feature: "f1", value: 0.5 }]);
  });

  it("builds LIME bars", () => {
    const bars = buildLimeBars({
      metadata: {} as never,
      feature_names: [],
      sample_prediction: { sample_index: 0, prediction: 1 },
      weights: [{ feature: "age", weight: -0.2, abs_weight: 0.2, direction: "negative", rank: 1 }],
      top_positive: [],
      top_negative: [],
      class_context: {},
    });

    expect(bars[0].feature).toBe("age");
    expect(bars[0].value).toBe(-0.2);
  });

  it("builds metric bars from chart payload", () => {
    const bars = buildMetricComparisonBars(
      {
        job_id: "job-1",
        task_type: "classification",
        status: "completed",
        models: [],
        leaderboard: [],
        chart_data: {
          metric_comparison: {
            labels: ["A", "B"],
            series: { accuracy: [0.8, 0.9] },
          },
          roc_curves: [],
          confusion_matrices: [],
        },
      },
      "accuracy"
    );

    expect(bars).toEqual([
      { label: "A", value: 0.8 },
      { label: "B", value: 0.9 },
    ]);
  });
});
