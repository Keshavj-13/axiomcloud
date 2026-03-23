"use client";
import { useState, useEffect } from "react";
import { modelsAPI, predictionsAPI } from "@/lib/api";
import { InferenceFeatureTemplate, InferenceTemplate, PredictionResult, TrainedModel } from "@/types";
import toast from "react-hot-toast";
import { Zap, Loader2, CheckCircle, BarChart2, Shuffle, RotateCcw } from "lucide-react";
import PanelHeader from "@/components/ui/PanelHeader";
import StatusBadge from "@/components/ui/StatusBadge";

export default function PredictPage() {
  const [deployedModels, setDeployedModels] = useState<TrainedModel[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<number | null>(null);
  const [selectedModel, setSelectedModel] = useState<TrainedModel | null>(null);
  const [template, setTemplate] = useState<InferenceTemplate | null>(null);
  const [features, setFeatures] = useState<Record<string, string>>({});
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loadingTemplate, setLoadingTemplate] = useState(false);
  const [predicting, setPredicting] = useState(false);

  useEffect(() => {
    modelsAPI.listDeployed().then(r => {
      setDeployedModels(r.data);
    }).catch(() => {});
  }, []);

  useEffect(() => {
    if (!selectedModelId) return;
    const loadSelection = async () => {
      setLoadingTemplate(true);
      try {
        const modelRes = await modelsAPI.get(selectedModelId);
        setSelectedModel(modelRes.data);

        try {
          const templateRes = await modelsAPI.inferenceTemplate(selectedModelId, true);
          setTemplate(templateRes.data);
          const init: Record<string, string> = {};
          (templateRes.data.features || []).forEach((f: InferenceFeatureTemplate) => {
            init[f.name] = String(f.default_value ?? "");
          });
          setFeatures(init);
        } catch {
          const fi = modelRes.data.feature_importance || {};
          const fallbackFeatures = Object.keys(fi);
          const fallbackTemplate: InferenceTemplate = {
            model_id: modelRes.data.id,
            model_name: modelRes.data.model_name,
            task_type: modelRes.data.task_type,
            job_id: modelRes.data.job_id,
            generated_at: new Date().toISOString(),
            features: fallbackFeatures.map((name) => ({
              name,
              dtype: "unknown",
              input_type: "text",
              default_value: "",
            })),
          };
          setTemplate(fallbackTemplate);
          const init: Record<string, string> = {};
          fallbackFeatures.forEach((name) => {
            init[name] = "";
          });
          setFeatures(init);
          toast.error("Using basic feature inputs; dataset defaults unavailable for this model");
        }

        setResult(null);
      } catch {
        toast.error("Failed to load selected model");
      } finally {
        setLoadingTemplate(false);
      }
    };
    loadSelection();
  }, [selectedModelId]);

  const randomizeDefaults = async () => {
    if (!selectedModelId) return;
    setLoadingTemplate(true);
    try {
      const templateRes = await modelsAPI.inferenceTemplate(selectedModelId, true);
      setTemplate(templateRes.data);
      const randomized: Record<string, string> = {};
      (templateRes.data.features || []).forEach((f: InferenceFeatureTemplate) => {
        randomized[f.name] = String(f.default_value ?? "");
      });
      setFeatures(randomized);
      toast.success("Feature defaults randomized");
    } catch {
      toast.error("Could not randomize defaults");
    } finally {
      setLoadingTemplate(false);
    }
  };

  const resetToMeanDefaults = async () => {
    if (!selectedModelId) return;
    setLoadingTemplate(true);
    try {
      const templateRes = await modelsAPI.inferenceTemplate(selectedModelId, false);
      setTemplate(templateRes.data);
      const centered: Record<string, string> = {};
      (templateRes.data.features || []).forEach((f: InferenceFeatureTemplate) => {
        centered[f.name] = String(f.default_value ?? "");
      });
      setFeatures(centered);
      toast.success("Defaults reset from dataset profile");
    } catch {
      toast.error("Could not reset defaults");
    } finally {
      setLoadingTemplate(false);
    }
  };

  const runPrediction = async () => {
    if (!selectedModelId) return toast.error("Select a model");
    const parsedFeatures: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(features)) {
      const num = Number(v);
      parsedFeatures[k] = isNaN(num) || v === "" ? v || null : num;
    }
    setPredicting(true);
    try {
      const res = await predictionsAPI.predict(selectedModelId, parsedFeatures);
      setResult(res.data);
      toast.success("Prediction complete!");
    } catch (e: any) {
      toast.error(e.response?.data?.detail || "Prediction failed");
    } finally { setPredicting(false); }
  };

  const featureSpecs = template?.features || [];

  return (
    <div className="mx-auto max-w-7xl">
      <div className="mb-8">
        <div className="section-kicker mb-2">Inference</div>
        <h1 className="font-display mb-1 text-4xl font-semibold text-text-primary">Inference Sandbox</h1>
        <p className="text-text-muted">Run ad-hoc predictions against active deployed models.</p>
      </div>

      {/* Model selector */}
      <div className="panel mb-6 rounded-xl p-6">
        <PanelHeader title="Model Selection" subtitle="Only deployed models are available for inference." icon={Zap} className="mb-4" />
        {deployedModels.length === 0 ? (
          <div className="rounded-lg bg-surface-variant/30 p-4 text-sm text-text-muted">
            No deployed models yet. Go to the <a href="/dashboard/models" className="text-primary underline">Leaderboard</a> and deploy a model first.
          </div>
        ) : (
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-3">
            {deployedModels.map(m => (
              <button
                key={m.id}
                onClick={() => setSelectedModelId(m.id)}
                className={`p-3 rounded-lg border text-left transition-all
                  ${selectedModelId === m.id
                    ? "border-primary/60 bg-primary/10 text-text-primary"
                    : "border-outline/25 bg-surface-variant/30 text-text-muted hover:border-primary/45"
                  }`}
              >
                <div className="font-medium text-sm">{m.model_name}</div>
                <div className="mt-1 text-xs"><StatusBadge label={m.task_type} tone="idle" /></div>
              </button>
            ))}
          </div>
        )}
      </div>

      {selectedModel && featureSpecs.length > 0 && (
        <div className="grid lg:grid-cols-2 gap-6">
          {/* Feature inputs */}
          <div className="panel p-6">
            <PanelHeader title="Feature Input" subtitle="Defaults are sampled from the model's training dataset." className="mb-4" />
            <div className="mb-4 rounded-lg border border-outline/20 bg-surface-variant/30 p-3 text-xs text-text-muted">
              <div><span className="text-text-primary">Model:</span> {selectedModel.model_name}</div>
              <div><span className="text-text-primary">Dataset:</span> {template?.dataset_name || "Unknown"}</div>
              {template?.target_column && <div><span className="text-text-primary">Target:</span> {template.target_column}</div>}
            </div>
            <div className="mb-4 flex gap-2">
              <button
                onClick={randomizeDefaults}
                disabled={loadingTemplate}
                className="btn-ghost px-3 py-2 text-xs disabled:opacity-60"
              >
                {loadingTemplate ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Shuffle className="h-3.5 w-3.5" />}
                Randomize Defaults
              </button>
              <button
                onClick={resetToMeanDefaults}
                disabled={loadingTemplate}
                className="btn-ghost px-3 py-2 text-xs disabled:opacity-60"
              >
                <RotateCcw className="h-3.5 w-3.5" />
                Reset to Stable Defaults
              </button>
            </div>
            <div className="space-y-3 max-h-[400px] overflow-y-auto pr-1">
              {featureSpecs.map((spec) => (
                <div key={spec.name}>
                  <label className="mb-1 block font-mono text-xs text-text-muted">{spec.name}</label>
                  {spec.input_type === "select" ? (
                    (spec.options && spec.options.length > 0) ? (
                    <select
                      value={features[spec.name] ?? ""}
                      onChange={e => setFeatures(prev => ({ ...prev, [spec.name]: e.target.value }))}
                      className="w-full rounded-lg border border-outline/25 bg-surface px-3 py-2 font-mono text-sm text-text-primary focus:outline-none focus:border-primary/50"
                    >
                      {(spec.options || []).map(option => (
                        <option key={option} value={option}>{option}</option>
                      ))}
                    </select>
                    ) : (
                      <input
                        type="text"
                        value={features[spec.name] ?? ""}
                        onChange={e => setFeatures(prev => ({ ...prev, [spec.name]: e.target.value }))}
                        placeholder="Enter category value..."
                        className="w-full rounded-lg border border-outline/25 bg-surface px-3 py-2 font-mono text-sm text-text-primary placeholder:text-text-muted focus:outline-none focus:border-primary/50"
                      />
                    )
                  ) : (
                    <input
                      type={spec.input_type === "number" ? "number" : "text"}
                      value={features[spec.name] ?? ""}
                      onChange={e => setFeatures(prev => ({ ...prev, [spec.name]: e.target.value }))}
                      min={spec.min}
                      max={spec.max}
                      step={spec.input_type === "number" ? (spec.integer_like ? 1 : "any") : undefined}
                      placeholder={spec.input_type === "number"
                        ? `Range ${spec.min ?? "?"} to ${spec.max ?? "?"}`
                        : "Enter value..."}
                      className="w-full rounded-lg border border-outline/25 bg-surface px-3 py-2 font-mono text-sm text-text-primary placeholder:text-text-muted focus:outline-none focus:border-primary/50"
                    />
                  )}
                  {spec.input_type === "number" && spec.min !== undefined && spec.max !== undefined && (
                    <div className="mt-1 text-[11px] text-text-muted">
                      observed min/max: {spec.min.toFixed(4)} / {spec.max.toFixed(4)}
                    </div>
                  )}
                </div>
              ))}
            </div>
            <button
              onClick={runPrediction}
              disabled={predicting}
              className="btn-primary w-full justify-center mt-5 disabled:opacity-50"
            >
              {predicting ? <Loader2 className="w-4 h-4 animate-spin" /> : <Zap className="w-4 h-4" />}
              {predicting ? "Predicting..." : "Run Prediction"}
            </button>
          </div>

          {/* Result */}
          <div className="panel p-6">
            <PanelHeader title="Inference Result" icon={BarChart2} className="mb-4" />
            {!result ? (
              <div className="py-16 text-center text-text-muted">
                <Zap className="mx-auto mb-3 h-10 w-10 opacity-20" />
                <p className="text-sm">Fill features and click Predict</p>
              </div>
            ) : (
              <div>
                <div className="flex items-center gap-3 mb-6">
                  <CheckCircle className="h-8 w-8 text-emerald-300" />
                  <div>
                    <div className="text-xs text-text-muted">Prediction</div>
                    <div className="font-display text-2xl font-bold gradient-text">
                      {String(result.prediction)}
                    </div>
                  </div>
                </div>

                {result.confidence !== undefined && result.confidence !== null && (
                  <div className="mb-4">
                    <div className="mb-1.5 flex justify-between text-xs text-text-muted">
                      <span>Confidence</span>
                      <span className="font-mono text-text-primary">{(result.confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div className="h-2 overflow-hidden rounded-full bg-surface-variant">
                      <div
                        className="h-full rounded-full bg-gradient-to-r from-primary-deep to-primary"
                        style={{ width: `${(result.confidence * 100).toFixed(0)}%` }}
                      />
                    </div>
                  </div>
                )}

                {result.probability && (
                  <div>
                    <div className="mb-2 text-xs text-text-muted">Class Probabilities</div>
                    <div className="space-y-2">
                      {Object.entries(result.probability)
                        .sort((a, b) => b[1] - a[1])
                        .map(([cls, prob]) => (
                          <div key={cls}>
                            <div className="flex justify-between text-xs mb-1">
                              <span className="font-mono text-text-primary">{cls}</span>
                              <span className="font-mono text-text-primary">{(prob * 100).toFixed(1)}%</span>
                            </div>
                            <div className="h-1.5 overflow-hidden rounded-full bg-surface-variant">
                              <div
                                className="h-full rounded-full bg-primary-deep transition-all"
                                style={{ width: `${(prob * 100).toFixed(0)}%` }}
                              />
                            </div>
                          </div>
                        ))}
                    </div>
                  </div>
                )}

                <div className="mt-4 rounded bg-surface-variant/35 p-3 text-xs font-mono text-text-muted">
                  Model: {result.model_name}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
