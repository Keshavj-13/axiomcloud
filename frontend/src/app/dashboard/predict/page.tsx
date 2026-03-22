"use client";
import { useState, useEffect } from "react";
import { modelsAPI, predictionsAPI } from "@/lib/api";
import { PredictionResult } from "@/types";
import toast from "react-hot-toast";
import { Zap, Loader2, CheckCircle, BarChart2 } from "lucide-react";
import PanelHeader from "@/components/ui/PanelHeader";
import StatusBadge from "@/components/ui/StatusBadge";

export default function PredictPage() {
  const [deployedModels, setDeployedModels] = useState<any[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<number | null>(null);
  const [selectedModel, setSelectedModel] = useState<any>(null);
  const [features, setFeatures] = useState<Record<string, string>>({});
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [predicting, setPredicting] = useState(false);

  useEffect(() => {
    modelsAPI.listDeployed().then(r => {
      setDeployedModels(r.data);
    }).catch(() => {});
  }, []);

  useEffect(() => {
    if (!selectedModelId) return;
    modelsAPI.get(selectedModelId).then(r => {
      setSelectedModel(r.data);
      // Initialize feature inputs
      const fi = r.data.feature_importance || {};
      const init: Record<string, string> = {};
      Object.keys(fi).forEach(k => init[k] = "");
      setFeatures(init);
      setResult(null);
    }).catch(() => {});
  }, [selectedModelId]);

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

  const featureKeys = Object.keys(features);

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

      {selectedModel && featureKeys.length > 0 && (
        <div className="grid lg:grid-cols-2 gap-6">
          {/* Feature inputs */}
          <div className="panel p-6">
            <PanelHeader title="Feature Input" subtitle="Provide values for the selected model schema." className="mb-4" />
            <div className="space-y-3 max-h-[400px] overflow-y-auto pr-1">
              {featureKeys.map(feat => (
                <div key={feat}>
                  <label className="mb-1 block font-mono text-xs text-text-muted">{feat}</label>
                  <input
                    value={features[feat]}
                    onChange={e => setFeatures(prev => ({ ...prev, [feat]: e.target.value }))}
                    placeholder="Enter value..."
                    className="w-full rounded-lg border border-outline/25 bg-surface px-3 py-2 font-mono text-sm text-text-primary placeholder:text-text-muted focus:outline-none focus:border-primary/50"
                  />
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
