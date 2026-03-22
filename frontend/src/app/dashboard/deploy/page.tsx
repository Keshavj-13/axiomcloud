"use client";
import { useState, useEffect } from "react";
import { modelsAPI, predictionsAPI } from "@/lib/api";
import { InferenceFeatureTemplate, PredictionResult, TrainedModel } from "@/types";
import toast from "react-hot-toast";
import { Rocket, Download, CheckCircle, XCircle, Trash2, Loader2, Shield } from "lucide-react";
import PanelHeader from "@/components/ui/PanelHeader";
import StatusBadge from "@/components/ui/StatusBadge";
import DataTable from "@/components/ui/DataTable";

export default function DeployPage() {
  const [allModels, setAllModels] = useState<TrainedModel[]>([]);
  const [loading, setLoading] = useState(true);
  const [endpointModelId, setEndpointModelId] = useState<number | "">("");
  const [endpointPayload, setEndpointPayload] = useState("{}");
  const [endpointResult, setEndpointResult] = useState<PredictionResult | null>(null);
  const [endpointBusy, setEndpointBusy] = useState(false);
  const [endpointHint, setEndpointHint] = useState<string | null>(null);

  const fetchModels = async () => {
    try {
      const res = await modelsAPI.list();
      setAllModels(res.data);

      const deployed = (res.data as TrainedModel[]).filter((m) => m.is_deployed);
      if (deployed.length > 0) {
        setEndpointModelId((prev) => (prev === "" ? deployed[0].id : prev));
      } else {
        setEndpointModelId("");
      }
    } catch (e: any) {
      const msg = e?.response?.data?.detail || e?.message || "Could not load models";
      toast.error(msg);
      console.error("Failed to fetch models:", e);
    }
    finally { setLoading(false); }
  };

  useEffect(() => { fetchModels(); }, []);

  const deploy = async (id: number) => {
    try {
      await modelsAPI.deploy(id);
      toast.success("Model deployed!");
      fetchModels();
    } catch (e: any) {
      toast.error(e?.response?.data?.detail || "Deploy failed");
    }
  };

  const undeploy = async (id: number) => {
    try {
      await modelsAPI.undeploy(id);
      toast.success("Model undeployed");
      fetchModels();
    } catch (e: any) {
      toast.error(e?.response?.data?.detail || "Undeploy failed");
    }
  };

  const deleteModel = async (id: number) => {
    if (!confirm("Delete this model?")) return;
    try {
      await modelsAPI.delete(id);
      toast.success("Deleted");
      fetchModels();
    } catch (e: any) {
      toast.error(e?.response?.data?.detail || "Delete failed");
    }
  };

  const deployedModels = allModels.filter(m => m.is_deployed);

  useEffect(() => {
    if (endpointModelId === "") {
      setEndpointPayload("{}");
      setEndpointHint(null);
      return;
    }

    modelsAPI.inferenceTemplate(Number(endpointModelId), true)
      .then((res) => {
        const features: Record<string, unknown> = {};
        (res.data.features || []).forEach((f: InferenceFeatureTemplate) => {
          features[f.name] = f.default_value;
        });
        setEndpointPayload(JSON.stringify(features, null, 2));
        if (res.data.artifact_compatible === false) {
          setEndpointHint("Loaded fallback template (artifact compatibility warning). Consider retraining this model.");
        } else {
          setEndpointHint(null);
        }
      })
      .catch(async () => {
        try {
          const modelRes = await modelsAPI.get(Number(endpointModelId));
          const fi = modelRes.data?.feature_importance || {};
          const fallback = Object.keys(fi).reduce<Record<string, unknown>>((acc, key) => {
            acc[key] = null;
            return acc;
          }, {});
          setEndpointPayload(JSON.stringify(fallback, null, 2));
          setEndpointHint("Template unavailable for this model. Using fallback payload from feature schema.");
        } catch {
          setEndpointPayload("{}");
          setEndpointHint("Template unavailable. Enter payload manually.");
        }
      });
  }, [endpointModelId]);

  const runEndpointOperation = async () => {
    if (endpointModelId === "") return toast.error("Select a deployed model first");

    let parsed: Record<string, unknown>;
    try {
      parsed = JSON.parse(endpointPayload || "{}");
    } catch {
      return toast.error("Payload must be valid JSON");
    }

    setEndpointBusy(true);
    try {
      const res = await predictionsAPI.predict(Number(endpointModelId), parsed);
      setEndpointResult(res.data);
      toast.success("Endpoint operation succeeded");
    } catch (e: any) {
      toast.error(e?.response?.data?.detail || "Endpoint call failed");
    } finally {
      setEndpointBusy(false);
    }
  };

  return (
    <div className="mx-auto max-w-7xl">
      <div className="mb-8">
        <div className="section-kicker mb-2">Deployments</div>
        <h1 className="font-display mb-1 text-4xl font-semibold text-text-primary">Endpoint Operations</h1>
        <p className="text-text-muted">Operate deployed models with status visibility and safe actions.</p>
      </div>

      {/* Deployed banner */}
      <div className="panel mb-6 rounded-xl p-6">
        <PanelHeader title="Deployed Endpoints" icon={Rocket} className="mb-3" />
        {deployedModels.length === 0 ? (
          <p className="text-sm text-text-muted">No models deployed. Deploy one below to serve predictions.</p>
        ) : (
          <div className="flex flex-wrap gap-3">
            {deployedModels.map(m => (
              <div key={m.id} className="flex items-center gap-2 rounded-lg border border-emerald-500/30 bg-emerald-500/10 px-3 py-2">
                <CheckCircle className="h-4 w-4 text-emerald-300" />
                <span className="text-sm font-medium text-text-primary">{m.model_name}</span>
                <StatusBadge label={m.task_type} tone="healthy" />
              </div>
            ))}
          </div>
        )}
      </div>

      {/* API info */}
      <div className="panel mb-6 rounded-xl border-primary/20 p-6">
        <PanelHeader title="API Control" subtitle="Once deployed, call the prediction endpoint." icon={Shield} className="mb-3" />
        <div className="rounded-lg bg-surface-variant/35 p-4 font-mono text-sm mb-4">
          <div className="text-primary">POST</div>
          <div className="text-text-primary">/api/predict</div>
          <div className="mt-2 text-text-muted">{"{"}</div>
          <div className="ml-4 text-text-primary">{`  "model_id": 1,`}</div>
          <div className="ml-4 text-text-primary">{`  "features": {"feature1": value, ...}`}</div>
          <div className="text-text-muted">{"}"}</div>
        </div>

        <div className="grid gap-3">
          <div>
            <label className="mb-1.5 block text-xs text-text-muted">Deployed model</label>
            <select
              value={endpointModelId}
              onChange={(e) => setEndpointModelId(e.target.value ? Number(e.target.value) : "")}
              className="w-full rounded-lg border border-outline/25 bg-surface px-3 py-2 text-sm text-text-primary focus:outline-none focus:border-primary/50"
            >
              <option value="">Select deployed model...</option>
              {deployedModels.map((m) => (
                <option key={m.id} value={m.id}>{m.model_name} (id:{m.id})</option>
              ))}
            </select>
          </div>
          <div>
            <label className="mb-1.5 block text-xs text-text-muted">Features JSON</label>
            <textarea
              value={endpointPayload}
              onChange={(e) => setEndpointPayload(e.target.value)}
              rows={8}
              className="w-full rounded-lg border border-outline/25 bg-surface px-3 py-2 font-mono text-xs text-text-primary focus:outline-none focus:border-primary/50"
            />
          </div>
          <button
            onClick={runEndpointOperation}
            disabled={endpointBusy || endpointModelId === ""}
            className="btn-primary w-fit disabled:opacity-50"
          >
            {endpointBusy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Rocket className="h-4 w-4" />} Run Endpoint Operation
          </button>

          {endpointHint && (
            <div className="rounded-lg border border-amber-500/25 bg-amber-500/10 p-2 text-xs text-amber-200">
              {endpointHint}
            </div>
          )}

          {endpointResult && (
            <div className="rounded-lg border border-outline/25 bg-surface-variant/25 p-3 text-xs">
              <div className="mb-1 text-text-muted">Response</div>
              <pre className="max-h-48 overflow-auto text-text-primary">{JSON.stringify(endpointResult, null, 2)}</pre>
            </div>
          )}
        </div>
      </div>

      {/* All models table */}
      <div className="panel rounded-xl p-6">
        <PanelHeader title="Runtime Endpoints" subtitle="Status, latency readiness and control actions." className="mb-4" />
        {loading ? (
          <div className="flex justify-center py-10">
            <Loader2 className="h-6 w-6 animate-spin text-primary" />
          </div>
        ) : allModels.length === 0 ? (
          <div className="py-12 text-center text-sm text-text-muted">
            No models yet. Start training from the Training page.
          </div>
        ) : (
          <DataTable
            data={allModels}
            rowKey={(m) => m.id}
            columns={[
              { key: "model", header: "Model", render: (m) => <span className="font-medium text-text-primary">{m.model_name}</span> },
              { key: "type", header: "Type", className: "text-xs text-text-muted", render: (m) => m.model_type },
              { key: "task", header: "Task", render: (m) => <StatusBadge label={m.task_type} tone="primary" /> },
              {
                key: "metric",
                header: "Primary Metric",
                align: "right",
                className: "font-mono text-xs text-text-primary",
                render: (m) => m.task_type === "classification"
                  ? m.accuracy ? `Acc ${(m.accuracy * 100).toFixed(1)}%` : "–"
                  : m.r2_score ? `R² ${m.r2_score.toFixed(3)}` : "–",
              },
              {
                key: "status",
                header: "Status",
                render: (m) => m.is_deployed
                  ? <StatusBadge label="deployed" tone="healthy" icon={<CheckCircle className="h-3 w-3" />} />
                  : <StatusBadge label="idle" tone="idle" />,
              },
              {
                key: "actions",
                header: "Actions",
                align: "right",
                render: (m) => (
                  <div className="flex items-center justify-end gap-2">
                    {m.is_deployed ? (
                      <button onClick={() => undeploy(m.id)}
                        className="btn-outline border-red-500/30 text-xs text-red-300 hover:border-red-500/50">
                        <XCircle className="h-3 w-3" /> Undeploy
                      </button>
                    ) : (
                      <button onClick={() => deploy(m.id)} className="btn-primary text-xs">
                        <Rocket className="h-3 w-3" /> Deploy
                      </button>
                    )}
                    <a href={modelsAPI.download(m.id)} download
                      className="rounded p-1.5 text-text-muted transition-colors hover:bg-surface-variant hover:text-text-primary">
                      <Download className="h-3.5 w-3.5" />
                    </a>
                    <button onClick={() => deleteModel(m.id)}
                      className="rounded p-1.5 text-text-muted transition-colors hover:bg-red-500/20 hover:text-red-300">
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  </div>
                ),
              },
            ]}
          />
        )}
      </div>
    </div>
  );
}
