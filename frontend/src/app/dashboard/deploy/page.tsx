"use client";
import { useState, useEffect } from "react";
import { modelsAPI } from "@/lib/api";
import toast from "react-hot-toast";
import { Rocket, Download, CheckCircle, XCircle, Trash2, Loader2, Shield } from "lucide-react";
import PanelHeader from "@/components/ui/PanelHeader";
import StatusBadge from "@/components/ui/StatusBadge";
import DataTable from "@/components/ui/DataTable";

export default function DeployPage() {
  const [allModels, setAllModels] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchModels = async () => {
    try {
      const res = await modelsAPI.list();
      setAllModels(res.data);
    } catch { toast.error("Could not load models"); }
    finally { setLoading(false); }
  };

  useEffect(() => { fetchModels(); }, []);

  const deploy = async (id: number) => {
    try {
      await modelsAPI.deploy(id);
      toast.success("Model deployed!");
      fetchModels();
    } catch { toast.error("Deploy failed"); }
  };

  const undeploy = async (id: number) => {
    try {
      await modelsAPI.undeploy(id);
      toast.success("Model undeployed");
      fetchModels();
    } catch { toast.error("Failed"); }
  };

  const deleteModel = async (id: number) => {
    if (!confirm("Delete this model?")) return;
    try {
      await modelsAPI.delete(id);
      toast.success("Deleted");
      fetchModels();
    } catch { toast.error("Delete failed"); }
  };

  const deployedModels = allModels.filter(m => m.is_deployed);

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
        <div className="rounded-lg bg-surface-variant/35 p-4 font-mono text-sm">
          <div className="text-primary">POST</div>
          <div className="text-text-primary">/api/predict</div>
          <div className="mt-2 text-text-muted">{"{"}</div>
          <div className="ml-4 text-text-primary">{`  "model_id": 1,`}</div>
          <div className="ml-4 text-text-primary">{`  "features": {"feature1": value, ...}`}</div>
          <div className="text-text-muted">{"}"}</div>
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
