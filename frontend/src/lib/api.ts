import axios from "axios";
import { logToFile } from "./logger";

const API_URL = (process.env.NEXT_PUBLIC_API_URL || "").replace(/\/$/, "");
const API_BASE_URL = API_URL ? `${API_URL}/api` : "/api";


export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: { "Content-Type": "application/json" },
  timeout: 120000, // 2 min for training
});

// Log errors to file (Node.js only)
if (typeof window === "undefined") {
  api.interceptors.response.use(
    (response) => response,
    (error) => {
      const msg = error?.message || "Unknown error";
      const url = error?.config?.url || "unknown";
      const method = error?.config?.method || "unknown";
      const status = error?.response?.status || "no-status";
      logToFile(`API error: [${method}] ${url} status=${status} msg=${msg}`,'error');
      return Promise.reject(error);
    }
  );
}

// ─── Dataset APIs ─────────────────────────────────────────────────────────────
export const datasetsAPI = {
  upload: (formData: FormData) =>
    api.post("/upload-dataset", formData, { headers: { "Content-Type": "multipart/form-data" } }),
  list: () => api.get("/datasets"),
  get: (id: number) => api.get(`/datasets/${id}`),
  qualityReport: (id: number) => api.get(`/datasets/${id}/quality-report`),
  cleanPreview: (id: number, previewRows: number = 10) =>
    api.get(`/datasets/${id}/clean-preview`, { params: { preview_rows: previewRows } }),
  cleanAndSave: (id: number, nameSuffix: string = "cleaned") =>
    api.post(`/datasets/${id}/clean-and-save`, null, { params: { name_suffix: nameSuffix } }),
  delete: (id: number) => api.delete(`/datasets/${id}`),
  loadExample: (key: string) => api.post(`/load-example/${key}`),
  listExamples: () => api.get("/example-datasets"),
};

// ─── Training APIs ────────────────────────────────────────────────────────────
export const trainingAPI = {
  train: (config: {
    dataset_id: number;
    target_column: string;
    task_type?: string;
    test_size?: number;
    cv_folds?: number;
  }) => api.post("/train-model", config),
  getStatus: (jobId: string) => api.get(`/training-status/${jobId}`),
  listJobs: () => api.get("/training-jobs"),
};

// ─── Model APIs ───────────────────────────────────────────────────────────────
export const modelsAPI = {
  list: (jobId?: string) => api.get("/models", { params: jobId ? { job_id: jobId } : {} }),
  get: (id: number) => api.get(`/models/${id}`),
  shap: (id: number, nsamples: number = 100) => api.get(`/models/${id}/shap`, { params: { nsamples } }),
  lime: (id: number, nsamples: number = 5) => api.get(`/models/${id}/lime`, { params: { nsamples } }),
  monitoring: (id: number, compareDatasetId?: number) =>
    api.get(`/models/${id}/monitoring`, { params: compareDatasetId ? { compare_dataset_id: compareDatasetId } : {} }),
  deploy: (id: number) => api.post(`/models/${id}/deploy`),
  undeploy: (id: number) => api.post(`/models/${id}/undeploy`),
  download: (id: number) => API_URL ? `${API_URL}/api/models/${id}/download` : `/api/models/${id}/download`,
  delete: (id: number) => api.delete(`/models/${id}`),
  listDeployed: () => api.get("/deployed-models"),
};

// ─── Metrics APIs ─────────────────────────────────────────────────────────────
export const metricsAPI = {
  getJobMetrics: (jobId: string) => api.get(`/metrics/${jobId}`),
  getDashboardSummary: () => api.get("/dashboard/summary"),
};

// ─── Prediction APIs ──────────────────────────────────────────────────────────
export const predictionsAPI = {
  predict: (modelId: number, features: Record<string, unknown>) =>
    api.post("/predict", { model_id: modelId, features }),
};

export default api;
