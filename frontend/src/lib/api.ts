import axios from "axios";

export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000/api";
export const BACKEND_ROOT_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://127.0.0.1:8000";

export interface ImageUploadResponse {
  image_id: string;
  filename: string;
  original_filename: string;
  width: number;
  height: number;
  file_size_bytes: number;
  mime_type: string;
  sha256_hash: string;
  preview_url: string;
  created_at: string;
}

export interface ProtectionConfigRequest {
  image_id: string;
  protection_mode: "artwork" | "general" | "portrait";
  surrogate_model: string;
  evaluation_models: string[];
  epsilon: number;
  iterations: number;
  step_size: number;
  loss_objective?: string;
  random_init?: boolean;
  random_seed?: number;
}

export interface PredictionItem {
  class_idx: number;
  style_name: string;
  confidence: number;
  confidence_percent: number;
}

export interface SingleModelPrediction {
  style_name: string;
  class_idx: number;
  confidence: number;
  confidence_percent: number;
  top5: PredictionItem[];
}

export interface ClassificationDisruptionDetail {
  original_prediction: SingleModelPrediction;
  cloaked_prediction: SingleModelPrediction;
  top1_prediction_changed: boolean;
  classification_disrupted: boolean;
  confidence_drop: number;
  confidence_drop_percent: number;
  embedding_cosine_distance: number;
  embedding_l2_distance: number;
  scientific_disclaimer: string;
}

export interface PerceptualQualityMetrics {
  ssim?: number | null;
  psnr_db?: number | null;
  lpips?: number | null;
  lpips_status?: string | null;
  linf_norm?: number | null;
  l2_norm?: number | null;
  rmse?: number | null;
  mae?: number | null;
  perturbation_percent?: number | null;
  epsilon_configured?: number | null;
  epsilon_tolerance?: number | null;
  epsilon_passed?: boolean | null;
  color_space?: string;
  input_range?: string;
}

export interface ModelEvaluationDetail {
  model_name: string;
  is_surrogate: boolean;
  status: "evaluated" | "not_configured" | "loading" | "unavailable" | "failed";
  status_detail?: string | null;
  original_top1_class?: string | null;
  original_top1_confidence?: number | null;
  protected_top1_class?: string | null;
  protected_top1_confidence?: number | null;
  confidence_drop?: number | null;
  classification_disrupted?: boolean | null;
  embedding_cosine_distance?: number | null;
  embedding_l2_distance?: number | null;
  evaluation_type?: "classifier" | "zero_shot_multimodal" | null;
  prompt_template?: string | null;
}

export interface RobustnessEvaluationDetail {
  transformation_name: string;
  transformation_params?: Record<string, any>;
  retained_protection_rate: number;
  top1_class_after_transform?: string | null;
  top1_confidence_after_transform?: number | null;
  prediction_disrupted: boolean;
  status: string;
  ssim_after_transform?: number | null;
}

export interface ProtectionJobResponse {
  job_id: string;
  image_id: string;
  config_id: string;
  status: "pending" | "processing" | "completed" | "failed";
  progress: number;
  error_message?: string | null;
  original_image_url?: string | null;
  protected_image_url?: string | null;
  protected_image_id?: string | null;
  difference_map_url?: string | null;
  overall_protection_score?: number | null;
  ai_disruption_score?: number | null;
  transferability_score?: number | null;
  robustness_score?: number | null;
  perceptual_quality_score?: number | null;
  
  // Detailed scientific evaluation payload
  classification?: ClassificationDisruptionDetail | null;
  quality_metrics?: PerceptualQualityMetrics | null;
  evaluations?: ModelEvaluationDetail[];
  robustness?: RobustnessEvaluationDetail[];
  report_url?: string | null;

  created_at: string;
  completed_at?: string | null;
}

export interface HealthCheckResponse {
  status: string;
  project: string;
  version: string;
  database_connected: boolean;
  torch_available: boolean;
  cuda_available: boolean;
  available_surrogates: string[];
  available_eval_models: string[];
}

export interface JobSummaryItem {
  job_id: string;
  image_id: string;
  original_filename: string;
  original_image_url: string;
  protected_image_url?: string | null;
  difference_map_url?: string | null;
  surrogate_model: string;
  status: "pending" | "processing" | "completed" | "failed";
  progress: number;
  overall_protection_score?: number | null;
  ai_disruption_score?: number | null;
  transferability_score?: number | null;
  perceptual_quality_score?: number | null;
  ssim?: number | null;
  top1_prediction_changed?: boolean | null;
  original_style?: string | null;
  cloaked_style?: string | null;
  created_at: string;
  completed_at?: string | null;
}

export interface CustomRobustnessRequest {
  transformation: "jpeg" | "resize" | "crop" | "blur" | "brightness" | "contrast";
  param_value: number;
}

export interface CustomRobustnessResponse {
  job_id: string;
  transformation_name: string;
  param_value: number;
  top1_class_after_transform: string;
  top1_confidence_after_transform: number;
  prediction_disrupted: boolean;
  retained_protection_rate: number;
  ssim_after_transform?: number | null;
  preview_url?: string | null;
}

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

export async function uploadImage(file: File): Promise<ImageUploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await apiClient.post<ImageUploadResponse>("/upload", formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });
  return response.data;
}

export async function startProtection(config: ProtectionConfigRequest): Promise<ProtectionJobResponse> {
  const response = await apiClient.post<ProtectionJobResponse>("/protect", config);
  return response.data;
}

export async function getJobStatus(jobId: string): Promise<ProtectionJobResponse> {
  const response = await apiClient.get<ProtectionJobResponse>(`/jobs/${jobId}`);
  return response.data;
}

export async function listJobs(): Promise<JobSummaryItem[]> {
  const response = await apiClient.get<JobSummaryItem[]>("/jobs");
  return response.data;
}

export async function deleteJob(jobId: string): Promise<{ status: string; job_id: string }> {
  const response = await apiClient.delete<{ status: string; job_id: string }>(`/jobs/${jobId}`);
  return response.data;
}

export async function testCustomRobustness(
  jobId: string,
  req: CustomRobustnessRequest
): Promise<CustomRobustnessResponse> {
  const response = await apiClient.post<CustomRobustnessResponse>(`/jobs/${jobId}/robustness`, req);
  return response.data;
}

export async function getReportJson(jobId: string): Promise<any> {
  const response = await apiClient.get<any>(`/reports/${jobId}`);
  return response.data;
}

export async function getHealth(): Promise<HealthCheckResponse> {
  const response = await apiClient.get<HealthCheckResponse>("/health");
  return response.data;
}

// ----------------------------------------------------------------------------
// Guardian Intelligence Interfaces & API Client
// ----------------------------------------------------------------------------

export interface GuardianPlanStep {
  step_number: number;
  tool_name: string;
  description: string;
  parameters: Record<string, any>;
  requires_approval: boolean;
  status: "pending" | "running" | "completed" | "failed" | "skipped";
}

export interface GuardianResourceEstimate {
  gpu_vram_gb: number;
  estimated_runtime_seconds: number;
  disk_mb: number;
  hardware_device: string;
}

export interface GuardianSuccessCriteria {
  target_metric: string;
  condition: string;
  threshold: number;
  description: string;
}

export interface GuardianPlanResponse {
  plan_id: string;
  version: number;
  title: string;
  goal: string;
  objective_type: string;
  status: "proposed" | "awaiting_approval" | "approved" | "queued" | "running" | "completed" | "failed" | "blocked" | "rejected";
  requires_approval: boolean;
  approval_status: "pending" | "approved" | "rejected" | "invalidated";
  config_hash: string;
  steps: GuardianPlanStep[];
  resource_estimate: GuardianResourceEstimate;
  success_criteria: GuardianSuccessCriteria[];
  missing_prerequisites: string[];
  clarification_needed?: string | null;
  created_at?: string | null;
}

export interface GuardianTimelineLog {
  timestamp: string;
  step_number: number;
  tool_name: string;
  status: "running" | "completed" | "failed";
  message: string;
  measurements?: Record<string, any> | null;
}

export interface GuardianEvidenceSummary {
  cited_experiment_ids: string[];
  empirical_observations: string[];
  hypotheses_tested: string[];
  grounded_recommendations: string[];
  uncertainties_and_caveats: string[];
}

export interface GuardianJobStatusResponse {
  execution_id: string;
  plan_id: string;
  plan_title?: string | null;
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  current_step: number;
  total_steps: number;
  progress: number;
  logs: GuardianTimelineLog[];
  results: Record<string, any>;
  evidence?: GuardianEvidenceSummary | null;
  error_message?: string | null;
  started_at?: string | null;
  completed_at?: string | null;
}

export interface GuardianChatResponse {
  reply: string;
  intent: string;
  plan?: GuardianPlanResponse | null;
  evidence?: GuardianEvidenceSummary | null;
  suggested_actions: string[];
}

export async function chatWithGuardian(req: {
  message: string;
  plan_id?: string;
  image_id?: string;
}): Promise<GuardianChatResponse> {
  const response = await apiClient.post<GuardianChatResponse>("/guardian/chat", req);
  return response.data;
}

export async function createGuardianPlan(req: {
  goal: string;
  image_id?: string;
  constraints?: Record<string, any>;
}): Promise<GuardianPlanResponse> {
  const response = await apiClient.post<GuardianPlanResponse>("/guardian/plan", req);
  return response.data;
}

export async function getGuardianPlan(planId: string): Promise<GuardianPlanResponse> {
  const response = await apiClient.get<GuardianPlanResponse>(`/guardian/plans/${planId}`);
  return response.data;
}

export async function listGuardianPlans(): Promise<any[]> {
  const response = await apiClient.get<any[]>("/guardian/plans");
  return response.data;
}

export async function approveGuardianPlan(req: {
  plan_id: string;
  config_hash: string;
  decision?: string;
  notes?: string;
}): Promise<any> {
  const response = await apiClient.post<any>("/guardian/approve", req);
  return response.data;
}

export async function executeGuardianPlan(req: {
  plan_id: string;
  config_hash: string;
}): Promise<{ success: boolean; execution_id: string; plan_id: string; status: string; message: string }> {
  const response = await apiClient.post<any>("/guardian/execute", req);
  return response.data;
}

export async function getGuardianJobStatus(executionId: string): Promise<GuardianJobStatusResponse> {
  const response = await apiClient.get<GuardianJobStatusResponse>(`/guardian/jobs/${executionId}`);
  return response.data;
}

export async function getGuardianEvidence(planId: string): Promise<any> {
  const response = await apiClient.get<any>(`/guardian/evidence/${planId}`);
  return response.data;
}

export async function listGuardianExperiments(): Promise<any[]> {
  const response = await apiClient.get<any[]>("/guardian/experiments");
  return response.data;
}

export async function compareGuardianExperiments(experimentIds: string[]): Promise<any> {
  const response = await apiClient.post<any>("/guardian/compare", { experiment_ids: experimentIds });
  return response.data;
}

export async function getGuardianModels(): Promise<any> {
  const response = await apiClient.get<any>("/guardian/models");
  return response.data;
}

export async function getGuardianTools(): Promise<any[]> {
  const response = await apiClient.get<any[]>("/guardian/tools");
  return response.data;
}
