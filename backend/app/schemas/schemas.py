from datetime import datetime
from typing import Any, Optional, List, Dict
from pydantic import BaseModel, Field


class ImageUploadResponse(BaseModel):
    image_id: str
    filename: str
    original_filename: str
    width: int
    height: int
    file_size_bytes: int
    mime_type: str
    sha256_hash: str
    preview_url: str
    created_at: datetime

    model_config = {"from_attributes": True}


class ProtectionConfigRequest(BaseModel):
    image_id: str
    protection_mode: str = Field(default="artwork", description="artwork, general, or portrait")
    surrogate_model: str = Field(default="resnet50")
    evaluation_models: list[str] = Field(default_factory=lambda: ["resnet50", "convnext_tiny", "vit_b_16", "clip_vit_b32"])
    epsilon: float = Field(default=8.0/255.0, ge=0.001, le=0.2)
    iterations: int = Field(default=20, ge=1, le=100)
    step_size: float = Field(default=2.0/255.0, ge=0.0001, le=0.1)
    loss_objective: str = Field(default="classification_disruption")
    random_init: bool = True
    random_seed: int = 42


class PredictionItem(BaseModel):
    class_idx: int
    style_name: str
    confidence: float
    confidence_percent: float


class SingleModelPrediction(BaseModel):
    style_name: str
    class_idx: int
    confidence: float
    confidence_percent: float
    top5: list[PredictionItem] = []


class ClassificationDisruptionDetail(BaseModel):
    original_prediction: SingleModelPrediction
    cloaked_prediction: SingleModelPrediction
    top1_prediction_changed: bool
    classification_disrupted: bool
    confidence_drop: float
    confidence_drop_percent: float
    embedding_cosine_distance: float
    embedding_l2_distance: float
    scientific_disclaimer: str


class PerceptualQualityMetrics(BaseModel):
    ssim: float | None = None
    psnr_db: float | None = None
    lpips: float | None = None
    lpips_status: str | None = None
    linf_norm: float | None = None
    l2_norm: float | None = None
    rmse: float | None = None
    mae: float | None = None
    perturbation_percent: float | None = None
    epsilon_configured: float | None = None
    epsilon_tolerance: float | None = None
    epsilon_passed: bool | None = None
    color_space: str = "sRGB"
    input_range: str = "[0.0, 1.0]"


class ModelEvaluationDetail(BaseModel):
    model_name: str
    is_surrogate: bool
    status: str = "evaluated"  # evaluated, unavailable, failed
    status_detail: str | None = None
    original_top1_class: str | None = None
    original_top1_confidence: float | None = None
    protected_top1_class: str | None = None
    protected_top1_confidence: float | None = None
    confidence_drop: float | None = None
    classification_disrupted: bool | None = None
    embedding_cosine_distance: float | None = None
    embedding_l2_distance: float | None = None
    evaluation_type: str = "classifier"
    prompt_template: str | None = None


class RobustnessEvaluationDetail(BaseModel):
    transformation_name: str
    transformation_params: dict[str, Any] = Field(default_factory=dict)
    retained_protection_rate: float
    top1_class_after_transform: str | None = None
    top1_confidence_after_transform: float | None = None
    prediction_disrupted: bool = False
    status: str = "Robust"
    ssim_after_transform: float | None = None


class ProtectionJobResponse(BaseModel):
    job_id: str
    image_id: str
    config_id: str
    status: str
    progress: float
    error_message: str | None = None
    original_image_url: str | None = None
    protected_image_url: str | None = None
    protected_image_id: str | None = None
    difference_map_url: str | None = None
    overall_protection_score: float | None = None
    ai_disruption_score: float | None = None
    transferability_score: float | None = None
    robustness_score: float | None = None
    perceptual_quality_score: float | None = None
    
    # Detailed scientific evaluation payload
    classification: Optional[ClassificationDisruptionDetail] = None
    quality_metrics: Optional[PerceptualQualityMetrics] = None
    evaluations: List[ModelEvaluationDetail] = Field(default_factory=list)
    robustness: List[RobustnessEvaluationDetail] = Field(default_factory=list)
    report_url: Optional[str] = None

    created_at: datetime
    completed_at: datetime | None = None

    model_config = {"from_attributes": True}


class ProtectionResultResponse(BaseModel):
    job: ProtectionJobResponse
    original_image_url: str
    protected_image_url: str | None = None
    difference_map_url: str | None = None
    evaluations: list[ModelEvaluationDetail] = []
    robustness: list[RobustnessEvaluationDetail] = []
    quality_metrics: PerceptualQualityMetrics = Field(default_factory=PerceptualQualityMetrics)
    classification: Optional[ClassificationDisruptionDetail] = None
    disclaimer: str = (
        "Adversarial cloaking provides empirical, non-cryptographic protection against supported models and tasks. "
        "Effectiveness depends on model architectures, attack hyper-parameters, and subsequent transformations (e.g. heavy compression or filtering)."
    )


class HealthCheckResponse(BaseModel):
    status: str
    project: str
    version: str
    database_connected: bool
    torch_available: bool
    cuda_available: bool
    available_surrogates: list[str]
    available_eval_models: list[str]


class JobSummaryItem(BaseModel):
    job_id: str
    image_id: str
    original_filename: str
    original_image_url: str
    protected_image_url: str | None = None
    difference_map_url: str | None = None
    surrogate_model: str
    status: str
    progress: float
    overall_protection_score: float | None = None
    ai_disruption_score: float | None = None
    transferability_score: float | None = None
    perceptual_quality_score: float | None = None
    ssim: float | None = None
    top1_prediction_changed: bool | None = None
    original_style: str | None = None
    cloaked_style: str | None = None
    created_at: datetime
    completed_at: datetime | None = None


class CustomRobustnessRequest(BaseModel):
    transformation: str  # jpeg, resize, crop, blur, brightness, contrast
    param_value: float


class CustomRobustnessResponse(BaseModel):
    job_id: str
    transformation_name: str
    param_value: float
    top1_class_after_transform: str
    top1_confidence_after_transform: float
    prediction_disrupted: bool
    retained_protection_rate: float
    ssim_after_transform: float | None = None
    preview_url: str | None = None
