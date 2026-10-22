import uuid
from datetime import datetime, timezone
from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    Boolean,
    DateTime,
    ForeignKey,
    JSON,
    Text,
)
from sqlalchemy.orm import relationship

from backend.app.database import Base

def generate_uuid() -> str:
    return str(uuid.uuid4())

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

class ImageRecord(Base):
    __tablename__ = "images"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    mime_type = Column(String(64), nullable=False)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    sha256_hash = Column(String(64), nullable=False, index=True)
    created_at = Column(DateTime, default=utc_now)
    is_protected_output = Column(Boolean, default=False)
    
    # Relationships
    jobs = relationship(
        "ProtectionJob",
        primaryjoin="ImageRecord.id == ProtectionJob.image_id",
        back_populates="image",
        cascade="all, delete-orphan"
    )


class ProtectionConfig(Base):
    __tablename__ = "protection_configs"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    protection_mode = Column(String(64), nullable=False, default="artwork")  # artwork, general, portrait
    surrogate_model = Column(String(64), nullable=False, default="resnet50")
    evaluation_models = Column(JSON, nullable=False)  # list of model identifiers
    epsilon = Column(Float, nullable=False, default=8.0/255.0)
    iterations = Column(Integer, nullable=False, default=20)
    step_size = Column(Float, nullable=False, default=2.0/255.0)
    loss_objective = Column(String(64), nullable=False, default="classification_disruption")
    random_init = Column(Boolean, default=True)
    random_seed = Column(Integer, default=42)
    created_at = Column(DateTime, default=utc_now)

    # Relationships
    job = relationship("ProtectionJob", back_populates="config", uselist=False)


class ProtectionJob(Base):
    __tablename__ = "protection_jobs"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    image_id = Column(String(36), ForeignKey("images.id"), nullable=False)
    config_id = Column(String(36), ForeignKey("protection_configs.id"), nullable=False)
    status = Column(String(32), default="pending")  # pending, processing, completed, failed
    progress = Column(Float, default=0.0)
    error_message = Column(Text, nullable=True)
    
    # Output image reference
    protected_image_id = Column(String(36), ForeignKey("images.id"), nullable=True)
    difference_map_path = Column(String(512), nullable=True)
    
    # Composite metrics summary
    overall_protection_score = Column(Float, nullable=True)
    ai_disruption_score = Column(Float, nullable=True)
    transferability_score = Column(Float, nullable=True)
    robustness_score = Column(Float, nullable=True)
    perceptual_quality_score = Column(Float, nullable=True)
    
    created_at = Column(DateTime, default=utc_now)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    image = relationship("ImageRecord", foreign_keys=[image_id], back_populates="jobs")
    protected_image = relationship("ImageRecord", foreign_keys=[protected_image_id])
    config = relationship("ProtectionConfig", back_populates="job")
    evaluations = relationship("EvaluationResult", back_populates="job", cascade="all, delete-orphan")
    robustness_results = relationship("RobustnessResult", back_populates="job", cascade="all, delete-orphan")
    report = relationship("ReportRecord", back_populates="job", uselist=False, cascade="all, delete-orphan")


class EvaluationResult(Base):
    __tablename__ = "evaluation_results"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    job_id = Column(String(36), ForeignKey("protection_jobs.id"), nullable=False)
    model_name = Column(String(64), nullable=False)
    is_surrogate = Column(Boolean, default=False)
    status = Column(String(32), default="evaluated")
    status_detail = Column(Text, nullable=True)
    evaluation_type = Column(String(32), default="classifier")
    prompt_template = Column(String(128), nullable=True)
    
    # Baseline results
    original_top1_class = Column(String(128), nullable=True)
    original_top1_confidence = Column(Float, nullable=True)
    
    # Protected results
    protected_top1_class = Column(String(128), nullable=True)
    protected_top1_confidence = Column(Float, nullable=True)
    
    # Metrics
    confidence_drop = Column(Float, nullable=True)
    classification_disrupted = Column(Boolean, nullable=True)
    embedding_cosine_distance = Column(Float, nullable=True)
    embedding_l2_distance = Column(Float, nullable=True)
    
    created_at = Column(DateTime, default=utc_now)

    # Relationships
    job = relationship("ProtectionJob", back_populates="evaluations")


class RobustnessResult(Base):
    __tablename__ = "robustness_results"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    job_id = Column(String(36), ForeignKey("protection_jobs.id"), nullable=False)
    transformation_name = Column(String(64), nullable=False)
    transformation_params = Column(JSON, nullable=True)
    
    retained_protection_rate = Column(Float, nullable=False)
    top1_class_after_transform = Column(String(128), nullable=True)
    top1_confidence_after_transform = Column(Float, nullable=True)
    embedding_distance_after_transform = Column(Float, nullable=True)
    
    created_at = Column(DateTime, default=utc_now)

    # Relationships
    job = relationship("ProtectionJob", back_populates="robustness_results")


class ReportRecord(Base):
    __tablename__ = "reports"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    job_id = Column(String(36), ForeignKey("protection_jobs.id"), nullable=False)
    report_json = Column(JSON, nullable=False)
    pdf_path = Column(String(512), nullable=True)
    created_at = Column(DateTime, default=utc_now)

    # Relationships
    job = relationship("ProtectionJob", back_populates="report")
