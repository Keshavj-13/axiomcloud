"""
Axiom Cloud AI - SQLAlchemy Database Models
"""
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, JSON, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    file_size = Column(Integer)
    num_rows = Column(Integer)
    num_columns = Column(Integer)
    columns_info = Column(JSON)  # column names, types, stats
    target_column = Column(String(255))
    task_type = Column(String(50))  # classification / regression
    is_example = Column(Boolean, default=False)
    preview_data = Column(JSON)  # first 10 rows
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    training_jobs = relationship("TrainingJob", back_populates="dataset")


class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String(100), unique=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    task_type = Column(String(50))
    target_column = Column(String(255))
    status = Column(String(50), default="pending")  # pending, running, completed, failed
    progress = Column(Integer, default=0)
    error_message = Column(Text)
    config = Column(JSON)  # training config
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True))

    # Relationships
    dataset = relationship("Dataset", back_populates="training_jobs")
    trained_models = relationship("TrainedModel", back_populates="training_job")


class TrainedModel(Base):
    __tablename__ = "trained_models"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String(100), ForeignKey("training_jobs.job_id"))
    model_name = Column(String(255), nullable=False)
    model_type = Column(String(100))  # e.g., RandomForest, XGBoost
    task_type = Column(String(50))
    file_path = Column(String(512))
    is_deployed = Column(Boolean, default=False)

    # Metrics
    accuracy = Column(Float)
    f1_score = Column(Float)
    roc_auc = Column(Float)
    rmse = Column(Float)
    mae = Column(Float)
    r2_score = Column(Float)

    # Additional data
    metrics = Column(JSON)           # Full metrics dict
    feature_importance = Column(JSON)  # Feature importance scores
    confusion_matrix = Column(JSON)    # Confusion matrix
    roc_curve_data = Column(JSON)      # ROC curve points
    cv_scores = Column(JSON)           # Cross-validation scores
    training_time = Column(Float)      # Training time in seconds

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    training_job = relationship("TrainingJob", back_populates="trained_models")
