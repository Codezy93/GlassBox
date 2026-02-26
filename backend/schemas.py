"""
Pydantic request/response schemas for the GlassBox FastAPI service.
"""

from pydantic import BaseModel
from typing import Optional


class PredictionRequest(BaseModel):
    """Profile input for prediction / explanation."""
    credit_limit: float
    sex: int
    education: int
    marriage: int
    age: int
    repayment_sep: int
    repayment_aug: int
    repayment_jul: int
    repayment_jun: int
    repayment_may: int
    repayment_apr: int
    bill_sep: float
    bill_aug: float
    bill_jul: float
    bill_jun: float
    bill_may: float
    bill_apr: float
    pay_sep: float
    pay_aug: float
    pay_jul: float
    pay_jun: float
    pay_may: float
    pay_apr: float

    model_config = {"json_schema_extra": {
        "examples": [{
            "credit_limit": 50000,
            "sex": 2,
            "education": 2,
            "marriage": 1,
            "age": 35,
            "repayment_sep": 0,
            "repayment_aug": 0,
            "repayment_jul": -1,
            "repayment_jun": -1,
            "repayment_may": -1,
            "repayment_apr": -1,
            "bill_sep": 45000,
            "bill_aug": 42000,
            "bill_jul": 38000,
            "bill_jun": 35000,
            "bill_may": 30000,
            "bill_apr": 28000,
            "pay_sep": 2000,
            "pay_aug": 2000,
            "pay_jul": 1500,
            "pay_jun": 1500,
            "pay_may": 1000,
            "pay_apr": 1000,
        }]
    }}


class PredictionResponse(BaseModel):
    """Standard prediction result with Conformal Uncertainty."""
    prediction: int
    probability_default: float
    probability_no_default: float
    label: str
    conformal_set_95: list[int]  # Certified set at 5% error rate
    conformal_set_99: list[int]  # Certified set at 1% error rate


class FeatureChange(BaseModel):
    """A single feature change in a counterfactual."""
    original: float
    suggested: float


class Counterfactual(BaseModel):
    """A single counterfactual explanation path."""
    changes: dict[str, FeatureChange]
    probability_no_default: float
    probability_default: float
    full_profile: dict[str, float]


class ExplanationResponse(BaseModel):
    """Response from the /explain endpoint with recourse certification."""
    prediction: int
    probability_default: float
    probability_no_default: float
    label: str
    counterfactuals: list[Counterfactual]
    local_importance: dict[str, float]  # SHAP values
    conformal_set_95: list[int]
    conformal_set_99: list[int]
    certification: dict  # Formal proof from Z3


class FeatureMetadata(BaseModel):
    """Metadata for a single feature (for the frontend)."""
    name: str
    type: str
    immutable: bool
    min: float
    max: float
    mean: float
    median: float
    values: Optional[list[float]] = None
    permitted_range: Optional[list[float]] = None
    labels: Optional[dict[str, str]] = None


class GlobalInsightsResponse(BaseModel):
    """Global feature importance (SHAP)."""
    importance: dict[str, float]


class FairnessAuditResponse(BaseModel):
    """Result of a fairness audit."""
    feature: str
    demographic_parity_difference: float
    demographic_parity_ratio: float
    group_selection_rates: dict[str, float]
    status: str


class ComprehensiveFairnessResponse(BaseModel):
    """Audit across all attributes."""
    results: dict[str, FairnessAuditResponse]


class CausalEdge(BaseModel):
    source: str
    target: str


class CausalNode(BaseModel):
    id: str


class CausalGraphResponse(BaseModel):
    """Graph structure for D3 visualization."""
    nodes: list[CausalNode]
    links: list[CausalEdge]


class ManifoldProjectionResponse(BaseModel):
    """2D Manifold projection for visualization."""
    points: list[list[float]]
    user_point: list[float]


class StabilityAuditResponse(BaseModel):
    """Performative stability metrics."""
    strategic_gaming_ratio: float
    distribution_shift_score: float
    adoption_rate: float
    gameability_report: dict[str, float]


class PrivacyAuditResponse(BaseModel):
    """Differential privacy budget."""
    epsilon: float
    delta: float
    mechanism: str
    certified: bool


class RobustnessResponse(BaseModel):
    """Adversarial robustness metrics."""
    robustness_score: float
    epsilon_radius: float
    feature_sensitivity_rank: dict[str, float]
    n_perturbations: int
