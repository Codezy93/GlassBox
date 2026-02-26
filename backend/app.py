"""
GlassBox FastAPI Application

Endpoints:
    GET  /features  → feature metadata for the frontend
    POST /predict   → model prediction (< 100ms)
    POST /explain   → counterfactual explanations
"""

import os
import sys
import time
import traceback

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from schemas import (
    PredictionRequest,
    PredictionResponse,
    ExplanationResponse,
    GlobalInsightsResponse,
    ComprehensiveFairnessResponse,
    CausalGraphResponse,
    ManifoldProjectionResponse,
    StabilityAuditResponse,
    PrivacyAuditResponse,
    RobustnessResponse
)  # noqa: E402
from engine.counterfactual import CounterfactualEngine, get_feature_metadata  # noqa: E402
from engine.interpretability import InterpretabilityEngine  # noqa: E402
from engine.fairness import FairnessAuditor  # noqa: E402
from engine.causal_engine import CausalEngine  # noqa: E402
from engine.conformal_engine import ConformalEngine  # noqa: E402
from engine.manifold_vae import ManifoldEngine  # noqa: E402
from engine.formal_verifier import FormalVerifier  # noqa: E402
from engine.performative_audit import PerformativeAudit  # noqa: E402
from engine.privacy_engine import PrivacyAuditEngine  # noqa: E402
from engine.robustness import RobustnessEngine  # noqa: E402

# ── Configure loguru ─────────────────────────────────────

logger.remove()  # remove default handler
logger.add(sys.stderr, level="DEBUG", format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — <level>{message}</level>")
logger.add("glassbox.log", rotation="5 MB", level="DEBUG")

# ── App setup ────────────────────────────────────────────

app = FastAPI(
    title="GlassBox API",
    description="Counterfactual Explanations for Credit Default Prediction",
    version="1.0.0",
)

DEFAULT_ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
]

origins_env = os.getenv("GLASSBOX_ALLOWED_ORIGINS")
allowed_origins = (
    [o.strip() for o in origins_env.split(",") if o.strip()]
    if origins_env
    else DEFAULT_ALLOWED_ORIGINS
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials="*" not in allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-loaded engines
_engine: CounterfactualEngine | None = None
_interpret_engine: InterpretabilityEngine | None = None
_fairness_auditor: FairnessAuditor | None = None
_causal_engine: CausalEngine | None = None
_conformal_engine: ConformalEngine | None = None
_manifold_engine: ManifoldEngine | None = None
_formal_verifier: FormalVerifier | None = None
_stability_audit: PerformativeAudit | None = None
_privacy_audit: PrivacyAuditEngine | None = None
_robustness_engine: RobustnessEngine | None = None

# Paths to models/data
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models", "saved")
EBM_PATH = os.path.join(MODELS_DIR, "ebm_model.pkl")
SPLITS_PATH = os.path.join(MODELS_DIR, "raw_splits.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")


def get_engine() -> CounterfactualEngine:
    global _engine
    if _engine is None:
        logger.info("🔄 Loading CounterfactualEngine …")
        _engine = CounterfactualEngine()
        logger.success("✅ CounterfactualEngine ready.")
    return _engine


def get_interpret_engine() -> InterpretabilityEngine:
    global _interpret_engine
    if _interpret_engine is None:
        logger.info("🔄 Loading InterpretabilityEngine (SHAP) …")
        _interpret_engine = InterpretabilityEngine(
            model_path=EBM_PATH,
            data_path=SPLITS_PATH,
        )
        logger.success("✅ InterpretabilityEngine ready.")
    return _interpret_engine


def get_fairness_auditor() -> FairnessAuditor:
    global _fairness_auditor
    if _fairness_auditor is None:
        logger.info("🔄 Loading FairnessAuditor …")
        _fairness_auditor = FairnessAuditor(
            model_path=EBM_PATH,
            data_path=SPLITS_PATH
        )
        logger.success("✅ FairnessAuditor ready.")
    return _fairness_auditor


def get_causal_engine() -> CausalEngine:
    global _causal_engine
    if _causal_engine is None:
        logger.info("🔄 Loading CausalEngine …")
        _causal_engine = CausalEngine(data_path=SPLITS_PATH)
        logger.success("✅ CausalEngine ready.")
    return _causal_engine


def get_conformal_engine() -> ConformalEngine:
    global _conformal_engine
    if _conformal_engine is None:
        logger.info("🔄 Loading ConformalEngine …")
        _conformal_engine = ConformalEngine(
            model_path=EBM_PATH,
            data_path=SPLITS_PATH
        )
        logger.success("✅ ConformalEngine ready.")
    return _conformal_engine


def get_manifold_engine() -> ManifoldEngine:
    global _manifold_engine
    if _manifold_engine is None:
        logger.info("🔄 Loading ManifoldEngine (VAE) …")
        _manifold_engine = ManifoldEngine(data_path=SPLITS_PATH)
        logger.success("✅ ManifoldEngine ready.")
    return _manifold_engine


def get_formal_verifier() -> FormalVerifier:
    global _formal_verifier
    if _formal_verifier is None:
        logger.info("🔄 Loading FormalVerifier (Z3) …")
        _formal_verifier = FormalVerifier(
            model_path=EBM_PATH,
            scaler_path=SCALER_PATH
        )
        logger.success("✅ FormalVerifier ready.")
    return _formal_verifier


def get_stability_audit() -> PerformativeAudit:
    global _stability_audit
    if _stability_audit is None:
        logger.info("🔄 Loading PerformativeAudit …")
        _stability_audit = PerformativeAudit(
            model_path=EBM_PATH,
            data_path=SPLITS_PATH
        )
        logger.success("✅ PerformativeAudit ready.")
    return _stability_audit


def get_privacy_audit() -> PrivacyAuditEngine:
    global _privacy_audit
    if _privacy_audit is None:
        logger.info("🔄 Loading PrivacyAuditEngine …")
        _privacy_audit = PrivacyAuditEngine()
        logger.success("✅ PrivacyAuditEngine ready.")
    return _privacy_audit


def get_robustness_engine() -> RobustnessEngine:
    global _robustness_engine
    if _robustness_engine is None:
        logger.info("🔄 Loading RobustnessEngine …")
        _robustness_engine = RobustnessEngine(
            model_path=EBM_PATH,
            scaler_path=SCALER_PATH
        )
        logger.success("✅ RobustnessEngine ready.")
    return _robustness_engine


# ── Middleware for request logging ───────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"→ {request.method} {request.url.path} from {request.client.host}")
    start = time.perf_counter()
    try:
        response = await call_next(request)
        elapsed = (time.perf_counter() - start) * 1000
        logger.info(f"← {request.method} {request.url.path} → {response.status_code} ({elapsed:.0f}ms)")
        return response
    except Exception as e:
        logger.error(f"💥 {request.method} {request.url.path} FAILED: {e}")
        raise


# ── Endpoints ────────────────────────────────────────────

@app.get("/")
async def root():
    logger.debug("Root endpoint hit")
    return {
        "name": "GlassBox API",
        "version": "1.0.0",
        "endpoints": [
            "/features", "/predict", "/explain", "/global-insights", 
            "/fairness", "/causal-graph", "/manifold-projection", 
            "/stability-audit", "/privacy-audit", "/robustness-audit"
        ],
    }


@app.get("/features")
async def features():
    """Return feature metadata for the frontend form/sliders."""
    logger.info("📋 /features requested")
    try:
        meta = get_feature_metadata()
        logger.success(f"📋 /features returning {len(meta)} features")
        return {"features": meta}
    except Exception as e:
        logger.error(f"📋 /features FAILED: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Fast prediction endpoint (target: < 100ms)."""
    logger.info(f"⚡ /predict called with credit_limit={request.credit_limit}, age={request.age}")
    start = time.perf_counter()

    try:
        engine = get_engine()
        input_data = request.model_dump()
        result = engine.predict(input_data)
        
        # Add conformal sets
        conf = get_conformal_engine()
        set95 = conf.get_prediction_set(input_data, alpha=0.05)
        set99 = conf.get_prediction_set(input_data, alpha=0.01)
        
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.success(f"⚡ /predict → {result['label']} (prob={result['probability_default']:.2%}) in {elapsed_ms:.0f}ms")
        
        return PredictionResponse(
            **result,
            conformal_set_95=set95,
            conformal_set_99=set99
        )
    except Exception as e:
        logger.error(f"⚡ /predict FAILED: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain", response_model=ExplanationResponse)
async def explain(request: PredictionRequest):
    """Heavy endpoint: generates counterfactual explanations."""
    logger.info(f"🧠 /explain called with credit_limit={request.credit_limit}, age={request.age}")
    start = time.perf_counter()

    try:
        engine = get_engine()
        input_data = request.model_dump()

        logger.debug("Running prediction…")
        prediction = engine.predict(input_data)
        logger.debug(f"Prediction done: {prediction['label']}")

        logger.debug("Generating counterfactuals…")
        counterfactuals = engine.generate(input_data, num_cfs=4)
        logger.debug(f"Counterfactuals generated: {len(counterfactuals)}")

        logger.debug("Generating local SHAP importance…")
        interpret = get_interpret_engine()
        local_imp = interpret.get_local_explanation(input_data)
        
        # Get conformal sets for explanation as well
        conf = get_conformal_engine()
        set95 = conf.get_prediction_set(input_data, alpha=0.05)
        set99 = conf.get_prediction_set(input_data, alpha=0.01)

        logger.debug("Running Formal Verification (Z3)…")
        verifier = get_formal_verifier()
        certification = verifier.certify_recourse(input_data)

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.success(f"🧠 /explain → {prediction['label']}, {len(counterfactuals)} CFs, Certified={certification['certified']} in {elapsed_ms:.0f}ms")

        return ExplanationResponse(
            prediction=prediction["prediction"],
            probability_default=prediction["probability_default"],
            probability_no_default=prediction["probability_no_default"],
            label=prediction["label"],
            counterfactuals=counterfactuals,
            local_importance=local_imp,
            conformal_set_95=set95,
            conformal_set_99=set99,
            certification=certification
        )
    except Exception as e:
        logger.error(f"🧠 /explain FAILED: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/global-insights", response_model=GlobalInsightsResponse)
async def global_insights():
    """Return global feature importance (SHAP)."""
    logger.info("🌍 /global-insights requested")
    try:
        interpret = get_interpret_engine()
        importance = interpret.get_global_importance()
        return GlobalInsightsResponse(importance=importance)
    except Exception as e:
        logger.error(f"🌍 /global-insights FAILED: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/fairness", response_model=ComprehensiveFairnessResponse)
async def fairness_audit():
    """Return model fairness audit results."""
    logger.info("⚖️ /fairness audit requested")
    try:
        auditor = get_fairness_auditor()
        results = auditor.get_comprehensive_audit()
        return ComprehensiveFairnessResponse(results=results)
    except Exception as e:
        logger.error(f"⚖️ /fairness audit FAILED: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/causal-graph", response_model=CausalGraphResponse)
async def causal_graph():
    """Return the discovered causal structure for D3."""
    logger.info("🔭 /causal-graph requested")
    try:
        causal = get_causal_engine()
        graph = causal.get_graph_json()
        return CausalGraphResponse(**graph)
    except Exception as e:
        logger.error(f"🔭 /causal-graph FAILED: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/manifold-projection", response_model=ManifoldProjectionResponse)
async def manifold_projection(request: PredictionRequest):
    """Return latent space projection of data and the specific user."""
    logger.info("🌀 /manifold-projection requested")
    try:
        manifold = get_manifold_engine()
        points = manifold.get_manifold_projection(n_samples=500)
        user_point = manifold.get_latent_coords(request.model_dump())
        return ManifoldProjectionResponse(points=points, user_point=user_point)
    except Exception as e:
        logger.error(f"🌀 /manifold-projection FAILED: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stability-audit", response_model=StabilityAuditResponse)
async def stability_audit():
    """Run performative risk and strategic gameability audit."""
    logger.info("🔄 /stability-audit requested")
    try:
        audit = get_stability_audit()
        results = audit.simulate_strategic_response(adoption_rate=0.3)
        gameability = audit.get_gameability_report()
        return StabilityAuditResponse(**results, gameability_report=gameability)
    except Exception as e:
        logger.error(f"🔄 /stability-audit FAILED: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/privacy-audit", response_model=PrivacyAuditResponse)
async def privacy_audit():
    """Verify differential privacy (DP-SGD) budget."""
    logger.info("🔒 /privacy-audit requested")
    try:
        audit = get_privacy_audit()
        return PrivacyAuditResponse(**audit.audit_privacy_budget())
    except Exception as e:
        logger.error(f"🔒 /privacy-audit FAILED: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/robustness-audit", response_model=RobustnessResponse)
async def robustness_audit(request: PredictionRequest):
    """Run adversarial robustness audit for a specific profile."""
    logger.info("🛡️ /robustness-audit requested")
    try:
        robust = get_robustness_engine()
        results = robust.audit_robustness(request.model_dump())
        return RobustnessResponse(**results)
    except Exception as e:
        logger.error(f"🛡️ /robustness-audit FAILED: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=3100, reload=True)
