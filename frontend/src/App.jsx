import { useState } from "react";
import ProfileForm from "./components/ProfileForm";
import PredictionResult from "./components/PredictionResult";
import CounterfactualCards from "./components/CounterfactualCards";
import WhatIfSandbox from "./components/WhatIfSandbox";
import LocalImportance from "./components/LocalImportance";
import GlobalInsights from "./components/GlobalInsights";
import FairnessAudit from "./components/FairnessAudit";
import CausalGraphView from "./components/CausalGraphView";
import UncertaintyAudit from "./components/UncertaintyAudit";
import ManifoldProjector from "./components/ManifoldProjector";
import VerificationCertificate from "./components/VerificationCertificate";
import StabilityPrivacyAudit from "./components/StabilityPrivacyAudit";
import RobustnessAudit from "./components/RobustnessAudit";
import { explain } from "./api";
import "./index.css";

export default function App() {
  const [prediction, setPrediction] = useState(null);
  const [counterfactuals, setCounterfactuals] = useState([]);
  const [localImportance, setLocalImportance] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [inputData, setInputData] = useState(null);
  const [features, setFeatures] = useState([]);
  const [activeTab, setActiveTab] = useState("individual");

  const handleAnalyze = async (formData) => {
    console.log("[App] handleAnalyze called:", formData);
    setLoading(true);
    setError(null);
    setPrediction(null);
    setCounterfactuals([]);
    setLocalImportance(null);
    setInputData(formData);

    try {
      const res = await explain(formData);
      console.log("[App] Got response:", res);
      setPrediction({
        prediction: res.prediction,
        probability_default: res.probability_default,
        probability_no_default: res.probability_no_default,
        label: res.label,
        conformal_set_95: res.conformal_set_95,
        conformal_set_99: res.conformal_set_99,
        certification: res.certification,
      });
      setCounterfactuals(res.counterfactuals || []);
      setLocalImportance(res.local_importance);
      if (activeTab !== "individual") setActiveTab("individual");
    } catch (err) {
      console.error("[App] Error:", err);
      setError(err.message || "Analysis failed. Is the API server running?");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="header">
        <div className="header-icon">🔬</div>
        <h1>GlassBox <span style={{ fontSize: "1rem", opacity: 0.6, verticalAlign: "middle" }}>Neural-Causal</span></h1>
        <p>
          Decision intelligence with actionable recourse, calibrated uncertainty,
          causal structure, and operational risk audits.
        </p>
      </header>

      {error && <div className="error-msg">⚠ {error}</div>}

      <div className="tabs">
        <button
          className={`tab-btn ${activeTab === "individual" ? "active" : ""}`}
          onClick={() => setActiveTab("individual")}
        >
          👤 Analysis
        </button>
        <button
          className={`tab-btn ${activeTab === "global" ? "active" : ""}`}
          onClick={() => setActiveTab("global")}
        >
          🌍 Drivers
        </button>
        <button
          className={`tab-btn ${activeTab === "fairness" ? "active" : ""}`}
          onClick={() => setActiveTab("fairness")}
        >
          ⚖️ Fairness
        </button>
        <button
          className={`tab-btn ${activeTab === "causal" ? "active" : ""}`}
          onClick={() => setActiveTab("causal")}
        >
          🧬 Causality
        </button>
        <button
          className={`tab-btn ${activeTab === "manifold" ? "active" : ""}`}
          onClick={() => setActiveTab("manifold")}
        >
          🌀 Latent Space
        </button>
        <button
          className={`tab-btn ${activeTab === "stability" ? "active" : ""}`}
          onClick={() => setActiveTab("stability")}
        >
          📈 Strategic Audit
        </button>
      </div>

      <div className="layout">
        <div className="sidebar">
          <ProfileForm
            onAnalyze={handleAnalyze}
            loading={loading}
            onFeaturesLoaded={setFeatures}
          />
        </div>

        <div className="results">
          {activeTab === "individual" && (
            <>
              {!prediction && !loading && (
                <div className="card empty">
                  <div className="big-icon">🎯</div>
                  <p>
                    Enter parameters for high-fidelity research auditing.
                  </p>
                </div>
              )}

              {loading && (
                <div className="card empty">
                  <div className="big-icon"><span className="spin" style={{ width: 32, height: 32, borderWidth: 3, display: 'inline-block' }}></span></div>
                  <p>Running model analysis and generating recourse options…<br />Please wait a moment.</p>
                </div>
              )}

              {prediction && (
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.5rem", marginBottom: "1.5rem" }}>
                  <PredictionResult data={prediction} />
                  <VerificationCertificate certification={prediction.certification} />
                </div>
              )}

              {prediction && (
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(350px, 1fr))", gap: "1.5rem", marginBottom: "1.5rem" }}>
                  <UncertaintyAudit prediction={prediction} />
                  <RobustnessAudit inputData={inputData} />
                </div>
              )}

              {prediction && (
                <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: "1.5rem", marginBottom: "1.5rem" }}>
                  {localImportance && <LocalImportance importance={localImportance} />}
                </div>
              )}

              {prediction && (
                <div style={{ marginBottom: "1.5rem" }}>
                  <CounterfactualCards items={counterfactuals} />
                </div>
              )}

              {prediction && (
                <WhatIfSandbox
                  inputData={inputData}
                  baseProbability={prediction.probability_no_default}
                  features={features}
                />
              )}
            </>
          )}

          {activeTab === "global" && <GlobalInsights />}
          {activeTab === "fairness" && <FairnessAudit />}
          {activeTab === "causal" && <CausalGraphView />}
          {activeTab === "manifold" && <ManifoldProjector inputData={inputData} />}
          {activeTab === "stability" && <StabilityPrivacyAudit />}
        </div>
      </div>
    </div>
  );
}
