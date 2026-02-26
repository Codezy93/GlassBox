import { useState, useEffect } from "react";
import { auditRobustness } from "../api";

export default function RobustnessAudit({ inputData }) {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        if (inputData) {
            setLoading(true);
            auditRobustness(inputData)
                .then(setData)
                .catch(err => {
                    console.error(err);
                    setError("Robustness audit failed. Ensure profile is valid.");
                })
                .finally(() => setLoading(false));
        }
    }, [inputData]);

    if (loading) return <div className="card empty"><div className="spin"></div><p>Auditing adversarial robustness radius…</p></div>;
    if (error) return <div className="error-msg">⚠ {error}</div>;
    if (!data) return null;

    const getScoreBadgeStyle = (score) => {
        if (score > 0.9) {
            return {
                background: "var(--emerald)",
                color: "#042016",
                border: "1px solid rgba(74, 222, 128, 0.45)",
            };
        }
        if (score > 0.7) {
            return {
                background: "var(--amber)",
                color: "#2d1b00",
                border: "1px solid rgba(245, 158, 11, 0.5)",
            };
        }
        return {
            background: "var(--rose)",
            color: "#fff",
            border: "1px solid rgba(251, 113, 133, 0.5)",
        };
    };

    return (
        <div className="card">
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
                <h2>🛡️ Adversarial Robustness</h2>
                <div className="badge" style={{ ...getScoreBadgeStyle(data.robustness_score), flexShrink: 0, whiteSpace: "nowrap" }}>
                    {(data.robustness_score * 100).toFixed(1)}% Stable
                </div>
            </div>

            <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem", marginBottom: "1.5rem" }}>
                Quantifies model stability under local noise (ε={data.epsilon_radius}). A high score indicates the prediction is <strong>formally robust</strong> to minor feature perturbations.
            </p>

            <div style={{ marginBottom: "1.5rem" }}>
                <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginBottom: "0.5rem", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                    Feature Sensitivity Rank (Local Gradient)
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                    {Object.entries(data.feature_sensitivity_rank)
                        .sort((a, b) => b[1] - a[1])
                        .slice(0, 5)
                        .map(([f, sens]) => (
                            <div key={f}>
                                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.75rem", marginBottom: "0.2rem" }}>
                                    <span>{f}</span>
                                    <span style={{ fontWeight: 600 }}>{(sens * 100).toFixed(1)}%</span>
                                </div>
                                <div style={{ height: "4px", background: "rgba(255,255,255,0.05)", borderRadius: "2px", overflow: "hidden" }}>
                                    <div style={{
                                        width: `${sens * 100}%`,
                                        height: "100%",
                                        background: sens > 0.5 ? "var(--amber)" : "var(--accent)",
                                        boxShadow: "0 0 8px var(--accent-light)"
                                    }} />
                                </div>
                            </div>
                        ))}
                </div>
            </div>

            <div style={{
                padding: "0.75rem",
                background: "rgba(16, 185, 129, 0.05)",
                borderRadius: "6px",
                fontSize: "0.75rem",
                color: "var(--text-secondary)",
                border: "1px solid rgba(16, 185, 129, 0.1)"
            }}>
                💡 <strong>Method note:</strong> This audit uses local Monte Carlo perturbations to estimate stability and rank the most sensitivity-driving features.
            </div>
        </div>
    );
}
