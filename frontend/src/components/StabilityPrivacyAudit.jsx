import { useState, useEffect } from "react";
import { getStabilityAudit, getPrivacyAudit } from "../api";

export default function StabilityPrivacyAudit() {
    const [stability, setStability] = useState(null);
    const [privacy, setPrivacy] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        Promise.all([getStabilityAudit(), getPrivacyAudit()])
            .then(([s, p]) => {
                setStability(s);
                setPrivacy(p);
                setLoading(false);
            })
            .catch((err) => {
                console.error(err);
                setError("Audit failed. Ensure backend is running Phase 12.");
                setLoading(false);
            });
    }, []);

    if (loading) return <div className="card empty"><div className="spin"></div><p>Auditing strategic stability and privacy budget…</p></div>;
    if (error) return <div className="error-msg">⚠ {error}</div>;

    return (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.5rem" }}>
            <div className="card">
                <h2>🌀 Performative Stability Audit</h2>
                <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem", marginBottom: "1.5rem" }}>
                    Models distribution shift and **Strategic Gaming**. If users follow the suggested recourse, how much does the model's performance degrade?
                </p>

                <div className="cf-item">
                    <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginBottom: "0.25rem", textTransform: "uppercase" }}>Strategic Gaming Ratio</div>
                    <div style={{ fontSize: "1.8rem", fontWeight: 700, color: "var(--rose)" }}>{(stability.strategic_gaming_ratio * 100).toFixed(1)}%</div>
                    <div style={{ fontSize: "0.7rem", color: "var(--text-secondary)", marginTop: "0.25rem" }}>
                        Percentage of default-risk users who can "game" the model into approval.
                    </div>
                </div>

                <div style={{ marginTop: "1.5rem" }}>
                    <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginBottom: "0.5rem", textTransform: "uppercase" }}>Feature Gameability Index</div>
                    <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                        {Object.entries(stability.gameability_report).map(([f, score]) => (
                            <div key={f} style={{ display: "flex", justifyContent: "space-between", fontSize: "0.8rem", background: "rgba(255,255,255,0.03)", padding: "0.4rem 0.8rem", borderRadius: "4px" }}>
                                <span>{f}</span>
                                <span style={{ fontWeight: 600, color: score > 0.1 ? "var(--amber)" : "var(--emerald)" }}>{(score * 100).toFixed(1)}% Sensitivity</span>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            <div className="card">
                <h2>🔒 Differential Privacy (DP-SGD)</h2>
                <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem", marginBottom: "1.5rem" }}>
                    Certifies that individual training data cannot be reconstructed from explanations.
                </p>

                <div className="cf-item" style={{ border: "1px solid var(--accent)" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                        <span style={{ fontSize: "0.75rem", letterSpacing: "0.05em", color: "var(--text-muted)" }}>CERTIFIED BUDGET</span>
                        <span className="badge ok">ACTIVE</span>
                    </div>
                    <div style={{ margin: "1rem 0", textAlign: "center" }}>
                        <div style={{ fontSize: "0.7rem", color: "var(--text-secondary)" }}>Privacy Loss (ε)</div>
                        <div style={{ fontSize: "2.5rem", fontWeight: 800, color: "var(--accent-light)" }}>{privacy.epsilon.toFixed(2)}</div>
                        <div style={{ fontSize: "0.7rem", color: "var(--text-secondary)" }}>Failure Prob (δ): {privacy.delta}</div>
                    </div>
                </div>

                <div style={{
                    marginTop: "1.5rem",
                    padding: "1rem",
                    background: "rgba(99, 102, 241, 0.05)",
                    borderRadius: "8px",
                    fontSize: "0.8rem",
                    color: "var(--text-secondary)",
                    border: "1px solid var(--border)"
                }}>
                    💡 <strong>Privacy Guarantee:</strong> This model was retrained using **Opacus** with gradient clipping and noise injection. An epsilon of 1.0 represents "Strong Privacy."
                </div>
            </div>
        </div>
    );
}
