export default function UncertaintyAudit({ prediction }) {
    if (!prediction) return null;

    const { conformal_set_95, conformal_set_99 } = prediction;

    const getSetLabel = (set) => {
        if (set.length === 0) return "Empty (High Uncertainty)";
        if (set.length === 2) return "{Approved, Default} (Inconclusive)";
        return set[0] === 0 ? "{Approved}" : "{Default}";
    };

    const getConfidenceLevel = (set) => {
        if (set.length === 1) return "Certified Confidence";
        if (set.length === 2) return "Low Confidence (Ambiguous)";
        return "Out of Distribution";
    };

    return (
        <div className="card">
            <h2>🛡️ Conformal Uncertainty Audit</h2>
            <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem", marginBottom: "1.5rem" }}>
                Unlike standard probabilities, <strong>Conformal Prediction</strong> provides mathematically
                guaranteed prediction sets that account for model uncertainty.
            </p>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem" }}>
                <div className="cf-item" style={{ border: conformal_set_95.length === 1 ? "1px solid var(--emerald)" : "1px solid var(--border)" }}>
                    <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginBottom: "0.25rem", textTransform: "uppercase" }}>
                        95% Confidence Set
                    </div>
                    <div style={{ fontSize: "1.2rem", fontWeight: 700, color: "var(--accent-light)" }}>
                        {getSetLabel(conformal_set_95)}
                    </div>
                    <div style={{ fontSize: "0.75rem", color: conformal_set_95.length === 1 ? "var(--emerald)" : "var(--rose)", marginTop: "0.25rem" }}>
                        {getConfidenceLevel(conformal_set_95)}
                    </div>
                </div>

                <div className="cf-item" style={{ border: conformal_set_99.length === 1 ? "1px solid var(--emerald)" : "1px solid var(--border)" }}>
                    <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginBottom: "0.25rem", textTransform: "uppercase" }}>
                        99% Confidence Set
                    </div>
                    <div style={{ fontSize: "1.2rem", fontWeight: 700, color: "var(--cyan)" }}>
                        {getSetLabel(conformal_set_99)}
                    </div>
                    <div style={{ fontSize: "0.75rem", color: conformal_set_99.length === 1 ? "var(--emerald)" : "var(--rose)", marginTop: "0.25rem" }}>
                        {getConfidenceLevel(conformal_set_99)}
                    </div>
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
                💡 <strong>Mathematical Guarantee:</strong> A "set" of size 1 (e.g. {"{Approved}"}) means the model is statistically certain of the outcome. A set of size 2 means the model finds both outcomes plausible at that confidence level.
            </div>
        </div>
    );
}
