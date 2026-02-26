import { useState, useEffect } from "react";
import { getGlobalInsights } from "../api";

export default function GlobalInsights() {
    const [importance, setImportance] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        getGlobalInsights()
            .then((res) => {
                setImportance(res.importance);
                setLoading(false);
            })
            .catch((err) => {
                console.error(err);
                setError("Failed to load global insights.");
                setLoading(false);
            });
    }, []);

    if (loading) return <div className="card empty"><div className="spin"></div><p>Calculating global SHAP values…</p></div>;
    if (error) return <div className="error-msg">{error}</div>;

    const entries = Object.entries(importance || {});
    if (entries.length === 0) return <div className="card empty"><p>No global insights available yet.</p></div>;
    const maxVal = Math.max(...entries.map((e) => e[1]), 0.0001);

    return (
        <div className="card">
            <h2>🌍 Global Model Drivers</h2>
            <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem", marginBottom: "1.5rem" }}>
                This chart shows the top factors that influence the credit risk model across the <strong>entire population</strong>.
                Higher values indicate a stronger overall impact on the model's decisions.
            </p>

            <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
                {entries.map(([name, val]) => {
                    const pct = (val / maxVal) * 100;
                    return (
                        <div key={name} style={{ display: "flex", flexDirection: "column", gap: "0.35rem" }}>
                            <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.8rem", fontWeight: 500 }}>
                                <span>{name}</span>
                                <span style={{ color: "var(--accent-light)" }}>{val.toFixed(4)}</span>
                            </div>
                            <div style={{
                                height: "10px",
                                background: "rgba(255,255,255,0.05)",
                                borderRadius: "5px",
                                overflow: "hidden"
                            }}>
                                <div style={{
                                    height: "100%",
                                    width: `${pct}%`,
                                    background: "linear-gradient(90deg, var(--accent), var(--cyan))",
                                    borderRadius: "5px",
                                    boxShadow: "0 0 10px rgba(99, 102, 241, 0.3)"
                                }} />
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
}
