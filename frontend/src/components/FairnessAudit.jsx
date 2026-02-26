import { useState, useEffect } from "react";
import { getFairnessAudit } from "../api";

export default function FairnessAudit() {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        getFairnessAudit()
            .then((res) => {
                setData(res.results);
                setLoading(false);
            })
            .catch((err) => {
                console.error(err);
                setError("Failed to load fairness audit.");
                setLoading(false);
            });
    }, []);

    if (loading) return <div className="card empty"><div className="spin"></div><p>Auditing model for demographic bias…</p></div>;
    if (error) return <div className="error-msg">{error}</div>;

    const results = Object.values(data || {});

    return (
        <div className="card">
            <h2>⚖️ Bias & Fairness Audit</h2>
            <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem", marginBottom: "1.5rem" }}>
                Transparency requires checking for algorithmic bias. This audit measures <strong>Demographic Parity</strong>
                across protected attributes in the test population.
            </p>

            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: "1rem" }}>
                {results.map((r) => (
                    <div key={r.feature} className="cf-item" style={{ border: r.status === "FAIL" ? "1px solid var(--rose)" : r.status === "WARNING" ? "1px solid var(--amber)" : "1px solid var(--border)" }}>
                        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.5rem" }}>
                            <span style={{ fontWeight: 700, textTransform: "uppercase", fontSize: "0.75rem", letterSpacing: "0.05em" }}>
                                Attribute: {r.feature}
                            </span>
                            <span className={`badge ${r.status === "FAIL" ? "bad" : "ok"}`} style={{ fontSize: "0.65rem", padding: "0.2rem 0.5rem" }}>
                                {r.status}
                            </span>
                        </div>

                        <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem", fontSize: "0.82rem" }}>
                            <div style={{ display: "flex", justifyContent: "space-between" }}>
                                <span style={{ color: "var(--text-secondary)" }}>Parity Difference:</span>
                                <span style={{ fontWeight: 600, color: r.demographic_parity_difference > 0.1 ? "var(--amber)" : "var(--emerald)" }}>
                                    {r.demographic_parity_difference.toFixed(4)}
                                </span>
                            </div>
                            <div style={{ display: "flex", justifyContent: "space-between" }}>
                                <span style={{ color: "var(--text-secondary)" }}>Parity Ratio:</span>
                                <span style={{ fontWeight: 600 }}>{r.demographic_parity_ratio.toFixed(4)}</span>
                            </div>

                            <div style={{ marginTop: "0.5rem" }}>
                                <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginBottom: "0.25rem", textTransform: "uppercase" }}>Selection Rates by Group</div>
                                <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem" }}>
                                    {Object.entries(r.group_selection_rates).map(([group, rate]) => (
                                        <div key={group} style={{ background: "rgba(255,255,255,0.04)", padding: "0.25rem 0.5rem", borderRadius: "4px", flex: "1 1 auto", textAlign: "center" }}>
                                            <div style={{ fontSize: "0.65rem", color: "var(--text-secondary)" }}>Group {group}</div>
                                            <div style={{ fontWeight: 700, color: "var(--accent-light)" }}>{(rate * 100).toFixed(1)}%</div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    </div>
                ))}
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
                💡 <strong>Note:</strong> Demographic Parity Difference measures the gap in approval rates between groups. Ideally, it should be close to 0. A high difference (e.g. &gt; 0.2) may indicate systemic bias.
            </div>
        </div>
    );
}
