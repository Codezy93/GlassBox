export default function LocalImportance({ importance }) {
    if (!importance) return null;

    // Convert to sorted array
    const entries = Object.entries(importance)
        .map(([name, val]) => ({ name, val }))
        .sort((a, b) => Math.abs(b.val) - Math.abs(a.val))
        .slice(0, 10); // Show top 10

    const maxVal = Math.max(...entries.map((e) => Math.abs(e.val)));

    return (
        <div className="card">
            <h2>🔍 Key Drivers (Local SHAP)</h2>
            <p style={{ color: "var(--text-secondary)", fontSize: "0.82rem", marginBottom: "1rem" }}>
                Feature contributions to this specific prediction.
                <span style={{ color: "var(--emerald)" }}> Green adds to approval</span>,
                <span style={{ color: "var(--rose)" }}> red shifts towards default</span>.
            </p>

            <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
                {entries.map((e) => {
                    const pct = (Math.abs(e.val) / maxVal) * 100;
                    const isPositive = e.val < 0; // In our model, higher SHAP on class 1 (default) is bad. 
                    // Wait, KernelExplainer might return values for class 1 by default. 
                    // If val > 0, it pushes towards Default (bad). If val < 0, it pushes towards Approved (good).

                    return (
                        <div key={e.name} style={{ display: "flex", flexDirection: "column", gap: "0.25rem" }}>
                            <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.75rem" }}>
                                <span>{e.name}</span>
                                <span style={{ fontWeight: 600, color: e.val > 0 ? "var(--rose)" : "var(--emerald)" }}>
                                    {e.val > 0 ? "+" : ""}{e.val.toFixed(4)}
                                </span>
                            </div>
                            <div style={{
                                height: "8px",
                                background: "rgba(255,255,255,0.05)",
                                borderRadius: "4px",
                                overflow: "hidden",
                                position: "relative"
                            }}>
                                <div style={{
                                    height: "100%",
                                    width: `${pct}%`,
                                    background: e.val > 0 ? "var(--rose)" : "var(--emerald)",
                                    borderRadius: "4px",
                                    transition: "width 1s ease-out"
                                }} />
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
}
