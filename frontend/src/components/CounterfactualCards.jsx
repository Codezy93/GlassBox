const NAMES = {
    credit_limit: "Credit Limit", marriage: "Marriage",
    repayment_sep: "Repay (Sep)", repayment_aug: "Repay (Aug)",
    repayment_jul: "Repay (Jul)", repayment_jun: "Repay (Jun)",
    repayment_may: "Repay (May)", repayment_apr: "Repay (Apr)",
    bill_sep: "Bill (Sep)", bill_aug: "Bill (Aug)",
    bill_jul: "Bill (Jul)", bill_jun: "Bill (Jun)",
    bill_may: "Bill (May)", bill_apr: "Bill (Apr)",
    pay_sep: "Payment (Sep)", pay_aug: "Payment (Aug)",
    pay_jul: "Payment (Jul)", pay_jun: "Payment (Jun)",
    pay_may: "Payment (May)", pay_apr: "Payment (Apr)",
};

function fmt(name, v) {
    const n = Number(v);
    if (name === "marriage") return { 1: "Married", 2: "Single", 3: "Others" }[Math.round(n)] || n;
    if (name.startsWith("repayment_")) {
        const m = { "-1": "On time", "0": "Revolving", "1": "1mo late", "2": "2mo late", "3": "3mo late" };
        return m[String(Math.round(n))] || `${Math.round(n)}mo`;
    }
    if (name.startsWith("bill_") || name.startsWith("pay_") || name === "credit_limit")
        return `$${Number(n).toLocaleString("en-US", { maximumFractionDigits: 0 })}`;
    return Math.round(n);
}

export default function CounterfactualCards({ items }) {
    if (!items || items.length === 0) return null;

    return (
        <div className="card">
            <h2>🛤️ Paths to Approval</h2>
            <p style={{ color: "var(--text-secondary)", fontSize: "0.82rem", marginBottom: "1rem" }}>
                <strong>{items.length}</strong> diverse ways to change the outcome. Each path shows the minimal changes needed.
            </p>

            <div className="cf-grid">
                {items.map((cf, i) => {
                    const changes = Object.entries(cf.changes || {});
                    return (
                        <div key={i} className="cf-item">
                            <div className="cf-head">
                                <span className="tag">Path {i + 1}</span>
                                <span className="pct">{(cf.probability_no_default * 100).toFixed(1)}%</span>
                            </div>
                            {changes.length === 0 ? (
                                <div style={{ color: "var(--text-muted)", fontSize: "0.82rem", fontStyle: "italic" }}>No changes needed</div>
                            ) : (
                                changes.map(([feat, vals]) => (
                                    <div key={feat} className="cf-row">
                                        <span className="fname">{NAMES[feat] || feat}</span>
                                        <span className="old">{fmt(feat, vals.original)}</span>
                                        <span className="arr">→</span>
                                        <span className="new">{fmt(feat, vals.suggested)}</span>
                                    </div>
                                ))
                            )}
                        </div>
                    );
                })}
            </div>
        </div>
    );
}
