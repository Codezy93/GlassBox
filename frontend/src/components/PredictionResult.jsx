export default function PredictionResult({ data }) {
    if (!data) return null;

    const { probability_no_default, probability_default, prediction } = data;
    const approved = prediction === 0;
    const pct = probability_no_default;

    // Semi-circle gauge math
    const R = 80;
    const C = Math.PI * R;
    const offset = C - pct * C;
    const color = pct > 0.65 ? "var(--emerald)" : pct > 0.4 ? "var(--amber)" : "var(--rose)";

    return (
        <div className="card prediction">
            <h2>📊 Prediction Result</h2>

            <div className="gauge-wrap">
                <svg viewBox="0 0 200 110">
                    <path className="arc-bg" d="M 20 100 A 80 80 0 0 1 180 100" />
                    <path
                        className="arc-fill"
                        d="M 20 100 A 80 80 0 0 1 180 100"
                        style={{
                            stroke: color,
                            strokeDasharray: C,
                            strokeDashoffset: offset,
                        }}
                    />
                </svg>
                <div className="gauge-pct" style={{ color }}>
                    {(pct * 100).toFixed(1)}%
                </div>
            </div>

            <div className={`badge ${approved ? "ok" : "bad"}`}>
                {approved ? "✅ APPROVED — No Default" : "⛔ REJECTED — Default Risk"}
            </div>

            <div className="prob-row">
                <span>Approval: <strong style={{ color: "var(--emerald)" }}>{(probability_no_default * 100).toFixed(1)}%</strong></span>
                <span>Default: <strong style={{ color: "var(--rose)" }}>{(probability_default * 100).toFixed(1)}%</strong></span>
            </div>
        </div>
    );
}
