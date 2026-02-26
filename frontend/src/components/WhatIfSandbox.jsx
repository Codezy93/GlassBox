import { useState, useEffect, useCallback } from "react";
import { predict } from "../api";

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

function debounce(fn, ms) {
    let t;
    return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}

export default function WhatIfSandbox({ inputData, baseProbability, features }) {
    const [sliders, setSliders] = useState({});
    const [prob, setProb] = useState(null);
    const [busy, setBusy] = useState(false);

    const mutable = features ? features.filter((f) => !f.immutable) : [];

    useEffect(() => {
        if (inputData) {
            const init = {};
            mutable.forEach((f) => { init[f.name] = inputData[f.name]; });
            setSliders(init);
            setProb(baseProbability);
        }
    }, [inputData]);

    // eslint-disable-next-line react-hooks/exhaustive-deps
    const debouncedPredict = useCallback(
        debounce(async (data) => {
            try {
                setBusy(true);
                const res = await predict(data);
                setProb(res.probability_no_default);
            } catch { /* silent */ }
            finally { setBusy(false); }
        }, 350),
        []
    );

    const onChange = (name, value) => {
        const next = { ...sliders, [name]: parseFloat(value) };
        setSliders(next);
        debouncedPredict({ ...inputData, ...next });
    };

    if (!inputData) return null;

    const delta = prob !== null ? prob - baseProbability : 0;
    const deltaClass = delta > 0.005 ? "delta-up" : delta < -0.005 ? "delta-down" : "delta-flat";
    const probColor = prob > 0.65 ? "var(--emerald)" : prob > 0.4 ? "var(--amber)" : "var(--rose)";

    return (
        <div className="card">
            <div className="sandbox-header">
                <h2>🧪 What-If Sandbox</h2>
                <div className="sandbox-prob">
                    <span className="val" style={{ color: probColor }}>
                        {prob !== null ? `${(prob * 100).toFixed(1)}%` : "—"}
                    </span>
                    {Math.abs(delta) > 0.005 && (
                        <span className={`delta ${deltaClass}`}>
                            {delta > 0 ? "+" : ""}{(delta * 100).toFixed(1)}%
                        </span>
                    )}
                    {busy && <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>⏳</span>}
                </div>
            </div>

            <p style={{ color: "var(--text-secondary)", fontSize: "0.82rem", marginBottom: "1rem" }}>
                Drag the sliders to see how changes affect the approval probability in real time.
            </p>

            <div className="slider-grid">
                {mutable.map((f) => {
                    const val = sliders[f.name] ?? f.mean;
                    const lo = f.permitted_range ? f.permitted_range[0] : f.min;
                    const hi = f.permitted_range ? f.permitted_range[1] : f.max;
                    const step = f.type === "continuous" ? Math.max(1, (hi - lo) / 200) : 1;
                    const display = f.labels && f.labels[String(Math.round(val))]
                        ? f.labels[String(Math.round(val))]
                        : f.type === "continuous"
                            ? `$${Number(val).toLocaleString("en-US", { maximumFractionDigits: 0 })}`
                            : Math.round(val);

                    return (
                        <div key={f.name} className="slider-group">
                            <label>
                                <span>{NAMES[f.name] || f.name}</span>
                                <span className="sv">{display}</span>
                            </label>
                            <input
                                type="range"
                                min={lo}
                                max={hi}
                                step={step}
                                value={val}
                                onChange={(e) => onChange(f.name, e.target.value)}
                            />
                        </div>
                    );
                })}
            </div>
        </div>
    );
}
