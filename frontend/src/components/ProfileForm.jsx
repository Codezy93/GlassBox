import { useState, useEffect } from "react";
import { getFeatures } from "../api";

const GROUPS = [
    { label: "Personal Info", keys: ["credit_limit", "sex", "education", "marriage", "age"] },
    { label: "Repayment History", keys: ["repayment_sep", "repayment_aug", "repayment_jul", "repayment_jun", "repayment_may", "repayment_apr"] },
    { label: "Bill Amounts", keys: ["bill_sep", "bill_aug", "bill_jul", "bill_jun", "bill_may", "bill_apr"] },
    { label: "Payment Amounts", keys: ["pay_sep", "pay_aug", "pay_jul", "pay_jun", "pay_may", "pay_apr"] },
];

const LABELS = {
    credit_limit: "Credit Limit ($)",
    sex: "Sex", education: "Education", marriage: "Marriage", age: "Age",
    repayment_sep: "September", repayment_aug: "August", repayment_jul: "July",
    repayment_jun: "June", repayment_may: "May", repayment_apr: "April",
    bill_sep: "Sep", bill_aug: "Aug", bill_jul: "Jul",
    bill_jun: "Jun", bill_may: "May", bill_apr: "Apr",
    pay_sep: "Sep", pay_aug: "Aug", pay_jul: "Jul",
    pay_jun: "Jun", pay_may: "May", pay_apr: "Apr",
};

const DEFAULTS = {
    credit_limit: 50000, sex: 2, education: 2, marriage: 1, age: 35,
    repayment_sep: 0, repayment_aug: 0, repayment_jul: -1, repayment_jun: -1, repayment_may: -1, repayment_apr: -1,
    bill_sep: 45000, bill_aug: 42000, bill_jul: 38000, bill_jun: 35000, bill_may: 30000, bill_apr: 28000,
    pay_sep: 2000, pay_aug: 2000, pay_jul: 1500, pay_jun: 1500, pay_may: 1000, pay_apr: 1000,
};

export default function ProfileForm({ onAnalyze, loading, onFeaturesLoaded }) {
    const [meta, setMeta] = useState([]);
    const [form, setForm] = useState(DEFAULTS);
    const [err, setErr] = useState(null);

    useEffect(() => {
        console.log("[ProfileForm] Fetching feature metadata…");
        getFeatures()
            .then((res) => {
                console.log("[ProfileForm] Features loaded:", res.features?.length);
                setMeta(res.features || []);
                if (onFeaturesLoaded) onFeaturesLoaded(res.features || []);
            })
            .catch((e) => {
                console.error("[ProfileForm] Fetch error:", e);
                setErr("Could not load features. Is the API running?");
            });
    }, []);

    const set = (k, v) => setForm((p) => ({ ...p, [k]: parseFloat(v) }));

    const handleClick = () => {
        console.log("[ProfileForm] ✅ Button clicked! Sending data…");
        if (onAnalyze) onAnalyze(form);
    };

    const getMeta = (n) => meta.find((f) => f.name === n) || {};

    const renderField = (name) => {
        const m = getMeta(name);
        const label = LABELS[name] || name;

        if (m.labels && Object.keys(m.labels).length > 0) {
            return (
                <div className="field" key={name}>
                    <label htmlFor={name}>
                        {label}
                        {m.immutable && <span className="lock">LOCKED</span>}
                    </label>
                    <select id={name} name={name} value={form[name] ?? ""} onChange={(e) => set(name, e.target.value)}>
                        {Object.entries(m.labels).map(([v, l]) => (
                            <option key={v} value={v}>{l}</option>
                        ))}
                    </select>
                </div>
            );
        }

        return (
            <div className="field" key={name}>
                <label htmlFor={name}>
                    {label}
                    {m.immutable && <span className="lock">LOCKED</span>}
                </label>
                <input
                    id={name}
                    name={name}
                    type="number"
                    value={form[name] ?? ""}
                    onChange={(e) => set(name, e.target.value)}
                    min={m.min}
                    max={m.max}
                    step={m.type === "continuous" ? 1000 : 1}
                />
            </div>
        );
    };

    return (
        <div className="card form-card">
            <h2>👤 Applicant Profile</h2>

            {err && <div className="error-msg">{err}</div>}

            {GROUPS.map((g) => (
                <div key={g.label}>
                    <div className="section-label">{g.label}</div>
                    <div className="fields">{g.keys.map(renderField)}</div>
                </div>
            ))}

            <button type="button" className="btn" disabled={loading} onClick={handleClick}>
                {loading && <span className="spin"></span>}
                {loading ? "Analyzing…" : "🔍 Analyze & Explain"}
            </button>
        </div>
    );
}
