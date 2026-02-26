export default function VerificationCertificate({ certification }) {
    if (!certification) return null;

    const { certified, message, proof_id } = certification;

    return (
        <div className="card" style={{ border: certified ? "1px solid var(--emerald)" : "1px solid var(--rose)", background: certified ? "rgba(16, 185, 129, 0.05)" : "rgba(244, 63, 94, 0.05)" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                <div>
                    <h2 style={{ color: certified ? "var(--emerald)" : "var(--rose)", margin: 0 }}>
                        {certified ? "📜 Certified Recourse Path" : "⚠️ Verification Failure"}
                    </h2>
                    <p style={{ fontSize: "0.85rem", opacity: 0.8, marginTop: "0.5rem" }}>
                        Formal SMT Reachability Audit (Z3-Solver)
                    </p>
                </div>
                {certified && (
                    <div style={{ fontSize: "2rem" }}>✅</div>
                )}
            </div>

            <div style={{ margin: "1.5rem 0", padding: "1rem", background: "rgba(0,0,0,0.2)", borderRadius: "6px", fontFamily: "monospace", fontSize: "0.8rem" }}>
                <div style={{ color: "var(--text-secondary)", marginBottom: "0.5rem" }}>REPORT:</div>
                <div style={{ color: "#fff" }}>{message}</div>
                {proof_id && (
                    <div style={{ marginTop: "1rem", color: "var(--text-muted)", fontSize: "0.7rem" }}>
                        SAT_PROOF_ID: {proof_id}
                    </div>
                )}
            </div>

            <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)" }}>
                💡 <strong>Method note:</strong> This certificate comes from an SMT satisfiability check over local action bounds using a model-fitted local surrogate.
            </div>
        </div>
    );
}
