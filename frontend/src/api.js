const API_BASE =
  import.meta.env.VITE_API_BASE ||
  `${window.location.protocol}//${window.location.hostname}:3100`;

export async function getFeatures() {
  console.log("[API] GET /features …");
  const r = await fetch(`${API_BASE}/features`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function predict(data) {
  console.log("[API] POST /predict …");
  const r = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function explain(data) {
  console.log("[API] POST /explain …");
  const r = await fetch(`${API_BASE}/explain`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function getGlobalInsights() {
  console.log("[API] GET /global-insights …");
  const r = await fetch(`${API_BASE}/global-insights`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export const getFairnessAudit = async () => {
  const response = await fetch(`${API_BASE}/fairness`);
  if (!response.ok) throw new Error("Fairness audit failed");
  return response.json();
};

export const getCausalGraph = async () => {
  const response = await fetch(`${API_BASE}/causal-graph`);
  if (!response.ok) throw new Error("Causal graph fetch failed");
  return response.json();
};

export const getManifoldProjection = async (profile) => {
  const response = await fetch(`${API_BASE}/manifold-projection`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(profile),
  });
  if (!response.ok) throw new Error("Manifold projection failed");
  return response.json();
};

export const getStabilityAudit = async () => {
  const response = await fetch(`${API_BASE}/stability-audit`);
  if (!response.ok) throw new Error("Stability audit failed");
  return response.json();
};

export const getPrivacyAudit = async () => {
  const response = await fetch(`${API_BASE}/privacy-audit`);
  if (!response.ok) throw new Error("Privacy audit failed");
  return response.json();
};

export const auditRobustness = async (profile) => {
  const response = await fetch(`${API_BASE}/robustness-audit`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(profile),
  });
  if (!response.ok) throw new Error("Robustness audit failed");
  return response.json();
};
