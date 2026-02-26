import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";
import { getManifoldProjection } from "../api";

export default function ManifoldProjector({ inputData }) {
    const svgRef = useRef(null);
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    useEffect(() => {
        if (!inputData) {
            setLoading(false);
            setData(null);
            setError(null);
            return;
        }

        setLoading(true);
        setError(null);
        getManifoldProjection(inputData)
            .then(setData)
            .catch((err) => {
                console.error(err);
                setError("Failed to compute latent projection for the current profile.");
            })
            .finally(() => setLoading(false));
    }, [inputData]);

    useEffect(() => {
        if (!data || !svgRef.current) return;

        const width = 600;
        const height = 400;
        const svg = d3.select(svgRef.current);
        svg.selectAll("*").remove();

        const margin = { top: 20, right: 20, bottom: 40, left: 40 };
        const chartWidth = width - margin.left - margin.right;
        const chartHeight = height - margin.top - margin.bottom;

        const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

        const x = d3.scaleLinear()
            .domain(d3.extent(data.points, d => d[0]))
            .range([0, chartWidth]);

        const y = d3.scaleLinear()
            .domain(d3.extent(data.points, d => d[1]))
            .range([chartHeight, 0]);

        // Data points (population)
        g.selectAll("circle.point")
            .data(data.points)
            .join("circle")
            .attr("class", "point")
            .attr("cx", d => x(d[0]))
            .attr("cy", d => y(d[1]))
            .attr("r", 2.5)
            .attr("fill", "var(--accent)")
            .attr("opacity", 0.15);

        // User point (Current applicant)
        if (data.user_point) {
            const [ux, uy] = data.user_point;
            g.append("circle")
                .attr("cx", x(ux))
                .attr("cy", y(uy))
                .attr("r", 8)
                .attr("fill", "var(--cyan)")
                .attr("stroke", "#fff")
                .attr("stroke-width", 2)
                .style("filter", "drop-shadow(0 0 10px var(--cyan))");

            g.append("text")
                .attr("x", x(ux) + 12)
                .attr("y", y(uy) + 4)
                .text("Current Applicant")
                .attr("fill", "#fff")
                .style("font-size", "0.75rem")
                .style("font-weight", "600");
        }

        // Axes
        g.append("g")
            .attr("transform", `translate(0,${chartHeight})`)
            .call(d3.axisBottom(x).ticks(5))
            .attr("color", "var(--text-muted)");

        g.append("g")
            .call(d3.axisLeft(y).ticks(5))
            .attr("color", "var(--text-muted)");

    }, [data]);

    if (!inputData) {
        return (
            <div className="card empty">
                <div className="big-icon">🧭</div>
                <p>Run an individual analysis first to project a profile into latent space.</p>
            </div>
        );
    }

    if (loading) return <div className="card empty"><div className="spin"></div><p>Projecting to Latent Manifold…</p></div>;
    if (error) return <div className="error-msg">⚠ {error}</div>;

    return (
        <div className="card">
            <h2>🌀 Latent Manifold Projection</h2>
            <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem", marginBottom: "1rem" }}>
                Visualization of the <strong>Beta-VAE</strong> latent space ($z$). The blue dot shows exactly where this
                applicant sits on the "Manifold of Financial Profiles."
            </p>
            <div style={{ background: "rgba(0,0,0,0.1)", borderRadius: "8px", border: "1px solid var(--border)" }}>
                <svg ref={svgRef} width="100%" height="400" />
            </div>
        </div>
    );
}
