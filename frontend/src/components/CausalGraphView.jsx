import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";
import { getCausalGraph } from "../api";

export default function CausalGraphView() {
    const svgRef = useRef(null);
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        getCausalGraph()
            .then((res) => {
                setData(res);
                setLoading(false);
            })
            .catch((err) => {
                console.error(err);
                setError("Failed to load causal structure.");
                setLoading(false);
            });
    }, []);

    useEffect(() => {
        if (!data || !svgRef.current) return;

        const width = 800;
        const height = 500;
        const svg = d3.select(svgRef.current);
        svg.selectAll("*").remove();

        // Define arrowhead
        svg.append("defs").append("marker")
            .attr("id", "arrow")
            .attr("viewBox", "0 -5 10 10")
            .attr("refX", 25)
            .attr("refY", 0)
            .attr("markerWidth", 6)
            .attr("markerHeight", 6)
            .attr("orient", "auto")
            .append("path")
            .attr("d", "M0,-5L10,0L0,5")
            .attr("fill", "var(--text-muted)");

        const container = svg.append("g");

        const simulation = d3.forceSimulation(data.nodes)
            .force("link", d3.forceLink(data.links).id(d => d.id).distance(150))
            .force("charge", d3.forceManyBody().strength(-400))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collision", d3.forceCollide().radius(60));

        // Add Zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.1, 8])
            .on("zoom", (event) => {
                container.attr("transform", event.transform);
            });

        svg.call(zoom);

        const link = container.append("g")
            .selectAll("line")
            .data(data.links)
            .join("line")
            .attr("stroke", "rgba(255,255,255,0.1)")
            .attr("stroke-width", 1.5)
            .attr("marker-end", "url(#arrow)");

        const node = container.append("g")
            .selectAll("g")
            .data(data.nodes)
            .join("g")
            .call(drag(simulation));

        node.append("circle")
            .attr("r", 15)
            .attr("fill", "var(--accent)")
            .attr("stroke", "var(--accent-light)")
            .attr("stroke-width", 2);

        node.append("text")
            .text(d => d.id)
            .attr("x", 20)
            .attr("y", 5)
            .attr("fill", "var(--text-primary)")
            .style("font-size", "0.75rem")
            .style("font-weight", "600")
            .style("pointer-events", "none")
            .style("text-transform", "capitalize")
            .style("text-shadow", "0 0 10px rgba(0,0,0,0.8)");

        simulation.on("tick", () => {
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);

            node.attr("transform", d => `translate(${d.x},${d.y})`);
        });

        function drag(sim) {
            function started(event) {
                if (!event.active) sim.alphaTarget(0.3).restart();
                event.subject.fx = event.subject.x;
                event.subject.fy = event.subject.y;
            }
            function dragged(event) {
                event.subject.fx = event.x;
                event.subject.fy = event.y;
            }
            function ended(event) {
                if (!event.active) sim.alphaTarget(0);
                event.subject.fx = null;
                event.subject.fy = null;
            }
            return d3.drag().on("start", started).on("drag", dragged).on("end", ended);
        }
    }, [data]);

    if (loading) return <div className="card empty"><div className="spin"></div><p>Simulating Causal Discovery…</p></div>;
    if (error) return <div className="error-msg">{error}</div>;

    return (
        <div className="card" style={{ overflow: "hidden" }}>
            <h2>🧬 Structural Causal Model (DAG)</h2>
            <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem", marginBottom: "1rem" }}>
                Discovered via GES algorithm. This graph shows the <strong>causal dependencies</strong>
                inferred from the credit history. Nodes are features, arrows indicate direction of influence.
            </p>
            <div style={{ background: "rgba(0,0,0,0.2)", borderRadius: "8px", border: "1px solid var(--border)" }}>
                <svg ref={svgRef} width="100%" height="500" style={{ cursor: "move" }} />
            </div>
        </div>
    );
}
