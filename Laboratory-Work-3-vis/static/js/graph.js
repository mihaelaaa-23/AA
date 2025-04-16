let originalGraphData = null;
let lastGraphType = "finite";
let lastNodeCount = 10;

async function generateGraph() {
  lastGraphType = document.getElementById("graphType").value;
  lastNodeCount = parseInt(document.getElementById("nodeCount").value);

  const response = await fetch("/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      graphType: lastGraphType,
      algorithm: "NONE",
      size: lastNodeCount
    })
  });

  const data = await response.json();
  originalGraphData = structuredClone(data.graph);
  drawGraph(data.graph, [], []);
}

console.log("Drawing graph with nodes:", Object.keys(graph).length);


async function runTraversal(algo) {
  if (!originalGraphData) {
    alert("Graph not generated yet.");
    return;
  }

  console.log("Running traversal:", algo);
  console.log("Graph:", originalGraphData);

  const response = await fetch("/api/traverse", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      graph: originalGraphData,
      algorithm: algo,
      graphType: lastGraphType
    })
  });

  if (!response.ok) {
    console.error("Traversal failed:", response.statusText);
    return;
  }

  const data = await response.json();
  console.log("Traversal result:", data);

  drawGraph(originalGraphData, data.traversal, data.edges, data.checked);
}


function resetGraph() {
  if (originalGraphData) {
    drawGraph(originalGraphData, [], []);
  }
}

function drawGraph(graph, traversal, edges = [], checked = []) {
  const svg = d3.select("#graph");
  svg.selectAll("*").remove();
  const width = +svg.attr("width");
  const height = +svg.attr("height");
  const seenEdges = new Set();
  const confirmedEdges = new Set();
  const confirmedNodes = new Set();

  // Determine if this is a directed graph based on graph type
  const isDirected = lastGraphType === "directed" || lastGraphType === "acyclic";

  // Define arrow markers for directed graphs
  if (isDirected) {
    // Add arrow marker definition
    svg.append("defs").append("marker")
      .attr("id", "arrowhead")
      .attr("viewBox", "0 -5 10 10")
      .attr("refX", 25) // Position the arrowhead away from the end of the line
      .attr("refY", 0)
      .attr("orient", "auto")
      .attr("markerWidth", 6)
      .attr("markerHeight", 6)
      .attr("xoverflow", "visible")
      .append("path")
      .attr("d", "M 0,-5 L 10,0 L 0,5")
      .attr("fill", "#999")
      .attr("stroke", "none");
  }

  const nodes = Object.keys(graph).map(d => ({ id: +d }));
  const links = [];

  // Create links based on the graph data
  for (const [src, targets] of Object.entries(graph)) {
    for (const tgt of targets) {
      if (Array.isArray(tgt)) {
        const [target, weight] = tgt;
        // For directed graphs, don't filter by source < target
        if (isDirected || +src < +target) {
          links.push({ source: +src, target: +target, weight });
        }
      } else {
        // For directed graphs, don't filter by source < target
        if (isDirected || +src < +tgt) {
          links.push({ source: +src, target: +tgt });
        }
      }
    }
  }

  const simulation = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(links).distance(100).id(d => d.id))
    .force("charge", d3.forceManyBody().strength(-300))
    .force("center", d3.forceCenter(width / 2, height / 2));

  // Create links with arrows if directed
  const link = svg.append("g")
    .attr("stroke", "#999")
    .attr("stroke-opacity", 0.6)
    .selectAll("line")
    .data(links)
    .join("line")
    .attr("stroke-width", 2)
    .attr("marker-end", isDirected ? "url(#arrowhead)" : ""); // Add arrowhead marker for directed graphs

  const linkLabels = svg.append("g")
    .selectAll("text")
    .data(links.filter(d => d.weight))
    .join("text")
    .attr("font-size", 12)
    .attr("fill", "black")
    .text(d => d.weight);

  const node = svg.append("g")
    .attr("stroke", "#fff")
    .attr("stroke-width", 1.5)
    .selectAll("circle")
    .data(nodes)
    .join("circle")
    .attr("r", 15)
    .attr("fill", "#ccc")
    .call(drag(simulation));

  const label = svg.append("g")
    .selectAll("text")
    .data(nodes)
    .join("text")
    .text(d => d.id)
    .attr("font-size", 12)
    .attr("dy", 4)
    .attr("text-anchor", "middle");

  checked.forEach(([from, to], i) => {
    const selector = d =>
      (d.source.id === from && d.target.id === to) ||
      (!isDirected && d.source.id === to && d.target.id === from);

    const delay = i * 600;

    // 🟠 Colorează portocaliu (doar dacă NU e deja verde)
    setTimeout(() => {
      if (
        confirmedEdges.has(`${from}-${to}`) ||
        (!isDirected && confirmedEdges.has(`${to}-${from}`))
      ) return;

      link.filter(selector)
        .filter(function () {
          return d3.select(this).attr("data-status") !== "confirmed";
        })
        .attr("stroke", "#ffaa00")
        .attr("stroke-width", 2)
        .attr("data-status", "checked");
    }, delay);
  });

  edges.forEach(([from, to], i) => {
    const selector = d =>
      (d.source.id === from && d.target.id === to) ||
      (!isDirected && d.source.id === to && d.target.id === from);

    const delay = i * 600 + 300;

    setTimeout(() => {
      // ✅ Colorează muchia, doar dacă nu a fost confirmată
      link.filter(selector)
        .attr("stroke", "#00cc66")
        .attr("stroke-width", 4)
        .attr("data-status", "confirmed");

      // If it's a directed graph, also update the arrowhead color
      if (isDirected) {
        // Update arrow marker for the confirmed path
        svg.select("defs").append("marker")
          .attr("id", "arrowhead-confirmed")
          .attr("viewBox", "0 -5 10 10")
          .attr("refX", 25)
          .attr("refY", 0)
          .attr("orient", "auto")
          .attr("markerWidth", 6)
          .attr("markerHeight", 6)
          .attr("xoverflow", "visible")
          .append("path")
          .attr("d", "M 0,-5 L 10,0 L 0,5")
          .attr("fill", "#00cc66")
          .attr("stroke", "none");

        link.filter(selector)
          .attr("marker-end", "url(#arrowhead-confirmed)");
      }

      // ⏳ Apoi colorezi nodul
      setTimeout(() => {
        node.filter(d => d.id === to)
          .filter(function () {
            return d3.select(this).attr("data-status") !== "confirmed";
          })
          .attr("fill", "#00cc66")
          .attr("data-status", "confirmed");
      }, 200);
    }, delay);
  });

  simulation.on("tick", () => {
    // For directed graphs, adjust the line endpoints to not overlap with the arrowhead
    if (isDirected) {
      link
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", function(d) {
          // Calculate the angle
          const dx = d.target.x - d.source.x;
          const dy = d.target.y - d.source.y;
          const angle = Math.atan2(dy, dx);
          // Move the endpoint back by the radius of the node
          const radius = 15;
          return d.target.x - (Math.cos(angle) * radius);
        })
        .attr("y2", function(d) {
          const dx = d.target.x - d.source.x;
          const dy = d.target.y - d.source.y;
          const angle = Math.atan2(dy, dx);
          const radius = 15;
          return d.target.y - (Math.sin(angle) * radius);
        });
    } else {
      link
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);
    }

    node
      .attr("cx", d => d.x)
      .attr("cy", d => d.y);

    label
      .attr("x", d => d.x)
      .attr("y", d => d.y);

    linkLabels
      ?.attr("x", d => (d.source.x + d.target.x) / 2)
      ?.attr("y", d => (d.source.y + d.target.y) / 2);
  });

  simulation.on("end", () => {
    const minX = Math.min(...nodes.map(d => d.x));
    const maxX = Math.max(...nodes.map(d => d.x));
    const minY = Math.min(...nodes.map(d => d.y));
    const maxY = Math.max(...nodes.map(d => d.y));

    const offsetX = width / 2 - (minX + maxX) / 2;
    const offsetY = height / 2 - (minY + maxY) / 2;

    // Aplicăm offset doar la poziționare SVG, nu stricăm `d.x` / `d.y`
    node.attr("cx", d => d.x + offsetX)
        .attr("cy", d => d.y + offsetY);

    label.attr("x", d => d.x + offsetX)
         .attr("y", d => d.y + offsetY);

    // Adjust the arrow position based on directed graph
    if (isDirected) {
      link.attr("x1", d => d.source.x + offsetX)
          .attr("y1", d => d.source.y + offsetY)
          .attr("x2", function(d) {
            const dx = d.target.x - d.source.x;
            const dy = d.target.y - d.source.y;
            const angle = Math.atan2(dy, dx);
            const radius = 15;
            return (d.target.x - (Math.cos(angle) * radius)) + offsetX;
          })
          .attr("y2", function(d) {
            const dx = d.target.x - d.source.x;
            const dy = d.target.y - d.source.y;
            const angle = Math.atan2(dy, dx);
            const radius = 15;
            return (d.target.y - (Math.sin(angle) * radius)) + offsetY;
          });
    } else {
      link.attr("x1", d => d.source.x + offsetX)
          .attr("y1", d => d.source.y + offsetY)
          .attr("x2", d => d.target.x + offsetX)
          .attr("y2", d => d.target.y + offsetY);
    }

    linkLabels?.attr("x", d => (d.source.x + d.target.x) / 2 + offsetX)
               ?.attr("y", d => (d.source.y + d.target.y) / 2 + offsetY);
  });

  traversal.forEach((id, i) => {
    setTimeout(() => {
      node.filter(d => d.id === id).attr("fill", "#00cc66");
    }, i * 500);
  });
}

function drag(simulation) {
  function dragstarted(event) {
    if (!event.active) simulation.alphaTarget(0.3).restart();
    event.subject.fx = event.subject.x;
    event.subject.fy = event.subject.y;
  }

  function dragged(event) {
    event.subject.fx = event.x;
    event.subject.fy = event.y;
  }

  function dragended(event) {
    if (!event.active) simulation.alphaTarget(0);
    event.subject.fx = null;
    event.subject.fy = null;
  }

  return d3.drag().on("start", dragstarted).on("drag", dragged).on("end", dragended);
}