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
    body: JSON.stringify({ graph: originalGraphData, algorithm: algo })
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


  const nodes = Object.keys(graph).map(d => ({ id: +d }));
  const links = [];

  checked.forEach(([from, to], i) => {
    const selector = d =>
      (d.source.id === from && d.target.id === to) ||
      (d.source.id === to && d.target.id === from);

    const delay = i * 600;

    // 🟠 Colorează portocaliu (doar dacă NU e deja verde)
    setTimeout(() => {
      if (
        confirmedEdges.has(`${from}-${to}`) ||
        confirmedEdges.has(`${to}-${from}`)
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
      (d.source.id === to && d.target.id === from);

    const delay = i * 600 + 300;

    setTimeout(() => {
      // ✅ Colorează muchia, doar dacă nu a fost confirmată

      link.filter(selector)
        .attr("stroke", "#00cc66")
        .attr("stroke-width", 4)
        .attr("data-status", "confirmed");

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


  for (const [src, targets] of Object.entries(graph)) {
  for (const tgt of targets) {
    if (Array.isArray(tgt)) {
      const [target, weight] = tgt;
      if (+src < +target) links.push({ source: +src, target: +target, weight });
    } else {
      if (+src < +tgt) links.push({ source: +src, target: +tgt });
    }
  }
}

  const simulation = d3.forceSimulation(nodes)
  .force("link", d3.forceLink(links).distance(100).id(d => d.id))
  .force("charge", d3.forceManyBody().strength(-300))
  .force("center", d3.forceCenter(width / 2, height / 2));

  const link = svg.append("g")
    .attr("stroke", "#999")
    .attr("stroke-opacity", 0.6)
    .selectAll("line")
    .data(links)
    .join("line")
    .attr("stroke-width", 2);

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

  simulation.on("tick", () => {
    link
      .attr("x1", d => d.source.x)
      .attr("y1", d => d.source.y)
      .attr("x2", d => d.target.x)
      .attr("y2", d => d.target.y);

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

    link.attr("x1", d => d.source.x + offsetX)
        .attr("y1", d => d.source.y + offsetY)
        .attr("x2", d => d.target.x + offsetX)
        .attr("y2", d => d.target.y + offsetY);

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