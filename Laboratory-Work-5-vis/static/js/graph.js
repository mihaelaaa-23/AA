// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
  console.log("DOM fully loaded");
});

let originalGraphData = null;
let lastGraphType = "sparse";  // Default value
let totalMSTWeight = 0;  // Track the total weight of the MST

async function generateGraph() {
  console.log("Generate graph function called");

  const graphTypeElement = document.getElementById("graphType");
  const nodesElement = document.getElementById("nodeCount");

  if (!graphTypeElement || !nodesElement) {
    console.error("Could not find required elements");
    alert("Error: Could not find required form elements");
    return;
  }

  const graphTypeSelect = graphTypeElement.value;
  lastGraphType = graphTypeSelect;
  const isDirected = graphTypeSelect.includes("directed");
  const size = parseInt(nodesElement.value);

  console.log("Graph type:", graphTypeSelect);
  console.log("Number of nodes:", size);

  try {
    const res = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        graphType: graphTypeSelect,
        size: size
      })
    });

    if (!res.ok) {
      throw new Error(`HTTP error! Status: ${res.status}`);
    }

    const data = await res.json();
    console.log("Received graph data:", data);

    originalGraphData = JSON.parse(JSON.stringify(data.graph)); // Deep clone
    document.getElementById("algorithm-legend").style.display = "none"; // Hide legend on new graph
    drawGraph(data.graph, [], [], []);
  } catch (error) {
    console.error("Error generating graph:", error);
    alert("Error generating graph: " + error.message);
  }
}

async function runTraversal(algo) {
  if (!originalGraphData) {
    alert("Please generate a graph first!");
    return;
  }

  try {
    const res = await fetch("/api/traverse", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        graph: originalGraphData,
        algorithm: algo
      })
    });

    if (!res.ok) {
      throw new Error(`HTTP error! Status: ${res.status}`);
    }

    const data = await res.json();

    // Update algorithm legend with the current algorithm name
    document.getElementById("algorithm-legend").style.display = "block";
    document.getElementById("algo-title").textContent =
      algo === "prim" ? "Prim's Algorithm" : "Kruskal's Algorithm";

    // Calculate MST total weight
    totalMSTWeight = calculateMSTWeight(originalGraphData, data.selected);
    document.getElementById("mst-weight").textContent = `MST Total Weight: ${totalMSTWeight}`;

    drawGraph(originalGraphData, data.traversal, data.selected, data.checked);
  } catch (error) {
    console.error("Error running traversal:", error);
    alert("Error running traversal: " + error.message);
  }
}

function calculateMSTWeight(graph, selectedEdges) {
  let totalWeight = 0;

  // For each selected edge in the MST
  for (const [u, v] of selectedEdges) {
    const sourceNode = parseInt(u);
    const targetNode = parseInt(v);

    // Find the weight of this edge in the original graph
    const neighbors = graph[sourceNode];
    for (const [neighbor, weight] of neighbors) {
      if (parseInt(neighbor) === targetNode) {
        totalWeight += weight;
        break;
      }
    }
  }

  return totalWeight;
}

function resetGraph() {
  if (originalGraphData) {
    document.getElementById("algorithm-legend").style.display = "none";
    drawGraph(originalGraphData, [], [], []);
  } else {
    alert("No graph to reset!");
  }
}

function drawGraph(graph, traversal, selected = [], checked = []) {
  console.log("Drawing graph with data:", graph);

  const svg = d3.select("#graph");
  svg.selectAll("*").remove();

  const width = svg.node().clientWidth || 1000;
  const height = svg.node().clientHeight || 600;

  // Determine if this is a directed graph - get from the data
  const isDirected = lastGraphType.includes("directed");
  console.log("Is directed:", isDirected);

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

    // Add marker for confirmed paths
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
      .attr("fill", "#00cc66") // Green for confirmed
      .attr("stroke", "none");

    // Add marker for checked paths
    svg.select("defs").append("marker")
      .attr("id", "arrowhead-checked")
      .attr("viewBox", "0 -5 10 10")
      .attr("refX", 25)
      .attr("refY", 0)
      .attr("orient", "auto")
      .attr("markerWidth", 6)
      .attr("markerHeight", 6)
      .attr("xoverflow", "visible")
      .append("path")
      .attr("d", "M 0,-5 L 10,0 L 0,5")
      .attr("fill", "#ffaa00") // Orange for checked
      .attr("stroke", "none");
  }

  // Convert graph data to D3 format
  const nodes = Object.keys(graph).map(d => ({ id: parseInt(d) }));
  const links = [];
  const edgeKey = new Set(); // Track edges to avoid duplicates

  for (const [src, neighbors] of Object.entries(graph)) {
    for (const [dst, weight] of neighbors) {
      const srcId = parseInt(src);
      const dstId = parseInt(dst);

      // Create a unique key for this edge
      const key = srcId < dstId ? `${srcId}-${dstId}` : `${dstId}-${srcId}`;

      // For undirected graphs or MST visualization, only add each edge once
      if (edgeKey.has(key)) continue;

      edgeKey.add(key);

      links.push({
        source: srcId,
        target: dstId,
        weight: weight,
        isReverse: false
      });
    }
  }

  console.log("Nodes:", nodes);
  console.log("Links:", links);

  // Create simulation with increased distance between nodes and edges
  const simulation = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(links).id(d => d.id).distance(180).strength(0.4)) // Increased distance
    .force("charge", d3.forceManyBody().strength(-500)) // Stronger repulsion
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("collision", d3.forceCollide().radius(40)); // Increased collision radius

  // Create a group for the links
  const linkGroup = svg.append("g");

  // Create link group
  const link = linkGroup.selectAll("line")
    .data(links)
    .join("line")
    .attr("stroke", "#999")
    .attr("stroke-opacity", 0.6)
    .attr("stroke-width", 2)
    .attr("marker-end", isDirected ? "url(#arrowhead)" : "")
    .attr("data-source", d => d.source.id)
    .attr("data-target", d => d.target.id)
    .attr("data-status", "normal");

  // Create a group for the weight labels with a background
  const labelGroup = svg.append("g");

  // First create background rectangles for the labels
  const labelBackgrounds = labelGroup.selectAll("rect")
    .data(links.filter(d => d.weight)) // Only edges with weights
    .join("rect")
    .attr("fill", "white")
    .attr("fill-opacity", 0.8)
    .attr("rx", 4)
    .attr("ry", 4)
    .attr("width", 20) // Default, will be updated
    .attr("height", 20) // Default, will be updated
    .attr("class", "weight-background");

  // Create weight labels - make them more visible
  const linkLabels = labelGroup.selectAll("text")
    .data(links.filter(d => d.weight))
    .join("text")
    .attr("font-size", 14)
    .attr("font-weight", "bold")
    .attr("fill", "black")
    .attr("text-anchor", "middle")
    .attr("dominant-baseline", "middle")
    .attr("class", "weight")
    .text(d => d.weight);

  // Create nodes with special highlight for start node (node 0)
  const node = svg.append("g")
    .selectAll("circle")
    .data(nodes)
    .join("circle")
    .attr("r", d => {
      if (d.id === 0) return 18; // Start node
      return 15; // Regular nodes
    })
    .attr("fill", d => {
      if (d.id === 0) return "#ffcc00"; // Start node is yellow
      return "#ccc"; // Regular nodes
    })
    .attr("stroke", "#fff")
    .attr("stroke-width", 1.5)
    .attr("data-id", d => d.id)
    .attr("data-status", d => {
      if (d.id === 0) return "start";
      return "normal";
    })
    .call(drag(simulation));

  // Create node labels
  const nodeLabels = svg.append("g")
    .selectAll("text")
    .data(nodes)
    .join("text")
    .text(d => d.id)
    .attr("font-size", d => {
      if (d.id === 0) return 14; // Larger font for start node
      return 12; // Normal font for other nodes
    })
    .attr("font-weight", "bold")
    .attr("dy", 4)
    .attr("text-anchor", "middle")
    .attr("fill", "#000");

  // Define helper function for finding edges
  const findEdge = (from, to) => {
    return d =>
      (d.source.id === from && d.target.id === to) ||
      (d.source.id === to && d.target.id === from);
  };

  // Keep track of confirmed nodes for vertex coloring logic
  const confirmedNodes = new Set();
  if (traversal.length > 0) {
    confirmedNodes.add(0); // Start node is always confirmed
  }

  // Reset all nodes to initial state
  node.attr("fill", d => {
    if (d.id === 0) return "#ffcc00"; // Start node is yellow
    return "#ccc"; // Regular nodes
  });

  // First, prepare transition for checked edges with delay
  const totalAnimationDuration = Math.max(checked.length, selected.length) * 400;

  // For smoother transitions, use D3 transitions
  checked.forEach(([from, to], i) => {
    const delay = i * 400;

    setTimeout(() => {
      // Only color if not already confirmed
      const edgeElement = link.filter(findEdge(from, to));

      if (edgeElement.attr("data-status") !== "confirmed") {
        edgeElement
          .transition()
          .duration(300)
          .attr("stroke", "#ffaa00") // Orange
          .attr("stroke-width", 2)
          .attr("data-status", "checked");

        if (isDirected) {
          edgeElement.attr("marker-end", "url(#arrowhead-checked)");
        }
      }
    }, delay);
  });

  // Then, prepare transitions for the selected/confirmed edges
  selected.forEach(([from, to], i) => {
    const delay = i * 400 + Math.min(checked.length * 100, 2000); // Cap the delay to avoid too long waits for large graphs

    setTimeout(() => {
      // Mark the source node as confirmed if not already
      if (!confirmedNodes.has(from)) {
        confirmedNodes.add(from);
      }

      // Mark the target node as confirmed
      confirmedNodes.add(to);

      // Color the edge green with transition
      const edgeElement = link.filter(findEdge(from, to))
        .transition()
        .duration(300)
        .attr("stroke", "#00cc66") // Green
        .attr("stroke-width", 4)
        .attr("data-status", "confirmed");

      if (isDirected) {
        edgeElement.attr("marker-end", "url(#arrowhead-confirmed)");
      }

      // Update both nodes connected by this edge
      setTimeout(() => {
        node.filter(d => {
          return confirmedNodes.has(d.id);
        })
          .transition()
          .duration(200)
          .attr("fill", d => d.id === 0 ? "#ffaa00" : "#00cc66"); // Keep start node yellowish but confirmed
      }, 200);
    }, delay);
  });

  // Helper function to calculate edge midpoint for weight labels
  function calculateLabelPosition(d) {
    if (!d.source || !d.target) return { x: 0, y: 0 };

    const dx = d.target.x - d.source.x;
    const dy = d.target.y - d.source.y;
    const angle = Math.atan2(dy, dx);

    // Base position is the midpoint
    let x = (d.source.x + d.target.x) / 2;
    let y = (d.source.y + d.target.y) / 2;

    // Apply a small offset perpendicular to the edge
    const offsetDistance = 10;
    x += Math.sin(angle) * offsetDistance;
    y -= Math.cos(angle) * offsetDistance;

    return { x, y };
  }

  // Update positions on tick
  simulation.on("tick", () => {
    // Extra spacing for directed edges to account for arrowheads
    const nodeRadius = 18; // Use the larger radius for extra spacing

    // For directed graphs, adjust the line endpoints to not overlap with the arrowhead
    if (isDirected) {
      link
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", function(d) {
          // Calculate angle and adjust endpoint
          const dx = d.target.x - d.source.x;
          const dy = d.target.y - d.source.y;
          const angle = Math.atan2(dy, dx);
          return d.target.x - (Math.cos(angle) * nodeRadius);
        })
        .attr("y2", function(d) {
          const dx = d.target.x - d.source.x;
          const dy = d.target.y - d.source.y;
          const angle = Math.atan2(dy, dx);
          return d.target.y - (Math.sin(angle) * nodeRadius);
        });
    } else {
      // For undirected graphs, have a bit of gap from the node
      link
        .attr("x1", function(d) {
          const dx = d.target.x - d.source.x;
          const dy = d.target.y - d.source.y;
          const angle = Math.atan2(dy, dx);
          return d.source.x + (Math.cos(angle) * (nodeRadius - 5));
        })
        .attr("y1", function(d) {
          const dx = d.target.x - d.source.x;
          const dy = d.target.y - d.source.y;
          const angle = Math.atan2(dy, dx);
          return d.source.y + (Math.sin(angle) * (nodeRadius - 5));
        })
        .attr("x2", function(d) {
          const dx = d.target.x - d.source.x;
          const dy = d.target.y - d.source.y;
          const angle = Math.atan2(dy, dx);
          return d.target.x - (Math.cos(angle) * (nodeRadius - 5));
        })
        .attr("y2", function(d) {
          const dx = d.target.x - d.source.x;
          const dy = d.target.y - d.source.y;
          const angle = Math.atan2(dy, dx);
          return d.target.y - (Math.sin(angle) * (nodeRadius - 5));
        });
    }

    // Update node and label positions
    node.attr("cx", d => d.x).attr("cy", d => d.y);
    nodeLabels.attr("x", d => d.x).attr("y", d => d.y);

    // Update the weight labels and their backgrounds
    linkLabels.each(function(d) {
      const pos = calculateLabelPosition(d);

      d3.select(this)
        .attr("x", pos.x)
        .attr("y", pos.y);

      // Find the matching background for this label
      const labelWidth = this.getBBox().width;
      const labelHeight = this.getBBox().height;

      // Find index of this label in the data
      const labelIndex = links.findIndex(link =>
        link.source.id === d.source.id &&
        link.target.id === d.target.id &&
        link.weight === d.weight);

      // Update the matching background rectangle
      labelBackgrounds.filter((_, i) => i === labelIndex)
        .attr("x", pos.x - labelWidth/2 - 4)
        .attr("y", pos.y - labelHeight/2 - 2)
        .attr("width", labelWidth + 8)
        .attr("height", labelHeight + 4);
    });
  });

  // Center the graph after force simulation ends
  simulation.alpha(1).restart(); // Ensure simulation starts properly

  simulation.on("end", () => {
    const minX = Math.min(...nodes.map(d => d.x));
    const maxX = Math.max(...nodes.map(d => d.x));
    const minY = Math.min(...nodes.map(d => d.y));
    const maxY = Math.max(...nodes.map(d => d.y));

    const graphWidth = maxX - minX;
    const graphHeight = maxY - minY;

    // Calculate offset to center, with some padding
    const offsetX = (width - graphWidth) / 2 - minX;
    const offsetY = (height - graphHeight) / 2 - minY;

    // Apply the offset to center the graph with smooth transition
    node.transition()
        .duration(500)
        .attr("cx", d => d.x + offsetX)
        .attr("cy", d => d.y + offsetY);

    nodeLabels.transition()
         .duration(500)
         .attr("x", d => d.x + offsetX)
         .attr("y", d => d.y + offsetY);

    link.transition()
        .duration(500)
        .attr("x1", d => {
          if (d.x1) return d.x1 + offsetX;
          return d.source.x + offsetX;
        })
        .attr("y1", d => {
          if (d.y1) return d.y1 + offsetY;
          return d.source.y + offsetY;
        })
        .attr("x2", d => {
          if (d.x2) return d.x2 + offsetX;
          return d.target.x + offsetX;
        })
        .attr("y2", d => {
          if (d.y2) return d.y2 + offsetY;
          return d.target.y + offsetY;
        });

    // Update weight labels and their backgrounds with the new positions
    linkLabels.transition()
      .duration(500)
      .attr("x", d => calculateLabelPosition(d).x + offsetX)
      .attr("y", d => calculateLabelPosition(d).y + offsetY);

    labelBackgrounds.transition()
      .duration(500)
      .attr("x", function() {
        const label = d3.select(linkLabels.nodes()[labelBackgrounds.data().indexOf(d3.select(this).datum())]);
        const bbox = label.node().getBBox();
        return (calculateLabelPosition(d3.select(this).datum()).x + offsetX) - bbox.width/2 - 4;
      })
      .attr("y", function() {
        const label = d3.select(linkLabels.nodes()[labelBackgrounds.data().indexOf(d3.select(this).datum())]);
        const bbox = label.node().getBBox();
        return (calculateLabelPosition(d3.select(this).datum()).y + offsetY) - bbox.height/2 - 2;
      });

    // Update node and edge positions in the data structure
    nodes.forEach(node => {
      node.x += offsetX;
      node.y += offsetY;
    });
  });
}

// Drag functionality for nodes
function drag(simulation) {
  function dragstarted(event, d) {
    if (!event.active) simulation.alphaTarget(0.3).restart();
    d.fx = d.x;
    d.fy = d.y;
  }

  function dragged(event, d) {
    d.fx = event.x;
    d.fy = event.y;
  }

  function dragended(event, d) {
    if (!event.active) simulation.alphaTarget(0);
    d.fx = null;
    d.fy = null;
  }

  return d3.drag()
    .on("start", dragstarted)
    .on("drag", dragged)
    .on("end", dragended);
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
  // Set up event listeners
  document.getElementById('generate-button').addEventListener('click', generateGraph);
  document.getElementById('prim-button').addEventListener('click', () => runTraversal('prim'));
  document.getElementById('kruskal-button').addEventListener('click', () => runTraversal('kruskal'));
  document.getElementById('reset-button').addEventListener('click', resetGraph);

  // Optional: Generate an initial graph
  generateGraph();
});