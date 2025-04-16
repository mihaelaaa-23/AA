// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
  console.log("DOM fully loaded");
});

let originalGraphData = null;
let lastGraphType = "sparse";  // Changed default value

async function generateGraph() {
  console.log("Generate graph function called");

  const graphTypeElement = document.getElementById("graphType");
  const nodesElement = document.getElementById("nodeCount"); // Changed from "nodes" to "nodeCount"

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
    drawGraph(originalGraphData, data.traversal, data.selected, data.checked);
  } catch (error) {
    console.error("Error running traversal:", error);
    alert("Error running traversal: " + error.message);
  }
}

function resetGraph() {
  if (originalGraphData) {
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

  // Get the end node (last node in traversal) if available
  const endNode = traversal.length > 0 ? traversal[traversal.length - 1] : null;
  console.log("End node:", endNode);

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

      // For undirected graphs, only add each edge once
      if (!isDirected && edgeKey.has(key)) continue;

      edgeKey.add(key);

      links.push({
        source: srcId,
        target: dstId,
        weight: weight,
        isReverse: false
      });

      // For undirected, we still need to track the reverse edge for weights
      if (!isDirected) {
        // Check if reverse edge exists with a different weight
        const reverseEdge = neighbors.find(n => parseInt(n[0]) === dstId);
        if (reverseEdge && reverseEdge[1] !== weight) {
          links.push({
            source: srcId,
            target: dstId,
            weight: reverseEdge[1],
            isReverse: true
          });
        }
      }
    }
  }

  console.log("Nodes:", nodes);
  console.log("Links:", links);

  // Create simulation with increased distance between nodes and edges
  const simulation = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(links).id(d => d.id).distance(180).strength(0.4)) // Further increased distance
    .force("charge", d3.forceManyBody().strength(-500)) // Stronger repulsion
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("collision", d3.forceCollide().radius(40)); // Increased collision radius

  // Create a group for the links
  const linkGroup = svg.append("g");

  // Create link group
  const link = linkGroup.selectAll("line")
    .data(links.filter(d => !d.isReverse)) // Filter out reverse duplicates for rendering
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
    .attr("class", d => d.isReverse ? "reverse-weight" : "weight")
    .text(d => d.weight);

  // Create nodes - add special highlight for start node (node 0) and end node (if applicable)
  const node = svg.append("g")
    .selectAll("circle")
    .data(nodes)
    .join("circle")
    .attr("r", d => {
      if (d.id === 0) return 18; // Start node
      if (endNode !== null && d.id === endNode) return 18; // End node
      return 15; // Regular nodes
    })
    .attr("fill", d => {
      if (d.id === 0) return "#ffcc00"; // Start node is yellow
      if (endNode !== null && d.id === endNode) return "#ff3366"; // End node is red
      return "#ccc"; // Regular nodes
    })
    .attr("stroke", d => {
      if (endNode !== null && d.id === endNode) return "#990033"; // Darker border for end node
      return "#fff"; // Normal border for other nodes
    })
    .attr("stroke-width", d => {
      if (endNode !== null && d.id === endNode) return 2.5; // Thicker border for end node
      return 1.5; // Normal for other nodes
    })
    .attr("data-id", d => d.id)
    .attr("data-status", d => {
      if (d.id === 0) return "start";
      if (endNode !== null && d.id === endNode) return "end";
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
      if (d.id === 0 || (endNode !== null && d.id === endNode)) return 14; // Larger font for start/end nodes
      return 12; // Normal font for other nodes
    })
    .attr("font-weight", "bold")
    .attr("dy", 4)
    .attr("text-anchor", "middle")
    .attr("fill", d => {
      if (endNode !== null && d.id === endNode) return "#fff"; // White text for end node
      return "#000"; // Black text for other nodes
    });

  // Define helper function for finding edges
  const findEdge = (from, to) => {
    return d =>
      (d.source.id === from && d.target.id === to) ||
      (!isDirected && d.source.id === to && d.target.id === from);
  };

  // Keep track of confirmed nodes for vertex coloring logic
  const confirmedNodes = new Set();
  if (traversal.length > 0) {
    confirmedNodes.add(0); // Start node is always confirmed
    if (endNode !== null) confirmedNodes.add(endNode); // End node is also confirmed
  }

  // Reset all nodes to initial state
  node.attr("fill", d => {
    if (d.id === 0) return "#ffcc00"; // Start node is yellow
    if (endNode !== null && d.id === endNode) return "#ff3366"; // End node is red
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
    const delay = i * 400 + checked.length * 100;

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
          // Skip coloring the end node - keep its distinctive color
          if (endNode !== null && d.id === endNode) return false;
          return confirmedNodes.has(d.id);
        })
          .transition()
          .duration(200)
          .attr("fill", d => d.id === 0 ? "#ffaa00" : "#00cc66"); // Keep start node yellowish but confirmed
      }, 200);
    }, delay);
  });

  // Add a final animation to make the end node pulse if it exists
  if (endNode !== null) {
    const finalDelay = (checked.length + selected.length) * 400;

    setTimeout(() => {
      // Add pulsing effect to end node
      node.filter(d => d.id === endNode)
        .transition()
        .duration(500)
        .attr("r", 22) // Expand
        .transition()
        .duration(500)
        .attr("r", 18) // Contract
        .transition()
        .duration(500)
        .attr("r", 20) // Settle at medium size
        .attr("fill", "#ff3366") // Ensure end node stays red
        .attr("stroke-width", 3); // Thicker border
    }, finalDelay);
  }

  // Helper function to calculate edge midpoint with offset for bidirectional different weights
  function calculateLabelPosition(d, isReverse = false) {
    if (!d.source || !d.target) return { x: 0, y: 0 };

    const dx = d.target.x - d.source.x;
    const dy = d.target.y - d.source.y;
    const angle = Math.atan2(dy, dx);

    // Base position is the midpoint
    let x = (d.source.x + d.target.x) / 2;
    let y = (d.source.y + d.target.y) / 2;

    // Increased offset for better label separation
    const offsetDistance = isReverse ? 25 : -25;

    // Always apply an offset perpendicular to the edge to avoid overlapping
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
      const isReverse = d.isReverse;
      const pos = calculateLabelPosition(d, isReverse);

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
        link.weight === d.weight &&
        link.isReverse === d.isReverse);

      // Update the matching background rectangle
      labelBackgrounds.filter((_, i) => i === labelIndex)
        .attr("x", pos.x - labelWidth/2 - 4)
        .attr("y", pos.y - labelHeight/2 - 2)
        .attr("width", labelWidth + 8)
        .attr("height", labelHeight + 4);
    });
  });

  // Add legend for start/end nodes
  if (traversal.length > 0 && endNode !== null) {
    const legend = svg.append("g")
      .attr("transform", "translate(20, 20)");

    // Start node legend
    legend.append("circle")
      .attr("cx", 10)
      .attr("cy", 10)
      .attr("r", 10)
      .attr("fill", "#ffcc00");

    legend.append("text")
      .attr("x", 25)
      .attr("y", 15)
      .text("Start Node (0)")
      .attr("font-size", 12);

    // End node legend
    legend.append("circle")
      .attr("cx", 10)
      .attr("cy", 40)
      .attr("r", 10)
      .attr("fill", "#ff3366")
      .attr("stroke", "#990033")
      .attr("stroke-width", 2);

    legend.append("text")
      .attr("x", 25)
      .attr("y", 45)
      .text(`End Node (${endNode})`)
      .attr("font-size", 12);
  }

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

    // Update labels and backgrounds after centering
    linkLabels.transition()
        .duration(500)
        .attr("x", d => {
          const pos = calculateLabelPosition(d, d.isReverse);
          return pos.x + offsetX;
        })
        .attr("y", d => {
          const pos = calculateLabelPosition(d, d.isReverse);
          return pos.y + offsetY;
        });

    labelBackgrounds.transition()
        .duration(500)
        .attr("x", function() {
          return parseFloat(d3.select(this).attr("x")) + offsetX;
        })
        .attr("y", function() {
          return parseFloat(d3.select(this).attr("y")) + offsetY;
        });
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

  return d3.drag()
    .on("start", dragstarted)
    .on("drag", dragged)
    .on("end", dragended);
}