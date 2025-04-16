from flask import Flask, render_template, request, jsonify
import random
import heapq
from collections import defaultdict

app = Flask(__name__)
graph_data = {}


def generate_graph(num_nodes, graph_type):
    # For MST algorithms, we only need undirected graphs
    # If directed is selected, we'll still create undirected edges but keep the directed UI
    is_directed_ui = "directed" in graph_type
    is_sparse = "sparse" in graph_type

    adjacency = {i: [] for i in range(num_nodes)}
    added = set()

    # Define appropriate connection density
    if is_sparse:
        # For sparse graphs, connect to approximately 2-3 nodes
        connection_ratio = 0.2  # Connect to about 20% of other nodes
    else:
        # For dense graphs, connect to most other nodes
        connection_ratio = 0.7  # Connect to about 70% of other nodes

    for i in range(num_nodes):
        possible_connections = [j for j in range(num_nodes) if i != j]

        # Determine how many connections to make based on density
        num_connections = max(1, int(len(possible_connections) * connection_ratio))
        # Make sure we don't try to sample more than available
        num_connections = min(num_connections, len(possible_connections))

        connections = random.sample(possible_connections, num_connections) if possible_connections else []

        for j in connections:
            # Check if edge already exists (for undirected graphs)
            if (i, j) in added or (j, i) in added:
                continue

            weight = random.randint(1, 10)
            adjacency[i].append([j, weight])
            added.add((i, j))

            # MST algorithms need undirected graphs, so add the reverse edge too
            # even if the UI might show directed edges
            adjacency[j].append([i, weight])
            added.add((j, i))

    return adjacency


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.json
    size = int(data.get("size", 10))
    graph_type = data.get("graphType", "undirected-sparse")
    graph = generate_graph(size, graph_type)
    graph_data["graph"] = graph
    return jsonify({"graph": graph})


@app.route("/api/traverse", methods=["POST"])
def traverse():
    data = request.json
    graph = data.get("graph", {})
    algorithm = data.get("algorithm", "prim")

    if algorithm == "prim":
        return jsonify(simulate_prim(graph))
    elif algorithm == "kruskal":
        return jsonify(simulate_kruskal(graph))

    return jsonify({"error": "Invalid algorithm"}), 400


@app.route("/api/reset", methods=["POST"])
def reset():
    return jsonify({"graph": graph_data.get("graph", {})})


def simulate_prim(graph):
    # Convert string keys to integers
    graph = {int(k): v for k, v in graph.items()}

    # Start from node 0
    start_node = 0

    # Initialize data structures
    num_nodes = len(graph)
    visited = {start_node}
    mst_edges = []
    traversal = [start_node]
    checked = []

    # Priority queue for edges
    edge_queue = []

    # Add all edges from start node to queue
    for neighbor, weight in graph[start_node]:
        heapq.heappush(edge_queue, (weight, start_node, neighbor))
        checked.append([start_node, neighbor])

    # While there are unvisited nodes
    while edge_queue and len(visited) < num_nodes:
        # Get the edge with minimum weight
        weight, u, v = heapq.heappop(edge_queue)

        # If the destination is already visited, skip
        if v in visited:
            continue

        # Add the destination to visited
        visited.add(v)
        traversal.append(v)

        # Add the edge to MST
        mst_edges.append([u, v])

        # Add all edges from the new node to the queue
        for neighbor, weight in graph[v]:
            if neighbor not in visited:
                heapq.heappush(edge_queue, (weight, v, neighbor))
                checked.append([v, neighbor])

    return {
        "traversal": traversal,
        "checked": checked,
        "selected": mst_edges
    }


def simulate_kruskal(graph):
    # Convert string keys to integers
    graph = {int(k): v for k, v in graph.items()}

    # Get all edges
    edges = []
    checked = []
    traversal = list(graph.keys())  # All nodes are considered in Kruskal

    for u in graph:
        for v, weight in graph[u]:
            # Only add each edge once (u,v where u < v)
            if u < v:
                edges.append((weight, u, v))
                checked.append([u, v])

    # Sort edges by weight
    edges.sort()

    # Initialize disjoint set
    parent = {node: node for node in graph}
    rank = {node: 0 for node in graph}

    # Find operation with path compression
    def find(node):
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    # Union operation with rank
    def union(u, v):
        root_u = find(u)
        root_v = find(v)

        if root_u != root_v:
            if rank[root_u] < rank[root_v]:
                parent[root_u] = root_v
            elif rank[root_u] > rank[root_v]:
                parent[root_v] = root_u
            else:
                parent[root_v] = root_u
                rank[root_u] += 1

    # Run Kruskal's algorithm
    mst_edges = []

    # For visualization, we want to show the edges in the order they're considered
    edge_order = []

    for weight, u, v in edges:
        if find(u) != find(v):
            union(u, v)
            mst_edges.append([u, v])
            edge_order.append([u, v])

    return {
        "traversal": traversal,
        "checked": checked,
        "selected": edge_order
    }


if __name__ == "__main__":
    app.run(debug=True, port=5005)