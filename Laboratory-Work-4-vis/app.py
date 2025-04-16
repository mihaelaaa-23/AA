from flask import Flask, render_template, request, jsonify
import random
import heapq
import math

app = Flask(__name__)
graph_data = {}


def generate_graph(num_nodes, graph_type):
    is_directed = "directed" in graph_type
    is_sparse = "sparse" in graph_type  # Changed to check for sparse specifically

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
            # For undirected graphs, check if edge already exists
            if not is_directed and ((i, j) in added or (j, i) in added):
                continue

            weight = random.randint(1, 10)
            adjacency[i].append([j, weight])
            added.add((i, j))

            # For undirected graphs, add the reverse edge too
            if not is_directed:
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
    algorithm = data.get("algorithm", "dijkstra")

    # Always use node 0 as start node
    start_node = 0

    if algorithm == "dijkstra":
        return jsonify(simulate_dijkstra(graph, start_node))
    elif algorithm == "floyd":
        return jsonify(simulate_floyd_warshall(graph))

    return jsonify({"error": "Invalid algorithm"}), 400


@app.route("/api/reset", methods=["POST"])
def reset():
    return jsonify({"graph": graph_data.get("graph", {})})


def simulate_dijkstra(graph, start_node=0):
    # Convert string keys to integers
    graph = {int(k): v for k, v in graph.items()}

    # Initialize data structures
    distances = {node: float('infinity') for node in graph}
    distances[start_node] = 0
    priority_queue = [(0, start_node)]
    previous = {node: None for node in graph}
    visited = set()

    # For visualization
    traversal = [start_node]
    checked = []
    selected = []

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_node in visited:
            continue

        visited.add(current_node)

        for neighbor, weight in graph[current_node]:
            if neighbor in visited:
                continue

            checked.append([current_node, neighbor])

            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))
                traversal.append(neighbor)

    # Reconstruct the selected edges
    for node in graph:
        if previous[node] is not None:
            selected.append([previous[node], node])

    return {
        "traversal": traversal,
        "checked": checked,
        "selected": selected
    }


def simulate_floyd_warshall(graph):
    # Convert string keys to integers
    graph = {int(k): v for k, v in graph.items()}
    nodes = list(graph.keys())
    n = len(nodes)

    # Initialize distance matrix
    dist = [[float('infinity') for _ in range(n)] for _ in range(n)]
    next_node = [[None for _ in range(n)] for _ in range(n)]

    # Set diagonal to 0
    for i in range(n):
        dist[i][i] = 0

    # Initialize with direct edges
    checked = []
    for u in nodes:
        for v, weight in graph[u]:
            dist[u][v] = weight
            next_node[u][v] = v
            checked.append([u, v])

    # Floyd-Warshall algorithm
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    next_node[i][j] = next_node[i][k]

    # Extract paths for visualization
    selected = []
    for i in range(n):
        for j in range(n):
            if i != j and next_node[i][j] is not None:
                u = i
                while u != j:
                    v = next_node[u][j]
                    pair = [u, v]
                    if pair not in selected:
                        selected.append(pair)
                    u = v

    traversal = nodes  # All nodes are considered in Floyd-Warshall

    return {
        "traversal": traversal,
        "checked": checked,
        "selected": selected
    }


if __name__ == "__main__":
    app.run(debug=True, port=5050)