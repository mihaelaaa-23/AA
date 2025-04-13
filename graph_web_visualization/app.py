from flask import Flask, render_template, request, jsonify
import random

app = Flask(__name__)

def generate_graph(graph_type, size):
    graph = {i: [] for i in range(size)}
    if graph_type == "finite":
        for i in range(size - 1):
            graph[i].append(i + 1)
            graph[i + 1].append(i)
    elif graph_type == "infinite":
        for i in range(size):
            if i + 1 < size:
                graph[i].append(i + 1)
    elif graph_type == "trivial":
        return {0: []}
    elif graph_type == "null":
        return {i: [] for i in range(size)}
    elif graph_type == "complete":
        for i in range(size):
            for j in range(size):
                if i != j:
                    graph[i].append(j)
    elif graph_type == "tree":
        graph = {i: [] for i in range(size)}
        for i in range(1, size):
            parent = (i - 1) // 2  # părintele în arbore binar complet
            graph[i].append(parent)
            graph[parent].append(i)
    elif graph_type == "sparse":
        for i in range(size):
            for _ in range(2):
                j = random.randint(0, size - 1)
                if j != i and j not in graph[i]:
                    graph[i].append(j)
                    graph[j].append(i)
    elif graph_type == "dense":
        for i in range(size):
            for j in range(size):
                if i != j and random.random() < 0.7:
                    if j not in graph[i]:
                        graph[i].append(j)
    elif graph_type == "bipartite":
        half = size // 2
        for i in range(half):
            for j in range(half, size):
                if random.random() < 0.5:
                    graph[i].append(j)
                    graph[j].append(i)
    elif graph_type == "cyclic":
        for i in range(size):
            graph[i].append((i + 1) % size)
            graph[(i + 1) % size].append(i)
    elif graph_type == "acyclic":
        for i in range(size):
            for j in range(i + 1, size):
                if random.random() < 0.3:
                    graph[i].append(j)  # doar i → j, deci fără cicluri
    elif graph_type == "regular":
        degree = min(3, size - 1)
        nodes = list(graph.keys()) * degree
        random.shuffle(nodes)
        attempts = 0
        while nodes and attempts < 10000:
            u = nodes.pop()
            v = nodes.pop()
            if v not in graph[u] and u != v:
                graph[u].append(v)
                graph[v].append(u)
            else:
                nodes.extend([u, v])
                random.shuffle(nodes)
                attempts += 1
    elif graph_type == "weighted":
        graph = {}
        for i in range(size):
            graph[i] = []
        for i in range(size):
            for j in range(i + 1, size):
                if random.random() < 0.3:
                    weight = random.randint(1, 10)
                    graph[i].append((j, weight))
                    graph[j].append((i, weight))
    elif graph_type == "directed":
        for i in range(size):
            for _ in range(2):
                j = random.randint(0, size - 1)
                if j != i and j not in graph[i]:
                    graph[i].append(j)
    elif graph_type == "multigraph":
        for i in range(size):
            for _ in range(2):  # 2 muchii posibile per nod
                j = random.randint(0, size - 1)
                if j != i:
                    graph[i].append(j)
                    graph[j].append(i)
                    graph[i].append(j)  # a doua muchie identică
                    graph[j].append(i)
    elif graph_type == "pseudograph":
        for i in range(size):
            if random.random() < 0.3:
                graph[i].append(i)  # self-loop
            for _ in range(2):
                j = random.randint(0, size - 1)
                graph[i].append(j)
    return graph


def traverse(graph, start, algorithm):
    visited = []
    used_edges = []
    checked_edges = []
    if algorithm == "DFS":
        stack = [(start, None)]
        seen = set()
        while stack:
            node, from_node = stack.pop()
            if node not in seen:
                visited.append(node)
                if from_node is not None:
                    used_edges.append((from_node, node))
                seen.add(node)
                neighbors = graph[node]
                for neighbor in reversed(neighbors):
                    n = neighbor[0] if isinstance(neighbor, tuple) else neighbor
                    checked_edges.append((node, n))
                    if n not in seen:
                        stack.append((n, node))
    else:  # BFS
        from collections import deque
        queue = deque([(start, None)])
        seen = set([start])
        while queue:
            node, from_node = queue.popleft()
            visited.append(node)
            if from_node is not None:
                used_edges.append((from_node, node))
            neighbors = graph[node]
            for neighbor in neighbors:
                n = neighbor[0] if isinstance(neighbor, tuple) else neighbor
                checked_edges.append((node, n))
                if n not in seen:
                    seen.add(n)
                    queue.append((n, node))
    return visited, used_edges, checked_edges


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/generate", methods=["POST"])
def api_generate():
    data = request.json
    graph_type = data.get("graphType", "finite")
    algorithm = data.get("algorithm", "DFS")
    size = int(data.get("size", 10))

    graph = generate_graph(graph_type, size)
    traversal, used_edges, checked_edges = traverse(graph, 0, algorithm)
    return jsonify({
        "graph": graph,
        "traversal": traversal,
        "edges": used_edges,
        "checked": checked_edges
    })

@app.route("/api/traverse", methods=["POST"])
def api_traverse():
    data = request.json
    graph = data["graph"]
    algorithm = data.get("algorithm", "DFS")

    # convert keys to int, and tuples to lists (for safety)
    graph = {int(k): [tuple(n) if isinstance(n, list) and len(n) == 2 else n for n in v] for k, v in graph.items()}

    traversal, used_edges, checked_edges = traverse(graph, 0, algorithm)

    return jsonify({
        "traversal": traversal,
        "edges": used_edges,
        "checked": checked_edges
    })

if __name__ == "__main__":
    app.run(debug=True)