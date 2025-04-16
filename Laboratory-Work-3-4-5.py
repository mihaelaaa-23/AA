import time
import random
import matplotlib.pyplot as plt
from collections import deque
from typing import List, Dict, Tuple, Union
from prettytable import PrettyTable
import heapq
import sys


class GraphAnalyzer:
    def __init__(self):
        self.metrics = {
            'execution_time': [],
            'memory_usage': [],
            'vertices_visited': [],
            'edges_explored': []
        }

    def dfs(self, adj: Union[List[List[int]], List[Dict[int, int]]], start: int) -> Tuple[List[int], Dict[str, int]]:
        visited = [False] * len(adj)
        stack = [start]
        traversal_order = []
        metrics = {'vertices_visited': 0, 'edges_explored': 0, 'memory_used': 0}
        max_stack_size = 0

        while stack:
            max_stack_size = max(max_stack_size, len(stack))
            vertex = stack.pop()
            if not visited[vertex]:
                visited[vertex] = True
                traversal_order.append(vertex)
                metrics['vertices_visited'] += 1

                neighbors = adj[vertex].keys() if isinstance(adj[vertex], dict) else adj[vertex]
                for neighbor in reversed(list(neighbors)):
                    metrics['edges_explored'] += 1
                    if not visited[neighbor]:
                        stack.append(neighbor)

        metrics['memory_used'] = max_stack_size
        return traversal_order, metrics

    def bfs(self, adj: Union[List[List[int]], List[Dict[int, int]]], start: int) -> Tuple[List[int], Dict[str, int]]:
        visited = [False] * len(adj)
        queue = deque([start])
        visited[start] = True
        traversal_order = []
        metrics = {'vertices_visited': 0, 'edges_explored': 0, 'memory_used': 0}
        max_queue_size = 0

        while queue:
            max_queue_size = max(max_queue_size, len(queue))
            vertex = queue.popleft()
            traversal_order.append(vertex)
            metrics['vertices_visited'] += 1

            neighbors = adj[vertex].keys() if isinstance(adj[vertex], dict) else adj[vertex]
            for neighbor in neighbors:
                metrics['edges_explored'] += 1
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)

        metrics['memory_used'] = max_queue_size
        return traversal_order, metrics

    def generate_graph(self, graph_type: str, size: int):
        adj = [[] for _ in range(size)]

        if graph_type == "finite":
            for i in range(size):
                edges = random.sample(range(size), min(3, size))
                for j in edges:
                    if j != i and j not in adj[i]:
                        adj[i].append(j)
                        adj[j].append(i)
            return adj, "Finite Graph"

        elif graph_type == "complete":
            for i in range(size):
                adj[i] = [j for j in range(size) if j != i]
            return adj, "Complete Graph"

        elif graph_type == "tree":
            for i in range(size):
                left = 2 * i + 1
                right = 2 * i + 2
                if left < size:
                    adj[i].append(left)
                    adj[left].append(i)
                if right < size:
                    adj[i].append(right)
                    adj[right].append(i)
            return adj, "Tree Graph"


        elif graph_type == "sparse":
            for i in range(size):
                for _ in range(2):  # doar 2 edges per nod = sparse
                    j = random.randint(0, size - 1)
                    if j != i and j not in adj[i]:
                        adj[i].append(j)
                        adj[j].append(i)
            return adj, "Sparse Graph"

        elif graph_type == "dense":
            for i in range(size):
                for j in range(i + 1, size):
                    if random.random() < 0.7:
                        adj[i].append(j)
                        adj[j].append(i)
            return adj, "Dense Graph"

        elif graph_type == "bipartite":
            group1 = range(0, size // 2)
            group2 = range(size // 2, size)
            for i in group1:
                for j in group2:
                    if random.random() < 0.3:
                        adj[i].append(j)
                        adj[j].append(i)
            return adj, "Bipartite Graph"

        elif graph_type == "cyclic":
            for i in range(size):
                adj[i].append((i + 1) % size)
                adj[(i + 1) % size].append(i)
            return adj, "Cyclic Graph"

        elif graph_type == "acyclic":
            adj = [[] for _ in range(size)]
            for i in range(size):
                for j in range(i + 1, size):
                    if random.random() < 0.2:
                        adj[i].append(j)  # Directed edge from i to j
            return adj, "Directed Acyclic Graph (DAG)"


        elif graph_type == "multigraph":
            for _ in range(size * 2):
                u = random.randint(0, size - 1)
                v = random.randint(0, size - 1)
                if u != v:
                    adj[u].append(v)
                    adj[v].append(u)
            return adj, "Multigraph"

        elif graph_type == "pseudograph":
            for i in range(size):
                adj[i].append(i)
                for j in range(i + 1, size):
                    if random.random() < 0.2:
                        adj[i].append(j)
                        adj[j].append(i)
            return adj, "Pseudograph"

        elif graph_type == "regular":
            degree = min(3, size - 1)

            if (size * degree) % 2 != 0:
                raise ValueError("Cannot generate k-regular graph: n*k must be even")

            adj = [[] for _ in range(size)]
            vertices = list(range(size)) * degree
            random.shuffle(vertices)
            attempts = 0
            max_attempts = 10000  # limit to avoid infinite loops

            while vertices and attempts < max_attempts:
                u = vertices.pop()
                v = vertices.pop()

                if u != v and v not in adj[u] and len(adj[u]) < degree and len(adj[v]) < degree:
                    adj[u].append(v)
                    adj[v].append(u)
                else:
                    vertices.extend([u, v])
                    random.shuffle(vertices)
                    attempts += 1

            if attempts == max_attempts:
                raise ValueError("Failed to generate a regular graph after many attempts")

            return adj, "Regular Graph"


        elif graph_type == "weighted":
            adj = [{} for _ in range(size)]
            for i in range(size):
                for j in range(i + 1, size):
                    if random.random() < 0.3:
                        weight = random.randint(1, 10)
                        adj[i][j] = weight
                        adj[j][i] = weight
            return adj, "Weighted Graph"

        elif graph_type == "directed":
            for i in range(size):
                for _ in range(2):
                    j = random.randint(0, size - 1)
                    if j != i and j not in adj[i]:
                        adj[i].append(j)
            return adj, "Directed Graph"

        else:
            raise ValueError(f"Unknown graph type: {graph_type}")

    def print_results_table(self, graph_sizes, dfs_times, bfs_times, dfs_memory, bfs_memory, graph_name):
        table = PrettyTable()
        table.title = f"Performance Comparison: {graph_name}"
        table.field_names = [
            "Graph Size", "DFS Time (s)", "BFS Time (s)", "DFS Memory", "BFS Memory", "Time Ratio", "Memory Ratio"
        ]

        for i in range(len(graph_sizes)):
            dfs_time = dfs_times[i]
            bfs_time = bfs_times[i]
            dfs_mem = dfs_memory[i]
            bfs_mem = bfs_memory[i]
            time_ratio = bfs_time / dfs_time if dfs_time != 0 else float('inf')
            mem_ratio = bfs_mem / dfs_mem if dfs_mem != 0 else float('inf')

            table.add_row([
                graph_sizes[i], f"{dfs_time:.6f}", f"{bfs_time:.6f}", dfs_mem, bfs_mem, f"{time_ratio:.2f}", f"{mem_ratio:.2f}"
            ])

        print(table)

    def analyze_algorithms(self, graph_sizes: List[int], graph_type: str):
        dfs_times, bfs_times, dfs_memory, bfs_memory = [], [], [], []

        for size in graph_sizes:
            adj, graph_name = self.generate_graph(graph_type, size)
            start_node = 0

            if isinstance(adj[0], list) and not adj[0]:
                if size > 1:
                    adj[0].append(1)
                    adj[1].append(0)
            elif isinstance(adj[0], dict) and not adj[0]:
                if size > 1:
                    adj[0][1] = 1
                    adj[1][0] = 1

            start_time = time.time()
            _, dfs_metrics = self.dfs(adj, start_node)
            dfs_time = time.time() - start_time

            start_time = time.time()
            _, bfs_metrics = self.bfs(adj, start_node)
            bfs_time = time.time() - start_time

            dfs_times.append(dfs_time)
            bfs_times.append(bfs_time)
            dfs_memory.append(dfs_metrics['memory_used'])
            bfs_memory.append(bfs_metrics['memory_used'])

        self.print_results_table(graph_sizes, dfs_times, bfs_times, dfs_memory, bfs_memory, graph_name)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(graph_sizes, dfs_times, label='DFS')
        plt.plot(graph_sizes, bfs_times, label='BFS')
        plt.xlabel('Graph Size')
        plt.ylabel('Execution Time')
        plt.title(f'Execution Time ({graph_name})')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(graph_sizes, dfs_memory, label='DFS')
        plt.plot(graph_sizes, bfs_memory, label='BFS')
        plt.xlabel('Graph Size')
        plt.ylabel('Memory Usage')
        plt.title(f'Memory Usage ({graph_name})')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def dijkstra(self, adj: List[Dict[int, int]], start: int) -> Tuple[Dict[int, int], float, int]:
        dist = {i: float('inf') for i in range(len(adj))}
        dist[start] = 0
        visited = set()
        heap = [(0, start)]

        start_time = time.time()

        while heap:
            d, u = heapq.heappop(heap)
            if u in visited:
                continue
            visited.add(u)

            for v in adj[u]:
                if dist[u] + adj[u][v] < dist[v]:
                    dist[v] = dist[u] + adj[u][v]
                    heapq.heappush(heap, (dist[v], v))

        exec_time = time.time() - start_time

        memory_used = (
                sys.getsizeof(dist) +
                sys.getsizeof(visited) +
                sys.getsizeof(heap)
        )

        return dist, exec_time, memory_used

    def floyd_warshall(self, adj: List[Dict[int, int]]) -> Tuple[List[List[float]], float, int]:
        size = len(adj)
        dist = [[float('inf')] * size for _ in range(size)]
        for i in range(size):
            dist[i][i] = 0
            for j in adj[i]:
                dist[i][j] = adj[i][j]

        start_time = time.time()

        for k in range(size):
            for i in range(size):
                for j in range(size):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]

        exec_time = time.time() - start_time
        memory_used = sys.getsizeof(dist) + sum(sys.getsizeof(row) for row in dist)

        return dist, exec_time, memory_used
    def generate_weighted_graph(self, size: int, graph_type: str) -> List[Dict[int, int]]:
        adj = [{} for _ in range(size)]

        if graph_type == "sparse":
            for i in range(size):
                connections = random.sample(range(size), min(2, size - 1))
                for j in connections:
                    if j != i and j not in adj[i]:
                        weight = random.randint(1, 10)
                        adj[i][j] = weight
                        adj[j][i] = weight

        elif graph_type == "dense":
            for i in range(size):
                for j in range(size):
                    if i != j and random.random() < 0.7:
                        weight = random.randint(1, 10)
                        adj[i][j] = weight

        return adj

    def analyze_shortest_path(self, graph_sizes: List[int], trials: int = 5) -> Dict[str, List]:
        results = {
            "size": [],
            "floyd_sparse": [], "floyd_sparse_mem": [],
            "floyd_dense": [], "floyd_dense_mem": [],
            "dijkstra_sparse": [], "dijkstra_sparse_mem": [],
            "dijkstra_dense": [], "dijkstra_dense_mem": []
        }

        for gtype in ["sparse", "dense"]:
            for size in graph_sizes:
                fw_times = []
                fw_mems = []
                dj_times = []
                dj_mems = []

                for _ in range(trials):
                    adj, _ = self.generate_weighted_graph_4(size, gtype)

                    if not fw_times:
                        _, fw_time, fw_mem = self.floyd_warshall(adj)
                        fw_times.append(fw_time)
                        fw_mems.append(fw_mem)

                    _, dj_time, dj_mem = self.dijkstra(adj, 0)
                    dj_times.append(dj_time)
                    dj_mems.append(dj_mem)

                avg_fw = sum(fw_times) / len(fw_times)
                avg_fw_mem = sum(fw_mems) // len(fw_mems)
                avg_dj = sum(dj_times) / trials
                avg_dj_mem = sum(dj_mems) // trials

                if gtype == "sparse":
                    results["floyd_sparse"].append(avg_fw)
                    results["floyd_sparse_mem"].append(avg_fw_mem)
                    results["dijkstra_sparse"].append(avg_dj)
                    results["dijkstra_sparse_mem"].append(avg_dj_mem)
                else:
                    results["floyd_dense"].append(avg_fw)
                    results["floyd_dense_mem"].append(avg_fw_mem)
                    results["dijkstra_dense"].append(avg_dj)
                    results["dijkstra_dense_mem"].append(avg_dj_mem)

                if size not in results["size"]:
                    results["size"].append(size)

        return results

    def display_shortest_path_results(self, results: Dict[str, List]):
        # --- Sparse Graphs ---
        table_sparse = PrettyTable()
        table_sparse.title = "Empirical Performance on Sparse Graphs"
        table_sparse.field_names = [
            "Graph Size",
            "Dijkstra Time (s)", "Floyd–W. Time (s)",
            "Dijkstra Mem (B)", "Floyd Mem (B)"
        ]

        for i in range(len(results["size"])):
            table_sparse.add_row([
                results["size"][i],
                f"{results['dijkstra_sparse'][i]:.6f}",
                f"{results['floyd_sparse'][i]:.6f}",
                results["dijkstra_sparse_mem"][i],
                results["floyd_sparse_mem"][i]
            ])

        print(table_sparse)

        # --- Dense Graphs ---
        table_dense = PrettyTable()
        table_dense.title = "Empirical Performance on Dense Graphs"
        table_dense.field_names = [
            "Graph Size",
            "Dijkstra Time (s)", "Floyd–W. Time (s)",
            "Dijkstra Mem (B)", "Floyd Mem (B)"
        ]

        for i in range(len(results["size"])):
            table_dense.add_row([
                results["size"][i],
                f"{results['dijkstra_dense'][i]:.6f}",
                f"{results['floyd_dense'][i]:.6f}",
                results["dijkstra_dense_mem"][i],
                results["floyd_dense_mem"][i]
            ])

        print(table_dense)

        # --- Execution Time Plot ---
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))

        ax[0].plot(results["size"], results["dijkstra_sparse"], marker='o', label="Dijkstra")
        ax[0].plot(results["size"], results["floyd_sparse"], marker='s', label="Floyd–Warshall")
        ax[0].set_title("Sparse Graphs - Execution Time")
        ax[0].set_xlabel("Graph Size")
        ax[0].set_ylabel("Time (s)")
        ax[0].legend()
        ax[0].grid(True)

        ax[1].plot(results["size"], results["dijkstra_dense"], marker='o', label="Dijkstra")
        ax[1].plot(results["size"], results["floyd_dense"], marker='s', label="Floyd–Warshall")
        ax[1].set_title("Dense Graphs - Execution Time")
        ax[1].set_xlabel("Graph Size")
        ax[1].set_ylabel("Time (s)")
        ax[1].legend()
        ax[1].grid(True)

        plt.tight_layout()
        plt.show()

        # --- Memory Usage Plot ---
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))

        ax[0].plot(results["size"], results["dijkstra_sparse_mem"], marker='o', label="Dijkstra Mem")
        ax[0].plot(results["size"], results["floyd_sparse_mem"], marker='s', label="Floyd Mem")
        ax[0].set_title("Sparse Graphs - Memory Usage")
        ax[0].set_xlabel("Graph Size")
        ax[0].set_ylabel("Memory (bytes)")
        ax[0].legend()
        ax[0].grid(True)

        ax[1].plot(results["size"], results["dijkstra_dense_mem"], marker='o', label="Dijkstra Mem")
        ax[1].plot(results["size"], results["floyd_dense_mem"], marker='s', label="Floyd Mem")
        ax[1].set_title("Dense Graphs - Memory Usage")
        ax[1].set_xlabel("Graph Size")
        ax[1].set_ylabel("Memory (bytes)")
        ax[1].legend()
        ax[1].grid(True)

        plt.tight_layout()
        plt.show()

    def generate_weighted_graph_4(self, size: int, graph_type: str) -> List[Dict[int, int]]:
        adj = [{} for _ in range(size)]

        if graph_type == "sparse":
            for i in range(size):
                connections = random.sample(range(size), min(2, size - 1))
                for j in connections:
                    if j != i and j not in adj[i]:
                        weight = random.randint(1, 10)
                        adj[i][j] = weight
                        adj[j][i] = weight

        elif graph_type == "dense":
            for i in range(size):
                for j in range(size):
                    if i != j and random.random() < 0.7:
                        weight = random.randint(1, 10)
                        adj[i][j] = weight

        return adj, graph_type.capitalize() + " Graph"

    def prim(self, adj: List[Dict[int, int]], start: int) -> Tuple[List[Tuple[int, int]], float, int]:
        mst = []
        total_weight = 0
        visited = [False] * len(adj)
        min_heap = [(0, start)]
        start_time = time.time()

        while min_heap:
            weight, u = heapq.heappop(min_heap)
            if visited[u]:
                continue

            visited[u] = True
            total_weight += weight

            if weight > 0:
                mst.append((u, weight))

            for v, edge_weight in adj[u].items():
                if not visited[v]:
                    heapq.heappush(min_heap, (edge_weight, v))

        end_time = time.time()
        memory_used = sys.getsizeof(mst) + sys.getsizeof(visited) + sys.getsizeof(min_heap)
        return mst, end_time - start_time, memory_used

    def kruskal(self, adj: List[Dict[int, int]]) -> Tuple[List[Tuple[int, int]], float, int]:
        edges = []
        for u in range(len(adj)):
            for v, weight in adj[u].items():
                if u < v:
                    edges.append((weight, u, v))

        edges.sort()

        parent = list(range(len(adj)))
        rank = [0] * len(adj)

        def find(u):
            if parent[u] != u:
                parent[u] = find(parent[u])
            return parent[u]

        def union(u, v):
            root_u = find(u)
            root_v = find(v)
            if root_u != root_v:
                if rank[root_u] > rank[root_v]:
                    parent[root_v] = root_u
                elif rank[root_u] < rank[root_v]:
                    parent[root_u] = root_v
                else:
                    parent[root_v] = root_u
                    rank[root_u] += 1

        mst = []
        total_weight = 0
        start_time = time.time()

        for weight, u, v in edges:
            if find(u) != find(v):
                union(u, v)
                mst.append((u, v, weight))
                total_weight += weight

        end_time = time.time()
        memory_used = sys.getsizeof(mst) + sys.getsizeof(parent) + sys.getsizeof(rank) + sys.getsizeof(edges)
        return mst, end_time - start_time, memory_used

    def print_results_table(self, graph_sizes, prim_times, kruskal_times, prim_memory, kruskal_memory, graph_name):
        table = PrettyTable()
        table.title = f"Performance Comparison: {graph_name}"
        table.field_names = [
            "Graph Size", "Prim’s Time (s)", "Kruskal’s Time (s)", "Prim’s Memory", "Kruskal’s Memory", "Time Ratio",
            "Memory Ratio"
        ]

        for i in range(len(graph_sizes)):
            prim_time = prim_times[i]
            kruskal_time = kruskal_times[i]
            prim_mem = prim_memory[i]
            kruskal_mem = kruskal_memory[i]

            # Calculate the time and memory ratios
            time_ratio = kruskal_time / prim_time if prim_time != 0 else float('inf')
            mem_ratio = kruskal_mem / prim_mem if prim_mem != 0 else float('inf')

            table.add_row([
                graph_sizes[i], f"{prim_time:.6f}", f"{kruskal_time:.6f}", prim_mem, kruskal_mem, f"{time_ratio:.2f}",
                f"{mem_ratio:.2f}"
            ])

        print(table)

    def plot_comparison(self, graph_sizes, prim_times, kruskal_times, prim_memory, kruskal_memory, graph_name):
        # Plot execution time comparison
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(graph_sizes, prim_times, label="Prim’s Algorithm", marker='o')
        plt.plot(graph_sizes, kruskal_times, label="Kruskal’s Algorithm", marker='s')
        plt.xlabel("Graph Size")
        plt.ylabel("Execution Time (s)")
        # plt.yscale("log")  # Log scale for the Y-axis
        # plt.xscale("log")  # Log scale for the X-axis (optional)
        plt.title(f"Execution Time Comparison ({graph_name})")
        plt.legend()
        plt.grid(True)

        # Plot memory usage comparison
        plt.subplot(1, 2, 2)
        plt.plot(graph_sizes, prim_memory, label="Prim’s Algorithm Memory", marker='o')
        plt.plot(graph_sizes, kruskal_memory, label="Kruskal’s Algorithm Memory", marker='s')
        plt.xlabel("Graph Size")
        plt.ylabel("Memory Usage (bytes)")
        # plt.yscale("log")  # Log scale for memory usage
        # plt.xscale("log")  # Log scale for graph size
        plt.title(f"Memory Usage Comparison ({graph_name})")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def analyze_greedy_algorithms(self, graph_sizes: List[int], graph_type: str, trials: int = 5):
        prim_times, kruskal_times = [], []
        prim_memory, kruskal_memory = [], []

        for size in graph_sizes:
            adj, graph_name = self.generate_weighted_graph_4(size, graph_type)

            prim_time_avg = 0
            kruskal_time_avg = 0
            prim_mem_avg = 0
            kruskal_mem_avg = 0
            for _ in range(trials):
                start_node = 0
                _, prim_time, prim_mem = self.prim(adj, start_node)
                prim_time_avg += prim_time
                prim_mem_avg += prim_mem

                _, kruskal_time, kruskal_mem = self.kruskal(adj)
                kruskal_time_avg += kruskal_time
                kruskal_mem_avg += kruskal_mem

            prim_times.append(prim_time_avg / trials)
            kruskal_times.append(kruskal_time_avg / trials)
            prim_memory.append(prim_mem_avg / trials)
            kruskal_memory.append(kruskal_mem_avg / trials)

        self.print_results_table(graph_sizes, prim_times, kruskal_times, prim_memory, kruskal_memory, graph_name)
        self.plot_comparison(graph_sizes, prim_times, kruskal_times, prim_memory, kruskal_memory, graph_name)

if __name__ == "__main__":
    analyzer = GraphAnalyzer()

    graph_types = [
        "finite", "complete", "tree", "sparse", "dense", "bipartite", "cyclic",
        "acyclic", "multigraph", "pseudograph", "regular", "weighted", "directed"
    ]

    graph_sizes = [10, 50, 100, 200, 500, 800]

    # lab 3
    for gtype in graph_types:
        print(f"\nAnalyzing {gtype} graph...")
        analyzer.analyze_algorithms(graph_sizes, gtype)

    # lab 4
    # graph_sizes = [10, 50, 100, 200]
    results = analyzer.analyze_shortest_path(graph_sizes, trials=5)
    analyzer.display_shortest_path_results(results)

    # lab 5
    for gtype in ["sparse", "dense"]:
        print(f"\nAnalyzing {gtype} graph...")
        analyzer.analyze_greedy_algorithms(graph_sizes, gtype)