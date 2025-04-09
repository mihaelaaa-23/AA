import time
import random
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import heapq


class ShortestPathAnalyzer:
    def __init__(self):
        pass

    def generate_weighted_graph(self, graph_type: str, size: int):
        adj = {i: {} for i in range(size)}

        if graph_type == "sparse":
            edges = size * 2  # ~2 edges per node
        elif graph_type == "dense":
            edges = (size * (size - 1)) // 2  # Complete graph
        else:
            raise ValueError("Unknown graph type")

        edge_set = set()
        while len(edge_set) < edges:
            u = random.randint(0, size - 1)
            v = random.randint(0, size - 1)
            if u != v and (u, v) not in edge_set and (v, u) not in edge_set:
                weight = random.randint(1, 10)
                adj[u][v] = weight
                adj[v][u] = weight  # Undirected
                edge_set.add((u, v))

        return adj

    def dijkstra(self, adj, start):
        distances = {node: float('inf') for node in adj}
        distances[start] = 0
        pq = [(0, start)]
        relaxations = 0

        while pq:
            dist_u, u = heapq.heappop(pq)
            if dist_u > distances[u]:
                continue

            for v in adj[u]:
                if distances[v] > distances[u] + adj[u][v]:
                    distances[v] = distances[u] + adj[u][v]
                    heapq.heappush(pq, (distances[v], v))
                    relaxations += 1

        return distances, relaxations

    def floyd_warshall(self, adj):
        nodes = list(adj.keys())
        dist = {u: {v: float('inf') for v in nodes} for u in nodes}
        relaxations = 0

        for u in nodes:
            dist[u][u] = 0
            for v in adj[u]:
                dist[u][v] = adj[u][v]

        for k in nodes:
            for i in nodes:
                for j in nodes:
                    if dist[i][j] > dist[i][k] + dist[k][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
                        relaxations += 1

        return dist, relaxations

    def analyze(self, sizes, graph_type):
        dijkstra_times = []
        floyd_times = []
        dijkstra_relax = []
        floyd_relax = []

        for size in sizes:
            adj = self.generate_weighted_graph(graph_type, size)
            start = 0

            # Dijkstra
            start_time = time.time()
            _, relax1 = self.dijkstra(adj, start)
            d_time = time.time() - start_time

            # Floyd–Warshall
            start_time = time.time()
            _, relax2 = self.floyd_warshall(adj)
            f_time = time.time() - start_time

            dijkstra_times.append(d_time)
            floyd_times.append(f_time)
            dijkstra_relax.append(relax1)
            floyd_relax.append(relax2)

            print(f"{graph_type.title()} Graph - Nodes: {size} | Dijkstra: {d_time:.4f}s | Floyd: {f_time:.4f}s")

        self.show_table(sizes, dijkstra_times, floyd_times, dijkstra_relax, floyd_relax, graph_type)
        self.plot_graph(sizes, dijkstra_times, floyd_times, graph_type)

    def show_table(self, sizes, d_times, f_times, d_relax, f_relax, gtype):
        table = PrettyTable()
        table.title = f"Performance Comparison on {gtype.title()} Graphs"
        table.field_names = ["Nodes", "Dijkstra Time", "Floyd Time", "Dijkstra Relax", "Floyd Relax"]

        for i in range(len(sizes)):
            table.add_row([
                sizes[i],
                f"{d_times[i]:.6f}",
                f"{f_times[i]:.6f}",
                d_relax[i],
                f_relax[i]
            ])
        print(table)

    def plot_graph(self, sizes, d_times, f_times, gtype):
        plt.figure(figsize=(8, 5))
        plt.plot(sizes, d_times, marker='o', label='Dijkstra')
        plt.plot(sizes, f_times, marker='s', label='Floyd-Warshall')
        plt.xlabel("Number of Nodes")
        plt.ylabel("Execution Time (seconds)")
        plt.title(f"Performance on {gtype.title()} Graphs")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    analyzer = ShortestPathAnalyzer()

    graph_sizes = [50, 100, 200, 300, 400, 500, 600, 800, 1000]
    for graph_type in ['sparse', 'dense']:
        print(f"\nAnalyzing {graph_type.upper()} graphs...")
        analyzer.analyze(graph_sizes, graph_type)
