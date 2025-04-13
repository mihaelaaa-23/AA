import time
import random
import matplotlib.pyplot as plt
from collections import deque
from typing import List, Dict, Tuple, Union
from prettytable import PrettyTable


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
                for _ in range(2):  # doar 2 muchii per nod = sparse
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


if __name__ == "__main__":
    analyzer = GraphAnalyzer()

    graph_types = [
        "finite", "complete", "tree", "sparse", "dense", "bipartite", "cyclic",
        "acyclic", "multigraph", "pseudograph", "regular", "weighted", "directed"
    ]

    graph_sizes = [10, 50, 100, 200, 500, 800]

    for gtype in graph_types:
        print(f"\nAnalyzing {gtype} graph...")
        analyzer.analyze_algorithms(graph_sizes, gtype)
