import time
import random
import matplotlib.pyplot as plt
from collections import deque
from typing import List, Dict, Tuple
from prettytable import PrettyTable


class GraphAnalyzer:
    def __init__(self):
        self.metrics = {
            'execution_time': [],
            'memory_usage': [],
            'vertices_visited': [],
            'edges_explored': []
        }
        self.graph_types = []

    def dfs(self, adj: List[List[int]], start: int) -> Tuple[List[int], Dict[str, int]]:
        visited = [False] * len(adj)
        stack = [start]
        traversal_order = []
        metrics = {
            'vertices_visited': 0,
            'edges_explored': 0,
            'memory_used': 0
        }

        max_stack_size = 0

        while stack:
            max_stack_size = max(max_stack_size, len(stack))
            vertex = stack.pop()
            if not visited[vertex]:
                visited[vertex] = True
                traversal_order.append(vertex)
                metrics['vertices_visited'] += 1

                # Push adjacent vertices in reverse order to maintain left-to-right traversal
                for neighbor in reversed(adj[vertex]):
                    metrics['edges_explored'] += 1
                    if not visited[neighbor]:
                        stack.append(neighbor)

        metrics['memory_used'] = max_stack_size
        return traversal_order, metrics

    def bfs(self, adj: List[List[int]], start: int) -> Tuple[List[int], Dict[str, int]]:
        visited = [False] * len(adj)
        queue = deque([start])
        visited[start] = True
        traversal_order = []
        metrics = {
            'vertices_visited': 0,
            'edges_explored': 0,
            'memory_used': 0
        }

        max_queue_size = 0

        while queue:
            max_queue_size = max(max_queue_size, len(queue))
            vertex = queue.popleft()
            traversal_order.append(vertex)
            metrics['vertices_visited'] += 1

            for neighbor in adj[vertex]:
                metrics['edges_explored'] += 1
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)

        metrics['memory_used'] = max_queue_size
        return traversal_order, metrics

    def generate_graph(self, graph_type: str, size: int) -> Tuple[List[List[int]], str]:
        adj = [[] for _ in range(size)]

        if graph_type == "finite":
            # Random finite graph with average degree 3
            for i in range(size):
                edges = random.sample(range(size), min(3, size))
                for j in edges:
                    if j != i and j not in adj[i]:
                        adj[i].append(j)
                        adj[j].append(i)
            return adj, "Finite Graph"

        elif graph_type == "complete":
            # Complete graph where every node is connected to every other node
            for i in range(size):
                adj[i] = [j for j in range(size) if j != i]
            return adj, "Complete Graph"

        elif graph_type == "tree":
            # Binary tree structure
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
            # Sparse graph with only a few edges
            for i in range(min(size, 10)):
                for j in range(i + 1, min(size, i + 10)):
                    if j < size:
                        adj[i].append(j)
                        adj[j].append(i)
            return adj, "Sparse Graph"

        elif graph_type == "dense":
            # Dense graph with many edges
            for i in range(size):
                for j in range(i + 1, size):
                    if random.random() < 0.7:  # 70% chance of connection
                        adj[i].append(j)
                        adj[j].append(i)
            return adj, "Dense Graph"

        elif graph_type == "bipartite":
            # Bipartite graph
            group1 = range(0, size // 2)
            group2 = range(size // 2, size)
            for i in group1:
                for j in group2:
                    if random.random() < 0.3:  # 30% chance of connection
                        adj[i].append(j)
                        adj[j].append(i)
            return adj, "Bipartite Graph"

        elif graph_type == "cyclic":
            # Cyclic graph
            for i in range(size):
                adj[i].append((i + 1) % size)
                adj[(i + 1) % size].append(i)
            return adj, "Cyclic Graph"

        else:
            raise ValueError("Unknown graph type")

    def print_results_table(self, graph_sizes: List[int], dfs_times: List[float], bfs_times: List[float],
                            dfs_memory: List[int], bfs_memory: List[int], graph_name: str):
        table = PrettyTable()
        table.title = f"Performance Comparison: {graph_name}"
        table.field_names = [
            "Graph Size",
            "DFS Time (s)",
            "BFS Time (s)",
            "DFS Memory",
            "BFS Memory",
            "Time Ratio (BFS/DFS)",
            "Memory Ratio (BFS/DFS)"
        ]

        for i in range(len(graph_sizes)):
            size = graph_sizes[i]
            dfs_time = dfs_times[i]
            bfs_time = bfs_times[i]
            dfs_mem = dfs_memory[i]
            bfs_mem = bfs_memory[i]

            time_ratio = bfs_time / dfs_time if dfs_time != 0 else float('inf')
            mem_ratio = bfs_mem / dfs_mem if dfs_mem != 0 else float('inf')

            table.add_row([
                size,
                f"{dfs_time:.6f}",
                f"{bfs_time:.6f}",
                dfs_mem,
                bfs_mem,
                f"{time_ratio:.2f}",
                f"{mem_ratio:.2f}"
            ])

        print(table)

    def analyze_algorithms(self, graph_sizes: List[int], graph_type: str):
        dfs_times = []
        bfs_times = []
        dfs_memory = []
        bfs_memory = []

        for size in graph_sizes:
            adj, graph_name = self.generate_graph(graph_type, size)
            start_node = 0

            # Measure DFS
            start_time = time.time()
            _, dfs_metrics = self.dfs(adj, start_node)
            dfs_time = time.time() - start_time

            # Measure BFS
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
        plt.xlabel('Graph Size (vertices)')
        plt.ylabel('Execution Time (seconds)')
        plt.title(f'Execution Time Comparison\n({graph_name})')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(graph_sizes, dfs_memory, label='DFS')
        plt.plot(graph_sizes, bfs_memory, label='BFS')
        plt.xlabel('Graph Size (vertices)')
        plt.ylabel('Memory Usage (max queue/stack size)')
        plt.title(f'Memory Usage Comparison\n({graph_name})')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    analyzer = GraphAnalyzer()

    graph_types = ["finite", "complete", "tree", "sparse", "dense", "bipartite", "cyclic"]
    graph_sizes = [10, 50, 100, 200, 500, 10000]

    for gtype in graph_types:
        try:
            print(f"\nAnalyzing {gtype} graphs...")
            analyzer.analyze_algorithms(graph_sizes, gtype)
        except ValueError as e:
            print(f"Skipping {gtype}: {str(e)}")