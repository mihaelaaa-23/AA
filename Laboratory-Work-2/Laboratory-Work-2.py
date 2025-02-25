import time
import random
import matplotlib.pyplot as plt
import heapq
from prettytable import PrettyTable

# Global variables to track metrics
operations_count = 0
swap_count = 0
recursion_depth = 0

def reset_metrics():
    global operations_count, swap_count, recursion_depth
    operations_count = 0
    swap_count = 0
    recursion_depth = 0

def measure_time_and_metrics(sort_function, arr):
    global operations_count, swap_count, recursion_depth
    reset_metrics()
    start_time = time.time()
    sorted_arr = sort_function(arr.copy())
    end_time = time.time()
    return end_time - start_time, operations_count, swap_count, recursion_depth

# Sorting Algorithms
def quicksort(arr):
    global operations_count, swap_count, recursion_depth
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    operations_count += len(arr)  # Counting comparisons
    swap_count += len(left) + len(right)  # Counting swaps (approximation)
    recursion_depth += 1  # Tracking recursion depth
    return quicksort(left) + middle + quicksort(right)

def mergesort(arr):
    global operations_count, swap_count, recursion_depth
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = mergesort(arr[:mid])
    right = mergesort(arr[mid:])
    recursion_depth += 1  # Tracking recursion depth
    return merge(left, right)

def merge(left, right):
    global operations_count, swap_count
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        operations_count += 1  # Counting comparisons
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
        swap_count += 1  # Counting swaps (approximation)
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def heapsort(arr):
    global operations_count, swap_count
    heapq.heapify(arr)
    operations_count += len(arr)  # Counting heapify operations
    sorted_arr = []
    for _ in range(len(arr)):
        sorted_arr.append(heapq.heappop(arr))
        swap_count += 1  # Counting swaps
    return sorted_arr

def bubblesort(arr):
    global operations_count, swap_count
    n = len(arr)
    for i in range(n - 1):
        swapped = False
        for j in range(n - i - 1):
            operations_count += 1  # Counting comparisons
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swap_count += 1  # Counting swaps
                swapped = True
        if not swapped:
            break
    return arr

# Generate different input data sizes and types
sizes = [100, 500, 1000, 5000, 10000]
algorithms = {"QuickSort": quicksort, "MergeSort": mergesort, "HeapSort": heapsort, "BubbleSort": bubblesort}

# Define data types
data_types = {
    "Random Integers": lambda size: [random.randint(1, 10000) for _ in range(size)],
    "Random Floats": lambda size: [random.uniform(1, 10000) for _ in range(size)],
    "Random Negative Integers": lambda size: [random.randint(-10000, -1) for _ in range(size)],
    "Random Mixed Numbers": lambda size: [random.randint(-5000, 5000) for _ in range(size)],
    "Sorted": lambda size: sorted([random.randint(1, 10000) for _ in range(size)]),
    "Reverse Sorted": lambda size: sorted([random.randint(1, 10000) for _ in range(size)], reverse=True),
    "Half Sorted": lambda size: sorted([random.randint(1, 10000) for _ in range(size // 2)]) + [random.randint(1, 10000) for _ in range(size // 2)],
}

results = {alg: {data_type: {"time": [], "operations": [], "swaps": [], "recursion_depth": []} for data_type in data_types.keys()} for alg in algorithms.keys()}

for size in sizes:
    for data_type, generator in data_types.items():
        data = generator(size)
        for alg_name, alg_func in algorithms.items():
            time_taken, operations, swaps, depth = measure_time_and_metrics(alg_func, data)
            results[alg_name][data_type]["time"].append(time_taken)
            results[alg_name][data_type]["operations"].append(operations)
            results[alg_name][data_type]["swaps"].append(swaps)
            results[alg_name][data_type]["recursion_depth"].append(depth)

# Print results in a table
for data_type in data_types.keys():
    print(f"\n=== {data_type} Data ===")
    table = PrettyTable()
    table.field_names = ["Algorithm", "Input Size", "Time (s)", "Operations", "Swaps", "Recursion Depth"]
    for alg in algorithms.keys():
        for i, size in enumerate(sizes):
            table.add_row([
                alg,
                size,
                f"{results[alg][data_type]['time'][i]:.6f}",
                results[alg][data_type]['operations'][i],
                results[alg][data_type]['swaps'][i],
                results[alg][data_type]['recursion_depth'][i]
            ])
    print(table)

# Plot the results
for data_type in data_types.keys():
    plt.figure()
    for alg in algorithms.keys():
        plt.plot(sizes, results[alg][data_type]["time"], marker='o', label=alg)
    plt.xlabel("Input Size")
    plt.ylabel("Time (seconds)")
    plt.title(f"Sorting Algorithm Performance - {data_type}")
    plt.legend()
    plt.show()