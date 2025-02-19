import time
import sys
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from decimal import Decimal, getcontext

# Increase recursion limit (optional, use with caution)
sys.setrecursionlimit(10000)

# 1) Naive recursion
def fib_naive(n):
    if n <= 1:
        return n
    return fib_naive(n - 1) + fib_naive(n - 2)


# 2) Memoization approach
def fib_memo(n, memo=None):
    if memo is None:
        memo = [-1] * (n + 1)
    if n <= 1:
        return n
    if memo[n] != -1:
        return memo[n]
    memo[n] = fib_memo(n - 1, memo) + fib_memo(n - 2, memo)
    return memo[n]


# 3) Bottom-up approach
def fib_bottom_up(n):
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]


# 4) Space-optimized approach
def fib_space_optimized(n):
    if n <= 1:
        return n
    prev2 = 0
    prev1 = 1
    for i in range(2, n + 1):
        curr = prev1 + prev2
        prev2 = prev1
        prev1 = curr
    return prev1

# 5) Matrix exponentiation
def multiply(mat1, mat2):
    x = mat1[0][0] * mat2[0][0] + mat1[0][1] * mat2[1][0]
    y = mat1[0][0] * mat2[0][1] + mat1[0][1] * mat2[1][1]
    z = mat1[1][0] * mat2[0][0] + mat1[1][1] * mat2[1][0]
    w = mat1[1][0] * mat2[0][1] + mat1[1][1] * mat2[1][1]

    mat1[0][0], mat1[0][1] = x, y
    mat1[1][0], mat1[1][1] = z, w


def matrix_power(mat, n):
    if n == 0 or n == 1:
        return
    M = [[1, 1],
         [1, 0]]
    matrix_power(mat, n // 2)
    multiply(mat, mat)
    if n % 2 != 0:
        multiply(mat, M)


def fib_matrix(n):
    if n <= 1:
        return n
    F = [[1, 1],
         [1, 0]]
    matrix_power(F, n - 1)
    return F[0][0]


# 6) Binet's formula
# Increase precision to avoid floating-point overflows
getcontext().prec = 1000  # Adjust precision as needed

def fib_binet(n):
    phi = Decimal((1 + Decimal(5).sqrt()) / 2)  # golden ratio
    psi = Decimal((1 - Decimal(5).sqrt()) / 2)  # = -1/phi
    return int(round((phi ** n - psi ** n) / Decimal(5).sqrt()))

# Function to measure and return execution time of a method
def measure_time(func, n):
    start = time.time()
    func(n)
    end = time.time()
    return end - start


def plot_graphs(ns, times_naive, times_memo, times_bottom, times_space, times_matrix, times_binet, title):
    plt.figure(figsize=(12, 8))
    plt.plot(ns, times_naive, label="Naive Recursion", marker='o')
    plt.plot(ns, times_memo, label="Memoization", marker='o')
    plt.plot(ns, times_bottom, label="Bottom-Up", marker='o')
    plt.plot(ns, times_space, label="Space-Optimized", marker='o')
    plt.plot(ns, times_matrix, label="Matrix Exponentiation", marker='o')
    plt.plot(ns, times_binet, label="Binet's Formula", marker='o')

    plt.xlabel("n")
    plt.ylabel("Execution Time (seconds)")
    plt.title(title)
    plt.yscale("log")  # Use logarithmic scale for better visualization

    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    # 1) Table: All methods (including naive) up to n=30
    ns = range(31)
    table = PrettyTable([
        "n",
        "Naive (s)",
        "Memo (s)",
        "Bottom-up (s)",
        "Space-Opt (s)",
        "Matrix (s)",
        "Binet (s)"
    ])

    for n in ns:
        t_naive = measure_time(fib_naive, n)
        t_memo = measure_time(fib_memo, n)
        t_bottom = measure_time(fib_bottom_up, n)
        t_space = measure_time(fib_space_optimized, n)
        t_matrix = measure_time(fib_matrix, n)
        t_binet = measure_time(fib_binet, n)

        table.add_row([
            n,
            f"{t_naive:.6e}",
            f"{t_memo:.6e}",
            f"{t_bottom:.6e}",
            f"{t_space:.6e}",
            f"{t_matrix:.6e}",
            f"{t_binet:.6e}"
        ])

    print("=== Fibonacci Timing Comparison (n up to 30) ===")
    print(table)

    # 2) Table: Other methods (0..500 in steps of 50)
    ns_others = range(0, 501, 50)
    table_others = PrettyTable(["n",
                                "Memo (sec)",
                                "Bottom-up (sec)",
                                "Space-opt (sec)",
                                "Matrix (sec)",
                                "Binet (sec)"])

    for n in ns_others:
        time_memo = measure_time(fib_memo, n)
        time_bottom = measure_time(fib_bottom_up, n)
        time_space = measure_time(fib_space_optimized, n)
        time_matrix = measure_time(fib_matrix, n)
        time_binet = measure_time(fib_binet, n)

        table_others.add_row([
            n,
            f"{time_memo:.6e}",
            f"{time_bottom:.6e}",
            f"{time_space:.6e}",
            f"{time_matrix:.6e}",
            f"{time_binet:.6e}"
        ])

    print("\n=== Other Methods Times (n up to 500) ===")
    print(table_others)

    # 3) Table: Other methods for large n
    large_ns = [501, 631, 794, 1000, 1259, 1585, 1995, 2512,
                3162, 3981, 5012, 6310, 7943, 10000, 12589, 15849,
                19950, 25120, 31620, 39810, 50120, 63100, 79430, 100000,
                125890, 158490, 199500]

    table_big_others = PrettyTable(["n",
                                    "Memo (sec)",
                                    "Bottom-up (sec)",
                                    "Space-opt (sec)",
                                    "Matrix (sec)",
                                    "Binet (sec)"])

    for n in large_ns:
        # Use iterative methods for large n to avoid recursion depth issues
        time_bottom = measure_time(fib_bottom_up, n)
        time_space = measure_time(fib_space_optimized, n)
        time_matrix = measure_time(fib_matrix, n)
        time_binet = measure_time(fib_binet, n)

        table_big_others.add_row([
            n,
            "N/A",  # Skip memoization for large n
            f"{time_bottom:.6e}",
            f"{time_space:.6e}",
            f"{time_matrix:.6e}",
            f"{time_binet:.6e}"
        ])

    print("\n=== Other Methods Times for Larger n ===")
    print(table_big_others)

    # 4) Print Time Complexity Summary
    complexity_table = PrettyTable(["Method", "Time Complexity"])
    complexity_table.add_row(["Naive recursion", "O(2^n)"])
    complexity_table.add_row(["Memoization", "O(n)"])
    complexity_table.add_row(["Bottom-up", "O(n)"])
    complexity_table.add_row(["Space-optimized", "O(n)"])
    complexity_table.add_row(["Matrix exponentiation", "O(log n)"])
    complexity_table.add_row(["Binet's formula", "O(1)"])

    print("\n=== Time Complexity Summary ===")
    print(complexity_table)

    # 5) Plot Graphs
    # Data for n up to 30
    ns = range(31)
    times_naive = [measure_time(fib_naive, n) for n in ns]
    times_memo = [measure_time(fib_memo, n) for n in ns]
    times_bottom = [measure_time(fib_bottom_up, n) for n in ns]
    times_space = [measure_time(fib_space_optimized, n) for n in ns]
    times_matrix = [measure_time(fib_matrix, n) for n in ns]
    times_binet = [measure_time(fib_binet, n) for n in ns]

    plot_graphs(ns, times_naive, times_memo, times_bottom, times_space, times_matrix, times_binet,
                "Fibonacci Methods Execution Time (n up to 30)")

    # Data for n up to 500 (steps of 50)
    ns_others = range(0, 501, 50)
    times_memo_others = [measure_time(fib_memo, n) for n in ns_others]
    times_bottom_others = [measure_time(fib_bottom_up, n) for n in ns_others]
    times_space_others = [measure_time(fib_space_optimized, n) for n in ns_others]
    times_matrix_others = [measure_time(fib_matrix, n) for n in ns_others]
    times_binet_others = [measure_time(fib_binet, n) for n in ns_others]

    plot_graphs(ns_others, [0] * len(ns_others), times_memo_others, times_bottom_others, times_space_others,
                times_matrix_others, times_binet_others, "Fibonacci Methods Execution Time (n up to 500)")

    # Data for large n
    large_ns = [501, 631, 794, 1000, 1259, 1585, 1995, 2512, 3162, 3981, 5012, 6310, 7943, 10000, 12589, 15849, 19950,
                25120, 31620, 39810, 50120, 63100, 79430, 100000, 125890, 158490, 199500]
    times_bottom_large = [measure_time(fib_bottom_up, n) for n in large_ns]
    times_space_large = [measure_time(fib_space_optimized, n) for n in large_ns]
    times_matrix_large = [measure_time(fib_matrix, n) for n in large_ns]
    times_binet_large = [measure_time(fib_binet, n) for n in large_ns]

    plot_graphs(large_ns, [0] * len(large_ns), [0] * len(large_ns), times_bottom_large, times_space_large,
                times_matrix_large, times_binet_large, "Fibonacci Methods Execution Time (Large n)")


if __name__ == "__main__":
    main()