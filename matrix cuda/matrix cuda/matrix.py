import numpy as np
from numba import cuda
import time
import matplotlib.pyplot as plt


# GPU перемножение матриц
@cuda.jit
def matrix_multiply_gpu(A, B, result):
    row, col = cuda.grid(2)

    if row < result.shape[0] and col < result.shape[1]:
        tmp = 0.0
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        result[row, col] = tmp


# Функция для генерации случайных матриц
def generate_matrices(size):
    return np.random.rand(size, size).astype(np.float32)


# Функция для измерения времени выполнения перемножения матриц
def measure_time(multiply_func, *args):
    start_time = time.time()
    multiply_func(*args)
    end_time = time.time()
    return end_time - start_time


# Основная функция
def main():
    sizes = [100, 200, 400, 800, 1000]
    cpu_times = []
    gpu_times = []

    for size in sizes:
        # Генерация случайных матриц
        A = generate_matrices(size)
        B = generate_matrices(size)
        result_cpu = np.zeros((size, size), dtype=np.float32)
        result_gpu = np.zeros((size, size), dtype=np.float32)

        # Время работы на CPU
        cpu_time = measure_time(np.dot, A, B)
        cpu_times.append(cpu_time)

        # Время работы на GPU
        threads_per_block = (16, 16)
        blocks_per_grid_x = (size + threads_per_block[0] - 1) // threads_per_block[0]
        blocks_per_grid_y = (size + threads_per_block[1] - 1) // threads_per_block[1]
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        d_A = cuda.to_device(A)
        d_B = cuda.to_device(B)
        d_result = cuda.device_array_like(result_gpu)

        gpu_time = measure_time(matrix_multiply_gpu[blocks_per_grid, threads_per_block], d_A, d_B, d_result)
        cuda.synchronize()
        gpu_times.append(gpu_time)

        # Проверка корректности перемножения
        np.testing.assert_allclose(np.dot(A, B), d_result.copy_to_host(), rtol=1e-5)

    # Построение графиков времени CPU и GPU
    plt.figure(figsize=(12, 6))

    # План для CPU
    plt.subplot(1, 2, 1)
    plt.plot(sizes, cpu_times, marker='o', label='CPU Time', color='b', linestyle='-')
    plt.xlabel('Размер матрицы')
    plt.ylabel('CPU время (секунды)')
    plt.title('CPU время для перемножения матриц')
    plt.grid(True)

    # План для GPU
    plt.subplot(1, 2, 2)
    plt.plot(sizes, gpu_times, marker='o', label='GPU Time', color='r', linestyle='-')
    plt.xlabel('Размер матрицы')
    plt.ylabel('Время работы GPU (секунд)')
    plt.title('Время GPU для перемножения матриц')
    plt.grid(True)

    # Отдельное окно для таблицы результатов
    plt.figure(figsize=(12, 4))
    table_data = [
        [f"{size}x{size}" for size in sizes],
        [f"{time:.6f}" for time in cpu_times],
        [f"{time:.6f}" for time in gpu_times],
        [f"{speedup:.2f}" if gpu_time != 0 else "N/A" for gpu_time, speedup in
         zip(gpu_times,
             [cpu_time / gpu_time if gpu_time != 0 else 0 for cpu_time, gpu_time in zip(cpu_times, gpu_times)])]
    ]

    table = plt.table(cellText=table_data,
                      rowLabels=['Размер матрицы', 'CPU Время (сек)', 'GPU Время (сек)', 'Ускорение'],
                      colLabels=[f"{size}x{size}" for size in sizes],
                      loc='center', cellLoc='center', rowLoc='center', bbox=[0, 0, 1, 1])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)  # Размер таблицы

    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()
