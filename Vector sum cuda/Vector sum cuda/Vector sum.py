
import matplotlib.pyplot as plt
import numpy as np
from numba import cuda

# GPU суммирования векторов
@cuda.jit
def sum_vector_gpu(vec, result):
    tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    stride = cuda.gridDim.x * cuda.blockDim.x

    partial_sum = 0.0
    for i in range(tid, len(vec), stride):
        partial_sum += vec[i]

    cuda.atomic.add(result, 0, partial_sum)

# Функция для генерации случайного вектора
def generate_vector(size):
    return np.random.rand(size).astype(np.float32)

# Функц для измерения времени выполнения функц суммирования векторов
def measure_time(sum_vector_func, *args):
    start_event = cuda.event()
    end_event = cuda.event()

    start_event.record()
    sum_vector_func(*args)
    end_event.record()
    end_event.synchronize()

    return start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds

# Основная функция
def main():
    sizes = [100000, 200000, 400000, 800000, 1000000]
    cpu_times = []
    gpu_times = []

    for size in sizes:
        vec = generate_vector(size)

        # Время работы CPU
        cpu_time = measure_time(np.sum, vec)
        cpu_times.append(cpu_time)

        # Время работы GPU
        threads_per_block = 256
        blocks_per_grid = (size + threads_per_block - 1) // threads_per_block

        d_vec = cuda.to_device(vec)
        d_result = cuda.device_array(1, dtype=np.float32)

        gpu_time = measure_time(sum_vector_gpu[blocks_per_grid, threads_per_block], d_vec, d_result)
        gpu_times.append(gpu_time)

    # Построение графиков времени CPU и GPU
    plt.figure(figsize=(12, 6))

    # План для CPU
    plt.subplot(1, 2, 1)
    plt.plot(sizes, cpu_times, marker='o', label='CPU Time', color='b', linestyle='-')
    plt.xscale('log')  # Логарифмическая шкала
    plt.xlabel('Размер вектора (логарифмическая шкала)')
    plt.ylabel('CPU время (секунды)')
    plt.title('CPU время для суммирования векторов')
    plt.grid(True)

    # План для GPU
    plt.subplot(1, 2, 2)
    plt.plot(sizes, gpu_times, marker='o', label='GPU Time', color='r', linestyle='-')
    plt.xscale('log')  # Логарифмическая шкала
    plt.xlabel('Размер вектора (логарифмическая шкала)')
    plt.ylabel('Время работы GPU (секунд)')
    plt.title('Время GPU для суммирования векторов')
    plt.grid(True)

    # Отдельное окно для таблицы результатов
    plt.figure(figsize=(12, 4))
    table_data = [[f"{size:,}" for size in sizes],
                  [f"{time:.6f}" for time in cpu_times],
                  [f"{time:.6f}" for time in gpu_times],
                  [f"{speedup:.2f}" if gpu_time != 0 else "N/A" for gpu_time, speedup in zip(gpu_times, [cpu_time/gpu_time if gpu_time != 0 else 0 for cpu_time, gpu_time in zip(cpu_times, gpu_times)])]]

    table = plt.table(cellText=table_data,
                      rowLabels=['Размер вектора', 'CPU Время (сек)', 'GPU Время (сек)', 'Ускорение'],
                      colLabels=[f"{size:,}" for size in sizes],
                      loc='center', cellLoc='center', rowLoc='center', bbox=[0, 0, 1, 1])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)  # Регулировка размера таблицы

    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
