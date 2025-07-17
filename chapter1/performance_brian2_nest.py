import time
import matplotlib.pyplot as plt
import numpy as np
import psutil
from brian2 import PoissonInput, NeuronGroup, Synapses, SpikeMonitor, run, start_scope, ms, second, Hz
import nest

def get_memory_usage():
    return psutil.Process().memory_info().rss / (1024 ** 2)  # в мегабайтах

def run_brian2_simulation(N):
    start_scope()
    start_mem = get_memory_usage()
    
    # Инициализация модели
    eqs = 'dv/dt = -v/(10*ms) : 1'
    G = NeuronGroup(N, eqs, threshold='v>1', reset='v=0', method='euler')
    
    # Создание случайных соединений
    S = Synapses(G, G, on_pre='v_post += 0.5')
    S.connect(condition='i != j', p=0.15)
    num_synapses = len(S)

    # Входные сигналы
    PoissonInput(G, 'v', 1, 100*Hz, 1)
    
    # Замер времени и памяти
    start_time = time.time()
    run(1*second)
    elapsed_time = time.time() - start_time
    used_mem = abs(get_memory_usage() - start_mem)
    
    return elapsed_time, used_mem, num_synapses

def run_nest_simulation(N):
    nest.ResetKernel()
    start_mem = get_memory_usage()
    
    # Параметры модели
    neuron_params = {
        'V_th': 1.0, 'V_reset': 0.0,
        'tau_m': 10.0, 'C_m': 250.0,
        'E_L': 0.0, 'V_m': 0.0
    }
    
    # Создание нейронов
    neurons = nest.Create('iaf_psc_alpha', N, params=neuron_params)
    
    # Создание случайных соединений
    nest.Connect(neurons, neurons,
                 conn_spec={
                     'rule': 'pairwise_bernoulli',
                     'p': 0.15,
                     'allow_autapses': False
                 },
                 syn_spec={'weight': 0.5})
    num_synapses = len(nest.GetConnections())
    # Входные сигналы
    noise = nest.Create('poisson_generator', 1, {'rate': 100.0})
    nest.Connect(noise, neurons, syn_spec={'weight': 1.0})
    
    # Замер времени и памяти
    start_time = time.time()
    nest.Simulate(1000.0)  # 1000 ms = 1 second
    elapsed_time = time.time() - start_time
    used_mem = abs(get_memory_usage() - start_mem)
    
    return elapsed_time, used_mem, num_synapses

# Параметры тестирования
sizes = [500, 1000, 5000, 10000, 20000]
results = {
    'brian_time': [],
    'nest_time': [],
    'brian_mem': [],
    'nest_mem': []
}

# Выполнение тестов
for N in sizes:
    print(f"\nTesting N = {N}")
    
    # Brian2
    b_time, b_mem, b_num_syn = run_brian2_simulation(N)
    results['brian_time'].append(b_time)
    results['brian_mem'].append(b_mem)
    
    # NEST
    n_time, n_mem, n_num_syn = run_nest_simulation(N)
    results['nest_time'].append(n_time)
    results['nest_mem'].append(n_mem)
    
    print(f"Brian2: {b_time:.2f} sec, {b_mem:.2f} MB, кол-во синапсов:{b_num_syn}")
    print(f"NEST:   {n_time:.2f} sec, {n_mem:.2f} MB, кол-во синапсов:{n_num_syn}")

# Визуализация результатов
plt.figure(figsize=(15, 6))

# График времени выполнения
plt.subplot(121)
plt.plot(sizes, results['brian_time'], 'o-', label='Brian2')
plt.plot(sizes, results['nest_time'], 's-', label='NEST')
plt.xlabel('Количество нейронов')
plt.ylabel('Время (сек)')
plt.title('Сравнение времени выполнения')
plt.grid(True, which='both', linestyle='--')
plt.legend()

# График использования памяти
plt.subplot(122)
plt.plot(sizes, results['brian_mem'], 'o-', label='Brian2')
plt.plot(sizes, results['nest_mem'], 's-', label='NEST')
plt.xlabel('Количество нейронов')
plt.ylabel('Использование памяти (мбайт)')
plt.title('Сравнение потребления памяти')
plt.grid(True, which='both', linestyle='--')
plt.legend()

plt.tight_layout()
plt.show()