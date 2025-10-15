Чтобы получить графики сравнения режимов из 500 нейронов для top centrality, proxy, hybrid
1. Извлечь архивы командой python unzip.py
   results_ext_test1_one_cluster_bez_boost.zip       — топ-узлы внутри целевого кластера без буста;
   results_ext_test1_one_cluster_s_boost.zip         — топ-узлы внутри целевого кластера с бустом;
   results_ext_test1_random_neighbors_bez_boost.zip  — случайные соседи 3 самых центральных узлов остальных кластеров без буста;
   results_ext_test1_random_neighbors_s_boost.zip    — случайные соседи 3 самых центральных узлов остальных кластеров с бустом;
     
2. Построить графики
   python graph_2.py
   python graph_3.py
   python graph_4.py

Чтобы получить данные как в архивах нужно:
1. Произвести 4 запуска программы python sim_copy_old.py  в различных режимах mode: str = 'one_cluster_bez_boost' и т.д.
При этом в каждой симуляции появляется папка results_ext_test1, ее нужно переименовать в папку соответсвующего режима.
Например: results_ext_test1_one_cluster_bez_boost
Когда все папки с полученными результатами сгенерированы
Последовательно выполняем запуск
2. python test1.py - доп. данные (значения изменения центральностей)
3. python test2.py - доп. данные (матрица пересечений управляющих нейронов)
4. python graph_1.py
5. python graph_2.py
6. python graph_3.py
7. python graph_4.py

Чтобы получить графики сравнения режимов сети из 50 нейронов без stdp и stdp  
1. Извлечь 
   results_ext_test1.zip
2. Построить графики
   python graph_1_stdp.py
