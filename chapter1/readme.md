Чтобы получить графики сравнения режимов из 500 нейронов для top centrality, proxy, hybrid
1. Извлечь архивы
   results_ext_test1_one_cluster_bez_boost.zip       — топ-узлы внутри целевого кластера без буста;
   results_ext_test1_one_cluster_s_boost.zip         — топ-узлы внутри целевого кластера с бустом;
   results_ext_test1_random_neighbors_bez_boost.zip  — случайные соседи 3 самых центральных узлов остальных кластеров без буста;
   results_ext_test1_random_neighbors_s_boost.zip    — случайные соседи 3 самых центральных узлов остальных кластеров с бустом;
   results_ext_test1_top_neighbors_bez_boost.zip     — топовые соседи 3 самых центральных узлов остальных кластеров без буста;
   results_ext_test1_top_neighbors_s_boost.zip       — топовые соседи 3 самых центральных узлов остальных кластеров с бустом.
2. Построить графики
   graph_1old.py
   graph_2old.py
   graph_3old.py
   graph_4old.py

3. Чтобы получить данные как в архивах нужно запускать python sim_copy_old.py в различных режимах
mode: str = 'one_cluster_bez_boost' и т.д.
4. test1.py - доп. данные (значения изменения центральностей)
5. test2.py - доп. данные (матрица пересечений управляющих нейронов)

Чтобы получить графики сравнения режимов сети из 50 нейронов без stdp и stdp
1. Извлечь архивы
   results_ext_test1.zip 
2. Построить графики
   graph_1new.py
   graph_2new.py
   graph_3new.py
   graph_4new.py

3. Чтобы получить данные как в архивах нужно запускать python sim_copy_new.py в различных режимах
use_stdp_values = [True] или [False]
