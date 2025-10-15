```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # если есть
```


# Репликация графиков сравнения режимов (500/50 нейронов)

Ниже приведён воспроизводимый протокол для получения всех графиков по режимам **top-centrality**, **proxy** и (опционально) **hybrid**, а также для набора из **50 нейронов** (без STDP/со STDP). Инструкции разделены на: быстрый старт из архивов и полный цикл с повторными симуляциями.

---

## Требования

* **Python** ≥ 3.9
* Библиотеки: `numpy`, `scipy`, `pandas`, `matplotlib`, `networkx`, `brian2` (для симуляций), при необходимости `seaborn` (если используется в ваших `graph_*`).
* Рекомендуется изолированная среда:

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # если есть
# иначе:
pip install numpy scipy pandas matplotlib networkx brian2
```

---

## Обозначения режимов

| Имя папки/режима                                             | Смысл                                                                                   |
| ------------------------------------------------------------ | --------------------------------------------------------------------------------------- |
| `one_cluster_bez_boost` / `one_cluster_s_boost`              | **Top-centrality**: стимуляция топ-узлов целевого кластера (без/с бустом центральности) |
| `random_neighbors_bez_boost` / `random_neighbors_s_boost`    | **Proxy**: случайные соседи 3 самых центральных узлов других кластеров (без/с бустом)   |
| `top_neighbors_bez_boost` / `top_neighbors_s_boost` *(опц.)* | **Hybrid**: соседи наиболее центральных узлов целевого и/или смежных кластеров          |

> Если «hybrid» не используется в вашей серии, просто опустите соответствующие пункты.

---

## A. Быстрый старт (из готовых архивов) — **500 нейронов**

1. **Распаковка архивов**

```bash
python unzip.py
```

Ожидаемые архивы (минимальный набор для top-centrality и proxy):

* `results_ext_test1_one_cluster_bez_boost.zip`
* `results_ext_test1_one_cluster_s_boost.zip`
* `results_ext_test1_random_neighbors_bez_boost.zip`
* `results_ext_test1_random_neighbors_s_boost.zip`

*(Опционально для hybrid: `results_ext_test1_top_neighbors_bez_boost.zip`, `results_ext_test1_top_neighbors_s_boost.zip`.)*

> **Важно (о структуре путей):** проверяйте, что после распаковки нет лишней вложенности вида
> `.../results_ext_test1/results_ext_test1/plots`.
> Правильный целевой путь:
> `phd/chapter1/results_ext_test1_<mode>/plots`
> Если распаковщик создал лишнюю вложенность, поднимите внутренний каталог на уровень выше:
>
> ```bash
> # пример исправления двойной вложенности для одного режима
> mv phd/chapter1/unpacked/results_ext_test1/results_ext_test1/* phd/chapter1/results_ext_test1_one_cluster_bez_boost/
> rmdir phd/chapter1/unpacked/results_ext_test1/results_ext_test1
> ```

2. **Построение сравнительных графиков**

```bash
python graph_2old.py
python graph_3old.py
python graph_4old.py
```

Графики будут сохранены в соответствующие подпапки `plots/` внутри каждой директории режима и/или в сводные каталоги (в зависимости от реализации скриптов).

---

## B. Полная репродукция (симуляции → доп. данные → графики) — **500 нейронов**

1. **Запуск симуляций в 4 режимах**
   В файле `sim_copy_old.py` установите параметр:

```python
mode: str = 'one_cluster_bez_boost'  # затем меняйте на каждый режим по очереди
```

Выполните **четыре** запуска, последовательно подставляя:

* `one_cluster_bez_boost`
* `one_cluster_s_boost`
* `random_neighbors_bez_boost`
* `random_neighbors_s_boost`

*(Опционально для hybrid — ещё два запуска: `top_neighbors_bez_boost`, `top_neighbors_s_boost`.)*

Каждый запуск создаёт папку `results_ext_test1`. После каждого запуска **сразу переименуйте** её согласно режиму:

```bash
python sim_copy_old.py
mv results_ext_test1 results_ext_test1_one_cluster_bez_boost

# затем следующий режим:
# (сменить mode в sim_copy_old.py)
python sim_copy_old.py
mv results_ext_test1 results_ext_test1_one_cluster_s_boost

# и так далее для остальных режимов
```

2. **Дополнительные данные**

```bash
python test1.py   # метрики изменения центральностей
python test2.py   # матрица пересечений управляющих нейронов
```

3. **Построение графиков (полный набор)**

```bash
python graph_1old.py
python graph_2old.py
python graph_3old.py
python graph_4old.py
```

> Убедитесь, что имена папок **строго** соответствуют ожидаемым в `graph_*`-скриптах.
> Если скрипты используют фиксированные пути, скорректируйте константы или символические ссылки.

---

## C. Набор **50 нейронов** (без STDP и со STDP)

1. **Распаковка**

```bash
unzip results_ext_test1.zip
```

2. **Графики**

```bash
python graph_1new.py
```

---

## Ожидаемая структура каталогов (пример)

```text
phd/
└── chapter1/
    ├── results_ext_test1_one_cluster_bez_boost/
    │   ├── data/              # метрики, матрицы, логи (если формируются)
    │   └── plots/             # графики по режиму
    ├── results_ext_test1_one_cluster_s_boost/
    │   └── plots/
    ├── results_ext_test1_random_neighbors_bez_boost/
    │   └── plots/
    ├── results_ext_test1_random_neighbors_s_boost/
    │   └── plots/
    ├── results_ext_test1_top_neighbors_bez_boost/    # (опц.)
    │   └── plots/
    ├── results_ext_test1_top_neighbors_s_boost/      # (опц.)
    │   └── plots/
    ├── results_ext_test1/        # (для набора 50 нейронов из архива)
    │   └── plots/
    ├── sim_copy_old.py
    ├── test1.py
    ├── test2.py
    ├── graph_1old.py
    ├── graph_2old.py
    ├── graph_3old.py
    ├── graph_4old.py
    ├── graph_1new.py
    └── unzip.py
```

---

## Частые проблемы и их решение

* **Двойная вложенность при распаковке** (`.../results_ext_test1/results_ext_test1/plots`):
  запускайте `unzip.py` из нужного родительского каталога **или** вручную переместите внутренний `results_ext_test1` на уровень выше (см. пример выше).

* **Несоответствие имён директорий ожиданиям скриптов**:
  проверьте константы путей в `graph_*` и `test*`-скриптах. Наименования должны совпадать **байт-в-байт**.

* **Повторяемость результатов**:
  чтобы исключить дрожание метрик, задавайте фиксированный `SEED` в симуляциях и генераторах графов (если такой параметр предусмотрен).

---

## Минимальные команды (шпаргалка)

**Из архивов (500 нейронов):**

```bash
python unzip.py
python graph_2old.py
python graph_3old.py
python graph_4old.py
```

**Полный цикл (500 нейронов):**

```bash
# 4 (или 6) симуляций с переименованием результатов:
python sim_copy_old.py && mv results_ext_test1 results_ext_test1_one_cluster_bez_boost
# сменить mode ...
python sim_copy_old.py && mv results_ext_test1 results_ext_test1_one_cluster_s_boost
python sim_copy_old.py && mv results_ext_test1 results_ext_test1_random_neighbors_bez_boost
python sim_copy_old.py && mv results_ext_test1 results_ext_test1_random_neighbors_s_boost
# (опц.) ещё 2 режима hybrid

# доп. данные + графики
python test1.py
python test2.py
python graph_1old.py
python graph_2old.py
python graph_3old.py
python graph_4old.py
```

**Набор 50 нейронов:**

```bash
unzip results_ext_test1.zip
python graph_1new.py
```

---

## Примечание о «hybrid»

Если в проекте присутствуют архивы/режимы `top_neighbors_*`, добавьте их в распаковку и в сводные скрипты наравне с остальными. В противном случае раздел «hybrid» опустите — сравнение будет выполнено для **top-centrality** и **proxy**.

---

Если потребуется, могу добавить вариант **Makefile**/`invoke`-тасков для автоматизации всего конвейера (симуляции → переименование → доп. данные → графики) без ручного редактирования `mode`.


Запуск:
```bash
python VAR_version_fs_100.py
```
