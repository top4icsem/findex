# ЛР №8 — профілювання (заповнюється за фактичними вимірами)

Запуск: `bash scripts/profile_lab08.sh` (корпус `data/`, `index.bin`).

| Місце / функція | Час / частка до змін, cProfile | Гіпотеза | Python чи нативний код, за Scalene | Після NumPy |
|---|---:|---|---|---:|
| 1. Заповнити з `docs/lab08_profile/legacy_search.txt` | — | — | — | — |
| 2. Заповнити з `docs/lab08_profile/build.txt` | — | — | — | — |
| 3. Заповнити з py-spy flame graph | — | — | — | — |

Не називайте у звіті це *реальними результатами*, доки не заповнили після власних вимірювань.

- cProfile: `docs/lab08_profile/{build,legacy_search,numpy_search}.{pstats,txt}`.
- py-spy: `docs/lab08_profile/build_flame.svg` та `search_flame.svg`.
- Scalene: `docs/lab08_profile/scalene_search.txt` (якщо підтримується ОС).
- Межа вимірювання: скрипт профілює **наявний legacy-код** Lab 03 як еталон; знімок фактичного Git-коміту до будь-якого рефакторингу зберігайте окремо.
