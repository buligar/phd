#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Безопасная пакетная разархивация результатов с авто-выравниванием корневой папки.
Цель: избежать дублирования вида a/b/results_ext.../results_ext.../plots.
"""

from __future__ import annotations
import argparse, concurrent.futures, shutil, sys, os
from pathlib import Path, PurePosixPath
from zipfile import ZipFile, BadZipFile

DEFAULT_ARCHIVES = [
    "results_ext_test1_one_cluster_bez_boost.zip",
    "results_ext_test1_one_cluster_s_boost.zip",
    "results_ext_test1_random_neighbors_bez_boost.zip",
    "results_ext_test1_random_neighbors_s_boost.zip",
    "results_ext_test1_top_neighbors_bez_boost.zip",
    "results_ext_test1_top_neighbors_s_boost.zip",
    "results_ext_test1.zip",
]

MARKER_NAME = ".extracted_ok"

def is_within(base: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except Exception:
        return False

def detect_single_root(zf: ZipFile) -> str | None:
    """Вернёт имя единственной корневой папки (если она одна), иначе None."""
    roots = set()
    for n in zf.namelist():
        if not n or n.startswith("__MACOSX"):
            continue
        p = PurePosixPath(n)
        if len(p.parts) == 0:
            continue
        roots.add(p.parts[0])
        if len(roots) > 1:
            return None
    return next(iter(roots)) if roots else None

def safe_extract_zip(zip_path: Path, out_dir: Path, strip_components: int = 0, chunk: int = 1024 * 1024) -> None:
    with ZipFile(zip_path, 'r') as zf:
        bad = zf.testzip()
        if bad is not None:
            raise BadZipFile(f"CRC error in '{bad}' внутри {zip_path.name}")
        for info in zf.infolist():
            p = PurePosixPath(info.filename)
            parts = p.parts[strip_components:]
            if not parts:
                continue  # пустая запись/каталог
            target = (out_dir.joinpath(*parts)).resolve()
            if not is_within(out_dir, target):
                print(f"[!] Пропуск небезопасного пути: {info.filename}")
                continue
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, 'r') as src, open(target, 'wb') as dst:
                shutil.copyfileobj(src, dst, length=chunk)

def process_archive(arc_path: Path, dst_root: Path, force: bool = False) -> tuple[str, str]:
    try:
        with ZipFile(arc_path, 'r') as zf:
            root_in_zip = detect_single_root(zf)

        # Папка верхнего уровня, где должны оказаться данные архива после распаковки
        target_root = dst_root / arc_path.stem
        marker = target_root / MARKER_NAME

        # Выравнивание корня: избавляемся от двойных уровней
        if root_in_zip and root_in_zip == arc_path.stem:
            # В ZIP уже есть корневая папка с именем архива → извлекаем в dst_root без strip,
            # чтобы получился dst_root/arc.stem/...
            out_dir = dst_root
            strip_components = 0
        elif root_in_zip:
            # В ZIP один корень, но имя другое → извлекаем в target_root со strip=1
            out_dir = target_root
            strip_components = 1
        else:
            # В ZIP нет единого корня → извлекаем в target_root без strip
            out_dir = target_root
            strip_components = 0

        # Если уже есть маркер и не просили перезапись — пропускаем
        if marker.exists() and not force:
            return (arc_path.name, "пропущен (уже распакован)")

        # Перезапись: очищаем именно target_root, не корневой dst_root
        if force and target_root.exists():
            shutil.rmtree(target_root)

        # Гарантируем существование каталога назначения для извлечения
        out_dir.mkdir(parents=True, exist_ok=True)

        safe_extract_zip(arc_path, out_dir, strip_components=strip_components)

        # Создаём маркер в target_root
        target_root.mkdir(parents=True, exist_ok=True)
        marker.write_text("ok", encoding="utf-8")

        return (arc_path.name, "успех")

    except FileNotFoundError:
        return (arc_path.name, "не найден")
    except BadZipFile as e:
        return (arc_path.name, f"ошибка ZIP/CRC: {e}")
    except Exception as e:
        return (arc_path.name, f"ошибка: {e}")


def main():
    ap = argparse.ArgumentParser(description="Пакетная разархивация ZIP с авто-выравниванием корня.")
    ap.add_argument("--src", type=Path, default=Path("."), help="Где лежат ZIP-файлы (по умолчанию: текущий).")
    ap.add_argument("--dst", type=Path, default=Path("."), help="Куда распаковывать (по умолчанию: текущий каталог).")
    ap.add_argument("--only", type=str, default="", help="Список имён через запятую для выборочной распаковки.")
    ap.add_argument("--force", action="store_true", help="Перезаписать уже распакованные данные.")
    ap.add_argument("--workers", type=int, default=1, help="Число параллельных потоков (архивов).")
    args = ap.parse_args()

    dst: Path = args.dst
    dst.mkdir(parents=True, exist_ok=True)

    wanted = [x.strip() for x in args.only.split(",") if x.strip()] if args.only.strip() else DEFAULT_ARCHIVES
    tasks = [args.src / name for name in wanted]

    print("[i] Источник:", args.src.resolve())
    print("[i] Назначение:", dst.resolve())
    print("[i] К распаковке:", ", ".join([t.name for t in tasks]))

    results = []
    if args.workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(process_archive, t, dst, args.force) for t in tasks]
            for f in concurrent.futures.as_completed(futs):
                results.append(f.result())
    else:
        for t in tasks:
            results.append(process_archive(t, dst, args.force))

    print("\n=== Сводка ===")
    maxn = max((len(n) for n, _ in results), default=0)
    for name, status in results:
        print(f"{name.ljust(maxn)} : {status}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[!] Прервано пользователем.", file=sys.stderr)
        sys.exit(130)
