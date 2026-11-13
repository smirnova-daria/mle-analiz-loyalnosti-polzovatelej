"""Точка входа в приложение: анализ лояльности пользователей Яндекс Афиши.

Запуск:
    python3 main.py
"""
from __future__ import annotations
import sys
from src.processing import DataProcessor


def print_report(dp: DataProcessor, ret: dict, corr: dict) -> None:
    """Краткий итоговый отчёт в терминал."""
    profile = dp.profile
    overall = ret["overall_repeat"]

    print("=" * 64)
    print(" АНАЛИЗ ЛОЯЛЬНОСТИ ПОЛЬЗОВАТЕЛЕЙ ЯНДЕКС АФИШИ")
    print("=" * 64)

    print("\n[Данные]")
    print(f"  заказов после очистки:   {len(dp.df):>8_}")
    print(f"  пользователей в профиле: {profile.shape[0]:>8_}")
    print(f"  доля вернувшихся:        {overall:>8.1%}")

    # --- Сегменты по устройству ---
    print("\n[Retention по устройству первой покупки]")
    dev = ret["segments"]["first_device_type"]
    for _, row in dev.iterrows():
        marker = " ←" if row["repeat_share"] > overall else ""
        print(f"  {row['first_device_type']:<10} "
              f"n={row['users']:>6}  вернулись {row['repeat_share']:>6.1%}{marker}")

    # --- Сегменты по типу мероприятия ---
    print("\n[Retention по типу первого мероприятия]")
    ev = ret["segments"]["first_event_type"]
    for _, row in ev.iterrows():
        marker = " ←" if row["repeat_share"] > overall else ""
        print(f"  {row['first_event_type']:<12} "
              f"n={row['users']:>6}  вернулись {row['repeat_share']:>6.1%}{marker}")

    # --- Гипотезы ---
    print("\n[Гипотеза 1: спорт vs концерты]")
    h1 = ret["hypotheses"]["sport_vs_concert"]
    print(f"  спорт:    n={h1['sport']['n']:>5}  вернулись {h1['sport']['share']}%")
    print(f"  концерты: n={h1['concert']['n']:>5}  вернулись {h1['concert']['share']}%")
    print(f"  разница:  {h1['diff_pp']:+.1f} п.п. → {h1['conclusion']}")

    print("\n[Гипотеза 2: топ-регионы vs остальные]")
    h2 = ret["hypotheses"]["top_regions"]
    print(f"  топ-{h2['top_n']}:    n={h2['top']['n']:>5}  вернулись {h2['top']['share']}%")
    print(f"  остальные: n={h2['rest']['n']:>5}  вернулись {h2['rest']['share']}%")
    print(f"  разница:   {h2['diff_pp']:+.1f} п.п. → {h2['conclusion']}")

    # --- Корреляции ---
    print("\n[φ_K: связь с числом заказов (total_orders)]")
    print("  " + corr["top_total"].to_string().replace("\n", "\n  "))

    print("\n[φ_K: связь с сегментом по числу заказов (orders_group)]")
    print("  " + corr["top_group"].to_string().replace("\n", "\n  "))

    # --- Рекомендации ---
    print("\n[Рекомендации]")
    print("  1. Сфокусироваться на втором заказе: ~39% пользователей не возвращаются.")
    print("  2. Приоритет удержания: театр и десктоп — устойчиво выше среднего.")
    print("  3. Мобильные (83% входов) возвращаются реже — нужны стимулы.")
    print("  4. Не сегментировать по дню недели и региону: эффект слабый.")
    print("  5. 216 аномальных аккаунтов (51% заказов) — отдельная стратегия.")
    print("  6. При построении модели возврата использовать целевую переменную ")
    print("     «повторный заказ в течение N дней» вместо «2+ заказа».")
    print("=" * 64)


def main() -> int:
    dp = DataProcessor()
    try:
        dp.load_data_from_db()
        dp.load_rates_from_csv()
        dp.preprocess()
        dp.build_user_profile()
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"[ошибка] {exc}", file=sys.stderr)
        return 1

    ret = dp.segment_retention(
        ["first_device_type", "first_event_type", "first_region"]
    )
    ret["hypotheses"] = {
        "sport_vs_concert": dp.test_hypothesis_sport_vs_concert(),
        "top_regions":      dp.test_hypothesis_top_regions(),
    }
    corr = dp.analyze_correlation()

    print_report(dp, ret, corr)

    return 0


if __name__ == "__main__":
    sys.exit(main())