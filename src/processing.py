import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine
from pathlib import Path


class DataProcessor:
    DEFAULT_QUERY = """
    SELECT 
        user_id,
        device_type_canonical,
        order_id,
        created_dt_msk AS order_dt,
        created_ts_msk AS order_ts,
        currency_code,
        revenue,
        tickets_count,
        created_dt_msk::date - (LAG(created_dt_msk) OVER (PARTITION BY user_id ORDER BY created_ts_msk))::date AS days_since_prev,
        event_id,
        event_name_code AS event_name,
        event_type_main,
        service_name,
        region_name,
        city_name
    FROM afisha.purchases p 
    JOIN afisha.events e USING (event_id)
    JOIN afisha.city c USING (city_id)
    JOIN afisha.regions r USING (region_id)
    WHERE p.device_type_canonical IN('mobile', 'desktop') 
        AND event_type_main <> 'фильм';
    """
    DEFAULT_DATA_PATH = "data/afisha_purchases.csv"
    DEFAULT_RATES_PATH = "data/final_tickets_tenge_df.csv"

    def __init__(self, data_path: str = None, rates_path: str = None):
        load_dotenv()
        self.db_config = {
            "user": os.getenv("DB_USER"),
            "pwd": os.getenv("DB_PWD"),
            "host": os.getenv("DB_HOST"),
            "port": os.getenv("DB_PORT"),
            "db": os.getenv("DB_NAME"),
        }
        self.data_path = data_path or self.DEFAULT_DATA_PATH
        self.rates_path = rates_path or self.DEFAULT_RATES_PATH
        self.df: pd.DataFrame | None = None
        self.df_rates: pd.DataFrame | None = None
        self.profile: pd.DataFrame | None = None

    def load_data_from_db(self, query: str | None = None) -> pd.DataFrame:
        missing = [key for key, value in self.db_config.items() if value is None]
        if missing:
            raise RuntimeError(
                f"Не заданы переменные окружения: {', '.join(missing)}. "
                "Проверьте, что в корне проекта есть файл .env, созданный "
                "из .env.example, и в нём заполнены DB_USER, DB_PWD, DB_HOST, "
                "DB_PORT, DB_NAME."
            )

        connection_string = "postgresql+psycopg2://{}:{}@{}:{}/{}".format(
            self.db_config["user"],
            self.db_config["pwd"],
            self.db_config["host"],
            self.db_config["port"],
            self.db_config["db"],
        )

        engine = create_engine(connection_string)
        query = query or self.DEFAULT_QUERY
        self.df = pd.read_sql_query(query, con=engine)
        return self.df

    def load_data_from_csv(self, path: str | None = None) -> pd.DataFrame:
        csv_path = Path(path or self.data_path)
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Файл данных не найден: {csv_path}. "
                "Скачайте выгрузку и положите её в папку data/. "
                "Подробности — в README.md, раздел «Подготовка данных»."
            )
        self.df = pd.read_csv(csv_path)
        return self.df

    def load_rates_from_csv(self, path: str | None = None) -> pd.DataFrame:
        csv_path = Path(path or self.rates_path)
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Файл курсов не найден: {csv_path}. "
                "Скачайте final_tickets_tenge_df.csv и положите его в папку data/. "
                "Подробности — в README.md, раздел «Подготовка данных»."
            )
        self.df_rates = pd.read_csv(csv_path)
        return self.df_rates

    def _convert_currency(self): 
        # 1. Готовим таблицу курсов
        rates = self.df_rates.copy()
        rates['rate_date'] = pd.to_datetime(rates['data']).dt.normalize()

        rates['rate_per_unit'] = rates['curs'] / rates['nominal']   # курс за 1 тенге
        rates = rates[['rate_date', 'rate_per_unit']].drop_duplicates('rate_date')

        # 2. Приводим дату заказа к дате без времени
        self.df['order_date'] = self.df['order_dt'].dt.normalize()

        # 3. Присоединяем курс по дате
        self.df = self.df.merge(rates, how='left', left_on='order_date', right_on='rate_date')

        # 4. Считаем выручку в рублях
        self.df['revenue_rub'] = np.where(
            self.df['currency_code'] == 'rub',
            self.df['revenue'],
            self.df['revenue'] * self.df['rate_per_unit']
        ).round(2)

        # 5. Убираем служебные столбцы
        self.df = self.df.drop(columns=['order_date', 'rate_date', 'rate_per_unit'])

        if self.df['revenue_rub'].isna().any():
            n = self.df['revenue_rub'].isna().sum()
            print(f"Внимание: {n} заказов без курса, revenue_rub = NaN")

    def _optimize_types(self): 
        # Оптимизируем тип tickets_count
        self.df['tickets_count'] = pd.to_numeric(self.df['tickets_count'], downcast='integer')
        # Оптимизируем тип days_since_prev
        self.df['days_since_prev'] = self.df['days_since_prev'].astype('Int64')

    def _drop_duplicates(self):
        # Содержательные поля, определяющие заказ (без order_id и days_since_prev)
        order_key = [
            'user_id', 'order_ts', 'device_type_canonical', 'currency_code',
            'revenue', 'tickets_count', 'event_id', 'service_name'
        ]

        # Удаление: оставляем первое вхождение
        rows_before = len(self.df)
        self.df = self.df.drop_duplicates(subset=order_key, keep='first').copy()
        rows_after = len(self.df)

        if rows_before - rows_after > 0:
            print(f"Удалено строк-дубликатов: {rows_before - rows_after} ({(rows_before - rows_after)/rows_before:.2%})")

    def _remove_negative_revenue(self): 
        rows_before = len(self.df)
        self.df = self.df[self.df['revenue_rub'] >= 0].copy()

        if rows_before - len(self.df) > 0:
            print(f"Удалено заказов с отрицательной выручкой: {rows_before - len(self.df)}")

    def _filter_p99_revenue(self): 
        perc_99 = self.df['revenue_rub'].quantile(0.99)
        old_shape = self.df.shape[0]
        self.df = self.df[self.df['revenue_rub'] <= perc_99].copy()

        if old_shape - self.df.shape[0] > 0:
            print(f'Удалено строк по выручке (>99-го перцентиля): {old_shape - self.df.shape[0]} ({(old_shape - self.df.shape[0]) / old_shape:.2%})')

    def _recalc_days_since_prev(self):
        # Пересчитываем интервалы между заказами уже по очищенным данным
        self.df = self.df.sort_values(['user_id', 'order_ts', 'order_id']).reset_index(drop=True)
        self.df['days_since_prev'] = self.df.groupby('user_id')['order_dt'].diff().dt.days.astype('Int64')

    def preprocess(self) -> pd.DataFrame:
        if self.df is None:
            raise RuntimeError("Сначала загрузите данные.")
        self._convert_currency()
        self._drop_duplicates()
        self._optimize_types()
        self._remove_negative_revenue()
        self._filter_p99_revenue()
        self._recalc_days_since_prev()
        print(f"[preprocess] заказов: {len(self.df)}, пользователей: {self.df['user_id'].nunique()}")
        return self.df

    def build_user_profile(self) -> pd.DataFrame:
        if self.df is None:
            raise RuntimeError("Сначала вызовите preprocess()")
        # Сортируем заказы по времени, чтобы выбрать первый заказ пользователя методом first
        df_sorted = self.df.sort_values(["order_ts", "order_id"]).reset_index(drop=True)
        # Строим профиль
        self.profile = df_sorted.groupby("user_id", as_index=False).agg(
            first_order=("order_dt", "min"),
            last_order=("order_dt", "max"),
            first_device_type=("device_type_canonical", "first"),
            first_region=("region_name", "first"),
            first_event_type=("event_type_main", "first"),
            total_orders=("order_id", "count"),
            avg_revenue_rub=("revenue_rub", "mean"),
            avg_tickets=("tickets_count", "mean"),
            avg_time_between_orders=("days_since_prev", "mean"),
        )

        # Бинарный признак: совершил ли пользователь 2 и более заказа
        self.profile["is_two"] = self.profile["total_orders"] >= 2

        self._filter_profile_p99()
        print(f"[build profile] пользователей: {self.profile.shape[0]}")
        return self.profile

    def _filter_profile_p99(self):
        if self.profile is None:
            raise RuntimeError("Сначала постройте профиль пользователя")
        # Фильтрация: оставляем пользователей с числом заказов не выше 99-го перцентиля
        orders_p99 = self.profile["total_orders"].quantile(0.99)
        users_before = len(self.profile)

        self.profile = self.profile[self.profile["total_orders"] <= orders_p99].copy()

        if users_before - len(self.profile) > 0:
            print(f"Удалено пользователей (>99 перцентиля заказов): {users_before - len(self.profile)} ({(users_before - len(self.profile)) / users_before:.2%})")
