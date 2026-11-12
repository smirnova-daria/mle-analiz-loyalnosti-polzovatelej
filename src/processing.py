import os
import pandas as pd
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
