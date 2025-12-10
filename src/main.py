# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Тестовое задание для Lead Sniper
# Автор: Сулейманова Алия
# Дата: 10.12.2025
# -----------------------------------------------------------------------------

import json
import os
import re
import time

import numpy as np
import pandas as pd
import requests

# --- 1. КОНСТАНТЫ И НАСТРОЙКИ ---
print("--- [1/7] Инициализация скрипта ---")
API_KEY = "ТВОЙ_НОВЫЙ_КЛЮЧ_API"

# Умное определение путей
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
except NameError:  # Если запускаем в Jupyter, __file__ не определен
    project_root = os.getcwd()  # Используем текущую рабочую директорию

RAW_DATA_PATH = os.path.join(project_root, "data", "raw", "raw_companies.csv")
INTERIM_DATA_PATH = os.path.join(project_root, "data", "interim", "enriched_data.csv")
FINAL_DATA_PATH = os.path.join(project_root, "data", "companies.csv")
MIN_REVENUE = 200_000_000

# --- 2. ФУНКЦИИ-ПОМОЩНИКИ ---


def get_financials(inn, api_key):
    """Получает финансовые данные (выручку и год) для компании по ИНН через API ФНС."""
    try:
        url = f"https://api-fns.ru/api/bo?req={inn}&key={api_key}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if inn not in data:
            return None, None
        company_data = data[inn]
        if not isinstance(company_data, dict) or not company_data:
            return None, None
        latest_year = sorted(company_data.keys())[-1]
        latest_report = company_data[latest_year]
        if "2110" in latest_report:
            revenue_year = int(latest_year)
            revenue = latest_report["2110"]
            return revenue_year, int(revenue) * 1000
    except Exception as e:
        print(f"  -> Ошибка при получении фин. данных для ИНН {inn}: {e}")
    return None, None


def get_okved(inn, api_key):
    """Получает основной ОКВЭД для компании (ЮЛ или ИП) по ИНН через API ФНС."""
    try:
        url = f"https://api-fns.ru/api/egr?req={inn}&key={api_key}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if "items" in data and data["items"]:
            item = data["items"][0]
            company_data = item.get("ЮЛ", item.get("ИП"))
            if (
                company_data
                and "ОснВидДеят" in company_data
                and "Код" in company_data["ОснВидДеят"]
            ):
                return company_data["ОснВидДеят"]["Код"]
    except Exception as e:
        print(f"  -> Ошибка при получении ОКВЭД для ИНН {inn}: {e}")
    return None


def extract_email(text):
    """Извлекает первый попавшийся email из строки."""
    if isinstance(text, str):
        match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
        return match.group(0).lower() if match else None
    return None


def extract_phone(text):
    """Извлекает первый попавшийся телефон из строки и приводит к формату +7 (версия 2)."""
    if isinstance(text, str):
        # Новое, более гибкое регулярное выражение:
        # Ищет последовательности из 10-11 цифр, которые могут быть разделены пробелами, скобками, дефисами.
        # Может начинаться с 7, 8 или +7.
        match = re.search(
            r"(\+?[78])?[\s\-(]*(\d{3})[\s\-)]*(\d{3})[\s\-]*(\d{2})[\s\-]*(\d{2})",
            text,
        )

        if match:
            # Собираем номер из найденных групп цифр
            digits_only = "".join(filter(str.isdigit, match.group(0)))

            # Если номер 11-значный и начинается с 8, заменяем на 7
            if len(digits_only) == 11 and digits_only.startswith("8"):
                digits_only = "7" + digits_only[1:]

            # Если номер 10-значный (без 7 или 8 в начале), добавляем 7
            if len(digits_only) == 10:
                digits_only = "7" + digits_only

            # Добавляем + в начале, если его нет
            if digits_only.startswith("7"):
                return "+" + digits_only

            return digits_only  # На всякий случай
    return None


# --- 3. ЗАГРУЗКА И ПРЕДВАРИТЕЛЬНАЯ ОЧИСТКА ---
print(f"--- [2/7] Загрузка и очистка исходных данных из {RAW_DATA_PATH} ---")
try:
    df_full = pd.read_csv(RAW_DATA_PATH)
except FileNotFoundError:
    print(f"ОШИБКА: Файл {RAW_DATA_PATH} не найден.")
    exit()

# БЕРЕМ ТОЛЬКО ПЕРВЫЕ 50 КОМПАНИЙ
df = df_full.head(50).copy()
print(f"Для обработки взяты первые {len(df)} компаний из списка.")

df["name"] = df["name"].str.strip()
df["legal_person"] = df["legal_person"].str.strip()
df["inn"] = df["inn"].astype(str)
df = df.rename(columns={"РРАР_score": "rating_ref"})

tag_mapper = {
    "BTL агентства": "BTL",
    "Агентства полного цикла": "FULL_CYCLE",
    "Сувенирная продукция": "SOUVENIR",
    "Event-management": "EVENT",
    "Мерчандайзинг": "MERCHANDISING",
    "Оформление мест продаж POS": "POS",
    "PR агентства": "PR, COMM_GROUP",
}
df["segment_tag"] = df["category"].apply(lambda x: tag_mapper.get(x, "UNKNOWN"))
df = df.drop(columns=["category"])
print("Исходные данные очищены и подготовлены.")

# --- 4. ПРОГРАММНОЕ ОБОГАЩЕНИЕ ДАННЫХ ---
print("--- [3/7] Начинаем обогащение данных через API. Это может занять время... ---")
enriched_df = df.copy()
enriched_df["revenue_year"], enriched_df["revenue"], enriched_df["okved_main"] = (
    np.nan,
    np.nan,
    np.nan,
)

for index, row in enriched_df.iterrows():
    inn = row["inn"]
    print(f"({index + 1}/{len(enriched_df)}) Обрабатываем: {row['name']} (ИНН: {inn})")
    revenue_year, revenue = get_financials(inn, API_KEY)
    if revenue is not None:
        enriched_df.loc[index, "revenue_year"] = revenue_year
        enriched_df.loc[index, "revenue"] = revenue
    okved = get_okved(inn, API_KEY)
    if okved is not None:
        enriched_df.loc[index, "okved_main"] = okved
    time.sleep(1.2)

os.makedirs(os.path.dirname(INTERIM_DATA_PATH), exist_ok=True)
enriched_df.to_csv(INTERIM_DATA_PATH, index=False)
print(f"Промежуточные результаты сохранены в {INTERIM_DATA_PATH}")

# --- 5. ФИНАЛЬНАЯ ОБРАБОТКА И ФИЛЬТРАЦИЯ ---
print("--- [4/7] Финальная обработка, дедупликация и фильтрация ---")
processed_df = enriched_df.copy()

processed_df = (
    processed_df.sort_values(by="revenue", ascending=False)
    .groupby("inn")
    .agg(
        {
            "name": "first",
            "legal_person": "first",
            "segment_tag": lambda x: ", ".join(sorted(x.unique())),
            "region": "first",
            "description": "first",
            "site": "first",
            "contacts": "first",
            "source": "first",
            "rating_ref": "first",
            "revenue": "first",
            "revenue_year": "first",
            "okved_main": "first",
        }
    )
    .reset_index()
)
print(f"Обработаны дубликаты. Уникальных компаний: {len(processed_df)}")

# Нормализация контактов
processed_df["email"] = processed_df["contacts"].apply(extract_email)
processed_df["phone"] = processed_df["contacts"].apply(extract_phone)
print("Контакты извлечены в отдельные столбцы.")

# Фильтрация по выручке
processed_df["revenue"] = pd.to_numeric(processed_df["revenue"], errors="coerce")
processed_df = processed_df.dropna(subset=["revenue"])
processed_df = processed_df[processed_df["revenue"] >= MIN_REVENUE]
print(f"Отфильтровано по выручке >= 200 млн. Осталось: {len(processed_df)}")

# Приведение типов
if not processed_df.empty:
    processed_df["revenue_year"] = processed_df["revenue_year"].astype(int)
    processed_df["revenue"] = processed_df["revenue"].astype(int)
    print("Типы данных нормализованы.")

# --- 6. СОЗДАНИЕ ФИНАЛЬНОГО ФАЙЛА ---
print(f"--- [5/7] Сохранение финального датасета в {FINAL_DATA_PATH} ---")
final_columns_in_order = [
    # Обязательные поля
    "inn",
    "name",
    "revenue_year",
    "revenue",
    "segment_tag",
    "source",
    # Желательные поля
    "okved_main",
    "employees",
    "site",
    "description",
    "region",
    "contacts",  # Исходный столбец контактов
    "phone",  # Очищенный телефон
    "email",  # Очищенный email
    "rating_ref",
]
output_df = pd.DataFrame(columns=final_columns_in_order)
for col in final_columns_in_order:
    if col in processed_df.columns:
        output_df[col] = processed_df[col]
output_df = output_df.fillna("")

os.makedirs(os.path.dirname(FINAL_DATA_PATH), exist_ok=True)
output_df.to_csv(FINAL_DATA_PATH, index=False)
print(f"Файл успешно сохранен. Итоговое количество компаний: {len(output_df)}")

# --- 7. ЗАВЕРШЕНИЕ ---
print("--- [6/7] Скрипт успешно завершил работу! ---")
print("\nПервые 5 строк финального файла:")
print(output_df.head())
