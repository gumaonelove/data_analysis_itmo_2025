# Fraud EDA & Baseline — ITMO Task

Репозиторий для разведочного анализа данных и прототипирования базовой модели выявления мошенничества по датасету `transaction_fraud_data.parquet` с конвертацией валют по `historical_currency_exchange.parquet`.

## Цели
1. Провести EDA и выявить ключевые драйверы риска.
2. Нормализовать суммы транзакций до единой валюты (USD).
3. Сформулировать продуктовые и технические гипотезы и план их валидации.
4. Обучить базовую модель (baseline) и оценить её по метрикам ROC-AUC / PR-AUC / Recall@Precision.
5. Подготовить артефакты: отчёт, метрики, reproducible-пайплайн.

## Быстрый старт
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Положите файлы в data/
# data/transaction_fraud_data.parquet
# data/historical_currency_exchange.parquet

# Запуски
jupyter lab  # notebooks/01_eda.ipynb
python src/train_baseline.py  # отчёт метрик в reports/metrics.json
```

## Структура
```
fraud-eda-itmo-2025/
├── data/                                  # сюда положите parquet-файлы
├── notebooks/
│   └── 01_eda.ipynb                       # воспроизводимый EDA
├── src/
│   ├── features.py                        # преобразования и фичи
│   ├── currency.py                        # конвертация валют
│   └── train_baseline.py                  # baseline-модель + метрики
├── reports/
│   ├── REPORT.md                          # гипотезы, выводы EDA, план экспериментов
│   └── metrics.json                       # метрики модели
├── .github/workflows/ci.yml               # линт в CI (ruff)
├── .gitignore
├── Makefile
├── pyproject.toml
└── requirements.txt
```

## Ключевые гипотезы (кратко)
- **H1 (Product):** Скорость операций и cross-border транзакции повышают риск → шаг-up аутентификация при превышении порогов. **НЕ подтвердилось**
- **H2 (Product):** Редкие/новые устройства и отсутствующая карта (CNP) ↑ риск → device binding и лимиты. **Подтвердилось**
- **H3 (Tech):** Конвертация сумм в USD улучшает стабильность модели и переносимость порогов. **Подтвердилось**
- **H4 (Tech):** Графовые признаки по `device_fingerprint`/`ip_address`/`card_number` увеличивают PR-AUC за счёт выявления колец. **Подтвердилось**
- **H5 (Tech):** Калибровка вероятностей + пороги по требуемой точности (например, Precision≥0.9) снижает операционные издержки. **Подтвердилось**


Развёрнутые гипотезы и план валидации — в `reports/REPORT.md`.

## Метрики
- ROC-AUC, PR-AUC (основная), Precision@k, Recall@P, Lift@k.
- Бизнес-метрики: предотвращённые потери (USD), нагрузка на ручную проверку, среднее время ответа.

## Лицензия
MIT
