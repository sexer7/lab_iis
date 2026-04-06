# ЛР1-ЛР2. Анализ данных и MLflow для предсказания цены автомобиля

## Описание проекта

Проект посвящён задаче регрессии: по характеристикам автомобиля нужно предсказать цену продажи `Selling_Price`.

На текущем этапе выполнены:

- ЛР1: разведочный анализ, очистка данных и сохранение подготовленного датасета.
- ЛР2: построение baseline-модели, генерация признаков, отбор признаков, тюнинг гиперпараметров, логирование экспериментов в MLflow и подготовка Production-модели для следующей лабораторной работы.

## Запуск

Ниже приведены команды для Windows PowerShell.

### 1. Клонирование репозитория

```powershell
git clone https://github.com/sexer7/lab_iis.git
cd lab_iis
```

### 2. Создание и активация виртуального окружения

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Установка зависимостей

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Подключение окружения как Jupyter kernel

```powershell
python -m ipykernel install --user --name lab_iis --display-name "lab_iis (.venv)"
```

### 5. Запуск ноутбуков проекта

```powershell
code .
```

После открытия проекта в VS Code:

1. Для ЛР1 откройте `eda/eda.ipynb`.
2. Для ЛР2 откройте `research/research.ipynb`.
3. Выберите kernel `lab_iis (.venv)`.
4. Запускайте ячейки по порядку.

Основной исследовательский сценарий ЛР2 выполняется через `research/research.ipynb`.
Файлы `research/*.py` оставлены как вспомогательные скрипты для воспроизводимого запуска тех же этапов без ноутбука.

### 6. Запуск MLflow

Скрипт запуска находится в [mlflow/start_mlflow.sh](C:/Users/mishu/Mohov/lab1/mlflow/start_mlflow.sh).

Команда для Git Bash:

```bash
cd mlflow
sh start_mlflow.sh
```

Эквивалентная команда для PowerShell:

```powershell
cd mlflow
python -m mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 127.0.0.1 --port 5000 --workers 1
```

После запуска интерфейс доступен по адресам:

- `http://127.0.0.1:5000`
- `http://localhost:5000/`

### 7. Выполнение ЛР2 через ноутбук

Основной путь выполнения ЛР2:

1. Запустите MLflow.
2. Откройте `research/research.ipynb`.
3. Выполняйте ячейки ноутбука последовательно:
   baseline-модель;
   модель с дополнительными признаками;
   forward-отбор признаков;
   подбор гиперпараметров;
   подготовка финальной Production-модели.

### 8. Скрипты для воспроизведения этапов ЛР2

Если нужно повторить отдельные шаги без ноутбука, можно использовать скрипты:

```powershell
python research/train_baseline.py
python research/train_featured_model.py
python research/train_selected_model.py
python research/train_tuned_model.py
python research/train_production_model.py
```

## Структура проекта

- `data/car_data.csv` - исходный датасет.
- `data/car_data_cleaned.csv` - очищенный датасет после EDA.
- `data/car_data_cleaned.pkl` - очищенный датасет в формате pickle.
- `eda/eda.ipynb` - ноутбук с разведочным анализом данных.
- `research/research.ipynb` - основной ноутбук с исследованиями и экспериментами ЛР2.
- `research/common.py` - общие функции для пайплайнов, метрик и служебных операций.
- `research/train_baseline.py` - baseline-модель.
- `research/train_featured_model.py` - модель с новыми признаками из `sklearn`.
- `research/train_selected_model.py` - модель с `mlxtend.SequentialFeatureSelector`.
- `research/train_tuned_model.py` - тюнинг гиперпараметров через `Optuna`.
- `research/train_production_model.py` - обучение Production-модели на всей выборке.
- `research/featured_feature_names.txt` - имена признаков после `fit_transform`.
- `research/selected_feature_names.txt` - имена признаков, выбранных SFS.
- `research/selected_feature_indices.txt` - индексы признаков, выбранных SFS.
- `research/tuning_trials.csv` - история trials при тюнинге.
- `research/best_tuned_params.txt` - лучшие найденные гиперпараметры.
- `mlflow/start_mlflow.sh` - запуск локального MLflow server.
- `requirements.txt` - актуальный список зависимостей.

## Результаты и выводы EDA

### Загрузка данных и знакомство с ними

- Исходный датасет содержит `301` запись и `9` столбцов.
- Числовые признаки: `Year`, `Present_Price`, `Driven_kms`.
- Категориальные признаки: `Car_Name`, `Fuel_Type`, `Selling_type`, `Transmission`, `Owner`.
- Целевая переменная: `Selling_Price`.
- В исходном датасете средняя цена продажи составляет `4.66`, медианная `3.60`, диапазон значений от `0.10` до `35.00`.

### Очистка данных

- удалены `2` полных дубликата
- проверены диапазоны значений для `Year`, `Driven_kms` и `Selling_Price`
- проверены пропуски, пропущенных значений не обнаружено
- признак `Owner` приведён к категориальному типу

### Полезные закономерности

- Наиболее сильная связь с целевой переменной наблюдается у `Present_Price`.
- Более новые автомобили в среднем продаются дороже.
- `Driven_kms` полезен в комбинации с другими признаками.
- Автомобили с автоматической коробкой в среднем дороже.
- Продажи через дилера в среднем дороже продаж от частных лиц.

## Результаты исследования

### Baseline

- модель: `StandardScaler + OrdinalEncoder + RandomForestRegressor`
- метрики: `MAE = 1.0808`, `MAPE = 0.3060`, `MSE = 7.2714`

### Модель с новыми признаками

- добавлены `PolynomialFeatures` и `KBinsDiscretizer`
- метрики: `MAE = 1.0590`, `MAPE = 0.3060`, `MSE = 6.8207`

### Лучшая модель по качеству

Лучший результат по целевой метрике `MAE` показал run `6d6108e1e17f425cb06afab37d7c53e0`.

Характеристики лучшей модели:

- модель: `RandomForestRegressor`
- параметры: `n_estimators = 200`, `max_depth = 12`, `min_samples_leaf = 2`
- preprocessing: `PolynomialFeatures + KBinsDiscretizer + OrdinalEncoder`
- отбор признаков: `mlxtend.SequentialFeatureSelector`
- метрики: `MAE = 1.0393`, `MAPE = 0.3062`, `MSE = 6.2757`

Выбранные столбцы:

- `num_scaled__Year`
- `num_poly__Year^2`
- `num_poly__Year Present_Price`
- `num_bins__Year_0.0`
- `num_bins__Year_1.0`
- `num_bins__Year_2.0`
- `num_bins__Year_3.0`
- `num_bins__Driven_kms_3.0`
- `cat__Car_Name`
- `cat__Fuel_Type`

### Тюнинг гиперпараметров

Для лучшей модели был выполнен тюнинг через `Optuna` по `10` trials.

- оптимизировались `n_estimators`, `max_depth`, `max_features`
- в коде явно указано `direction="minimize"`, потому что `MAE` для регрессии нужно минимизировать
- лучший CV trial: `best_cv_mae = 0.7661`
- лучшие параметры trial: `n_estimators = 51`, `max_depth = 15`, `max_features = 0.8190`

На тестовой выборке tuned-версия не улучшила `MAE` относительно лучшей SFS-модели, поэтому в Production взята именно модель из run `6d6108e1e17f425cb06afab37d7c53e0`.

### Production-модель

Лучшая модель была переобучена на всей очищенной выборке и зарегистрирована как Production-версия.

- production run_id: `6e05235bc5aa464db3d168d2249fbb42`
- зарегистрированная модель: `car-price-rf-featured-sfs`
- версия в реестре: `3`
- тэг версии: `status = Production`
- alias модели: `Production`

В Production-run залогированы:

- сигнатура модели
- пример входных данных
- `requirements.txt`
- список используемых столбцов

## Что важно поддерживать актуальным

- При добавлении новых библиотек необходимо обновлять `requirements.txt`.
- После изменения структуры проекта нужно актуализировать разделы `Запуск` и `Структура проекта`.
- README должен соответствовать текущему состоянию проекта на каждом коммите.
