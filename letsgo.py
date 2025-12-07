import marimo

__generated_with = "0.18.3"
app = marimo.App()

# =======================
# ЯЧЕЙКА 1: Импорт библиотек
# =======================
@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
    from sklearn.cluster import KMeans
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.utils import shuffle
    
    return (
        mo, pd, np, train_test_split, StandardScaler, 
        KNeighborsRegressor, NearestNeighbors, KMeans,
        mean_squared_error, mean_absolute_error, shuffle
    )

# =======================
# ЯЧЕЙКА 2: Загрузка, очистка и подготовка данных
# =======================
@app.cell
def _(
    KMeans, KNeighborsRegressor, NearestNeighbors, StandardScaler, 
    mean_absolute_error, mean_squared_error, mo, np, pd, shuffle, train_test_split
):
    # Загрузка данных
    PATH = "all_v2.csv"
    df = pd.read_csv(PATH)
    print("Исходный датасет:", df.shape)
    
    # Проверяем наличие колонки "building_type"
    has_building_type = 'building_type' in df.columns
    print(f"Есть колонка 'building_type': {has_building_type}")
    
    # Проверяем регионы
    print("\n=== АНАЛИЗ РЕГИОНОВ ===")
    df['region'] = df['region'].astype(str).replace('nan', 'Неизвестно')
    print("Топ-15 регионов по количеству объявлений:")
    region_counts = df['region'].value_counts().head(15)
    
    # Создаем маппинг кодов регионов на названия
    region_code_to_name = {
        '77': 'Москва',
        '78': 'Санкт-Петербург',
        '50': 'Московская область',
        '23': 'Краснодарский край',
        '16': 'Республика Татарстан',
        '66': 'Свердловская область',
        '61': 'Ростовская область',
        '63': 'Самарская область',
        '52': 'Нижегородская область',
        '02': 'Республика Башкортостан',
        '26': 'Ставропольский край',
        '74': 'Челябинская область',
        '54': 'Новосибирская область',
        '55': 'Омская область',
        '56': 'Оренбургская область',
        '24': 'Красноярский край',
        '59': 'Пермский край',
        '72': 'Тюменская область',
        '33': 'Владимирская область',
        '36': 'Воронежская область',
    }
    
    # Берем топ-12 регионов
    top_regions = region_counts.head(12).index.tolist()
    
    # Создаем читабельные названия для UI
    region_display_names = {}
    region_code_mapping = {}  # Для обратного маппинга
    
    for i, region_code in enumerate(top_regions, 1):
        clean_region = str(region_code).strip()
        
        # Получаем название региона
        if clean_region in region_code_to_name:
            region_name = region_code_to_name[clean_region]
        else:
            # Если код не найден в маппинге, используем как есть
            region_name = f"Регион {clean_region}"
        
        # Создаем отображаемое имя
        display_name = f"{i} -- {region_name}"
        
        region_display_names[display_name] = region_code
        region_code_mapping[region_code] = region_name
        
        print(f"  {i:2d}. {display_name}: {region_counts[region_code]:,} объявлений")
    
    print(f"\nИспользуем {len(region_display_names)} регионов для анализа")
    
    # ========== ОЧИСТКА ДАННЫХ ==========
    df_clean = df.copy()
    df_clean = df_clean[df_clean['region'].isin(top_regions)]
    df_clean = df_clean[df_clean["price"].between(1_000_000, 40_000_000)]
    df_clean = df_clean[df_clean["area"].between(15, 200) & df_clean["kitchen_area"].between(4, 50)]
    df_clean = df_clean[df_clean["kitchen_area"] <= df_clean["area"]]
    
    # Комнаты
    df_clean["rooms"] = pd.to_numeric(df_clean["rooms"], errors='coerce')
    df_clean = df_clean.dropna(subset=["rooms"])
    df_clean["rooms"] = df_clean["rooms"].astype(int)
    df_clean = df_clean[df_clean["rooms"].between(1, 5)]
    
    # Этажи
    df_clean["level"] = pd.to_numeric(df_clean["level"], errors='coerce')
    df_clean["levels"] = pd.to_numeric(df_clean["levels"], errors='coerce')
    df_clean = df_clean.dropna(subset=["level", "levels"])
    df_clean["level"] = df_clean["level"].astype(int)
    df_clean["levels"] = df_clean["levels"].astype(int)
    
    df_clean = df_clean[
        (df_clean["level"] >= 1) &
        (df_clean["levels"] >= 1) &
        (df_clean["level"] <= df_clean["levels"]) &
        (df_clean["levels"] <= 40)
    ]
    
    # Площадь на комнату
    df_clean["area_per_room"] = np.where(
        df_clean["rooms"] > 0,
        df_clean["area"] / df_clean["rooms"],
        df_clean["area"]
    )
    df_clean = df_clean[df_clean["area_per_room"].between(10, 80)]
    
    # Важные поля без NaN
    key_cols = ["price","area","kitchen_area","rooms","level","levels","geo_lat","geo_lon","region"]
    df_clean = df_clean.dropna(subset=key_cols)
    print(f"\nПосле очистки: {df_clean.shape}")
    
    # ========== СОЗДАНИЕ ПРИЗНАКОВ ==========
    if 'date' in df_clean.columns:
        df_clean["date"] = pd.to_datetime(df_clean["date"], errors="coerce")
        df_clean = df_clean.dropna(subset=["date"])
        df_clean["day_of_year"] = df_clean["date"].dt.dayofyear
        df_clean["month"] = df_clean["date"].dt.month
        season_map = {12:1,1:1,2:1,3:2,4:2,5:2,6:3,7:3,8:3,9:4,10:4,11:4}
        df_clean["season"] = df_clean["month"].map(season_map).astype(int)
    else:
        df_clean["day_of_year"] = 180
        df_clean["season"] = 2
    
    df_clean["is_first"] = (df_clean["level"] == 1).astype(int)
    df_clean["is_last"] = (df_clean["level"] == df_clean["levels"]).astype(int)
    df_clean["building_height_log"] = np.log1p(df_clean["levels"])
    df_clean["area_log"] = np.log1p(df_clean["area"])
    df_clean["kitchen_ratio"] = df_clean["kitchen_area"] / df_clean["area"]
    
    # OHE регионов (используем коды регионов)
    df_clean["region"] = df_clean["region"].astype(str)
    df_model = pd.get_dummies(df_clean, columns=["region"], prefix="region", drop_first=False)
    
    # ========== ПОДГОТОВКА ДЛЯ МОДЕЛИ ==========
    target = "price"
    base_numeric_features = [
        "geo_lat","geo_lon","level","levels","is_first","is_last",
        "area","kitchen_area","area_per_room","kitchen_ratio",
        "building_height_log","area_log","day_of_year","season"
    ]
    
    region_features = [c for c in df_model.columns if c.startswith("region_")]
    feature_cols = base_numeric_features + region_features
    
    X_all = df_model[feature_cols].reset_index(drop=True)
    y_all = np.log1p(df_model[target].values)
    print(f"Размер X_all: {X_all.shape}")
    print(f"Количество признаков: {len(feature_cols)}")
    
    if len(X_all) == 0:
        raise ValueError("После обработки данных не осталось записей!")
    
    # ========== РАЗДЕЛЕНИЕ ДАННЫХ ==========
    X_all_shuffled, y_all_shuffled = shuffle(X_all, y_all, random_state=42)
    MAX_ROWS = min(400_000, len(X_all_shuffled))
    X_small = X_all_shuffled[:MAX_ROWS]
    y_small = y_all_shuffled[:MAX_ROWS]
    
    X_train_base, X_valid_base, y_train, y_valid = train_test_split(
        X_small, y_small, test_size=0.2, random_state=42
    )
    print(f"Train: {X_train_base.shape}, Valid: {X_valid_base.shape}")
    
    # ========== ГЕО-ФИЧИ ==========
    coords_train = X_train_base[["geo_lat","geo_lon"]].values
    coords_valid = X_valid_base[["geo_lat","geo_lon"]].values
    
    N_CLUSTERS = min(50, len(X_train_base) // 100)
    if N_CLUSTERS > 1:
        kmeans_geo = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
        kmeans_geo.fit(coords_train)
        X_train_base["geo_cluster"] = kmeans_geo.labels_
        X_valid_base["geo_cluster"] = kmeans_geo.predict(coords_valid)
    else:
        X_train_base["geo_cluster"] = 0
        X_valid_base["geo_cluster"] = 0
    
    if len(X_train_base) > 10:
        knn_geo = NearestNeighbors(n_neighbors=min(10, len(X_train_base)))
        knn_geo.fit(coords_train)
        dist_train, _ = knn_geo.kneighbors(coords_train)
        dist_valid, _ = knn_geo.kneighbors(coords_valid)
        X_train_base["geo_density"] = 1.0 / (dist_train.mean(axis=1) + 1e-6)
        X_valid_base["geo_density"] = 1.0 / (dist_valid.mean(axis=1) + 1e-6)
    else:
        X_train_base["geo_density"] = 1.0
        X_valid_base["geo_density"] = 1.0
    
    feature_cols_extended = feature_cols + ["geo_cluster","geo_density"]
    X_train = X_train_base[feature_cols_extended].copy()
    X_valid = X_valid_base[feature_cols_extended].copy()
    print(f"Добавлены гео-признаки. Всего признаков: {len(feature_cols_extended)}")
    
    # ========== МАСШТАБИРОВАНИЕ ==========
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    
    # ========== ОБУЧЕНИЕ МОДЕЛИ ==========
    best_params = {"n_neighbors": min(15, len(X_train_scaled) // 10), "weights": "distance", "p": 1}
    best_knn = KNeighborsRegressor(**best_params, metric="minkowski", n_jobs=-1)
    best_knn.fit(X_train_scaled, y_train)
    print("Модель KNN обучена")
    
    # ========== МЕТРИКИ ==========
    y_val_pred_log = best_knn.predict(X_valid_scaled)
    y_val_true = np.expm1(y_valid)
    y_val_pred = np.expm1(y_val_pred_log)
    
    rmse = np.sqrt(mean_squared_error(y_val_true, y_val_pred))
    mae = mean_absolute_error(y_val_true, y_val_pred)
    mape = np.mean(np.abs(y_val_true - y_val_pred) / y_val_true)
    
    print(f"RMSE по цене (рубли): {rmse:,.0f}")
    print(f"MAE  по цене (рубли): {mae:,.0f}")
    print(f"MAPE по цене: {mape:.2%}")
    
    # ========== СОБИРАЕМ СТАТИСТИКУ ПО РЕГИОНАМ ==========
    region_medians = {}
    for display_name, region_code in region_display_names.items():
        region_name = region_code
        region_col = f"region_{region_name}"
        
        if region_col in df_model.columns:
            region_mask = df_model[region_col] == 1
            
            if region_mask.any():
                region_data = df_model[region_mask]
                region_medians[region_name] = {
                    'display_name': display_name,
                    'region_name': region_code_mapping.get(region_code, f"Регион {region_code}"),
                    'geo_lat': float(region_data['geo_lat'].median()),
                    'geo_lon': float(region_data['geo_lon'].median()),
                    'area': float(region_data['area'].median()),
                    'kitchen_area': float(region_data['kitchen_area'].median()),
                    'rooms': float(region_data['rooms'].median()),
                    'level': float(region_data['level'].median()),
                    'levels': float(region_data['levels'].median()),
                    'price': float(region_data['price'].median()),
                    'kitchen_ratio': float(region_data['kitchen_area'].median() / region_data['area'].median() 
                                          if region_data['area'].median() > 0 else 0.15),
                    'count': int(region_mask.sum())
                }
    
    print(f"\nСобрана статистика по {len(region_medians)} регионам")
    
    return (
        best_knn, feature_cols_extended, has_building_type,
        mae, mape, mo, np, pd, rmse, scaler,
        region_display_names, region_medians, region_code_mapping
    )

# =======================
# ЯЧЕЙКА 3: Создание интерфейса фильтров
# =======================
@app.cell
def _(mo, region_display_names, region_medians):
    # Берем медианы из первого региона
    first_region_key = list(region_medians.keys())[0] if region_medians else None
    default_values = region_medians.get(first_region_key, {}) if first_region_key else {}
    
    # Основные фичи с разумными диапазонами
    feature_configs = {
        'area': {'min': 30, 'max': 150, 'step': 1, 'label': 'Площадь (м²)'},
        'kitchen_area': {'min': 6, 'max': 20, 'step': 0.5, 'label': 'Кухня (м²)'},
        'rooms': {'min': 1, 'max': 4, 'step': 1, 'label': 'Комнат'},
        'level': {'min': 1, 'max': 25, 'step': 1, 'label': 'Этаж'},
        'levels': {'min': 5, 'max': 25, 'step': 1, 'label': 'Этажей в доме'},
    }
    
    # Создаем слайдеры
    filters = {}
    
    for feature, config in feature_configs.items():
        # Берем значение из медиан или из конфига
        if feature in default_values:
            initial_value = default_values[feature]
            initial_value = max(config['min'], min(config['max'], initial_value))
        else:
            initial_value = (config['min'] + config['max']) / 2
        
        # Округляем целочисленные значения
        if feature in ['rooms', 'level', 'levels']:
            initial_value = int(round(initial_value))
        
        # Создаем слайдер
        filters[feature] = mo.ui.slider(
            start=config['min'],
            stop=config['max'],
            step=config['step'],
            value=initial_value,
            label=config['label']
        )
    
    # Выпадающий список для региона (с красивыми названиями)
    if region_display_names and len(region_display_names) > 0:
        filters['region'] = mo.ui.dropdown(
            options=list(region_display_names.keys()),
            value=list(region_display_names.keys())[0],
            label="Регион"
        )
    
    # Сезон продаж
    season_options = {
        "1 -- Зима (дек-фев)": 1,
        "2 -- Весна (мар-май)": 2,
        "3 -- Лето (июн-авг)": 3,
        "4 -- Осень (сен-ноя)": 4
    }
    
    filters['season'] = mo.ui.dropdown(
        options=list(season_options.keys()),
        value=list(season_options.keys())[1],
        label="Сезон продажи"
    )
    
    # РАДИОКНОПКИ для этажа (без надписи)
    floor_options = [
        ("Обычный этаж", "regular"),
        ("Первый этаж", "first"),
        ("Последний этаж", "last")
    ]
    
    # Убираем label для радиокнопок
    filters['floor_type'] = mo.ui.radio(
        options=[opt[0] for opt in floor_options],
        value="Обычный этаж"
        # Убрали label
    )
    
    # Кнопка для предсказания
    predict_button = mo.ui.button(label="Рассчитать цену", kind="success")
    
    return (
        filters, floor_options, predict_button, 
        region_display_names, season_options
    )

# =======================
# ЯЧЕЙКА 4: Функция для расчета цены с ВАЛИДАЦИЕЙ
# =======================
@app.cell
def _(best_knn, feature_cols_extended, filters, floor_options, mo, np, pd, scaler, season_options, region_display_names):
    
    # Функция для создания входных данных
    def create_input_data():
        # Получаем значения из UI
        region_display = None
        season_value = None
        floor_type_value = "regular"
        
        if 'region' in filters:
            region_display = filters['region'].value
        
        if 'season' in filters:
            selected_season_option = filters['season'].value
            season_value = season_options.get(selected_season_option)
        
        if 'floor_type' in filters:
            selected_floor_option = filters['floor_type'].value
            for display_name, value in floor_options:
                if display_name == selected_floor_option:
                    floor_type_value = value
                    break
        
        # Получаем числовые значения
        area_val = float(filters['area'].value) if 'area' in filters else 70.0
        kitchen_val = float(filters['kitchen_area'].value) if 'kitchen_area' in filters else 12.0
        rooms_val = int(filters['rooms'].value) if 'rooms' in filters else 2
        level_val = int(filters['level'].value) if 'level' in filters else 5
        levels_val = int(filters['levels'].value) if 'levels' in filters else 10
        
        # ВАЛИДАЦИЯ: проверяем логические ошибки
        validation_errors = []
        
        # 1. Кухня не может быть больше общей площади
        if kitchen_val > area_val:
            validation_errors.append(f"❌ Кухня ({kitchen_val:.1f} м²) больше общей площади ({area_val:.1f} м²)")
        
        # 2. Этаж не может быть больше этажности
        if level_val > levels_val:
            validation_errors.append(f"❌ Этаж ({level_val}) больше этажности ({levels_val})")
        
        # 3. Кухня не может быть слишком маленькой (меньше 4 м²)
        if kitchen_val < 4:
            validation_errors.append(f"❌ Кухня слишком маленькая ({kitchen_val:.1f} м²)")
        
        # 4. Комнат не может быть 0
        if rooms_val < 1:
            validation_errors.append(f"❌ Количество комнат должно быть не менее 1")
        
        if validation_errors:
            print("\n".join(validation_errors))
            return None  # Возвращаем None если есть ошибки
        
        # Автоматический расчет пропорций
        kitchen_ratio_val = 0.15
        area_per_room_val = 30.0
        
        if area_val > 0:
            kitchen_ratio_val = kitchen_val / area_val
            kitchen_ratio_val = max(0.08, min(0.3, kitchen_ratio_val))
        
        if rooms_val > 0:
            area_per_room_val = area_val / rooms_val
            area_per_room_val = max(15, min(50, area_per_room_val))
        
        # Определяем регион
        region_code = None
        if region_display:
            # Извлекаем код региона из display name
            if region_display in region_display_names:
                region_code = region_display_names[region_display]
        
        # Создаем базовый словарь со всеми признаками
        input_data = {}
        for col in feature_cols_extended:
            input_data[col] = 0
        
        # Устанавливаем основные значения
        input_data['area'] = area_val
        input_data['kitchen_area'] = kitchen_val
        input_data['rooms'] = rooms_val
        input_data['level'] = level_val
        input_data['levels'] = levels_val
        input_data['kitchen_ratio'] = kitchen_ratio_val
        input_data['area_per_room'] = area_per_room_val
        
        # Гео-координаты (средние)
        input_data['geo_lat'] = 55.75
        input_data['geo_lon'] = 37.61
        
        # Сезон
        if season_value:
            input_data['season'] = season_value
        else:
            input_data['season'] = 2
        
        # День года
        input_data['day_of_year'] = 180
        
        # Производные признаки
        input_data['building_height_log'] = np.log1p(levels_val)
        input_data['area_log'] = np.log1p(area_val)
        
        # Фичи для этажа
        input_data['is_first'] = 0
        input_data['is_last'] = 0
        
        if floor_type_value == "first":
            input_data['level'] = 1
            input_data['is_first'] = 1
        elif floor_type_value == "last":
            input_data['level'] = levels_val
            input_data['is_last'] = 1
        
        # Гео-фичи
        input_data['geo_cluster'] = 0
        input_data['geo_density'] = 1.0
        
        # Устанавливаем регион
        if region_code:
            region_col_name = f"region_{region_code}"
            if region_col_name in feature_cols_extended:
                input_data[region_col_name] = 1
        
        return input_data
    
    # Функция для расчета цены
    def calculate_price():
        try:
            # Получаем входные данные
            input_data = create_input_data()
            
            # Если есть ошибки валидации, возвращаем 0
            if input_data is None:
                print("⛔ Возвращаем 0 рублей из-за ошибок валидации")
                return 0
            
            # Создаем DataFrame с правильным порядком колонок
            input_df = pd.DataFrame([input_data])
            
            # Убедимся, что все колонки присутствуют в правильном порядке
            missing_cols = set(feature_cols_extended) - set(input_df.columns)
            if missing_cols:
                for col in missing_cols:
                    input_df[col] = 0
            
            # Сортируем колонки в правильном порядке
            input_df = input_df[feature_cols_extended]
            
            # Проверяем порядок
            if list(input_df.columns) != feature_cols_extended:
                print("Предупреждение: порядок колонок не совпадает!")
                input_df = input_df.reindex(columns=feature_cols_extended, fill_value=0)
            
            # Масштабируем и предсказываем
            input_scaled = scaler.transform(input_df)
            y_pred_log = best_knn.predict(input_scaled)[0]
            price_pred = np.expm1(y_pred_log)
            
            # Выводим информацию
            rooms_val = input_data['rooms']
            area_val = input_data['area']
            kitchen_val = input_data['kitchen_area']
            level_val = input_data['level']
            levels_val = input_data['levels']
            
            print(f"✓ Расчет: {rooms_val}к, {area_val:.0f}м², кухня {kitchen_val:.1f}м², {level_val}/{levels_val} эт. → {price_pred:,.0f}₽")
            
            return price_pred
        
        except Exception as e:
            print(f"✗ Ошибка при расчете: {e}")
            import traceback
            traceback.print_exc()
            return 0
    
    return calculate_price, create_input_data

# =======================
# ЯЧЕЙКА 5: Создание интерфейса с ОПИСАНИЕМ РЕГИОНОВ
# =======================
@app.cell
def _(calculate_price, filters, mo, predict_button, region_display_names):
    # Функция для расчета пропорций
    def calculate_proportions():
        try:
            area_val = float(filters['area'].value) if 'area' in filters else 0
            kitchen_val = float(filters['kitchen_area'].value) if 'kitchen_area' in filters else 0
            rooms_val = int(filters['rooms'].value) if 'rooms' in filters else 1
            
            kitchen_ratio = 0
            area_per_room = 0
            
            if area_val > 0:
                kitchen_ratio = kitchen_val / area_val
                kitchen_ratio = max(0.08, min(0.3, kitchen_ratio))
            
            if rooms_val > 0:
                area_per_room = area_val / rooms_val
            
            return kitchen_ratio, area_per_room
        except:
            return 0.15, 30.0
    
    # Функция для проверки валидации
    def check_validation():
        try:
            area_val = float(filters['area'].value) if 'area' in filters else 0
            kitchen_val = float(filters['kitchen_area'].value) if 'kitchen_area' in filters else 0
            level_val = int(filters['level'].value) if 'level' in filters else 0
            levels_val = int(filters['levels'].value) if 'levels' in filters else 0
            rooms_val = int(filters['rooms'].value) if 'rooms' in filters else 0
            
            errors = []
            
            if kitchen_val > area_val:
                errors.append(f"❌ Кухня больше общей площади!")
            
            if level_val > levels_val:
                errors.append(f"❌ Этаж больше этажности!")
            
            if kitchen_val < 4:
                errors.append(f"❌ Кухня слишком маленькая!")
            
            if rooms_val < 1:
                errors.append(f"❌ Должна быть хотя бы 1 комната!")
            
            return errors
        except:
            return []
    
    # Создаем текст с описанием регионов
    def create_regions_description():
        if not region_display_names:
            return ""
        
        description_lines = ["### 📍 Используемые регионы:"]
        for display_name in region_display_names.keys():
            description_lines.append(f"- {display_name}")
        
        return "\n".join(description_lines)
    
    # Простая функция для создания интерфейса
    def create_prediction_interface():
        # Получаем текущую цену
        current_price = calculate_price()
        
        # Получаем пропорции
        kitchen_ratio, area_per_room = calculate_proportions()
        
        # Проверяем валидацию
        validation_errors = check_validation()
        
        # Получаем описание регионов
        regions_description = create_regions_description()
        
        # Формируем интерфейс
        interface_elements = [
            mo.md("# 🏠 Калькулятор стоимости квартиры"),
            mo.md("### Настройте параметры квартиры:"),
        ]
        
        # Основные параметры
        interface_elements.append(mo.md("#### 📏 Основные параметры:"))
        
        # Первая строка: площадь и кухня
        row1 = []
        if 'area' in filters:
            row1.append(filters['area'])
        if 'kitchen_area' in filters:
            row1.append(filters['kitchen_area'])
        if 'rooms' in filters:
            row1.append(filters['rooms'])
        
        if row1:
            interface_elements.append(mo.hstack(row1, gap=2, justify="start"))
        
        # Вторая строка: этажи
        row2 = []
        if 'level' in filters:
            row2.append(filters['level'])
        if 'levels' in filters:
            row2.append(filters['levels'])
        
        if row2:
            interface_elements.append(mo.hstack(row2, gap=2, justify="start"))
        
        # Автоматически рассчитываемые значения
        interface_elements.append(mo.md("#### 🔢 Автоматически рассчитывается:"))
        
        info_row = [
            mo.md(f"**Доля кухни:** {kitchen_ratio:.2f} ({kitchen_ratio*100:.0f}%)"),
            mo.md(f"**Площадь на комнату:** {area_per_room:.1f} м²")
        ]
        
        interface_elements.append(mo.hstack(info_row, gap=2, justify="start"))
        
        # Регион и сезон
        interface_elements.append(mo.md("#### 📍 Расположение и время:"))
        location_widgets = []
        if 'region' in filters:
            location_widgets.append(filters['region'])
        if 'season' in filters:
            location_widgets.append(filters['season'])
        if location_widgets:
            interface_elements.append(mo.hstack(location_widgets, gap=2, justify="start"))
        
        # Тип этажа (без заголовка)
        if 'floor_type' in filters:
            interface_elements.append(mo.md("#### 🏢 Тип этажа:"))
            interface_elements.append(mo.hstack([filters['floor_type']], justify="start"))
        
        # Описание регионов (вместо заголовка "Тип этажа")
        interface_elements.append(mo.md(regions_description))
        
        # Кнопка и результат
        interface_elements.extend([
            mo.md("---"),
            mo.hstack([predict_button], justify="center"),
        ])
        
        # Показываем ошибки валидации если они есть
        if validation_errors:
            interface_elements.append(mo.md("#### ⚠️ Ошибки валидации:"))
            for error in validation_errors:
                interface_elements.append(mo.md(f"- {error}"))
            interface_elements.append(mo.md("**Цена будет 0 рублей пока ошибки не исправлены**"))
        
        interface_elements.extend([
            mo.md("---"),
            mo.md(f"## 💰 Прогнозируемая цена:"),
            mo.md(f"# **{current_price:,.0f} ₽**"),
            mo.md(f"*Примерно {current_price/1000000:.1f} млн рублей*"),
        ])
        
        return mo.vstack(interface_elements, gap=2)
    
    # Просто возвращаем интерфейс
    prediction_interface = create_prediction_interface()
    
    return create_prediction_interface, prediction_interface

# =======================
# ЯЧЕЙКА 6: Выбор фичи для анализа
# =======================
@app.cell
def _(filters, mo):
    # Выбор фичи для анализа (только числовые слайдеры)
    numeric_features = [f for f, w in filters.items() 
                      if hasattr(w, 'start') and hasattr(w, 'stop')]
    
    feature_selector = mo.ui.dropdown(
        options=numeric_features,
        value=numeric_features[0] if numeric_features else None,
        label="Выберите признак для анализа влияния"
    )
    
    return feature_selector, numeric_features

# =======================
# ЯЧЕЙКА 7: РАБОЧИЙ ГРАФИК ВЛИЯНИЯ ПРИЗНАКА НА ЦЕНУ (ИСПРАВЛЕННЫЙ)
# =======================
@app.cell
def _(best_knn, feature_cols_extended, filters, floor_options, mo, np, pd, scaler, season_options, region_display_names):
    
    def create_feature_impact_plot(feature_name):
        import plotly.graph_objects as go
        
        if feature_name not in filters:
            return mo.md(f"Признак '{feature_name}' не найден в фильтрах")
        
        # Получаем текущие значения фильтров
        widget = filters[feature_name]
        
        # Получаем остальные значения из UI
        region_display = None
        season_value = None
        floor_type_value = "regular"
        
        if 'region' in filters:
            region_display = filters['region'].value
        
        if 'season' in filters:
            selected_season_option = filters['season'].value
            season_value = season_options.get(selected_season_option)
        
        if 'floor_type' in filters:
            selected_floor_option = filters['floor_type'].value
            for display_name, value in floor_options:
                if display_name == selected_floor_option:
                    floor_type_value = value
                    break
        
        # Получаем остальные числовые значения ИЗ ТЕКУЩИХ ФИЛЬТРОВ
        # (а не фиксированные значения)
        area_val = float(filters['area'].value) if 'area' in filters else 70.0
        kitchen_val = float(filters['kitchen_area'].value) if 'kitchen_area' in filters else 12.0
        rooms_val = int(filters['rooms'].value) if 'rooms' in filters else 2
        level_val = int(filters['level'].value) if 'level' in filters else 5
        levels_val = int(filters['levels'].value) if 'levels' in filters else 10
        
        # Получаем диапазон значений для графика
        start_val = float(widget.start)
        stop_val = float(widget.stop)
        current_val = float(widget.value)
        
        # Генерируем 8 точек для графика
        values = np.linspace(start_val, stop_val, 8)
        
        # Сохраняем цены
        prices = []
        
        # Для каждой точки строим прогноз
        for val in values:
            try:
                # Создаем базовый словарь со всеми признаками
                input_data = {}
                for col in feature_cols_extended:
                    input_data[col] = 0
                
                # Устанавливаем основные значения (кроме текущего признака)
                # ИСПРАВЛЕНИЕ: используем значения из фильтров для каждого признака
                input_data['area'] = area_val if feature_name != 'area' else val
                input_data['kitchen_area'] = kitchen_val if feature_name != 'kitchen_area' else val
                input_data['rooms'] = rooms_val if feature_name != 'rooms' else int(val)
                input_data['level'] = level_val if feature_name != 'level' else int(val)
                input_data['levels'] = levels_val if feature_name != 'levels' else int(val)
                
                # Автоматический расчет пропорций
                kitchen_ratio_val = 0.15
                area_per_room_val = 30.0
                
                if input_data['area'] > 0:
                    kitchen_ratio_val = input_data['kitchen_area'] / input_data['area']
                    kitchen_ratio_val = max(0.08, min(0.3, kitchen_ratio_val))
                
                if input_data['rooms'] > 0:
                    area_per_room_val = input_data['area'] / input_data['rooms']
                    area_per_room_val = max(15, min(50, area_per_room_val))
                
                input_data['kitchen_ratio'] = kitchen_ratio_val
                input_data['area_per_room'] = area_per_room_val
                
                # Гео-координаты (средние)
                input_data['geo_lat'] = 55.75
                input_data['geo_lon'] = 37.61
                
                # Сезон
                if season_value:
                    input_data['season'] = season_value
                else:
                    input_data['season'] = 2
                
                # День года
                input_data['day_of_year'] = 180
                
                # Производные признаки
                input_data['building_height_log'] = np.log1p(input_data['levels'])
                input_data['area_log'] = np.log1p(input_data['area'])
                
                # Фичи для этажа
                input_data['is_first'] = 0
                input_data['is_last'] = 0
                
                if floor_type_value == "first":
                    input_data['level'] = 1
                    input_data['is_first'] = 1
                elif floor_type_value == "last":
                    input_data['level'] = input_data['levels']
                    input_data['is_last'] = 1
                
                # Гео-фичи
                input_data['geo_cluster'] = 0
                input_data['geo_density'] = 1.0
                
                # Устанавливаем регион
                if region_display and region_display in region_display_names:
                    region_code = region_display_names[region_display]
                    region_col_name = f"region_{region_code}"
                    if region_col_name in feature_cols_extended:
                        input_data[region_col_name] = 1
                
                # Создаем DataFrame
                input_df = pd.DataFrame([input_data])
                
                # Убедимся, что все колонки присутствуют
                for col in feature_cols_extended:
                    if col not in input_df.columns:
                        input_df[col] = 0
                
                # Сортируем колонки
                input_df = input_df[feature_cols_extended]
                
                # Масштабируем и предсказываем
                input_scaled = scaler.transform(input_df)
                y_pred_log = best_knn.predict(input_scaled)[0]
                price_pred = np.expm1(y_pred_log)
                
                prices.append(price_pred)
                
            except Exception as e:
                print(f"Ошибка при расчете для {feature_name}={val:.1f}: {e}")
                prices.append(0)
        
        # Создаем график
        fig = go.Figure()
        
        # Убираем точки, где цена = 0 (ошибки)
        valid_indices = [i for i, price in enumerate(prices) if price > 0]
        valid_values = [values[i] for i in valid_indices]
        valid_prices = [prices[i] for i in valid_indices]
        
        if len(valid_values) > 1:
            fig.add_trace(go.Scatter(
                x=valid_values,
                y=valid_prices,
                mode='lines+markers',
                name='Зависимость цены',
                line=dict(color='#3498db', width=3),
                marker=dict(size=8, color='#2980b9')
            ))
        
        # Добавляем текущее значение
        if len(values) > 0 and len(prices) > 0:
            # Находим индекс ближайшего значения
            idx = np.abs(values - current_val).argmin()
            if idx < len(prices) and prices[idx] > 0:
                current_price = prices[idx]
                
                fig.add_trace(go.Scatter(
                    x=[current_val],
                    y=[current_price],
                    mode='markers',
                    name='Текущее значение',
                    marker=dict(size=15, color='#e74c3c', symbol='circle')
                ))
        
        # Настройки графика
        fig.update_layout(
            title=f"Влияние '{getattr(widget, 'label', feature_name)}' на стоимость квартиры",
            xaxis_title=getattr(widget, 'label', feature_name),
            yaxis_title="Цена (рубли)",
            template="plotly_white",
            height=500,
            showlegend=True,
            hovermode='x unified'
        )
        
        # Форматирование оси Y
        fig.update_yaxes(tickformat=",.0f")
        
        # Проверяем, есть ли данные для отображения
        if len(valid_values) < 2:
            return mo.md(f"**Не удалось построить график для '{feature_name}'.** Возможно, есть проблемы с данными.")
        
        return mo.ui.plotly(fig)
    
    return create_feature_impact_plot,

# =======================
# ЯЧЕЙКА 8: Статистика данных
# =======================
@app.cell
def _(mae, mape, mo, rmse):
    # Интерфейс статистики
    statistics_interface = mo.vstack([
        mo.md("# 📊 Статистика данных"),
        mo.md("### Общая информация:"),
        mo.md(f"- **RMSE модели:** {rmse:,.0f} ₽"),
        mo.md(f"- **MAE модели:** {mae:,.0f} ₽"),
        mo.md(f"- **MAPE модели:** {mape:.2%}"),
        mo.md("### Особенности модели:"),
        mo.md("- Используется K-Nearest Neighbors Regressor"),
        mo.md("- 12 основных регионов России"),
        mo.md("- Учет площади, комнат, этажа и сезона"),
        mo.md("- Автоматический расчет пропорций"),
        mo.md("### Правила валидации:"),
        mo.md("- Кухня не может быть больше общей площади"),
        mo.md("- Этаж не может быть больше этажности"),
        mo.md("- Кухня не может быть меньше 4 м²"),
        mo.md("- Должна быть хотя бы 1 комната"),
        mo.md("- При нарушении правил цена = 0 ₽")
    ])
    
    return statistics_interface,

# =======================
# ЯЧЕЙКА 9: Основной интерфейс с табами
# =======================
@app.cell
def _(create_feature_impact_plot, feature_selector, mae, mape, mo, prediction_interface, rmse, statistics_interface):
    # Создаем график для выбранного признака
    plot_display = create_feature_impact_plot(feature_selector.value)
    
    # Создаем табы для разных разделов
    tabs = mo.ui.tabs({
        "🎯 Предсказание цены": prediction_interface,
        "📈 Анализ влияния": mo.vstack([
            mo.md("## 📈 Анализ влияния отдельных признаков на цену"),
            mo.md("Выберите признак, чтобы увидеть как его изменение влияет на стоимость квартиры:"),
            feature_selector,
            mo.md("---"),
            plot_display
        ]),
        "📊 Статистика": statistics_interface,
        "ℹ️ О модели": mo.vstack([
            mo.md("# ℹ️ Информация о модели"),
            mo.md("### Используемая модель: K-Nearest Neighbors Regressor"),
            mo.md("**Параметры модели:**"),
            mo.md("- n_neighbors: 15"),
            mo.md("- weights: distance"),
            mo.md("- metric: minkowski (p=1)"),
            mo.md("### Метрики качества:"),
            mo.md(f"- RMSE: {rmse:,.0f} ₽"),
            mo.md(f"- MAE: {mae:,.0f} ₽"),
            mo.md(f"- MAPE: {mape:.2%}"),
            mo.md("### Особенности:"),
            mo.md("- Используется масштабирование признаков (StandardScaler)"),
            mo.md("- Добавлены гео-признаки: кластеры и плотность"),
            mo.md("- One-Hot Encoding для 12 стандартных регионов"),
            mo.md("- Логарифмирование цены для нормализации распределения"),
        ])
    })
    
    # Главный интерфейс
    main_interface = mo.vstack([
        mo.md("# 🏢 Интерактивный калькулятор стоимости квартир"),
        mo.md("Используйте фильтры для расчета стоимости квартиры"),
        mo.md("---"),
        tabs
    ])
    
    return main_interface, tabs

# =======================
# ЯЧЕЙКА 10: Отображение интерфейса
# =======================
@app.cell
def _(main_interface, mo):
    # Отображаем основной интерфейс
    mo.vstack([
        main_interface,
        mo.md("---"),
        mo.md("*Приложение создано с использованием Marimo*")
    ])
