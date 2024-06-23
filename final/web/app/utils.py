import numpy as np
import pandas as pd


def transform_feature(le, values, values_type=None):
    """Кодирует новые значения, используя обученный LabelEncoder.

    Аргументы:
    le -- обученный LabelEncoder
    values -- значения для кодирования
    values_type -- тип значений, в который необходимо привести значения перед кодированием (например, str или int).
                Если None, то преобразование не выполняется.

    Возвращает:
    Закодированные значения
    """
    if values_type:
        values = values.astype(values_type)
    classes = le.classes_
    new_values = np.setdiff1d(values, classes)
    if new_values.size > 0:
        # обработка новых значений, не наблюдаемых во время обучения
        le.classes_ = np.append(classes, new_values)
    return le.transform(values)


def convert_status(val):
    """Преобразует значение статуса недвижимости в одну из категорий.

    Аргументы:
    val -- исходное значение статуса недвижимости

    Возвращает:
    Преобразованное значение статуса недвижимости
    """
    if val is None:
        return None

    val = str(val).lower()

    if val == 'nan':
        return None
    if 'under contract' in val:
        return 'under_contract'
    if 'for sale' in val or 'active' in val or val == 'for_sale':
        return 'for_sale'
    if 'auction' in val:
        return 'auction'
    if 'new' in val:
        return 'new'
    if 'foreclosure' in val or val == 'foreclosed':
        return 'foreclosure'
    if 'pending' in val or val == 'p':
        return 'pending'

    return 'other'


def convert_property_type(val):
    """Преобразует значение типа недвижимости в одну из категорий.

    Аргументы:
    val -- исходное значение типа недвижимости

    Возвращает:
    Преобразованное значение типа недвижимости
    """
    if val is None:
        return 'unknown'

    val = str(val).lower()

    if val == 'nan':
        return 'unknown'
    if 'single' in val:
        return 'single'
    if 'condo' in val:
        return 'condo'
    if 'multi' in val:
        return 'multi'
    if 'coop' in val:
        return 'coop'
    if 'land' in val:
        return 'land'
    if 'traditional' in val:
        return 'traditional'
    if 'townhouse' in val:
        return 'townhouse'

    return 'other'


def prepare_data(df, le_zipcode, le_state, onehot_encoder):
    """Готовит данные для предсказания, включая преобразования категориальных признаков.

    Аргументы:
    df -- исходный DataFrame с данными
    le_zipcode -- обученный LabelEncoder для zipcode
    le_state -- обученный LabelEncoder для state
    onehot_encoder -- обученный OneHotEncoder для категориальных признаков

    Возвращает:
    DataFrame с преобразованными признаками, готовыми для предсказания
    """
    data = df.copy()

    if 'sqft' in data:
        log_columns = ['sqft', 'average_school_distance', 'num_schools']
        for column in log_columns:
            data[f'{column}_log'] = np.log1p(data[column])
        data.drop(['sqft', 'average_school_distance', 'num_schools'], axis=1, inplace=True)

    data['state'] = transform_feature(le_state, data['state'], None)
    data['zipcode'] = transform_feature(le_zipcode, data['zipcode'], int)

    # Применяем OneHotEncoder к признакам status и propertyType
    if 'status' in data:
        # Применяем преобразование `status`
        data['status'] = data['status'].apply(convert_status)
        data['status'] = data['status'].fillna('unknown')
        data['status'] = data['status'].astype('category')

        # Применяем преобразование `propertyType`
        data['propertyType'] = data['propertyType'].apply(convert_property_type)
        data['propertyType'] = data['propertyType'].astype('category')

        categorical_columns = ['status', 'propertyType']
        encoded_new_data = onehot_encoder.transform(data[categorical_columns])
        encoded_columns = onehot_encoder.get_feature_names_out(categorical_columns)
        encoded_df = pd.DataFrame(encoded_new_data, columns=encoded_columns)

        # Проверка и добавление столбцов для неизвестных категорий
        if 'status_unknown' not in encoded_columns:
            encoded_df['status_unknown'] = 0
        if 'propertyType_unknown' not in encoded_columns:
            encoded_df['propertyType_unknown'] = 0

        # Обработка неизвестных категорий в status
        for index, row in data.iterrows():
            if row['status'] not in onehot_encoder.categories_[0]:
                encoded_df.loc[index, 'status_unknown'] = 1
                encoded_df.loc[index, [col for col in encoded_df.columns if col.startswith('status_') and col != 'status_unknown']] = 0

        # Обработка неизвестных категорий в propertyType
        for index, row in data.iterrows():
            if row['propertyType'] not in onehot_encoder.categories_[1]:
                encoded_df.loc[index, 'propertyType_unknown'] = 1
                encoded_df.loc[index, [col for col in encoded_df.columns if col.startswith('propertyType_') and col != 'propertyType_unknown']] = 0

        # Удаление исходных категориальных признаков из новых данных
        data = data.drop(columns=['status', 'propertyType'])

        # Объединение закодированных признаков с остальными данными
        data = pd.concat([data.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)


    return data


def get_random_data(num_rows=1):
    """
    Генерация случайных данных для заданного количества строк.

    Аргументы:
    num_rows (int): Количество строк для генерации данных. По умолчанию 1.

    Возвращает:
    pd.DataFrame: DataFrame со случайными данными.
    """

    # Список возможных значений для категориальных признаков
    status_values = ['for_sale', 'foreclosure', 'new', 'other', 'pending', 'under_contract', 'unknown']
    property_type_values = ['single', 'condo', 'multi', 'coop', 'land', 'traditional', 'townhouse', 'other', 'unknown']
    state_values = ['CA', 'TX', 'NY', 'FL', 'IL', 'PA', 'OH', 'MI', 'GA', 'NC']
    zipcodes = [str(np.random.randint(10000, 99999)) for _ in range(num_rows)]

    # Генерация случайных данных
    data = {
        'status': np.random.choice(status_values, num_rows),
        'baths': np.random.randint(1, 5, num_rows),
        'fireplace': np.random.randint(0, 2, num_rows),
        'zipcode': np.random.choice(zipcodes, num_rows),
        'beds': np.random.randint(1, 6, num_rows),
        'state': np.random.choice(state_values, num_rows),
        'PrivatePool': np.random.randint(0, 2, num_rows),
        'year_built': np.random.randint(1900, 2023, num_rows),
        'parking': np.random.randint(0, 4, num_rows),
        'is_remodeled': np.random.randint(0, 2, num_rows),
        'is_year_built_missing': np.random.randint(0, 2, num_rows),
        'is_heating': np.random.randint(0, 2, num_rows),
        'is_heating_gas': np.zeros(num_rows, dtype=int),
        'is_heating_electric': np.zeros(num_rows, dtype=int),
        'is_heating_central': np.zeros(num_rows, dtype=int),
        'is_cooling': np.random.randint(0, 2, num_rows),
        'average_school_rating': np.random.randint(1, 11, num_rows),
        'has_private_school': np.random.randint(0, 2, num_rows),
        'is_average_school_rating_missing': np.random.randint(0, 2, num_rows),
        'sqft': np.random.randint(500, 5000, num_rows),
        'average_school_distance': np.random.uniform(0.1, 5.0, num_rows),
        'num_schools': np.random.randint(1, 10, num_rows),
        'propertyType': np.random.choice(property_type_values, num_rows)
    }

    # Применяем логику зависимости для признаков 'is_heating_XXX'
    for i in range(num_rows):
        if data['is_heating'][i] == 1:
            heating_options = ['is_heating_gas', 'is_heating_electric', 'is_heating_central']
            chosen_heating = np.random.choice(heating_options + [None])
            if chosen_heating:
                data[chosen_heating][i] = 1

    # Создание DataFrame
    return pd.DataFrame(data)