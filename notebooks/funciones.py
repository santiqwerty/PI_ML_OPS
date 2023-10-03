import demjson3
import pandas as pd
import ast

def load_and_normalize_user_reviews(file_path):

    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.readlines()
    # Utilizo demjson.decode en lugar de json.loads o eval
    records = [demjson3.decode(line.strip()) for line in data]
    df = pd.DataFrame(records)

    data_desanidada = []

    for index, row in df.iterrows(): 
        user_id = row['user_id']
        user_url = row['user_url']
        reviews = row['reviews']

        for review in reviews:
            new_row = {
                'user_id': user_id,
                'user_url': user_url,
                'funny': review.get('funny', ''),
                'posted': review.get('posted', ''),
                'last_edited': review.get('last_edited', ''),
                'item_id': review.get('item_id', ''),
                'helpful': review.get('helpful', ''),
                'recommend': review.get('recommend', False),
                'review': review.get('review', ''),
            }
            data_desanidada.append(new_row)

    return pd.DataFrame(data_desanidada)

def load_and_normalize_user_items(file_path):
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.readlines()
    records = [eval(line.strip()) for line in data]
    df = pd.DataFrame(records)

    data_desanidada = []

    for index, row in df.iterrows():
        user_id = row['user_id']
        steam_id = row['steam_id']
        items_count = row['items_count']
        user_url = row['user_url']
        items = row['items']
        
        for item in items:
            new_row = {
                'user_id': user_id,
                'user_url': user_url,
                'steam_id': steam_id,
                'item_count': items_count,
                'item_id': item.get('item_id', ''),
                'item_name': item.get('item_name', ''),
                'playtime_forever': item.get('playtime_forever', ''),
                'playtime_2weeks': item.get('playtime_2weeks', ''),
        
            }
            data_desanidada.append(new_row)
    return pd.DataFrame(data_desanidada)


def load_and_normalize_steam_games(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.readlines()
    # Utiliza demjson.decode en lugar de eval
    records = [demjson3.decode(line.strip()) for line in data]
    df = pd.DataFrame(records)
    return df

def process_file(file_path, output_csv_path, load_and_normalize_function):
    
    normalized_data = load_and_normalize_function(file_path)
    
    # Save to a CSV file
    normalized_data.to_csv(output_csv_path, index=False)
    
    # Return the DataFrame
    return normalized_data

def convert_price(price):
    # Verificar si el precio es NaN antes de tratar de convertirlo
    if pd.isna(price):
        return price  # Retornar NaN si el precio es NaN
    try:
        return float(price)
    except ValueError:
        if price == 'Starting at $499.00':
            return 499.00
        elif price == 'Starting at $449.00':
            return 449.00
        else:
            return 0.0  # Retornar 0 para valores atípicos que no son los especificados

# Función para convertir la cadena que representa una lista a una lista de Python
def convert_string_to_list(value):
    if isinstance(value, list):
        return value
    elif value == 'Valor Desconocido (NULL)' or pd.isnull(value):
        return []  # Retornar una lista vacía si el valor es 'Valor Desconocido (NULL)' o NaN
    else:
        try:
            # Utiliza ast.literal_eval para una conversión más segura
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            # Captura excepciones y retorna una lista vacía como una acción de gracia
            return []
        
def extract_percentage(value):
    start = value.find('(')
    end = value.find('%')
    if start != -1 and end != -1:
        percentage = value[start+1:end]
        return int(percentage)
    return 0