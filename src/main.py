from fastapi import FastAPI, HTTPException
import pandas as pd

app = FastAPI()

# Cargar los DataFrames y otras inicializaciones necesarias
# Asumiendo que los archivos están en la ruta especificada
genre_playtime_df = pd.read_parquet('../data/processed/genre_playtime.parquet')

recommendations_df = pd.read_parquet('../data/processed/recommendations.parquet')
game_cosine_sim_df = pd.read_parquet('../data/processed/recomendacion_juego.parquet')
user_cosine_sim_df = pd.read_parquet('../data/processed/recomendacion_usuario.parquet')
final_merged_df = pd.read_parquet('../data/processed/merge.parquet') 
reduced_df = pd.read_parquet('../data/processed/reduced_df.parquet')  

async def play_time_genre(genero: str):
    try:
        # Filtrar solo las filas donde el género especificado es 1 (presente)
        genre_df = genre_playtime_df[genre_playtime_df[genero] == 1]      
        # Agrupar por año de lanzamiento y sumar el tiempo de juego
        year_playtime = genre_df.groupby('release_year')['playtime_forever'].sum()     
        # Encontrar el año con el máximo tiempo de juego
        max_playtime_year = year_playtime.idxmax()   

        max_playtime_year = int(max_playtime_year)

        return {"Año de lanzamiento con más horas jugadas para Género " + genero : max_playtime_year}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/user_for_genre/{genero}")
async def user_for_genre(genero: str):
    try:
        # Filtrar solo las filas donde el género especificado es 1 (presente)
        genre_df = genre_playtime_df[genre_playtime_df[genero] == 1]        
        # Agrupar por usuario y sumar el tiempo de juego
        user_playtime = genre_df.groupby('user_id')['playtime_forever'].sum()      
        # Encontrar el usuario con el máximo tiempo de juego
        max_playtime_user = user_playtime.idxmax()      
        # Filtrar las filas para el usuario específico
        user_df = genre_df[genre_df['user_id'] == max_playtime_user]     
        # Agrupar por año de lanzamiento y sumar el tiempo de juego
        user_year_playtime = user_df.groupby('release_year')['playtime_forever'].sum().reset_index()     
        # Crear la lista de horas jugadas por año
        hours_played = user_year_playtime.to_dict('records')
        return {"Usuario con más horas jugadas para Género " + genero : max_playtime_user, "Horas jugadas": hours_played}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/users_recommend/{year}")
async def users_recommend(year: int):
    try:
        # Filtrar las filas para el año especificado y las recomendaciones positivas
        year_recommend_df = recommendations_df[(recommendations_df['release_year'] == year) & (recommendations_df['recommend'] == True)]   
        # Agrupar por nombre de la aplicación y contar las recomendaciones
        game_recommend_count = year_recommend_df.groupby('app_name').size() 
        # Ordenar los juegos por la cantidad de recomendaciones y tomar los 3 primeros
        top_3_games = game_recommend_count.sort_values(ascending=False).head(3).index.tolist()
        return [{"Puesto 1": top_3_games[0]}, {"Puesto 2": top_3_games[1]}, {"Puesto 3": top_3_games[2]}]
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/users_not_recommend/{year}")
async def users_not_recommend(year: int):
    try:
        year_not_recommend_df = recommendations_df[(recommendations_df['release_year'] == year) & (recommendations_df['recommend'] == False)]
        game_not_recommend_count = year_not_recommend_df.groupby('app_name').size()
        bottom_3_games = game_not_recommend_count.sort_values().head(3).index.tolist()
        return [{"Puesto 1": bottom_3_games[0]}, {"Puesto 2": bottom_3_games[1]}, {"Puesto 3": bottom_3_games[2]}]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/sentiment_analysis/{year}")
async def sentiment_analysis(year: int):
    try:
        # Filtrar las filas para el año especificado
        year_sentiment_df = recommendations_df[recommendations_df['release_year'] == year]
        # Contar los registros de análisis de sentimiento
        sentiment_counts = year_sentiment_df['sentiment_analysis'].value_counts()
        return {
            "Negative": int(sentiment_counts.get(0, 0)),
            "Neutral": int(sentiment_counts.get(1, 0)),
            "Positive": int(sentiment_counts.get(2, 0))
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/recomendacion_juego/{item_id}")
async def recomendacion_juego(item_id: int):
    try:
        # Obtener las similitudes del juego dado
        sim_scores = game_cosine_sim_df[item_id]
        # Ordenar los juegos basados en la similitud del coseno
        recommended_games_ids = sim_scores.sort_values(ascending=False)[1:6].index  # Excluir el juego mismo
        # Buscar los nombres de los juegos recomendados
        recommended_games_names = reduced_df.set_index('item_id').loc[recommended_games_ids, 'app_name'].unique()
        # Devolver los nombres de los juegos recomendados
        return {"Juegos recomendados": recommended_games_names.tolist()}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/recomendacion_usuario/{user_id}")
async def recomendacion_usuario(user_id: str):
    try:
        # Obtener las similitudes del usuario dado
        sim_scores = user_cosine_sim_df[user_id]
        # Ordenar los usuarios basados en la similitud del coseno
        similar_users = sim_scores.sort_values(ascending=False)[1:]  # Excluir el usuario mismo
        
        # Obtener los juegos jugados por los usuarios similares
        similar_users_list = similar_users.index.tolist()
        recommended_games_ids = final_merged_df[final_merged_df['user_id'].isin(similar_users_list)]['item_id'].unique()
        
        # Ordenar los juegos recomendados basados en la similitud del coseno
        recommended_games_ids = sorted(recommended_games_ids, key=lambda x: similar_users_list.index(final_merged_df[final_merged_df['item_id'] == x]['user_id'].iloc[0]))
        
        # Obtener los nombres de los 5 juegos más recomendados
        # Asegurarse de que 'item_id' sea el índice en final_merged_df
        final_merged_df.set_index('item_id', inplace=True)
        recommended_game_names = final_merged_df.loc[recommended_games_ids[:5], 'app_name'].unique()
        
        # Resetear el índice de final_merged_df para futuras consultas
        final_merged_df.reset_index(inplace=True)
        
        return {"Juegos recomendados": recommended_game_names.tolist()}
    
    except Exception as e:
        print(e)
        raise HTTPException(status_code=400, detail=str(e)) 


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

