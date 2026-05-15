from typing import Dict
from .catalogo import CatalogoMusical
from .fitness import FitnessEvaluator
from .algoritmo import AlgoritmoGenetico

class PlaylistGenerator:
    """
    Classe Facade para encapsular a complexidade da Camada III.
    Projetada para ser facilmente consumida pelo backend em FastAPI.
    """
    
    def __init__(self, catalogo: CatalogoMusical):
        self.catalogo = catalogo

    def gerar(self, target_valence: float, target_arousal: float, tamanho_playlist: int = 10,
              n_geracoes: int = 50, tamanho_populacao: int = 50) -> Dict:
        """
        Gera uma playlist utilizando o Algoritmo Genético.
        Retorna um dicionário JSON-friendly.
        """
        evaluator = FitnessEvaluator(target_valence, target_arousal, self.catalogo)
        ag = AlgoritmoGenetico(evaluator, tamanho_populacao=tamanho_populacao)
        ag.tamanho_playlist = tamanho_playlist
        
        melhor_individuo = ag.evoluir(n_geracoes=n_geracoes)
        
        df_playlist = self.catalogo.df.iloc[melhor_individuo.cromossomo]
        
        # Constrói o formato de saída
        faixas = []
        for _, row in df_playlist.iterrows():
            faixas.append({
                "musica": row["track_name"],
                "artista": row["artist_name"],
                "energia": round(row["energy"], 4),
                "valencia": round(row["valence"], 4),
                "bpm": round(row["tempo"], 2)
            })

        return {
            "score_valencia_alvo": target_valence,
            "score_arousal_alvo": target_arousal,
            "fitness_alcançado": round(melhor_individuo.fitness, 4),
            "estatisticas": {
                "energia_media": round(df_playlist["energy"].mean(), 4),
                "valencia_media": round(df_playlist["valence"].mean(), 4),
                "bpm_medio": round(df_playlist["tempo"].mean(), 2)
            },
            "faixas": faixas,
            "historico_convergencia": ag.historico_melhor_fitness
        }
