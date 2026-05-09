import numpy as np
from .catalogo import CatalogoMusical
from .individuo import PlaylistIndividual

class FitnessEvaluator:
    """
    Cálculo do Fitness da playlist baseado em 3 componentes:
    1. Compatibilidade Emocional (Aproximação dos Targets do Fuzzy)
    2. Diversidade de Artistas (Penaliza repetição)
    3. Suavidade de Transição (Evita saltos abruptos de energia/bpm entre faixas adjacentes)
    """

    def __init__(self, score_humor: float, catalogo: CatalogoMusical,
                 peso_compat: float = 0.5, peso_div: float = 0.3, peso_suav: float = 0.2):
        self.score_humor = score_humor
        self.catalogo = catalogo
        self.peso_compat = peso_compat
        self.peso_div = peso_div
        self.peso_suav = peso_suav
        
        # Mapeamento do score do Fuzzy (0 a 1) para features musicais alvo
        self.target_valence = score_humor
        self.target_energy = score_humor
        self.target_tempo_norm = score_humor

    def avaliar(self, individuo: PlaylistIndividual) -> float:
        df_playlist = self.catalogo.df.iloc[individuo.cromossomo]
        
        # 1. Compatibilidade Emocional (Baseada no Erro Quadrático Médio - MSE)
        avg_valence = df_playlist["valence"].mean()
        avg_energy = df_playlist["energy"].mean()
        avg_tempo_norm = df_playlist["tempo_norm"].mean()
        
        mse = (
            (avg_valence - self.target_valence)**2 +
            (avg_energy - self.target_energy)**2 +
            (avg_tempo_norm - self.target_tempo_norm)**2
        ) / 3.0
        f_compat = 1.0 - np.sqrt(mse) # Quanto menor o erro, maior o fitness (máx 1.0)
        
        # 2. Diversidade de Artistas
        artistas_unicos = df_playlist["artist_name"].nunique()
        tamanho_playlist = len(individuo.cromossomo)
        f_div = artistas_unicos / tamanho_playlist
        
        # 3. Suavidade de Transição (Diferença absoluta média entre faixas adjacentes)
        if tamanho_playlist > 1:
            diffs = []
            for i in range(1, tamanho_playlist):
                m1 = df_playlist.iloc[i-1]
                m2 = df_playlist.iloc[i]
                diff_e = abs(m1["energy"] - m2["energy"])
                diff_t = abs(m1["tempo_norm"] - m2["tempo_norm"])
                diffs.append((diff_e + diff_t) / 2.0)
            avg_diff = sum(diffs) / len(diffs)
            f_suav = 1.0 - avg_diff # Penaliza grandes saltos
        else:
            f_suav = 1.0

        # Fitness Agregado Ponderado
        fitness_total = (
            (self.peso_compat * f_compat) +
            (self.peso_div * f_div) +
            (self.peso_suav * f_suav)
        )
        
        # Armazena o valor no indivíduo para cache
        individuo.fitness = max(0.0, min(1.0, fitness_total))
        return individuo.fitness
