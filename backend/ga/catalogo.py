import os
import random
import numpy as np
import pandas as pd
from typing import Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

class CatalogoMusical:
    """
    Responsável pela ingestão, normalização e disponibilidade das músicas.
    Padrão de Projeto: Singleton (simplificado para uso acadêmico).
    """

    def __init__(self, caminho_csv: Optional[str] = None):
        self.caminho_csv = caminho_csv or os.path.join(DATA_DIR, "spotify_catalog.csv")
        self.df = None
        self._carregar_dados()

    def _carregar_dados(self):
        """Carrega do CSV se existir, senão gera um fallback realista."""
        if os.path.exists(self.caminho_csv):
            print(f"[Catálogo] Carregando dataset do Spotify em: {self.caminho_csv}")
            try:
                self.df = pd.read_csv(self.caminho_csv)
                self._limpar_e_normalizar()
            except Exception as e:
                print(f"[Catálogo] Erro ao ler CSV: {e}. Usando fallback.")
                self._gerar_fallback()
        else:
            print(f"[Catálogo] Dataset não encontrado em {self.caminho_csv}.")
            print("[Catálogo] Kaggle: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset")
            print("[Catálogo] Gerando dataset sintético (fallback)...")
            self._gerar_fallback()

    def _gerar_fallback(self, n_samples: int = 5000):
        """Gera um dataset fake mas estatisticamente realista para testes."""
        np.random.seed(42)
        random.seed(42)
        
        artistas = [
            "The Weeknd", "Taylor Swift", "Drake", "Arctic Monkeys", "Queen", 
            "Anitta", "Dua Lipa", "Ed Sheeran", "Coldplay", "Bruno Mars", 
            "Billie Eilish", "Imagine Dragons", "Post Malone", "Maroon 5", "Eminem",
            "Kendrick Lamar", "Bad Bunny", "Rosalía", "J Balvin", "Shakira",
            "Beyoncé", "Rihanna", "Lady Gaga", "Katy Perry", "Justin Bieber",
            "Ariana Grande", "Harry Styles", "Adele", "David Guetta", "Calvin Harris"
        ]
        
        musicas = [
            "Blinding Lights", "Shape of You", "Dance Monkey", "Rockstar", "One Dance", 
            "Closer", "Sunflower", "Someone You Loved", "Senorita", "Bad Guy", 
            "Perfect", "Thinking Out Loud", "God's Plan", "Lucid Dreams", "Photograph", 
            "Starboy", "Love Yourself", "Havana", "Believer", "Shallow", 
            "Thunder", "Watermelon Sugar", "Stay", "Levitating", "Peaches", 
            "Good 4 U", "Drivers License", "Montero", "Save Your Tears", "Kiss Me More",
            "Industry Baby", "Heat Waves", "As It Was", "About Damn Time", "Anti-Hero", 
            "Flowers", "Kill Bill", "Creepin'", "Calm Down", "Die For You", 
            "Cruel Summer", "Paint The Town Red", "Water", "Greedy", "Strangers"
        ]

        # Gera milhares de combinações diferentes de títulos reais com sufixos
        sufixos = ["", "", "", "", " (Remix)", " (Acoustic)", " (Live)", " (Radio Edit)"]
        
        track_names = [f"{random.choice(musicas)}{random.choice(sufixos)}" for _ in range(n_samples)]
        artist_names = [random.choice(artistas) for _ in range(n_samples)]

        dados = {
            "track_name": track_names,
            "artist_name": artist_names,
            "valence": np.random.beta(a=2, b=2, size=n_samples),       
            "energy": np.random.beta(a=2.5, b=2, size=n_samples),      
            "danceability": np.random.beta(a=5, b=2, size=n_samples),  
            "tempo": np.random.normal(loc=120, scale=20, size=n_samples) 
        }
        self.df = pd.DataFrame(dados)
        
        # Garante limites reais
        self.df["tempo"] = self.df["tempo"].clip(lower=60, upper=200)
        
        os.makedirs(DATA_DIR, exist_ok=True)
        self.df.to_csv(self.caminho_csv, index=False)
        self._limpar_e_normalizar()

    def _limpar_e_normalizar(self):
        """Filtra dados inválidos e normaliza variáveis para o intervalo [0, 1]."""
        colunas_necessarias = ["track_name", "artist_name", "valence", "energy", "danceability", "tempo"]
        self.df = self.df.dropna(subset=colunas_necessarias)
        
        # Normalização Min-Max para 'tempo', para ficar no range [0, 1]
        min_tempo = 60.0
        max_tempo = 200.0
        
        self.df["tempo_norm"] = (self.df["tempo"] - min_tempo) / (max_tempo - min_tempo)
        self.df["tempo_norm"] = self.df["tempo_norm"].clip(0, 1)

    def obter_tamanho(self) -> int:
        return len(self.df)

    def obter_musica(self, indice: int) -> pd.Series:
        return self.df.iloc[indice]
