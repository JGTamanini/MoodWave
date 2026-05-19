import os
import matplotlib.pyplot as plt
from .algoritmo import AlgoritmoGenetico

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class ExperimentosGA:
    """
    Classe utilitária para gerar os gráficos exigidos pelo caráter acadêmico do trabalho.
    """
    
    @staticmethod
    def plotar_convergencia(ag: AlgoritmoGenetico, nome_arquivo="convergencia_ga.png"):
        plt.figure(figsize=(10, 6))
        plt.plot(ag.historico_melhor_fitness, label="Melhor Fitness", color="green", linewidth=2)
        plt.plot(ag.historico_media_fitness, label="Média da População", color="orange", linestyle="--")
        plt.title("Curva de Convergência do Algoritmo Genético", fontsize=14)
        plt.xlabel("Geração", fontsize=12)
        plt.ylabel("Fitness", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        caminho = os.path.join(BASE_DIR, nome_arquivo)
        plt.savefig(caminho, dpi=300, bbox_inches="tight")
        print(f"[Experimentos] Gráfico de convergência salvo em: {caminho}")
        plt.close()
