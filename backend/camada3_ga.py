from typing import Dict, Optional
from ga import (
    CatalogoMusical,
    FitnessEvaluator,
    AlgoritmoGenetico,
    PlaylistGenerator,
    ExperimentosGA
)

def gerar_playlist(score_humor: float, catalogo: Optional[CatalogoMusical] = None) -> Dict:
    """
    Função Helper principal solicitada na especificação do projeto.
    
    Args:
        score_humor (float): Valor entre 0 e 1 oriundo do Fuzzy.
        catalogo (CatalogoMusical, opcional): Instância do catálogo carregado. 
                                              Se None, inicializa um novo (menos eficiente).
    """
    cat = catalogo if catalogo else CatalogoMusical()
    generator = PlaylistGenerator(cat)
    return generator.gerar(score_humor=score_humor)


if __name__ == "__main__":
    print("=" * 60)
    print("INICIANDO CAMADA III - ALGORITMO GENÉTICO (Modularizado)")
    print("=" * 60)

    # 1. Carrega catálogo
    catalogo = CatalogoMusical()
    print(f"[Sistema] Catálogo pronto com {catalogo.obter_tamanho()} músicas.\n")

    # 2. Simula uma saída do Fuzzy
    score_teste = 0.85
    print(f"[Sistema] Recebido Score Humor do Fuzzy: {score_teste}")
    print("[Sistema] Iniciando otimização evolutiva...\n")

    # 3. Execução Controlada para gerar gráfico
    evaluator = FitnessEvaluator(score_teste, catalogo)
    ag = AlgoritmoGenetico(evaluator, tamanho_populacao=100, elitismo=True)
    melhor_ind = ag.evoluir(n_geracoes=80)
    
    # Gera gráfico obrigatório
    ExperimentosGA.plotar_convergencia(ag)

    # 4. Geração via Fachada
    resultado = gerar_playlist(score_teste, catalogo)
    
    print("\n" + "=" * 60)
    print(f"PLAYLIST GERADA (Fitness: {resultado['fitness_alcançado']})")
    print("=" * 60)
    print(f"Estatísticas da Playlist:")
    print(f" - Energia Média : {resultado['estatisticas']['energia_media']} (Ideal próximo a {score_teste})")
    print(f" - Valência Média: {resultado['estatisticas']['valencia_media']} (Ideal próximo a {score_teste})")
    print(f" - BPM Médio     : {resultado['estatisticas']['bpm_medio']} BPM")
    print("-" * 60)
    
    for i, m in enumerate(resultado['faixas'], 1):
        print(f"{i:2d}. {m['musica']} ({m['artista']}) | BPM: {m['bpm']} | Ene: {m['energia']} | Val: {m['valencia']}")
    
    print("=" * 60)
