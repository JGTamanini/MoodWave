from typing import Dict, Optional
from ga import (
    CatalogoMusical,
    FitnessEvaluator,
    AlgoritmoGenetico,
    PlaylistGenerator,
    ExperimentosGA
)

def gerar_playlist(target_valence: float, target_arousal: float, catalogo: Optional[CatalogoMusical] = None) -> Dict:
    """
    Função Helper principal solicitada na especificação do projeto.
    
    Args:
        target_valence (float): Valor entre 0 e 1 de valência oriundo do Fuzzy.
        target_arousal (float): Valor entre 0 e 1 de arousal oriundo do Fuzzy.
        catalogo (CatalogoMusical, opcional): Instância do catálogo carregado. 
                                              Se None, inicializa um novo (menos eficiente).
    """
    cat = catalogo if catalogo else CatalogoMusical()
    generator = PlaylistGenerator(cat)
    return generator.gerar(target_valence=target_valence, target_arousal=target_arousal)


if __name__ == "__main__":
    print("=" * 60)
    print("INICIANDO CAMADA III - ALGORITMO GENÉTICO (Modularizado)")
    print("=" * 60)

    # 1. Carrega catálogo
    catalogo = CatalogoMusical()
    print(f"[Sistema] Catálogo pronto com {catalogo.obter_tamanho()} músicas.\n")

    # 2. Simula uma saída do Fuzzy
    valencia_teste = 0.80  # Ex: Alegria
    arousal_teste = 0.90
    print(f"[Sistema] Recebido do Fuzzy -> Valência: {valencia_teste} | Arousal: {arousal_teste}")
    print("[Sistema] Iniciando otimização evolutiva...\n")

    # 3. Execução Controlada para gerar gráfico
    evaluator = FitnessEvaluator(valencia_teste, arousal_teste, catalogo)
    ag = AlgoritmoGenetico(evaluator, tamanho_populacao=100, elitismo=True)
    melhor_ind = ag.evoluir(n_geracoes=80)
    
    # Gera gráfico obrigatório
    ExperimentosGA.plotar_convergencia(ag)

    # 4. Geração via Fachada
    resultado = gerar_playlist(valencia_teste, arousal_teste, catalogo)
    
    print("\n" + "=" * 60)
    print(f"PLAYLIST GERADA (Fitness: {resultado['fitness_alcançado']})")
    print("=" * 60)
    print(f"Estatísticas da Playlist:")
    print(f" - Energia Média : {resultado['estatisticas']['energia_media']} (Ideal próximo a {arousal_teste})")
    print(f" - Valência Média: {resultado['estatisticas']['valencia_media']} (Ideal próximo a {valencia_teste})")
    print(f" - BPM Médio     : {resultado['estatisticas']['bpm_medio']} BPM")
    print("-" * 60)
    
    for i, m in enumerate(resultado['faixas'], 1):
        print(f"{i:2d}. {m['musica']} ({m['artista']}) | BPM: {m['bpm']} | Ene: {m['energia']} | Val: {m['valencia']}")
    
    print("=" * 60)
