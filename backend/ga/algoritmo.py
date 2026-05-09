import random
from typing import List, Tuple
from .individuo import PlaylistIndividual
from .fitness import FitnessEvaluator

class AlgoritmoGenetico:
    """
    Motor principal da Evolução. Orquestra as gerações, seleção, crossover e mutação.
    """

    def __init__(self, evaluator: FitnessEvaluator, tamanho_populacao: int = 50,
                 taxa_mutacao: float = 0.1, taxa_crossover: float = 0.8,
                 elitismo: bool = True):
        self.evaluator = evaluator
        self.tamanho_populacao = tamanho_populacao
        self.taxa_mutacao = taxa_mutacao
        self.taxa_crossover = taxa_crossover
        self.elitismo = elitismo
        
        self.tamanho_playlist = 10  # Pode ser parametrizado futuramente
        self.max_id = evaluator.catalogo.obter_tamanho()
        self.populacao: List[PlaylistIndividual] = []
        self.historico_melhor_fitness = []
        self.historico_media_fitness = []

    def inicializar_populacao(self):
        self.populacao = [
            PlaylistIndividual(self.tamanho_playlist, self.max_id)
            for _ in range(self.tamanho_populacao)
        ]

    def avaliar_populacao(self):
        for ind in self.populacao:
            if ind.fitness == -1.0: # Só avalia se não tiver cache
                self.evaluator.avaliar(ind)

    def selecao_torneio(self, k: int = 3) -> PlaylistIndividual:
        """Seleciona o melhor indivíduo de um subgrupo aleatório (Torneio)."""
        torneio = random.sample(self.populacao, k)
        melhor = max(torneio, key=lambda ind: ind.fitness)
        return melhor

    def crossover_um_ponto(self, p1: PlaylistIndividual, p2: PlaylistIndividual) -> Tuple[PlaylistIndividual, PlaylistIndividual]:
        """Troca os segmentos genéticos entre dois pais após um ponto de corte aleatório."""
        if random.random() > self.taxa_crossover:
            return PlaylistIndividual(self.tamanho_playlist, self.max_id, p1.cromossomo), \
                   PlaylistIndividual(self.tamanho_playlist, self.max_id, p2.cromossomo)

        corte = random.randint(1, self.tamanho_playlist - 2)
        f1_genes = p1.cromossomo[:corte] + p2.cromossomo[corte:]
        f2_genes = p2.cromossomo[:corte] + p1.cromossomo[corte:]
        
        # O construtor do PlaylistIndividual já cuida da reparação de duplicatas
        f1 = PlaylistIndividual(self.tamanho_playlist, self.max_id, f1_genes)
        f2 = PlaylistIndividual(self.tamanho_playlist, self.max_id, f2_genes)
        return f1, f2

    def mutacao(self, individuo: PlaylistIndividual):
        """
        Mutação Híbrida:
        1. Swap (troca duas músicas de ordem) - ajuda na suavidade.
        2. Substituição (troca uma música por outra aleatória do catálogo) - ajuda na diversidade e aproximação.
        """
        for i in range(self.tamanho_playlist):
            if random.random() < self.taxa_mutacao:
                if random.random() < 0.5:
                    # Mutação de Swap
                    j = random.randint(0, self.tamanho_playlist - 1)
                    individuo.cromossomo[i], individuo.cromossomo[j] = individuo.cromossomo[j], individuo.cromossomo[i]
                else:
                    # Mutação de Substituição
                    novo_gene = random.randint(0, self.max_id - 1)
                    while novo_gene in individuo.cromossomo:
                        novo_gene = random.randint(0, self.max_id - 1)
                    individuo.cromossomo[i] = novo_gene
                individuo.fitness = -1.0 # Invalida o cache do fitness

    def evoluir(self, n_geracoes: int = 50):
        self.inicializar_populacao()
        self.avaliar_populacao()

        for geracao in range(n_geracoes):
            # Ordena a população (maior fitness primeiro)
            self.populacao.sort(key=lambda ind: ind.fitness, reverse=True)
            
            # Registro para gráficos
            melhor_fit = self.populacao[0].fitness
            media_fit = sum(ind.fitness for ind in self.populacao) / self.tamanho_populacao
            self.historico_melhor_fitness.append(melhor_fit)
            self.historico_media_fitness.append(media_fit)

            nova_populacao = []
            
            # Elitismo
            if self.elitismo:
                nova_populacao.append(self.populacao[0])

            # Reprodução
            while len(nova_populacao) < self.tamanho_populacao:
                p1 = self.selecao_torneio()
                p2 = self.selecao_torneio()
                
                f1, f2 = self.crossover_um_ponto(p1, p2)
                
                self.mutacao(f1)
                self.mutacao(f2)
                
                nova_populacao.append(f1)
                if len(nova_populacao) < self.tamanho_populacao:
                    nova_populacao.append(f2)

            self.populacao = nova_populacao
            self.avaliar_populacao()

        # Garante que no fim está ordenado
        self.populacao.sort(key=lambda ind: ind.fitness, reverse=True)
        return self.populacao[0]
