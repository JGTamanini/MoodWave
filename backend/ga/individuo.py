import random
from typing import List, Optional

class PlaylistIndividual:
    """
    Representa uma solução candidata (Cromossomo).
    Cromossomo: Lista de índices inteiros mapeando para linhas no dataset.
    Restrição: Não são permitidos índices duplicados (mesma música 2x).
    """

    def __init__(self, tamanho: int, max_id: int, genes: Optional[List[int]] = None):
        if genes is None:
            # Inicialização aleatória sem reposição (sem duplicatas)
            self.cromossomo = random.sample(range(max_id), tamanho)
        else:
            self.cromossomo = genes.copy()
            self._reparar_duplicatas(max_id)
            
        self.fitness: float = -1.0

    def _reparar_duplicatas(self, max_id: int):
        """Substitui genes repetidos por novos genes aleatórios."""
        vistos = set()
        for i in range(len(self.cromossomo)):
            if self.cromossomo[i] in vistos:
                novo_gene = random.randint(0, max_id - 1)
                while novo_gene in vistos or novo_gene in self.cromossomo:
                    novo_gene = random.randint(0, max_id - 1)
                self.cromossomo[i] = novo_gene
            vistos.add(self.cromossomo[i])

    def __repr__(self):
        return f"PlaylistIndividual(fit={self.fitness:.4f}, genes={self.cromossomo})"
