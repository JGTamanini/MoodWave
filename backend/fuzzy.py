import numpy as np

class SistemaFuzzy:
    def __init__(self):
        self.saidas = {
            "melancolico": 0.10,
            "calmo": 0.35,
            "neutro": 0.50,
            "animado": 0.65,
            "euforico": 0.90
        }

    def _trapezoide(self, x, a, b, c, d):
        if a is None: a = -np.inf
        if b is None: b = -np.inf
        if c is None: c = np.inf
        if d is None: d = np.inf

        if x <= a:
            return 0
        elif x > a and x <= b:
            return (x - a) / (b - a)
        elif x > b and x <= c:
            return 1
        elif x > c and x <= d:
            return (d - x) / (d - c)
        else:
            return 0
    
    def _fp_sentimento(self, prob_sentimento):
        neg = self._trapezoide(prob_sentimento, None, None, 0.275, 0.425)
        neu = self._trapezoide(prob_sentimento, 0.275, 0.425, 0.575, 0.725)
        pos = self._trapezoide(prob_sentimento, 0.575, 0.725, None, None)
        return neg, neu, pos
    
    def _fp_energia(self, energia):
        bai = self._trapezoide(energia, None, None, 0.275, 0.425)
        med = self._trapezoide(energia, 0.275, 0.425, 0.575, 0.725)
        alt = self._trapezoide(energia, 0.575, 0.725, None, None)
        return bai, med, alt
    
    def fuzzificar(self, prob_sentimento, energia):
        s_neg, s_neu, s_pos = self._fp_sentimento(prob_sentimento)
        e_bai, e_med, e_alt = self._fp_energia(energia)

        return {
            "s_neg": s_neg, "s_neu": s_neu, "s_pos": s_pos,
            "e_bai": e_bai, "e_med": e_med, "e_alt": e_alt,
        }

    def inferir (self, graus):
        # graus é o dicionário que veio do fuzzificar (return)
        return {
            "melancolico": min(graus["s_neg"], graus["e_bai"]),
            "calmo": max(min(graus["s_neu"], graus["e_bai"]), min(graus["s_neg"], graus["e_med"])),
            "neutro": max(min(graus["s_pos"], graus["e_bai"]), min(graus["s_neu"], graus["e_med"]), min(graus["s_neg"], graus["e_alt"])),
            "animado": max(min(graus["s_pos"], graus["e_med"]), min(graus["s_neu"], graus["e_alt"])),
            "euforico": min(graus["s_pos"], graus["e_alt"])
        }
    
    def defuzzificar(self, forcas):
        # feito em Weighted Average
        # score = Σ(força × valor_saída) / Σ(força)
        # forcas é o dicionário que veio do inferir (return)
        numerador = 0
        denominador = 0
        for estado, forca in forcas.items():
            valor_saida = self.saidas[estado]
            numerador += forca * valor_saida
            denominador += forca
        return numerador/denominador if denominador != 0 else 0
        

    def calcular_humor(self, prob_sentimento, energia):
        graus = self.fuzzificar(prob_sentimento, energia)
        forcas = self.inferir(graus)
        humor = self.defuzzificar(forcas)
        return humor

# teste rápido — fora da classe
if __name__ == "__main__":
    fis = SistemaFuzzy()

    print(fis.calcular_humor(0.1, 0.1))  # esperado: ~0.10 melancólico
    print(fis.calcular_humor(0.5, 0.5))  # esperado: ~0.50 neutro
    print(fis.calcular_humor(0.9, 0.9))  # esperado: ~0.90 eufórico
    # na fronteira entre melancólico e calmo
    print(fis.calcular_humor(0.2, 0.4))
    # sentimento negativo mas energia alta — deve puxar para neutro
    print(fis.calcular_humor(0.1, 0.9))
    # feliz mas sem energia — deve puxar para baixo
    print(fis.calcular_humor(0.9, 0.1))