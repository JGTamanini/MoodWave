"""
Camada II — Sistema de Inferência Fuzzy (Sugeno)
MoodWave — Sistema de Recomendação de Músicas por Humor
 
Interface pública:
    fis = SistemaFuzzy()
    resultado = fis.analisar(probabilidades)
    → {"score": float, "label": str}
 
Métodos internos (disponíveis mas não necessários para integração):
    fis.calcular_humor(valencia, arousal) → float [0, 1]
    fis.classificar(score_humor)          → str
 
Entradas:
    probabilidades : dict — vem da Camada I (Felipe)
        {
            "alegria":  float [0, 1],
            "tristeza": float [0, 1],
            "raiva":    float [0, 1],
            "medo":     float [0, 1],
            "aversao":  float [0, 1],
            "surpresa": float [0, 1],
            "desprezo": float [0, 1]
        }
 
Saída:
    score  : float [0, 1]
    label  : str — "Alegria", "Surpresa", "Raiva", "Medo/Desprezo",
                   "Tristeza/Aversão"
 
Nota sobre fusão de pares:
    Tristeza e Aversão compartilham valência baixa e arousal baixo no
    modelo de Russell — são indistinguíveis por esses dois eixos.
    O mesmo vale para Medo e Desprezo (valência baixa, arousal médio).
    Os pares são fundidos em estados únicos no Fuzzy; a distinção fina
    entre eles, quando necessária, deve usar a classe retornada pela
    Camada I diretamente.
 
Base teórica:
    Valência e Arousal calculados via média ponderada segundo
    o modelo Circumplex de Russell (1980) e Ekman (1992).
"""
 
import numpy as np
 
 
class SistemaFuzzy:
    def __init__(self):
        # Cinco estados de saída baseados nas emoções de Ekman,
        # com os pares indistinguíveis no plano valência×arousal fundidos:
        #   tristeza + aversão  → valência baixa, arousal baixo
        #   medo     + desprezo → valência baixa, arousal médio
        self.saidas = {
            "tristeza_aversao": 0.10,   # valência baixa, arousal baixo
            "medo_desprezo":    0.30,   # valência baixa, arousal médio
            "raiva":            0.50,   # valência baixa, arousal alto
            "surpresa":         0.70,   # valência média, arousal alto
            "alegria":          0.90,   # valência alta,  arousal alto
        }
 
    # ------------------------------------------------------------------
    # NOVO — tradução das 7 emoções para valência e arousal
    # ------------------------------------------------------------------
 
    def _calcular_valencia_arousal(self, prob: dict) -> tuple[float, float]:
        """
        Converte o vetor de 7 probabilidades em dois floats contínuos:
            valencia : quão positiva é a emoção  [0, 1]
            arousal  : quão agitada é a emoção   [0, 1]
 
        Pesos baseados no modelo Circumplex de Russell (1980)
        mapeado para as emoções de Ekman (1992).
        """
        valencia = (
            prob["alegria"]  * 0.90 +
            prob["surpresa"] * 0.60 +
            prob["medo"]     * 0.20 +
            prob["tristeza"] * 0.15 +
            prob["aversao"]  * 0.10 +
            prob["raiva"]    * 0.10 +
            prob["desprezo"] * 0.05
        )
 
        arousal = (
            prob["alegria"]  * 0.85 +
            prob["raiva"]    * 0.90 +
            prob["surpresa"] * 0.80 +
            prob["medo"]     * 0.60 +
            prob["desprezo"] * 0.40 +
            prob["tristeza"] * 0.20 +
            prob["aversao"]  * 0.15
        )
 
        # garante que os pesos somam ≤ 1; normaliza se necessário
        soma_prob = sum(prob.values())
        if soma_prob > 0:
            valencia = valencia / soma_prob
            arousal  = arousal  / soma_prob
 
        return float(np.clip(valencia, 0, 1)), float(np.clip(arousal, 0, 1))
 
    # ------------------------------------------------------------------
    # Núcleo Fuzzy — sem alterações em relação à versão anterior
    # ------------------------------------------------------------------
 
    def _trapezoide(self, x, a, b, c, d):
        if a is None: a = -np.inf
        if b is None: b = -np.inf
        if c is None: c =  np.inf
        if d is None: d =  np.inf
 
        if x <= a:
            return 0
        elif a < x <= b:
            return (x - a) / (b - a)
        elif b < x <= c:
            return 1
        elif c < x <= d:
            return (d - x) / (d - c)
        else:
            return 0
 
    def _fp_valencia(self, valencia):
        """Funções de pertinência para valência (substitui prob_sentimento)."""
        neg = self._trapezoide(valencia, None, None, 0.275, 0.425)
        neu = self._trapezoide(valencia, 0.275, 0.425, 0.575, 0.725)
        pos = self._trapezoide(valencia, 0.575, 0.725, None, None)
        return neg, neu, pos
 
    def _fp_arousal(self, arousal):
        """Funções de pertinência para arousal (substitui energia)."""
        bai = self._trapezoide(arousal, None, None, 0.275, 0.425)
        med = self._trapezoide(arousal, 0.275, 0.425, 0.575, 0.725)
        alt = self._trapezoide(arousal, 0.575, 0.725, None, None)
        return bai, med, alt
 
    def fuzzificar(self, valencia: float, arousal: float) -> dict:
        v_neg, v_neu, v_pos = self._fp_valencia(valencia)
        a_bai, a_med, a_alt = self._fp_arousal(arousal)
 
        return {
            "v_neg": v_neg, "v_neu": v_neu, "v_pos": v_pos,
            "a_bai": a_bai, "a_med": a_med, "a_alt": a_alt,
        }
 
    def inferir(self, graus: dict) -> dict:
        """
        Matriz de regras valência × arousal mapeada para as emoções de Ekman
        (com pares fundidos onde o modelo de Russell não distingue).
 
        Tabela de regras:
            v_neg × a_bai → tristeza/aversão
            v_neg × a_med → medo/desprezo
            v_neg × a_alt → raiva
            v_neu × a_alt → surpresa
            v_pos × a_alt → alegria
            v_pos × a_med → alegria (positivo mas menos agitado)
            v_neu × a_med → surpresa (neutro com agitação média)
            v_pos × a_bai → tristeza/aversão (positivo sem energia — borda)
            v_neu × a_bai → tristeza/aversão (neutro sem energia — borda)
        """
        return {
            "tristeza_aversao": max(
                min(graus["v_neg"], graus["a_bai"]),
                min(graus["v_neu"], graus["a_bai"]),
                min(graus["v_pos"], graus["a_bai"]),  # positivo sem arousal → apatia
            ),
            "medo_desprezo": min(graus["v_neg"], graus["a_med"]),
            "raiva":         min(graus["v_neg"], graus["a_alt"]),
            "surpresa":      max(
                min(graus["v_neu"], graus["a_alt"]),
                min(graus["v_neu"], graus["a_med"]),
            ),
            "alegria":       max(
                min(graus["v_pos"], graus["a_alt"]),
                min(graus["v_pos"], graus["a_med"]),
            ),
        }
 
    def defuzzificar(self, forcas: dict) -> float:
        """Weighted Average (Sugeno) — sem alterações."""
        numerador   = sum(f * self.saidas[e] for e, f in forcas.items())
        denominador = sum(forcas.values())
        return numerador / denominador if denominador != 0 else 0.0
 
    def calcular_humor(self, valencia: float, arousal: float) -> float:
        graus  = self.fuzzificar(valencia, arousal)
        forcas = self.inferir(graus)
        return self.defuzzificar(forcas)
 
    def classificar(self, score_humor: float) -> str:
        """
        Thresholds alinhados com os valores de saída de self.saidas:
            0.10 → Tristeza/Aversão
            0.30 → Medo/Desprezo
            0.50 → Raiva
            0.70 → Surpresa
            0.90 → Alegria
        """
        if score_humor < 0.20:
            return "Tristeza/Aversão"
        elif score_humor < 0.40:
            return "Medo/Desprezo"
        elif score_humor < 0.60:
            return "Raiva"
        elif score_humor < 0.80:
            return "Surpresa"
        else:
            return "Alegria"
 
    # ------------------------------------------------------------------
    # Interface pública — nova assinatura
    # ------------------------------------------------------------------
 
    def analisar(self, probabilidades: dict) -> dict:
        """
        Entrada: dicionário com as 7 probabilidades da Camada I.
        Saída:   {"score": float, "label": str}
        """
        valencia, arousal = self._calcular_valencia_arousal(probabilidades)
        score = self.calcular_humor(valencia, arousal)
        label = self.classificar(score)
        return {"score": round(score, 4), "label": label}
 
 
# ----------------------------------------------------------------------
# Testes rápidos
# ----------------------------------------------------------------------
if __name__ == "__main__":
    fis = SistemaFuzzy()
 
    casos = [
        ("Alegria pura",     {"alegria": 0.90, "tristeza": 0.02, "raiva": 0.01,
                              "medo": 0.02, "aversao": 0.01, "surpresa": 0.03, "desprezo": 0.01}),
        ("Tristeza pura",    {"alegria": 0.01, "tristeza": 0.85, "raiva": 0.03,
                              "medo": 0.04, "aversao": 0.03, "surpresa": 0.02, "desprezo": 0.02}),
        ("Aversão pura",     {"alegria": 0.01, "tristeza": 0.03, "raiva": 0.03,
                              "medo": 0.04, "aversao": 0.85, "surpresa": 0.02, "desprezo": 0.02}),
        ("Raiva pura",       {"alegria": 0.01, "tristeza": 0.05, "raiva": 0.80,
                              "medo": 0.05, "aversao": 0.05, "surpresa": 0.02, "desprezo": 0.02}),
        ("Medo puro",        {"alegria": 0.01, "tristeza": 0.05, "raiva": 0.03,
                              "medo": 0.82, "aversao": 0.03, "surpresa": 0.04, "desprezo": 0.02}),
        ("Desprezo puro",    {"alegria": 0.01, "tristeza": 0.05, "raiva": 0.03,
                              "medo": 0.03, "aversao": 0.03, "surpresa": 0.03, "desprezo": 0.82}),
        ("Surpresa pura",    {"alegria": 0.05, "tristeza": 0.02, "raiva": 0.02,
                              "medo": 0.03, "aversao": 0.02, "surpresa": 0.84, "desprezo": 0.02}),
        ("Misto neutro",     {"alegria": 0.15, "tristeza": 0.15, "raiva": 0.15,
                              "medo": 0.15, "aversao": 0.15, "surpresa": 0.13, "desprezo": 0.12}),
    ]
 
    print(f"{'Caso':<22} {'Valência':>9} {'Arousal':>9} {'Score':>7} {'Label'}")
    print("-" * 70)
    for nome, prob in casos:
        v, a = fis._calcular_valencia_arousal(prob)
        r    = fis.analisar(prob)
        print(f"{nome:<22} {v:>9.3f} {a:>9.3f} {r['score']:>7.4f}  {r['label']}")