"""
=============================================================
N2 - Sistema Inteligente | Camada I: PLN com Naive Bayes
=============================================================
Responsável : Felipe
Disciplina  : Inteligência Artificial
Professor   : Claudinei Dias (Ney)

Descrição:
    Classificador de emoções baseado nas 7 categorias de Ekman
    (alegria, tristeza, raiva, medo, aversao, surpresa, neutro).
    Treinado com o dataset GoEmotions (Google, 2020).
    Saída: vetor de probabilidades para o Sistema Fuzzy (Camada II).

─────────────────────────────────────────────────────────────
SETUP — cada membro do grupo faz isso UMA VEZ na própria máquina:

    1. pip install nltk scikit-learn pandas numpy joblib

    2. python -c "import nltk; nltk.download('stopwords');
                  nltk.download('punkt'); nltk.download('punkt_tab')"

    3. Baixe o dataset em:
       https://www.kaggle.com/datasets/debarshichanda/goemotions
       Arquivos necessários (pasta data/):
           train.tsv
           test.tsv
           emotions.txt
           ekman_mapping.json
       Coloque em: MoodWave/backend/data/goemotions/

    4. python pln.py

DATASET: GoEmotions (Google, 2020)
    Formato  : TSV sem cabeçalho (texto | label_id | id_reddit)
    Emoções  : 27 originais → reduzidas a 7 via ekman_mapping.json
    Idioma   : Inglês (comentários do Reddit)
    Amostras : ~43k treino | ~5k teste

MAPEAMENTO EKMAN (confirmado via ekman_mapping.json):
    joy      → Alegria   (joy, amusement, approval, excitement,
                           gratitude, love, optimism, relief,
                           pride, admiration, desire, caring)
    sadness  → Tristeza  (sadness, disappointment, embarrassment,
                           grief, remorse)
    anger    → Raiva     (anger, annoyance, disapproval)
    fear     → Medo      (fear, nervousness)
    disgust  → Aversao   (disgust)
    surprise → Surpresa  (surprise, realization, confusion, curiosity)
    neutral  → Neutro    (neutral — label 27)

ESTRUTURA DE PASTAS:
    MoodWave/backend/
    ├── pln.py
    ├── data/
    │   └── goemotions/
    │       ├── train.tsv              (subir no GitHub — 3.4MB)
    │       ├── test.tsv               (subir no GitHub — 427KB)
    │       ├── emotions.txt           (subir no GitHub — 1KB)
    │       └── ekman_mapping.json     (subir no GitHub — 1KB)
    └── modelo_nb.pkl                  (no .gitignore)
=============================================================
"""

import re
import os
import json
import joblib
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from sklearn.pipeline import Pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────
# 1. PRÉ-PROCESSADOR DE TEXTO
# ─────────────────────────────────────────────

class PreProcessador:
    """
    Pipeline de limpeza e normalização de texto em inglês.

    GoEmotions é em inglês (Reddit), então:
        - Stop words em inglês
        - Porter Stemmer (padrão para inglês)
        - Sem RSLP (era específico para português)

    Etapas:
        1. Lowercase
        2. Remoção de URLs, menções (@) e hashtags (#)
        3. Remoção de pontuação e números
        4. Tokenização
        5. Remoção de stop words (inglês)
        6. Stemming com Porter Stemmer
    """

    def __init__(self):
        nltk.download("stopwords", quiet=True)
        nltk.download("punkt",     quiet=True)
        nltk.download("punkt_tab", quiet=True)

        self.stop_words = set(stopwords.words("english"))
        self.stemmer    = PorterStemmer()

    def limpar(self, texto: str) -> str:
        texto = texto.lower()
        texto = re.sub(r"http\S+|www\S+",      "", texto)  # URLs
        texto = re.sub(r"@\w+",                 "", texto)  # menções
        texto = re.sub(r"#\w+",                 "", texto)  # hashtags
        texto = re.sub(r"[^a-z\s]",             "", texto)  # pontuação/números
        texto = re.sub(r"\s+",                  " ", texto).strip()
        return texto

    def tokenizar(self, texto: str) -> list:
        tokens = word_tokenize(texto)
        return [t for t in tokens if t not in self.stop_words and len(t) > 2]

    def stemmizar(self, tokens: list) -> list:
        return [self.stemmer.stem(t) for t in tokens]

    def processar(self, texto: str) -> str:
        """Pipeline completo — retorna string processada para o TF-IDF."""
        texto  = self.limpar(texto)
        tokens = self.tokenizar(texto)
        tokens = self.stemmizar(tokens)
        return " ".join(tokens)


# ─────────────────────────────────────────────
# 2. CARREGADOR DE DATASET
# ─────────────────────────────────────────────

class CarregadorDataset:
    """
    Carrega e processa o GoEmotions reduzindo 27 emoções para 7 (Ekman).

    Fluxo:
        1. Cache existe  →  carrega direto
        2. TSV existe    →  processa com mapeamento Ekman e gera cache
        3. Nenhum        →  lança erro orientando o usuário

    Sobre multilabel:
        O GoEmotions permite múltiplos labels por texto (ex: "2,14").
        Estratégia adotada: pega o PRIMEIRO label — o anotador principal.
        Isso simplifica para classificação single-label com Naive Bayes.
    """

    PASTA_GOEMOTIONS = os.path.join(BASE_DIR, "data", "goemotions")
    ARQUIVO_TREINO   = os.path.join(BASE_DIR, "data", "goemotions", "train.tsv")
    ARQUIVO_EMOCOES  = os.path.join(BASE_DIR, "data", "goemotions", "emotions.txt")
    ARQUIVO_EKMAN    = os.path.join(BASE_DIR, "data", "goemotions", "ekman_mapping.json")
    ARQUIVO_CACHE    = os.path.join(BASE_DIR, "data", "goemotions", "cache_ekman.csv")

    # Tradução das 6 emoções Ekman + neutro para português
    TRADUCAO = {
        "joy":      "Alegria",
        "sadness":  "Tristeza",
        "anger":    "Raiva",
        "fear":     "Medo",
        "disgust":  "Aversao",
        "surprise": "Surpresa",
        "neutral":  "Neutro",
    }

    def _construir_mapa_id_ekman(self) -> dict:
        """
        Constrói dicionário: id_numerico → emoção_ekman_em_português

        Exemplo: 17 (joy) → "Alegria"
                  0 (admiration, que mapeia para joy) → "Alegria"
                 27 (neutral) → "Neutro"
        """
        # Carrega lista de emoções (índice = id numérico)
        with open(self.ARQUIVO_EMOCOES, encoding="utf-8") as f:
            emocoes = [linha.strip() for linha in f]  # lista com 28 emoções

        # Carrega mapeamento Ekman (ekman → [lista de emoções originais])
        with open(self.ARQUIVO_EKMAN, encoding="utf-8") as f:
            ekman_map = json.load(f)

        # Inverte: emoção_original → ekman
        emocao_para_ekman = {}
        for ekman, lista in ekman_map.items():
            for emocao in lista:
                emocao_para_ekman[emocao] = ekman

        # Monta id → ekman_pt
        mapa = {}
        for idx, emocao in enumerate(emocoes):
            if emocao == "neutral":
                mapa[idx] = "Neutro"
            elif emocao in emocao_para_ekman:
                ekman = emocao_para_ekman[emocao]
                mapa[idx] = self.TRADUCAO[ekman]
            # ids sem mapeamento são ignorados (dropna depois)

        return mapa

    def _processar_tsv(self) -> pd.DataFrame:
        """Lê train.tsv, aplica mapeamento Ekman e salva cache."""
        print(f"[Dataset] Lendo '{self.ARQUIVO_TREINO}'...")
        df = pd.read_csv(
            self.ARQUIVO_TREINO,
            sep="\t",
            header=None,
            names=["text", "labels", "id"],
        )

        # GoEmotions pode ter múltiplos labels separados por vírgula
        # Estratégia: pega o primeiro label (anotador principal)
        mapa_id_ekman = self._construir_mapa_id_ekman()
        df["label_id"] = df["labels"].astype(str).str.split(",").str[0].astype(int)
        df["emocao"]   = df["label_id"].map(mapa_id_ekman)
        df = df.dropna(subset=["emocao"])
        df = df[["text", "emocao"]].rename(columns={"emocao": "sentiment"})

        os.makedirs(self.PASTA_GOEMOTIONS, exist_ok=True)
        df.to_csv(self.ARQUIVO_CACHE, index=False, encoding="utf-8")
        print(f"[Dataset] Cache salvo em '{self.ARQUIVO_CACHE}'")
        return df

    def carregar(self) -> tuple:
        """
        Ponto de entrada principal. Retorna (textos, labels).
        """
        if os.path.exists(self.ARQUIVO_CACHE):
            print(f"[Dataset] Cache encontrado. Carregando...")
            df = pd.read_csv(self.ARQUIVO_CACHE, encoding="utf-8")
        elif os.path.exists(self.ARQUIVO_TREINO):
            df = self._processar_tsv()
        else:
            raise FileNotFoundError(
                "\n"
                "╔══════════════════════════════════════════════════════════╗\n"
                "║           Arquivos do GoEmotions não encontrados!        ║\n"
                "╠══════════════════════════════════════════════════════════╣\n"
                "║  Baixe em: kaggle.com/datasets/debarshichanda/goemotions ║\n"
                "║  Coloque em: MoodWave/backend/data/goemotions/           ║\n"
                "║  Arquivos: train.tsv | emotions.txt | ekman_mapping.json ║\n"
                "╚══════════════════════════════════════════════════════════╝\n"
            )

        df      = df[["text", "sentiment"]].dropna()
        textos  = df["text"].astype(str).tolist()
        labels  = df["sentiment"].tolist()

        print(f"[Dataset] {len(textos)} amostras carregadas.")
        print(f"[Dataset] Distribuição:\n{pd.Series(labels).value_counts().to_string()}\n")
        return textos, labels


# ─────────────────────────────────────────────
# 3. CLASSIFICADOR DE EMOÇÕES
# ─────────────────────────────────────────────

class ClassificadorSentimentos:
    """
    Classificador Naive Bayes com TF-IDF para as 7 emoções de Ekman.

    Pipeline interno:
        TfidfVectorizer (unigrams + bigrams) → ComplementNB

    Por que ComplementNB?
        O GoEmotions é severamente desbalanceado (Alegria tem 29x mais
        amostras que Aversão). O ComplementNB foi desenvolvido exatamente
        para esse cenário — ele treina usando o complemento de cada classe,
        sendo matematicamente superior ao MultinomialNB em dados desiguais.

    Saída principal para a Camada II (Fuzzy do João):
        {
            "classe": "Alegria",
            "probabilidades": {
                "Alegria":  0.72,
                "Tristeza": 0.05,
                "Raiva":    0.03,
                "Medo":     0.08,
                "Aversao":  0.02,
                "Surpresa": 0.07,
                "Neutro":   0.03,
            }
        }
    """

    EMOCOES = ["Alegria", "Aversao", "Medo", "Neutro", "Raiva", "Surpresa", "Tristeza"]

    def __init__(self, caminho_modelo: str = os.path.join(BASE_DIR, "modelo_nb.pkl")):
        self.caminho_modelo = caminho_modelo
        self.preprocessador = PreProcessador()
        self.pipeline       = None

    def treinar(self, textos: list, labels: list,
                test_size: float = 0.2, random_state: int = 42) -> dict:
        """
        Treina o modelo e retorna métricas de avaliação.

        Args:
            textos       : lista de strings brutas
            labels       : lista com as 7 emoções de Ekman em português
            test_size    : proporção do conjunto de teste (padrão 20%)
            random_state : semente para reprodutibilidade

        Returns:
            dict com accuracy, f1_macro, classification_report, confusion_matrix
        """
        print("[Treino] Pré-processando textos...")
        textos_proc = [self.preprocessador.processar(t) for t in textos]

        # ── Balanceamento por undersampling ──────────────────────────
        import random as _random
        _random.seed(random_state)

        df_treino  = pd.DataFrame({"text": textos_proc, "label": labels})
        min_classe = df_treino["label"].value_counts().min()
        limite     = min(min_classe * 4, 3000)

        partes = []
        for classe in df_treino["label"].unique():
            subset = df_treino[df_treino["label"] == classe]
            partes.append(subset.sample(min(len(subset), limite),
                                        random_state=random_state))

        df_balanceado = pd.concat(partes).sample(frac=1, random_state=random_state)
        textos_proc   = df_balanceado["text"].tolist()
        labels        = df_balanceado["label"].tolist()
        
        
        print(f"[Treino] Distribuição após balanceamento:")
        print(pd.Series(labels).value_counts().to_string())
        print()

        X_train, X_test, y_train, y_test = train_test_split(
            textos_proc, labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels,
        )

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=50_000,
                min_df=2,
                sublinear_tf=True,
            )),
            # ComplementNB — superior ao MultinomialNB para dados desbalanceados
            ("nb", ComplementNB(alpha=1.0)),
        ])

        print("[Treino] Treinando MultinomialNB (7 classes Ekman)...")
        self.pipeline.fit(X_train, y_train)

        y_pred   = self.pipeline.predict(X_test)
        metricas = {
            "accuracy":              round(accuracy_score(y_test, y_pred), 4),
            "f1_macro":              round(f1_score(y_test, y_pred, average="macro"), 4),
            "classification_report": classification_report(y_test, y_pred, zero_division=0),
            "confusion_matrix":      confusion_matrix(y_test, y_pred).tolist(),
        }

        print(f"\n[Avaliação] Accuracy : {metricas['accuracy']}")
        print(f"[Avaliação] F1 Macro : {metricas['f1_macro']}")
        print(f"\n{metricas['classification_report']}")
        return metricas

    def salvar(self):
        """Serializa o modelo treinado em disco com joblib."""
        if self.pipeline is None:
            raise RuntimeError("Modelo não treinado. Chame treinar() primeiro.")
        joblib.dump(self.pipeline, self.caminho_modelo)
        print(f"[Modelo] Salvo em '{self.caminho_modelo}'")

    def carregar(self):
        """Carrega modelo serializado do disco."""
        if not os.path.exists(self.caminho_modelo):
            raise FileNotFoundError(
                f"Modelo não encontrado em '{self.caminho_modelo}'.\n"
                "Treine o modelo antes de carregar."
            )
        self.pipeline = joblib.load(self.caminho_modelo)
        print(f"[Modelo] Carregado de '{self.caminho_modelo}'")

    def analisar_texto(self, texto: str) -> dict:
        """
        ⭐ INTERFACE COM A CAMADA II — chame esta função no Fuzzy do João.

        Args:
            texto : string em inglês digitada pelo usuário

        Returns:
            {
                "classe": "Alegria",          ← emoção dominante
                "probabilidades": {
                    "Alegria":  0.72,         ← João usa este vetor no Fuzzy
                    "Tristeza": 0.05,
                    "Raiva":    0.03,
                    "Medo":     0.08,
                    "Aversao":  0.02,
                    "Surpresa": 0.07,
                    "Neutro":   0.03,
                }
            }

        Exemplo de uso pelo João (Camada II):
            from pln import ClassificadorSentimentos

            clf = ClassificadorSentimentos()
            clf.carregar()

            resultado = clf.analisar_texto("I feel amazing today!")
            emocao_dominante = resultado["classe"]          # "Alegria"
            prob_alegria     = resultado["probabilidades"]["Alegria"]  # 0.72
        """
        if self.pipeline is None:
            raise RuntimeError("Modelo não carregado. Chame treinar() ou carregar().")

        texto_proc = self.preprocessador.processar(texto)
        classe     = self.pipeline.predict([texto_proc])[0]
        probs      = self.pipeline.predict_proba([texto_proc])[0]
        classes    = self.pipeline.classes_

        probabilidades = {
            cls: round(float(prob), 4)
            for cls, prob in zip(classes, probs)
        }

        # Garante que todas as 7 emoções estão presentes (mesmo que com 0.0)
        for emocao in self.EMOCOES:
            probabilidades.setdefault(emocao, 0.0)

        return {
            "classe":         classe,
            "probabilidades": probabilidades,
        }

    def analisar_lote(self, textos: list) -> list:
        """Analisa uma lista de textos de uma vez."""
        return [self.analisar_texto(t) for t in textos]


# ─────────────────────────────────────────────
# 4. BLOCO PRINCIPAL
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # 1. Carrega dataset
    carregador     = CarregadorDataset()
    textos, labels = carregador.carregar()

    # 2. Treina e avalia
    classificador = ClassificadorSentimentos()
    classificador.treinar(textos, labels)

    # 3. Salva modelo serializado
    classificador.salvar()

    # 4. Testa com frases de exemplo
    print("\n" + "="*60)
    print("TESTE COM FRASES DE EXEMPLO")
    print("="*60)

    frases_teste = [
        ("I feel amazing today, everything is going great!", "Alegria esperada"),
        ("I am so angry, this is completely unacceptable.",  "Raiva esperada"),
        ("I'm terrified of what might happen next.",         "Medo esperado"),
        ("This is disgusting, I can't believe it.",          "Aversao esperada"),
        ("I'm heartbroken and devastated by the news.",      "Tristeza esperada"),
        ("Wow, I had no idea that would happen!",            "Surpresa esperada"),
        ("I went to the store and bought some groceries.",   "Neutro esperado"),
    ]

    for texto, esperado in frases_teste:
        r = classificador.analisar_texto(texto)
        print(f"\nTexto    : {texto}")
        print(f"Esperado : {esperado}")
        print(f"Classe   : {r['classe']}")
        print("Probabilidades:")
        for emocao, prob in sorted(r["probabilidades"].items(),
                                   key=lambda x: x[1], reverse=True):
            barra = "█" * int(prob * 30)
            print(f"  {emocao:<10} {prob:.4f} {barra}")

    # 5. Interface com a Camada II
    print("\n" + "="*60)
    print("INTERFACE COM CAMADA II (Fuzzy — João)")
    print("="*60)
    texto_usuario = "I feel so happy and excited about this!"
    saida = classificador.analisar_texto(texto_usuario)
    print(f"Texto          : '{texto_usuario}'")
    print(f"Classe         : {saida['classe']}")
    print(f"Probabilidades : {saida['probabilidades']}")
    print("\nJoão recebe o dicionário 'probabilidades' em calcular_humor()")