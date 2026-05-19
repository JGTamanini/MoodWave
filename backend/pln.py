"""
=============================================================
N2 - Sistema Inteligente | Camada I: PLN com Naive Bayes
=============================================================
Responsável : Felipe
Disciplina  : Inteligência Artificial
Professor   : Claudinei Dias (Ney)

Descrição:
    Classificador de emoções baseado nas categorias de Ekman
    (Alegria, Tristeza, Raiva, Medo, Aversao, Surpresa).
    Dataset híbrido: ISEAR + Emotion Dataset (Parulpandey).
    Balanceamento via SMOTE (oversampling sintético).
    Saída: vetor de probabilidades para o Sistema Fuzzy (Camada II).

─────────────────────────────────────────────────────────────
SETUP — cada membro do grupo faz isso UMA VEZ na própria máquina:

    1. pip install nltk scikit-learn pandas numpy joblib imbalanced-learn

    2. python -c "import nltk; nltk.download('stopwords');
                  nltk.download('punkt'); nltk.download('punkt_tab')"

    3. Coloque os arquivos nas pastas:
       MoodWave/backend/data/isear/
           train_data.csv
           test_data.csv
       MoodWave/backend/data/emotion-dataset/
           training.csv
           test.csv        (opcional)
           validation.csv  (opcional)

    4. python pln.py

DATASETS:
    ISEAR (International Survey on Emotion Antecedents and Reactions)
        ~6.7k amostras | 5 classes | perfeitamente balanceado
        Fonte: kaggle.com/datasets/dalopeza/isear-dataset

    Emotion Dataset (Parulpandey / CARER - Saravia et al. 2018)
        ~16k amostras | 6 classes | tweets em inglês
        Fonte: kaggle.com/datasets/parulpandey/emotion-dataset

    Por que híbrido?
        ISEAR tem qualidade alta e Raiva/Medo bem representados.
        Emotion Dataset adiciona Surpresa (ausente no ISEAR) e
        aumenta o volume geral, especialmente de Alegria e Tristeza.

MAPEAMENTO → EKMAN (português):
    ISEAR  : joy→Alegria | anger→Raiva | fear→Medo
             sadness→Tristeza | disgust→Aversao
             guilt→Tristeza | shame→Tristeza
    Emotion: 0(sadness)→Tristeza | 1(joy)→Alegria | 2(love)→Alegria
             3(anger)→Raiva | 4(fear)→Medo | 5(surprise)→Surpresa

    Nota: Aversao vem só do ISEAR. Neutro não está nos datasets
    — retorna 0.0 na saída (João deve tratar no Fuzzy).

ESTRUTURA DE PASTAS:
    MoodWave/backend/
    ├── pln.py
    ├── data/
    │   ├── isear/
    │   │   ├── train_data.csv         (subir no GitHub)
    │   │   └── test_data.csv          (subir no GitHub)
    │   └── emotion-dataset/
    │       └── training.csv           (subir no GitHub — 1.5MB)
    └── modelo_nb.pkl                  (no .gitignore)
=============================================================
"""

import re
import os
import joblib
import pandas as pd
import numpy as np

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
        texto = re.sub(r"http\S+|www\S+", "", texto)
        texto = re.sub(r"@\w+",           "", texto)
        texto = re.sub(r"#\w+",           "", texto)
        texto = re.sub(r"[^a-z\s]",       "", texto)
        texto = re.sub(r"\s+",            " ", texto).strip()
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
# 2. CARREGADOR DE DATASET HÍBRIDO
# ─────────────────────────────────────────────

class CarregadorDataset:
    """
    Carrega e combina ISEAR + Emotion Dataset, mapeando para Ekman.

    Fluxo:
        1. Cache existe  →  carrega direto
        2. CSVs existem  →  processa, combina e gera cache
        3. Só ISEAR      →  usa só ISEAR (fallback parcial)
        4. Nenhum        →  lança erro
    """

    # ISEAR
    ISEAR_TREINO  = os.path.join(BASE_DIR, "data", "isear", "train_data.csv")
    ISEAR_TESTE   = os.path.join(BASE_DIR, "data", "isear", "test_data.csv")

    # Emotion Dataset
    EMOTION_TREINO = os.path.join(BASE_DIR, "data", "emotion-dataset", "training.csv")
    EMOTION_TESTE  = os.path.join(BASE_DIR, "data", "emotion-dataset", "test.csv")
    EMOTION_VAL    = os.path.join(BASE_DIR, "data", "emotion-dataset", "validation.csv")

    # Cache
    ARQUIVO_CACHE = os.path.join(BASE_DIR, "data", "cache_hibrido.csv")

    # Mapeamento ISEAR → Ekman português
    MAPA_ISEAR = {
        "joy":     "Alegria",
        "anger":   "Raiva",
        "fear":    "Medo",
        "sadness": "Tristeza",
        "disgust": "Aversao",
        "guilt":   "Tristeza",
        "shame":   "Tristeza",
    }

    # Mapeamento Emotion Dataset (numérico) → Ekman português
    MAPA_EMOTION = {
        0: "Tristeza",  # sadness
        1: "Alegria",   # joy
        2: "Alegria",   # love → subcategoria de alegria em Ekman
        3: "Raiva",     # anger
        4: "Medo",      # fear
        5: "Surpresa",  # surprise
    }

    def _carregar_isear(self) -> pd.DataFrame:
        """Carrega e combina train + test do ISEAR."""
        partes = []
        for caminho in [self.ISEAR_TREINO, self.ISEAR_TESTE]:
            if os.path.exists(caminho):
                df = pd.read_csv(caminho, encoding="utf-8")
                partes.append(df)

        if not partes:
            return pd.DataFrame(columns=["text", "sentiment"])

        df = pd.concat(partes, ignore_index=True)
        df["sentiment"] = df["emotion"].str.strip().str.lower().map(self.MAPA_ISEAR)
        df = df.dropna(subset=["sentiment"])
        print(f"[Dataset] ISEAR: {len(df)} amostras")
        return df[["text", "sentiment"]]

    def _carregar_emotion(self) -> pd.DataFrame:
        """Carrega training + test + validation do Emotion Dataset."""
        partes = []
        for caminho in [self.EMOTION_TREINO, self.EMOTION_TESTE, self.EMOTION_VAL]:
            if os.path.exists(caminho):
                df = pd.read_csv(caminho, encoding="utf-8")
                partes.append(df)

        if not partes:
            return pd.DataFrame(columns=["text", "sentiment"])

        df = pd.concat(partes, ignore_index=True)
        df["sentiment"] = df["label"].map(self.MAPA_EMOTION)
        df = df.dropna(subset=["sentiment"])
        print(f"[Dataset] Emotion Dataset: {len(df)} amostras")
        return df[["text", "sentiment"]]

    def _processar_hibrido(self) -> pd.DataFrame:
        """Combina os dois datasets e salva cache."""
        print("[Dataset] Processando dataset híbrido ISEAR + Emotion Dataset...")
        df_isear   = self._carregar_isear()
        df_emotion = self._carregar_emotion()

        if df_isear.empty and df_emotion.empty:
            raise FileNotFoundError(
                "\n"
                "╔══════════════════════════════════════════════════════════╗\n"
                "║              Nenhum dataset encontrado!                  ║\n"
                "╠══════════════════════════════════════════════════════════╣\n"
                "║  ISEAR   : backend/data/isear/train_data.csv             ║\n"
                "║  Emotion : backend/data/emotion-dataset/training.csv     ║\n"
                "╚══════════════════════════════════════════════════════════╝\n"
            )

        df = pd.concat([df_isear, df_emotion], ignore_index=True)
        df = df.dropna(subset=["text", "sentiment"])
        df = df[df["text"].str.strip() != ""]

        os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
        df.to_csv(self.ARQUIVO_CACHE, index=False, encoding="utf-8")
        print(f"[Dataset] Cache híbrido salvo em '{self.ARQUIVO_CACHE}'")
        return df

    def carregar(self) -> tuple:
        """Ponto de entrada principal. Retorna (textos, labels)."""
        if os.path.exists(self.ARQUIVO_CACHE):
            print(f"[Dataset] Cache encontrado. Carregando...")
            df = pd.read_csv(self.ARQUIVO_CACHE, encoding="utf-8")
        else:
            df = self._processar_hibrido()

        df     = df[["text", "sentiment"]].dropna()
        textos = df["text"].astype(str).tolist()
        labels = df["sentiment"].tolist()

        print(f"[Dataset] {len(textos)} amostras carregadas.")
        print(f"[Dataset] Distribuição:\n{pd.Series(labels).value_counts().to_string()}\n")
        return textos, labels


# ─────────────────────────────────────────────
# 3. CLASSIFICADOR DE EMOÇÕES
# ─────────────────────────────────────────────

class ClassificadorSentimentos:
    """
    Classificador ComplementNB com TF-IDF + SMOTE para as emoções de Ekman.

    Pipeline:
        Texto → TF-IDF → SMOTE (oversampling) → ComplementNB

    Por que SMOTE?
        Gera amostras sintéticas interpolando vizinhos próximos no espaço
        TF-IDF, sem simplesmente duplicar exemplos (oversampling ingênuo).
        Superior ao undersampling porque não descarta dados reais.

    Saída para a Camada II (Fuzzy do João):
        {
            "classe": "Alegria",
            "probabilidades": {
                "Alegria":  0.72,
                "Tristeza": 0.10,
                "Raiva":    0.08,
                "Medo":     0.05,
                "Aversao":  0.03,
                "Surpresa": 0.02,
                "Neutro":   0.00,
            }
        }
    """

    EMOCOES_EKMAN = ["Alegria", "Aversao", "Medo", "Neutro",
                     "Raiva", "Surpresa", "Tristeza"]

    def __init__(self, caminho_modelo: str = os.path.join(BASE_DIR, "modelo_nb.pkl")):
        self.caminho_modelo = caminho_modelo
        self.preprocessador = PreProcessador()
        self.pipeline       = None
        self.classes_       = None

    def treinar(self, textos: list, labels: list,
                test_size: float = 0.2, random_state: int = 42) -> dict:
        """
        Treina o modelo com SMOTE para balancear classes.

        Args:
            textos       : lista de strings brutas
            labels       : lista com emoções em português
            test_size    : proporção do conjunto de teste (padrão 20%)
            random_state : semente para reprodutibilidade

        Returns:
            dict com accuracy, f1_macro, classification_report, confusion_matrix
        """
        from imblearn.over_sampling import SMOTE

        print("[Treino] Pré-processando textos...")
        textos_proc = [self.preprocessador.processar(t) for t in textos]

        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            textos_proc, labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels,
        )

        # Vetoriza antes do SMOTE (SMOTE opera em espaço numérico)
        print("[Treino] Vetorizando com TF-IDF...")
        tfidf = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=60_000,
            min_df=2,
            sublinear_tf=True,
        )
        X_train_tfidf = tfidf.fit_transform(X_train_raw)
        X_test_tfidf  = tfidf.transform(X_test_raw)

        # SMOTE — balanceia classes minoritárias
        print("[Treino] Aplicando SMOTE (oversampling sintético)...")
        contagem = pd.Series(y_train).value_counts()
        k = max(1, min(5, contagem.min() - 1))  # k_neighbors seguro
        smote = SMOTE(random_state=random_state, k_neighbors=k)
        X_bal, y_bal = smote.fit_resample(X_train_tfidf, y_train)

        print(f"[Treino] Distribuição após SMOTE:")
        print(pd.Series(y_bal).value_counts().to_string())
        print()

        # Treina ComplementNB
        print("[Treino] Treinando ComplementNB...")
        nb = ComplementNB(alpha=0.3)
        nb.fit(X_bal, y_bal)

        # Guarda tfidf + nb separados (SMOTE não entra no pipeline de inferência)
        self.pipeline = {"tfidf": tfidf, "nb": nb}
        self.classes_ = nb.classes_

        # Avaliação no conjunto de teste (sem SMOTE — dados reais)
        y_pred   = nb.predict(X_test_tfidf)
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
        """Serializa tfidf + nb em disco com joblib."""
        if self.pipeline is None:
            raise RuntimeError("Modelo não treinado. Chame treinar() primeiro.")
        joblib.dump({"pipeline": self.pipeline, "classes": self.classes_},
                    self.caminho_modelo)
        print(f"[Modelo] Salvo em '{self.caminho_modelo}'")

    def carregar(self):
        """Carrega modelo serializado do disco."""
        if not os.path.exists(self.caminho_modelo):
            raise FileNotFoundError(
                f"Modelo não encontrado em '{self.caminho_modelo}'.\n"
                "Treine o modelo antes de carregar."
            )
        dados          = joblib.load(self.caminho_modelo)
        self.pipeline  = dados["pipeline"]
        self.classes_  = dados["classes"]
        print(f"[Modelo] Carregado de '{self.caminho_modelo}'")

    # Limiar abaixo do qual o texto é considerado neutro.
    # Se nenhuma emoção ultrapassa 35%, o modelo está inseguro
    # e provavelmente é uma frase sem carga emocional clara.
    LIMIAR_NEUTRO = 0.28

    def analisar_texto(self, texto: str) -> dict:
        """
        ⭐ INTERFACE COM A CAMADA II — chame esta função no Fuzzy do João.

        Args:
            texto : string em inglês digitada pelo usuário

        Returns:
            {
                "classe": "Alegria",          ← emoção dominante (ou "Neutro")
                "probabilidades": {
                    "Alegria":  0.72,
                    "Tristeza": 0.10,
                    "Raiva":    0.08,
                    "Medo":     0.05,
                    "Aversao":  0.03,
                    "Surpresa": 0.02,
                    "Neutro":   0.00,         ← 1.0 - prob_max se neutro
                }
            }

        Detecção de Neutro:
            Se a probabilidade máxima for menor que LIMIAR_NEUTRO (0.35),
            o modelo está inseguro — provavelmente o texto não tem carga
            emocional clara. Nesse caso, classe = "Neutro".

        Exemplo de uso pelo João (Camada II):
            from pln import ClassificadorSentimentos

            clf = ClassificadorSentimentos()
            clf.carregar()

            resultado = clf.analisar_texto("I feel amazing today!")
            emocao    = resultado["classe"]               # "Alegria"
            probs     = resultado["probabilidades"]
            alegria   = probs["Alegria"]                  # ex: 0.72
        """
        if self.pipeline is None:
            raise RuntimeError("Modelo não carregado. Chame treinar() ou carregar().")

        tfidf = self.pipeline["tfidf"]
        nb    = self.pipeline["nb"]

        texto_proc = self.preprocessador.processar(texto)
        X          = tfidf.transform([texto_proc])
        classe     = nb.predict(X)[0]
        probs      = nb.predict_proba(X)[0]

        probabilidades = {
            str(cls): round(float(prob), 4)
            for cls, prob in zip(self.classes_, probs)
        }

        # Garante todas as 7 emoções de Ekman (ausentes = 0.0)
        for emocao in self.EMOCOES_EKMAN:
            probabilidades.setdefault(emocao, 0.0)

        # ── Detecção de Neutro por limiar ────────────────────────────
        prob_maxima = max(probabilidades[e] for e in self.EMOCOES_EKMAN
                         if e != "Neutro")

        if prob_maxima < self.LIMIAR_NEUTRO:
            classe = "Neutro"
            probabilidades["Neutro"] = round(1.0 - prob_maxima, 4)

        return {
            "classe":         str(classe),
            "probabilidades": probabilidades,
        }

    def analisar_lote(self, textos: list) -> list:
        """Analisa uma lista de textos de uma vez."""
        return [self.analisar_texto(t) for t in textos]


# ─────────────────────────────────────────────
# 4. BLOCO PRINCIPAL
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # 1. Carrega dataset híbrido
    carregador     = CarregadorDataset()
    textos, labels = carregador.carregar()

    # 2. Treina com SMOTE + ComplementNB
    classificador = ClassificadorSentimentos()
    classificador.treinar(textos, labels)

    # 3. Salva modelo
    classificador.salvar()

    # 4. Testa com frases de exemplo
    print("\n" + "="*60)
    print("TESTE COM FRASES DE EXEMPLO")
    print("="*60)

    frases_teste = [
        ("I feel amazing today, everything is going great!", "Alegria"),
        ("I am so angry, this is completely unacceptable.",  "Raiva"),
        ("I'm terrified of what might happen next.",         "Medo"),
        ("This is disgusting, I can't believe it.",          "Aversao"),
        ("I'm heartbroken and devastated by the news.",      "Tristeza"),
        ("Wow, I had no idea that would happen!",            "Surpresa"),
        ("I feel so guilty about what I did to them.",       "Tristeza"),
        ("I went to the store and bought some groceries.",   "Neutro"),
        ("The meeting is scheduled for Tuesday.",            "Neutro"),
        ("It is currently raining outside.",                 "Neutro"),
    ]

    acertos = 0
    for texto, esperado in frases_teste:
        r = classificador.analisar_texto(texto)
        acertou = "✓" if r["classe"] == esperado else "✗"
        if r["classe"] == esperado:
            acertos += 1
        print(f"\n{acertou} Texto    : {texto}")
        print(f"  Esperado : {esperado} | Classe: {r['classe']}")
        print("  Probabilidades:")
        for emocao, prob in sorted(r["probabilidades"].items(),
                                   key=lambda x: x[1], reverse=True):
            if prob > 0:
                barra = "█" * int(prob * 40)
                print(f"    {emocao:<10} {prob:.4f} {barra}")

    print(f"\n[Teste] Acertos: {acertos}/{len(frases_teste)}")

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