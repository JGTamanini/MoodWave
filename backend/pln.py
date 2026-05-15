"""
=============================================================
N2 - Sistema Inteligente | Camada I: PLN com Naive Bayes
=============================================================
Responsável : Felipe
Disciplina  : Inteligência Artificial
Professor   : Claudinei Dias (Ney)

Descrição:
    Classificador de emoções baseado nas 7 categorias de Ekman
    (Alegria, Tristeza, Raiva, Medo, Aversao, Surpresa, Neutro).
    Treinado com o dataset ISEAR — perfeitamente balanceado.
    Saída: vetor de probabilidades para o Sistema Fuzzy (Camada II).

─────────────────────────────────────────────────────────────
SETUP — cada membro do grupo faz isso UMA VEZ na própria máquina:

    1. pip install nltk scikit-learn pandas numpy joblib

    2. python -c "import nltk; nltk.download('stopwords');
                  nltk.download('punkt'); nltk.download('punkt_tab')"

    3. Baixe o dataset em:
       https://www.kaggle.com/datasets/dalopeza/isear-dataset
       Arquivos necessários:
           train_data.csv
           test_data.csv   (opcional — para avaliação extra)
       Coloque em: MoodWave/backend/data/isear/

    4. python pln.py

DATASET: ISEAR (International Survey on Emotion Antecedents and Reactions)
    Formato  : CSV com cabeçalho (text | emotion)
    Emoções  : 7 classes originais → mapeadas para Ekman em português
    Idioma   : Inglês
    Amostras : ~6k treino | ~600 teste (balanceado ~860/classe)

    Por que ISEAR e não GoEmotions?
        O GoEmotions tem desbalanceamento severo (Alegria com 29x mais
        amostras que Aversão), forçando undersampling agressivo que
        prejudica o modelo. O ISEAR tem distribuição quase perfeita entre
        as 7 classes (~860 por classe), resultando em F1-Macro superior
        sem necessidade de técnicas de balanceamento.

MAPEAMENTO ISEAR → EKMAN (português):
    joy     → Alegria
    anger   → Raiva
    fear    → Medo
    sadness → Tristeza
    disgust → Aversao
    guilt   → Tristeza  (culpa é subcategoria de tristeza em Ekman)
    shame   → Tristeza  (vergonha é subcategoria de tristeza em Ekman)

    Nota: O ISEAR não tem classe "surpresa" nem "neutro".
    O sistema retorna probabilidade 0.0 para essas classes,
    o que é academicamente justificável — o dataset não as contempla.

ESTRUTURA DE PASTAS:
    MoodWave/backend/
    ├── pln.py
    ├── data/
    │   └── isear/
    │       ├── train_data.csv    (subir no GitHub — 687KB)
    │       └── test_data.csv     (subir no GitHub — 90KB)
    └── modelo_nb.pkl             (no .gitignore)
=============================================================
"""

import re
import os
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

    ISEAR é em inglês, então:
        - Stop words em inglês
        - Porter Stemmer (padrão para inglês)

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
        texto = re.sub(r"http\S+|www\S+", "", texto)   # URLs
        texto = re.sub(r"@\w+",           "", texto)   # menções
        texto = re.sub(r"#\w+",           "", texto)   # hashtags
        texto = re.sub(r"[^a-z\s]",       "", texto)   # pontuação/números
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
# 2. CARREGADOR DE DATASET
# ─────────────────────────────────────────────

class CarregadorDataset:
    """
    Carrega o ISEAR e mapeia as 7 emoções originais para Ekman em português.

    Fluxo:
        1. Cache existe  →  carrega direto
        2. CSV existe    →  processa, mapeia e gera cache
        3. Nenhum        →  lança erro orientando o usuário
    """

    PASTA_ISEAR    = os.path.join(BASE_DIR, "data", "isear")
    ARQUIVO_TREINO = os.path.join(BASE_DIR, "data", "isear", "train_data.csv")
    ARQUIVO_TESTE  = os.path.join(BASE_DIR, "data", "isear", "test_data.csv")
    ARQUIVO_CACHE  = os.path.join(BASE_DIR, "data", "isear", "cache_ekman.csv")
    ARQUIVO_CSV    = os.path.join(BASE_DIR, "data", "isear", "train_data.csv")  # compatibilidade

    # Mapeamento ISEAR → Ekman em português
    # guilt e shame são subcategorias de tristeza no modelo de Ekman
    MAPA_EMOCOES = {
        "joy":     "Alegria",
        "anger":   "Raiva",
        "fear":    "Medo",
        "sadness": "Tristeza",
        "disgust": "Aversao",
        "guilt":   "Tristeza",   # culpa → tristeza (Ekman)
        "shame":   "Tristeza",   # vergonha → tristeza (Ekman)
    }

    def _processar_csv(self) -> pd.DataFrame:
        """Lê train_data.csv + test_data.csv, aplica mapeamento e salva cache."""
        print(f"[Dataset] Lendo '{self.ARQUIVO_TREINO}'...")
        df_treino = pd.read_csv(self.ARQUIVO_TREINO, encoding="utf-8")

        # Melhoria 1: concatena test_data se existir — mais dados = modelo melhor
        if os.path.exists(self.ARQUIVO_TESTE):
            print(f"[Dataset] Concatenando '{self.ARQUIVO_TESTE}'...")
            df_teste = pd.read_csv(self.ARQUIVO_TESTE, encoding="utf-8")
            df = pd.concat([df_treino, df_teste], ignore_index=True)
        else:
            print("[Dataset] test_data.csv não encontrado, usando só train_data.")
            df = df_treino

        print(f"[Dataset] Colunas encontradas: {list(df.columns)}")
        print(f"[Dataset] Classes originais: {df['emotion'].unique().tolist()}")

        df["sentiment"] = df["emotion"].str.strip().str.lower().map(self.MAPA_EMOCOES)
        df = df.dropna(subset=["sentiment"])
        df = df[["text", "sentiment"]]

        os.makedirs(self.PASTA_ISEAR, exist_ok=True)
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
        elif os.path.exists(self.ARQUIVO_CSV):
            df = self._processar_csv()
        else:
            raise FileNotFoundError(
                "\n"
                "╔══════════════════════════════════════════════════════════╗\n"
                "║           Arquivos do ISEAR não encontrados!             ║\n"
                "╠══════════════════════════════════════════════════════════╣\n"
                "║  Baixe em: kaggle.com/datasets/dalopeza/isear-dataset    ║\n"
                "║  Coloque em: MoodWave/backend/data/isear/                ║\n"
                "║  Arquivo necessário: train_data.csv                      ║\n"
                "╚══════════════════════════════════════════════════════════╝\n"
            )

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
    Classificador ComplementNB com TF-IDF para as emoções de Ekman.

    Pipeline interno:
        TfidfVectorizer (unigrams + bigrams) → ComplementNB

    Por que ComplementNB?
        Matematicamente superior ao MultinomialNB para classificação
        multiclasse — treina usando o complemento de cada classe,
        reduzindo o viés em distribuições assimétricas.

    Saída para a Camada II (Fuzzy do João):
        {
            "classe": "Alegria",
            "probabilidades": {
                "Alegria":  0.72,
                "Tristeza": 0.10,
                "Raiva":    0.05,
                "Medo":     0.08,
                "Aversao":  0.05,
                "Surpresa": 0.00,
                "Neutro":   0.00,
            }
        }

    Nota: Surpresa e Neutro sempre retornam 0.0 pois o ISEAR
    não contempla essas classes. O Fuzzy do João deve tratar esse caso.
    """

    # Todas as 7 emoções de Ekman — garante chaves consistentes na saída
    EMOCOES_EKMAN = ["Alegria", "Aversao", "Medo", "Neutro",
                     "Raiva", "Surpresa", "Tristeza"]

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
            labels       : lista com emoções em português
            test_size    : proporção do conjunto de teste (padrão 20%)
            random_state : semente para reprodutibilidade

        Returns:
            dict com accuracy, f1_macro, classification_report, confusion_matrix
        """
        print("[Treino] Pré-processando textos...")
        textos_proc = [self.preprocessador.processar(t) for t in textos]

        X_train, X_test, y_train, y_test = train_test_split(
            textos_proc, labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels,        # mantém proporção de classes no split
        )

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),     # unigrams + bigrams
                max_features=60_000,    # levemente aumentado
                min_df=2,               # ignora termos que aparecem < 2x
                sublinear_tf=True,      # aplica log na frequência do termo
            )),
            ("nb", ComplementNB(alpha=0.3)),  # alpha ajustado — melhor que 1.0 e 0.1
        ])

        print("[Treino] Treinando ComplementNB...")
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
                "classe": "Alegria",
                "probabilidades": {
                    "Alegria":  0.72,   ← João usa este vetor no Fuzzy
                    "Tristeza": 0.10,
                    "Raiva":    0.05,
                    "Medo":     0.08,
                    "Aversao":  0.05,
                    "Surpresa": 0.00,   ← sempre 0.0 (ISEAR não tem essa classe)
                    "Neutro":   0.00,   ← sempre 0.0 (ISEAR não tem essa classe)
                }
            }

        Exemplo de uso pelo João (Camada II):
            from pln import ClassificadorSentimentos

            clf = ClassificadorSentimentos()
            clf.carregar()

            resultado = clf.analisar_texto("I feel amazing today!")
            emocao    = resultado["classe"]                         # "Alegria"
            probs     = resultado["probabilidades"]                 # dict com 7 emoções
            alegria   = probs["Alegria"]                           # ex: 0.72
        """
        if self.pipeline is None:
            raise RuntimeError("Modelo não carregado. Chame treinar() ou carregar().")

        texto_proc = self.preprocessador.processar(texto)
        classe     = self.pipeline.predict([texto_proc])[0]
        probs      = self.pipeline.predict_proba([texto_proc])[0]
        classes    = self.pipeline.classes_

        # Monta dict com probabilidades das classes treinadas
        probabilidades = {
            cls: round(float(prob), 4)
            for cls, prob in zip(classes, probs)
        }

        # Garante todas as 7 emoções de Ekman na saída (ausentes = 0.0)
        for emocao in self.EMOCOES_EKMAN:
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
        ("I feel so guilty about what I did to them.",       "Tristeza esperada (guilt)"),
        ("I'm ashamed of my behavior, it was wrong.",        "Tristeza esperada (shame)"),
    ]

    for texto, esperado in frases_teste:
        r = classificador.analisar_texto(texto)
        print(f"\nTexto    : {texto}")
        print(f"Esperado : {esperado}")
        print(f"Classe   : {r['classe']}")
        print("Probabilidades:")
        for emocao, prob in sorted(r["probabilidades"].items(),
                                   key=lambda x: x[1], reverse=True):
            if prob > 0:
                barra = "█" * int(prob * 40)
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