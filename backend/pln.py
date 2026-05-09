"""
=============================================================
N2 - Sistema Inteligente | Camada I: PLN com Naive Bayes
=============================================================
Responsável : Felipe
Disciplina  : Inteligência Artificial
Professor   : Claudinei Dias (Ney)

Descrição:
    Classificador de sentimentos (Positivo / Negativo / Neutro)
    treinado com tweets em português brasileiro.
    Saída normalizada para alimentar o Sistema Fuzzy (Camada II).

─────────────────────────────────────────────────────────────
SETUP — cada membro do grupo faz isso UMA VEZ na própria máquina:

    1. pip install nltk scikit-learn pandas numpy joblib

    2. python -c "import nltk; nltk.download('stopwords');
                  nltk.download('rslp'); nltk.download('punkt')"

    3. Baixe o dataset em:
       https://www.kaggle.com/datasets/augustop/portuguese-tweets-for-sentiment-analysis
       Arquivo necessário: Train3Classes.csv
       Coloque em: MoodWave/backend/data/Train3Classes.csv

    4. python pln.py

DATASET: Portuguese Tweets for Sentiment Analysis
    Separador : ponto e vírgula (;)
    Colunas   : id | tweet_text | tweet_date | sentiment | query_used
    Labels    : 0 = Negativo | 1 = Positivo | 2 = Neutro
    Amostras  : ~100k tweets balanceados (33k por classe)

    Justificativa da escolha: o sistema recebe texto informal do
    usuário (como "tô arrasado hoje"), linguagem muito próxima de
    tweets. Datasets de reviews ou notícias não capturam esse padrão.
    O Train3Classes foi o único arquivo do dataset com as 3 classes
    necessárias e balanceamento adequado.

ESTRUTURA DE PASTAS:
    MoodWave/backend/
    ├── pln.py
    ├── data/
    │   ├── Train3Classes.csv          (baixar do Kaggle — no .gitignore)
    │   └── dataset_sentimentos.csv    (cache gerado — no .gitignore)
    └── modelo_nb.pkl                  (gerado ao treinar — no .gitignore)
=============================================================
"""

import re
import os
import joblib
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
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
    Pipeline de limpeza e normalização de texto em português.

    Etapas:
        1. Lowercase
        2. Remoção de URLs, menções (@) e hashtags (#)
        3. Remoção de pontuação e números
        4. Tokenização
        5. Remoção de stop words (português)
        6. Stemming com RSLP (algoritmo nativo para português)
    """

    def __init__(self):
        nltk.download("stopwords", quiet=True)
        nltk.download("rslp",      quiet=True)
        nltk.download("punkt",     quiet=True)
        nltk.download("punkt_tab", quiet=True)

        self.stop_words = set(stopwords.words("portuguese"))
        self.stemmer    = RSLPStemmer()

    def limpar(self, texto: str) -> str:
        texto = texto.lower()
        texto = re.sub(r"http\S+|www\S+",          "", texto)  # URLs
        texto = re.sub(r"@\w+",                     "", texto)  # menções
        texto = re.sub(r"#\w+",                     "", texto)  # hashtags
        texto = re.sub(r"[^a-záàâãéêíóôõúüçñ\s]", "", texto)  # pontuação/números
        texto = re.sub(r"\s+",                      " ", texto).strip()
        return texto

    def tokenizar(self, texto: str) -> list:
        tokens = word_tokenize(texto, language="portuguese")
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
    Carrega o dataset Train3Classes.csv e gera um cache padronizado.

    Fonte: https://www.kaggle.com/datasets/augustop/portuguese-tweets-for-sentiment-analysis

    Fluxo:
        1. Se cache (dataset_sentimentos.csv) existe → carrega direto
        2. Se Train3Classes.csv existe → processa e gera cache
        3. Se nenhum dos dois → usa fallback embutido no código
    """

    ARQUIVO_CSV   = os.path.join(BASE_DIR, "data", "Train3Classes.csv")
    ARQUIVO_CACHE = os.path.join(BASE_DIR, "data", "dataset_sentimentos.csv")
    PASTA_DATA    = os.path.join(BASE_DIR, "data")

    # Mapeamento de labels numéricos → classes em português
    # Confirmado inspecionando os tweets reais do dataset:
    #   0 → tweets com :( → Negativo
    #   1 → tweets com :) → Positivo
    #   2 → notícias/tweets neutros → Neutro
    MAPA_LABELS = {
        0: "Negativo", "0": "Negativo",
        1: "Positivo", "1": "Positivo",
        2: "Neutro",   "2": "Neutro",
    }

    def _processar_csv(self) -> pd.DataFrame:
        """Lê Train3Classes.csv, aplica mapeamento e salva cache."""
        print(f"[Dataset] Lendo '{self.ARQUIVO_CSV}'...")
        df = pd.read_csv(
            self.ARQUIVO_CSV,
            sep=";",
            encoding="utf-8",
            on_bad_lines="skip",
        )
        print(f"[Dataset] Colunas encontradas: {list(df.columns)}")

        df = df[["tweet_text", "sentiment"]].dropna()
        df["sentiment"] = df["sentiment"].map(self.MAPA_LABELS)
        df = df.dropna(subset=["sentiment"])
        df = df.rename(columns={"tweet_text": "text"})

        os.makedirs(self.PASTA_DATA, exist_ok=True)
        df.to_csv(self.ARQUIVO_CACHE, index=False, encoding="utf-8")
        print(f"[Dataset] Cache salvo em '{self.ARQUIVO_CACHE}'")
        return df

    def _fallback_embutido(self) -> pd.DataFrame:
        """
        Fallback mínimo para rodar sem o CSV do Kaggle.
        Útil para testes rápidos ou demonstração offline.
        """
        print("[Dataset] AVISO: Train3Classes.csv não encontrado!")
        print("[Dataset] Baixe em: https://www.kaggle.com/datasets/augustop/portuguese-tweets-for-sentiment-analysis")
        print("[Dataset] Usando dataset embutido reduzido como fallback...")

        dados = [
            ("Amei demais, superou todas as minhas expectativas!", "Positivo"),
            ("Entrega rápida e produto igual à descrição. Recomendo!", "Positivo"),
            ("Passei na faculdade! Estou muito feliz e aliviado.", "Positivo"),
            ("Que dia maravilhoso, tudo deu certo hoje!", "Positivo"),
            ("Estou eufórico, a melhor notícia do ano!", "Positivo"),
            ("O show foi fantástico, emocionante do começo ao fim.", "Positivo"),
            ("Produto veio com defeito e o suporte não ajudou.", "Negativo"),
            ("Que dia horrível, tô completamente esgotado.", "Negativo"),
            ("Péssima qualidade, quebrou na primeira semana.", "Negativo"),
            ("Odeio quando isso acontece, que raiva enorme.", "Negativo"),
            ("Fui enganado na compra, produto falsificado.", "Negativo"),
            ("Atendimento grosseiro, me senti desrespeitado.", "Negativo"),
            ("Fui ao mercado e comprei pão.", "Neutro"),
            ("O tempo está nublado hoje.", "Neutro"),
            ("A reunião durou cerca de duas horas.", "Neutro"),
            ("O pacote chegou na segunda-feira.", "Neutro"),
            ("Enviei o formulário por e-mail.", "Neutro"),
            ("A entrega foi feita pelo correio.", "Neutro"),
        ] * 40  # ~720 amostras balanceadas

        df = pd.DataFrame(dados, columns=["text", "sentiment"])
        os.makedirs(self.PASTA_DATA, exist_ok=True)
        df.to_csv(self.ARQUIVO_CACHE, index=False, encoding="utf-8")
        return df

    def carregar(self) -> tuple:
        """
        Ponto de entrada principal. Retorna (textos, labels) prontos para treino.
        """
        if os.path.exists(self.ARQUIVO_CACHE):
            print(f"[Dataset] Cache encontrado. Carregando...")
            df = pd.read_csv(self.ARQUIVO_CACHE, encoding="utf-8")
        elif os.path.exists(self.ARQUIVO_CSV):
            df = self._processar_csv()
        else:
            df = self._fallback_embutido()

        df      = df[["text", "sentiment"]].dropna()
        textos  = df["text"].astype(str).tolist()
        labels  = df["sentiment"].tolist()

        print(f"[Dataset] {len(textos)} amostras carregadas.")
        print(f"[Dataset] Distribuição:\n{pd.Series(labels).value_counts().to_string()}\n")
        return textos, labels


# ─────────────────────────────────────────────
# 3. CLASSIFICADOR DE SENTIMENTOS
# ─────────────────────────────────────────────

class ClassificadorSentimentos:
    """
    Classificador Naive Bayes com TF-IDF para análise de sentimentos.

    Pipeline interno:
        TfidfVectorizer (unigrams + bigrams) → MultinomialNB

    Resultado obtido com Train3Classes.csv (100k tweets):
        Accuracy : 0.7514
        F1 Macro : 0.7508

    Saída principal para a Camada II (Fuzzy do João):
        prob_positivo : float entre 0.0 e 1.0
    """

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
            labels       : lista com "Positivo", "Negativo" ou "Neutro"
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
                max_features=50_000,
                min_df=2,               # ignora termos que aparecem < 2x
                sublinear_tf=True,      # aplica log na frequência do termo
            )),
            ("nb", MultinomialNB(alpha=1.0)),   # alpha = suavização de Laplace
        ])

        print("[Treino] Treinando MultinomialNB...")
        self.pipeline.fit(X_train, y_train)

        y_pred   = self.pipeline.predict(X_test)
        metricas = {
            "accuracy":              round(accuracy_score(y_test, y_pred), 4),
            "f1_macro":              round(f1_score(y_test, y_pred, average="macro"), 4),
            "classification_report": classification_report(y_test, y_pred),
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
            texto : string bruta digitada pelo usuário

        Returns:
            {
                "classe"        : "Positivo" | "Negativo" | "Neutro",
                "prob_positivo" : float,   ← João usa este no calcular_humor()
                "prob_negativo" : float,
                "prob_neutro"   : float,
            }

        Exemplo de uso pelo João (Camada II):
            from pln import ClassificadorSentimentos

            clf = ClassificadorSentimentos()
            clf.carregar()

            resultado  = clf.analisar_texto("Estou muito feliz!")
            prob_fuzzy = resultado["prob_positivo"]  # ex: 0.87
            score      = calcular_humor(prob_fuzzy, energia_bpm)
        """
        if self.pipeline is None:
            raise RuntimeError("Modelo não carregado. Chame treinar() ou carregar().")

        texto_proc = self.preprocessador.processar(texto)
        classe     = self.pipeline.predict([texto_proc])[0]
        probs      = self.pipeline.predict_proba([texto_proc])[0]
        mapa_prob  = dict(zip(self.pipeline.classes_, probs))

        return {
            "classe":        classe,
            "prob_positivo": round(float(mapa_prob.get("Positivo", 0.0)), 4),
            "prob_negativo": round(float(mapa_prob.get("Negativo", 0.0)), 4),
            "prob_neutro":   round(float(mapa_prob.get("Neutro",   0.0)), 4),
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
    print("\n" + "="*55)
    print("TESTE COM FRASES DE EXEMPLO")
    print("="*55)

    frases_teste = [
        "Hoje foi um dia incrível, me sinto muito feliz!",
        "Que dia horrível, tô completamente esgotado.",
        "Fui ao mercado e comprei pão.",
        "Passei na faculdade!!! Estou eufórico demais!",
        "Odeio quando isso acontece, que raiva.",
        "O tempo está nublado hoje.",
    ]

    for frase in frases_teste:
        r = classificador.analisar_texto(frase)
        print(f"\nTexto : {frase}")
        print(f"Classe: {r['classe']}")
        print(f"  prob_positivo : {r['prob_positivo']:.4f}  ← entra no Fuzzy")
        print(f"  prob_negativo : {r['prob_negativo']:.4f}")
        print(f"  prob_neutro   : {r['prob_neutro']:.4f}")

    # 5. Interface com a Camada II
    print("\n" + "="*55)
    print("INTERFACE COM CAMADA II (Fuzzy — João)")
    print("="*55)
    texto_usuario = "Estou me sentindo muito animado hoje!"
    saida = classificador.analisar_texto(texto_usuario)
    print(f"Texto do usuário : '{texto_usuario}'")
    print(f"prob_sentimento  : {saida['prob_positivo']}  → variável de entrada do Fuzzy")
    print("João recebe esse valor em calcular_humor(prob_sentimento, energia_bpm)")