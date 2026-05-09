"""
=============================================================
N2 - Sistema Inteligente | Camada I: PLN com Naive Bayes
=============================================================
Responsável : Felipe
Disciplina  : Inteligência Artificial
Professor   : Claudinei Dias (Ney)
 
Descrição:
    Classificador de sentimentos (Positivo / Negativo / Neutro)
    treinado com o dataset TweetSentBR.
    Saída normalizada para alimentar o Sistema Fuzzy (Camada II).
 
Dependências:
    pip install nltk scikit-learn pandas numpy joblib
 
Download dos dados NLTK (rodar uma vez):
    python -c "import nltk; nltk.download('stopwords'); nltk.download('rslp'); nltk.download('punkt')"
=============================================================
"""

import re
import os 
import zipfile
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
    f1_score
)

from sklearn.pipeline import Pipeline


# -------------------------------
# 1. Pré-processamento de texto
# -------------------------------

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
        nltk.download('stopwords', quiet=True)
        nltk.download('rslp', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_table', quiet=True)

        self.stop_words = set(stopwords.words('portuguese'))
        self.stemmer = RSLPStemmer()

def limpar(self, texto: str) -> str:
        texto = texto.lower()
        texto = re.sub(r"https\S+|www\.\S+", "", texto)  # Remove URLs
        texto = re.sub(r"@\w+", "", texto)  # Remove menções
        texto = re.sub(r"#\w+", "", texto)  # Remove hashtags
        texto = re.sub(r"[^a-záàâãéêíóôõúüçñ\s]", "", texto)
        texto = re.sub(r"\s+", " ", texto).strip()

        return texto

def tokenizar(self, texto: str) -> list:
       tokens = word_tokenize(texto, language='portuguese')
       return [token for token in tokens if token not in self.stop_words and len(token) > 2]

def stemmizar (self, tokens: list) -> list:
        return [self.stemmer.stem(token) for token in tokens]

def processar(self, texto: str) -> str:
        texto = self.limpar(texto)
        tokens = self.tokenizar(texto)
        tokens_stemmizados = self.stemmizar(tokens)
        return " ".join(tokens_stemmizados)

# -------------------------------
# 2. Carregar DataSet (em teoria download automático)
# -------------------------------

"""
Baixa e carrega o TweetSentBR do Kaggle automaticamente.

Na primeira execução: faz o download e extrai o CSV em data/.
Nas execuções seguintes: usa o arquivo já existente (sem re-download).

Cada membro do grupo precisa apenas do seu próprio kaggle.json
configurado — o código cuida do resto.
"""

KAGGLE_DATASET = "leandrodoze/tweets-from-brazilian-companies"
PASTA_DATA     = "data"
NOME_ZIP       = "tweets-from-brazilian-companies.zip"

MAPA_LABELS = {
    "positive": "Positivo", "positivo": "Positivo",
    "pos": "Positivo", "1": "Positivo", 1: "Positivo",
    "negative": "Negativo", "negativo": "Negativo",
    "neg": "Negativo", "-1": "Negativo", -1: "Negativo",
    "neutral": "Neutro", "neutro": "Neutro",
    "neu": "Neutro", "0": "Neutro", 0: "Neutro",
}
 
def __init__(self,
                coluna_texto: str = "tweet_text",
                coluna_label: str = "sentiment"):
    self.coluna_texto = coluna_texto
    self.coluna_label = coluna_label

def _verificar_kaggle_json(self):
    """Checa se o kaggle.json existe. Lança erro claro se não encontrar."""
    caminhos = [
        os.path.expanduser("~/.kaggle/kaggle.json"),
        os.path.join(os.environ.get("USERPROFILE", ""), ".kaggle", "kaggle.json"),
    ]
    for c in caminhos:
        if os.path.exists(c):
            return

    raise FileNotFoundError(
        "\n"
        "╔══════════════════════════════════════════════════════════╗\n"
        "║              kaggle.json não encontrado!                 ║\n"
        "╠══════════════════════════════════════════════════════════╣\n"
        "║  Siga os passos abaixo:                                  ║\n"
        "║  1. Acesse https://www.kaggle.com e faça login           ║\n"
        "║  2. Vá em Account > Settings > API > Create New Token    ║\n"
        "║  3. Salve o kaggle.json em:                              ║\n"
        "║     Linux/Mac : ~/.kaggle/kaggle.json                    ║\n"
        "║     Windows   : C:\\Users\\SEU_USUARIO\\.kaggle\\kaggle.json ║\n"
        "║  4. Execute o script novamente                           ║\n"
        "╚══════════════════════════════════════════════════════════╝\n"
    )

def _encontrar_csv(self):
    """Procura qualquer .csv dentro da pasta data/."""
    if not os.path.exists(self.PASTA_DATA):
        return None
    for arquivo in os.listdir(self.PASTA_DATA):
        if arquivo.endswith(".csv"):
            return os.path.join(self.PASTA_DATA, arquivo)
    return None

def _baixar_dataset(self):
    """Faz o download via Kaggle API e extrai o ZIP."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise ImportError(
            "Biblioteca 'kaggle' não instalada.\n"
            "Rode: pip install kaggle"
        )

    self._verificar_kaggle_json()
    os.makedirs(self.PASTA_DATA, exist_ok=True)

    print(f"[Dataset] Baixando '{self.KAGGLE_DATASET}' do Kaggle...")
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(
        self.KAGGLE_DATASET,
        path=self.PASTA_DATA,
        quiet=False,
    )

    caminho_zip = os.path.join(self.PASTA_DATA, self.NOME_ZIP)
    if os.path.exists(caminho_zip):
        print(f"[Dataset] Extraindo {self.NOME_ZIP}...")
        with zipfile.ZipFile(caminho_zip, "r") as zf:
            zf.extractall(self.PASTA_DATA)
        os.remove(caminho_zip)

    print(f"[Dataset] Concluído. Arquivos em '{self.PASTA_DATA}/'")

def carregar(self) -> tuple:
    """
    Ponto de entrada principal.

    - CSV já existe em data/  →  carrega direto (sem download).
    - CSV não existe          →  baixa do Kaggle e depois carrega.

    Returns:
        (textos, labels) — listas prontas para treino.
    """
    caminho_csv = self._encontrar_csv()

    if caminho_csv is None:
        print("[Dataset] CSV não encontrado localmente. Iniciando download...")
        self._baixar_dataset()
        caminho_csv = self._encontrar_csv()

    if caminho_csv is None:
        raise RuntimeError(
            "Não foi possível encontrar o CSV após o download.\n"
            "Verifique a pasta data/ manualmente."
        )

    print(f"[Dataset] Carregando '{caminho_csv}'...")
    df = pd.read_csv(caminho_csv, encoding="utf-8")

    # Detecção automática de colunas se os nomes padrão não existirem
    if self.coluna_texto not in df.columns:
        candidatos = [c for c in df.columns
                        if "text" in c.lower() or "tweet" in c.lower()]
        if not candidatos:
            raise ValueError(
                f"Coluna de texto não encontrada.\n"
                f"Colunas disponíveis: {list(df.columns)}\n"
                f"Ajuste 'coluna_texto' no CarregadorDataset."
            )
        self.coluna_texto = candidatos[0]
        print(f"[Dataset] Coluna de texto detectada: '{self.coluna_texto}'")

    if self.coluna_label not in df.columns:
        candidatos = [c for c in df.columns
                        if any(k in c.lower() for k in ["sentiment", "label", "class"])]
        if not candidatos:
            raise ValueError(
                f"Coluna de label não encontrada.\n"
                f"Colunas disponíveis: {list(df.columns)}\n"
                f"Ajuste 'coluna_label' no CarregadorDataset."
            )
        self.coluna_label = candidatos[0]
        print(f"[Dataset] Coluna de label detectada: '{self.coluna_label}'")

    df = df[[self.coluna_texto, self.coluna_label]].dropna()
    df[self.coluna_label] = df[self.coluna_label].map(self.MAPA_LABELS)
    df = df.dropna(subset=[self.coluna_label])

    textos = df[self.coluna_texto].astype(str).tolist()
    labels = df[self.coluna_label].tolist()

    print(f"[Dataset] {len(textos)} amostras carregadas.")
    print(f"[Dataset] Distribuição:\n{pd.Series(labels).value_counts().to_string()}\n")

    return textos, labels

# -------------------------------
# 3. Classificador de Sentimentos
# -------------------------------

class ClassificadorSentimentos:
    """
    Classificador Naive Bayes com TF-IDF para análise de sentimentos.
 
    Pipeline interno:
        TfidfVectorizer (unigrams + bigrams) → MultinomialNB
 
    Saída principal para a Camada II (Fuzzy do João):
        prob_positivo : float entre 0.0 e 1.0
    """
 
    def __init__(self, caminho_modelo: str = "modelo_nb.pkl"):
        self.caminho_modelo = caminho_modelo
        self.preprocessador = PreProcessador()
        self.pipeline       = None
 
    # ── Treinamento ──────────────────────────
 
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
            stratify=labels,        # garante proporção igual de classes no split
        )
 
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),     # unigrams + bigrams
                max_features=50_000,
                min_df=2,               # ignora termos que aparecem < 2x
                sublinear_tf=True,      # log na frequência do termo
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
 
    # ── Inferência ───────────────────────────
 
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
            from camada1_nlp import ClassificadorSentimentos
 
            clf = ClassificadorSentimentos()
            clf.carregar()
 
            resultado   = clf.analisar_texto("Estou muito feliz!")
            prob_fuzzy  = resultado["prob_positivo"]  # ex: 0.87
            score_humor = calcular_humor(prob_fuzzy, energia_bpm)
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
 
    CAMINHO_MODELO = "modelo_nb.pkl"
 
    # 1. Carrega dataset (baixa automaticamente se necessário)
    carregador     = CarregadorDataset()
    textos, labels = carregador.carregar()
 
    # 2. Treina e avalia
    classificador = ClassificadorSentimentos(caminho_modelo=CAMINHO_MODELO)
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
 
    # 5. Demonstração da interface com a Camada II
    print("\n" + "="*55)
    print("INTERFACE COM CAMADA II (Fuzzy — João)")
    print("="*55)
    texto_usuario = "Estou me sentindo muito animado hoje!"
    saida = classificador.analisar_texto(texto_usuario)
    print(f"Texto do usuário : '{texto_usuario}'")
    print(f"prob_sentimento  : {saida['prob_positivo']}  → variável de entrada do Fuzzy")
    print("João recebe esse valor em calcular_humor(prob_sentimento, energia_bpm)")