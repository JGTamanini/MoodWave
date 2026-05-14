"""
=============================================================
MoodWave — Backend FastAPI
=============================================================
Responsável : Eric Gabriel Caetano
Disciplina  : Inteligência Artificial
Professor   : Claudinei Dias (Ney)

Descrição:
    API REST que integra as três camadas do sistema MoodWave:
        Camada I  — PLN (Felipe): classifica sentimento do texto
        Camada II — Fuzzy (João): estima humor com sentimento + energia
        Camada III— AG (Gabriel): gera playlist compatível com o humor

Endpoints:
    GET  /health    → verifica se a API está no ar
    POST /analisar  → recebe texto + energia e retorna humor + playlist

Para rodar:
    cd backend
    uvicorn main:app --reload
=============================================================
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

from pln import ClassificadorSentimentos
from fuzzy import SistemaFuzzy
from camada3_ga import gerar_playlist, CatalogoMusical


# ─────────────────────────────────────────────
# Modelos de dados (Pydantic)
# ─────────────────────────────────────────────

class AnalisarRequest(BaseModel):
    texto: str = Field(..., min_length=1, description="Texto descrevendo o humor do usuário")
    energia: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Nível de energia do usuário (0 = esgotado, 1 = muito energético)",
    )


class Probabilidades(BaseModel):
    positivo: float
    neutro: float
    negativo: float


class Faixa(BaseModel):
    titulo: str
    artista: str
    energia: float
    valencia: float
    bpm: float


class AnalisarResponse(BaseModel):
    humor: str
    score_humor: float
    probabilidades: Probabilidades
    playlist: list[Faixa]
    estatisticas: dict


# ─────────────────────────────────────────────
# Estado global (modelos e catálogo)
# ─────────────────────────────────────────────

_estado = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega modelos pesados uma única vez na inicialização."""
    print("[MoodWave] Inicializando modelos...")

    clf = ClassificadorSentimentos()
    clf.carregar()
    _estado["clf"] = clf

    _estado["fis"] = SistemaFuzzy()

    catalogo = CatalogoMusical()
    _estado["catalogo"] = catalogo

    print("[MoodWave] Todos os modelos carregados. API pronta.")
    yield
    _estado.clear()


# ─────────────────────────────────────────────
# Aplicação FastAPI
# ─────────────────────────────────────────────

app = FastAPI(
    title="MoodWave API",
    description="Sistema de recomendação de músicas por humor (PLN + Fuzzy + AG)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/health", tags=["Status"])
def health():
    """Verifica se a API está no ar."""
    return {"status": "ok", "modelos_carregados": bool(_estado)}


@app.post("/analisar", response_model=AnalisarResponse, tags=["MoodWave"])
def analisar(req: AnalisarRequest):
    """
    Pipeline completo MoodWave:
        1. PLN — classifica sentimento do texto (Camada I)
        2. Fuzzy — estima humor com sentimento + energia (Camada II)
        3. AG — gera playlist compatível com o humor (Camada III)
    """
    if not _estado:
        raise HTTPException(status_code=503, detail="Modelos ainda não carregados.")

    clf: ClassificadorSentimentos = _estado["clf"]
    fis: SistemaFuzzy = _estado["fis"]
    catalogo: CatalogoMusical = _estado["catalogo"]

    # Camada I — PLN
    nlp = clf.analisar_texto(req.texto)

    # Camada II — Fuzzy (sentimento do texto + energia do slider)
    fuzzy = fis.analisar(nlp["prob_positivo"], req.energia)

    # Camada III — Algoritmo Genético
    resultado_ga = gerar_playlist(fuzzy["score"], catalogo)

    playlist = [
        Faixa(
            titulo=f["musica"],
            artista=f["artista"],
            energia=f["energia"],
            valencia=f["valencia"],
            bpm=f["bpm"],
        )
        for f in resultado_ga["faixas"]
    ]

    return AnalisarResponse(
        humor=fuzzy["label"],
        score_humor=round(fuzzy["score"], 4),
        probabilidades=Probabilidades(
            positivo=nlp["prob_positivo"],
            neutro=nlp["prob_neutro"],
            negativo=nlp["prob_negativo"],
        ),
        playlist=playlist,
        estatisticas=resultado_ga["estatisticas"],
    )
