# 🎵 MoodWave — Sistema de Recomendação de Músicas por Humor

> Descreva como você está se sentindo. Receba uma playlist feita para o seu momento.

---

## 📌 Sobre o Projeto

O **MoodTune** é um sistema inteligente que combina três camadas de IA para recomendar músicas com base no humor do usuário. O usuário digita uma frase livre descrevendo como está se sentindo — o sistema analisa, classifica e gera uma playlist personalizada.

---

## 🧠 Arquitetura

```
Texto do usuário
      ↓
┌─────────────────────────┐
│  Camada I — PLN         │  Processa o texto e classifica o sentimento
│  Naive Bayes            │  → prob_sentimento (0 a 1)
└────────────┬────────────┘
             ↓
┌─────────────────────────┐
│  Camada II — Fuzzy      │  Combina sentimento + energia e estima o humor
│  Sugeno                 │  → score_humor (0 a 1)
└────────────┬────────────┘
             ↓
┌─────────────────────────┐
│  Camada III — AG        │  Seleciona músicas compatíveis com o humor
│  Algoritmo Genético     │  → playlist ordenada
└────────────┬────────────┘
             ↓
┌─────────────────────────┐
│  Backend — FastAPI      │  Expõe as camadas via API REST
│  + Frontend React       │  Interface para o usuário
└─────────────────────────┘
```

---

## 👥 Equipe

| Membro | Responsabilidade |
|--------|-----------------|
| Felipe da Silva Chawischi | Camada I — PLN e Naive Bayes |
| João Guilherme T. Dalmarco | Camada II — Sistema Fuzzy Sugeno |
| Gabriel Felipe Alves Bandoch | Camada III — Algoritmo Genético |
| Eric Gabriel Caetano | Backend FastAPI + Integração + Relatório + Frontend |

---

## ⚙️ Como Rodar Localmente

### Pré-requisitos
- Python 3.10+
- Node.js 18+

### Backend

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/moodwave.git
cd moodwave/backend

# Instale as dependências
pip install -r requirements.txt

# Suba o servidor
uvicorn main:app --reload
```

A API estará disponível em `http://localhost:8000`

### Frontend

```bash
cd moodtune/frontend

# Instale as dependências
npm install

# Suba o servidor de desenvolvimento
npm run dev
```

A interface estará disponível em `http://localhost:5173`

---

## 📡 Endpoints da API

| Método | Rota | Descrição |
|--------|------|-----------|
| `POST` | `/analisar` | Recebe texto e retorna humor + playlist |
| `GET` | `/health` | Verifica se a API está no ar |

### Exemplo de requisição

```json
POST /analisar
{
  "texto": "Estou me sentindo ansioso mas com energia hoje"
}
```

### Exemplo de resposta

```json
{
  "humor": "Animado",
  "score_humor": 0.72,
  "probabilidades": {
    "positivo": 0.65,
    "neutro": 0.25,
    "negativo": 0.10
  },
  "playlist": [
    { "titulo": "Levitating", "artista": "Dua Lipa", "energia": 0.80, "bpm": 103 },
    { "titulo": "Blinding Lights", "artista": "The Weeknd", "energia": 0.73, "bpm": 171 }
  ]
}
```

---

## 📦 Dependências Principais

### Backend
```
fastapi
uvicorn
pydantic
scikit-learn
nltk
scikit-fuzzy
deap
pandas
numpy
joblib
```

### Frontend
```
react
vite
tailwindcss
axios
chart.js
```

---

## 📄 Licença

Projeto acadêmico — Inteligência Artificial — Católica de Santa Catarina
