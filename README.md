# 🚀 Movie Recommendation API — API REST com FastAPI

API REST para recomendação de filmes construída com **FastAPI**, servindo o sistema híbrido de Machine Learning desenvolvido no [Dia 4](https://github.com/Bernardinense/movie-recommendation-system) do desafio.

> **Parte do desafio [#7DaysOfCode](https://7daysofcode.io/) de Data Science — Dia 5/7**

---

## 📋 Sobre o Projeto

Este projeto transforma o modelo de recomendação do Dia 4 em uma **API REST profissional**, demonstrando como servir modelos de Machine Learning em produção. A API recebe um ID de usuário e retorna recomendações personalizadas de filmes.

### Endpoints

| Método | Rota | Descrição |
|--------|------|-----------|
| `GET` | `/` | Endpoint raiz — informações da API |
| `GET` | `/health` | Health check — verifica se o modelo está carregado |
| `POST` | `/recomendar` | **Principal** — gera recomendações para um usuário |
| `GET` | `/stats` | Estatísticas do dataset |

### Lógica de recomendação

O endpoint `/recomendar` aceita 3 parâmetros:
- **user_id** — ID do usuário (1–943)
- **n** — Número de recomendações (1–20, padrão: 5)
- **method** — Método: `auto`, `collaborative`, `popularity`, `svd`

No modo `auto`, a API decide automaticamente:
- Usuário com ≥5 avaliações → Filtragem Colaborativa
- Usuário com <5 avaliações → Popularidade (cold start)

### Exemplo de resposta

```json
{
  "user_id": 196,
  "method_used": "collaborative",
  "recommendations": [
    {"movie_id": 144, "title": "Die Hard (1988)", "score": 2.69},
    {"movie_id": 780, "title": "Dumb & Dumber (1994)", "score": 2.63},
    {"movie_id": 89, "title": "Blade Runner (1982)", "score": 2.51}
  ],
  "total": 3
}
```

---

## 🛠️ Tecnologias Utilizadas

- **FastAPI** — Framework web assíncrono
- **Uvicorn** — Servidor ASGI
- **Pydantic** — Validação de dados e schemas
- **Pandas** — Manipulação de dados
- **Scikit-learn** — Modelos de ML (similaridade de cosseno, TruncatedSVD)
- **joblib** — Serialização/deserialização do modelo

---

## 📁 Estrutura do Projeto

```
movie-recommendation-api/
├── app/
│   ├── __init__.py          # Pacote Python
│   ├── main.py              # FastAPI — endpoints e configuração
│   ├── models.py            # Schemas Pydantic (validação)
│   ├── ml_model.py          # Serviço de recomendação
│   └── recommenders.py      # Classes dos modelos ML
├── setup_data.py            # Download automático do dataset
├── requirements.txt         # Dependências
├── LICENSE                  # Arquivo de Licença
├── README.md                # Este arquivo
└── .gitignore               # Arquivos ignorados
```

> O dataset e o modelo treinado são configurados localmente (não versionados no Git).

---

## 🚀 Como Executar

### 1. Clone o repositório
```bash
git clone https://github.com/Bernardinense/movie-recommendation-api.git
cd movie-recommendation-api
```

### 2. Crie e ative o ambiente virtual
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
```

### 3. Instale as dependências
```bash
pip install -r requirements.txt
```

### 4. Baixe o dataset
```bash
python setup_data.py
```

### 5. Gere o modelo treinado

Esta API depende do modelo híbrido treinado no [Dia 4 (movie-recommendation-system)](https://github.com/Bernardinense/movie-recommendation-system):

1. Clone e execute o notebook do [Dia 4](https://github.com/Bernardinense/movie-recommendation-system) para treinar os modelos
2. O notebook gera o arquivo `hybrid_system.joblib` na pasta `models/`
3. Crie a pasta `modelo/` na raiz deste projeto e copie o arquivo:

```
movie-recommendation-api/
├── modelo/
│   └── hybrid_system.joblib   ← copiar aqui
└── ...
```

> O arquivo `recommenders.py` recria as mesmas classes do Dia 4, necessárias para o pickle desserializar o modelo corretamente.

### 6. Inicie a API
```bash
uvicorn app.main:app --reload
```

### 7. Acesse a documentação
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 🔗 Parte do Desafio #7DaysOfCode

Este projeto é o **Dia 5** de um desafio de 7 dias cobrindo o pipeline completo de Data Science:

| Dia | Projeto | Tema |
|-----|---------|------|
| 1 | [ceaps-data-wrangling](https://github.com/Bernardinense/ceaps-data-wrangling) | Limpeza e Tratamento de Dados |
| 2 | [ceaps-storytelling](https://github.com/Bernardinense/ceaps-storytelling) | Visualização e Storytelling |
| 3 | [ceaps-forecasting](https://github.com/Bernardinense/ceaps-forecasting) | Previsão com Prophet e Sklearn |
| 4 | [movie-recommendation-system](https://github.com/Bernardinense/movie-recommendation-system) | Sistema de Recomendação |
| **5** | **movie-recommendation-api** | **API REST com FastAPI** |
| 6 | ab-testing-hypothesis | Teste A/B e Validação de Hipóteses |

📌 Veja a jornada completa: [7DaysOfCode-DataScience](https://github.com/Bernardinense/7DaysOfCode-DataScience)

---

## 👤 Autor

**Bruno Corrêa** —  Engenheiro | Especialista em Ciência de Dados

[![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/bfpc7/)
[![GitHub](https://img.shields.io/badge/GitHub-black?style=flat&logo=github)](https://github.com/Bernardinense)

---

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.