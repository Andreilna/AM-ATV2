# AT2 — Rede Neural MLP e API de Inferência

Projeto de classificação da sobrevivência de passageiros do Titanic usando um
pipeline Scikit-Learn com `MLPClassifier`, disponibilizado por uma API REST
FastAPI.

## Dataset

A base tem 891 registros e é derivada dos dados públicos da competição Titanic
do Kaggle: [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/data).
O arquivo `data/titanic.csv` é uma cópia pública da base de treinamento,
disponibilizada em [Data Science Dojo](https://github.com/datasciencedojo/datasets/blob/master/titanic.csv).

O alvo `Survived` indica se o passageiro sobreviveu (1) ou não (0). O modelo
utiliza `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare` e `Embarked`. A base
contém valores ausentes em idade e porto de embarque; eles são imputados dentro
do pipeline, sem vazamento entre treino e teste. Por ser um conjunto histórico,
o modelo é demonstrativo e não deve ser usado para decisões reais sobre pessoas.

## Reproduzir

Requisitos: Python 3.15 e [UV](https://docs.astral.sh/uv/).

```powershell
uv sync
uv run jupyter notebook notebooks/analise_treinamento.ipynb
```

Execute as células do notebook em ordem para explorar os dados, treinar e avaliar
o pipeline. A última etapa salva o artefato em
`models/pipeline_rede_neural.joblib`, necessário para iniciar a API.

## API

```powershell
uv run fastapi dev src/app.py
```

- Documentação interativa: <http://127.0.0.1:8000/docs>
- Diagnóstico: `GET /diagnostico/modelo`
- Inferência: `POST /predict`
- Exemplos válidos e inválido: [requisicoes.http](requisicoes.http)

A API carrega o pipeline uma única vez no `lifespan` de inicialização. O corpo
de `POST /predict` deve fornecer os sete atributos em snake_case. A resposta
contém `categoria_prevista` (0 ou 1) e `score_confianca` (probabilidade máxima
entre 0 e 1). O Pydantic rejeita tipos, categorias e limites inválidos com HTTP
422.

## Estrutura

```text
data/titanic.csv
models/pipeline_rede_neural.joblib
notebooks/analise_treinamento.ipynb
src/app.py
src/schemas.py
requisicoes.http
```