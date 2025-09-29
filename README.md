# Projeto Final - Machine Learning

Este repositório contém o código do meu projeto final de graduação em Ciência da Computação.  
O objetivo é desenvolver e avaliar modelos de **aprendizado de máquina aplicados a séries temporais econômicas**, com foco em backtesting e previsão.

## Estrutura do Repositório

```
projfinal-ml/
│── app/                # Código principal do projeto
│── .gitignore          # Arquivos ignorados pelo Git
```

## Funcionalidades

- Processamento de séries temporais econômicas.
- Implementação de modelos de machine learning.
- Backtesting e avaliação de performance.
- API para interação com os modelos.

## Tecnologias Utilizadas

- **Python 3.11+**
- **pandas**, **numpy**, **scikit-learn**
- **FastAPI**
- **Uvicorn**
- **plotly** (visualização)

## Como Executar

Clone o repositório:

```bash
git clone https://github.com/olintoeduardo/projfinal-ml.git
cd projfinal-ml
```

Crie um ambiente virtual e instale as dependências:

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

Execute a aplicação:

```bash
uvicorn app.main:app --reload
```

Acesse em: [http://localhost:8000](http://localhost:8000)

## Próximos Passos

- [ ] Adicionar testes unitários
- [ ] Documentar os endpoints da API
- [ ] Criar notebooks de experimentos
- [ ] Incluir exemplos de uso

## Autor

Projeto desenvolvido por **Eduardo Olinto**.