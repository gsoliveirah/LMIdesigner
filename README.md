# alloy-inverse-design

Design inverso de **Aços Avançados de Alta Resistência (AHSS)** combinando
**termodinâmica computacional (CALPHAD)**, **aprendizado de máquina
físico-informado (PINN)** e **otimização multiobjetivo (NSGA-III)**.

```
 LHS (composição × processo)
        │
        ▼
 CALPHAD (pycalphad / empírico) ──► equilíbrio na T de recozimento (f_γ, C_γ, fases indesejadas)
        │
        ▼
 Modelo de transformação ──► ferrita / bainita / martensita / austenita retida ──► propriedades
        │                                                                   (dados sintéticos)
        ▼
 Pré-processamento ──► limpeza, IsolationForest, C_eq, Pcm, Ac1/Ac3, Ms
        │
        ▼
 PINN (PyTorch) ──► perda = dados + fases + λ·[YS≤UTS, regra das misturas, lei de alongamento, monotonicidade]
        │             Optuna (TPE + MedianPruner) · GroupKFold por liga
        ▼
 NSGA-III (pymoo) ──► max UTS, max El, min custo, min CO₂  s.a. UTS/El mínimos, C_eq, filtro de risco
        │
        ▼
 Filtro termodinâmico ──► triagem rápida no laço + validação CALPHAD no conjunto de Pareto
        │
        ▼
 Dashboard Streamlit
```

## Estrutura

```text
├── data/
│   ├── raw/                  # Dados experimentais brutos
│   ├── synthetic/            # Dados gerados via CALPHAD
│   └── processed/            # Datasets limpos + estatísticas de normalização
├── models/                   # Checkpoints da PINN (*.pt)
├── src/
│   ├── config.py             # Constantes físicas, limites, preços, fatores de CO₂
│   ├── metallurgy.py         # Equações metalúrgicas (NumPy *e* PyTorch, diferenciáveis)
│   ├── cli.py                # CLI: generate | train | optimize
│   ├── data_pipeline/
│   │   ├── calphad_gen.py    # Backends termodinâmicos + LHS + gerador sintético
│   │   └── preprocessing.py  # Limpeza, IsolationForest, C_eq, normalização
│   ├── models/
│   │   ├── pinn.py           # Arquitetura PINN, perdas físicas, MC-dropout, checkpoint
│   │   └── train.py          # Treinador com Optuna + GroupKFold
│   ├── optimization/
│   │   ├── inverse_design.py # Problema NSGA-III, custo, CO₂, solução de compromisso (ASF)
│   │   └── thermo_filter.py  # Regras de estabilidade de fases e risco metalúrgico
│   └── app/
│       └── dashboard.py      # Interface Streamlit
└── tests/                    # pytest
```

> `src/metallurgy.py` e `src/cli.py` foram adicionados à estrutura proposta:
> o primeiro concentra as equações físicas usadas *tanto* na geração de dados
> quanto nas perdas da PINN (garantindo consistência); o segundo orquestra o
> pipeline pela linha de comando.

## Instalação

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Uso rápido

```bash
# 1. Dados sintéticos (backend empírico, instantâneo)
python -m src.cli generate --n-alloys 1000 --n-process 4

#    ...ou com equilíbrio CALPHAD real (lento: ~1 s/ponto)
python -m src.cli generate --tdb /caminho/mc_fe.tdb --n-alloys 200 --n-process 4

# 2. Treino (20 trials Optuna, GroupKFold 5 folds)
python -m src.cli train --n-trials 20 --epochs 300
#    combinando dados sintéticos e experimentais:
python -m src.cli train --data data/synthetic/synthetic_ahss.csv data/raw/experimental.csv

# 3. Design inverso: ≥1180 MPa, ≥10 % de alongamento
python -m src.cli optimize --min-uts 1180 --min-el 10 --generations 200 --fe-route EAF

# 4. Dashboard
streamlit run src/app/dashboard.py

# Testes
pytest -q
```

### Formato de dados experimentais (`data/raw/*.csv`)

| coluna | unidade | obrigatória |
|---|---|---|
| `C, Mn, Si, Cr, Ni, Mo, V, Nb, Ti, Al, Cu, B, N` | wt% (não reportado → 0) | sim |
| `T_anneal_C` | °C | sim |
| `cooling_rate_Cps` | °C/s | sim |
| `grain_size_um` | µm | sim |
| `YS_MPa, UTS_MPa, El_pct` | MPa, MPa, % | sim |
| `f_ferrite, f_bainite, f_martensite, f_retained_austenite` | fração | opcional (supervisiona a cabeça de fases se todas presentes) |
| `alloy_id` | — | opcional (senão o grupo é inferido pela composição) |

## Componentes

### Geração CALPHAD (`calphad_gen.py`)
* `latin_hypercube` — LHS com otimização de discrepância (`scipy.stats.qmc`), escala
  log10 para a taxa de resfriamento.
* `PyCalphadBackend` — equilíbrio de Gibbs com a base TDB fornecida; separa
  austenita de carbonitretos MX (ambos `FCC_A1`), extrai C na austenita e soma
  fases indesejadas (σ, Laves, µ, χ).
* `EmpiricalThermoBackend` — Andrews + regra da alavanca, vetorizado (prototipagem).
* `CalphadDataGenerator` — cada liga recebe várias rotas de processo
  (`alloy_id` = grupo), microestrutura via temperabilidade + Koistinen–Marburger +
  estabilização da austenita por Si/Al (TRIP), propriedades via regra das misturas,
  ruído experimental configurável.

### PINN (`pinn.py`)
* Normalização *dentro* do modelo (buffers) → autograd em unidades físicas.
* Tronco MLP residual + cabeça mecânica `[YS, UTS, El]` + cabeça de fases (softmax:
  Σf = 1 imposto rigidamente).
* Resíduos físicos aplicados nos dados **e em pontos de colocação** amostrados no
  espaço de projeto inteiro — regulariza onde o otimizador vai explorar:
  `ys_le_uts`, `rule_of_mixtures` (com balanço de massa de C), `elongation_law`,
  `monotonicity` (∂UTS/∂C ≥ 0, ∂UTS/∂Mn ≥ 0, ∂UTS/∂(taxa) ≥ 0, ∂YS/∂d ≤ 0 — Hall–Petch),
  `non_negative`.
* Incerteza epistêmica por MC-dropout (`predict(..., n_mc_samples=N)`), utilizável
  como restrição no design inverso (`OptimizationConfig.max_uts_cv`).

### Treinamento (`train.py`)
* `GroupKFold` por liga (sem vazamento composicional), estatísticas de normalização
  ajustadas só no fold de treino, early stopping, AdamW + cosine annealing.
* Optuna (TPE + MedianPruner por fold) sobre largura, profundidade, dropout, lr,
  weight decay e λ_física. Modelo final treinado em todos os dados pelo nº médio de
  épocas ótimas da CV.

### Design inverso (`inverse_design.py`)
* 4 objetivos: −UTS, −El, custo (USD/kg, com rendimento de adição), CO₂ (kg/kg,
  rota BF-BOF ou EAF). Direções de referência Das-Dennis.
* Restrições: UTS/El mínimos, C_eq IIW, zero regras críticas, `risk_score` máximo,
  incerteza opcional.
* Elementos podem ser fixados (`OptimizationConfig.fixed_values={"Ni": 0, "Mo": 0}`).
* `select_compromise` — Achievement Scalarizing Function com pesos do usuário.

### Filtro termodinâmico (`thermo_filter.py`)
Regras vetorizadas com severidade (CRITICAL/WARNING): soldabilidade (C_eq IIW, Pcm),
recozimento abaixo de Ac1 / acima de Ac3+50 °C, janela intercrítica estreita,
bandeamento de Mn, galvanizabilidade (Si), lingotamento (Al, Nb+Ti+V), fragilização
por B, B desprotegido (Ti/N < 3,42), hot shortness por Cu, e fases intermetálicas
previstas por CALPHAD. `validate_with_calphad` recalcula o equilíbrio apenas no
conjunto de Pareto.

## Limitações e próximos passos

* **Os dados sintéticos são tão bons quanto os modelos que os geram.** O backend
  empírico e as leis constitutivas (Pickering, Krauss, Andrews, KM) são
  aproximações semiquantitativas; use-os para validar o pipeline e como *prior*,
  e calibre com dados experimentais (transfer learning/fine-tuning em `data/raw`).
* Bases TDB de aço completas (TCFE, MatCalc `mc_fe`) não são distribuídas aqui. A
  base de teste do pycalphad (`mc_fecocrnbti.tdb`) é truncada e só serve para testes
  unitários (transição α/γ do Fe puro).
* Preços (`ELEMENT_PRICE_USD_PER_KG`) e fatores de CO₂ (`ELEMENT_CO2_KG_PER_KG`, base
  Nuss & Eckelman 2014) são **indicativos** — atualize com cotações LME/ferroligas e
  com o LCA da sua planta.
* Extensões naturais: cinética de partição (Q&P, DICTRA), tempo de encharque,
  revenimento, textura/anisotropia, otimização bayesiana com aquisição ativa de
  experimentos.
