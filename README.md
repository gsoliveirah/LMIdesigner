# alloy-inverse-design

Design inverso de **ligas de alumínio secundárias** (fundição Al-Si-Mg/Cu, oriundas
de reciclagem) combinando **termodinâmica computacional (CALPHAD)**, **aprendizado de
máquina físico-informado (PINN)** e **otimização multiobjetivo (NSGA-III)**.

O problema central do refino de alumínio reciclado é uma assimetria: elementos em
*falta* corrigem-se por adição (Si, Mg, ligas-mãe AlMn/AlSr/AlTi), mas as impurezas
acumuladas na sucata (**Fe, Zn, Ni, Cr** e, para ligas como a A356, **Cu**) não saem
do banho; só podem ser diluídas com alumínio primário, caro e intensivo em carbono
(~16 kg CO₂/kg contra ~0,5 kg CO₂/kg da refusão). O sistema busca composições e
rotas de tratamento térmico que **toleram mais impurezas** (por exemplo, neutralizando
o Fe com Mn) e assim **maximizam o conteúdo reciclado**, sem queimar a peça na
solubilização nem formar sludge no forno.

```
 LHS (composição × processo; T_solubilização relativa ao eutético)
        │
        ▼
 CALPHAD (pycalphad / empírico) ──► equilíbrio na T de solubilização (Cu/Mg dissolvidos, líquido)
        │                            + na T do forno de espera (sludge)
        ▼
 Modelo de solidificação/T6 ──► SDAS, partição do Fe β/α, Al2Cu residual, envelhecimento (P)
        │                        ──► YS, UTS, El  (dados sintéticos)
        ▼
 Pré-processamento ──► limpeza, gate MAD + IsolationForest, Si_eq ("C_eq"), SF, Mn/Fe, Fe_crit…
        │
        ▼
 PINN (PyTorch) ──► perda = dados + fases + λ·[superposição de resistência, lei de ductilidade,
        │             encruamento, balanço de massa Fe/Cu/Si, monotonicidade, YS ≤ UTS]
        │             Optuna (TPE + MedianPruner) · GroupKFold por liga
        ▼
 NSGA-III (pymoo) ──► max UTS, max El, min custo, min CO₂
        │              s.a. UTS/El/YS mínimos, % reciclado mínimo, carga atingível, filtro de risco
        │              (custo/CO₂ vêm do modelo de carga: sucata + Al primário + adições)
        ▼
 Filtro termodinâmico ──► triagem rápida no laço + validação CALPHAD no conjunto de Pareto
        │
        ▼
 Carga ótima multi-sucata (LP/HiGHS) ──► Dashboard Streamlit
```

## Estrutura

```text
├── data/
│   ├── raw/                  # Dados experimentais brutos
│   ├── synthetic/            # Dados gerados via CALPHAD
│   └── processed/            # Datasets limpos + estatísticas de normalização
├── models/                   # Checkpoints da PINN (*.pt)
├── src/
│   ├── config.py             # Constantes físicas, limites, preços LME, CO₂, classes de sucata
│   ├── metallurgy.py         # Equações metalúrgicas (NumPy *e* PyTorch, diferenciáveis)
│   ├── cli.py                # CLI: generate | train | optimize | charge
│   ├── data_pipeline/
│   │   ├── calphad_gen.py    # Backends termodinâmicos + LHS + gerador sintético
│   │   └── preprocessing.py  # Limpeza, outliers (MAD + IsolationForest), C_eq, normalização
│   ├── models/
│   │   ├── pinn.py           # Arquitetura PINN, perdas físicas, MC-dropout, checkpoint
│   │   └── train.py          # Treinador com Optuna + GroupKFold
│   ├── optimization/
│   │   ├── inverse_design.py # Problema NSGA-III, solução de compromisso (ASF)
│   │   ├── charge.py         # Modelo de carga do forno (forma fechada + LP multi-sucata)
│   │   └── thermo_filter.py  # Regras de estabilidade de fases e risco metalúrgico
│   └── app/
│       └── dashboard.py      # Interface Streamlit
└── tests/                    # pytest (56 testes)
```

> Três módulos foram adicionados à estrutura proposta:
> * `src/metallurgy.py` concentra as equações físicas usadas *tanto* na geração de dados
>   quanto nas perdas da PINN, o que garante consistência entre as duas;
> * `src/optimization/charge.py` traduz composição-alvo → carga do forno → custo/CO₂/%
>   reciclado, que é o elo entre o design da liga e a reciclagem;
> * `src/cli.py` orquestra o pipeline pela linha de comando.

## Instalação

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Uso rápido

```bash
# 1. Dados sintéticos (backend empírico, ~4 s para 2000 pontos)
python -m src.cli generate --n-alloys 1000 --n-process 4

#    ...ou com equilíbrio CALPHAD real (lento: 2 equilíbrios/ponto)
python -m src.cli generate --tdb /caminho/cost507.tdb --n-alloys 200 --n-process 4

# 2. Treino (20 trials Optuna, GroupKFold 5 folds)
python -m src.cli train --n-trials 20 --epochs 300
#    combinando dados sintéticos e experimentais:
python -m src.cli train --data data/synthetic/synthetic_al.csv data/raw/experimental.csv

# 3. Design inverso: ≥ 280 MPa, ≥ 5 % de alongamento, mistura de sucatas,
#    ≥ 60 % de conteúdo reciclado, molde permanente (2 °C/s)
python -m src.cli optimize --min-uts 280 --min-el 5 \
    --scrap tense_cast=0.6 wheels_a356=0.4 --min-recycled 0.6 \
    --fix cooling_rate_Cps=2 --generations 200

# 4. Carga ótima multi-sucata para uma composição (minimizando CO₂, com faixas de especificação)
python -m src.cli charge --target Si=7 Fe=0.3 Cu=0.5 Mn=0.15 Mg=0.35 Zn=0.4 Ti=0.1 Sr=0.015 \
    --tolerance Fe=0.05 Cu=0.1 --objective co2

# 5. Dashboard
streamlit run src/app/dashboard.py

# Testes (o teste marcado como slow usa pycalphad + base Al-Cu de teste do pycalphad)
pytest -q
pytest -q -m "not slow"
```

### Formato de dados experimentais (`data/raw/*.csv`)

| coluna | unidade | obrigatória |
|---|---|---|
| `Si, Fe, Cu, Mn, Mg, Zn, Cr, Ni, Ti, Sr` | wt% (não reportado → 0) | sim |
| `cooling_rate_Cps` | °C/s (ou converta de SDAS: CR = (SDAS/39,4)^(−1/0,317)) | sim |
| `T_solution_C`, `T_aging_C` | °C | sim |
| `t_aging_h` | h | sim |
| `YS_MPa, UTS_MPa, El_pct` | MPa, MPa, % | sim |
| `f_alpha_Al, f_Si_eutectic, f_beta_AlFeSi, f_alpha_AlFeMnSi, f_Al2Cu` | fração mássica | opcional (supervisiona a cabeça de fases se todas presentes) |
| `alloy_id` | — | opcional (senão o grupo é inferido pela composição) |

## Componentes

### Física compartilhada (`metallurgy.py`)
Funções agnósticas ao backend: as mesmas equações rodam em NumPy no gerador e em
PyTorch, diferenciáveis, nas perdas da PINN.

| Fenômeno | Modelo |
|---|---|
| Liquidus / "C_eq" | Si equivalente via inclinações liquidus dos binários Al-X + polinômio de Djurdjevic |
| Fusão incipiente | 577 °C (Al-Si) → 525 °C (Al-Si-Al₂Cu) → 507 °C (+Mg, fase Q) |
| Química do Fe | Fe crítico de Taylor `0,075·(Si−4,05)`; sludge factor `Fe+2Mn+3Cr`; partição β/α sigmoidal em (Mn+2Cr)/Fe, com limiar que cai com a taxa de resfriamento |
| Solidificação | SDAS = 39,4·CR^−0,317 µm |
| Solubilização | solvus de Arrhenius para Cu (5,65 % a 548 °C) e Mg; Al₂Cu residual por estequiometria |
| Envelhecimento | Shercliff–Ashby: P = t_eq/t_pico (Q = 130 kJ/mol; Cu retarda), S(P) = [2√P/(1+P)]^½ |
| YS | σ₀ + solução sólida + S·√(σ_Mg² + σ_Cu²) + k/√SDAS + Si eutético/α-Fe |
| Alongamento | dano exponencial por β-AlFeSi (escala com √SDAS), α-Fe, Al₂Cu, Si não modificado (Sr ou CR alto) |
| UTS | YS + H·(1 − e^(−El/4)): fundidos fraturam antes da estricção |

Valores de referência reproduzidos: A356-T6 em molde permanente ≈ 205/295 MPa e 7 %;
o Fe a 0,6 % derruba o alongamento de 3,4 % para 1,1 % em areia, e 0,35 % de Mn
recupera parte da perda; uma 319 solubilizada a 540 °C queima.

### Geração CALPHAD (`calphad_gen.py`)
* `latin_hypercube`: LHS com otimização de discrepância (`scipy.stats.qmc`), em escala
  log10 para a taxa de resfriamento e o tempo de envelhecimento.
* A **T de solubilização é amostrada como margem abaixo do eutético** da própria liga
  (`solution_margin_C = (−15, 75) °C`), como na prática industrial. Cerca de 20 % das
  amostras saem "queimadas", para que o modelo aprenda o dano.
* `PyCalphadBackend`: equilíbrio de Gibbs com a base TDB fornecida, em **duas
  temperaturas** (solubilização e forno de espera). As fases são classificadas por
  padrão de nome (`classify_phase`: COST507/TCAL/PanAl usam nomes diferentes), e o
  Cu/Mg dissolvidos vêm da composição da matriz FCC rica em Al. Elementos ausentes da
  base são ignorados, com aviso. Validado contra o solvus Al-Cu real: 2,40 % Cu a
  450 °C (CALPHAD) contra 2,55 % (empírico).
* `EmpiricalThermoBackend`: mesmas saídas, vetorizado e instantâneo.

### Pré-processamento (`preprocessing.py`)
* `compute_c_eq(df, method)`: **"C_eq" para alumínio**. Nos aços, o carbono
  equivalente resume o efeito combinado da liga. Os índices análogos em Al-Si são
  `SI_EQ` (padrão: liquidus/fluidez), `SLUDGE` (Fe+2Mn+3Cr) e `MN_FE` (neutralização
  do β).
* Outliers em duas camadas: um **gate MAD** univariado nos alvos (erros grosseiros,
  como ksi digitado como MPa, que o IsolationForest deixa passar em 17 dimensões) e
  depois o **IsolationForest** multivariado.
* Features derivadas: Fe_crit, Fe_excess, T_liquidus, T_incipiente, intervalo de
  solidificação, margem de solubilização, SDAS, log10(P).

### PINN (`pinn.py`)
* Normalização *dentro* do modelo (buffers), de modo que o autograd opera em unidades físicas.
* Tronco MLP residual + cabeça mecânica `[YS, UTS, El]` + cabeça de fases (softmax de
  5 fases; Σf = 1 imposto rigidamente). A supervisão de fases usa a **distância de
  Hellinger** (MSE em √f), sensível às fases minoritárias (β ~0,5 %).
* Resíduos físicos aplicados nos dados **e em pontos de colocação** espalhados pelo
  espaço de projeto inteiro: `strength_model`, `ductility_law`, `work_hardening`,
  `mass_balance` (o Fe é insolúvel, então todo ele tem de estar em β/α; Cu no Al₂Cu
  ≤ Cu total; Si nas fases ≤ Si total), `monotonicity` (∂YS/∂Mg ≥ 0, ∂YS/∂Cu ≥ 0,
  ∂El/∂Fe ≤ 0, ∂El/∂log CR ≥ 0), `ys_le_uts`, `non_negative`.
* Incerteza epistêmica por MC-dropout (`predict(..., n_mc_samples=N)`), utilizável
  como restrição no design inverso (`OptimizationConfig.max_uts_cv`).

### Treinamento (`train.py`)
* `GroupKFold` por liga (sem vazamento composicional) e estatísticas de normalização
  ajustadas só no fold de treino. Early stopping, AdamW + cosine annealing.
* Optuna (TPE + MedianPruner por fold) sobre largura, profundidade, dropout, lr,
  weight decay e λ_física. O modelo final é treinado em todos os dados pelo número
  médio de épocas ótimas da CV.

### Modelo de carga (`charge.py`)
* `ChargeModel`: sucata (ou mistura fixa de classes) + Al primário + adições,
  vetorizado e em **forma fechada**. Para cada alvo, calcula a fração máxima de sucata
  limitada pela impureza mais restritiva. Também calcula custo (com rendimento metálico
  da sucata e recuperação das adições), CO₂ e `charge_violation_wt` (alvo inatingível).
* **Elementos não adicionáveis** (Fe, Zn, Ni, Cr: impurezas exceto Cu): o alvo é
  um *teto*, e a PINN recebe o teor que a carga de fato entrega. Sem isso, o otimizador
  "adicionaria" Zn para deslocar Al primário e baixar o CO₂ contábil.
* `optimize_charge_lp`: programação linear (HiGHS) com várias classes de sucata,
  disponibilidade por classe, faixas de especificação e objetivo de custo, CO₂ ou
  custo + preço de carbono. Coincide com a forma fechada para uma única sucata (testado).

### Design inverso (`inverse_design.py`)
* 4 objetivos: −UTS, −El, custo (USD/kg de banho), CO₂ (kg/kg). Direções de referência Das-Dennis.
* Restrições: UTS/El/YS mínimos, fração mínima de sucata, carga atingível, nenhuma
  regra crítica, `risk_score` máximo e, opcionalmente, incerteza.
* Variáveis podem ser fixadas, por exemplo a taxa de resfriamento do processo da fundição:
  `OptimizationConfig.fixed_values={"cooling_rate_Cps": 2.0}`.
* `select_compromise`: Achievement Scalarizing Function com pesos do usuário.

### Filtro termodinâmico (`thermo_filter.py`)
Regras vetorizadas com severidade:

| Código | Severidade | Critério |
|---|---|---|
| `INCIPIENT_MELTING` | CRÍTICA | T_sol > T_incipiente − 5 °C |
| `SLUDGE_FORMATION` | CRÍTICA | SF > 1,8 + 0,008·(T_espera − 650) |
| `INCIPIENT_MELTING_CALPHAD` / `UNDESIRED_PHASES_CALPHAD` | CRÍTICA | líquido na solubilização / sludge previstos por CALPHAD |
| `FE_ABOVE_CRITICAL`, `MN_FE_RATIO_LOW`, `BETA_PHASE_FRACTION` | AVISO | química do Fe |
| `INCOMPLETE_DISSOLUTION`, `WIDE_FREEZING_RANGE` | AVISO | processo/fundibilidade |
| `ZN_TRAMP`, `NI_TRAMP`, `SR_OVERMODIFICATION`, `TI_PRIMARY_TIAL3`, `HYPEREUTECTIC` | AVISO | impurezas e aditivos |
| `CU_CORROSION` | INFO | Cu > 1 % |

`validate_with_calphad` recalcula o equilíbrio apenas no conjunto de Pareto.

## Limitações e próximos passos

* **Os dados sintéticos valem o que valem os modelos que os geram.** As leis
  constitutivas são semiquantitativas, calibradas para reproduzir ordens de grandeza
  de A356/319. Use-as para validar o pipeline e como *prior*, e calibre com dados
  experimentais (fine-tuning em `data/raw`).
* Bases TDB completas de Al (COST 507, TCAL, PanAl) não são distribuídas aqui. O teste
  CALPHAD usa a base Al-Cu-Y de teste do pycalphad.
* `NP` do pycalphad é fração molar de fase, aproximada como fração mássica (o erro é
  pequeno para ligas diluídas e maior para intermetálicos com Fe/Cu).
* Preços (`ELEMENT_PRICE_USD_PER_KG`, `SCRAP_GRADES`) e fatores de CO₂ são
  **indicativos**. Atualize com cotações LME/Fastmarkets, com a análise OES do seu
  pátio de sucata e com o LCA da sua planta.
* Fora do escopo atual: porosidade por hidrogênio e bifilmes (que dominam a dispersão
  de ductilidade real), Scheil (pycalphad-scheil) para a sequência de solidificação,
  remoção de Fe por sludging deliberado (sedimentação a baixa temperatura), elementos
  traço (Pb, Sn, Ca, P), fadiga, e otimização bayesiana com aquisição ativa de experimentos.
