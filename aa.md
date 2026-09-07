# Plano de Execução — Correção do Estimador de Entropia (ESAE-IS)

> **Documento autocontido.** Não depende de `PLANO_CORRECOES_V2.md`.
>
> **Repositório:** https://github.com/bdlemos/Unsupervised_IS
> ⚠️ **O clone público está atrás da máquina de experimentos.** Trabalhar sempre na cópia
> local, que contém os métodos parametrizados de exp2–exp4. Ver §2.3.
> **Objetivo:** corrigir o estimador que define a taxa de redução adaptativa, que hoje
> colapsa em datasets grandes.
> **Escopo:** apenas o mecanismo entropia→taxa. Não mexer em autoencoder, sublinear
> sampling, classificação ou texto do paper.

---

## 1. O problema, em uma frase

O ESAE-IS remove **9,8%** das instâncias no `medline` (economia líquida de 3,7% de tempo)
enquanto o baseline supervisionado `biois` remove 32,8% (economia de 27,3%). O mecanismo poda
**menos** justamente nos datasets grandes, onde poderia poupar mais — invertendo a premissa
econômica do método.

### 1.1 Evidência

Taxa de redução e entropia recuperada (invertendo a Eq. 3, ver §2.2):

| dataset | redução | H |
|---|---|---|
| **medline** | 9,76% | **0,9856** ← maior H do conjunto |
| ohsumed | 12,02% | 0,9818 |
| trec | 13,99% | 0,9783 |
| books | 18,08% | 0,9705 |
| **agnews** | 20,88% | 0,9646 |
| 20ng | 21,83% | 0,9625 |
| reuters90 | 23,10% | 0,9595 |
| sst2 | 23,28% | 0,9591 |

### 1.2 Diagnóstico

```
H(dataset, K) = f(estrutura do dado)  +  g(K, N)
                └── o que queremos ──┘  └─ contaminação ─┘
```

`g` é viés de tamanho finito. A ocupação média por cluster é `N/K`; quanto maior, menores as
flutuações relativas de ocupação, mais uniforme a distribuição, maior `H`.

Confirmado empiricamente por regressão com efeitos fixos de dataset sobre os runs de
sensitivity (K ∈ {50, 100, 200} × 6 datasets):

```
(1 − H) ~ log K        coef = +0,00805    p = 1,4e-03    R² intra-dataset = 0,619
```

De K=50 a K=200, `H` varia **0,0112**. A variação de `H` **entre datasets** (a K=200) é
**0,0373**. Ou seja: **K move H em ~30% da magnitude do sinal que se quer medir.**

Como `K = √N`, a ocupação média é `√N`, que cresce com N. Datasets grandes → clusters maiores
→ ocupação mais uniforme → `H → 1` → `r → 0`.

### 1.3 A premissa da Eq. 3 NÃO está errada

Isto é importante para não jogar fora o mecanismo. Testando a premissa central contra a
tolerância empírica (inclinação de Macro-F1 vs. taxa de redução, medida nas curvas de taxa
fixa de 10% a 40%):

| dataset | H | tolerância empírica (pp F1 / pp removido) |
|---|---|---|
| sst2 | 0,9591 | −0,0030 |
| reuters90 | 0,9595 | +0,0132 |
| 20ng | 0,9625 | −0,0782 |
| books | 0,9705 | −0,0732 |
| trec | 0,9783 | −0,0107 |
| ohsumed | 0,9818 | −0,0871 |

```
Spearman(H, tolerância)              = −0,714   p = 0,111
Spearman(taxa escolhida, tolerância) = +0,714   p = 0,111
```

Ambos **na direção que a Eq. 3 prevê**. Com n=6 o Spearman exige |ρ| ≥ 0,886 para p<0,05, então
ρ=0,714 é efeito forte mas subpotente — não é nulo.

> **Conclusão do diagnóstico:** o problema é **estimador contaminado**, não premissa quebrada.
> A correção é estatística, não uma reengenharia do método.

---

## 2. Estado atual do código

### 2.1 Arquivos relevantes

| Caminho | Papel |
|---|---|
| `unsupervised-is/src/main/python/iSel/entropy_sublinear_ae_is.py` | Classe `EntropySublinearAEIS`. Contém a Eq. 3 e o cálculo de `H` |
| `unsupervised-is/src/main/python/iSel/sublinear_ae_is.py` | `SublinearAEIS` — versão de taxa fixa |
| `unsupervised-is/src/main/python/iSel/autoencoder_is.py` | `AutoencoderIS` — produz `reconstruction_errors_` |
| `unsupervised-is/scripts/run_generateSplit.py` | `get_selector()` registra métodos; loop de folds |
| `run_pipeline.sh` | Orquestrador. Define `RESULTS_DIR` |

### 2.2 Onde vive a Eq. 3

`entropy_sublinear_ae_is.py`, linhas 181–183:

```python
H = _compute_normalized_entropy(sizes)
target_reduction = self.r_max * (1.0 - H ** self.alpha)
target_reduction = max(0.05, min(target_reduction, self.r_max))  # Clamp to [5%, r_max]
```

E `_compute_normalized_entropy`, linhas 70–87:

```python
nonzero = sizes[sizes > 0]
k = len(nonzero)               # <-- normaliza por clusters NÃO-VAZIOS, não por K pedido
probs = nonzero / total
entropy = -np.sum(probs * np.log(probs))
max_entropy = np.log(k)
return entropy / max_entropy
```

**Detalhe que importa:** a normalização usa `log(k_não_vazios)`, não `log(K_pedido)`.
MiniBatchKMeans pode produzir clusters vazios. Os dois valores precisam ser logados
separadamente (§4).

### 2.3 ⚠️ O GitHub está atrás da máquina de experimentos

O clone público em `github.com/bdlemos/Unsupervised_IS` **não reflete o código que rodou os
experimentos**. Duas divergências conhecidas, ambas já resolvidas na máquina local e apenas
não sincronizadas:

**(a) `n_clusters` aparece como código morto no clone.** Linhas 146–147:

```python
# n_clusters = max(2, min(self.n_clusters, n_samples // 10))
n_clusters = np.sqrt(n_samples).astype(int)  # Use sqrt(N) for micro-clusters
```

No clone, o parâmetro do construtor é ignorado. **Na máquina local ele é respeitado** — os
resultados de `exp3-sensitivity/K_{50,100,200,sqrtN}` mostram taxas de redução distintas por K,
o que só é possível com o parâmetro ativo.

**(b) Métodos parametrizados ausentes no clone.** Não existem `iSel/ablation_methods.py` nem
registros `esae-K*`, `esae-alpha*`, `esae-gamma*`, `sae-rate-*`, `ablation-*` em
`get_selector()`. **Existem na máquina local** — foi o que produziu exp2, exp3 e exp4.

> **Trabalhar sempre na cópia da máquina de experimentos, nunca no clone do GitHub.**
> T0 e T1 (§5) são tarefas de **verificação de contrato**, não de implementação do zero.

A resolução de `n_clusters` que o resto deste plano assume — verificar se bate com a local e
estender se necessário:

```python
if self.n_clusters is None or self.n_clusters == 'sqrt':
    n_clusters = max(2, int(np.sqrt(n_samples)))
elif isinstance(self.n_clusters, str) and self.n_clusters.startswith('n_over_'):
    m = int(self.n_clusters.split('_')[-1])
    n_clusters = max(2, int(n_samples / m))
else:
    n_clusters = max(2, min(int(self.n_clusters), n_samples - 1))
```

A sentinela `'n_over_<m>'` é **nova** e provavelmente não existe nem na local — ela habilita a
opção K ∝ N (ocupação média constante), testada em T2.

### 2.4 Detalhe que sobrevive à dessincronia

A observação da §2.2 sobre `_compute_normalized_entropy` normalizar por `log(k_não_vazios)` em
vez de `log(K_pedido)` é sobre a **semântica do estimador**, não sobre versão de código.
Confirmar na cópia local, mas o plano assume que continua valendo.

### 2.5 Direção da amostragem

`entropy_sublinear_ae_is.py`, linhas 219–221:

```python
inv_scores = 1.0 / (c_scores + 1e-8)
p_remove = inv_scores / np.sum(inv_scores)
```

A probabilidade de **remoção** é inversamente proporcional ao erro de reconstrução. Ou seja,
o método **retém preferencialmente instâncias de alto erro**. Registrar isto — não alterar
neste plano.

### 2.6 Tesselações já implementadas

O construtor aceita `clustering_method` ∈ `{'minibatchkmeans', 'gmm', 'birch', 'dbscan'}`
(linhas 150–170). A branch DBSCAN usa `eps=0.5, min_samples=5` hardcoded e recalcula
`n_clusters` a partir dos labels. Isso reduz o trabalho da Tarefa T5.

---

## 3. Organização dos resultados — OBRIGATÓRIA

Seguir **exatamente** a convenção já usada em `exp1-reconstruction-error/`,
`exp2-ablation/`, `exp3-sensitivity/`, `exp4-fixed-rates/`, `exp5-entropy-vs-empirical/`.

Cada experimento novo é um diretório `results/exp<N>-<slug>/`. Não inventar outra estrutura,
não achatar, não renomear.

### 3.1 Experimentos deste plano

```
results/
├── exp7-entropy-instrumentation/     # T2 — dumps brutos de H, K, N, ocupação
├── exp8-entropy-estimator/           # T3 — análise offline dos estimadores + V1–V3
├── exp9-tessellation/                # T5 — tesselações alternativas (CONDICIONAL)
└── exp10-rate-recalibration/         # T6 — varredura r_max/α + aceitação V4
```

### 3.2 Layout de `exp7-entropy-instrumentation/`

Segue o padrão de `exp1-reconstruction-error/`: **um subdiretório por dataset**, arquivos de
análise dentro.

```
results/exp7-entropy-instrumentation/
├── <dataset>/
│   ├── occupancy_fold<F>_K<KSPEC>.csv      # vetor de ocupação, uma linha por cluster
│   └── entropy_metrics.csv                 # uma linha por (fold, KSPEC)
└── instrumentation_summary.csv             # consolidado de todos os datasets
```

`<dataset>` ∈ os 19 originais + `agnews` + `medline`.
`<F>` = índice do fold (0-based).
`<KSPEC>` = especificação de K como string segura para nome de arquivo:
`sqrt`, `50`, `100`, `200`, `400`, `800`, `n_over_25`, `n_over_50`, `n_over_100`.

**Schema de `occupancy_fold<F>_K<KSPEC>.csv`:**

| coluna | tipo | descrição |
|---|---|---|
| `cluster_id` | int | índice do cluster, 0-based |
| `size` | int | número de instâncias atribuídas |

Incluir clusters vazios (`size=0`). O número de linhas deve ser exatamente `K_requested`.

**Schema de `entropy_metrics.csv`:**

| coluna | tipo | descrição |
|---|---|---|
| `dataset` | str | nome do dataset |
| `fold` | int | 0-based |
| `k_spec` | str | `sqrt`, `200`, `n_over_50`, … |
| `k_requested` | int | K pedido ao algoritmo |
| `k_nonempty` | int | clusters com `size > 0` |
| `n_samples` | int | N do fold de treino |
| `H_lognonempty` | float | **estimador atual** — normalizado por `log(k_nonempty)` |
| `H_logk` | float | normalizado por `log(k_requested)` |
| `target_reduction` | float | taxa que a Eq. 3 produz, em [0,1] |
| `achieved_reduction` | float | taxa efetivamente alcançada após o sublinear |
| `clustering_method` | str | `minibatchkmeans`, `dbscan`, … |
| `random_state` | int | semente |
| `selection_time_s` | float | segundos da seleção completa |

### 3.3 Layout de `exp8-entropy-estimator/`

Segue o padrão de `exp5-entropy-vs-empirical/`: análise agregada na raiz, plots por dataset.

```
results/exp8-entropy-estimator/
├── estimators.csv                  # todos os estimadores por (dataset, fold, k_spec)
├── tolerance.csv                   # tolerância empírica por dataset
├── V1_k_dependence.csv             # resultado do teste V1
├── V2_n_dependence.csv             # resultado do teste V2
├── V3_tolerance_correlation.csv    # resultado do teste V3
├── verdict.json                    # aprovação/reprovação por estimador
├── plot_V1_estimator_vs_K.png
├── plot_V2_estimator_vs_N.png
└── plot_V3_estimator_vs_tolerance.png
```

### 3.4 Layout de `exp9-tessellation/` e `exp10-rate-recalibration/`

Seguem o padrão de `exp3-sensitivity/` — **um subdiretório por configuração**, cada um com a
estrutura completa de pipeline:

```
results/exp9-tessellation/
└── <method>_<param>/                    # ex: voxel_eps05, leader_eps03, hdbscan_mcs25
    ├── instance_selection/
    │   ├── selection_summary.csv
    │   ├── selection_summary.json
    │   ├── logs/
    │   │   ├── <dataset>_<method>.log
    │   │   ├── failed_methods.txt
    │   │   └── failed_pairs.csv
    │   └── selection/
    │       └── <dataset>/
    │           ├── saida_<method>
    │           ├── saida_<method>.json
    │           ├── split_10_<method>.pkl
    │           └── split_10_<method>_idxinfold.pkl
    └── classificacao/
        └── results/
            ├── results_modernbert.csv
            └── times_modernbert.csv

results/exp10-rate-recalibration/
└── rmax<R>_alpha<A>/                    # ex: rmax06_alpha10, rmax05_alpha08
    ├── instance_selection/  (idem)
    └── classificacao/       (idem)
```

### 3.5 Schemas dos CSVs de pipeline — NÃO ALTERAR

**`classificacao/results/results_modernbert.csv`**

```
dataset,method,reduction,class,mac0,mac1,mac2,mac3,mac4,mac5,mac6,mac7,mac8,mac9,macro_avg_ic
```

**`classificacao/results/times_modernbert.csv`**

```
dataset,method,reduction,class,time0,time1,time2,time3,time4,time5,time6,time7,time8,time9,time_avg_ic
```

**`instance_selection/selection_summary.csv`**

```
dataset,method,time_mean,time_error,time_n,reduction_mean,reduction_error,reduction_n
```

Notas:
- `mac<i>` = Macro-F1 do fold `i`, em [0,1] com 10 casas decimais.
- `time<i>` = segundos do fold `i`.
- Datasets com 5 folds (`agnews`, `medline`) deixam `mac5..mac9` e `time5..time9` **vazios**,
  não zero. É o comportamento atual e a análise depende dele.
- `macro_avg_ic` e `time_avg_ic` ficam vazios — também é o comportamento atual.
- `reduction` é fração em [0,1], não percentual.

> **Atenção ao sufixo.** `run_pipeline.sh:190` gera
> `results_${OUTPUT_SUFFIX}.csv`. Os diretórios `exp2`–`exp4` usam `results_modernbert.csv`;
> a pasta principal `results/jina-v5/` usa `results_from_outputs.csv`. **Para os experimentos
> deste plano, usar `results_modernbert.csv`**, consistente com `exp2`–`exp4`.

### 3.6 Como direcionar a saída

Igual aos experimentos anteriores — via `RESULTS_DIR`:

```bash
RESULTS_DIR=/app/results/exp10-rate-recalibration/rmax06_alpha10 \
  bash run_pipeline.sh \
    --methods "esae-rmax06-alpha10" \
    --datasets "sst2,ohsumed,reuters90,books,trec,20ng,agnews,medline" \
    --inputrep "jina-v5"
```

`run_pipeline.sh:94` respeita `RESULTS_DIR` se já estiver no ambiente.

---

## 4. Definições matemáticas — implementar exatamente assim

Seja `s = (s_1, …, s_K)` o vetor de ocupação, `N = Σ s_c`, `p_c = s_c / N`.

### 4.1 E0 — entropia normalizada (atual, baseline de comparação)

```
K_ne = #{c : s_c > 0}
H    = −(1 / ln K_ne) · Σ_{c: s_c>0} p_c ln p_c
```

Convenção `0·ln0 = 0`. Se `K_ne ≤ 1`, retornar `1.0`.

Calcular **também** `H_logk` usando `ln K_requested` no denominador, e logar os dois.

### 4.2 E1 — ΔH, entropia em excesso sobre o acaso ⭐ candidato principal

Monte Carlo, determinístico dada a semente:

```
Para b = 1..B (B = 1000, seed = 13):
    sortear s^(b) ~ Multinomial(N, uniforme sobre K_requested bins)
    H_b = entropia normalizada de s^(b), normalizada por ln(#{c: s^(b)_c > 0})
H_null(N, K) = média(H_b)
ΔH           = H_null − H_obs
ΔH_norm      = ΔH / H_null            # em [0, 1] na prática
```

**Por que funciona:** `H_null` captura exatamente o viés de tamanho finito `g(K, N)`.
Subtraí-lo remove a contaminação sem tocar na tesselação.

Cachear `H_null(N, K)` — depende só de `(N, K, B, seed)`, não dos dados. Um dicionário em
memória basta.

**Direção:** `ΔH` alto = mais concentração que o acaso = mais redundância = **podar mais**.
É o inverso da direção de `H`. Cuidado ao plugar na Eq. 3.

### 4.3 E2 — Gini da ocupação

```
s_ord = sort(s)  (ascendente, incluindo zeros)
Gini  = (2·Σ_{i=1}^{K} i·s_ord_i) / (K·Σ s) − (K+1)/K
```

Gini alto = ocupação concentrada = redundante. Mesma direção de `ΔH`.

### 4.4 E3 — razão de clusters efetivos

```
K_eff = exp(−Σ_{c: s_c>0} p_c ln p_c)      # número efetivo de clusters (perplexidade)
E3    = K_eff / K_requested                 # em (0, 1]
```

E3 baixo = poucos clusters dominam = redundante. Direção **inversa** a `ΔH`.

### 4.5 E4 — CV da ocupação relativo ao nulo

```
CV      = desvio_padrão(s) / média(s)
CV_null = √((1 − 1/K) / (N/K))              # analítico, multinomial uniforme
E4      = CV / CV_null
```

E4 alto = mais disperso que o acaso = redundante. Mesma direção de `ΔH`.

### 4.6 E5 — Gini dos erros de reconstrução (não usa clustering)

Sobre o vetor de erros `ε = (ε_1, …, ε_N)` do autoencoder, mesma fórmula de Gini de E2.

**Vantagem:** livre de escala por construção, independe de K, e reaproveita computação que já
existe. **Só implementar se E1 reprovar** — exige persistir `ε` por fold, o que aumenta o
escopo. Os erros já existem para 18 datasets em
`marcus_fb/exp1-reconstruction-error/<dataset>/per_instance_metrics.csv`, coluna
`reconstruction_error`, mas **apenas do fold 0** e **sem** `agnews`/`medline`.

### 4.7 Tolerância empírica (alvo dos testes)

Para cada dataset, sobre os 7 pontos de taxa fixa de `exp4-fixed-rates/`:

```
xs = achieved_reduction × 100      (percentual, das 7 taxas)
ys = média sobre folds de Macro-F1 × 100
tolerância = coeficiente angular de OLS(ys ~ xs)     # pp de F1 por pp removido
```

Mais negativo = menos tolerante = deve podar menos.

**Para `agnews` e `medline` não existem curvas de 7 pontos.** Usar estimativa de 2 pontos a
partir de `results/_tmp_modernbert/classificacao/results/results_from_outputs.csv`:

```
tolerância ≈ média sobre m ∈ {sublinear-ae-is, autoencoder-is} de:
             (F1(m) − F1(no-is)) / (reduction(m) × 100)
```

Marcar essas duas linhas com `tolerance_source = "2point"` e as demais com `"7point"` em
`tolerance.csv`. **Não misturar sem essa marcação** — a de 2 pontos é bem mais ruidosa.

---

## 5. Tarefas

### T0 — Verificar `n_clusters` e estender com `n_over_<m>` 🔴 BLOQUEANTE

**Arquivo:** `unsupervised-is/src/main/python/iSel/entropy_sublinear_ae_is.py`
**Trabalhar na cópia da máquina de experimentos, não no clone do GitHub (§2.3).**

1. **Verificar** que o parâmetro `n_clusters` do construtor é respeitado (deve ser — é o que
   produziu `exp3-sensitivity/K_*`). Se a cópia local ainda tiver o hardcode das linhas
   146–147, aplicar o bloco de resolução da §2.3.
2. **Estender** com a sentinela `'n_over_<m>'` — é nova e provavelmente não existe nem na
   local. Habilita K ∝ N, testada em T2.
3. Confirmar que o default é `'sqrt'` (ou equivalente).
4. Guardar `self.n_clusters_used_ = n_clusters` logo após a resolução.

**Teste de regressão obrigatório:** rodar `entropy-sublinear-ae-is` com o default em `sst2` e
confirmar `achieved_reduction ≈ 0,2328` e em `ohsumed` ≈ `0,1202`. Se não bater, a cópia
divergiu do que gerou `exp3-sensitivity/K_sqrtN` e **parar** — todo o resto do plano compara
contra esses números.

**Não** alterar o comportamento default: `'sqrt'` deve reproduzir exatamente os números atuais.

---

### T1 — Verificar e completar os métodos parametrizados 🔴 BLOQUEANTE

**Arquivo:** `unsupervised-is/scripts/run_generateSplit.py`, função `get_selector()`
(a partir da linha ~54).

Os registros de `exp2`–`exp4` **existem na máquina local** (§2.3b). Esta tarefa é:
(i) verificar que batem com o contrato de nomes abaixo, e
(ii) **adicionar os que ainda não existem**, que são necessários para T2 e T6:

| Padrão | Existe? | Necessário para |
|---|---|---|
| `esae-K<n>`, `esae-alpha<a>`, `esae-gamma<g>` | sim (local) | verificar contrato |
| `sae-rate-<10..40>` | sim (local) | verificar contrato |
| `ablation-*` | sim (local) | não usado neste plano |
| **`esae-Kn_over_<m>`** | **não** | T2 — grade de instrumentação |
| **`esae-rmax<R>-alpha<A>`** | **não** | T6 — recalibração |
| **`sae-rate-50`, `sae-rate-60`** | **não** | T6 — estender curva de tolerância |

Contrato de referência, por **parsing de prefixo**, antes dos `if method == ...` literais.
Se a versão local usar outra convenção de nome, **manter a local** e ajustar os comandos das
seções seguintes — o que não pode é divergir silenciosamente:

```python
import re

# esae-K<spec>      -> esae-K50, esae-K200, esae-Ksqrt, esae-Kn_over_50
m = re.fullmatch(r'esae-K(.+)', method)
if m:
    spec = m.group(1)
    kval = spec if spec in ('sqrt',) or spec.startswith('n_over_') else int(spec)
    return entropy_sublinear_ae_is.EntropySublinearAEIS(
        r_max=0.50, alpha=15.0, n_clusters=kval, gamma=0.5,
        ae_epochs=50, random_state=13)

# esae-alpha<A>     -> esae-alpha5 ... esae-alpha25
m = re.fullmatch(r'esae-alpha(\d+)', method)
if m:
    return entropy_sublinear_ae_is.EntropySublinearAEIS(
        r_max=0.50, alpha=float(m.group(1)), n_clusters='sqrt', gamma=0.5,
        ae_epochs=50, random_state=13)

# esae-gamma<G>     -> esae-gamma03, esae-gamma05, esae-gamma07  (dois dígitos = /10)
m = re.fullmatch(r'esae-gamma(\d+)', method)
if m:
    return entropy_sublinear_ae_is.EntropySublinearAEIS(
        r_max=0.50, alpha=15.0, n_clusters='sqrt',
        gamma=int(m.group(1)) / 10.0, ae_epochs=50, random_state=13)

# esae-rmax<R>-alpha<A>  -> esae-rmax06-alpha10   (dois dígitos = /10)
m = re.fullmatch(r'esae-rmax(\d+)-alpha(\d+)', method)
if m:
    return entropy_sublinear_ae_is.EntropySublinearAEIS(
        r_max=int(m.group(1)) / 10.0, alpha=float(m.group(2)),
        n_clusters='sqrt', gamma=0.5, ae_epochs=50, random_state=13)

# sae-rate-<R>      -> sae-rate-10 ... sae-rate-60  (percentual inteiro)
m = re.fullmatch(r'sae-rate-(\d+)', method)
if m:
    return sublinear_ae_is.SublinearAEIS(
        target_reduction=int(m.group(1)) / 100.0, n_clusters=200,
        gamma=0.5, ae_epochs=50, random_state=13)
```

⚠️ `random_state=13` em tudo — é o valor usado nos experimentos anteriores. Mudar quebra a
comparabilidade.

⚠️ `SublinearAEIS` mantém `n_clusters=200` porque é assim que `exp4-fixed-rates` foi rodado.
**Não** trocar para `'sqrt'` sem rerodar o exp4 inteiro.

---

### T2 — Instrumentação 🔴 BLOQUEANTE

Produz `results/exp7-entropy-instrumentation/` (§3.2).

**Arquivo 1:** `entropy_sublinear_ae_is.py` — expor os atributos após `select_data`:

```python
self.n_clusters_used_   = n_clusters          # já em T0
self.k_nonempty_        = int(np.sum(sizes > 0))
self.cluster_sizes_     = sizes               # já existe (linha 174)
self.entropy_           = H                   # já existe (linha 185)
self.entropy_logk_      = _normalized_entropy_logk(sizes, n_clusters)
self.n_samples_         = n_samples
self.target_reduction_  = target_reduction    # renomear de self.target_reduction
```

Adicionar a função auxiliar (não modificar `_compute_normalized_entropy`, que é o baseline E0):

```python
def _normalized_entropy_logk(sizes, k_requested):
    total = np.sum(sizes)
    if total == 0 or k_requested <= 1:
        return 1.0
    nz = sizes[sizes > 0]
    p = nz / total
    return float(-np.sum(p * np.log(p)) / np.log(k_requested))
```

**Arquivo 2:** `run_generateSplit.py` — no loop de folds (a partir da linha ~153), após
`idxs_docs = get_selection(...)`, capturar do seletor e acumular. O seletor está dentro de
`get_selection()`; alterar essa função para **retornar também o objeto seletor**, ou expor via
atributo no `info`.

Escrever, por dataset:
- `occupancy_fold<F>_K<KSPEC>.csv` — uma linha por cluster, incluindo vazios
- `entropy_metrics.csv` — append de uma linha por (fold, k_spec)

E ao final, consolidar tudo em `instrumentation_summary.csv` na raiz do exp7.

**Grade a rodar:**

| eixo | valores |
|---|---|
| datasets | os 19 originais + `agnews` + `medline` |
| k_spec | `sqrt`, `50`, `100`, `200`, `400`, `800`, `n_over_25`, `n_over_50`, `n_over_100` |
| folds | todos (10; 5 para agnews/medline) |

> **Isto NÃO exige rodar classificação.** É só a fase de seleção. Rodar apenas o passo 1 do
> pipeline (`--steps "1"` ou `--steps "selection"`, ver `run_pipeline.sh:74`). Isso torna a
> grade viável: são clusterizações, não fine-tunings.

---

### T3 — Análise offline dos estimadores

Produz `results/exp8-entropy-estimator/` (§3.3).

**Novo arquivo:** `unsupervised-is/analysis/exp8_entropy_estimator.py`

Lê `exp7-entropy-instrumentation/*/occupancy_*.csv`, calcula E0–E4 (§4.1–4.5) para cada
`(dataset, fold, k_spec)`, e escreve `estimators.csv`:

| coluna | descrição |
|---|---|
| `dataset`, `fold`, `k_spec`, `k_requested`, `k_nonempty`, `n_samples` | identificação |
| `E0_H`, `E1_dH`, `E1_dH_norm`, `E2_gini`, `E3_keff_ratio`, `E4_cv_ratio` | estimadores |
| `H_null` | valor Monte Carlo usado em E1 |

Depois lê as curvas de `exp4-fixed-rates/` e o CSV dos grandes, monta `tolerance.csv`
(§4.7) e roda os testes abaixo.

---

### T4 — Testes de validação V1–V3

Todos rodam offline, sem GPU. Resultados em CSV + `verdict.json`.

#### V1 — independência de K (o teste que separa estimador limpo de contaminado)

Para cada estimador `E`, regressão com **efeitos fixos de dataset** (centralização
intra-dataset), usando só as k_specs numéricas fixas (`50`, `100`, `200`, `400`, `800`):

```
E ~ log(k_requested) + dataset FE
```

Reportar: coeficiente, IC 95%, p, e **R² intra-dataset**.

| critério | aprovação |
|---|---|
| p do coeficiente | > 0,05 |
| R² intra-dataset | < 0,10 |

Referência do estimador atual: **R² = 0,619**, p = 1,4e-03. Qualquer coisa perto disso reprova.

#### V2 — independência de N

Uma observação por dataset, no k_spec default (`sqrt`), média sobre folds:

```
E ~ log(n_samples)
```

| critério | aprovação |
|---|---|
| p do coeficiente | > 0,05 |
| \|ρ Spearman(E, N)\| | < 0,4 |

**Este é o teste que o estimador atual reprova e que causa o problema do `medline`.**
Incluir `agnews` e `medline` — sem eles o teste não tem alcance de escala.

#### V3 — correlação com a tolerância empírica

```
ρ = Spearman(E, tolerância)
```

Sinal esperado depende da direção do estimador (§4):
- `E0_H`, `E3_keff_ratio`: ρ **negativo** (alto = intolerante)
- `E1_dH`, `E1_dH_norm`, `E2_gini`, `E4_cv_ratio`: ρ **positivo** (alto = tolerante/redundante)

| critério | aprovação |
|---|---|
| \|ρ\| | ≥ 0,7 no sinal correto |

⚠️ **O alvo é a INCLINAÇÃO, nunca o argmax.** Correlacionar contra a melhor taxa empírica
(argmax) dá ρ=0,27 (p=0,60) e é um alvo degenerado: a melhor taxa fixa é 10% — a menor
testada — em 4 de 6 datasets, porque a curva é monótona decrescente onde tem inclinação e
plana onde não tem. Usar argmax reproduz um resultado nulo espúrio.

⚠️ Com n=6 (ou n=8 com os grandes), V3 é subpotente por construção. **V3 não é critério de
reprovação** — é diagnóstico de direção. V1 e V2 são intra-dataset e têm poder suficiente com
os dados existentes; **são eles que decidem.**

#### `verdict.json`

```json
{
  "estimators": {
    "E1_dH_norm": {
      "V1": {"coef": 0.0, "ci95": [0.0, 0.0], "p": 0.0, "r2_within": 0.0, "pass": true},
      "V2": {"coef": 0.0, "p": 0.0, "spearman_N": 0.0, "pass": true},
      "V3": {"spearman": 0.0, "p": 0.0, "expected_sign": "positive", "pass": true},
      "overall": "APPROVED"
    }
  },
  "recommended_estimator": "E1_dH_norm",
  "decision": "PROCEED_TO_T6"
}
```

`decision` ∈ `{"PROCEED_TO_T6", "PROCEED_TO_T5"}`:
- **algum** estimador aprova V1 **e** V2 → `PROCEED_TO_T6` (pular T5)
- **nenhum** aprova → `PROCEED_TO_T5`

---

### T5 — Tesselações alternativas ⚠️ CONDICIONAL

**Só executar se `verdict.json.decision == "PROCEED_TO_T5"`.**

Produz `results/exp9-tessellation/` (§3.4).

A tesselação faz **dois trabalhos** em `EntropySublinearAEIS`:
(a) comparar erros de reconstrução localmente;
(b) fornecer a ocupação para `H`.

**Só (b) está quebrado.** Se T4 achar um estimador limpo, (a) continua bem servido por
MiniBatchKMeans e T5 é desnecessário. Por isso é condicional.

Se for necessária, ordem de prioridade:

| # | Método | Por quê | Custo |
|---|---|---|---|
| 1 | **Voxel grid / LSH em raio fixo** | Ocupação = densidade real; nº de células emergente; O(N). Já citado no related work do paper (Rusu et al., PCL) | Baixo |
| 2 | **Leader / canopy com raio ε** | Mesma propriedade, uma passada só | Baixo |
| 3 | **HDBSCAN** | Genuinamente density-aware | Médio, traz parâmetros próprios |
| 4 | **Bisecting k-means** | Dá hierarquia, granularidade por critério de parada | Médio |

❌ **Não investir em k-d tree.** É balanceado por construção, então a entropia de ocupação é
degenerada (`H ≈ 1` sempre) — inútil como estimador.

ℹ️ As branches `gmm`, `birch` e `dbscan` já existem (§2.6). DBSCAN usa `eps=0.5,
min_samples=5` hardcoded nas linhas 162; parametrizar antes de varrer.

ℹ️ **Observação para o texto do paper (não implementar):** a seção 3.4 rejeita DBSCAN dizendo
que ele "colapsa espaços densos em clusters monolíticos, levando H→0 e neutralizando a função
adaptativa". Mas `H→0` significa "redundância máxima detectada", o que é informativo, não uma
falha. A justificativa está frágil.

---

### T6 — Recalibração da taxa + aceitação

Produz `results/exp10-rate-recalibration/` (§3.4).

#### T6.1 — Plugar o estimador aprovado

Substituir a Eq. 3 pela forma correspondente à direção do estimador. Para `E1_dH_norm`
(direção "alto = mais redundante"):

```python
target_reduction = self.r_max * (dH_norm ** self.alpha)
```

Manter o clamp `max(0.05, min(target_reduction, self.r_max))`.

#### T6.2 — Ajustar o mapeamento, não chutar

⚠️ `α=15` foi escolhido para funcionar na faixa estreita que `H` produzia. Com estimador novo
a faixa muda e **α=15 não faz mais sentido**.

**Ajustar `α` e `r_max` por leave-one-dataset-out sobre a tolerância empírica.** Para cada
dataset deixado de fora, ajustar nos demais e prever a taxa dele. Reportar o erro
out-of-sample. Isso responde diretamente à crítica de "os valores foram escolhidos usando os
próprios datasets e depois chamados de zero-parameter".

Grade inicial sugerida: `r_max` ∈ {0,3; 0,4; 0,5; 0,6} × `α` ∈ {2; 4; 6; 8; 10; 15}.
Refinar depois do primeiro passe.

#### T6.3 — Critério de aceitação V4

Rodar seleção + classificação em `sst2, ohsumed, reuters90, books, trec, 20ng, agnews, medline`.

| # | Critério | Meta |
|---|---|---|
| V4.1 | Taxa no `medline` | ≥ 20% (hoje: 9,8%) |
| V4.2 | Economia líquida no `medline` | ≥ 15% (hoje: 3,7%) |
| V4.3 | Macro-F1 no `medline` | não-inferior ao full-data, δ=0,025 (hoje: −0,07 pp, passa) |
| V4.4 | Macro-F1 nos 6 pequenos | nenhuma queda > 0,5 pp vs. configuração atual |
| V4.5 | Taxa nos 6 pequenos | permanece em 12–30% |

**Economia líquida** = `(T_full − T_metodo) / T_full`, onde `T = tempo_treino + tempo_seleção`.
O overhead de seleção **não** é desprezível (447 s no medline) e precisa entrar na conta.

---

## 6. Ordem de execução

```
T0 (bug n_clusters)  ──┐
T1 (get_selector)    ──┼──> T2 (instrumentação, só seleção)
                       │         │
                       │         v
                       │    T3 (estimadores offline)
                       │         │
                       │         v
                       │    T4 (V1–V3)  ──> verdict.json
                       │                        │
                       │            ┌───────────┴───────────┐
                       │            v                       v
                       │    PROCEED_TO_T6            PROCEED_TO_T5
                       │            │                       │
                       │            │                  T5 (tesselação)
                       │            │                       │
                       │            └───────────┬───────────┘
                       │                        v
                       └──────────────────> T6 (recalibração + V4)
```

**T0–T4 não exigem GPU.** T2 é só clusterização; T3 e T4 são análise de CSV. O ponto de
decisão (`verdict.json`) chega **antes** de qualquer fine-tuning novo.

---

## 7. Erros a não cometer

0. **Não trabalhar no clone do GitHub.** Ele está atrás da máquina de experimentos e não tem
   os métodos parametrizados. Confirmar a origem do código antes de qualquer edição (§2.3).
1. **Não trocar o default `sqrt`** em T0. Ele precisa reproduzir os números atuais
   bit-a-bit para que a comparação com `exp2`–`exp4` valha — o teste de regressão de T0
   (`sst2 ≈ 0,2328`, `ohsumed ≈ 0,1202`) é a trava.
2. **Não usar `random_state` diferente de 13.** Quebra comparabilidade com tudo que já rodou.
3. **Não normalizar por `log(k_nonempty)` e por `log(k_requested)` indistintamente.**
   São grandezas diferentes; logar as duas separadamente (`H_lognonempty`, `H_logk`).
4. **Não correlacionar contra o argmax da curva** em V3. Ver o aviso em T4/V3.
5. **Não preencher `mac5..mac9` com zeros** em `agnews`/`medline`. Deixar vazio — a análise
   depende disso para saber que são 5 folds.
6. **Não misturar tolerância de 7 pontos com a de 2 pontos** sem a coluna
   `tolerance_source`.
7. **Não rodar classificação em T2.** A grade de 9 k_specs × 21 datasets × 10 folds é viável
   só porque é apenas seleção.
8. **Não manter `α=15`** depois de trocar o estimador. Reajustar por LODO (T6.2).
9. **Não mudar `SublinearAEIS` para `n_clusters='sqrt'`** sem rerodar `exp4-fixed-rates`
   inteiro — as curvas de tolerância vêm de lá e são o alvo de V3.
10. **Não alterar a direção da amostragem** (`p_remove ∝ 1/ε`, §2.5). É outro problema, fora
    do escopo deste plano.

---

## 8. Referências de dados

| Caminho | Conteúdo |
|---|---|
| `marcus_fb/exp1-reconstruction-error/<ds>/per_instance_metrics.csv` | RE por instância, fold 0, 18 datasets. Colunas: `reconstruction_error`, `centroid_distance`, `lr_confidence`, `lr_correct`, `knn_purity`, `true_label`, `predicted_label` |
| `marcus_fb/exp3-sensitivity/K_*/instance_selection/selection_summary.csv` | Taxas por K ∈ {50,100,200,√N} — origem do teste V1 de referência |
| `marcus_fb/exp4-fixed-rates/rate_*/classificacao/results/results_modernbert.csv` | Curvas de taxa fixa 10–40%, 6 datasets — origem da tolerância de 7 pontos |
| `results/_tmp_modernbert/classificacao/results/results_from_outputs.csv` | `agnews`, `medline` + baselines supervisionados. 5 folds |
| `results/_tmp_modernbert/instance_selection/selection_summary.csv` | Tempo e redução da seleção, incl. grandes |

**Convenções:** `mac0..mac9` = Macro-F1 por fold em [0,1]. `reduction` = fração em [0,1].
`time*` em segundos. `agnews`/`medline` têm 5 folds; os demais, 10.
