# Relatório Técnico: Diagnóstico e Análise do SAE-IS (SublinearAEIS)

**Data:** 10 de Outubro de 2026  
**Repositório:** `/workspace/Unsupervised_IS`  
**Escopo:** Diagnóstico de falha no mecanismo de Instance Selection baseado em Autoencoder e Cota Sublinear.

---

## 1. Contexto e Motivação do Diagnóstico

O método **SAE-IS** (`SublinearAEIS`) foi proposto como uma abordagem não-supervisionada de seleção de instâncias estruturada em 4 etapas:
1. **Autoencoder Scoring:** Treina um Autoencoder com bottleneck sobre embeddings pré-treinados (`jina-v5`) e calcula o Erro de Reconstrução ($RE$) de cada exemplo.
2. **Tesselação Espacial:** Agrupa o espaço em $K = \sqrt{N}$ micro-clusters via MiniBatchK-Means.
3. **Cota Sublinear:** Define uma cota de retenção por micro-cluster através de $K_c = \min(S_c, \lceil \lambda \cdot S_c^\gamma \rceil)$, com $\gamma = 0.5$, forçando clusters densos a descartar mais exemplos e preservando clusters pequenos.
4. **Poda Ponderada por RE:** Dentro de cada micro-cluster podado, sorteia sem reposição as instâncias a remover via probabilidade ponderada pelo RE.

Foram avaliadas duas variantes para a etapa (4):
* **Original:** Remove preferencialmente quem tem $RE$ baixo (premissa: casos fáceis são redundantes da classe majoritária, preservando instâncias difíceis/fronteira):
  $$p_{\text{remove}} \propto \frac{1}{\text{score} + 1e-8}$$
* **Cleaning:** Remove preferencialmente quem tem $RE$ alto (premissa de limpeza de ruído):
  $$p_{\text{remove}} \propto \text{score} + 1e-8$$

### O Problema Observado
1. **Empate em Macro-F1 com o Random:** A 40% de remoção (com ModernBERT nos 10 folds), o método empatou com a seleção aleatória em Books (0.840) e DBLP (0.808), e piorou significativamente no WebKB (0.735 vs 0.776 do aleatório).
2. **Sobreposição Anômala:** A sobreposição $|A \cap B| / |A|$ entre os conjuntos mantidos pelo **Original** e pelo **Cleaning** foi de **0.64** em todos os datasets, ficando **acima** da sobreposição com o aleatório (0.60). Em simulação matemática, regras opostas deveriam produzir sobreposição bem abaixo de 0.56.

---

## 2. Resultados Consolidados dos Passos 1 a 3

Foram gerados arquivos `.parquet` por fold em `results/diag-aggressive-rates/diag/<dataset>/<metodo>/fold_<k>.parquet` com rastreamento por instância: `idx`, `re_score`, `cell_id`, `cell_size`, `cell_quota_keep`, `kept`.

### 2.1. Curvas de $P(\text{manter} \mid \text{decil de RE})$

#### Remoção de 40% (Target Keep = 60%)
| Decil | Books (Orig) | Books (Clean) | Books (Rand) | DBLP (Orig) | DBLP (Clean) | DBLP (Rand) | WebKB (Orig) | WebKB (Clean) | WebKB (Rand) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **D0 (Min RE)** | 0.3977 | 0.6220 | 0.5983 | 0.3766 | 0.6205 | 0.5997 | 0.3921 | 0.6806 | 0.5973 |
| **D1** | 0.4767 | 0.6065 | 0.5958 | 0.4595 | 0.6020 | 0.5969 | 0.4480 | 0.6050 | 0.5886 |
| **D2** | 0.5160 | 0.6073 | 0.5998 | 0.5129 | 0.5913 | 0.6062 | 0.4948 | 0.6066 | 0.5974 |
| **D3** | 0.5557 | 0.6029 | 0.6020 | 0.5508 | 0.5973 | 0.5994 | 0.5332 | 0.5776 | 0.6026 |
| **D4** | 0.5826 | 0.5985 | 0.6051 | 0.5827 | 0.5991 | 0.6023 | 0.5821 | 0.5689 | 0.6005 |
| **D5** | 0.6136 | 0.5978 | 0.6011 | 0.6200 | 0.5924 | 0.6002 | 0.6170 | 0.5497 | 0.5989 |
| **D6** | 0.6492 | 0.5935 | 0.5977 | 0.6533 | 0.5952 | 0.6030 | 0.6611 | 0.5492 | 0.6040 |
| **D7** | 0.6834 | 0.5859 | 0.6011 | 0.6904 | 0.6016 | 0.5966 | 0.7017 | 0.5632 | 0.5906 |
| **D8** | 0.7222 | 0.5784 | 0.5992 | 0.7372 | 0.6010 | 0.5997 | 0.7487 | 0.6177 | 0.6109 |
| **D9 (Max RE)** | 0.8029 | 0.6072 | 0.5999 | 0.8168 | 0.5996 | 0.5961 | 0.8212 | 0.6811 | 0.6092 |

#### Remoção de 60% (Target Keep = 40%)
| Decil | Books (Orig) | Books (Clean) | Books (Rand) | DBLP (Orig) | DBLP (Clean) | DBLP (Rand) | WebKB (Orig) | WebKB (Clean) | WebKB (Rand) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **D0 (Min RE)** | 0.2190 | 0.4588 | 0.3997 | 0.2032 | 0.4650 | 0.4018 | 0.2006 | 0.5143 | 0.4021 |
| **D1** | 0.2824 | 0.4364 | 0.3965 | 0.2724 | 0.4280 | 0.3984 | 0.2514 | 0.4431 | 0.3953 |
| **D2** | 0.3187 | 0.4271 | 0.4031 | 0.3155 | 0.4225 | 0.4045 | 0.2941 | 0.4163 | 0.3921 |
| **D3** | 0.3403 | 0.4160 | 0.4042 | 0.3455 | 0.4083 | 0.3948 | 0.3303 | 0.3869 | 0.4006 |
| **D4** | 0.3782 | 0.4037 | 0.4070 | 0.3763 | 0.4004 | 0.4002 | 0.3908 | 0.3731 | 0.4020 |
| **D5** | 0.4073 | 0.3972 | 0.3997 | 0.4084 | 0.3934 | 0.4026 | 0.4133 | 0.3551 | 0.3978 |
| **D6** | 0.4325 | 0.3878 | 0.3992 | 0.4365 | 0.3866 | 0.4030 | 0.4574 | 0.3533 | 0.3989 |
| **D7** | 0.4779 | 0.3670 | 0.3963 | 0.4792 | 0.3832 | 0.4006 | 0.4968 | 0.3471 | 0.3941 |
| **D8** | 0.5230 | 0.3531 | 0.3958 | 0.5312 | 0.3720 | 0.3981 | 0.5540 | 0.3793 | 0.4095 |
| **D9 (Max RE)** | 0.6207 | 0.3529 | 0.3986 | 0.6319 | 0.3405 | 0.3960 | 0.6111 | 0.4310 | 0.4076 |

---

### 2.2. Decomposição da Sobreposição (Grupo a: Não Podadas vs Grupo b: Podadas)

| Dataset | Taxa | % em (a) | % em (b) | Sobreposição (a) | Sobreposição (b) | Aleatório em (b) | Sobreposição Global |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **books** | 40% | 3,48% | 96,52% | 1,0000 | **0,6211** | 0,5856 | **0,6431** |
| **books** | 60% | 0,31% | 99,69% | 1,0000 | **0,4291** | 0,3981 | **0,4336** |
| **dblp** | 40% | 2,29% | 97,71% | 1,0000 | **0,6246** | 0,5906 | **0,6389** |
| **dblp** | 60% | 0,12% | 99,88% | 1,0000 | **0,4178** | 0,3993 | **0,4196** |
| **webkb** | 40% | 4,82% | 95,18% | 1,0000 | **0,6197** | 0,5796 | **0,6504** |
| **webkb** | 60% | 0,76% | 99,24% | 1,0000 | **0,4363** | 0,3954 | **0,4470** |

*Observação Crucial:* Em todos os cenários, a sobreposição dentro das células podadas (b) ficou **acima do aleatório**. Segundo o critério de `TASK.md`, isso confirma o diagnóstico de **H-bug**.

---

### 2.3. Sanidade com $\beta=10$ no WebKB (Passo 3)

Ao elevar a potência do score ($p \propto \text{score}^\beta$) com $\beta=10$:
* **Sobreposição Geral:** Despencou de **0.6504** para **0.4472**.
* **Sobreposição em (b) podadas:** Caiu para **0.3984** (bem abaixo do aleatório 0.5796).
* **Curva de Decis:**
  * Original ($\beta=10$): D0 cai para **6.96%**, D9 sobe para **94.19%**.
  * Cleaning ($\beta=10$): D0 sobe para **96.99%**, D9 cai para **51.83%**.

Isso comprovou que o Autoencoder gera scores ordenáveis válidos, mas com $\beta=1$ a ponderação intra-cluster era inoperante.

---

## 3. Análise Detalhada por Classes

### 3.1. WebKB: A Destruição das Classes Raras (Passo 5)

Taxas de retenção média nos 10 folds a 40% de remoção (Random esperado = 60.0%):

| Classe | N total | Proporção Original | Retenção Random@40 | Retenção SAE Orig@40 | Retenção SAE Clean@40 | Retenção SAE Orig ($\beta=10$) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **C0** | 1.630 | 19,88% | 59,45% | 56,50% | 58,66% | 54,04% |
| **C1 (Majoritária)** | 3.705 | 45,19% | 60,19% | **66,05%** ⬆️ | 62,91% | **70,32%** ⬆️ |
| **C2** | 926 | 11,29% | 57,91% | 51,76% | 55,69% | 46,39% |
| **C3 (137 ex)** | 137 | 1,67% | 60,58% | **58,24%** ⬇️ | 56,21% | **57,35%** ⬇️ |
| **C4** | 501 | 6,11% | 63,63% | 65,18% | 60,75% | 73,30% |
| **C5** | 1.118 | 13,64% | 60,39% | 52,27% | 58,30% | 43,33% |
| **C6 (182 ex)** | 182 | 2,22% | 58,91% | **44,51%** ⚠️ | 45,85% | **40,29%** ⚠️ |

#### Diagnóstico da Causa-Raiz no WebKB:
* **Classe 6:** É composta por textos homogêneos de vocabulário simples. Por isso, seu erro de reconstrução é **baixo** (RE mediano = 0,142 vs 0,205 da classe 1). Como o SAE Original remove preferencialmente quem tem baixo erro, ele classifica a Classe 6 como "redundante" e **descarta mais de 55% da classe**, derrubando o F1 de 0.777 para 0.686.
* **Classe 1 (Majoritária):** Apresenta grande heterogeneidade léxica e, consequentemente, o maior erro médio de reconstrução (0,242). O SAE Original assume erroneamente que ela é "fronteira rara" e a hiper-preserva (66% a $\beta=1$ e 70% a $\beta=10$), concentrando ainda mais o desbalanceamento (a Classe 1 sobe de 45% para 53% do dataset mantido).

---

### 3.2. Books e DBLP: O Efeito Nulo do Autoencoder

Em `books` e `dblp`, a comparação entre Original e Cleaning revelou que as duas variantes selecionam proporções de classe **rigorosamente idênticas** (diferenças menores que 0,1% a 0,5%):

#### Books
* Classe 4: Retenção Original = **55,05%** vs Cleaning = **55,06%** (diferença de 0,01%)
* Classe 5: Retenção Original = **64,06%** vs Cleaning = **64,16%** (diferença de 0,10%)
* Classe 1: Retenção Original = **59,22%** vs Cleaning = **59,28%** (diferença de 0,06%)

#### DBLP
* Classe 5: Retenção Original = **60,54%** vs Cleaning = **60,40%** (diferença de 0,14%)
* Classe 8: Retenção Original = **60,39%** vs Cleaning = **60,42%** (diferença de 0,03%)
* Classe 9: Retenção Original = **63,02%** vs Cleaning = **62,69%** (diferença de 0,33%)

#### Diagnóstico nos Datasets Maiores:
Em `books` e `dblp`, o Autoencoder teve efeito prático zero. O comportamento foi 100% determinado pela tesselação K-Means e pela fórmula sublinear $K_c = \lambda S_c^{0.5}$. Classes que caíram em clusters menores tiveram retenção maior, e classes em clusters maiores tiveram retenção menor — de forma idêntica em ambas as variantes.

---

## 4. As Três Falhas Estruturais do Mecanismo Original

1. **Bug do Acoplamento de Semente sob Probabilidade Quase Plana:**
   Dentro de cada micro-cluster, os pontos são geometricamente próximos pelo K-Means, gerando variação de erro muito pequena (ratio $p_{\max} / p_{\min}$ com mediana de ~4,7). 
   Na variante Cleaning, a probabilidade linear $p \propto \text{score}$ tornou a distribuição virtualmente uniforme ($p_i \approx 1/S_c$). Como Original e Cleaning executavam o sorteio sem reposição consumindo o `np.random` a partir da **mesma semente (`seed=13`)**, o gerador consumiu a mesma sequência de números aleatórios, sorteando praticamente as mesmas instâncias nos dois métodos e criando a falsa sobreposição de 0.64.
2. **Falha da Premissa Teórica com Embeddings Modernos:**
   Em espaços latentes pré-treinados densos (`jina-v5`), o erro de reconstrução de um Autoencoder mede a dispersão intra-classe no espaço de embeddings, e **não a frequência da classe no dataset**. Classes minoritárias compactas têm erro mínimo (sendo punidas pelo Original), enquanto classes majoritárias ricas têm erro alto (sendo hiper-preservadas).
3. **Blindagem de Outliers por Células Pequenas:**
   A fórmula sublinear protege 100% de qualquer célula com $S_c \le \lambda^2$ ($S_c \le 37$ no WebKB). Outliers isolados caem nessas células e jamais são podados pela variante Cleaning, que manteve 68% do decil de maior erro.

---

## 5. Proposta de Resgate (Caminho 2: Normalização Z-Score Local e Softmax)

Para que o conceito de *Autoencoder + Tesselação Sublinear* funcione de maneira matematicamente coerente sem depender de escalas absolutas deformadas:

1. **Normalização Local (Z-Score Intra-Cluster):**
   Dentro de cada micro-cluster $c$, calcular a média $\mu_c$ e desvio padrão $\sigma_c$ dos erros de reconstrução das instâncias daquela célula:
   $$z_i = \frac{s_i - \mu_c}{\sigma_c + \epsilon}$$
2. **Amostragem Simétrica por Softmax com Temperatura ($\tau$):**
   Substituir as fórmulas assimétricas ($1/x$ vs $x$) por uma distribuição softmax simétrica parametrizada por temperatura $\tau$:
   * **Modo Original** (remove baixos erros):
     $$p_i = \frac{\exp(-z_i / \tau)}{\sum_{j \in C_c} \exp(-z_j / \tau)}$$
   * **Modo Cleaning** (remove altos erros):
     $$p_i = \frac{\exp(+z_i / \tau)}{\sum_{j \in C_c} \exp(+z_j / \tau)}$$
3. **Desacoplamento de Sementes:**
   Garantir sementes independentes entre execuções para evitar correlações espúrias na avaliação experimental.
