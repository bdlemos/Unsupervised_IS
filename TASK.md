# Tarefa: diagnosticar por que o erro de reconstrução não altera a seleção do SAE-IS

## Contexto

O repositório é `/workspace/Unsupervised_IS`. O método **SAE-IS** (`SublinearAEIS`) seleciona
instâncias em quatro etapas: (1) treina um autoencoder sobre os embeddings `jina-v5` e calcula o
erro de reconstrução (RE) de cada instância; (2) tessela o espaço em K=√N micro-clusters;
(3) define uma cota sublinear por célula (γ=0.5); (4) dentro de cada célula, sorteia o que
remover, com probabilidade ponderada pelo RE.

Existem duas variantes da etapa (4), que diferem só na direção do peso:

| Variante | Regra | Resultados em |
|---|---|---|
| **original** | remove com mais probabilidade quem tem RE **baixo** (mantém os casos difíceis) | `results/exp-aggressive-rates/rate_40/` |
| **cleaning** | remove com mais probabilidade quem tem RE **alto**: `p_remove = (c_scores + 1e-8) / np.sum(c_scores + 1e-8)` | `results/exp-aggressive-rates-cleaning/rate_{40,60}/` |

Há também um controle `random-rate-40` em `results/exp-aggressive-rates/rate_40/`. Os splits são
os mesmos em todas as variantes (`datasets/<ds>/splits/split_10.pkl`, 10 folds).

## O que foi observado

**1. Em Macro-F1, as duas variantes empatam com o aleatório.** Remoção de 40%, ModernBERT,
média de 10 folds:

| | books | dblp | webkb |
|---|---|---|---|
| Random@40 | 0.8404 | 0.8080 | 0.7758 |
| SAE@40 original | 0.8407 | 0.8085 | 0.7349 |
| SAE@40 cleaning | 0.8392 | 0.8070 | 0.7616 |

**2. As duas variantes escolhem quase o mesmo conjunto.** Medi a sobreposição dos conjuntos
mantidos (`train_idxs` dos `split_10_*.pkl`), como |A∩B|/|A|, em média por fold:

| Par | Sobreposição |
|---|---|
| original × random | 0.600 |
| cleaning × random | 0.600 |
| original × cleaning | **0.64**, nos três datasets |

**3. Uma simulação diz que isso não deveria acontecer.** Simulei a regra com scores lognormais
calibrados pelo log do webkb (RE mín 0.03, mediana 0.166, máx 1.0), uma única célula, remoção de
40%, sorteio sem reposição e `p ∝ score^β`:

| β | Sobreposição original×cleaning esperada | P(manter \| decil de RE mais baixo): original / cleaning |
|---|---|---|
| 1 | 0.56 | 0.29 / 0.82 |
| 3 | 0.44 | 0.01 / 0.97 |
| 10 | 0.34 | 0.00 / 1.00 |

Duas regras opostas deveriam ter sobreposição **abaixo** de 0.60. A medida real, 0.64, está
**acima**, e isso indica que alguma coisa na implementação anula o efeito do RE.

## Hipóteses, da mais à menos provável

- **H-cota:** as cotas sublineares mantêm células pequenas inteiras nas duas variantes. Isso
  infla a sobreposição, e nessas células o RE não tem efeito nenhum.
- **H-margem:** nas células podadas, a cota deixa pouca liberdade de escolha (remove quase tudo
  ou quase nada da célula), e o peso do RE muda pouco o resultado.
- **H-bug:** a inversão de direção não faz o que deveria. Por exemplo, a original não usa de fato
  o inverso do score, os scores estão desalinhados dos índices da célula, ou a seed/ordem faz o
  sorteio ignorar `p`.

## Passos

Os passos 1 a 3 **só rodam a seleção** (6 a 30 s por fold), sem treinar o classificador. Faça-os
antes de qualquer coisa que use GPU.

### Passo 1: instrumentar a seleção

1. Localize a implementação: `grep -rn "SublinearAEIS" --include=*.py .`.
2. Leia a etapa (4) e escreva **literalmente** como a variante original calcula `p_remove`.
   Escreva também como a cota de cada célula é calculada.
3. Adicione uma opção (flag ou variável de ambiente, **desligada por padrão**) que salve, por
   fold, `diag/<ds>/<method>/fold_<k>.parquet` com as colunas:
   `idx` (índice global), `re_score`, `cell_id`, `cell_size`, `cell_quota_keep`, `kept` (bool).
4. Rode a seleção com essa opção para `books`, `dblp` e `webkb`, nas variantes original e
   cleaning, com remoção de 40% e 60%, usando a mesma seed nas duas. Grave as saídas num
   diretório **novo**, `results/diag-aggressive-rates/`.

### Passo 2: análise

Para cada combinação de (dataset, taxa), calcule e reporte:

1. **P(manter | decil de RE)** para original, cleaning e random. Faça uma tabela de 10 linhas e,
   se puder, um gráfico.
2. **Decomposição da sobreposição** original×cleaning em dois grupos:
   - (a) células com `cell_quota_keep >= cell_size`, mantidas inteiras;
   - (b) células podadas.

   Reporte a fração de instâncias em (a) e a sobreposição dentro de (b). Para comparar, dentro de
   (b) o aleatório daria a taxa de manutenção da própria célula.
3. **Distribuição das cotas:** um histograma de `cell_quota_keep / cell_size`.

Para ler o resultado:
- Se a sobreposição em (b) ficar **abaixo** do aleatório e as curvas do item 1 forem opostas,
  o RE age, e a explicação é H-cota ou H-margem.
- Se a sobreposição em (b) ficar **perto ou acima** do aleatório, a explicação provável é
  H-bug. Nesse caso, ache o bug e descreva-o, **sem corrigir** o método principal.

### Passo 3: sanidade com β extremo

Adicione `beta` como parâmetro (padrão 1, para não mudar o comportamento atual) e use
`p ∝ score^beta`. Rode a seleção com β=10, nas duas direções, a 40%, no webkb. A sobreposição
esperada é perto de 0.34, se a cota não dominar. Se continuar perto de 0.64, a seleção não está
sendo feita pelo score.

### Passo 4: classificação

**Só faça este passo depois de reportar os passos 1 a 3 e receber confirmação.** Comece pelo
webkb, que é o mais barato e o único dataset com sinal. Rode SAE com β∈{1, 3, 10}, nas duas
direções, contra `random-rate-<taxa>`, a 40% e 60%. Use o mesmo pipeline de classificação
ModernBERT já usado nos experimentos acima.

Antes disso, falta rodar `random-rate-60` (e `sae-rate-60` na direção original) para books, dblp
e webkb: sem esses controles, o `sae-rate-60` cleaning não tem com o que ser comparado. Também
falta a classificação de `acm` e `20ng` a 40%. A seleção já existe, mas o log do acm termina num
`KeyboardInterrupt` durante o `import torch`.

### Passo 5: classes no webkb

Com os labels de `datasets/webkb/`, compare a distribuição de classes do conjunto mantido em
cada variante a 40%. As classes com 137 e 182 exemplos de teste (em 10 folds) são as que mais
perdem F1 na variante original (0.516→0.461 e 0.777→0.686, contra o random). Quero saber se o
conjunto mantido pela original descarta mais exemplos dessas classes.

## Restrições

- **Não sobrescreva** nada em `results/jina-v5/`, `results/exp-aggressive-rates*/` ou
  `experiments/feedback-response/`. Toda saída nova vai em `results/diag-aggressive-rates/`.
- As mudanças no código são aditivas e ficam desligadas por padrão. O comportamento do método
  original não pode mudar.
- Não invente números. Se algo falhar, reporte o erro com o trecho do log.
- Não compare tempos entre rodadas de máquinas diferentes. Os tempos variam até 10pp entre
  execuções da mesma configuração.

## Formato da resposta

1. Trecho de código da etapa (4) e da cota, com arquivo e linha.
2. Tabelas do passo 2 (P(manter | decil), decomposição da sobreposição, fração de células
   inteiras).
3. Resultado do passo 3.
4. Veredito: qual hipótese (H-cota, H-margem ou H-bug) os dados sustentam, e com que evidência.
5. Lista de arquivos alterados ou criados.
