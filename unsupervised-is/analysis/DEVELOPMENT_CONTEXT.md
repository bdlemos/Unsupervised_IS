# Evolução do Unsupervised Instance Selection (Pipeline de Proteção de Minorias)

Este documento resume a jornada de desenvolvimento empírico focada em criar um método de Instance Selection (IS) Não-Supervisionado capaz de nivelar o balanceamento de classes e proteger classes minoritárias sem acessar os rótulos (`y`).

## 1. O Ponto de Partida e o Paradoxo da Esparsidade
A literatura clássica de detecção de anomalias e seleção não-supervisionada assume que **"Classes Minoritárias = Regiões Esparsas"** e **"Classes Majoritárias = Regiões Densas"**.
Para testar isso, desenvolvemos o método `DensityAwareIS`, que protegia as instâncias em regiões de baixa densidade (avaliadas via distância k-NN).

**A Descoberta Crítica (Iteração 1):**
Quando testamos em embeddings profundos (Jina V5) no dataset **SST1**, descobrimos que a premissa é **falsa**. Classes minoritárias de sentimento extremo (Classes 0 e 4) eram na verdade **as mais densas** do espaço, pois usavam vocabulários altamente restritos e coesos. Ao proteger as regiões esparsas, o método acabou protegendo a classe Neutra (majoritária) e varrendo as minorias do mapa (Protection Ratio < 1.0).

## 2. A Correção Espacial (Iteração 2: BoundaryIS)
Para corrigir o viés da densidade, abandonamos a medição de densidade absoluta. Em vez disso, desenvolvemos o `BoundaryIS`:
1. **Tesselação:** Dividimos o espaço em centenas de micro-clusters ($K=200$) usando K-Means.
2. **Amostragem Sublinear:** Adotamos uma função sublinear ($S_c^{0.5}$) para ditar o limite de retenção de cada micro-cluster.
Isso nivelou a representatividade geográfica, achatando regiões massivamente populosas (maiorias) e mantendo intactas as micro-regiões coesas (minorias densas).

**Resultado:** O `BoundaryIS` atingiu Protection Ratios excelentes no *Ohsumed* (1.66) e consertou o *SST1* (0.99). Porém, revelou uma nova fraqueza: instâncias ultra-minoritárias (2 a 5 casos, como no *Reuters90*) que, por azar, caíssem perto do centro geométrico de um grande cluster majoritário, eram apagadas inadvertidamente.

## 3. A Solução Definitiva (Iteração 3: SublinearAEIS)
Para proteger as "agulhas no palheiro" (long-tail extremo do Reuters/ACM), casamos a elegante Amostragem Sublinear do `BoundaryIS` com o Score Latente do Autoencoder.

- **O "Quanto" Remover (Sublinear K-Means):** A amostragem sublinear continua punindo clusters superpovoados e protegendo pequenos.
- **"Quem" Remover (Autoencoder Scoring):** Em vez de remover as instâncias pelo centro geométrico, usamos o **Erro de Reconstrução do Autoencoder**.
Como o Autoencoder aprende a estrutura majoritária do dataset, uma instância minoritária isolada (mesmo que imersa geograficamente em um cluster gigante) terá um erro de reconstrução altíssimo e gritará como uma "anomalia semântica". O algoritmo deleta os erros baixos (redundância) e salva os erros altos (fronteiras e minorias raras).

**Resultados do SublinearAEIS (Redução de ~35%):**
- **ACM:** Protection Ratio de **1.5494** (Fenomenal)
- **Ohsumed:** Protection Ratio de **1.5363** (Massivo ganho frente aos 1.17 do baseline antigo)
- **Twitter:** Protection Ratio de **1.3405** (Excelente)
- **SST1 / Reuters90:** Melhorias cirúrgicas na retenção de dezenas de classes do long-tail (salvando pontualmente 100% de classes pequenas que métodos anteriores dizimavam).

## 4. O Caminho para Publicação
Essa narrativa compõe um paper robusto (ex: *"Beyond Sparsity: Sublinear Spatial Autoencoding for Unsupervised Instance Selection"*). 
O grande valor está em provar o viés da densidade escondida nas representações de texto denso moderno (Jina/BERT) e superá-lo integrando geometria (Voronoi/K-Means Sublinear) com aprendizado de representação profundo (Autoencoder).
