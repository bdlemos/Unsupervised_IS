# Relatório Consolidado: Curva de Orçamento Extremo e Novos Paradigmas

Este relatório compara diretamente todos os métodos avaliados sob o mesmo protocolo experimental rigoroso (10 folds, Proxy validado com embeddings `jina-v5`).

## 1. Tabela Comparativa de Macro-F1 Médio

| dataset | rate | cluster-random-g0.5 | diffusion-cover | random | sae-clean | sae-orig | typ-cover-g0.5 | typ-cover-g1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20ng | 90 | 0.6762 | 0.6689 | 0.6746 | 0.6727 | 0.6704 | 0.6724 | 0.6713 |
| 20ng | 95 | 0.6622 | 0.6595 | 0.6608 | 0.6596 | 0.6523 | 0.6588 | 0.6588 |
| 20ng | 98 | 0.6266 | 0.6338 | 0.6295 | 0.6421 | 0.6236 | 0.6346 | 0.6346 |
| 20ng | 99 | 0.5866 | 0.5815 | 0.5735 | 0.6070 | 0.5814 | 0.6156 | 0.6156 |
| acm | 90 | 0.6262 | 0.6294 | 0.6224 | 0.6243 | 0.6471 | 0.6144 | 0.6159 |
| acm | 95 | 0.5969 | 0.5952 | 0.5894 | 0.5992 | 0.6073 | 0.5896 | 0.5864 |
| acm | 98 | 0.5681 | 0.5753 | 0.5489 | 0.5618 | 0.5476 | 0.5698 | 0.5698 |
| acm | 99 | 0.5366 | 0.5277 | 0.5166 | 0.5387 | 0.5136 | 0.5447 | 0.5447 |
| books | 90 | 0.7787 | 0.7675 | 0.7723 | 0.7725 | 0.7775 | 0.7735 | 0.7723 |
| books | 95 | 0.7651 | 0.7595 | 0.7648 | 0.7627 | 0.7627 | 0.7650 | 0.7645 |
| books | 98 | 0.7496 | 0.7491 | 0.7450 | 0.7489 | 0.7435 | 0.7529 | 0.7529 |
| books | 99 | 0.7303 | 0.7333 | 0.7270 | 0.7355 | 0.7343 | 0.7416 | 0.7416 |
| dblp | 90 | 0.7956 | 0.7893 | 0.7953 | 0.7950 | 0.7950 | 0.7925 | 0.7914 |
| dblp | 95 | 0.7854 | 0.7812 | 0.7846 | 0.7849 | 0.7860 | 0.7856 | 0.7843 |
| dblp | 98 | 0.7709 | 0.7717 | 0.7641 | 0.7743 | 0.7659 | 0.7737 | 0.7737 |
| dblp | 99 | 0.7493 | 0.7519 | 0.7435 | 0.7520 | 0.7416 | 0.7582 | 0.7582 |
| sst1 | 90 | 0.3921 | 0.3912 | 0.3922 | - | - | 0.3835 | 0.3841 |
| sst1 | 95 | 0.3691 | 0.3517 | 0.3693 | - | - | 0.3641 | 0.3641 |
| sst1 | 98 | 0.3142 | 0.3222 | 0.3396 | - | - | 0.3282 | 0.3282 |
| sst1 | 99 | 0.2934 | 0.2855 | 0.2980 | - | - | 0.2887 | 0.2887 |
| webkb | 90 | 0.4956 | 0.4917 | 0.5287 | 0.5075 | 0.5009 | 0.4918 | 0.4918 |
| webkb | 95 | 0.4235 | 0.4104 | 0.4665 | 0.4251 | 0.4133 | 0.4091 | 0.4091 |
| webkb | 98 | 0.3337 | 0.3615 | 0.3808 | 0.3650 | 0.3055 | 0.3523 | 0.3523 |
| webkb | 99 | 0.2682 | 0.3041 | 0.2952 | 0.2965 | 0.1831 | 0.3177 | 0.3177 |

## 2. Testes de Significância Estatística vs Random (p-valor, t-pareado)

| dataset | rate | method | mean_diff | p_value | sig_0.05 |
| --- | --- | --- | --- | --- | --- |
| 20ng | 90 | cluster-random-g0.5 | 0.0016 | 0.6488 | não |
| 20ng | 90 | diffusion-cover | -0.0058 | 0.1257 | não |
| 20ng | 90 | sae-clean | -0.0020 | 0.5351 | não |
| 20ng | 90 | sae-orig | -0.0043 | 0.1336 | não |
| 20ng | 90 | typ-cover-g0.5 | -0.0023 | 0.4861 | não |
| 20ng | 90 | typ-cover-g1 | -0.0033 | 0.4145 | não |
| 20ng | 95 | cluster-random-g0.5 | 0.0014 | 0.6434 | não |
| 20ng | 95 | diffusion-cover | -0.0013 | 0.7843 | não |
| 20ng | 95 | sae-clean | -0.0012 | 0.7128 | não |
| 20ng | 95 | sae-orig | -0.0085 | 0.0059 | PIOR (*) |
| 20ng | 95 | typ-cover-g0.5 | -0.0020 | 0.4524 | não |
| 20ng | 95 | typ-cover-g1 | -0.0020 | 0.4524 | não |
| 20ng | 98 | cluster-random-g0.5 | -0.0029 | 0.7404 | não |
| 20ng | 98 | diffusion-cover | 0.0044 | 0.4758 | não |
| 20ng | 98 | sae-clean | 0.0127 | 0.0819 | não |
| 20ng | 98 | sae-orig | -0.0059 | 0.3194 | não |
| 20ng | 98 | typ-cover-g0.5 | 0.0052 | 0.5275 | não |
| 20ng | 98 | typ-cover-g1 | 0.0052 | 0.5275 | não |
| 20ng | 99 | cluster-random-g0.5 | 0.0131 | 0.3151 | não |
| 20ng | 99 | diffusion-cover | 0.0080 | 0.6249 | não |
| 20ng | 99 | sae-clean | 0.0334 | 0.0101 | SIM (*) |
| 20ng | 99 | sae-orig | 0.0079 | 0.4984 | não |
| 20ng | 99 | typ-cover-g0.5 | 0.0421 | 0.0100 | SIM (*) |
| 20ng | 99 | typ-cover-g1 | 0.0421 | 0.0100 | SIM (*) |
| acm | 90 | cluster-random-g0.5 | 0.0038 | 0.7535 | não |
| acm | 90 | diffusion-cover | 0.0071 | 0.5283 | não |
| acm | 90 | sae-clean | 0.0019 | 0.8840 | não |
| acm | 90 | sae-orig | 0.0247 | 0.0163 | SIM (*) |
| acm | 90 | typ-cover-g0.5 | -0.0080 | 0.4362 | não |
| acm | 90 | typ-cover-g1 | -0.0065 | 0.6311 | não |
| acm | 95 | cluster-random-g0.5 | 0.0075 | 0.2333 | não |
| acm | 95 | diffusion-cover | 0.0058 | 0.3898 | não |
| acm | 95 | sae-clean | 0.0099 | 0.2947 | não |
| acm | 95 | sae-orig | 0.0180 | 0.1751 | não |
| acm | 95 | typ-cover-g0.5 | 0.0002 | 0.9786 | não |
| acm | 95 | typ-cover-g1 | -0.0030 | 0.6849 | não |
| acm | 98 | cluster-random-g0.5 | 0.0192 | 0.1433 | não |
| acm | 98 | diffusion-cover | 0.0264 | 0.0080 | SIM (*) |
| acm | 98 | sae-clean | 0.0129 | 0.0425 | SIM (*) |
| acm | 98 | sae-orig | -0.0013 | 0.8763 | não |
| acm | 98 | typ-cover-g0.5 | 0.0209 | 0.0121 | SIM (*) |
| acm | 98 | typ-cover-g1 | 0.0209 | 0.0121 | SIM (*) |
| acm | 99 | cluster-random-g0.5 | 0.0200 | 0.0030 | SIM (*) |
| acm | 99 | diffusion-cover | 0.0111 | 0.2333 | não |
| acm | 99 | sae-clean | 0.0221 | 0.0276 | SIM (*) |
| acm | 99 | sae-orig | -0.0030 | 0.8130 | não |
| acm | 99 | typ-cover-g0.5 | 0.0281 | 0.0065 | SIM (*) |
| acm | 99 | typ-cover-g1 | 0.0281 | 0.0065 | SIM (*) |
| books | 90 | cluster-random-g0.5 | 0.0064 | 0.0062 | SIM (*) |
| books | 90 | diffusion-cover | -0.0048 | 0.0232 | PIOR (*) |
| books | 90 | sae-clean | 0.0002 | 0.9164 | não |
| books | 90 | sae-orig | 0.0052 | 0.0258 | SIM (*) |
| books | 90 | typ-cover-g0.5 | 0.0012 | 0.5366 | não |
| books | 90 | typ-cover-g1 | 0.0000 | 0.9876 | não |
| books | 95 | cluster-random-g0.5 | 0.0004 | 0.8127 | não |
| books | 95 | diffusion-cover | -0.0053 | 0.0083 | PIOR (*) |
| books | 95 | sae-clean | -0.0021 | 0.2656 | não |
| books | 95 | sae-orig | -0.0021 | 0.3776 | não |
| books | 95 | typ-cover-g0.5 | 0.0002 | 0.8957 | não |
| books | 95 | typ-cover-g1 | -0.0003 | 0.8569 | não |
| books | 98 | cluster-random-g0.5 | 0.0046 | 0.2230 | não |
| books | 98 | diffusion-cover | 0.0041 | 0.1458 | não |
| books | 98 | sae-clean | 0.0039 | 0.2335 | não |
| books | 98 | sae-orig | -0.0015 | 0.6674 | não |
| books | 98 | typ-cover-g0.5 | 0.0079 | 0.0349 | SIM (*) |
| books | 98 | typ-cover-g1 | 0.0079 | 0.0349 | SIM (*) |
| books | 99 | cluster-random-g0.5 | 0.0034 | 0.2982 | não |
| books | 99 | diffusion-cover | 0.0063 | 0.0919 | não |
| books | 99 | sae-clean | 0.0085 | 0.0264 | SIM (*) |
| books | 99 | sae-orig | 0.0073 | 0.0771 | não |
| books | 99 | typ-cover-g0.5 | 0.0146 | 0.0015 | SIM (*) |
| books | 99 | typ-cover-g1 | 0.0146 | 0.0015 | SIM (*) |
| dblp | 90 | cluster-random-g0.5 | 0.0003 | 0.8291 | não |
| dblp | 90 | diffusion-cover | -0.0060 | 0.0129 | PIOR (*) |
| dblp | 90 | sae-clean | -0.0003 | 0.8795 | não |
| dblp | 90 | sae-orig | -0.0003 | 0.8521 | não |
| dblp | 90 | typ-cover-g0.5 | -0.0028 | 0.0993 | não |
| dblp | 90 | typ-cover-g1 | -0.0039 | 0.0556 | não |
| dblp | 95 | cluster-random-g0.5 | 0.0008 | 0.7609 | não |
| dblp | 95 | diffusion-cover | -0.0034 | 0.2573 | não |
| dblp | 95 | sae-clean | 0.0003 | 0.8873 | não |
| dblp | 95 | sae-orig | 0.0014 | 0.5702 | não |
| dblp | 95 | typ-cover-g0.5 | 0.0010 | 0.5679 | não |
| dblp | 95 | typ-cover-g1 | -0.0002 | 0.8891 | não |
| dblp | 98 | cluster-random-g0.5 | 0.0068 | 0.0141 | SIM (*) |
| dblp | 98 | diffusion-cover | 0.0076 | 0.0009 | SIM (*) |
| dblp | 98 | sae-clean | 0.0101 | 0.0054 | SIM (*) |
| dblp | 98 | sae-orig | 0.0017 | 0.5448 | não |
| dblp | 98 | typ-cover-g0.5 | 0.0096 | 0.0058 | SIM (*) |
| dblp | 98 | typ-cover-g1 | 0.0096 | 0.0058 | SIM (*) |
| dblp | 99 | cluster-random-g0.5 | 0.0058 | 0.5037 | não |
| dblp | 99 | diffusion-cover | 0.0085 | 0.1611 | não |
| dblp | 99 | sae-clean | 0.0086 | 0.1924 | não |
| dblp | 99 | sae-orig | -0.0019 | 0.8658 | não |
| dblp | 99 | typ-cover-g0.5 | 0.0147 | 0.0109 | SIM (*) |
| dblp | 99 | typ-cover-g1 | 0.0147 | 0.0109 | SIM (*) |
| sst1 | 90 | cluster-random-g0.5 | -0.0000 | 0.9965 | não |
| sst1 | 90 | diffusion-cover | -0.0009 | 0.9152 | não |
| sst1 | 90 | typ-cover-g0.5 | -0.0086 | 0.2096 | não |
| sst1 | 90 | typ-cover-g1 | -0.0081 | 0.2404 | não |
| sst1 | 95 | cluster-random-g0.5 | -0.0002 | 0.9764 | não |
| sst1 | 95 | diffusion-cover | -0.0176 | 0.0757 | não |
| sst1 | 95 | typ-cover-g0.5 | -0.0052 | 0.5110 | não |
| sst1 | 95 | typ-cover-g1 | -0.0052 | 0.5110 | não |
| sst1 | 98 | cluster-random-g0.5 | -0.0254 | 0.0995 | não |
| sst1 | 98 | diffusion-cover | -0.0174 | 0.0694 | não |
| sst1 | 98 | typ-cover-g0.5 | -0.0114 | 0.1268 | não |
| sst1 | 98 | typ-cover-g1 | -0.0114 | 0.1268 | não |
| sst1 | 99 | cluster-random-g0.5 | -0.0046 | 0.4934 | não |
| sst1 | 99 | diffusion-cover | -0.0124 | 0.3438 | não |
| sst1 | 99 | typ-cover-g0.5 | -0.0092 | 0.3419 | não |
| sst1 | 99 | typ-cover-g1 | -0.0092 | 0.3419 | não |
| webkb | 90 | cluster-random-g0.5 | -0.0331 | 0.0007 | PIOR (*) |
| webkb | 90 | diffusion-cover | -0.0370 | 0.0261 | PIOR (*) |
| webkb | 90 | sae-clean | -0.0213 | 0.2263 | não |
| webkb | 90 | sae-orig | -0.0278 | 0.2288 | não |
| webkb | 90 | typ-cover-g0.5 | -0.0369 | 0.0315 | PIOR (*) |
| webkb | 90 | typ-cover-g1 | -0.0369 | 0.0315 | PIOR (*) |
| webkb | 95 | cluster-random-g0.5 | -0.0430 | 0.0395 | PIOR (*) |
| webkb | 95 | diffusion-cover | -0.0561 | 0.0039 | PIOR (*) |
| webkb | 95 | sae-clean | -0.0414 | 0.0350 | PIOR (*) |
| webkb | 95 | sae-orig | -0.0532 | 0.0301 | PIOR (*) |
| webkb | 95 | typ-cover-g0.5 | -0.0574 | 0.0055 | PIOR (*) |
| webkb | 95 | typ-cover-g1 | -0.0574 | 0.0055 | PIOR (*) |
| webkb | 98 | cluster-random-g0.5 | -0.0471 | 0.0322 | PIOR (*) |
| webkb | 98 | diffusion-cover | -0.0193 | 0.1582 | não |
| webkb | 98 | sae-clean | -0.0159 | 0.4176 | não |
| webkb | 98 | sae-orig | -0.0754 | 0.0010 | PIOR (*) |
| webkb | 98 | typ-cover-g0.5 | -0.0285 | 0.1671 | não |
| webkb | 98 | typ-cover-g1 | -0.0285 | 0.1671 | não |
| webkb | 99 | cluster-random-g0.5 | -0.0270 | 0.1121 | não |
| webkb | 99 | diffusion-cover | 0.0088 | 0.5564 | não |
| webkb | 99 | sae-clean | 0.0012 | 0.8962 | não |
| webkb | 99 | sae-orig | -0.1121 | 0.0001 | PIOR (*) |
| webkb | 99 | typ-cover-g0.5 | 0.0224 | 0.1870 | não |
| webkb | 99 | typ-cover-g1 | 0.0224 | 0.1870 | não |
