# Curva de Orçamento em Representação TF-IDF

Comparativo entre Random, Cluster-Random, Typ-Cover e Diffusion-Cover utilizando embeddings esparsos TF-IDF.

## Macro-F1 Médio (Proxy: Regressão Logística em TF-IDF)

| dataset | rate | cluster-random-g0.5 | diffusion-cover | random | typ-cover-g0.5 |
| --- | --- | --- | --- | --- | --- |
| 20ng | 50 | 0.8332 | 0.7953 | 0.8343 | 0.8068 |
| 20ng | 80 | 0.7718 | 0.6918 | 0.7818 | 0.7047 |
| 20ng | 90 | 0.6756 | 0.5868 | 0.7148 | 0.5550 |
| 20ng | 95 | 0.5039 | 0.4802 | 0.5835 | 0.4887 |
| 20ng | 98 | 0.2210 | 0.2242 | 0.2780 | 0.2226 |
| 20ng | 99 | 0.1013 | 0.1073 | 0.1079 | 0.1142 |
| acm | 50 | 0.5431 | 0.5360 | 0.5640 | 0.5342 |
| acm | 80 | 0.4653 | 0.4442 | 0.4904 | 0.4433 |
| acm | 90 | 0.3918 | 0.3592 | 0.3903 | 0.3509 |
| acm | 95 | 0.2874 | 0.2690 | 0.2611 | 0.2528 |
| acm | 98 | 0.1248 | 0.1249 | 0.1240 | 0.1294 |
| acm | 99 | 0.0726 | 0.0717 | 0.0685 | 0.0769 |
| books | 50 | 0.7831 | 0.7667 | 0.7868 | 0.7698 |
| books | 80 | 0.7396 | 0.6759 | 0.7419 | 0.6420 |
| books | 90 | 0.6879 | 0.5818 | 0.6865 | 0.5334 |
| books | 95 | 0.6025 | 0.4667 | 0.6041 | 0.4440 |
| books | 98 | 0.4297 | 0.3921 | 0.4674 | 0.4012 |
| books | 99 | 0.2592 | 0.2625 | 0.3230 | 0.2708 |
| dblp | 50 | 0.7530 | 0.7364 | 0.7577 | 0.7363 |
| dblp | 80 | 0.6966 | 0.6489 | 0.6966 | 0.6531 |
| dblp | 90 | 0.6145 | 0.5658 | 0.6144 | 0.5732 |
| dblp | 95 | 0.5112 | 0.4753 | 0.5069 | 0.4696 |
| dblp | 98 | 0.3248 | 0.3057 | 0.3206 | 0.2990 |
| dblp | 99 | 0.1321 | 0.1434 | 0.1521 | 0.1529 |
| sst1 | 50 | 0.3101 | 0.3184 | 0.3157 | 0.3211 |
| sst1 | 80 | 0.2486 | 0.2794 | 0.2514 | 0.2787 |
| sst1 | 90 | 0.2139 | 0.2487 | 0.2177 | 0.2375 |
| sst1 | 95 | 0.1883 | 0.1898 | 0.1837 | 0.1978 |
| sst1 | 98 | 0.1470 | 0.1422 | 0.1436 | 0.1424 |
| sst1 | 99 | 0.1287 | 0.1290 | 0.1323 | 0.1277 |
| webkb | 50 | 0.5593 | 0.5530 | 0.5726 | 0.5559 |
| webkb | 80 | 0.4327 | 0.4396 | 0.4256 | 0.4356 |
| webkb | 90 | 0.3475 | 0.3368 | 0.3306 | 0.3421 |
| webkb | 95 | 0.2737 | 0.2532 | 0.2346 | 0.2502 |
| webkb | 98 | 0.1441 | 0.1534 | 0.1190 | 0.1451 |
| webkb | 99 | 0.1084 | 0.1180 | 0.0973 | 0.1208 |

## Testes de Significância Estatística vs Random (p-valor, t-pareado)

| dataset | rate | method | mean_diff | p_value | sig_0.05 |
| --- | --- | --- | --- | --- | --- |
| 20ng | 50 | cluster-random-g0.5 | -0.0011 | 0.6821 | não |
| 20ng | 50 | diffusion-cover | -0.0390 | 0.0000 | PIOR (*) |
| 20ng | 50 | typ-cover-g0.5 | -0.0274 | 0.0000 | PIOR (*) |
| 20ng | 80 | cluster-random-g0.5 | -0.0100 | 0.0064 | PIOR (*) |
| 20ng | 80 | diffusion-cover | -0.0899 | 0.0000 | PIOR (*) |
| 20ng | 80 | typ-cover-g0.5 | -0.0771 | 0.0000 | PIOR (*) |
| 20ng | 90 | cluster-random-g0.5 | -0.0392 | 0.0001 | PIOR (*) |
| 20ng | 90 | diffusion-cover | -0.1281 | 0.0000 | PIOR (*) |
| 20ng | 90 | typ-cover-g0.5 | -0.1598 | 0.0000 | PIOR (*) |
| 20ng | 95 | cluster-random-g0.5 | -0.0796 | 0.0004 | PIOR (*) |
| 20ng | 95 | diffusion-cover | -0.1033 | 0.0001 | PIOR (*) |
| 20ng | 95 | typ-cover-g0.5 | -0.0948 | 0.0034 | PIOR (*) |
| 20ng | 98 | cluster-random-g0.5 | -0.0570 | 0.0277 | PIOR (*) |
| 20ng | 98 | diffusion-cover | -0.0538 | 0.0486 | PIOR (*) |
| 20ng | 98 | typ-cover-g0.5 | -0.0553 | 0.0519 | não |
| 20ng | 99 | cluster-random-g0.5 | -0.0066 | 0.6176 | não |
| 20ng | 99 | diffusion-cover | -0.0006 | 0.9658 | não |
| 20ng | 99 | typ-cover-g0.5 | 0.0062 | 0.6855 | não |
| acm | 50 | cluster-random-g0.5 | -0.0209 | 0.0000 | PIOR (*) |
| acm | 50 | diffusion-cover | -0.0280 | 0.0000 | PIOR (*) |
| acm | 50 | typ-cover-g0.5 | -0.0298 | 0.0000 | PIOR (*) |
| acm | 80 | cluster-random-g0.5 | -0.0251 | 0.0004 | PIOR (*) |
| acm | 80 | diffusion-cover | -0.0461 | 0.0000 | PIOR (*) |
| acm | 80 | typ-cover-g0.5 | -0.0471 | 0.0001 | PIOR (*) |
| acm | 90 | cluster-random-g0.5 | 0.0015 | 0.7658 | não |
| acm | 90 | diffusion-cover | -0.0312 | 0.0016 | PIOR (*) |
| acm | 90 | typ-cover-g0.5 | -0.0394 | 0.0001 | PIOR (*) |
| acm | 95 | cluster-random-g0.5 | 0.0263 | 0.0570 | não |
| acm | 95 | diffusion-cover | 0.0079 | 0.5019 | não |
| acm | 95 | typ-cover-g0.5 | -0.0083 | 0.4045 | não |
| acm | 98 | cluster-random-g0.5 | 0.0008 | 0.9537 | não |
| acm | 98 | diffusion-cover | 0.0009 | 0.9488 | não |
| acm | 98 | typ-cover-g0.5 | 0.0054 | 0.6973 | não |
| acm | 99 | cluster-random-g0.5 | 0.0042 | 0.7209 | não |
| acm | 99 | diffusion-cover | 0.0032 | 0.7439 | não |
| acm | 99 | typ-cover-g0.5 | 0.0084 | 0.4493 | não |
| books | 50 | cluster-random-g0.5 | -0.0037 | 0.0993 | não |
| books | 50 | diffusion-cover | -0.0202 | 0.0000 | PIOR (*) |
| books | 50 | typ-cover-g0.5 | -0.0170 | 0.0000 | PIOR (*) |
| books | 80 | cluster-random-g0.5 | -0.0023 | 0.5651 | não |
| books | 80 | diffusion-cover | -0.0661 | 0.0000 | PIOR (*) |
| books | 80 | typ-cover-g0.5 | -0.0999 | 0.0001 | PIOR (*) |
| books | 90 | cluster-random-g0.5 | 0.0014 | 0.8344 | não |
| books | 90 | diffusion-cover | -0.1047 | 0.0000 | PIOR (*) |
| books | 90 | typ-cover-g0.5 | -0.1531 | 0.0001 | PIOR (*) |
| books | 95 | cluster-random-g0.5 | -0.0016 | 0.9201 | não |
| books | 95 | diffusion-cover | -0.1374 | 0.0000 | PIOR (*) |
| books | 95 | typ-cover-g0.5 | -0.1601 | 0.0000 | PIOR (*) |
| books | 98 | cluster-random-g0.5 | -0.0377 | 0.1360 | não |
| books | 98 | diffusion-cover | -0.0753 | 0.0107 | PIOR (*) |
| books | 98 | typ-cover-g0.5 | -0.0663 | 0.0383 | PIOR (*) |
| books | 99 | cluster-random-g0.5 | -0.0638 | 0.0480 | PIOR (*) |
| books | 99 | diffusion-cover | -0.0606 | 0.0454 | PIOR (*) |
| books | 99 | typ-cover-g0.5 | -0.0522 | 0.1015 | não |
| dblp | 50 | cluster-random-g0.5 | -0.0047 | 0.0159 | PIOR (*) |
| dblp | 50 | diffusion-cover | -0.0213 | 0.0000 | PIOR (*) |
| dblp | 50 | typ-cover-g0.5 | -0.0213 | 0.0000 | PIOR (*) |
| dblp | 80 | cluster-random-g0.5 | 0.0001 | 0.9864 | não |
| dblp | 80 | diffusion-cover | -0.0477 | 0.0000 | PIOR (*) |
| dblp | 80 | typ-cover-g0.5 | -0.0435 | 0.0000 | PIOR (*) |
| dblp | 90 | cluster-random-g0.5 | 0.0001 | 0.9803 | não |
| dblp | 90 | diffusion-cover | -0.0485 | 0.0000 | PIOR (*) |
| dblp | 90 | typ-cover-g0.5 | -0.0411 | 0.0001 | PIOR (*) |
| dblp | 95 | cluster-random-g0.5 | 0.0042 | 0.6400 | não |
| dblp | 95 | diffusion-cover | -0.0316 | 0.0058 | PIOR (*) |
| dblp | 95 | typ-cover-g0.5 | -0.0373 | 0.0019 | PIOR (*) |
| dblp | 98 | cluster-random-g0.5 | 0.0042 | 0.7262 | não |
| dblp | 98 | diffusion-cover | -0.0149 | 0.1858 | não |
| dblp | 98 | typ-cover-g0.5 | -0.0217 | 0.0355 | PIOR (*) |
| dblp | 99 | cluster-random-g0.5 | -0.0201 | 0.2873 | não |
| dblp | 99 | diffusion-cover | -0.0087 | 0.7126 | não |
| dblp | 99 | typ-cover-g0.5 | 0.0008 | 0.9681 | não |
| sst1 | 50 | cluster-random-g0.5 | -0.0056 | 0.3013 | não |
| sst1 | 50 | diffusion-cover | 0.0027 | 0.3685 | não |
| sst1 | 50 | typ-cover-g0.5 | 0.0054 | 0.2699 | não |
| sst1 | 80 | cluster-random-g0.5 | -0.0027 | 0.5774 | não |
| sst1 | 80 | diffusion-cover | 0.0280 | 0.0014 | SIM (*) |
| sst1 | 80 | typ-cover-g0.5 | 0.0274 | 0.0015 | SIM (*) |
| sst1 | 90 | cluster-random-g0.5 | -0.0039 | 0.5376 | não |
| sst1 | 90 | diffusion-cover | 0.0309 | 0.0012 | SIM (*) |
| sst1 | 90 | typ-cover-g0.5 | 0.0197 | 0.0226 | SIM (*) |
| sst1 | 95 | cluster-random-g0.5 | 0.0045 | 0.3638 | não |
| sst1 | 95 | diffusion-cover | 0.0061 | 0.2705 | não |
| sst1 | 95 | typ-cover-g0.5 | 0.0141 | 0.0293 | SIM (*) |
| sst1 | 98 | cluster-random-g0.5 | 0.0035 | 0.8301 | não |
| sst1 | 98 | diffusion-cover | -0.0013 | 0.9368 | não |
| sst1 | 98 | typ-cover-g0.5 | -0.0012 | 0.9401 | não |
| sst1 | 99 | cluster-random-g0.5 | -0.0036 | 0.7688 | não |
| sst1 | 99 | diffusion-cover | -0.0033 | 0.7490 | não |
| sst1 | 99 | typ-cover-g0.5 | -0.0046 | 0.6343 | não |
| webkb | 50 | cluster-random-g0.5 | -0.0132 | 0.0596 | não |
| webkb | 50 | diffusion-cover | -0.0196 | 0.0130 | PIOR (*) |
| webkb | 50 | typ-cover-g0.5 | -0.0166 | 0.0581 | não |
| webkb | 80 | cluster-random-g0.5 | 0.0070 | 0.5876 | não |
| webkb | 80 | diffusion-cover | 0.0139 | 0.1114 | não |
| webkb | 80 | typ-cover-g0.5 | 0.0099 | 0.2999 | não |
| webkb | 90 | cluster-random-g0.5 | 0.0169 | 0.0039 | SIM (*) |
| webkb | 90 | diffusion-cover | 0.0062 | 0.4592 | não |
| webkb | 90 | typ-cover-g0.5 | 0.0115 | 0.1154 | não |
| webkb | 95 | cluster-random-g0.5 | 0.0391 | 0.0205 | SIM (*) |
| webkb | 95 | diffusion-cover | 0.0185 | 0.1979 | não |
| webkb | 95 | typ-cover-g0.5 | 0.0156 | 0.2356 | não |
| webkb | 98 | cluster-random-g0.5 | 0.0251 | 0.0053 | SIM (*) |
| webkb | 98 | diffusion-cover | 0.0345 | 0.0305 | SIM (*) |
| webkb | 98 | typ-cover-g0.5 | 0.0261 | 0.0884 | não |
| webkb | 99 | cluster-random-g0.5 | 0.0111 | 0.1832 | não |
| webkb | 99 | diffusion-cover | 0.0207 | 0.1432 | não |
| webkb | 99 | typ-cover-g0.5 | 0.0235 | 0.1374 | não |
