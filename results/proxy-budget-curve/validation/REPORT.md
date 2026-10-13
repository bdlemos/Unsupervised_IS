# Validação do proxy (Regressão Logística) vs ModernBERT

## Médias por configuração (só folds com resultado do ModernBERT)

| dataset | rate | method | proxy_macro | bert_macro | n_folds |
| --- | --- | --- | --- | --- | --- |
| acm | 40 | sae-rate | 0.6948 | 0.6667 | 10 |
| acm | 40 | sae-softmax-rate | 0.6699 | 0.6607 | 3 |
| acm | 60 | random-rate | 0.6826 | 0.6486 | 10 |
| acm | 60 | sae-rate | 0.6822 | 0.6494 | 10 |
| acm | 60 | sae-softmax-rate | 0.6663 | 0.6443 | 10 |
| books | 40 | random-rate | 0.7966 | 0.8404 | 10 |
| books | 40 | sae-rate | 0.7965 | 0.8407 | 10 |
| books | 40 | sae-softmax-rate | 0.7956 | 0.8332 | 10 |
| books | 60 | random-rate | 0.7929 | 0.8176 | 10 |
| books | 60 | sae-rate | 0.7926 | 0.8200 | 10 |
| books | 60 | sae-softmax-rate | 0.7919 | 0.8158 | 10 |
| dblp | 40 | random-rate | 0.8121 | 0.8080 | 10 |
| dblp | 40 | sae-rate | 0.8134 | 0.8085 | 10 |
| dblp | 40 | sae-softmax-rate | 0.8114 | 0.8083 | 10 |
| dblp | 60 | random-rate | 0.8086 | 0.7996 | 10 |
| dblp | 60 | sae-rate | 0.8089 | 0.8012 | 10 |
| dblp | 60 | sae-softmax-rate | 0.8068 | 0.7934 | 10 |
| webkb | 40 | random-rate | 0.6333 | 0.7758 | 10 |
| webkb | 40 | sae-rate | 0.6404 | 0.7349 | 10 |
| webkb | 40 | sae-softmax-rate | 0.6255 | 0.7738 | 10 |
| webkb | 60 | random-rate | 0.6127 | 0.7115 | 10 |
| webkb | 60 | sae-rate | 0.6138 | 0.7146 | 10 |
| webkb | 60 | sae-softmax-rate | 0.6073 | 0.7016 | 10 |

## Proxy com 100% dos dados (referência)

| dataset | proxy_macro |
| --- | --- |
| acm | 0.7058 |
| books | 0.8014 |
| dblp | 0.8146 |
| webkb | 0.6632 |

## Correlação de nível (Spearman, 23 configs): rho=0.614 p=0.00184
(Fácil: é dominada pela diferença entre datasets.)

## Diferenças método − Random (o teste que importa)

| dataset | rate | method | d_proxy | d_bert |
| --- | --- | --- | --- | --- |
| acm | 60 | sae-rate | -0.0004 | +0.0008 |
| acm | 60 | sae-softmax-rate | -0.0164 | -0.0043 |
| books | 40 | sae-rate | -0.0001 | +0.0003 |
| books | 40 | sae-softmax-rate | -0.0010 | -0.0072 |
| books | 60 | sae-rate | -0.0003 | +0.0024 |
| books | 60 | sae-softmax-rate | -0.0010 | -0.0018 |
| dblp | 40 | sae-rate | +0.0014 | +0.0004 |
| dblp | 40 | sae-softmax-rate | -0.0007 | +0.0003 |
| dblp | 60 | sae-rate | +0.0003 | +0.0016 |
| dblp | 60 | sae-softmax-rate | -0.0018 | -0.0062 |
| webkb | 40 | sae-rate | +0.0071 | -0.0408 |
| webkb | 40 | sae-softmax-rate | -0.0078 | -0.0020 |
| webkb | 60 | sae-rate | +0.0011 | +0.0031 |
| webkb | 60 | sae-softmax-rate | -0.0054 | -0.0099 |

Spearman(Δproxy, Δbert) = 0.437 (p=0.118); concordância de sinal = 64% (14 pares)
