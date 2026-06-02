# converte_blocos.py

entrada = "texts.txt"
saida = "texts_clean.txt"

with open(entrada, "r", encoding="utf-8") as f:
    conteudo = f.read()

# separa pelos blocos (duas quebras de linha)
blocos = conteudo.strip().split("\n\n")

linhas = []

for bloco in blocos:
    # remove espaços extras e junta tudo em uma linha
    linha = " ".join(bloco.split())
    
    # ignora linhas vazias
    if linha:
        linhas.append(linha)

with open(saida, "w", encoding="utf-8") as f:
    for linha in linhas:
        f.write(linha + "\n")

print(f"Convertido! Resultado salvo em: {saida}")