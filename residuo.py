import numpy as np


def tabela_residuos(y_true, y_pred, nome_csv:str="./tabela_residuos.csv"):
    residuos = y_true - y_pred
    observacao = np.arange(1, len(y_true)+1)

    tabela = np.column_stack(
        [observacao,
        y_true,
        y_pred,
        residuos]
    )

    np.savetxt(
        nome_csv,
        tabela,
        delimiter=";",
        fmt=["%d","%.4f","%.4f","%.4f"],
        header="observacao;valor_observado;valor_ajustado;residuo",
        comments=""
    )
    print("tabela de residuos criada")