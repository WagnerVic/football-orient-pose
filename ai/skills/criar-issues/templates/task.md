# Task — [Título descritivo da entrega]

| Campo | Valor |
|---|---|
| **Label** | task |
| **Assignee** | WagnerVic |

---

## Descrição

> O que fazer, por que é necessário, como se conecta com a US pai.
> Contexto suficiente pra implementar sem perguntar nada.
>
> Se depende de outra task: "Depende de #XX (pré-processamento)."
> Se envolve modelo: incluir formato de entrada/saída esperado.
> Se envolve script: descrever argumentos e saídas.
> Se envolve avaliação: especificar métrica e split usado.

## Entregáveis

> Paths concretos de arquivos ou artefatos.
>
> ❌ "Implementar o pipeline"
> ✅ "`src/pipeline/preprocess.py` — normalização de frames"
> ✅ "`notebooks/eval_vitpose.ipynb` — inferência e métricas"
> ✅ "`models/vitpose_b_finetuned.pth` — checkpoint salvo"

## Critérios de aceite

> Específicos a esta task (não da US inteira).
>
> ✅ "Script roda sem erro em todos os vídeos do val split"
> ✅ "Saída está no formato correto para o próximo estágio do pipeline"
> ✅ "PDJ@0.5 calculado e registrado no notebook"

## US pai

> Parte de: #XX — [título da US]
