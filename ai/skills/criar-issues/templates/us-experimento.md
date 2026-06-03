# US - [Título do experimento]

| Campo | Valor |
|---|---|
| **Label** | user-story, enhancement |
| **Assignees** | WagnerVic |
| **Épico** | #XX |

---

## Narrativa

> "Como pesquisador, quero [executar experimento X],
> para [medir/validar/comparar Y] e [tomar decisão Z]."
>
> ✅ "Como pesquisador, quero fine-tunar o ViTPose-B no dataset anotado,
>     para medir o ganho de PDJ em relação ao baseline zero-shot e decidir
>     se o custo de anotação se justifica"

## Contexto

> Estado atual: qual baseline existe, quais resultados já foram obtidos.
> Motivação: por que este experimento agora?
> Dependências: qual US/task precisa estar pronta antes.

## Hipótese

> O que se espera provar ou refutar.
>
> ✅ "Fine-tuning com 500 frames anotados eleva PDJ@0.5 acima de 75%
>     (vs. ~55% do baseline zero-shot)"

## Configuração do Experimento

> Parâmetros fixos e variáveis do experimento.
>
> | Parâmetro | Valor |
> |---|---|
> | Modelo base | ViTPose-B (mmpose) |
> | Dataset | val split — X frames |
> | Épocas | 50 |
> | Learning rate | 1e-4 |
> | Batch size | 16 |
> | Métrica principal | PDJ@0.5 |

## Entregáveis

> O que será produzido.
>
> ✅ "Checkpoint do modelo fine-tunado salvo em `models/`"
> ✅ "Notebook de avaliação com métricas PDJ/PCK/OKS"
> ✅ "Tabela comparativa: zero-shot vs. fine-tuned"
> ✅ "Decisão registrada: avançar, ajustar hiperparâmetros ou descartar"

## Tasks

> Sub-issues. Atômicas, ordenadas por dependência.
> Formato: `- [ ] #XX — [título]`

## Critérios de aceite

> Verificáveis com métricas concretas.
>
> ✅ "Treino completa sem erro e checkpoint salvo"
> ✅ "Métricas PDJ/PCK/OKS calculadas no val split e registradas"
> ✅ "Comparação com baseline documentada na issue"

## Fora de escopo

> Limites explícitos do experimento.
>
> ✅ "Otimização de hiperparâmetros — será US separada se o baseline for promissor"

## Referências

> `#XX` para issues relacionadas, `[texto](url)` para papers/docs.
> Resultados de benchmarks anteriores, configs de referência.
