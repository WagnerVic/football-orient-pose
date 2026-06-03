# US - [Título do estudo/POC]

| Campo | Valor |
|---|---|
| **Label** | user-story, enhancement |
| **Assignees** | WagnerVic |
| **Épico** | #XX |

---

## Narrativa

> Para POC o ator é o pesquisador ou o time:
> "Como pesquisador, quero validar [abordagem/modelo],
> para [decisão que será tomada com o resultado]."
>
> ✅ "Como pesquisador, quero validar o uso do ViTPose em imagens de
>     futebol sem fine-tuning, para decidir se é viável como baseline
>     zero-shot antes de investir em anotação"

## Contexto

> Que decisão está bloqueada sem este estudo?
> Que dúvida técnica ou de viabilidade precisa ser respondida?
> Qual o estado atual do conhecimento sobre o tema?

## Escopo da POC

> Lista de pontos de validação. Cada item é uma pergunta concreta.
>
> ✅ "O modelo generaliza para frames de broadcast sem re-treino?"
> ✅ "PDJ@0.5 supera 50% no val split com inferência zero-shot?"
>
> O que NÃO será validado:
> ✅ "Não testaremos performance com vídeos de câmera lateral — apenas broadcast"

## Modelos / Abordagens a Avaliar

> Libs, checkpoints ou estratégias. Se comparativo, usar tabela:
>
> | Critério | Modelo A | Modelo B |
> |---|---|---|
> | Pré-treinado em pessoas | ✅ | ✅ |
> | Suporte a oclusão | ✅ | ⚠️ |
> | Inferência < 100ms/frame | ❌ | ✅ |

## Entregáveis

> O que será produzido.
>
> ✅ "Notebook com inferência e visualização de keypoints"
> ✅ "Tabela comparativa de métricas (PDJ, PCK, OKS) no val split"
> ✅ "Decisão registrada na issue ou ADR: qual modelo avançar para fine-tuning"

## Tasks

> Sub-issues.

## Critérios de aceite

> Critérios de aprendizado, não de funcionalidade final.
>
> ✅ "Métricas PDJ/PCK/OKS calculadas e registradas para cada modelo avaliado"
> ✅ "Decisão técnica registrada com justificativa (avançar ou descartar)"

## Fora de escopo

> O que esta POC NÃO cobre.

## Referências

> `#XX` para issues, `[texto](url)` para links externos.
> Papers dos modelos, notebooks de referência, dataset utilizado.
