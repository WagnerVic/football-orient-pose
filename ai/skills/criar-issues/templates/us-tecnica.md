# US - [Título da implementação técnica]

| Campo | Valor |
|---|---|
| **Label** | user-story |
| **Assignees** | WagnerVic |
| **Épico** | #XX |

---

## Narrativa

> Ator pode ser o pipeline ou pesquisador:
> "Como pipeline de estimação, quero [capacidade técnica],
> para [benefício arquitetural ou de qualidade do modelo]."
>
> ✅ "Como pipeline de inferência, quero um módulo de pré-processamento
>     de frames que normalize resolução e formato de entrada,
>     para garantir compatibilidade com todos os backbones avaliados"

## Contexto

> Situação técnica atual, motivação, dependências.
> Se integra com modelo externo: descrever contrato de entrada/saída.
> Se há decisão arquitetural: referenciar issue ou ADR.
> Se depende de outra US: mencionar explicitamente.

## Escopo MVP

> Decisões de escopo técnico com justificativa.
>
> ✅ "Inferência em batch offline (não real-time — latência não é requisito na v1)"
> ✅ "Suporte apenas a vídeos MP4 — outros formatos fora de escopo nesta US"

## Estratégia Técnica

> Stack/libs (com versão), padrão de implementação, formato de dados
> de entrada/saída, contrato de API interna se expõe/consome.
> Se há decisão já registrada em outra issue, referenciar e NÃO repetir.

## Tasks

> Sub-issues. Atômicas, ordenadas por dependência.
> Formato: `- [ ] #XX — [título]`

## Critérios de aceite

> Técnicos e verificáveis.
>
> ✅ "Script de pré-processamento roda sem erro em todos os vídeos do val split"
> ✅ "Saída segue o formato H3WB-17 definido em `.refs/`"
> ✅ "Tempo de processamento < 2s por frame em CPU"

## Fora de escopo

> Limites técnicos explícitos.
>
> ✅ "Inferência em tempo real — depende de otimização de latência, US futura"

## Referências

> `#XX` para issues, `[texto](url)` para links externos.
> Papers, notebooks de referência, configs de modelo, ADRs.
