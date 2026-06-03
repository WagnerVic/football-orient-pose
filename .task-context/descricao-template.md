# Descrição — Contexto detalhado

> Copie este arquivo para `.task-context/input/descricao.md` e preencha.
> Este arquivo é OBRIGATÓRIO para a skill funcionar.
>
> A estrutura abaixo é sugerida — adapte, remova ou adicione seções
> conforme o contexto da sua US. O que importa é que a IA tenha
> contexto real suficiente para não gerar issues genéricas.

---

## O que é

<!-- Visão geral do que será construído ou investigado. 2–4 frases. -->

## Problema / Motivação

<!-- Qual dor, limitação ou questão em aberto esta US resolve?
     Por que fazer isso agora? O que está bloqueado sem isso? -->

## Modelo base / Abordagem

<!-- Qual modelo ou técnica será usada (ex: ViTPose-B, mmpose, YOLO-Pose).
     Versão, checkpoint, configuração relevante. -->

## Dataset / Split

<!-- Qual dataset será usado. Qual split (train/val/test).
     Quantos frames/anotações. Formato das anotações (ex: H3WB-17, COCO). -->

## Formato de entrada/saída

<!-- O que o pipeline recebe e o que produz.
     Ex: "entrada: frame RGB 1920×1080 → saída: 17 keypoints (x, y, conf) por pessoa detectada" -->

## Pipeline do experimento

<!-- Passo a passo técnico: pré-processamento → inferência → pós-processamento → avaliação.
     Foco no que é novo ou não óbvio. -->

## Métricas esperadas

<!-- Quais métricas serão calculadas (PDJ, PCK, OKS, MPJPE).
     Se há baseline de comparação, indicar os valores.
     Ex: "baseline zero-shot: PDJ@0.5 = 55% — meta: > 70% com fine-tuning" -->

## Decisões técnicas já tomadas

<!-- Libs, versões, padrões já definidos que a IA deve respeitar.
     Evita a IA sugerir stack diferente do que o projeto usa. -->

## Escopo MVP

<!-- O que está incluído e o que está fora (com justificativa curta).
     Ex: "apenas vídeos de broadcast — câmera lateral é US futura" -->

## Dependências

<!-- Outras US, épicos ou artefatos que esta US depende.
     Ex: "depende de #12 (pipeline de detecção de jogadores)" -->
