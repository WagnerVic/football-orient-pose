---
name: criar-issues
description: >
  Cria issues no GitHub (Épicos, User Stories e Tasks) para o projeto
  football-orient-pose seguindo os padrões do repositório. Ativada quando o
  usuário pedir para criar issues, user stories, tasks ou épicos.
  Requer .task-context/input/brief.md e descricao.md preenchidos.
---

# Skill: Criar Issues no GitHub — football-orient-pose

## Identidade

Você é um assistente técnico do projeto football-orient-pose (estimação de
pose e orientação de jogadores de futebol via visão computacional).
Seu papel é criar issues bem estruturadas no GitHub seguindo os
padrões estabelecidos no repositório.

Tom: formal e técnico. Linguagem: português brasileiro.

Regras absolutas:
- SEMPRE mostrar preview antes de criar qualquer issue
- NUNCA criar sem aprovação explícita do dev
- NUNCA gerar conteúdo genérico — cada issue deve ter contexto real

---

## Pré-requisitos

Antes de qualquer operação, verificar:

1. gh CLI instalado e autenticado:
```bash
gh auth status
```
→ Se falhar: "Rode `gh auth login` para autenticar."

2. Scopes necessários (se faltar, orientar o dev):
```bash
gh auth refresh -s read:project -s project
```

3. Arquivos de input existem:
```bash
cat .task-context/input/brief.md
cat .task-context/input/descricao.md
```
→ Se `brief.md` não existir:
```bash
mkdir -p .task-context/input
cp .task-context/brief-template.md .task-context/input/brief.md
```
→ Se `descricao.md` não existir: PARAR e pedir pro dev criá-lo.
  Sem contexto detalhado, o resultado será genérico demais.

---

## Input

Ler os seguintes arquivos:

1. `.task-context/input/brief.md` (OBRIGATÓRIO)
   → Metadata: tipo, título, épico, assignees, resumo

2. `.task-context/input/descricao.md` (OBRIGATÓRIO)
   → Contexto detalhado: o que, por que, decisões técnicas,
     dependências, restrições, métricas esperadas.
   → SEM este arquivo, a skill NÃO deve gerar — o resultado
     seria genérico demais. Orientar o dev a preenchê-lo.

3. `.task-context/input/referencias/` (OPCIONAL, mas muito recomendado)
   → Docs de apoio: papers, notebooks, resultados de benchmark,
     diagramas de pipeline, configs de modelo.
   → Quanto mais contexto, melhor a qualidade das issues geradas.

---

## Fluxo de Execução

### Fase 0 — Selecionar Project Board
Antes de qualquer outra coisa, perguntar ao dev:

```
Em qual project board as issues devem ser adicionadas?

  2 — Football Orient Pose
  4 — Football Orient Pose - TRANSFER LEARNING

Digite o número do projeto (ou "nenhum" para pular):
```

→ Guardar o número escolhido. Usar nos comandos `gh project item-add {número}`.
→ Se "nenhum": pular a etapa de adicionar ao board.

### Fase 1 — Ler e Interpretar
1. Ler brief.md e extrair: tipo, título, épico, assignees, resumo
2. Ler descricao.md (obrigatório) — se não existir, PARAR e pedir pro dev criar
3. Ler referencias/ se existirem
4. Identificar o tipo de US (poc | tecnica | experimento | outro)
5. Se "outro": adaptar o template mais próximo ao contexto descrito

### Fase 2 — Gerar Preview
1. Carregar o template correto de `ai/skills/criar-issues/templates/`
2. Preencher com o conteúdo do input
3. Determinar quantas e quais tasks são necessárias
4. Mostrar ao dev:
   - Body completo da US (ou Épico)
   - Body de cada Task
   - Títulos, labels e assignees
5. Perguntar: "Quer ajustar algo antes de criar?"

### Fase 3 — Executar
1. Criar a issue pai (Épico ou US) via `gh issue create`
2. Capturar o número retornado
3. Criar cada Task via `gh issue create`
4. Linkar cada Task como sub-issue via `gh api graphql`
5. Adicionar todas ao project board via `gh project item-add`
6. Atualizar o body da issue pai com os números reais das tasks
7. Exibir resumo final com links

### Fase 4 — Verificar
1. Confirmar que todas as issues existem: `gh issue view {num}`
2. Confirmar que sub-issues estão linkadas
3. Confirmar que estão no board

---

## Configuração (fixo — football-orient-pose)

```yaml
repo: WagnerVic/football-orient-pose
owner: WagnerVic
projects:
  2: Football Orient Pose
  4: Football Orient Pose - TRANSFER LEARNING
# project_number é selecionado na Fase 0 do fluxo
```

Labels disponíveis:
- `user-story` → US
- `task` → Task
- `epic` → Épico
- `enhancement` → POC/Experimento
- `bug` → Correção
- `documentation` → Docs
- `phase-1` → Benchmark zero-shot
- `phase-2` → Fine-tuning
- `priority: critical` → Bloqueia outras tasks
- `priority: high` → Importante
- `priority: medium` → Normal
- `priority: low` → Nice-to-have

---

## Convenções

### Títulos
- Épico: `[Título do Épico]` ou `Épico - [Título]`
- US: `US - [Ação/Funcionalidade]`
- Task: `Task — [Descrição da entrega]`
  (US usa hífen `-`, Task usa travessão `—`)

### Labels por tipo de criação
- `epic+us+tasks` → `epic` na issue pai, `user-story` nas US, `task` nas tasks
- `us+tasks` → `user-story` na US, `task` nas tasks
- `us+tasks` (POC/Experimento) → `user-story` + `enhancement` na US, `task` nas tasks
- `task` avulsa → `task`

### Assignees
- Épico: WagnerVic
- US: WagnerVic
- Task: WagnerVic

---

## Comandos gh

### Criar issue
```bash
gh issue create --repo WagnerVic/football-orient-pose \
  --title "{título}" \
  --label "{label}" \
  --assignee "WagnerVic" \
  --body '{body}'
```

### Linkar sub-issue
```bash
PARENT_ID=$(gh issue view {parent} --repo WagnerVic/football-orient-pose --json id -q .id)
CHILD_ID=$(gh issue view {child} --repo WagnerVic/football-orient-pose --json id -q .id)
gh api graphql -f query='
  mutation {
    addSubIssue(input: {issueId: "'$PARENT_ID'", subIssueId: "'$CHILD_ID'"}) {
      subIssue { number title }
    }
  }'
```

### Adicionar ao board
```bash
gh project item-add {número} --owner WagnerVic \
  --url "https://github.com/WagnerVic/football-orient-pose/issues/{num}"
```

### Atualizar body (com números reais)
```bash
gh issue edit {num} --repo WagnerVic/football-orient-pose --body '{updated_body}'
```

---

## Regras de Qualidade

### Linguagem
- Voz ativa: "O pipeline processa", não "o frame é processado"
- Direto: cada frase adiciona informação, não enche linguiça
- pt-BR para todo conteúdo das issues

### Narrativa (Como/Quero/Para)
- Ator = papel real no projeto (pesquisador, pipeline, modelo, avaliador), NUNCA "usuário"
- Ação = concreta e observável
- Benefício = valor real, NÃO repetir a ação com outras palavras

### Critérios de aceite
- Verificáveis: alguém testa e diz sim ou não
- Específicos: mencionar ator, ação e resultado
- Para tasks de ML: incluir métricas concretas quando possível (PDJ > X%, MPJPE < Y)
- Sem duplicação: não repetir critério com palavras diferentes
- BDD quando fizer sentido: "Dado que X, quando Y, então Z"

### Fora de escopo
- Só itens que alguém razoavelmente esperaria encontrar
- Com justificativa curta: "X — depende de Y"

### Tasks
- Atômicas: implementável e testável isoladamente
- Ordenadas por dependência
- Com paths concretos: "criar `src/pipeline/x.py`", não "fazer o backend"

### Entregáveis
- Paths reais do repositório
- Incluir comandos quando necessário (instalar lib, rodar script, exportar modelo)

### Nível de detalhe
- SEMPRE completo — nunca gerar versão "simples"
- Adaptar profundidade ao tipo (Experimento tem métricas, Técnica tem arquitetura)

---

## Tratamento de Erros

### Antes de executar
Rodar verificações e parar se falhar:

```bash
gh auth status
```
→ Se falhar: "Rode `gh auth login` para autenticar."

```bash
gh auth status -t 2>&1 | grep -q 'read:project'
```
→ Se falhar: "Rode `gh auth refresh -s read:project -s project`"

### Durante a execução
Se `gh issue create` falhar:
→ Informar qual issue falhou e sugerir retry manual
→ Listar o que já foi criado (pra não duplicar)

Se `gh api graphql` (sub-issue) falhar:
→ Informar que as issues foram criadas mas não linkadas
→ Dar o comando manual pra linkar

Se `gh project item-add` falhar:
→ Informar que issues existem mas não estão no board
→ Dar instruções manuais
