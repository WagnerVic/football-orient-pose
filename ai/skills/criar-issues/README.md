# Skill: Criar Issues no GitHub

> Automatiza a criação de Épicos, User Stories e Tasks no GitHub
> seguindo os padrões do projeto football-orient-pose.

## Pré-requisitos

1. **gh CLI** instalado:
   ```bash
   # Linux (Debian/Ubuntu)
   sudo apt install gh

   # Verificar instalação
   gh --version
   ```

2. **Autenticado no GitHub:**
   ```bash
   gh auth login
   gh auth status
   ```

3. **Scopes de projeto habilitados:**
   ```bash
   gh auth refresh -s read:project -s project
   ```

---

## Como usar

### 1. Prepare o contexto

```bash
mkdir -p .task-context/input
cp .task-context/brief-template.md .task-context/input/brief.md
# edite brief.md e crie descricao.md com o contexto detalhado
```

| Arquivo | Obrigatório | O que colocar |
|---|---|---|
| `input/brief.md` | ✅ | Tipo, título, épico, resumo. Copie do `brief-template.md`. |
| `input/descricao.md` | ✅ | Contexto que a IA precisa para não gerar genérico: o problema, decisões técnicas, pipeline envolvido, métricas esperadas, dependências. **Sem este arquivo a skill recusa gerar.** |
| `input/referencias/` | ⚠️ Recomendado | Papers, notebooks anteriores, resultados de benchmark, configs de modelo. |

### 2. Invoque a skill na sua IA

```
Siga a skill ai/skills/criar-issues/SKILL.md
e crie as issues com base em .task-context/input/
```

A IA vai:
1. Ler seu contexto
2. Gerar um **preview completo** das issues
3. Pedir sua aprovação antes de criar qualquer coisa
4. Criar tudo via `gh` CLI — issues, sub-issues linkadas e board

### 3. Revise e aprove

A IA sempre mostra preview antes de criar. Você pode pedir ajustes antes de confirmar.

---

## Tipos de US suportados

| Tipo | Quando usar |
|---|---|
| `poc` | Estudo ou validação de viabilidade de um modelo/abordagem |
| `tecnica` | Infraestrutura, módulo interno, integração de pipeline |
| `experimento` | Fine-tuning, ablation study, comparativo de métricas |
| `outro` | Nenhum dos acima — descreva o tipo no brief |

---

## Compatibilidade

| IDE | Como referenciar a skill |
|---|---|
| Claude Code | `/skills criar-issues` ou referencie `ai/skills/criar-issues/SKILL.md` no chat |
| GitHub Copilot | `@ai/skills/criar-issues/SKILL.md` |
| Cursor | `@ai/skills/criar-issues/SKILL.md` |
| Aider | `/add ai/skills/criar-issues/SKILL.md` |
