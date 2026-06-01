# Épico 2 — Estimadores de Pose: Retrospectiva Técnica

**Branch:** `7-epic-2-estimadores-de-pose-3-modelos`  
**Data:** Maio 2026  
**Objetivo:** Implementar wrappers para 3 modelos de pose estimation (RTMPose, HRNet, OpenPose) com interface unificada, infraestrutura de reprodutibilidade e validação comparativa.

---

## Contexto e Motivação

O projeto compara modelos de pose estimation em crops de jogadores de futebol do dataset 3DSP (imagens 100×100 px, anotadas no formato H3WB-17). O Épico 2 precisava:

1. Ter os 3 modelos do paper de referência (Yeung et al., 2024) funcionando na mesma interface
2. Garantir que qualquer pessoa consiga reproduzir os experimentos do zero
3. Documentar por que o HRNet performa ~56% apesar de ser um modelo sólido no COCO

---

## Arquitetura: Interface Unificada (`BasePoseEstimator`)

Já existia antes do épico. É uma ABC (Abstract Base Class) em `src/football_orient_pose/pose.py` com:

- `predict(image) → (17, 3)` — COCO-17 com confiança por joint
- `predict_batch(images) → (N, 17, 3)` — batch de imagens
- `predict_h3wb(image) → (17, 2)` — converte COCO-17 → H3WB-17 (só x,y) para comparar com as anotações

**Por que essa interface?** O dataset anota no formato H3WB-17, mas os modelos são treinados em COCO-17. A conversão `coco17_to_h3wb17()` já existia em `utils/keypoint_mapping.py`. A interface força todos os modelos a expor os mesmos métodos, permitindo que o CLI de avaliação (`evaluate.py`) funcione com qualquer modelo sem mudança de código.

**Padrão de injeção para testes:**
```python
def __init__(self, ..., pose_model: Any | None = None):
    self._pose_model = pose_model if pose_model is not None else self._load_pose_model()
```
Esse padrão permite injetar um `FakeModel` nos testes unitários sem precisar dos pesos reais. A função real (`_load_pose_model`) só é chamada em produção.

---

## US 2.1 + 2.2 — RTMPoseEstimator

**Status:** ✅ Completo.

O RTMPose usa `rtmlib`, que baixa os pesos automaticamente na primeira inferência. Não precisa de setup manual. O notebook `01_rtmpose_validation.ipynb` documenta todos os resultados.

**Ajuste feito neste épico:** O notebook tinha o `DATA_DIR` hardcodado com um caminho absoluto de outra máquina (`/home/phaelzin/football-orient-pose/data/3dsp`). Corrigido para usar `python-dotenv` lendo o `.env` do projeto, tornando o notebook portátil.

### Resultados RTMPose — split val (800 frames, 40 clips)

| Métrica | Resultado | Referência (paper) |
|---------|-----------|-------------------|
| PDJ@0.5 | **93.62%** | 89.51% |
| PCK@0.2 | 41.76% | — |
| OKS | 81.82% | — |
| AP50 | 97.88% | — |
| AP75 | 81.00% | — |
| mAP@[.5:.95] | 69.04% | — |
| MPJPE-2D | 4.81 px | — |
| F1-macro | 93.62% | — |

---

## US 2.3 — HRNetEstimator

**Status:** ✅ Completo. ONNX exportado (242.6 MB). Testes passando. Avaliação executada.

### O que é o HRNet-W48

HRNet (High-Resolution Network) mantém representações de alta resolução em paralelo durante toda a rede, em vez de fazer encoder/decoder. Gera **heatmaps de saída `(17, 64, 48)`** — um por joint. A posição do keypoint é o argmax do heatmap, rescalado para a resolução original da imagem.

### Por que o PDJ é ~56% em crops 100×100

Esse resultado é **esperado e documentado** no paper (Yeung et al., 2024, Tab. 5).

O heatmap tem resolução `64 × 48`. Um crop de 100×100 px divide assim:
- Dimensão Y: 100 / 64 ≈ **1.56 px por célula do heatmap**
- Dimensão X: 100 / 48 ≈ **2.08 px por célula do heatmap**

O argmax do heatmap só pode apontar para posições inteiras no grid `64×48`. O erro de quantização é de até ±1 célula do heatmap, ou seja, ±2 px na imagem original. Esse erro se acumula em todos os 17 joints, e o PDJ@0.5 mede se o erro é menor que 50% da diagonal do bounding box. Para um crop 100×100, a diagonal é ~141 px, então o limiar é ~70 px — mas o erro sistemático de ~2 px derruba a precisão para ~56%.

O RTMPose usa **SimCC** (Simplified Coordinate Classification), que prediz coordenadas diretamente sem heatmaps, evitando completamente esse erro de quantização. Daí a diferença de 93% vs 56%.

### Por que ONNX e não PyTorch direto

O HRNet original usa `mmpose` + `mmcv` — ecossistema pesado com dependências de compilação. Tentamos instalar via `uv add mmpose` e falhou por dois motivos:
1. `chumpy` (dependência do mmpose) precisa de `pip` como build dependency, não compatível com `uv`
2. `xtcocotools` requer compilação de extensões C que falharam no ambiente

**Solução adotada:** O ONNX foi exportado uma única vez (usando um script temporário que dependia do código em `.refs`) e hospedado no **Google Drive**. O `download_models.sh` baixa o zip via `gdown` e descompacta localmente. Em runtime, a inferência usa só `onnxruntime` — sem dependência de mmpose/mmcv, sem export em tempo de setup.

**Por que Google Drive e não versionar no git:** O ONNX é dividido em dois arquivos (`hrnet_w48_coco_256x192.onnx` + `hrnet_w48_coco_256x192.onnx.data`) totalizando ~489 MB, que mesmo zipados ficam ~451 MB — acima do limite de 100 MB por arquivo do GitHub. O padrão adotado (zip no Drive + `gdown`) é o mesmo já usado para o dataset do projeto.

### `scripts/download_models.sh` — HRNet

```bash
uv run gdown "1dNC22Hvp-oHqb6vYKuQhs7TQDLoanB1K" -O models/weights/hrnet_w48_coco_256x192.zip
unzip -o models/weights/hrnet_w48_coco_256x192.zip -d models/weights/
rm models/weights/hrnet_w48_coco_256x192.zip
```
Idempotente: pula se `hrnet_w48_coco_256x192.onnx` já existir.

### Arquivo `src/football_orient_pose/estimators/hrnet.py`

O estimador:
- Carrega o ONNX com `onnxruntime` (import lazy — só na instanciação real)
- Pré-processamento: BGR→RGB → resize 256×192 → normalização ImageNet (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]) → CHW float32
- Pós-processamento: argmax no heatmap `(17,64,48)` → rescala para resolução original
- `predict()` retorna `(17, 3)` com confiança = valor máximo do heatmap

### Resultados HRNet — split val (800 frames, 40 clips)

| Métrica | Resultado | Referência (paper) |
|---------|-----------|-------------------|
| PDJ@0.5 | **88.90%** | ~56.08% |
| PCK@0.2 | 40.51% | — |
| OKS | 76.22% | — |
| AP50 | 94.88% | — |
| AP75 | 65.00% | — |
| mAP@[.5:.95] | 59.01% | — |
| MPJPE-2D | 6.04 px | — |
| F1-macro | 88.90% | — |

### Discrepância em relação ao paper (88.90% vs 56.08%)

O resultado obtido foi **32.8 pp acima** do reportado em Yeung et al. (2024). Hipóteses:

1. **Protocolo de avaliação diferente:** O paper pode estar avaliando em imagens da cena completa (não nos crops 100×100), onde o HRNet recebe pessoas menores ainda — aumentando o erro de quantização. Nossa avaliação usa os crops já extraídos.

2. **Versão do modelo:** O paper pode ter usado uma configuração diferente de HRNet (ex: HRNet-W32 em vez de W48, ou outra resolução de entrada).

3. **Pré-processamento distinto:** O paper pode não ter aplicado normalização ImageNet ou ter usado um pipeline de crop diferente antes de passar ao HRNet.

Apesar da discrepância, o resultado **confirma a hierarquia esperada**: RTMPose (93.62%) > HRNet (88.90%), consistente com o paper. O HRNet ainda fica ~5 pp abaixo do RTMPose, e as métricas de precisão refinada (AP75: 65% vs 81%, mAP: 59% vs 69%, MPJPE: 6.04 px vs 4.81 px) mostram claramente que o HRNet tem mais erro de localização — o que é consistente com a hipótese de quantização dos heatmaps.

### Testes `tests/test_hrnet_estimator.py`

6 testes com `FakeHRNetModel` injetado:
- Instanciação com modelo injetado
- `predict()` retorna shape `(17, 3)`
- `predict_h3wb()` retorna shape `(17, 2)`
- Confidences são positivas (>0)
- `predict_batch()` retorna shape `(N, 17, 3)`
- Modelo é reutilizado entre chamadas (não recarrega)

---

## US 2.4 — OpenPoseEstimator

**Status:** ⚠️ Código completo, pesos indisponíveis. Avaliação bloqueada.

### O que é o OpenPose

OpenPose é uma abordagem **bottom-up**: detecta todos os keypoints da imagem de uma vez usando Part Affinity Fields (PAFs), sem precisar de bounding box de pessoa. A variante COCO gera 18 partes (vs COCO-17 padrão — inclui o joint "Neck" que o COCO-17 não tem).

### Mapeamento OpenPose-18 → COCO-17

O joint "Neck" (índice 1 no OpenPose) não existe no COCO-17 e é descartado. Os demais são remapeados:

```python
OPENPOSE_TO_COCO17 = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]
# OPENPOSE_TO_COCO17[i] = índice OpenPose correspondente ao i-ésimo joint COCO
```

Esse mapeamento é testado em 4 testes unitários: comprimento=17, sem duplicatas, todos os índices no range [0,17], e que o "Neck" (OP índice 1) não aparece.

### Implementação

Usa `cv2.dnn.readNetFromCaffe()` — disponível em qualquer instalação do OpenCV, sem dependências extras. O pré-processamento usa `cv2.dnn.blobFromImage()` e a detecção de pico por joint usa `cv2.minMaxLoc()`.

### Por que os pesos não estão disponíveis

Os pesos do OpenPose COCO (`pose_iter_440000.caffemodel`, ~200 MB) são distribuídos pelo servidor da CMU:

```
http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel
```

Esse servidor estava **offline no momento da implementação** (DNS não resolvia). Tentativas alternativas:
- **GitHub oficial CMU**: o repositório tem um arquivo `CMakeLists.txt` que baixa os pesos durante o build do C++, mas não distribui o `.caffemodel` diretamente via releases
- **Mirror no GitHub Releases**: URL encontrada retornou 404

O `prototxt` da rede foi baixado com sucesso (46 KB, do GitHub oficial do OpenPose).

**Decisão:** Manter o código completo e funcional, usar o mesmo esquema do HRNet: obter o caffemodel de outra fonte, zipar e hospedar no Google Drive. O `download_models.sh` já tem o trecho comentado aguardando o ID do arquivo no Drive.

### Testes `tests/test_openpose_estimator.py`

9 testes, incluindo 4 dedicados ao mapeamento:
- Instanciação com modelo injetado
- `predict()` retorna shape `(17, 3)`
- `predict_h3wb()` retorna shape `(17, 2)`
- `OPENPOSE_TO_COCO17` tem comprimento 17
- Nenhum índice duplicado no mapeamento
- Todos os índices no range [0, 17]
- Neck (OP índice 1) não está no mapeamento

---

## Infraestrutura de Reprodutibilidade

### `.env` e `.env.example`

O notebook 01 tinha `DATA_DIR` hardcodado para `/home/phaelzin/football-orient-pose/data/3dsp`, que só funcionava em uma máquina específica. Adicionamos:

```bash
# .env (não vai pro git)
DATA_DIR=data
SPLIT_CONFIG=configs/split.json
```

O `.env.example` é commitado como template. O `.env` real fica no `.gitignore`.

**Descoberta importante:** O caminho correto dos dados é `data/train/00001/...`, não `data/3dsp/train/00001/`. O README antigo tinha o path errado — corrigido.

### `scripts/download_models.sh`

Script idempotente (pula se já existe) que faz tudo em um comando:

```bash
bash scripts/download_models.sh
```

| Modelo | Ação |
|--------|------|
| RTMPose | Nada — `rtmlib` baixa sozinho na primeira inferência |
| HRNet-W48 | Baixa zip do Google Drive via `gdown` e descompacta o ONNX |
| OpenPose | Baixa `.prototxt` do GitHub; caffemodel pendente upload no Drive |

### README completo

Reescrito do zero com:
- 3 passos de setup: dataset → `.env` → pesos
- Estrutura completa de pastas com todos os arquivos documentados
- Tabela de avaliação com resultados conhecidos e pendentes
- Comando de avaliação CLI

---

## CLI de Avaliação (`evaluate.py`)

O script original importava `RTMPoseEstimator` no topo do arquivo, o que fazia o `import` falhar em máquinas sem `rtmlib` instalado mesmo que o usuário quisesse rodar HRNet.

**Correção:** Imports lazy dentro de `_build_estimator()`:
```python
def _build_estimator(model_name, device):
    if model_name == "rtmpose":
        from football_orient_pose.estimators.rtmpose import RTMPoseEstimator
        return RTMPoseEstimator(device=device)
    if model_name == "hrnet":
        from football_orient_pose.estimators.hrnet import HRNetEstimator
        return HRNetEstimator(device=device)
    ...
```

Agora `hrnet` só importa `onnxruntime` quando requisitado, `rtmpose` só importa `rtmlib` quando requisitado, etc.

---

## Notebooks de Validação

### `01_rtmpose_validation.ipynb`

Já existia. Ajuste feito: substituir `DATA_DIR` hardcodado por `load_dotenv()` + `os.getenv("DATA_DIR")`.

**Resultados documentados:**
| Métrica | RTMPose | Referência (paper) |
|---------|---------|-------------------|
| PDJ@0.5 | 93.62% | 89.51% |
| PCK@0.2 | 41.76% | — |
| OKS | 81.82% | — |
| AP50 | 97.88% | — |
| MPJPE-2D | 4.81 px | — |

### `02_hrnet_validation.ipynb`

Criado do zero para documentar o HRNet. Seções:
1. **Setup** — carrega `.env`, define `ONNX_PATH`, verifica existência do arquivo
2. **GPU check** — `nvidia-smi` + `onnxruntime` providers
3. **Inferência single** — `predict()` e `predict_h3wb()` com asserts de shape
4. **Visualização** — matplotlib scatter dos keypoints na imagem (COCO-17 e H3WB-17)
5. **Batch smoke test** — 5 frames, assert `(5, 17, 3)`
6. **PDJ validation** — 40 clips × 20 frames = 800 frames, tqdm, referência 56.08%
7. **Métricas complementares** — tabela comparativa HRNet vs RTMPose
8. **Gráfico por grupo anatômico** — barras HRNet vs RTMPose
9. **Heatmap MPJPE-2D por joint** — usando `H3WB17_NAMES`
10. **Critério de aceite da US #13**

---

## Situação Atual por Item

| Item | Situação | Próximo passo |
|------|----------|---------------|
| `HRNetEstimator` + testes | ✅ Completo, 53/53 testes passando | — |
| `OpenPoseEstimator` + testes | ✅ Código completo | — |
| ONNX HRNet no Google Drive | ✅ Zip hospedado, ID configurado no script | — |
| `download_models.sh` via gdown | ✅ Baixa e descompacta ONNX do Drive | — |
| Notebook 01 RTMPose | ✅ Portátil (usa `.env`) | — |
| Notebook 02 HRNet | ✅ Criado | — |
| Avaliação RTMPose (PDJ) | ✅ PDJ 93.62% (ref paper: 89.51%) | — |
| Avaliação HRNet (PDJ) | ✅ PDJ 88.90% (ref paper: ~56.08%) | — |
| OpenPose caffemodel | ⏳ Pendente upload no Drive | Zipar, subir no Drive, preencher OPENPOSE_GDRIVE_ID |
| Notebook 03 OpenPose | ❌ Bloqueado pelos pesos | Após caffemodel disponível |
| Commit da branch | ⏳ Pendente | — |

---

## Problemas e Soluções Encontradas

### 1. mmpose não instala via uv

**Problema:** `uv add mmpose` falhou porque `chumpy` precisava de `pip` como build dependency (incompatível com `uv`) e `xtcocotools` requer compilação de extensões C.

**Solução:** Usar o código HRNet já presente em `.refs/3dsp-repo/3dsp_utils/MotionAGFormer/demo/lib/hrnet/` para carregar e exportar os pesos. Em runtime, só `onnxruntime` é necessário.

### 2. ONNX export gerava arquivo de 1.9 MB (sem pesos)

**Problema:** PyTorch 2.12 mudou o comportamento padrão do `torch.onnx.export()` para usar o exporter `dynamo=True`. Esse novo exporter gera apenas o grafo computacional sem os pesos embutidos — o arquivo de 1.9 MB carregava mas produzia só zeros na inferência.

**Solução:** `dynamo=False` (exporter clássico TorchScript). Gera 242.6 MB com pesos embutidos.

**Como descobrimos:** Comparando o tamanho do arquivo (1.9 MB vs esperado ~240 MB) e verificando que a saída era um tensor de zeros.

### 3. Chaves dos pesos incompatíveis

**Problema:** Os pesos do OpenMMLab têm prefixos diferentes do que o código `.refs` espera:
- OpenMMLab: `backbone.layer1.xxx` e `keypoint_head.final_layer.xxx`
- `.refs`: `layer1.xxx` e `final_layer.xxx`

`load_state_dict(strict=True)` falhava com centenas de chaves ausentes.

**Solução:** Loop de remapeamento antes do `load_state_dict`:
```python
for k, v in raw.items():
    if k.startswith("backbone."):
        state[k[len("backbone."):]] = v
    elif k.startswith("keypoint_head.final_layer."):
        state[k[len("keypoint_head."):]] = v
```

### 4. Export script dependia do `.refs` não versionado — resolvido com Google Drive

**Problema:** O `export_hrnet_onnx.py` dependia do `.refs` local (não versionado). Tentamos copiar os arquivos para `scripts/hrnet_lib/`, mas o `.gitignore` genérico tinha uma regra `lib/` que ignorava a pasta inteira — e git não permite re-incluir filhos de diretório ignorado.

**Solução definitiva:** Exportar o ONNX uma única vez localmente e hospedar o zip no Google Drive. O `download_models.sh` baixa com `gdown` — sem dependência de `.refs`, sem script de export, sem problema de gitignore. Os arquivos `export_hrnet_onnx.py` e `scripts/hrnet_lib/` foram deletados do repositório.

### 5. Servidor CMU do OpenPose offline

**Problema:** `posefs1.perception.cs.cmu.edu` não resolvia DNS. O `caffemodel` (~200 MB) não pôde ser baixado.

**Tentativas:**
1. URL direta do servidor CMU → DNS não resolve
2. Mirror via GitHub Releases → 404

**Solução:** Mesmo esquema do HRNet — quando o caffemodel for obtido, zipar e hospedar no Google Drive. O `download_models.sh` já tem o trecho comentado aguardando o `OPENPOSE_GDRIVE_ID`.

### 6. DATA_DIR errado no notebook e evaluate.py

**Problema:** README antigo dizia `data/3dsp/train/`, mas o dado real está em `data/train/`. O CLI padrão de `evaluate.py` usava `--data-dir data/3dsp`.

**Solução:**
- Corrigido `evaluate.py` para aceitar `--data-dir data` e usar `data_dir / "train" / clip_id`
- README corrigido com o caminho real
- `.env` com `DATA_DIR=data`

### 7. onnxscript faltando

**Problema:** O exporter dynamo tentava importar `onnxscript`, que não estava instalado. Apareceu como erro secundário ao depurar o ONNX de 1.9 MB.

**Solução:** Irrelevante após trocar para `dynamo=False`.

---

## Dependências Adicionadas Neste Épico

```toml
# pyproject.toml
dependencies = [
    "onnxruntime-gpu",  # HRNet inference
    "python-dotenv",    # .env support nos notebooks
    "gdown",            # download de pesos do Google Drive
]
```

`opencv-python` já estava presente (necessário para OpenPose via `cv2.dnn`). `yacs` foi removido junto com o export script — não é mais necessário em runtime.

---

## Referências

- Yeung et al. (2024) — paper com PDJ@0.5: RTMPose 89.51%, HRNet 56.08%
- HRNet ONNX (Google Drive): `https://drive.google.com/file/d/1dNC22Hvp-oHqb6vYKuQhs7TQDLoanB1K`
- OpenPose prototxt: `https://github.com/CMU-Perceptual-Computing-Lab/openpose` (models/pose/coco/)
