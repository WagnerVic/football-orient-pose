# ==============================================================================
# setup_data.ps1 - Descompacta os datasets zipados para a pasta data/
#
# Uso:
#   .\setup_data.ps1              # descompacta todos os .zip encontrados
#   .\setup_data.ps1 -Clean       # remove data/ para re-extrair do zero
#
# Requisitos: PowerShell 5.1+ (já incluso no Windows 10/11)
# ==============================================================================

param(
    [switch]$Clean
)

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$DataDir = Join-Path $ProjectRoot "data"

function Write-Info  { param($msg) Write-Host "[INFO]  $msg" -ForegroundColor Green }
function Write-Warn  { param($msg) Write-Host "[WARN]  $msg" -ForegroundColor Yellow }
function Write-Err   { param($msg) Write-Host "[ERROR] $msg" -ForegroundColor Red }

# --- modo clean ---
if ($Clean) {
    if (Test-Path $DataDir) {
        Remove-Item $DataDir -Recurse -Force
        Write-Info "Pasta data/ removida. Execute o script novamente para re-extrair."
    } else {
        Write-Warn "Pasta data/ nao existe."
    }
    exit 0
}

# Garante que data/ existe
New-Item -ItemType Directory -Force -Path $DataDir | Out-Null

# Busca todos os .zip na raiz do projeto
$zips = Get-ChildItem -Path $ProjectRoot -Filter "*.zip" -File

if ($zips.Count -eq 0) {
    Write-Warn "Nenhum arquivo .zip encontrado na raiz do projeto."
    exit 0
}

foreach ($zip in $zips) {
    $zipName  = $zip.BaseName
    $marker   = Join-Path $DataDir ".extracted_$zipName"

    # Idempotencia: pula se ja foi extraido
    if (Test-Path $marker) {
        Write-Warn "Dataset '$zipName' ja foi extraido. Para re-extrair, execute: .\setup_data.ps1 -Clean"
        continue
    }

    Write-Info "Extraindo '$($zip.Name)' para data/..."

    # Extrai para pasta temporaria
    $tmpDir = Join-Path $env:TEMP ("3dsp_extract_" + [System.IO.Path]::GetRandomFileName())
    New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null

    try {
        Expand-Archive -Path $zip.FullName -DestinationPath $tmpDir -Force

        # Detecta a pasta raiz dentro do zip (ex: "3dsp/")
        $topLevel = Get-ChildItem -Path $tmpDir | Select-Object -First 1

        if ($topLevel -and $topLevel.PSIsContainer) {
            # Copia o conteudo da pasta raiz direto para data/
            # (train/ e test/ ficam em data/, sem a pasta intermediaria)
            $innerItems = Get-ChildItem -Path $topLevel.FullName
            foreach ($item in $innerItems) {
                $dest = Join-Path $DataDir $item.Name
                if (-not (Test-Path $dest)) {
                    Copy-Item -Path $item.FullName -Destination $DataDir -Recurse
                }
            }
        } else {
            # Zip sem pasta raiz: copia tudo direto
            Copy-Item -Path "$tmpDir\*" -Destination $DataDir -Recurse
        }

        # Cria marcador de controle
        Get-Date -Format "o" | Out-File -FilePath $marker -Encoding utf8
        Write-Info "Dataset '$zipName' extraido com sucesso!"

    } finally {
        Remove-Item $tmpDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Write-Info "Setup concluido! Dados disponiveis em: $DataDir"
