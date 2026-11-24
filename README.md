# Pneumonia em Radiografias de Tórax: CNN do Zero vs ResNet-50 (Transfer Learning)

Comparação estatística entre duas abordagens de classificação binária (**NORMAL** vs **PNEUMONIA**) em radiografias de tórax pediátricas:

1. **CNN treinada do zero** (5 blocos convolucionais).
2. **ResNet-50 pré-treinada no ImageNet**, com o backbone totalmente congelado (só a nova cabeça é treinada).

Cada modelo é treinado **30 vezes** (seeds 0 a 29). As métricas no conjunto de teste são comparadas com Shapiro-Wilk e Mann-Whitney U.

## Estrutura

```
pneumonia-xray/
├── main.py                      # ponto de entrada: laço de modelos x seeds
├── requirements.txt
├── scripts/
│   └── unify_dataset.py         # unifica train/val/test em NORMAL/ e PNEUMONIA/
├── src/
│   ├── dataset.py               # leitura, divisão 70/15/15, Dataset e DataLoaders
│   ├── train.py                 # hiperparâmetros, treino, avaliação, run_one
│   ├── analysis.py              # estatística e gráficos
│   └── models/
│       ├── __init__.py          # build_model("CNN" | "ResNet50")
│       ├── scratch_cnn.py
│       └── frozen_resnet50.py
├── data/                        # dataset (ignorado pelo git)
└── results/                     # saídas (ignorado pelo git)
```

## Requisitos

- Python 3.10+ (testado com 3.11)
- GPU NVIDIA recomendada. O treino completo são 60 execuções; em CPU é muito lento.
- Para placas da série RTX 50 (Blackwell): driver 570+ e PyTorch com CUDA 12.8 (`cu128`).

## Instalação

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Verifique se a GPU foi detectada:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Dataset

Dataset: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) (Mooney, 2018), originalmente de Kermany et al. (2018).

```bash
pip install kaggle        # requer ~/.kaggle/kaggle.json
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p data
unzip data/chest-xray-pneumonia.zip -d data/archive
```

O zip do Kaggle traz `chest_xray/` com `train/`, `val/` e `test/` e, dentro dela, **uma segunda cópia** `chest_xray/chest_xray/`. As duas cópias têm o mesmo conteúdo. Use **apenas uma** ao unificar, senão as imagens ficam duplicadas e podem vazar entre treino e teste.

### Unificação

A divisão original da validação tem só 16 imagens, o que é insuficiente. O projeto junta tudo e faz uma divisão própria (70/15/15 estratificada):

```bash
python scripts/unify_dataset.py \
    --root ./data/archive/chest_xray \
    --dst  ./data/chest_xray_unificado
```

Resultado esperado:

```
data/chest_xray_unificado/
├── NORMAL/       (1.583 imagens)
└── PNEUMONIA/    (4.273 imagens)     # total: 5.856
```

Confira as contagens com `ls data/chest_xray_unificado/NORMAL | wc -l`. Se aparecer o dobro, as duas cópias foram incluídas. Apague a pasta e rode de novo. A opção `--move` move os arquivos em vez de copiar.

## Execução

Rode sempre a partir da raiz do projeto.

```bash
# teste rápido (seeds 0 e 1)
python main.py --data_dir ./data/chest_xray_unificado --seeds 2

# experimento completo (seeds 0 a 29)
python main.py --data_dir ./data/chest_xray_unificado --seeds 30 --workers 6
```

Em segundo plano (Linux):

```bash
nohup python main.py --data_dir ./data/chest_xray_unificado --seeds 30 --workers 6 > treino.log 2>&1 &
tail -f treino.log
```

### Argumentos

| Argumento | Padrão | Descrição |
|---|---|---|
| `--data_dir` | (obrigatório) | Pasta unificada com `NORMAL/` e `PNEUMONIA/` |
| `--out_dir` | `results` | Pasta de saída |
| `--seeds` | `30` | Número de seeds; roda de `0` a `seeds-1` |
| `--workers` | `4` | Processos do DataLoader |
| `--ref_seed` | `7` | Seed usada nas curvas de treino e na ROC. Deve estar no intervalo de seeds executadas |
| `--models` | `CNN ResNet50` | Modelos a treinar (ex.: `--models CNN`) |

### Retomada

Os resultados são gravados em `results/runs.csv` a cada execução. Se o treino for interrompido, rodar o mesmo comando de novo **pula** as execuções já concluídas. Se mudar hiperparâmetros, dataset ou código dos modelos, apague `results/` antes, senão resultados antigos se misturam com os novos.

## Saídas (`results/`)

| Arquivo | Conteúdo |
|---|---|
| `runs.csv` | Uma linha por execução: loss, acurácia, F1 (macro, Normal, Pneumonia), AUC, tempo |
| `stats_summary.csv` | Média, desvio-padrão, mediana, p-valor de Shapiro-Wilk e de Mann-Whitney por métrica |
| `boxplot_f1.png` | Distribuição do F1 macro nas seeds |
| `curvas_CNN_old.png`, `curvas_ResNet50_old.png` | Loss e acurácia de treino/validação (seed de referência) |
| `roc.png` | Curvas ROC comparativas (seed de referência) |
| `histories/`, `probs/` | Histórico por época e probabilidades de teste de cada execução |

A estatística e os gráficos só são gerados quando `runs.csv` contém os dois modelos.

## Detalhes do experimento

**Dados:** redimensionamento para 224x224, normalização com média/desvio do ImageNet e, no treino, flip horizontal e rotação aleatória de até 10°. A divisão 70/15/15 é estratificada e **fixa** (`random_state=42`). As seeds controlam apenas a inicialização dos pesos e a ordem dos lotes.

**CNN do zero:** 5 blocos (`Conv 3x3` → `BatchNorm` → `ReLU` → `MaxPool 2x2`) com 32→64→128→256→512 canais, `AdaptiveAvgPool2d(1)` e cabeça totalmente conectada 512→256→128→64→2, com dropout 0,3 antes da camada final.

**ResNet-50:** pesos `IMAGENET1K_V1`, todos os parâmetros do backbone congelados (incluindo BatchNorm em modo `eval`) e `fc` substituída por `Dropout(0.2)` + `Linear(2)`.

| Hiperparâmetro | CNN | ResNet-50 |
|---|---|---|
| Learning rate | 5e-5 | 5e-4 |
| Batch size | 64 | 64 |
| Épocas | 10 | 10 |
| Otimizador | Adam | Adam |
| `class_weight` | não | não |

**Avaliação:** o modelo da **última época** é avaliado no conjunto de teste, com limiar 0,5. As métricas por seed são comparadas com o teste de Mann-Whitney U (bilateral), após Shapiro-Wilk para verificar normalidade (α = 0,05).

## Reprodutibilidade

- Seeds fixadas em `random`, `numpy` e `torch` (CPU e CUDA), e `Generator` próprio no DataLoader de treino.
- `cudnn.benchmark = True` (em `main.py`) acelera o treino, mas pode causar pequenas diferenças entre execuções com a mesma seed. Para reprodutibilidade estrita, use `torch.backends.cudnn.deterministic = True` e `benchmark = False`, ao custo de velocidade.
- Resultados também variam entre GPUs, drivers e versões do PyTorch.

## Problemas comuns

| Sintoma | Causa provável |
|---|---|
| `torch.cuda.is_available()` retorna `False` | Driver desatualizado ou PyTorch instalado sem CUDA (confira o índice `cu128`) |
| `no kernel image is available` | PyTorch sem suporte à arquitetura da GPU; use CUDA 12.8+ |
| `Nenhuma imagem em .../{NORMAL,PNEUMONIA}` | `--data_dir` não aponta para a pasta unificada |
| `Total` maior que 5.856 | As duas cópias do Kaggle foram unificadas |
| Sem curvas/ROC nos resultados | `--ref_seed` fora do intervalo de seeds executadas |
| Erro de multiprocessing no Windows | Rode via `python main.py` (o guard `__main__` já existe) ou use `--workers 0` |
| GPU com baixo uso | Gargalo na leitura das imagens: aumente `--workers` e use SSD |

## Próximos Passos

- Atualizar o relatório com os novos resultados obtidos
- Adição de LR scheduler para ambos os modelos
- Fine-tuning do ResNet50

## Referências

- Kermany et al. *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning.* Cell, 2018.
- Mooney, P. T. *Chest X-Ray Images (Pneumonia).* Kaggle, 2018.