# Classificação de tumores cerebrais com CNN — BRISC 2025

Projeto acadêmico de classificação de imagens de ressonância magnética cerebral utilizando **PyTorch e ResNet18 com Transfer Learning**. O trabalho compara um treinamento com a divisão original do BRISC e um segundo experimento com filtragem conservadora de imagens visualmente semelhantes entre partições.

> **Aviso:** este repositório é um protótipo de pesquisa/ensino. Não é um dispositivo médico, não fornece diagnóstico e não deve ser utilizado para orientar decisões clínicas.

## Sumário

- [1. Objetivo](#1-objetivo)
- [2. Classes e conjunto de dados](#2-classes-e-conjunto-de-dados)
- [3. Requisitos](#3-requisitos)
- [4. Instalação](#4-instalação)
- [5. Organização das pastas](#5-organização-das-pastas)
- [6. Fluxo principal: modelo original](#6-fluxo-principal-modelo-original)
- [7. Experimentos de interpretação e análise](#7-experimentos-de-interpretação-e-análise)
- [8. Auditoria e treinamento filtrado](#8-auditoria-e-treinamento-filtrado)
- [9. Comparação dos modelos e análise dos erros](#9-comparação-dos-modelos-e-análise-dos-erros)
- [10. Executar com CPU ou GPU](#10-executar-com-cpu-ou-gpu)
- [11. Resultados desta execução](#11-resultados-desta-execução)
- [12. Limitações metodológicas](#12-limitações-metodológicas)
- [13. Solução de problemas](#13-solução-de-problemas)
- [14. Referências](#14-referências)

## 1. Objetivo

O objetivo é desenvolver e avaliar um classificador que recebe uma fatia de MRI cerebral e prevê uma entre quatro categorias do conjunto BRISC:

1. **Glioma**;
2. **Meningioma**;
3. **Sem tumor (`no_tumor`)**;
4. **Tumor hipofisário (`pituitary`)**.

A arquitetura principal é a **ResNet18**, inicializada com pesos pré-treinados no ImageNet. A camada final é substituída por uma camada com quatro saídas e o modelo é ajustado para as classes do BRISC.

O projeto também inclui análises complementares de Grad-CAM, comparação com máscaras, desempenho por orientação anatômica, calibração de probabilidades e auditoria de possíveis imagens duplicadas ou semelhantes entre as partições.

**Forma de uso desta versão:** os experimentos são executados por scripts no terminal. Uma interface interativa `app.py`/Streamlit não faz parte da versão atualmente documentada.

## 2. Classes e conjunto de dados

### 2.1 BRISC

O projeto utiliza o **BRISC 2025**, conjunto público de fatias de MRI cerebral ponderadas em T1. A versão utilizada nesta execução contém 6.000 imagens para classificação: 5.000 em `train` e 1.000 em `test`. O conjunto também disponibiliza uma tarefa separada de segmentação com imagens e máscaras pareadas.

- [BRISC 2025 — Mendeley Data, versão 3](https://data.mendeley.com/datasets/c7dj4pwhcv/3)
- [Artigo do BRISC — Scientific Data](https://doi.org/10.1038/s41597-026-06753-y)

Baixe o conjunto e extraia-o para `data/brisc2025/`. A estrutura usada pelos scripts deve ficar assim:

```text
 data/
 └── brisc2025/
     ├── classification_task/
     │   ├── train/
     │   │   ├── glioma/
     │   │   ├── meningioma/
     │   │   ├── no_tumor/
     │   │   └── pituitary/
     │   └── test/
     │       ├── glioma/
     │       ├── meningioma/
     │       ├── no_tumor/
     │       └── pituitary/
     └── segmentation_task/
         ├── train/
         │   ├── images/
         │   └── masks/
         └── test/
             ├── images/
             └── masks/
```

A tarefa `classification_task` é necessária para treinamento e avaliação. A pasta `segmentation_task` é necessária somente para executar a comparação entre Grad-CAM e máscaras tumorais. Nas imagens e máscaras de segmentação, os nomes-base correspondem; normalmente a imagem é `.jpg` e a máscara é `.png`.

Os nomes das imagens indicam classe, plano e sequência. Os sufixos `ax`, `co` e `sa` representam, respectivamente, os planos axial, coronal e sagital. Os scripts de análise por plano utilizam essa convenção de nomes.

**Importante:** o download do conjunto de dados não está incluído neste repositório. Cada pessoa deve obtê-lo na fonte oficial e respeitar a licença e as condições de uso indicadas na página do dataset.

## 3. Requisitos

- Linux ou outro sistema compatível com Python e PyTorch;
- Python 3.12 foi utilizado nos experimentos deste projeto;
- memória RAM suficiente para carregar lotes de imagens;
- espaço em disco para o conjunto de dados, checkpoints e resultados;
- GPU NVIDIA com CUDA é recomendada para treinamento mais rápido, mas não é obrigatória.

### Ambiente utilizado nos experimentos relatados

- Python 3.12;
- PyTorch 2.15.0 (`+cu132` na instalação CUDA utilizada);
- TorchVision 0.29.2;
- NVIDIA GeForce RTX 3050 Laptop GPU com 6 GB;
- CUDA reconhecida pelo PyTorch.

As versões instaladas podem variar. Para escolher a combinação correta de PyTorch, TorchVision e plataforma de processamento, consulte o [instalador oficial do PyTorch](https://pytorch.org/get-started/locally/).

## 4. Instalação

Os comandos abaixo são para Linux, a partir da pasta raiz do projeto.

### 4.1 Criar e ativar o ambiente virtual

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 4.2 Instalar PyTorch e TorchVision

Escolha **uma** das opções abaixo.

**Opção A — computador sem GPU CUDA (CPU):**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Opção B — GPU NVIDIA:**

Acesse o [seletor oficial do PyTorch](https://pytorch.org/get-started/locally/), selecione o sistema operacional, Pip, Python e a versão CUDA compatível com o computador. Execute o comando gerado pelo site. Não escolha uma versão CUDA aleatoriamente: ela deve ser compatível com o ambiente e o driver da máquina.

### 4.3 Instalar as demais dependências

```bash
pip install numpy pandas matplotlib scikit-learn pillow tqdm seaborn scipy ImageHash
```

`ImageHash` é necessário para a auditoria pHash. As demais bibliotecas são utilizadas no carregamento de imagens, treinamento, métricas, gráficos e calibração.

### 4.4 Verificar a instalação

```bash
python - <<'PY'
import torch
import torchvision

print("PyTorch:", torch.__version__)
print("TorchVision:", torchvision.__version__)
print("CUDA disponível:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

Se `CUDA disponível` for `False`, o PyTorch utilizará CPU nos scripts com seleção automática de dispositivo.

## 5. Organização das pastas

Os scripts devem ficar na raiz do projeto, no mesmo nível de `data/`, `checkpoints/` e `results/`.

```text
 tumor_cnn/
 ├── .venv/                         # ambiente virtual (não versionar)
 ├── data/
 │   └── brisc2025/
 │       ├── classification_task/
 │       └── segmentation_task/     # necessário para análise de máscaras
 ├── checkpoints/                  # pesos produzidos pelo treinamento
 ├── results/                       # relatórios, CSVs e gráficos
 ├── inspect_dataset.py
 ├── data_pipeline.py
 ├── train.py
 ├── evaluate.py
 ├── gradcam.py
 ├── investigate_planes_gradcam.py
 ├── compare_gradcam_masks.py
 ├── calibrate_model.py
 ├── classwise_calibration.py
 ├── audit_brisc_leakage.py
 ├── inspect_leakage_membership.py
 ├── prepare_leakage_filtered_split.py
 ├── verify_leakage_manifests.py
 ├── train_filtered.py
 ├── evaluate_models_comparison.py
 └── analyze_model_errors.py
```

As pastas `checkpoints/` e `results/` são criadas pelos programas quando necessário. Os pesos `.pth`, o dataset e os resultados gerados não precisam ser enviados junto com o código-fonte, mas são necessários para reproduzir etapas específicas.

## 6. Fluxo principal: modelo original

Esta é a sequência recomendada para executar o experimento original desde o início.

### 6.1 Inspecionar o dataset

```bash
python inspect_dataset.py
```

Verifica a quantidade de arquivos por classe, dimensões, modos de cor e imagens ilegíveis. Gera `results/amostras_dataset.png`.

### 6.2 Conferir o pré-processamento e os DataLoaders

```bash
python data_pipeline.py
```

O pipeline:

- lê `classification_task/train` e `classification_task/test`;
- separa 20% da pasta `train` para validação, de forma estratificada;
- utiliza semente 42;
- redimensiona as imagens para 224 × 224 preservando a proporção, com preenchimento preto;
- converte as imagens para RGB;
- aplica tensorização e normalização com média `(0.485, 0.456, 0.406)` e desvio `(0.229, 0.224, 0.225)`;
- utiliza lotes de 16 imagens.

Gera `results/preprocess_preview.png`. A pasta oficial `test` não participa do ajuste dos pesos.

### 6.3 Treinar a ResNet18 original

```bash
python train.py
```

Parâmetros disponíveis:

```bash
python train.py --help
```

| Parâmetro | Padrão | Significado |
|---|---:|---|
| `--epochs` | 15 | Número máximo de épocas |
| `--lr` | `0.0001` | Taxa de aprendizado |
| `--patience` | 5 | Parada antecipada após épocas sem melhora |
| `--seed` | 42 | Semente aleatória |
| `--no-pretrained` | desativado | Não baixar os pesos ImageNet |

O melhor checkpoint é selecionado pela menor perda de validação e salvo em `checkpoints/best_resnet18.pth`. O treinamento também gera histórico, curvas, matriz de confusão e relatório de validação em `results/`.

Na primeira execução com pesos pré-treinados, é necessária conexão com a internet para baixar os pesos da ResNet18, caso ainda não estejam no cache do TorchVision.

### 6.4 Avaliar no teste oficial

Depois do treinamento:

```bash
python evaluate.py
```

Gera métricas, relatório por classe, matriz de confusão e `results/test_predictions.csv`. O script carrega `checkpoints/best_resnet18.pth` e não modifica os pesos.

**Execute a avaliação oficial somente depois de finalizar as decisões de treinamento.** Não utilize os resultados do teste para escolher épocas, hiperparâmetros ou versões do modelo.

## 7. Experimentos de interpretação e análise

Os scripts desta seção analisam o modelo original, salvo indicação em contrário.

### 7.1 Grad-CAM

```bash
python gradcam.py
```

Gera `results/gradcam_examples.png`, com imagens corretas e incorretas e os mapas de ativação. O programa utiliza as previsões geradas por `evaluate.py`.

Opções:

```bash
python gradcam.py --help
python gradcam.py --max-errors 4 --correct-per-class 1
```

Grad-CAM destaca regiões que influenciaram a pontuação de uma classe. **Não é uma máscara tumoral, não delimita precisamente uma lesão e não comprova que a rede utilizou evidência clinicamente relevante.**

### 7.2 Desempenho por plano anatômico

```bash
python investigate_planes_gradcam.py
```

Identifica o plano pelo sufixo do arquivo (`ax`, `co`, `sa`) e calcula métricas para axial, coronal e sagital. Gera tabelas e gráficos em `results/`, além de painéis Grad-CAM para exemplos corretos e erros.

### 7.3 Comparar Grad-CAM com máscaras tumorais

Requer a pasta `segmentation_task` do BRISC:

```bash
python compare_gradcam_masks.py
```

O script pareia imagens e máscaras, exclui a classe `no_tumor` desta comparação e calcula Dice, IoU, ROC-AUC por pixel e Average Precision. Também gera painéis visuais.

Opções:

```bash
python compare_gradcam_masks.py --help
```

O limiar padrão para Dice/IoU é 0,50. Resultados de Dice e IoU dependem do limiar e da forma como o mapa Grad-CAM é convertido em região binária. ROC-AUC e Average Precision avaliam a ordenação das ativações nos pixels; não equivalem à precisão de uma segmentação clínica.

### 7.4 Calibrar as probabilidades

```bash
python calibrate_model.py
python classwise_calibration.py
```

O primeiro script ajusta uma temperatura escalar utilizando somente o conjunto de validação e compara métricas antes/depois no teste. O segundo apresenta análises one-vs-rest por classe e por plano.

A temperatura encontrada nesta execução foi `1.205384`. Esse valor pertence ao checkpoint original e à divisão de validação utilizada nesta execução. Se o modelo ou a divisão mudar, a temperatura deve ser recalculada na validação correspondente.

Temperature Scaling ajusta a distribuição das probabilidades, mas não modifica os pesos da rede nem a classe de maior pontuação. ECE, NLL e Brier Score medem aspectos diferentes; uma calibração pode melhorar algumas métricas e piorar outras.

## 8. Auditoria e treinamento filtrado

Esta etapa foi criada após a identificação de possíveis duplicatas exatas e imagens visualmente semelhantes entre as partições do BRISC.

### 8.1 Auditar duplicatas e similaridades

```bash
python -m pip install ImageHash
python audit_brisc_leakage.py
```

O programa é somente leitura. Utiliza SHA-256 para detectar arquivos exatamente iguais e pHash de 64 bits, com distância de Hamming padrão até 6, para sinalizar imagens visualmente semelhantes dentro da mesma classe e orientação.

Resultados padrão em `results/leakage_audit/`:

- `audit_summary.json`;
- `exact_duplicate_groups.csv`;
- `near_duplicate_pairs.csv`;
- `train_candidates_to_review.csv`;
- `near_duplicate_contact_sheet.png`.

O pHash é uma triagem de similaridade visual, **não uma identificação de paciente ou exame**. Uma correspondência pode ser uma duplicata, uma fatia relacionada ou apenas uma imagem parecida. A ausência de correspondências também não garante independência por paciente.

### 8.2 Reconstruir a divisão interna original

```bash
python inspect_leakage_membership.py
```

Reproduz a divisão estratificada do pipeline original (80% treino efetivo / 20% validação dentro da pasta `train`) para identificar em qual parte estavam os candidatos sinalizados pela auditoria.

### 8.3 Preparar a divisão filtrada e agrupada

```bash
python prepare_leakage_filtered_split.py
```

Gera manifests CSV em `results/leakage_filtered/`, sem mover, apagar ou copiar as imagens originais. Na execução documentada, 1.191 imagens candidatas ligadas ao teste foram retiradas do pool de treino/validação, restando 3.809 imagens para uma divisão agrupada.

### 8.4 Validar os manifests

```bash
python verify_leakage_manifests.py
```

Confere existência dos caminhos, classes, ausência de imagens repetidas, reconstrução do pool e ausência de componentes compartilhados entre treino e validação. Só prossiga se as verificações terminarem com `OK`.

### 8.5 Treinar o modelo filtrado

```bash
python train_filtered.py
```

Opções:

```bash
python train_filtered.py --help
```

Exemplo forçando CPU e reduzindo o lote:

```bash
python train_filtered.py --device cpu --batch-size 8 --num-workers 0
```

O checkpoint filtrado é salvo separadamente em `checkpoints/best_resnet18_filtered.pth`. O checkpoint original não é sobrescrito. O teste oficial não é utilizado pelo script de treinamento filtrado.

## 9. Comparação dos modelos e análise dos erros

### 9.1 Comparar original e filtrado

```bash
python evaluate_models_comparison.py
```

A comparação usa as mesmas 1.000 imagens oficiais e o mesmo pré-processamento para os dois checkpoints. Não treina modelos nem ajusta parâmetros no teste.

Também realiza uma análise secundária removendo as 120 imagens de teste envolvidas em pares pHash internos detectados pela auditoria. Essa análise é apenas uma verificação de sensibilidade, pois o subconjunto resultante continua vindo do mesmo teste oficial.

Opções:

```bash
python evaluate_models_comparison.py --help
python evaluate_models_comparison.py --device cpu
```

Os resultados são salvos em `results/model_comparison/`.

### 9.2 Analisar os erros individuais

Depois da comparação:

```bash
python analyze_model_errors.py
```

Gera tabelas separando erros exclusivos do modelo original, exclusivos do filtrado e erros compartilhados. Também cruza os casos com os indicadores da auditoria pHash. A presença de uma sinalização pHash não demonstra que ela causou o erro.

## 10. Executar com CPU ou GPU

Os scripts com seleção automática de dispositivo utilizam CUDA quando disponível e CPU nos demais casos. Não é necessária uma versão separada do projeto para computadores sem GPU.

Para forçar CPU em scripts que selecionam o dispositivo automaticamente:

```bash
CUDA_VISIBLE_DEVICES="" python train.py
CUDA_VISIBLE_DEVICES="" python evaluate.py
CUDA_VISIBLE_DEVICES="" python gradcam.py
```

O treinamento filtrado e a comparação de modelos também possuem opção explícita:

```bash
python train_filtered.py --device cpu
python evaluate_models_comparison.py --device cpu
```

O processamento pela CPU é funcional, mas o treinamento tende a ser consideravelmente mais lento. Para reduzir o uso de memória, o treinamento filtrado permite diminuir `--batch-size` e `--num-workers`. No pipeline original, os valores `BATCH_SIZE` e `NUM_WORKERS` podem ser ajustados no arquivo `data_pipeline.py`.

## 11. Resultados desta execução

Os valores abaixo são os resultados observados na execução documentada deste projeto. Não são garantias de desempenho em outras instalações, divisões ou versões do dataset.

### 11.1 Teste oficial: modelo original e filtrado

| Métrica | ResNet18 original | ResNet18 filtrada |
|---|---:|---:|
| Imagens de treino | 4.000 | 3.047 |
| Imagens de validação | 1.000 | 762 |
| Acurácia no teste oficial | 98,70% | 95,50% |
| Precisão macro | 98,87% | 95,18% |
| Recall macro | 98,87% | 95,79% |
| F1-score macro | 98,87% | 95,42% |

A diferença observada de acurácia foi de 3,20 pontos percentuais. Ela não pode ser atribuída exclusivamente a vazamento: o modelo filtrado também foi treinado com menos imagens e com outra divisão de validação.

### 11.2 Auditoria de similaridade

Na auditoria realizada:

- 6.000 imagens legíveis;
- 9 pares exatos entre treino e teste;
- 891 pares pHash entre treino e teste, com Hamming até 6;
- 791 imagens de treino diretamente ligadas ao teste;
- 1.191 imagens de treino em componentes transitivos ligados ao teste;
- 120 imagens do teste envolvidas em pares pHash internos ao próprio teste.

Esses números são específicos da versão do dataset, arquivos e parâmetros usados nesta execução. Os candidatos pHash não são duplicatas confirmadas nem identificadores de pacientes.

### 11.3 Outras análises

- Acurácia original por plano: axial 98,74%; coronal 98,36%; sagital 98,99%.
- Grad-CAM versus máscaras: 860 imagens tumorais pareadas; Dice médio de 0,1346 com limiar 0,50; ROC-AUC por pixel médio de 0,9342; Average Precision média de 0,2840.
- Temperature Scaling do modelo original: temperatura `1.205384`; no teste, o ECE top-label passou de 0,008624 para 0,008131 e o NLL de 0,054806 para 0,050931. O Brier Score teve pequena piora, de 0,023947 para 0,024088.

## 12. Limitações metodológicas

1. **Não é ferramenta clínica.** O BRISC e este projeto não validam o uso do modelo para diagnóstico de pacientes.
2. **Sem identificadores completos de pacientes.** Não foi possível construir uma separação comprovadamente independente por paciente.
3. **pHash é heurístico.** Similaridade visual não comprova que duas imagens pertençam ao mesmo exame ou paciente.
4. **O teste oficial tem dependências internas detectadas.** A auditoria encontrou 68 pares pHash teste-teste, envolvendo 120 imagens. A análise secundária é de sensibilidade, não um novo teste independente.
5. **Grad-CAM não é segmentação.** As ativações ajudam a interpretar a decisão da CNN, mas não delimitam o tumor com precisão clínica.
6. **Diferença entre modelos tem múltiplas causas possíveis.** A queda de desempenho do modelo filtrado pode refletir a filtragem, a redução de dados e a nova divisão agrupada; o experimento não isola o efeito de cada fator.
7. **Domínio de imagens limitado.** O modelo foi treinado com imagens T1 do BRISC. O desempenho em outros protocolos, modalidades, hospitais ou equipamentos não foi estabelecido.

## 13. Solução de problemas

### `FileNotFoundError` para `classification_task`

Confira se a estrutura está exatamente em:

```text
data/brisc2025/classification_task/train/
data/brisc2025/classification_task/test/
```

### `Checkpoint não encontrado`

Execute primeiro o treinamento correspondente:

```bash
python train.py
```

ou, para o modelo filtrado:

```bash
python train_filtered.py
```

### `CUDA disponível: False`

O PyTorch não reconheceu uma GPU CUDA utilizável. Verifique a instalação de PyTorch e o driver NVIDIA. Se o computador não tiver GPU compatível, instale a versão CPU e execute normalmente ou force CPU conforme a [Seção 10](#10-executar-com-cpu-ou-gpu).

### Erro ao baixar pesos pré-treinados

A primeira execução de `train.py` ou `train_filtered.py` pode precisar baixar os pesos ImageNet da ResNet18. Verifique a conexão. Como alternativa, ambos aceitam `--no-pretrained`, mas isso muda o experimento: o modelo será inicializado sem os pesos pré-treinados.

### `ModuleNotFoundError: imagehash`

Instale a dependência específica da auditoria:

```bash
pip install ImageHash
```

### Memória insuficiente

No treinamento filtrado, tente:

```bash
python train_filtered.py --batch-size 8 --num-workers 0
```

No pipeline original, reduza `BATCH_SIZE` e, se necessário, `NUM_WORKERS` em `data_pipeline.py`.

## 14. Referências

1. **BRISC 2025 — Mendeley Data, versão 3.** DOI: [10.17632/c7dj4pwhcv.3](https://doi.org/10.17632/c7dj4pwhcv.3). Dataset de classificação e segmentação de MRI cerebral.
2. **BRISC: Annotated Dataset for Brain Tumor Segmentation and Classification.** *Scientific Data*. [https://doi.org/10.1038/s41597-026-06753-y](https://doi.org/10.1038/s41597-026-06753-y).
3. **PyTorch — Start Locally.** Instruções oficiais para instalar PyTorch com CPU ou plataforma de aceleração compatível. [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/).

---

**Projeto acadêmico — classificação de imagens médicas com aprendizado profundo.**
