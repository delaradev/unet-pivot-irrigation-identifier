# -*- coding: utf-8 -*-
"""cpic_training.ipynb

Driver do treinamento no Google Colab (GPU). Exportado automaticamente do notebook
oficial — ao editar o notebook no Colab, re-exporte e substitua este arquivo para
manter os dois em sincronia. Este script apenas clona o repositório e chama
src/main.py; toda a lógica de fato vive em src/ (ver README.md para a estrutura
completa do pipeline).

Original file is located at
    https://colab.research.google.com/drive/1y0ucXlT0CPE0VdhMSZDSgwqZvsCF5xiD
"""

from google.colab import drive
drive.mount('/content/drive')

!mkdir -p /content/cpic-brasil/code
!git clone https://github.com/delaradev/tcc.git /content/cpic-brasil/code

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/cpic-brasil/code
# ATENÇÃO: essa linha %cd é comentada automaticamente pela exportação do Colab p/ .py,
# mas é ESSENCIAL ao rodar de fato — sem ela, os comandos abaixo (mkdir data, pip
# install -r requirements.txt, python src/main.py) rodam relativos a /content, não a
# /content/cpic-brasil/code, e vão falhar por caminho não encontrado. Ao rodar célula a
# célula no notebook (.ipynb) real, a magic executa normalmente; ela só aparece
# comentada aqui porque este .py é um espelho exportado, não o que roda de fato.
!pip install -q -r requirements.txt

!mkdir -p data
!cp "/content/drive/MyDrive/dataset.zip" data/
!cd data && unzip -q dataset.zip
!ls -la data/dataset/train_images | head -5

!git pull

import os
os.environ['CPIC_OUTPUT_DIR'] = '/content/drive/MyDrive/cpic_runs'

!python src/main.py --mode train --config config/config.yaml

#!python src/main.py --mode train --config config/config.yaml --resume /content/drive/MyDrive/cpic_runs/CPIC_Brazil_Article_Reproducible_20260610_134010/last_model.keras