"""Executa o notebook em um kernel do Python atual e pode montar o ZIP local."""
from pathlib import Path
import argparse
import json
import os
import sys
import tempfile
import zipfile

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip', action='store_true', help='Gera entrega/TP1_Angel_Ottoni.zip')
    args = parser.parse_args()
    raiz = Path(__file__).resolve().parent
    for nome in ['food_truck.txt', 'imoveis.txt', 'estudantes.txt', 'microchips.txt']:
        if not (raiz / nome).is_file():
            parser.error(f'Falta {nome}: utilize os dados originais fornecidos na disciplina.')
    caminho = raiz / 'TP1_Modelos_Lineares.ipynb'
    notebook = nbformat.read(caminho, as_version=4)
    with tempfile.TemporaryDirectory(prefix='tp1-kernel-') as pasta:
        kernel = Path(pasta) / 'kernels' / 'tp1'
        kernel.mkdir(parents=True)
        (kernel / 'kernel.json').write_text(json.dumps({
            'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            'display_name': 'Python TP1', 'language': 'python'}))
        anterior = os.environ.get('JUPYTER_PATH')
        os.environ['JUPYTER_PATH'] = pasta + (os.pathsep + anterior if anterior else '')
        try:
            NotebookClient(notebook, timeout=180, kernel_name='tp1',
                           resources={'metadata': {'path': str(raiz)}}).execute()
        finally:
            if anterior is None:
                os.environ.pop('JUPYTER_PATH', None)
            else:
                os.environ['JUPYTER_PATH'] = anterior
    nbformat.validate(notebook)
    nbformat.write(notebook, caminho)
    print('Notebook executado e validado.')
    if args.zip:
        pasta_entrega = raiz / 'entrega'
        pasta_entrega.mkdir(exist_ok=True)
        arquivos = [caminho, raiz/'relatorio.md', raiz/'README.md', raiz/'requirements.txt',
                    raiz/'reproduzir.py', raiz/'resultados.json',
                    raiz/'experimentos_regularizacao.csv', raiz/'validacao_cruzada.csv']
        arquivos += [raiz/n for n in ['food_truck.txt', 'imoveis.txt', 'estudantes.txt', 'microchips.txt']]
        arquivos += sorted((raiz/'figuras').glob('*.png'))
        destino = pasta_entrega / 'TP1_Angel_Ottoni.zip'
        with zipfile.ZipFile(destino, 'w', zipfile.ZIP_DEFLATED) as pacote:
            for arquivo in arquivos:
                pacote.write(arquivo, arquivo.relative_to(raiz))
        with zipfile.ZipFile(destino) as pacote:
            assert pacote.testzip() is None
        print(f'ZIP criado e verificado: {destino}')


if __name__ == '__main__':
    main()
