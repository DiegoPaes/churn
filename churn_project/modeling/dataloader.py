import kagglehub
import shutil
import os
from pathlib import Path
from churn_project.logger import logging
from churn_project.exception import CustomException
import sys

def diretorio_raiz(start_path: Path = Path().resolve()) -> Path:
    """Sobe a hierarquia da pasta até encontrar o arquivo pyproject.toml"""
    try:
        logging.info('Iniciando a busca pelo diretório raiz')
        for parent in [start_path] + list(start_path.parents):
            if (parent / 'pyproject.toml').exists:
                return parent
    except Exception as e:
        raise CustomException(e, sys)

dir_raiz = str(diretorio_raiz())
RAW_DIR = dir_raiz + '/data/raw'

class DownloadDataset:
    def __init__(self, dataset_name="blastchar/telco-customer-churn", raw_data_dir=RAW_DIR):
        self.dataset_name = dataset_name
        self.raw_data_dir = raw_data_dir
        self.dataset_path = None
        self.arquivos_movidos = []

    def download_dataset(self) -> Path:
        try:
            logging.info('Iniciando download dataset')
            self.dataset_path = kagglehub.dataset_download(self.dataset_name)
            logging.info(f'Download concluído com sucesso em {self.dataset_path}')
        except Exception as e:
            raise CustomException(e, sys)

    def mover_para_raw(self):
        try:
            os.makedirs(self.raw_data_dir, exist_ok=True)
            logging.info('Inicio da transferecia do dataset')

            # Verifica se o diretório contém arquivos CSV
            for file in os.listdir(self.dataset_path):
                if file.endswith('.csv'):
                    src = os.path.join(self.dataset_path, file)
                    dst = os.path.join(self.raw_data_dir, file)
                    shutil.move(src, dst)
                    self.arquivos_movidos.append(dst)
                    logging.info(f'Arquivo movido de {src} para {dst}')
        except Exception as e:
            raise CustomException(e, sys)

            
if __name__ == '__main__':
    dataset = DownloadDataset()
    dataset.download_dataset()
    dataset.mover_para_raw()
    