import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from churn_project.logger import logging
from churn_project.exception import CustomException
from dataloader import diretorio_raiz
from typing import Dict, Type
import sys
import os

np.random.seed(42)

class ISalvadorStrategy(ABC):
    @abstractmethod
    def salvar(self, df: pd.DataFrame, caminho_saida: str) -> None:
        pass

class SalvadorCSVStrategy(ISalvadorStrategy):
    """Estratégia concreta para salvar DataFrames como arquivos CSV."""
    def salvar(self, df: pd.DataFrame, caminho_saida: str) -> None:
        try:
            logging.info(f"Utilizando a estratégia para salvar como CSV em '{caminho_saida}'...")
            df.to_csv(caminho_saida, index=False)
        except Exception as e:
            raise CustomException(e, sys)

class SalvadorParquetStrategy(ISalvadorStrategy):
    """
    Estratégia concreta para salvar DataFrames como arquivos Parquet.
    """
    def salvar(self, df: pd.DataFrame, caminho_saida: str) -> None:
        try:
            logging.info(f"Utilizando a estratégia para salvar como Parquet em '{caminho_saida}'...")
            df.to_parquet(caminho_saida, index=False, engine='pyarrow')
        except Exception as e:
            raise CustomException(e, sys)

class ILeitorStrategy(ABC):
    @abstractmethod
    def ler(self, caminho_arquivo: str) -> pd.DataFrame:
        """
        Método de leitura que será implementado por cada estratégia específica
        """
        pass

class LeitorCSVStrategy(ILeitorStrategy):
    """
    Leitura de arquivos CSV
    """
    def ler(self, caminho_arquivo: str) -> pd.DataFrame:
        try:
            logging.info('Iniciando leitura do CSV')
            return pd.read_csv(caminho_arquivo)
        except Exception as e:
            raise CustomException(e, sys)

class LeitorXLSXStrategy(ILeitorStrategy):
    """
    Leitura de arquivos xlsx
    """
    def ler(self, caminho_arquivo: str) -> pd.DataFrame:
        try:
            logging.info('Iniciando leitura do xlsx')
            return pd.read_excel(caminho_arquivo)
        except Exception as e:
            raise CustomException(e, sys)
        
class LeitorParquetStrategy(ILeitorStrategy):
    """
    Leitura de arquivos xlsx
    """
    def ler(self, caminho_arquivo: str) -> pd.DataFrame:
        try:
            logging.info('Iniciando leitura do parquet')
            return pd.read_parquet(caminho_arquivo)
        except Exception as e:
            raise CustomException(e, sys)

class EnriquecimentoDataset:
    """
    Classe responsável por aplicar enriquecimentos a um DataFrame.
    A criação do objeto é desacoplada da fonte de dados (arquivo).
    """
    _estrategia_leitura: Dict[str, Type[ILeitorStrategy]] = {
        '.csv': LeitorCSVStrategy,
        '.xlsx': LeitorXLSXStrategy,
        '.parquet': LeitorParquetStrategy
    }

    _estrategias_salvamento: Dict[str, Type[ISalvadorStrategy]] = {
        '.csv': SalvadorCSVStrategy,
        '.parquet': SalvadorParquetStrategy,
    }

    def __init__(self, dataframe: pd.DataFrame):
        """
        O construtor agora recebe um DataFrame.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise CustomException(TypeError('A entrada deve ser um dataframe pandas'), sys)
        self.df = dataframe.copy()
        logging.info('Objeto EnriquecimentoDataset criado com sucesso')

    @classmethod
    def from_file(cls, caminho_arquivo: str) -> 'EnriquecimentoDataset':
        logging.info(f'Iniciando a criação do dataset a partir do arquivo {caminho_arquivo}')
        _, extensao = os.path.splitext(caminho_arquivo)
        extensao = extensao.lower()

        try:
            estrategia_classe = cls._estrategia_leitura[extensao]
            leitor = estrategia_classe()
            dataframe_carregado = leitor.ler(caminho_arquivo)
            return cls(dataframe_carregado)
        except KeyError:
            print(f"ERRO: Nenhuma estratégia de leitura encontrada para a extensão '{extensao}'.")
            raise ValueError(f"Formato de arquivo '{extensao}' não é suportado.")
        except Exception as e:
            raise CustomException(e, sys)
        
    def get_df(self) -> pd.DataFrame:
        """Retorna o DataFrame atual."""
        return self.df
    
    def salvar(self, caminho_saida: str) -> None:
        """
        Salva o DataFrame do objeto em um arquivo, usando a estratégia
        apropriada baseada na extensão do arquivo de saída.
        """
        # 1. Garante que o diretório de destino exista
        
        diretorio_pai = os.path.dirname(caminho_saida)
        if diretorio_pai: # Garante que não é um caminho relativo sem pasta
            os.makedirs(diretorio_pai, exist_ok=True)
            logging.info(f"Diretório '{diretorio_pai}' verificado/criado.")

        # 2. Seleciona e usa a estratégia de salvamento
        _, extensao = os.path.splitext(caminho_saida)
        try:
            estrategia_classe = self._estrategias_salvamento[extensao.lower()]
            salvador = estrategia_classe()
            salvador.salvar(self.df, caminho_saida)
            logging.info(f"SUCESSO: Dataset salvo com sucesso em '{caminho_saida}'.")
        except KeyError:
            print(f"ERRO: Nenhuma estratégia de salvamento encontrada para a extensão '{extensao}'.")
            raise ValueError(f"Formato de arquivo para salvamento '{extensao}' não é suportado.")
        except Exception as e:
            raise CustomException(e, sys)
    
    def __str__(self):
        return f'Dataset com {self.df.shape[0]} linhas e {self.df.shape[1]} colunas.'
    
    # --- Método de enriquecimento ---

class EnriquecimentoDatasetTelco(EnriquecimentoDataset):
    """
    Subclasse especializada para aplicar as regras de negócio 
    """
    def __init__(self, dataframe: pd.DataFrame):
        # Chama o construtor da classe pai (EnriquecimentoDataset)
        super().__init__(dataframe)
        logging.info('Instância de Telco Churn pronta')

    def preparar_coluna_churn(self):
        """Converte a coluna churn do formato texto para o formato numerico"""
        try:
            logging.info("Preparando a coluna 'Churn' (Yes/No -> 1/0)")
            self.df['Churn'] = self.df['Churn'].map({'Yes': 1, 'No': 0})
            return self
        except Exception as e:
            raise CustomException(e, sys)

    def gerar_features_interacao(self):
        """Gera features sintéticas baseadas em interações do cliente."""
        try:
            logging.info('Gerando features de interação')
            self.df['num_logins_last_30d'] = np.where(self.df['Churn'] == 1,
                                                 np.random.poisson(2, len(self.df)),
                                                 np.random.poisson(10, len(self.df)))
            logging.info('features num_logins_las_30d concluída com sucesso')
            self.df['support_tickets_last_90d'] = np.where(self.df['Churn'] == 1,
                                                      np.random.poisson(3, len(self.df)),
                                                      np.random.poisson(1, len(self.df)))
            logging.info('features support_tickets_last_90d concluída com sucesso')
            self.df['last_interaction_days_ago'] = np.where(self.df['Churn'] == 1,
                                                       np.random.randint(15, 60, len(self.df)),
                                                       np.random.randint(1, 15, len(self.df)))
            logging.info('features last_interaction_days_ago concluída com sucesso')
            logging.info('features completas')
            return self
        except Exception as e:
            raise CustomException(e, sys)
        
    @staticmethod
    def _generate_usage_series(churned: int) -> list:
        """
        Função auxiliar encapsulada como um método estático privado, não depende de uma instância (self).
        """
        base = np.random.poisson(10, 6)
        trend = np.linspace(1, 0.3, 6) if churned == 1 else np.ones(6)
        return (base * trend).astype(int).tolist()
    
    def gerar_features_uso_recente(self):
        """Gera features complexas baseadas no histórico de uso dos últimos 6 meses."""
        try:
            logging.info("Gerando features de série de uso (últimos 6 meses)...")
            self.df['usage_last_6m'] = self.df['Churn'].apply(self._generate_usage_series)
            self.df['trend_usage'] = self.df['usage_last_6m'].apply(lambda x: np.polyfit(range(6), x, 1)[0])
            self.df['std_usage'] = self.df['usage_last_6m'].apply(np.std)
            self.df['avg_usage_last_3m'] = self.df['usage_last_6m'].apply(lambda x: np.mean(x[-3:]))
            logging.info("features de série criadas")
            return self
        except Exception as e:
            raise CustomException(e, sys)
        
    def aplicar_enriquecimento_padrao(self):
        logging.info("Aplicando fluxo de enriquecimento padrão para Telco Churn...")
        self.preparar_coluna_churn().gerar_features_interacao().gerar_features_uso_recente()
        logging.info("Fluxo de enriquecimento concluído.")
        return self
    
if __name__ == "__main__":
    # Definição dos caminhos de entrada e saída
    input_path = str(diretorio_raiz())+"/data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    output_path_csv = str(diretorio_raiz())+"/data/interim/telco_churn_enriched.csv"

    print("==========================================================")
    print("Pipeline Completo: Carregar, Enriquecer e Salvar")
    print("==========================================================")
    
    try:
        # 1. Carregar usando o método de fábrica herdado
        dataset_telco = EnriquecimentoDatasetTelco.from_file(input_path)
        
        # 2. Aplicar o enriquecimento específico
        dataset_telco.aplicar_enriquecimento_padrao()
        
        logging.info("Dataset Final a ser Salvo")

        # 3. Salvar o resultado final. O método `salvar` foi herdado.
        dataset_telco.salvar(output_path_csv)

    except Exception as e:
        raise CustomException(e, sys)