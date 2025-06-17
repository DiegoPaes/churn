import logging
import os
from datetime import datetime

# 1) Diretório fixo de logs
logs_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir, exist_ok=True)

# 2) Nome do arquivo de log (incluindo timestamp, se desejar)
log_filename = datetime.now().strftime("%Y_%m_%d") + ".log"
LOG_FILE_PATH = os.path.join(logs_dir, log_filename)

# 3) Configuração do logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    filemode='a',  # 'a' = append; 'w' = overwrite
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)