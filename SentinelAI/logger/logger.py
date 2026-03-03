import logging
import os
from datetime import datetime
from from_root import from_root

LOGFILE= f"datetime.now().strftime('%Y-%m-%d_%H-%M-%S')"
logs_path= os.path.join(os.getcwd(), "logs", LOGFILE)

os.makedirs(logs_path, exist_ok=True)
LOG_FILE_PATH= os.path.join(logs_path, exist_ok=True)
logging.basicConfig(
    filename= LOG_FILE_PATH,
    format= "[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
    level= logging.DEBUG
)