from aiusage.entity.config_entity import DataIngestionConfig
from aiusage import logger
import zipfile

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
    
    def extract_zip_file(self):
        with zipfile.ZipFile(self.config.zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(self.config.root_dir)
        logger.info("Zip file extacted!")
