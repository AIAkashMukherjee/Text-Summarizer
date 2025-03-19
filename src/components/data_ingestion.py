
from src.entity.config_entity import DataIngestionConfig
import urllib.request as request
import os
from dataclasses import dataclass
from src.logger.custom_logging import logger
from src.exceptions.expection import CustomException
import zipfile


class DataIngestion:
    def __init__(self,config:DataIngestionConfig):
        self.config=config

    def download_pdf(self):
        if not os.path.exists(self.config.local_data_file):
            filename,fileheader=request.urlretrieve(
                url=self.config.source_URL,
                filename=self.config.local_data_file
            )
            logger.info(f"File is downloaded")
        else:
            logger.info('File already exists')  

    def extract_zip_file(self):
        unzip_path=self.config.unzip_dir
        os.makedirs(unzip_path,exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file,'r')as f:
            f.extractall(unzip_path)   