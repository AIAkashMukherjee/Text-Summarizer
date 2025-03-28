
from datasets import load_from_disk
from transformers import AutoTokenizer
import os
from src.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self,config:DataTransformationConfig):
        self.config=config
        self.tokenizer=AutoTokenizer.from_pretrained(config.tokenizer_name)

    def convert_to_features(self,batch):
        input_encodings=self.tokenizer(batch["dialogue"],truncation=True,max_length=1024)

        with self.tokenizer.as_target_tokenizer():
            target_encodings=self.tokenizer(batch["summary"],truncation=True,max_length=128)

        return{
        'input_ids' : input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'labels': target_encodings['input_ids']    
        }
    
    def convert(self):
        dataset_samsum=load_from_disk(self.config.data_path)
        dataset_samsum=dataset_samsum.map(self.convert_to_features,batched=True)
        dataset_samsum.save_to_disk(os.path.join(self.config.root_dir,"samsum_dataset"))