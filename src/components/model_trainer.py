from src.config.configuration import ModelTrainerConfig
import torch,os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from datasets import load_from_disk


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        if torch.backends.mps.is_available():
            device = torch.device("mps")  # Use Metal Performance Shaders (MPS) for Apple Silicon
        else:
            device = torch.device("cpu")  # Fallback to CPU

        print(f"Using device: {device}")
        model=self.config.model_ckpt
        tokenizer=AutoTokenizer.from_pretrained(model)
        model_bart=AutoModelForSeq2SeqLM.from_pretrained(model).to(device)   
        seq_2_seq_collator=DataCollatorForSeq2Seq(tokenizer,model_bart) 
        print(f"eval_steps value: {self.config.eval_steps}")

        dataset_samsum = load_from_disk(self.config.data_path)
        training_args=TrainingArguments(
            output_dir=self.config.root_dir, num_train_epochs=self.config.num_train_epochs, warmup_steps=500,
            per_device_train_batch_size=1, per_device_eval_batch_size=1,
            weight_decay=0.01, logging_steps=10,
            evaluation_strategy='steps', eval_steps=500, save_steps=1e6,
            gradient_accumulation_steps=4)
        
        trainer = Trainer(model=model_bart, args=training_args,
                  tokenizer=tokenizer, data_collator=seq_2_seq_collator,
                  train_dataset=dataset_samsum["test"],
                  eval_dataset=dataset_samsum["validation"])
        trainer.train()

        model_bart.save_pretrained(os.path.join(self.config.root_dir, "bart-samsum-model"))
        tokenizer.save_pretrained(os.path.join(self.config.root_dir,"tokenizer"))
