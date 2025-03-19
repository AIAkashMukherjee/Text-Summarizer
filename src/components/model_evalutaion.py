from src.config.configuration import ModelEvaluationConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk
import torch,evaluate,os
from tqdm import tqdm
import pandas as pd

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def split_data_to_batchs(self, list_of_elements, batch_size):
        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i:i + batch_size]

    def calculate_metric_on_test_ds(self, dataset, metric, model, tokenizer,
                                     batch_size=16, device=torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"),
                                     column_text="article",
                                     column_summary="highlights"):
        article_batches = list(self.split_data_to_batchs(dataset[column_text], batch_size))
        target_batches = list(self.split_data_to_batchs(dataset[column_summary], batch_size))

        for article_batch, target_batch in tqdm(zip(article_batches, target_batches), total=len(article_batches)):
            inputs = tokenizer(article_batch, max_length=1024, truncation=True,
                               padding="max_length", return_tensors="pt")

            summaries = model.generate(input_ids=inputs["input_ids"].to(device),
                                       attention_mask=inputs["attention_mask"].to(device),
                                       length_penalty=0.8, num_beams=8, max_length=128)
            ''' parameter for length penalty ensures that the model does not generate sequences that are too long. '''

            # Finally, we decode the generated texts,
            # replace the token, and add the decoded texts with the references to the metric.
            decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True) for s in summaries]
            decoded_summaries = [d.replace("", " ") for d in decoded_summaries]

            metric.add_batch(predictions=decoded_summaries, references=target_batch)

        # Finally compute and return the ROUGE scores.
        score = metric.compute()
        return score



    def evaluate(self):
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        model_bart = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path).to(device)
       
        #loading data 
        dataset_samsum_pt = load_from_disk(self.config.data_path)


        rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

        rouge_metric = evaluate.load('rouge')

        #rouge_metric = rouge_metric

        score = self.calculate_metric_on_test_ds(
        dataset_samsum_pt['test'][0:10], rouge_metric, model_bart, tokenizer, batch_size = 2, column_text = 'dialogue', column_summary= 'summary'
            )

        # Directly use the scores without accessing fmeasure or mid
        rouge_dict = {rn: score[rn] for rn in rouge_names}

        df = pd.DataFrame(rouge_dict, index = ['bart'] )
        df.to_csv(self.config.metric_file_name, index=False)

