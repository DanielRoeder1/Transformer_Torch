from datasets import load_dataset
from transformers import AutoTokenizer, PretrainedConfig


class Wmt14Handler():
    def __init__(self, tokenizer,config, language_split) -> None:
        self.tokenizer = tokenizer
        self.config = config
        self.language_split = language_split
    
    def preprocess_function(self,examples):
        """
        Shifts the targets one token to the right using bos_token
        """
        lang1, lang2 = list(examples["translation"][0].keys())
        inputs = [ex[lang1] for ex in examples["translation"]]
        targets = [ex[lang2] for ex in examples["translation"]]

        model_inputs = self.tokenizer(
            inputs, text_target=targets, add_special_tokens= False, max_length= self.config.seq_len, truncation= True, return_attention_mask= False, return_token_type_ids= False, padding= "max_length"
        )
        model_inputs["labels"] = [[self.tokenizer.bos_token_id] + ids for ids in model_inputs["labels"]]
        return model_inputs

    def get_wmt14(self):
        dataset = load_dataset("wmt14", self.language_split)
        tokenized_dataset = dataset.map(self.preprocess_function, batched= True)
        tokenized_dataset = tokenized_dataset.with_format("torch")
        tokenized_dataset = tokenized_dataset.remove_columns("translation")
        print("Loaded Data!")
        return tokenized_dataset
    



if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("rossanez/t5-small-finetuned-de-en-lr2e-4")
    tokenizer.add_special_tokens({"bos_token": "<start>"})
    config = PretrainedConfig.from_json_file("config.json")

    t = Wmt14Handler(tokenizer, config, "de-en")
    data = t.get_wmt14()


    
