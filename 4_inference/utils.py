from typing import List

from torch.utils.data import Dataset
from transformers import pipeline


class ListDataset(Dataset):
    """ListDataset for usage in combination with huggingface pipelines 
    to visualize a progressbar with tqdm.

    Args:
        Dataset (ListDataset): Object which implements Dataset Class of tensorflow
    """
    def __init__(self, original_list: List[str]):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i: int):
        return self.original_list[i]


class Translation:
    def __init__(self):
        de_en_model = "Helsinki-NLP/opus-mt-de-en"
        self.de_en_pipeline = pipeline("text2text-generation", model=de_en_model, tokenizer=de_en_model)
        # https://github.com/huggingface/transformers/blob/9e71d4645526911f2ea9743aa4cf8e9d479fc840/src/transformers/pipelines/__init__.py#L214
        self.en_de_pipeline = pipeline("translation_en_to_de")


    def translate_de_to_en_multiple_results(self, texts: List[str], num_return_sequences: int) -> List[List[str]]:
        texts_en = [[text['generated_text'] for text in text_list] for text_list in self.de_en_pipeline(texts, 
                       num_beams=num_return_sequences, 
                       num_return_sequences=num_return_sequences)]
        return texts_en


    def translate_en_to_de_multiple_results(self, texts: List[str], num_return_sequences: int) -> List[List[str]]:
        texts_de = [[text['translation_text'] for text in text_list] for text_list in self.en_de_pipeline(texts, 
                       num_beams=num_return_sequences, 
                       num_return_sequences=num_return_sequences)]
        return texts_de


    def translate_de_to_en(self, texts: List[str]) -> List[str]:
        texts_en = [text['generated_text'] for text in self.de_en_pipeline(texts)]
        return texts_en


    def translate_en_to_de(self, texts: List[str]) -> List[str]:
        texts_de = [text['translation_text'] for text in self.en_de_pipeline(texts)]
        return texts_de
