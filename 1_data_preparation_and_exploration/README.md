# Data preparation
This folder contains scripts and jupyter notebooks to prepare data for the training with a text-to-text transformer model like [T5](https://huggingface.co/docs/transformers/model_doc/t5).

One drawback of the transformer architecture is that there are input length constraints while training a transformer model. Typical input lengths are 512 (used in [T5 paper](https://arxiv.org/pdf/1910.10683.pdf)) or 1024 tokens (used in [mT5 paper](https://arxiv.org/abs/2010.11934)). Labels need smaller token lengths like 128. The computational cost directly depends on the token lengths of input and label. With higher token lengths better hardware is needed. For most experiments in this work an input length of 1024 and a label length of 128 worked well.

To avoid problems with examples there are multiple ways to handle the token length constraints:
1. Truncate too long examples (This may result in information loss)
2. Remove examples that exceed the token length constraints defined for training (This may remove too many examples and be negative for the training)

Another point regarding the token length is the tokenizer which is used with the specific transformer model. Since the T5 tokenizer doesn't use whitespace or word segmentation but subword tokenization with [sentencepiece](https://github.com/google/sentencepiece) the training examples have to be tokenized and length checked before they can be further filtered. Since different transformer-based models use different tokenizers this has to be kept in mind. The scripts only uses the trained mt5 tokenizer but this can be changed in the code itself.

The following jupyter notebooks can be used to check the token lengths of a dataset and create a dataset for the task of question generation with filered examples. The best way to execute the notebooks is via google colab because they contain special [forms](https://colab.research.google.com/notebooks/forms.ipynb) which can be adjusted.

|Task| |
|---|---|
|Check token length distribution of a dataset|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TiloMichel/textgen-for-chatbot-training-german/blob/main/1_data_preparation_and_exploration/check_dataset_examples_token_lengths.ipynb) |
|Create datasets with examples filtered by token length|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TiloMichel/textgen-for-chatbot-training-german/blob/main/1_data_preparation_and_exploration/create_qg_dataset.ipynb) |
|question_generation_dataset_statistics|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TiloMichel/textgen-for-chatbot-training-german/blob/main/1_data_preparation_and_exploration/question_generation_dataset_statistics.ipynb)|

The `create_dataset_scripts` directory contains WIP code of scripts to create datasets. The scripts don't implement the token length filter feature yet which is to be changed in the future.

## Details
The data must be converted into a suitable format for training. Prefix, Input Text and Target Text are required for each example (or series). The preprocessed data set looks something like this:
| prefix              | input_text | target_text |
|---------------------|------------|-------------|
| generate question:  |Sichuan gilt vorwiegend als Reisanbaugebiet, ist aber auch bedeutender Produzent von Mais, Süßkartoffeln, Weizen, Raps und Soja. Der Anbau von Obst und Zitrusfrüchten ist ebenfalls verbreitet. An tierischen Produkten sind vor allem Schweinefleisch und Seidenkokons bedeutend. Des Weiteren wird in Sichuan Tee von internationalem Rang produziert. Dieser Tee ist ausschließlich für den Export bestimmt. In den Gebirgslagen der Provinz wird außerdem der großrahmige Jiulong-Yak gezüchtet. Eine Zuchtstation, die das Leistungsniveau dieser Yak-Rasse weiter verbessern soll, besteht seit 1980 in Jiulong. Der Wert der landwirtschaftlichen Produktion stieg von 1994 bis 1999 jährlich um durchschnittlich 5,6 % auf 144,5 Milliarden RMB.|Was wird in Sichuan angebaut? `<sep>` Welches für den Export bestimmte Produkt wird in Sichuan produziert? `<sep>` Was wird in den Gebirgen von Sichuan gezüchtet? `<sep>` Welche tierische Produkte werden in Sichuan produziert? `</s>`|
| paraphrase:|Durch die Zusammenlegung des Four Rivers Council und des Audubon Council entstand der Shawnee Trails Council.|Shawnee Trails Council entstand durch die Fusion zwischen dem Four Rivers Council und dem Audubon Council. `</s>`|

Through this format, any text-to-text tasks can be trained with the [mT5](https://huggingface.co/docs/transformers/model_doc/mt5) model, even multiple tasks simultaneously by using different prefixes.

The dataset should be in Arrow format, this can be loaded by the huggingface datasets library. With the methods [save_to_disk](https://huggingface.co/docs/datasets/master/en/package_reference/main_classes#datasets.Dataset.save_to_disk) and [load_from_disk](https://huggingface.co/docs/datasets/master/en/package_reference/main_classes#datasets.Dataset.load_from_disk) saving can be easily implemented. In the Python script preprocess_datasets.py examples are available how to implement a conversion from csv or json to an Arrow dataset.

It is possible to load records from huggingface or from csv/json files. However, the data schema must correspond to that of the table above.

The records are loaded in run_text2text.py as follows:
```python
# Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
# or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
# (the dataset will be downloaded automatically from the datasets Hub).
#
# In distributed training, the load_dataset function guarantee that only one local process can concurrently
# download the dataset.
if data_args.dataset_name is not None:
    # Downloading and loading a dataset from the hub.
    datasets = load_dataset(data_args.dataset_name,
                            data_args.dataset_config_name)
elif data_args.dataset_dir is not None:
    datasets = load_from_disk(data_args.dataset_dir)
else:
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
        extension = data_args.train_file.split(".")[-1]
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
        extension = data_args.validation_file.split(".")[-1]
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file
        extension = data_args.test_file.split(".")[-1]
    datasets = load_dataset(extension, data_files=data_files)
```
