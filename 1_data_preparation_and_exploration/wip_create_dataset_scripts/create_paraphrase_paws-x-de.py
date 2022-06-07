import argparse
from pathlib import Path
from Text2TextDatasetTransformers import PawsParaphrasing

def main(args):
    dataset_path = Path(args.dir)
    pawsx = dataset_path / "paws-x" / "de"
    paws_paraphrasing = PawsParaphrasing(
        task_prefix="paraphrase: ",
        tokenizer_eos="</s>",
        train_path=str(pawsx / "translated_train.tsv"),
        validation_path=str(pawsx / "dev_2k.tsv"),
        test_path=str(pawsx / "test_2k.tsv"))
    dataset_dict2 = paws_paraphrasing.create_dataset_dict()
    dataset_dict2.save_to_disk(str(dataset_path / "paws-x-de"))
    dataset_dict2.load_from_disk(str(dataset_path / "paws-x-de"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir", 
        type=str, 
        default="", 
        help="Path to the datasets directory.")
    args = parser.parse_args()
    main(args)
