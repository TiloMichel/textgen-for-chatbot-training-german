import argparse
from pathlib import Path
from Text2TextDatasetTransformers import GermanQuadEnd2EndQuestionGeneration


def main(args):
    dataset_path = Path(args.dir)
    germanquad_path = dataset_path / "germanquad"
    end_2_end_germanquad_transformator = GermanQuadEnd2EndQuestionGeneration(
        task_prefix="generate question: ",
        tokenizer_eos="</s>",
        train_path=str(germanquad_path / "GermanQuAD-train.json"),
        validation_path=str(germanquad_path / "GermanQuAD-test.json"))
    dataset_dict = end_2_end_germanquad_transformator.create_dataset_dict()
    dataset_dict.save_to_disk(str(dataset_path / "e2e-qg-germanquad"))
    dataset_dict.load_from_disk(str(dataset_path / "e2e-qg-germanquad"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir", 
        type=str, 
        default="", 
        help="Path to the datasets directory.")
    args = parser.parse_args()
    main(args)