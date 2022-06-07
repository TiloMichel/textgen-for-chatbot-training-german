import argparse
from pathlib import Path
from Text2TextDatasetTransformers import (
    SquadEnd2EndQuestionGeneration, 
    GermanQuadEnd2EndQuestionGeneration
)


def main(args):
    dataset_path = Path(args.dir)
    squad_path = dataset_path / "squad_v1.1"
    end_2_end_germanquad_transformator = SquadEnd2EndQuestionGeneration(
        task_prefix="generate question: ",
        tokenizer_eos="</s>",
        train_path=str(squad_path / "train-v1.1.json"),
        validation_path=str(squad_path / "dev-v1.1.json"))
    squad_dataset_dict = end_2_end_germanquad_transformator.create_dataset_dict()
    squad_dataset_dict.save_to_disk(str(dataset_path / "e2e-qg-squad"))
    squad_dataset_dict.load_from_disk(str(dataset_path / "e2e-qg-squad"))
    germanquad_path = dataset_path / "germanquad"
    end_2_end_germanquad_transformator = GermanQuadEnd2EndQuestionGeneration(
        task_prefix="generate question: ",
        tokenizer_eos="</s>",
        train_path=str(germanquad_path / "GermanQuAD-train.json"),
        validation_path=str(germanquad_path / "GermanQuAD-test.json"))
    germanquad_dataset_dict = end_2_end_germanquad_transformator.create_dataset_dict()
    germanquad_dataset_dict.save_to_disk(str(dataset_path / "e2e-qg-germanquad"))
    germanquad_dataset_dict.load_from_disk(str(dataset_path / "e2e-qg-germanquad"))
    #ToDo: Concetanate dataset splits
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir", 
        type=str, 
        default="", 
        help="Path to the datasets directory.")
    args = parser.parse_args()
    main(args)