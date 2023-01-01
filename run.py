import argparse
import yaml

from process.preprocess import Preprocess
from process.train import Train
from process.embed import Embed
from process.analysis import Analysis


def load_configuration(config_file_path):
    # Load configuration file(.yaml)
    with open(config_file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config


def pipeline(config_file_path):
    # Load configuration
    config = load_configuration(config_file_path)
    # Config pipeline
    processor_list = [Preprocess, Train, Embed, Analysis]
    pre_processor = None
    # Run
    for p in processor_list:
        cur_process = p(task_name=p.__name__, config=config, processor=pre_processor)
        cur_process.processing()
        pre_processor = cur_process


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OMICS2VEC test version")
    parser.add_argument(
        "--config",
        required=True,
        default="config/O2V_C.yaml",
        help="Configuration file for pipeline execution",
    )
    args = parser.parse_args()
    processor = int(
        input(
            """
        Please enter the desired process...
        0: Pipeline
        1: Preprocess
        2: Train
        3: Embed
        4: Analysis
        """
        )
    )
    pipeline(args.config)
