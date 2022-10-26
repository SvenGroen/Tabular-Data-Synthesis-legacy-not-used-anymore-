from table_evaluator import load_data, TableEvaluator
from tabular_synthesis.synthesizer.loading.util import get_dataset

import argparse
import os
from azureml.core import Run

def main():
    parser = argparse.ArgumentParser(description="Evaluate a synthetic dataset")
    parser.add_argument("--real_dataset_path", type=str, help="Path to the real dataset")
    parser.add_argument("--synthetic_dataset_path", type=str, help="Path to the synthetic dataset")
    parser.add_argument("--config_path", type=str, help="Path to the config file")
    parser.add_argument("--output_path", default=None,type=str, help="Path to the output file")
    args = parser.parse_args()
    run = Run.get_context()

    real, config = get_dataset(args.real_dataset_path, args.config_path)
    fake, _ = get_dataset(path=args.synthetic_dataset_path, config_path= args.config_path, skiprows=1)
    fake["income"] = fake["income"].apply(lambda x: x.replace(".", ""))
    # if len(real) > len(fake):
    #     real = real.sample(len(fake))
    # else:
    #     fake = fake.sample(len(real))
    # assert len(real) == len(fake)
    if args.output_path is None:
        import tempfile
        dirpath = tempfile.mkdtemp()

    table_evaluator = TableEvaluator(real, fake, cat_cols=config["dataset_config"]["categorical_columns"], verbose=True)
    table_evaluator.visual_evaluation(save_dir=dirpath if args.output_path is None else args.output_path)

    # print every file in dirpath
    for file in os.listdir(dirpath):
        run.log_image(file, os.path.join(dirpath, file))

    output = table_evaluator.evaluate(target_col=config["dataset_config"]["problem_type"]["Classification"], return_outputs=True)

    # save dict as json
    import json
    with open(os.path.join(dirpath, "output.json"), "w") as f:
        json.dump(output, f)

    if args.output_path is None:
        import shutil
        shutil.rmtree(dirpath)

    out = {k: v["result"] for k,v in output["Overview Results"].items()}
    run.log_table(name="Overview Results", value=out)


if __name__ == "__main__":
    main()
