import argparse
from pathlib import Path

from src.metric.utils import calc_cer, calc_wer
from src.utils.util import read_json


def evaluate_prediction_file(path: Path):
    predicted_answers = read_json(path)

    argmax_cers: list[float] = []
    argmax_wers: list[float] = []
    beam_search_cers: list[float] = []
    beam_search_wers: list[float] = []

    for test_example in predicted_answers:
        argmax_cers.append(
            calc_cer(test_example["ground_truth"], test_example["pred_text_argmax"])
        )
        argmax_wers.append(
            calc_wer(test_example["ground_truth"], test_example["pred_text_argmax"])
        )

        beam_search_cers.append(
            calc_cer(
                test_example["ground_truth"], test_example["pred_text_beam_search"]
            )
        )
        beam_search_wers.append(
            calc_wer(
                test_example["ground_truth"], test_example["pred_text_beam_search"]
            )
        )

    print("Evaluation results")
    print(
        "AVG CER (argmax)     ",
        "{:.2f}".format(sum(argmax_cers) / len(argmax_cers) * 100),
    )
    print(
        "AVG CER (beam search)",
        "{:.2f}".format(sum(beam_search_cers) / len(beam_search_cers) * 100),
    )
    print(
        "AVG WER (argmax)     ",
        "{:.2f}".format(sum(argmax_wers) / len(argmax_wers) * 100),
    )
    print(
        "AVG WER (beam search)",
        "{:.2f}".format(sum(beam_search_wers) / len(beam_search_wers) * 100),
    )


if __name__ == "__main__":
    """
    Expected format of output.json:

    [
      {
        "ground_truth": "...",
        "pred_text_argmax": "...",
        "pred_text_beam_search": "..."
      },
      {
        "ground_truth": "...",
        "pred_text_argmax": "...",
        "pred_text_beam_search": "..."
      },
      ...
    ]

    """
    parser = argparse.ArgumentParser(description="evaluate prediction file")
    parser.add_argument(
        "--prediction_path",
        type=Path,
        required=False,
        default=Path("output.json"),
        help="location of the prediction file",
    )

    args = parser.parse_args()
    evaluate_prediction_file(args.prediction_path)
