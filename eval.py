import argparse
import json
from hw_asr.metric.utils import calc_wer, calc_cer


def evaluate_prediction_file(path):
    predicted_answers = json.load(open(path, encoding='utf-8'))

    argmax_cers = []
    argmax_wers = []
    beam_search_cers = []
    beam_search_wers = []

    for test_example in predicted_answers:
        argmax_cers.append(calc_cer(test_example["ground_truth"], test_example["pred_text_argmax"]))
        argmax_wers.append(calc_wer(test_example["ground_truth"], test_example["pred_text_argmax"]))

        beam_search_cers.append(calc_cer(test_example["ground_truth"], test_example["pred_text_beam_search"]))
        beam_search_wers.append(calc_wer(test_example["ground_truth"], test_example["pred_text_beam_search"]))

    print("Evaluation results")
    print("AVG CER (argmax)     ", "{:.2f}".format(sum(argmax_cers) / len(argmax_cers) * 100))
    print("AVG CER (beam search)", "{:.2f}".format(sum(beam_search_cers) / len(beam_search_cers) * 100))
    print("AVG WER (argmax)     ", "{:.2f}".format(sum(argmax_wers) / len(argmax_wers) * 100))
    print("AVG WER (beam search)", "{:.2f}".format(sum(beam_search_wers) / len(beam_search_wers) * 100))


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
    parser.add_argument("--prediction_path",
                        type=str,
                        required=False,
                        default="output.json",
                        help="location of the prediction file")

    args = parser.parse_args()
    evaluate_prediction_file(args.prediction_path)
