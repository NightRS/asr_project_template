from typing import List
from ..text_encoder.ctc_char_text_encoder import Hypothesis

from hw_asr.base.base_metric import BaseMetric
from hw_asr.metric.utils import calc_wer


class OracleWERMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, beam_search: List[List[Hypothesis]], text: List[str], **kwargs):
        oracle_wers = []
        for target_text, beam_search_result in zip(text, beam_search):
            wers = [calc_wer(target_text, hypothesis.text) for hypothesis in beam_search_result]
            oracle_wers.append(min(wers))
        return sum(oracle_wers) / len(oracle_wers)
