from typing import List
from ..text_encoder.ctc_char_text_encoder import Hypothesis

from hw_asr.base.base_metric import BaseMetric
from hw_asr.metric.utils import calc_cer


class OracleCERMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, beam_search: List[List[Hypothesis]], text: List[str], **kwargs):
        oracle_cers = []
        for target_text, beam_search_result in zip(text, beam_search):
            cers = [calc_cer(target_text, hypothesis.text) for hypothesis in beam_search_result]
            oracle_cers.append(min(cers))
        return sum(oracle_cers) / len(oracle_cers)
