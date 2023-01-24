from typing import List
from ..text_encoder.ctc_char_text_encoder import Hypothesis

from hw_asr.base.base_metric import BaseMetric
from hw_asr.metric.utils import calc_wer


class BeamSearchWERMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, beam_search: List[List[Hypothesis]], text: List[str], **kwargs):
        beam_search_wers = []
        for target_text, beam_search_result in zip(text, beam_search):
            beam_search_wers.append(calc_wer(target_text, beam_search_result[0].text))
        return sum(beam_search_wers) / len(beam_search_wers)
