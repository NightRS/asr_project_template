import editdistance


def calc_cer(target_text: str, predicted_text: str) -> float:
    if target_text == "":
        return len(predicted_text)
    return editdistance.distance(target_text, predicted_text) / len(target_text)


def calc_wer(target_text: str, predicted_text: str) -> float:
    if target_text == "":
        return len(predicted_text.split(" "))
    splitted_target = target_text.split(" ")
    return editdistance.distance(splitted_target, predicted_text.split(" ")) / len(
        splitted_target
    )
