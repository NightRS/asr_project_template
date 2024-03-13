from collections.abc import Sequence
from datetime import datetime


def check1(
    timestamps: Sequence[datetime],
    grads: Sequence[float],
    losses: Sequence[float],
    metrics: Sequence[float],
    lrs: Sequence[float],
):
    return [
        # self.writer.set_step(-1, "train")
        # self.writer.log_manual({"train_metrics/epoch": 1}, step=0)
        (
            0,
            {
                "train_metrics/epoch": 1,
            },
        ),
        # self.writer.set_step(0, "train")
        # self.writer.add_scalars({"epoch": 1, "learning_rate": self.lr_scheduler.get_last_lr()[0]})
        # self.writer.add_scalars({"loss": ..., "grad_norm": ..., "CER (argmax)": ...})
        # self.writer.write()
        (
            0,
            {
                "steps_logged": 1,
                "train_steps_per_sec": (
                    1 / (timestamps[2] - timestamps[1]).total_seconds()
                ),
                "seconds_per_train_step": (
                    (timestamps[2] - timestamps[1]).total_seconds() / 1
                ),
                "total_time_sec": (timestamps[2] - timestamps[0]).total_seconds(),
                "total_time_hr": (
                    (timestamps[2] - timestamps[0]).total_seconds() / 3600
                ),
                "_time_usage_sec": (timestamps[1] - timestamps[0]).total_seconds(),
                "train_time_usage_sec": (
                    (timestamps[2] - timestamps[1]).total_seconds()
                ),
                "val_time_usage_sec": 0.0,
                "_time_usage_relative": (
                    (timestamps[1] - timestamps[0]).total_seconds()
                    / (timestamps[2] - timestamps[0]).total_seconds()
                ),
                "train_time_usage_relative": (
                    (timestamps[2] - timestamps[1]).total_seconds()
                    / (timestamps[2] - timestamps[0]).total_seconds()
                ),
                "val_time_usage_relative": 0.0,
                "train_metrics/epoch": 1,
                "train_metrics/learning_rate": lrs[0],
                "train_metrics/loss": losses[0],
                "train_metrics/grad_norm": grads[0],
                "train_metrics/CER (argmax)": metrics[0],
            },
        ),
        # self.writer.set_step(1, "train")
        # self.writer.add_scalars({"epoch": 1, "learning rate": self.lr_scheduler.get_last_lr()[0]})
        # self.writer.add_scalars({"loss": ..., "grad_norm": ..., "CER (argmax)": ...})
        # self.writer.write()
        (
            1,
            {
                "steps_logged": 1,
                "train_steps_per_sec": (
                    1 / (timestamps[3] - timestamps[2]).total_seconds()
                ),
                "seconds_per_train_step": (
                    (timestamps[3] - timestamps[2]).total_seconds() / 1
                ),
                "total_time_sec": (timestamps[3] - timestamps[0]).total_seconds(),
                "total_time_hr": (
                    (timestamps[3] - timestamps[0]).total_seconds() / 3600
                ),
                "_time_usage_sec": (timestamps[1] - timestamps[0]).total_seconds(),
                "train_time_usage_sec": (timestamps[3] - timestamps[1]).total_seconds(),
                "val_time_usage_sec": 0.0,
                "_time_usage_relative": (
                    (timestamps[1] - timestamps[0]).total_seconds()
                    / (timestamps[3] - timestamps[0]).total_seconds()
                ),
                "train_time_usage_relative": (
                    (timestamps[3] - timestamps[1]).total_seconds()
                    / (timestamps[3] - timestamps[0]).total_seconds()
                ),
                "val_time_usage_relative": 0.0,
                "train_metrics/epoch": 1,
                "train_metrics/learning_rate": lrs[1],
                "train_metrics/loss": losses[1],
                "train_metrics/grad_norm": grads[1],
                "train_metrics/CER (argmax)": metrics[1],
            },
        ),
        # self.writer.set_step(2, "train")
        # self.writer.add_scalars({"epoch": 1, "learning_rate": self.lr_scheduler.get_last_lr()[0]})
        # self.writer.add_scalars({"loss": ..., "grad_norm": ..., "CER (argmax)": ...})
        # self.writer.write()
        (
            2,
            {
                "steps_logged": 1,
                "train_steps_per_sec": (
                    1 / (timestamps[4] - timestamps[3]).total_seconds()
                ),
                "seconds_per_train_step": (
                    (timestamps[4] - timestamps[3]).total_seconds() / 1
                ),
                "total_time_sec": (timestamps[4] - timestamps[0]).total_seconds(),
                "total_time_hr": (
                    (timestamps[4] - timestamps[0]).total_seconds() / 3600
                ),
                "_time_usage_sec": (timestamps[1] - timestamps[0]).total_seconds(),
                "train_time_usage_sec": (timestamps[4] - timestamps[1]).total_seconds(),
                "val_time_usage_sec": 0.0,
                "_time_usage_relative": (
                    (timestamps[1] - timestamps[0]).total_seconds()
                    / (timestamps[4] - timestamps[0]).total_seconds()
                ),
                "train_time_usage_relative": (
                    (timestamps[4] - timestamps[1]).total_seconds()
                    / (timestamps[4] - timestamps[0]).total_seconds()
                ),
                "val_time_usage_relative": 0.0,
                "train_metrics/epoch": 1,
                "train_metrics/learning_rate": lrs[2],
                "train_metrics/loss": losses[2],
                "train_metrics/grad_norm": grads[2],
                "train_metrics/CER (argmax)": metrics[2],
            },
        ),
        # self.writer.set_step(3, "train")
        # self.writer.add_scalars({"epoch": 1, "learning_rate": self.lr_scheduler.get_last_lr()[0]})
        # self.writer.add_scalars({"loss": ..., "grad_norm": ..., "CER (argmax)": ...})
        # self.writer.write()
        (
            3,
            {
                "steps_logged": 1,
                "train_steps_per_sec": (
                    1 / (timestamps[5] - timestamps[4]).total_seconds()
                ),
                "seconds_per_train_step": (
                    (timestamps[5] - timestamps[4]).total_seconds() / 1
                ),
                "total_time_sec": (timestamps[5] - timestamps[0]).total_seconds(),
                "total_time_hr": (
                    (timestamps[5] - timestamps[0]).total_seconds() / 3600
                ),
                "_time_usage_sec": (timestamps[1] - timestamps[0]).total_seconds(),
                "train_time_usage_sec": (timestamps[5] - timestamps[1]).total_seconds(),
                "val_time_usage_sec": 0.0,
                "_time_usage_relative": (
                    (timestamps[1] - timestamps[0]).total_seconds()
                    / (timestamps[5] - timestamps[0]).total_seconds()
                ),
                "train_time_usage_relative": (
                    (timestamps[5] - timestamps[1]).total_seconds()
                    / (timestamps[5] - timestamps[0]).total_seconds()
                ),
                "val_time_usage_relative": 0.0,
                "train_metrics/epoch": 1,
                "train_metrics/learning_rate": lrs[3],
                "train_metrics/loss": losses[3],
                "train_metrics/grad_norm": grads[3],
                "train_metrics/CER (argmax)": metrics[3],
            },
        ),
        # self.writer.set_step(4, "train")
        # self.writer.add_scalars({"epoch": 1, "learning_rate": self.lr_scheduler.get_last_lr()[0]})
        # self.writer.add_scalars({"loss": ..., "grad_norm": ..., "CER (argmax)": ...})
        # self.writer.write()
        (
            4,
            {
                "steps_logged": 1,
                "train_steps_per_sec": (
                    1 / (timestamps[6] - timestamps[5]).total_seconds()
                ),
                "seconds_per_train_step": (
                    (timestamps[6] - timestamps[5]).total_seconds() / 1
                ),
                "total_time_sec": (timestamps[6] - timestamps[0]).total_seconds(),
                "total_time_hr": (
                    (timestamps[6] - timestamps[0]).total_seconds() / 3600
                ),
                "_time_usage_sec": (timestamps[1] - timestamps[0]).total_seconds(),
                "train_time_usage_sec": (timestamps[6] - timestamps[1]).total_seconds(),
                "val_time_usage_sec": 0.0,
                "_time_usage_relative": (
                    (timestamps[1] - timestamps[0]).total_seconds()
                    / (timestamps[6] - timestamps[0]).total_seconds()
                ),
                "train_time_usage_relative": (
                    (timestamps[6] - timestamps[1]).total_seconds()
                    / (timestamps[6] - timestamps[0]).total_seconds()
                ),
                "val_time_usage_relative": 0.0,
                "train_metrics/epoch": 1,
                "train_metrics/learning_rate": lrs[4],
                "train_metrics/loss": losses[4],
                "train_metrics/grad_norm": grads[4],
                "train_metrics/CER (argmax)": metrics[4],
            },
        ),
        # self.writer.set_step(5, "train")
        # self.writer.add_scalars({"epoch": 1, "learning_rate": self.lr_scheduler.get_last_lr()[0]})
        # self.writer.add_scalars({"loss": ..., "grad_norm": ..., "CER (argmax)": ...})
        # self.writer.write()
        (
            5,
            {
                "steps_logged": 1,
                "train_steps_per_sec": (
                    1 / (timestamps[7] - timestamps[6]).total_seconds()
                ),
                "seconds_per_train_step": (
                    (timestamps[7] - timestamps[6]).total_seconds() / 1
                ),
                "total_time_sec": (timestamps[7] - timestamps[0]).total_seconds(),
                "total_time_hr": (
                    (timestamps[7] - timestamps[0]).total_seconds() / 3600
                ),
                "_time_usage_sec": (timestamps[1] - timestamps[0]).total_seconds(),
                "train_time_usage_sec": (timestamps[7] - timestamps[1]).total_seconds(),
                "val_time_usage_sec": 0.0,
                "_time_usage_relative": (
                    (timestamps[1] - timestamps[0]).total_seconds()
                    / (timestamps[7] - timestamps[0]).total_seconds()
                ),
                "train_time_usage_relative": (
                    (timestamps[7] - timestamps[1]).total_seconds()
                    / (timestamps[7] - timestamps[0]).total_seconds()
                ),
                "val_time_usage_relative": 0.0,
                "train_metrics/epoch": 1,
                "train_metrics/learning_rate": lrs[5],
                "train_metrics/loss": losses[5],
                "train_metrics/grad_norm": grads[5],
                "train_metrics/CER (argmax)": metrics[5],
            },
        ),
        # self.writer.set_step(6, "train")
        # self.writer.add_scalars({"epoch": 1, "learning_rate": self.lr_scheduler.get_last_lr()[0]})
        # self.writer.add_scalars({"loss": ..., "grad_norm": ..., "CER (argmax)": ...})
        # self.writer.write()
        (
            6,
            {
                "steps_logged": 1,
                "train_steps_per_sec": (
                    1 / (timestamps[8] - timestamps[7]).total_seconds()
                ),
                "seconds_per_train_step": (
                    (timestamps[8] - timestamps[7]).total_seconds() / 1
                ),
                "total_time_sec": (timestamps[8] - timestamps[0]).total_seconds(),
                "total_time_hr": (
                    (timestamps[8] - timestamps[0]).total_seconds() / 3600
                ),
                "_time_usage_sec": (timestamps[1] - timestamps[0]).total_seconds(),
                "train_time_usage_sec": (
                    (timestamps[8] - timestamps[1]).total_seconds()
                ),
                "val_time_usage_sec": 0.0,
                "_time_usage_relative": (
                    (timestamps[1] - timestamps[0]).total_seconds()
                    / (timestamps[8] - timestamps[0]).total_seconds()
                ),
                "train_time_usage_relative": (
                    (timestamps[8] - timestamps[1]).total_seconds()
                    / (timestamps[8] - timestamps[0]).total_seconds()
                ),
                "val_time_usage_relative": 0.0,
                "train_metrics/epoch": 1,
                "train_metrics/learning_rate": lrs[6],
                "train_metrics/loss": losses[6],
                "train_metrics/grad_norm": grads[6],
                "train_metrics/CER (argmax)": metrics[6],
            },
        ),
        # self.writer.set_step(7, "train")
        # self.writer.add_scalars({"epoch": 1, "learning_rate": self.lr_scheduler.get_last_lr()[0]})
        # self.writer.add_scalars({"loss": ..., "grad_norm": ..., "CER (argmax)": ...})
        # self.writer.write()
        (
            7,
            {
                "steps_logged": 1,
                "train_steps_per_sec": (
                    1 / (timestamps[9] - timestamps[8]).total_seconds()
                ),
                "seconds_per_train_step": (
                    (timestamps[9] - timestamps[8]).total_seconds() / 1
                ),
                "total_time_sec": (timestamps[9] - timestamps[0]).total_seconds(),
                "total_time_hr": (
                    (timestamps[9] - timestamps[0]).total_seconds() / 3600
                ),
                "_time_usage_sec": (timestamps[1] - timestamps[0]).total_seconds(),
                "train_time_usage_sec": (
                    (timestamps[9] - timestamps[1]).total_seconds()
                ),
                "val_time_usage_sec": 0.0,
                "_time_usage_relative": (
                    (timestamps[1] - timestamps[0]).total_seconds()
                    / (timestamps[9] - timestamps[0]).total_seconds()
                ),
                "train_time_usage_relative": (
                    (timestamps[9] - timestamps[1]).total_seconds()
                    / (timestamps[9] - timestamps[0]).total_seconds()
                ),
                "val_time_usage_relative": 0.0,
                "train_metrics/epoch": 1,
                "train_metrics/learning_rate": lrs[7],
                "train_metrics/loss": losses[7],
                "train_metrics/grad_norm": grads[7],
                "train_metrics/CER (argmax)": metrics[7],
            },
        ),
        # self.writer.set_step(8, "train")
        # self.writer.add_scalars({"epoch": 1, "learning_rate": self.lr_scheduler.get_last_lr()[0]})
        # self.writer.add_scalars({"loss": ..., "grad_norm": ..., "CER (argmax)": ...})
        # self.writer.write()
        (
            8,
            {
                "steps_logged": 1,
                "train_steps_per_sec": (
                    1 / (timestamps[10] - timestamps[9]).total_seconds()
                ),
                "seconds_per_train_step": (
                    (timestamps[10] - timestamps[9]).total_seconds() / 1
                ),
                "total_time_sec": (timestamps[10] - timestamps[0]).total_seconds(),
                "total_time_hr": (
                    (timestamps[10] - timestamps[0]).total_seconds() / 3600
                ),
                "_time_usage_sec": (timestamps[1] - timestamps[0]).total_seconds(),
                "train_time_usage_sec": (
                    (timestamps[10] - timestamps[1]).total_seconds()
                ),
                "val_time_usage_sec": 0.0,
                "_time_usage_relative": (
                    (timestamps[1] - timestamps[0]).total_seconds()
                    / (timestamps[10] - timestamps[0]).total_seconds()
                ),
                "train_time_usage_relative": (
                    (timestamps[10] - timestamps[1]).total_seconds()
                    / (timestamps[10] - timestamps[0]).total_seconds()
                ),
                "val_time_usage_relative": 0.0,
                "train_metrics/epoch": 1,
                "train_metrics/learning_rate": lrs[8],
                "train_metrics/loss": losses[8],
                "train_metrics/grad_norm": grads[8],
                "train_metrics/CER (argmax)": metrics[8],
            },
        ),
        # self.writer.set_step(9, "train")
        # self.writer.add_scalars({"epoch": 1, "learning_rate": self.lr_scheduler.get_last_lr()[0]})
        # self.writer.add_scalars({"loss": ..., "grad_norm": ..., "CER (argmax)": ...})
        # self.writer.write()
        (
            9,
            {
                "steps_logged": 1,
                "train_steps_per_sec": (
                    1 / (timestamps[11] - timestamps[10]).total_seconds()
                ),
                "seconds_per_train_step": (
                    (timestamps[11] - timestamps[10]).total_seconds() / 1
                ),
                "total_time_sec": (timestamps[11] - timestamps[0]).total_seconds(),
                "total_time_hr": (
                    (timestamps[11] - timestamps[0]).total_seconds() / 3600
                ),
                "_time_usage_sec": (timestamps[1] - timestamps[0]).total_seconds(),
                "train_time_usage_sec": (
                    (timestamps[11] - timestamps[1]).total_seconds()
                ),
                "val_time_usage_sec": 0.0,
                "_time_usage_relative": (
                    (timestamps[1] - timestamps[0]).total_seconds()
                    / (timestamps[11] - timestamps[0]).total_seconds()
                ),
                "train_time_usage_relative": (
                    (timestamps[11] - timestamps[1]).total_seconds()
                    / (timestamps[11] - timestamps[0]).total_seconds()
                ),
                "val_time_usage_relative": 0.0,
                "train_metrics/epoch": 1,
                "train_metrics/learning_rate": lrs[9],
                "train_metrics/loss": losses[9],
                "train_metrics/grad_norm": grads[9],
                "train_metrics/CER (argmax)": metrics[9],
            },
        ),
        # self.writer.set_step(10, "train")
        # self.writer.add_scalars({"epoch": 1, "learning_rate": self.lr_scheduler.get_last_lr()[0]})
        # self.writer.add_scalars({"loss": ..., "grad_norm": ..., "CER (argmax)": ...})
        # self.writer.write()
        (
            10,
            {
                "steps_logged": 1,
                "train_steps_per_sec": (
                    1 / (timestamps[12] - timestamps[11]).total_seconds()
                ),
                "seconds_per_train_step": (
                    (timestamps[12] - timestamps[11]).total_seconds() / 1
                ),
                "total_time_sec": (timestamps[12] - timestamps[0]).total_seconds(),
                "total_time_hr": (
                    (timestamps[12] - timestamps[0]).total_seconds() / 3600
                ),
                "_time_usage_sec": (timestamps[1] - timestamps[0]).total_seconds(),
                "train_time_usage_sec": (
                    (timestamps[12] - timestamps[1]).total_seconds()
                ),
                "val_time_usage_sec": 0.0,
                "_time_usage_relative": (
                    (timestamps[1] - timestamps[0]).total_seconds()
                    / (timestamps[12] - timestamps[0]).total_seconds()
                ),
                "train_time_usage_relative": (
                    (timestamps[12] - timestamps[1]).total_seconds()
                    / (timestamps[12] - timestamps[0]).total_seconds()
                ),
                "val_time_usage_relative": 0.0,
                "train_metrics/epoch": 1,
                "train_metrics/learning_rate": lrs[10],
                "train_metrics/loss": losses[10],
                "train_metrics/grad_norm": grads[10],
                "train_metrics/CER (argmax)": metrics[10],
            },
        ),
        # self.writer.set_step(11, "train")
        # self.writer.add_scalars({"epoch": 1, "learning_rate": self.lr_scheduler.get_last_lr()[0]})
        # self.writer.add_scalars({"loss": ..., "grad_norm": ..., "CER (argmax)": ...})
        # self.writer.write()
        (
            11,
            {
                "steps_logged": 1,
                "train_steps_per_sec": (
                    1 / (timestamps[13] - timestamps[12]).total_seconds()
                ),
                "seconds_per_train_step": (
                    (timestamps[13] - timestamps[12]).total_seconds() / 1
                ),
                "total_time_sec": (timestamps[13] - timestamps[0]).total_seconds(),
                "total_time_hr": (
                    (timestamps[13] - timestamps[0]).total_seconds() / 3600
                ),
                "_time_usage_sec": (timestamps[1] - timestamps[0]).total_seconds(),
                "train_time_usage_sec": (
                    (timestamps[13] - timestamps[1]).total_seconds()
                ),
                "val_time_usage_sec": 0.0,
                "_time_usage_relative": (
                    (timestamps[1] - timestamps[0]).total_seconds()
                    / (timestamps[13] - timestamps[0]).total_seconds()
                ),
                "train_time_usage_relative": (
                    (timestamps[13] - timestamps[1]).total_seconds()
                    / (timestamps[13] - timestamps[0]).total_seconds()
                ),
                "val_time_usage_relative": 0.0,
                "train_metrics/epoch": 1,
                "train_metrics/learning_rate": lrs[11],
                "train_metrics/loss": losses[11],
                "train_metrics/grad_norm": grads[11],
                "train_metrics/CER (argmax)": metrics[11],
            },
        ),
        # self.writer.set_step(11, "val")
        # self.writer.add_scalars({"loss": ..., "CER (argmax)": ...})
        # self.writer.set_step(11, "")
        # self.writer.write()
        (
            11,
            {
                "total_time_sec": (timestamps[15] - timestamps[0]).total_seconds(),
                "total_time_hr": (
                    (timestamps[15] - timestamps[0]).total_seconds() / 3600
                ),
                "_time_usage_sec": (timestamps[1] - timestamps[0]).total_seconds(),
                "train_time_usage_sec": (
                    (timestamps[14] - timestamps[1]).total_seconds()
                ),
                "val_time_usage_sec": (timestamps[15] - timestamps[14]).total_seconds(),
                "_time_usage_relative": (
                    (timestamps[1] - timestamps[0]).total_seconds()
                    / (timestamps[15] - timestamps[0]).total_seconds()
                ),
                "train_time_usage_relative": (
                    (timestamps[14] - timestamps[1]).total_seconds()
                    / (timestamps[15] - timestamps[0]).total_seconds()
                ),
                "val_time_usage_relative": (
                    (timestamps[15] - timestamps[14]).total_seconds()
                    / (timestamps[15] - timestamps[0]).total_seconds()
                ),
                "val_metrics/loss": (
                    (losses[12] + losses[13] + losses[14] + losses[15]) / 4
                ),
                "val_metrics/CER (argmax)": (
                    (metrics[12] + metrics[13] + metrics[14] + metrics[15]) / 4
                ),
            },
        ),
        # self.writer.set_step(11, "train")
        # self.writer.wandb.log({"train_metrics/epoch": 2}, step=12)
        (
            12,
            {
                "train_metrics/epoch": 2,
            },
        ),
        # self.writer.set_step(12, "train")
        # self.writer.add_scalars({"epoch": 2, "learning_rate": self.lr_scheduler.get_last_lr()[0]})
        # self.writer.add_scalars({"loss": ..., "grad_norm": ..., "CER (argmax)": ...})
        # self.writer.write()
        (
            12,
            {
                "steps_logged": 1,
                "train_steps_per_sec": (
                    1 / (timestamps[17] - timestamps[16]).total_seconds()
                ),
                "seconds_per_train_step": (
                    (timestamps[17] - timestamps[16]).total_seconds() / 1
                ),
                "total_time_sec": (timestamps[17] - timestamps[0]).total_seconds(),
                "total_time_hr": (
                    (timestamps[17] - timestamps[0]).total_seconds() / 3600
                ),
                "_time_usage_sec": (
                    (timestamps[1] - timestamps[0]).total_seconds()
                    + (timestamps[16] - timestamps[15]).total_seconds()
                ),
                "train_time_usage_sec": (
                    (timestamps[14] - timestamps[1]).total_seconds()
                    + (timestamps[17] - timestamps[16]).total_seconds()
                ),
                "val_time_usage_sec": (timestamps[15] - timestamps[14]).total_seconds(),
                "_time_usage_relative": (
                    (
                        (timestamps[1] - timestamps[0]).total_seconds()
                        + (timestamps[16] - timestamps[15]).total_seconds()
                    )
                    / (timestamps[17] - timestamps[0]).total_seconds()
                ),
                "train_time_usage_relative": (
                    (
                        (timestamps[14] - timestamps[1]).total_seconds()
                        + (timestamps[17] - timestamps[16]).total_seconds()
                    )
                    / (timestamps[17] - timestamps[0]).total_seconds()
                ),
                "val_time_usage_relative": (
                    (timestamps[15] - timestamps[14]).total_seconds()
                    / (timestamps[17] - timestamps[0]).total_seconds()
                ),
                "train_metrics/epoch": 2,
                "train_metrics/learning_rate": lrs[12],
                "train_metrics/loss": losses[16],
                "train_metrics/grad_norm": grads[12],
                "train_metrics/CER (argmax)": metrics[16],
            },
        ),
        # self.writer.set_step(13, "train")
        # self.writer.add_scalars({"epoch": 2, "learning_rate": self.lr_scheduler.get_last_lr()[0]})
        # self.writer.add_scalars({"loss": ..., "grad_norm": ..., "CER (argmax)": ...})
        # self.writer.write()
        (
            13,
            {
                "steps_logged": 1,
                "train_steps_per_sec": (
                    1 / (timestamps[18] - timestamps[17]).total_seconds()
                ),
                "seconds_per_train_step": (
                    (timestamps[18] - timestamps[17]).total_seconds() / 1
                ),
                "total_time_sec": (timestamps[18] - timestamps[0]).total_seconds(),
                "total_time_hr": (
                    (timestamps[18] - timestamps[0]).total_seconds() / 3600
                ),
                "_time_usage_sec": (
                    (timestamps[1] - timestamps[0]).total_seconds()
                    + (timestamps[16] - timestamps[15]).total_seconds()
                ),
                "train_time_usage_sec": (
                    (timestamps[14] - timestamps[1]).total_seconds()
                    + (timestamps[18] - timestamps[16]).total_seconds()
                ),
                "val_time_usage_sec": (timestamps[15] - timestamps[14]).total_seconds(),
                "_time_usage_relative": (
                    (
                        (timestamps[1] - timestamps[0]).total_seconds()
                        + (timestamps[16] - timestamps[15]).total_seconds()
                    )
                    / (timestamps[18] - timestamps[0]).total_seconds()
                ),
                "train_time_usage_relative": (
                    (
                        (timestamps[14] - timestamps[1]).total_seconds()
                        + (timestamps[18] - timestamps[16]).total_seconds()
                    )
                    / (timestamps[18] - timestamps[0]).total_seconds()
                ),
                "val_time_usage_relative": (
                    (timestamps[15] - timestamps[14]).total_seconds()
                    / (timestamps[18] - timestamps[0]).total_seconds()
                ),
                "train_metrics/epoch": 2,
                "train_metrics/learning_rate": lrs[13],
                "train_metrics/loss": losses[17],
                "train_metrics/grad_norm": grads[13],
                "train_metrics/CER (argmax)": metrics[17],
            },
        ),
        # self.writer.set_step(14, "train")
        # self.writer.add_scalars({"epoch": 2, "learning_rate": self.lr_scheduler.get_last_lr()[0]})
        # self.writer.add_scalars({"loss": ..., "grad_norm": ..., "CER (argmax)": ...})
        # self.writer.write()
        (
            14,
            {
                "steps_logged": 1,
                "train_steps_per_sec": (
                    1 / (timestamps[19] - timestamps[18]).total_seconds()
                ),
                "seconds_per_train_step": (
                    (timestamps[19] - timestamps[18]).total_seconds() / 1
                ),
                "total_time_sec": (timestamps[19] - timestamps[0]).total_seconds(),
                "total_time_hr": (
                    (timestamps[19] - timestamps[0]).total_seconds() / 3600
                ),
                "_time_usage_sec": (
                    (timestamps[1] - timestamps[0]).total_seconds()
                    + (timestamps[16] - timestamps[15]).total_seconds()
                ),
                "train_time_usage_sec": (
                    (timestamps[14] - timestamps[1]).total_seconds()
                    + (timestamps[19] - timestamps[16]).total_seconds()
                ),
                "val_time_usage_sec": (timestamps[15] - timestamps[14]).total_seconds(),
                "_time_usage_relative": (
                    (
                        (timestamps[1] - timestamps[0]).total_seconds()
                        + (timestamps[16] - timestamps[15]).total_seconds()
                    )
                    / (timestamps[19] - timestamps[0]).total_seconds()
                ),
                "train_time_usage_relative": (
                    (
                        (timestamps[14] - timestamps[1]).total_seconds()
                        + (timestamps[19] - timestamps[16]).total_seconds()
                    )
                    / (timestamps[19] - timestamps[0]).total_seconds()
                ),
                "val_time_usage_relative": (
                    (timestamps[15] - timestamps[14]).total_seconds()
                    / (timestamps[19] - timestamps[0]).total_seconds()
                ),
                "train_metrics/epoch": 2,
                "train_metrics/learning_rate": lrs[14],
                "train_metrics/loss": losses[18],
                "train_metrics/grad_norm": grads[14],
                "train_metrics/CER (argmax)": metrics[18],
            },
        ),
        # self.writer.set_step(15, "train")
        # self.writer.add_scalars({"epoch": 2, "learning_rate": self.lr_scheduler.get_last_lr()[0]})
        # self.writer.add_scalars({"loss": ..., "grad_norm": ..., "CER (argmax)": ...})
        # self.writer.write()
        (
            15,
            {
                "steps_logged": 1,
                "train_steps_per_sec": (
                    1 / (timestamps[20] - timestamps[19]).total_seconds()
                ),
                "seconds_per_train_step": (
                    (timestamps[20] - timestamps[19]).total_seconds() / 1
                ),
                "total_time_sec": (timestamps[20] - timestamps[0]).total_seconds(),
                "total_time_hr": (
                    (timestamps[20] - timestamps[0]).total_seconds() / 3600
                ),
                "_time_usage_sec": (
                    (timestamps[1] - timestamps[0]).total_seconds()
                    + (timestamps[16] - timestamps[15]).total_seconds()
                ),
                "train_time_usage_sec": (
                    (timestamps[14] - timestamps[1]).total_seconds()
                    + (timestamps[20] - timestamps[16]).total_seconds()
                ),
                "val_time_usage_sec": (timestamps[15] - timestamps[14]).total_seconds(),
                "_time_usage_relative": (
                    (
                        (timestamps[1] - timestamps[0]).total_seconds()
                        + (timestamps[16] - timestamps[15]).total_seconds()
                    )
                    / (timestamps[20] - timestamps[0]).total_seconds()
                ),
                "train_time_usage_relative": (
                    (
                        (timestamps[14] - timestamps[1]).total_seconds()
                        + (timestamps[20] - timestamps[16]).total_seconds()
                    )
                    / (timestamps[20] - timestamps[0]).total_seconds()
                ),
                "val_time_usage_relative": (
                    (timestamps[15] - timestamps[14]).total_seconds()
                    / (timestamps[20] - timestamps[0]).total_seconds()
                ),
                "train_metrics/epoch": 2,
                "train_metrics/learning_rate": lrs[15],
                "train_metrics/loss": losses[19],
                "train_metrics/grad_norm": grads[15],
                "train_metrics/CER (argmax)": metrics[19],
            },
        ),
        # self.writer.set_step(16, "train")
        # self.writer.add_scalars({"epoch": 2, "learning_rate": self.lr_scheduler.get_last_lr()[0]})
        # self.writer.add_scalars({"loss": ..., "grad_norm": ..., "CER (argmax)": ...})
        # self.writer.write()
        (
            16,
            {
                "steps_logged": 1,
                "train_steps_per_sec": (
                    1 / (timestamps[21] - timestamps[20]).total_seconds()
                ),
                "seconds_per_train_step": (
                    (timestamps[21] - timestamps[20]).total_seconds() / 1
                ),
                "total_time_sec": (timestamps[21] - timestamps[0]).total_seconds(),
                "total_time_hr": (
                    (timestamps[21] - timestamps[0]).total_seconds() / 3600
                ),
                "_time_usage_sec": (
                    (timestamps[1] - timestamps[0]).total_seconds()
                    + (timestamps[16] - timestamps[15]).total_seconds()
                ),
                "train_time_usage_sec": (
                    (timestamps[14] - timestamps[1]).total_seconds()
                    + (timestamps[21] - timestamps[16]).total_seconds()
                ),
                "val_time_usage_sec": (timestamps[15] - timestamps[14]).total_seconds(),
                "_time_usage_relative": (
                    (
                        (timestamps[1] - timestamps[0]).total_seconds()
                        + (timestamps[16] - timestamps[15]).total_seconds()
                    )
                    / (timestamps[21] - timestamps[0]).total_seconds()
                ),
                "train_time_usage_relative": (
                    (
                        (timestamps[14] - timestamps[1]).total_seconds()
                        + (timestamps[21] - timestamps[16]).total_seconds()
                    )
                    / (timestamps[21] - timestamps[0]).total_seconds()
                ),
                "val_time_usage_relative": (
                    (timestamps[15] - timestamps[14]).total_seconds()
                    / (timestamps[21] - timestamps[0]).total_seconds()
                ),
                "train_metrics/epoch": 2,
                "train_metrics/learning_rate": lrs[16],
                "train_metrics/loss": losses[20],
                "train_metrics/grad_norm": grads[16],
                "train_metrics/CER (argmax)": metrics[20],
            },
        ),
        # self.writer.set_step(17, "train")
        # self.writer.add_scalars({"epoch": 2, "learning_rate": self.lr_scheduler.get_last_lr()[0]})
        # self.writer.add_scalars({"loss": ..., "grad_norm": ..., "CER (argmax)": ...})
        # self.writer.write()
        (
            17,
            {
                "steps_logged": 1,
                "train_steps_per_sec": (
                    1 / (timestamps[22] - timestamps[21]).total_seconds()
                ),
                "seconds_per_train_step": (
                    (timestamps[22] - timestamps[21]).total_seconds() / 1
                ),
                "total_time_sec": (timestamps[22] - timestamps[0]).total_seconds(),
                "total_time_hr": (
                    (timestamps[22] - timestamps[0]).total_seconds() / 3600
                ),
                "_time_usage_sec": (
                    (timestamps[1] - timestamps[0]).total_seconds()
                    + (timestamps[16] - timestamps[15]).total_seconds()
                ),
                "train_time_usage_sec": (
                    (timestamps[14] - timestamps[1]).total_seconds()
                    + (timestamps[22] - timestamps[16]).total_seconds()
                ),
                "val_time_usage_sec": (timestamps[15] - timestamps[14]).total_seconds(),
                "_time_usage_relative": (
                    (
                        (timestamps[1] - timestamps[0]).total_seconds()
                        + (timestamps[16] - timestamps[15]).total_seconds()
                    )
                    / (timestamps[22] - timestamps[0]).total_seconds()
                ),
                "train_time_usage_relative": (
                    (
                        (timestamps[14] - timestamps[1]).total_seconds()
                        + (timestamps[22] - timestamps[16]).total_seconds()
                    )
                    / (timestamps[22] - timestamps[0]).total_seconds()
                ),
                "val_time_usage_relative": (
                    (timestamps[15] - timestamps[14]).total_seconds()
                    / (timestamps[22] - timestamps[0]).total_seconds()
                ),
                "train_metrics/epoch": 2,
                "train_metrics/learning_rate": lrs[17],
                "train_metrics/loss": losses[21],
                "train_metrics/grad_norm": grads[17],
                "train_metrics/CER (argmax)": metrics[21],
            },
        ),
        # self.writer.set_step(18, "train")
        # self.writer.add_scalars({"epoch": 2, "learning_rate": self.lr_scheduler.get_last_lr()[0]})
        # self.writer.add_scalars({"loss": ..., "grad_norm": ..., "CER (argmax)": ...})
        # self.writer.write()
        (
            18,
            {
                "steps_logged": 1,
                "train_steps_per_sec": (
                    1 / (timestamps[23] - timestamps[22]).total_seconds()
                ),
                "seconds_per_train_step": (
                    (timestamps[23] - timestamps[22]).total_seconds() / 1
                ),
                "total_time_sec": (timestamps[23] - timestamps[0]).total_seconds(),
                "total_time_hr": (
                    (timestamps[23] - timestamps[0]).total_seconds() / 3600
                ),
                "_time_usage_sec": (
                    (timestamps[1] - timestamps[0]).total_seconds()
                    + (timestamps[16] - timestamps[15]).total_seconds()
                ),
                "train_time_usage_sec": (
                    (timestamps[14] - timestamps[1]).total_seconds()
                    + (timestamps[23] - timestamps[16]).total_seconds()
                ),
                "val_time_usage_sec": (timestamps[15] - timestamps[14]).total_seconds(),
                "_time_usage_relative": (
                    (
                        (timestamps[1] - timestamps[0]).total_seconds()
                        + (timestamps[16] - timestamps[15]).total_seconds()
                    )
                    / (timestamps[23] - timestamps[0]).total_seconds()
                ),
                "train_time_usage_relative": (
                    (
                        (timestamps[14] - timestamps[1]).total_seconds()
                        + (timestamps[23] - timestamps[16]).total_seconds()
                    )
                    / (timestamps[23] - timestamps[0]).total_seconds()
                ),
                "val_time_usage_relative": (
                    (timestamps[15] - timestamps[14]).total_seconds()
                    / (timestamps[23] - timestamps[0]).total_seconds()
                ),
                "train_metrics/epoch": 2,
                "train_metrics/learning_rate": lrs[18],
                "train_metrics/loss": losses[22],
                "train_metrics/grad_norm": grads[18],
                "train_metrics/CER (argmax)": metrics[22],
            },
        ),
        # self.writer.set_step(19, "train")
        # self.writer.add_scalars({"epoch": 2, "learning_rate": self.lr_scheduler.get_last_lr()[0]})
        # self.writer.add_scalars({"loss": ..., "grad_norm": ..., "CER (argmax)": ...})
        # self.writer.write()
        (
            19,
            {
                "steps_logged": 1,
                "train_steps_per_sec": (
                    1 / (timestamps[24] - timestamps[23]).total_seconds()
                ),
                "seconds_per_train_step": (
                    (timestamps[24] - timestamps[23]).total_seconds() / 1
                ),
                "total_time_sec": (timestamps[24] - timestamps[0]).total_seconds(),
                "total_time_hr": (
                    (timestamps[24] - timestamps[0]).total_seconds() / 3600
                ),
                "_time_usage_sec": (
                    (timestamps[1] - timestamps[0]).total_seconds()
                    + (timestamps[16] - timestamps[15]).total_seconds()
                ),
                "train_time_usage_sec": (
                    (timestamps[14] - timestamps[1]).total_seconds()
                    + (timestamps[24] - timestamps[16]).total_seconds()
                ),
                "val_time_usage_sec": (timestamps[15] - timestamps[14]).total_seconds(),
                "_time_usage_relative": (
                    (
                        (timestamps[1] - timestamps[0]).total_seconds()
                        + (timestamps[16] - timestamps[15]).total_seconds()
                    )
                    / (timestamps[24] - timestamps[0]).total_seconds()
                ),
                "train_time_usage_relative": (
                    (
                        (timestamps[14] - timestamps[1]).total_seconds()
                        + (timestamps[24] - timestamps[16]).total_seconds()
                    )
                    / (timestamps[24] - timestamps[0]).total_seconds()
                ),
                "val_time_usage_relative": (
                    (timestamps[15] - timestamps[14]).total_seconds()
                    / (timestamps[24] - timestamps[0]).total_seconds()
                ),
                "train_metrics/epoch": 2,
                "train_metrics/learning_rate": lrs[19],
                "train_metrics/loss": losses[23],
                "train_metrics/grad_norm": grads[19],
                "train_metrics/CER (argmax)": metrics[23],
            },
        ),
        # self.writer.set_step(20, "train")
        # self.writer.add_scalars({"epoch": 2, "learning_rate": self.lr_scheduler.get_last_lr()[0]})
        # self.writer.add_scalars({"loss": ..., "grad_norm": ..., "CER (argmax)": ...})
        # self.writer.write()
        (
            20,
            {
                "steps_logged": 1,
                "train_steps_per_sec": (
                    1 / (timestamps[25] - timestamps[24]).total_seconds()
                ),
                "seconds_per_train_step": (
                    (timestamps[25] - timestamps[24]).total_seconds() / 1
                ),
                "total_time_sec": (timestamps[25] - timestamps[0]).total_seconds(),
                "total_time_hr": (
                    (timestamps[25] - timestamps[0]).total_seconds() / 3600
                ),
                "_time_usage_sec": (
                    (timestamps[1] - timestamps[0]).total_seconds()
                    + (timestamps[16] - timestamps[15]).total_seconds()
                ),
                "train_time_usage_sec": (
                    (timestamps[14] - timestamps[1]).total_seconds()
                    + (timestamps[25] - timestamps[16]).total_seconds()
                ),
                "val_time_usage_sec": (timestamps[15] - timestamps[14]).total_seconds(),
                "_time_usage_relative": (
                    (
                        (timestamps[1] - timestamps[0]).total_seconds()
                        + (timestamps[16] - timestamps[15]).total_seconds()
                    )
                    / (timestamps[25] - timestamps[0]).total_seconds()
                ),
                "train_time_usage_relative": (
                    (
                        (timestamps[14] - timestamps[1]).total_seconds()
                        + (timestamps[25] - timestamps[16]).total_seconds()
                    )
                    / (timestamps[25] - timestamps[0]).total_seconds()
                ),
                "val_time_usage_relative": (
                    (timestamps[15] - timestamps[14]).total_seconds()
                    / (timestamps[25] - timestamps[0]).total_seconds()
                ),
                "train_metrics/epoch": 2,
                "train_metrics/learning_rate": lrs[20],
                "train_metrics/loss": losses[24],
                "train_metrics/grad_norm": grads[20],
                "train_metrics/CER (argmax)": metrics[24],
            },
        ),
        # self.writer.set_step(21, "train")
        # self.writer.add_scalars({"epoch": 2, "learning_rate": self.lr_scheduler.get_last_lr()[0]})
        # self.writer.add_scalars({"loss": ..., "grad_norm": ..., "CER (argmax)": ...})
        # self.writer.write()
        (
            21,
            {
                "steps_logged": 1,
                "train_steps_per_sec": (
                    1 / (timestamps[26] - timestamps[25]).total_seconds()
                ),
                "seconds_per_train_step": (
                    (timestamps[26] - timestamps[25]).total_seconds() / 1
                ),
                "total_time_sec": (timestamps[26] - timestamps[0]).total_seconds(),
                "total_time_hr": (
                    (timestamps[26] - timestamps[0]).total_seconds() / 3600
                ),
                "_time_usage_sec": (
                    (timestamps[1] - timestamps[0]).total_seconds()
                    + (timestamps[16] - timestamps[15]).total_seconds()
                ),
                "train_time_usage_sec": (
                    (timestamps[14] - timestamps[1]).total_seconds()
                    + (timestamps[26] - timestamps[16]).total_seconds()
                ),
                "val_time_usage_sec": (timestamps[15] - timestamps[14]).total_seconds(),
                "_time_usage_relative": (
                    (
                        (timestamps[1] - timestamps[0]).total_seconds()
                        + (timestamps[16] - timestamps[15]).total_seconds()
                    )
                    / (timestamps[26] - timestamps[0]).total_seconds()
                ),
                "train_time_usage_relative": (
                    (
                        (timestamps[14] - timestamps[1]).total_seconds()
                        + (timestamps[26] - timestamps[16]).total_seconds()
                    )
                    / (timestamps[26] - timestamps[0]).total_seconds()
                ),
                "val_time_usage_relative": (
                    (timestamps[15] - timestamps[14]).total_seconds()
                    / (timestamps[26] - timestamps[0]).total_seconds()
                ),
                "train_metrics/epoch": 2,
                "train_metrics/learning_rate": lrs[21],
                "train_metrics/loss": losses[25],
                "train_metrics/grad_norm": grads[21],
                "train_metrics/CER (argmax)": metrics[25],
            },
        ),
        # self.writer.set_step(22, "train")
        # self.writer.add_scalars({"epoch": 2, "learning_rate": self.lr_scheduler.get_last_lr()[0]})
        # self.writer.add_scalars({"loss": ..., "grad_norm": ..., "CER (argmax)": ...})
        # self.writer.write()
        (
            22,
            {
                "steps_logged": 1,
                "train_steps_per_sec": (
                    1 / (timestamps[27] - timestamps[26]).total_seconds()
                ),
                "seconds_per_train_step": (
                    (timestamps[27] - timestamps[26]).total_seconds() / 1
                ),
                "total_time_sec": (timestamps[27] - timestamps[0]).total_seconds(),
                "total_time_hr": (
                    (timestamps[27] - timestamps[0]).total_seconds() / 3600
                ),
                "_time_usage_sec": (
                    (timestamps[1] - timestamps[0]).total_seconds()
                    + (timestamps[16] - timestamps[15]).total_seconds()
                ),
                "train_time_usage_sec": (
                    (timestamps[14] - timestamps[1]).total_seconds()
                    + (timestamps[27] - timestamps[16]).total_seconds()
                ),
                "val_time_usage_sec": (timestamps[15] - timestamps[14]).total_seconds(),
                "_time_usage_relative": (
                    (
                        (timestamps[1] - timestamps[0]).total_seconds()
                        + (timestamps[16] - timestamps[15]).total_seconds()
                    )
                    / (timestamps[27] - timestamps[0]).total_seconds()
                ),
                "train_time_usage_relative": (
                    (
                        (timestamps[14] - timestamps[1]).total_seconds()
                        + (timestamps[27] - timestamps[16]).total_seconds()
                    )
                    / (timestamps[27] - timestamps[0]).total_seconds()
                ),
                "val_time_usage_relative": (
                    (timestamps[15] - timestamps[14]).total_seconds()
                    / (timestamps[27] - timestamps[0]).total_seconds()
                ),
                "train_metrics/epoch": 2,
                "train_metrics/learning_rate": lrs[22],
                "train_metrics/loss": losses[26],
                "train_metrics/grad_norm": grads[22],
                "train_metrics/CER (argmax)": metrics[26],
            },
        ),
        # self.writer.set_step(23, "train")
        # self.writer.add_scalars({"epoch": 2, "learning_rate": self.lr_scheduler.get_last_lr()[0]})
        # self.writer.add_scalars({"loss": ..., "grad_norm": ..., "CER (argmax)": ...})
        # self.writer.write()
        (
            23,
            {
                "steps_logged": 1,
                "train_steps_per_sec": (
                    1 / (timestamps[28] - timestamps[27]).total_seconds()
                ),
                "seconds_per_train_step": (
                    (timestamps[28] - timestamps[27]).total_seconds() / 1
                ),
                "total_time_sec": (timestamps[28] - timestamps[0]).total_seconds(),
                "total_time_hr": (
                    (timestamps[28] - timestamps[0]).total_seconds() / 3600
                ),
                "_time_usage_sec": (
                    (timestamps[1] - timestamps[0]).total_seconds()
                    + (timestamps[16] - timestamps[15]).total_seconds()
                ),
                "train_time_usage_sec": (
                    (timestamps[14] - timestamps[1]).total_seconds()
                    + (timestamps[28] - timestamps[16]).total_seconds()
                ),
                "val_time_usage_sec": (timestamps[15] - timestamps[14]).total_seconds(),
                "_time_usage_relative": (
                    (
                        (timestamps[1] - timestamps[0]).total_seconds()
                        + (timestamps[16] - timestamps[15]).total_seconds()
                    )
                    / (timestamps[28] - timestamps[0]).total_seconds()
                ),
                "train_time_usage_relative": (
                    (
                        (timestamps[14] - timestamps[1]).total_seconds()
                        + (timestamps[28] - timestamps[16]).total_seconds()
                    )
                    / (timestamps[28] - timestamps[0]).total_seconds()
                ),
                "val_time_usage_relative": (
                    (timestamps[15] - timestamps[14]).total_seconds()
                    / (timestamps[28] - timestamps[0]).total_seconds()
                ),
                "train_metrics/epoch": 2,
                "train_metrics/learning_rate": lrs[23],
                "train_metrics/loss": losses[27],
                "train_metrics/grad_norm": grads[23],
                "train_metrics/CER (argmax)": metrics[27],
            },
        ),
        # self.writer.set_step(23, "val")
        # self.writer.add_scalars({"loss": ..., "CER (argmax)": ...})
        # self.writer.set_step(23, "")
        # self.writer.write()
        (
            23,
            {
                "total_time_sec": (timestamps[30] - timestamps[0]).total_seconds(),
                "total_time_hr": (
                    (timestamps[30] - timestamps[0]).total_seconds() / 3600
                ),
                "_time_usage_sec": (
                    (timestamps[1] - timestamps[0]).total_seconds()
                    + (timestamps[16] - timestamps[15]).total_seconds()
                ),
                "train_time_usage_sec": (
                    (timestamps[14] - timestamps[1]).total_seconds()
                    + (timestamps[29] - timestamps[16]).total_seconds()
                ),
                "val_time_usage_sec": (
                    (timestamps[15] - timestamps[14]).total_seconds()
                    + (timestamps[30] - timestamps[29]).total_seconds()
                ),
                "_time_usage_relative": (
                    (
                        (timestamps[1] - timestamps[0]).total_seconds()
                        + (timestamps[16] - timestamps[15]).total_seconds()
                    )
                    / (timestamps[30] - timestamps[0]).total_seconds()
                ),
                "train_time_usage_relative": (
                    (
                        (timestamps[14] - timestamps[1]).total_seconds()
                        + (timestamps[29] - timestamps[16]).total_seconds()
                    )
                    / (timestamps[30] - timestamps[0]).total_seconds()
                ),
                "val_time_usage_relative": (
                    (
                        (timestamps[15] - timestamps[14]).total_seconds()
                        + (timestamps[30] - timestamps[29]).total_seconds()
                    )
                    / (timestamps[30] - timestamps[0]).total_seconds()
                ),
                "val_metrics/loss": (
                    (losses[28] + losses[29] + losses[30] + losses[31]) / 4
                ),
                "val_metrics/CER (argmax)": (
                    (metrics[28] + metrics[29] + metrics[30] + metrics[31]) / 4
                ),
            },
        ),
        # self.writer.set_step(23, "train")
        # self.writer.wandb.log({"train_metrics/epoch": 3}, step=24)
        (
            24,
            {
                "train_metrics/epoch": 3,
            },
        ),
        # self.writer.set_step(24, "train")
        # self.writer.add_scalars({"epoch": 3, "learning_rate": self.lr_scheduler.get_last_lr()[0]})
        # self.writer.add_scalars({"loss": ..., "grad_norm": ..., "CER (argmax)": ...})
        # self.writer.write()
        (
            24,
            {
                "steps_logged": 1,
                "train_steps_per_sec": (
                    1 / (timestamps[32] - timestamps[31]).total_seconds()
                ),
                "seconds_per_train_step": (
                    (timestamps[32] - timestamps[31]).total_seconds() / 1
                ),
                "total_time_sec": (timestamps[32] - timestamps[0]).total_seconds(),
                "total_time_hr": (
                    (timestamps[32] - timestamps[0]).total_seconds() / 3600
                ),
                "_time_usage_sec": (
                    (timestamps[1] - timestamps[0]).total_seconds()
                    + (timestamps[16] - timestamps[15]).total_seconds()
                    + (timestamps[31] - timestamps[30]).total_seconds()
                ),
                "train_time_usage_sec": (
                    (timestamps[14] - timestamps[1]).total_seconds()
                    + (timestamps[29] - timestamps[16]).total_seconds()
                    + (timestamps[32] - timestamps[31]).total_seconds()
                ),
                "val_time_usage_sec": (
                    (timestamps[15] - timestamps[14]).total_seconds()
                    + (timestamps[30] - timestamps[29]).total_seconds()
                ),
                "_time_usage_relative": (
                    (
                        (timestamps[1] - timestamps[0]).total_seconds()
                        + (timestamps[16] - timestamps[15]).total_seconds()
                        + (timestamps[31] - timestamps[30]).total_seconds()
                    )
                    / (timestamps[32] - timestamps[0]).total_seconds()
                ),
                "train_time_usage_relative": (
                    (
                        (timestamps[14] - timestamps[1]).total_seconds()
                        + (timestamps[29] - timestamps[16]).total_seconds()
                        + (timestamps[32] - timestamps[31]).total_seconds()
                    )
                    / (timestamps[32] - timestamps[0]).total_seconds()
                ),
                "val_time_usage_relative": (
                    (
                        (timestamps[15] - timestamps[14]).total_seconds()
                        + (timestamps[30] - timestamps[29]).total_seconds()
                    )
                    / (timestamps[32] - timestamps[0]).total_seconds()
                ),
                "train_metrics/epoch": 3,
                "train_metrics/learning_rate": lrs[24],
                "train_metrics/loss": losses[32],
                "train_metrics/grad_norm": grads[24],
                "train_metrics/CER (argmax)": metrics[32],
            },
        ),
        # self.writer.set_step(25, "train")
        # self.writer.add_scalars({"epoch": 3, "learning_rate": self.lr_scheduler.get_last_lr()[0]})
        # self.writer.add_scalars({"loss": ..., "grad_norm": ..., "CER (argmax)": ...})
        # self.writer.write()
        (
            25,
            {
                "steps_logged": 1,
                "train_steps_per_sec": (
                    1 / (timestamps[33] - timestamps[32]).total_seconds()
                ),
                "seconds_per_train_step": (
                    (timestamps[33] - timestamps[32]).total_seconds() / 1
                ),
                "total_time_sec": (timestamps[33] - timestamps[0]).total_seconds(),
                "total_time_hr": (
                    (timestamps[33] - timestamps[0]).total_seconds() / 3600
                ),
                "_time_usage_sec": (
                    (timestamps[1] - timestamps[0]).total_seconds()
                    + (timestamps[16] - timestamps[15]).total_seconds()
                    + (timestamps[31] - timestamps[30]).total_seconds()
                ),
                "train_time_usage_sec": (
                    (timestamps[14] - timestamps[1]).total_seconds()
                    + (timestamps[29] - timestamps[16]).total_seconds()
                    + (timestamps[33] - timestamps[31]).total_seconds()
                ),
                "val_time_usage_sec": (
                    (timestamps[15] - timestamps[14]).total_seconds()
                    + (timestamps[30] - timestamps[29]).total_seconds()
                ),
                "_time_usage_relative": (
                    (
                        (timestamps[1] - timestamps[0]).total_seconds()
                        + (timestamps[16] - timestamps[15]).total_seconds()
                        + (timestamps[31] - timestamps[30]).total_seconds()
                    )
                    / (timestamps[33] - timestamps[0]).total_seconds()
                ),
                "train_time_usage_relative": (
                    (
                        (timestamps[14] - timestamps[1]).total_seconds()
                        + (timestamps[29] - timestamps[16]).total_seconds()
                        + (timestamps[33] - timestamps[31]).total_seconds()
                    )
                    / (timestamps[33] - timestamps[0]).total_seconds()
                ),
                "val_time_usage_relative": (
                    (
                        (timestamps[15] - timestamps[14]).total_seconds()
                        + (timestamps[30] - timestamps[29]).total_seconds()
                    )
                    / (timestamps[33] - timestamps[0]).total_seconds()
                ),
                "train_metrics/epoch": 3,
                "train_metrics/learning_rate": lrs[25],
                "train_metrics/loss": losses[33],
                "train_metrics/grad_norm": grads[25],
                "train_metrics/CER (argmax)": metrics[33],
            },
        ),
        # self.writer.set_step(26, "train")
        # self.writer.add_scalars({"epoch": 3, "learning_rate": self.lr_scheduler.get_last_lr()[0]})
        # self.writer.add_scalars({"loss": ..., "grad_norm": ..., "CER (argmax)": ...})
        # self.writer.write()
        (
            26,
            {
                "steps_logged": 1,
                "train_steps_per_sec": (
                    1 / (timestamps[34] - timestamps[33]).total_seconds()
                ),
                "seconds_per_train_step": (
                    (timestamps[34] - timestamps[33]).total_seconds() / 1
                ),
                "total_time_sec": (timestamps[34] - timestamps[0]).total_seconds(),
                "total_time_hr": (
                    (timestamps[34] - timestamps[0]).total_seconds() / 3600
                ),
                "_time_usage_sec": (
                    (timestamps[1] - timestamps[0]).total_seconds()
                    + (timestamps[16] - timestamps[15]).total_seconds()
                    + (timestamps[31] - timestamps[30]).total_seconds()
                ),
                "train_time_usage_sec": (
                    (timestamps[14] - timestamps[1]).total_seconds()
                    + (timestamps[29] - timestamps[16]).total_seconds()
                    + (timestamps[34] - timestamps[31]).total_seconds()
                ),
                "val_time_usage_sec": (
                    (timestamps[15] - timestamps[14]).total_seconds()
                    + (timestamps[30] - timestamps[29]).total_seconds()
                ),
                "_time_usage_relative": (
                    (
                        (timestamps[1] - timestamps[0]).total_seconds()
                        + (timestamps[16] - timestamps[15]).total_seconds()
                        + (timestamps[31] - timestamps[30]).total_seconds()
                    )
                    / (timestamps[34] - timestamps[0]).total_seconds()
                ),
                "train_time_usage_relative": (
                    (
                        (timestamps[14] - timestamps[1]).total_seconds()
                        + (timestamps[29] - timestamps[16]).total_seconds()
                        + (timestamps[34] - timestamps[31]).total_seconds()
                    )
                    / (timestamps[34] - timestamps[0]).total_seconds()
                ),
                "val_time_usage_relative": (
                    (
                        (timestamps[15] - timestamps[14]).total_seconds()
                        + (timestamps[30] - timestamps[29]).total_seconds()
                    )
                    / (timestamps[34] - timestamps[0]).total_seconds()
                ),
                "train_metrics/epoch": 3,
                "train_metrics/learning_rate": lrs[26],
                "train_metrics/loss": losses[34],
                "train_metrics/grad_norm": grads[26],
                "train_metrics/CER (argmax)": metrics[34],
            },
        ),
        # self.writer.set_step(27, "train")
        # self.writer.add_scalars({"epoch": 3, "learning_rate": self.lr_scheduler.get_last_lr()[0]})
        # self.writer.add_scalars({"loss": ..., "grad_norm": ..., "CER (argmax)": ...})
        # self.writer.write()
        (
            27,
            {
                "steps_logged": 1,
                "train_steps_per_sec": (
                    1 / (timestamps[35] - timestamps[34]).total_seconds()
                ),
                "seconds_per_train_step": (
                    (timestamps[35] - timestamps[34]).total_seconds() / 1
                ),
                "total_time_sec": (timestamps[35] - timestamps[0]).total_seconds(),
                "total_time_hr": (
                    (timestamps[35] - timestamps[0]).total_seconds() / 3600
                ),
                "_time_usage_sec": (
                    (timestamps[1] - timestamps[0]).total_seconds()
                    + (timestamps[16] - timestamps[15]).total_seconds()
                    + (timestamps[31] - timestamps[30]).total_seconds()
                ),
                "train_time_usage_sec": (
                    (timestamps[14] - timestamps[1]).total_seconds()
                    + (timestamps[29] - timestamps[16]).total_seconds()
                    + (timestamps[35] - timestamps[31]).total_seconds()
                ),
                "val_time_usage_sec": (
                    (timestamps[15] - timestamps[14]).total_seconds()
                    + (timestamps[30] - timestamps[29]).total_seconds()
                ),
                "_time_usage_relative": (
                    (
                        (timestamps[1] - timestamps[0]).total_seconds()
                        + (timestamps[16] - timestamps[15]).total_seconds()
                        + (timestamps[31] - timestamps[30]).total_seconds()
                    )
                    / (timestamps[35] - timestamps[0]).total_seconds()
                ),
                "train_time_usage_relative": (
                    (
                        (timestamps[14] - timestamps[1]).total_seconds()
                        + (timestamps[29] - timestamps[16]).total_seconds()
                        + (timestamps[35] - timestamps[31]).total_seconds()
                    )
                    / (timestamps[35] - timestamps[0]).total_seconds()
                ),
                "val_time_usage_relative": (
                    (
                        (timestamps[15] - timestamps[14]).total_seconds()
                        + (timestamps[30] - timestamps[29]).total_seconds()
                    )
                    / (timestamps[35] - timestamps[0]).total_seconds()
                ),
                "train_metrics/epoch": 3,
                "train_metrics/learning_rate": lrs[27],
                "train_metrics/loss": losses[35],
                "train_metrics/grad_norm": grads[27],
                "train_metrics/CER (argmax)": metrics[35],
            },
        ),
        # self.writer.set_step(28, "train")
        # self.writer.add_scalars({"epoch": 3, "learning_rate": self.lr_scheduler.get_last_lr()[0]})
        # self.writer.add_scalars({"loss": ..., "grad_norm": ..., "CER (argmax)": ...})
        # self.writer.write()
        (
            28,
            {
                "steps_logged": 1,
                "train_steps_per_sec": (
                    1 / (timestamps[36] - timestamps[35]).total_seconds()
                ),
                "seconds_per_train_step": (
                    (timestamps[36] - timestamps[35]).total_seconds() / 1
                ),
                "total_time_sec": (timestamps[36] - timestamps[0]).total_seconds(),
                "total_time_hr": (
                    (timestamps[36] - timestamps[0]).total_seconds() / 3600
                ),
                "_time_usage_sec": (
                    (timestamps[1] - timestamps[0]).total_seconds()
                    + (timestamps[16] - timestamps[15]).total_seconds()
                    + (timestamps[31] - timestamps[30]).total_seconds()
                ),
                "train_time_usage_sec": (
                    (timestamps[14] - timestamps[1]).total_seconds()
                    + (timestamps[29] - timestamps[16]).total_seconds()
                    + (timestamps[36] - timestamps[31]).total_seconds()
                ),
                "val_time_usage_sec": (
                    (timestamps[15] - timestamps[14]).total_seconds()
                    + (timestamps[30] - timestamps[29]).total_seconds()
                ),
                "_time_usage_relative": (
                    (
                        (timestamps[1] - timestamps[0]).total_seconds()
                        + (timestamps[16] - timestamps[15]).total_seconds()
                        + (timestamps[31] - timestamps[30]).total_seconds()
                    )
                    / (timestamps[36] - timestamps[0]).total_seconds()
                ),
                "train_time_usage_relative": (
                    (
                        (timestamps[14] - timestamps[1]).total_seconds()
                        + (timestamps[29] - timestamps[16]).total_seconds()
                        + (timestamps[36] - timestamps[31]).total_seconds()
                    )
                    / (timestamps[36] - timestamps[0]).total_seconds()
                ),
                "val_time_usage_relative": (
                    (
                        (timestamps[15] - timestamps[14]).total_seconds()
                        + (timestamps[30] - timestamps[29]).total_seconds()
                    )
                    / (timestamps[36] - timestamps[0]).total_seconds()
                ),
                "train_metrics/epoch": 3,
                "train_metrics/learning_rate": lrs[28],
                "train_metrics/loss": losses[36],
                "train_metrics/grad_norm": grads[28],
                "train_metrics/CER (argmax)": metrics[36],
            },
        ),
        # self.writer.set_step(29, "train")
        # self.writer.add_scalars({"epoch": 3, "learning_rate": self.lr_scheduler.get_last_lr()[0]})
        # self.writer.add_scalars({"loss": ..., "grad_norm": ..., "CER (argmax)": ...})
        # self.writer.write()
        (
            29,
            {
                "steps_logged": 1,
                "train_steps_per_sec": (
                    1 / (timestamps[37] - timestamps[36]).total_seconds()
                ),
                "seconds_per_train_step": (
                    (timestamps[37] - timestamps[36]).total_seconds() / 1
                ),
                "total_time_sec": (timestamps[37] - timestamps[0]).total_seconds(),
                "total_time_hr": (
                    (timestamps[37] - timestamps[0]).total_seconds() / 3600
                ),
                "_time_usage_sec": (
                    (timestamps[1] - timestamps[0]).total_seconds()
                    + (timestamps[16] - timestamps[15]).total_seconds()
                    + (timestamps[31] - timestamps[30]).total_seconds()
                ),
                "train_time_usage_sec": (
                    (timestamps[14] - timestamps[1]).total_seconds()
                    + (timestamps[29] - timestamps[16]).total_seconds()
                    + (timestamps[37] - timestamps[31]).total_seconds()
                ),
                "val_time_usage_sec": (
                    (timestamps[15] - timestamps[14]).total_seconds()
                    + (timestamps[30] - timestamps[29]).total_seconds()
                ),
                "_time_usage_relative": (
                    (
                        (timestamps[1] - timestamps[0]).total_seconds()
                        + (timestamps[16] - timestamps[15]).total_seconds()
                        + (timestamps[31] - timestamps[30]).total_seconds()
                    )
                    / (timestamps[37] - timestamps[0]).total_seconds()
                ),
                "train_time_usage_relative": (
                    (
                        (timestamps[14] - timestamps[1]).total_seconds()
                        + (timestamps[29] - timestamps[16]).total_seconds()
                        + (timestamps[37] - timestamps[31]).total_seconds()
                    )
                    / (timestamps[37] - timestamps[0]).total_seconds()
                ),
                "val_time_usage_relative": (
                    (
                        (timestamps[15] - timestamps[14]).total_seconds()
                        + (timestamps[30] - timestamps[29]).total_seconds()
                    )
                    / (timestamps[37] - timestamps[0]).total_seconds()
                ),
                "train_metrics/epoch": 3,
                "train_metrics/learning_rate": lrs[29],
                "train_metrics/loss": losses[37],
                "train_metrics/grad_norm": grads[29],
                "train_metrics/CER (argmax)": metrics[37],
            },
        ),
        # self.writer.set_step(30, "train")
        # self.writer.add_scalars({"epoch": 3, "learning_rate": self.lr_scheduler.get_last_lr()[0]})
        # self.writer.add_scalars({"loss": ..., "grad_norm": ..., "CER (argmax)": ...})
        # self.writer.write()
        (
            30,
            {
                "steps_logged": 1,
                "train_steps_per_sec": (
                    1 / (timestamps[38] - timestamps[37]).total_seconds()
                ),
                "seconds_per_train_step": (
                    (timestamps[38] - timestamps[37]).total_seconds() / 1
                ),
                "total_time_sec": (timestamps[38] - timestamps[0]).total_seconds(),
                "total_time_hr": (
                    (timestamps[38] - timestamps[0]).total_seconds() / 3600
                ),
                "_time_usage_sec": (
                    (timestamps[1] - timestamps[0]).total_seconds()
                    + (timestamps[16] - timestamps[15]).total_seconds()
                    + (timestamps[31] - timestamps[30]).total_seconds()
                ),
                "train_time_usage_sec": (
                    (timestamps[14] - timestamps[1]).total_seconds()
                    + (timestamps[29] - timestamps[16]).total_seconds()
                    + (timestamps[38] - timestamps[31]).total_seconds()
                ),
                "val_time_usage_sec": (
                    (timestamps[15] - timestamps[14]).total_seconds()
                    + (timestamps[30] - timestamps[29]).total_seconds()
                ),
                "_time_usage_relative": (
                    (
                        (timestamps[1] - timestamps[0]).total_seconds()
                        + (timestamps[16] - timestamps[15]).total_seconds()
                        + (timestamps[31] - timestamps[30]).total_seconds()
                    )
                    / (timestamps[38] - timestamps[0]).total_seconds()
                ),
                "train_time_usage_relative": (
                    (
                        (timestamps[14] - timestamps[1]).total_seconds()
                        + (timestamps[29] - timestamps[16]).total_seconds()
                        + (timestamps[38] - timestamps[31]).total_seconds()
                    )
                    / (timestamps[38] - timestamps[0]).total_seconds()
                ),
                "val_time_usage_relative": (
                    (
                        (timestamps[15] - timestamps[14]).total_seconds()
                        + (timestamps[30] - timestamps[29]).total_seconds()
                    )
                    / (timestamps[38] - timestamps[0]).total_seconds()
                ),
                "train_metrics/epoch": 3,
                "train_metrics/learning_rate": lrs[30],
                "train_metrics/loss": losses[38],
                "train_metrics/grad_norm": grads[30],
                "train_metrics/CER (argmax)": metrics[38],
            },
        ),
        # self.writer.set_step(31, "train")
        # self.writer.add_scalars({"epoch": 3, "learning_rate": self.lr_scheduler.get_last_lr()[0]})
        # self.writer.add_scalars({"loss": ..., "grad_norm": ..., "CER (argmax)": ...})
        # self.writer.write()
        (
            31,
            {
                "steps_logged": 1,
                "train_steps_per_sec": (
                    1 / (timestamps[39] - timestamps[38]).total_seconds()
                ),
                "seconds_per_train_step": (
                    (timestamps[39] - timestamps[38]).total_seconds() / 1
                ),
                "total_time_sec": (timestamps[39] - timestamps[0]).total_seconds(),
                "total_time_hr": (
                    (timestamps[39] - timestamps[0]).total_seconds() / 3600
                ),
                "_time_usage_sec": (
                    (timestamps[1] - timestamps[0]).total_seconds()
                    + (timestamps[16] - timestamps[15]).total_seconds()
                    + (timestamps[31] - timestamps[30]).total_seconds()
                ),
                "train_time_usage_sec": (
                    (timestamps[14] - timestamps[1]).total_seconds()
                    + (timestamps[29] - timestamps[16]).total_seconds()
                    + (timestamps[39] - timestamps[31]).total_seconds()
                ),
                "val_time_usage_sec": (
                    (timestamps[15] - timestamps[14]).total_seconds()
                    + (timestamps[30] - timestamps[29]).total_seconds()
                ),
                "_time_usage_relative": (
                    (
                        (timestamps[1] - timestamps[0]).total_seconds()
                        + (timestamps[16] - timestamps[15]).total_seconds()
                        + (timestamps[31] - timestamps[30]).total_seconds()
                    )
                    / (timestamps[39] - timestamps[0]).total_seconds()
                ),
                "train_time_usage_relative": (
                    (
                        (timestamps[14] - timestamps[1]).total_seconds()
                        + (timestamps[29] - timestamps[16]).total_seconds()
                        + (timestamps[39] - timestamps[31]).total_seconds()
                    )
                    / (timestamps[39] - timestamps[0]).total_seconds()
                ),
                "val_time_usage_relative": (
                    (
                        (timestamps[15] - timestamps[14]).total_seconds()
                        + (timestamps[30] - timestamps[29]).total_seconds()
                    )
                    / (timestamps[39] - timestamps[0]).total_seconds()
                ),
                "train_metrics/epoch": 3,
                "train_metrics/learning_rate": lrs[31],
                "train_metrics/loss": losses[39],
                "train_metrics/grad_norm": grads[31],
                "train_metrics/CER (argmax)": metrics[39],
            },
        ),
        # self.writer.set_step(32, "train")
        # self.writer.add_scalars({"epoch": 3, "learning_rate": self.lr_scheduler.get_last_lr()[0]})
        # self.writer.add_scalars({"loss": ..., "grad_norm": ..., "CER (argmax)": ...})
        # self.writer.write()
        (
            32,
            {
                "steps_logged": 1,
                "train_steps_per_sec": (
                    1 / (timestamps[40] - timestamps[39]).total_seconds()
                ),
                "seconds_per_train_step": (
                    (timestamps[40] - timestamps[39]).total_seconds() / 1
                ),
                "total_time_sec": (timestamps[40] - timestamps[0]).total_seconds(),
                "total_time_hr": (
                    (timestamps[40] - timestamps[0]).total_seconds() / 3600
                ),
                "_time_usage_sec": (
                    (timestamps[1] - timestamps[0]).total_seconds()
                    + (timestamps[16] - timestamps[15]).total_seconds()
                    + (timestamps[31] - timestamps[30]).total_seconds()
                ),
                "train_time_usage_sec": (
                    (timestamps[14] - timestamps[1]).total_seconds()
                    + (timestamps[29] - timestamps[16]).total_seconds()
                    + (timestamps[40] - timestamps[31]).total_seconds()
                ),
                "val_time_usage_sec": (
                    (timestamps[15] - timestamps[14]).total_seconds()
                    + (timestamps[30] - timestamps[29]).total_seconds()
                ),
                "_time_usage_relative": (
                    (
                        (timestamps[1] - timestamps[0]).total_seconds()
                        + (timestamps[16] - timestamps[15]).total_seconds()
                        + (timestamps[31] - timestamps[30]).total_seconds()
                    )
                    / (timestamps[40] - timestamps[0]).total_seconds()
                ),
                "train_time_usage_relative": (
                    (
                        (timestamps[14] - timestamps[1]).total_seconds()
                        + (timestamps[29] - timestamps[16]).total_seconds()
                        + (timestamps[40] - timestamps[31]).total_seconds()
                    )
                    / (timestamps[40] - timestamps[0]).total_seconds()
                ),
                "val_time_usage_relative": (
                    (
                        (timestamps[15] - timestamps[14]).total_seconds()
                        + (timestamps[30] - timestamps[29]).total_seconds()
                    )
                    / (timestamps[40] - timestamps[0]).total_seconds()
                ),
                "train_metrics/epoch": 3,
                "train_metrics/learning_rate": lrs[32],
                "train_metrics/loss": losses[40],
                "train_metrics/grad_norm": grads[32],
                "train_metrics/CER (argmax)": metrics[40],
            },
        ),
        # self.writer.set_step(33, "train")
        # self.writer.add_scalars({"epoch": 3, "learning_rate": self.lr_scheduler.get_last_lr()[0]})
        # self.writer.add_scalars({"loss": ..., "grad_norm": ..., "CER (argmax)": ...})
        # self.writer.write()
        (
            33,
            {
                "steps_logged": 1,
                "train_steps_per_sec": (
                    1 / (timestamps[41] - timestamps[40]).total_seconds()
                ),
                "seconds_per_train_step": (
                    (timestamps[41] - timestamps[40]).total_seconds() / 1
                ),
                "total_time_sec": (timestamps[41] - timestamps[0]).total_seconds(),
                "total_time_hr": (
                    (timestamps[41] - timestamps[0]).total_seconds() / 3600
                ),
                "_time_usage_sec": (
                    (timestamps[1] - timestamps[0]).total_seconds()
                    + (timestamps[16] - timestamps[15]).total_seconds()
                    + (timestamps[31] - timestamps[30]).total_seconds()
                ),
                "train_time_usage_sec": (
                    (timestamps[14] - timestamps[1]).total_seconds()
                    + (timestamps[29] - timestamps[16]).total_seconds()
                    + (timestamps[41] - timestamps[31]).total_seconds()
                ),
                "val_time_usage_sec": (
                    (timestamps[15] - timestamps[14]).total_seconds()
                    + (timestamps[30] - timestamps[29]).total_seconds()
                ),
                "_time_usage_relative": (
                    (
                        (timestamps[1] - timestamps[0]).total_seconds()
                        + (timestamps[16] - timestamps[15]).total_seconds()
                        + (timestamps[31] - timestamps[30]).total_seconds()
                    )
                    / (timestamps[41] - timestamps[0]).total_seconds()
                ),
                "train_time_usage_relative": (
                    (
                        (timestamps[14] - timestamps[1]).total_seconds()
                        + (timestamps[29] - timestamps[16]).total_seconds()
                        + (timestamps[41] - timestamps[31]).total_seconds()
                    )
                    / (timestamps[41] - timestamps[0]).total_seconds()
                ),
                "val_time_usage_relative": (
                    (
                        (timestamps[15] - timestamps[14]).total_seconds()
                        + (timestamps[30] - timestamps[29]).total_seconds()
                    )
                    / (timestamps[41] - timestamps[0]).total_seconds()
                ),
                "train_metrics/epoch": 3,
                "train_metrics/learning_rate": lrs[33],
                "train_metrics/loss": losses[41],
                "train_metrics/grad_norm": grads[33],
                "train_metrics/CER (argmax)": metrics[41],
            },
        ),
        # self.writer.set_step(34, "train")
        # self.writer.add_scalars({"epoch": 3, "learning_rate": self.lr_scheduler.get_last_lr()[0]})
        # self.writer.add_scalars({"loss": ..., "grad_norm": ..., "CER (argmax)": ...})
        # self.writer.write()
        (
            34,
            {
                "steps_logged": 1,
                "train_steps_per_sec": (
                    1 / (timestamps[42] - timestamps[41]).total_seconds()
                ),
                "seconds_per_train_step": (
                    (timestamps[42] - timestamps[41]).total_seconds() / 1
                ),
                "total_time_sec": (timestamps[42] - timestamps[0]).total_seconds(),
                "total_time_hr": (
                    (timestamps[42] - timestamps[0]).total_seconds() / 3600
                ),
                "_time_usage_sec": (
                    (timestamps[1] - timestamps[0]).total_seconds()
                    + (timestamps[16] - timestamps[15]).total_seconds()
                    + (timestamps[31] - timestamps[30]).total_seconds()
                ),
                "train_time_usage_sec": (
                    (timestamps[14] - timestamps[1]).total_seconds()
                    + (timestamps[29] - timestamps[16]).total_seconds()
                    + (timestamps[42] - timestamps[31]).total_seconds()
                ),
                "val_time_usage_sec": (
                    (timestamps[15] - timestamps[14]).total_seconds()
                    + (timestamps[30] - timestamps[29]).total_seconds()
                ),
                "_time_usage_relative": (
                    (
                        (timestamps[1] - timestamps[0]).total_seconds()
                        + (timestamps[16] - timestamps[15]).total_seconds()
                        + (timestamps[31] - timestamps[30]).total_seconds()
                    )
                    / (timestamps[42] - timestamps[0]).total_seconds()
                ),
                "train_time_usage_relative": (
                    (
                        (timestamps[14] - timestamps[1]).total_seconds()
                        + (timestamps[29] - timestamps[16]).total_seconds()
                        + (timestamps[42] - timestamps[31]).total_seconds()
                    )
                    / (timestamps[42] - timestamps[0]).total_seconds()
                ),
                "val_time_usage_relative": (
                    (
                        (timestamps[15] - timestamps[14]).total_seconds()
                        + (timestamps[30] - timestamps[29]).total_seconds()
                    )
                    / (timestamps[42] - timestamps[0]).total_seconds()
                ),
                "train_metrics/epoch": 3,
                "train_metrics/learning_rate": lrs[34],
                "train_metrics/loss": losses[42],
                "train_metrics/grad_norm": grads[34],
                "train_metrics/CER (argmax)": metrics[42],
            },
        ),
        # self.writer.set_step(35, "train")
        # self.writer.add_scalars({"epoch": 3, "learning_rate": self.lr_scheduler.get_last_lr()[0]})
        # self.writer.add_scalars({"loss": ..., "grad_norm": ..., "CER (argmax)": ...})
        # self.writer.write()
        (
            35,
            {
                "steps_logged": 1,
                "train_steps_per_sec": (
                    1 / (timestamps[43] - timestamps[42]).total_seconds()
                ),
                "seconds_per_train_step": (
                    (timestamps[43] - timestamps[42]).total_seconds() / 1
                ),
                "total_time_sec": (timestamps[43] - timestamps[0]).total_seconds(),
                "total_time_hr": (
                    (timestamps[43] - timestamps[0]).total_seconds() / 3600
                ),
                "_time_usage_sec": (
                    (timestamps[1] - timestamps[0]).total_seconds()
                    + (timestamps[16] - timestamps[15]).total_seconds()
                    + (timestamps[31] - timestamps[30]).total_seconds()
                ),
                "train_time_usage_sec": (
                    (timestamps[14] - timestamps[1]).total_seconds()
                    + (timestamps[29] - timestamps[16]).total_seconds()
                    + (timestamps[43] - timestamps[31]).total_seconds()
                ),
                "val_time_usage_sec": (
                    (timestamps[15] - timestamps[14]).total_seconds()
                    + (timestamps[30] - timestamps[29]).total_seconds()
                ),
                "_time_usage_relative": (
                    (
                        (timestamps[1] - timestamps[0]).total_seconds()
                        + (timestamps[16] - timestamps[15]).total_seconds()
                        + (timestamps[31] - timestamps[30]).total_seconds()
                    )
                    / (timestamps[43] - timestamps[0]).total_seconds()
                ),
                "train_time_usage_relative": (
                    (
                        (timestamps[14] - timestamps[1]).total_seconds()
                        + (timestamps[29] - timestamps[16]).total_seconds()
                        + (timestamps[43] - timestamps[31]).total_seconds()
                    )
                    / (timestamps[43] - timestamps[0]).total_seconds()
                ),
                "val_time_usage_relative": (
                    (
                        (timestamps[15] - timestamps[14]).total_seconds()
                        + (timestamps[30] - timestamps[29]).total_seconds()
                    )
                    / (timestamps[43] - timestamps[0]).total_seconds()
                ),
                "train_metrics/epoch": 3,
                "train_metrics/learning_rate": lrs[35],
                "train_metrics/loss": losses[43],
                "train_metrics/grad_norm": grads[35],
                "train_metrics/CER (argmax)": metrics[43],
            },
        ),
        # self.writer.set_step(35, "val")
        # self.writer.add_scalars({"loss": ..., "CER (argmax)": ...})
        # self.writer.set_step(35, "")
        # self.writer.write()
        (
            35,
            {
                "total_time_sec": (timestamps[45] - timestamps[0]).total_seconds(),
                "total_time_hr": (
                    (timestamps[45] - timestamps[0]).total_seconds() / 3600
                ),
                "_time_usage_sec": (
                    (timestamps[1] - timestamps[0]).total_seconds()
                    + (timestamps[16] - timestamps[15]).total_seconds()
                    + (timestamps[31] - timestamps[30]).total_seconds()
                ),
                "train_time_usage_sec": (
                    (timestamps[14] - timestamps[1]).total_seconds()
                    + (timestamps[29] - timestamps[16]).total_seconds()
                    + (timestamps[44] - timestamps[31]).total_seconds()
                ),
                "val_time_usage_sec": (
                    (timestamps[15] - timestamps[14]).total_seconds()
                    + (timestamps[30] - timestamps[29]).total_seconds()
                    + (timestamps[45] - timestamps[44]).total_seconds()
                ),
                "_time_usage_relative": (
                    (
                        (timestamps[1] - timestamps[0]).total_seconds()
                        + (timestamps[16] - timestamps[15]).total_seconds()
                        + (timestamps[31] - timestamps[30]).total_seconds()
                    )
                    / (timestamps[45] - timestamps[0]).total_seconds()
                ),
                "train_time_usage_relative": (
                    (
                        (timestamps[14] - timestamps[1]).total_seconds()
                        + (timestamps[29] - timestamps[16]).total_seconds()
                        + (timestamps[44] - timestamps[31]).total_seconds()
                    )
                    / (timestamps[45] - timestamps[0]).total_seconds()
                ),
                "val_time_usage_relative": (
                    (
                        (timestamps[15] - timestamps[14]).total_seconds()
                        + (timestamps[30] - timestamps[29]).total_seconds()
                        + (timestamps[45] - timestamps[44]).total_seconds()
                    )
                    / (timestamps[45] - timestamps[0]).total_seconds()
                ),
                "val_metrics/loss": (
                    (losses[44] + losses[45] + losses[46] + losses[47]) / 4
                ),
                "val_metrics/CER (argmax)": (
                    (metrics[44] + metrics[45] + metrics[46] + metrics[47]) / 4
                ),
            },
        ),
    ]


def check5(
    timestamps: Sequence[datetime],
    grads: Sequence[float],
    losses: Sequence[float],
    metrics: Sequence[float],
    lrs: Sequence[float],
):
    return [
        # self.writer.set_step(-1, "train")
        # self.writer.log_manual({"train_metrics/epoch": 1}, step=0)
        (
            0,
            {
                "train_metrics/epoch": 1,
            },
        ),
        # self.writer.set_step(4, "train")
        # self.writer.add_scalars({"epoch": 1, "learning_rate": self.lr_scheduler.get_last_lr()[0]})
        # self.writer.add_scalars({"loss": ..., "grad_norm": ..., "CER (argmax)": ...})
        # self.writer.write()
        (
            4,
            {
                "steps_logged": 5,
                "train_steps_per_sec": (
                    5 / (timestamps[2] - timestamps[1]).total_seconds()
                ),
                "seconds_per_train_step": (
                    (timestamps[2] - timestamps[1]).total_seconds() / 5
                ),
                "total_time_sec": (timestamps[2] - timestamps[0]).total_seconds(),
                "total_time_hr": (
                    (timestamps[2] - timestamps[0]).total_seconds() / 3600
                ),
                "_time_usage_sec": (timestamps[1] - timestamps[0]).total_seconds(),
                "train_time_usage_sec": (
                    (timestamps[2] - timestamps[1]).total_seconds()
                ),
                "val_time_usage_sec": 0.0,
                "_time_usage_relative": (
                    (timestamps[1] - timestamps[0]).total_seconds()
                    / (timestamps[2] - timestamps[0]).total_seconds()
                ),
                "train_time_usage_relative": (
                    (timestamps[2] - timestamps[1]).total_seconds()
                    / (timestamps[2] - timestamps[0]).total_seconds()
                ),
                "val_time_usage_relative": 0.0,
                "train_metrics/epoch": 1,
                "train_metrics/learning_rate": lrs[4],
                "train_metrics/loss": (
                    (losses[0] + losses[1] + losses[2] + losses[3] + losses[4]) / 5
                ),
                "train_metrics/grad_norm": (
                    (grads[0] + grads[1] + grads[2] + grads[3] + grads[4]) / 5
                ),
                "train_metrics/CER (argmax)": (
                    (metrics[0] + metrics[1] + metrics[2] + metrics[3] + metrics[4]) / 5
                ),
            },
        ),
        # self.writer.set_step(9, "train")
        # self.writer.add_scalars({"epoch": 1, "learning_rate": self.lr_scheduler.get_last_lr()[0]})
        # self.writer.add_scalars({"loss": ..., "grad_norm": ..., "CER (argmax)": ...})
        # self.writer.write()
        (
            9,
            {
                "steps_logged": 5,
                "train_steps_per_sec": (
                    5 / (timestamps[3] - timestamps[2]).total_seconds()
                ),
                "seconds_per_train_step": (
                    (timestamps[3] - timestamps[2]).total_seconds() / 5
                ),
                "total_time_sec": (timestamps[3] - timestamps[0]).total_seconds(),
                "total_time_hr": (
                    (timestamps[3] - timestamps[0]).total_seconds() / 3600
                ),
                "_time_usage_sec": (timestamps[1] - timestamps[0]).total_seconds(),
                "train_time_usage_sec": (timestamps[3] - timestamps[1]).total_seconds(),
                "val_time_usage_sec": 0.0,
                "_time_usage_relative": (
                    (timestamps[1] - timestamps[0]).total_seconds()
                    / (timestamps[3] - timestamps[0]).total_seconds()
                ),
                "train_time_usage_relative": (
                    (timestamps[3] - timestamps[1]).total_seconds()
                    / (timestamps[3] - timestamps[0]).total_seconds()
                ),
                "val_time_usage_relative": 0.0,
                "train_metrics/epoch": 1,
                "train_metrics/learning_rate": lrs[9],
                "train_metrics/loss": (
                    (losses[5] + losses[6] + losses[7] + losses[8] + losses[9]) / 5
                ),
                "train_metrics/grad_norm": (
                    (grads[5] + grads[6] + grads[7] + grads[8] + grads[9]) / 5
                ),
                "train_metrics/CER (argmax)": (
                    (metrics[5] + metrics[6] + metrics[7] + metrics[8] + metrics[9]) / 5
                ),
            },
        ),
        # self.writer.set_step(11, "train")
        # self.writer.add_scalars({"epoch": 1, "learning_rate": self.lr_scheduler.get_last_lr()[0]})
        # self.writer.add_scalars({"loss": ..., "grad_norm": ..., "CER (argmax)": ...})
        # self.writer.write()
        (
            11,
            {
                "steps_logged": 2,
                "train_steps_per_sec": (
                    2 / (timestamps[4] - timestamps[3]).total_seconds()
                ),
                "seconds_per_train_step": (
                    (timestamps[4] - timestamps[3]).total_seconds() / 2
                ),
                "total_time_sec": (timestamps[4] - timestamps[0]).total_seconds(),
                "total_time_hr": (
                    (timestamps[4] - timestamps[0]).total_seconds() / 3600
                ),
                "_time_usage_sec": (timestamps[1] - timestamps[0]).total_seconds(),
                "train_time_usage_sec": (timestamps[4] - timestamps[1]).total_seconds(),
                "val_time_usage_sec": 0.0,
                "_time_usage_relative": (
                    (timestamps[1] - timestamps[0]).total_seconds()
                    / (timestamps[4] - timestamps[0]).total_seconds()
                ),
                "train_time_usage_relative": (
                    (timestamps[4] - timestamps[1]).total_seconds()
                    / (timestamps[4] - timestamps[0]).total_seconds()
                ),
                "val_time_usage_relative": 0.0,
                "train_metrics/epoch": 1,
                "train_metrics/learning_rate": lrs[11],
                "train_metrics/loss": (losses[10] + losses[11]) / 2,
                "train_metrics/grad_norm": (grads[10] + grads[11]) / 2,
                "train_metrics/CER (argmax)": (metrics[10] + metrics[11]) / 2,
            },
        ),
        # self.writer.set_step(11, "val")
        # self.writer.add_scalars({"loss": ..., "CER (argmax)": ...})
        # self.writer.set_step(11, "")
        # self.writer.write()
        (
            11,
            {
                "total_time_sec": (timestamps[6] - timestamps[0]).total_seconds(),
                "total_time_hr": (
                    (timestamps[6] - timestamps[0]).total_seconds() / 3600
                ),
                "_time_usage_sec": (timestamps[1] - timestamps[0]).total_seconds(),
                "train_time_usage_sec": (
                    (timestamps[5] - timestamps[1]).total_seconds()
                ),
                "val_time_usage_sec": (timestamps[6] - timestamps[5]).total_seconds(),
                "_time_usage_relative": (
                    (timestamps[1] - timestamps[0]).total_seconds()
                    / (timestamps[6] - timestamps[0]).total_seconds()
                ),
                "train_time_usage_relative": (
                    (timestamps[5] - timestamps[1]).total_seconds()
                    / (timestamps[6] - timestamps[0]).total_seconds()
                ),
                "val_time_usage_relative": (
                    (timestamps[6] - timestamps[5]).total_seconds()
                    / (timestamps[6] - timestamps[0]).total_seconds()
                ),
                "val_metrics/loss": (
                    (losses[12] + losses[13] + losses[14] + losses[15]) / 4
                ),
                "val_metrics/CER (argmax)": (
                    (metrics[12] + metrics[13] + metrics[14] + metrics[15]) / 4
                ),
            },
        ),
        # self.writer.set_step(11, "train")
        # self.writer.wandb.log({"train_metrics/epoch": 2}, step=12)
        (
            12,
            {
                "train_metrics/epoch": 2,
            },
        ),
        # self.writer.set_step(16, "train")
        # self.writer.add_scalars({"epoch": 2, "learning_rate": self.lr_scheduler.get_last_lr()[0]})
        # self.writer.add_scalars({"loss": ..., "grad_norm": ..., "CER (argmax)": ...})
        # self.writer.write()
        (
            16,
            {
                "steps_logged": 5,
                "train_steps_per_sec": (
                    5 / (timestamps[8] - timestamps[7]).total_seconds()
                ),
                "seconds_per_train_step": (
                    (timestamps[8] - timestamps[7]).total_seconds() / 5
                ),
                "total_time_sec": (timestamps[8] - timestamps[0]).total_seconds(),
                "total_time_hr": (
                    (timestamps[8] - timestamps[0]).total_seconds() / 3600
                ),
                "_time_usage_sec": (
                    (timestamps[1] - timestamps[0]).total_seconds()
                    + (timestamps[7] - timestamps[6]).total_seconds()
                ),
                "train_time_usage_sec": (
                    (timestamps[5] - timestamps[1]).total_seconds()
                    + (timestamps[8] - timestamps[7]).total_seconds()
                ),
                "val_time_usage_sec": (timestamps[6] - timestamps[5]).total_seconds(),
                "_time_usage_relative": (
                    (
                        (timestamps[1] - timestamps[0]).total_seconds()
                        + (timestamps[7] - timestamps[6]).total_seconds()
                    )
                    / (timestamps[8] - timestamps[0]).total_seconds()
                ),
                "train_time_usage_relative": (
                    (
                        (timestamps[5] - timestamps[1]).total_seconds()
                        + (timestamps[8] - timestamps[7]).total_seconds()
                    )
                    / (timestamps[8] - timestamps[0]).total_seconds()
                ),
                "val_time_usage_relative": (
                    (timestamps[6] - timestamps[5]).total_seconds()
                    / (timestamps[8] - timestamps[0]).total_seconds()
                ),
                "train_metrics/epoch": 2,
                "train_metrics/learning_rate": lrs[16],
                "train_metrics/loss": (
                    (losses[16] + losses[17] + losses[18] + losses[19] + losses[20]) / 5
                ),
                "train_metrics/grad_norm": (
                    (grads[12] + grads[13] + grads[14] + grads[15] + grads[16]) / 5
                ),
                "train_metrics/CER (argmax)": (
                    (
                        metrics[16]
                        + metrics[17]
                        + metrics[18]
                        + metrics[19]
                        + metrics[20]
                    )
                    / 5
                ),
            },
        ),
        # self.writer.set_step(21, "train")
        # self.writer.add_scalars({"epoch": 2, "learning_rate": self.lr_scheduler.get_last_lr()[0]})
        # self.writer.add_scalars({"loss": ..., "grad_norm": ..., "CER (argmax)": ...})
        # self.writer.write()
        (
            21,
            {
                "steps_logged": 5,
                "train_steps_per_sec": (
                    5 / (timestamps[9] - timestamps[8]).total_seconds()
                ),
                "seconds_per_train_step": (
                    (timestamps[9] - timestamps[8]).total_seconds() / 5
                ),
                "total_time_sec": (timestamps[9] - timestamps[0]).total_seconds(),
                "total_time_hr": (
                    (timestamps[9] - timestamps[0]).total_seconds() / 3600
                ),
                "_time_usage_sec": (
                    (timestamps[1] - timestamps[0]).total_seconds()
                    + (timestamps[7] - timestamps[6]).total_seconds()
                ),
                "train_time_usage_sec": (
                    (timestamps[5] - timestamps[1]).total_seconds()
                    + (timestamps[9] - timestamps[7]).total_seconds()
                ),
                "val_time_usage_sec": (timestamps[6] - timestamps[5]).total_seconds(),
                "_time_usage_relative": (
                    (
                        (timestamps[1] - timestamps[0]).total_seconds()
                        + (timestamps[7] - timestamps[6]).total_seconds()
                    )
                    / (timestamps[9] - timestamps[0]).total_seconds()
                ),
                "train_time_usage_relative": (
                    (
                        (timestamps[5] - timestamps[1]).total_seconds()
                        + (timestamps[9] - timestamps[7]).total_seconds()
                    )
                    / (timestamps[9] - timestamps[0]).total_seconds()
                ),
                "val_time_usage_relative": (
                    (timestamps[6] - timestamps[5]).total_seconds()
                    / (timestamps[9] - timestamps[0]).total_seconds()
                ),
                "train_metrics/epoch": 2,
                "train_metrics/learning_rate": lrs[21],
                "train_metrics/loss": (
                    (losses[21] + losses[22] + losses[23] + losses[24] + losses[25]) / 5
                ),
                "train_metrics/grad_norm": (
                    (grads[17] + grads[18] + grads[19] + grads[20] + grads[21]) / 5
                ),
                "train_metrics/CER (argmax)": (
                    (
                        metrics[21]
                        + metrics[22]
                        + metrics[23]
                        + metrics[24]
                        + metrics[25]
                    )
                    / 5
                ),
            },
        ),
        # self.writer.set_step(23, "train")
        # self.writer.add_scalars({"epoch": 2, "learning_rate": self.lr_scheduler.get_last_lr()[0]})
        # self.writer.add_scalars({"loss": ..., "grad_norm": ..., "CER (argmax)": ...})
        # self.writer.write()
        (
            23,
            {
                "steps_logged": 2,
                "train_steps_per_sec": (
                    2 / (timestamps[10] - timestamps[9]).total_seconds()
                ),
                "seconds_per_train_step": (
                    (timestamps[10] - timestamps[9]).total_seconds() / 2
                ),
                "total_time_sec": (timestamps[10] - timestamps[0]).total_seconds(),
                "total_time_hr": (
                    (timestamps[10] - timestamps[0]).total_seconds() / 3600
                ),
                "_time_usage_sec": (
                    (timestamps[1] - timestamps[0]).total_seconds()
                    + (timestamps[7] - timestamps[6]).total_seconds()
                ),
                "train_time_usage_sec": (
                    (timestamps[5] - timestamps[1]).total_seconds()
                    + (timestamps[10] - timestamps[7]).total_seconds()
                ),
                "val_time_usage_sec": (timestamps[6] - timestamps[5]).total_seconds(),
                "_time_usage_relative": (
                    (
                        (timestamps[1] - timestamps[0]).total_seconds()
                        + (timestamps[7] - timestamps[6]).total_seconds()
                    )
                    / (timestamps[10] - timestamps[0]).total_seconds()
                ),
                "train_time_usage_relative": (
                    (
                        (timestamps[5] - timestamps[1]).total_seconds()
                        + (timestamps[10] - timestamps[7]).total_seconds()
                    )
                    / (timestamps[10] - timestamps[0]).total_seconds()
                ),
                "val_time_usage_relative": (
                    (timestamps[6] - timestamps[5]).total_seconds()
                    / (timestamps[10] - timestamps[0]).total_seconds()
                ),
                "train_metrics/epoch": 2,
                "train_metrics/learning_rate": lrs[23],
                "train_metrics/loss": (losses[26] + losses[27]) / 2,
                "train_metrics/grad_norm": (grads[22] + grads[23]) / 2,
                "train_metrics/CER (argmax)": (metrics[26] + metrics[27]) / 2,
            },
        ),
        # self.writer.set_step(23, "val")
        # self.writer.add_scalars({"loss": ..., "CER (argmax)": ...})
        # self.writer.set_step(23, "")
        # self.writer.write()
        (
            23,
            {
                "total_time_sec": (timestamps[12] - timestamps[0]).total_seconds(),
                "total_time_hr": (
                    (timestamps[12] - timestamps[0]).total_seconds() / 3600
                ),
                "_time_usage_sec": (
                    (timestamps[1] - timestamps[0]).total_seconds()
                    + (timestamps[7] - timestamps[6]).total_seconds()
                ),
                "train_time_usage_sec": (
                    (timestamps[5] - timestamps[1]).total_seconds()
                    + (timestamps[11] - timestamps[7]).total_seconds()
                ),
                "val_time_usage_sec": (
                    (timestamps[6] - timestamps[5]).total_seconds()
                    + (timestamps[12] - timestamps[11]).total_seconds()
                ),
                "_time_usage_relative": (
                    (
                        (timestamps[1] - timestamps[0]).total_seconds()
                        + (timestamps[7] - timestamps[6]).total_seconds()
                    )
                    / (timestamps[12] - timestamps[0]).total_seconds()
                ),
                "train_time_usage_relative": (
                    (
                        (timestamps[5] - timestamps[1]).total_seconds()
                        + (timestamps[11] - timestamps[7]).total_seconds()
                    )
                    / (timestamps[12] - timestamps[0]).total_seconds()
                ),
                "val_time_usage_relative": (
                    (
                        (timestamps[6] - timestamps[5]).total_seconds()
                        + (timestamps[12] - timestamps[11]).total_seconds()
                    )
                    / (timestamps[12] - timestamps[0]).total_seconds()
                ),
                "val_metrics/loss": (
                    (losses[28] + losses[29] + losses[30] + losses[31]) / 4
                ),
                "val_metrics/CER (argmax)": (
                    (metrics[28] + metrics[29] + metrics[30] + metrics[31]) / 4
                ),
            },
        ),
        # self.writer.set_step(23, "train")
        # self.writer.wandb.log({"train_metrics/epoch": 3}, step=24)
        (
            24,
            {
                "train_metrics/epoch": 3,
            },
        ),
        # self.writer.set_step(28, "train")
        # self.writer.add_scalars({"epoch": 3, "learning_rate": self.lr_scheduler.get_last_lr()[0]})
        # self.writer.add_scalars({"loss": ..., "grad_norm": ..., "CER (argmax)": ...})
        # self.writer.write()
        (
            28,
            {
                "steps_logged": 5,
                "train_steps_per_sec": (
                    5 / (timestamps[14] - timestamps[13]).total_seconds()
                ),
                "seconds_per_train_step": (
                    (timestamps[14] - timestamps[13]).total_seconds() / 5
                ),
                "total_time_sec": (timestamps[14] - timestamps[0]).total_seconds(),
                "total_time_hr": (
                    (timestamps[14] - timestamps[0]).total_seconds() / 3600
                ),
                "_time_usage_sec": (
                    (timestamps[1] - timestamps[0]).total_seconds()
                    + (timestamps[7] - timestamps[6]).total_seconds()
                    + (timestamps[13] - timestamps[12]).total_seconds()
                ),
                "train_time_usage_sec": (
                    (timestamps[5] - timestamps[1]).total_seconds()
                    + (timestamps[11] - timestamps[7]).total_seconds()
                    + (timestamps[14] - timestamps[13]).total_seconds()
                ),
                "val_time_usage_sec": (
                    (timestamps[6] - timestamps[5]).total_seconds()
                    + (timestamps[12] - timestamps[11]).total_seconds()
                ),
                "_time_usage_relative": (
                    (
                        (timestamps[1] - timestamps[0]).total_seconds()
                        + (timestamps[7] - timestamps[6]).total_seconds()
                        + (timestamps[13] - timestamps[12]).total_seconds()
                    )
                    / (timestamps[14] - timestamps[0]).total_seconds()
                ),
                "train_time_usage_relative": (
                    (
                        (timestamps[5] - timestamps[1]).total_seconds()
                        + (timestamps[11] - timestamps[7]).total_seconds()
                        + (timestamps[14] - timestamps[13]).total_seconds()
                    )
                    / (timestamps[14] - timestamps[0]).total_seconds()
                ),
                "val_time_usage_relative": (
                    (
                        (timestamps[6] - timestamps[5]).total_seconds()
                        + (timestamps[12] - timestamps[11]).total_seconds()
                    )
                    / (timestamps[14] - timestamps[0]).total_seconds()
                ),
                "train_metrics/epoch": 3,
                "train_metrics/learning_rate": lrs[28],
                "train_metrics/loss": (
                    (losses[32] + losses[33] + losses[34] + losses[35] + losses[36]) / 5
                ),
                "train_metrics/grad_norm": (
                    (grads[24] + grads[25] + grads[26] + grads[27] + grads[28]) / 5
                ),
                "train_metrics/CER (argmax)": (
                    (
                        metrics[32]
                        + metrics[33]
                        + metrics[34]
                        + metrics[35]
                        + metrics[36]
                    )
                    / 5
                ),
            },
        ),
        # self.writer.set_step(33, "train")
        # self.writer.add_scalars({"epoch": 3, "learning_rate": self.lr_scheduler.get_last_lr()[0]})
        # self.writer.add_scalars({"loss": ..., "grad_norm": ..., "CER (argmax)": ...})
        # self.writer.write()
        (
            33,
            {
                "steps_logged": 5,
                "train_steps_per_sec": (
                    5 / (timestamps[15] - timestamps[14]).total_seconds()
                ),
                "seconds_per_train_step": (
                    (timestamps[15] - timestamps[14]).total_seconds() / 5
                ),
                "total_time_sec": (timestamps[15] - timestamps[0]).total_seconds(),
                "total_time_hr": (
                    (timestamps[15] - timestamps[0]).total_seconds() / 3600
                ),
                "_time_usage_sec": (
                    (timestamps[1] - timestamps[0]).total_seconds()
                    + (timestamps[7] - timestamps[6]).total_seconds()
                    + (timestamps[13] - timestamps[12]).total_seconds()
                ),
                "train_time_usage_sec": (
                    (timestamps[5] - timestamps[1]).total_seconds()
                    + (timestamps[11] - timestamps[7]).total_seconds()
                    + (timestamps[15] - timestamps[13]).total_seconds()
                ),
                "val_time_usage_sec": (
                    (timestamps[6] - timestamps[5]).total_seconds()
                    + (timestamps[12] - timestamps[11]).total_seconds()
                ),
                "_time_usage_relative": (
                    (
                        (timestamps[1] - timestamps[0]).total_seconds()
                        + (timestamps[7] - timestamps[6]).total_seconds()
                        + (timestamps[13] - timestamps[12]).total_seconds()
                    )
                    / (timestamps[15] - timestamps[0]).total_seconds()
                ),
                "train_time_usage_relative": (
                    (
                        (timestamps[5] - timestamps[1]).total_seconds()
                        + (timestamps[11] - timestamps[7]).total_seconds()
                        + (timestamps[15] - timestamps[13]).total_seconds()
                    )
                    / (timestamps[15] - timestamps[0]).total_seconds()
                ),
                "val_time_usage_relative": (
                    (
                        (timestamps[6] - timestamps[5]).total_seconds()
                        + (timestamps[12] - timestamps[11]).total_seconds()
                    )
                    / (timestamps[15] - timestamps[0]).total_seconds()
                ),
                "train_metrics/epoch": 3,
                "train_metrics/learning_rate": lrs[33],
                "train_metrics/loss": (
                    (losses[37] + losses[38] + losses[39] + losses[40] + losses[41]) / 5
                ),
                "train_metrics/grad_norm": (
                    (grads[29] + grads[30] + grads[31] + grads[32] + grads[33]) / 5
                ),
                "train_metrics/CER (argmax)": (
                    (
                        metrics[37]
                        + metrics[38]
                        + metrics[39]
                        + metrics[40]
                        + metrics[41]
                    )
                    / 5
                ),
            },
        ),
        # self.writer.set_step(35, "train")
        # self.writer.add_scalars({"epoch": 3, "learning_rate": self.lr_scheduler.get_last_lr()[0]})
        # self.writer.add_scalars({"loss": ..., "grad_norm": ..., "CER (argmax)": ...})
        # self.writer.write()
        (
            35,
            {
                "steps_logged": 2,
                "train_steps_per_sec": (
                    2 / (timestamps[16] - timestamps[15]).total_seconds()
                ),
                "seconds_per_train_step": (
                    (timestamps[16] - timestamps[15]).total_seconds() / 2
                ),
                "total_time_sec": (timestamps[16] - timestamps[0]).total_seconds(),
                "total_time_hr": (
                    (timestamps[16] - timestamps[0]).total_seconds() / 3600
                ),
                "_time_usage_sec": (
                    (timestamps[1] - timestamps[0]).total_seconds()
                    + (timestamps[7] - timestamps[6]).total_seconds()
                    + (timestamps[13] - timestamps[12]).total_seconds()
                ),
                "train_time_usage_sec": (
                    (timestamps[5] - timestamps[1]).total_seconds()
                    + (timestamps[11] - timestamps[7]).total_seconds()
                    + (timestamps[16] - timestamps[13]).total_seconds()
                ),
                "val_time_usage_sec": (
                    (timestamps[6] - timestamps[5]).total_seconds()
                    + (timestamps[12] - timestamps[11]).total_seconds()
                ),
                "_time_usage_relative": (
                    (
                        (timestamps[1] - timestamps[0]).total_seconds()
                        + (timestamps[7] - timestamps[6]).total_seconds()
                        + (timestamps[13] - timestamps[12]).total_seconds()
                    )
                    / (timestamps[16] - timestamps[0]).total_seconds()
                ),
                "train_time_usage_relative": (
                    (
                        (timestamps[5] - timestamps[1]).total_seconds()
                        + (timestamps[11] - timestamps[7]).total_seconds()
                        + (timestamps[16] - timestamps[13]).total_seconds()
                    )
                    / (timestamps[16] - timestamps[0]).total_seconds()
                ),
                "val_time_usage_relative": (
                    (
                        (timestamps[6] - timestamps[5]).total_seconds()
                        + (timestamps[12] - timestamps[11]).total_seconds()
                    )
                    / (timestamps[16] - timestamps[0]).total_seconds()
                ),
                "train_metrics/epoch": 3,
                "train_metrics/learning_rate": lrs[35],
                "train_metrics/loss": (losses[42] + losses[43]) / 2,
                "train_metrics/grad_norm": (grads[34] + grads[35]) / 2,
                "train_metrics/CER (argmax)": (metrics[42] + metrics[43]) / 2,
            },
        ),
        # self.writer.set_step(35, "val")
        # self.writer.add_scalars({"loss": ..., "CER (argmax)": ...})
        # self.writer.set_step(35, "")
        # self.writer.write()
        (
            35,
            {
                "total_time_sec": (timestamps[18] - timestamps[0]).total_seconds(),
                "total_time_hr": (
                    (timestamps[18] - timestamps[0]).total_seconds() / 3600
                ),
                "_time_usage_sec": (
                    (timestamps[1] - timestamps[0]).total_seconds()
                    + (timestamps[7] - timestamps[6]).total_seconds()
                    + (timestamps[13] - timestamps[12]).total_seconds()
                ),
                "train_time_usage_sec": (
                    (timestamps[5] - timestamps[1]).total_seconds()
                    + (timestamps[11] - timestamps[7]).total_seconds()
                    + (timestamps[17] - timestamps[13]).total_seconds()
                ),
                "val_time_usage_sec": (
                    (timestamps[6] - timestamps[5]).total_seconds()
                    + (timestamps[12] - timestamps[11]).total_seconds()
                    + (timestamps[18] - timestamps[17]).total_seconds()
                ),
                "_time_usage_relative": (
                    (
                        (timestamps[1] - timestamps[0]).total_seconds()
                        + (timestamps[7] - timestamps[6]).total_seconds()
                        + (timestamps[13] - timestamps[12]).total_seconds()
                    )
                    / (timestamps[18] - timestamps[0]).total_seconds()
                ),
                "train_time_usage_relative": (
                    (
                        (timestamps[5] - timestamps[1]).total_seconds()
                        + (timestamps[11] - timestamps[7]).total_seconds()
                        + (timestamps[17] - timestamps[13]).total_seconds()
                    )
                    / (timestamps[18] - timestamps[0]).total_seconds()
                ),
                "val_time_usage_relative": (
                    (
                        (timestamps[6] - timestamps[5]).total_seconds()
                        + (timestamps[12] - timestamps[11]).total_seconds()
                        + (timestamps[18] - timestamps[17]).total_seconds()
                    )
                    / (timestamps[18] - timestamps[0]).total_seconds()
                ),
                "val_metrics/loss": (
                    (losses[44] + losses[45] + losses[46] + losses[47]) / 4
                ),
                "val_metrics/CER (argmax)": (
                    (metrics[44] + metrics[45] + metrics[46] + metrics[47]) / 4
                ),
            },
        ),
    ]


def check12(
    timestamps: Sequence[datetime],
    grads: Sequence[float],
    losses: Sequence[float],
    metrics: Sequence[float],
    lrs: Sequence[float],
):
    return [
        # self.writer.set_step(-1, "train")
        # self.writer.log_manual({"train_metrics/epoch": 1}, step=0)
        (
            0,
            {
                "train_metrics/epoch": 1,
            },
        ),
        # self.writer.set_step(11, "train")
        # self.writer.add_scalars({"epoch": 1, "learning_rate": self.lr_scheduler.get_last_lr()[0]})
        # self.writer.add_scalars({"loss": ..., "grad_norm": ..., "CER (argmax)": ...})
        # self.writer.write()
        (
            11,
            {
                "steps_logged": 12,
                "train_steps_per_sec": (
                    12 / (timestamps[2] - timestamps[1]).total_seconds()
                ),
                "seconds_per_train_step": (
                    (timestamps[2] - timestamps[1]).total_seconds() / 12
                ),
                "total_time_sec": (timestamps[2] - timestamps[0]).total_seconds(),
                "total_time_hr": (
                    (timestamps[2] - timestamps[0]).total_seconds() / 3600
                ),
                "_time_usage_sec": (timestamps[1] - timestamps[0]).total_seconds(),
                "train_time_usage_sec": (
                    (timestamps[2] - timestamps[1]).total_seconds()
                ),
                "val_time_usage_sec": 0.0,
                "_time_usage_relative": (
                    (timestamps[1] - timestamps[0]).total_seconds()
                    / (timestamps[2] - timestamps[0]).total_seconds()
                ),
                "train_time_usage_relative": (
                    (timestamps[2] - timestamps[1]).total_seconds()
                    / (timestamps[2] - timestamps[0]).total_seconds()
                ),
                "val_time_usage_relative": 0.0,
                "train_metrics/epoch": 1,
                "train_metrics/learning_rate": lrs[11],
                "train_metrics/loss": sum(losses[0:12]) / 12,
                "train_metrics/grad_norm": sum(grads[0:12]) / 12,
                "train_metrics/CER (argmax)": sum(metrics[0:12]) / 12,
            },
        ),
        # self.writer.set_step(11, "val")
        # self.writer.add_scalars({"loss": ..., "CER (argmax)": ...})
        # self.writer.set_step(11, "")
        # self.writer.write()
        (
            11,
            {
                "total_time_sec": (timestamps[4] - timestamps[0]).total_seconds(),
                "total_time_hr": (
                    (timestamps[4] - timestamps[0]).total_seconds() / 3600
                ),
                "_time_usage_sec": (timestamps[1] - timestamps[0]).total_seconds(),
                "train_time_usage_sec": (
                    (timestamps[3] - timestamps[1]).total_seconds()
                ),
                "val_time_usage_sec": (timestamps[4] - timestamps[3]).total_seconds(),
                "_time_usage_relative": (
                    (timestamps[1] - timestamps[0]).total_seconds()
                    / (timestamps[4] - timestamps[0]).total_seconds()
                ),
                "train_time_usage_relative": (
                    (timestamps[3] - timestamps[1]).total_seconds()
                    / (timestamps[4] - timestamps[0]).total_seconds()
                ),
                "val_time_usage_relative": (
                    (timestamps[4] - timestamps[3]).total_seconds()
                    / (timestamps[4] - timestamps[0]).total_seconds()
                ),
                "val_metrics/loss": (
                    (losses[12] + losses[13] + losses[14] + losses[15]) / 4
                ),
                "val_metrics/CER (argmax)": (
                    (metrics[12] + metrics[13] + metrics[14] + metrics[15]) / 4
                ),
            },
        ),
        # self.writer.set_step(11, "train")
        # self.writer.wandb.log({"train_metrics/epoch": 2}, step=12)
        (
            12,
            {
                "train_metrics/epoch": 2,
            },
        ),
        # self.writer.set_step(23, "train")
        # self.writer.add_scalars({"epoch": 2, "learning_rate": self.lr_scheduler.get_last_lr()[0]})
        # self.writer.add_scalars({"loss": ..., "grad_norm": ..., "CER (argmax)": ...})
        # self.writer.write()
        (
            23,
            {
                "steps_logged": 12,
                "train_steps_per_sec": (
                    12 / (timestamps[6] - timestamps[5]).total_seconds()
                ),
                "seconds_per_train_step": (
                    (timestamps[6] - timestamps[5]).total_seconds() / 12
                ),
                "total_time_sec": (timestamps[6] - timestamps[0]).total_seconds(),
                "total_time_hr": (
                    (timestamps[6] - timestamps[0]).total_seconds() / 3600
                ),
                "_time_usage_sec": (
                    (timestamps[1] - timestamps[0]).total_seconds()
                    + (timestamps[5] - timestamps[4]).total_seconds()
                ),
                "train_time_usage_sec": (
                    (timestamps[3] - timestamps[1]).total_seconds()
                    + (timestamps[6] - timestamps[5]).total_seconds()
                ),
                "val_time_usage_sec": (timestamps[4] - timestamps[3]).total_seconds(),
                "_time_usage_relative": (
                    (
                        (timestamps[1] - timestamps[0]).total_seconds()
                        + (timestamps[5] - timestamps[4]).total_seconds()
                    )
                    / (timestamps[6] - timestamps[0]).total_seconds()
                ),
                "train_time_usage_relative": (
                    (
                        (timestamps[3] - timestamps[1]).total_seconds()
                        + (timestamps[6] - timestamps[5]).total_seconds()
                    )
                    / (timestamps[6] - timestamps[0]).total_seconds()
                ),
                "val_time_usage_relative": (
                    (timestamps[4] - timestamps[3]).total_seconds()
                    / (timestamps[6] - timestamps[0]).total_seconds()
                ),
                "train_metrics/epoch": 2,
                "train_metrics/learning_rate": lrs[23],
                "train_metrics/loss": sum(losses[16:28]) / 12,
                "train_metrics/grad_norm": sum(grads[12:24]) / 12,
                "train_metrics/CER (argmax)": sum(metrics[16:28]) / 12,
            },
        ),
        # self.writer.set_step(23, "val")
        # self.writer.add_scalars({"loss": ..., "CER (argmax)": ...})
        # self.writer.set_step(23, "")
        # self.writer.write()
        (
            23,
            {
                "total_time_sec": (timestamps[8] - timestamps[0]).total_seconds(),
                "total_time_hr": (
                    (timestamps[8] - timestamps[0]).total_seconds() / 3600
                ),
                "_time_usage_sec": (
                    (timestamps[1] - timestamps[0]).total_seconds()
                    + (timestamps[5] - timestamps[4]).total_seconds()
                ),
                "train_time_usage_sec": (
                    (timestamps[3] - timestamps[1]).total_seconds()
                    + (timestamps[7] - timestamps[5]).total_seconds()
                ),
                "val_time_usage_sec": (
                    (timestamps[4] - timestamps[3]).total_seconds()
                    + (timestamps[8] - timestamps[7]).total_seconds()
                ),
                "_time_usage_relative": (
                    (
                        (timestamps[1] - timestamps[0]).total_seconds()
                        + (timestamps[5] - timestamps[4]).total_seconds()
                    )
                    / (timestamps[8] - timestamps[0]).total_seconds()
                ),
                "train_time_usage_relative": (
                    (
                        (timestamps[3] - timestamps[1]).total_seconds()
                        + (timestamps[7] - timestamps[5]).total_seconds()
                    )
                    / (timestamps[8] - timestamps[0]).total_seconds()
                ),
                "val_time_usage_relative": (
                    (
                        (timestamps[4] - timestamps[3]).total_seconds()
                        + (timestamps[8] - timestamps[7]).total_seconds()
                    )
                    / (timestamps[8] - timestamps[0]).total_seconds()
                ),
                "val_metrics/loss": (
                    (losses[28] + losses[29] + losses[30] + losses[31]) / 4
                ),
                "val_metrics/CER (argmax)": (
                    (metrics[28] + metrics[29] + metrics[30] + metrics[31]) / 4
                ),
            },
        ),
        # self.writer.set_step(23, "train")
        # self.writer.wandb.log({"train_metrics/epoch": 3}, step=24)
        (
            24,
            {
                "train_metrics/epoch": 3,
            },
        ),
        # self.writer.set_step(35, "train")
        # self.writer.add_scalars({"epoch": 1, "learning_rate": self.lr_scheduler.get_last_lr()[0]})
        # self.writer.add_scalars({"loss": ..., "grad_norm": ..., "CER (argmax)": ...})
        # self.writer.write()
        (
            35,
            {
                "steps_logged": 12,
                "train_steps_per_sec": (
                    12 / (timestamps[10] - timestamps[9]).total_seconds()
                ),
                "seconds_per_train_step": (
                    (timestamps[10] - timestamps[9]).total_seconds() / 12
                ),
                "total_time_sec": (timestamps[10] - timestamps[0]).total_seconds(),
                "total_time_hr": (
                    (timestamps[10] - timestamps[0]).total_seconds() / 3600
                ),
                "_time_usage_sec": (
                    (timestamps[1] - timestamps[0]).total_seconds()
                    + (timestamps[5] - timestamps[4]).total_seconds()
                    + (timestamps[9] - timestamps[8]).total_seconds()
                ),
                "train_time_usage_sec": (
                    (timestamps[3] - timestamps[1]).total_seconds()
                    + (timestamps[7] - timestamps[5]).total_seconds()
                    + (timestamps[10] - timestamps[9]).total_seconds()
                ),
                "val_time_usage_sec": (
                    (timestamps[4] - timestamps[3]).total_seconds()
                    + (timestamps[8] - timestamps[7]).total_seconds()
                ),
                "_time_usage_relative": (
                    (
                        (timestamps[1] - timestamps[0]).total_seconds()
                        + (timestamps[5] - timestamps[4]).total_seconds()
                        + (timestamps[9] - timestamps[8]).total_seconds()
                    )
                    / (timestamps[10] - timestamps[0]).total_seconds()
                ),
                "train_time_usage_relative": (
                    (
                        (timestamps[3] - timestamps[1]).total_seconds()
                        + (timestamps[7] - timestamps[5]).total_seconds()
                        + (timestamps[10] - timestamps[9]).total_seconds()
                    )
                    / (timestamps[10] - timestamps[0]).total_seconds()
                ),
                "val_time_usage_relative": (
                    (
                        (timestamps[4] - timestamps[3]).total_seconds()
                        + (timestamps[8] - timestamps[7]).total_seconds()
                    )
                    / (timestamps[10] - timestamps[0]).total_seconds()
                ),
                "train_metrics/epoch": 3,
                "train_metrics/learning_rate": lrs[35],
                "train_metrics/loss": sum(losses[32:44]) / 12,
                "train_metrics/grad_norm": sum(grads[24:36]) / 12,
                "train_metrics/CER (argmax)": sum(metrics[32:44]) / 12,
            },
        ),
        # self.writer.set_step(35, "val")
        # self.writer.add_scalars({"loss": ..., "CER (argmax)": ...})
        # self.writer.set_step(35, "")
        # self.writer.write()
        (
            35,
            {
                "total_time_sec": (timestamps[12] - timestamps[0]).total_seconds(),
                "total_time_hr": (
                    (timestamps[12] - timestamps[0]).total_seconds() / 3600
                ),
                "_time_usage_sec": (
                    (timestamps[1] - timestamps[0]).total_seconds()
                    + (timestamps[5] - timestamps[4]).total_seconds()
                    + (timestamps[9] - timestamps[8]).total_seconds()
                ),
                "train_time_usage_sec": (
                    (timestamps[3] - timestamps[1]).total_seconds()
                    + (timestamps[7] - timestamps[5]).total_seconds()
                    + (timestamps[11] - timestamps[9]).total_seconds()
                ),
                "val_time_usage_sec": (
                    (timestamps[4] - timestamps[3]).total_seconds()
                    + (timestamps[8] - timestamps[7]).total_seconds()
                    + (timestamps[12] - timestamps[11]).total_seconds()
                ),
                "_time_usage_relative": (
                    (
                        (timestamps[1] - timestamps[0]).total_seconds()
                        + (timestamps[5] - timestamps[4]).total_seconds()
                        + (timestamps[9] - timestamps[8]).total_seconds()
                    )
                    / (timestamps[12] - timestamps[0]).total_seconds()
                ),
                "train_time_usage_relative": (
                    (
                        (timestamps[3] - timestamps[1]).total_seconds()
                        + (timestamps[7] - timestamps[5]).total_seconds()
                        + (timestamps[11] - timestamps[9]).total_seconds()
                    )
                    / (timestamps[12] - timestamps[0]).total_seconds()
                ),
                "val_time_usage_relative": (
                    (
                        (timestamps[4] - timestamps[3]).total_seconds()
                        + (timestamps[8] - timestamps[7]).total_seconds()
                        + (timestamps[12] - timestamps[11]).total_seconds()
                    )
                    / (timestamps[12] - timestamps[0]).total_seconds()
                ),
                "val_metrics/loss": (
                    (losses[44] + losses[45] + losses[46] + losses[47]) / 4
                ),
                "val_metrics/CER (argmax)": (
                    (metrics[44] + metrics[45] + metrics[46] + metrics[47]) / 4
                ),
            },
        ),
    ]
