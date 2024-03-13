# ASR Project

This project is a continuation of a homework of the course Deep Learning in Audio. It is a training framework for speech recognition models. It offers extensive `wandb` logging, Python data type support and a lot of tests.

## Installation guide

This project has been developed in a Docker container using VSCode. All necessary dependencies can be found in a docker file.

## Bug report

I have found a bug in the original template. The training loop seemed to do more steps than specified in the config. It turns out that in `asr_project_template/src/trainer/trainer.py`, line 87, the loop looks like this:

```python
for batch_idx in range(inf):
    # ...
    if batch_idx >= self.len_epoch:
        break
```

So basically the epoch had the length of `self.len_epoch + 1`, which is counterintuitive.
