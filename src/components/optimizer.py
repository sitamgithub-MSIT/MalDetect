# Import necessary libraries
import math
from keras import ops
from keras.optimizers import schedules
from keras.saving import register_keras_serializable
from typing import Dict, Any


@register_keras_serializable()
def lr_warmup_cosine_decay(
    global_step: int,
    warmup_steps: int,
    hold: int = 0,
    total_steps: int = 0,
    start_lr: float = 0.0,
    target_lr: float = 1e-2,
) -> float:
    """
    Applies a learning rate schedule that includes a warmup phase followed by a cosine decay.

    Args:
        - global_step (int): The current step in the training process.
        - warmup_steps (int): The number of steps to linearly increase the learning rate.
        - hold (int, optional): The number of steps to hold the learning rate at the target value before decaying. Defaults to 0.
        - total_steps (int, optional): The total number of training steps. Defaults to 0.
        - start_lr (float, optional): The initial learning rate at the start of warmup. Defaults to 0.0.
        - target_lr (float, optional): The target learning rate to reach after warmup and decay. Defaults to 1e-2.

    Returns:
        float: The calculated learning rate at the current global step.
    """
    # Cosine decay
    learning_rate = (
        0.5
        * target_lr
        * (
            1
            + ops.cos(
                math.pi
                * ops.convert_to_tensor(
                    global_step - warmup_steps - hold, dtype="float32"
                )
                / ops.convert_to_tensor(
                    total_steps - warmup_steps - hold, dtype="float32"
                )
            )
        )
    )

    warmup_lr = target_lr * (global_step / warmup_steps)

    if hold > 0:
        learning_rate = ops.where(
            global_step > warmup_steps + hold, learning_rate, target_lr
        )

    learning_rate = ops.where(global_step < warmup_steps, warmup_lr, learning_rate)
    return learning_rate


@register_keras_serializable()
class WarmUpCosineDecay(schedules.LearningRateSchedule):
    """
    A LearningRateSchedule that combines a warm-up phase with a cosine decay.

    Args:
        - warmup_steps (int): Number of steps to warm up the learning rate.
        - total_steps (int): Total number of steps for the learning rate schedule.
        - hold (int): Number of steps to hold the learning rate before decaying.
        - start_lr (float): Initial learning rate at the start of warm-up. Default is 0.0.
        - target_lr (float): Target learning rate at the end of warm-up. Default is 1e-2.

    Methods:
        __call__(step):
            Computes the learning rate at a given step.

        get_config():
            Returns the configuration of the learning rate schedule as a dictionary.

        from_config(config):
            Instantiates the learning rate schedule from a configuration dictionary.
    """

    def __init__(
        self,
        warmup_steps: int,
        total_steps: int,
        hold: int,
        start_lr: float = 0.0,
        target_lr: float = 1e-2,
    ) -> None:
        super().__init__()
        self.start_lr = start_lr
        self.target_lr = target_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.hold = hold

    def __call__(self, step: int) -> float:
        lr = lr_warmup_cosine_decay(
            global_step=step,
            total_steps=self.total_steps,
            warmup_steps=self.warmup_steps,
            start_lr=self.start_lr,
            target_lr=self.target_lr,
            hold=self.hold,
        )

        return ops.where(step > self.total_steps, 0.0, lr)

    def get_config(self) -> Dict[str, Any]:
        return {
            "start_lr": self.start_lr,
            "target_lr": self.target_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "hold": self.hold,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        return cls(**config)
