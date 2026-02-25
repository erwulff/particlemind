import math
import torch
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from lightning.fabric.utilities.rank_zero import rank_zero_only

@rank_zero_only
def print_rank0(*args, **kwargs):
    print(*args, **kwargs)

def configure_optimizers_base(self):
    # --- Optimizer --- #
    optimizer = torch.optim.AdamW(self.parameters(), **self.optimizer_kwargs)

    # --- Scheduler --- #
    if self.lr_scheduler_kwargs.get("use_scheduler", False):
        # --- compute total steps safely in setup / after trainer is attached --- #
        if not hasattr(self, "total_steps"):
            # number of devices (GPUs) and gradient accumulation
            num_gpus = self.trainer.num_devices
            accumulate_grad_batches = self.trainer.accumulate_grad_batches

            # global batch size per optimizer step
            global_batch_size = self.batch_size_per_gpu * num_gpus * accumulate_grad_batches

            # number of optimizer steps per epoch
            steps_per_epoch = math.ceil(self.num_train_events / global_batch_size)

            # total optimizer steps for all epochs
            total_steps = steps_per_epoch * self.trainer.max_epochs

            # warmup
            warmup_frac = self.lr_scheduler_kwargs.get("warmup_frac", 0.01)
            warmup_steps = max(int(total_steps * warmup_frac), 1)

            # cosine annealing
            cosine_steps = total_steps - warmup_steps

            # save for later use
            self.total_steps = total_steps
            self.warmup_steps = warmup_steps
            self.cosine_steps = cosine_steps

            print_rank0("warmup_steps:", warmup_steps)
            print_rank0("total_steps:", total_steps)
            print_rank0("cosine_steps:", cosine_steps)

        # --- Linear warmup scheduler --- #
        warmup_scheduler = LinearLR(
            optimizer, start_factor=1e-6, end_factor=1.0, total_iters=self.warmup_steps
        )

        # --- Cosine annealing scheduler --- #
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=self.cosine_steps, eta_min=1e-6
        )

        # --- Combine schedulers --- #
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.warmup_steps],
        )

        # --- Lightning dict format --- #
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # step-wise LR updates
                "frequency": 1,
                "name": "lr",
            },
        }

    return optimizer
