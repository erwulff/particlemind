import torch
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from lightning.fabric.utilities.rank_zero import rank_zero_only



@rank_zero_only
def print_rank0(*args, **kwargs):
    print(*args, **kwargs)

def configure_optimizers_base(self):
        # --- Optimizer --- #
        optimizer = torch.optim.AdamW(self.model.parameters(), **self.optimizer_kwargs)


        # --- Scheduler --- #
        if self.lr_scheduler_kwargs["use_scheduler"]:
            # Compute total_steps manually
            num_gpus = self.trainer.num_devices  # 4 in your case
            accumulate_grad_batches = self.trainer.accumulate_grad_batches  # 32

        
            global_batch_size = self.batch_size_per_gpu * num_gpus
            steps_per_epoch = (self.num_train_events + global_batch_size - 1) // global_batch_size
            total_steps = int(steps_per_epoch * self.trainer.max_epochs)
            warmup_frac = self.lr_scheduler_kwargs.get("warmup_frac", 0.01)
            warmup_steps = max(int(total_steps * warmup_frac), 1)
            cosine_steps = total_steps - warmup_steps
        
            print_rank0("warmup_steps:", warmup_steps)
            print_rank0("total_steps:", total_steps)
            print_rank0("cosine_steps:", cosine_steps)
    
            # Linear warmup
            warmup_scheduler = LinearLR(
                optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps
            )
            # Cosine decay
            cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=1e-6)
    
            # Combine schedulers
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps],
            )
    
            # Lightning dict format
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",      # step-wise LR updates
                    "frequency": 1,
                    "name": "lr",            # wandb logging name
                },
            }
    
        return optimizer