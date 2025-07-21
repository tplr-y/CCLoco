# Standard library
import contextlib
from abc import ABC, abstractmethod

# Third party
import torch
import torch.distributed as dist

# Local
import tplr


class InnerOuterStrategy(ABC):
    @abstractmethod
    def inner_step(self, model, loader, inner_optimizer=None, inner_scheduler=None):
        """Execute inner optimization step"""
        pass

    @abstractmethod
    def outer_step(self, model, optimizer, scheduler=None):
        """Execute outer optimization step"""
        pass


class SimpleAccum(InnerOuterStrategy):
    def __init__(self, device, world_size, global_rank, tokenizer, config):
        self.device = device
        self.world_size = world_size
        self.global_rank = global_rank
        self.tokenizer = tokenizer
        self.config = config
        self.compressive_optimizer = self.config.outer_optimizer in ["ccloco", "demo"]

    def inner_step(self, model, loader, inner_optimizer=None, inner_scheduler=None):
        total_loss = 0
        batch_tokens = 0
        batch_count = 0
        accum_batch_size = 0

        ddp_context = (
            model.no_sync()
            if hasattr(model, "no_sync") and self.world_size > 1
            else contextlib.nullcontext()
        )

        with ddp_context:
            for i, batch in enumerate(loader):
                input_ids = batch.to(self.device, non_blocking=True)
                labels = input_ids.clone()
                labels[labels == self.tokenizer.pad_token_id] = -100

                accum_batch_size += len(batch)

                with torch.amp.autocast(
                    device_type=self.device.type, dtype=torch.bfloat16
                ):
                    outputs = model(input_ids=input_ids, labels=labels)

                loss = outputs.loss / (
                    self.config.batch_size / self.config.micro_batch_size
                )
                loss.backward()

                total_loss += loss.item()
                batch_count += 1
                batch_tokens += (labels != -100).sum().item()

                if self.global_rank == 0 and i % 20 == 0:
                    tplr.logger.info(
                        f"Batch {i}, loss: {outputs.loss.item():.4f}, accum: {accum_batch_size}/{self.config.batch_size}"
                    )

                if accum_batch_size >= self.config.batch_size:
                    break

        return {
            "total_loss": total_loss,
            "loss_after_gather": total_loss,
            "batch_count": batch_count,
            "batch_tokens": batch_tokens,
        }

    def outer_step(self, model, optimizer, scheduler):
        # Compressive optimizers handle node communications internally
        if self.world_size > 1 and not self.compressive_optimizer:
            actual_model = model.module if hasattr(model, "module") else model
            for param in actual_model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

        optimizer.step()
        if scheduler:
            scheduler.step()


class Diloco(InnerOuterStrategy):
    def __init__(self, device, world_size, global_rank, tokenizer, config):
        self.device = device
        self.world_size = world_size
        self.global_rank = global_rank
        self.tokenizer = tokenizer
        self.config = config
        self.params_offloaded = None
        self.compressive_optimizer = self.config.outer_optimizer in ["ccloco", "demo"]

    def inner_step(self, model, loader, inner_optimizer, inner_scheduler):
        total_loss, batch_tokens, batch_count = 0, 0, 0
        loss_after_gather = 0.0

        self.params_offloaded = self._get_offloaded_param(model)

        ddp_context = (
            model.no_sync()
            if hasattr(model, "no_sync") and self.world_size > 1
            else contextlib.nullcontext()
        )

        inner_step_count = 0
        accum_batch_size = 0

        with ddp_context:
            for i, batch in enumerate(loader):
                input_ids = batch.to(self.device, non_blocking=True)
                labels = input_ids.clone()
                labels[labels == self.tokenizer.pad_token_id] = -100
                accum_batch_size += len(batch)

                with torch.amp.autocast(
                    device_type=self.device.type, dtype=torch.bfloat16
                ):
                    outputs = model(input_ids=input_ids, labels=labels)

                loss = outputs.loss / (
                    self.config.batch_size / self.config.micro_batch_size
                )  # aka accumulation steps
                loss.backward()

                total_loss += loss.item()
                batch_count += 1
                batch_tokens += (labels != -100).sum().item()

                if accum_batch_size >= self.config.batch_size:
                    if inner_step_count == 0:
                        loss_after_gather = total_loss
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    inner_optimizer.step()
                    inner_scheduler.step()
                    inner_optimizer.zero_grad()

                    if self.global_rank == 0 and inner_step_count % 5 == 0:
                        tplr.logger.info(
                            f"Inner Step {inner_step_count + 1}/{self.config.inner_steps}, loss: {outputs.loss.item():.4f}"
                        )

                    inner_step_count += 1
                    accum_batch_size = 0

                    if inner_step_count >= self.config.inner_steps:
                        break

        return {
            "total_loss": total_loss / inner_step_count,
            "loss_after_gather": loss_after_gather,
            "batch_count": batch_count,
            "batch_tokens": batch_tokens,
        }

    def outer_step(self, model, optimizer, scheduler=None):
        if self.params_offloaded is None:
            raise ValueError("No offloaded parameters found. Run inner_step first.")

        actual_model = model.module if hasattr(model, "module") else model

        for param_offloaded, param in zip(
            self.params_offloaded, actual_model.parameters()
        ):
            saved_param = param_offloaded.to(param.device)
            param.grad = saved_param - param.data

            if self.world_size > 1 and not self.compressive_optimizer:
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

            param.data.copy_(saved_param)

        optimizer.step()

    def _get_offloaded_param(self, model):
        actual_model = model.module if hasattr(model, "module") else model
        return [
            param.data.detach().clone().to("cpu") for param in actual_model.parameters()
        ]
