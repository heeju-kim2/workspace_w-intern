import math
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from datetime import datetime
from tqdm import tqdm
import torch
import os 
import gc 
import numpy as np
import evaluate
import random
import wandb
from peft import PeftModel
import json

def save_model_hook(models, weights, output_dir):
    for model in models:
        model.save_pretrained(output_dir)
        # make sure to pop weight so that corresponding model is not saved again
        weights.pop()

def load_model_hook(models, input_dir):
    while len(models) > 0:
        model = models.pop()
        # pop models so that they are not loaded again
        PeftModel.from_pretrained(model.base_model.model, input_dir)

def evaluation(model, eval_dataloader, processor, metric, forced_decoder_ids, accelerator):
    model.eval()
    predictions = []
    references = []
    normalized_predictions = []
    normalized_references = []
    normalizer = BasicTextNormalizer()
    for _, batch in enumerate(tqdm(eval_dataloader)):
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                generated_tokens = (
                    model.generate(
                        input_features=batch["input_features"],
                        forced_decoder_ids=forced_decoder_ids,
                        max_new_tokens=255,
                    )
                    .cpu()
                    .numpy()
                )
                labels = batch["labels"].cpu().numpy()
                labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
                decoded_preds = processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
                predictions.extend(decoded_preds)
                references.extend(decoded_labels)
                normalized_predictions.extend([normalizer(pred).strip() for pred in decoded_preds])
                normalized_references.extend([normalizer(label).strip() for label in decoded_labels])
            del generated_tokens, labels, batch
        gc.collect()
    wer = 100 * metric.compute(predictions=predictions, references=references)
    normalized_wer = 100 * metric.compute(predictions=normalized_predictions, references=normalized_references)
    eval_metrics = {"eval/wer": wer, "eval/normalized_wer": normalized_wer}
    if accelerator.get_tracker("wandb"):
        sample_size = min(len(predictions), 256)
        ids = [random.randint(0, len(predictions) - 1) for p in range(0, sample_size)]
        sample_predictions = [predictions[i] for i in ids]
        sample_references = [references[i] for i in ids]
        sample_normalized_predictions = [normalized_predictions[i] for i in ids]
        sample_normalized_references = [normalized_references[i] for i in ids]
        table_rows = [
            list(r)
            for r in zip(
                sample_predictions, sample_references, sample_normalized_predictions, sample_normalized_references
            )
        ]
        eval_metrics["eval_samples"] = wandb.Table(
            columns=["predictions", "references", "normalized_predictions", "normalized_references"],
            rows=table_rows,
        )
    return eval_metrics

def train(model,
        processor, 
        train_dataloader,
        eval_dataloader,
        optimizer,
        lr_scheduler,
        train_config,
        logger,
        accelerator,
    ):

    model.train()
    # prepares data and models for distributed processing by moving them to chips
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler)

    ## delete
    num_epochs = train_config.num_epochs
    gradient_accumulation_steps = train_config.gradient_accumulation_steps
    run_eval = train_config.run_eval
    max_train_step = train_config.max_train_step
    gradient_clipping = train_config.gradient_clipping
    gradient_clipping_threshold = train_config.gradient_clipping_threshold

    # Note here that the max steps is adjusted by the accelerator's num_processes
    max_train_step = math.ceil(train_config.max_train_step / accelerator.num_processes)
    if train_config.use_peft and train_config.peft_method == "adalora":
        model.base_model.peft_config["default"].total_step = args.max_train_steps
        # model.base_model.peft_config.total_step = args.max_train_steps

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if train_config.with_tracking:
        run_name = f"run-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        accelerator.init_trackers(
            "Whisper PEFT Fine-Tuning", init_kwargs={"wandb": {"name": run_name}}
        )

    # saving and loading checkpoints for resuming training
    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)


    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(max_train_step), disable=not accelerator.is_local_main_process)
    global_step = 0
    starting_epoch = 0
    best_metric = None
    resume_step = 0
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=train_config.language, task=train_config.task)

    # We need to adjust the progress bar to the current step
    tot_eval_metrics = dict()
    progress_bar.update(resume_step)
    for epoch in range(starting_epoch, num_epochs):
        model.train()
        if train_config.with_tracking:
            total_loss = 0
            running_loss = 0
        for step, batch in enumerate(accelerator.skip_first_batches(train_dataloader, num_batches=resume_step)):
            with accelerator.accumulate(model):
                """
                input_features
                labels
                """
                batch['input_features'] = batch['input_features'].to(torch.bfloat16)
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                
                # Update the importance of low-rank matrices
                # and allocate the budget accordingly.
                # This is only needed for AdaLora.
                # Note that this requires parameter gradients.
                # Hence being called before optimizer.zero_grad().
                if train_config.use_peft and train_config.peft_method == "adalora":
                    model.update_and_allocate(global_step)

                optimizer.zero_grad()
                global_step += 1
                progress_bar.update(1)

            if train_config.with_tracking:
                step_loss = accelerator.reduce(loss.detach().clone()).item()
                total_loss += step_loss
                running_loss += step_loss

            if global_step % train_config.logging_steps == 0:
                if train_config.with_tracking:
                    accelerator.log({"train/running_loss": running_loss / args.logging_steps}, step=global_step)
                    running_loss = 0

                
            if global_step >= max_train_step:
                break
        # epoch end
        lr_scheduler.step()
        output_dir = os.path.join(train_config.output_dir, f"epoch_{epoch}")
        accelerator.save_state(output_dir)

        metric = evaluate.load(train_config.eval_metric)
        eval_metrics = evaluation(
                    model, eval_dataloader, processor, metric, forced_decoder_ids, accelerator
                )
        eval_metrics.pop("eval_samples")
        tot_eval_metrics[f"epoch_{epoch}"] = eval_metrics 
        if train_config.with_tracking:
            logger.info(f"Step {global_step} eval metrics: {eval_metrics}")
            accelerator.log(eval_metrics, step=global_step)
        if best_metric is None or eval_metrics["eval/wer"] < best_metric:
            best_metric = eval_metrics["eval/wer"]
            accelerator.save_state(os.path.join(train_config.output_dir, "best_checkpoint"))

        if train_config.with_tracking:
            train_epoch_loss = total_loss / (step + 1)
            logger.info(f"Epoch {epoch} train loss: {train_epoch_loss}")
            accelerator.log({"epoch/train_loss": train_epoch_loss}, step=epoch)

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(train_config.output_dir, is_main_process=accelerator.is_main_process)
    if accelerator.is_main_process:
        processor.tokenizer.save_pretrained(train_config.output_dir)

    with open(os.path.join(train_config.output_dir, "all_results.json"), "w") as f:
        json.dump(tot_eval_metrics, f)
    
    return eval_metrics