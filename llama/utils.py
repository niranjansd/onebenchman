import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTTrainer
import params
import dataset_lib.dolly as dolly

def load_model(model_name):
    # Load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, params.bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=params.use_4bit,
        bnb_4bit_quant_type=params.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=params.use_nested_quant,
    )

    if compute_dtype == torch.float16 and params.use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
            print("=" * 80)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=params.device_map,
        quantization_config=bnb_config
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=params.lora_alpha,
        lora_dropout=params.lora_dropout,
        r=params.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer, peft_config


# template dataset to add prompt to each sample
def template_dataset(sample, tokenizer):
    sample["text"] = f"{dolly.format(sample)}{tokenizer.eos_token}"
    return sample


def load_hf_dataset(dataset_name="databricks/databricks-dolly-15k", split="train"):

    # apply prompt template per sample
    dataset = load_dataset(dataset_name, split=split)

    # Shuffle the dataset
    dataset_shuffled = dataset.shuffle(seed=42)

    # Select the first 50 rows from the shuffled dataset, comment if you want 15k
    dataset = dataset_shuffled.select(range(50))

    dataset = dataset.map(template_dataset, remove_columns=list(dataset.features))
    return dataset    


def finetune(params, model, dataset, tokenizer, peft_config):
    training_arguments = TrainingArguments(
        output_dir=params.output_dir,
        per_device_train_batch_size=params.per_device_train_batch_size,
        gradient_accumulation_steps=params.gradient_accumulation_steps,
        optim=params.optim,
        save_steps=params.save_steps,
        logging_steps=params.logging_steps,
        learning_rate=params.learning_rate,
        fp16=params.fp16,
        bf16=params.bf16,
        max_grad_norm=params.max_grad_norm,
        max_steps=params.max_steps,
        warmup_ratio=params.warmup_ratio,
        group_by_length=params.group_by_length,
        lr_scheduler_type=params.lr_scheduler_type,
        report_to="tensorboard"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=params.max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=params.packing,
    )

    trainer.train()
    trainer.model.save_pretrained(params.output_dir)


def load_lora(model_name, lora_dir, device_map):
    # Reload model in FP16 and merge it with LoRA weights
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    model = PeftModel.from_pretrained(base_model, lora_dir)
    model = model.merge_and_unload()

    # Reload tokenizer to save it
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"    
    return model, tokenizer