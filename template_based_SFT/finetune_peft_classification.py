import datetime, os, sys, json, torch, wandb
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from time import time
from datasets import load_dataset
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model, prepare_model_for_int8_training
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments, Trainer, TrainerCallback, set_seed
from template_dataset import get_prompt_dataset, get_eval_dataloader, collate_fn
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default="facebook/opt-125m",
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(
        default="nsmc", metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    block_size: Optional[int] = field(
        default=1024, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    max_len: Optional[int] = field(
        default=64, metadata={"help": "The maximum length of each data in dataset to use."}
    )
    max_label_len: Optional[int] = field(
        default=4, metadata={"help": "The maximum length of each label in dataset to use."}
    )
    prefix: Optional[str] = field(
        default="다음 문장은 긍정일까요 부정일까요?\n",
        metadata={"help": "The prefix string used for template based training."}
    )
    suffix: Optional[str] = field(
        default="\n정답:",
        metadata={"help": "The suffix string used for template based training."}
    )
    columns: Optional[List[str]] = field(
        default_factory=list,
        metadata={"help": "The column names in dataset used for template based training."}
    )
    labels_to_ids: Optional[Dict] = field(
        default_factory=dict,
        metadata={"help": "The labels with corresponding class id in classification"}
    )

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        json_file = os.path.abspath(sys.argv[1])
        model_args, data_args, training_args = parser.parse_json_file(json_file=json_file)
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    KST = datetime.timezone(datetime.timedelta(hours=9))
    timestamp = str(datetime.datetime.fromtimestamp(time(), tz=KST)).split()

    if not training_args.output_dir.endswith('/'):
        training_args.output_dir = training_args.output_dir + "/"
    training_args.output_dir = training_args.output_dir + timestamp[0] + "_" + timestamp[1][:8]
    training_args.run_name = training_args.run_name + "-" + timestamp[0] + "-" + timestamp[1][:8]

    os.makedirs(training_args.output_dir, exist_ok=True)
    with open(json_file, 'r') as openfile:
        config_object = json.load(openfile)
    with open(training_args.output_dir + '/running_config.json', "w") as outfile:
        json.dump(config_object, outfile)

    logger.add(training_args.output_dir + '/{time}_train.log')

    logger.info("Model arguments")
    for arg in str(model_args).split(','):
        logger.debug(arg.strip())

    logger.info("Data arguments")
    for arg in str(data_args).split(','):
        logger.debug(arg.strip())

    logger.info("Training arguments")
    for arg in str(training_args).split(','):
        logger.debug(arg.strip())

    # Set seed before initializing model.
    set_seed(training_args.seed)

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        load_in_8bit=True,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # ### Prepare model for training
    #
    # Some pre-processing needs to be done before training such an int8 model using `peft`, therefore let's import an utiliy function `prepare_model_for_int8_training` that will:
    # - Cast the layer norm in `float32` for stability purposes
    # - Add a `forward_hook` to the input embedding layer to enable gradient computation of the input hidden states
    # - Enable gradient checkpointing for more memory-efficient training
    # - Cast the output logits in `float32` for smoother sampling during the sampling procedure


    if "gpt-neox" in model_args.model_name_or_path:
        model = prepare_model_for_int8_training(
            model, output_embedding_layer_name="embed_out", layer_norm_names=["layer_norm", "layernorm"], cast_dtype=torch.float16
        )
    else:
        model = prepare_model_for_int8_training(model, cast_dtype=torch.float16)


    # ### Apply LoRA
    #
    # Here comes the magic with `peft`! Let's load a `PeftModel` and specify that we are going to use low-rank adapters (LoRA) using `get_peft_model` utility function from `peft`.
    def print_trainable_parameters(model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        logger.info(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )


    target_modules = None
    if "gpt-neox" in model_args.model_name_or_path:
        target_modules = ["query_key_value", "xxx"]  # workaround to use 8bit training on this model
    config = LoraConfig(
        r=16, lora_alpha=32, target_modules=target_modules, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)
    print_trainable_parameters(model)

    # load dataset
    ids_to_labels = {id:label for label, id in data_args.labels_to_ids.items()}
    if '/' in data_args.dataset_name:
        dataset_names = data_args.dataset_name.split('/')
        data = load_dataset(dataset_names[0], dataset_names[1])
    else:
        data = load_dataset(data_args.dataset_name)
    dataset_keys = [key for key in data.keys()]
    data = get_prompt_dataset(dataset=data,
                              tokenizer=tokenizer,
                              max_len=data_args.max_len,
                              max_label_len=data_args.max_label_len,
                              prefix=data_args.prefix,
                              suffix=data_args.suffix,
                              columns=data_args.columns,
                              ids_to_labels=ids_to_labels,                           
                            )
    
    if training_args.do_eval:
        eval_dataloader = get_eval_dataloader(data[dataset_keys[1]], training_args.per_device_eval_batch_size)

        def validation_step(model, tokenizer, batch, first):
            generated_ids = model.generate(
                input_ids=batch['input_ids'].to(model.device),
                attention_mask=batch['attention_mask'].to(model.device),
                max_new_tokens=data_args.max_label_len,
                eos_token_id = tokenizer.eos_token_id,
                pad_token_id = tokenizer.pad_token_id,
            )
            generated_txt = []
            for i, g in enumerate(generated_ids):
                decoded_txt = tokenizer.decode(g.tolist(), skip_special_tokens=True).split(data_args.suffix)
                generated_txt.append(decoded_txt[-1].strip())
            
            labels = batch['decoded_labels']

            if first:
                count = 0
                for gold, gen_txt in zip(labels, generated_txt):
                    logger.debug(f'gold: {ids_to_labels[gold]} pred: {gen_txt}')
                    count += 1
                    if count > 4:
                        break

            return {'generated': generated_txt, 'labels': labels}

        def validation_epoch_end(outputs):
            generated_txt = []
            labels = []
            preds = []

            for i in outputs:
                generated_txt.extend(i['generated'])
                labels.extend(i['labels'])
                for txt in i['generated']:
                    try:
                        pred_id = data_args.labels_to_ids[txt]
                    except:
                        pred_id = -100
                    preds.append(pred_id)

            class_ids = [key for key, value in ids_to_labels.items()]
            is_binary = False
            if -100 not in preds:
                if (0 in class_ids) and (1 in class_ids) and (len(class_ids) == 2):
                    is_binary = True
            else:
                class_ids.append(-100)

            acc = accuracy_score(labels, preds)
            if is_binary:
                f1 = f1_score(labels, preds)
            else:
                f1 = f1_score(y_true=labels, y_pred=preds, labels=class_ids, average="macro")

            metrics = {}
            metrics['accuracy'] = acc
            metrics['f1'] = f1
            metrics['error'] = accuracy_score([-100] * len(preds), preds)

            return metrics

        def evaluate(eval_dataloader):
            model.eval()
            eval_results = []
            for i, batch in tqdm(enumerate(eval_dataloader), "Generating predictions", total=len(eval_dataloader)):
                eval_results.append(validation_step(model, tokenizer, batch, i == 0))
            model.train()
            return validation_epoch_end(eval_results)

        class EvaluationCallback(TrainerCallback):
            def on_train_begin(self, args: TrainingArguments, state, control, logs=None, **kwargs):
                wandb.log({"accuracy" : 0.0, "f1": 0.0, "eval_epoch" : 0.0})
            
            def on_epoch_end(self, args, state, control, logs=None, **kwargs):
                logger.info(f"***Evaluation at epoch {state.epoch} begins***")
                metrics = evaluate(eval_dataloader)
                wandb.log({"accuracy" : metrics['accuracy'], "f1": metrics['f1'], "eval_epoch" : state.epoch})
                logger.info(f"***Evaluation results***")
                logger.info(f"accuracy: {metrics['accuracy']}")
                logger.info(f"f1: {metrics['f1']}")
                logger.info(f"error rate: {metrics['error']}")
                save_dir = training_args.output_dir + f"/checkpoint-{state.global_step}"
                os.makedirs(save_dir, exist_ok=True)
                model.save_pretrained(save_dir)

    # ### Training
    if training_args.do_train:
        model.train()
        model.gradient_checkpointing_enable()
        trainer = Trainer(
            model=model,
            train_dataset=data["train"],
            args=training_args,
            data_collator=collate_fn,
        )
        if training_args.do_eval:
            trainer.add_callback(EvaluationCallback)
        model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
        trainer.train()
        wandb.finish()
        model.save_pretrained(training_args.output_dir)
        model.config.use_cache = True

    # ## Share adapters on the 🤗 Hub
    # model.push_to_hub(training_args.output_dir, use_auth_token=True)

    # Load adapters from the Hub and generate some output texts:

    # peft_model_id = training_args.output_dir
    # config = PeftConfig.from_pretrained(peft_model_id)
    # model = AutoModelForCausalLM.from_pretrained(
    #     config.base_model_name_or_path, return_dict=True, load_in_8bit=True, device_map="auto"
    # )
    # tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    # 
    # # Load the Lora model
    # model = PeftModel.from_pretrained(model, peft_model_id)
    # # You can then directly use the trained model or the model that you have loaded from the 🤗 Hub for inference

    batch = tokenizer("다음 제목의 주제를 IT과학, 경제, 사회, 생활문화, 세계, 스포츠, 정치 중 하나로 분류하세요.\n기업들 현금 실탄 쌓자…코로나 위기에 자산 처분 잇따라\n주제:", return_tensors="pt")
    batch.to('cuda')
    with torch.cuda.amp.autocast():
        output_tokens = model.generate(input_ids = batch['input_ids'], max_new_tokens=data_args.max_label_len + 1)

    logger.info("\n\n", tokenizer.decode(output_tokens[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
