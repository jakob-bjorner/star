# Reasoning project for Deep Reinforcement Learning
Team with Atharva, Lorand, and Milad and Yixiong Hao, Ayush

Project Goal: to investigate the potential improvement in down stream general reasoning ability given modifications to Quiet-STaR esk training. We are tackling some problem settings which we expect to be very easy to initially improve over the naive performance, one such setting is multilingual MMLU, for which the performance degrades substantially, when asking MMLU questions in a different langauge.

Roadmap:
1. replicate STaR setting of QA for MMLU to see what signal can be gotten 

## Initial goal of Replicating STaR for multilingual MMLU

Below is an initial result on the english MMLU setting, where iterative finetuning showed only marginal improved performance of MMLU on a held out validation split. We don't expect MMLU reasoning generation on english to significantly change the performance given prior works show insignificant performance changes when using chain of thought style prompting on this simple benchmark. It is, however interesting to observe as a point of reference for non english experiments.

![alt text](image.png)

Below the STaR method (without rationalization) trained on Yoruba MMLU. The perforamce in this case starts below random (25 % for these four answer MCQs) due to degenerate outputs from LLama 3.1 8B instruct on the low resource langauge of Yoruba. 

Here are two example degenerate output answers:
```
Ṣeese esi lati ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ṣe ...

Ipa italẹ fun awọn ohun elo ti awọn asidi ọra polyunsaturated adayeba ninu awọn epo ẹfọ ni.\n\nAwọn asidi ọra polyunsaturated ni awọn asidi ọra ti o ni iwe ifowopamọ carbon-carbon meji. Awọn iwe ifowopamọ carbon-carbon meji ni awọn iwe ifowopamọ meji ti o ni awọn atomu carbon 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, ...
```

![alt text](image-1.png)

# some scripts with hydra for convience:
<!-- python run.py trainer.single_batch=true trainer.gen_batch_size=32 -->

// for seeing if I can fit to a single batch of data.
python run.py trainer.single_batch=true trainer.gen_batch_size=8 trainer.train_batch_size=8 trainer.agent.model_handler.warmup=0

python run.py -m trainer.gen_batch_size=24 trainer.train_batch_size=16 logger=CustomWandBLogger trainer.dataset_name=mmlu_YO-NG run_type=train_translate

python run.py logger=CustomWandBLogger trainer.train_batch_size=32 trainer.gen_batch_s
ize=64 run_type=train_english trainer.dataset_name=mmlu


python run.py trainer.single_batch=true trainer.gen_batch_size=8 trainer.train_batch_size=8 trainer.agent.model_handler.warmup=0

python run.py trainer.single_batch=true trainer.gen_batch_size=8 trainer.train_batch_size=8 trainer.agent.model_handler.warmup=0 trainer.train_strategy=dpo trainer.samples_per_train_question=5

python run.py -m run_type=train_dpo_english trainer.gen_batch_size=64 trainer.train_batch_size=6 trainer.train_strategy=dpo trainer.samples_per_train_question=2 logger=CustomWandBLogger trainer.dataset_name=mmlu trainer.agent.model_handler.optimizer_partial.lr=0.0002 trainer.dpo_beta=0.1
# tried learning rate 0.00002, now trying 0.0002, maybe also try different beta. maybe also try reinforce tuning before to make the ref model and starting model supervised on the winning responses.

<!-- python run.py run_type=val trainer.gen_batch_size=48 "trainer.dataset_name=mmlu_YO-NG" "+current_global_step=2" "+model_checkpoint_to_eval=/nethome/jbjorner3/dev/hallucination-fun/star/star_runs/train_2024-10-01/03-06-34_0/checkpoint_2"

python run.py run_type=val trainer.gen_batch_size=32 "trainer.dataset_name=mmlu_YO-NG" "+current_global_step=150" "+model_checkpoint_to_eval=/nethome/jbjorner3/dev/hallucination-fun/star/star_runs/train_2024-10-01/03-06-34_0/checkpoint_2" -->