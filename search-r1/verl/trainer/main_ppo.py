# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import ipdb
from verl import DataProto
import torch
from verl.utils.reward_score import qa_em
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
import re
import numpy as np
from data_gen.api import call_llm_batch


def _select_rm_score_fn(data_source):
    if data_source in [
            'nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa',
            'musique', 'bamboogle'
    ]:
        return qa_em.compute_score_em
    elif "bioasq" in data_source:
        return qa_em.compute_score_em
    elif "papersearchrl" in data_source.lower():
        return qa_em.compute_score_em
    elif "papersearchr1" in data_source.lower():
        return qa_em.compute_score_em
    else:
        raise NotImplementedError


class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, format_score=0.) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.format_score = format_score

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'],
                                         dtype=torch.float32)

        # all_scores = []

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch[
                'attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][
                prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model'][
                'ground_truth']

            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)

            score = compute_score_fn(solution_str=sequences_str,
                                     ground_truth=ground_truth,
                                     format_score=self.format_score)

            reward_tensor[i, valid_response_length - 1] = score
            # all_scores.append(score)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        return reward_tensor


class LLMJudgeRewardManager():
    """LLM Judge-based reward manager using batch API calls for efficiency with binary scoring."""

    def __init__(self,
                 tokenizer,
                 num_examine,
                 judge_model="openai/gpt-4.1-nano",
                 max_concurrent=10,
                 format_score=0.,
                 temperature=0.0) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.judge_model = judge_model
        self.max_concurrent = max_concurrent
        self.format_score = format_score
        self.temperature = temperature

    def _create_judge_prompts(self, sequences_strs, ground_truths,
                              data_sources):
        """Create batch of prompts for the LLM judge with binary scoring."""
        prompts = []

        for sequences_str, ground_truth, data_source in zip(
                sequences_strs, ground_truths, data_sources):
            ground_truth = set(ground_truth['target'])

            # Question  = text that follows "Question:" up to the first  <|im_end|>
            q_match = re.search(r'Question:\s*(.*?)\s*<\|im_end\|>',
                                sequences_str,
                                flags=re.DOTALL | re.IGNORECASE)
            question = q_match.group(1).strip() if q_match else ""

            # Answers   = all occurrences between <answer> … </answer>
            answers = re.findall(r'<answer>\s*(.*?)\s*</answer>',
                                 sequences_str,
                                 flags=re.DOTALL | re.IGNORECASE)

            # Take the 3-rd answer if it exists, otherwise fall back to last
            pred_answer = answers[2].strip() if len(answers) >= 3 \
                          else (answers[-1].strip() if answers else "")

            # Build the judge prompt
            prompt = f"""Please evaluate the following answer to a question.

Question: {question}
Ground Truth Answer: {ground_truth}
Predicted Answer: {pred_answer}

Please provide:
1. A correctness score either 0 or 1.

The predicted answer should be a single entity. 
- The answer is correct **only if it matches all key aspects, specificity, and detail of the ground truth, even if differently phrased (e.g., synonym or rewording that does not lose or broaden meaning).**
- If the prediction omits, generalizes, or adds information not present in the ground truth, it should be considered **incorrect**.
- **Even if the answer is a valid synonym, ensure it covers the same detail and scope.**
- For example, if the ground truth is "heart muscle," correct answers include "cardiac muscle" or "muscle of the heart," but NOT just "muscle" or "organ."
- **Any answer that is more vague, less specific, or encompasses a broader/narrower category than the ground truth should be marked as incorrect.**

Respond with exactly "1" if the response is correct and adequate, or "0" if it is incorrect or inadequate."""

            prompts.append(prompt)

        return prompts

    def __call__(self, data: DataProto):
        """Batch-based LLM judge reward computation with binary scoring."""

        # If there is rm score, we directly return rm score
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'],
                                         dtype=torch.float32)

        # Collect all data for batch processing
        sequences_strs = []
        ground_truths = []
        data_sources = []
        valid_response_lengths = []
        already_print_data_sources = {}

        # First pass: decode all sequences and collect data
        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch[
                'attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][
                prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model'][
                'ground_truth']
            data_source = data_item.non_tensor_batch['data_source']

            sequences_strs.append(sequences_str)
            ground_truths.append(ground_truth)
            data_sources.append(data_source)
            valid_response_lengths.append(valid_response_length)

            # Handle printing logic
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(f"[SAMPLE] {sequences_str}")

        # Create judge prompts for batch processing
        judge_prompts = self._create_judge_prompts(sequences_strs,
                                                   ground_truths, data_sources)

        try:
            # Batch LLM call for all judgments
            print(
                f"[LLM JUDGE] Processing {len(judge_prompts)} samples with {self.judge_model}"
            )
            judge_responses, costs = call_llm_batch(
                prompts=judge_prompts,
                model_name=self.judge_model,
                max_tokens=5,  # Very short response for just "0" or "1"
                temperature=self.temperature,
                max_concurrent=self.max_concurrent,
                include_cost=True,
                use_cache=True)

            total_cost = sum(costs) if costs else 0.0
            print(f"[LLM JUDGE] Total cost: ${total_cost:.6f}")

            # Parse binary scores and populate reward tensor
            correct_count = 0
            for i, (judge_response, valid_response_length) in enumerate(
                    zip(judge_responses, valid_response_lengths)):
                try:
                    # Extract binary score from judge response
                    response_clean = judge_response.strip()

                    # Try to parse as integer first
                    if response_clean in ['1', '1.0']:
                        score = 1.0
                        correct_count += 1
                    elif response_clean in ['0', '0.0']:
                        score = 0.0
                    else:
                        # Try to extract first number if response is longer
                        import re
                        numbers = re.findall(r'\b[01](?:\.0)?\b',
                                             response_clean)
                        if numbers:
                            score = 1.0 if numbers[0] in ['1', '1.0'] else 0.0
                            if score == 1.0:
                                correct_count += 1
                        else:
                            print(
                                f"[WARNING] Invalid judge score '{judge_response}' for sample {i}, using 0.0"
                            )
                            score = 0.0

                except (ValueError, TypeError):
                    print(
                        f"[WARNING] Invalid judge score '{judge_response}' for sample {i}, using 0.0"
                    )
                    score = 0.0

                reward_tensor[i, valid_response_length - 1] = score

            accuracy = correct_count / len(
                judge_responses) if judge_responses else 0.0
            print(
                f"[LLM JUDGE] Batch accuracy: {accuracy:.3f} ({correct_count}/{len(judge_responses)})"
            )

        except Exception as e:
            print(f"[ERROR] LLM judge batch call failed: {e}")
            print("[FALLBACK] Using default scores of 0.0")
            # Fallback to zero scores if LLM call fails
            for i, valid_response_length in enumerate(valid_response_lengths):
                reward_tensor[i, valid_response_length - 1] = 0.0

        return reward_tensor


import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    import os

    # Use environment variable to control debugging mode
    debug_mode = os.getenv('DEBUG_MODE', 'false').lower() == 'true'

    # Set required environment variables
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = '1'
        os.environ['RANK'] = '0'
        os.environ['LOCAL_RANK'] = '0'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        os.environ['WG_BACKEND'] = 'ray'

    if not ray.is_initialized():
        runtime_env = {
            'env_vars': {
                'TOKENIZERS_PARALLELISM': 'true',
                'NCCL_DEBUG': 'WARN',
                'WORLD_SIZE': os.environ['WORLD_SIZE'],
                'RANK': os.environ['RANK'],
                'LOCAL_RANK': os.environ['LOCAL_RANK'],
                'MASTER_ADDR': os.environ['MASTER_ADDR'],
                'MASTER_PORT': os.environ['MASTER_PORT'],
                'WG_BACKEND': 'ray',
                'RAY_DEBUG': 'legacy'
            }
        }

        ray.init(local_mode=False, runtime_env=runtime_env)

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    # DEBUG: Print what config we actually received
    print(
        f"[DEBUG] Inside main_task: n_gpus_per_node = {config.trainer.n_gpus_per_node}"
    )
    print(f"[DEBUG] Inside main_task: nnodes = {config.trainer.nnodes}")
    print(
        f"[DEBUG] Inside main_task: total GPUs = {config.trainer.n_gpus_per_node * config.trainer.nnodes}"
    )

    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(
        config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # env_class = ENV_CLASS_MAPPING[config.env.name]

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id:
        [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    use_llm_judge = getattr(config.reward_model,
                            'use_llm_judge', False) if hasattr(
                                config, 'reward_model') else False
    llm_judge_model = getattr(config.reward_model, 'llm_judge_model',
                              'openai/gpt-4.1-nano') if hasattr(
                                  config,
                                  'reward_model') else 'openai/gpt-4.1-nano'
    llm_judge_max_concurrent = getattr(
        config.reward_model, 'llm_judge_max_concurrent', 50) if hasattr(
            config, 'reward_model') else 50

    if use_llm_judge:
        print(f"[REWARD] Using LLM Judge with model: {llm_judge_model}")
        reward_fn = LLMJudgeRewardManager(
            tokenizer=tokenizer,
            num_examine=0,
            judge_model=llm_judge_model,
            max_concurrent=llm_judge_max_concurrent,
            temperature=0.0)
        val_reward_fn = LLMJudgeRewardManager(
            tokenizer=tokenizer,
            num_examine=1,
            judge_model=llm_judge_model,
            max_concurrent=llm_judge_max_concurrent,
            temperature=0.0)
    else:
        print("[REWARD] Using traditional rule-based reward manager")
        reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0)
        val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1)

    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec=resource_pool_spec, mapping=mapping)
    trainer = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
    )
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
