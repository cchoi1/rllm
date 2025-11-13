import asyncio
import concurrent.futures
import logging
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor

import torch

from rllm.agents.agent import Action, BaseAgent, Trajectory
from rllm.agents.utils import (
    convert_messages_to_tokens_and_masks,
    get_recent_assistant_user_messages,
)
from rllm.environments.base.base_env import BaseEnv
from rllm.environments.env_utils import (
    compute_mc_return,
    compute_trajectory_reward,
)
from rllm.misc import colorful_print
from rllm.parser import ChatTemplateParser

logger = logging.getLogger(__name__)


class AgentExecutionEngine:
    def __init__(
        self,
        engine_name="openai",
        tokenizer=None,
        rollout_engine=None,
        chat_parser=None,
        n_parallel_agents=1,
        trajectory_timeout=None,
        gamma=0.2,
        api_retries=3,
        retry_limit=3,
        max_steps=5,
        max_response_length=8192,
        max_prompt_length=1024,
        config=None,
        agent_class=None,
        env_class=None,
        agent_args=None,
        rollout_engine_args=None,
        env_args=None,
        max_workers=64,
        enforce_max_prompt_length=False,  # If enabled, applies max_prompt check per step
        overlong_filter=False,  # Filter for overlong trajectories (i.e. TRUNCATION, MAX_STEPS, TIMEOUT)
        **kwargs,
    ):
        if agent_args is None:
            agent_args = {}
        if rollout_engine_args is None:
            rollout_engine_args = {}
        if env_args is None:
            env_args = {}

        self.config = config
        self.tokenizer = tokenizer
        self.engine_name = engine_name
        self.n_parallel_agents = n_parallel_agents
        self.overlong_filter = overlong_filter

        # For interaction
        self.gamma = gamma
        self.retry_limit = retry_limit
        self.max_steps = max_steps
        self.max_response_length = max_response_length
        self.max_prompt_length = max_prompt_length
        self.enforce_max_prompt_length = enforce_max_prompt_length
        self.disable_thinking = self.config.get("rllm", {}).get("disable_thinking", False) if self.config is not None else False

        self.agent_class = agent_class
        self.agent_args = agent_args
        self.env_class = env_class
        self.env_args = env_args

        self.agents = [None for _ in range(n_parallel_agents)]
        self.envs = [None for _ in range(n_parallel_agents)]

        self.trajectory_timeout = trajectory_timeout
        if not trajectory_timeout:
            self.trajectory_timeout = int(1e9)

        if env_class is not None:
            assert env_class.is_multithread_safe(), "Environment must be multithread safe for async engine"

        if chat_parser is None:
            self.chat_parser = ChatTemplateParser.get_parser(self.tokenizer, disable_thinking=self.disable_thinking)
        else:
            self.chat_parser = chat_parser

        self.rollout_engine_args = rollout_engine_args
        self.sampling_params = kwargs.get("sampling_params", {})  # for openai api requests

        assert self.engine_name in ["openai", "verl"], "Currently only openai and verl are supported as rollout engine"
        if self.engine_name == "openai":
            from rllm.engine.rollout.openai_engine import OpenAIEngine

            self.rollout_engine = OpenAIEngine(
                **rollout_engine_args,
                api_retries=api_retries,
                tokenizer=self.tokenizer,
                max_prompt_length=self.max_prompt_length,
                max_response_length=self.max_response_length,
                disable_thinking=self.disable_thinking,
            )
        elif self.engine_name == "verl":
            from rllm.engine.rollout.verl_engine import VerlEngine

            self.rollout_engine = VerlEngine(
                config=self.config,
                rollout_manager=rollout_engine,
                tokenizer=self.tokenizer,
                disable_thinking=self.disable_thinking,
            )

        # Create a thread pool executor for environment interactions (i.e. step, reset, close)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    async def get_model_response(self, prompt, application_id, **kwargs) -> str:
        """
        Compute model response asynchronously based on the engine type.

        This function is multithread safe and routes the request to the appropriate
        engine-specific handler.

        Args:
            prompt: The input prompt to send to the model
            application_id: Unique identifier for the application
            **kwargs: Additional arguments to pass to the model

        Returns:
            The model's response text

        Raises:
            NotImplementedError: If the engine type is not supported
        """

        sampling_params = self.sampling_params.copy()
        sampling_params.update(kwargs)

        if self.engine_name == "openai":
            output = await self.rollout_engine.get_model_response(prompt, application_id=application_id, enforce_max_prompt_length=False, **sampling_params)
            return output.text
        elif self.engine_name == "verl":
            meta_data = sampling_params.pop("meta_info", {})
            validate = meta_data.get("validate", False)
            output = await self.rollout_engine.get_model_response(prompt, application_id=application_id, validate=validate, enforce_max_prompt_length=False, **sampling_params)
            return output.text
        else:
            raise NotImplementedError(f"Engine type '{self.engine_name}' not supported")

    def update_envs_and_agents(self, envs, agents):
        """
        Update the environments and agents.

        Args:
            envs: List of environments to use
            agents: List of agents to use
        """
        assert len(agents) == len(envs), f"Number of agents must equal to number of environments but received, {len(agents)} and {len(envs)}"
        self.envs = envs
        # For keeping track of the environment index in the batch.
        for idx, env in enumerate(envs):
            env.idx = idx
        self.agents = agents
        self.n_parallel_agents = len(envs)

    async def run_agent_trajectory_async(self, idx, application_id, seed=0, mode="Text", **kwargs):
        """Run a single agent's trajectory asynchronously"""
        agent = self.agents[idx]
        env = self.envs[idx]
        # env_id = env.env_id

        termination_reason = None
        prompt_token_len = 0
        prompt_tokens = []
        response_token_len = 0
        response_tokens = []
        response_masks = []
        total_time = 0.0
        reward_time = None
        llm_time = 0.0
        env_time = 0.0
        reward = 0.0

        # for step return
        episode_steps = []

        # Reset environment with the task using the executor
        loop = asyncio.get_event_loop()
        observation, info = await loop.run_in_executor(self.executor, env.reset)
        info["max_steps"] = self.max_steps

        # Reset agent
        agent.reset()
        # Update agent internal state from environment.
        agent.update_from_env(
            observation=observation,  # Raw observation from environment
            reward=0.0,
            done=False,
            info=info,
        )
        messages = agent.chat_completions
        prompt_tokens, _ = convert_messages_to_tokens_and_masks(messages, tokenizer=self.tokenizer, parser=self.chat_parser, contains_first_msg=True, contains_generation_msg=True)
        prompt_token_len = len(prompt_tokens)
        # Note, this should never happen!
        if prompt_token_len > self.max_prompt_length:
            agent.reset()
            raise Exception(f"Trajectory {idx}: initial prompt length {prompt_token_len} already exceeded max_prompt_length {self.max_prompt_length}, retrying")

        for step_idx in range(self.max_steps):
            # Get action from agent
            prompt_messages = agent.chat_completions.copy()
            # Max remaining tokens left for the response
            # For enforced max prompt at each step, no need to deduct here
            if not self.enforce_max_prompt_length:
                max_tokens = self.max_response_length - response_token_len
            else:
                max_tokens = self.max_response_length

                # since max prompt is enforced, we filter out too long prompts.
                prompt_str = self.chat_parser.parse(prompt_messages, add_generation_prompt=True, is_first_msg=True)
                prompt_len = len(self.tokenizer.encode(prompt_str, add_special_tokens=False))
                if prompt_len > self.max_prompt_length:
                    termination_reason = "PROMPT_TRUNCATION"
                    break

            kwargs["max_tokens"] = max_tokens

            start_time = time.time()
            response = await self.get_model_response(prompt_messages, application_id, **kwargs)
            delta_time = time.time() - start_time
            llm_time += delta_time
            total_time += delta_time
            # Update steps
            prompt_response_pair = {
                "prompt": self.chat_parser.parse(prompt_messages, add_generation_prompt=True, is_first_msg=True),
                "response": response,
            }
            episode_steps.append(prompt_response_pair)

            # Update agent with model response
            action: Action = agent.update_from_model(response)
            action = action.action

            # Take step in environment using the executor
            start_time = time.time()

            try:
                next_observation, reward, done, info = await asyncio.wait_for(loop.run_in_executor(self.executor, env.step, action), timeout=(self.trajectory_timeout - total_time))
            except asyncio.TimeoutError:
                termination_reason = "ENV_TIMEOUT"
                if step_idx == 0:
                    colorful_print(f"Warning: Trajectory {idx} completed due to: {termination_reason} before able to perform 1 complete action. This might cause unexpected behavior. Consider increasing trajectory timeout limit.\n", "red")
                reward = 0

                cur_step = agent.get_current_state()
                done = True
                cur_step.done = done
                break

            delta_time = time.time() - start_time
            env_time += delta_time
            total_time += delta_time
            info["max_steps"] = self.max_steps
            info["cur_tokens"] = response_token_len

            # Update agent internal state.
            agent.update_from_env(
                observation=next_observation,
                reward=reward,
                done=done,
                info=info,
            )

            cur_step = agent.get_current_state()
            cur_step.reward = reward
            cur_step.done = done
            cur_step.info.update(info)

            chat_completions_messages = agent.chat_completions
            assistant_message, env_messages = get_recent_assistant_user_messages(chat_completions_messages)

            # Check and convert to tokens if necessary
            assert assistant_message is not None or mode != "Token", "Assistant messages is none when accumulating token trajectories which should be conversations. This should not happen."
            assert env_messages is not None or mode != "Token", "Environment messages is none when accumulating token trajectories which should be conversations. This should not happen."
            assistant_msg_tokens, assistant_msg_masks = [], []
            env_msg_tokens, env_msg_masks = [], []
            if assistant_message:
                assistant_msg_tokens, assistant_msg_masks = convert_messages_to_tokens_and_masks([assistant_message], tokenizer=self.tokenizer, parser=self.chat_parser, contains_first_msg=False, contains_generation_msg=False)
            if env_messages:
                env_msg_tokens, env_msg_masks = convert_messages_to_tokens_and_masks(env_messages, tokenizer=self.tokenizer, parser=self.chat_parser, contains_first_msg=False, contains_generation_msg=True)

            # Update repsonse token length
            response_token_len += len(assistant_msg_tokens) + len(env_msg_tokens)
            # Reached maximum number of tokens for the trajectory
            if not self.enforce_max_prompt_length and response_token_len >= self.max_response_length:
                # Truncation length
                truncation_length = self.max_response_length - response_token_len
                # Truncate the response and masks
                if truncation_length < 0:
                    truncated_response_tokens = (assistant_msg_tokens + env_msg_tokens)[:truncation_length]
                    truncated_response_masks = (assistant_msg_masks + env_msg_masks)[:truncation_length]
                else:
                    # Edge case where the response is exactly the max response length.
                    truncated_response_tokens = assistant_msg_tokens + env_msg_tokens
                    truncated_response_masks = assistant_msg_masks + env_msg_masks
                # Update token collections
                response_tokens.extend(truncated_response_tokens)
                response_masks.extend(truncated_response_masks)

                cur_step = agent.get_current_state()
                if response_token_len - len(env_msg_tokens) > self.max_response_length:
                    cur_step.reward = 0.0
                cur_step.done = True
                termination_reason = "TRUNCATION"
                # handle returning
                break

            # Update the token version of trajectory
            response_tokens.extend(assistant_msg_tokens)
            response_masks.extend(assistant_msg_masks)
            observation = next_observation

            if total_time >= self.trajectory_timeout:
                termination_reason = "TIMEOUT"
                cur_step = agent.get_current_state()
                done = True
                cur_step.done = done
                break

            # Check if episode is done
            if done:
                termination_reason = "ENV_DONE"
                break

            response_tokens.extend(env_msg_tokens)
            response_masks.extend(env_msg_masks)

            if step_idx == self.max_steps - 1:
                termination_reason = "MAX_STEPS"

        masked_out = False
        if self.overlong_filter:
            if termination_reason == "TRUNCATION" or termination_reason == "MAX_STEPS" or termination_reason == "TIMEOUT":
                # Mask out the entire response for overlong trajectories if the reward is 0.
                response_masks = [0] * len(response_masks)
                masked_out = True

        if hasattr(env, "compute_final_reward") and not masked_out:
            cur_step = agent.get_current_state()
            start_time = time.time()
            reward = await loop.run_in_executor(self.executor, env.compute_final_reward)
            reward_time = time.time() - start_time
            cur_step.reward = reward
        # Closing environment using the executor.
        await loop.run_in_executor(self.executor, env.close)
        if termination_reason:
            if reward > 0:
                color = "green"
            else:
                color = "yellow"
            colorful_print(
                f"Trajectory {idx} completed due to: {termination_reason}. Reward is {reward}. \n",
                color,
            )
            if masked_out:
                colorful_print(f"Trajectory {idx} is masked out due to overlong filter.", "red")

        trajectory: Trajectory = agent.trajectory
        # Aggregate final trajectory statistics
        compute_trajectory_reward(trajectory)
        compute_mc_return(trajectory, gamma=self.gamma)

        if mode == "Text":
            return trajectory
        elif mode == "Token":
            token_result = {
                "prompt_tokens": torch.tensor(prompt_tokens, dtype=torch.long),
                "response_tokens": torch.tensor(response_tokens, dtype=torch.long),
                "response_masks": torch.tensor(response_masks, dtype=torch.long),
                "trajectory_reward": trajectory.reward,
                "idx": env.idx,
                "chat_completions": agent.chat_completions,
                "metrics": {
                    # Total number of steps taken in the trajectory
                    "steps": len(trajectory.steps),
                    # Time to calculate reward
                    "reward_time": reward_time,
                    # Total time spent in environment execution (env.step)
                    "env_time": env_time,
                    # Time to calculate response tokens
                    "llm_time": llm_time,
                    # Total time spent in the trajectory
                    "total_time": total_time,
                },
            }
            return token_result
        elif mode == "Conversation":
            return agent.chat_completions
        elif mode == "Step":
            steps_result = {
                "steps": episode_steps,
                "trajectory_reward": trajectory.reward,
                "idx": env.idx,
                "mc_returns": [step.mc_return for step in trajectory.steps][: len(episode_steps)],
            }
            return steps_result

    async def run_agent_trajectory_with_retry(self, idx, application_id, seed=0, mode="Text", **kwargs):
        for _ in range(self.retry_limit):
            try:
                return await asyncio.wait_for(self.run_agent_trajectory_async(idx, application_id=application_id, seed=seed, mode=mode, **kwargs), timeout=7200)
            except Exception:
                traceback.print_exc()
                continue
        traceback.print_exc()
        raise Exception(f"Trajectory {idx} cannot complete. Please check the log message")

    async def trajectory_generator(self, reset_seed=0, timing_raw=None, mode="Text", **kwargs):
        if timing_raw is None:
            timing_raw = {}
        assert all(env is not None and isinstance(env, BaseEnv) for env in self.envs), "All environments must be inheriting from BaseEnv"
        assert all(env.is_multithread_safe() for env in self.envs), "All environments must be multithread safe for async engine"  # type: ignore
        max_concurrency = self.n_parallel_agents
        self.executor = ThreadPoolExecutor(max_workers=max_concurrency)

        if self.engine_name == "verl":
            self.rollout_engine.wake_up()

        async def launch_one_trajectory_task(env_idx: int):
            try:
                application_id = str(uuid.uuid4())
                result = await self.run_agent_trajectory_with_retry(
                    idx=env_idx,
                    application_id=application_id,
                    seed=reset_seed,
                    mode=mode,
                    **kwargs,
                )
            except Exception as e:
                import traceback

                traceback.print_exc()
                raise e
            return result

        # Create all N conceptual tasks. Their execution will be throttled by the semaphore
        # and the availability of agent/env indices.
        tasks_to_run = [launch_one_trajectory_task(i) for i in range(len(self.envs))]

        tasks_completed = 0
        for coro in asyncio.as_completed(tasks_to_run):
            try:
                result = await coro
                tasks_completed += 1
                colorful_print(f"Number of Trajectories {tasks_completed}/{len(self.envs)} completed", "cyan")
                yield result
            except Exception as e:
                raise e

        if self.engine_name == "verl":
            self.rollout_engine.sleep()

        self.executor.shutdown(wait=False, cancel_futures=True)

    async def execute_tasks(self, tasks: list[dict]):
        """
        Run asynchronous interactions between the agent and environment where each agent
        has its own environment instance and can proceed independently.

        Args:
            tasks: List of tasks to process
            max_concurrent: Maximum number of concurrent tasks to process (defaults to self.n_parallel_agents)

        Returns:
            A list of trajectories, one for each task.
        """

        max_concurrent = self.n_parallel_agents

        # Initialize results list to store trajectories for all tasks
        all_trajectories = {}

        # Create a queue of tasks to process
        task_queue = list(enumerate(tasks))
        semaphore = asyncio.Semaphore(max_concurrent)
        index_queue: asyncio.Queue[int] = asyncio.Queue(maxsize=max_concurrent)
        for i in range(max_concurrent):
            index_queue.put_nowait(i)

        # Track completed trajectories
        completed = 0
        total = len(tasks)

        async def sem_wrapper(task_id, task):
            nonlocal completed
            async with semaphore:
                # Get an available index
                index = await index_queue.get()
                try:
                    self.envs[index] = self.env_class.from_dict({**task, **self.env_args})
                    self.agents[index] = self.agent_class(**self.agent_args)
                    assert self.agents[index] is not None and isinstance(self.agents[index], BaseAgent), "Agent is not initalized or not inheriting from BaseAgent"
                    self.agents[index].trajectory.task = task  # type: ignore
                    res = await self.run_agent_trajectory_async(index, application_id=task_id)
                    res.task = task
                    completed += 1
                    colorful_print(f"Progress: {completed}/{total} trajectories completed", "cyan")
                    return task_id, res
                finally:
                    # Put the index back in the queue when done
                    await index_queue.put(index)

        # Run all tasks concurrently
        results = await asyncio.gather(*[sem_wrapper(task_id, task) for task_id, task in task_queue])

        all_trajectories = {task_id: trajectory for task_id, trajectory in results}
        ordered_trajectories = [all_trajectories[i] for i in range(len(all_trajectories))]
        return ordered_trajectories

    def shutdown(self):
        if hasattr(self, "executor") and self.executor is not None:
            self.executor.shutdown()
            self.executor = None


class AsyncAgentExecutionEngine(AgentExecutionEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

# import asyncio
# import copy
# import concurrent.futures
# import logging
# import time
# import traceback
# import uuid
# from concurrent.futures import ThreadPoolExecutor

# import numpy as np
# import openai
# import torch
# from openai.types import Completion

# from rllm.agents.agent import Action, BaseAgent, Step, Trajectory
# from rllm.agents.utils import (
#     convert_messages_to_tokens_and_masks,
#     get_recent_assistant_user_messages,
# )
# from rllm.environments.base.base_env import BaseEnv
# from rllm.environments.env_utils import (
#     compute_mc_return,
#     compute_trajectory_reward,
# )
# from rllm.misc import colorful_print
# from rllm.parser.chat_template.parser import ChatTemplateParser
# from rllm.router.router import Router

# logger = logging.getLogger(__name__)


# class AgentExecutionEngine:
#     def __init__(
#         self,
#         engine_name="openai",
#         tokenizer=None,
#         rollout_engine=None,
#         chat_parser=None,
#         n_parallel_agents=1,
#         trajectory_timeout=None,
#         gamma=0.2,
#         api_retries=3,
#         retry_limit=3,
#         max_steps=5,
#         max_response_length=8192,
#         max_prompt_length=1024,
#         config=None,
#         agent_class=None,
#         env_class=None,
#         agent_args=None,
#         rollout_engine_args=None,
#         env_args=None,
#         max_workers=64,
#         enforce_max_prompt_length=False,  # If enabled, applies max_prompt check per step
#         overlong_filter=False,  # Filter for overlong trajectories (i.e. TRUNCATION, MAX_STEPS, TIMEOUT)
#         **kwargs,
#     ):
#         if agent_args is None:
#             agent_args = {}
#         if rollout_engine_args is None:
#             rollout_engine_args = {}
#         if env_args is None:
#             env_args = {}

#         self.config = config
#         self.rollout_engine = rollout_engine
#         self.tokenizer = tokenizer
#         self.engine_name = engine_name
#         self.n_parallel_agents = n_parallel_agents
#         self.overlong_filter = overlong_filter

#         # For interaction
#         self.gamma = gamma
#         self.retry_limit = retry_limit
#         self.api_retries = api_retries
#         self.max_steps = max_steps
#         self.max_response_length = max_response_length
#         self.max_prompt_length = max_prompt_length
#         self.enforce_max_prompt_length = enforce_max_prompt_length

#         self.agent_class = agent_class
#         self.agent_args = agent_args
#         self.env_class = env_class
#         self.env_args = env_args

#         self.agents = [None for _ in range(n_parallel_agents)]
#         self.envs = [None for _ in range(n_parallel_agents)]

#         self.trajectory_timeout = trajectory_timeout
#         if not trajectory_timeout:
#             self.trajectory_timeout = int(1e9)

#         if env_class is not None:
#             assert env_class.is_multithread_safe(), "Environment must be multithread safe for async engine"
#         # rollout engine args
#         self.rollout_engine_args = rollout_engine_args
#         self.sampling_params = kwargs.get("sampling_params", {})

#         assert self.engine_name in ["openai", "verl"], "Currently only openai and verl are supported as rollout engine"
#         if self.engine_name == "openai":
#             from openai import AsyncOpenAI

#             self.client = AsyncOpenAI(**self.rollout_engine_args)
#             # Disable httpx INFO logs that show HTTP requests
#             logging.getLogger("httpx").setLevel(logging.WARNING)
#         elif self.engine_name == "verl":
#             # All generation is done via scheduler. Currently only works for verl
#             self.server_addresses = getattr(self.rollout_engine, "server_addresses", [])
#             self.router = Router(config=self.config, tokenizer=self.tokenizer, addresses=self.server_addresses)

#         # Create a thread pool executor for environment interactions (i.e. step, reset, close)
#         self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

#         if chat_parser is None:
#             self.chat_parser = ChatTemplateParser.get_parser(self.tokenizer, disable_thinking=kwargs.get("disable_thinking", False))
#         else:
#             self.chat_parser = chat_parser

#     async def get_model_response(self, prompt, application_id, **kwargs):
#         """
#         Compute model response asynchronously based on the engine type.

#         This function is multithread safe and routes the request to the appropriate
#         engine-specific handler.

#         Args:
#             prompt: The input prompt to send to the model
#             application_id: Unique identifier for the application
#             **kwargs: Additional arguments to pass to the model

#         Returns:
#             The model's response text

#         Raises:
#             NotImplementedError: If the engine type is not supported
#         """
#         # print("PROMPT: ", prompt)
#         kwargs.setdefault("logprobs", True)
#         kwargs.setdefault("top_logprobs", 1)
        
#         if self.engine_name == "openai":
#             lp = kwargs.get("logprobs", None)
#             if isinstance(lp, bool):
#                 kwargs["logprobs"] = 1 if lp else None  # int or None
#             # not valid for Completions requests; drop it if present
#             kwargs.pop("top_logprobs", None)
#             # don't send Nones down to the SDK
#             kwargs = {k: v for k, v in kwargs.items() if v is not None}
#             return await self._get_openai_async(prompt, application_id, **kwargs)
#         elif self.engine_name == "verl":
#             return await self._get_verl_async(prompt, application_id, **kwargs)
#         else:
#             raise NotImplementedError(f"Engine type '{self.engine_name}' not supported")

#     def update_envs_and_agents(self, envs, agents):
#         """
#         Update the environments and agents.

#         Args:
#             envs: List of environments to use
#             agents: List of agents to use
#         """
#         assert len(agents) == len(envs), f"Number of agents must equal to number of environments but received, {len(agents)} and {len(envs)}"
#         self.envs = envs
#         # For keeping track of the environment index in the batch.
#         for idx, env in enumerate(envs):
#             env.idx = idx
#         self.agents = agents
#         self.n_parallel_agents = len(envs)

#     async def _get_verl_async(self, prompt, application_id, **kwargs):
#         batch = self._convert_prompt_verl([prompt], **kwargs)

#         if "max_tokens" in kwargs:
#             batch.meta_info["max_tokens"] = kwargs["max_tokens"]

#         # ask the router to return logprobs for the rollout policy if it can
#         batch.meta_info.setdefault("recompute_log_prob", True)
#         batch.meta_info.setdefault("return_logprob", True)

#         output = await self.router.generate_sequences(batch, application_id=application_id, **kwargs)

#         # Attention & tokens over the response window (after max_prompt_length)
#         attn = output.batch["attention_mask"][0, self.max_prompt_length:]
#         tokens = output.batch["responses"][0]

#         # Find last index where attention == 1 (end of response)
#         non_pad_indices = (attn == 1).nonzero(as_tuple=True)[0]
#         if len(non_pad_indices) == 0:
#             trimmed = tokens[:0]  # empty
#             resp_len = 0
#         else:
#             last_valid_idx = non_pad_indices[-1].item()
#             trimmed = tokens[: last_valid_idx + 1]  # include the last valid token
#             resp_len = last_valid_idx + 1

#         # Decode response (preserve internal specials; strip pad/eos from text like your original)
#         response = self.tokenizer.decode(trimmed, skip_special_tokens=False)
#         pad_token = getattr(self.tokenizer, "pad_token", "")
#         eos_token = getattr(self.tokenizer, "eos_token", "")
#         if pad_token:
#             response = response.replace(pad_token, "")
#         if eos_token:
#             response = response.replace(eos_token, "")

#         # --- rollout logprobs (θ_vllm) aligned to response tokens ---
#         response_logprobs = []
#         lp_tensor = None

#         # Prefer native rollout key if present; fall back to legacy if router names it differently
#         if "rollout_log_probs" in output.batch:
#             lp_tensor = output.batch["rollout_log_probs"][0]  # shape: [prompt+resp] or [resp]
#         elif "old_log_probs" in output.batch:
#             # Some backends still use this key; treat as rollout logprobs at collection time
#             lp_tensor = output.batch["old_log_probs"][0]

#         if lp_tensor is not None:
#             # If the tensor is full-length (prompt + response), slice the response region.
#             if lp_tensor.shape[0] == output.batch["attention_mask"].shape[1]:
#                 lp_resp = lp_tensor[self.max_prompt_length : self.max_prompt_length + resp_len]
#             else:
#                 # Already response-only; just trim/pad to resp_len
#                 lp_resp = lp_tensor[:resp_len]
#             response_logprobs = lp_resp.detach().cpu().tolist()

#         return {"text": response, "response_logprobs": response_logprobs}


#     async def _get_openai_async(self, prompt, _, **kwargs):
#         """
#         Get action from OpenAI API asynchronously with retry logic.

#         Args:
#             prompt: The input prompt in text format for completions API
#             application_id: Unique identifier for the application (unused for OpenAI)
#             **kwargs: Additional arguments to pass to the OpenAI API

#         Returns:
#             The response from OpenAI API
#         """

#         async def get_response(prompt_text: str):
#             retries = self.api_retries
#             while retries > 0:
#                 try:
#                     response = await self.client.completions.create(
#                         prompt=prompt_text,
#                         timeout=3600,
#                         **self.sampling_params,
#                         **kwargs,
#                     )
#                     return response
#                 except openai.RateLimitError:
#                     retries -= 1
#                     if retries == 0:
#                         return "Error: Rate limit reached and retries exhausted."
#                     logger.info("Sleep for 5 seconds for API limit.")
#                     await asyncio.sleep(5)
#                 except Exception as e:
#                     logger.error("Error: %s", e)
#                     return f"Error processing content: {e}"

#         # If prompt is in chat format, convert it to text format
#         prompt_text = prompt
#         if isinstance(prompt, list) and all(isinstance(msg, dict) for msg in prompt):
#             prompt_text = self.chat_parser.parse(prompt, add_generation_prompt=True, is_first_msg=True)

#         response = await get_response(prompt_text)
#         # if isinstance(response, Completion):
#         #     response = response.choices[0].text
#         # If SDK typed object:
#         if isinstance(response, Completion):
#             ch = response.choices[0]
#             text = ch.text
#             lp = getattr(ch, "logprobs", None)

#             response_logprobs = []
#             if lp and getattr(lp, "token_logprobs", None):
#                 # For Completions, token_logprobs align to the completion tokens directly (echo=False)
#                 response_logprobs = list(lp.token_logprobs)

#             if kwargs.get("return_logprobs", True):
#                 return {"text": text, "response_logprobs": response_logprobs}
#             return text
#         return response

#     async def run_agent_trajectory_async(self, idx, application_id, seed=0, mode="Text", **kwargs):
#         """Run a single agent's trajectory asynchronously"""
#         agent = self.agents[idx]
#         env = self.envs[idx]

#         termination_reason = None
#         prompt_token_len = 0
#         prompt_tokens = []
#         response_token_len = 0
#         response_tokens = []
#         response_masks = []
#         response_logprobs_all = []  # <-- collect per-token logprobs aligned with response_tokens
#         total_time = 0.0
#         reward_time = None
#         llm_time = 0.0
#         env_time = 0.0
#         reward = 0.0

#         # for step return
#         episode_steps = []

#         # Reset environment with the task using the executor
#         loop = asyncio.get_event_loop()
#         observation, info = await loop.run_in_executor(self.executor, env.reset)
#         info["max_steps"] = self.max_steps

#         # Reset agent
#         agent.reset()
#         # Update agent internal state from environment.
#         agent.update_from_env(
#             observation=observation,  # Raw observation from environment
#             reward=0.0,
#             done=False,
#             info=info,
#         )
    
#         messages = agent.chat_completions
#         prompt_tokens, _ = convert_messages_to_tokens_and_masks(
#             messages,
#             tokenizer=self.tokenizer,
#             parser=self.chat_parser,
#             contains_first_msg=True,
#             contains_generation_msg=True,
#         )
#         prompt_token_len = len(prompt_tokens)
#         if prompt_token_len > self.max_prompt_length:
#             agent.reset()
#             raise Exception(
#                 f"Trajectory {idx}: initial prompt length {prompt_token_len} already exceeded "
#                 f"max_prompt_length {self.max_prompt_length}, retrying"
#             )

#         for step_idx in range(self.max_steps):
#             # Build the prompt/messages for this step
#             prompt_messages = agent.chat_completions.copy()

#             # Decide how many tokens we can still emit
#             if not self.enforce_max_prompt_length:
#                 max_tokens = self.max_response_length - response_token_len
#             else:
#                 max_tokens = self.max_response_length
#                 # Enforce per-step prompt length if requested
#                 prompt_str = self.chat_parser.parse(prompt_messages, add_generation_prompt=True, is_first_msg=True)
#                 prompt_len = len(self.tokenizer.encode(prompt_str, add_special_tokens=False))
#                 if prompt_len > self.max_prompt_length:
#                     termination_reason = "PROMPT_TRUNCATION"
#                     break

#             kwargs["max_tokens"] = max_tokens

#             # === LLM call (unwrap {text, response_logprobs} if returned) ===
#             start_time = time.time()
#             model_out = await self.get_model_response(prompt_messages, application_id, **kwargs)
#             delta_time = time.time() - start_time
#             llm_time += delta_time
#             total_time += delta_time

#             if isinstance(model_out, dict):
#                 resp_text = model_out.get("text", "")
#                 resp_lps = model_out.get("response_logprobs", None)  # list[float] aligned to the assistant completion tokens
#             else:
#                 resp_text = model_out
#                 resp_lps = None

#             # Record step prompt/response text (for Step mode / debugging)
#             prompt_response_pair = {
#                 "prompt": self.chat_parser.parse(prompt_messages, add_generation_prompt=True, is_first_msg=True),
#                 "response": resp_text,
#             }
#             # carry per-step rollout logprobs (aligned/padded later in _transform_agent_steps)
#             if resp_lps is not None:
#                 # ensure JSON-serializable
#                 prompt_response_pair["response_logprobs"] = [float(x) for x in resp_lps]
#             episode_steps.append(prompt_response_pair)

#             # === Update agent with the model response ===
#             action: Action = agent.update_from_model(resp_text)
#             action = action.action

#             # Optional: extract think
#             thought = ""
#             if "</think>" in resp_text:
#                 try:
#                     think_text = resp_text.split("</think>")[0]
#                     if "<think>" in think_text:
#                         thought = think_text.split("<think>")[1].strip()
#                     else:
#                         thought = think_text.strip()
#                 except:
#                     thought = ""

#             # === Step the environment ===
#             start_time = time.time()
#             try:
#                 next_observation, reward, done, info = await asyncio.wait_for(
#                     loop.run_in_executor(self.executor, env.step, action),
#                     timeout=(self.trajectory_timeout - total_time),
#                 )
#             except asyncio.TimeoutError:
#                 termination_reason = "ENV_TIMEOUT"
#                 if step_idx == 0:
#                     colorful_print(
#                         f"Warning: Trajectory {idx} completed due to: {termination_reason} before able to perform "
#                         f"1 complete action. This might cause unexpected behavior. Consider increasing trajectory timeout limit.\n",
#                         "red",
#                     )
#                 reward = 0
#                 cur_step = agent.get_current_state()
#                 done = True
#                 cur_step.done = done
#                 break

#             delta_time = time.time() - start_time
#             env_time += delta_time
#             total_time += delta_time
#             info["max_steps"] = self.max_steps
#             info["cur_tokens"] = response_token_len

#             # Update agent internal state from env
#             agent.update_from_env(
#                 observation=next_observation,
#                 reward=reward,
#                 done=done,
#                 info=info,
#             )
#             cur_step = agent.get_current_state()
#             cur_step.reward = reward
#             cur_step.done = done
#             cur_step.info.update(info)

#             # Attach thought if available
#             if hasattr(cur_step, 'thought'):
#                 cur_step.thought = thought
#             else:
#                 setattr(cur_step, 'thought', thought)

#             # === Optional extras copied from observation ===
#             if hasattr(agent, '_current_obs') and isinstance(agent._current_obs, dict):
#                 # find the last agent chat completion with "role" == "user"
#                 last_user_msg = None
#                 for msg in reversed(agent.chat_completions):
#                     if msg["role"] == "user":
#                         last_user_msg = msg
#                         break
#                 if not hasattr(cur_step, 'extras') or cur_step.extras is None:
#                     cur_step.extras = {}

#                 obs_solver_prompt = agent._current_obs.get("solver_prompt", "")
#                 obs_solver_full_output = agent._current_obs.get("solver_full_output", "")
#                 obs_solver_code = agent._current_obs.get("solver_output", "")
#                 obs_verifier_results = agent._current_obs.get("verifier_results", "")
#                 obs_passed_tests = agent._current_obs.get("passed_tests", 0)
#                 obs_total_tests = agent._current_obs.get("total_tests", 0)
#                 obs_solved = agent._current_obs.get("solved", False)

#                 cur_step.extras.update({
#                     "solver_prompt": obs_solver_prompt,
#                     "solver_full_output": obs_solver_full_output,
#                     "solver_code": obs_solver_code,
#                     "verifier_results": obs_verifier_results,
#                     "passed_tests": obs_passed_tests,
#                     "total_tests": obs_total_tests,
#                     "solved": obs_solved,
#                     "context_manager_prompt": self.chat_parser.parse([last_user_msg], add_generation_prompt=True, is_first_msg=True) if last_user_msg else "",
#                 })

#             # === Tokenize the latest assistant + env/user messages ===
#             chat_completions_messages = agent.chat_completions
#             assistant_message, env_messages = get_recent_assistant_user_messages(chat_completions_messages)

#             assert assistant_message is not None or mode != "Token", \
#                 "Assistant messages is none when accumulating token trajectories which should be conversations."
#             assert env_messages is not None or mode != "Token", \
#                 "Environment messages is none when accumulating token trajectories which should be conversations."

#             assistant_msg_tokens, assistant_msg_masks = [], []
#             env_msg_tokens, env_msg_masks = [], []
#             if assistant_message:
#                 assistant_msg_tokens, assistant_msg_masks = convert_messages_to_tokens_and_masks(
#                     [assistant_message],
#                     tokenizer=self.tokenizer,
#                     parser=self.chat_parser,
#                     contains_first_msg=False,
#                     contains_generation_msg=False,
#                 )
#             if env_messages:
#                 env_msg_tokens, env_msg_masks = convert_messages_to_tokens_and_masks(
#                     env_messages,
#                     tokenizer=self.tokenizer,
#                     parser=self.chat_parser,
#                     contains_first_msg=False,
#                     contains_generation_msg=True,
#                 )

#             # === Update total response token count (assistant + env) ===
#             step_assist_len = len(assistant_msg_tokens)
#             step_env_len = len(env_msg_tokens)
#             response_token_len += step_assist_len + step_env_len

#             # Build aligned logprobs for assistant tokens; zeros for env tokens
#             if resp_lps is None:
#                 assist_lps = [0.0] * step_assist_len
#             else:
#                 # pad/trim to match assistant tokens
#                 if len(resp_lps) > step_assist_len:
#                     assist_lps = resp_lps[:step_assist_len]
#                 elif len(resp_lps) < step_assist_len:
#                     assist_lps = resp_lps + [0.0] * (step_assist_len - len(resp_lps))
#                 else:
#                     assist_lps = resp_lps
#             combined_step_lps = assist_lps + [0.0] * step_env_len

#             # === If we've exceeded max_response_length, truncate tokens/masks/logprobs consistently ===
#             if not self.enforce_max_prompt_length and response_token_len >= self.max_response_length:
#                 truncation_length = self.max_response_length - response_token_len
#                 # Build combined step lists (assistant + env) and truncate identically
#                 combined_tokens = (assistant_msg_tokens + env_msg_tokens)
#                 combined_masks  = (assistant_msg_masks + env_msg_masks)

#                 if truncation_length < 0:
#                     truncated_tokens = combined_tokens[:truncation_length]
#                     truncated_masks  = combined_masks[:truncation_length]
#                     truncated_lps    = combined_step_lps[:truncation_length]
#                 else:
#                     truncated_tokens = combined_tokens
#                     truncated_masks  = combined_masks
#                     truncated_lps    = combined_step_lps

#                 response_tokens.extend(truncated_tokens)
#                 response_masks.extend(truncated_masks)
#                 response_logprobs_all.extend(truncated_lps)

#                 cur_step = agent.get_current_state()
#                 if response_token_len - step_env_len > self.max_response_length:
#                     cur_step.reward = 0.0
#                 cur_step.done = True
#                 termination_reason = "TRUNCATION"
#                 break

#             # === Normal (non-truncated) append ===
#             response_tokens.extend(assistant_msg_tokens)
#             response_masks.extend(assistant_msg_masks)
#             response_logprobs_all.extend(assist_lps)

#             observation = next_observation

#             if total_time >= self.trajectory_timeout:
#                 termination_reason = "TIMEOUT"
#                 cur_step = agent.get_current_state()
#                 done = True
#                 cur_step.done = done
#                 break

#             if done:
#                 termination_reason = "ENV_DONE"
#                 # Still append env messages for visibility/faithful transcript (with zero logprobs)
#                 response_tokens.extend(env_msg_tokens)
#                 response_masks.extend(env_msg_masks)
#                 response_logprobs_all.extend([0.0] * step_env_len)
#                 break

#             # Not done: append env messages as part of the response stream
#             response_tokens.extend(env_msg_tokens)
#             response_masks.extend(env_msg_masks)
#             response_logprobs_all.extend([0.0] * step_env_len)

#             if step_idx == self.max_steps - 1:
#                 termination_reason = "MAX_STEPS"

#         # Optionally mask overlong trajectories
#         masked_out = False
#         if self.overlong_filter:
#             if termination_reason in {"TRUNCATION", "MAX_STEPS", "TIMEOUT"}:
#                 response_masks = [0] * len(response_masks)
#                 masked_out = True

#         # Final reward if available
#         if hasattr(env, "compute_final_reward") and not masked_out:
#             cur_step = agent.get_current_state()
#             start_time = time.time()
#             reward = await loop.run_in_executor(self.executor, env.compute_final_reward)
#             reward_time = time.time() - start_time
#             cur_step.reward = reward

#         # Close env
#         await loop.run_in_executor(self.executor, env.close)

#         if termination_reason:
#             color = "green" if reward > 0 else "yellow"
#             colorful_print(
#                 f"Trajectory {idx} completed due to: {termination_reason}. Reward is {reward}. \n",
#                 color,
#             )
#             if masked_out:
#                 colorful_print(f"Trajectory {idx} is masked out due to overlong filter.", "red")

#         trajectory: Trajectory = agent.trajectory
#         compute_trajectory_reward(trajectory)
#         compute_mc_return(trajectory, gamma=self.gamma)

#         if mode == "Text":
#             return trajectory
#         elif mode == "Token":
#             # Sanity: lengths must match
#             assert len(response_tokens) == len(response_masks) == len(response_logprobs_all), \
#                 f"Token/mask/logprob length mismatch: {len(response_tokens)} vs {len(response_masks)} vs {len(response_logprobs_all)}"

#             token_result = {
#                 "prompt_tokens": torch.tensor(prompt_tokens, dtype=torch.long),
#                 "response_tokens": torch.tensor(response_tokens, dtype=torch.long),
#                 "response_masks": torch.tensor(response_masks, dtype=torch.long),
#                 "response_logprobs": torch.tensor(response_logprobs_all, dtype=torch.float32),  # <-- NEW
#                 "trajectory_reward": trajectory.reward,
#                 "idx": env.idx,
#                 "chat_completions": agent.chat_completions,
#                 "metrics": {
#                     "steps": len(trajectory.steps),
#                     "reward_time": reward_time,
#                     "env_time": env_time,
#                     "llm_time": llm_time,
#                     "total_time": total_time,
#                 },
#             }
#             return token_result
#         elif mode == "Conversation":
#             return agent.chat_completions
#         elif mode == "Step":
#             steps_result = {
#                 "steps": episode_steps,
#                 "trajectory_reward": trajectory.reward,
#                 "idx": env.idx,
#                 "mc_returns": [step.mc_return for step in trajectory.steps][: len(episode_steps)],
#             }
#             return steps_result

#     async def run_agent_trajectory_with_retry(self, idx, application_id, seed=0, mode="Text", **kwargs):
#         for _ in range(self.retry_limit):
#             try:
#                 return await asyncio.wait_for(self.run_agent_trajectory_async(idx, application_id=application_id, seed=seed, mode=mode, **kwargs), timeout=7200)
#             except Exception:
#                 traceback.print_exc()
#                 continue
#         traceback.print_exc()
#         raise Exception(f"Trajectory {idx} cannot complete. Please check the log message")

#     async def trajectory_generator(self, reset_seed=0, timing_raw=None, mode="Text", **kwargs):
#         if timing_raw is None:
#             timing_raw = {}
#         assert all(env is not None and isinstance(env, BaseEnv) for env in self.envs), "All environments must be inheriting from BaseEnv"
#         assert all(env.is_multithread_safe() for env in self.envs), "All environments must be multithread safe for async engine"  # type: ignore
#         max_concurrency = self.n_parallel_agents
#         self.executor = ThreadPoolExecutor(max_workers=max_concurrency)

#         if self.engine_name == "verl":
#             # self.rollout_engine.wake_up()
#             wake = getattr(self.rollout_engine, "wake_up", None)
#             if callable(wake):
#                 wake()

#         async def launch_one_trajectory_task(env_idx: int):
#             try:
#                 application_id = str(uuid.uuid4())
#                 result = await self.run_agent_trajectory_with_retry(
#                     idx=env_idx,
#                     application_id=application_id,
#                     seed=reset_seed,
#                     mode=mode,
#                     **kwargs,
#                 )
#             except Exception as e:
#                 import traceback

#                 traceback.print_exc()
#                 raise e
#             return result

#         # Create all N conceptual tasks. Their execution will be throttled by the semaphore
#         # and the availability of agent/env indices.
#         tasks_to_run = [launch_one_trajectory_task(i) for i in range(len(self.envs))]

#         tasks_completed = 0
#         for coro in asyncio.as_completed(tasks_to_run):
#             try:
#                 result = await coro
#                 tasks_completed += 1
#                 colorful_print(f"Number of Trajectories {tasks_completed}/{len(self.envs)} completed", "cyan")
#                 yield result
#             except Exception as e:
#                 raise e

#         if self.engine_name == "verl":
#             self.rollout_engine.sleep()

#         self.executor.shutdown(wait=False, cancel_futures=True)

#     async def execute_tasks(self, tasks: list[dict]):
#         """
#         Run asynchronous interactions between the agent and environment where each agent
#         has its own environment instance and can proceed independently.

#         Args:
#             tasks: List of tasks to process
#             max_concurrent: Maximum number of concurrent tasks to process (defaults to self.n_parallel_agents)

#         Returns:
#             A list of trajectories, one for each task.
#         """

#         max_concurrent = self.n_parallel_agents

#         # Initialize results list to store trajectories for all tasks
#         all_trajectories = {}

#         # Create a queue of tasks to process
#         task_queue = list(enumerate(tasks))
#         semaphore = asyncio.Semaphore(max_concurrent)
#         index_queue: asyncio.Queue[int] = asyncio.Queue(maxsize=max_concurrent)
#         for i in range(max_concurrent):
#             index_queue.put_nowait(i)

#         # Track completed trajectories
#         completed = 0
#         total = len(tasks)

#         async def sem_wrapper(task_id, task):
#             nonlocal completed
#             async with semaphore:
#                 # Get an available index
#                 index = await index_queue.get()
#                 try:
#                     self.envs[index] = self.env_class.from_dict({**task, **self.env_args})
#                     self.agents[index] = self.agent_class(**self.agent_args)
#                     assert self.agents[index] is not None and isinstance(self.agents[index], BaseAgent), "Agent is not initalized or not inheriting from BaseAgent"
#                     self.agents[index].trajectory.task = task  # type: ignore
#                     res = await self.run_agent_trajectory_async(index, application_id=task_id)
#                     res.task = task
#                     completed += 1
#                     colorful_print(f"Progress: {completed}/{total} trajectories completed", "cyan")
#                     return task_id, res
#                 finally:
#                     # Put the index back in the queue when done
#                     await index_queue.put(index)

#         # Run all tasks concurrently
#         results = await asyncio.gather(*[sem_wrapper(task_id, task) for task_id, task in task_queue])

#         all_trajectories = {task_id: trajectory for task_id, trajectory in results}
#         ordered_trajectories = [all_trajectories[i] for i in range(len(all_trajectories))]
#         return ordered_trajectories

#     def _convert_prompt_verl(self, prompts, **kwargs):
#         """
#         Given a list of prompts in Chat template, convert to DataProto format in veRL

#         Args:
#             prompts: List of prompts to convert
#             **kwargs: Additional arguments

#         Returns:
#             DataProto object containing the converted prompts
#         """
#         from verl.protocol import DataProto, union_two_dict
#         from verl.utils.model import compute_position_id_with_mask
#         from verl.utils.torch_functional import pad_sequence_to_length

#         old_padding_side = self.tokenizer.padding_side
#         self.tokenizer.padding_side = "left"

#         formatted_prompts = [self.chat_parser.parse(prompt, add_generation_prompt=True, is_first_msg=True) for prompt in prompts]

#         # Tokenize the final processed strings
#         inputs = self.tokenizer(
#             formatted_prompts,
#             padding=True,
#             return_tensors="pt",
#             add_special_tokens=False,
#         )
#         self.tokenizer.padding_side = old_padding_side

#         input_ids = inputs["input_ids"]
#         attention_mask = inputs["attention_mask"]

#         # pad to max sizes
#         input_ids = pad_sequence_to_length(input_ids, max_seq_len=self.max_prompt_length, pad_token_id=self.tokenizer.pad_token_id, left_pad=True)
#         attention_mask = pad_sequence_to_length(attention_mask, max_seq_len=self.max_prompt_length, pad_token_id=0, left_pad=True)
#         position_ids = compute_position_id_with_mask(attention_mask)
#         batch_dict = {
#             "input_ids": input_ids,
#             "attention_mask": attention_mask,
#             "position_ids": position_ids,
#         }
#         data = DataProto.from_dict(batch_dict)
#         data.non_tensor_batch["formatted_prompts"] = np.array(formatted_prompts)

#         # original_batch contains the extra info needed for generation
#         if "meta_info" in kwargs and kwargs["meta_info"]:
#             meta_info = kwargs["meta_info"]
#             # only use the original_batch's meta_info since tensor_batch is from batch_dict and non_tensor_batch is not neeeded
#             data.meta_info = union_two_dict(data.meta_info, meta_info)

#         return data


# class AsyncAgentExecutionEngine(AgentExecutionEngine):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

# import asyncio
# import copy
# import concurrent.futures
# import logging
# import time
# import traceback
# import uuid
# from concurrent.futures import ThreadPoolExecutor

# import numpy as np
# import openai
# import torch
# from openai.types import Completion

# from rllm.agents.agent import Action, BaseAgent, Step, Trajectory
# from rllm.agents.utils import (
#     convert_messages_to_tokens_and_masks,
#     get_recent_assistant_user_messages,
# )
# from rllm.environments.base.base_env import BaseEnv
# from rllm.environments.env_utils import (
#     compute_mc_return,
#     compute_trajectory_reward,
# )
# from rllm.misc import colorful_print
# from rllm.parser.chat_template.parser import ChatTemplateParser
# from rllm.router.router import Router

# logger = logging.getLogger(__name__)


# class AgentExecutionEngine:
#     def __init__(
#         self,
#         engine_name="openai",
#         tokenizer=None,
#         rollout_engine=None,
#         chat_parser=None,
#         n_parallel_agents=1,
#         trajectory_timeout=None,
#         gamma=0.2,
#         api_retries=3,
#         retry_limit=3,
#         max_steps=5,
#         max_response_length=8192,
#         max_prompt_length=1024,
#         config=None,
#         agent_class=None,
#         env_class=None,
#         agent_args=None,
#         rollout_engine_args=None,
#         env_args=None,
#         max_workers=64,
#         enforce_max_prompt_length=False,  # If enabled, applies max_prompt check per step
#         overlong_filter=False,  # Filter for overlong trajectories (i.e. TRUNCATION, MAX_STEPS, TIMEOUT)
#         **kwargs,
#     ):
#         if agent_args is None:
#             agent_args = {}
#         if rollout_engine_args is None:
#             rollout_engine_args = {}
#         if env_args is None:
#             env_args = {}

#         self.config = config
#         self.rollout_engine = rollout_engine
#         self.tokenizer = tokenizer
#         self.engine_name = engine_name
#         self.n_parallel_agents = n_parallel_agents
#         self.overlong_filter = overlong_filter

#         # For interaction
#         self.gamma = gamma
#         self.retry_limit = retry_limit
#         self.api_retries = api_retries
#         self.max_steps = max_steps
#         self.max_response_length = max_response_length
#         self.max_prompt_length = max_prompt_length
#         self.enforce_max_prompt_length = enforce_max_prompt_length

#         self.agent_class = agent_class
#         self.agent_args = agent_args
#         self.env_class = env_class
#         self.env_args = env_args

#         self.agents = [None for _ in range(n_parallel_agents)]
#         self.envs = [None for _ in range(n_parallel_agents)]

#         self.trajectory_timeout = trajectory_timeout
#         if not trajectory_timeout:
#             self.trajectory_timeout = int(1e9)

#         if env_class is not None:
#             assert env_class.is_multithread_safe(), "Environment must be multithread safe for async engine"
#         # rollout engine args
#         self.rollout_engine_args = rollout_engine_args
#         self.sampling_params = kwargs.get("sampling_params", {})

#         assert self.engine_name in ["openai", "verl"], "Currently only openai and verl are supported as rollout engine"
#         if self.engine_name == "openai":
#             from openai import AsyncOpenAI

#             self.client = AsyncOpenAI(**self.rollout_engine_args)
#             # Disable httpx INFO logs that show HTTP requests
#             logging.getLogger("httpx").setLevel(logging.WARNING)
#         elif self.engine_name == "verl":
#             # All generation is done via scheduler. Currently only works for verl
#             self.server_addresses = getattr(self.rollout_engine, "server_addresses", [])
#             self.router = Router(config=self.config, tokenizer=self.tokenizer, addresses=self.server_addresses)

#         # Create a thread pool executor for environment interactions (i.e. step, reset, close)
#         self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

#         if chat_parser is None:
#             self.chat_parser = ChatTemplateParser.get_parser(self.tokenizer, disable_thinking=kwargs.get("disable_thinking", False))
#         else:
#             self.chat_parser = chat_parser

#     async def get_model_response(self, prompt, application_id, **kwargs):
#         """
#         Compute model response asynchronously based on the engine type.

#         This function is multithread safe and routes the request to the appropriate
#         engine-specific handler.

#         Args:
#             prompt: The input prompt to send to the model
#             application_id: Unique identifier for the application
#             **kwargs: Additional arguments to pass to the model

#         Returns:
#             The model's response text

#         Raises:
#             NotImplementedError: If the engine type is not supported
#         """
#         # print("PROMPT: ", prompt)
#         if self.engine_name == "openai":
#             return await self._get_openai_async(prompt, application_id, **kwargs)
#         elif self.engine_name == "verl":
#             return await self._get_verl_async(prompt, application_id, **kwargs)
#         else:
#             raise NotImplementedError(f"Engine type '{self.engine_name}' not supported")

#     def update_envs_and_agents(self, envs, agents):
#         """
#         Update the environments and agents.

#         Args:
#             envs: List of environments to use
#             agents: List of agents to use
#         """
#         assert len(agents) == len(envs), f"Number of agents must equal to number of environments but received, {len(agents)} and {len(envs)}"
#         self.envs = envs
#         # For keeping track of the environment index in the batch.
#         for idx, env in enumerate(envs):
#             env.idx = idx
#         self.agents = agents
#         self.n_parallel_agents = len(envs)

#     async def _get_verl_async(self, prompt, application_id, **kwargs):
#         batch = self._convert_prompt_verl([prompt], **kwargs)

#         if "max_tokens" in kwargs:
#             batch.meta_info["max_tokens"] = kwargs["max_tokens"]

#         output = await self.router.generate_sequences(batch, application_id=application_id, **kwargs)

#         attn = output.batch["attention_mask"][0, self.max_prompt_length :]
#         tokens = output.batch["responses"][0]

#         # Find last index where attention == 1
#         non_pad_indices = (attn == 1).nonzero(as_tuple=True)[0]
#         if len(non_pad_indices) == 0:
#             trimmed = tokens[:0]  # empty
#         else:
#             last_valid_idx = non_pad_indices[-1].item()
#             trimmed = tokens[: last_valid_idx + 1]  # include the last valid token

#         response = self.tokenizer.decode(trimmed, skip_special_tokens=False)

#         pad_token = self.tokenizer.pad_token
#         eos_token = self.tokenizer.eos_token
#         response = response.replace(pad_token, "").replace(eos_token, "")
#         return response

#     async def _get_openai_async(self, prompt, _, **kwargs):
#         """
#         Get action from OpenAI API asynchronously with retry logic.

#         Args:
#             prompt: The input prompt in text format for completions API
#             application_id: Unique identifier for the application (unused for OpenAI)
#             **kwargs: Additional arguments to pass to the OpenAI API

#         Returns:
#             The response from OpenAI API
#         """

#         async def get_response(prompt_text: str):
#             retries = self.api_retries
#             while retries > 0:
#                 try:
#                     response = await self.client.completions.create(
#                         prompt=prompt_text,
#                         timeout=3600,
#                         **self.sampling_params,
#                         **kwargs,
#                     )
#                     return response
#                 except openai.RateLimitError:
#                     retries -= 1
#                     if retries == 0:
#                         return "Error: Rate limit reached and retries exhausted."
#                     logger.info("Sleep for 5 seconds for API limit.")
#                     await asyncio.sleep(5)
#                 except Exception as e:
#                     logger.error("Error: %s", e)
#                     return f"Error processing content: {e}"

#         # If prompt is in chat format, convert it to text format
#         prompt_text = prompt
#         if isinstance(prompt, list) and all(isinstance(msg, dict) for msg in prompt):
#             prompt_text = self.chat_parser.parse(prompt, add_generation_prompt=True, is_first_msg=True)

#         response = await get_response(prompt_text)
#         if isinstance(response, Completion):
#             response = response.choices[0].text
#         return response

#     async def run_agent_trajectory_async(self, idx, application_id, seed=0, mode="Text", **kwargs):
#         """Run a single agent's trajectory asynchronously"""
#         agent = self.agents[idx]
#         env = self.envs[idx]
#         # env_id = env.env_id

#         termination_reason = None
#         prompt_token_len = 0
#         prompt_tokens = []
#         response_token_len = 0
#         response_tokens = []
#         response_masks = []
#         total_time = 0.0
#         reward_time = None
#         llm_time = 0.0
#         env_time = 0.0
#         reward = 0.0

#         # for step return
#         episode_steps = []

#         # Reset environment with the task using the executor
#         loop = asyncio.get_event_loop()
#         observation, info = await loop.run_in_executor(self.executor, env.reset)
#         info["max_steps"] = self.max_steps

#         # Reset agent
#         agent.reset()
#         # Update agent internal state from environment.
#         agent.update_from_env(
#             observation=observation,  # Raw observation from environment
#             reward=0.0,
#             done=False,
#             info=info,
#         )
        
#         messages = agent.chat_completions
#         prompt_tokens, _ = convert_messages_to_tokens_and_masks(messages, tokenizer=self.tokenizer, parser=self.chat_parser, contains_first_msg=True, contains_generation_msg=True)
#         prompt_token_len = len(prompt_tokens)
#         # Note, this should never happen!
#         if prompt_token_len > self.max_prompt_length:
#             agent.reset()
#             raise Exception(f"Trajectory {idx}: initial prompt length {prompt_token_len} already exceeded max_prompt_length {self.max_prompt_length}, retrying")

#         for step_idx in range(self.max_steps):
#             # Get action from agent
#             prompt_messages = agent.chat_completions.copy()
#             # Max remaining tokens left for the response
#             # For enforced max prompt at each step, no need to deduct here
#             if not self.enforce_max_prompt_length:
#                 max_tokens = self.max_response_length - response_token_len
#             else:
#                 max_tokens = self.max_response_length

#                 # since max prompt is enforced, we filter out too long prompts.
#                 prompt_str = self.chat_parser.parse(prompt_messages, add_generation_prompt=True, is_first_msg=True)
#                 prompt_len = len(self.tokenizer.encode(prompt_str, add_special_tokens=False))
#                 if prompt_len > self.max_prompt_length:
#                     termination_reason = "PROMPT_TRUNCATION"
#                     break

#             kwargs["max_tokens"] = max_tokens

#             start_time = time.time()
#             response = await self.get_model_response(prompt_messages, application_id, **kwargs)
#             delta_time = time.time() - start_time
#             llm_time += delta_time
#             total_time += delta_time
#             # Update steps
#             prompt_response_pair = {
#                 "prompt": self.chat_parser.parse(prompt_messages, add_generation_prompt=True, is_first_msg=True),
#                 "response": response,
#             }
#             episode_steps.append(prompt_response_pair)

#             # Update agent with model response
#             action: Action = agent.update_from_model(response)
#             action = action.action
            
#             # Extract thinking from response if present
#             thought = ""
#             if "</think>" in response:
#                 try:
#                     # Get everything before </think>
#                     think_text = response.split("</think>")[0]
#                     # Strip <think> tag if it exists
#                     if "<think>" in think_text:
#                         thought = think_text.split("<think>")[1].strip()
#                     else:
#                         thought = think_text.strip()
#                 except:
#                     thought = ""

#             # Take step in environment using the executor
#             start_time = time.time()

#             try:
#                 next_observation, reward, done, info = await asyncio.wait_for(loop.run_in_executor(self.executor, env.step, action), timeout=(self.trajectory_timeout - total_time))
#             except asyncio.TimeoutError:
#                 termination_reason = "ENV_TIMEOUT"
#                 if step_idx == 0:
#                     colorful_print(f"Warning: Trajectory {idx} completed due to: {termination_reason} before able to perform 1 complete action. This might cause unexpected behavior. Consider increasing trajectory timeout limit.\n", "red")
#                 reward = 0

#                 cur_step = agent.get_current_state()
#                 done = True
#                 cur_step.done = done
#                 break

#             delta_time = time.time() - start_time
#             env_time += delta_time
#             total_time += delta_time
#             info["max_steps"] = self.max_steps
#             info["cur_tokens"] = response_token_len

#             # Update agent internal state.
#             agent.update_from_env(
#                 observation=next_observation,
#                 reward=reward,
#                 done=done,
#                 info=info,
#             )

#             cur_step = agent.get_current_state()
#             cur_step.reward = reward
#             cur_step.done = done
#             cur_step.info.update(info)
            
#             # Add thought field to the step
#             if hasattr(cur_step, 'thought'):
#                 cur_step.thought = thought
#             else:
#                 # If thought attribute doesn't exist, add it dynamically
#                 setattr(cur_step, 'thought', thought)
            
#             # Update step extras with solver information from the observation
#             if hasattr(agent, '_current_obs') and isinstance(agent._current_obs, dict):
#                 # find the last agent chat completion with "role" == "user"
#                 last_user_msg = None
#                 for msg in reversed(agent.chat_completions):
#                     if msg["role"] == "user":
#                         last_user_msg = msg
#                         break
#                 context_manager_prompt = last_user_msg
#                 if not hasattr(cur_step, 'extras') or cur_step.extras is None:
#                     cur_step.extras = {}

#                 # Base fields from current observation
#                 obs_solver_prompt = agent._current_obs.get("solver_prompt", "")
#                 print(f"[agent_execution_engine] Obs solver prompt: {obs_solver_prompt}")
#                 obs_solver_full_output = agent._current_obs.get("solver_full_output", "")
#                 obs_solver_code = agent._current_obs.get("solver_output", "")
#                 obs_verifier_results = agent._current_obs.get("verifier_results", "")
#                 obs_passed_tests = agent._current_obs.get("passed_tests", 0)
#                 obs_total_tests = agent._current_obs.get("total_tests", 0)
#                 obs_solved = agent._current_obs.get("solved", False)
#                 obs_round_idx = agent._current_obs.get("round_idx", 0)

#                 cur_step.extras.update({
#                     "solver_prompt": obs_solver_prompt,
#                     "solver_full_output": obs_solver_full_output,
#                     "solver_code": obs_solver_code,
#                     "verifier_results": obs_verifier_results,
#                     "passed_tests": obs_passed_tests,
#                     "total_tests": obs_total_tests,
#                     "solved": obs_solved,
#                     "context_manager_prompt": self.chat_parser.parse([last_user_msg], add_generation_prompt=True, is_first_msg=True),
#                 })

#             chat_completions_messages = agent.chat_completions
#             assistant_message, env_messages = get_recent_assistant_user_messages(chat_completions_messages)

#             # Check and convert to tokens if necessary
#             assert assistant_message is not None or mode != "Token", "Assistant messages is none when accumulating token trajectories which should be conversations. This should not happen."
#             assert env_messages is not None or mode != "Token", "Environment messages is none when accumulating token trajectories which should be conversations. This should not happen."
#             assistant_msg_tokens, assistant_msg_masks = [], []
#             env_msg_tokens, env_msg_masks = [], []
#             if assistant_message:
#                 assistant_msg_tokens, assistant_msg_masks = convert_messages_to_tokens_and_masks([assistant_message], tokenizer=self.tokenizer, parser=self.chat_parser, contains_first_msg=False, contains_generation_msg=False)
#             if env_messages:
#                 env_msg_tokens, env_msg_masks = convert_messages_to_tokens_and_masks(env_messages, tokenizer=self.tokenizer, parser=self.chat_parser, contains_first_msg=False, contains_generation_msg=True)

#             # Update repsonse token length
#             response_token_len += len(assistant_msg_tokens) + len(env_msg_tokens)
#             # Reached maximum number of tokens for the trajectory
#             if not self.enforce_max_prompt_length and response_token_len >= self.max_response_length:
#                 # Truncation length
#                 truncation_length = self.max_response_length - response_token_len
#                 # Truncate the response and masks
#                 if truncation_length < 0:
#                     truncated_response_tokens = (assistant_msg_tokens + env_msg_tokens)[:truncation_length]
#                     truncated_response_masks = (assistant_msg_masks + env_msg_masks)[:truncation_length]
#                 else:
#                     # Edge case where the response is exactly the max response length.
#                     truncated_response_tokens = assistant_msg_tokens + env_msg_tokens
#                     truncated_response_masks = assistant_msg_masks + env_msg_masks
#                 # Update token collections
#                 response_tokens.extend(truncated_response_tokens)
#                 response_masks.extend(truncated_response_masks)

#                 cur_step = agent.get_current_state()
#                 if response_token_len - len(env_msg_tokens) > self.max_response_length:
#                     cur_step.reward = 0.0
#                 cur_step.done = True
#                 termination_reason = "TRUNCATION"
#                 # handle returning
#                 break

#             # Update the token version of trajectory
#             response_tokens.extend(assistant_msg_tokens)
#             response_masks.extend(assistant_msg_masks)
#             observation = next_observation

#             if total_time >= self.trajectory_timeout:
#                 termination_reason = "TIMEOUT"
#                 cur_step = agent.get_current_state()
#                 done = True
#                 cur_step.done = done
#                 break

#             # Check if episode is done
#             if done:
#                 termination_reason = "ENV_DONE"
#                 break

#             response_tokens.extend(env_msg_tokens)
#             response_masks.extend(env_msg_masks)

#             if step_idx == self.max_steps - 1:
#                 termination_reason = "MAX_STEPS"

#         masked_out = False
#         if self.overlong_filter:
#             if termination_reason == "TRUNCATION" or termination_reason == "MAX_STEPS" or termination_reason == "TIMEOUT":
#                 # Mask out the entire response for overlong trajectories if the reward is 0.
#                 response_masks = [0] * len(response_masks)
#                 masked_out = True

#         if hasattr(env, "compute_final_reward") and not masked_out:
#             cur_step = agent.get_current_state()
#             start_time = time.time()
#             reward = await loop.run_in_executor(self.executor, env.compute_final_reward)
#             reward_time = time.time() - start_time
#             cur_step.reward = reward
#         # Closing environment using the executor.
#         await loop.run_in_executor(self.executor, env.close)
#         if termination_reason:
#             if reward > 0:
#                 color = "green"
#             else:
#                 color = "yellow"
#             colorful_print(
#                 f"Trajectory {idx} completed due to: {termination_reason}. Reward is {reward}. \n",
#                 color,
#             )
#             if masked_out:
#                 colorful_print(f"Trajectory {idx} is masked out due to overlong filter.", "red")

#         trajectory: Trajectory = agent.trajectory
#         # Aggregate final trajectory statistics
#         compute_trajectory_reward(trajectory)
#         compute_mc_return(trajectory, gamma=self.gamma)

#         if mode == "Text":
#             return trajectory
#         elif mode == "Token":
#             token_result = {
#                 "prompt_tokens": torch.tensor(prompt_tokens, dtype=torch.long),
#                 "response_tokens": torch.tensor(response_tokens, dtype=torch.long),
#                 "response_masks": torch.tensor(response_masks, dtype=torch.long),
#                 "trajectory_reward": trajectory.reward,
#                 "idx": env.idx,
#                 "chat_completions": agent.chat_completions,
#                 "metrics": {
#                     # Total number of steps taken in the trajectory
#                     "steps": len(trajectory.steps),
#                     # Time to calculate reward
#                     "reward_time": reward_time,
#                     # Total time spent in environment execution (env.step)
#                     "env_time": env_time,
#                     # Time to calculate response tokens
#                     "llm_time": llm_time,
#                     # Total time spent in the trajectory
#                     "total_time": total_time,
#                 },
#             }
#             return token_result
#         elif mode == "Conversation":
#             return agent.chat_completions
#         elif mode == "Step":
#             steps_result = {
#                 "steps": episode_steps,
#                 "trajectory_reward": trajectory.reward,
#                 "idx": env.idx,
#                 "mc_returns": [step.mc_return for step in trajectory.steps][: len(episode_steps)],
#             }
#             return steps_result

#     async def run_agent_trajectory_with_retry(self, idx, application_id, seed=0, mode="Text", **kwargs):
#         for _ in range(self.retry_limit):
#             try:
#                 return await asyncio.wait_for(self.run_agent_trajectory_async(idx, application_id=application_id, seed=seed, mode=mode, **kwargs), timeout=7200)
#             except Exception:
#                 traceback.print_exc()
#                 continue
#         traceback.print_exc()
#         raise Exception(f"Trajectory {idx} cannot complete. Please check the log message")

#     async def trajectory_generator(self, reset_seed=0, timing_raw=None, mode="Text", **kwargs):
#         if timing_raw is None:
#             timing_raw = {}
#         assert all(env is not None and isinstance(env, BaseEnv) for env in self.envs), "All environments must be inheriting from BaseEnv"
#         assert all(env.is_multithread_safe() for env in self.envs), "All environments must be multithread safe for async engine"  # type: ignore
#         max_concurrency = self.n_parallel_agents
#         self.executor = ThreadPoolExecutor(max_workers=max_concurrency)

#         if self.engine_name == "verl":
#             # self.rollout_engine.wake_up()
#             wake = getattr(self.rollout_engine, "wake_up", None)
#             if callable(wake):
#                 wake()

#         async def launch_one_trajectory_task(env_idx: int):
#             try:
#                 application_id = str(uuid.uuid4())
#                 result = await self.run_agent_trajectory_with_retry(
#                     idx=env_idx,
#                     application_id=application_id,
#                     seed=reset_seed,
#                     mode=mode,
#                     **kwargs,
#                 )
#             except Exception as e:
#                 import traceback

#                 traceback.print_exc()
#                 raise e
#             return result

#         # Create all N conceptual tasks. Their execution will be throttled by the semaphore
#         # and the availability of agent/env indices.
#         tasks_to_run = [launch_one_trajectory_task(i) for i in range(len(self.envs))]

#         tasks_completed = 0
#         for coro in asyncio.as_completed(tasks_to_run):
#             try:
#                 result = await coro
#                 tasks_completed += 1
#                 colorful_print(f"Number of Trajectories {tasks_completed}/{len(self.envs)} completed", "cyan")
#                 yield result
#             except Exception as e:
#                 raise e

#         if self.engine_name == "verl":
#             self.rollout_engine.sleep()

#         self.executor.shutdown(wait=False, cancel_futures=True)

#     async def execute_tasks(self, tasks: list[dict]):
#         """
#         Run asynchronous interactions between the agent and environment where each agent
#         has its own environment instance and can proceed independently.

#         Args:
#             tasks: List of tasks to process
#             max_concurrent: Maximum number of concurrent tasks to process (defaults to self.n_parallel_agents)

#         Returns:
#             A list of trajectories, one for each task.
#         """

#         max_concurrent = self.n_parallel_agents

#         # Initialize results list to store trajectories for all tasks
#         all_trajectories = {}

#         # Create a queue of tasks to process
#         task_queue = list(enumerate(tasks))
#         semaphore = asyncio.Semaphore(max_concurrent)
#         index_queue: asyncio.Queue[int] = asyncio.Queue(maxsize=max_concurrent)
#         for i in range(max_concurrent):
#             index_queue.put_nowait(i)

#         # Track completed trajectories
#         completed = 0
#         total = len(tasks)

#         async def sem_wrapper(task_id, task):
#             nonlocal completed
#             async with semaphore:
#                 # Get an available index
#                 index = await index_queue.get()
#                 try:
#                     self.envs[index] = self.env_class.from_dict({**task, **self.env_args})
#                     self.agents[index] = self.agent_class(**self.agent_args)
#                     assert self.agents[index] is not None and isinstance(self.agents[index], BaseAgent), "Agent is not initalized or not inheriting from BaseAgent"
#                     self.agents[index].trajectory.task = task  # type: ignore
#                     res = await self.run_agent_trajectory_async(index, application_id=task_id)
#                     res.task = task
#                     completed += 1
#                     colorful_print(f"Progress: {completed}/{total} trajectories completed", "cyan")
#                     return task_id, res
#                 finally:
#                     # Put the index back in the queue when done
#                     await index_queue.put(index)

#         # Run all tasks concurrently
#         results = await asyncio.gather(*[sem_wrapper(task_id, task) for task_id, task in task_queue])

#         all_trajectories = {task_id: trajectory for task_id, trajectory in results}
#         ordered_trajectories = [all_trajectories[i] for i in range(len(all_trajectories))]
#         return ordered_trajectories

#     def _convert_prompt_verl(self, prompts, **kwargs):
#         """
#         Given a list of prompts in Chat template, convert to DataProto format in veRL

#         Args:
#             prompts: List of prompts to convert
#             **kwargs: Additional arguments

#         Returns:
#             DataProto object containing the converted prompts
#         """
#         from verl.protocol import DataProto, union_two_dict
#         from verl.utils.model import compute_position_id_with_mask
#         from verl.utils.torch_functional import pad_sequence_to_length

#         old_padding_side = self.tokenizer.padding_side
#         self.tokenizer.padding_side = "left"

#         formatted_prompts = [self.chat_parser.parse(prompt, add_generation_prompt=True, is_first_msg=True) for prompt in prompts]

#         # Tokenize the final processed strings
#         inputs = self.tokenizer(
#             formatted_prompts,
#             padding=True,
#             return_tensors="pt",
#             add_special_tokens=False,
#         )
#         self.tokenizer.padding_side = old_padding_side

#         input_ids = inputs["input_ids"]
#         attention_mask = inputs["attention_mask"]

#         # pad to max sizes
#         input_ids = pad_sequence_to_length(input_ids, max_seq_len=self.max_prompt_length, pad_token_id=self.tokenizer.pad_token_id, left_pad=True)
#         attention_mask = pad_sequence_to_length(attention_mask, max_seq_len=self.max_prompt_length, pad_token_id=0, left_pad=True)
#         position_ids = compute_position_id_with_mask(attention_mask)
#         batch_dict = {
#             "input_ids": input_ids,
#             "attention_mask": attention_mask,
#             "position_ids": position_ids,
#         }
#         data = DataProto.from_dict(batch_dict)
#         data.non_tensor_batch["formatted_prompts"] = np.array(formatted_prompts)

#         # original_batch contains the extra info needed for generation
#         if "meta_info" in kwargs and kwargs["meta_info"]:
#             meta_info = kwargs["meta_info"]
#             # only use the original_batch's meta_info since tensor_batch is from batch_dict and non_tensor_batch is not neeeded
#             data.meta_info = union_two_dict(data.meta_info, meta_info)

#         return data


# class AsyncAgentExecutionEngine(AgentExecutionEngine):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)