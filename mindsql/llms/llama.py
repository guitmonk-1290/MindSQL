from llama_cpp import Llama, LLAMA_SPLIT_LAYER, LLAMA_DEFAULT_SEED, LLAMA_ROPE_SCALING_UNSPECIFIED

from .._utils.constants import LLAMA_VALUE_ERROR, LLAMA_PROMPT_EXCEPTION
from .illm import ILlm


class LlamaCpp(ILlm):
    def __init__(self, config=None):
        """
        Initialize the class with an optional config parameter.

        Parameters:
            config (any): The configuration parameter.

        Returns:
            None
        """
        if config is None:
            raise ValueError("")

        if 'model_path' not in config:
            raise ValueError(LLAMA_VALUE_ERROR)
        path = config['model_path']
        
        if 'llama_params' in config:
            llama_params = config.get('llama_params', {})
            default_params = {
                'n_gpu_layers': 0,
                'split_mode': LLAMA_SPLIT_LAYER,
                'main_gpu': 0,
                'tensor_split': None,
                'vocab_only': False,
                'use_mmap': True,
                'use_mlock': False,
                'kv_overrides': None,
                # Context Params
                'seed': LLAMA_DEFAULT_SEED,
                'n_ctx': 512,
                'n_batch': 512,
                'n_threads': None,
                'n_threads_batch': None,
                'rope_scaling_type': LLAMA_ROPE_SCALING_UNSPECIFIED,
                'rope_freq_base': 0.0,
                'rope_freq_scale': 0.0,
                'yarn_ext_factor': -1.0,
                'yarn_attn_factor': 1.0,
                'yarn_beta_fast': 32.0,
                'yarn_beta_slow': 1.0,
                'yarn_orig_ctx': 0,
                'mul_mat_q': True,
                'logits_all': False,
                'embedding': False,
                'offload_kqv': True,
                # Sampling Params
                'last_n_tokens_size': 64,
                # LoRA Params
                'lora_base': None,
                'lora_scale': 1.0,
                'lora_path': None,
                # Backend Params
                'numa': False,
                # Chat Format Params
                'chat_format': None,
                'chat_handler': None,
                # Speculative Decoding
                'draft_model': None,
                # Tokenizer Override
                'tokenizer': None,
                # Misc
                'verbose': True,
            }

            # Combine parameters using custom values or defaults
            params = {key: llama_params.get(key, default_params[key]) for key in default_params}

            self.model = Llama(model_path=path, **params)
        else:
            self.model = Llama(model_path=path)

    def system_message(self, message: str) -> any:
        """
        Create a system message.

        Parameters:
            message (str): The content of the system message.

        Returns:
            any: A formatted system message.

        Example:
            system_msg = system_message("System update: Server maintenance scheduled.")
        """
        return {"role": "system", "content": message}

    def user_message(self, message: str) -> any:
        """
        Create a user message.

        Parameters:
            message (str): The content of the user message.

        Returns:
            any: A formatted user message.
        """
        return {"role": "user", "content": message}

    def assistant_message(self, message: str) -> any:
        """
        Create an assistant message.

        Parameters:
            message (str): The content of the assistant message.

        Returns:
            any: A formatted assistant message.
        """
        return {"role": "assistant", "content": message}

    def invoke(self, prompt, **kwargs) -> str:
        """
        Submit a prompt to the model for generating a response.

        Parameters:
            prompt (str): The prompt parameter.
            **kwargs: Additional keyword arguments (optional).
                - temperature (float): The temperature parameter for controlling randomness in generation.

        Returns:
            str: The generated response from the model.
        """
        if prompt is None or len(prompt) == 0:
            raise Exception(LLAMA_PROMPT_EXCEPTION)

        temperature = kwargs.get("temperature", 0.1)
        return self.model(prompt=prompt, temperature=temperature, echo=False)["choices"][0]["text"]
