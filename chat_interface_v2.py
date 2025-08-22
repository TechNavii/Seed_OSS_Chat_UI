#!/usr/bin/env python3
"""
Improved AI Chat Interface for Seed-OSS-36B Model
Version 2: Better handling of thinking phase and generation
"""

import os
import sys
import threading
import time
import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import gradio as gr
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    StoppingCriteria,
    StoppingCriteriaList,
)

warnings.filterwarnings("ignore", category=UserWarning)


class DeviceType(Enum):
    """Supported device types for model execution."""
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"
    AUTO = "auto"


@dataclass
class GenerationConfig:
    """Configuration for text generation parameters."""
    max_tokens: int = 9000  # Higher default to leave room after thinking
    temperature: float = 1.0  # Use 1.0 for better numerical stability
    top_p: float = 0.95  # Slightly higher for stability
    thinking_budget: int = 1024  # Used to cap thinking; clamped to leave answer room
    use_sampling: bool = True
    repetition_penalty: float = 1.02  # Gentle penalty; <=1.05 per tests
    min_response_tokens: int = 800  # Minimum tokens reserved for response


class DeviceManager:
    """Manages device detection and optimization."""
    
    @staticmethod
    def detect_best_device() -> DeviceType:
        """Auto-detect the best available device."""
        if torch.cuda.is_available():
            print("✅ CUDA GPU detected")
            return DeviceType.CUDA
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("✅ Apple Silicon (MPS) detected")
            return DeviceType.MPS
        else:
            print("ℹ️  Using CPU")
            return DeviceType.CPU
    
    @staticmethod
    def get_device_string(device_type: DeviceType) -> str:
        """Get device string for PyTorch."""
        if device_type == DeviceType.AUTO:
            device_type = DeviceManager.detect_best_device()
        return device_type.value if device_type != DeviceType.AUTO else "cpu"


class ModelConfig:
    """Model configuration with device-specific optimizations."""
    
    def __init__(self, device_type: DeviceType = DeviceType.AUTO, quantization: Optional[int] = None):
        if device_type == DeviceType.AUTO:
            device_type = DeviceManager.detect_best_device()
        
        self.device_type = device_type
        self.device_string = DeviceManager.get_device_string(device_type)
        self.quantization = quantization
        
        # Set device-specific configurations
        self._configure_for_device()
    
    def _configure_for_device(self):
        """Configure settings based on device type."""
        if self.device_type == DeviceType.CUDA:
            self.device_map = "auto"
            self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            self.low_cpu_mem_usage = True
        elif self.device_type == DeviceType.MPS:
            self.device_map = "auto"
            self.dtype = torch.float16
            self.low_cpu_mem_usage = True
        else:
            self.device_map = "cpu"
            self.dtype = torch.float32
            self.low_cpu_mem_usage = True
    
    def get_load_kwargs(self) -> Dict:
        """Get model loading kwargs based on configuration."""
        kwargs = {
            "device_map": self.device_map,
            "torch_dtype": self.dtype,
            "low_cpu_mem_usage": self.low_cpu_mem_usage,
            "trust_remote_code": True,
        }
        
        if self.device_type == DeviceType.CUDA and self.quantization:
            kwargs["load_in_8bit"] = (self.quantization == 8)
            kwargs["load_in_4bit"] = (self.quantization == 4)
        
        return kwargs


class ResponseParser:
    """Parses model responses and handles special tokens."""
    
    SEED_TAGS = {
        'think_start': '<seed:think>',
        'think_end': '</seed:think>',
        'eos': '<seed:eos>',
        'bos_assistant': '<seed:bosassistant>',
        'eos_assistant': '<seed:eosassistant>',
    }
    
    def extract_components(self, text: str) -> Tuple[str, str]:
        """Extract thinking and response components from text."""
        if not text:
            return "", ""
        
        # Find thinking section
        reasoning = ""
        if self.SEED_TAGS['think_start'] in text:
            start_idx = text.find(self.SEED_TAGS['think_start'])
            end_idx = text.find(self.SEED_TAGS['think_end'], start_idx)
            
            if end_idx > start_idx:
                reasoning = text[start_idx + len(self.SEED_TAGS['think_start']):end_idx].strip()
            elif start_idx >= 0:
                # Thinking tag started but not closed yet
                reasoning = text[start_idx + len(self.SEED_TAGS['think_start']):].strip()
        
        # Extract actual response (after thinking)
        response = ""
        if self.SEED_TAGS['think_end'] in text:
            response_start = text.find(self.SEED_TAGS['think_end']) + len(self.SEED_TAGS['think_end'])
            response_text = text[response_start:].strip()
            
            # Debug: Check what comes after thinking
            if not response_text and response_start < len(text):
                print(f"[Debug] After think_end, found: {repr(text[response_start:response_start+100])}")
            
            response = response_text
        elif self.SEED_TAGS['think_start'] not in text:
            # No thinking phase, entire text is response
            response = text.strip()
        
        # Clean up response from special tokens (but preserve actual content)
        # Only remove seed tags, not the actual response
        for tag_name, tag_value in self.SEED_TAGS.items():
            if tag_name != 'think_end':  # Don't remove content after think_end
                response = response.replace(tag_value, '')
        
        response = response.strip()
        
        return reasoning, response
    
    def is_thinking_complete(self, text: str) -> bool:
        """Check if thinking phase is complete."""
        if self.SEED_TAGS['think_start'] not in text:
            return True
        if self.SEED_TAGS['think_end'] in text:
            return True
        return False


class ModelManager:
    """Manages model loading and initialization."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
    
    def load_model(self, model_path: str) -> None:
        """Load model and tokenizer with appropriate settings."""
        model_path = self._resolve_model_path(model_path)
        
        print(f"📦 Loading tokenizer from: {model_path}")
        self._load_tokenizer(model_path)
        
        print(f"🤖 Loading model on {self.config.device_type.value}...")
        self._load_model(model_path)
        
        # Verify model loaded correctly
        if self.model is not None:
            print("✅ Model loaded and ready!")
        else:
            raise RuntimeError("Failed to load model")
    
    def _resolve_model_path(self, model_path: str) -> str:
        """Resolve and validate model path."""
        # Convert relative paths to absolute
        if model_path.startswith(('./', '../', '~/')):
            model_path = str(Path(model_path).expanduser().resolve())
        elif not model_path.startswith('/') and '/' not in model_path:
            # Assume it's a relative path
            model_path = str(Path(model_path).resolve())
        
        # Check if it's a local path
        if os.path.exists(model_path):
            print(f"✅ Using local model at: {model_path}")
            return model_path
        
        # Otherwise assume it's a HuggingFace model ID
        print(f"📥 Using model from HuggingFace: {model_path}")
        return model_path
    
    def _load_tokenizer(self, model_path: str) -> None:
        """Load and configure tokenizer."""
        use_local = os.path.exists(model_path)
        
        tokenizer_kwargs = {
            "trust_remote_code": True,
            "use_fast": True,
        }
        
        if use_local:
            tokenizer_kwargs["local_files_only"] = True
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _load_model(self, model_path: str) -> None:
        """Load model with error handling and fallback."""
        use_local = os.path.exists(model_path)
        
        try:
            load_kwargs = self.config.get_load_kwargs()
            if use_local:
                load_kwargs["local_files_only"] = True
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **load_kwargs
            )
            
            # Apply device-specific optimizations
            if self.config.device_type == DeviceType.MPS:
                self._optimize_for_mps()
                
        except ValueError as e:
            if "seed_oss" in str(e):
                self._handle_seed_oss_error()
            else:
                self._try_cpu_fallback(model_path, use_local, e)
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def _optimize_for_mps(self) -> None:
        """Apply MPS-specific optimizations."""
        if hasattr(self.model, 'to'):
            self.model = self.model.to('mps')
        
        if hasattr(self.model.config, 'use_cache'):
            self.model.config.use_cache = True
    
    def _handle_seed_oss_error(self) -> None:
        """Handle seed_oss architecture error."""
        print("\n" + "=" * 60)
        print("⚠️  IMPORTANT: Seed-OSS Requires Custom Transformers Fork")
        print("=" * 60)
        print("\nInstall the custom fork:")
        print("   pip uninstall transformers -y")
        print("   pip install git+https://github.com/Fazziekey/transformers.git@seed-oss")
        print("=" * 60)
        raise ValueError("Seed-OSS model requires custom transformers fork")
    
    def _try_cpu_fallback(self, model_path: str, use_local: bool, original_error: Exception) -> None:
        """Try falling back to CPU if GPU loading fails."""
        if self.config.device_type != DeviceType.CPU:
            print(f"❌ Error: {original_error}")
            print("Attempting CPU fallback...")
            
            self.config.device_type = DeviceType.CPU
            self.config._configure_for_device()
            
            try:
                load_kwargs = self.config.get_load_kwargs()
                if use_local:
                    load_kwargs["local_files_only"] = True
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    **load_kwargs
                )
                print("✅ Model loaded on CPU (fallback)")
            except Exception as cpu_error:
                print(f"❌ CPU fallback failed: {cpu_error}")
                raise original_error
        else:
            raise original_error


class ImprovedChatStreamer:
    """Enhanced streaming with better handling of thinking phase."""
    
    def __init__(self, model_manager: ModelManager, parser: ResponseParser):
        self.model_manager = model_manager
        self.parser = parser
        self.generation_timeout = 1800  # 30 minutes timeout for long tasks
    
    def generate_response(
        self,
        message: str,
        history: list,
        generation_config: GenerationConfig
    ) -> Iterator[str]:
        """Generate streaming response with improved handling."""
        try:
            if self.model_manager.model is None:
                yield "❌ Model not loaded. Please check the setup."
                return
            
            # Build messages
            messages = self._build_messages(history, message)
            
            # Tokenize with bounded thinking budget (ensures room for answer)
            input_ids = self._tokenize_with_thinking(messages, generation_config)
            
            # Create streamer
            streamer = TextIteratorStreamer(
                self.model_manager.tokenizer,
                timeout=30.0,
                skip_prompt=True,
                skip_special_tokens=False
            )
            
            # Start generation with an interrupt event so we can stop cleanly
            stop_event = threading.Event()
            generation_thread = self._start_generation_thread(
                input_ids, streamer, generation_config, stop_event
            )
            
            # Stream response with improved handling
            yield from self._stream_response_improved(
                streamer,
                generation_thread,
                stop_event,
                message,
                history,
                generation_config,
            )
            
        except Exception as e:
            print(f"❌ Generation error: {e}")
            import traceback
            traceback.print_exc()
            yield f"❌ Error: {str(e)}"
    
    def _build_messages(self, history: list, current: str) -> list:
        """Build message list from history."""
        messages = []
        for user_msg, assistant_msg in history:
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if assistant_msg:
                messages.append({"role": "assistant", "content": assistant_msg})
        messages.append({"role": "user", "content": current})
        return messages
    
    def _tokenize_with_thinking(
        self, 
        messages: list, 
        config: GenerationConfig
    ) -> torch.Tensor:
        """Tokenize with a SAFE thinking budget to avoid indefinite thinking.

        We clamp the thinking_budget so that it never exceeds
        (max_tokens - min_response_tokens), ensuring there is always
        room for the actual answer.
        """
        max_for_thinking = max(0, config.max_tokens - max(0, config.min_response_tokens))
        safe_thinking_budget = max(0, min(config.thinking_budget, max_for_thinking))
        print(
            f"[Debug] Tokenizing with thinking_budget={safe_thinking_budget} (max_for_thinking={max_for_thinking})"
        )
        
        try:
            if safe_thinking_budget == 0:
                # No-think mode: avoid injecting any think tags via template
                input_ids = self.model_manager.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
            else:
                try:
                    # Prefer passing thinking_budget if supported by the custom fork
                    input_ids = self.model_manager.tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt",
                        thinking_budget=safe_thinking_budget,
                    )
                except TypeError:
                    # Fallback for environments without thinking_budget support
                    print("[Debug] Tokenizer.apply_chat_template doesn't accept thinking_budget; falling back without it")
                    input_ids = self.model_manager.tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt",
                    )
            print("[Debug] Successfully tokenized")
        except Exception as e:
            print(f"[Debug] Tokenization error: {e}")
            raise
        
        return input_ids
    
    def _start_generation_thread(
        self,
        input_ids: torch.Tensor,
        streamer: TextIteratorStreamer,
        config: GenerationConfig,
        stop_event: threading.Event,
    ) -> threading.Thread:
        """Start generation in a separate thread."""
        # Get device
        device = self._get_device()
        input_ids = input_ids.to(device)
        
        # Create attention mask
        attention_mask = torch.ones_like(input_ids)
        
        # Generation kwargs
        # Ensure minimum tokens for both thinking AND response (guidance only)
        # Don't set min_new_tokens too high as it can cause freezing
        estimated_thinking = min(1000, config.thinking_budget or 1000)
        min_tokens = min(estimated_thinking + config.min_response_tokens, 4000)  # Cap to prevent lockups
        
        # Ensure max_tokens is high enough but reasonable
        actual_max_tokens = config.max_tokens
        
        # Ensure min_tokens doesn't exceed max_tokens
        if min_tokens > actual_max_tokens:
            min_tokens = min(500, actual_max_tokens // 2)  # More conservative
        
        # For code generation, we need different parameters
        # Ensure numerical stability
        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": actual_max_tokens,  # Use configured max
            "min_new_tokens": None,  # Don't force minimum - let model decide when to stop
            "temperature": max(0.1, min(2.0, config.temperature)),  # Clamp to valid range
            "top_p": max(0.1, min(1.0, config.top_p)),  # Clamp to valid range
            "do_sample": config.use_sampling,
            "repetition_penalty": max(1.0, min(2.0, config.repetition_penalty)),  # Clamp to valid range
            "pad_token_id": self.model_manager.tokenizer.pad_token_id,
            "eos_token_id": self.model_manager.tokenizer.eos_token_id,
            "streamer": streamer,
            # Add for numerical stability
            "top_k": 50,  # Limit vocab for stability
            "num_beams": 1,  # Use greedy for stability
        }

        # In no-think mode, discourage think/budget tokens entirely
        if config.thinking_budget == 0:
            banned = []
            tok = self.model_manager.tokenizer
            try:
                for token_str in ("<seed:think>", "<seed:cot_budget_reflect>"):
                    ids = tok.encode(token_str, add_special_tokens=False)
                    if ids:
                        banned.append(ids)
            except Exception:
                pass
            if banned:
                gen_kwargs["bad_words_ids"] = banned
        
        print(f"[Debug] Generation config: max={actual_max_tokens}, min=None (not forcing), thinking_est={estimated_thinking}")

        # Interrupt generation cleanly when requested
        class _EventStop(StoppingCriteria):
            def __init__(self, event: threading.Event):
                super().__init__()
                self._event = event

            def __call__(self, input_ids: torch.LongTensor, scores: Optional[torch.FloatTensor] = None, **kwargs) -> bool:  # type: ignore[override]
                return self._event.is_set()

        gen_kwargs["stopping_criteria"] = StoppingCriteriaList([_EventStop(stop_event)])
        
        # Wrap generation in try-catch for better error handling
        def safe_generate():
            try:
                self.model_manager.model.generate(**gen_kwargs)
            except RuntimeError as e:
                if "probability tensor" in str(e):
                    print(f"[Error] Numerical instability in generation: {e}")
                    print("[Info] Retrying with safer parameters...")
                    # Retry with safer parameters
                    gen_kwargs["temperature"] = 1.0
                    gen_kwargs["top_p"] = 0.95
                    gen_kwargs["repetition_penalty"] = 1.0
                    gen_kwargs["do_sample"] = False  # Use greedy decoding
                    self.model_manager.model.generate(**gen_kwargs)
                else:
                    raise
        
        thread = threading.Thread(target=safe_generate)
        thread.start()
        return thread
    
    def _get_device(self) -> torch.device:
        """Get model device."""
        if hasattr(self.model_manager.model, 'device'):
            return self.model_manager.model.device
        
        try:
            return next(self.model_manager.model.parameters()).device
        except StopIteration:
            return torch.device(self.model_manager.config.device_string)

    def _force_answer_after_think(
        self,
        message: str,
        history: list,
        reasoning_text: str,
        base_config: GenerationConfig,
    ) -> Iterator[str]:
        """Fallback: prefill reasoning and generate only the answer.

        Strategy:
        - Build a new message list with the conversation history and the current user message.
        - Prefill an assistant message containing only the captured reasoning (closed with think_end).
        - Start a fresh generation with thinking_budget=0 to encourage immediate answering.
        - Stream only the response part to the UI.
        """
        try:
            # Build messages with prefilled reasoning
            messages = []
            for user_msg, assistant_msg in history:
                if user_msg:
                    messages.append({"role": "user", "content": user_msg})
                if assistant_msg:
                    messages.append({"role": "assistant", "content": assistant_msg})
            messages.append({"role": "user", "content": message})

            if reasoning_text:
                messages.append({
                    "role": "assistant",
                    "reasoning_content": reasoning_text,
                    "content": "",
                })

            # Use zero thinking_budget to skip further thinking
            try:
                input_ids = self.model_manager.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    thinking_budget=0,
                )
            except TypeError:
                # Fallback if tokenizer doesn't accept thinking_budget kwarg
                input_ids = self.model_manager.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )

            # New streamer and thread
            streamer = TextIteratorStreamer(
                self.model_manager.tokenizer,
                timeout=30.0,
                skip_prompt=True,
                skip_special_tokens=False,
            )
            stop_event = threading.Event()
            thread = self._start_generation_thread(
                input_ids, streamer, base_config, stop_event
            )

            # Stream only the response part
            accumulated = ""
            start_time = time.time()
            total_tokens = 0
            first_token_time = None
            last_output = ""

            for chunk in streamer:
                if not chunk:
                    continue
                accumulated += chunk
                if first_token_time is None:
                    first_token_time = time.time()
                total_tokens += max(1, len(chunk) // 4)

                _, response = self.parser.extract_components(accumulated)
                current_time = time.time() - start_time
                speed = total_tokens / current_time if current_time > 0 else 0
                ttft = (first_token_time - start_time) if first_token_time else 0

                if response:
                    mini = f"\n\n---\n⚡ {speed:.1f} tok/s | ⏱️ {current_time:.1f}s | 🚀 TTFT: {ttft:.3f}s | 📝 {total_tokens} tokens"
                    output = f"💬 **Response:**\n\n{response}{mini}"
                else:
                    output = "⏳ Generating answer..."

                if output != last_output:
                    last_output = output
                    yield output

            thread.join(timeout=10)
            if thread.is_alive():
                stop_event.set()
                thread.join(timeout=5)

            # Finalize
            _, response = self.parser.extract_components(accumulated)
            if response and response != last_output:
                yield f"💬 **Response:**\n\n{response}"

        except Exception as e:
            print(f"[Error] Force-answer fallback failed: {e}")
            yield "❌ Failed to force answer generation. Try again or adjust settings."
    
    def _stream_response_improved(
        self, 
        streamer: TextIteratorStreamer,
        thread: threading.Thread,
        stop_event: threading.Event,
        message: str,
        history: list,
        config: GenerationConfig,
    ) -> Iterator[str]:
        """Improved streaming with better thinking phase handling and telemetry."""
        accumulated = ""
        last_output = ""
        chunk_count = 0
        empty_count = 0
        thinking_phase = False
        response_started = False
        error_occurred = False
        
        # Telemetry metrics
        start_time = time.time()
        first_token_time = None
        thinking_tokens = 0
        response_tokens = 0
        total_tokens = 0
        
        try:
            # Compute a safe cap for thinking tokens to avoid endless reasoning
            safe_thinking_cap = max(0, min(config.thinking_budget, config.max_tokens - max(0, config.min_response_tokens)))
            safe_thinking_cap = max(0, safe_thinking_cap)
            for chunk in streamer:
                chunk_count += 1
                elapsed = time.time() - start_time
                
                # Timeout check
                if elapsed > self.generation_timeout:
                    print(f"[Warning] Generation timeout after {elapsed:.1f}s")
                    # Stop generation thread cleanly
                    stop_event.set()
                    yield "⚠️ Generation timed out. Please try again."
                    break
                
                # Check if stuck in thinking for too long (cap by config)
                if thinking_phase and (thinking_tokens >= safe_thinking_cap + 32):
                    print(f"[Warning] Thinking exceeded cap: {thinking_tokens} >= {safe_thinking_cap}")
                    # Request stop and surface a warning
                    stop_event.set()
                    # Wait briefly for the generator to stop before fallback
                    for _ in range(10):  # up to ~10s
                        if not thread.is_alive():
                            break
                        thread.join(timeout=1)
                    yield "⚠️ Thinking budget reached; switching to answer mode."
                    # Try to force answer generation in a second pass
                    reasoning_text, _ = self.parser.extract_components(accumulated)
                    yield from self._force_answer_after_think(
                        message, history, reasoning_text, config
                    )
                    return
                
                # Handle empty chunks
                if not chunk:
                    empty_count += 1
                    if empty_count > 50:  # Reduced threshold
                        print(f"[Debug] Too many empty chunks ({empty_count}), checking status")
                        # If we're past thinking and the model is still alive, keep waiting
                        if thinking_phase and len(accumulated) > 5000:
                            print("[Warning] Very long thinking phase, may be stuck")
                            stop_event.set()
                            # Wait briefly for the generator to stop before fallback
                            for _ in range(6):  # up to ~6s
                                if not thread.is_alive():
                                    break
                                thread.join(timeout=1)
                            yield "⚠️ Extended thinking detected; switching to answer mode."
                            reasoning_text, _ = self.parser.extract_components(accumulated)
                            yield from self._force_answer_after_think(
                                message, history, reasoning_text, config
                            )
                            return
                        # Only break if the generation thread has actually finished
                        if not thread.is_alive() and accumulated and self.parser.is_thinking_complete(accumulated):
                            break
                        # Otherwise, reset the counter and continue waiting
                        empty_count = 0
                    continue
                
                accumulated += chunk
                
                # Track first token time
                if chunk and first_token_time is None:
                    first_token_time = time.time()
                
                # Count tokens approximately (rough estimate based on characters)
                if chunk:
                    # Rough token estimation (1 token ≈ 4 characters)
                    chunk_tokens = max(1, len(chunk) // 4)
                    total_tokens += chunk_tokens
                    
                    if thinking_phase:
                        thinking_tokens += chunk_tokens
                    elif response_started:
                        response_tokens += chunk_tokens
                
                # Debug logging for first chunks
                if chunk_count <= 5:
                    print(f"[Debug] Chunk {chunk_count}: {repr(chunk[:50])}")
                
                # Detect thinking phase
                if '<seed:think>' in accumulated and not thinking_phase:
                    thinking_phase = True
                    print("[Debug] Entered thinking phase")
                
                # Check if thinking is complete
                if thinking_phase and '</seed:think>' in accumulated:
                    thinking_phase = False
                    response_started = True
                    print("[Debug] Thinking complete, response starting")
                
                # Parse content
                reasoning, response = self.parser.extract_components(accumulated)
                
                # Calculate current telemetry
                current_time = time.time() - start_time
                current_speed = total_tokens / current_time if current_time > 0 else 0
                ttft = first_token_time - start_time if first_token_time else 0
                
                # Mini telemetry for real-time display
                if first_token_time and ttft > 0:
                    mini_telemetry = f"\n\n---\n⚡ {current_speed:.1f} tok/s | ⏱️ {current_time:.1f}s | 🚀 TTFT: {ttft:.3f}s | 📝 {total_tokens} tokens"
                else:
                    mini_telemetry = f"\n\n---\n⚡ {current_speed:.1f} tok/s | ⏱️ {current_time:.1f}s | 📝 {total_tokens} tokens"
                
                # Format output based on state
                if thinking_phase:
                    # Show thinking progress with live metrics (truncate very long thinking)
                    truncated_reasoning = reasoning[:1000] + '...' if len(reasoning) > 1000 else reasoning
                    output = f"🧠 **Thinking...**\n\n```\n{truncated_reasoning}\n```\n\n⏳ Processing...{mini_telemetry}"
                elif response and reasoning:
                    # Show BOTH reasoning and response with metrics
                    output = f"🧠 **Thinking Process:**\n\n```\n{reasoning}\n```\n\n💬 **Response:**\n\n{response}{mini_telemetry}"
                elif response:
                    # Show just response if no reasoning
                    output = f"💬 **Response:**\n\n{response}{mini_telemetry}"
                elif reasoning and response_started:
                    # Show reasoning if we have it but no response yet
                    output = f"🧠 **Reasoning:**\n\n```\n{reasoning}\n```\n\n⏳ Generating response...{mini_telemetry}"
                else:
                    # Still waiting
                    output = f"⏳ Generating...{mini_telemetry if total_tokens > 0 else ''}"
                
                # Yield if different from last
                if output != last_output:
                    last_output = output
                    yield output
            
            # Wait for thread completion (graceful)
            # Always signal stop once streaming ended
            stop_event.set()
            deadline = time.time() + 30  # wait up to 30s
            while thread.is_alive() and time.time() < deadline:
                thread.join(timeout=1)
            
            # Check if thread had an error
            if thread.is_alive():
                # Avoid alarming log spam; note but continue
                print("[Debug] Generation thread still running after stream end; proceeding")
                error_occurred = True
            
            # Final output with telemetry
            if accumulated and not error_occurred:
                reasoning, response = self.parser.extract_components(accumulated)
                
                # Debug logging for empty response
                if not response and reasoning:
                    print(f"[Debug] No response after thinking. Accumulated text length: {len(accumulated)}")
                    print(f"[Debug] Thinking complete: {self.parser.is_thinking_complete(accumulated)}")
                    print(f"[Debug] Last 200 chars of accumulated: {repr(accumulated[-200:])}")
                
                # Calculate final telemetry
                total_time = time.time() - start_time
                time_to_first_token = (first_token_time - start_time) if first_token_time else 0
                tokens_per_second = total_tokens / total_time if total_time > 0 else 0
                
                # Get actual token counts using tokenizer
                if response:
                    actual_response_tokens = len(self.model_manager.tokenizer.encode(response))
                else:
                    actual_response_tokens = 0
                
                if reasoning:
                    actual_thinking_tokens = len(self.model_manager.tokenizer.encode(reasoning))
                else:
                    actual_thinking_tokens = 0
                
                actual_total_tokens = actual_response_tokens + actual_thinking_tokens
                
                # Check if response is suspiciously short
                response_warning = ""
                if actual_response_tokens < 10 and actual_thinking_tokens > 0:
                    response_warning = "\n⚠️ **Response is very short!** Try increasing 'Minimum Response Tokens' in settings."
                
                # Format telemetry
                telemetry = f"""
📊 **Generation Metrics:**
- ⏱️ Total time: {total_time:.2f}s
- 🚀 Time to first token: {time_to_first_token:.3f}s
- 📝 Total tokens: {actual_total_tokens} (thinking: {actual_thinking_tokens}, response: {actual_response_tokens})
- ⚡ Speed: {tokens_per_second:.1f} tokens/sec (avg)
- 🔄 Chunks processed: {chunk_count}{response_warning}
"""
                
                if response and reasoning:
                    # Show BOTH reasoning and response in final output
                    final = f"🧠 **Thinking Process:**\n\n```\n{reasoning}\n```\n\n💬 **Response:**\n\n{response}\n{telemetry}"
                elif response:
                    final = f"💬 **Response:**\n\n{response}\n{telemetry}"
                elif reasoning:
                    final = f"🧠 **Reasoning:**\n\n```\n{reasoning}\n```\n\n💬 **Response:** [No response generated]\n{telemetry}"
                else:
                    final = f"📝 **Raw output:**\n\n```\n{accumulated[:1000]}\n```\n{telemetry}"
                
                if final != last_output:
                    yield final
            elif not last_output or last_output == "⏳ Generating...":
                yield "⚠️ No response generated. Try adjusting parameters or reloading the model."
                
        except Exception as e:
            print(f"[Error] Streaming error: {e}")
            import traceback
            traceback.print_exc()
            
            # Try to provide helpful error message
            if "probability tensor" in str(e):
                yield "❌ Error: Numerical instability detected. Try adjusting temperature to 1.0 and using greedy decoding."
            elif "out of memory" in str(e).lower():
                yield "❌ Error: Out of memory. Try reducing max tokens or batch size."
            else:
                yield f"❌ Error during generation: {str(e)}\n\nTry adjusting generation parameters or restarting the interface."


class ChatApplication:
    """Main chat application with improved UI handling."""
    
    def __init__(self, model_path: str, config: ModelConfig):
        self.model_path = model_path
        self.config = config
        self.model_manager = ModelManager(config)
        self.parser = ResponseParser()
        self.streamer = ImprovedChatStreamer(self.model_manager, self.parser)
        
        # Load model
        print("\n" + "=" * 60)
        print("🚀 Initializing Seed-OSS Chat Interface v2")
        print("=" * 60)
        self._print_system_info()
        
        try:
            self.model_manager.load_model(model_path)
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            sys.exit(1)
    
    def _print_system_info(self):
        """Print system and configuration information."""
        print("\n📊 System Information:")
        print(f"  Platform: {sys.platform}")
        print(f"  Python: {sys.version.split()[0]}")
        print(f"  PyTorch: {torch.__version__}")
        print("\n⚙️  Configuration:")
        print(f"  Device: {self.config.device_type.value}")
        print(f"  Data Type: {self.config.dtype}")
        if self.config.quantization:
            print(f"  Quantization: {self.config.quantization}-bit")
    
    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface with improved handling."""
        with gr.Blocks(
            title="Seed-OSS Chat v2",
            theme=gr.themes.Soft(),
            css=self._get_custom_css()
        ) as interface:
            gr.Markdown("# 🌱 Seed-OSS-36B Chat Interface v2")
            gr.Markdown("Enhanced interface with improved thinking phase handling")
            
            # Chat interface
            chatbot = gr.Chatbot(
                height=500,
                show_label=False,
                elem_classes="chatbot"
            )
            
            # Input row
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Type your message here...",
                    show_label=False,
                    lines=3,
                    max_lines=5,
                    elem_classes="message-input",
                    scale=9
                )
                
                with gr.Column(scale=1):
                    submit_btn = gr.Button(
                        "Send",
                        variant="primary",
                        elem_classes="send-button"
                    )
                    clear_btn = gr.Button(
                        "Clear",
                        variant="secondary"
                    )
            
            # Generation settings
            with gr.Accordion("⚙️ Generation Settings", open=False):
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    label="Temperature",
                    info="1.0 is recommended for stability"
                )
                
                max_tokens = gr.Slider(
                    minimum=50,
                    maximum=10000,
                    value=9000,  # Higher default to ensure room after thinking
                    step=50,
                    label="Max Tokens (Total for thinking + response)",
                    info="Controls total generation length. Higher = longer responses"
                )
                
                thinking_budget = gr.Slider(
                    minimum=0,
                    maximum=4096,
                    value=1024,  # Keep at 1024 per user requirement
                    step=256,
                    label="Thinking Budget (UI reference only)",
                    info="Note: Actual thinking uses max_tokens to avoid early stopping. This value is for display only."
                )
                
                min_response = gr.Slider(
                    minimum=50,
                    maximum=3000,
                    value=800,  # Ensure stronger minimum for responses
                    step=50,
                    label="Minimum Response Tokens",
                    info="Minimum tokens to generate for the response after thinking (increase for code generation)"
                )
                
                timeout_mins = gr.Slider(
                    minimum=1,
                    maximum=60,
                    value=30,
                    step=5,
                    label="Generation Timeout (minutes)",
                    info="Maximum time for generation before timeout"
                )
            
            # Event handlers
            def handle_message(message, history, temp, max_tok, think_budget, min_resp, timeout_minutes):
                """Handle message submission with improved input preservation."""
                if not message.strip():
                    yield history, message
                    return
                
                # Detect code generation requests
                code_keywords = ['code', 'html', 'webpage', 'website', 'javascript', 'python', 'css', 'program', 'script', 'function', 'implement']
                is_code_request = any(keyword in message.lower() for keyword in code_keywords)
                
                if is_code_request:
                    print("[Info] Code generation request detected - adjusting parameters")
                    # For code generation, ensure higher minimums
                    if min_resp < 800:
                        min_resp = 800
                        print(f"[Info] Increased min_response to {min_resp} for code generation")
                    if max_tok < 6000:
                        max_tok = 6000
                        print(f"[Info] Increased max_tokens to {max_tok} for code generation")
                
                # Validate token settings
                if think_budget + min_resp > max_tok:
                    warning_msg = f"⚠️ Warning: Max tokens ({max_tok}) should be >= thinking budget ({think_budget}) + min response ({min_resp})"
                    print(warning_msg)
                    history = history + [[message, warning_msg]]
                    yield history, message
                    # Continue anyway but with warning
                
                # Store original message
                original_msg = message
                
                # Add to history
                history = history + [[message, ""]]
                yield history, original_msg  # Keep input visible
                
                # Set timeout for this generation
                self.streamer.generation_timeout = timeout_minutes * 60  # Convert to seconds
                
                # Generate response
                config = GenerationConfig(
                    temperature=temp,
                    max_tokens=max_tok,
                    thinking_budget=think_budget,
                    min_response_tokens=min_resp
                )
                
                print(f"[Info] Generating with max_tokens={max_tok}, thinking_budget={think_budget}, min_response={min_resp}")
                
                first_chunk = True
                for response in self.streamer.generate_response(message, history[:-1], config):
                    history[-1][1] = response
                    
                    # Only clear input after first real response
                    if first_chunk and response and response != "⏳ Generating...":
                        yield history, ""
                        first_chunk = False
                    else:
                        yield history, original_msg if first_chunk else ""
            
            # Connect events
            msg.submit(
                handle_message,
                inputs=[msg, chatbot, temperature, max_tokens, thinking_budget, min_response, timeout_mins],
                outputs=[chatbot, msg],
                queue=True
            )
            
            submit_btn.click(
                handle_message,
                inputs=[msg, chatbot, temperature, max_tokens, thinking_budget, min_response, timeout_mins],
                outputs=[chatbot, msg],
                queue=True
            )
            
            clear_btn.click(
                lambda: ([], ""),
                outputs=[chatbot, msg]
            )
            
            # Footer
            gr.Markdown("""
            ---
            💡 **Tips:**
            - The model uses a "thinking" phase before responding
            - **Improved**: Thinking budget is applied safely and clamped to leave room for the answer
            - **For Code Generation**: System auto-adjusts to min 800 response tokens & 8000+ max tokens
            - The thinking phase typically uses ~1500 tokens, response uses remaining tokens
            - Recommended settings:
              - Text generation: Max Tokens = 4000-6000, Min Response = 500-800
              - Code/HTML generation: Max Tokens = 8000-10000, Min Response = 800-2000
            - Lower repetition penalty (1.02) for better code patterns
            - Adjust Temperature (0.3-0.5 for precise code, 0.7-1.0 for creative text)
            - Clear the chat if responses seem stuck or repetitive
            """)
        
        return interface
    
    def _get_custom_css(self) -> str:
        """Get custom CSS for the interface."""
        return """
        .chatbot {
            font-family: 'SF Pro Text', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        .message-input {
            font-size: 16px;
        }
        .send-button {
            height: 80px;
            font-size: 18px;
            font-weight: 600;
        }
        """
    
    def launch(self, share: bool = False, port: int = 7860):
        """Launch the Gradio interface."""
        interface = self.create_interface()
        
        print("\n" + "=" * 60)
        print("🎉 Launching Chat Interface v2")
        print("=" * 60)
        
        interface.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=share,
            show_error=True
        )


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Seed-OSS Chat Interface v2")
    parser.add_argument(
        "--model-path",
        type=str,
        default="./Seed-OSS-36B-Instruct",
        help="Path to model (local or HuggingFace ID)"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Device to use"
    )
    parser.add_argument(
        "--quantization",
        type=int,
        choices=[4, 8],
        help="Quantization bits (CUDA only)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for web interface"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public share link"
    )
    
    args = parser.parse_args()
    
    # Create config
    device_type = DeviceType[args.device.upper()]
    config = ModelConfig(device_type, args.quantization)
    
    # Create and launch app
    app = ChatApplication(args.model_path, config)
    app.launch(share=args.share, port=args.port)


if __name__ == "__main__":
    main()
