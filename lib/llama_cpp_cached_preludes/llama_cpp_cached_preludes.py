from __future__ import annotations

import os
import sys
# import pickle
import time
import json
import ctypes
from typing import * # type: ignore


import llama_cpp as LlC
import llama_cpp.llama_cpp as LlCLib
import numpy as np




class LlamaCachedPreludes:
    
    def __init__(
        self, 
        llama: LlC.Llama,
        cache_dir: str,
        verbose: bool = True,
    ) -> None:
        self.llama = llama
        self.llama.set_cache(None)
        self.cache_dir = cache_dir
        self.verbose = verbose
        
    
    def precache_prelude(self,
        prelude_id_and_tokens: Tuple[str, List[int] | str],
    ) -> bool:
        prelude_id, tokens = prelude_id_and_tokens

        time_start = 0.

        cache_path = os.path.join(self.cache_dir, prelude_id)
        tokens = self.convert_to_tokens(tokens, True)

        if not self.should_update_cache(cache_path, tokens):
            print(f"Cache \"{prelude_id}\" seems to be up to date.", file=sys.stderr)
            return False

        if self.verbose:
            time_start = time.time()
        
        self.llama.reset()
        self.llama.eval(tokens)

        if self.verbose:
            time_passed = time.time() - time_start
            
            print(f"Evaluated the prompt in {time_passed:.3f}s, {time_passed / len(tokens):.3f} " +
                f"s/t.", file=sys.stderr)
            
            time_start = time.time()

        self.write_state_to_disk(tokens, cache_path)

        if self.verbose:

            print(f"Saved the state into \"{prelude_id}\" in " +
                f"{time.time() - time_start:.3f}s.", file=sys.stderr)
            
        return True


    def precache_prelude_by_appending_to(self,
        prelude_id_and_tokens: Tuple[str, List[int] | str],
        parent_prelude_id_and_tokens: Tuple[str, List[int] | str],
    ) -> bool:
        prelude_id, tokens = prelude_id_and_tokens
        parent_prelude_id, parent_prelude_tokens = parent_prelude_id_and_tokens

        time_start = 0.

        tokens = self.convert_to_tokens(tokens, True)
        parent_prelude_tokens = self.convert_to_tokens(parent_prelude_tokens, True)
        cache_path = os.path.join(self.cache_dir, prelude_id)

        if len(tokens) < len(parent_prelude_tokens)  or\
                tokens[:len(parent_prelude_tokens)] != parent_prelude_tokens:
            raise Exception("Parent prelude is not the heading of the new prelude")
                
        if not self.should_update_cache(cache_path, tokens):
            print(f"Cache \"{prelude_id}\" seems to be up to date.", file=sys.stderr)
            return False
        
        if self.verbose:
            time_start = time.time()

        self.precache_prelude(parent_prelude_id_and_tokens)

        parent_cache_path = os.path.join(self.cache_dir, parent_prelude_id)
        self.read_state_from_disk(parent_cache_path)

        if self.verbose:
            
            print(f"Restored the state for new prelude \"{prelude_id}\" from "+
                f"\"{parent_prelude_id}\" in " +
                f"{time.time() - time_start:.3f}s.", file=sys.stderr)        

        if self.verbose:
            time_start = time.time()
        
        self.llama.eval(tokens[len(parent_prelude_tokens):])
        
        if self.verbose:
            time_passed = time.time() - time_start
            
            print(f"Evaluated the prompt in {time_passed:.3f}s, {time_passed / len(tokens):.3f} " +
                f"s/t.", file=sys.stderr)
            
            time_start = time.time()

        self.write_state_to_disk(tokens, cache_path)

        if self.verbose:

            print(f"Saved the state into \"{prelude_id}\" in " +
                f"{time.time() - time_start:.3f}s.", file=sys.stderr)
            
        return True
    

    def to_tokens(self, content: str, prepend_bos: bool = False) -> List[int]:
        return self.llama.tokenize(content.encode("utf-8"), add_bos=prepend_bos)
    

    def convert_to_tokens(self, content: str | List[int], prepend_bos: bool = False) -> List[int]:

        if isinstance(content, str):
            return self.to_tokens(content, prepend_bos)
        else:
            return content


    def generate(self, 
        prelude_id_and_tokens: Tuple[str, List[int] | str],
        new_tokens: List[int] | str, 
        inference_parameters: dict[Any, Any] = dict(),
        new_tokens_appended_to_prelude: bool = True,
        load_state: bool = True,
    ) -> Generator[int, Sequence[int] | None, None]:
        time_start = 0.

        prelude_id, prelude_tokens = prelude_id_and_tokens
        prelude_tokens = self.convert_to_tokens(prelude_tokens, True)

        if new_tokens_appended_to_prelude:
            new_tokens = self.convert_to_tokens(new_tokens)
            tokens = prelude_tokens + new_tokens
        else:
            new_tokens = self.convert_to_tokens(new_tokens, True)
            tokens = new_tokens

        self.precache_prelude(prelude_id_and_tokens)

        if load_state:        
            if self.verbose:
                time_start = time.time()

            cache_path = os.path.join(self.cache_dir, prelude_id)
            self.read_state_from_disk(cache_path)

            if self.verbose:
                
                print(f"Restored the state from \"{prelude_id}\" in " +
                    f"{time.time() - time_start:.3f}s.", file=sys.stderr)
                
        assert(not self.llama.ctx is None)
        LlCLib.llama_set_rng_seed(self.llama.ctx, ctypes.c_int(-1)) # type: ignore

        return self.llama.generate(tokens, **inference_parameters)
    

    def generate_and_fetch_to_string(self, 
        prelude_id_and_tokens: Tuple[str, List[int] | str],
        new_tokens: List[int] | str, 
        inference_parameters: dict[Any, Any] = dict(),
        new_tokens_appended_to_prelude: bool = True,
        load_state: bool = True,
    ) -> str:
        gen = self.generate(prelude_id_and_tokens, new_tokens, inference_parameters, 
            new_tokens_appended_to_prelude, load_state)
        
        return self.fetch_generator_to_the_end(gen)


    def fetch_generator_to_the_end(self, gen: Generator[int, Sequence[int] | None, None]) -> str:
        time_start = 0.

        buf = ""
        eos_token = self.llama.token_eos()
        token_num = 0

        if self.verbose:
            time_start = time.time()
            

        for token in gen:
            token_str = self.llama.detokenize([token]).decode("utf-8", errors="ignore")
            
            if token == eos_token:
                break

            buf += token_str
            token_num += 1

            if self.verbose:
                print(token_str, end="", flush=True, file=sys.stderr)

        if self.verbose:
            time_elaps = time.time() - time_start
            
            print("\n\n", file=sys.stderr)
            print(f"Generated {token_num} tokens / {len(buf)} chars in  " +
                f"{time_elaps:.3f}s, {time_elaps / token_num:.3f} s/t.", file=sys.stderr)

        return buf
    

    def write_state_to_disk(self, tokens: List[int], path_and_base_filename: str):
        assert self.llama.ctx is not None
        session_path = f"{path_and_base_filename}.session"

        LlCLib.llama_save_session_file(
            ctx = self.llama.ctx,
            path_session = bytes(session_path.encode("utf-8")), # type: ignore
            tokens = (LlCLib.llama_token * len(tokens))(*tokens),
            n_token_count = ctypes.c_size_t(len(tokens)),    
        )

        np.save(f"{path_and_base_filename}.input_ids", self.llama.input_ids)
        np.save(f"{path_and_base_filename}.scores", self.llama.scores[0:self.llama.n_tokens,:])

        meta_part = dict(
            tokens = tokens,
            n_tokens = self.llama.n_tokens,
        )

        json.dump(meta_part, open(f"{path_and_base_filename}.cache_meta.json", "w"))
        
        
    def read_state_from_disk(self, path_and_base_filename: str):
        assert self.llama.ctx is not None
        session_path = f"{path_and_base_filename}.session"

        meta_part = json.load(open(f"{path_and_base_filename}.cache_meta.json", "r"))
        
        tokens_buf = (LlCLib.llama_token * meta_part["n_tokens"])()
        tokens_written = ctypes.c_size_t(0)
        pointer = ctypes.pointer(tokens_written)

        LlCLib.llama_load_session_file(
            ctx = self.llama.ctx,
            path_session = bytes(session_path.encode("utf-8")), # type: ignore
            tokens_out = tokens_buf,
            n_token_capacity = ctypes.c_size_t(meta_part["n_tokens"]),
            n_token_count_out = pointer,
        )

        self.llama.n_tokens = meta_part["n_tokens"]
        self.llama.input_ids = np.load(f"{path_and_base_filename}.input_ids.npy")
        buf = np.load(f"{path_and_base_filename}.scores.npy")
        self.llama.scores[0:self.llama.n_tokens,:] = buf
    

    def should_update_cache(self, path_and_base_filename: str, tokens: List[int]) -> bool:
        path = f"{path_and_base_filename}.cache_meta.json"
        
        if not os.path.isfile(path):
            return True
        
        with open(path, "r") as f:
            meta = json.load(f)
            return meta["tokens"] != tokens




    
    

