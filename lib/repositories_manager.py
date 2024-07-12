from __future__ import annotations
from typing import * # type: ignore

from . import llama_cpp_cached_preludes as LCP

import os
import sys
import re


class Repository:
    def __init__(self,
        repo_dir: str,
        preludes_from: str | None = None,
    ):
        self.repo_dir = repo_dir
        self.preludes_from = preludes_from

        # check id_from_file / file_from_id if want to change
        self.repo_files_filter: Callable[[str], bool] = lambda x: x.endswith(".txt")


class RepositoriesManager:
    def __init__(self, lccp: LCP.LlamaCachedPreludes, verbose: bool = True):
        self.repositories: dict[str, Repository] = dict()
        self.lccp = lccp
        self.verbose = verbose


    def add_repository(self, repo_id: str, repo: Repository, force_replacement: bool = False):
        
        if not force_replacement  and  repo_id in self.repositories:
            raise Exception(f"Repository with id \"{repo_id}\" already exists")

        self.repositories[repo_id] = repo


    def id_from_file(self, filename: str) -> str:
        
        if not filename.endswith(".txt"):
            raise Exception(f"File \"{filename}\" is expected to have extension `.txt`")
        
        return filename[:-4]
    

    def file_from_id(self, id: str) -> str:
        return f"{id}.txt"


    def generate_outputs_for_preludes(self, 
        repo_id: str, 
        outputs_dir: str,
        output_index_generator: Iterable[Any],
        inference_parameters: dict[Any, Any],
        force_regardless_of_mtime: bool = False,
    ):
        repo = self.repositories[repo_id]

        for elem in os.listdir(repo.repo_dir):
            path = os.path.join(repo.repo_dir, elem)

            if not( os.path.isfile(path)  and  repo.repo_files_filter(elem) ):
                continue

            prelude_id = f"{repo_id}.{self.id_from_file(elem)}"

            if self.verbose:
                print(f"Evaluating file \"{elem}\" in repository \"{repo_id}\"", file=sys.stderr)

            cont = open(path).read()
            prelude = (prelude_id, cont)                
            
            is_first = True

            for index in output_index_generator:
                
                output_path = os.path.join(outputs_dir, 
                    f"{self.id_from_file(elem)}.output-{index}.txt")

                if not force_regardless_of_mtime  and  os.path.isfile(output_path)  and  \
                        os.path.getmtime(output_path) >= os.path.getmtime(path):
                    
                    if self.verbose:

                        print(f"Output file \"{output_path}\" for prelude \"{prelude_id}\" is "+
                            f"new enough, skipping", file=sys.stderr)
                    
                    continue
                
                if self.verbose:
                    
                    print(f"Generating output file \"{output_path}\" for prelude \"{prelude_id}\"", 
                        file=sys.stderr)
                
                output_cont = self.lccp.generate_and_fetch_to_string(prelude, "",
                    inference_parameters, load_state=is_first)
                
                is_first = False                
                open(output_path, "w").write(output_cont)


    def precache(self, repo_id: str):
            repo = self.repositories[repo_id]
            
            for elem in os.listdir(repo.repo_dir):
                path = os.path.join(repo.repo_dir, elem)
                prelude_id = f"{repo_id}.{self.id_from_file(elem)}"
                
                if not( os.path.isfile(path)  and  repo.repo_files_filter(elem) ):
                    continue

                if self.verbose:

                    print(f"Evaluating file \"{elem}\" in repository \"{repo_id}\", precaching "+
                        f"as prelude \"{prelude_id}\"", file=sys.stderr)
                
                cont = open(path).read()
                prelude = (prelude_id, cont)
                self.lccp.precache_prelude(prelude)


    def generate(self, prompt: str, repo_id: str, prelude_id: str, 
            inference_parameters: dict[Any, Any] = dict(), is_first: bool = True) -> str:
        repo = self.repositories[repo_id]
        prel_path = os.path.join(repo.repo_dir, self.file_from_id(prelude_id))
        prel_text = open(prel_path).read()
        prelude = (f"{repo_id}.{prelude_id}", prel_text)
        return self.lccp.generate_and_fetch_to_string(prelude, prompt, inference_parameters,
            load_state=is_first)

    # Seems to be an exact copy of generate_outputs_for_preludes ??? How has that happened ???
    #    
    # def generate_outputs_from_preludes(self, 
    #     repo_id: str, 
    #     outputs_dir: str,
    #     output_index_generator: Iterable[Any],
    #     inference_parameters: dict[Any, Any],
    #     force_regardless_of_mtime: bool = False,
    # ):
    #     repo = self.repositories[repo_id]
        
    #     for elem in os.listdir(repo.repo_dir):
    #         path = os.path.join(repo.repo_dir, elem)
    #         prelude_id = f"{repo_id}.{self.id_from_file(elem)}"
            
    #         if not( os.path.isfile(path)  and  repo.repo_files_filter(elem) ):
    #             continue

    #         if self.verbose:
    #             print(f"Evaluating file \"{elem}\" in repository \"{repo_id}\"", file=sys.stderr)

    #         cont = open(path).read()
    #         prelude = (prelude_id, cont)                
            
    #         for index in output_index_generator:
                
    #             output_path = os.path.join(outputs_dir, 
    #                 f"{self.file_from_id(elem)}.output-{index}.txt")

    #             if not force_regardless_of_mtime  and  os.path.isfile(output_path)  and  \
    #                     os.path.getmtime(output_path) >= os.path.getmtime(path):
                    
    #                 if self.verbose:

    #                     print(f"Output file \"{output_path}\" for prelude \"{prelude_id}\" is "+
    #                         f"new enough, skipping", file=sys.stderr)
                    
    #                 continue
                
    #             if self.verbose:
                    
    #                 print(f"Generating output file \"{output_path}\" for prelude \"{prelude_id}\"", 
    #                     file=sys.stderr)
                
    #             output_cont = self.lccp.generate_and_fetch_to_string(prelude, "",
    #                 inference_parameters)
                
    #             open(output_path, "w").write(output_cont)


    def generate_outputs_as_from_prompts(self, 
        repo_id: str, 
        outputs_dir: str,
        output_index_generator: Iterable[Any],
        inference_parameters: dict[Any, Any],
        force_regardless_of_mtime: bool = False,
    ):
        """
        Input filename format: `{prelude_filename_from_prelude_repository}---{prompt_id}.txt`
        """

        repo = self.repositories[repo_id]
        
        if repo.preludes_from is None:

            raise Exception(f"Trying to generate outputs as from prompts for repository "+
                f"\"{repo_id}\", but the repository of preludes (`preludes_from`) is `None`")

        repo_prel = self.repositories[repo.preludes_from]

        for elem in os.listdir(repo.repo_dir):
            path = os.path.join(repo.repo_dir, elem)
            
            if not( os.path.isfile(path)  and  repo.repo_files_filter(elem) ):
                continue

            if self.verbose:
                print(f"Evaluating file \"{elem}\" in repository \"{repo_id}\"", file=sys.stderr)


            matches = re.match(r"^(.*)---.*$", elem)
            
            if matches is None:

                if self.verbose:

                    print(f"Cannot find the prelude name component in filename \"{elem}\" --- "+
                        "skipping", file=sys.stderr)

                continue

            prelude_id = matches[1]
            prelude_path = os.path.join(repo_prel.repo_dir, self.file_from_id(prelude_id))
            prelude_id = f"{repo.preludes_from}.{prelude_id}"

            prelude_text = open(prelude_path).read()
            prelude = (prelude_id, prelude_text)
            prompt_text = open(path).read()

            prelude_mtime = os.path.getmtime(prelude_path)
            prompt_mtime = os.path.getmtime(path)
            
            is_first = True

            for index in output_index_generator:
                
                output_path = os.path.join(outputs_dir, 
                    f"{self.id_from_file(elem)}.output-{index}.txt")
                
                if not force_regardless_of_mtime  and  os.path.isfile(output_path):
                    mtime = os.path.getmtime(output_path) 

                    if mtime >= prompt_mtime  and  mtime >= prelude_mtime:

                        if self.verbose:

                            print(f"Output file \"{output_path}\" is "+
                                f"new enough, skipping", file=sys.stderr)
                            
                        continue
                
                output_text = self.lccp.generate_and_fetch_to_string(prelude, prompt_text, 
                    inference_parameters, load_state=is_first)
                
                is_first = False                
                open(output_path, "w").write(output_text)


    def show_stats_on_tokens_number(self, 
        repo_id: str, 
        outputs_dir: str | None,
    ):

        repo = self.repositories[repo_id]
        
        for elem in os.listdir(repo.repo_dir):
            path = os.path.join(repo.repo_dir, elem)
            
            if not( os.path.isfile(path)  and  repo.repo_files_filter(elem) ):
                continue

            file_text = open(path).read()
            file_tokens = len(self.lccp.to_tokens(file_text, repo.preludes_from is None))
            
            print(f"\"{path}\" tokens: {file_tokens}")
            
            prelude_tokens: int | None = None

            if not repo.preludes_from is None:
                matches = re.match(r"^(.*)---.*$", elem)
            
                if not matches is None:
                    prelude_id = matches[1]
                    repo_prel = self.repositories[repo.preludes_from]
                    prelude_path = os.path.join(repo_prel.repo_dir, self.file_from_id(prelude_id))

                    if os.path.isfile(prelude_path):
                        prelude_text = open(prelude_path).read()
                        prelude_tokens = len(self.lccp.to_tokens(prelude_text, True))

                        print(f"- with its prelude \"{prelude_path}\": {prelude_tokens} + "+
                            f"{file_tokens} = {prelude_tokens + file_tokens}")
                    else:
                        print(f"- its prelude \"{prelude_path}\" is absent")

            if not outputs_dir is None:
                
                for out_elem in os.listdir(outputs_dir):
                    out_path = os.path.join(outputs_dir, out_elem)
                    
                    if not out_elem.startswith( f"{self.id_from_file(elem)}." ):
                        continue

                    out_text = open(out_path).read()
                    out_tokens = len(self.lccp.to_tokens(out_text, False))

                    if repo.preludes_from is None:
                    
                        print(f"* with its output \"{out_path}\": {out_tokens} + {file_tokens} = "+
                            f"{file_tokens + out_tokens}")
                    
                    else:

                        if prelude_tokens is None:

                            print(f"  * with its output \"{out_path}\": {out_tokens} + ??? + "+
                                f"{file_tokens} = {file_tokens + out_tokens} + ???")
                            
                        else:

                            print(f"  * with its output \"{out_path}\": {out_tokens} + "+
                                f"{prelude_tokens} + {file_tokens} = "+
                                f"{prelude_tokens + file_tokens + out_tokens}")
                            
            print()




