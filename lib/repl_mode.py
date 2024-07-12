from __future__ import annotations
from typing import * # type: ignore

from . import stdio_json_interface as SJI
from . import repositories_manager as ReM
from . import llama_cpp_cached_preludes as LCP

OK = "Ok"
ERROR = "SystemError"

class REPLMode:
    def __init__(self, rep_man: ReM.RepositoriesManager, lccp: LCP.LlamaCachedPreludes):
        self.stdio_inter = SJI.StdIOJsonInterface()
        self.rep_man = rep_man
        self.lccp = lccp

    def loop(self):
        while True:
            cmd = self.stdio_inter.get_command()

            if cmd is None:
                return

            args = cmd["arguments"]

            if cmd["command"] == "single_prompt":

                if not(
                    "prelude_id" in args  and  
                    "prelude" in args  and  
                    "prompt" in args
                ):
                    self.stdio_inter.raise_exception(f"Wrong arguments for the command.")

                number_of_outputs = args.get("number_of_random_outputs", 1)
                add_nonrandom_output_to_front = args.get("add_nonrandom_output_to_front", False)
                inf_params: dict[Any, Any] = args.get("inference_parameters", {})
                
                load_state = True
                res: list[str] = []

                if add_nonrandom_output_to_front:
                    
                    x = self.lccp.generate_and_fetch_to_string(
                        (args["prelude_id"], args["prelude"]),
                        args["prompt"],
                        { **inf_params, "temp": 0.0 },
                        load_state= load_state,
                    )

                    load_state = False
                    res.append(x)

                for _i in range(number_of_outputs):
                    
                    x = self.lccp.generate_and_fetch_to_string(
                        (args["prelude_id"], args["prelude"]),
                        args["prompt"],
                        inf_params,
                        load_state= load_state,
                    )

                    load_state = False
                    res.append(x)

                self.stdio_inter.send_response(OK, res, "")
            
            else:
                self.stdio_inter.raise_exception(f"Command `{cmd['command']}` is unknown")
        
