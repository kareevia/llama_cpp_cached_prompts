from __future__ import annotations
from typing import * # type: ignore

import sys
import pyjson5

class StdIOJsonInterface:
    def __init__(self):
        pass
        
    def get_command(self) -> dict[str, Any] | None:
        input_str = sys.stdin.buffer.readline()

        if len(input_str) == 0:
            return None

        try:
            command: dict[str, Any] = pyjson5.decode_buffer(input_str, wordlength= 0)

            if not(
                isinstance(command, dict) and  # type: ignore
                "command" in command  and
                "arguments" in command and
                isinstance(command["command"], str)
            ):
                raise RuntimeError("Expected JSON of format `{command: \"<command>\", "+
                    "arguments: <arguments>}` as input")
            
            return command
            
        except Exception as err:
            
            self.send_response(
                "SystemError",
                None,
                "Exception: {}\n\n{}".format(type(err), err),
            )

            raise err


    def raise_exception(self, message: str) -> NoReturn:
        self.send_response(
            "SystemError",
            None,
            message,
        )

        raise Exception(f"As result of input command: {message}")


    def send_response(self, status: str, value: Any, message: str):

        response_json = {
            "status": status,
            "value": value,
            "message": message
        }

        self.message_out(response_json)

    
    def message_out(self, data: List[Any] | Dict[str, Any]):
        byt: bytes = pyjson5.encode_bytes(data, tojson= None, mappingtypes= None) # type: ignore
        sys.stdout.buffer.write(byt)
        sys.stdout.buffer.write(b"\n")
        sys.stdout.flush()
    
