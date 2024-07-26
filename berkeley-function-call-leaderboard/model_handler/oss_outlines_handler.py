import json
import os

from model_handler.constant import GORILLA_TO_OPENAPI
from model_handler.handler import BaseHandler
from model_handler.model_style import ModelStyle
from model_handler.utils import (
    ast_parse,
    augment_prompt_by_languge,
    language_specific_pre_processing,
)

def _cast_to_openai_type(properties, mapping, test_category):
    for key, value in properties.items():
        if "type" not in value:
            properties[key]["type"] = "string"
        else:
            var_type = value["type"]
            if mapping == GORILLA_TO_OPENAPI and var_type == "float":
                properties[key]["format"] = "float"
                properties[key]["description"] += " This is a float type value."
            if var_type in mapping:
                properties[key]["type"] = mapping[var_type]
            else:
                properties[key]["type"] = "string"

        # Currently support:
        # - list of any
        # - list of list of any
        # - list of dict
        # - list of list of dict
        # - dict of any

        if properties[key]["type"] == "array" or properties[key]["type"] == "object":
            if "properties" in properties[key]:
                properties[key]["properties"] = _cast_to_openai_type(
                    properties[key]["properties"], mapping, test_category
                )
            elif "items" in properties[key]:
                properties[key]["items"]["type"] = mapping[
                    properties[key]["items"]["type"]
                ]
                if (
                    properties[key]["items"]["type"] == "array"
                    and "items" in properties[key]["items"]
                ):
                    properties[key]["items"]["items"]["type"] = mapping[
                        properties[key]["items"]["items"]["type"]
                    ]
                elif (
                    properties[key]["items"]["type"] == "object"
                    and "properties" in properties[key]["items"]
                ):
                    properties[key]["items"]["properties"] = _cast_to_openai_type(
                        properties[key]["items"]["properties"], mapping, test_category
                    )
    return properties

# allow `\"`, `\\`, or any character which isn't a control sequence
STRING_INNER = r'([^"\\\x00-\x1F\x7F-\x9F]|\\["\\])'
STRING = f'"{STRING_INNER}*"'

INTEGER = r"(-)?(0|[1-9][0-9]*)"
# NUMBER = rf"({INTEGER})(\.[0-9]+)?([eE][+-][0-9]+)?"
NUMBER = rf"({INTEGER})(\.[0-9]+)([eE][+-][0-9]+)?"
BOOLEAN = r"(true|false)"
NULL = r"null"
WHITESPACE = r"[ ]?"

# _type_to_regex = {
#     "string": STRING,
#     "integer": INTEGER,
#     "number": NUMBER,
#     "boolean": BOOLEAN,
#     "null": NULL,
# }

def build_dict_regex(props):
    out_re = r'\{'
    args_part  = ", ".join([
        f'"{prop}": '+type_to_regex(props[prop])
        for prop in props
    ])
    return out_re+args_part+r"\}"

# this should be more universal
def type_to_regex(arg_meta):
    basic_map = {
        "string": f'"{STRING_INNER}{{1,55}}"',  # might need to be longer?
        "integer": INTEGER,
        "number": NUMBER,
        "float": NUMBER,  # change this later
        "boolean": BOOLEAN,
        "null": NULL,
    }
    arg_type = arg_meta["type"]
    if arg_type == "object":
        arg_type = "dict"
    if arg_type == "dict":
        if "properties" in arg_meta.keys():
            result = build_dict_regex(arg_meta["properties"])
        else:
            result = r"""\[("([^"\\\x00-\x1F\x7F-\x9F]|\\["\\]){1,55}", ){0,8}"([^"\\\x00-\x1F\x7F-\x9F]|\\["\\]){1,55}"\]"""
    # Note, this currently won't pass the empty list
    elif arg_type == "array":
        pattern = type_to_regex(arg_meta["items"])
        result = array_regex = r"\[(" + pattern + ", ){0,8}" + pattern + "\]"
    else:
        result = basic_map[arg_type]
    return result

def build_fc_regex(function_data):
    out_re = r'\{"name": "' + function_data["name"] + '", "arguments": \{'
    args_part = ", ".join(
        [
            f'"{arg}": ' + type_to_regex(value)
            for arg, value in _cast_to_openai_type(
                function_data["parameters"]["properties"], GORILLA_TO_OPENAPI, "simple"
            ).items()
            # do any of the examples use the option parameter?
            # Easy enough to add in!
            if arg in function_data["parameters"]["required"]
        ]
    )
    optional_part = "".join(
        [
            f'(, "{arg}": ' + type_to_regex(value) + r")?"
            for arg, value in _cast_to_openai_type(
                function_data["parameters"]["properties"], GORILLA_TO_OPENAPI, "simple"
            ).items()
            if not (arg in function_data["parameters"]["required"])
        ]
    )
    fc_regex = out_re + args_part + optional_part + "}}"
    return fc_regex
    # return out_re + args_part + "}}"



class OSSOutlinesHandler(BaseHandler):
    def __init__(self, model_name, temperature=0.7, top_p=1, max_tokens=1000, dtype="float16") -> None:
        super().__init__(model_name, temperature, top_p, max_tokens)
        self.model_style = ModelStyle.OSSMODEL
        self.dtype = dtype

    def _format_prompt(prompt, function, test_category):
        SYSTEM_PROMPT = """
            You are an helpful assistant who has access to the following functions to help the user, you can use the functions if needed-
        """
        functions = ""
        if isinstance(function, list):
            for idx, func in enumerate(function):
                functions += "\n" + str(func)
        else:
            functions += "\n" + str(function)
        return f"SYSTEM: {SYSTEM_PROMPT}\n{functions}\nUSER: {prompt}\nASSISTANT: "

    @staticmethod
    def _batch_generate(
        test_question,
        fc_regexes,
        model_path,
        temperature,
        max_tokens,
        top_p,
        dtype,
        stop_token_ids=None,
        max_model_len=None,
        num_gpus=8,
        gpu_memory_utilization=0.9,
    ):
        import vllm
        from vllm import LLM, SamplingParams
        from outlines import models, generate

        print("start generating, test question length: ", len(test_question))

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop_token_ids=stop_token_ids,
        )
        llm = vllm.LLM(
            model=model_path,
            dtype=dtype,
            trust_remote_code=True,
            disable_custom_all_reduce=True,
            max_model_len=max_model_len,
            tensor_parallel_size=num_gpus,
            gpu_memory_utilization=gpu_memory_utilization
        )
        model = models.VLLM(llm)
        final_ans_jsons = []
        for i, question in enumerate(test_question[:3]):
            for attempt in range(5):
                try:
                    fc_regex = fc_regexes[i]
                    generator = generate.regex(model, fc_regex)
                    outputs = generator(prompts=question,sampling_params=sampling_params)
                except:
                    continue
                else:
                    break
            else:
                outputs = "Error"
            final_ans_jsons.append(outputs)
        return final_ans_jsons

    @staticmethod
    def process_input(test_question, format_prompt_func=_format_prompt):
        prompts = []
        fc_regexes = []
        for question in test_question:
            test_category = question["id"].rsplit("_", 1)[0]
            prompt = augment_prompt_by_languge(question["question"], test_category)
            functions = language_specific_pre_processing(
                question["function"], test_category
            )
            prompts.append(format_prompt_func(prompt, functions, test_category))
            fc_regex = build_fc_regex(question["function"])
            fc_regexes.append(fc_regex)

        return prompts, fc_regexes

    def inference(
        self,
        test_question,
        num_gpus,
        gpu_memory_utilization,
        format_prompt_func=_format_prompt,
        stop_token_ids=None,
        max_model_len=None,
    ):
        test_question, fc_regexes = self.process_input(test_question, format_prompt_func)

        ans_jsons = self._batch_generate(
            test_question=test_question,
            fc_regexes=fc_regexes,
            model_path=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            dtype=self.dtype,
            stop_token_ids=stop_token_ids,
            max_model_len=max_model_len,
            num_gpus=num_gpus,
            gpu_memory_utilization=gpu_memory_utilization,
        )

        return ans_jsons, {"input_tokens": 0, "output_tokens": 0, "latency": 0}

    def decode_ast(self, result, language="Python"):
        func = result
        if " " == func[0]:
            func = func[1:]
        if not func.startswith("["):
            func = "[" + func
        if not func.endswith("]"):
            func = func + "]"
        decode_output = ast_parse(func, language)
        return decode_output

    def decode_execute(self, result):
        return result
