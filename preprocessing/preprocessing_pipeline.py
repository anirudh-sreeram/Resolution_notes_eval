import logging
from typing import List

import logging

# import constants
#from debug_response_details import DebugMetadataKey
from preprocessing.common import PreProcessedInput
from preprocessing.summarization.summarization_preprocessing import (
    ExportPrompt,
    ProcessTags,
    RemoveCreatedDate,
    RemoveHtmlTags,
    RemoveLinks,
    RemovePreviousSummary,
    ReplaceInstructions,
    ReplacePreviousModelTags,
    TruncateInput,
    RemoveChatSpecificBoilerTemplateCode,
)
from transformers import AutoTokenizer


"""This is a utility file that holds methods that can be considered for re-use.
"""
import re

from transformers import AutoTokenizer


def calculate_token(text: str, tokenizer: AutoTokenizer) -> int:
    """Retrieve token size. Returns -1 if there is an error"""
    try:
        tokens = tokenizer(text, return_tensors="pt")
        token_count = tokens["input_ids"].shape[1]
    except Exception as exp:
        token_count = -1
    return token_count


def word_replacement(text_to: str, text_from: str, generated_output: str) -> str:
    """Replaces a string with another string. This is not case-sensitive
    For example: text_to = r'incident', text_from = r'case(s)'
    generated_output = "my name is case value and case intuit. Casey Case-Maker Data provided\n********cases********"
    replaced_output = "my name is incident value and incident intuit. Casey incident-Maker Data provided\n********incident********"
    """
    regex_str_pre = r"(?<![a-zA-Z])"
    regex_str_post = r"?(?![a-zA-Z])"
    replacement_regex_str = regex_str_pre + text_from + regex_str_post
    replaced_output = re.sub(replacement_regex_str, text_to, generated_output, flags=re.IGNORECASE)
    return replaced_output


# def get_glide_bool(input_metadata: dict, key: str, default) -> bool:
#     return str(input_metadata.get(key, default)).lower() == str(True).lower()

class PreProcessingPipeline:
    preprocessors = {
        "prompt": ExportPrompt(),
        "replace_previous_model_tags": ReplacePreviousModelTags(),
        "remove_previous_summary": RemovePreviousSummary(),
        "remove_created_date": RemoveCreatedDate(),
        "html_tags": RemoveHtmlTags(),
        "remove_links": RemoveLinks(),
        "truncate_input": TruncateInput(),
        "replace_instructions": ReplaceInstructions(),
        "process_tags": ProcessTags(),
        "remove_chat_specific_boiler_template_code": RemoveChatSpecificBoilerTemplateCode()
    }

    def preprocess(self, input: str, tokenizer: AutoTokenizer, options: dict) -> PreProcessedInput:
        LLM_DEBUG_ENABLED = "enable_debug"
        LLM_DEBUG_ENABLED_DEFAULT = False
        LLM_SUMMARIZATION_PRE_PROCESSING_STEPS = "Record Resolution"

        input = PreProcessedInput(input)#), get_glide_bool(options, LLM_DEBUG_ENABLED, LLM_DEBUG_ENABLED_DEFAULT))
        #input.add_metadata(LLM_SUMMARIZATION_OPTIONS_TENSOR_NAME, options)
        steps = options[LLM_SUMMARIZATION_PRE_PROCESSING_STEPS]

        # apply steps one by one
        for step in steps:
            if step not in PreProcessingPipeline.preprocessors:
                continue
            try:
                temp_pred_metadata_holder = {"name": step}

                input_token_count = self.get_token_count_metric(input, tokenizer)
                temp_pred_metadata_holder["input_token_count"] = input_token_count
                input = PreProcessingPipeline.preprocessors[step].preprocess(input, tokenizer, temp_pred_metadata_holder)
                output_token_count = self.get_token_count_metric(input, tokenizer)
                temp_pred_metadata_holder["output_token_count"] = output_token_count

            except Exception as exp:
                pass

        return input

    def get_token_count_metric(self, input: PreProcessedInput, tokenizer: AutoTokenizer) -> int:
        """This method will add token count before and after pre-processing (to metadata)"""
        token_count = calculate_token(input.preprocessed_input, tokenizer)
        return token_count


# import logging
# from typing import List
# from transformers import AutoTokenizer
# from preprocessing.summarization.summarization_preprocessing import (
#     RemovePreviousSummary,
#     ExportPrompt,
#     RemoveHtmlTags,
#     RemoveLinks,
#     TruncateInput,
# )
# from preprocessing.common import PreProcessedInput

# _logger = logging.getLogger(__name__)

# class PreProcessingPipeline:
#     preprocessors = {
#         "prompt": ExportPrompt(),
#         "remove_previous_summary": RemovePreviousSummary(),
#         "html_tags": RemoveHtmlTags(),
#         "remove_links": RemoveLinks(),
#         "truncate_input": TruncateInput(),
#     }
#     default_steps = ["prompt", "remove_links", "html_tags", "remove_previous_summary", "truncate_input"]

#     def preprocess(
#         self, input: str, tokenizer: AutoTokenizer, steps: List[str]
#     ) -> PreProcessedInput:
#         input = PreProcessedInput(input)
#         # apply steps one by one
#         for step in steps:
#             if step not in PreProcessingPipeline.preprocessors:
#                 continue
#             try:
#                 input = PreProcessingPipeline.preprocessors[step].preprocess(
#                     input, tokenizer
#                 )
#             except Exception as exp:
#                 _logger.error(f"Error in preprocessing step {step}: {exp}")
#         return input
