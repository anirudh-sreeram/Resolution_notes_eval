import json
import re

import bs4
from typing import List
import preprocessing.constants as constants
from preprocessing.capability_definition import CapabilityOptions
from preprocessing.common import LLM_LINKS, LLM_SUMMARIZATION_MAX_TOKENS, LLM_SUMMARIZATION_PROMPT, PreProcessedInput, PreProcessStep
from transformers import AutoTokenizer


class RemoveLinks(PreProcessStep):
    """Remove links from the input"""

    http_regex = "https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)"

    def preprocess(self, input: PreProcessedInput, tokenizer: AutoTokenizer, temp_pred_metadata_holder: dict) -> PreProcessedInput:
        # check number of tokens
        tokens = tokenizer(
            input.preprocessed_input,
            return_tensors="pt",
        )
        if tokens["input_ids"].shape[1] < LLM_SUMMARIZATION_MAX_TOKENS:
            temp_pred_metadata_holder["links_removed"] = str(False).lower()
            return input
        # replace links with dummy links
        link_dict = {}
        all_links = re.findall(RemoveLinks.http_regex, input.preprocessed_input)
        counter = 0
        for link in all_links:
            if link in link_dict:
                continue
            link_dict[link] = "https://link" + str(counter) + ".servicenow.com"
            counter += 1
        # replace links
        for link in link_dict:
            input.preprocessed_input = input.preprocessed_input.replace(link, link_dict[link])
        temp_pred_metadata_holder["links_removed"] = str(True).lower()
        input.add_metadata(LLM_LINKS, link_dict)
        return input


class RemoveHtmlTags(PreProcessStep):
    """Remove HTML tags from the input"""

    def preprocess(self, input: PreProcessedInput, tokenizer: AutoTokenizer, temp_pred_metadata_holder: dict) -> PreProcessedInput:
        # Create a BeautifulSoup object
        soup = bs4.BeautifulSoup(input.preprocessed_input, "html.parser")

        # Iterate over the BeautifulSoup object and remove the tags
        for tag in soup.find_all("*"):
            tag.decompose()
        # Print the text content of the BeautifulSoup object
        input.preprocessed_input = soup.text
        return input


class ReplacePreviousModelTags(PreProcessStep):
    """Replace the previous model tags with the current model tags"""

    def preprocess(self, input: PreProcessedInput, tokenizer: AutoTokenizer, temp_pred_metadata_holder: dict) -> PreProcessedInput:
        input.preprocessed_input = (
            input.preprocessed_input.replace("<|system|>", "<|system|>\n")
            .replace("<|endoftext|><|customer|>", "<|end|>\n<|user|>\n")
            .replace("<|endoftext|><|agent|>", "<|end|>\n<|assistant|>")
            .replace("<|endoftext|>", "<|end|>")
        )
        return input


class RemovePreviousSummary(PreProcessStep):
    """Remove the previous summary from the input"""

    default_regex_str = (
        "[\n\s]*?[*]+\sGenerated by NowLLM\s[*]+[\n\s]*?[\d\D]+?[\n\s]*?[\n\s]*?[*]+\sGenerated by NowLLM\s[*]+[\n\s]*?"  # noqa: W605
    )

    def preprocess(self, input: PreProcessedInput, tokenizer: AutoTokenizer, temp_pred_metadata_holder: dict) -> PreProcessedInput:
        input.preprocessed_input = re.sub(RemovePreviousSummary.default_regex_str, "", input.preprocessed_input)
        custom_watermark = input.metadata[constants.LLM_SUMMARIZATION_OPTIONS_TENSOR_NAME].get(constants.LLM_SUMMARIZATION_WATERMARK)
        if custom_watermark is not None:
            custom_pattern = (
                r"[\n\s]*?" + re.escape(custom_watermark) + r"[\n\s]*?[\d\D]+?[\n\s]*?[\n\s]*?" + re.escape(custom_watermark) + r"[\n\s]*?"
            )
            input.preprocessed_input = re.sub(custom_pattern, "", input.preprocessed_input)
        return input


class RemoveCreatedDate(PreProcessStep):
    """Remove the created date from the input like Creation date: 2023-06-16 14:21:31"""

    regex_str = r"(?m)^Creation date: (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) "

    def preprocess(self, input: PreProcessedInput, tokenizer: AutoTokenizer, temp_pred_metadata_holder: dict) -> PreProcessedInput:
        input.preprocessed_input = re.sub(RemoveCreatedDate.regex_str, "", input.preprocessed_input)
        return input


class ExportPrompt(PreProcessStep):
    """Export the prompt from the input"""

    regex_str = "<\|end\|>\s*<\|user\|>([\D\d]*?)<\|end\|>\s*<\|assistant\|>"  # noqa: W605

    def preprocess(self, input: PreProcessedInput, tokenizer: AutoTokenizer, temp_pred_metadata_holder: dict) -> PreProcessedInput:
        prompt = re.search(ExportPrompt.regex_str, input.preprocessed_input).group(1)
        input.add_metadata(LLM_SUMMARIZATION_PROMPT, prompt)
        return input


class TruncateInput(PreProcessStep):
    """Truncates the input Text token count to less than the max token count"""

    end_of_text_token = "<|end|>\n<|user|>"

    def single_input_total_token_size(self, preprocessed_input: str, tokenizer: AutoTokenizer) -> int:
        # Calculate token size of the input text
        token_counter = tokenizer(preprocessed_input, return_tensors="pt")
        return token_counter["input_ids"].shape[1]

    def extract_activity_stream_start_id(self, preprocessed_input: str, temp_pred_metadata_holder: dict) -> List[int]:
        """
        This method extracts the index of locations where comments and work notes tags are applied.
        2 regular expressions are used.
        v1 is based on the example below:
            comment by agent:
            work_notes by agent:
            comments by user_1:
            work_note by agent_abcde_1234:
        v0 is the pre-existing truncation format (default). The deault regular expression is used if there are no matches with v1
        :param preprocessed_input: input text
        :param temp_pred_metadata_holder: Metadata Holder
        :return: list of indexes for truncation
        """
        regex_pattern_v0 = "(comments:|work_notes:)"  # noqa: W605
        # Comments and work_notes starting with new line
        regex_pattern_v1 = r"(?m)(^(comments by [\w]+:)|^(work_notes by [\w]+:))"
        activity_stream_start_index_list = self.extract_pattern_start_index(regex_pattern_v1, preprocessed_input)
        temp_pred_metadata_holder["truncation_pattern_version"] = "truncation_regex_pattern_v1"
        if len(activity_stream_start_index_list) == 0:
            activity_stream_start_index_list = self.extract_pattern_start_index(regex_pattern_v0, preprocessed_input)
            temp_pred_metadata_holder["truncation_pattern_version"] = "truncation_regex_pattern_v0"
        return activity_stream_start_index_list

    def extract_pattern_start_index(self, regex_pattern: str, preprocessed_input: str) -> List[int]:
        activity_stream_start_index = []
        if regex_pattern:
            for match in re.finditer(regex_pattern, preprocessed_input):
                activity_stream_start_index.append(match.start())
        return activity_stream_start_index

    def truncate_input(self, input: PreProcessedInput, tokenizer: AutoTokenizer, temp_pred_metadata_holder: dict) -> PreProcessedInput:
        # Calculate total token size and return input if less than max
        input_text_token_count = self.single_input_total_token_size(input.preprocessed_input, tokenizer)
        input_max_token_option_value = int(
            input.metadata[constants.LLM_SUMMARIZATION_OPTIONS_TENSOR_NAME][constants.LLM_SUMMARIZATION_MAX_NEW_TOKENS]
        )
        max_token_value = LLM_SUMMARIZATION_MAX_TOKENS - min(input_max_token_option_value, 1000)
        if input_text_token_count < max_token_value:
            temp_pred_metadata_holder["truncation"] = str(False).lower()
            return input
        temp_pred_metadata_holder["truncation"] = str(True).lower()
        comments_start_index = self.extract_activity_stream_start_id(input.preprocessed_input, temp_pred_metadata_holder)
        comments_start_index_list_length = len(comments_start_index)
        iterator = 1
        processed_input = ""
        token_calc = input_text_token_count
        temp_pred_metadata_holder["all_worknotes_comments_truncated"] = str(False).lower()
        while token_calc > max_token_value:
            if iterator >= comments_start_index_list_length - 1:
                eot_token_index = input.preprocessed_input.find(TruncateInput.end_of_text_token)
                sub_string_input = input.preprocessed_input[comments_start_index[0] : eot_token_index]
                processed_input = input.preprocessed_input.replace(sub_string_input, "")
                temp_pred_metadata_holder["all_worknotes_comments_truncated"] = str(True).lower()
                break
            sub_string_input = input.preprocessed_input[comments_start_index[0] : comments_start_index[iterator]]
            processed_input = input.preprocessed_input.replace(sub_string_input, "")
            token_calc = self.single_input_total_token_size(processed_input, tokenizer)
            iterator += 1
        input.preprocessed_input = processed_input
        return input

    def preprocess(self, input: PreProcessedInput, tokenizer: AutoTokenizer, temp_pred_metadata_holder: dict) -> PreProcessedInput:
        input = self.truncate_input(input, tokenizer, temp_pred_metadata_holder)
        return input


class ReplaceInstructions(PreProcessStep):
    """
    Replace instruction framework
    This Framework allows for Instruction replacement based on the optional parameters or text properties such as token length
    """

    def get_state_from_sections(self, optional_parameters: dict) -> str:
        """
        This function is specifically for small text instruction replacement
        The InstructionBank in Constants will contain 3 different instructions, one for each 'state'
        State will be New, WIP or Resolved
        if 'resolution' is in the output sections list then the state will default to resolved,
        else if 'actions taken' is in the output sections list then the state will be WIP,
        else the state will be New.
        :param optional_parameters: Specifically using LLM_SUMMARIZATION_OUTPUT_SECTIONS_JSON_INPUT
        :return: state
        """
        sectional_format = optional_parameters[constants.LLM_SUMMARIZATION_OUTPUT_SECTIONS_JSON_INPUT]
        sections_received = [section.lower() for section in sectional_format[constants.LLM_SUMMARIZATION_OUTPUT_SECTIONS]]
        if constants.LLM_SUMMARIZATION_RESOLUTION in sections_received:
            state = constants.LLM_SUMMARIZATION_SMALL_CONTEXT_INSTRUCTION_DEFAULT
        elif constants.LLM_SUMMARIZATION_ACTIONS_TAKEN in sections_received:
            state = constants.LLM_SUMMARIZATION_SMALL_CONTEXT_INSTRUCTION_WIP
        else:
            if len(sections_received) > 0:
                state = constants.LLM_SUMMARIZATION_SMALL_CONTEXT_INSTRUCTION_NEW
            else:
                state = constants.LLM_SUMMARIZATION_SMALL_CONTEXT_INSTRUCTION_DEFAULT
        return state

    def low_word_count(self, input: PreProcessedInput, optional_parameters: dict, temp_pred_metadata_holder: dict) -> bool:
        """
        Calculate if the word count is low
        Word count is obtained by splitting the text with a 'space' delimiter
        The instruction is not included in the word count calculation
        :param input: Input data
        :param optional_parameters: specifically using a new optional paramater LLM_SUMMARIZATION_SMALL_INSTRUCTION_WORD_COUNT
        :return: True/False
        """
        small_instruction_word_limit = optional_parameters[constants.LLM_SUMMARIZATION_SMALL_CONTEXT_INSTRUCTION_WORD_COUNT]
        processed_text = input.preprocessed_input.replace(input.metadata[LLM_SUMMARIZATION_PROMPT], "")
        processed_text = processed_text.split(" ")
        word_count = len(processed_text)
        temp_pred_metadata_holder["word_count"] = word_count
        if word_count <= small_instruction_word_limit:
            return True
        return False

    def preprocess(self, input: PreProcessedInput, tokenizer: AutoTokenizer, temp_pred_metadata_holder: dict) -> PreProcessedInput:
        # Extract Optional Parameters
        optional_parameters = input.metadata[constants.LLM_SUMMARIZATION_OPTIONS_TENSOR_NAME]
        use_peft_key = optional_parameters[constants.LLM_SUMMARIZATION_USE_PEFT]
        capability = optional_parameters[constants.LLM_CAPABILITY]
        word_count_low = self.low_word_count(input, optional_parameters, temp_pred_metadata_holder)  # Calculate if the word count is low
        # check if it is record summarization
        # If token count for the preprocessed input is less than 2000 then replace ths instruction
        if capability == CapabilityOptions.RECORD_SUMMARIZATION.name:
            if word_count_low:
                # Extract State (WIP, NEW or RESOLVED)
                state = self.get_state_from_sections(optional_parameters)  # Retrieve state from Instructions
                input.preprocessed_input = re.sub(
                    input.metadata[LLM_SUMMARIZATION_PROMPT],
                    constants.InstructionReplacementBank[state],
                    input.preprocessed_input,
                )
                temp_pred_metadata_holder["small_context_instruction_replacement"] = str(True).lower()
                input.add_metadata(LLM_SUMMARIZATION_PROMPT, constants.InstructionReplacementBank[state])
        # Replace instructions for PEFT
        if use_peft_key in constants.InstructionReplacementBank:
            input.preprocessed_input = re.sub(
                input.metadata[LLM_SUMMARIZATION_PROMPT], constants.InstructionReplacementBank[use_peft_key], input.preprocessed_input
            )
            temp_pred_metadata_holder["peft_instruction_replacement"] = str(True).lower()
            input.add_metadata(LLM_SUMMARIZATION_PROMPT, constants.InstructionReplacementBank[use_peft_key])
        # Replace Instruction if Incident table instead of Case table
        # Replaces the word "case" or "cases" in the instruction to "incident". This is specific to ITSM
        if constants.LLM_TABLE_KEY in optional_parameters:
            if optional_parameters[constants.LLM_TABLE_KEY] == constants.LLM_INCIDENT_TABLE:
                case_replacement_expression = r"(?<![a-zA-Z])case(s)?(?![a-zA-Z])"
                new_summarization_instruction = re.sub(
                    case_replacement_expression, constants.LLM_INCIDENT_TABLE, input.metadata[LLM_SUMMARIZATION_PROMPT], flags=re.IGNORECASE
                )
                input.preprocessed_input = re.sub(
                    input.metadata[LLM_SUMMARIZATION_PROMPT], new_summarization_instruction, input.preprocessed_input
                )
                input.add_metadata(LLM_SUMMARIZATION_PROMPT, new_summarization_instruction)
                input.preprocessed_input = re.sub(
                    constants.LLM_SHORT_DESCRIPTION_SENTENCE_CASE, constants.LLM_SHORT_DESCRIPTION_SENTENCE_INCIDENT, input.preprocessed_input
                )
                temp_pred_metadata_holder["itsm_instruction_replacement"] = str(True).lower()
        temp_pred_metadata_holder["instructions_applied"] = input.metadata[LLM_SUMMARIZATION_PROMPT]
        return input


class ProcessTags(PreProcessStep):
    """
    The goal of this step is to remove tags for on the fly experimentation
    Below are few examples of how it might look in the prompt...

    Example 1: <{category1.state}>State: resolved<{category1.state}/>
    Example 2: <{category1.priority}>Priority: Critical<{category1.priority}/>
    Example 3: comment<{category1.created_by}> by agent<{category1.created_by}/>: My laptop is not working
    Example 4: <{category2.instruction}>Generate a SUMMARY DOCUMENT ...<{category2.instruction}/>


    Below is the logic for preprocessing

    In options you would get a field processTags: [state, created_by]
    In the prompt, search for expressions of the form <{a.b}>text<{a.b}/>
    If a = category2 OR b in processTags, just remove the surrounding <{a.b}> and <{a.b}/>
    E.g., in Example 1, <{category1.state}>State: resolved<{category1.state}/> would get modified to State: resolved
    In example 3:
    comment<{category1.created_by}> by agent<{category1.created_by}/>: My laptop is not working would get modified to
    comment by agent: My laptop is not working

    Else, remove the entire matched expression <{a.b}>text<{a.b}/>
    """

    process_pattern = r"(\<\{[\d\D]*?\}\>)[\D\d]*?(\<\{[\d\D]*?\}\/\>)"  # Identify <{category1.state}> pattern across input
    tag_pattern = r"[^a-zA-Z\d.]"  # To identify 'category' and 'tag' within  <{category1.state}>

    def preprocess(self, input: PreProcessedInput, tokenizer: AutoTokenizer, temp_pred_metadata_holder: dict) -> PreProcessedInput:
        options = input.metadata[constants.LLM_SUMMARIZATION_OPTIONS_TENSOR_NAME]
        processed_input_text = input.preprocessed_input
        process_tags = options.get(constants.LLM_PROCESS_TAGS, [])  # Extract process tags
        for matches in re.finditer(self.process_pattern, processed_input_text):
            existing_tags = re.sub(self.tag_pattern, r"", matches.group(1)).split(".")
            category_tag = existing_tags[0]  # category1 or category2
            process_tag = existing_tags[1] if len(existing_tags) > 1 else existing_tags[0]  # Find process tag (state, priority etc)
            if category_tag == constants.LLM_CATEGORY2_TAG or (process_tag in process_tags):
                processed_input_text = processed_input_text.replace(matches.group(1), "").replace(matches.group(2), "")
            else:
                processed_input_text = processed_input_text.replace(matches.group(0), "")
        input.preprocessed_input = processed_input_text
        return input


class RemoveChatSpecificBoilerTemplateCode(PreProcessStep):
    """
    The goal of this step is to remove chat specific boiler template code.

    1.	Remove repeated boilerplate text by VA (virtual agent)
        [timestamp]Virtual Agent: Please click on Show me everything that I can assist you with:
        [timestamp]Virtual Agent: Show Me Everything
        [timestamp]Virtual Agent: I am sorry but I didn't understand your request.
        [timestamp]Virtual Agent: Please try giving me your request in a different way. I'm currently better at understanding short sentences.
        [timestamp]Virtual Agent: What’s your issue or request? Or take a look at what I can help with.
        [timestamp]Virtual Agent: Thank you for using our support chat.
    """

    lines_to_remove = ["hi, i'm your virtual agent. let me know how i can help you today.",
                       "what's your issue or request? or take a look at what i can help with.",
                       "i am sorry but i didn't understand your request.",
                       "please try giving me your request in a different way. i'm currently better at understanding short sentences.",
                       'show me everything',
                       'i want to be sure i got this right. what item best describes what you want to do?',
                       "no problem, let's try again. select one that matches what you want to do.",
                       'please stand by while i connect you to a live agent.',
                       'it seems you have left the conversation.',
                       'thank you for using our support chat.',
                       'please stand by a while i connect you to a live agent.',
                       'thank you for contacting support. i am looking into your question now',
                       'are you still there? in an effort to provide the best service, this chat will time-out after three minutes of inactivity.']

    def preprocess(self, input: PreProcessedInput, tokenizer: AutoTokenizer, temp_pred_metadata_holder: dict) -> PreProcessedInput:
        indexes_to_removed = {}
        options = input.metadata[constants.LLM_SUMMARIZATION_OPTIONS_TENSOR_NAME]
        additional_lines_to_remove = options.get(constants.LLM_CAPABILITY_CHAT_SUMMARIZATION_LINES_TO_REMOVE, [])

        local_lines_to_remove = {}
        for line in self.lines_to_remove:
            local_lines_to_remove[line.lower()] = 1
        for add_line in additional_lines_to_remove:
            local_lines_to_remove[add_line.lower()] = 1

        personas = {}
        if constants.LLM_CAPABILITY_CHAT_SUMMARIZATION_PERSONA in options:
            personas = json.loads(options.get(constants.LLM_CAPABILITY_CHAT_SUMMARIZATION_PERSONA, ""))

        persona_replacer = {}
        if constants.LLM_CAPABILITY_CHAT_SUMMARIZATION_PERSONA_VIRTUAL_AGENT in personas:
            persona_replacer[personas[constants.LLM_CAPABILITY_CHAT_SUMMARIZATION_PERSONA_VIRTUAL_AGENT].lower()] = 'Virtual Agent'

        if constants.LLM_CAPABILITY_CHAT_SUMMARIZATION_PERSONA_CUSTOMER in personas:
            persona_replacer[personas[constants.LLM_CAPABILITY_CHAT_SUMMARIZATION_PERSONA_CUSTOMER].lower()] = 'Customer'

        if constants.LLM_CAPABILITY_CHAT_SUMMARIZATION_PERSONA_LIVEAGENT in personas:
            live_agents = json.loads(personas[constants.LLM_CAPABILITY_CHAT_SUMMARIZATION_PERSONA_LIVEAGENT])
            for a_live_agent in live_agents:
                persona_replacer[a_live_agent.lower()] = 'Agent'

        unsplit_lines = input.preprocessed_input

        for apersona in persona_replacer.keys():
            if apersona in unsplit_lines.lower():
                compiled = re.compile(re.escape(apersona), re.IGNORECASE)
                unsplit_lines = str(compiled.sub(persona_replacer[apersona], unsplit_lines))

        lines = unsplit_lines.splitlines()
        for i in range(len(lines)):
            for rline in local_lines_to_remove:
                if rline.lower() in lines[i].lower():
                    indexes_to_removed[i] = 1
                if "system:" in lines[i].lower():
                    indexes_to_removed[i] = 1

        for i in sorted(indexes_to_removed, reverse=True):
            del lines[i]
        input.preprocessed_input = "\n".join(lines)
        return input



# import re
# import bs4
# from preprocessing.common import (
#     PreProcessStep,
#     PreProcessedInput,
#     LLM_SUMMARIZATION_PROMPT,
#     LLM_SUMMARIZATION_MAX_TOKENS,
#     LLM_LINKS,
# )
# from transformers import AutoTokenizer

# class RemoveLinks(PreProcessStep):
#     """Remove links from the input"""

#     http_regex = "https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)"

#     def preprocess(
#         self, input: PreProcessedInput, tokenizer: AutoTokenizer
#     ) -> PreProcessedInput:
#         # replace links with dummy links
#         link_dict = {}
#         all_links = re.findall(RemoveLinks.http_regex, input.preprocessed_input)
#         counter = 0
#         for link in all_links:
#             if link in link_dict:
#                 continue
#             link_dict[link] = "https://link" + str(counter) + ".servicenow.com"
#             counter += 1
#         # replace links
#         for link in link_dict:
#             input.preprocessed_input = input.preprocessed_input.replace(
#                 link, link_dict[link]
#             )
#         input.add_metadata(LLM_LINKS, link_dict)
#         return input


# class RemoveHtmlTags(PreProcessStep):
#     """Remove HTML tags from the input"""

#     def preprocess(
#         self, input: PreProcessedInput, tokenizer: AutoTokenizer
#     ) -> PreProcessedInput:
#         # Create a BeautifulSoup object
#         soup = bs4.BeautifulSoup(input.preprocessed_input, "html.parser")

#         # Iterate over the BeautifulSoup object and remove the tags
#         for tag in soup.find_all("*"):
#             tag.decompose()
#         # Print the text content of the BeautifulSoup object
#         input.preprocessed_input = soup.text
#         return input


# class RemovePreviousSummary(PreProcessStep):
#     """Remove the previous summary from the input"""

#     regex_str = "[\n\s]*?[*]+\sGenerated by NowLLM\s[*]+[\n\s]*?[\d\D]+?[\n\s]*?[\n\s]*?[*]+\sGenerated by NowLLM\s[*]+[\n\s]*?"

#     def preprocess(
#         self, input: PreProcessedInput, tokenizer: AutoTokenizer
#     ) -> PreProcessedInput:
#         input.preprocessed_input = re.sub(
#             RemovePreviousSummary.regex_str, "", input.preprocessed_input
#         )
#         return input


# class ExportPrompt(PreProcessStep):
#     """Export the prompt from the input"""

#     regex_str = "<\|endoftext\|><\|customer\|>([\D\d]*?)<\|endoftext\|><\|agent\|>"

#     def preprocess(
#         self, input: PreProcessedInput, tokenizer: AutoTokenizer
#     ) -> PreProcessedInput:
#         prompt = re.search(ExportPrompt.regex_str, input.preprocessed_input).group(1)
#         input.add_metadata(LLM_SUMMARIZATION_PROMPT, prompt)
#         return input


# class TruncateInput(PreProcessStep):
#     """Truncates the input Text token count to less than the max token count"""

#     regex_str = "(comments:|work_notes:)"
#     end_of_text_token = "<|endoftext|><|customer|>"

#     def single_input_total_token_size(
#         self, preprocessed_input: str, tokenizer: AutoTokenizer
#     ) -> int:
#         # Calculate token size of the input text
#         token_counter = tokenizer(preprocessed_input, return_tensors="pt")
#         return token_counter["input_ids"].shape[1]

#     def truncate_input(
#         self, input: PreProcessedInput, tokenizer: AutoTokenizer
#     )-> PreProcessedInput:
#         # Calculate total token size and return input if less than max
#         input_text_token_count = self.single_input_total_token_size(
#             input.preprocessed_input, tokenizer)
#         input.add_prediction_metadata("input_token_count", str(input_text_token_count))
#         if input_text_token_count < LLM_SUMMARIZATION_MAX_TOKENS:
#             return input
#         comments_start_index = []
#         for match in re.finditer(TruncateInput.regex_str, input.preprocessed_input):
#             comments_start_index.append(match.start())
#         comments_start_index_list_length = len(comments_start_index)
#         iterator = 1
#         processed_input = ""
#         token_calc = input_text_token_count
#         while token_calc > LLM_SUMMARIZATION_MAX_TOKENS:
#             if iterator == comments_start_index_list_length - 1:
#                 eot_token_index = input.preprocessed_input.find(TruncateInput.end_of_text_token)
#                 sub_string_input = input.preprocessed_input[comments_start_index[0]:eot_token_index]
#                 processed_input = input.preprocessed_input.replace(sub_string_input, "")
#                 token_calc = self.single_input_total_token_size(processed_input, tokenizer)
#                 input.add_prediction_metadata("token_count_final", str(token_calc))
#                 break
#             sub_string_input = input.preprocessed_input[comments_start_index[0]:comments_start_index[iterator]]
#             processed_input = input.preprocessed_input.replace(sub_string_input, "")
#             token_calc = self.single_input_total_token_size(processed_input, tokenizer)
#             input.add_prediction_metadata("token_count_final", str(token_calc))
#             iterator += 1
#         input.preprocessed_input = processed_input
#         return input

#     def preprocess(
#         self, input: PreProcessedInput, tokenizer: AutoTokenizer
#     ) -> PreProcessedInput:
#         input = self.truncate_input(input, tokenizer)
#         return input
