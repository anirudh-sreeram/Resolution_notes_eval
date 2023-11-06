# from capability_definition import CapabilityOptions

# NowLLM Summarization Constants
LLM_SUMMARIZATION_INPUT_TENSOR_NAME = "text"
LLM_SUMMARIZATION_METADATA_TENSOR_NAME = "request_metadata"
LLM_SUMMARIZATION_OUTPUT_TENSOR_NAME = "summary"
LLM_SUMMARIZATION_ERROR_TENSOR_NAME = "error"
LLM_SUMMARIZATION_OPTIONS_TENSOR_NAME = "options"
LLM_SUMMARIZATION_END_OF_TEXT_TOKEN = "eos_token"
LLM_SUMMARIZATION_END_OF_TEXT_TOKEN_DEFAULT = "<|end|>"
LLM_SUMMARIZATION_HYPER_PARAMETERS = "hyperparameters"
LLM_SUMMARIZATION_TEMPERATURE = "temperature"
LLM_SUMMARIZATION_TEMPERATURE_DEFAULT = "0.2"
LLM_SUMMARIZATION_MAX_NEW_TOKENS = "max_new_tokens"
LLM_SUMMARIZATION_MAX_NEW_TOKENS_DEFAULT = "500"
LLM_SUMMARIZATION_USE_PEFT = "use_peft"
LLM_SUMMARIZATION_USE_PEFT_DEFAULT = ""
LLM_SUMMARIZATION_DO_SAMPLE = "do_sample"
LLM_SUMMARIZATION_DO_SAMPLE_DEFAULT = True
LLM_SUMMARIZATION_NUM_BEAMS = "num_beams"
LLM_SUMMARIZATION_NUM_BEAMS_DEFAULT = "1"
LLM_SUMMARIZATION_NO_REPEAT_NGRAM_SIZE = "no_repeat_ngram_size"
LLM_SUMMARIZATION_NO_REPEAT_NGRAM_SIZE_DEFAULT = "10"
LLM_SUMMARIZATION_RESPONSE_METADATA_TENSOR_NAME = "response_metadata"
LLM_SUMMARIZATION_POST_PROCESSING_STEPS = "post_processing_steps"
LLM_SUMMARIZATION_PRE_PROCESSING_STEPS = "pre_processing_steps"
LLM_SUMMARIZATION_REPETITION_PENALTY = "repetition_penalty"
LLM_SUMMARIZATION_REPETITION_PENALTY_DEFAULT = "1.05"
LLM_SUMMARIZATION_TOKEN_LIMIT_FOR_MULTIPLE_SUMMARIES = "token_limit_for_multiple_summaries"
LLM_SUMMARIZATION_TOKEN_LIMIT_FOR_MULTIPLE_SUMMARIES_DEFAULT = "4000"
LLM_SUMMARIZATION_SMALL_CONTEXT_INSTRUCTION_NEW = "small_context_instruction_new"
LLM_SUMMARIZATION_SMALL_CONTEXT_INSTRUCTION_WIP = "small_context_instruction_wip"
LLM_SUMMARIZATION_SMALL_CONTEXT_INSTRUCTION_DEFAULT = "small_context_instruction_default"
LLM_SUMMARIZATION_SMALL_CONTEXT_INSTRUCTION_WORD_COUNT = "small_context_word_count"
LLM_SUMMARIZATION_SMALL_CONTEXT_INSTRUCTION_WORD_COUNT_LIMIT = 350
LLM_SUMMARIZATION_RESOLUTION = "resolution"
LLM_SUMMARIZATION_ACTIONS_TAKEN = "actions taken"
LLM_SUMMARIZATION_ISSUE = "issue"
LLM_SUMMARIZATION_PEFT_WIP = "CS_CSM_WIP"
LLM_SUMMARIZATION_PEFT_NEW = "CS_CSM_NEW"
LLM_SUMMARIZATION_PEFT_DEFAULT = "CS_CSM_DEFAULT"
InstructionReplacementBank = {
    LLM_SUMMARIZATION_PEFT_WIP: "\nSummarize and create exactly 2 sections which is Issue and Actions Taken. Start with Issue:",
    LLM_SUMMARIZATION_PEFT_NEW: "\nSummarize and create exactly 1 section which is Issue",
    LLM_SUMMARIZATION_PEFT_DEFAULT: "\nSummarize and create three sections Issue, Actions Taken and Resolution. "
    "Do not repeat same sentence. Start with Issue:",
    LLM_SUMMARIZATION_SMALL_CONTEXT_INSTRUCTION_WIP: """Generate a SUMMARY DOCUMENT that contains the following sections, if applicable for
    the case: 1. Issue - The Issue section should represent what the case is about. Answer in 1 or 2 sentences.
    2. Actions Taken - The Actions Taken section should provide a bulleted list of significant actions performed so far to investigate the
    case only if there are important actions in comments and work notes. Do not consider logs, alerts, attachments, stack traces, json outputs,
    unix shell outputs and source code in the given input text. Comments start with "comments:" and work notes start with "work_notes:".
    Do not generate actions taken if comments or work notes are not available. Instead, say no key actions are recorded in the case.
    Include ONLY "Issue" and "Actions Taken" sections. Exclude ALL other sections. For each section, only use the provided information and do not generate any information which is not provided. If you have no available
    information for a section, respond with N/A for that section.""",
    LLM_SUMMARIZATION_SMALL_CONTEXT_INSTRUCTION_NEW: """Provide the issue that represents what the case is about.
    Answer concisely in 1 or 2 sentences but do not extract directly from the above text. Begin your response with Issue:""",
    LLM_SUMMARIZATION_SMALL_CONTEXT_INSTRUCTION_DEFAULT: """Generate a SUMMARY DOCUMENT that contains the following sections, if they are
    applicable for the case: 1. Issue - The Issue section should represent what the case is about. Answer in 1 or 2 sentences.
    2. Actions Taken - The Actions Taken section should provide a bulleted list of significant actions performed so far to investigate the case
    only if there are important actions in comments and work notes. Do not consider logs, alerts, attachments, stack traces, json outputs,
    unix shell outputs and source code in the given input text. Comments start with "comments:" and work notes start with "work_notes:".
    Do not generate actions taken if comments or work notes are not available. Instead, say no key actions are recorded in the case.
    3. Resolution - The Resolution section should highlight the action or actions only if absolutely sure the problem was resolved and is
    explicitly stated in comments or work notes. Do not consider logs, alerts, attachments, stack traces, json outputs, unix shell outputs and
    source code in the given input text. Answer in 1 or 2 sentences. Comments start with "comments:" and work notes start with "work_notes:".
    Do not generate resolution if comments or work notes are not available. Instead, say no resolution has been recorded in the case.
    Include ONLY "Issue" "Actions Taken" and "Resolution" sections. Exclude ALL other sections. For each section, only use the provided information. For each section, do not generate any information which is not provided.
    For each section, if you are not absolutely sure, return N/A.""",
}
LLM_SUMMARIZATION_OUTPUT_SECTIONS_JSON_INPUT = "section_format"
LLM_SUMMARIZATION_OUTPUT_SECTIONS = "sections"
LLM_SUMMARIZATION_OUTPUT_SECTIONS_DEFAULT = []
LLM_SUMMARIZATION_OUTPUT_SECTIONS_FORMAT = "format"
LLM_SUMMARIZATION_OUTPUT_SECTIONS_FORMAT_DEFAULT = "json"
LLM_SUMMARIZATION_OUTPUT_SECTIONS_VERSION = "version"
LLM_SUMMARIZATION_OUTPUT_SECTIONS_VERSION_DEFAULT = 1.0
LLM_SUMMARIZATION_OUTPUT_SECTIONS_JSON_INPUT_DEFAULT = {
    LLM_SUMMARIZATION_OUTPUT_SECTIONS: LLM_SUMMARIZATION_OUTPUT_SECTIONS_DEFAULT,
    LLM_SUMMARIZATION_OUTPUT_SECTIONS_FORMAT: LLM_SUMMARIZATION_OUTPUT_SECTIONS_FORMAT_DEFAULT,
    LLM_SUMMARIZATION_OUTPUT_SECTIONS_VERSION: LLM_SUMMARIZATION_OUTPUT_SECTIONS_VERSION_DEFAULT,
}
LLM_SUMMARIZATION_WATERMARK = "watermark"
LLM_SUMMARIZATION_ENABLE_WATERMARK = "enable_watermark"
LLM_SUMMARIZATION_ENABLE_WATERMARK_DEFAULT = "False"
LLM_SUMMARIZATION_WATERMARK_DEFAULT = "********** Generated by NowLLM **********"
LLM_CAPABILITY = "capability"
# LLM_CAPABILITY_DEFAULT = CapabilityOptions.RECORD_SUMMARIZATION.value
LLM_CSM_TABLE = "sn_customerservice_case"
LLM_INCIDENT_TABLE = "incident"
LLM_TABLE_KEY = "table"
WORDREPLACEMENTBANKFORBU = {
    LLM_CSM_TABLE: r"case(s)",
    LLM_INCIDENT_TABLE: r"incident(s)",
}
LLM_DEBUG_ENABLED = "enable_debug"
LLM_DEBUG_ENABLED_DEFAULT = False
LLM_SHORT_DESCRIPTION_SENTENCE_CASE = "A case was opened with a short description"
LLM_SHORT_DESCRIPTION_SENTENCE_INCIDENT = "An incident was opened with a short description"
LLM_ITSM_POSTPROCESS_ENABLE = "enable_itsm_postprocess"
LLM_ITSM_POSTPROCESS_ENABLE_DEFAULT = False
LLM_CAPABILITY_CASE_SUMMARIZATION = ["Record Summarization", "Record Resolution"]
LLM_CAPABILITY_CHAT_SUMMARIZATION_EOS_TOKEN = 49155  # Specifically for Chat Summarization and non Record data
LLM_CAPABILITY_CHAT_SUMMARIZATION_LINES_TO_REMOVE = "lines_to_remove"
LLM_CAPABILITY_CHAT_SUMMARIZATION_PERSONA = "persona"
LLM_CAPABILITY_CHAT_SUMMARIZATION_PERSONA_VIRTUAL_AGENT = "virtualAgent"
LLM_CAPABILITY_CHAT_SUMMARIZATION_PERSONA_CUSTOMER = "customer"
LLM_CAPABILITY_CHAT_SUMMARIZATION_PERSONA_LIVEAGENT = "liveAgent"
LLM_PROCESS_TAGS = "processTags"
LLM_CATEGORY2_TAG = "category2"
