import enum


class CapabilityOptions(enum.Enum):
    RECORD_SUMMARIZATION = "Record Summarization"
    RECORD_RESOLUTION = "Record Resolution"
    CHAT_SUMMARIZATION = "Chat Summarization"
    GENERATE_KB_ARTICLE = "Generate Knowledge Article"
    GENERIC = "Generic"

    @classmethod
    def get_key_from_value(cls, value):
        for key, enum_value in cls.__members__.items():
            if enum_value.value == value:
                return key
        return None


class CapabilityDefinition:
    def __init__(self, selected_capability, preprocessing_steps=None, postprocessing_steps=None):
        # Initialize capability using the provided value or default to RECORD_SUMMARIZATION
        self.capability = CapabilityOptions.get_key_from_value(selected_capability) or CapabilityOptions.RECORD_SUMMARIZATION.name

        # Define preprocessing steps based on capability
        self.preprocessing_steps = preprocessing_steps or {
            CapabilityOptions.RECORD_SUMMARIZATION.name: [
                "replace_previous_model_tags",
                "remove_created_date",
                "prompt",
                "replace_instructions",
                "remove_links",
                "html_tags",
                "remove_previous_summary",
                "process_tags",
                "truncate_input",
            ],
            CapabilityOptions.RECORD_RESOLUTION.name: [
                "replace_previous_model_tags",
                "remove_created_date",
                "prompt",
                "replace_instructions",
                "remove_links",
                "html_tags",
                "remove_previous_summary",
                "process_tags",
                "truncate_input",
            ],
            CapabilityOptions.CHAT_SUMMARIZATION.name: [
                "replace_previous_model_tags",
                "remove_created_date",
                "prompt",
                "replace_instructions",
                "remove_links",
                "html_tags",
                "remove_previous_summary",
                "process_tags",
                "remove_chat_specific_boiler_template_code",
                "truncate_input",
            ],
            CapabilityOptions.GENERATE_KB_ARTICLE.name: [
                "replace_previous_model_tags",
                "remove_created_date",
                "prompt",
                "replace_instructions",
                "remove_links",
                "html_tags",
                "remove_previous_summary",
                "truncate_input",
            ],
            CapabilityOptions.GENERIC.name: [],
        }

        # Define postprocessing steps based on capability
        self.postprocessing_steps = postprocessing_steps or {
            CapabilityOptions.RECORD_SUMMARIZATION.name: [
                "prompt_repeat_check",
                "consecutive_repeat_check",
                "replace_links",
                "bu_specific_processing",
                "output_sections_generator",
                "water_mark_addition",
            ],
            CapabilityOptions.RECORD_RESOLUTION.name: [
                "prompt_repeat_check",
                "consecutive_repeat_check",
                "replace_links",
                "bu_specific_processing",
                "output_sections_generator",
                "water_mark_addition",
            ],
            CapabilityOptions.CHAT_SUMMARIZATION.name: [
                "prompt_repeat_check",
                "consecutive_repeat_check",
                "replace_links",
                "bu_specific_processing",
                "output_sections_generator",
                "water_mark_addition",
            ],
            CapabilityOptions.GENERATE_KB_ARTICLE.name: [
                "prompt_repeat_check",
                "consecutive_repeat_check",
                "replace_links",
                "bu_specific_processing",
                "output_sections_generator",
                "water_mark_addition",
            ],
            CapabilityOptions.GENERIC.name: [],
        }

    def get_preprocessing_steps(self):
        return self.preprocessing_steps.get(self.capability)

    def get_postprocessing_steps(self):
        return self.postprocessing_steps.get(self.capability)
