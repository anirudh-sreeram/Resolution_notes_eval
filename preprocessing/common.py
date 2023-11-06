from abc import ABC, abstractmethod

#import constants
#from debug_response_details import DebugResponseDetails

LLM_SUMMARIZATION_PROMPT = "prompt"
LLM_LINKS = "links"
LLM_SUMMARIZATION_MAX_TOKENS = 8132  # account for the prefix


class PreProcessedInput:
    def __init__(self, input: str, should_store_debug=False):
        self.original_input = input
        self.preprocessed_input = input
        self.metadata = {}
        self.prediction_metadata = []
        #self.debug_response_details = DebugResponseDetails(should_store_debug)

    def add_metadata(self, key: str, value):
        self.metadata[key] = value

    def add_prediction_metadata(self, prediction_data: dict):
        self.prediction_metadata.append(prediction_data)

    def __str__(self):
        return self.preprocessed_input


class PreProcessStep(ABC):
    @abstractmethod
    def preprocess(self, s: PreProcessedInput) -> PreProcessedInput:
        pass


# from abc import ABC, abstractmethod

# LLM_SUMMARIZATION_PROMPT = "prompt"
# LLM_LINKS = "links"
# LLM_SUMMARIZATION_MAX_TOKENS = 7500 # account for the prefix


# class PreProcessedInput:
#     def __init__(self, input: str):
#         self.original_input = input
#         self.preprocessed_input = input
#         self.metadata = {}
#         self.prediction_metadata = {}

#     def add_metadata(self, key: str, value: str):
#         self.metadata[key] = value

#     def add_prediction_metadata(self, key: str, value: str):
#         self.prediction_metadata[key] = value

#     def __str__(self):
#         return self.preprocessed_input


# class PreProcessStep(ABC):
#     @abstractmethod
#     def preprocess(self, s: PreProcessedInput) -> PreProcessedInput:
#         pass
