import enum

# Evaluations are question answering, summarization
class EvalTypes(enum.Enum):
    QUESTION_ANSWERING = "question_answering"
    SUMMARIZATION = "summarization"