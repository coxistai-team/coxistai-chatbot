from transformers import pipeline
import logging

logging.info("Loading zero-shot classification model...")
classifier = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")
logging.info("Zero-shot model loaded.")

NON_EDU_KEYWORDS = [
    "movie", "theft", "robbery", "netflix", "watch", "download", "shopping", "travel",
    "celebrity", "repair my", "fix my", "broken", "not working", "how much to repair",
    "where to fix", "price of", "how much does", "cost of", "which phone", "which mobile",
    "which game", "best ice cream", "recommend a", "which brand", "better option",
    "should I buy", "top rated"
]
EDU_KEYWORDS = [
    "explain", "how", "science", "language", "history", "scientific", "logic",
    "architecture", "design principles", "engineering", "teach me", "learning",
    "pedagogy", "types of", "list of", "classification of", "define", "difference between"
]

def is_educational(question: str) -> bool:
    """
    Checks if a question is educational using a hybrid keyword and zero-shot model approach.
    """
    if not question:
        return False

    question_lower = question.lower()

    # Fast keyword check first
    if any(keyword in question_lower for keyword in NON_EDU_KEYWORDS):
        return False
    if any(keyword in question_lower for keyword in EDU_KEYWORDS):
        return True

    # If keywords are not decisive, use the pre-loaded zero-shot model
    try:
        labels = [
            "educational: factual knowledge, explanations, technology, critical thinking, science, maths, sports, coding, general knowledge, or academic concepts",
            "non-educational: requests product comparisons, unsafe, Consumer Product Advice, shopping advice, entertainment, random_fun, plant_motivation, absurd_request, gaming, or personal opinions"
        ]
        result = classifier(question, labels)

        # Check if the top-scoring label is the 'educational' one
        if "educational" in result['labels'][0] and result['scores'][0] > 0.85:
            return True
        else:
            return False
    except Exception as e:
        logging.error(f"Classifier error: {e}")
        return True # Fail safely
    
# Test code removed to prevent execution during import
# questions =[
#     "what is python"
#     ]
# for q in questions:
#     print(f"Q: {q}")
#     print("Educational" if is_educational(q) else "Non-educational")
#     print("---")