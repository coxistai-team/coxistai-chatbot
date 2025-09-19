import os
from typing import Optional
from openai import OpenAI

class SmartDeepSeek:
    def __init__(self, api_key: str):
        """Initializes the client and your original model tiers."""
        if not api_key:
            raise ValueError("API key is required.")
        self.client = OpenAI(api_key=api_key)

        # Your original model tiers
        self.free_model = "gpt-3.5-turbo"
        self.paid_model = "gpt-4o-mini"  # Updated to valid OpenAI model
        self.reason_model = "gpt-4o"  # Updated to valid OpenAI model

        # Your original logic for model switching
        self.complexity_threshold = 15
        self.dissatisfaction_triggers = [
            "not satisfied", "explain better",
            "more detail", "incomplete answer"
        ]

    def needs_paid_model(self, question: str, previous_response: str = "") -> bool:
        """This is your original logic to determine if a question needs a paid model."""
        question_lower = question.lower()

        if any(trigger in previous_response.lower() for trigger in self.dissatisfaction_triggers):
            return True

        if len(question.split()) > self.complexity_threshold:
            return True

        technical_terms = [
            "explain in detail", "step-by-step",
            "prove that", "compare and contrast"
        ]
        if any(term in question_lower for term in technical_terms):
            return True

        return False

    def query_model(self, model: str, question: str, system_prompt: Optional[str] = None) -> Optional[str]:
        """This is the token-efficient API call function we developed."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": question})
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.5,
                max_tokens=2000  # Add token limit for cost control
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API Error for model '{model}': {e}")
            print(f"Error type: {type(e).__name__}")
            # Log more detailed error information
            if hasattr(e, 'response'):
                print(f"Response status: {e.response.status_code if hasattr(e.response, 'status_code') else 'N/A'}")
                print(f"Response text: {e.response.text if hasattr(e.response, 'text') else 'N/A'}")
            return None

    def get_response(self, question: str, previous_response: str = "", system_prompt: Optional[str] = None) -> str:
        """This function now combines your switching logic with our efficient API calls."""
        
        if self.needs_paid_model(question, previous_response):
            # Try the main paid model first
            response = self.query_model(self.paid_model, question, system_prompt=system_prompt)
            # If it fails, fall back to the reason model
            if not response:
                response = self.query_model(self.reason_model, question, system_prompt=system_prompt)
        else:
            # Use the free model for simple questions
            response = self.query_model(self.free_model, question, system_prompt=system_prompt)
            
        return response or "I apologize, but I encountered an error while generating a response."