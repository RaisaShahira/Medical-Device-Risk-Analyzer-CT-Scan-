from llm import gpt_model_call
import json


# ===============================
# Brainstorming Agent
# ===============================
class BrainstormingAgent:
    def __init__(self):
        self.prompt_template = """
You are a biomedical risk analysis assistant specialized in performing
Failure Mode and Effects Analysis (FMEA) for a CT Scan medical imaging system.

The CT Scan system includes hardware, software, electrical, mechanical,
radiation safety, and operational components.

Here is the current FMEA table:
{{fmea_table}}

You are focusing on the FMEA field "{{dic_key_value}}" in row {{selected_row}}.

The user is editing the field "{{dic}}" and has entered the following text:
"{{user_text}}"

Your task:
- Help refine or expand this entry.
- Brainstorm **5 alternative variants**, each addressing a different possible aspect
  (e.g., hardware failure, software issue, human error, environmental condition, safety impact).
- The output MUST be relevant to CT Scan systems.

IMPORTANT RULES:
- ONLY generate content for TEXTUAL FMEA fields:
  - Failure Mode
  - Effects of Failure
  - Potential Cause
  - Current Controls
  - Recommended Actions
  - Responsible
  - Actions Taken
- DO NOT generate or infer Severity, Occurrence, Detection, or RPN values.
- This is a decision-support tool, NOT a final authority.

Output MUST be valid JSON and strictly follow this template.
Do NOT include explanations outside the JSON.

Output template:
{
  "output": [
    {"reason": "different_aspect", "content": "variant_text", "comment": "concise_explanation"},
    {"reason": "different_aspect", "content": "variant_text", "comment": "concise_explanation"},
    {"reason": "different_aspect", "content": "variant_text", "comment": "concise_explanation"},
    {"reason": "different_aspect", "content": "variant_text", "comment": "concise_explanation"},
    {"reason": "different_aspect", "content": "variant_text", "comment": "concise_explanation"}
  ]
}

Output:
"""

    def generate_output(
        self,
        fmea_table,
        dic_key_value,
        selected_row,
        user_text,
        dic,
        model="gpt-4.1-mini"
    ):
        prompt = self.prompt_template
        prompt = prompt.replace("{{fmea_table}}", fmea_table)
        prompt = prompt.replace("{{dic_key_value}}", dic_key_value)
        prompt = prompt.replace("{{selected_row}}", str(selected_row))
        prompt = prompt.replace("{{dic}}", dic)
        prompt = prompt.replace("{{user_text}}", user_text)

        print("***********************")
        print("Brainstorming Agent running")
        print("***********************")

        try:
            text_output = gpt_model_call(prompt, model=model)
        except Exception as e:
            print(f"LLM Error: {e}")
            return None

        try:
            result_json = json.loads(text_output)
        except json.JSONDecodeError as e:
            print("JSON parsing failed:", e)
            print(text_output)
            return None

        with open("brainstorming_agent_output.json", "w") as f:
            json.dump(result_json, f, indent=4)

        return result_json


# ===============================
# Completing Agent
# ===============================
class CompletingAgent:
    def __init__(self):
        self.prompt_template = """
You are a biomedical risk analysis assistant supporting FMEA completion
for a CT Scan medical imaging system.

Here is the current FMEA table:
{{fmea_table}}

You are focusing on the FMEA field "{{dic_key_value}}" in row {{selected_row}}.

Your task:
- Provide **3 alternative textual entries** for the field "{{dic}}".
- Each alternative should reflect a different plausible aspect relevant to CT Scan systems.

IMPORTANT RULES:
- ONLY generate TEXTUAL FMEA fields.
- DO NOT generate Severity, Occurrence, Detection, or RPN.
- Keep all outputs concise and professional.
- Output MUST be valid JSON only.

Output template:
{
  "output": [
    {"reason": "different_aspect", "content": "variant_text", "comment": "concise_explanation"},
    {"reason": "different_aspect", "content": "variant_text", "comment": "concise_explanation"},
    {"reason": "different_aspect", "content": "variant_text", "comment": "concise_explanation"}
  ]
}

Output:
"""

    def generate_output(
        self,
        fmea_table,
        dic_key_value,
        selected_row,
        dic,
        model="gpt-4.1-mini"
    ):
        prompt = self.prompt_template
        prompt = prompt.replace("{{fmea_table}}", fmea_table)
        prompt = prompt.replace("{{dic_key_value}}", dic_key_value)
        prompt = prompt.replace("{{selected_row}}", str(selected_row))
        prompt = prompt.replace("{{dic}}", dic)

        print("***********************")
        print("Completing Agent running")
        print("***********************")

        try:
            text_output = gpt_model_call(prompt, model=model)
        except Exception as e:
            print(f"LLM Error: {e}")
            return None

        try:
            result_json = json.loads(text_output)
        except json.JSONDecodeError as e:
            print("JSON parsing failed:", e)
            print(text_output)
            return None

        with open("completing_agent_output.json", "w") as f:
            json.dump(result_json, f, indent=4)

        return result_json