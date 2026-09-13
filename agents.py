from llm import gpt_model_call
import json
from similarity_engine import FailureModeSimilarityEngine

# =========================
# Similarity Engine Global
# =========================
sim_engine = FailureModeSimilarityEngine("FMEACT.csv")


# =========================
# Brainstorming Agent
# =========================
class BrainstormingAgent:
    def __init__(self):
        self.prompt_template = """
You are a biomedical risk analysis assistant helping create and refine
Failure Mode and Effects Analysis (FMEA) for a CT Scan medical imaging system.

All knowledge must be derived logically and based on patterns from the provided dataset.

Here is the user's FMEA table:
{{fmea_table}}

Here are historically similar rows from the user's dataset (matched based on Failure Mode similarity):
{{similar_rows}}

You are focusing on the field "{{dic_key_value}}" in row {{selected_row}}.
User is currently editing "{{dic}}" and has typed:
"{{user_text}}"

Your task:
- Brainstorm 5 alternative options for this field.
- Variants should reflect:
  hardware / software / operator / environment / patient safety aspects

Rules:
- Only textual FMEA fields
- No Severity / Occurrence / Detection / RPN
- JSON only

{
 "output":[
   {"reason":"","content":"","comment":""}
 ]
}
Output:
"""


# =========================
# Completing Agent
# =========================
class CompletingAgent:
    def __init__(self):
        self.prompt_template = """
You are a biomedical risk analysis assistant supporting FMEA completion
for a CT Scan system.

Current FMEA table:
{{fmea_table}}

Historically similar dataset rows:
{{similar_rows}}

You are completing the field "{{dic_key_value}}" in row {{selected_row}}.

TASK:
- Provide 3 realistic entries for "{{dic}}"

Rules:
- Only textual FMEA fields
- No Severity / Occurrence / Detection / RPN
- JSON output only

{
 "output":[
   {"reason":"","content":"","comment":""}
 ]
}
Output:
"""

    # -----------------------
    # LLM Completing Mode
    # -----------------------
    def generate_output(self, fmea_table, dic_key_value,
                        selected_row, dic, similar_rows,
                        model="gpt-4.1-mini"):

        prompt = self.prompt_template \
            .replace("{{fmea_table}}", fmea_table) \
            .replace("{{dic_key_value}}", dic_key_value) \
            .replace("{{selected_row}}", str(selected_row)) \
            .replace("{{dic}}", dic) \
            .replace("{{similar_rows}}", similar_rows)

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


    # -----------------------
    # SIMILARITY MODE
    # -----------------------
    def completing_agent_with_similarity(self, query_failure_mode):

        match = sim_engine.find_best_match(query_failure_mode)

        if not match:
            return {
                "output":[
                    {
                        "reason":"no_match",
                        "content":"No similar failure found",
                        "comment":"Similarity score too low"
                    }
                ]
            }

        return {
            "output":[
                {
                    "reason":"matched_failure_mode",
                    "content": match.get("Failure Mode",""),
                    "comment":"Auto suggested based on similarity score"
                },
                {
                    "reason":"effects_reference",
                    "content": match.get("Effects of Failure",""),
                    "comment":"Pulled from dataset"
                },
                {
                    "reason":"recommended_action_reference",
                    "content": match.get("Recommended Actions",""),
                    "comment":"Suggested mitigation"
                }
            ]
        }
