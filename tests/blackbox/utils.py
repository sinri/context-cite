import csv
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, GenerationConfig

from context_cite.black_box import BlackBoxCitationAnalyzer


class LLM:
    def __init__(self, analyzer: BlackBoxCitationAnalyzer):
        # self.__tokenizer = AutoTokenizer.from_pretrained(model_source)
        # self.__model = AutoModelForCausalLM.from_pretrained(
        #     model_source,
        #     torch_dtype="auto",
        #     device_map="auto"
        # )
        self.__model, self.__tokenizer = analyzer.get_model_and_tokenizer()

        # self.__pipe = pipeline("text-generation", model=model_source)
        self.__config = GenerationConfig(
            max_new_tokens=2048,
            repetition_penalty=1.5,
        )

    def generate(self, prompt: str, enable_thinking=False, remove_think=True):
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = self.__tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,  # Switches between thinking and non-thinking modes. Default is True.
        )

        model_inputs = self.__tokenizer([text], return_tensors="pt").to(self.__model.device)
        # conduct text completion
        generated_ids = self.__model.generate(
            **model_inputs,
            generation_config=self.__config,
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        thinking_content = self.__tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self.__tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        if remove_think:
            return content
        else:
            return thinking_content + "\n\n" + content


# model_source="E:\\sinri\\DeepSeek-R1-Distill-Qwen-1.5B"
# model_source = "E:\\sinri\\HuggingFace\\Qwen3-1.7B"
# llm = LLM(model_source)

class ReportCsvHelper:
    def __init__(self, target_csv_file_path: str):
        self.__target_csv_file_path = target_csv_file_path
        self.__records = []

    def add_record(self,
                   input_1: str,
                   input_2: str,
                   output_1: str,
                   output_2: str,
                   influence_1: float,
                   influence_2: float,
                   ):
        self.__records.append({
            'input_1': input_1,
            'input_2': input_2,
            'output_1': output_1,
            'output_2': output_2,
            'influence_1': influence_1,
            'influence_2': influence_2,
        })

    def save(self):
        with open(self.__target_csv_file_path, 'w',newline="", encoding="gbk",errors="replace") as csvfile:
            fieldnames = ['input_1', 'input_2', 'output_1', 'output_2', 'influence_1', 'influence_2']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for record in self.__records:
                writer.writerow(record)
