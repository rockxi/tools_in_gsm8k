import os
import requests
import json
from datasets import load_dataset

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
MODEL_NAME = ["openai/gpt-3.5-turbo", "openai/gpt-4o-mini",]
MAX_SAMPLES = 250
OUTPUT_FILE_NO_TOOLS = "gsm8k_no_tools"
OUTPUT_FILE_WITH_TOOLS = "gsm8k_tools"

costs_dict = {
    'openai/gpt-3.5-turbo' : (0.5 / 1_000_000, 1.5 / 1_000_000),
    'openai/gpt-4o-mini' : (0.15 / 1_000_000, 0.6 / 1_000_000),
}

def calculator(expression):
    """
    A simple calculator function that evaluates mathematical expressions.

    Args:
        expression (str): A mathematical expression to evaluate

    Returns:
        float: The result of the calculation
    """
    try:
        # Using eval with a clean namespace for safety
        # In a production environment, you'd want to use a more secure approach
        result = eval(expression, {"__builtins__": {}}, {})
        return result
    except Exception as e:
        return f"Error calculating: {str(e)}"

def run_test_no_tools(api_key, model_name, max_samples, output_file):
    dataset = load_dataset("gsm8k", 'main')
    for k in dataset:
        dataset = dataset[k]; break

    results = []
    correct_count = 0

    for i in range(min(max_samples, len(dataset))):
        example = dataset[i]
        question = example["question"]
        answer = example["answer"]

        prompt = f"Solve the math problem and write only answer no solution:\n\n{question}\n\nAnswer:" # Простой prompt

        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        data = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}]
        }

        try:
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
            response.raise_for_status() # Проверка на HTTP ошибки
            response_json = response.json()

            if 'usage' in response_json:
                usages = response_json['usage']['completion_tokens'], response_json['usage']['prompt_tokens']
                costs = usages[0] * costs_dict[model_name][0] + usages[1] * costs_dict[model_name][1]
            else: costs = 0

            model_response = response_json['choices'][0]['message']['content']

            prompt = ('Пользователь отправляет правильный ответ на вопрос и свой ответ.'
                      'Верни True если ответ пользователя совпадает с правильным и False если нет.'
                      'Не возвращай ничего кроме True или False')

            data = {
                "model": 'openai/gpt-4o',
                "messages": [{"role": "system", "content": prompt},
                             {"role": "system", "content": f'Правильный ответ: {answer}'},
                             {"role": "user", "content": f'мой ответ: {model_response}'}]
            }

            response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
            response.raise_for_status() # Проверка на HTTP ошибки
            response_json = response.json()

            is_correct = response_json['choices'][0]['message']['content'] == 'True'
            if is_correct: correct_count += 1


            results.append({
                "question": question,
                "gold_answer": answer,
                "model_response": model_response,
                "is_correct": is_correct,
                "costs" : costs
            })

            print(f"Задача {i+1}/{max_samples}: Correct={is_correct}")

        except requests.exceptions.RequestException as e:
            print(f"Ошибка API для задачи {i+1}: {e}")
            results.append({
                "question": question,
                "gold_answer": answer,
                "model_response": "API Error",
                "is_correct": False,
                "error": str(e)
            })

    accuracy = (correct_count / max_samples) * 100 if max_samples > 0 else 0
    print(f"\nТочность БЕЗ инструментов: {accuracy:.2f}% ({correct_count}/{max_samples})")

    # Сохранение результатов в JSONL формате
    name = model_name.split('/')[1] if '/' in model_name else model_name
    output_file = output_file + '_' + name + '.jsonl'
    json.dump(results,
              open(output_file, 'w'),
              indent = 4)
    print(f"Результаты сохранены в: {output_file}")

def run_test_with_tools(api_key, model_name, max_samples, output_file):
    dataset = load_dataset("gsm8k", 'main')
    for k in dataset:
        dataset = dataset[k]; break

    results = []
    dialogs = []
    correct_count = 0

    for i in range(min(max_samples, len(dataset))):
        example = dataset[i]
        question = example["question"]
        answer = example["answer"]

        # Prompt that asks the model to solve the problem using the calculator
        prompt = f"Solve the following math problem step by step. Use the calculator when you need to perform calculations. At the end, provide ONLY the final numerical answer.\n\n{question}\n\nAnswer:"

        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

        # Define the calculator tool
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Calculate the result of a mathematical expression",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "The mathematical expression to evaluate"
                            }
                        },
                        "required": ["expression"]
                    }
                }
            }
        ]

        # Start a conversation with the model
        messages = [{"role": "user", "content": prompt}]
        conversation_history = [{"role": "user", "content": prompt}]

        try:
            # Keep interacting until we get a final answer (no more tool calls)
            costs = 0
            while True:
                data = {
                    "model": model_name,
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": "auto"  # Let the model decide when to use the tool
                }

                response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
                response.raise_for_status()
                response_json = response.json()

                if 'usage' in response_json:
                    usages = response_json['usage']['completion_tokens'], response_json['usage']['prompt_tokens']
                    costs += usages[0] * costs_dict[model_name][0] + usages[1] * costs_dict[model_name][1]
                else:
                    costs += 0

                assistant_message = response_json['choices'][0]['message']
                conversation_history.append(assistant_message)

                # Check if there are tool calls
                if "tool_calls" in assistant_message:
                    # Add the assistant's message to conversation
                    messages.append(assistant_message)

                    # Process each tool call
                    for tool_call in assistant_message["tool_calls"]:
                        if tool_call["function"]["name"] == "calculator":
                            # Parse the arguments
                            arguments = json.loads(tool_call["function"]["arguments"])
                            expression = arguments.get("expression", "")

                            # Call the calculator function
                            calc_result = calculator(expression)

                            # Create a response message from the tool
                            tool_response = {
                                "role": "tool",
                                "tool_call_id": tool_call["id"],
                                "name": "calculator",
                                "content": str(calc_result)
                            }

                            # Add to conversation and history
                            messages.append(tool_response)
                            conversation_history.append(tool_response)
                else:
                    # No tool calls, we have the final answer
                    model_response = assistant_message["content"]
                    break

            # Check if the answer is correct
            prompt = ('Пользователь отправляет правильный ответ на вопрос и свой ответ.'
                      'Верни True если ответ пользователя совпадает с правильным и False если нет.'
                      'Не возвращай ничего кроме True или False')

            data = {
                "model": 'openai/gpt-4o',
                "messages": [{"role": "system", "content": prompt},
                             {"role": "system", "content": f'Правильный ответ: {answer}'},
                             {"role": "user", "content": f'мой ответ: {model_response}'}]
            }

            response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
            response.raise_for_status()
            response_json = response.json()

            is_correct = response_json['choices'][0]['message']['content'] == 'True'
            if is_correct: correct_count += 1

            results.append({
                "question": question,
                "gold_answer": answer,
                "model_response": model_response,
                "is_correct": is_correct,
                "costs": costs
            })
            dialogs.append({
                "conversation": conversation_history
            })

            print(f"Задача {i + 1}/{max_samples}: Correct={is_correct}")

        except requests.exceptions.RequestException as e:
            print(f"Ошибка API для задачи {i + 1}: {e}")
            results.append({
                "question": question,
                "gold_answer": answer,
                "model_response": "API Error",
                "is_correct": False,
                "error": str(e),
            })

    accuracy = (correct_count / max_samples) * 100 if max_samples > 0 else 0
    print(f"\nТочность С инструментами: {accuracy:.2f}% ({correct_count}/{max_samples})")

    # Сохранение результатов в JSONL формате
    name = model_name.split('/')[1] if '/' in model_name else model_name
    output_file = output_file + '_' + name + '.jsonl'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    with open('dialogs.jsonl', 'w') as f:
        json.dump(dialogs, f, indent=4)
    print(f"Результаты сохранены в: {output_file}")

if __name__ == "__main__":
    for model_name_ in MODEL_NAME:
        run_test_no_tools(OPENROUTER_API_KEY, model_name_, MAX_SAMPLES, OUTPUT_FILE_NO_TOOLS)
        run_test_with_tools(OPENROUTER_API_KEY, model_name_, MAX_SAMPLES, OUTPUT_FILE_WITH_TOOLS)
