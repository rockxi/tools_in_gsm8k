import json
import matplotlib.pyplot as plt
import seaborn as sns
import os


def analyze_json_files(*file_paths):
    """
    Анализирует несколько JSON-файлов, подсчитывает количество правильных ответов и среднюю стоимость запросов.

    Args:
        file_paths (str): Пути к JSON-файлам.

    Returns:
        tuple:
            - labels (list): Метки для графиков (имена файлов/моделей).
            - correct_counts (list): Количество правильных ответов.
            - avg_costs (list): Средние стоимости запросов.
    """
    labels = []
    correct_counts = []
    avg_costs = []

    for file_path in file_paths:
        correct_answers = 0
        total_cost = 0
        total_items = 0

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                total_items = len(data)

                for item in data:
                    if item.get('is_correct', False):
                        correct_answers += 1
                    cost_value = item.get('costs', 0)
                    if isinstance(cost_value, (int, float)):
                        total_cost += cost_value
                    else:
                        print(f"Предупреждение: некорректное значение 'costs' в файле {file_path}, объект: {item}")

                avg_cost = total_cost / total_items if total_items > 0 else 0
                correct_counts.append(correct_answers)
                avg_costs.append(avg_cost)

        except FileNotFoundError:
            print(f"Ошибка: Файл не найден - {file_path}")
            correct_counts.append(0)
            avg_costs.append(0)
        except json.JSONDecodeError:
            print(f"Ошибка: JSON не удалось декодировать в файле {file_path}")
            correct_counts.append(0)
            avg_costs.append(0)
        except Exception as e:
            print(f"Ошибка при обработке файла {file_path}: {e}")
            correct_counts.append(0)
            avg_costs.append(0)

        labels.append(os.path.basename(file_path).replace('.jsonl', ''))

    return labels, correct_counts, avg_costs


def create_bar_charts(labels, correct_counts, avg_costs, output_dir='images'):
    """
    Создает столбчатые диаграммы для количества правильных ответов и средней стоимости запросов.
    """
    sns.set_theme()
    os.makedirs(output_dir, exist_ok=True)

    # График количества правильных ответов
    plt.figure(figsize=(12, 6))
    barplot = sns.barplot(x=labels, y=correct_counts, palette="viridis")
    plt.ylabel("Количество правильных ответов")
    plt.title("Сравнение количества правильных ответов")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for bar, height in zip(barplot.patches, correct_counts):
        barplot.annotate(f'{height}', (bar.get_x() + bar.get_width() / 2, height),
                         ha='center', va='bottom', xytext=(0, 5), textcoords='offset points')
    plt.ylim(0, 250)
    # plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correct_answers_comparison.png'))
    plt.close()

    # График средней стоимости запросов
    plt.figure(figsize=(12, 6))
    plt.ylim(0, 0.0025)
    barplot_costs = sns.barplot(x=labels, y=avg_costs, palette="viridis")
    plt.ylabel("Средняя стоимость запроса")
    plt.title("Сравнение средней стоимости запросов")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for bar, height in zip(barplot_costs.patches, avg_costs):
        barplot_costs.annotate(f'{height:.5f}', (bar.get_x() + bar.get_width() / 2, height),
                               ha='center', va='bottom', xytext=(0, 5), textcoords='offset points')

    # plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'average_costs_comparison.png'))
    plt.close()

    print(f"Графики сохранены в папку '{output_dir}'")


if __name__ == "__main__":
    files = [
        'gsm8k_tools_gpt-4o-mini.jsonl',
        'gsm8k_no_tools_gpt-4o-mini.jsonl',
        'gsm8k_tools_gpt-3.5-turbo.jsonl',
        'gsm8k_no_tools_gpt-3.5-turbo.jsonl'
    ]
    labels, correct_counts, avg_costs = analyze_json_files(*files)
    create_bar_charts(['gpt-4o-mini + Tool', 'gpt-4o-mini', 'gpt-3.5-turbo + Tool', 'gpt-3.5-turbo'],
                      correct_counts, avg_costs)
    print("\nАнализ завершен.")
