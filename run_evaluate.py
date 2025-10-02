import asyncio
import pandas as pd
import evaluate
from dotenv import load_dotenv

# Import the SearchEngine from your project's core logic
from core.search_engine import SearchEngine

def calculate_retrieval_metrics(retrieved_urls: list, ground_truth_urls: list) -> dict:
    """Calculates precision, recall, and F1 score for the retrieval step."""
    # Ensure lists are not empty and handle strings
    if not retrieved_urls: retrieved_urls = []
    if not ground_truth_urls: ground_truth_urls = []
    
    retrieved_set = set(retrieved_urls)
    ground_truth_set = set(ground_truth_urls)

    true_positives = len(retrieved_set.intersection(ground_truth_set))
    false_positives = len(retrieved_set - ground_truth_set)
    false_negatives = len(ground_truth_set - retrieved_set)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1_score": f1_score}


async def run_evaluation():
    """
    Loads a manual dataset, runs each query through the search engine,
    calculates metrics, and saves the results to a CSV file.
    """
    print("--- Starting Evaluation ---")
    
    # 1. Initialize your system
    load_dotenv()
    engine = SearchEngine()
    rouge_metric = evaluate.load('rouge')
    
    # 2. Load your manual dataset
    try:
        dataset_df = pd.read_csv("summaries.csv")
    except FileNotFoundError:
        print("ERROR: 'my_test_data.csv' not found. Please create it first.")
        return

    results_list = []

    print(f"Running evaluation on {len(dataset_df)} examples from your CSV...")
    for index, row in dataset_df.iterrows():
        query = row['query']
        reference_summary = row['reference_summary']
        # Split the URLs string into a list
        ground_truth_urls = row['relevant_urls'].split(';') if isinstance(row['relevant_urls'], str) else []
        
        print(f"Processing query {index+1}/{len(dataset_df)}: {query[:80]}...")
        
        # Get the list of links the fetcher found
        retrieved_links_dicts = await engine.web_fetcher.search_google_api(query)
        if not retrieved_links_dicts:
             retrieved_links_dicts = await engine.web_fetcher.search_ddg(query)
        retrieved_urls = [d['href'] for d in retrieved_links_dicts]
        
        # Get the generated summary
        result_data = await engine.search(query=query, age_group='adult')
        generated_summary = result_data['summary']

        # Calculate ROUGE scores
        rouge_scores = rouge_metric.compute(predictions=[generated_summary], references=[reference_summary])
        
        # Calculate retrieval metrics
        retrieval_scores = calculate_retrieval_metrics(retrieved_urls, ground_truth_urls)

        results_list.append({
            'query': query,
            'reference_summary': reference_summary,
            'generated_summary': generated_summary,
            'retrieved_urls': ";".join(retrieved_urls),
            'ground_truth_urls': ";".join(ground_truth_urls),
            'rougeL': rouge_scores['rougeL'],
            'precision': retrieval_scores['precision'],
            'recall': retrieval_scores['recall'],
            'f1_score': retrieval_scores['f1_score'],
        })

    print("--- Evaluation Complete ---")
    results_df = pd.DataFrame(results_list)
    
    # Calculate and print average scores
    avg_rougeL = results_df['rougeL'].mean()
    avg_precision = results_df['precision'].mean()
    avg_recall = results_df['recall'].mean()
    avg_f1 = results_df['f1_score'].mean()
    
    print(f"\n--- Average Scores ---")
    print(f"Summarization Quality (ROUGE-L): {avg_rougeL:.4f}")
    print(f"Retrieval Precision:               {avg_precision:.4f}")
    print(f"Retrieval Recall:                  {avg_recall:.4f}")
    print(f"Retrieval F1-Score:                {avg_f1:.4f}\n")
    
    output_path = "evaluation_results.csv"
    results_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Full results saved to {output_path}")

if __name__ == "__main__":
    asyncio.run(run_evaluation())