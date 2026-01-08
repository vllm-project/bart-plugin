#!/usr/bin/env python3
"""
Evaluate BART model on CNN/DailyMail summarization task.

This script evaluates facebook/bart-large-cnn on the CNN/DailyMail dataset
and computes ROUGE scores for comparison with published benchmarks.

Reference results for facebook/bart-large-cnn on CNN/DailyMail (test set):
- ROUGE-1: 44.16
- ROUGE-2: 21.28
- ROUGE-L: 40.90

"""
import argparse
import time
from typing import List

# import vllm_bart_plugin
from datasets import load_dataset
from vllm import LLM, SamplingParams
from vllm.sampling_params import BeamSearchParams
from rouge_score import rouge_scorer
from transformers import AutoTokenizer


def prepare_inputs(articles: List[str]) -> List[dict]:
    """
    Prepare inputs in the format required by vLLM BART plugin.

    Args:
        articles: List of article texts to summarize

    Returns:
        List of input dictionaries with encoder/decoder prompts
    """
    inputs = []
    for article in articles:
        inputs.append({
            "encoder_prompt": {
                "prompt": "",
                "multi_modal_data": {
                    "text": article,
                },
            },
            "decoder_prompt": "<s>",  # BOS token to start generation
        })
    return inputs


def compute_rouge(predictions: List[str], references: List[str]) -> dict:
    """
    Compute ROUGE scores.

    Args:
        predictions: List of generated summaries
        references: List of reference summaries

    Returns:
        Dictionary with ROUGE-1, ROUGE-2, and ROUGE-L F1 scores
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)

    return {
        'rouge1': sum(rouge1_scores) / len(rouge1_scores) * 100,
        'rouge2': sum(rouge2_scores) / len(rouge2_scores) * 100,
        'rougeL': sum(rougeL_scores) / len(rougeL_scores) * 100,
    }


def main():
    """Run BART evaluation on CNN/DailyMail."""
    parser = argparse.ArgumentParser(description='Evaluate BART on CNN/DailyMail')
    parser.add_argument(
        '--model',
        type=str,
        default='facebook/bart-large-cnn',
        help='Model name (default: facebook/bart-large-cnn)'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['validation', 'test'],
        help='Dataset split to evaluate (default: test)'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=None,
        help='Number of samples to evaluate (default: all)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for inference (default: 8)'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=142,
        help='Maximum tokens to generate (default: 142)'
    )
    parser.add_argument(
        '--min-tokens',
        type=int,
        default=30,
        help='Minimum tokens to generate (default: 30)'
    )
    parser.add_argument(
        '--length-penalty',
        type=float,
        default=2.0,
        help='Length penalty for beam search (default: 2.0)'
    )
    parser.add_argument(
        '--num-beams',
        type=int,
        default=4,
        help='Number of beams for beam search (default: 4)'
    )
    parser.add_argument(
        '--gpu-memory-utilization',
        type=float,
        default=0.8,
        help='GPU memory utilization (default: 0.9)'
    )

    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"Loading {args.model}...")
    llm = LLM(
        model=args.model,
        enforce_eager=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="float16",
    )

    # Sampling params matching BART paper settings
    sampling_params = SamplingParams(
        temperature=0.0,  # Greedy decoding
        max_tokens=args.max_tokens,
        min_tokens=args.min_tokens,
    )
    # TODO once beam search is integrated
    # beam_params = BeamSearchParams(
    #     beam_width=args.num_beams, 
    #     max_tokens=args.max_tokens,     
    #     length_penalty=args.length_penalty,
    # )

    print(f"Loading CNN/DailyMail {args.split} set...")
    dataset = load_dataset("cnn_dailymail", "3.0.0", split=args.split)

    if args.num_samples:
        dataset = dataset.select(range(min(args.num_samples, len(dataset))))
    
    # Filter out articles that are too long
    # NOTE we filter out articles that are too long to fit in the model context window
    max_article_length = 1024 - args.max_tokens
    print(f"Filtering articles longer than {max_article_length} tokens...")
    original_size = len(dataset)
    dataset = dataset.filter(lambda x: len(tokenizer(x['article'], add_special_tokens=False)['input_ids']) <= max_article_length)
    filtered_size = len(dataset)
    print(f"Kept {filtered_size}/{original_size} samples ({filtered_size/original_size*100:.1f}%)")

    print(f"Evaluating on {len(dataset)} samples...")

    # Process in batches
    all_predictions = []
    all_references = []

    start_time = time.time()

    for i in range(0, len(dataset), args.batch_size):
        batch = dataset[i:min(i + args.batch_size, len(dataset))]
        articles = batch['article']
        highlights = batch['highlights']

        # Prepare inputs
        inputs = prepare_inputs(articles)

        # Generate summaries
        outputs = llm.generate(inputs, sampling_params=sampling_params)


        # Extract generated text
        predictions = [o.outputs[0].text.strip() for o in outputs]

        all_predictions.extend(predictions)
        all_references.extend(highlights)

        if (i // args.batch_size + 1) % 10 == 0:
            elapsed = time.time() - start_time
            samples_per_sec = (i + len(articles)) / elapsed
            print(f"Processed {i + len(articles)}/{len(dataset)} samples ({samples_per_sec:.2f} samples/sec)")

    total_time = time.time() - start_time

    # Compute ROUGE scores
    print("\nComputing ROUGE scores...")
    rouge_scores = compute_rouge(all_predictions, all_references)

    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Dataset: CNN/DailyMail {args.split}")
    print(f"Samples: {len(dataset)}")
    print(f"Total time: {total_time:.2f}s ({len(dataset)/total_time:.2f} samples/sec)")
    print("-"*60)
    print(f"ROUGE-1: {rouge_scores['rouge1']:.2f}")
    print(f"ROUGE-2: {rouge_scores['rouge2']:.2f}")
    print(f"ROUGE-L: {rouge_scores['rougeL']:.2f}")
    print("-"*60)
    print("\nReference scores for facebook/bart-large-cnn (from original paper, https://arxiv.org/pdf/1910.13461):")
    print("ROUGE-1: 44.16")
    print("ROUGE-2: 21.28")
    print("ROUGE-L: 40.90")
    print("="*60)

    # Print a few examples
    print("\nExample outputs (first 3):")
    for i in range(min(3, len(all_predictions))):
        print(f"\n--- Example {i+1} ---")
        print(f"Article (first 200 chars): {dataset[i]['article'][:200]}...")
        print(f"\nReference: {all_references[i]}")
        print(f"\nGenerated: {all_predictions[i]}")


if __name__ == "__main__":
    main()
