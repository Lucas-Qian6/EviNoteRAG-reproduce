import os
import glob
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

def default_prefix_func(content):
    """Default prefix function"""
    return f"[PREFIX] {content}"

def process_prompt(prompt_array, question, prefix_func):
    """
    Modify the prompt field and add a prefix.
    
    Args:
        prompt_array (list): Original prompt array
        question (str): Question text
        prefix_func (function): Custom prefix function
    
    Returns:
        list: Modified prompt array
    """
    try:
        new_prompt = [
            {
                "role": item.get("role", "user"),
                "content": prefix_func(question)
            }
            for item in prompt_array
        ]
        return new_prompt
    except Exception as e:
        print(f"❌ Error occurred while processing prompt: {e}")
        return prompt_array  # Return the original data to avoid task failure

def process_parquet_files(input_dir, output_dir, prefix_func=default_prefix_func):
    """
    Automated processing of Parquet files (pure Python implementation)
    
    Args:
        input_dir (str): Input directory path
        output_dir (str): Output directory path
        prefix_func (function): Custom prefix handling function
    """
    try:
        # Recursively find all Parquet files
        file_paths = glob.glob(os.path.join(input_dir, "**/*.parquet"), recursive=True)
        if not file_paths:
            print("⚠️ No Parquet files found")
            return

        for file_path in file_paths:
            # Construct output path
            relative_path = os.path.relpath(file_path, input_dir)
            base_name, _ = os.path.splitext(relative_path)
            # output_file = os.path.join(output_dir, f"{base_name}_p.parquet")
            output_file = os.path.join(output_dir, f"{base_name}_dotraining.parquet")
            
            # Create output directories
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Read Parquet file
            table = pq.read_table(file_path)
            df = table.to_pandas()
            
            # Validate data structure
            if "prompt" not in df.columns or "question" not in df.columns:
                print(f"⚠️ Skipping file (missing required fields): {file_path}")
                continue
            
            # Add prefix
            df["prompt"] = df.apply(
                lambda row: process_prompt(row["prompt"], row["question"], prefix_func),
                axis=1
            )
            
            # Convert back to Arrow Table
            table = pa.Table.from_pandas(df)
            
            # Save result
            pq.write_table(table, output_file)
            print(f"✅ Processed and saved: {output_file}")
    
    except Exception as e:
        print(f"❌ Processing failed: {e}")

    # ## Note-Taking Rules  
    # When retrieving information enclosed in `<information>`, summarize its content and use the following markers to highlight key or uncertain elements:

    # There are two types of markers:
    # 1. `-` (Uncertainty): Marks ambiguous or uncertain information.  
    # Example: `-He picked up Jenny-` (Uncertain who "he" refers to).
    # 2. `*` (Key Info): Highlights important or critical details.  
    # Example: *Built in 1900* (The year is essential).

    #         return f"""
    # ## Background Information  
    # # Role Definition  
    # You are a specialized **Information Retrieval Agent**. Perform reasoning and use the search tool before providing the final answer.
    # You should continue searching until all the required information has been retrieved, and then provide the final answer.

    # When retrieving information enclosed in `<information>`, organize its content into atomic claims and the relationships between them, then write the result in a `<summary>` block.

    # Steps:
    # 1. **Decompose**: extract each distinct factual statement as a separate claim, labeled with its source (e.g. `C1 [Doc1]`).
    # 2. **Filter**: drop claims that are not relevant to the question.
    # 3. **Relate**: for pairs of relevant claims, mark one of:
    #    - **corroborates**: claims reinforce each other (same or near-identical fact).
    #    - **contradicts**: claims conflict or give incompatible information.
    #    - **complements**: claims address different aspects of the question without overlap.
    #    - **subsumes**: one claim is strictly more specific than the other on the same point.
    # 4. **Resolve**: merge corroborating claims into one; on contradiction prefer the more specific claim and note the conflict; keep complementary claims as separate points; on subsumption keep the more specific claim and drop the generic one.

    # ## Format Instructions  
    # - Use `<search>Your query</search>` to call the search tool.
    # - For each `<information>Search result</information>`, provide a structured claim-level summary inside `<summary>`, following the steps above.
    # - Only output the final answer inside `<answer></answer>`. Do not include explanations, reasoning, or extra text.
    # - If it's a yes/no question, respond only with `yes` or `no`.
    # - Always follow this format strictly.
    # - **Answer must be in English. Only English responses will be accepted.**
    # Note: No searches allowed after answer submission. So avoid answering when uncertain – verify accuracy thoroughly before answering
    # Question: {question}
    # """

    #         When retrieving information enclosed in `<information>`, organize its content into atomic claims and the relationships between them, then write the result in a `<summary>` block following the steps below:
    #     The <summary> block should:
    #     - Decompose useful retrieved evidence into atomic claims, labeled by source, such as C1 [Doc1].
    #     - Filter out claims that are irrelevant to the question.
    #     - Relate important claims using: corroborates, contradicts, complements, or subsumes.
    #     - Resolve:
    #     For each relation:
    #         - corroborates: merge the claims into one concise claim.
    #         - complements: keep both claims if both help answer the question.
    #         - subsumes: keep the more specific claim and drop the generic one.
    #         - contradicts:
    #         a. Check whether the claims differ only because of time, location, entity, definition, or scope.
    #             If so, keep both with their scope clearly stated.
    #         b. If they truly conflict, choose one only if the evidence clearly supports it more.
    #             Evidence is stronger when it is directly relevant to the question, supported by multiple documents,
    #             more specific to the asked entity/time/scope, or from a more relevant document.
    #         c. If there is no clear reason to choose, mark the conflict as UNRESOLVED.
    #             Do not invent a resolution.
    #             Search for more evidence to resolve the conflict.


if __name__ == "__main__":
    def my_custom_prefix(question):
        """prefix function"""
        question = question.strip()
        if question[-1] != '?':
            question += '?'
        return f"""
        ## Background Information  
        # Role Definition  
        You are a specialized **Information Retrieval Agent**. Perform reasoning and use the search tool before providing the final answer.
        You should continue searching until all the required information has been retrieved, and then provide the final answer.

        ## Note-Taking Rules
        When retrieving information enclosed in `<information>`, write claim-level notes in a <summary> block:
        - Decompose retrieved evidence into atomic factual claims, labeled by source (e.g., Claim: fact [Doc1]).
        - When claims from different sources agree, merge them into one claim citing both sources.
        - When claims conflict, note the conflict and assess which is more specific. If unresolved, search for more evidence.

        ## Format Instructions  
        - Use `<search>Your query</search>` to call the search tool.
        - For each `<information>Search result</information>`, provide a structured claim-level summary inside `<summary>`, following the steps above.
        - Only output the final answer inside `<answer></answer>`. Do not include explanations, reasoning, or extra text.
        - If it's a yes/no question, respond only with `yes` or `no`.
        - Always follow this format strictly.
        - **Answer must be in English. Only English responses will be accepted.**
        Note: No searches allowed after answer submission. So avoid answering when uncertain – verify accuracy thoroughly before answering
        Question: {question}
        """

    input_directory = "./data/"
    output_directory = "./data/"

    process_parquet_files(input_directory, output_directory, my_custom_prefix)
