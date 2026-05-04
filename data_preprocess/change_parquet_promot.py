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
            output_file = os.path.join(output_dir, f"{base_name}_dotraining3.parquet")
            
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
    ## Role
    You are an Information Retrieval Agent. Use the search tool to find evidence before answering.

    ## Claim-Level Evidence Reasoning
    Evidence in `<information>` is already decomposed into atomic claims, such as:
    C1 [Doc1]: ...
    C2 [Doc2]: ...

    After each `<information>`, write a `<summary>` by reasoning over the claims.

    ### Step 1: Filter
    Drop claims that are irrelevant to the question.

    ### Step 2: Relate and Resolve
    For relevant claims, identify their relationships and resolve them as follows:

    1. **corroborates**: two claims state the same fact.
    Action: merge them into one note and cite both sources.
    Example:
    C1 [Doc1]: The Eiffel Tower is 330 meters tall.
    C4 [Doc2]: The Eiffel Tower has a height of 330 m.
    Summary note:
    *Answer* The Eiffel Tower is 330 meters tall. [Doc1, Doc2]

    2. **contradicts**: two claims give incompatible facts about the same entity, time, or attribute.
    Action: keep both values and mark the note as uncertain. Do not guess which one is correct.
    Example:
    C2 [Doc1]: The bridge opened in 1937.
    C5 [Doc3]: The bridge opened in 1936.
    Summary note:
    -Noise/Uncertain- Opening year conflict: 1937 [Doc1] vs 1936 [Doc3].

    3. **complements**: claims provide different useful facts that together help answer the question.
    Action: keep them as separate notes.
    Example:
    C1 [Doc1]: Marie Curie discovered radium.
    C3 [Doc2]: Marie Curie won the Nobel Prize in Chemistry in 1911.
    Summary notes:
    *Bridge* Marie Curie discovered radium. [Doc1]
    *Answer* Marie Curie won the Nobel Prize in Chemistry in 1911. [Doc2]

    4. **subsumes**: one claim contains all useful information from another claim plus more detail.
    Action: keep only the more detailed claim.
    Example:
    C2 [Doc1]: Prague is in Europe.
    C6 [Doc3]: Prague is the capital of the Czech Republic in Central Europe.
    Summary note:
    *Answer* Prague is the capital of the Czech Republic in Central Europe. [Doc3]

    ### Step 3: Tag Resolved Notes
    Use only these markers:
    - `*Answer*`: a claim that directly supports the final answer.
    - `*Bridge*`: an intermediate claim needed to connect the question to the answer.
    - `-Noise/Uncertain-`: irrelevant, conflicting, ambiguous, or weak evidence.

    ## Search Strategy
    Before issuing each new search, do the following:

    1. Break the original question into the sub-questions it implies (e.g., who, what, when, where, how many).
    2. Review the `*Answer*` and `*Bridge*` notes accumulated across **all previous summaries**.
    3. Identify which sub-question is least covered or still uncertain.
    4. Write a query that targets exactly that gap — not the whole question again.

    Only search for the missing piece. Do not repeat a broad query if partial evidence already exists.
    If a `-Noise/Uncertain-` conflict affects the answer, the next search should target resolving that specific conflict.

    ## Format Instructions
    - Use `<search>Your query</search>` to call the search tool.
    - After each `<information>...</information>`, write one `<summary>...</summary>`.
    - When ready, output the final answer only inside `<answer></answer>`.
    - If it is a yes/no question, respond only with `yes` or `no`.
    - Always follow this format strictly.
    - Answer must be in English.

    No searches allowed after answer submission.

    Question: {question}
    """

    input_directory = "./data/"
    output_directory = "./data/"

    process_parquet_files(input_directory, output_directory, my_custom_prefix)
