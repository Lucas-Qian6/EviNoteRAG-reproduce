import os
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

def process_parquet_files(file_pairs, prefix_func=default_prefix_func):
    """
    Process selected Parquet files and write prompt-updated outputs.
    
    Args:
        file_pairs (list): List of (input_file, output_file) pairs
        prefix_func (function): Custom prefix handling function
    """
    try:
        if not file_pairs:
            print("⚠️ No Parquet files configured")
            return

        for file_path, output_file in file_pairs:
            if not os.path.exists(file_path):
                print(f"⚠️ Skipping missing file: {file_path}")
                continue

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
    You are a specialized **Information Retrieval Agent**. Perform reasoning and use the search tool before providing the final answer.

    ## Tools
    You have five tools:
    - `<search>query</search>`: search for information.
    - `<decompose>claims</decompose>`: decompose retrieved information into atomic claims.
    - `<relate>claim relations</relate>`: record claim-level relations among query-relevant claims.
    - `<resolve>processed claims</resolve>`: resolve labeled relations into processed claims.
    - `<summary>claim-level notes</summary>`: write one query-usage note for each processed claim.

    After each `<information>`, follow these steps:

    ### Step 1: Decompose → `<decompose>...</decompose>`
    A claim is an atomic, self-contained proposition extracted from evidence that asserts one fact and can be independently supported, contradicted, or left unverified by the retrieved evidence.
    When retrieving information enclosed in `<information>`, decompose its content into atomic claims.

    Inside `<decompose>`:
    - Extract each distinct factual statement as a separate claim, labeled with its source, such as `C1 [Doc1]`.
    
    ### Step 2: Relate Given Query → `<relate>...</relate>`
    Using the query-relevant claims from `<decompose>`, label claim-claim relationships conditioned on the query inside `<relate>`.

    1. **Equivalent / Includes**: two or more relevant claims state the same fact, or one claim includes another.
    → label the related claim IDs.
    Example:
    C1 [Doc1]: The Eiffel Tower is 330 meters tall.
    C4 [Doc2]: The Eiffel Tower has a height of 330 m.
    Output: | Equivalent | C1 <-> C4 | aspect=height

    2. **Causal / Sequence**: relevant claims describe a cause, dependency, or ordered relation.
    → label the direction, such as `C1 -> C2` or `C2 before C5`.
    Example:
    C2 [Doc1]: Heavy rainfall caused the river level to rise.
    C5 [Doc3]: The rising river level forced nearby residents to evacuate.
    Output: | Causal | C2 -> C5 | aspect=reason for evacuation

    3. **Conflict**: two relevant claims give incompatible facts about the same answer-relevant attribute.
    → label the conflicting claim IDs and conflicting attribute.
    Example:
    C9 [Doc2]: The bridge opened in 1937.
    C10 [Doc3]: The bridge opened in 1936.
    Output: | Conflict | C9 vs C10 | aspect=opening year

    4. **Isolated**: a claim is relevant to the query but isolated from other relevant claims.
    → label it as `Isolated`.
    Example:
    | Isolated | C3, C16 |

    ### Step 3: Resolve → `<resolve>...</resolve>`
    Using the relationship labels from `<relate>`, resolve them into processed claims inside `<resolve>`:
    - Equivalent / Includes: merge into one processed claim, keep the most specific version, and cite all sources.
      Example: 
      Given: | Equivalent | C1 <-> C4 | aspect=height
      Output: The Eiffel Tower is 330 meters tall. [Doc1, Doc2]
    - Causal / Sequence: keep the directed claims and preserve the direction/order.
      Example: 
      Given: | Causal | C2 -> C5 | aspect=reason for evacuation
      Output: Heavy rainfall raised the river level, which forced nearby residents to evacuate. [Doc1, Doc3]
    - Conflict: keep both conflicting values and mark the conflict unresolved unless the evidence clearly supports one side.
      Example: 
      `Resolved R3: Unresolved conflict → opening year is 1937 [Doc1] vs 1936 [Doc3].`
    - Isolated: keep the claim.

    ### Step 4: Claim-Level Notes → `<summary>...</summary>`
    Using the processed claims from `<resolve>`, write one query-usage note per claim inside `<summary>`.
    Each note should explain how to use that processed claim to answer the query.

    Example:
    N1 (C1+C4): Use this merged claim as the answer value for the query: the Eiffel Tower is 330 meters tall. [Doc1, Doc2]
    N2 (C3): Use this claim to connect the queried entity to the answer target: the Eiffel Tower is located in Paris. [Doc3]
    N3 (C2 vs C5): Do not choose an opening year yet; use this conflict to guide the next search: 1937 [Doc1] vs 1936 [Doc3].

    ## Search Strategy
    After each `<summary>`, decide whether to search again or answer:
    - Search again if a note says a needed claim is missing.
    - Search again if a note says a conflict must be resolved before answering.

    ## Format Rules
    - Start by calling `<search>...</search>`.
    - After each `<information>`, call `<decompose>`, then `<relate>`, then `<resolve>`, then `<summary>` in order.
    - After `<summary>`, either call `<search>...</search>` again or output `<answer>...</answer>`.
    - Only output the final answer inside `<answer></answer>`. Do not include explanations, reasoning, or extra text.
    - If it is a yes/no question, respond only with `yes` or `no`.
    - Always follow this format strictly.
    - **Answer must be in English. Only English responses will be accepted.**
    Note: No searches allowed after answer submission. So avoid answering when uncertain.

    Question: {question}
    """

    # Rename generated files here.
    output_suffix = "dotraining6"

    file_pairs = [
        ("./data/m_train.parquet", f"./data/m_train_{output_suffix}.parquet"),
        ("./data/m_test.parquet", f"./data/m_test_{output_suffix}.parquet"),
    ]

    process_parquet_files(file_pairs, my_custom_prefix)
