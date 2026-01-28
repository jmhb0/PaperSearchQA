"""
python -m ipdb data_gen/generate_questions_from_abstracts.py
Options 


Script to generate question-answer datasets from PubMed abstracts using GPT.

PREREQUISITES:
    Before running this script, you need to create the PubMed abstracts database:
    1. Run: python data_gen/allMesh_to_parquet.py
    2. This will create the required file: data/allMeSH_2022.parquet


This script:
1. Loads PubMed abstracts from the parquet database
2. Samples n_samples abstracts randomly
3. Processes them in batches using GPT-4 to generate Q&A pairs
4. Generates golden_answers (synonyms) for each answer
5. Creates train/test split and pushes to HuggingFace Hub
"""

import os
import pandas as pd
import numpy as np
import csv
from typing import List, Dict, Any
import random
import json
from tqdm import tqdm
import re
from datasets import Dataset, DatasetDict
import ipdb

from allMesh_to_parquet import return_indexer
from api import call_llm, call_llm_batch

# Configuration for HuggingFace uploads
# Set ENABLE_HF_UPLOAD=1 to explicitly enable uploads (requires HF_USERNAME and HF_TOKEN)
ENABLE_HF_UPLOAD = os.environ.get("ENABLE_HF_UPLOAD", "0") == "1"
HF_USERNAME = os.environ.get("HF_USERNAME", "your_username")  # Used for dataset naming
OUTPUT_DIR = "output"

# Template definitions
TEMPLATE_1 = """
BACKGROUND
You are a domain-expert biomedical NLP assistant.
You are helping me to create an open-domain QA dataset. 
The downstream task will read a query and require an agent to search over Pubmed abstracts

--------
YOUR TASK 
I will provide you with title and abstract of a Pubmed article. 
Your task is to create a new question-answer pair. 

--------
TYPES OF QUESTIONS
The questions should be 'factoid based'. 
The answer should be a simple entity. 
It should not be ambiguous.
Don't be pretentious. 

--------
IMPORTANT NOTES
The question-answer pair will be used to evaluation question-answering systems with retrieval. Ths means the target system does not know which paper the question was sourced from. So an inappropriate question would be "What technology is used in this study to ...".
If the question contains acronyms that are not well known, then explain the acronym.

--------
EXAMPLE CATEGORIES 
Below are sample categories with sample questions. 

Category: 1 - Genetic inheritance & disease-linked mutations
question: What gene is mutated in Sickle Cell Anemia?
answer: HBB
question: Which ultraconserved element is associated with Embryonic Stem Cells (ESC) self-renewal?
answer: T-UCstem1
question: Is Huntington's disease caused by a dominate or recessive gene?
answer: dominant

Category: 2 - Therapeutics, indications & clinical evidence
question: What is the most effective drug for oxaliplatin-induced neuropathy?
answer: Duloxetine
question: Which cancer is the BCG vaccine used for?
answer: Non-muscle Invasive Bladder Cancer
question: How many injections of CLS-TA did the patients participating in the PEACHTREE trial receive?
answer: two

Category: 3 - Protein function, localization & signalling/enzymatic interactions
question: Which histone mark distinguishes active from inactive enhancers?
answer: H3K27ac
question: Which component of the Influenza A Virus affects mRNA transcription termination?
answer: NS1
question: Which is the main calcium binding protein of the sarcoplasmic reticulum?
answer: Calsequestrin

Category: 4 - Experimental & computational methods, resources & acronyms
question: Which algorithm has been proposed for efficient storage of WGS variant calls?
answer: SeqArray
question: What is an acceptable sequence coverage(depth) required for human whole-exome sequencing?
answer: 30x-60x

Category: 5 - Disease causation & pathogens
question: Which is the most common disease attributed to malfunction or absence of primary cilia?
answer: ['Polycystic kidney disease', 'PKD']
question: What organism causes scarlet fever also known as scarletina?
answer: ['Group A Streptococcus', 'Streptococcus pyogenes']
question: The pathogen Fusarium graminearum affects what type of plant species?
answer: cereal crops

Category: 6 - Biomarkers & diagnostic tests
question: Salivary Cortisol is a biomarker for what disease/syndrome/condition?
answer: stress
question: What is the gold standard for a diagnosis of narcolepsy?
answer: ['Sleep study', 'overnight polysomnography']

Category: 7 - Bioinformatics databases & curated resources
question: Which R/bioconductor package has been developed to aid in epigenomic analysis?
answer: DeepBlueR
question: Which database associates human noncoding SNPs with their three-dimensional interacting genes?
answer: 3DSNP
question: What is the RESID database?
question: Which is the literature-based database of phenotypes?
answer: PheneBank

Category: 8 - Clinical grading & diagnostic scales / classification systems
question: What can be predicted with the Wells criteria?
answer: pulmonary embolism
question: Symptoms of which disorder are evaluated with the Davidson Trauma Scale?
answer: ['post-traumatic stress disorder', 'PTSD']
question: Which value of nuchal translucency thickness is set as the threshold for high-risk for Down Syndrome?
answer: 3mm

Category: 9 - Anatomical / cellular structures & localisation
question: Where is corticosterone synthesized?
answer: Adrenal glands
question: Which is the chromosome area that the human gene coding for the dopamine transporter (DAT1) is located to?
answer: 5p15.3
question: Where is the respirasome located?
answer: inner mitochondrial membrane

--------

OUTPUT FORMAT
Return question inside tag `<question>...</question>`, answer inside `<answer>...</answer>`. 
If the QA corresponde to one of the above categories put its number in <cat_num>...</cat_num> and category description in <cat>...</cat>

--------
TITLE AND ABSTRACT
{title_abstract}
"""

# this template is adjusted after Duo's feedback
TEMPLATE_2 = """
BACKGROUND
You are a domain-expert biomedical NLP assistant.
You are helping me to create an open-domain QA dataset. 
The downstream task will read a query and require an agent to search over Pubmed abstracts

--------
YOUR TASK 
I will provide you with title and abstract of a Pubmed article. 
Your task is to create a new question-answer pair. 

--------
TYPES OF QUESTIONS
The questions should be 'factoid based'. 
The answer should be a simple entity. 
It should not be ambiguous.
Don't be pretentious. 

--------
IMPORTANT NOTES
The question-answer pair will be used to evaluation question-answering systems with retrieval. Ths means the target system does not know which paper the question was sourced from. So an inappropriate question would be "What technology is used in this study to ...". or "what type of treatment is assessed in this study?" (where the study name is not specifified).
If the question contains acronyms that are not well known, then explain the acronym.

--------
EXAMPLE CATEGORIES 
Below are sample categories with sample questions. 

Category: 1 - Genetic inheritance & disease-linked mutations
question: What gene is mutated in Sickle Cell Anemia?
answer: HBB
question: Which ultraconserved element is associated with Embryonic Stem Cells (ESC) self-renewal?
answer: T-UCstem1
question: Is Huntington's disease caused by a dominate or recessive gene?
answer: dominant

Category: 2 - Therapeutics, indications & clinical evidence
question: What is the most effective drug for oxaliplatin-induced neuropathy?
answer: Duloxetine
question: Which cancer is the BCG vaccine used for?
answer: Non-muscle Invasive Bladder Cancer
question: How many injections of CLS-TA did the patients participating in the PEACHTREE trial receive?
answer: two

Category: 3 - Protein function, localization & signalling/enzymatic interactions
question: Which histone mark distinguishes active from inactive enhancers?
answer: H3K27ac
question: Which component of the Influenza A Virus affects mRNA transcription termination?
answer: NS1
question: Which is the main calcium binding protein of the sarcoplasmic reticulum?
answer: Calsequestrin

Category: 4 - Experimental & computational methods, resources & acronyms
question: Which algorithm has been proposed for efficient storage of WGS variant calls?
answer: SeqArray
question: What is an acceptable sequence coverage(depth) required for human whole-exome sequencing?
answer: 30x-60x

Category: 5 - Disease causation & pathogens
question: Which is the most common disease attributed to malfunction or absence of primary cilia?
answer: ['Polycystic kidney disease', 'PKD']
question: What organism causes scarlet fever also known as scarletina?
answer: ['Group A Streptococcus', 'Streptococcus pyogenes']
question: The pathogen Fusarium graminearum affects what type of plant species?
answer: cereal crops

Category: 6 - Biomarkers & diagnostic tests
question: Salivary Cortisol is a biomarker for what disease/syndrome/condition?
answer: stress
question: What is the gold standard for a diagnosis of narcolepsy?
answer: ['Sleep study', 'overnight polysomnography']

Category: 7 - Bioinformatics databases & curated resources
question: Which R/bioconductor package has been developed to aid in epigenomic analysis?
answer: DeepBlueR
question: Which database associates human noncoding SNPs with their three-dimensional interacting genes?
answer: 3DSNP
question: What is the RESID database?
question: Which is the literature-based database of phenotypes?
answer: PheneBank

Category: 8 - Clinical grading & diagnostic scales / classification systems
question: What can be predicted with the Wells criteria?
answer: pulmonary embolism
question: Symptoms of which disorder are evaluated with the Davidson Trauma Scale?
answer: ['post-traumatic stress disorder', 'PTSD']
question: Which value of nuchal translucency thickness is set as the threshold for high-risk for Down Syndrome?
answer: 3mm

Category: 9 - Anatomical / cellular structures & localisation
question: Where is corticosterone synthesized?
answer: Adrenal glands
question: Which is the chromosome area that the human gene coding for the dopamine transporter (DAT1) is located to?
answer: 5p15.3
question: Where is the respirasome located?
answer: inner mitochondrial membrane

Category: 10 - Psychology and behavioral health
Question: Which psychomotor domain showed a significant difference between institutionalized and non-institutionalized sheltered children and adolescents?
Answer: Body awareness
Question: What ethical principle justifies actions that have both good and harmful effects, as long as the harm is not intended but only foreseen?
Answer: Rule of Double Effect
Questions: What psychological process during an incubation period is associated with enhanced creative problem solving?
Answer: Mind-wandering

--------

OUTPUT FORMAT
Return question inside tag `<question>...</question>`, answer inside `<answer>...</answer>`. 
If the QA corresponde to one of the above categories put its number in <cat_num>...</cat_num> and category description in <cat>...</cat>

--------
TITLE AND ABSTRACT
{title_abstract}
"""

TEMPLATE_3 = """
BACKGROUND
You are a domain-expert biomedical NLP assistant.
You are helping me to create an open-domain QA dataset. 
The downstream task will read a query and require an agent to search over Pubmed abstracts

--------
YOUR TASK 
I will provide you with title and abstract of a Pubmed article. 
Your task is to create 3 new question-answer pairs. 

--------
TYPES OF QUESTIONS
The questions should be 'factoid based'. 
The answer should be a simple entity. 
It should not be ambiguous.
Don't be pretentious. 

--------
IMPORTANT NOTES
The question-answer pair will be used to evaluation question-answering systems with retrieval. Ths means the target system does not know which paper the question was sourced from. So an inappropriate question would be "What technology is used in this study to ...". or "what type of treatment is assessed in this study?" (where the study name is not specifified).
If the question contains acronyms that are not well known, then explain the acronym.

--------
EXAMPLE CATEGORIES 
Below are sample categories with sample questions. 

Category: 1 - Genetic inheritance & disease-linked mutations
question: What gene is mutated in Sickle Cell Anemia?
answer: HBB
question: Which ultraconserved element is associated with Embryonic Stem Cells (ESC) self-renewal?
answer: T-UCstem1
question: Is Huntington's disease caused by a dominate or recessive gene?
answer: dominant

Category: 2 - Therapeutics, indications & clinical evidence
question: What is the most effective drug for oxaliplatin-induced neuropathy?
answer: Duloxetine
question: Which cancer is the BCG vaccine used for?
answer: Non-muscle Invasive Bladder Cancer
question: How many injections of CLS-TA did the patients participating in the PEACHTREE trial receive?
answer: two

Category: 3 - Protein function, localization & signalling/enzymatic interactions
question: Which histone mark distinguishes active from inactive enhancers?
answer: H3K27ac
question: Which component of the Influenza A Virus affects mRNA transcription termination?
answer: NS1
question: Which is the main calcium binding protein of the sarcoplasmic reticulum?
answer: Calsequestrin

Category: 4 - Experimental & computational methods, resources & acronyms
question: Which algorithm has been proposed for efficient storage of WGS variant calls?
answer: SeqArray
question: What is an acceptable sequence coverage(depth) required for human whole-exome sequencing?
answer: 30x-60x

Category: 5 - Disease causation & pathogens
question: Which is the most common disease attributed to malfunction or absence of primary cilia?
answer: ['Polycystic kidney disease', 'PKD']
question: What organism causes scarlet fever also known as scarletina?
answer: ['Group A Streptococcus', 'Streptococcus pyogenes']
question: The pathogen Fusarium graminearum affects what type of plant species?
answer: cereal crops

Category: 6 - Biomarkers & diagnostic tests
question: Salivary Cortisol is a biomarker for what disease/syndrome/condition?
answer: stress
question: What is the gold standard for a diagnosis of narcolepsy?
answer: ['Sleep study', 'overnight polysomnography']

Category: 7 - Bioinformatics databases & curated resources
question: Which R/bioconductor package has been developed to aid in epigenomic analysis?
answer: DeepBlueR
question: Which database associates human noncoding SNPs with their three-dimensional interacting genes?
answer: 3DSNP
question: What is the RESID database?
question: Which is the literature-based database of phenotypes?
answer: PheneBank

Category: 8 - Clinical grading & diagnostic scales / classification systems
question: What can be predicted with the Wells criteria?
answer: pulmonary embolism
question: Symptoms of which disorder are evaluated with the Davidson Trauma Scale?
answer: ['post-traumatic stress disorder', 'PTSD']
question: Which value of nuchal translucency thickness is set as the threshold for high-risk for Down Syndrome?
answer: 3mm

Category: 9 - Anatomical / cellular structures & localisation
question: Where is corticosterone synthesized?
answer: Adrenal glands
question: Which is the chromosome area that the human gene coding for the dopamine transporter (DAT1) is located to?
answer: 5p15.3
question: Where is the respirasome located?
answer: inner mitochondrial membrane

Category: 10 - Psychology and behavioral health
Question: Which psychomotor domain showed a significant difference between institutionalized and non-institutionalized sheltered children and adolescents?
Answer: Body awareness
Question: What ethical principle justifies actions that have both good and harmful effects, as long as the harm is not intended but only foreseen?
Answer: Rule of Double Effect
Questions: What psychological process during an incubation period is associated with enhanced creative problem solving?
Answer: Mind-wandering

--------

OUTPUT FORMAT
A single QA has tags `<question>...</question>`, answer inside `<answer>...</answer>`. 
If the QA corresponds to one of the above categories put its number in <cat_num>...</cat_num> and category description in <cat>...</cat>. 
Each QA should exist in its own tag <qa>...</qa>

Therefore the first 2 questions would be:
<qas>
   <qa> <question> ... </question>
      <answer> ... </answer>
      <cat_num> ... </cat_num>
      <cat> ... </cat>
   </qa>
   <qa> 
       .....
   </qa>
   ...
</qas>

--------
TITLE AND ABSTRACT
{title_abstract}
"""

TEMPLATE_4 = """
BACKGROUND  
You are a domain-expert biomedical NLP assistant helping to build an open-domain factoid QA set.  
At evaluation time, the QA system will NOT see the source article—only PubMed as a whole.

INPUT  
You receive one PubMed article (title + abstract).

TASK  
Produce **3** new question–answer pairs that meet all guidelines below.

GUIDELINES  
1. **Factoid only** – the answer is a single, unambiguous biomedical entity (gene, drug, disease, method, etc.).  
2. **Independence** – word questions so they do *not* rely on the given article being visible.  
   *Avoid phrases like "in this study", "according to the article", or any hint that a specific paper is required.*  
3. **Knowledge scope** – choose facts that are (a) stated in the abstract **and** (b) well supported elsewhere in PubMed, so retrieval is feasible.  
4. **Acronyms** – spell out uncommon acronyms on first mention, e.g. "extracorporeal membrane oxygenation (ECMO)".  
5. **Tone** – clear, direct, non-pretentious.  
6. **Category tag** – if the QA fits one of the 10 categories below, include that number and label; otherwise use 0 / "Other".

CATEGORIES  
1 Genetic mutations 2 Therapeutics & clinical evidence 3 Protein function & signalling  
4 Methods & resources 5 Disease causation & pathogens 6 Biomarkers & diagnostics  
7 Bioinformatics databases 8 Clinical scales & classifications  
9 Anatomy & cellular localisation 10 Psychology & behavioural health

OUTPUT  
Return exactly:

<qas>  
  <qa>  
    <question> … </question>  
    <answer> … </answer>  
    <cat_num> … </cat_num>  
    <cat> … </cat>  
  </qa>  
  … ×3  
</qas>

EXAMPLES  

*Good*  
Q "What imaging technique is commonly used to assess pulmonary artery involvement in Behçet's disease?"  
A "Pulmonary angiography"

*Bad*  
Q "Which chemical was used to induce lung tumours *in this study*?" ← refers to the unseen paper.

TITLE AND ABSTRACT
{title_abstract}
"""

# it's actually the same as template 4 but I want a fresh version
TEMPLATE_5 = """
BACKGROUND  
You are a domain-expert biomedical NLP assistant helping to build an open-domain factoid QA set.  
At evaluation time, the QA system will NOT see the source article—only PubMed as a whole.

INPUT  
You receive one PubMed article (title + abstract).

TASK  
Produce **3** new question–answer pairs that meet all guidelines below.

GUIDELINES  
1. **Factoid only** – the answer is a single, unambiguous biomedical entity (gene, drug, disease, method, etc.).  
2. **Independence** – word questions so they do *not* rely on the given article being visible.  
   *Avoid phrases like "in this study", "according to the article", or any hint that a specific paper is required.*  
3. **Knowledge scope** – choose facts that are (a) stated in the abstract **and** (b) well supported elsewhere in PubMed, so retrieval is feasible.  
4. **Acronyms** – spell out uncommon acronyms on first mention, e.g. "extracorporeal membrane oxygenation (ECMO)".  
5. **Tone** – clear, direct, non-pretentious.  
6. **Category tag** – if the QA fits one of the 10 categories below, include that number and label; otherwise use 0 / "Other".

CATEGORIES  
1 Genetic mutations 2 Therapeutics & clinical evidence 3 Protein function & signalling  
4 Methods & resources 5 Disease causation & pathogens 6 Biomarkers & diagnostics  
7 Bioinformatics databases 8 Clinical scales & classifications  
9 Anatomy & cellular localisation 10 Psychology & behavioural health

OUTPUT  
Return exactly:

<qas>  
  <qa>  
    <question> … </question>  
    <answer> … </answer>  
    <cat_num> … </cat_num>  
    <cat> … </cat>  
  </qa>  
  … ×3  
</qas>

EXAMPLES  

*Good*  
Q "What imaging technique is commonly used to assess pulmonary artery involvement in Behçet's disease?"  
A "Pulmonary angiography"

*Bad*  
Q "Which chemical was used to induce lung tumours *in this study*?" ← refers to the unseen paper.

TITLE AND ABSTRACT
{title_abstract}
"""

# Golden answers template
GOLDEN_ANSWERS_TEMPLATE_1 = """I am generating a dataset for biological question answering.
Below I'll give a question with its target answer.

They are questions that have simple factoid answers - meaning the answer is a single entity.
However the same entity might have synonyms that are equivalent.
Your task is to return a python list of synonyms for each unique entity mentioned in the answers.

For example, if the target answer was "c-Jun NH2-terminal kinase", then you should return a list of synonyms ["JNK", "c-Jun N-terminal kinase", "c-Jun amino-terminal kinase", "c-Jun NH2-terminal kinase"]
If the answer was "two" then you should return ["two","2"]

OTHER GUIDANCE
The answers are correct and you may not disagree with the answer. 
Upper and lower case letters are treated the same.

OUTPUT FORMAT
Return a python list of all the common synonyms. Return as many as you can think of. Include the original answer(s) in the list as well.

You must respond with valid JSON list containing only the list of synonyms.

QUESTION
{question}
ANSWER
{answer}
"""

GOLDEN_ANSWERS_TEMPLATE_2 = """
You are an expert biomedical‐ontology assistant with exhaustive knowledge of gene, protein, chemical, numeric, and clinical terminology.

TASK  
You will be given a question and its **gold answer**.  
The gold answer may contain **one or more distinct entities** (genes, proteins, chemicals, numeric values, etc.).  
For **each** entity, return **every widely used synonym, alias, abbreviation, spelling or punctuation variant** that appears in established biomedical sources (HGNC, UniProt, GeneCards, MeSH, DrugBank, PubChem, literature).  
* Include the gold answer itself.  
* Treat upper- vs lower-case, hyphens, spaces, "α/alpha" variants, and singular/plural forms as distinct synonyms if they occur in the literature.  
* Do **not** invent new terms or include database IDs (e.g. Q9Y6K9).  
* If an entity truly has < 3 known synonyms, return the ones that exist.  
* Preserve the order: start with the original answer, then common abbreviations, then longer or older names.

OUTPUT  
Return **valid JSON only**.  
Return a flat list of strings.

EXAMPLES  
• Input answer: `c-Jun NH2-terminal kinase`  
  Output is a json with key `golden_answers` and value a list of strings:  
  {{"golden_answers": ["c-Jun NH2-terminal kinase", "c-Jun N-terminal kinase", "c-Jun amino-terminal kinase", "JNK", "JNK1", "SAPK1"]}}

QUESTION
{question}
ANSWER
{answer}
"""

GOLDEN_ANSWERS_TEMPLATE_3 = """
**Biomedical Ontology QA Golden Answer Expansion Task**

You are an expert biomedical ontology assistant with exhaustive knowledge of genes, proteins, chemicals, numeric, and clinical terminology.

**TASK**  
You will be given a question and its corresponding **golden_answer**.  
The **golden_answer** may refer to one or more biomedical entities (e.g., gene, protein, chemical, disease, peptide, etc.).

For **each entity**, return **every widely used synonym, established alias, abbreviation, alternate spelling, or punctuation variant** that is found in authoritative biomedical sources (such as HGNC, UniProt, GeneCards, MeSH, DrugBank, PubChem, and peer-reviewed literature).

For **each answer**, you must:

- Include the original golden answer as given.
- Include all common abbreviations, aliases, spelling or punctuation variants found in the literature or biomedical databases.
- Include all singular and plural forms, if they are attested in the literature or resources.
- Do **not** invent new terms; only include real, attested synonyms or variants.
- Do **not** include database IDs or accessions.
- Different upper/lower case forms are **not** required, as matching will be case-insensitive.
- Preserve the order: start with the original answer, then common abbreviations, then longer or older names, then plural forms.
- If an entity has < 3 known valid synonyms, return **all that exist**.
- Return a **flat list of strings** in **valid JSON**.
- Remember, if you miss any, then very bad things will happen, so don't forget any.

**OUTPUT**  
Return valid JSON only with key "golden_answers" and value a list of strings.
Return all discovered and attested names as a flat list of strings.

**EXAMPLES**
- If the golden answer is "gamma-aminobutyric acid", accepted output includes {{"golden_answers" : ["gamma-aminobutyric acid", "GABA", "4-aminobutyric acid", "gamma-aminobutyric acids", "GABA", "GABAs"]  }}
- If the golden answer is "acetylcholine", accepted output includes {{"golden_answers" : ["acetylcholine", "ACh", "acetyl choline", "acetylcholines", "AChs"]}}

QUESTION
{question}
ANSWER
{answer}
"""

PARAPHRASE_TEMPLATE_1 = """
You are given a question that was written using a particular document as its main source. Your task is to rewrite the question so that it retains the original meaning and would result in the same correct answer, but uses different wording and phrasing. Important constraints:
Do not broaden or narrow the scope of the question.
Do not introduce ambiguity or alter clinical/technical context.
Make sure the correct answer remains exactly the same.
Your goal is to change the surface wording so that simple bag-of-words search (like BM25) may not easily match the original document, while an expert human or strong language model could still answer correctly.
Avoid copying any significant phrase (three or more words in sequence) from the original question.

Example: 
- Original: What congenital abnormality can cause unilateral hydrocephalus in the perinatal period? 
- Edited: Which birth defect present during the perinatal stage may result in hydrocephalus affecting only one side of the brain?

Output should be in tags like <question> ... </question>

Question: {question}
Answer: {answer}
"""

# Template mapping
TEMPLATES = {
    1: TEMPLATE_1,
    2: TEMPLATE_2,
    3: TEMPLATE_3,
    4: TEMPLATE_4,
    5: TEMPLATE_5
}
GOLDEN_TEMPLATES = {
    1: GOLDEN_ANSWERS_TEMPLATE_1,
    2: GOLDEN_ANSWERS_TEMPLATE_2,
    3: GOLDEN_ANSWERS_TEMPLATE_3
}
PARAPHRASE_TEMPLATES = {1: PARAPHRASE_TEMPLATE_1}

TEMPLATES_MULTIQUESTION = [3, 4, 5]


def parse_llm_response(response: str) -> Dict[str, str]:
    """
    Parse LLM response that contains XML-like tags for question, answer, category, etc.
    
    Expected format:
    <question>...</question>
    <answer>...</answer>
    <cat_num>...</cat_num>
    <cat>...</cat>
    
    Returns:
        Dict with keys: question, answer, cat_num, cat
    """
    result = {'question': '', 'answer': '', 'cat_num': '', 'cat': ''}

    # Define regex patterns for each tag
    patterns = {
        'question': r'<question>(.*?)</question>',
        'answer': r'<answer>(.*?)</answer>',
        'cat_num': r'<cat_num>(.*?)</cat_num>',
        'cat': r'<cat>(.*?)</cat>'
    }

    # Extract content from each tag
    for key, pattern in patterns.items():
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            result[key] = match.group(1).strip()

    return result


def parse_multi_qa_response(response: str) -> List[Dict[str, str]]:
    """
    Parse LLM response that contains multiple Q&A pairs in XML-like tags.
    
    Expected format:
    <qas>
       <qa> 
           <question>...</question>
           <answer>...</answer>
           <cat_num>...</cat_num>
           <cat>...</cat>
       </qa>
       <qa> 
           ...
       </qa>
       ...
    </qas>
    
    Returns:
        List of dicts, each with keys: question, answer, cat_num, cat
    """
    results = []

    # Find all <qa>...</qa> blocks
    qa_pattern = r'<qa>(.*?)</qa>'
    qa_matches = re.findall(qa_pattern, response, re.DOTALL | re.IGNORECASE)

    # Parse each Q&A block
    for qa_content in qa_matches:
        result = {'question': '', 'answer': '', 'cat_num': '', 'cat': ''}

        # Define regex patterns for each tag within a Q&A block
        patterns = {
            'question': r'<question>(.*?)</question>',
            'answer': r'<answer>(.*?)</answer>',
            'cat_num': r'<cat_num>(.*?)</cat_num>',
            'cat': r'<cat>(.*?)</cat>'
        }

        # Extract content from each tag
        for key, pattern in patterns.items():
            match = re.search(pattern, qa_content, re.DOTALL | re.IGNORECASE)
            if match:
                result[key] = match.group(1).strip()

        # Only add if we have at least a question
        if result['question']:
            results.append(result)

    return results


def process_batch(batch_data: List[Dict],
                  template: str,
                  template_key: int = None) -> List[Dict]:
    """Process a batch of abstracts through GPT using batch mode"""
    if not batch_data:
        return []

    # First, generate all prompts for the batch
    prompts = []
    for item in batch_data:
        title = item.get('title', '')
        abstract = item.get('abstractText', '')
        pmid = item.get('pmid', '')
        year = item.get('year', '')

        # Inline the format_title_abstract function
        title_abstract = f"TITLE: {title}. ABSTRACT: {abstract}. YEAR: {year}"

        # Replace template variable
        prompt = template.replace("{title_abstract}", title_abstract)
        prompts.append(prompt)

    # Send entire batch to GPT
    try:
        responses, _ = call_llm_batch(prompts=prompts,
                                      model_name="openai/gpt-4.1",
                                      max_tokens=1000,
                                      temperature=0.7,
                                      use_cache=True)

    except Exception as e:
        print(f"Error calling GPT-4 batch: {e}")
        # Return empty results for the batch if API call fails
        return []

    # Check if we're using a multi-question template
    is_multi_question = template_key in TEMPLATES_MULTIQUESTION if template_key is not None else False

    # Process responses and match them back to original items
    results = []
    for i, (item, response) in enumerate(zip(batch_data, responses)):
        title = item.get('title', '')
        pmid = item.get('pmid', '')

        # Parse response using appropriate parsing function
        try:
            if is_multi_question:
                # Parse multiple Q&A pairs
                parsed_list = parse_multi_qa_response(response)

                # Create separate result entries for each Q&A pair
                for parsed in parsed_list:
                    if parsed['question']:
                        results.append({
                            'question': parsed['question'],
                            'answer': parsed['answer'],
                            'cat_num': parsed['cat_num'],
                            'cat': parsed['cat'],
                            'pmid': pmid,
                            'paper_title': title,
                        })

                if not parsed_list:
                    print(
                        f"Warning: No questions found in multi-QA response for PMID {pmid}"
                    )

            else:
                # Parse single Q&A pair (existing logic)
                parsed = parse_llm_response(response)

                # Only include if we got at least a question
                if parsed['question']:
                    results.append({
                        'question': parsed['question'],
                        'answer': parsed['answer'],
                        'cat_num': parsed['cat_num'],
                        'cat': parsed['cat'],
                        'pmid': pmid,
                        'paper_title': title,
                        'raw_response':
                        response  # Keep raw response for debugging
                    })
                else:
                    print(
                        f"Warning: No question found in response for PMID {pmid}"
                    )

        except Exception as e:
            print(f"Error parsing response for PMID {pmid}: {e}")
            print(f"Raw response: {response[:200]}...")
            continue

    return results


def filter_this_study_answers(qa_pairs: List[Dict]) -> List[Dict]:
    """
    Filter out Q&A pairs where the question or answer contains 'this study' or 'the study' (case insensitive).
    Returns filtered list and prints how many were removed.
    """
    original_count = len(qa_pairs)

    # Filter out pairs where question or answer contains "this study" or "the study" (case insensitive)
    filtered_pairs = []
    removed_count = 0

    for qa in qa_pairs:
        question = qa.get('question', '')
        answer = qa.get('answer', '')

        # Check both question and answer for "this study" and "the study"
        question_has_this_study = isinstance(
            question, str) and 'this study' in question.lower()
        answer_has_this_study = isinstance(
            answer, str) and 'this study' in answer.lower()
        question_has_the_study = isinstance(
            question, str) and 'the study' in question.lower()
        answer_has_the_study = isinstance(
            answer, str) and 'the study' in answer.lower()

        if question_has_this_study or answer_has_this_study or question_has_the_study or answer_has_the_study:
            removed_count += 1
        else:
            filtered_pairs.append(qa)

    print(
        f"Filtered out {removed_count} Q&A pairs containing 'this study' or 'the study' in the question or answer"
    )
    print(
        f"Remaining pairs: {len(filtered_pairs)} (originally {original_count})"
    )

    return filtered_pairs


def generate_golden_answers(qa_pairs: List[Dict],
                            golden_key: int = 1) -> List[Dict]:
    """Generate golden_answers (synonyms) for each Q&A pair in batches """
    print("Generating golden answers (synonyms)...")

    # Get golden template
    if golden_key not in GOLDEN_TEMPLATES:
        raise ValueError(
            f"Golden template key {golden_key} not found. Available keys: {list(GOLDEN_TEMPLATES.keys())}"
        )

    golden_template = GOLDEN_TEMPLATES[golden_key]

    batch_size = 1000
    updated_qa_pairs = []

    # Process in batches of 50
    for i in tqdm(range(0, len(qa_pairs), batch_size),
                  desc="Generating golden answers"):
        batch = qa_pairs[i:i + batch_size]

        # Create prompts for the entire batch
        prompts = []
        for qa in batch:
            prompt = golden_template.format(question=qa['question'],
                                            answer=qa['answer'])
            prompts.append(prompt)

        # Call LLM for the entire batch
        try:
            responses, _ = call_llm_batch(prompts=prompts,
                                          model_name="openai/gpt-4.1",
                                          max_tokens=500,
                                          temperature=0.3,
                                          max_concurrent=50,
                                          use_cache=True,
                                          json_mode=True)

            # Process each response in the batch
            for qa, golden_answers in zip(batch, responses):
                updated_qa = qa.copy()
                try:
                    golden_answers = json.loads(
                        golden_answers)['golden_answers']
                    # With json_mode=True, `call_llm_batch` should return parsed JSON objects.
                    # We validate that we got a non-empty list or dictionary.
                    if (isinstance(golden_answers,
                                   (list, dict))) and golden_answers:
                        updated_qa['golden_answers'] = golden_answers
                    else:
                        # Fallback to original answer if response is empty or wrong type
                        updated_qa['golden_answers'] = [qa['answer']]
                        print(
                            f"Warning: Empty or invalid golden_answers format for question: {qa['question'][:50]}..."
                        )
                except Exception as e:
                    # Fallback for any unexpected error during processing of a single response
                    updated_qa['golden_answers'] = [qa['answer']]
                    print(
                        f"Warning: Unexpected error processing golden_answers for question: {qa['question'][:50]}... Error: {e}"
                    )
                updated_qa_pairs.append(updated_qa)

        except Exception as e:
            print(f"Error calling LLM batch for golden answers: {e}")
            # Add fallback for entire batch if LLM call fails
            for qa in batch:
                updated_qa = qa.copy()
                updated_qa['golden_answers'] = [qa['answer']]
                updated_qa_pairs.append(updated_qa)

    print(f"Generated golden answers for {len(updated_qa_pairs)} Q&A pairs")
    return updated_qa_pairs


def apply_paraphrasing(qa_pairs: List[Dict],
                       paraphrase_key: int = 1,
                       paraphrase_pcnt: float = 0.5) -> List[Dict]:
    """Apply paraphrasing to questions with given probability"""
    print(
        f"Applying paraphrasing with probability {paraphrase_pcnt} using template {paraphrase_key}..."
    )

    # Get paraphrase template
    if paraphrase_key not in PARAPHRASE_TEMPLATES:
        raise ValueError(
            f"Paraphrase template key {paraphrase_key} not found. Available keys: {list(PARAPHRASE_TEMPLATES.keys())}"
        )

    paraphrase_template = PARAPHRASE_TEMPLATES[paraphrase_key]

    # Add question_original column and determine which questions to paraphrase
    updated_qa_pairs = []
    questions_to_paraphrase = []
    indices_to_paraphrase = []

    random.seed(42)  # For reproducibility

    for i, qa in enumerate(qa_pairs):
        # Create updated QA with question_original
        updated_qa = qa.copy()
        updated_qa['question_original'] = qa['question']

        # Decide whether to paraphrase this question
        if random.random() < paraphrase_pcnt:
            # Mark for paraphrasing
            prompt = paraphrase_template.format(question=qa['question'],
                                                answer=qa['answer'])
            questions_to_paraphrase.append(prompt)
            indices_to_paraphrase.append(i)

        updated_qa_pairs.append(updated_qa)

    print(
        f"Selected {len(questions_to_paraphrase)} questions for paraphrasing out of {len(qa_pairs)}"
    )

    if not questions_to_paraphrase:
        return updated_qa_pairs

    # Process paraphrasing in batches
    batch_size = 1000
    paraphrased_questions = []

    for i in tqdm(range(0, len(questions_to_paraphrase), batch_size),
                  desc="Paraphrasing questions"):
        batch_prompts = questions_to_paraphrase[i:i + batch_size]

        try:
            responses, _ = call_llm_batch(prompts=batch_prompts,
                                          model_name="openai/gpt-4.1",
                                          max_tokens=500,
                                          temperature=0.7,
                                          use_cache=True)

            # Parse responses to extract paraphrased questions
            for response in responses:
                # Extract question from <question>...</question> tags
                question_match = re.search(r'<question>(.*?)</question>',
                                           response, re.DOTALL | re.IGNORECASE)
                if question_match:
                    paraphrased_question = question_match.group(1).strip()
                    paraphrased_questions.append(paraphrased_question)
                else:
                    # Fallback to original if parsing fails
                    paraphrased_questions.append(None)
                    print(
                        f"Warning: Could not parse paraphrased question from response: {response[:100]}..."
                    )

        except Exception as e:
            print(f"Error calling LLM batch for paraphrasing: {e}")
            # Add None for entire batch if LLM call fails
            paraphrased_questions.extend([None] * len(batch_prompts))

    # Apply paraphrased questions back to the dataset
    paraphrase_count = 0
    for i, (qa_idx, paraphrased_q) in enumerate(
            zip(indices_to_paraphrase, paraphrased_questions)):
        if paraphrased_q is not None:
            updated_qa_pairs[qa_idx]['question'] = paraphrased_q
            paraphrase_count += 1
        # If paraphrasing failed, keep original question

    print(f"Successfully paraphrased {paraphrase_count} questions")
    return updated_qa_pairs


def create_and_push_dataset(qa_pairs: List[Dict],
                            n_test: int = 200,
                            dataset_name_suffix: str = None):
    """
    Create train/test split and save locally (optionally push to HuggingFace Hub).

    Args:
        qa_pairs: List of Q&A dictionaries
        n_test: Number of test examples
        dataset_name_suffix: Dataset name suffix (will be prefixed with HF_USERNAME if uploading)
    """
    print(
        f"Creating dataset with {len(qa_pairs)} examples, test size: {n_test}")

    # Convert to DataFrame first to check for mixed types
    df = pd.DataFrame(qa_pairs)

    # Filter and fix golden_answers column
    if 'golden_answers' in df.columns:
        original_length = len(df)

        def fix_golden_answers(x):
            """Fix golden_answers by flattening nested lists and filtering non-lists"""
            if not isinstance(x, list):
                return None  # Mark for removal

            # Flatten nested lists
            flattened = []
            for item in x:
                if isinstance(item, list):
                    flattened.extend(item)  # Flatten nested list
                else:
                    flattened.append(item)  # Keep single item

            return flattened

        # Apply the fix function
        df['golden_answers_fixed'] = df['golden_answers'].apply(
            fix_golden_answers)

        # Count how many needed flattening vs removal
        non_list_count = df['golden_answers_fixed'].isnull().sum()
        flattened_count = 0

        # Count rows that needed flattening (had nested lists)
        for i, (original, fixed) in enumerate(
                zip(df['golden_answers'], df['golden_answers_fixed'])):
            if fixed is not None and isinstance(original, list):
                # Check if any item in original was a list
                if any(isinstance(item, list) for item in original):
                    flattened_count += 1

        # Remove rows where golden_answers couldn't be fixed (weren't lists)
        df = df[df['golden_answers_fixed'].notnull()]

        # Replace the original column with the fixed one
        df['golden_answers'] = df['golden_answers_fixed']
        df = df.drop('golden_answers_fixed', axis=1)

        print(
            f"Filtered out {non_list_count} rows where golden_answers was not a list"
        )
        print(
            f"Flattened {flattened_count} rows where golden_answers contained nested lists"
        )
        print(
            f"Original dataset size: {original_length}, Filtered dataset size: {len(df)}"
        )

        # Convert back to list of dictionaries
        qa_pairs = df.to_dict('records')

    # Split into train and test (last n_test examples as test)
    if len(qa_pairs) > n_test:
        train_data = qa_pairs[:-n_test]
        test_data = qa_pairs[-n_test:]
    else:
        print(
            f"Warning: Not enough data for test split. Using all data as train."
        )
        train_data = qa_pairs
        test_data = []

    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")

    # Create HuggingFace datasets
    dataset_dict = {}

    if train_data:
        train_dataset = Dataset.from_list(train_data)
        dataset_dict['train'] = train_dataset

    if test_data:
        test_dataset = Dataset.from_list(test_data)
        dataset_dict['test'] = test_dataset

    # Create DatasetDict
    dataset = DatasetDict(dataset_dict)

    # Build dataset name with username prefix
    dataset_name_with_prefix = f"{HF_USERNAME}/{dataset_name_suffix}"

    # Always save locally as parquet
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for split_name, split_dataset in dataset.items():
        output_file = os.path.join(OUTPUT_DIR, f"{dataset_name_with_prefix.replace('/', '_')}_{split_name}.parquet")
        split_dataset.to_parquet(output_file)
        print(f"✅ Saved {split_name} split to: {output_file}")

    # Optionally push to HuggingFace Hub if explicitly enabled
    if ENABLE_HF_UPLOAD:
        print(f"\nPushing dataset to HuggingFace Hub as: {dataset_name_with_prefix}")
        dataset.push_to_hub(dataset_name_with_prefix, token=os.getenv("HF_TOKEN"))
        print(f"✅ Uploaded to: https://huggingface.co/datasets/{dataset_name_with_prefix}")
    else:
        print("\nℹ️  To upload to HuggingFace, set ENABLE_HF_UPLOAD=1:")
        print("   export ENABLE_HF_UPLOAD=1")
        print("   export HF_USERNAME=your_username")
        print("   export HF_TOKEN=your_token")

    return dataset


def generate_dataset_from_abstracts(key: int = 1,
                                    golden_key: int = 1,
                                    n_samples: int = 1000,
                                    n_test: int = 200,
                                    hub_name: str = None,
                                    do_paraphrase: bool = True,
                                    paraphrase_key: int = 1,
                                    paraphrase_pcnt: float = 0.5):
    """Main function to generate dataset from abstracts"""
    print(
        f"Starting dataset generation with key={key}, golden_key={golden_key}, n_samples={n_samples}, "
        f"do_paraphrase={do_paraphrase}, paraphrase_key={paraphrase_key}, paraphrase_pcnt={paraphrase_pcnt}"
    )

    # Inline setup_results_directory
    results_dir = "results/generate_dataset_from_abstracts"
    os.makedirs(results_dir, exist_ok=True)

    # Get template
    if key not in TEMPLATES:
        raise ValueError(
            f"Template key {key} not found. Available keys: {list(TEMPLATES.keys())}"
        )

    template = TEMPLATES[key]

    # Load the indexer
    print("Loading PubMed abstracts database...")
    parquet_file = 'data/allMeSH_2022.parquet'
    if not os.path.exists(parquet_file):
        raise FileNotFoundError(f"Parquet file not found: {parquet_file}")

    indexer = return_indexer(parquet_file)
    total_length = len(indexer)
    print(f"Database loaded. Total articles: {total_length}")

    # Generate a large random sample once, then take first n_samples
    # This ensures that the first 10 samples are always the same whether you take 10, 20, or more
    random.seed(0)
    np.random.seed(0)

    # Generate a large master list of random indices (e.g., 50k or use total_length)
    max_samples = total_length
    master_indices = random.sample(range(total_length), max_samples)

    # Take only the first n_samples from the master list
    indices = master_indices[:n_samples]
    print(
        f"Selected first {len(indices)} indices from master random list of {len(master_indices)}"
    )

    # Process in batches of 50
    batch_size = 1000
    all_results = []

    for i in tqdm(range(0, len(indices), batch_size),
                  desc="Processing batches"):
        batch_indices = indices[i:i + batch_size]

        # Get batch data
        batch_data = []
        for idx in batch_indices:
            try:
                item = indexer.iloc(idx)
                batch_data.append(item)
            except Exception as e:
                print(f"Error getting item at index {idx}: {e}")
                continue

        if not batch_data:
            continue

        # Process batch - pass the template key
        batch_results = process_batch(batch_data, template, template_key=key)
        all_results.extend(batch_results)

        # Save intermediate results
        if i % (batch_size * 10) == 0:  # Save every 10 batches
            temp_file = os.path.join(
                results_dir, f"temp_results_batch_{i//batch_size}.csv")
            pd.DataFrame(all_results).to_csv(temp_file, index=False)
            print(f"Saved intermediate results to {temp_file}")

    # Save initial results
    if all_results:
        df = pd.DataFrame(all_results)

        # Create and push to HuggingFace Hub (moved up to get hub_name early)
        if hub_name is None:
            hub_name = f"PaperSearchRL_v{key}_gv{golden_key}_n{n_samples}_test{n_test}"

        # Extract dataset name from hub_name (remove "jmhb/" prefix)
        dataset_name = hub_name.split('/')[-1] if '/' in hub_name else hub_name

        output_file = os.path.join(results_dir, f"{dataset_name}_initial.csv")
        df.to_csv(output_file, index=False)
        print(f"Initial Q&A results saved to {output_file}")
        print(f"Generated {len(all_results)} Q&A pairs")

        # Filter out answers containing "this study"
        all_results = filter_this_study_answers(all_results)

        # Generate golden answers
        all_results_with_golden = generate_golden_answers(
            all_results, golden_key=golden_key)

        # Apply paraphrasing if enabled
        if do_paraphrase:
            all_results_with_golden = apply_paraphrasing(
                all_results_with_golden,
                paraphrase_key=paraphrase_key,
                paraphrase_pcnt=paraphrase_pcnt)

        # Add is_paraphrased column
        if do_paraphrase:
            for qa in all_results_with_golden:
                qa['is_paraphrased'] = qa['question_original'] != qa[
                    'question']
        else:
            # If no paraphrasing was done, add the column as False for all entries
            for qa in all_results_with_golden:
                qa['is_paraphrased'] = False

        # Set debug breakpoint as requested

        # Save final results with golden answers (and possibly paraphrasing)
        df_final = pd.DataFrame(all_results_with_golden)

        # Reorder columns to put question_original first if it exists
        if 'question_original' in df_final.columns:
            # Define desired column order
            cols = [
                'question_original', 'question', 'answer', 'is_paraphrased'
            ]
            # Add remaining columns
            remaining_cols = [
                col for col in df_final.columns if col not in cols
            ]
            cols.extend(remaining_cols)
            df_final = df_final[cols]
        elif 'is_paraphrased' in df_final.columns:
            # If no question_original but is_paraphrased exists, put is_paraphrased after answer
            cols = ['question', 'answer', 'is_paraphrased']
            remaining_cols = [
                col for col in df_final.columns if col not in cols
            ]
            cols.extend(remaining_cols)
            df_final = df_final[cols]

        final_output_file = os.path.join(results_dir, f"{dataset_name}.csv")
        df_final.to_csv(final_output_file, index=False)
        print(
            f"Final results with golden answers{' and paraphrasing' if do_paraphrase else ''} saved to {final_output_file}"
        )

        dataset = create_and_push_dataset(all_results_with_golden,
                                          n_test=n_test,
                                          dataset_name_suffix=hub_name)
        print(f"Dataset creation complete: {hub_name}")

        # Check data types for each column
        print("Column data types:")
        print(df_final.dtypes)

        # Check for mixed types in each column
        print("\nChecking for mixed types in each column:")
        for col in df_final.columns:
            unique_types = df_final[col].apply(
                lambda x: type(x).__name__).unique()
            if len(unique_types) > 1:
                print(f"Column '{col}' has mixed types: {unique_types}")
                # Show some examples
                print("  Examples:")
                for utype in unique_types:
                    example = df_final[df_final[col].apply(
                        lambda x: type(x).__name__ == utype)][col].iloc[0]
                    print(f"    {utype}: {repr(example)}")
                print()

        # Check specifically for the golden_answers column (most likely culprit)
        if 'golden_answers' in df_final.columns:
            print(f"\nDetailed check for 'golden_answers' column:")
            ga_types = df_final['golden_answers'].apply(
                lambda x: type(x).__name__)
            print(f"Types found: {ga_types.value_counts()}")

            # Show examples of each type
            for type_name in ga_types.unique():
                example_idx = ga_types[ga_types == type_name].index[0]
                example_value = df_final.loc[example_idx, 'golden_answers']
                print(f"  {type_name} example: {repr(example_value)}")

    else:
        print("No results generated")

    pass


if __name__ == "__main__":
    # You can modify these parameters or add command line argument parsing
    n_samples = 20000
    n_test = 5000
    key = 5
    golden_key = 3
    do_paraphrase = True
    paraphrase_key = 1
    paraphrase_pcnt = 0.5

    # Build hub_name with paraphrasing info if applicable
    base_hub_name = f"PaperSearchRL_v{key}_gv{golden_key}_n{n_samples}_test{n_test}"
    if do_paraphrase:
        pcnt_rounded = round(paraphrase_pcnt * 100)
        hub_name = f"{base_hub_name}_parav{paraphrase_key}pcnt{pcnt_rounded}"
    else:
        hub_name = base_hub_name
    print(f"hub_name: {hub_name}")

    print(f"generating dataset with hub_name: {hub_name}")
    generate_dataset_from_abstracts(key=key,
                                    golden_key=golden_key,
                                    n_samples=n_samples,
                                    n_test=n_test,
                                    hub_name=hub_name,
                                    do_paraphrase=do_paraphrase,
                                    paraphrase_key=paraphrase_key,
                                    paraphrase_pcnt=paraphrase_pcnt)
