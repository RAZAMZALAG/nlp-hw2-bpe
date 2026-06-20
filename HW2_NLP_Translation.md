# Natural Language Processing - Homework 2

## Task Description
In this exercise, you will implement a Byte Pair Encoding (BPE) tokenizer as learned in the lecture, for two datasets from different domains. You will also be tested on a third hidden domain. The tokenizer will serve an existing Named Entity Recognition (NER) model and you will evaluate its impact on the task's performance.

## Performance Metric (F1 and Efficiency)
* **F1 Metric:** Refers to binary F1 accuracy at the word level.
* **Tokenizer Efficiency:** Refers to the number of tokens each sentence encodes to and its inference time in the virtual machine environment.

## Grade Structure

| Section | Percentage | Description |
| :--- | :--- | :--- |
| **Tokenizer Implementation** | 20% | Full implementation of the three tokenizers, and meeting the F1 score threshold of at least 0.5 on the `domain_1` dev files and 1 on `domain_2`. |
| **Competition** | 45% | Weighted competition on tokenizer efficiency, tokenization speed, and F1 score in tagging the hidden competition files (one from each domain). |
| **Tokenizer 3 Evaluation** | 15% | Evaluation of the third tokenizer's performance (the hidden domain) in the report. |
| **Submission Report** | 20% | Writing a concise report and full compliance with format requirements. |

## Attached Files
The exercise data files are:
* **Tokenizer training files:** `domain_1_train.txt`, `domain_2_train.txt`. Each sentence is separated by a newline. You can use them as you wish; matching `dev` files are also attached.
* **NER task training and evaluation files (for an existing model):**
  * Domain 1: `tagged.train_1_binary` (training), `tagged.dev_1_binary` (development)
  * Domain 2: `tagged.train_2_binary` (training), `tagged.dev_2_binary` (development)

**Note:** You are not required to read and process the NER files yourselves, but to ensure that in the `train_ner_model.py` file you pass the correct parameters according to the domain.
Apart from file routing, no run parameters should be changed - your tokenizer will be tested against a uniform model trained in the exact same way for everyone. Further explanation on how to use the files can be found in the `README` file.

## Implementation Requirements

### Training Tokenizers
* **Algorithm:** The implementation of all tokenizers must be based on the Byte Pair Encoding (BPE) algorithm - meaning it starts from breaking down the sentence to the character level.
* **Token limit:** The tokenizer must allow tokens up to the bigram level (two adjacent words receiving one token). You must ensure each tokenizer generates at least one bigram. Failure to meet this condition will result in the disqualification of the exercise.
* **Data usage:** You may only use the provided data for training. Using additional data from external sources is prohibited.

### Interface and Base Class
* **Base class:** Each tokenizer must inherit from the `BaseTokenizer` base class.
* **Functions:** All required functions of the abstract class must be implemented.
* **Space character:** You must add the `space_token` attribute to the class, with your chosen character for it (e.g., `' '` or `'_'`).
* **Recreation:** You must implement your tokenizer in the `code/bpe_tokenizer.py` file (a class inheriting from `BaseTokenizer`). Tokenizers must be reproducible using the provided `generate_tokenizers.py` script, which runs your implementation on the provided data. Do not edit or submit this script.

**Note:** You need to save the trained tokenizer in the given format, and ensure other scripts can read it without knowing your class (`train_ner_model.py` & `test_tokenizer.py`), only the inherited class.

### First and Second Tokenizers
Use the provided training files (`train_1_domain.txt` - `train_2_domain.txt`). The task files (NER) come from the same domain.

### Third Tokenizer - Hidden Domain
Use the training files from the previous tokenizers. You must think about how to deal with the fact that they were sampled from a different domain than the one you will be tested on.

### Validation
For the two given domains, report on the dev file:
* Tagging speed and efficiency.
* The F1-score on the `tagged.dev` file.

### Competition
The three tokenizers will be tested on hidden competition files. The competition is weighted on tokenizer efficiency, tokenization speed, and F1 score. Each component has an equal weight and there might be a trade-off between these components.

### Environment
The exercise must run in the provided `uv` environment (detailed requirements in `pyproject.toml`). Do not install additional libraries without course staff approval. You must initialize the workspace on the provided machines by running the command `bash init.sh`.

**Note:** The provided machines have a total runtime limit of 10 hours, so ensure you leave them on only when necessary (it is recommended to debug your code before running it on the machine and to run on it only when you already need a GPU).
You can use the `time_left` command in the terminal to see how much runtime is left for your machine.

## Submission
The assignment submission will be done in a zip file only, named `HW2_123456789.zip`.

**The report evaluation will be based on:**
* **Creativity:** Improving tokenizer efficiency and speed while maintaining F1 performance.
* **Writing and phrasing:** Clarity of reporting in the report.
* **Rigorousity:** Use of proper evaluation methods.
* **Innovation:** Original ideas and their implementation.

**Format requirements:**
* Up to one A4 page.
* Standard margins - 2.54 cm on all sides.
* Single-column format.
* Arial font, size 10.
* Standard line spacing (1.15).
* A single blank line between paragraphs.
* Every image must be centered, without text wrap around, and must have a caption below the image.
* Text inside images (legends, titles, etc.) must be at least 9 pt. If it requires zooming over 100% to read, the image will not be considered.

**Files to submit:**
* The **`report_123456789.pdf`** file containing concise explanations, reporting, and analysis of results.
  * **Mandatory content:** Author name and ID, performance evaluation report for model 2, explanation of parameter selection, features, and final improvements in both models.
  * **Mandatory content:**
    * **Training:** Explanation of each implemented tokenizer. How it differs from the base algorithm, and how it handles the hidden domain.
    * **Bigrams:** Report on the most common (up to 5) and least common (up to 5) bigrams in the training set.
    * **Validation:** F1 score report on the dev files for each model, and tokenizer efficiency.
    * **Evaluation:** Your evaluation of the tokenizer's performance on the hidden domain.
* **Your source code (in the `code/` folder):** Must include `bpe_tokenizer.py` and any auxiliary module it imports, as well as scripts you edited (e.g., `train_tokenizer.py`, `test_tokenizer.py`). Do not include `generate_tokenizers.py`, `train_ner_model.py`, `base_tokenizer.py`, or `check_submission.py` - evaluation is done against their original versions. The code must be documented and readable.
* **Trained tokenizers (in the `trained_tokenizers/` folder):** Saved as `tokenizer_1.pkl` (1, 2, and 3) in the provided format.

**Note:** The scripts `generate_tokenizers.py` and `check_submission.py` are provided to you - do not edit or submit them. Your three tokenizers must be reproducible using `generate_tokenizers.py` based on the code in `./code`.
**Note:** File format mismatch means a grade of 0.
*For the avoidance of doubt - an example submission file is attached containing exactly the file structure that should appear in your final zip.*

## Plagiarism
Code passing between students is strictly prohibited. AI tools (such as chatGPT) are allowed, but the code is your responsibility and you must not pass prompts or code snippets generated by AI.
