# Natural Language Processing - Homework 2

## Task Description
In this exercise, you will implement a BPE (Byte Pair Encoding) tokenizer as taught in the lecture, for two datasets from different domains. You will also be evaluated on a hidden third domain. The tokenizer will be used by an existing Named Entity Recognition (NER) model, and you will evaluate its impact on the task's performance.

## Performance Metrics (F1 and Efficiency)
* **F1 Metric:** Refers to the binary F1 score at the word level.
* **Tokenizer Efficiency:** Refers to the number of tokens each sentence is encoded into and its inference time in the virtual machine environment.

## Grading Structure

| Section | Percentage of Grade | Description |
| :--- | :--- | :--- |
| **Tokenizer Implementation** | 20% | Full implementation of the three tokenizers, meeting a threshold F1 score of at least 0.4 on the `domain_1_dev` and `domain_2_dev` files. |
| **Competition** | 45% | A weighted competition on tokenizer efficiency, tokenization speed, and F1 score when tagging the hidden competition files (one from each domain). |
| **Tokenizer 3 Evaluation** | 15% | Evaluation of the performance of the third tokenizer (the hidden domain) in the report. |
| **Submission Report** | 20% | Writing a concise report and strictly adhering to the formatting requirements. |

## Attached Files
The data files for the exercise are:
* **Tokenizer training files:** `domain_1_train.txt` and `domain_2_train.txt`. Each sentence is separated by a newline. You may use them as you wish. Corresponding `dev` files are also included.
* **NER training and evaluation files (for an existing model):**
    * **Domain 1:** `train_1_binary.tagged` (training), `dev_1_binary.tagged` (development).
    * **Domain 2:** `train_2_binary.tagged` (training), `dev_2_binary.tagged` (development).

**Note:** You are not required to read and process the NER files yourselves, but rather ensure that in the `train_ner_model.py` file you pass the correct parameters according to the domain. Aside from routing the files, do not change any running parameters - your tokenizer will be evaluated against a uniform model trained in the same way for everyone. Further explanation on how to use the files can be found in the README file.

---

## Implementation Requirements

### Training the Tokenizers
* **Algorithm:** The implementation of all tokenizers must be based on the BPE - Byte Pair Encoding algorithm. Meaning - it starts from breaking down the sentence to the character level.
* **Token Limit:** The tokenizer must allow tokens up to the bigram level (two adjacent words receiving one token). You must ensure that each tokenizer generates at least one bigram. Failure to meet this condition will result in disqualification of the exercise.
* **Data Usage:** You may only use the provided data for training. Using additional data from external sources is strictly prohibited.

### Interface and Base Class
* **Base Class:** Every tokenizer must inherit from the base class `BaseTokenizer`.
* **Functions:** You must implement all functions required by the abstract class.
* **Space Character:** You must add the `space_token` attribute to the class, with the character you chose for it (for example, `▁` or `_`).
* **Reconstruction:** You must implement your tokenizer in the `code/bpe_tokenizer.py` file (a class inheriting from `BaseTokenizer`). The tokenizers must be reproducible using the provided script `generate_tokenizers.py`, which runs your implementation on the provided data. Do not edit or submit this script.

**Note** - You need to save the trained tokenizer in the given format, and ensure that the other scripts (`train_ner_model.py` & `test_tokenizer.py`) are able to load it without knowing your specific class, only the class it inherited from.

### First and Second Tokenizers
Use the training files you received (`domain_1_train.txt` and `domain_2_train.txt`). The task files (NER) come from the same domain.

### Third Tokenizer - Hidden Domain
Use the training files from the previous tokenizers. You need to think about how to deal with the fact that they were sampled from a different domain than the one you will be tested on.

### Validation
For the two given domains, report on the `dev` file:
* Tagging speed and efficiency.
* The F1 score on `dev.tagged`.

### Competition
All three tokenizers will be tested on hidden competition files. The competition is weighted on tokenizer efficiency, tokenization speed, and F1 score. Each component has an equal weight and there may be a trade-off between these components.

---

## Workspace Environment
The exercise must run in the provided `uv` environment (detailed requirements in `pyproject.toml`). Do not install additional libraries without permission from the course staff. You must initialize the workspace on the machines provided to you by running the command `bash init.sh`.

**Note** - The machines provided to you have a total runtime limit of 10 hours, so you must ensure you leave them on only during essential use (it is recommended to debug your code before running it on the machine and only run it there when you need a GPU).
You can use the `time_left` command in the terminal to see how much runtime is left for your machine.

---

## Submission
Submission of the work will be done in a zip file only, named `HW2_123456789.zip`.

* **The report evaluation will be based on:**
    * **Creativity:** In improving tokenizer efficiency and speed while maintaining F1 performance.
    * **Writing and Phrasing:** Clarity of reporting in the report.
    * **Rigorousity:** Use of correct evaluation methods.
    * **Innovation:** Original ideas and their implementation.
* **Format Requirements:**
    * Up to one page of A4 size.
    * Standard margins - 2.54 cm on all sides.
    * Single-column format.
    * Arial font, size 10.
    * Standard line spacing (1.15).
    * A single blank line spacing between paragraphs.
    * Every image must be centered on the page, with no text wrap around it, and must have a caption below the image.
    * Text inside images (legends, titles, etc.) must be at least 9 pt. If you need to zoom over 100% to read the text in the image, it will not be counted.
    * File name: `report_123456789.pdf` containing concise explanations, reporting, and results analysis.
    * **Mandatory Content:** Author's name and ID, performance evaluation report for model 2, explanation of parameter selection, final characteristics and improvements in both models.
* **Mandatory Content (continued):**
    * **Training:** An explanation of each implemented tokenizer. How it differs from the basic algorithm, and how it handles the hidden domain.
    * **Bigrams:** Reporting on the most common bigrams (up to 5) and least common (up to 5) in the training set.
    * **Validation:** Reporting the F1 metric on the `dev` files for each of the models, and the efficiency of the tokenizers.
    * **Evaluation:** Your evaluation for the tokenizer's performance on the hidden domain.
* **Your Source Code (in the `code/` folder):** Must include `bpe_tokenizer.py` and any helper module it imports, as well as scripts you edited (e.g., `train_tokenizer.py`, `test_tokenizer.py`). **Do not include** the files - `base_tokenizer.py`, `train_ner_model.py`, `generate_tokenizers.py`, `check_submission.py` - evaluation is done against their original versions. The code must be documented and readable.
* **The Trained Tokenizers (in the `trained_tokenizers/` folder):** Saved under the names `tokenizer_1.pkl` (and 2, 3) in the format provided to you.
* **Training Commands File (`train_commands.txt` in the zip root):** A plain text file containing the exact commands you used to train and save the three tokenizers - including running `generate_tokenizers.py` with the arguments you chose (e.g., `--vocab_size`, `--train_files_3`). This file is required so we can reproduce your tokenizers.

**Note:** The scripts `generate_tokenizers.py` and `check_submission.py` are provided to you - do not edit or submit them. Your three tokenizers must be reproducible using `generate_tokenizers.py` based on the code in `code/` and using the commands you will document in `train_commands.txt`.

**Note:** A mismatch in file format means a grade of 0.
For the avoidance of doubt - a sample submission file is attached containing exactly the file structure that should appear in your final zip.

## Plagiarism
It is strictly forbidden to share code between students. You may use AI tools (such as chatGPT), but the code is your responsibility and you are not allowed to share prompts or code snippets generated by AI.