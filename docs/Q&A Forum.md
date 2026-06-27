# NLP Homework 2 Q&A Forum

## Weighting of Efficiency-Speed and F1
**By: Hanin Naran** | Date: Tuesday, June 23, 2026, 9:28 PM

**Question:**
It is written that the three components—efficiency, speed, and F1 of the tokenizer—will receive equal weight in the evaluation. I wanted to ask what the weighting formula is, given that these 3 numbers are not on the same scale.

**Answer (By: Dvir Lefer | Date: Tuesday, June 23, 2026, 10:18 PM):**
Because you are measured on performance within a competition, the values are normalized relative to the other students. Only then is the weight distributed, so effectively, the weight of each of them out of the homework grade will be 15%.

---

## Token Limit and Dev Files
**By: Ofek Nisan** | Date: Tuesday, June 23, 2026, 11:18 PM

**Question:**
Hi, as part of the homework instructions, it says that the tokenizer must allow tokens up to the bigram level (two adjacent words receiving one token) and that we must ensure each tokenizer generates at least one bigram, otherwise the exercise will be disqualified.
I would appreciate knowing what this means or getting an example. I am afraid I misunderstood.
Additionally, regarding the dev files, is there a reason we actually need them?

**Answer (By: Dvir Lefer | Date: Wednesday, June 24, 2026, 1:14 PM):**
Hi Ofek, as written in the exercise instructions - two adjacent words receiving a single token. For example, "Ofek Nisan" would be represented by a single token.
Regarding the dev files - they are basically just smaller files for your convenience. It is common practice to use such files for debugging, and you can do whatever you see fit with them.

**Follow-up Question (By: Ofek Nisan | Date: Wednesday, June 24, 2026, 1:58 PM):**
Hi, thank you for the answer.
So as part of the restriction, you wouldn't want us to create a token for 3 adjacent words (obviously after running a few merges)?

**Answer (By: Dvir Lefer | Date: Wednesday, June 24, 2026, 2:35 PM):**
Yes, that is the meaning of "up to the bigram level".

---

## Question Regarding Vocabulary Size
**By: Noa Bamberger** | Date: Saturday, June 20, 2026, 2:24 PM

**Question:**
Hi,
According to what is written in the homework guidelines and the sample submission file, it seems we can edit `train_tokenizer` and `test_tokenizer` to adapt them to the task.
To my understanding, they can also be adapted to train differently between `tokenizer_1`, `tokenizer_2`, and `tokenizer_3`. Is this correct?
However - the `generate_tokenizers` code *does not* use either of these files (neither `train_tokenizer` nor `test_tokenizer`), but rather reads the Tokenizer itself and *inserts the model's hyperparameters by itself*, for example, `vocab_size`.
I would appreciate understanding if I can edit the training hyperparameters, and if so, whether to do it using the train and test files or in a different way?
Thanks!

**Answer (By: Dvir Lefer | Date: Sunday, June 21, 2026, 11:08 AM):**
Hi Noa,
Thank you for the note - you can edit the hyperparameters, of course. I will upload an updated version that allows full reproduction of the tokenizers according to how you train them (subject to the restrictions written in the guidelines).

**Update (By: Dvir Lefer | Date: Sunday, June 21, 2026, 12:35 PM):**
Hi Noa,
In order not to make things difficult with major changes to the files, we added a small requirement - you need to attach your commands for running `generate_tokenizer.py` in order to reproduce your tokenizers.
Note that the training logic takes place within the class you are required to submit anyway, so this shouldn't prevent you from doing anything.
Good luck!

**Follow-up Question (By: Ron Preminger | Date: Wednesday, June 24, 2026, 6:34 PM):**
Since we are not allowed to edit `generate_tokenizer.py`, does this mean we have to train all tokenizers on the same vocab size and the same parameters?

**Answer (By: Dvir Lefer | Date: Wednesday, June 24, 2026, 7:41 PM):**
Note that these are arguments that can be defined when running the script, which is why we allowed uploading a text file with the exact commands you used.

---

## F1 Threshold Update
**By: Dvir Lefer** | Date: Sunday, June 21, 2026, 9:39 PM

**Announcement:**
Hello everyone,
We have updated the F1 passing threshold to 0.4 instead of 0.5 - the guidelines file has been updated accordingly.
Good luck!