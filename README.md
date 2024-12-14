# Transformer-Implementation-for-Machine-Translation
Implement a Transformer model for machine translation using the UMC005: English-Urdu Parallel Corpus. The task involves training a model that can translate English sentences into Urdu. The assignment will focus on understanding the core components of the Transformer architecture and evaluating its performance on the given dataset. The following tasks are required the least:
•
Preprocess the dataset by cleaning and tokenizing the text. Ensure that English and Urdu texts are properly aligned.
•
Use suitable tokenizers for English and Urdu. You can experiment with Byte Pair Encoding (BPE) or other suitable tokenization techniques.
•
Implement the Transformer model from scratch using a deep learning framework like TensorFlow or PyTorch. The architecture should include an encoder and a decoder with multiple layers.
•
Tune hyperparameters such as the number of layers, hidden units, heads in multi-head attention, dropout rates, learning rates, and batch size. Use techniques like early stopping and learning rate scheduling to improve training efficiency.
•
Evaluate the trained model on the test set using translation quality metrics BLEU Score and ROUGE (mandatory). You can add additional scores if you want (optional)
•
Comparative Analysis: Implement an LSTM model for translation using the same dataset.
•
Compare the performance of two models in terms of their translation accuracy, training time, memory usage, inference speed, perplexity.
•
Deploy your model on a local machine or cloud and create a simple GUI like ChatGPT UI. The user will be able to input English language text which will be aligned to the left, then press enter, and the model will translate the text. The translated Urdu text should appear below the source sentence and be right-aligned. The source and target language texts' history should remain visible to the user, similar to ChatGPT.
•
Plot training and validation loss curves to illustrate the training process to see if the model converges effectively.
•
Present detailed results in a report, summarizing the methodology, findings, and challenges.
•
Provide the code in a well-documented format, clearly explaining each step.
•
Implement attention visualization to show which parts of the English sentence the model focused on when generating the Urdu translation.
•
Bonus: Experiment with fine-tuning a pretrained model and compare the results with your custom-trained model.
UMC005: English-Urdu Parallel Corpus UMC005 English-Urdu is a parallel corpus of texts in English and Urdu language with sentence
alignments. The corpus can be used for experiments with statistical machine translation. For details, please visit the following website: English Urdu Dataset
