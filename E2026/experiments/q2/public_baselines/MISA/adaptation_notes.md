# MISA aligned-50 adaptation

Source: https://github.com/declare-lab/MISA at `ec42faddde0d210cf7368aebf2118fe9570e7102`. License: MIT.

Precomputed BERT text feature replaces tokenizer/BERT encoder; masked mean A/V utterance vectors; shared/private decomposition, CMD, difference and reconstruction losses; 3-class and regression heads.

Code is a new adaptation of the published mechanism. The input is Attachment2 precomputed 50-step features. The common task uses train-only balanced CE plus SmoothL1; MISA retains its original core auxiliary objectives with published default coefficients. Padding comes from `text_bert[:,1,:]`; native all-zero vectors remain data. Models are selected on clean validation selection score, never test.
