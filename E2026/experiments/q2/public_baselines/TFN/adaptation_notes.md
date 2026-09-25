# TFN aligned-50 adaptation

Source: https://github.com/Justin1904/TensorFusionNetworks at `ef0e78b5583159de9b74ef2cdef6031bd9f94b37`. License: No LICENSE file in inspected revision; architecture reference only; no source code copied.

Packed text LSTM; masked-mean audio/vision utterance vectors; 33^3 outer product; 3-class and regression heads.

Code is a new adaptation of the published mechanism. The input is Attachment2 precomputed 50-step features. The common task uses train-only balanced CE plus SmoothL1; MISA retains its original core auxiliary objectives with published default coefficients. Padding comes from `text_bert[:,1,:]`; native all-zero vectors remain data. Models are selected on clean validation selection score, never test.
