# **RMLNMT**

code implements for paper **[Improving both domain robustness and domain adaptability in machine translation(COLING 2022)](https://arxiv.org/abs/2112.08288)**, 
the code is based on public code: [fairseq](https://github.com/facebookresearch/fairseq), [Meta-Curriculum](https://github.com/NLP2CT/Meta-Curriculum) and [Meta-MT](https://www.dropbox.com/s/jguxb75utg1dmxl/meta-mt.zip?dl=0),
we provide the implement of different classifier, and word-level domain mixing.

------

**Requirements**

1. Fairseq (v0.6.0)
2. Pytorch
2. all requirements are shown in ```requirements.txt```, you can install using ```pip install -r requirements.txt ```

------

**Pipeline**

1. Train a domain classifier based on BERT/ CNN etc in ```domain_classification/Bert_classfier.py``` or ```domain_classification/main.py```

2. Score the sentence to represent the domain similarity with general domains:

   ``````python
   python meta_score_prepare.py \
   --num_labels 11 \
   --device_id 7 \
   --model_name bert-base-uncased \
   --input_path $YOUR_INPUT_PATH \
   --cls_data $YOUR_CLASSIFICATION_PATH \
   --out_data $YOUR_OUTPUT_PATH \
   --script_path $SCRIPT_PATH
   ``````

3. Run baseline systems using [fairseq](https://github.com/pytorch/fairseq), [Meta-MT](https://www.dropbox.com/s/jguxb75utg1dmxl/meta-mt.zip?dl=0) and [Meta-curriculum](https://github.com/NLP2CT/Meta-Curriculum).

4. code related to the word-lvel domain mixing is in ```word_moudles```, and please use the following command to reproduce the results in our paper:

   ```python
   python -u $code_dir/meta_ws_adapt_training.py $DARA_DIR \
       --train-subset meta-train-spm $META_DEV \
       --damethod bayesian \
       --arch transformer_da_bayes_iwslt_de_en \
       --criterion $CRITERION $BASELINE \
       --domains $DOMAINS --max-tokens 1 \
       --user-dir $user_dir \
       --domain-nums 5 \
       --translation-task en2de \
       --source-lang en --target-lang de \
       --is-curriculum --split-by-cl --distributed-world-size $GPUS \
       --required-batch-size-multiple 1 \
       --tensorboard-logdir $TF_BOARD \
       --optimizer $OPTIMIZER --lr $META_LR $DO_SAVE \
       --save-dir $PT_OUTPUT_DIR --save-interval-updates $SAVEINTERVALUPDATES \
       --max-epoch 20 \
       --skip-invalid-size-inputs-valid-test \
       --flush-secs 1 --train-percentage 0.99 --restore-file $PRE_TRAIN --log-format json \
       --- --task word_adapt_new --is-curriculum \
       --train-subset support --test-subset query --valid-subset dev_sub \
       --max-tokens 2000 --skip-invalid-size-inputs-valid-test \
       --update-freq 10000 \
       --domain-nums 5 \
       --translation-task en2de \
       --distributed-world-size 1 --max-epoch 1 --optimizer adam \
       --damethod bayesian --criterion cross_entropy_da \
       --lr 5e-05 --lr-scheduler inverse_sqrt --no-save \
       --support-tokens 8000 --query-tokens 16000 \
       --source-lang en --label-smoothing 0.1 \
       --adam-betas '(0.9, 0.98)' --warmup-updates 4000 \
       --warmup-init-lr '1e-07' --weight-decay 0.0001 \
       --target-lang de \
       --user-dir $user_dir
   ```

****
If you find our paper useful, please kindly cite our paper. Thanks!
```bibtex
@article{lai2021improving,
  title={Improving both domain robustness and domain adaptability in machine translation},
  author={Lai, Wen and Libovick{\`y}, Jind{\v{r}}ich and Fraser, Alexander},
  journal={arXiv preprint arXiv:2112.08288},
  year={2021}
}
```
   
### Contact
If you have any questions about our paper, please feel convenient to let me know through email: [lavine@cis.lmu.de](mailto:lavine@cis.lmu.de) 

   

