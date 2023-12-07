#GPT2 + EnCodec24K + EnCodecMAE-Base + NSynth
ginpipe configs/base/train_lm.gin \
		configs/datasets/nsynth_24k.gin \
        configs/features/encodecmae_prompt.gin \
        configs/features/wav_and_prompt.gin \
        configs/models/encodecgpt2.gin \
		--module_list configs/imports \
		--project_name encodecgpt2 \
		--experiment_name nsynth-24k-encodecmaebase

#GPT2 + EnCodec24K + EnCodecMAE-Base + Librilight
#ginpipe configs/base/train_lm.gin \
#		configs/datasets/librilight_24k.gin \
#        configs/features/encodecmae_prompt.gin \
#        configs/features/wav_and_prompt.gin \
#        configs/models/encodecgpt2.gin \
#		--module_list configs/imports \
#		--project_name encodecgpt2 \
#		--experiment_name librilight-24k-encodecmaebase