# TSMixer: reproduce

This repo contains codes to reproduce the main results of TSMixer.

Currently support four ETDataset(ETTh1, ETTh2, ETTm1, ETTm2) and two models(linear baseline and TSMixer).

To run the experiment:
```python
python main.py --model tsmixer --n_mixer 1 \
	-L 512 -T 96 --name ETTh1
```