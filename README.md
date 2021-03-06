# Digit Recognizer
My solution for [Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) Kaggle competition based on @cdeotte [experiments](https://www.kaggle.com/cdeotte/how-to-choose-cnn-architecture-mnist).
# Implementation Details
Input images were augmented with random rotation up to 20 degrees and shift up to 3 pixels. Here I didn't use random horizontal flip because digit '2' could transform into '5', and learning could be messed up. Data augmentation usually improves DL model performance so I didn't compare current model with model on non-augmented data.

I took original [LeNet-5](https://github.com/trushkin-ilya/digit-recognizer/blob/f32054a656620d7d9110799f30018d55842f186e/lenet.py#L4) and compared adjustments in above-mentioned experiments. All comparisons were built up to 10 epochs. Then, [final model](https://github.com/trushkin-ilya/digit-recognizer/blob/f32054a656620d7d9110799f30018d55842f186e/lenet.py#L184) was trained for 100 epochs (46 hours on Intel Core i5-4670@3400MHz CPU) and got accuracy score 99.6%. Solution was submitted to Kaggle and took 370th place with accuracy of 0.99557.

To skip implementation of visualizing, I used [Weights & Biases tool](https://www.wandb.com/).


![text](https://github.com/trushkin-ilya/digit-recognizer/blob/master/W&B%20Chart%2011_9_2019,%2011_12_22%20PM.png?raw=true)

Visualizations, network architechtures and execution logs available at [Weights & Biases project](https://app.wandb.ai/ilya-trushkin/digit-recognizer).
# Running scripts locally
Make sure to install dependencies from `requirements.txt`. After that run `train.py` from root directory which is equivalent to:

``train.py --epochs=100 --lr=0.01 --momentum=0.5 --use-wandb=False --output-dir=. --predict_every=1``
