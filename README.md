# Speech Recognition on the Free Spoken Digit Dataset

<img src="./assets/fsdd.svg" width="25%" align="right"/>

<p align="center">
    <a href="#dataset-description-and-classification-problem"><b>Dataset Description and Classification Problem</b></a> ·
    <a href="#methodology"><b>Methodology</b></a> ·
    <a href="#results"><b>Results</b></a><br/>
    <a href="#improvements">Improvements</a> ·
    <a href="#setup-and-requirements">Setup and Requirements</a> ·
    <a href="#resources">Resources</a>
</p>

- Experimenting with PyTorch to broaden my knowledge of deep learning frameworks.
- Creating data loaders/generators to efficiently load and iterate through training and validation data batches of a specific dataset.
- Handling variable-length multivariate sequential data in PyTorch, using [`PackedSequence`](https://pytorch.org/docs/master/generated/torch.nn.utils.rnn.PackedSequence.html) objects for processing with RNNs.
- Using common audio feature extraction methods on raw mono signals to produce meaningful and discriminative representations such as [Mel-Frequency Cepstral Coefficients (MFCCs)](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum) for speech recognition.
- Investigating the effectiveness of my recent DTW k-NN implementation (see [Sequentia](https://github.com/eonu/sequentia)) on audio classification.

## Dataset Description and Classification Problem

<!-- MNIST vs FSDD picture -->

TODO

<table>
    <tbody>
        <tr>
            <th align="center" width="50%">
                <img src="./assets/mnist-7.png" height="100px"/>
                <br/>
                MNIST
            </th>
            <th align="center" width="50%">
                <img src="./assets/fsdd-7.svg" height="100px"/>
                <br/>
                FSDD
            </th>
		</tr>
    </tbody>
</table>

<p align="center">
    <img src="./assets/mnist-7.png" height="100px"/>
    <img src="./assets/blank.png" width="70px"/>
    <img src="./assets/fsdd-7.svg" height="100px"/>
</p>

## Methodology

<!-- Talk about MFCCs (how many?) (how preprocessed) -->

<!-- Train-val-test split size -->
<!-- Measures (accuracy) -->

<!-- Talk about each method -->
<!-- Link to notebooks -->

### Approximate Dynamic Time Warpking k-Nearest Neighbors (DTW k-NN)

<!-- Talk about method + sequentia -->
<!-- Link to notebook -->

### Bidirectional Recurrent Neural Networks with Long Short-Term Memory (BiLSTM-RNN)

<!-- Talk about method -->
<!-- NN diagram? -->
<!-- Link to notebook -->

## Results

<!-- Table -->

## Improvements

<!-- Talk about Mel spectrograms -->
<!-- Talk about how it can be improved by sigment + specaugment, delta+deltadelta -->
<!-- Deeper NN -->
<!-- CNN-LSTM (temporal convolutions) -->

## Setup and Requirements

## Resources

<!-- Useful resources (Haytham Fayek, MusicInformationRetrieval, etc.) -->