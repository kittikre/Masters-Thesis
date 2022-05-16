# Masters-Thesis


The process of manually identifying EEG recordings as epileptic (‘abnormal’) or nonepileptic
(‘normal’) is very tedious: not only is it time-consuming, but also costly with
often high error rates. However, Deep Learning might offer potential assistance in classifying
EEGs. Our work investigates the suitability of such methods for detecting abnormal
brain activity by building, training, and testing multiple Deep Neural Networks. The data
we utilize for the analysis is the Temple University Hospital (TUH) Abnormal EEG Corpus,
which contains around 3000 labeled EEG recordings. We compare the performance
of a shallowCNN, deepCNN, and hybrid CNN-LSTM architecture, where the CNN is expected
to account for the spatial, and the LSTM for the sequential nature of the EEG.
Based on our results, the deepCNN shows the highest performance (with an accuracy
of 79.4%). Furthermore, the cost in terms of training time and computational resources
required is considerably lower for the deepCNN than it is for the hybrid CNN-LSTM.
With the goal of understanding better what factors play a central role in this binary classification
task, we set up three hypotheses regarding the depth of the model, the length
of the used EEG recordings, and the patients’ age. Based on these, we conclude that
deeper models and longer recording times positively affect the accuracy. As for patients’
age, the experiment carried out on only adult data yields inferior results compared to
results retrieved by using the entire dataset. It was difficult to draw clear confusions due
to computational resource constraints, as we could not test the computationally heavy
hybrid model on the same length of recordings, as we did for the shallowCNN and deep-
CNN. Hence, in the future, we suggest further research into this direction. Also, trying
out different architectural choices using these models as a basis can lead to improved performance.
With our efforts, we hope to contribute to the progression towards a clinically
useful automatic detection of abnormal EEG activity.
