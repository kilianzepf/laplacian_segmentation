import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from netcal.presentation import ReliabilityDiagram
from netcal.metrics import ACE

"""
This script reads all confidence and accuracy predictions from the model and generates the 
(1) Visual reliability diagram with ECE metric
(2) ACE metric
"""

if __name__ == "__main__":
    file_path = '/home/selmawanna/PycharmProjects/iccv_2023_cleanup/src/training_scripts/ssn_flipout_ood_derm_test_reliability_diagram.csv'
    df = pd.read_csv(file_path)
    labels = np.squeeze(df.iloc[:, 0].to_numpy())
    confidence = np.squeeze(df.iloc[:, 1].to_numpy())

    n_bins = 15

    ace_metric = ACE(n_bins)
    print('ACE metric:')
    print(ace_metric.measure(confidence, labels))

    diagram = ReliabilityDiagram(n_bins)
    diagram.plot(confidence, labels)
    plt.show()