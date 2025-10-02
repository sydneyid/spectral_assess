#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 17:35:40 2025

@author: sydneydolan
"""

import matplotlib.pyplot as plt

layers_cifar =[1,2,3]
layers_cluster = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
layers_pattern = [1,2,3,4,5,6]

pl_cluster =[12,15,15,17,13,18,12,10,13,14,15,21,24,22,23,27]
pl_cifar = [19,12,22]
pl_pattern=[19,19,18,21,17,19]


alpha_cifar = [ 1.688,3.0955,2.027]
alpha_cluster =[4.49, 3.856, 3.644, 3.804, 4.556, 3.959, 4.397, 5.275, 5.107, 3.859, 3.063, 3.156, 2.897, 2.425, 2.393, 2.134]
alpha_pattern = [ 5.226, 4.205, 4.352, 3.972, 4.431, 4.320]

plt.plot(layers_cifar, alpha_cifar, label ='CIFAR-10', marker='o')
plt.plot(layers_cluster, alpha_cluster, label ='CLUSTER', marker='s')
plt.plot(layers_pattern, alpha_pattern, label ='PATTERN', marker='o')

# Shade the area between y=2 and y=4 in green
plt.axhspan(2, 4, color='green', alpha=0.15)

# Set axis labels with font size 16
plt.xlabel('Graph Transformer Layer', fontsize=16)
plt.ylabel(r'$\alpha$', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.legend(fontsize=16)
plt.show()

plt.figure()
plt.scatter(layers_cluster, pl_cluster, label ='CLUSTER', marker='s')
plt.scatter(layers_pattern, pl_pattern,label ='PATTERN', marker='o')
plt.scatter(layers_cifar,pl_cifar, label ='CIFAR-10', marker='o')