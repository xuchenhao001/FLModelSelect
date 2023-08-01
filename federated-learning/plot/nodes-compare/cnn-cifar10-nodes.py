import sys

from plot.utils import plot_node_acc

# s = 2.45, random scheme
rdm_03_min = [8.08, 10.0, 10.0, 10.0, 10.0, 16.56, 17.36, 19.71, 20.22, 27.45, 30.83, 30.13, 25.76, 30.82, 31.72, 35.94, 36.0, 34.17, 33.83, 33.61, 36.64, 36.03, 35.05, 37.26, 39.5, 36.94, 40.81, 35.32, 48.45, 38.48, 38.07, 41.11, 41.61, 44.15, 44.4, 40.4, 41.03, 42.25, 41.19, 43.95, 43.71, 43.3, 45.3, 43.87, 43.84, 41.36, 40.51, 38.11, 43.05, 42.61, 45.7, 44.39, 46.32, 42.13, 46.54, 45.09, 44.16, 40.57, 42.97, 43.89, 42.07, 43.35, 43.23, 42.04, 43.55, 46.44, 43.56, 44.78, 39.25, 43.81, 45.48, 44.89, 43.19, 44.85, 45.33, 44.22, 42.36, 39.46, 43.1, 45.23, 42.79, 44.57, 45.0, 42.06, 40.5, 44.02, 44.58, 43.4, 42.84, 41.46, 39.24, 44.47, 44.07, 44.97, 43.33, 41.73, 42.4, 43.67, 43.11, 41.36]
rdm_03_max = [15.93, 10.0, 16.84, 18.41, 24.42, 22.58, 26.58, 30.77, 28.7, 35.0, 35.13, 36.08, 37.16, 37.67, 37.01, 39.1, 40.12, 38.79, 37.43, 40.32, 39.34, 39.04, 38.46, 41.89, 43.74, 40.83, 42.82, 40.36, 49.89, 44.48, 42.07, 43.65, 43.82, 46.37, 47.94, 43.97, 45.73, 44.54, 44.98, 46.99, 46.07, 45.85, 49.3, 45.7, 46.27, 43.7, 43.19, 40.81, 44.36, 45.4, 48.38, 46.85, 48.51, 44.93, 49.62, 47.39, 46.89, 42.94, 46.46, 45.79, 45.52, 44.36, 45.58, 43.24, 46.53, 49.34, 47.05, 47.5, 41.76, 46.24, 48.42, 48.12, 46.28, 46.79, 48.55, 47.53, 45.22, 41.1, 45.88, 47.8, 44.54, 47.02, 46.94, 45.4, 42.01, 46.11, 47.45, 45.84, 44.69, 43.65, 40.91, 47.16, 46.18, 47.66, 46.99, 44.22, 44.54, 45.96, 45.31, 43.04]
rdm_03_mean = [10.8, 10.0, 13.83, 15.94, 16.83, 19.31, 20.98, 24.11, 24.27, 31.67, 32.99, 32.43, 32.44, 33.54, 33.89, 37.67, 37.4, 36.08, 35.54, 36.14, 37.64, 37.57, 36.73, 39.29, 41.93, 38.18, 41.91, 37.58, 49.1, 41.32, 39.8, 41.99, 42.79, 45.39, 46.01, 42.58, 43.56, 43.92, 43.42, 45.34, 45.15, 44.51, 47.25, 44.75, 45.41, 42.66, 42.16, 39.86, 43.74, 44.13, 46.67, 45.65, 47.47, 43.65, 48.14, 46.23, 45.03, 41.5, 44.53, 44.84, 43.23, 43.77, 44.32, 42.48, 45.04, 47.48, 44.4, 45.9, 40.28, 44.84, 46.95, 46.29, 44.23, 45.9, 46.86, 45.53, 44.48, 40.28, 44.44, 46.64, 43.88, 45.8, 45.67, 43.29, 41.58, 44.96, 45.97, 44.55, 43.79, 42.67, 40.26, 45.87, 45.12, 45.82, 45.53, 42.64, 43.54, 44.64, 44.43, 42.47]

rdm_05_min = [7.08, 10.0, 10.0, 10.0, 16.16, 10.66, 16.43, 18.36, 21.83, 25.35, 21.94, 25.45, 28.22, 27.16, 31.64, 34.56, 36.66, 33.99, 35.08, 39.0, 35.92, 38.91, 37.34, 36.75, 40.47, 36.59, 41.37, 39.95, 44.07, 44.85, 43.64, 42.3, 40.46, 44.65, 44.19, 42.49, 45.89, 44.98, 44.07, 47.07, 44.95, 46.35, 46.81, 45.07, 44.82, 46.5, 47.04, 43.8, 45.64, 46.21, 46.49, 45.03, 46.38, 46.33, 46.59, 44.16, 46.79, 45.35, 47.51, 42.01, 45.31, 45.93, 44.42, 43.95, 45.81, 45.66, 45.76, 45.25, 45.83, 44.85, 47.05, 47.45, 45.18, 47.13, 46.22, 46.81, 45.08, 43.9, 46.22, 46.51, 45.06, 45.24, 45.85, 46.36, 43.12, 45.96, 45.29, 46.61, 44.37, 46.11, 43.66, 45.9, 45.0, 44.95, 44.91, 45.8, 44.01, 46.11, 46.25, 44.93]
rdm_05_max = [11.92, 10.0, 14.9, 16.45, 17.38, 15.94, 22.68, 25.52, 28.9, 31.77, 31.07, 32.44, 34.99, 31.53, 36.36, 38.87, 40.59, 37.14, 38.88, 42.15, 39.22, 41.91, 41.28, 39.33, 42.66, 39.99, 43.08, 41.93, 45.8, 46.66, 44.28, 45.67, 43.4, 46.66, 45.52, 44.42, 47.4, 46.25, 46.51, 48.27, 46.59, 48.37, 48.0, 47.15, 46.63, 48.03, 48.62, 45.59, 47.39, 46.86, 48.76, 47.92, 47.07, 47.79, 49.28, 45.68, 47.5, 47.53, 49.05, 44.16, 46.7, 47.14, 45.58, 45.36, 48.56, 46.27, 47.24, 47.2, 46.92, 46.81, 48.35, 48.36, 46.91, 48.37, 47.66, 48.54, 47.77, 44.91, 47.91, 47.85, 46.26, 46.32, 47.18, 48.39, 44.3, 48.01, 46.39, 47.63, 45.73, 47.0, 44.54, 47.73, 46.28, 46.27, 46.37, 47.49, 45.52, 46.65, 47.98, 46.74]
rdm_05_mean = [9.8, 10.0, 11.16, 12.53, 16.89, 13.49, 18.61, 20.98, 24.53, 28.13, 25.51, 28.84, 31.08, 29.27, 34.23, 35.79, 38.04, 34.97, 36.51, 40.08, 36.77, 40.0, 39.04, 38.25, 41.44, 38.02, 42.4, 40.94, 45.12, 45.74, 43.98, 43.91, 41.8, 45.67, 45.07, 43.44, 46.53, 45.65, 45.04, 47.75, 45.84, 47.73, 47.23, 46.22, 45.76, 47.22, 47.77, 44.68, 46.76, 46.65, 47.87, 46.3, 46.64, 47.05, 48.27, 44.99, 47.12, 46.22, 48.37, 43.35, 45.97, 46.45, 45.23, 44.71, 46.79, 45.83, 46.73, 46.47, 46.39, 45.9, 47.64, 47.8, 46.03, 47.89, 47.29, 47.93, 46.54, 44.29, 47.31, 47.42, 45.85, 45.88, 46.64, 47.1, 43.68, 46.83, 46.1, 47.22, 45.05, 46.57, 44.23, 46.93, 45.89, 45.87, 45.74, 46.74, 44.63, 46.45, 46.97, 45.86]

rdm_07_min = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 11.69, 16.57, 16.92, 19.09, 18.71, 22.43, 27.79, 25.7, 32.26, 30.02, 33.35, 32.7, 34.46, 37.34, 35.32, 38.35, 34.7, 38.47, 39.69, 38.39, 40.74, 41.49, 42.09, 43.93, 44.01, 41.95, 44.35, 43.44, 42.74, 43.96, 44.55, 43.16, 45.36, 45.39, 46.73, 46.59, 45.19, 46.92, 46.24, 47.72, 47.06, 44.49, 45.76, 46.86, 46.28, 46.63, 47.07, 46.46, 46.64, 44.96, 46.82, 46.56, 46.38, 45.72, 46.17, 47.06, 45.59, 45.76, 46.92, 46.77, 46.93, 45.47, 45.92, 46.14, 46.5, 46.31, 45.87, 46.84, 46.99, 47.17, 45.76, 45.72, 46.6, 45.97, 44.05, 46.06, 46.29, 45.77, 44.76, 46.0, 44.9, 46.02, 44.22, 45.44, 45.11, 46.43, 46.47, 46.34, 44.77, 46.17, 45.44, 46.79, 46.52, 46.03]
rdm_07_max = [10.01, 10.0, 10.0, 10.0, 13.65, 17.57, 20.27, 25.07, 25.53, 26.12, 24.91, 29.57, 33.24, 33.85, 36.23, 34.23, 36.96, 37.12, 36.41, 42.05, 39.6, 41.59, 39.2, 41.07, 43.39, 42.07, 43.38, 45.12, 45.61, 47.44, 48.46, 45.35, 48.01, 48.04, 47.53, 48.81, 47.97, 46.04, 49.8, 49.17, 48.66, 49.57, 47.69, 48.38, 48.17, 50.41, 49.48, 48.0, 48.49, 49.88, 48.53, 48.84, 48.96, 49.55, 48.1, 47.84, 48.57, 48.44, 48.49, 47.97, 48.25, 48.32, 46.85, 48.21, 49.09, 48.35, 48.59, 46.94, 48.21, 48.07, 48.26, 48.04, 47.4, 48.72, 48.06, 48.58, 47.31, 47.16, 48.25, 47.18, 46.53, 47.68, 47.97, 47.54, 46.6, 47.66, 46.91, 47.46, 45.91, 46.95, 47.12, 48.08, 47.94, 47.98, 46.38, 47.94, 47.14, 47.95, 47.22, 47.11]
rdm_07_mean = [10.0, 10.0, 10.0, 10.0, 11.5, 14.21, 17.61, 21.15, 21.36, 22.54, 21.73, 26.17, 30.13, 28.85, 34.02, 32.05, 35.24, 35.5, 35.43, 39.66, 37.46, 39.62, 37.51, 39.64, 41.5, 40.39, 42.19, 43.14, 43.95, 45.61, 46.37, 43.64, 46.55, 45.53, 45.37, 46.58, 46.55, 45.05, 47.36, 47.46, 47.6, 47.79, 46.43, 47.68, 47.37, 48.95, 48.12, 46.29, 46.99, 48.44, 47.23, 47.91, 47.98, 48.27, 47.31, 46.11, 47.75, 47.4, 47.37, 46.91, 47.29, 47.48, 45.98, 46.91, 47.99, 47.36, 47.56, 46.1, 47.03, 47.13, 47.28, 47.34, 46.66, 47.96, 47.62, 47.87, 46.37, 46.53, 47.36, 46.83, 45.54, 47.08, 47.08, 46.71, 46.01, 46.9, 45.95, 46.7, 45.14, 46.3, 46.19, 47.07, 47.4, 47.07, 45.6, 46.91, 46.34, 47.39, 46.88, 46.55]

acc_03_min = [7.87, 10.0, 10.0, 14.56, 17.68, 17.88, 17.46, 18.05, 18.3, 18.12, 18.08, 18.29, 18.58, 18.34, 18.59, 19.67, 20.66, 21.23, 21.86, 24.21, 23.77, 26.11, 26.4, 26.96, 26.42, 27.68, 28.4, 27.94, 28.3, 28.8, 29.93, 28.76, 30.87, 30.82, 31.15, 30.88, 31.59, 30.2, 30.58, 31.3, 31.6, 30.99, 32.7, 30.68, 32.13, 31.8, 30.91, 32.8, 32.82, 32.79, 32.67, 32.23, 33.65, 32.79, 33.64, 32.15, 33.24, 33.18, 33.24, 33.24, 33.0, 33.98, 32.68, 32.73, 33.23, 32.46, 33.62, 32.45, 32.77, 33.11, 32.31, 33.7, 33.45, 33.08, 33.44, 33.52, 33.24, 33.15, 33.07, 33.32, 33.66, 33.64, 33.57, 33.24, 33.77, 33.44, 33.67, 33.72, 34.07, 32.95, 33.72, 33.37, 33.41, 33.28, 33.65, 33.24, 33.04, 33.48, 33.44, 33.55]
acc_03_max = [10.33, 14.38, 17.79, 18.15, 18.4, 19.17, 19.8, 20.1, 21.52, 29.75, 23.52, 22.72, 25.22, 26.57, 26.78, 26.03, 26.73, 24.31, 31.21, 29.85, 30.94, 30.93, 30.03, 35.11, 30.1, 33.6, 32.85, 30.95, 33.36, 33.6, 33.08, 36.42, 34.39, 35.03, 33.52, 35.29, 34.23, 33.87, 37.82, 34.27, 36.61, 36.44, 34.74, 34.61, 34.42, 37.37, 34.02, 34.97, 35.31, 36.06, 36.7, 35.21, 35.04, 37.8, 34.43, 35.06, 36.97, 35.73, 34.98, 37.69, 37.9, 35.11, 36.4, 37.49, 34.62, 37.84, 37.84, 34.82, 37.9, 34.88, 34.93, 36.15, 36.98, 35.0, 35.23, 35.11, 35.26, 35.69, 35.25, 35.26, 34.8, 37.47, 34.81, 34.89, 36.57, 35.65, 36.93, 37.53, 37.71, 34.7, 35.99, 35.43, 37.92, 35.34, 37.35, 34.99, 34.89, 34.93, 38.21, 34.85]
acc_03_mean = [9.64, 10.88, 14.17, 17.03, 18.04, 18.21, 18.26, 18.64, 19.34, 21.07, 19.36, 19.66, 20.06, 21.67, 21.19, 21.6, 22.41, 22.56, 24.84, 25.97, 26.61, 27.89, 27.61, 29.76, 27.98, 29.39, 30.29, 29.49, 30.48, 31.36, 31.67, 31.94, 31.86, 32.45, 32.35, 32.62, 33.07, 32.42, 33.46, 32.69, 33.37, 33.62, 33.68, 32.92, 33.06, 34.17, 33.26, 33.95, 34.03, 34.77, 34.34, 33.84, 34.3, 35.27, 34.1, 33.72, 34.88, 34.15, 34.24, 34.61, 35.25, 34.62, 34.25, 35.02, 33.95, 34.66, 35.24, 34.09, 34.54, 34.25, 33.66, 34.62, 35.05, 34.07, 34.3, 34.36, 34.3, 34.79, 34.1, 34.41, 34.29, 35.22, 34.21, 34.34, 34.97, 34.59, 34.86, 34.86, 36.34, 34.04, 34.85, 34.78, 34.97, 34.31, 34.68, 34.2, 34.26, 34.3, 34.86, 34.44]

acc_05_min = [10.0, 10.0, 10.0, 10.0, 10.96, 17.89, 18.39, 18.9, 17.47, 19.49, 19.37, 21.0, 19.67, 22.26, 23.73, 24.61, 25.24, 27.22, 25.61, 28.1, 26.24, 28.14, 28.85, 28.82, 29.02, 30.11, 29.97, 30.62, 31.71, 31.01, 32.66, 33.01, 32.9, 33.8, 33.78, 34.83, 35.52, 34.42, 36.01, 36.18, 36.72, 37.07, 37.18, 36.83, 37.88, 37.62, 37.13, 36.68, 37.53, 37.28, 37.42, 37.89, 36.7, 38.01, 37.81, 38.11, 37.62, 37.94, 37.49, 37.51, 38.13, 37.59, 37.5, 37.72, 37.37, 37.5, 38.01, 37.29, 37.73, 37.28, 37.62, 37.58, 37.96, 37.59, 37.91, 37.94, 37.84, 37.87, 37.82, 38.28, 37.71, 37.77, 37.81, 37.34, 37.75, 38.14, 37.79, 37.88, 37.19, 37.37, 37.58, 38.13, 37.85, 37.42, 37.83, 37.3, 37.88, 37.52, 37.66, 37.47]
acc_05_max = [10.0, 10.0, 12.43, 17.29, 19.49, 20.84, 20.73, 21.29, 23.53, 24.15, 24.87, 25.04, 26.66, 26.16, 28.04, 27.26, 28.13, 28.93, 31.92, 30.58, 29.99, 30.53, 31.43, 32.31, 32.97, 32.93, 33.76, 34.79, 35.28, 36.25, 35.94, 36.24, 36.5, 35.94, 37.36, 38.33, 37.9, 38.74, 38.98, 37.99, 39.35, 39.92, 38.78, 39.3, 39.69, 40.15, 39.52, 40.57, 40.22, 39.44, 40.16, 40.12, 39.75, 40.2, 39.8, 39.67, 39.79, 39.66, 39.81, 39.69, 40.38, 39.82, 39.87, 39.41, 40.27, 40.09, 39.62, 39.87, 39.67, 40.15, 39.85, 40.16, 39.77, 39.79, 39.88, 39.8, 39.53, 39.7, 39.64, 39.61, 39.64, 39.93, 39.7, 39.34, 39.49, 39.38, 39.72, 39.56, 39.53, 39.99, 39.92, 39.48, 40.14, 39.63, 39.84, 39.84, 39.96, 40.05, 39.85, 39.9]
acc_05_mean = [10.0, 10.0, 10.49, 12.65, 16.0, 18.76, 19.38, 19.96, 20.01, 21.13, 21.45, 22.59, 23.2, 24.32, 26.25, 26.06, 26.64, 27.95, 28.37, 29.14, 28.64, 29.13, 30.19, 30.29, 30.93, 31.92, 32.28, 32.21, 33.1, 33.9, 33.96, 34.28, 35.42, 34.88, 36.07, 36.4, 36.71, 36.68, 37.46, 37.42, 37.93, 38.52, 38.21, 38.21, 38.75, 38.66, 38.12, 38.62, 39.01, 38.66, 38.73, 38.92, 38.54, 39.18, 38.85, 39.1, 39.14, 38.95, 38.94, 38.86, 39.17, 38.93, 38.86, 38.82, 38.99, 39.09, 38.93, 38.94, 38.91, 38.95, 39.0, 38.95, 39.08, 38.9, 38.92, 39.06, 38.87, 38.87, 38.95, 38.99, 38.92, 39.07, 38.94, 38.68, 38.9, 38.87, 38.93, 38.91, 38.71, 38.86, 38.95, 39.06, 39.1, 38.99, 39.1, 38.84, 39.14, 39.07, 39.2, 39.13]

acc_07_min = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.38, 11.83, 13.08, 14.56, 18.41, 19.09, 18.67, 18.69, 19.55, 19.74, 21.03, 21.83, 23.86, 26.13, 27.43, 27.43, 28.7, 30.53, 29.74, 31.95, 31.67, 32.73, 33.39, 32.74, 33.84, 34.46, 34.75, 34.79, 36.01, 35.56, 36.82, 37.36, 37.42, 36.88, 38.5, 38.61, 39.06, 38.44, 38.43, 39.38, 39.2, 38.66, 40.24, 39.82, 40.37, 39.95, 40.21, 39.72, 40.73, 40.56, 40.65, 40.33, 40.95, 40.79, 40.81, 40.69, 40.29, 40.78, 40.67, 40.98, 40.89, 40.42, 40.67, 40.92, 40.54, 40.42, 40.33, 41.19, 40.76, 40.9, 40.56, 40.28, 40.85, 40.84, 40.65, 40.83, 40.78, 40.99, 40.5, 40.55, 41.03, 40.46, 40.76, 40.61, 40.51, 40.51, 40.8, 40.28, 40.42, 40.74, 40.35, 40.27, 40.11, 40.64]
acc_07_max = [10.0, 10.0, 10.0, 14.81, 18.66, 18.8, 19.38, 20.25, 20.22, 20.43, 20.8, 22.27, 26.63, 27.86, 28.68, 29.67, 31.37, 32.27, 33.14, 33.36, 34.11, 35.07, 34.85, 35.51, 34.82, 37.89, 36.68, 38.39, 39.15, 38.15, 39.29, 39.46, 40.89, 40.08, 40.43, 40.26, 41.87, 41.06, 41.13, 41.0, 41.65, 41.49, 41.68, 41.52, 42.06, 41.75, 42.15, 41.69, 42.32, 41.72, 41.95, 42.21, 42.33, 41.67, 41.98, 42.04, 41.96, 42.18, 42.07, 42.22, 42.1, 42.26, 42.0, 41.98, 42.12, 42.52, 41.93, 42.54, 42.35, 41.9, 41.78, 42.01, 42.03, 42.0, 42.33, 42.05, 41.91, 42.02, 42.18, 41.86, 41.88, 42.03, 41.89, 41.96, 42.23, 42.03, 42.04, 41.61, 42.16, 42.26, 41.96, 42.11, 41.87, 41.74, 41.7, 41.86, 42.22, 42.02, 42.01, 42.05]
acc_07_mean = [10.0, 10.0, 10.0, 10.96, 11.74, 11.84, 14.76, 16.83, 18.08, 18.7, 19.74, 20.38, 21.82, 23.13, 24.35, 24.94, 26.76, 27.17, 28.69, 29.71, 30.89, 31.25, 31.73, 32.64, 32.66, 34.06, 34.28, 35.15, 35.79, 35.75, 36.34, 36.7, 37.46, 37.13, 37.98, 38.23, 38.77, 39.19, 39.18, 39.08, 40.04, 39.63, 40.41, 39.81, 39.9, 40.77, 40.2, 40.45, 40.95, 40.83, 40.87, 41.16, 41.29, 40.87, 41.26, 41.35, 41.25, 41.31, 41.77, 41.27, 41.37, 41.44, 41.08, 41.31, 41.31, 41.43, 41.26, 41.22, 41.4, 41.32, 41.2, 41.23, 41.27, 41.54, 41.42, 41.35, 41.31, 41.28, 41.42, 41.38, 41.19, 41.32, 41.37, 41.4, 41.21, 41.17, 41.33, 41.19, 41.32, 41.26, 40.95, 41.14, 41.09, 41.01, 41.08, 41.2, 41.19, 41.12, 41.14, 41.18]

data = {
    "rdm_03_min": rdm_03_min,
    "rdm_03_max": rdm_03_max,
    "rdm_03_mean": rdm_03_mean,
    "rdm_05_min": rdm_05_min,
    "rdm_05_max": rdm_05_max,
    "rdm_05_mean": rdm_05_mean,
    "rdm_07_min": rdm_07_min,
    "rdm_07_max": rdm_07_max,
    "rdm_07_mean": rdm_07_mean,
    "acc_03_min": acc_03_min,
    "acc_03_max": acc_03_max,
    "acc_03_mean": acc_03_mean,
    "acc_05_min": acc_05_min,
    "acc_05_max": acc_05_max,
    "acc_05_mean": acc_05_mean,
    "acc_07_min": acc_07_min,
    "acc_07_max": acc_07_max,
    "acc_07_mean": acc_07_mean,
}

save_path = None
if len(sys.argv) == 3 and sys.argv[1] and sys.argv[1] == "save":
    save_path = sys.argv[2]

plot_node_acc("", data, legend_pos="in", save_path=save_path, plot_size="2")