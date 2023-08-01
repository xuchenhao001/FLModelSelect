import sys

from plot.utils import plot_round_acc

accuracy_min = [10.0, 10.0, 10.0, 10.0, 10.0, 10.07, 10.49, 17.28, 18.62, 18.75, 17.88, 19.2, 19.61, 19.4, 20.1, 20.65, 21.36, 24.57, 25.88, 27.61, 29.26, 29.38, 30.49, 31.13, 31.86, 32.06, 33.0, 33.24, 33.67, 34.11, 34.33, 35.47, 35.17, 35.56, 36.19, 37.12, 37.18, 36.81, 38.68, 38.05, 38.98, 38.43, 39.08, 39.5, 38.96, 39.16, 39.44, 38.08, 40.03, 40.27, 40.1, 40.15, 40.83, 41.02, 40.4, 40.03, 41.14, 40.28, 40.42, 40.79, 40.33, 40.43, 40.94, 40.66, 40.64, 40.48, 40.82, 40.43, 41.04, 40.96, 40.89, 39.93, 40.44, 40.0, 41.19, 40.68, 40.15, 41.15, 40.26, 40.47, 40.93, 40.05, 40.75, 40.71, 40.34, 40.68, 41.0, 40.0, 40.45, 41.05, 40.8, 40.41, 40.26, 40.4, 40.63, 40.28, 40.79, 40.8, 40.65, 40.23]
accuracy_max = [10.04, 10.0, 10.0, 15.0, 18.08, 19.36, 20.09, 20.21, 19.98, 20.15, 21.62, 23.17, 24.03, 27.28, 28.16, 29.55, 31.21, 31.94, 31.69, 32.65, 34.13, 33.6, 33.83, 34.06, 34.54, 36.28, 36.59, 37.22, 37.26, 37.29, 37.9, 38.67, 38.51, 39.44, 39.99, 39.78, 41.04, 41.06, 41.03, 41.78, 42.91, 41.68, 41.8, 42.81, 42.35, 42.11, 42.2, 41.79, 42.93, 42.32, 42.43, 42.76, 43.48, 42.27, 43.38, 43.28, 43.7, 43.25, 43.88, 43.04, 43.39, 42.69, 43.32, 43.42, 43.12, 42.88, 42.94, 43.0, 42.88, 42.48, 42.7, 43.06, 42.73, 42.81, 43.1, 42.94, 42.96, 43.09, 43.03, 42.95, 42.67, 43.41, 42.33, 42.59, 43.16, 42.67, 42.5, 42.94, 42.7, 42.64, 42.41, 43.01, 42.99, 42.58, 42.34, 42.54, 42.39, 42.78, 42.62, 42.83]
accuracy_mean = [10.01, 10.0, 10.0, 11.25, 12.06, 14.3, 16.86, 18.53, 19.55, 19.24, 19.48, 20.48, 21.3, 22.36, 23.61, 24.98, 26.58, 28.06, 29.0, 30.05, 31.26, 31.18, 32.44, 32.84, 33.58, 34.31, 34.83, 35.01, 35.58, 35.76, 36.03, 36.96, 37.19, 37.75, 38.06, 38.29, 38.6, 39.01, 39.88, 39.66, 40.54, 39.94, 40.3, 40.91, 40.86, 40.56, 40.58, 40.37, 41.19, 41.17, 41.11, 41.48, 41.8, 41.62, 41.92, 41.68, 42.16, 42.03, 42.12, 41.91, 42.0, 41.5, 41.87, 41.96, 41.77, 41.74, 41.65, 41.64, 41.96, 41.91, 41.89, 41.73, 41.52, 41.55, 41.88, 41.67, 41.72, 42.05, 41.76, 41.4, 41.73, 41.48, 41.44, 41.6, 41.62, 41.65, 41.77, 41.58, 41.52, 41.67, 41.5, 41.7, 41.54, 41.45, 41.42, 41.32, 41.62, 41.56, 41.43, 41.5]
datasize_min = [10.0, 10.0, 10.38, 16.85, 18.44, 19.66, 23.0, 29.3, 29.32, 32.93, 33.16, 34.72, 35.94, 37.74, 39.79, 40.49, 40.63, 42.74, 43.27, 44.66, 45.35, 46.15, 46.49, 44.61, 46.57, 46.59, 46.94, 46.61, 47.02, 46.59, 45.92, 45.97, 46.39, 46.26, 46.62, 46.46, 46.1, 46.61, 45.19, 46.43, 46.18, 46.17, 45.99, 45.65, 46.21, 44.83, 46.31, 46.18, 45.36, 45.98, 45.96, 45.33, 45.73, 45.6, 45.35, 45.28, 45.22, 45.3, 45.29, 45.43, 45.17, 45.56, 45.55, 45.46, 44.78, 45.15, 45.21, 45.01, 45.25, 45.46, 45.33, 45.57, 45.13, 45.45, 45.55, 45.29, 45.02, 45.28, 45.42, 45.4, 45.54, 45.13, 43.71, 44.97, 45.59, 45.39, 46.02, 45.39, 45.27, 45.44, 45.64, 45.17, 45.18, 45.58, 45.46, 45.26, 45.31, 45.38, 45.53, 45.39]
datasize_max = [10.0, 11.97, 15.74, 19.26, 23.91, 26.05, 28.18, 32.15, 34.07, 36.08, 36.56, 39.48, 39.4, 42.46, 42.09, 43.3, 44.95, 45.3, 45.73, 46.61, 47.45, 47.31, 47.12, 47.89, 48.67, 47.73, 48.3, 48.9, 48.91, 47.81, 48.6, 48.14, 47.86, 47.95, 48.01, 48.08, 48.08, 48.38, 48.29, 48.07, 47.28, 47.49, 47.62, 47.14, 47.03, 47.77, 48.09, 47.54, 47.46, 47.99, 47.3, 47.53, 46.84, 47.51, 47.33, 47.37, 47.52, 47.44, 47.37, 47.47, 47.35, 47.39, 47.24, 47.45, 47.27, 47.35, 47.4, 46.76, 47.39, 47.36, 46.9, 47.18, 47.41, 47.32, 47.06, 47.2, 47.17, 47.19, 46.73, 47.0, 47.42, 47.57, 47.08, 47.65, 47.21, 46.86, 47.34, 47.25, 46.9, 46.54, 46.93, 47.25, 47.18, 47.69, 47.53, 47.32, 47.28, 47.68, 47.34, 47.0]
datasize_mean = [10.0, 10.49, 14.16, 17.83, 20.45, 23.33, 26.46, 30.88, 32.59, 34.62, 35.14, 37.3, 38.14, 39.8, 40.7, 41.94, 42.82, 43.98, 44.49, 45.68, 46.32, 46.68, 46.8, 46.86, 47.43, 47.08, 47.3, 47.47, 47.72, 47.29, 47.6, 47.25, 47.38, 47.15, 47.27, 47.47, 47.0, 47.46, 47.05, 47.38, 46.92, 47.04, 47.05, 46.72, 46.53, 46.75, 47.07, 47.07, 46.7, 47.02, 46.61, 46.7, 46.5, 46.81, 46.54, 46.44, 46.6, 46.38, 46.4, 46.49, 46.54, 46.56, 46.71, 46.66, 46.21, 46.42, 46.27, 46.13, 46.53, 46.43, 46.4, 46.35, 46.27, 46.33, 46.14, 46.3, 46.26, 46.27, 46.22, 46.46, 46.41, 46.52, 45.85, 46.42, 46.45, 46.19, 46.56, 46.28, 46.31, 46.19, 46.36, 46.26, 45.95, 46.4, 46.34, 46.18, 46.21, 46.56, 46.36, 46.35]
entropy_max_min = [10.0, 10.0, 10.0, 10.0, 10.11, 14.93, 15.63, 16.17, 24.32, 23.41, 27.79, 30.72, 31.99, 33.42, 34.61, 33.85, 36.5, 38.19, 39.73, 40.57, 40.29, 41.5, 42.66, 43.31, 45.13, 45.13, 44.74, 45.83, 46.37, 45.29, 47.13, 46.93, 44.65, 48.02, 46.88, 46.73, 45.97, 47.22, 47.0, 45.32, 46.63, 46.23, 46.0, 46.71, 46.84, 46.81, 46.02, 46.93, 46.28, 46.33, 46.99, 45.36, 46.51, 46.22, 45.29, 45.07, 46.01, 45.75, 45.09, 46.02, 45.67, 45.53, 44.69, 45.97, 45.84, 45.01, 44.56, 45.88, 44.24, 45.87, 44.34, 46.2, 44.52, 44.59, 45.29, 44.44, 44.55, 44.38, 44.2, 45.26, 45.02, 46.76, 43.73, 46.16, 44.39, 45.61, 45.3, 45.26, 44.76, 45.58, 45.17, 45.07, 44.99, 43.63, 45.64, 46.03, 44.82, 45.77, 45.34, 45.54]
entropy_max_max = [10.0, 10.0, 10.0, 16.9, 17.02, 18.67, 21.91, 25.78, 31.67, 34.02, 35.45, 35.59, 37.88, 38.4, 39.69, 40.5, 42.05, 43.07, 44.26, 43.8, 45.75, 46.15, 45.99, 45.1, 46.49, 48.21, 47.97, 48.99, 48.12, 48.52, 48.35, 48.45, 47.62, 48.59, 48.67, 49.45, 48.15, 49.12, 47.87, 48.91, 49.17, 48.9, 47.75, 47.67, 48.86, 49.07, 49.12, 48.67, 47.45, 48.26, 48.05, 48.33, 47.46, 47.95, 48.06, 47.76, 48.28, 49.02, 48.65, 47.32, 48.18, 47.86, 48.37, 47.64, 46.65, 48.23, 46.97, 48.15, 46.93, 48.0, 46.13, 47.91, 46.92, 47.45, 46.57, 48.31, 46.83, 48.5, 48.18, 47.02, 46.26, 47.88, 47.37, 46.68, 47.66, 47.02, 47.37, 47.43, 47.17, 47.21, 47.62, 46.94, 47.01, 46.53, 46.53, 47.96, 46.45, 47.1, 47.02, 47.28]
entropy_max_mean = [10.0, 10.0, 10.0, 11.73, 12.4, 16.9, 19.48, 21.68, 27.04, 27.96, 30.68, 32.6, 34.86, 36.26, 36.75, 37.44, 38.91, 40.5, 41.56, 41.8, 43.18, 43.74, 43.87, 44.64, 45.69, 46.37, 46.16, 47.58, 46.98, 46.82, 47.88, 47.54, 46.5, 48.16, 47.59, 48.1, 47.24, 47.96, 47.55, 47.1, 48.05, 47.2, 46.97, 47.34, 47.98, 47.72, 47.76, 47.56, 47.09, 47.56, 47.5, 47.14, 47.1, 46.84, 46.67, 46.27, 47.1, 47.02, 46.88, 46.88, 46.86, 46.53, 46.5, 46.54, 46.22, 46.75, 45.81, 47.16, 45.74, 47.07, 45.24, 46.8, 45.64, 46.51, 45.9, 46.47, 46.0, 46.51, 45.67, 46.22, 45.6, 47.24, 45.4, 46.45, 45.76, 46.58, 45.99, 46.36, 46.19, 46.45, 46.38, 46.28, 46.02, 45.44, 45.99, 46.94, 45.6, 46.52, 46.42, 46.41]
entropy_min_min = [10.0, 9.92, 9.59, 10.0, 10.0, 10.0, 9.94, 10.64, 17.82, 18.11, 17.92, 18.07, 17.97, 18.24, 18.32, 19.34, 20.63, 22.66, 24.89, 25.11, 27.29, 27.35, 28.9, 29.04, 29.69, 30.34, 31.61, 31.68, 32.75, 32.96, 35.07, 35.93, 35.56, 35.58, 35.52, 37.37, 37.78, 37.1, 39.33, 39.07, 39.35, 38.52, 39.76, 39.66, 40.25, 39.15, 39.64, 40.78, 40.6, 40.89, 40.59, 40.42, 41.36, 40.32, 41.1, 41.13, 41.23, 40.67, 40.06, 40.76, 40.96, 41.25, 41.17, 41.37, 41.35, 41.06, 40.9, 41.45, 41.13, 41.08, 41.48, 40.96, 41.17, 40.61, 41.24, 40.87, 40.75, 40.78, 39.94, 40.96, 41.05, 40.92, 40.66, 40.51, 40.16, 40.53, 40.42, 40.09, 40.59, 40.09, 39.87, 40.62, 40.46, 40.49, 40.2, 39.45, 39.67, 40.15, 40.3, 40.64]
entropy_min_max = [10.0, 10.0, 13.37, 15.74, 19.02, 18.58, 18.74, 18.34, 18.87, 19.56, 19.92, 24.22, 26.7, 28.21, 27.96, 28.53, 28.09, 30.24, 30.62, 30.2, 31.77, 33.24, 32.94, 33.97, 34.24, 34.67, 34.33, 36.67, 36.71, 35.94, 35.62, 37.29, 37.53, 39.26, 38.94, 39.68, 40.09, 40.26, 41.35, 41.37, 41.07, 41.76, 41.82, 42.03, 41.41, 42.45, 42.02, 43.93, 43.69, 43.73, 42.2, 42.74, 42.52, 43.13, 42.77, 42.63, 42.89, 43.16, 42.88, 43.17, 42.7, 43.05, 42.4, 42.44, 42.57, 42.45, 41.95, 41.6, 42.56, 43.03, 42.75, 42.55, 43.71, 42.71, 43.16, 42.08, 41.62, 42.32, 42.11, 41.59, 41.4, 42.18, 41.7, 41.8, 41.71, 41.68, 42.26, 42.19, 41.9, 42.17, 42.12, 41.98, 42.07, 42.27, 42.1, 42.17, 42.06, 43.21, 43.45, 42.47]
entropy_min_mean = [10.0, 9.98, 10.74, 11.44, 13.67, 14.37, 14.25, 15.18, 18.44, 18.92, 18.84, 20.04, 20.85, 21.92, 22.36, 23.71, 24.44, 25.72, 26.89, 27.46, 29.4, 29.62, 30.58, 31.42, 31.54, 32.25, 32.95, 33.43, 34.1, 34.63, 35.39, 36.49, 36.66, 37.54, 37.52, 38.1, 39.03, 39.02, 39.91, 39.87, 40.06, 40.34, 40.73, 40.74, 40.74, 40.82, 41.07, 42.0, 42.29, 41.98, 41.51, 41.6, 42.0, 41.88, 41.99, 41.82, 41.84, 41.89, 41.62, 41.58, 41.83, 42.03, 41.74, 41.97, 41.82, 41.6, 41.59, 41.53, 41.61, 41.91, 41.93, 41.87, 42.05, 41.8, 41.96, 41.45, 41.25, 41.31, 41.08, 41.34, 41.26, 41.34, 41.19, 41.0, 40.88, 41.08, 41.73, 41.31, 41.22, 41.04, 40.66, 41.32, 41.0, 40.97, 40.7, 40.6, 40.63, 41.03, 41.16, 41.22]
gradiv_max_min = [10.0, 10.0, 10.0, 10.31, 15.07, 17.17, 18.25, 20.31, 22.81, 27.97, 31.09, 35.32, 35.37, 37.53, 37.05, 39.43, 38.31, 40.55, 39.67, 40.02, 39.97, 41.85, 42.31, 43.23, 43.56, 44.35, 44.16, 46.13, 44.83, 45.78, 46.24, 45.95, 46.76, 46.02, 46.88, 46.59, 46.11, 45.89, 44.84, 46.38, 45.26, 45.88, 45.5, 45.97, 45.58, 44.87, 46.57, 45.42, 45.29, 46.04, 43.99, 45.05, 44.0, 45.19, 45.96, 43.11, 45.76, 44.42, 44.59, 45.99, 44.95, 45.8, 44.81, 46.21, 45.17, 45.29, 44.23, 45.95, 45.8, 43.92, 44.47, 44.52, 44.92, 45.54, 45.33, 44.54, 45.08, 43.68, 44.25, 43.7, 45.25, 45.31, 43.07, 43.55, 42.6, 45.11, 44.63, 45.17, 44.11, 44.87, 43.98, 45.34, 44.67, 44.71, 43.49, 44.4, 45.09, 44.43, 44.53, 44.41]
gradiv_max_max = [10.0, 10.0, 14.62, 16.97, 18.44, 21.02, 25.74, 28.87, 33.83, 35.72, 37.09, 38.99, 37.51, 39.79, 39.32, 41.92, 42.85, 42.53, 43.06, 43.28, 45.32, 46.17, 45.42, 46.87, 47.68, 47.34, 48.47, 48.39, 48.5, 48.1, 48.07, 49.44, 48.24, 49.32, 49.08, 48.56, 50.14, 47.98, 48.82, 48.84, 49.75, 49.08, 48.81, 47.58, 49.65, 47.61, 49.25, 47.26, 47.39, 49.26, 48.0, 46.91, 48.02, 47.99, 47.21, 49.35, 48.29, 48.49, 47.98, 47.51, 48.73, 46.3, 49.01, 47.8, 47.72, 47.55, 47.66, 47.38, 47.3, 47.64, 48.08, 46.56, 47.44, 46.38, 48.3, 46.41, 48.41, 45.57, 48.1, 47.7, 47.36, 47.99, 47.63, 47.74, 48.57, 46.31, 47.47, 47.13, 47.74, 46.64, 46.86, 46.5, 47.9, 47.11, 46.81, 46.71, 47.57, 46.59, 47.43, 47.75]
gradiv_max_mean = [10.0, 10.0, 11.16, 12.22, 16.95, 18.57, 21.83, 25.47, 28.86, 32.62, 34.53, 36.58, 36.71, 38.54, 38.68, 40.64, 40.84, 41.58, 41.84, 41.95, 43.04, 44.09, 44.11, 45.17, 45.54, 46.24, 46.68, 47.44, 46.85, 47.16, 47.18, 47.28, 47.47, 47.76, 47.84, 47.48, 48.18, 47.35, 47.42, 47.67, 47.42, 47.64, 47.16, 46.87, 47.67, 46.39, 47.85, 46.14, 46.42, 47.36, 46.21, 46.38, 45.99, 46.36, 46.5, 46.34, 46.68, 46.39, 45.61, 46.63, 46.59, 45.95, 46.72, 46.89, 46.43, 46.18, 45.9, 46.68, 46.47, 46.16, 45.83, 45.6, 46.42, 45.87, 46.36, 45.26, 46.26, 44.89, 45.82, 45.51, 46.1, 46.9, 45.64, 45.21, 46.08, 45.71, 45.93, 46.04, 45.89, 45.8, 45.7, 45.65, 45.95, 46.05, 45.29, 45.68, 45.85, 45.77, 45.51, 45.82]
gradiv_min_min = [10.0, 8.38, 12.54, 17.91, 18.02, 18.95, 20.28, 23.93, 26.04, 28.18, 29.15, 30.18, 30.65, 31.17, 32.23, 33.05, 34.42, 34.3, 34.72, 34.81, 36.14, 37.01, 36.79, 37.1, 37.73, 38.18, 38.69, 39.32, 39.42, 39.53, 39.33, 39.89, 39.49, 40.13, 40.58, 40.08, 40.51, 40.87, 41.04, 39.98, 40.29, 40.32, 41.14, 40.18, 40.93, 41.03, 41.43, 40.46, 40.68, 39.84, 39.98, 40.1, 39.84, 39.46, 38.95, 39.41, 39.65, 39.46, 38.26, 38.92, 38.53, 38.03, 38.25, 37.58, 37.59, 37.94, 37.49, 37.05, 37.55, 37.46, 37.55, 37.1, 37.14, 36.85, 37.32, 37.02, 36.93, 37.07, 37.45, 36.92, 36.72, 36.92, 37.5, 37.29, 37.29, 37.5, 37.17, 36.8, 36.73, 36.64, 36.91, 37.3, 36.56, 37.15, 36.98, 36.46, 36.86, 36.83, 36.53, 36.74]
gradiv_min_max = [10.01, 17.55, 18.27, 18.96, 22.64, 26.47, 27.72, 28.43, 30.01, 30.18, 30.85, 32.53, 33.41, 34.48, 35.32, 35.65, 36.03, 36.86, 37.69, 38.25, 39.01, 39.24, 39.22, 39.52, 40.03, 40.52, 40.3, 41.14, 41.26, 41.42, 41.14, 41.63, 40.83, 40.51, 42.18, 41.67, 42.67, 41.89, 43.03, 41.75, 41.8, 42.11, 41.83, 42.2, 42.24, 42.76, 41.96, 42.59, 42.38, 42.21, 41.97, 42.48, 41.65, 41.77, 41.65, 42.04, 41.63, 42.17, 40.99, 40.48, 40.91, 40.22, 40.45, 40.58, 39.82, 39.55, 39.61, 39.47, 39.51, 39.66, 39.2, 38.95, 39.12, 38.84, 39.15, 38.99, 39.26, 39.06, 39.24, 39.24, 38.74, 39.04, 38.92, 39.03, 39.01, 38.85, 38.82, 38.92, 39.13, 38.59, 38.92, 39.09, 38.91, 38.85, 39.11, 38.86, 38.91, 38.78, 38.85, 39.06]
gradiv_min_mean = [10.0, 13.6, 16.76, 18.34, 20.68, 23.26, 25.09, 26.51, 27.98, 28.95, 30.19, 30.96, 32.02, 33.08, 33.47, 34.39, 34.88, 35.62, 36.32, 36.79, 37.48, 38.07, 38.15, 38.61, 39.34, 39.63, 39.33, 40.33, 40.16, 40.61, 40.08, 40.65, 40.27, 40.32, 41.21, 40.93, 41.45, 41.45, 41.84, 41.0, 41.34, 41.33, 41.45, 41.27, 41.56, 41.73, 41.69, 41.62, 41.47, 41.34, 40.99, 41.36, 41.14, 41.0, 40.56, 40.73, 40.57, 40.61, 39.66, 39.7, 39.6, 39.17, 39.28, 39.26, 38.88, 38.64, 38.44, 38.47, 38.5, 38.42, 38.37, 38.17, 38.11, 38.06, 38.14, 38.06, 38.14, 38.08, 38.3, 37.98, 37.86, 37.99, 38.06, 37.95, 38.04, 38.11, 37.92, 38.04, 38.01, 37.63, 37.94, 38.04, 37.7, 38.02, 38.12, 37.72, 37.96, 37.86, 37.8, 37.96]
loss_max_min = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.23, 11.0, 12.55, 13.4, 15.58, 17.71, 18.52, 19.87, 19.95, 22.44, 22.45, 25.4, 27.65, 28.31, 29.34, 30.72, 30.3, 31.52, 31.09, 32.12, 31.77, 31.96, 32.7, 33.02, 33.81, 34.28, 35.1, 36.44, 37.11, 36.53, 37.26, 38.51, 38.4, 39.0, 38.41, 39.01, 39.98, 39.48, 39.32, 40.16, 39.89, 39.95, 39.74, 40.43, 39.6, 40.76, 40.63, 41.41, 40.8, 41.42, 41.5, 41.13, 40.46, 41.27, 41.06, 41.24, 40.79, 40.84, 41.17, 41.41, 40.69, 40.94, 40.97, 41.19, 40.88, 41.18, 41.1, 41.41, 41.16, 40.84, 41.24, 41.32, 41.03, 40.34, 40.81, 41.19, 41.1, 40.84, 41.13, 41.4, 40.7, 41.03, 41.01, 41.22, 41.06, 41.06, 40.95, 40.78, 40.49, 40.85, 40.44, 41.09, 40.63, 40.76]
loss_max_max = [10.0, 10.0, 10.0, 12.57, 18.46, 18.94, 18.85, 19.59, 19.33, 19.9, 19.23, 20.18, 21.11, 23.18, 23.82, 25.11, 28.04, 29.75, 29.65, 31.59, 30.19, 32.91, 33.46, 33.94, 34.14, 35.04, 36.42, 35.67, 36.53, 37.04, 37.42, 37.92, 39.13, 37.84, 39.62, 38.71, 39.67, 40.24, 40.28, 40.43, 40.87, 41.08, 41.77, 41.13, 40.77, 41.77, 42.34, 41.3, 42.47, 41.57, 41.7, 41.44, 42.07, 42.1, 42.52, 42.37, 42.01, 41.74, 42.21, 42.32, 42.36, 42.89, 42.34, 42.62, 41.73, 42.12, 42.13, 43.13, 43.12, 43.06, 42.69, 42.49, 42.42, 42.33, 42.73, 42.6, 42.81, 42.73, 42.8, 42.41, 42.49, 42.7, 42.8, 42.37, 42.53, 42.83, 43.12, 42.72, 42.49, 42.73, 43.12, 42.73, 42.7, 42.84, 42.43, 42.25, 42.72, 42.94, 42.54, 42.77]
loss_max_mean = [10.0, 10.0, 10.0, 10.64, 12.12, 12.42, 14.44, 15.62, 16.79, 17.52, 18.13, 19.0, 19.99, 21.36, 22.09, 23.68, 25.64, 26.97, 28.29, 29.84, 29.85, 31.68, 31.56, 32.18, 32.46, 33.54, 33.66, 33.8, 34.88, 35.22, 35.84, 35.81, 36.79, 37.18, 38.24, 37.67, 38.37, 39.29, 39.35, 39.64, 39.6, 40.3, 40.5, 40.06, 40.24, 40.74, 41.42, 40.74, 40.84, 40.96, 40.78, 41.07, 41.5, 41.78, 41.66, 41.89, 41.79, 41.54, 41.5, 41.98, 41.89, 41.88, 41.37, 41.65, 41.49, 41.64, 41.39, 41.75, 41.85, 41.91, 41.7, 41.76, 41.66, 41.68, 41.73, 41.66, 41.8, 41.88, 41.76, 41.4, 41.44, 41.71, 41.55, 41.48, 41.59, 41.9, 41.6, 41.6, 41.57, 41.7, 41.77, 41.63, 41.64, 41.58, 41.2, 41.53, 41.43, 41.72, 41.33, 41.47]
loss_min_min = [10.0, 10.0, 10.0, 10.0, 10.0, 10.34, 11.84, 15.59, 18.54, 18.55, 18.82, 18.85, 19.59, 20.77, 20.52, 23.87, 25.87, 26.32, 28.69, 30.71, 29.61, 30.72, 31.31, 32.72, 32.93, 33.68, 34.12, 34.56, 35.15, 36.32, 36.86, 36.91, 36.49, 38.43, 37.79, 39.03, 39.04, 38.95, 40.01, 38.11, 38.79, 39.77, 37.47, 39.85, 39.8, 40.25, 40.35, 40.38, 40.83, 41.02, 40.9, 40.84, 40.66, 40.95, 40.97, 40.56, 40.95, 41.24, 40.93, 40.57, 41.32, 40.57, 40.93, 41.26, 40.93, 40.79, 40.93, 40.9, 40.69, 40.94, 41.13, 40.58, 40.84, 40.68, 40.87, 40.93, 40.84, 40.94, 40.49, 40.44, 40.53, 40.84, 40.85, 40.34, 40.57, 40.66, 40.62, 40.89, 40.86, 40.63, 40.44, 40.98, 40.42, 40.33, 40.69, 39.88, 40.45, 40.29, 40.58, 40.29]
loss_min_max = [10.0, 10.0, 10.0, 10.0, 14.98, 18.6, 19.5, 20.12, 20.53, 21.52, 22.11, 24.42, 24.93, 25.91, 27.96, 28.9, 29.3, 30.85, 31.8, 31.38, 33.16, 32.83, 33.9, 35.36, 35.19, 35.64, 36.69, 37.07, 37.52, 38.49, 38.13, 39.87, 39.38, 39.54, 39.5, 40.14, 40.23, 40.52, 41.09, 41.79, 41.77, 42.36, 41.49, 42.09, 42.04, 42.31, 42.68, 42.22, 42.29, 42.56, 43.25, 42.19, 42.34, 43.72, 42.99, 43.3, 42.66, 43.05, 42.87, 42.96, 43.17, 42.75, 42.49, 43.12, 43.4, 42.95, 43.37, 43.14, 42.84, 42.96, 42.74, 42.76, 43.05, 43.03, 42.8, 42.89, 42.85, 42.67, 42.8, 42.71, 42.77, 43.04, 42.82, 43.22, 42.72, 42.82, 42.7, 43.1, 42.56, 43.1, 42.82, 42.77, 42.87, 42.95, 42.48, 42.91, 42.2, 42.87, 42.83, 42.51]
loss_min_mean = [10.0, 10.0, 10.0, 10.0, 11.85, 16.0, 17.19, 18.92, 19.78, 20.38, 20.75, 21.49, 22.84, 24.01, 25.83, 27.33, 28.4, 29.43, 30.48, 31.06, 31.67, 31.79, 33.01, 33.86, 34.1, 34.78, 35.26, 35.71, 36.26, 37.04, 37.56, 37.9, 37.86, 38.89, 38.8, 39.46, 39.39, 39.8, 40.36, 39.87, 40.47, 40.92, 40.16, 41.13, 40.89, 41.18, 41.5, 41.53, 41.4, 41.58, 41.89, 41.45, 41.43, 41.88, 41.8, 41.76, 41.82, 41.98, 41.91, 41.6, 42.04, 41.57, 41.49, 42.09, 41.8, 41.68, 41.82, 41.64, 41.62, 41.78, 41.78, 41.53, 41.74, 41.75, 41.83, 41.68, 41.76, 41.73, 41.62, 41.61, 41.52, 41.55, 41.68, 41.59, 41.58, 41.34, 41.71, 41.84, 41.53, 41.78, 41.69, 41.84, 41.57, 41.66, 41.7, 41.52, 41.53, 41.59, 41.51, 41.65]
random_min = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 12.97, 18.97, 17.39, 18.33, 19.14, 23.13, 26.44, 26.21, 30.92, 28.62, 31.8, 34.46, 34.97, 38.52, 36.68, 38.95, 37.37, 38.47, 40.09, 40.59, 40.55, 41.42, 43.0, 45.0, 44.24, 42.68, 45.75, 45.11, 45.01, 44.71, 45.28, 44.05, 46.48, 46.75, 46.61, 46.85, 46.26, 47.27, 46.3, 47.25, 46.46, 45.8, 46.25, 47.12, 46.57, 46.75, 47.09, 47.8, 46.63, 45.56, 46.14, 46.52, 46.73, 45.36, 46.27, 47.12, 45.01, 46.23, 47.33, 46.8, 46.35, 45.52, 46.39, 46.63, 46.45, 47.36, 45.67, 46.82, 46.76, 47.46, 45.83, 46.19, 46.95, 46.39, 45.01, 46.61, 46.61, 46.67, 45.29, 46.54, 44.98, 45.96, 44.03, 45.7, 45.67, 46.33, 47.0, 46.22, 45.23, 46.64, 46.06, 46.77, 46.18, 45.82]
random_max = [10.0, 10.0, 15.86, 15.68, 15.83, 18.72, 21.31, 25.23, 28.25, 26.91, 26.2, 29.71, 33.61, 34.37, 36.59, 35.05, 37.62, 39.65, 38.14, 42.86, 38.13, 41.8, 40.93, 42.02, 43.1, 41.46, 43.78, 45.83, 46.35, 47.41, 47.11, 45.06, 47.24, 46.1, 46.22, 47.02, 46.53, 46.51, 47.54, 47.06, 47.81, 47.94, 46.7, 48.24, 47.41, 48.98, 48.58, 46.52, 46.92, 48.63, 47.38, 47.77, 47.71, 48.23, 47.6, 46.44, 47.63, 47.31, 47.42, 47.18, 47.44, 47.52, 46.52, 47.18, 48.39, 47.46, 47.48, 46.59, 46.97, 47.67, 47.7, 47.62, 46.24, 48.14, 47.61, 48.11, 46.93, 46.79, 48.03, 47.74, 46.17, 47.06, 47.11, 47.43, 46.1, 46.93, 46.43, 46.91, 45.14, 46.12, 46.77, 47.14, 47.55, 47.1, 46.03, 47.26, 46.66, 47.43, 46.99, 46.26]
random_mean = [10.0, 10.0, 11.46, 11.42, 12.46, 13.99, 17.98, 21.46, 21.92, 22.41, 22.17, 26.06, 30.58, 30.02, 33.94, 32.33, 35.15, 36.56, 36.08, 40.4, 37.44, 40.08, 38.95, 40.22, 41.67, 41.01, 42.0, 43.49, 44.56, 46.14, 45.84, 43.5, 46.46, 45.63, 45.78, 46.11, 46.0, 45.19, 46.95, 46.89, 47.36, 47.44, 46.5, 47.68, 46.92, 48.42, 47.62, 46.26, 46.6, 48.09, 47.02, 47.24, 47.53, 48.01, 46.93, 46.12, 46.91, 46.89, 47.06, 46.21, 46.7, 47.32, 45.6, 46.64, 47.9, 47.1, 47.08, 46.08, 46.66, 47.24, 47.08, 47.49, 46.04, 47.54, 47.22, 47.71, 46.3, 46.63, 47.38, 47.04, 45.62, 46.79, 46.8, 46.94, 45.83, 46.67, 45.96, 46.28, 44.75, 45.85, 46.09, 46.75, 47.36, 46.79, 45.58, 46.86, 46.36, 47.17, 46.52, 46.08]

data = {
    "accuracy_min": accuracy_min,
    "accuracy_max": accuracy_max,
    "accuracy_mean": accuracy_mean,
    "datasize_min": datasize_min,
    "datasize_max": datasize_max,
    "datasize_mean": datasize_mean,
    "entropy_max_min": entropy_max_min,
    "entropy_max_max": entropy_max_max,
    "entropy_max_mean": entropy_max_mean,
    "entropy_min_min": entropy_min_min,
    "entropy_min_max": entropy_min_max,
    "entropy_min_mean": entropy_min_mean,
    "gradiv_max_min": gradiv_max_min,
    "gradiv_max_max": gradiv_max_max,
    "gradiv_max_mean": gradiv_max_mean,
    "gradiv_min_min": gradiv_min_min,
    "gradiv_min_max": gradiv_min_max,
    "gradiv_min_mean": gradiv_min_mean,
    "loss_max_min": loss_max_min,
    "loss_max_max": loss_max_max,
    "loss_max_mean": loss_max_mean,
    "loss_min_min": loss_min_min,
    "loss_min_max": loss_min_max,
    "loss_min_mean": loss_min_mean,
    "random_min": random_min,
    "random_max": random_max,
    "random_mean": random_mean,
}

save_path = None
if len(sys.argv) == 3 and sys.argv[1] and sys.argv[1] == "save":
    save_path = sys.argv[2]

plot_round_acc("", data, legend_pos="out", save_path=save_path, plot_size="2")