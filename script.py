# -*- coding: utf-8 -*-
import cv2
import numpy as np
from skimage import exposure
from sklearn.metrics import mean_squared_error
import random
import matplotlib.pyplot as plt
import os
import shutil

RESULTS_DIR = 'results'
GRAPHS_DIR = os.path.join(RESULTS_DIR, 'graphs')

def setup_directories():
    if os.path.exists(RESULTS_DIR):
        shutil.rmtree(RESULTS_DIR)
    os.makedirs(GRAPHS_DIR, exist_ok=True)

setup_directories()

def rgb_to_lab(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

def apply_clahe(image, clipLimit=2.0, grid=(8, 8)):
    try:
        image = rgb_to_lab(image)
        lab_planes = list(cv2.split(image))
        if len(lab_planes) != 3:
            raise ValueError("LAB color space conversion failed. Expected 3 channels.")
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=grid)
        if lab_planes[0] is None:
            raise ValueError("L channel is empty. Check the image conversion.")
        if lab_planes[0].dtype != np.uint8:
            lab_planes[0] = cv2.convertScaleAbs(lab_planes[0])
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab_img = cv2.merge(lab_planes)
        return cv2.cvtColor(lab_img, cv2.COLOR_LAB2RGB)
    except cv2.error as e:
        print(f"OpenCV error: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise

def evaluate_fitness(original_image, processed_image):
    ogn_flat = original_image.flatten()
    pcs_flat = processed_image.flatten()
    mse = np.mean((ogn_flat - pcs_flat) ** 2)
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr, mse

def generate_population(chromosome_length, population_size):
    population = []
    for _ in range(population_size):
        chromosome = [random.randint(0, 1) for _ in range(chromosome_length)]
        population.append(chromosome)
    return population

def binary_to_decimal(binary):
    decimal = 0
    for digit in binary:
        decimal = decimal * 2 + int(digit)
    return decimal

def decode_chromosome(chromosome):
    n_bit = len(chromosome) // 2
    m_bits = len(chromosome) - n_bit
    m_part = chromosome[:m_bits]
    n_part = chromosome[n_bit:]
    m = binary_to_decimal(m_part)
    n = binary_to_decimal(n_part)
    return m, n

def sort_arrays(arr1, arr2):
    arr1_with_index = [(value, index) for index, value in enumerate(arr1)]
    arr1_with_index.sort(key=lambda x: x[0])
    sorted_arr1 = [tup[0] for tup in arr1_with_index]
    sorted_arr2 = [arr2[tup[1]] for tup in arr1_with_index]
    return sorted_arr1, sorted_arr2

def cfCalculate(fitness_val):
    fitness_cf = np.zeros_like(fitness_val)
    total = 0
    for i in range(len(fitness_val)):
        total += fitness_val[i]
        fitness_cf[i] = total
    return fitness_cf

def chromosomeSelection(fitness_val, population):
    fitness_cf = cfCalculate(fitness_val)
    dice = random.uniform(0, fitness_cf[-1])
    for i in range(len(fitness_cf)):
        if dice < fitness_cf[i]:
            return population[i]

def generation_fitness(population, image, gen):
    fitness = np.zeros(len(population))
    mse_array = []
    for i, selected_chromosome in enumerate(population):
        m, n = decode_chromosome(selected_chromosome)
        grid = (m, n)
        processed_img = apply_clahe(image, grid=grid)
        psnr, mse = evaluate_fitness(image, processed_img)
        fitness[i] = psnr
        mse_array.append(mse)

    psnr_graph(fitness, len(population), gen)
    mse_graph(mse_array, len(population), gen)

    return fitness, mse_array

def Crossover(parent1, parent2, crossover_type="one_point"):
    if crossover_type == "one_point":
        crossover_point = random.randint(0, len(parent1))
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
    elif crossover_type == "two_point":
        crossover_point1 = random.randint(0, len(parent1))
        crossover_point2 = random.randint(crossover_point1, len(parent1))
        child1 = parent1[:crossover_point1] + parent2[crossover_point1:crossover_point2] + parent1[crossover_point2:]
        child2 = parent2[:crossover_point1] + parent1[crossover_point1:crossover_point2] + parent2[crossover_point2:]
    else:
        raise ValueError("Invalid crossover type. Choose either 'one_point' or 'two_point'.")
    return child1, child2

def NewGeneration(population, child1, child2):
    population[0] = child1
    population[1] = child2

def Mutation(generation, population_size, times=1):
    for _ in range(times):
        random_chromosome = random.randint(0, population_size - 1)
        random_position = random.randint(0, len(generation[0]) - 1)
        chromosome_list = list(map(str, generation[random_chromosome]))  # Convert each element to string
        if chromosome_list[random_position] == '1':
            chromosome_list[random_position] = '0'
        else:
            chromosome_list[random_position] = '1'
        generation[random_chromosome] = ''.join(chromosome_list)

def Algorithm(image, population_size, generations, psnr_array, mse_final):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    max_m = image.shape[0] // 4
    max_n = image.shape[1] // 4
    chromosome_length = len(bin(max(max_m, max_n))[2:]) * 2
    population = generate_population(chromosome_length, population_size)

    for i in range(generations):
        fitness_val, mse_array = generation_fitness(population, image, i+1)
        fitness_val, population = sort_arrays(fitness_val, population)
        mse_array = sorted(list(mse_array))
        psnr_array.append(fitness_val[population_size - 1])
        mse_final.append(mse_array[0])
        parent1 = chromosomeSelection(fitness_val, population)
        parent2 = chromosomeSelection(fitness_val, population)
        child1, child2 = Crossover(parent1, parent2)
        NewGeneration(population, child1, child2)

    processed = apply_latest_clahe(image, population=population, population_size=population_size, max_m=max_m, max_n=max_n)
    return processed

def apply_latest_clahe(image, population, population_size, max_m, max_n):
    latest_chromosome = population[population_size - 1]
    m, n = decode_chromosome(latest_chromosome)
    m = max(min(m, max_m), 2)
    n = max(min(n, max_n), 2)
    grid = (m, n)
    processed_img = apply_clahe(image, grid=grid)
    return processed_img

def psnr_graph(psnr_array, generations, gen=1):
    if len(psnr_array) > generations:
        psnr_array = psnr_array[:generations]
    labels = list(range(1, generations+1))
    plt.plot(labels, psnr_array, marker='o')
    plt.xlabel('Generation')
    plt.ylabel('PSNR')
    plt.title('PSNR vs. Generation')
    plt.grid(True)
    plt.savefig(os.path.join(GRAPHS_DIR, f'psnr_graph_{gen}.png'))
    plt.close()

def mse_graph(mse_array, generations, gen=1):
    if len(mse_array) > generations:
        mse_array = mse_array[:generations]
    labels = list(range(1, generations+1))
    plt.plot(labels, mse_array, marker='o')
    plt.xlabel('Generation')
    plt.ylabel('MSE')
    plt.title('MSE vs. Generation')
    plt.grid(True)
    plt.savefig(os.path.join(GRAPHS_DIR, f'mse_graph_{gen}.png'))
    plt.close()

def process_image_from_path(image_path, population_size, generations):
    psnr_array = []
    mse_final = []

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Error loading image: {image_path}")

    result_img = Algorithm(image, population_size, generations, psnr_array, mse_final)

    return result_img, psnr_array[-1] if psnr_array else None, mse_final[-1] if mse_final else None
