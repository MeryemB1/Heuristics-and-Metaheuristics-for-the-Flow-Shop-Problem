import numpy as np
import pandas as pd


#evaluation fonction 
def evaluate_sequence(sequence, processing_times):
    _, num_machines = processing_times.shape
    num_jobs = len(sequence)

    # Check if the sequence is empty
    if num_jobs == 0:
        # Return a default value (you may choose 0 or another suitable value)
        return 0

    completion_times = np.zeros((num_jobs, num_machines))

    # Calculate the completion times for the first machine
    completion_times[0][0] = processing_times[sequence[0]][0]
    for i in range(1, num_jobs):
        completion_times[i][0] = completion_times[i-1][0] + processing_times[sequence[i]][0]

    # Calculate the completion times for the remaining machines
    for j in range(1, num_machines):
        completion_times[0][j] = completion_times[0][j-1] + processing_times[sequence[0]][j]
        for i in range(1, num_jobs):
            completion_times[i][j] = max(completion_times[i-1][j], completion_times[i][j-1]) + processing_times[sequence[i]][j]

    # Return the total completion time, which is the completion time of the last job in the last machine
    return completion_times[num_jobs-1][num_machines-1]

# gupta  : 
import numpy as np

def gupta_heuristic(processing_times):
    # Get the number of jobs and machines
    jobs, machines = processing_times.shape

    # Calculate the total processing time for each job
    total_times = np.sum(processing_times, axis=1)

    # Calculate the differences in processing times between consecutive machines for each job
    differences = np.diff(processing_times, axis=1)

    # Calculate the ratio of setup time to total processing time for each job
    ratios = differences[:, :-1].sum(axis=1) / total_times

    # Sort jobs based on the ratios and total processing times
    order = sorted(range(jobs), key=lambda k: (ratios[k], total_times[k]))

    # Evaluate the makespan of the sequence
    cmax = evaluate_sequence(order, processing_times)

    return order, cmax


#CDC 

def johnson_method(processing_times):
    # Nombre de tâches et de machines
    tâches, machines = processing_times.shape
    # Copie des temps de traitement
    temps_de_traitement_copie = processing_times.copy()
    # Valeur maximale pour la comparaison
    maximum = processing_times.max() + 1
    # Listes pour stocker les indices des tâches pour chaque machine
    machine_1 = []
    machine_2 = []

    # Vérifie que le nombre de machines est égal à 2, sinon lève une exception
    if machines != 2:
        raise Exception("La méthode de Johnson fonctionne uniquement avec deux machines")

    # Itère sur le nombre de tâches
    for i in range(tâches):
        # Trouve le temps de traitement minimum restant
        minimum = temps_de_traitement_copie.min()
        # Trouve la position du temps de traitement minimum
        position = np.where(temps_de_traitement_copie == minimum)

        # Si le minimum se trouve sur la première machine, ajoute l'indice de la tâche à machine_1
        if position[1][0] == 0:
            machine_1.append(position[0][0])
        # Sinon, ajoute l'indice de la tâche à machine_2
        else:
            machine_2.insert(0, position[0][0])

        # Marque le temps de traitement minimum comme traité en lui attribuant une valeur maximale
        temps_de_traitement_copie[position[0][0]] = maximum

    # Concatène les deux listes des indices des tâches et retourne la séquence optimisée
    return machine_1 + machine_2


def CDS_heuristic(processing_times):
    # Nombre de tâches et de machines
    tâches, machines = processing_times.shape
    # Nombre de paires de machines
    nombre_paires = machines - 1
    # Matrice pour stocker les temps de traitement pour la méthode de Johnson
    temps_de_traitement_johnson = np.zeros((tâches, 2))
    # Meilleur coût initialisé à une valeur infinie
    meilleur_coût = np.inf
    # Meilleure séquence initialisée à une liste vide
    meilleure_séquence = []

    # Itère sur chaque paire de machines
    for k in range(nombre_paires):
        # Calcule les temps de traitement cumulatifs pour la méthode de Johnson
        temps_de_traitement_johnson[:, 0] += processing_times[:, k]
        temps_de_traitement_johnson[:, 1] += processing_times[:, -k - 1]
        # Applique la méthode de Johnson pour obtenir une séquence optimisée
        séquence = johnson_method(temps_de_traitement_johnson)
        # Évalue le coût de la séquence obtenue
        coût = evaluate_sequence(séquence, processing_times)
        # Met à jour la meilleure séquence et le meilleur coût si le coût actuel est meilleur
        if coût < meilleur_coût:
            meilleur_coût = coût
            meilleure_séquence = séquence

    # Retourne la meilleure séquence et le meilleur coût obtenus par l'heuristique CDS
    return meilleure_séquence, meilleur_coût


#Palmer 
import numpy as np

def palmer_heuristic(matrice_p):
    # Obtenir le nombre de tâches et de machines à partir de la matrice d'entrée
    tâches, machines = matrice_p.shape

    # Initialiser une liste pour stocker les indices de priorité pour chaque tâche
    indices = []

    # Itérer sur chaque tâche
    for i in range(tâches):
        # Calculer le temps de traitement total pour la tâche actuelle
        temps_traitement_total = np.sum(matrice_p[i])

        # Initialiser l'indice de priorité pour la tâche actuelle
        fi = 0

        # Itérer sur chaque machine pour la tâche actuelle
        for j in range(machines):
            # Calculer la contribution de la machine actuelle à l'indice de priorité
            fi += (machines - 2*j + 1) * matrice_p[i][j] / temps_traitement_total

        # Ajouter l'indice de priorité de la tâche actuelle à la liste
        indices.append(fi)

    # Trier la liste des indices pour déterminer l'ordre des tâches en fonction de la priorité
    ordre = sorted(range(tâches), key=lambda k: indices[k])

    # Retourner l'ordre des tâches
    return ordre


#johnon 

import numpy as np

def johnson_method(processing_times):
    # Get the number of jobs and machines from the input matrix
    jobs, machines = processing_times.shape

    # Create a copy of the processing times matrix
    copy_processing_times = processing_times.copy()

    # Define a value greater than any processing time for comparison
    maximum = processing_times.max() + 1

    # Initialize lists to store the jobs assigned to machine 1 and machine 2
    m1 = []
    m2 = []

    # Ensure the number of machines is 2, as Johnson's method only works with two machines
    if machines != 2:
        raise Exception("Johnson's method only works with two machines")

    # Iterate over each job
    for i in range(jobs):
        # Find the minimum processing time remaining in the copy matrix
        minimum = copy_processing_times.min()

        # Find the position of the minimum processing time
        position = np.where(copy_processing_times == minimum)

        # Determine which machine to assign the job to based on its position in the original matrix
        if position[1][0] == 0:
            m1.append(position[0][0])  # Add the job to machine 1
        else:
            m2.insert(0, position[0][0])  # Add the job to machine 2 at the beginning of the list

        # Mark the processing time as completed by assigning it a value greater than any processing time
        copy_processing_times[position[0][0]] = maximum

    # Return the sequence of jobs for machine 1 followed by the sequence for machine 2
    return m1 + m2


#Ham 
import numpy as np

def ham_heuristic(processing_time):
    # Get the number of jobs and machines
    jobs, machines = processing_time.shape

    # Create a sequence of jobs from 0 to the total number of jobs
    sequence = list(range(jobs))

    # Calculate the total processing time for the first half of machines (P1)
    P1 = processing_time[:,:machines//2].sum(axis=1)

    # Calculate the total processing time for the second half of machines (P2)
    P2 = processing_time[:,machines//2:].sum(axis=1)

    # Calculate the difference between P2 and P1
    P2_P1 = P2 - P1

    # Sort the jobs in descending order of the difference (P2 - P1)
    solution_1 = [job for _ , job in sorted(zip(P2_P1, sequence), reverse=True)]

    # Separate jobs into two groups: positive and negative differences
    positives = np.argwhere(P2_P1 >= 0).flatten()
    negatives = np.argwhere(P2_P1 < 0).flatten()

    # Sort jobs in the positive group based on the total processing time for the first half of machines (P1)
    positive_indices = [job for _ , job in sorted(zip(P1[positives], positives))]

    # Sort jobs in the negative group based on the total processing time for the second half of machines (P2) in reverse order
    negative_indices = [job for _ , job in sorted(zip(P2[negatives], negatives), reverse=True)]

    # Combine the sorted lists of jobs
    positive_indices.extend(negative_indices)

    # Calculate the makespan for both solutions
    Cmax1 = evaluate_sequence(solution_1, processing_time)
    Cmax2 = evaluate_sequence(positive_indices, processing_time)

    # Return the solution with the minimum makespan
    if Cmax1 < Cmax2:
        return solution_1, Cmax1
    else:
        return positive_indices, Cmax2


#PRSKE 

def skewness(processing_times):
    jobs, machines = processing_times.shape
    skewnesses = []
    # Calculate the skewness for each job
    for i in range(jobs):
        avg = np.mean(processing_times[i,:])
        numerator = 0
        denominator = 0
        for j in range(machines):
            m = (processing_times[i,j] - avg)
            numerator += m**3
            denominator += m**2
        # Actually calculating the skewness
        numerator = numerator*(1/machines)
        denominator = (np.sqrt(denominator*(1/machines)))**3
        skewnesses.append(numerator/denominator)
    return np.array(skewnesses)


import numpy as np

def PRSKE_heuristic(processing_times):
    # Calculate the average processing time for each job
    avg = np.mean(processing_times, axis=1)

    # Calculate the standard deviation of processing time for each job
    std = np.std(processing_times, axis=1, ddof=1)

    # Calculate the skewness of processing time for each job
    skw = skewness(processing_times)  # Assuming this function calculates skewness

    # Calculate the priority order based on the sum of skewness, standard deviation, and average
    order = skw + std + avg

    # Sort the jobs in descending order of priority
    sequence = [job for _ , job in sorted(zip(order, list(range(processing_times.shape[0]))),reverse=True)]

    # Evaluate the sequence to get the makespan
    makespan = evaluate_sequence(sequence, processing_times)

    # Return the sorted sequence and its corresponding makespan
    return sequence, makespan


#Artificial Heuristic 

import numpy as np

def artificial_heuristic(processing_times):

    jobs, machines = processing_times.shape
    r = 1
    best_cost = np.inf  # Initialize the best cost to infinity
    best_seq = []  # Initialize the best sequence as an empty list

    # Iterate until the number of considered machines equals the total number of machines
    while r != machines:
        # Create a matrix to calculate the weights (wi)
        wi = np.zeros((jobs, machines - r))
        for i in range(jobs):
            for j in range(0, machines - r):
                wi[i, j] = (machines - r) - (j)

        # Create a matrix to store the weighted processing times for each job
        am = np.zeros((jobs, 2))

        # Calculate the weighted processing times for the first and last machines
        am[:, 0] = np.sum(wi[:, :machines - r] * processing_times[:, :machines - r], axis=1)

        # Calculate the weighted processing times for the middle machines
        for i in range(jobs):
            for j in range(0, machines - r):
                am[i, 1] += wi[i, j] * processing_times[i, machines - j - 1]

        # Obtain a sequence using the Johnson's method with the calculated weighted processing times
        seq = johnson_method(am)

        # Evaluate the cost (makespan) of the obtained sequence
        cost = evaluate_sequence(seq, processing_times)

        # Update the best sequence and cost if the current cost is better
        if cost < best_cost:
            best_cost = cost
            best_seq = seq

        # Move to the next configuration by considering one additional machine
        r += 1

    # Return the best sequence and its corresponding cost
    return best_seq, best_cost


