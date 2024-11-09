import subprocess
import matplotlib.pyplot as plt

# Configuration
executable = 'openmp_part_1/tp_pi_critical'  
C_values = [4, 6] 
N_values = [10**6, 10**7, 10**8, 10**9]  
execution_times = []

C = C_values[0]
for N in N_values:
    # Préparer la commande
    command = [executable, '-N', str(N), '-C', str(C)]
    
    # Exécuter la commande et récupérer le temps d'exécution
    result = subprocess.run(command, capture_output=True, text=True)
    
    # Extraire le temps d'exécution de la sortie
    output = result.stdout
    error = result.stderr
    print(output)  
    print(error)

    # Trouver le temps d'exécution
    for line in output.splitlines():
        if 'in' in line:
            time_str = line.split('in')[-1].strip().split()[0]
            execution_times.append(float(time_str))
            break

# Tracer les résultats
plt.plot(N_values, execution_times, label=f'C={N_values}')

plt.xscale('log')
plt.xlabel('Nombre de steps (N)')
plt.ylabel('Temps d\'exécution (s)')
plt.title('Temps d\'exécution en fonction de N')
plt.legend()
plt.grid()
plt.show()
