import numpy as np
import matplotlib.pyplot as plt

def fourier_coefficients(image_values, start=0, end=2*np.pi, a_terms=5, b_terms=4):
    N = len(image_values)
    t = np.linspace(start, end, N)  # Intervalo de 'start' a 'end'
    
    a_0 = np.sum(image_values) / N  # Calcula a_0
    a_n = [2/N * np.sum(image_values * np.cos(n * t)) for n in range(1, a_terms)]  # Calcula a_n para n > 0
    a = [a_0] + a_n  # Combina a_0 y a_n
    
    b = [0] + [2/N * np.sum(image_values * np.sin(n * t)) for n in range(1, b_terms+1)]
    
    return a, b
def plot_coefficients(a, b):
    n_a = len(a)
    n_b = len(b)
    width = 0.35  # ancho de las barras
    x_a = np.arange(n_a)
    x_b = np.arange(n_b) + width  # Desplaza las barras 'b' a la derecha
    
    plt.bar(x_a, a, width, color='blue', alpha=0.7, label='a_n')
    plt.bar(x_b, b, width, color='red', alpha=0.7, label='b_n')
    plt.xlabel('n')
    plt.ylabel('Valor del coeficiente')
    plt.legend()
    plt.savefig("bar.png")


def apply_function_to_points(points, function):
    # Convertimos los puntos a un array de numpy
    np_points = np.array(points)
    # Aplicamos la función a los puntos
    result_vector = function(np_points)
    return result_vector
def f(t):
    #return np.exp(32*t)
    return 8 + 3*np.cos(t) + 2*np.cos(2*t) + + np.cos(3*t) + 2*np.sin(t) + 4*np.sin(2*t) + 3*np.sin(3*t)
# Define el inicio y el final del rango
inicio = 0
final = 2 * np.pi

# Genera la lista de valores
t = np.linspace(inicio, final, num=1000)
print("Función f con inicio=" + inicio.__str__() + " y final=" + final.__str__())
y = apply_function_to_points(t, f)

# Calcula el espacio entre los puntos x (delta_x)
delta_x = t[1] - t[0]

# Calcula el área bajo la curva utilizando el método de los trapecios
area_trapz = np.trapz(y, dx=delta_x)
print("Área calculada con el método de los trapecios: ", area_trapz)

# Calcula el área bajo la curva utilizando el método de los rectángulos
area_rect = np.sum(y) * delta_x
print("Área calculada con el método de los rectángulos: ", area_rect)

# Calcula los coeficientes de Fourier
a, b = fourier_coefficients(y, start=inicio,end=final)

print("Coeficientes a_n: ", a)
print("Coeficientes b_n: ", b)

# Representa los coeficientes en un diagrama de barras
plot_coefficients(a, b)
