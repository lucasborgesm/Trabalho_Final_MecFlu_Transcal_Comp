import numpy as np
import matplotlib.pyplot as plt
import meshio
import matplotlib.tri as mtri

# Função para calcular a área dos elementos triangulares
def calcula_area_tria(x, y, ien):
    areas = []
    for tri in ien:
        x1, y1 = x[tri[0]], y[tri[0]]
        x2, y2 = x[tri[1]], y[tri[1]]
        x3, y3 = x[tri[2]], y[tri[2]]
        area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        areas.append(area)
    return np.array(areas)

# Função para calcular a razão de aspecto
def calcula_razao_aspecto_tria(x, y, ien):
    razoes = []
    for tri in ien:
        x1, y1 = x[tri[0]], y[tri[0]]
        x2, y2 = x[tri[1]], y[tri[1]]
        x3, y3 = x[tri[2]], y[tri[2]]
        a = np.linalg.norm([x2 - x1, y2 - y1])
        b = np.linalg.norm([x3 - x2, y3 - y2])
        c = np.linalg.norm([x3 - x1, y3 - y1])
        razoes.append(max(a, b, c) / min(a, b, c))
    return np.array(razoes)

# Função para contar os vizinhos de cada nó
def contar_vizinhos(ien, npoints):
    vizinhos = [set() for _ in range(npoints)]
    for elem in ien:
        for i in elem:
            vizinhos[i].update(elem)
            vizinhos[i].remove(i)
    return np.array([len(v) for v in vizinhos])

# Ler a malha gerada pelo GMSH
msh_file = "msh_NACA0012_0.msh"  # Substitua pelo caminho correto do arquivo .msh
mesh = meshio.read(f"Malhas/{msh_file}")

# Extração das coordenadas dos nós e conectividade
x = mesh.points[:, 0]
y = mesh.points[:, 1]
ien = mesh.cells_dict["triangle"]

# Análise da malha
areas = calcula_area_tria(x, y, ien)
razoes_aspecto = calcula_razao_aspecto_tria(x, y, ien)
vizinhos = contar_vizinhos(ien, len(x))

# Plotagem da malha
plt.figure(figsize=(8, 8))
plt.triplot(x, y, ien, color="gray", alpha=0.7)
plt.xlabel("X")
plt.ylabel("Y")
plt.axis("equal")
plt.show()

# Histograma das áreas
plt.figure(figsize=(8, 6))
plt.hist(areas, bins=20, alpha=0.75, color="blue", label="Áreas")
plt.title("Distribuição de Áreas - Malha GMSH")
plt.xlabel("Área")
plt.ylabel("Frequência")
plt.legend()
plt.show()

# Impressão de parâmetros
print("\nParâmetros da Malha GMSH:")
print(f"  Média da Área: {np.mean(areas):.4f}")
print(f"  Média da Razão de Aspecto: {np.mean(razoes_aspecto):.4f}")
print(f"  Média de Vizinhos: {np.mean(vizinhos):.2f}")
