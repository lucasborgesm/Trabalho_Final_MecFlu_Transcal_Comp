import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

# Funções de geração de malhas
def malha_uniforme_2D(tamx, nx, tamy, ny):
    x = np.linspace(0, tamx, nx)
    y = np.linspace(0, tamy, ny)
    x, y = np.meshgrid(x, y)
    return np.ravel(x), np.ravel(y)

def malha_nao_uniforme_2D(tamx, nx, tamy, ny):
    xv = tamx * np.sin(np.linspace(0, np.pi / 2, nx))
    yv = tamy * np.sin(np.linspace(0, np.pi / 2, ny))
    x, y = np.meshgrid(xv, yv)
    return np.ravel(x), np.ravel(y)

def malha_aleatoria_2D(tamx, tamy, npoints):
    x = np.random.rand(npoints) * tamx
    y = np.random.rand(npoints) * tamy

    borda_x = np.concatenate((np.linspace(0, tamx, int(np.sqrt(npoints))),
                              np.linspace(0, tamx, int(np.sqrt(npoints))),
                              np.zeros(int(np.sqrt(npoints))),
                              np.ones(int(np.sqrt(npoints))) * tamx))
    borda_y = np.concatenate((np.zeros(int(np.sqrt(npoints))),
                              np.ones(int(np.sqrt(npoints))) * tamy,
                              np.linspace(0, tamy, int(np.sqrt(npoints))),
                              np.linspace(0, tamy, int(np.sqrt(npoints)))))

    x = np.concatenate((x, borda_x))
    y = np.concatenate((y, borda_y))

    triang = mtri.Triangulation(x, y)
    return x, y, triang.triangles

def IEN_quad(nx, ny):
    nelem = (nx - 1) * (ny - 1)
    ien = np.zeros((nelem, 4), dtype=int)
    for j in range(ny - 1):
        for i in range(nx - 1):
            idx = j * (nx - 1) + i
            ien[idx] = [j * nx + i, j * nx + i + 1, (j + 1) * nx + i + 1, (j + 1) * nx + i]
    return ien

def IEN_tria(ien_quad):
    nelem = len(ien_quad) * 2
    ien_tria = np.zeros((nelem, 3), dtype=int)
    for i, quad in enumerate(ien_quad):
        ien_tria[2 * i] = [quad[0], quad[1], quad[2]]
        ien_tria[2 * i + 1] = [quad[0], quad[2], quad[3]]
    return ien_tria

# Funções de análise de malha
def calcula_area_quad(x, y, ien_quad):
    areas = []
    for quad in ien_quad:
        x1, y1 = x[quad[0]], y[quad[0]]
        x2, y2 = x[quad[1]], y[quad[1]]
        x3, y3 = x[quad[2]], y[quad[2]]
        x4, y4 = x[quad[3]], y[quad[3]]
        area1 = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        area2 = 0.5 * abs((x3 - x1) * (y4 - y1) - (x4 - x1) * (y3 - y1))
        areas.append(area1 + area2)
    return np.array(areas)

def calcula_area_tria(x, y, ien_tria):
    areas = []
    for tri in ien_tria:
        x1, y1 = x[tri[0]], y[tri[0]]
        x2, y2 = x[tri[1]], y[tri[1]]
        x3, y3 = x[tri[2]], y[tri[2]]
        area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        areas.append(area)
    return np.array(areas)

def calcula_razao_aspecto_quad(x, y, ien_quad):
    razoes = []
    for quad in ien_quad:
        x1, y1 = x[quad[0]], y[quad[0]]
        x2, y2 = x[quad[1]], y[quad[1]]
        x3, y3 = x[quad[2]], y[quad[2]]
        x4, y4 = x[quad[3]], y[quad[3]]
        dist_h = max(np.linalg.norm([x2 - x1, y2 - y1]), np.linalg.norm([x4 - x3, y4 - y3]))
        dist_v = max(np.linalg.norm([x4 - x1, y4 - y1]), np.linalg.norm([x3 - x2, y3 - y2]))
        razoes.append(max(dist_h, dist_v) / min(dist_h, dist_v))
    return np.array(razoes)

def calcula_razao_aspecto_tria(x, y, ien_tria):
    razoes = []
    for tri in ien_tria:
        x1, y1 = x[tri[0]], y[tri[0]]
        x2, y2 = x[tri[1]], y[tri[1]]
        x3, y3 = x[tri[2]], y[tri[2]]
        a = np.linalg.norm([x2 - x1, y2 - y1])
        b = np.linalg.norm([x3 - x2, y3 - y2])
        c = np.linalg.norm([x3 - x1, y3 - y1])
        razoes.append(max(a, b, c) / min(a, b, c))
    return np.array(razoes)

def contar_vizinhos(ien, npoints):
    vizinhos = [set() for _ in range(npoints)]
    for elem in ien:
        for i in elem:
            vizinhos[i].update(elem)
            vizinhos[i].remove(i)
    return np.array([len(v) for v in vizinhos])

# Configurações do problema
Lx, Ly = 1, 1
nx, ny = 20, 20
npoints_aleatoria = 400

# Geração de malhas e conectividades
malhas = {
    "Uniforme": malha_uniforme_2D(Lx, nx, Ly, ny),
    "Não Uniforme": malha_nao_uniforme_2D(Lx, nx, Ly, ny),
    "Aleatória": malha_aleatoria_2D(Lx, Ly, npoints_aleatoria),
}

analises = {}

# Visualização das malhas
fig, axes = plt.subplots(3, 2, figsize=(15, 18))
fig.suptitle("Visualização das Malhas", fontsize=16)

# Plotagem das malhas Quad (Uniforme e Não Uniforme)
x_uni, y_uni = malhas["Uniforme"]
x_nuni, y_nuni = malhas["Não Uniforme"]
ien_quad_uni = IEN_quad(nx, ny)
ien_quad_nuni = IEN_quad(nx, ny)

# Plotagem das malhas Tria (Uniforme e Não Uniforme)
ien_tria_uni = IEN_tria(ien_quad_uni)
ien_tria_nuni = IEN_tria(ien_quad_nuni)

# Plotagem da malha Aleatória
x_ale, y_ale, ien_tria_ale = malhas["Aleatória"]

plt.figure(figsize=(8, 6))
for quad in ien_quad_uni:
    plt.plot(x_uni[quad[[0, 1, 2, 3, 0]]], y_uni[quad[[0, 1, 2, 3, 0]]], color="gray")
plt.title("Malha Quad - Uniforme")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.show()

# Malha Quad - Não Uniforme
plt.figure(figsize=(8, 6))
for quad in ien_quad_nuni:
    plt.plot(x_nuni[quad[[0, 1, 2, 3, 0]]], y_nuni[quad[[0, 1, 2, 3, 0]]], color="gray")
plt.title("Malha Quad - Não Uniforme")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.show()

# Malha Tria - Uniforme
plt.figure(figsize=(8, 6))
plt.triplot(x_uni, y_uni, ien_tria_uni, color="gray")
plt.title("Malha Tria - Uniforme")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.show()

# Malha Tria - Não Uniforme
plt.figure(figsize=(8, 6))
plt.triplot(x_nuni, y_nuni, ien_tria_nuni, color="gray")
plt.title("Malha Tria - Não Uniforme")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.show()

# Malha Tria - Aleatória
plt.figure(figsize=(8, 6))
plt.triplot(x_ale, y_ale, ien_tria_ale, color="gray")
plt.title("Malha Tria - Aleatória")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.show()

# Histogramas de áreas para Não Uniforme e Aleatória
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Distribuição de Áreas - Malhas Não Uniformes e Aleatória", fontsize=16)

areas_quad_nuni = calcula_area_quad(x_nuni, y_nuni, ien_quad_nuni)
areas_tria_nuni = calcula_area_tria(x_nuni, y_nuni, ien_tria_nuni)
areas_tria_ale = calcula_area_tria(x_ale, y_ale, ien_tria_ale)

axes[0].hist(areas_quad_nuni, bins=15, alpha=0.75, color="blue", label="Quad Não Uniforme")
axes[0].set_title("Quad - Não Uniforme")
axes[0].set_xlabel("Área")
axes[0].set_ylabel("Frequência")
axes[0].legend()

axes[1].hist(areas_tria_nuni, bins=15, alpha=0.75, color="green", label="Tria Não Uniforme")
axes[1].set_title("Tria - Não Uniforme")
axes[1].set_xlabel("Área")
axes[1].set_ylabel("Frequência")
axes[1].legend()

axes[2].hist(areas_tria_ale, bins=15, alpha=0.75, color="red", label="Tria Aleatória")
axes[2].set_title("Tria - Aleatória")
axes[2].set_xlabel("Área")
axes[2].set_ylabel("Frequência")
axes[2].legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Impressão de outros parâmetros
print("\nParâmetros das Malhas:")

print("\nUniforme:")
print(f"  Média da Área (Quad): {np.mean(areas_quad_nuni):.4f}")
print(f"  Média da Área (Tria): {np.mean(areas_tria_nuni):.4f}")
print(f"  Média da Razão de Aspecto (Quad): {np.mean(calcula_razao_aspecto_quad(x_uni, y_uni, ien_quad_uni)):.4f}")
print(f"  Média da Razão de Aspecto (Tria): {np.mean(calcula_razao_aspecto_tria(x_uni, y_uni, ien_tria_uni)):.4f}")
print(f"  Média de Vizinhos (Quad): {np.mean(contar_vizinhos(ien_quad_uni, len(x_uni))):.2f}")
print(f"  Média de Vizinhos (Tria): {np.mean(contar_vizinhos(ien_tria_uni, len(x_uni))):.2f}")

print("\nNão Uniforme:")
print(f"  Média da Área (Quad): {np.mean(areas_quad_nuni):.4f}")
print(f"  Média da Área (Tria): {np.mean(areas_tria_nuni):.4f}")
print(f"  Média da Razão de Aspecto (Quad): {np.mean(calcula_razao_aspecto_quad(x_nuni, y_nuni, ien_quad_nuni)):.4f}")
print(f"  Média da Razão de Aspecto (Tria): {np.mean(calcula_razao_aspecto_tria(x_nuni, y_nuni, ien_tria_nuni)):.4f}")
print(f"  Média de Vizinhos (Quad): {np.mean(contar_vizinhos(ien_quad_nuni, len(x_nuni))):.2f}")
print(f"  Média de Vizinhos (Tria): {np.mean(contar_vizinhos(ien_tria_nuni, len(x_nuni))):.2f}")

print("\nAleatória:")
print(f"  Média da Área (Tria): {np.mean(areas_tria_ale):.4f}")
print(f"  Média da Razão de Aspecto (Tria): {np.mean(calcula_razao_aspecto_tria(x_ale, y_ale, ien_tria_ale)):.4f}")
print(f"  Média de Vizinhos (Tria): {np.mean(contar_vizinhos(ien_tria_ale, len(x_ale))):.2f}")
