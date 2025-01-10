import numpy as np
import pandas as pd


def rotate_points(points, center, angle_deg):
    """
    Rotaciona um conjunto de pontos em torno de um ponto central.

    Parameters:
    - points: array de coordenadas dos pontos [[x1, y1], [x2, y2], ...].
    - center: ponto central de rotação [xc, yc].
    - angle_deg: ângulo de rotação em graus.

    Returns:
    - rotated_points: array de coordenadas dos pontos rotacionados.
    """
    angle_deg = -angle_deg
    angle_rad = np.radians(angle_deg)
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                [np.sin(angle_rad), np.cos(angle_rad)]])
    rotated_points = []
    for point in points:
        translated_point = np.array(point) - np.array(center)
        rotated_point = rotation_matrix @ translated_point + np.array(center)
        rotated_points.append(rotated_point)
    return np.array(rotated_points)


def calculate_geometric_center(points):
    """
    Calcula o centro geométrico de um conjunto de pontos.

    Parameters:
    - points: array de coordenadas dos pontos [[x1, y1], [x2, y2], ...].

    Returns:
    - center: coordenadas do centro geométrico [xc, yc].
    """
    points = np.array(points)
    xc = np.mean(points[:, 0])  # Média das coordenadas x
    yc = np.mean(points[:, 1])  # Média das coordenadas y
    return [xc, yc]


# DEFININDO CONTORNOS DA SIMULACAO

lf = 1   # Comprimento à frente
lb = 3   # Comprimento atrás
h = 2    # Altura do domínio

# DEFININDO PARAMETROS

eSizeEdges = 0.5
eSizeAirf = 0.25

# DEFININDO PONTOS DO PERFIL

file_name = "NACA0012"

airfoil = np.array(pd.read_csv(f"Pontos/{file_name}.csv", sep=','))

# Ângulo de rotação (em graus)
angle = 30

airfPoints = []

for line in airfoil:
    airfPoints.append([line[0], line[1]])

# CONSTRUINDO A GEOMETRIA

# Pontos
pointsStr = ''

# Pontos dos contornos
pointsStr += f"Point(1) = {{-{lf:.3f}, -{h/2:.3f}, 0, {eSizeEdges:.3f}}};\n"
pointsStr += f"Point(2) = {{{lb:.3f}, -{h/2:.3f}, 0, {eSizeEdges:.3f}}};\n"
pointsStr += f"Point(3) = {{{lb:.3f}, {h/2:.3f}, 0, {eSizeEdges:.3f}}};\n"
pointsStr += f"Point(4) = {{-{lf:.3f}, {h/2:.3f}, 0, {eSizeEdges:.3f}}};\n"

# Calcula o centro geométrico dos pontos do aerofólio
center = calculate_geometric_center(airfPoints)

# Rotacionar os pontos do aerofólio em torno do centro geométrico
rotated_airfPoints = rotate_points(airfPoints, center, angle)

# Substituir os pontos rotacionados no loop de construção da geometria
for i, point in enumerate(rotated_airfPoints, start=5):
    pointsStr += f"Point({i}) = {{{point[0]:.6f}, {point[1]:.6f}, 0, {eSizeAirf:.6f}}};\n"

# Salvar os pontos rotacionados em um novo arquivo CSV
rotated_df = pd.DataFrame(rotated_airfPoints, columns=["x", "y"])
rotated_df.to_csv(f"Pontos/{file_name}_{angle}.csv", index=False)
print(f"Os pontos rotacionados foram salvos no arquivo 'Pontos/{file_name}_{angle}.csv'")

# Linhas
linesStr = ''

# Contorno do domínio
linesStr += "Line(1) = {1, 2};\n"
linesStr += "Line(2) = {2, 3};\n"
linesStr += "Line(3) = {3, 4};\n"
linesStr += "Line(4) = {4, 1};\n"

# Linhas do aerofólio
for i in range(len(rotated_airfPoints)):
    start = i + 5
    end = 5 if i == len(rotated_airfPoints) - 1 else start + 1
    linesStr += f"Line({i+5}) = {{{start}, {end}}};\n"

# Curvas e Superfícies
curvesStr = "\nLine Loop(1) = {1, 2, 3, 4};\n"
curvesStr += f"Line Loop(2) = {{{', '.join(str(i+5) for i in range(len(rotated_airfPoints)))}}};\n"
curvesStr += "Plane Surface(1) = {1, 2};\n"

# Grupos Físicos
physicalStr = "\n"
physicalStr += "Physical Line(\"inlet\") = {1};\n"
physicalStr += "Physical Line(\"outlet\") = {3};\n"
physicalStr += "Physical Line(\"lower_wall\") = {2};\n"
physicalStr += "Physical Line(\"upper_wall\") = {4};\n"
physicalStr += "Physical Line(\"airfoil\") = {"
physicalStr += ", ".join(str(i+5) for i in range(len(rotated_airfPoints)))
physicalStr += "};\n"
physicalStr += "Physical Surface(\"fluid\") = {1};\n"

# Juntando Tudo
meshStr = pointsStr + linesStr + curvesStr + physicalStr

# Escrevendo o Arquivo GEO
with open(f'Geometrias/msh_{file_name}_{angle}.geo', 'w') as mesh:
    mesh.write(meshStr)

print(f"A geometria foi salva no arquivo 'Geometrias/msh_{file_name}_{angle}.geo'")
