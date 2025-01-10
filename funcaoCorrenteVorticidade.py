import meshio
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as mtri
import pandas as pd
from collections import defaultdict


def extract_boundary_nodes(mesh):
    """
    Extrai os nós associados a cada grupo físico no arquivo .msh (versão 2.2).

    Parameters:
    - mesh: objeto meshio lido do arquivo .msh.

    Returns:
    - boundary_nodes: dicionário {nome_do_grupo: [índices_dos_nós]}.
    """
    boundary_nodes = {}

    # Procurar elementos do tipo 'line'
    if "line" in mesh.cells_dict:
        line_data = mesh.cells_dict["line"]
    else:
        raise ValueError("Nenhum elemento do tipo 'line' encontrado na malha.")

    # Mapear grupos físicos para os elementos do tipo linha
    if "gmsh:physical" in mesh.cell_data_dict and "line" in mesh.cell_data_dict["gmsh:physical"]:
        physical_ids = mesh.cell_data_dict["gmsh:physical"]["line"]
    else:
        raise ValueError("Os IDs dos grupos físicos não estão definidos para elementos 'line'.")

    # Iterar sobre os IDs dos grupos físicos para associar os nós
    for group_id in np.unique(physical_ids):
        group_name = f"group_{group_id}"  # Nome genérico, pode ser ajustado
        group_lines = line_data[physical_ids == group_id]  # Linhas no grupo
        nodes = np.unique(group_lines.flatten())  # Extrair nós únicos
        boundary_nodes[group_name] = nodes

    return boundary_nodes


def plot_mesh_with_boundaries(X, Y, IEN, boundary_nodes):
    """
    Plota a malha com os nós destacados para cada grupo de contorno.

    Parameters:
    - X, Y: coordenadas dos nós.
    - IEN: matriz de conectividade elementar.
    - boundary_nodes: dicionário com os nós para cada grupo.
    """
    plt.figure(figsize=(10, 8))
    plt.triplot(X, Y, IEN, color="gray", alpha=0.7)

    colors = ["red", "blue", "green", "yellow", "black", "purple"]

    for idx, (group, nodes) in enumerate(boundary_nodes.items()):
        group_label = group
        if group == "group_4":
            group_label = "inlet"
        if group == "group_3":
            group_label = "outlet"
        if group == "group_1":
            group_label = "lower_wall"
        if group == "group_2":
            group_label = "upper_wall"
        if group == "group_5":
            group_label = "airfoil"
        plt.scatter(X[nodes], Y[nodes], label=group_label, color=colors[idx % len(colors)], s=10)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.axis("equal")
    plt.title("Malha com Contornos Destacados")
    plt.show()


def solve_stream_vorticity(mesh, inlet, outlet, lower_wall, upper_wall, surface, Re, max_iter=5000, tol=1e-5, max_tol=1e+5):
    """
    Resolve as equações de função corrente-vorticidade explicitamente.

    Parameters:
    - mesh: tupla com coordenadas (X, Y) e conectividade IEN.
    - inlet, outlet, lower_wall, upper_wall, surface: índices dos contornos.
    - Re: número de Reynolds.
    - max_iter: número máximo de iterações.
    - tol: tolerância para a convergência.

    Returns:
    - PSI: solução da função corrente.
    - OMEGA: solução da vorticidade.
    """
    (X, Y), IEN = mesh
    npoints = len(X)

    # Inicialização
    PSI = np.zeros(npoints)  # Função corrente
    OMEGA = np.zeros(npoints)  # Vorticidade

    # Matriz de massa e rigidez
    M = np.zeros((npoints, npoints))
    K = np.zeros((npoints, npoints))
    Gx = np.zeros((npoints, npoints))
    Gy = np.zeros((npoints, npoints))

    for e in range(len(IEN)):
        i, j, k = IEN[e]

        Ae = (1 / 2) * np.linalg.det(np.array([[1, X[i], Y[i]],
                                               [1, X[j], Y[j]],
                                               [1, X[k], Y[k]]]))

        if np.isclose(Ae, 0):
            print(f"Elemento degenerado detectado: {e}")

        bi, bj, bk = Y[j] - Y[k], Y[k] - Y[i], Y[i] - Y[j]
        ci, cj, ck = X[k] - X[j], X[i] - X[k], X[j] - X[i]

        me = (Ae / 12) * np.array([[2, 1, 1],
                                   [1, 2, 1],
                                   [1, 1, 2]])

        ke = (1 / (4 * Ae)) * np.array([[bi * bi + ci * ci, bi * bj + ci * cj, bi * bk + ci * ck],
                                        [bj * bi + cj * ci, bj * bj + cj * cj, bj * bk + cj * ck],
                                        [bk * bi + ck * ci, bk * bj + ck * cj, bk * bk + ck * ck]])

        gxe = (1 / 6) * np.array([[bi, bj, bk],
                                  [bi, bj, bk],
                                  [bi, bj, bk]])

        gye = (1 / 6) * np.array([[ci, cj, ck],
                                  [ci, cj, ck],
                                  [ci, cj, ck]])

        for ilocal in range(3):
            iglobal = IEN[e][ilocal]
            for jlocal in range(3):
                jglobal = IEN[e][jlocal]
                M[iglobal, jglobal] += me[ilocal, jlocal]
                K[iglobal, jglobal] += ke[ilocal, jlocal]
                Gx[iglobal, jglobal] += gxe[ilocal, jlocal]
                Gy[iglobal, jglobal] += gye[ilocal, jlocal]

    if np.linalg.cond(M) > 1 / np.finfo(M.dtype).eps:
        print("A matriz M é singular ou próxima de ser singular.")

    # Inicialização do vetor de erros
    error_psi_list = []
    error_omega_list = []

    yp = 0

    for k in surface:
        yp += Y[k]

    yp = yp/len(surface)

    print(yp)

    # Iteração para resolver PSI e OMEGA
    for iteration in range(max_iter):
        # Atualizar PSI (resolvendo a equação de Poisson)
        A_psi = K.copy()
        b_psi = M @ OMEGA
        for j in np.concatenate([inlet, outlet, lower_wall, upper_wall, surface]):
            A_psi[j, :] = 0
            A_psi[j, j] = 1
            if j in np.concatenate([inlet, lower_wall, upper_wall]):
                b_psi[j] = Y[j]
            elif j in outlet:
                A_psi[j, :] = K[j, :]
                b_psi[j] = 0
            elif j in surface:
                b_psi[j] = yp
        PSI_new = np.linalg.solve(A_psi, b_psi)

        # Atualizar OMEGA (resolvendo a equação de transporte)
        vx = np.linalg.solve(M, Gy @ PSI_new)
        vy = -np.linalg.solve(M, Gx @ PSI_new)

        A_omega = M + (1 / Re) * K
        b_omega = M @ OMEGA - (np.multiply(vx, Gx @ OMEGA) + np.multiply(vy, Gy @ OMEGA))
        for j in np.concatenate([inlet, outlet, lower_wall, upper_wall, surface]):
            A_omega[j, :] = 0
            A_omega[j, j] = 1
            if j in lower_wall or j in upper_wall or j in surface:
                b_omega[j] = -2 * (Gy @ PSI_new)[j]
        OMEGA_new = np.linalg.solve(A_omega, b_omega)

        # Verificar convergência
        error_psi = np.linalg.norm(PSI_new - PSI, ord=2)
        error_omega = np.linalg.norm(OMEGA_new - OMEGA, ord=2)

        error_psi_list.append(error_psi)
        error_omega_list.append(error_omega)

        print(f"Iteração {iteration}: Erro de PSI = {error_psi:.6e}, Erro de OMEGA = {error_omega:.6e}")

        # Atualizar os valores de PSI e OMEGA
        PSI, OMEGA = PSI_new, OMEGA_new

        if error_psi < tol and error_omega < tol:
            print(f"Convergiu na iteração {iteration}")
            break

        elif error_psi > max_tol or error_omega > max_tol:
            print(f"Foi identificada uma não convergênicia na iteração {iteration}")
            break

    return PSI, OMEGA, vx, vy


def calculate_pressure(vx, vy, U_inf, rho=1.0, p_inf=0.0):
    """
    Calcula o coeficiente de pressão (C_p) e a pressão (p) em cada ponto.

    Parameters:
    - vx: componente da velocidade em x.
    - vy: componente da velocidade em y.
    - U_inf: velocidade do fluxo livre.
    - rho: densidade do fluido (default: 1.0).
    - p_inf: pressão no fluxo livre (default: 0.0).

    Returns:
    - C_p: coeficiente de pressão em cada ponto.
    - p: pressão em cada ponto.
    """
    velocity_magnitude_squared = vx**2 + vy**2
    C_p = 1 - (velocity_magnitude_squared / U_inf**2)
    p = p_inf + 0.5 * rho * U_inf**2 * C_p
    return C_p, p


def calculate_lift_and_drag(surface, X, Y, p):
    """
    Calcula as forças de sustentação (L) e arrasto (D) no aerofólio.

    Parameters:
    - surface: índices dos pontos que formam a superfície do aerofólio.
    - X: coordenadas X de todos os nós.
    - Y: coordenadas Y de todos os nós.
    - p: pressão em cada nó.

    Returns:
    - L: força de sustentação.
    - D: força de arrasto.
    """
    L, D = 0, 0

    for i in range(len(surface) - 1):
        # Coordenadas dos pontos do segmento
        x1, y1 = X[surface[i]], Y[surface[i]]
        x2, y2 = X[surface[i + 1]], Y[surface[i + 1]]

        # Comprimento do segmento
        ds = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        # Vetor normal ao segmento (sentido horário)
        nx = y2 - y1
        ny = -(x2 - x1)

        # Média da pressão nos dois pontos do segmento
        p_avg = 0.5 * (p[surface[i]] + p[surface[i + 1]])

        # Contribuições para as forças
        D += -p_avg * nx * ds
        L += -p_avg * ny * ds

    return L, D


# Ler o arquivo .msh
msh_file = "msh_NACA0012_30.msh"
mesh = meshio.read(f"Malhas/{msh_file}")

# Extraindo coordenadas dos nós e conectividade dos elementos
X = mesh.points[:, 0]
Y = mesh.points[:, 1]
IEN = mesh.cells_dict["triangle"]

# Extraindo nós associados aos grupos físicos
boundary_nodes = extract_boundary_nodes(mesh)

# Verificando se os grupos físicos estão bem definidos
print("Grupos físicos encontrados:")
for group, nodes in boundary_nodes.items():
    print(f"Grupo {group}: {len(nodes)} nós")

# Plotando a malha com os nós destacados
plot_mesh_with_boundaries(X, Y, IEN, boundary_nodes)

# Usando os grupos físicos no restante do código
inlet = boundary_nodes.get("group_4", [])
outlet = boundary_nodes.get("group_3", [])
lower_wall = boundary_nodes.get("group_1", [])
upper_wall = boundary_nodes.get("group_2", [])
surface = boundary_nodes.get("group_5", [])

inlet = np.array(inlet)
outlet = np.array(outlet)
lower_wall = np.array(lower_wall)
upper_wall = np.array(upper_wall)
surface = np.array(surface)

# print(f"Nós de inlet: {inlet}")
# print(f"Nós de outlet: {outlet}")
# print(f"Nós de lower_wall: {lower_wall}")
# print(f"Nós de upper_wall: {upper_wall}")
# print(f"Nós de surface: {surface}")

mesh_prop = (X, Y), IEN

triang = mtri.Triangulation(X, Y, IEN)

Re = 5  # Número de Reynolds
PSI, OMEGA, vx, vy = solve_stream_vorticity(mesh_prop, inlet, outlet, lower_wall, upper_wall, surface, Re)

# Calculando o coeficiente de pressão e a pressão
U_inf = 1.0  # Ajuste para o valor correto de U_inf
C_p, p = calculate_pressure(vx, vy, U_inf)

# Cálculo das forças de sustentação e arrasto
L, D = calculate_lift_and_drag(surface, X, Y, p)
print(f"Força de Sustentação (L): {L:.3f}")
print(f"Força de Arrasto (D): {D:.3f}")


# Gráfico da função corrente (PSI)
plt.figure(figsize=(8, 6))
plt.tricontourf(triang, PSI, levels=50, cmap="jet")  # Faixa limitada
plt.colorbar(label="Função Corrente (ψ)")
plt.tricontour(triang, PSI, levels=20, colors="black", linewidths=0.5)  # Linhas em preto
plt.title("Linhas de Corrente ao Redor do Aerofólio")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis("equal")
plt.show()

# Gráfico da vorticidade (OMEGA)
plt.figure(figsize=(8, 6))
plt.tricontourf(triang, OMEGA, cmap="jet", levels=50)
plt.colorbar(label="Vorticidade (ω)")
plt.title("Distribuição de Vorticidade ao Redor do Aerofólio")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis("equal")
plt.show()

# Gráfico de vx (componente da velocidade em x)
plt.figure(figsize=(8, 6))
plt.tricontourf(triang, vx, cmap="jet", levels=50)
plt.colorbar(label="Velocidade vx")
plt.title("Componente da Velocidade vx ao Redor do Aerofólio")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis("equal")
plt.show()

# Gráfico de vy (componente da velocidade em y)
plt.figure(figsize=(8, 6))
plt.tricontourf(triang, vy, cmap="jet", levels=50)
plt.colorbar(label="Velocidade vy)")
plt.title("Componente da Velocidade vy ao Redor do Aerofólio")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis("equal")
plt.show()

# Gráfico da pressão (p)
plt.figure(figsize=(8, 6))
plt.tricontourf(triang, p, levels=50, cmap="jet")
plt.colorbar(label="Pressão (p)")
plt.title("Distribuição da Pressão")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis("equal")
plt.show()
