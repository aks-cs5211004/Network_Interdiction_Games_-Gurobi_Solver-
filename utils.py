import numpy as np
import torch


def simplex_projection(y):
    u = y[np.argsort(-y)]
    d = y.shape[0]
    rho_idx = -1
    cusum = 0
    for i in range(d):
        cusum += u[i]
        temp = u[i] + (1 - cusum) / (i+1)
        if(temp > 0):
            rho_idx = i
    l = (1 - torch.sum(u[:rho_idx+1])) / (rho_idx + 1)
    dt = (y + l).dtype
    res = torch.max(y + l,torch.zeros(d, dtype=dt))
    # eps = 1e-2
    # res += eps
    return res

def f_util(x, S, L, A, B):
    idx = 0
    arr = torch.zeros(S)
    for i in range(S):
        if(len(L) and i == L[idx]):
            arr[i] = -A[i] * x[idx] - B[i]
            idx += 1
            idx %= len(L)
        else:
            arr[i] = -B[i]
    return arr

def l_util(x, L, C, D):
    arr = torch.zeros(len(L))
    for i in range(len(L)):
        arr[i] = C[i] * x[i] + D[i]
    return arr

# Solve OPT-Homo using baselines after getting the sampling
# Function to draw trajectory and flag no. of nodes from L

def draw_trajectory(N, L):
    T = [0]
    last = 0
    while (last < len(N) - 1):
        last = np.random.choice(N[last])
        T.append(last)
    # if T[-1] != len(N) - 1
    # T.append(len(N)-1)
    flag = []
    sta = 0
    for i in T:
        for j in range(sta, len(L), 1):
            if i == L[j]:
                flag.append(j)
                sta = j + 1
                break
        # if i in L:
            # flag += 1
    return T, flag

# Solving Approx-OPT
# Function to generate adjaceny lists after removing L \ {s}
def sub_graph(N, L, s):
    L_mod = []
    for i in L:
        if i != s:
            L_mod.append(i)
    G = []
    for i in range(len(N)):
        temp = []
        if i in L_mod:
            G.append([])
            continue
        else:
            for j in N[i]:
                if j in L_mod:
                    continue
                else:
                    temp.append(j)
        G.append(temp)
    # G.append([])
    return G

# H(s) can be computed solving the Z system of linear eqns
# Write optimization problem g(y, delta) where inner problem is concave, update delta likwise

def approx_opt(x, S, L, N, f_util, l_util, mu, delta, A, B, C, D, all_G):
    g_tot = 0.0
    n_tot = 0.0
    d_tot = 0.0
    x = torch.tensor(x, requires_grad=True)
    idx = 0
    for s in L:
        G = all_G[idx]
        # G = sub_graph(N, L, s)
        n_s, d_s, g_s, _ = SNIG(x[idx:idx+1], S, [s], G, f_util, l_util, mu, delta, A, B, C, D)
        g_tot += g_s
        n_tot += n_s.detach()
        d_tot += d_s.detach()
        idx += 1
    # G = sub_graph(N, L, -1)
    G = all_G[idx]
    residue = denominator(None, S, [], G, f_util, l_util, mu, delta, A, B, C, D)
    d_tot -= len(L) * residue.detach()
    g_tot += delta * len(L) * residue
    return n_tot, d_tot, g_tot, x

def homo_opt_gd(x, Ts, flags, L, A, B, C, D, mu):
    num = 0
    den = 0
    for i in range(len(flags)):
        if(len(flags[i]) > 0):
            r_leader = 0
            U = 0
            for j in Ts[i]:
                U -= B[j]
            for j in range(len(flags[i])):
                U -= C[flags[i][j]] * x[flags[i][j]]
                r_leader += A[flags[i][j]] * x[flags[i][j]] + B[flags[i][j]]
            U /= mu
            U = torch.exp(U)
            # print(U)
            den += U
            num += r_leader * U
        else:
            U = 0
            for j in Ts[i]:
                U -= B[j]
            U /= mu
            U = torch.exp(U)
            # print(U)
            den += U
    return num / den

def SNIG(x, S, L, N, f_util, l_util, mu, delta, A, B, C, D):
    # x is L dim
    # convention 0 is origin, S - 1 is destination
    # L : list of nodes that can be protected (keep it sorted)
    # N : list of list representing valid state transitions
    # x = torch.tensor(x, requires_grad=True)
    v = f_util(x, S, L, A, B) # S dim return
    r = l_util(x, L, C, D) # L dim return
    # print(r)
    M = torch.zeros(S, S)
    for s in range(S):
        for s1 in N[s]:
            M[s][s1] =  torch.exp(v[s] / mu)
    # print(M)
    B = torch.zeros(S, len(L) + 1)
    B[S - 1][len(L)] = 1
    idx = 0
    for s in L:
        B[s][idx] = 1
        idx += 1
    I = torch.eye(S)
    temp_inv, info = torch.linalg.inv_ex(I - M)
    # temp_inv = torch.cholesky_inverse(I - M)
    # print(temp_inv)
    H = torch.matmul(temp_inv, B)
    # H = torch.matmul(torch.inverse(I - M), B)
    n = torch.dot((r * H[0][:len(L)]), H[L,len(L)])
    # d = H[0][len(L)] * torch.sum(r)
    d = H[0][len(L)]
    g = n - delta * d
    # g.backward()
    # print(x.grad)
    return n.detach(), d.detach(), g, x
    # return True

def denominator(x, S, L, N, f_util, l_util, mu, delta, A, B, C, D):
    # x is L dim
    # convention 0 is origin, S - 1 is destination
    # L : list of nodes that can be protected (keep it sorted)
    # N : list of list representing valid state transitions
    # x = torch.tensor(x, requires_grad=True)
    v = f_util(x, S, L, A, B) # S dim return
    # r = l_util(x, L, C, D) # L dim return
    # print(r)
    M = torch.zeros(S, S)
    for s in range(S):
        for s1 in N[s]:
            M[s][s1] =  torch.exp(v[s] / mu)
    # print(M)
    B = torch.zeros(S, len(L) + 1)
    B[S - 1][len(L)] = 1
    idx = 0
    for s in L:
        B[s][idx] = 1
        idx += 1
    I = torch.eye(S)
    temp_inv, info = torch.linalg.inv_ex(I - M)
    # temp_inv = torch.cholesky_inverse(I - M)
    # print(temp_inv)# def simplex_projection(y, M = 2):
#     gamma = torch.min(y) - 0.5
#     d = y.shape[0]
#     dt = y.dtype
#     t = 0
#     while (t < 10):
#         v = y - gamma * torch.ones(d, dtype=dt)
#         v_pos = torch.max(v,torch.zeros(d, dtype=dt))
#         c = 0 
#         for i in range(d): 
#             if (0 <= v[i] and v[i] <= 1):
#                 c += 1
#         gamma -= (M - sum(torch.min(v_pos, torch.ones(d, dtype=dt)))) / c
#         t += 1
#     return torch.min(v_pos, torch.ones(d, dtype=dt))

    H = torch.matmul(temp_inv, B)
    # H = torch.matmul(torch.inverse(I - M), B)
    # n = torch.dot((r * H[0][:len(L)]), H[L,len(L)])
    # d = H[0][len(L)] * torch.sum(r)
    d = H[0][len(L)]
    # g = n - delta * d
    # g.backward()
    # print(x.grad)
    # return n.detach(), d.detach(), g, x
    # return True
    return d

def generalized_isNAN(x):
    x = torch.tensor(x)
    ret = torch.isnan(x)
    # print(ret)
    for y in ret:
        if y == True:
            # print("True")
            return True
    return False

