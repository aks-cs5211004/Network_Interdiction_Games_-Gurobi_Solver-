import os

from utils import *

# np.random.seed(0)
trial = 0
while(trial < 20):
    sig = 0
    S = 20
    # L is defendable nodes
    num_L = int(0.8 * S)
    lr1 = 1e-4
    lr2 = 1e-3
    # lr1 = 1e-6
    # lr2 = 1e-5
    # ensure source and destination are not defendable
    L = list(np.random.choice(1 + np.arange(S-1), size=num_L, replace=False))
    L.sort()
    print(len(L))
    # Generate utility parameters
    high_A = 1.0
    high_B = 1.0
    high_C = 1.0
    high_D = 1.0
    C = np.random.uniform(low=0, high=high_C, size=len(L))
    D = np.random.uniform(low=0, high=high_D, size=len(L))
    A = np.random.uniform(low=0, high=high_A, size=S)
    B = np.random.uniform(low=0, high=high_B, size=S)

    A = torch.tensor(A)
    B = torch.tensor(B)
    C = torch.tensor(C)
    D = torch.tensor(D)

    # N is our DAG
    p = 0.8
    N = []
    # max_out_edges = 4
    # max_out_edges_freq = 7
    sta = 0
    for i in range(S-1):
        temp = []
        for j in range(i+1, S-1, 1):
            if (np.random.rand() > p):
                temp.append(j)
        if(len(temp) == 0):
            temp.append(i+1)
        N.append(temp)
    N.append([])

    # print(N)

    mu = 2.0
    num_epochs = 50
    # lr = 1e-5
    for mod in range(num_L):
        # lr = 1e-5
        all_G = []
        for s in L:
            all_G.append(sub_graph(N, L, s))
        all_G.append(sub_graph(N, L, -1))
        xt = simplex_projection(torch.rand(len(L)))
        n, d, g, x = approx_opt(xt, S, L, N, f_util, l_util, mu, 0.1, A, B, C, D, all_G)
        deltat = n / d
        for rep in range(num_epochs):
            n, d, g, x = approx_opt(xt, S, L, N, f_util, l_util, mu, deltat, A, B, C, D, all_G)
            g.backward()
            xt = simplex_projection(xt + lr1 * x.grad.detach())
            if(generalized_isNAN(xt)):
                sig = 1
                break
            deltat = n / d
            # print(xt)
            # print(deltat)
        x_approx_opt = torch.tensor(xt)
        if (sig):
            break
    if(sig):
        continue
        # Initialize
        # lr = 1e-4
        xt = simplex_projection(x_approx_opt)
        n, d, g, x = SNIG(xt, S, L, N, f_util, l_util, mu, 0.1, A, B, C, D)
        deltat = n / d
        for rep in range(num_epochs):
            xc = torch.tensor(xt, requires_grad=True)
            n, d, g, x = SNIG(xc, S, L, N, f_util, l_util, mu, deltat, A, B, C, D)
            g.backward()
            xt = simplex_projection(xt + lr2 * x.grad.detach())
            if(generalized_isNAN(xt)):
                sig = 1
                break
            deltat = n / d
            # print(xt)
            # print(deltat)
        x_approx_opt = torch.tensor(xt)
        if (sig):
            break
    if(sig):
        continue
        print(x_approx_opt)
        n, d, g, x = SNIG(x_approx_opt, S, L, N, f_util, l_util, mu, 0.1, A, B, C, D)
        print("Approx OPT : ", n / d)
        if(min(x_approx_opt) > 0):
            break
        change_L = torch.argmax(xt)
        change_A = L[change_L]
        # print("A : ", A[change_A])
        # print("B : ", B[change_A])
        # print("C : ", C[change_L])
        # print("D : ", D[change_L])
        reduceby = 0.5
        A[change_A] *= (1 + reduceby)
        B[change_A] *= (1 + reduceby)
        C[change_L] *= (1 - reduceby)
        D[change_L] *= (1 - reduceby)

    # Now run on the all the algorithms on the network
    all_T = []
    flags = []
    for i in range(1000):
        T, flag = draw_trajectory(N, L)
        all_T.append(T)
        flags.append(flag)

    num_epochs = 300
    num_epochs_gd = 200
    # Baseline
    xt = simplex_projection(torch.rand(len(L)))
    lr = 1e-1
    for rep in range(num_epochs_gd):
        xc = torch.tensor(xt, requires_grad=True)
        g = homo_opt_gd(xc, all_T, flags, L, A, B, C, D, mu)
        # print("Epoch : ", rep + 1, "OPT : ", g.detach())
        g.backward()
        xt = simplex_projection(xt + lr * xc.grad.detach())
        if(generalized_isNAN(xt)):
            sig = 1
            break
    x_homo_gd = torch.tensor(xt)
    if(sig):
        continue
    
    # Approach 1
    xt = simplex_projection(torch.rand(len(L)))
    # print(xt)
    n, d, g, x = SNIG(xt, S, L, N, f_util, l_util, mu, 0.1, A, B, C, D)
    deltat = n / d
    # Loop
    # lr = 1e-5
    for rep in range(2 * num_epochs):
        xc = torch.tensor(xt, requires_grad=True)
        n, d, g, x = SNIG(xc, S, L, N, f_util, l_util, mu, deltat, A, B, C, D)
        g.backward()
        print(sum(xt))
        xt = simplex_projection(xt + lr1 * x.grad.detach())
        if(generalized_isNAN(xt)):
            sig = 1
            break
        deltat = n / d
    if(sig):
        continue
        # print("Epoch : ", rep + 1, "Delta : ", deltat)

    x_homo_opt = torch.tensor(xt)
    
    # Approach 2
    # Step 1
    # lr = 1e-5
    all_G = []
    for s in L:
        all_G.append(sub_graph(N, L, s))
    all_G.append(sub_graph(N, L, -1))
    xt = simplex_projection(torch.rand(len(L)))
    n, d, g, x = approx_opt(xt, S, L, N, f_util, l_util, mu, 0.1, A, B, C, D, all_G)
    deltat = n / d
    print("Initial : ", deltat)
    # lr = 1e-10
    for rep in range(num_epochs):
        n, d, g, x = approx_opt(xt, S, L, N, f_util, l_util, mu, deltat, A, B, C, D, all_G)
        g.backward()
        # print(x.grad)
        # print(xt)
        xt = simplex_projection(xt + lr2 * x.grad.detach())
        if(generalized_isNAN(xt)):
            sig = 1
            break
        # print(x.grad)
        # print(xt + lr * x.grad.detach())
        # print("X : ", xt)
        deltat = n / d
        # print("Epoch : ", rep + 1, "Delta : ", deltat)
    if(sig):
        continue

    x_approx_opt = torch.tensor(xt)

    # Step 2
    # Initialize
    # lr = 1e-4
    xt = simplex_projection(x_approx_opt)
    # print(xt)
    n, d, g, x = SNIG(xt, S, L, N, f_util, l_util, mu, 0.1, A, B, C, D)
    deltat = n / d
    for rep in range(num_epochs):
        xc = torch.tensor(xt, requires_grad=True)
        n, d, g, x = SNIG(xc, S, L, N, f_util, l_util, mu, deltat, A, B, C, D)
        g.backward()
        # print(xt)
        xt = simplex_projection(xt + lr2 * x.grad.detach())
        if(generalized_isNAN(xt)):
            sig = 1
            break
        deltat = n / d
        # print("Epoch : ", rep + 1, "Delta : ", deltat)
    if(sig):
        continue
    x_approx_opt = torch.tensor(xt)
    results_location = './simulations/{}_S_{}_L_{}_p_{}_mu_{}_trial.npy'.format(S, num_L, p, mu, trial)
    if not os.path.isfile(results_location):
            open(results_location,"w+")
    log = {}
    n, d, g, x = SNIG(x_homo_gd, S, L, N, f_util, l_util, mu, 0.1, A, B, C, D)
    print("Trial : ", trial)
    print("Homo OPT GD : ", n / d)
    log['B'] = n / d
    n, d, g, x = SNIG(x_approx_opt, S, L, N, f_util, l_util, mu, 0.1, A, B, C, D)
    print("Approx OPT : ", n / d)
    log['A2'] = n / d
    n, d, g, x = SNIG(x_homo_opt, S, L, N, f_util, l_util, mu, 0.1, A, B, C, D)
    print("Homo OPT : ", n / d)
    log['A1'] = n / d
    np.save(results_location, log)
    trial += 1


