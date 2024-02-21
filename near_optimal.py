import math

import numpy as np
from gurobipy import GRB, Model, quicksum

#multidict ({combinations:value,..})
#m=Model('RAP'  )

#no. of nodes

f3 = open("near_opt_results", "w+")
for S in range(50,201,50):

    # Sample L
    num_L = int(0.8 * S)
    #L=16
    L = list(np.random.choice(1 + np.arange(S-1), size=num_L, replace=False))
    L.sort()
    L_dict={}
    for i in range(len(L)):
        L_dict[L[i]]=i;


    print("L=",L)


    #Generate partitions
    #take no. of partitions as 1
    M=1


    # Generate utility parameters randomly
    high_A = 1.0
    high_B = 1.0
    high_C = 1.0
    high_D = 1.0
    #Defender
    #C,D is positive
    C = np.random.uniform(low=0, high=high_C, size=len(L))
    D = np.random.uniform(low=0, high=high_D, size=len(L))
    #Adversary
    #A is negative
    A = np.random.uniform(low=0, high=high_A, size=S)
    B = np.random.uniform(low=0, high=high_B, size=S)

    #Get the Graph (*Directed Acyclic?)
    p=0.8
    Graph=[]
    for i in range(S-1):
            temp = []
            for j in range(i+1, S-1, 1):
                if (np.random.rand() > p):
                    temp.append(j)
            if(len(temp) == 0):
                temp.append(i+1)
            Graph.append(temp)
    Graph.append([])


    #Sample N paths from D(x_0) (?)
    #Lets suppose N=20 and K=20
    #constants
    N=30

    #take mu
    mu=2.0






    #Sample N Paths (from the Distribution now)
    #all paths start from only one source '0'  and end at destination "N=19".

    #Calculate Z
    M_mat=np.zeros((S,S))
    for s in range(S):
        for s1 in (Graph[s]):
            M_mat[s][s1]=np.exp((-A[s]*(1/len(L))+B[s])/mu)
    b_mat=np.zeros((S))
    I_mat=np.identity(S)
    b_mat[19]=1
    Z_mat=np.linalg.solve(I_mat-M_mat,b_mat)
        
    paths=[]

    #sampling paths from the distribution
    for i in range(N):
        path = [0]
        last = [0]
        while (last[0] < S - 1):
            #creating the probability distribution
            p=[((math.exp(-A[last[0]]*1/len(L)+B[last[0]])/mu)*Z_mat[j])/Z_mat[last[0]] for j in Graph[last[0]]]
            #sampling from the distributio
            last = np.random.choice(Graph[last[0]],1,p)
            path.append(last[0])
        paths.append(path)

    print(paths)



    
    #Define constants and Bounds
    #x_s=1/L
    L_v=[-0 for i in range(N)]
    U_v=[0 for i in range(N)]
    L_u=[-0 for i in range(N)]
    U_u=[0 for i in range(N)]
    L_x=0
    U_x=1

    max_U_u=0
    for i in range(len(paths)):
        for k in paths[i]:
            if k in L:
                j=L_dict[k]
                U_u[i]+=C[j]+D[j]
                L_u[i]+=D[j]
            U_v[i]+=(A[k]*1/len(L))
            if k in L:
                L_v[i]+=(-A[k])+A[k]*1/len(L)
            else:
                L_v[i]+=A[k]*1/len(L)
            
        U_v[i]=(U_v[i]/mu)
        L_v[i]=(L_v[i]/mu)
        
        
    for i in range(N):
        max_U_u=max(max_U_u,U_u[i])

        

    #To find constants for binary search
    U_optimal=0
    L_optimal=2

    for i in range(len(L)):
        L_optimal=min(L_optimal,D[i])
        U_optimal=max(U_optimal,D[i]+C[i])
    #Doubt here ...
    U_optimal=U_optimal*len(L)
        
    #subject to change
    epsilon=0.01

    #Binary Search Loop
    f1 = open("bounds.txt", "w+")
    f2 = open("optimals.txt", "w+")

    for K in range(40,201,40):

        while(U_optimal-L_optimal>=epsilon):
            wr=str(L_optimal)+" "+str(U_optimal)+"\n"
            f1.write(wr)

            labda=(U_optimal+L_optimal)/2
            
            #Define Model
            model=None
            model=Model('mips')

            #Define variables:-
            z=[]
            u=[]
            s=[]
            x=[]
            kappa=[]
            v=[]

            #x
            for i in range(len(L)):
                x.append(model.addVar(lb=L_x,ub=U_x,vtype=GRB.CONTINUOUS,name=f'x_{i}'))
                
            for i in range(N):
                #z,delta,s
                temp_z=[]
                temp_s=[]
                for j in range (K):
                    temp_z.append(model.addVar(ub=1,lb=0,vtype=GRB.BINARY,name=f'z_{i}_{j}'))
                    #No upper lower bound added here?
                    temp_s.append(model.addVar(lb=-float("inf"),ub=float("inf"),vtype=GRB.CONTINUOUS,name=f's_{i}_{j}'))
                #z
                z.append(temp_z)
                #s
                s.append(temp_s)
                #u
                u.append(model.addVar(lb=-float("inf"),ub=float("inf"),vtype=GRB.CONTINUOUS,name=f'u_{i}'))
                #k
                kappa.append(model.addVar(lb=-float("inf"),ub=float("inf"),vtype=GRB.CONTINUOUS,name=f'k_{i}'))
            temp=model.addVars(N,lb=-float("inf"),ub=float("inf"),vtype=GRB.INTEGER)

            #Define Constraints:-
            model.addConstrs(u[i]== sum((C[L_dict[j]]*(x[L_dict[j]])+D[L_dict[j]] if (j in L) else 0) for j in paths[i])  for i in range(N))

            for i in range(N):
                model.addConstr(temp[i]== quicksum(z[i][k] for k in range(K)))
                model.addConstr((L_v[i]+((U_v[i]-L_v[i])*temp[i])/K + kappa[i])==(quicksum(-A[j]*(x[L_dict[j]]) +A[j]*1/len(L) if (j in L) else A[j]*1/len(L) for j in paths[i])/mu))
            for i in range(N):
                model.addConstr(kappa[i]<=(U_v[i]-L_v[i])/K)
                model.addConstr(kappa[i]>=0)
            for i in range(N):
                for j in range(K-1):
                    model.addConstr(z[i][j]>=z[i][j+1])

            for i in range(N):
                model.addConstrs(s[i][j]<=(U_u[i]-labda)*z[i][j] for j in range(K))
                model.addConstrs(s[i][j]>=(L_u[i]-labda)*z[i][j] for j in range(K))
                model.addConstrs(s[i][j]<=(u[i]-labda)-(L_u[i]-labda)*(1-z[i][j]) for j in range(K))
                model.addConstrs(s[i][j]>=(u[i]-labda)-(U_u[i]-labda)*(1-z[i][j]) for j in range(K))
            model.addConstr(sum(x[i] for i in range(len(L)))<=M)
            
            #Define objective:-
            delta=[]
            for i in range(N):
                col = []
                for j in range(K):
                    col.append(0)
                delta.append(col)
                
            for i in range(N):
                for j in range(K):
                    delta[i][j]=(math.exp(L_v[i]+(j+1)*(U_v[i]-L_v[i])/K)-math.exp(L_v[i]+(j)*(U_v[i]-L_v[i])/K))/((U_v[i]-L_v[i])/K)
            objective=(1/N)* sum(((u[i]-labda)* math.exp(L_v[i]) + ((U_v[i]-L_v[i])/K)*sum(delta[i][j]*s[i][j]  for j in range(K)) ) for i in range (N)) 

            model.setObjective(objective,GRB.MAXIMIZE)
            model.write('./Result.lp')
            model.optimize()
            step_optimal=model.objVal
            
            wr=str(step_optimal)+"\n"
            f2.write(wr)
            
            if step_optimal>=0:
                L_optimal=labda
            else:
                U_optimal=labda
                
                
        f3.write("No. of nodes="+str(S)+" N="+str(N)+" K="+str(K) +" Optimal value="+ str(L_optimal)+"\n")
        
        






        
        
        

        
            
            

        

        










