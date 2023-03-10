import numpy
import numpy as np
import pandas as pd
from itertools import combinations
import time
# =============================================================================
from scipy import spatial
# =============================================================================
import matplotlib.pyplot as plt
import random
import math




def One_Zero_Relocate(path,cost):
    flag = False
    minCost = Calculate_Cost(path,cost)
    temp_minCost = minCost

    temp_path = path.copy()
    path_list = path.tolist()
    # print("Path type",path.dtype)
    path_list_row = []
    inner_row_temp = []
    path_joker = [[]]
    path_joker = path.tolist()
    biggest_row = 0
    temp_client = 0

    for rows in range((np.shape(path)[0])-1,0,-1):
        for i in range(np.size(np.nonzero(path[rows,:]))-1,1,-1 ):
            improved_count = 0
            if path[rows,i] ==1 or path[rows,i]==0 : break
            ##Me auth tsekaroyme thn veltisth topothetish toy client gia na mhn kratame thn prwth improved diadromh alla thn kaluterh
            temp_row_no = rows
            temp_row_2nd = path_list[rows][:]
            path_list_row = path_list[rows][:]
            # print("LOL_1",i, path_list_row)
            client = path_list_row.pop(i)
            if client==0 or client==1:
                break
            else:
                path_list_row.append(0)

            for rows_inner in range((np.shape(path)[0])): # topothetoume ton pelath pou vrhkame se oles tis pithanes theseis kai psaxnoume thn veltisth allagh
                for i_inner in range(1, np.size((np.nonzero(path[rows, :])))):
                    if path_list[rows_inner][i_inner] == 1 or path_list[rows_inner][i_inner] == 0 : break # den allazoume pote thn apothiki kai ston 0 stamataei na psaxnei
                    if rows == rows_inner: # mhn pareis apthn idia diadromh
                        break
                    inner_row_temp = path_list[rows_inner][:]
                    inner_row_temp.insert(i_inner,client) #ftiaxnoume mia prosorini lista me ton pelath poy topothetisame se mia nea diadromi
                    sum_c = 0
                    for idx in range(len(path)):
                        if rows_inner == idx:
                            sum_c = sum_c+ Row_Cost_Calculator(np.array(inner_row_temp),costos) # to kostos ths grammhs poy prosthesame enan pelath
                            continue
                        elif rows == idx :
                            sum_c = sum_c + Row_Cost_Calculator(np.array(path_list_row),costos) # to kostos ths grammhs poy afairesame ena pelath
                            continue
                        else:
                            sum_c = sum_c + Row_Cost_Calculator(np.array(path_list[idx][:]),costos) # upologizoume to kostos gia tis grammes poy exoun parameinei idies
                    if (T_max<Row_Cost_Calculator(np.array(inner_row_temp),costos)) | (Q_max<( demand[client-1] + Qpath[rows_inner] )):##Ean oi allages sta path den ksepernoyn Qman,Tmax synexise
                        continue
                    elif(sum_c<temp_minCost): # exei to arxiko kostos tou pinaka prin kanei opoiadipote allagh
                        flag = True
                        temp_minCost = sum_c # kanoume antikatastash tou kostous os to trexon elaxisto
                        temp_row_no = rows_inner # kratame ton arithmo ths grammhs pou valame ton pelath mas
                        temp_path_list_row = path_list_row # kratame th grammh pou vgalame enan pelath mas
                        improved_count+=1
                        temp_row_2nd = inner_row_temp # kratame thn olh thn grammh
                        temp_client = client # kratame ton sugkekrimeno pelath


            if flag: # exoume vrei th veltisth diadromh meta apo olous tous pithanous sindiasmous kai monimopoioume tis allages
                path_list[rows] = temp_path_list_row
                path_list[temp_row_no][:] = temp_row_2nd #Meta apo olous tous dundiasmous topothetishs toy client , krathsame ta stoixeia ths kalyterhs diadromhs kai ta efarmozoume sto kanoniko path mas
                Qpath[temp_row_no] = Qpath[temp_row_no] + demand[temp_client - 1]
                Tpath = Row_Cost_Calculator(np.array(temp_row_2nd), costos)
                flag=False
                # print("New Path",path_list)
                # print("Row",temp_path_list_row,temp_row_2nd)
            # temp_minCost = Calculate_Cost(np.array(path_list),costos)

    for rows_x in range((np.shape(path)[0])):
        while np.array(path_list[rows_x]).size>path[rows_x].size :
            path_list[rows_x][:] = np.delete(np.array(path_list[rows_x]),np.array(path_list[rows_x]).size-1)#svinoume mhdenika etsi vste na exoume opoiomorfia stis grammes
        path[rows_x][:] = np.array(path_list[rows_x]).copy()# metatrepoume th lista se pinaka
    return path,temp_minCost



def two_opt(path,costos):
    #best = path.copy()
    improved = True
    i_counter=0
    while improved:
        improved = False
        for i in range(1, len(path) - 2):
            if path[i]==1 :i_counter+=1
            for j in range(i + 1, len(path)):
                # if j - i == 1: continue  # changes nothing, skip then
                if path[j-1]==0: break
                # if path[j - 1] == 1: break
                if (path[j]==1): break
                new_route = np.array(path[:]).copy()
                new_route[i:j] = np.array(path[j - 1:i - 1:-1]).copy()  # this is the 2woptSwap
                if Row_Cost_Calculator(new_route,costos) < Row_Cost_Calculator(path,costos):  # what should cost be?
                    path = new_route.copy()
                    # print("yoloo",path,best)
                    improved = True
        #path = best.copy()
    return path

def Row_Cost_Calculator(row,costos):
    ST = s_time
    DT = s_time
    sum = 0
    counter_one=0
    # allaksa range 1 -> 0
    for i in range(0,len(row[np.nonzero(row)])-1 ):
        if row[i] != 0:
            sum = sum + costos[row[i]-1,row[i+1]-1]+ST
        if row[i]==1 : counter_one+=1
        if (row[i] == 1) and (counter_one > 1) and (i < len(row[np.nonzero(row)]) - 1):
            sum = sum - ST + DT
    return sum


#allazw calculate_cost -> if path[rows,i]!=0
def Calculate_Cost(path,costos):
    number_of_routes = len(path)
    F0 = Nu * 50
    F1 = (number_of_routes - Nu) * 150
    Total_Fixed = F0 + F1

    ST = s_time
    DT = s_time
    sum=0

    for rows in range((np.shape(path)[0])):
        counter_one = 0
        for i in range(0,np.size((np.nonzero(path[rows,:]))) - 1):
            if path[rows,i]!=0 :
                sum = sum + costos[path[rows,i]-1,path[rows,i+1]-1] + ST
            if path[rows,i]==1: counter_one+=1
            if (path[rows,i]==1) and (counter_one>1) and (i<np.size((np.nonzero(path[rows,:]))) - 1):
                sum = sum - ST + DT
    return sum + Total_Fixed




def plot_coords(coords,path, plot_title):
    plt.figure(figsize=((10,8)))
    for path_rows in range(len(path)):
        temp = np.transpose(path[path_rows,np.nonzero(path[path_rows,:])]-1)
        xs = coords[temp,0]
        ys = coords[temp,1]
        # print("TEMP VALUE",temp, xs , ys)
        plt.scatter(xs, ys,linewidths=1)
        plt.plot(xs,ys)
        for index in range(len(temp)):
            plt.text(xs[index][0], ys[index][0] + 0.5, '{}'.format(temp[index][0]+1))
        plt.title(plot_title)
    plt.show()
    return


def Nearest_Neighbor(XYa, costos , Nu):
    square = np.empty([2, 2])
    
    for x in range(len(np.array(XYa))):
        for yy in range(len(np.array(XYa))):
            square[0] = XYa[x]
            square[1] = XYa[yy]
            costos[x][yy] = spatial.distance.pdist(square, metric='euclidean').copy()

    
    costos_const = np.copy(costos)
    costos[:, 0] = np.inf
    np.fill_diagonal(costos, np.inf)

    i = 0
    c = 1
    m = 1
    QPath = np.zeros([MAX_DIADROMES, 1])
    TPath = np.zeros([MAX_DIADROMES, 1])
    path = np.zeros([MAX_DIADROMES, N-1], dtype='int')
    path[0, 0] = 1
    forthgo = 0
    flag = 1
    delay_time = s_time 
    
    while (c < N):
        if forthgo >= Nu: flag = 0
        
        min_Y = np.amin(costos[i, :])
        komvos = np.argmin(costos[i, :])
        
        if ((TPath[forthgo] + min_Y + s_time + costos[0,komvos] * flag) > T_max):
            path[forthgo,m] = flag
            forthgo += 1
            path[forthgo, 0] = 1
            m = 1
            i = 0
            continue
        
        elif ((QPath[forthgo] + demand[komvos] ) > Q_max):
            min_Y = np.amin(costos[0, :])
            komvos = np.argmin(costos[0, :])
            if ((TPath[forthgo] + costos_const[0,i] + delay_time + min_Y + s_time + costos[0,komvos] * flag) <= T_max):
                path[forthgo,m] = 1
                m = m + 1
                QPath[forthgo] = 0
            else:
                path[forthgo,m] = flag
                forthgo += 1
                path[forthgo, 0] = 1
                m = 1
                i = 0
                continue
            
        QPath[forthgo] = QPath[forthgo] + demand[komvos]
        TPath[forthgo] = TPath[forthgo] + min_Y + s_time

        path[forthgo, m] = komvos+1
        m += 1
        costos[:, komvos] = np.inf

        i = komvos
        c += 1  # TELOS WHILE

    costos = costos_const.copy()
    path = path[~np.all(path == 0, axis=1)]
    path = np.transpose(path)
    path = path[~np.all(path == 0, axis=1)]
    path = np.transpose(path)

    return path,costos,QPath,TPath



def Simulated_Annealing(path,initial_cost):
    x = []
    for i in range(1, N):
        x.append(i)

    comb_list =  list(combinations((x), 2))#dose mas enan pithano sundiasmo 2 komvon poy tha allaxoume theseis
    currentPath= path.copy()
    bestSolution = path.copy()
    bestCost=initial_cost;
    Tstart = 500
    Tmin = 10
    a = 0.95
    f = 0
    f1 = 0
    f2 = 0
    f3 = 0
    T = Tstart


    cost = 0
    for i in range(path.shape[0]):
        cost = cost + Row_Cost_Calculator(path[i,:],costos)
    minCost = np.array(cost).copy()

    
    while T > Tmin :

        for combinatio in comb_list: # gia olous toys pithanous sindiasmous
            rand_i = combinatio[0] # edo vazoume ton prwto komvo apton sindiasmo mas
            rand_j = combinatio[1] # edo vazoume ton deutero komvo apton sindiasmo mas
            route_i, position_i = np.where(currentPath== (rand_i + 1))
            route_j, position_j = np.where(currentPath== (rand_j + 1))
            if route_i == route_j : #allages ginontai mono apo diaforetikes diadromes
                continue


            if (Qpath[route_j] - demand[rand_j] +demand[rand_i]) > Q_max:
                continue

            if (Qpath[route_i] - demand[rand_i] + demand[rand_j]) > Q_max:
                continue

            temp_Path =currentPath.copy()
            temp_Path[route_i,position_i] = rand_j+1
            temp_Path[route_j,position_j] = rand_i+1

            temp_Cost_i = Row_Cost_Calculator(temp_Path[route_i,:].ravel(),costos)
            temp_Cost_j = Row_Cost_Calculator(temp_Path[route_j,:].ravel(),costos)


            if temp_Cost_i > T_max:
                continue
            temp_Cost_j = Row_Cost_Calculator(temp_Path[route_j, :].ravel(), costos)
            if temp_Cost_j > T_max:
                continue
            cost = Calculate_Cost(temp_Path,costos)

            f3 = f3 +1

            # AN EINAI MIKROTERO KANOYME PANTA THN KINISI
            if cost < minCost:
                f2 = f2 + 1
                minCost = cost
                currentPath= temp_Path.copy()
                Qpath[route_j] = Qpath[route_j] - demand[rand_j] + demand[rand_i]
                Qpath[route_i] = Qpath[route_i] - demand[rand_i] + demand[rand_j]
            else:
            ##  ALLIWS ME PITHANOTITA
                delta = cost - minCost
                accept_proba = math.exp(-delta/T)
                if random.uniform(0,1) < accept_proba:
                    f1 = f1 +1
                    currentPath= temp_Path.copy()
                    Qpath[route_j] = Qpath[route_j] - demand[rand_j] + demand[rand_i]
                    Qpath[route_i] = Qpath[route_i] - demand[rand_i] + demand[rand_j]
        f = f + 1
        T = T - f*a

        ###============= LOCAL SEARCH =============###
        candidate_path =currentPath.copy()
        while True:
            cost_before_ls=Calculate_Cost(candidate_path,costos)
            candidate_path,_= One_Zero_Relocate(candidate_path,costos)
            for i in range(np.shape(candidate_path)[0]):
                candidate_path[i,:] = two_opt(candidate_path[i,:],costos)
            cost_after_ls=Calculate_Cost(candidate_path,costos)
            # print("LS improvement: ", cost_before_ls-cost_after_ls)
            if not cost_before_ls-cost_after_ls>0 :
                break;




        ###============= KRATAME TO KALYTERO =============###
        candidate_cost=Calculate_Cost(candidate_path,costos) 
        if candidate_cost <bestCost:
              bestCost=candidate_cost
              bestSolution =candidate_path.copy()
              print("NEW BEST COST:",bestCost, " Temp=",T)


        # print("ALL_THE_FS\n",f,f1,f2,f3,"\nAnnealing path:\n",bestPath,"\ncost:\n",minCost)
            #plot_coords(coords,bestPath, "Routes after Annealing")
    return bestSolution







###========= ARXIKOPOIISI DEDOMENWN===========================##
MAX_DIADROMES = 20
Nu = 4
N :int = pd.read_excel("data.xlsx", skiprows=1 - 1, usecols="A", nrows=1, header=None, names=["Value"]).iloc[0]["Value"]
Q_max :float = pd.read_excel("data.xlsx", skiprows=1 - 1, usecols="B", nrows=1, header=None, names=["Value"]).iloc[0]["Value"]
T_max :float= pd.read_excel("data.xlsx", skiprows=1 - 1, usecols="C", nrows=1, header=None, names=["Value"]).iloc[0]["Value"]
s_time = pd.read_excel("data.xlsx", skiprows=1 - 1, usecols="D", nrows=1, header=None, names=["Value"]).iloc[0]["Value"]

coords = np.array(pd.read_excel("data.xlsx", header=None, index_col=0, usecols="E:G"))
demand = np.array(pd.read_excel("data.xlsx", header=None, index_col=0, usecols="G:H"))
costos = np.empty([N, N], dtype='float32')


    
st = time.time()
###========= ARXIKI LUSI===========================##
path,costos,Qpath,Tpath= Nearest_Neighbor(coords, costos , Nu) ###=====NEAREST NEIGHBOR
print(len(path),path)
print("Nearest Neighbor path:\n",path)
plot_coords(coords,path, "Best Routes")
init_cost=Calculate_Cost(path,costos)
print("Initial Cost: ", init_cost)
###============= Simulated Annealing optimization
best_path= Simulated_Annealing(path,init_cost)
print("Best Cost:",Calculate_Cost(best_path,costos))
plot_coords(coords,best_path, "Best Routes")

et = time.time()
total_time = et - st
print("Exectution Time in seconds : ",total_time)
