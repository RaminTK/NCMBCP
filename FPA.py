import numpy as np
from copy import deepcopy as dc
from time import process_time
import random
from math import exp, floor

def Read_location(file):
    f = open('/content/drive/MyDrive/Colab Notebooks/MBCP/'+file, 'r')
    data = []
    for line in f:
        data.append([float(v) for v in line.split()])
    f.close()
    return data

def Read_donation(file):
    f = open('/content/drive/MyDrive/Colab Notebooks/MBCP/'+file, 'r')
    data = []
    for line in f:
        data.append([float(v) for v in line.split()])
    f.close()
    return data
    
def duration(rout,total_distance):
    dur = 0
    for i in range(len(rout) -1) :
        dur+= total_distance[rout[i]][rout[i+1]]
    return dur

def degree_n_poss(old_poss,first_list,total_distance):
    new_poss = []
    for i in old_poss :
        for j in first_list :
            if duration (i+[j,0],total_distance) <= spt and j not in i:
                new_poss.append(i+[j])
    return new_poss

def minTest(roots,total_distance):
    minimumTest = []
    for i in roots:
        minimumTest.append(duration(i,total_distance))
    minimum = min(minimumTest)
    return minimum

def meetTime(rout,start_time,total_distance):
    meet_time = [start_time]
    for i in range(len(rout)-1):
        meet_time.append(start_time + duration([rout[i],rout[i+1]],total_distance))
        start_time = meet_time[-1]
    return meet_time

def multi_healthy_collected_plt (rout,start_time,donation_data,spt,total_distance):
    don_data = dc(donation_data)
    pick_list = []
    for i in range(len(rout)):
        meet_time = meetTime(rout[i],start_time[i],total_distance)
        pickup = []
        for j in range(len(rout[i])):
            p = 0
            for k in range(len(don_data[rout[i][j]])):
                if don_data[rout[i][j]][k] <= meet_time[j] and don_data[rout[i][j]][k] >= meet_time[-1]-spt:
                    p += 1
                    don_data[rout[i][j]][k] = 99999
            pickup.append(p)
        pick_list.append(sum(pickup))
    return sum(pick_list)

def random_start (remain,multipleRout,total_distance):
    size = len(multipleRout) + 1
    temp = np.random.dirichlet(np.ones(size),size=1)[0]
    finalRemain = [np.floor(num) for num in (temp * (remain+1))]
    finalRemain.pop()
    for i in range(len(finalRemain)-1):
        finalRemain[i+1] += finalRemain[i]
    times = [0]
    for rout in multipleRout:
        times.append(meetTime(rout,times[-1],total_distance)[-1])
    times.pop()
    for i in range(len(times)):
        times[i] = times[i] + finalRemain[i]
    return times

def improved_st (initial_sol,initial_time,donation_data,spt,total_distance):
    solution = dc(initial_sol)
    start_time = dc(initial_time)
    rem_time = work_hours - sum([duration(i,total_distance) for i in solution])
    if rem_time < 0 :
        return initial_time
    else:
        for i in range(15):
            temp_start_time = random_start (rem_time,solution,total_distance)
            if multi_healthy_collected_plt(solution,start_time,donation_data,spt,total_distance) < multi_healthy_collected_plt(solution,temp_start_time,donation_data,spt,total_distance):
                start_time = dc(temp_start_time)
        return start_time

def initial_timing (sol,total_distance):
    initial_time = ([duration(i,total_distance) for i in sol])
    for i in range(len(initial_time)-1) :
        initial_time[i+1] = initial_time[i]+initial_time[i+1]
    initial_time = [0]+initial_time
    initial_time.pop()
    return initial_time

def updating_donation_list (rout,start_time,donation_data,total_distance):
    don_data = dc(donation_data)
    global spt
    for i in range(len(rout)):
        meet_time = meetTime(rout[i],start_time[i],total_distance)
        for j in range(len(rout[i])):
            for k in range(len(don_data[rout[i][j]])):
                if don_data[rout[i][j]][k] <= meet_time[j] and don_data[rout[i][j]][k] >= meet_time[-1]-spt:
                    don_data[rout[i][j]][k] = 99999
    return don_data

def initial_population(population_size,roots,min_test,work_hours,donation_data,spt,total_distance):
    population = []
    population_timing = []
    for i in range(population_size):
        multiSeq = []
        multiTime = [0]
        a = dc(random.choice (roots)) 
        multiTime.append(duration(a,total_distance)+multiTime[-1])
        multiSeq.append(a)
        while  multiTime[-1] + min_test < work_hours :
            a = dc(random.choice (roots))
            multiSeq.append(a)
            multiTime.append(multiTime[-1]+duration(a,total_distance))
        multiTime = multiTime[:-2]
        multiSeq.pop()
        population.append(multiSeq)
        population_timing.append(improved_st (multiSeq,multiTime,donation_data,spt,total_distance))
    return population,population_timing

def chrom_pool (donation_data,spt,the_size,total_distance):
    x = list(range(1,len(donation_data)))
    first_list = []
    for i in x:
        if duration([i,0],total_distance) <= spt :
            first_list.append(i)
    poss_one = [[i]for i in first_list]
    old_poss = poss_one
    final_list = poss_one
    while len(final_list) < the_size and len(old_poss) > 0 :
        old_poss = degree_n_poss(old_poss,first_list,total_distance)
        final_list.extend(old_poss)      
    routs =final_list
    roots = [[0]+i+[0] for i in routs]
    return roots
 
def counting_check (rout,start_time,donation_data,total_distance,spt,best_result):
    don_data = dc(donation_data)
    for z in range(len(rout)):
        for i in range(len(rout[z])):
            meet_time = meetTime(rout[z][i],start_time[z][i],total_distance)
            for j in range(len(rout[z][i])):
                for k in range(len(don_data[rout[z][i][j]])):
                    if don_data[rout[z][i][j]][k] <= meet_time[j] and don_data[rout[z][i][j]][k] >= meet_time[-1]-spt:
                        don_data[rout[z][i][j]][k] = 88888
    counter  = 0
    for row in don_data :
        for item in row :
            if item == 88888 :
                counter +=1
    if counter != sum(best_result):
        print('there is a conflict')

def CP_check(donation_data,best_result):
    counter = 0
    for i in donation_data:
        for j in i :
            if j == 99999:
                counter+=1
    if counter != sum(best_result):
        ('there is a conflict in CP counting')
        
def time_conflict(routes,timing,total_distance):
    for i in range(len(routes)):
        flag = 0
        for j in range(len(routes[i])-1):
            if timing[i][j]+duration(routes[i][j],total_distance)> timing [i][j+1]:
            #     pass
            # else:
                flag = 1
                break
        if flag == 1 :
            print('there is a conflict between timings for vehicle: ', i+1)
            print(timing[i])
            print(routes[i])
            break
        
  
def new_solution (tour_org,total_distance,spt, iter, MaxIter, sigma_initial, sigma_final, roots, best_sol):
    tour = dc(tour_org)
    best = dc(best_sol)
    p = dc(random.uniform(0,1))
#######################  abiotic/self-pollination 
    if  p < 0.7:
        opt1 = random.randint(0,len(tour)-1)
        opt2 = random.randint(0,len(tour)-1)
        tour[opt1],tour[opt2] = tour[opt2],tour[opt1]
#######################  biotic/cross-pollination 
    elif p < 0.9:
        opt1 = random.randint(0,len(best)-1)
        opt2 = random.randint(0,len(tour)-1)
        tour[opt2] = best[opt1]        
    else:
#######################  abiotic/self-pollination 
        opt1 = random.randint(0,len(tour)-1)
        new_route = dc(random.choice (roots))
        tour[opt1] = new_route
    timing = initial_timing (tour,total_distance)
    timing = spatial_st (tour,timing,donation_data,spt,total_distance, iter, MaxIter, sigma_initial,sigma_final)
    return tour, timing



def weak_del (fitness,population,population_timing,p_min):
    for i in range(len(fitness)):
        if len(fitness) <= p_min:
            break
        weak_id = dc(np.argmin(fitness))
        population.pop(weak_id)
        population_timing.pop(weak_id)
        fitness.pop(weak_id)
    return fitness,population,population_timing



def spatial_st (initial_sol,initial_time,donation_data,spt,total_distance,iter, MaxIter, sigma_initial,sigma_final):
    solution = dc(initial_sol)
    start_time = dc(initial_time)
    rem_time = work_hours - sum([duration(i,total_distance) for i in solution])
    if rem_time < 0 :
        return initial_time
    else:
        spatial_range = dc(spatial(iter, MaxIter, sigma_initial,sigma_final))
        for i in range(spatial_range):
            temp_start_time = random_start (rem_time,solution,total_distance)
            if multi_healthy_collected_plt(solution,start_time,donation_data,spt,total_distance) < multi_healthy_collected_plt(solution,temp_start_time,donation_data,spt,total_distance):
                start_time = dc(temp_start_time)
        return start_time


def spatial(iter, MaxIter, sigma_initial,sigma_final):
    sigma = (((MaxIter - iter)/(MaxIter - 1))**2) * (sigma_initial - sigma_final) + sigma_final;
    return (int(sigma*15)+1)




dons = ('don1.txt','don2.txt','don3.txt','don4.txt','don5.txt','don6.1.txt','don7.txt')

locs = ('loc1.1.txt','loc2.1.txt','loc3.1.txt','loc4.1.txt','loc5.1.txt','loc6.1.txt','loc7.1.txt')

vehicle_num = (6,6,7,8,9,10,10)
spoil_time = (300,500,700)

sigma_initial = 1
sigma_final = 0.01
MaxIter = 125      ###################### 30  ==> 25


for i in range(10):                                                              #################### change ####################### 10-1
    for spt in spoil_time :
        the_size = spt*200 ############## 200 ==>
        for prb in range(7):                                                
            start = process_time()
            donation_da = Read_donation(dons[prb]) 
            donation_data = np.array([np.array(xi) for xi in donation_da])
            location_data = Read_location(locs[prb])  
            total_distance = np.array([np.array(xi) for xi in location_data])
            backup_donation_data = dc(donation_data)
            roots = chrom_pool (donation_data,spt,the_size,total_distance)
            population_size = int(10 * np.ceil(np.log(len(roots))))             ############### 10 ===> 
            work_hours = 1500+spt
            min_test = dc(minTest(roots,total_distance))
            best_result = []
            best_sequence = []
            best_time = []
            for vehicle in range(vehicle_num[prb]):    

        # '''============================|initial population|=================================='''
                population,population_timing = initial_population(population_size,roots,min_test,work_hours,donation_data,spt,total_distance)
        # '''============================|initial fitness|=================================='''
                fitness = []
                for i in range(len(population)): 
                    fitness.append(multi_healthy_collected_plt(population[i],population_timing[i],donation_data,spt,total_distance))    
################### ok
                f_min, f_max = min(fitness),max(fitness)
                p_max = population_size   
                best_sol = population[np.argmax(fitness)]
                best_sol_time = population_timing [np.argmax(fitness)]

                # seed_num = []
                # for i in range(len(population)):
                #     seed_num.append(int(seed_min + (seed_max - seed_min)*(fitness[i]-f_min)/(f_max-f_min)))
################## تعداد سید ها چقد تولیدمثل کنن معلوم شد
                for iter in range(MaxIter):#############
                    for j in range(p_max):
                    
                        temp_pop,temp_time = dc(new_solution(population[i],total_distance,spt, iter, MaxIter, sigma_initial,sigma_final,roots, best_sol))
                        population.append(temp_pop)
                        population_timing.append(temp_time)
                        fitness.append(multi_healthy_collected_plt(temp_pop,temp_time,donation_data,spt,total_distance)) 
################### سید های جدید به جمعیت اولیه افزوده شد
                    fitness,population,population_timing = dc(weak_del (fitness,population,population_timing,p_max))

################### بهترین جواب آپدیت شد
                    ele = dc(np.argmax(fitness))
                    best_sol = population[ele]
                    best_sol_time = population_timing [ele]
################### ضعیف ها حذف شدند
                    for k in range(len(population)):
                        if len(population[k]) != len(population_timing[k]):
                            print('this is the mistake')
        # '''============================|result|=================================='''
                maxpos = fitness.index(max(fitness)) 
                best_result.append(fitness[maxpos])
                best_sequence.append(population[maxpos])
                best_time.append(population_timing[maxpos])
                donation_data = updating_donation_list (best_sequence[-1],best_time[-1],donation_data,total_distance)
            end = process_time()
            # print(f'instance #{prb+1} CP is {sum(best_result)} in {end - start} seconds')
            print(f'{spt},{prb+1},{sum(best_result)},{end - start}')

            print()
            time_conflict(best_sequence,best_time,total_distance)
            CP_check(donation_data,best_result)
            counting_check (best_sequence,best_time,backup_donation_data,total_distance,spt,best_result)
