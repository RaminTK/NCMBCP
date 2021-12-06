import numpy as np
from copy import deepcopy as dc
from time import process_time
from tqdm import tqdm
import random
from math import sqrt, exp, floor


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

def SA_random_start (multipleRout,multipleTime,total_distance,donation_data):
    for i in range(10):
        selected_rout = dc(random.randrange(len(multipleTime)))
        temp_time = dc(multipleTime)
        if selected_rout == 0 :
            # print(multipleTime)
            if len(multipleTime) == 1:
                V_min = 0
                V_max = dc(work_hours  - duration(multipleRout[selected_rout],total_distance) - 1)
            else:             
                V_min = 0
                V_max = dc(multipleTime[selected_rout+1] - duration(multipleRout[selected_rout],total_distance) - 1)
            # print('1',V_min, V_max,multipleTime[selected_rout])

        elif selected_rout == len(multipleTime)-1:
            V_min = dc(multipleTime[selected_rout-1]+duration(multipleRout[selected_rout - 1],total_distance))
            V_max = dc(1500+spt - duration(multipleRout[selected_rout],total_distance) - 1)
            # print('2',V_min, V_max,multipleTime[selected_rout])
        else:
            V_min = dc(multipleTime[selected_rout-1]+duration(multipleRout[selected_rout-1],total_distance))
            V_max = dc(multipleTime[selected_rout+1]  - duration(multipleRout[selected_rout],total_distance) - 1)
            # print('3',V_min, V_max,multipleTime[selected_rout])
        if V_min > V_max :
            continue

        temperature = 10
        flag = 1
        while temperature > 1 :
            if flag == 1 :
                new_time = dc(multipleTime[selected_rout] + int(0.1 *(V_max - multipleTime[selected_rout])))
                temp_time [selected_rout] = dc(new_time)

                new_cost = dc(multi_healthy_collected_plt(multipleRout,temp_time,donation_data,spt,total_distance))
                old_cost = dc(multi_healthy_collected_plt(multipleRout,multipleTime,donation_data,spt,total_distance))
                if new_cost >= old_cost and new_time <= V_max:
                    multipleTime = dc(temp_time)
                # elif exp((new_cost - old_cost)/temperature) >= random.uniform(0.9, 1):
                #     multipleTime = dc(temp_time)
                else:
                    flag = 0
            else:
                new_time = dc(multipleTime[selected_rout] - int(0.1 *(multipleTime[selected_rout] - V_min)))
                temp_time [selected_rout] = dc(new_time)
                new_cost = dc(multi_healthy_collected_plt(multipleRout,temp_time,donation_data,spt,total_distance))
                old_cost = dc(multi_healthy_collected_plt(multipleRout,multipleTime,donation_data,spt,total_distance))
                if new_cost >= old_cost and new_time >= V_min:
                    multipleTime = dc(temp_time)
                # elif exp((new_cost - old_cost)/temperature) >= random.uniform(0.9, 1):
                #     multipleTime = dc(temp_time)
                else:
                    flag = 1
            temperature -= 0.1
    return multipleTime

# def SA_random_start (multipleRout,multipleTime,total_distance,donation_data):
#     for i in range(5):
#         selected_rout = dc(random.randrange(len(multipleTime)))
#         temp_time = dc(multipleTime)
#         if selected_rout == 0 :
#             V_min = 0
#             V_max = dc(multipleTime[selected_rout+1]-1)
#         elif selected_rout == len(multipleTime)-1:
#             V_min = dc(multipleTime[selected_rout-1]+duration(multipleRout[selected_rout],total_distance)+1)
#             V_max = dc(1500+spt)
#         else:
#             V_min = dc(multipleTime[selected_rout-1]+duration(multipleRout[selected_rout],total_distance))
#             V_max = dc(multipleTime[selected_rout+1]-1)

#         temperature = 10
#         flag = 1
#         while temperature > 1 :
#             if flag == 1 :
#                 new_time = dc(multipleTime[selected_rout] + int(0.1 *(V_max - multipleTime[selected_rout])))
#                 if new_time > V_max:
#                     continue
#                 temp_time [selected_rout] = dc(new_time)
#                 new_cost = dc(multi_healthy_collected_plt(multipleRout,temp_time,donation_data,spt,total_distance))
#                 old_cost = dc(multi_healthy_collected_plt(multipleRout,multipleTime,donation_data,spt,total_distance))
#                 if new_cost >= old_cost and new_time <= V_max:
#                     multipleTime = dc(temp_time)
#                 # elif exp((new_cost - old_cost)/temperature) >= random.uniform(0.9, 1):
#                 #     multipleTime = dc(temp_time)
#                 else:
#                     flag = 0
#             else:
#                 new_time = dc(multipleTime[selected_rout] + int(0.1 *(multipleTime[selected_rout] - V_min)))
#                 if new_time < V_min:
#                     continue
#                 temp_time [selected_rout] = dc(new_time)
#                 new_cost = dc(multi_healthy_collected_plt(multipleRout,temp_time,donation_data,spt,total_distance))
#                 old_cost = dc(multi_healthy_collected_plt(multipleRout,multipleTime,donation_data,spt,total_distance))
#                 if new_cost >= old_cost and new_time >= V_min:
#                     multipleTime = dc(temp_time)
#                 # elif exp((new_cost - old_cost)/temperature) >= random.uniform(0.9, 1):
#                 #     multipleTime = dc(temp_time)
#                 else:
#                     flag = 1
#             temperature -= 0.1
#     return multipleTime

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
    
    def mutation(fitness,population,population_timing,roots,work_hours,donation_data,spt,total_distance,rate_of_mutation):
    fit = np.array(fitness)
    for ele in sorted(fit.argsort()[:rate_of_mutation], reverse = True):  
        del population[ele]
        del population_timing[ele]
    for i in range(rate_of_mutation):
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

def crossover (parent1,parent2,population,population_timing,donation_data,spt,total_distance):
    if len(parent1[0]) > 1 :  
        cross_point_1 = np.random.randint(1,len(parent1[0]))
    else:
        cross_point_1 = 0
    if len(parent2[0]) > 1 :  
        cross_point_2 = np.random.randint(1,len(parent2[0]))
    else:
        cross_point_2 = 0
    child1 = parent1[0][:cross_point_1] + parent2[0][cross_point_2:]
    child2 = parent2[0][:cross_point_2] + parent1[0][cross_point_1:]
    timing_child1 = improved_st (child1,initial_timing(child1,total_distance),donation_data,spt,total_distance)
    timing_child2 = improved_st (child2,initial_timing(child2,total_distance),donation_data,spt,total_distance)
    population.append(child1)
    population.append(child2)
    population_timing.append(timing_child1)
    population_timing.append(timing_child2)
    fitness = []
    for i in range(len(population)): 
        fitness.append(multi_healthy_collected_plt(population[i],population_timing[i],donation_data,spt,total_distance))
    fit = np.array(fitness)
    for ele in sorted(fit.argsort()[:2], reverse = True):  
        del population[ele]
        del population_timing[ele]
    return population,population_timing

def selection(fitness, population, population_timing):
    fit =dc(fitness)
    total = sum(fit)
    fitness_probability = fit
    if total != 0 :
        for i in range(len(fit)-1):
            fitness_probability[i+1] = fit[i]+fit[i+1]
        fitness_probability = [i/total for i in fitness_probability]
    else:
        fitness_probability = [1/len(fitness_probability) for i in fitness_probability]
    rand = dc(np.random.uniform(0,1))
    counter = 0
    for prob in fitness_probability :
        if rand > prob:
            counter +=1
    parent1 = [population[counter],population_timing[counter]]
    rand = dc(np.random.uniform(0,1))
    counter = 0
    for prob in fitness_probability :
        if rand > prob:
            counter +=1
    parent2 = [population[counter],population_timing[counter]]
    return parent1,parent2
    
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
            
            
# generation_limit = 1000
mutation_rate = 0.2
crossover_rate = 0.1
rate_of_mutation = 10


dons = ('don1.txt','don2.txt','don3.txt','don4.txt','don5.txt','don6.1.txt','don7.txt')

locs = ('loc1.1.txt','loc2.1.txt','loc3.1.txt','loc4.1.txt','loc5.1.txt','loc6.1.txt','loc7.1.txt')

vehicle_num = (6,6,7,8,9,10,10)

spoil_time = (300,500,700)
# spoil_time = ([300])#,500,700)


for i in range(10):                                                              #################### change ####################### 10-1
    for spt in spoil_time :
        the_size = spt*2000
        for prb in range(7):                                                
            start = process_time()
            donation_da = Read_donation(dons[prb]) 
            donation_data = np.array([np.array(xi) for xi in donation_da])
            location_data = Read_location(locs[prb])  
            total_distance = np.array([np.array(xi) for xi in location_data])
            backup_donation_data = dc(donation_data)
            roots = chrom_pool (donation_data,spt,the_size,total_distance)
            generation_limit = int(136.9*np.log(len(roots)))          

            population_size = 50 #int(10 * np.ceil(np.log(len(roots))))
            work_hours = 1500+spt
            min_test = dc(minTest(roots,total_distance))
            best_result = []
            best_sequence = []
            best_time = []
            # vehicle_history = []
            for vehicle in range(vehicle_num[prb]):
        # '''============================|initial population|=================================='''
                population,population_timing = initial_population(population_size,roots,min_test,work_hours,donation_data,spt,total_distance)
                time_conflict(population,population_timing,total_distance)

        # '''============================|initial fitness|=================================='''
                fitness = []
                for i in range(len(population)): 
                    fitness.append(multi_healthy_collected_plt(population[i],population_timing[i],donation_data,spt,total_distance))    
                
        # '''============================|evolution|=================================='''
                # result_history = []
                for i in range(generation_limit):
                # for i in tqdm(list(range(generation_limit))):
                    rand_num = dc(np.random.uniform(0,1))

        # '''============================|mutation|=================================='''
                    if rand_num < mutation_rate : 
                        population,population_timing = mutation(fitness,population,population_timing,roots,work_hours,donation_data,spt,total_distance,rate_of_mutation)
  
        # '''============================|selection|=================================='''
                    else :
                        parent1,parent2 = selection(fitness,population,population_timing)

        # '''============================|crossover|=================================='''
                        population,population_timing = crossover (parent1,parent2,population,population_timing,donation_data,spt,total_distance)

        # '''============================|new fitness|=================================='''
                    fitness = []
                    for i in range(len(population)): 
                        fitness.append(multi_healthy_collected_plt(population[i],population_timing[i],donation_data,spt,total_distance))

        # '''============================|result|=================================='''
                # for i in range(len(population)):
                #     population_timing[i] = SA_random_start (population[i],population_timing[i],total_distance, donation_data)
                maxpos = fitness.index(max(fitness)) 
                best_result.append(fitness[maxpos])
                best_sequence.append(population[maxpos])
                best_time.append(population_timing[maxpos])

                # print(best_time[-1])
                # before = dc(multi_healthy_collected_plt(best_sequence[-1],best_time[-1],donation_data,spt,total_distance))

                best_time[-1] = dc((best_sequence[-1],best_time[-1],total_distance, donation_data))
                # print(best_time[-1])
                # after = dc(multi_healthy_collected_plt(best_sequence[-1],best_time[-1],donation_data,spt,total_distance))
                # if before != after:
                    # print('+', after - before)
                best_result[-1] = dc(multi_healthy_collected_plt(best_sequence[-1],best_time[-1],donation_data,spt,total_distance))
                donation_data = updating_donation_list (best_sequence[-1],best_time[-1],donation_data,total_distance)
            end = process_time()
            print(f'instance #{prb+1} CP is {sum(best_result)} in {end - start} seconds')
            print()

            for i in range(vehicle_num[prb]):
              print(i)
              print(best_sequence[i])
              print(best_time[i])
              print(best_result[i])
              print()
            time_conflict(best_sequence,best_time,total_distance)
            CP_check(donation_data,best_result)
            counting_check (best_sequence,best_time,backup_donation_data,total_distance,spt,best_result)
