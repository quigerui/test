# 导入pandas库，用于数据处理
import pandas as pd
# 导入numpy库，用于数值计算
import numpy as np
# 导入copy库，用于复制数据
import copy
# 导入math库，用于数学计算
import math
# 导入time库，用于时间计算
import time
# 导入tqdm库，用于进度条显示
from tqdm import tqdm
# 导入seaborn库，用于数据可视化
import seaborn as sns
# 导入matplotlib.pyplot库，用于数据可视化
import matplotlib.pyplot as plt

# Seaborn styles
sns.set()

def read_file(filename):
    # 读取文件
    path = 'Data/'+filename#文件路径
    f = open(path).read().splitlines()#读取文件内容

    # 初始化变量
    for i,line in enumerate(f):
        line = line.split()#按空格分割
        if i == 4:#读取机器数和工件数
            j = int(line[0])#机器数
            m = int(line[1])#工件数
            p_t = np.zeros((j,m))#初始化加工时间
            m_seq = np.zeros((j,m), dtype = np.int32)#初始化操作序列
        elif i > 4:
            # 读取操作序列和加工时间
            for k in range(len(line)):#遍历每一行
                if k % 2 == 0:#如果k是偶数
                    m_seq[i-5,int(k/2)] = int(line[k])#将line[k]赋值给m_seq[i-5,int(k/2)]
                elif k % 2 == 1:
                    # 将line[k]的值赋给p_t[i-5,int(k/2)]，即p_t的第i-5行，第int(k/2)列
                    p_t[i-5,int(k/2)] = int(line[k])
    
    # 返回j，m，p_t，m_seq的整数值
    return int(j), int(m), p_t, m_seq

def compute_makespan(chromosome, p_t, m_seq):
    # 计算最大完工时间
    op_count = np.zeros(p_t.shape[0], dtype = np.int32)  # 初始化操作计数器
    j_time = np.zeros(p_t.shape[0])  # 初始化机器j的完成时间
    m_time = np.zeros(p_t.shape[1])  # 初始化工件m的完成时间

    for j in chromosome:
        # 计算工件j的完成时间
        completion_t = max(j_time[j], m_time[m_seq[j,op_count[j]]]) + p_t[j,op_count[j]]
        j_time[j] = completion_t  # 更新工件j的完成时间
        m_time[m_seq[j,op_count[j]]] = completion_t  # 更新工件m的完成时间
        op_count[j] += 1  # 更新操作计数器

    makespan = max(j_time)  # 计算最大完工时间

    return makespan  # 返回最大完工时间
            
def generate_init_pop(population_size, j, m):
    # 生成初始种群
    population_list = np.zeros((population_size, int(j*m)), dtype = np.int32)#初始化种群列表
    chromosome = np.zeros(j*m)#初始化染色体
    start = 0#初始化开始位置
    for i in range(j):#遍历工件
        # 将i赋值给chromosome的start到start+m的元素
        chromosome[start:start+m] = i
        start += m#更新开始位置

    for i in range(population_size):
        # 随机打乱chromosome的元素顺序
        np.random.shuffle(chromosome)
        # 将打乱后的chromosome赋值给population_list的第i行
        population_list[i] = chromosome
        # 返回种群列表
    return population_list

def two_point_crossover(populationlist, crossover_rate):
    # 两点交叉
    parentlist = copy.deepcopy(populationlist)#复制种群列表
    childlist = copy.deepcopy(populationlist)#复制种群列表
    for i in range(len(parentlist),2):#遍历种群列表
        sample_prob=np.random.rand()
        # 随机生成一个概率，如果小于等于交叉率，则进行交叉
        if sample_prob <= crossover_rate:
            # 随机选择两个交叉点
            cutpoint = np.random.choice(2, parentlist.shape[1], replace = False)
            cutpoint.sort()
            # 选择两个父代
            parent_1 = parentlist[i]
            parent_2 = parentlist[i+1]
            # 复制两个父代
            child_1 = copy.deepcopy(parent_1)
            child_2 = copy.deepcopy(parent_2)
            # 交换两个父代的交叉点之间的基因
            child_1[cutpoint[0]:cutpoint[1]] = parent_2[cutpoint[0]:cutpoint[1]]
            child_2[cutpoint[0]:cutpoint[1]] = parent_1[cutpoint[0]:cutpoint[1]]
            # 将两个子代添加到子代列表中
            childlist[i] = child_1
            childlist[i+1] = child_2
    
    return parentlist, childlist

def job_order_crossover(populationlist, j, crossover_rate):
    # 工序交叉
    parentlist = copy.deepcopy(populationlist)
    childlist = copy.deepcopy(populationlist)
    for i in range(len(parentlist),2):
        sample_prob=np.random.rand()
        # 随机生成一个概率，如果小于交叉率，则进行交叉
        if sample_prob <= crossover_rate:
            # 随机选择两个父代
            parent_id = np.random.choice(len(populationlist), 2, replace=False)
            # 随机选择一个工序
            select_job = np.random.choice(j, 1, replace=False)[0]
            # 进行交叉操作，生成两个子代
            child_1 = job_order_implementation(parentlist[parent_id[0]], parentlist[parent_id[1]], select_job)
            child_2 = job_order_implementation(parentlist[parent_id[1]], parentlist[parent_id[0]], select_job)
            # 将子代添加到子代列表中
            childlist[i] = child_1
            childlist[i+1] = child_2

    return parentlist, childlist

def job_order_implementation(parent1, parent2, select_job):
    # 工序交叉实现
    other_job_order = []
    child = np.zeros(len(parent1))
    # 遍历parent2，将不等于select_job的元素添加到other_job_order中
    for j in parent2:
        if j != select_job:
            other_job_order.append(j)
    k = 0
    # 遍历parent1，如果元素等于select_job，则将child对应位置赋值为select_job，否则将other_job_order中的元素赋值给child对应位置
    for i,j in enumerate(parent1):
        if j == select_job:
            child[i] = j
        else:
            child[i] = other_job_order[k]
            k += 1
    
    return child

def repair(chromosome, j, m):
    # 修复染色体
    job_count = np.zeros(j)
    # 初始化一个长度为j的数组，用于记录每个任务的个数
    for j in chromosome:
        job_count[j] += 1
    
    # 遍历染色体，将每个任务的个数记录到job_count数组中
    job_count = job_count - m

    # 将job_count数组中的每个元素减去m，得到每个任务的实际个数
    much_less = [[],[]]
    is_legall = True
    # 初始化两个空列表，用于记录个数大于0和小于0的任务
    for j,count in enumerate(job_count):
        # 遍历job_count数组，将个数大于0和小于0的任务分别记录到much_less列表中
        if count > 0:
            is_legall = False
            much_less[0].append(j)
        elif count < 0:
            is_legall = False
            much_less[1].append(j)

    if is_legall == False:
        # 如果存在个数大于0或小于0的任务，则进行修复
        for m in much_less[0]:
            # 遍历个数大于0的任务
            for j in range(len(chromosome)):
                # 遍历染色体
                if chromosome[j] == m:
                    # 如果当前任务等于个数大于0的任务
                    less_id = np.random.choice(len(much_less[1]),1)[0]
                    # 随机选择一个个数小于0的任务
                    chromosome[j] = much_less[1][less_id]
                    # 将当前任务替换为个数小于0的任务
                    job_count[m] -= 1
                    # 将个数大于0的任务的个数减1
                    job_count[much_less[1][less_id]] += 1

                    # 将个数小于0的任务的个数加1
                    if job_count[much_less[1][less_id]] == 0:
                        # 如果个数小于0的任务的个数变为0，则将其从much_less列表中移除
                        much_less[1].remove(much_less[1][less_id])

                    if job_count[m] == 0:
                        # 如果个数大于0的任务的个数变为0，则跳出循环
                        break
    
def mutation(childlist, num_mutation_jobs, mutation_rate, p_t, m_seq):
    # 变异
    for chromosome in childlist:
        # 随机生成一个概率
        sample_prob = np.random.rand()
        # 如果概率小于等于变异率
        if sample_prob <= mutation_rate:
            # 随机选择变异点
            mutationpoints = np.random.choice(len(chromosome), num_mutation_jobs, replace = False)
            # 深度拷贝染色体
            chrom_copy = copy.deepcopy(chromosome)
            # 遍历变异点
            for i in range(len(mutationpoints)-1):
                # 将变异点后的基因替换为变异点前的基因
                chromosome[mutationpoints[i+1]] = chrom_copy[mutationpoints[i]]

            # 将变异点前的基因替换为变异点后的基因
            chromosome[mutationpoints[0]] = chrom_copy[mutationpoints[-1]]
    """
    makespan_list = np.zeros(len(childlist))
    for i,chromosome in enumerate(childlist):
        makespan_list[i] = compute_makespan(chromosome, p_t, m_seq)
    
    num_all_mut = int(0.1*len(childlist))
    zipped = list(zip(makespan_list, np.arange(len(makespan_list))))
    sorted_zipped = sorted(zipped, key=lambda x: x[0])
    zipped = zip(*sorted_zipped)
    partial_mut_id = np.asarray(list(zipped)[1])[:-num_all_mut]
    all_mut = generate_init_pop(num_all_mut, p_t.shape[0], p_t.shape[1])
    childlist = np.concatenate((all_mut,copy.deepcopy(childlist)[partial_mut_id]), axis = 0)
    """
    
def selection(populationlist, makespan_list):
    # 选择
    num_self_select = int(0.2*len(populationlist)/2)  # 计算自选个体数量
    num_roulette_wheel = int(len(populationlist)/2) - num_self_select  # 计算轮盘赌选择个体数量
    zipped = list(zip(makespan_list, np.arange(len(makespan_list))))  # 将makespan_list和索引打包成元组
    sorted_zipped = sorted(zipped, key=lambda x: x[0])  # 按makespan_list的值进行排序
    zipped = zip(*sorted_zipped)  # 将排序后的元组解包
    self_select_id = np.asarray(list(zipped)[1])[:num_self_select]  # 取出前num_self_select个索引作为自选个体索引
    
    makespan_list = 1/makespan_list  # 将makespan_list取倒数
    selection_prob = makespan_list/sum(makespan_list)  # 计算选择概率
    roulette_wheel_id = np.random.choice(len(populationlist), size = num_roulette_wheel, p = selection_prob)  # 根据选择概率随机选择个体索引
    new_population = np.concatenate((copy.deepcopy(populationlist)[self_select_id],copy.deepcopy(populationlist)[roulette_wheel_id]), axis=0)  # 将自选个体和轮盘赌选择的个体合并成新种群
    
    return new_population  # 返回新种群

def binary_selection(populationlist, makespan_list):
    # 二元选择
    new_population = np.zeros((int(len(populationlist)/2), populationlist.shape[1]), dtype = np.int32)
    
    # 计算自选择和二元选择的数量
    num_self_select = int(0.1*len(populationlist)/2)
    num_binary = int(len(populationlist)/2) - num_self_select
    # 将makespan_list和对应的索引打包成元组，并按makespan_list排序
    zipped = list(zip(makespan_list, np.arange(len(makespan_list))))
    sorted_zipped = sorted(zipped, key=lambda x: x[0])
    zipped = zip(*sorted_zipped)
    # 取出前num_self_select个索引
    self_select_id = np.asarray(list(zipped)[1])[:num_self_select]
    
    # 进行二元选择
    for i in range(num_binary):
        select_id = np.random.choice(len(makespan_list), 2, replace=False)
        if makespan_list[select_id[0]] < makespan_list[select_id[1]]:
            new_population[i] = populationlist[select_id[0]]
        else:
            new_population[i] = populationlist[select_id[1]]
    
    # 将自选择的个体复制到新种群中
    new_population[-num_self_select:] = copy.deepcopy(populationlist)[self_select_id]
    
    return new_population

def get_critical_path(chromosome,p_t, m_seq):
    # 获取关键路径
    critical_path = []
    start_t = np.zeros(len(chromosome))
    end_t = np.zeros(len(chromosome))

    op_count = np.zeros(p_t.shape[0], dtype = np.int32)
    j_time = np.zeros(p_t.shape[0])
    m_time = np.zeros(p_t.shape[1])

    # 遍历染色体中的每个基因
    for i,j in enumerate(chromosome):
        # 计算当前基因的完成时间
        completion_t = max(j_time[j], m_time[m_seq[j,op_count[j]]]) + p_t[j,op_count[j]]
        # 记录当前基因的开始时间
        start_t[i] = max(j_time[j], m_time[m_seq[j,op_count[j]]])
        # 记录当前基因的结束时间
        end_t[i] = completion_t
        # 更新当前基因的加工时间
        j_time[j] = completion_t
        # 更新当前机器的加工时间
        m_time[m_seq[j,op_count[j]]] = completion_t
        # 更新当前基因的操作计数
        op_count[j] += 1

    # 计算最大完工时间
    makespan = max(j_time)
    last_end_t = makespan
    # 从后往前遍历染色体中的每个基因，找到关键路径
    for i in range(len(chromosome) - 1, -1, -1):
        if end_t[i] == last_end_t:
            critical_path.insert(0, i)
            last_end_t = start_t[i]

    return critical_path
    
def get_neighbors(chromosome, indices):
    # 定义一个空列表，用于存储邻居
    neighbors = []
    # 遍历indices列表，除了最后一个元素
    for i in range(len(indices)-1):
        # 复制chromosome列表
        neighbor = chromosome[:]
        # 交换neighbor列表中indices[i]和indices[i+1]位置的元素
        neighbor[indices[i]], neighbor[indices[i+1]] = neighbor[indices[i+1]], neighbor[indices[i]]
        # 如果neighbor列表中indices[i]和indices[i+1]位置的元素不相等，则将neighbor添加到neighbors列表中
        if neighbor[indices[i]] != neighbor[indices[i]+1]:
            neighbors.append(neighbor)

    # 返回neighbors列表
    return neighbors

def tabu_search(population_list, makespan_list, p_t, m_seq, max_iterations, tabu_tenure):
    # 初始化禁忌列表
    tabu_list = []

    # 遍历种群列表
    for i, chromosome in enumerate(population_list):
        # 当前解
        current_solution = chromosome
        # 当前解的Makespan
        best_makespan = makespan_list[i]
        # 当前解
        best_solution = chromosome
        # 无改进次数
        unimprovement = 0
        # 迭代次数
        for iteration in range(max_iterations):
            # 获取当前解的关键路径
            critical_path = get_critical_path(current_solution, p_t, m_seq)
            # 获取当前解的邻居解
            neighbors = get_neighbors(current_solution, critical_path)
            # 最佳邻居解
            best_neighbor = None
            # 最佳邻居解的Makespan
            best_neighbor_makespan = float('inf')
            
            # 遍历邻居解
            for neighbor in neighbors:
                # 如果邻居解不在禁忌列表中
                if not any(np.array_equal(neighbor, tabu) for tabu in tabu_list):
                    # 计算邻居解的Makespan
                    neighbor_makespan = compute_makespan(neighbor, p_t, m_seq)
                    # 如果邻居解的Makespan小于最佳邻居解的Makespan
                    if neighbor_makespan < best_neighbor_makespan:
                        # 更新最佳邻居解
                        best_neighbor = neighbor
                        best_neighbor_makespan = neighbor_makespan
                        
            
            # 如果存在最佳邻居解
            if best_neighbor is not None:
                # 更新当前解
                current_solution = best_neighbor
                current_makespan = best_neighbor_makespan
              
                # 如果当前解的makespan小于最佳解的makespan，则更新最佳解
                if current_makespan < best_makespan:
                    best_solution = current_solution
                    best_makespan = current_makespan
                    population_list[i] = best_solution
                    makespan_list[i] = best_makespan
                else:
                    # 否则，未改进次数加1
                    unimprovement += 1

                # 如果未改进次数大于20，则跳出循环
                if unimprovement > 20:
                    break
                
                # 将当前解加入禁忌表
                tabu_list.append(current_solution)
                # 如果禁忌表长度大于禁忌期限，则删除最早加入的解
                if len(tabu_list) > tabu_tenure:
                    tabu_list.pop(0)  
            else:
                # 如果没有找到更好的邻居，则跳出循环
                break 

    # 返回种群列表和最大完工时间列表
    return  population_list, makespan_list

def draw_bar_plot(filename, listx):
    # 读取csv文件
    df = pd.read_csv(filename)
    

    # 设置柱状图的宽度
    width = 0.25
    # 计算x轴的坐标
    listx1 = [x - (width / 2) for x in range(len(listx))]  
    listx2 = [x + (width / 2) for x in range(len(listx))]  
  
    # 获取y轴的数据
    listy1 = df["GA"]
    listy2 = df["Optimal"]

    # 绘制柱状图
    plt.bar(listx1, listy1, width, label="GA")
    plt.bar(listx2, listy2, width, label="Opt")
    # 设置x轴的刻度标签
    plt.xticks(range(len(listx)), labels=listx)
    # 添加图例
    plt.legend()
    # 设置标题
    plt.title("Makespan Comparison on OR-Library Instances")
    # 设置y轴标签
    plt.ylabel("Makesapn")
    # 设置x轴标签
    plt.xlabel("15*15 OR Library Instances")
    # 保存图片
    plt.savefig("Makespan Comparison_on OR-Library Instances Bar Chart")
    # 显示图片
    plt.show()

def draw_gantt_chart(chromosome, p_t, m_seq, title):
    # 初始化甘特图数据
    gantt_data = []
    # 初始化每个作业的开始时间
    start_t = np.zeros(p_t.shape[0])  
    # 初始化每个机器的时间
    m_time = np.zeros(p_t.shape[1])   
    # 初始化每个作业的完成时间
    j_time = np.zeros(p_t.shape[0])   
    # 初始化每个作业的操作计数
    op_count = np.zeros(p_t.shape[0], dtype=np.int32)  

   
    # 遍历染色体中的每个基因
    for i, gene in enumerate(chromosome):
        # 获取当前作业
        job = gene  
        # 获取当前作业的机器
        machine = m_seq[job, op_count[job]]  

        
        # 计算当前作业的开始时间
        start_time = max(j_time[job], m_time[machine])
        # 计算当前作业的完成时间
        completion_time = start_time + p_t[job, op_count[job]]

       
        # 更新当前作业的完成时间
        j_time[job] = completion_time
        # 更新当前机器的时间
        m_time[machine] = completion_time

        
        # 将当前作业的信息添加到甘特图数据中
        gantt_data.append((f'Job {job}', start_time, completion_time, f'Machine {machine+1}'))
        # 更新当前作业的操作计数
        op_count[job] += 1

   
    # 将甘特图数据转换为DataFrame
    df = pd.DataFrame(gantt_data, columns=['Job', 'Start', 'Finish', 'Machine'])

    # 将机器信息转换为机器ID
    df['Machine_ID'] = df['Machine'].apply(lambda x: int(x.split(' ')[1]))  
    # 按照机器ID排序
    df = df.sort_values(by='Machine_ID')  

   
    # 创建甘特图
    fig, ax = plt.subplots(figsize=(10, 6))

    # 遍历DataFrame中的每一行
    for i, row in df.iterrows():
        # 绘制甘特图中的条形
        ax.barh(row['Machine'], row['Finish'] - row['Start'], left=row['Start'], height=0.4, label=row['Job'])

   
    # 设置x轴和y轴标签
    ax.set_xlabel('Time')
    ax.set_ylabel('Machine')
    # 设置标题
    ax.set_title(title)

    # 保存甘特图
    plt.savefig("GanttPlot/"+title)

if __name__ == "__main__":
    # 读取Excel文件中的数据
    df = pd.read_excel("./Data/lower_bounds.xlsx", sheet_name="orb")
    time_list = []
    ratio_list = []
    outDf = pd.DataFrame()
    # 遍历1到10的数字
    for index in range(1,11):
        # 根据数字生成实例名称
        instance_name = "orb0"+str(index)
        if(index >= 10):instance_name = "orb"+str(index)
        # 读取文件中的数据
        j, m, p_t, m_seq = read_file(instance_name)
        # 设置种群大小
        population_size = 200
        # 生成初始种群
        population_list = generate_init_pop(population_size, j ,m)
        # 设置交叉率和变异率
        crossover_rate = 0.8
        mutation_rate = 0.1
        mutation_selection_rate = 0.1
        # 计算变异作业数量
        num_mutation_jobs=round(j*m*mutation_selection_rate)
        # 设置迭代次数
        num_iteration = 100 #可改次数
        min_makespan_record = []#记录最小makespan
        avg_makespan_record = []#记录平均makespan
        min_makespan = 9999999#初始化最小makespan
        

        # 记录开始时间
        begint = time.time()
        # 迭代
        for i in tqdm(range(num_iteration)):
            # 交叉
            parentlist, childlist = job_order_crossover(population_list, j, crossover_rate)
            # 变异
            mutation(childlist, num_mutation_jobs, mutation_rate, p_t, m_seq)
            # 合并父代和子代
            population_list = np.concatenate((parentlist, childlist), axis=0)
            # 计算makespan
            makespan_list = np.zeros(len(population_list))
            for k in range(len(population_list)):
                makespan_list[k] = compute_makespan(population_list[k], p_t, m_seq)
                # 更新最小makespan
                if makespan_list[k] < min_makespan:
                    min_m = makespan_list[k]
                    min_makespan = makespan_list[k]
                    min_c = population_list[k]
   
            # 选择
            population_list = binary_selection(population_list, makespan_list)
            # 禁忌搜索
            population_list, makespan_list = tabu_search(population_list, makespan_list, p_t, m_seq, 1000, 9)
            # 更新最小makespan
            min_makespan = min(makespan_list)
            min_makespan_record.append(min_makespan)
            avg_makespan_record.append(np.average(makespan_list))
        
        # 计算时间消耗
        time_consume = time.time() - begint
        # 计算比率
        ratio = (min_makespan - df["lower bound"][index-1])/df["lower bound"][index-1]
        time_list.append(time)
        ratio_list.append(ratio)
        # 打印最小makespan和下界
        print(min_makespan, " ", df["lower bound"][index-1])
        # 绘制甘特图
        draw_gantt_chart(min_c, p_t, m_seq, "Gantt Chart for GA on orb" + str(index)+ " instance")
        # 将结果保存到数据框中
        row_input = pd.DataFrame([[min_makespan, df["lower bound"][index-1], time_consume]], columns = ["GA","Optimal","time(sec)"])
        outDf = pd.concat([outDf, row_input], ignore_index=True)

    # 将结果数据框保存为csv文件
    outDf.to_csv("Result.csv")
    # 绘制柱状图，横轴为orb1到orb10，纵轴为对应的值
    draw_bar_plot("Result.csv", ["orb"+str(i+1) for i in range(1, 11)])

    
    

        



        





    
