from agent import *
from message import *
import copy
import random
import numpy as np
import matplotlib.pyplot as plt

agent_num = 30
dsa_num_iterations = 100
mgm_num_iterations = int(dsa_num_iterations / 2)


# generates agent_num=30 agents
def generate_agents():
    agents = []
    for i in range(agent_num):
        rand_number = random.randint(0, 9)
        agent = Agent(i, rand_number)  # sets random assignment
        agents.append(agent)
    return agents


# generates for each neighbors matrix that contains random values from 0-10
def create_constraints_table():
    return np.random.randint(10, size=(10, 10))


# calculate for each connection the assignment cost value
def assign_value_to_key(agents):
    for agent in agents:
        assign_val = 0
        for key, value in agent.get_neighbors().items():
            assign_val += value[agent.get_curr_assig(), agents[key].get_curr_assig()]
        agent.set_assignment_val(assign_val)


# generates neighbors for each agent if random value is lower than requested
def initiate_neighborhood(agents, prob):
    for i in range(agent_num):
        for j in range(i + 1, agent_num):
            rand = random.random()
            if rand < prob:
                temp_table = create_constraints_table() # create constraints table for each connection
                agents[i].set_neighbor(j, temp_table)
                agents[j].set_neighbor(i, np.transpose(temp_table))  # we inserted into the agent that we've create connection with the same table in transpose
    assign_value_to_key(agents)


# if the probability is less then a random number -  were checking what is the best assignment by summing by rows in
# all neighbors columns - and choosong the minimum value per that iteration
def optimal_assig_iter(agent, agent_curr_mess, prob_to_change_assig):
    dict_values = dict([(x, 0) for x in range(10)])
    current_cost = 0
    for message in agent_curr_mess:
        curr_matrix = agent.get_neighbors()[message.get_from_who()][:, message.get_value()]
        current_cost = current_cost + curr_matrix[agent.get_curr_assig()]  # we want to add current assign if there is change in the asign
        for index in range(len(curr_matrix)):
            dict_values[index] = dict_values[index] + curr_matrix[index]  # adding to current key the assign with neighbors
    if random.random() < prob_to_change_assig:
        key_with_min_value_in_dict = min(dict_values, key=dict_values.get)  # returns related key of minimum value
        agent.set_curr_assig(key_with_min_value_in_dict)
        min_value = dict_values[key_with_min_value_in_dict]
        return min_value  # calculates minimum value and return it
    return current_cost


def dsa_algorithm(agents, prob_to_change_assig):  # prob_to_change_assig = 0.7 OR 0.4
    cost_per_iteration_list = []  # list that contains 100 costs each per iteration
    for i in range(dsa_num_iterations):
        curr_total_cost = 0
        messages_bucket = []
        for agent in agents:  # we generate here messages with current assign
            for key in agent.get_neighbors().keys():
                messages_bucket.append(Message(agent, agents[key], agent.get_curr_assig()))  # creates messages with current assignment of the agent
        for agent in agents:
            agent_curr_mess = []
            for message in messages_bucket:  # we adding each agent relevant messages for him
                if message.get_to_who() == agent.get_id():
                    agent_curr_mess.append(message)
            curr_total_cost += optimal_assig_iter(agent, agent_curr_mess, prob_to_change_assig) # incremental calculation of total cost per iteration
        cost_per_iteration_list.append(curr_total_cost)
    return cost_per_iteration_list


def calc_alternative_option(agent, agent_curr_mess):
    dict_values = dict([(x, 0) for x in range(10)])
    for message in agent_curr_mess:
        curr_matrix = agent.get_neighbors()[message.get_from_who()][:, message.get_value()]
        for index in range(len(curr_matrix)):
            dict_values[index] = dict_values[index] + curr_matrix[index]  # adding to current key the assign with neighbors
    key_with_min_value_in_dict = min(dict_values, key=dict_values.get)  # returns related key of minimum value
    agent.set_alternative_best_assin(key_with_min_value_in_dict, dict_values[key_with_min_value_in_dict])  # sets the assignment and its value


def send_agents_assign(agents):
    messages_bucket_mgm = []
    for agent in agents:  # we generate here messages with current assign
        for key in agent.get_neighbors().keys():
            messages_bucket_mgm.append(Message(agent, agents[key], agent.get_curr_assig()))  # creates messages with current assignment of the agent
    for agent in agents:
        agent_curr_mess = []
        for message in messages_bucket_mgm:  # we adding each agent relevant messages for him
            if message.get_to_who() == agent.get_id():  # if the message related to current agent
                agent_curr_mess.append(message)
        calc_alternative_option(agent, agent_curr_mess)  # calculates best alternative option


# sends r value to all agents neighbors and if it stands in condition we change the assignment
def send_r_value(agents):
    messages_bucket_r = [] # we creates bucket that contains r messages
    for agent in agents:
        for key in agent.get_neighbors().keys():
            r = agent.get_assignment_val() - agent.get_alternative_best_assin_value()  # calculates r
            messages_bucket_r.append(Message(agent, agents[key], r))  # creates messages with r value of the agent
    for agent in agents:
        found_optimal_r_val = True # flag that indicates if curr r is stands in the condition below
        r_curr_agent = agent.get_assignment_val() - agent.get_alternative_best_assin_value()  # r of current agent
        for message in messages_bucket_r:  # we adding each agent relevant messages for him
            if message.get_to_who() == agent.get_id(): # checks if the message relevant to current agent
                r_neighbor = message.get_value() # gets r value in the message
                if r_curr_agent <= r_neighbor or r_curr_agent <= 0:
                    found_optimal_r_val = False
                    break
        if found_optimal_r_val: # if the flag is true we changing current assignment
            agent.set_curr_assig(agent.get_alternative_best_assin())
            agent.set_assignment_val(agent.get_alternative_best_assin_value())


def mgm_algorithm(agents):
    cost_per_iteration_list = []  # list that contains 100 costs each per iteration
    for i in range(mgm_num_iterations):
        send_agents_assign(agents)  # sends all messages to a bucket and each agent calculates his best alternative
        send_r_value(agents)  # if the iteration is odd - so we will sent the reduced cost to all agents
        cost_per_iteration = 0
        for agent in agents:
            cost_per_iteration += agent.get_assignment_val()  # calculates total cost per iteration
        cost_per_iteration_list.append(cost_per_iteration)
        cost_per_iteration_list.append(cost_per_iteration)  # we added the same value twice so so that we will fit to x axis values of the dsa algorithm
    return cost_per_iteration_list


# generates plot - for y axis we averaging all iteration by the relevant index - for example: mean on all first
# iterations in all 10 lists
def plot_input_func(all_iterations_prob1_1, all_iterations_prob2_1, mgm_cost_list_1, all_iterations_prob1_2, all_iterations_prob2_2, mgm_cost_list_2):
    x = [i for i in range(dsa_num_iterations)]  # create x axis (by num of iterations)
    # we splited by two because we counted both neighbors cost
    y1_1 = [np.mean(k)/2 for k in zip(*all_iterations_prob1_1)]  # averaging sum of related cells in all iteration -first problem
    y2_1 = [np.mean(k)/2 for k in zip(*all_iterations_prob2_1)]  # averaging sum of related cells in all iteration -first problem
    y3_1 = [np.mean(k)/2 for k in zip(*mgm_cost_list_1)]  # averaging sum of related cells in all iteration -first problem
    y1_2 = [np.mean(k)/2 for k in zip(*all_iterations_prob1_2)]  # averaging sum of related cells in all iteration -second problem
    y2_2 = [np.mean(k)/2 for k in zip(*all_iterations_prob2_2)]  # averaging sum of related cells in all iteration -second problem
    y3_2 = [np.mean(k)/2 for k in zip(*mgm_cost_list_2)]  # averaging sum of related cells in all iteration -second problem
    plt.subplot(1, 2, 1)  # row 1, col 2 index 1
    plt.title("problem 1 -> prob = 0.2")
    plt.plot(x, y1_1, label="dsa 0.7", color='orange')
    plt.plot(x, y2_1, label="dsa 0.4", color='cyan')
    plt.plot(x, y3_1, label="mgm", color='blue')
    plt.xlabel("Iteration")  # x lable
    plt.ylabel("Total Cost")  # y lable
    plt.legend()  # line describer (dsa/mgm line)
    plt.subplot(1, 2, 2)  # row 1, col 2 index 2
    plt.title("problem 2 -> prob = 0.5")
    plt.plot(x, y1_2, label="dsa 0.7")
    plt.plot(x, y2_2, label="dsa 0.4")
    plt.plot(x, y3_2, label="mgm")
    plt.xlabel("Iteration")  # x lable
    plt.ylabel("Total Cost")  # y lable
    plt.legend()  # line describer (dsa/mgm line)
    plt.show()


def initiate_problem(prob_for_neighbor, dsa_prob_1, dsa_prob_2):
    agents = generate_agents()  # generates 30 agents
    initiate_neighborhood(agents, prob_for_neighbor)  # initiate neighborhood - creates constrains beween agents
    dsa_cost_list_prob_1 = []  # list for each iteration for problem 1(100 iterations and 0.2 probability to create neighbors)
    dsa_cost_list_prob_2 = []  # list for each iteration for problem 2(100 iterations and 0.5 probability to create neighbors)
    mgm_cost_list = []
    for iter in range(10):
        agents1 = copy.deepcopy(agents)  # coping original agents list
        agents2 = copy.deepcopy(agents)  # coping original agents list
        agents3 = copy.deepcopy(agents)  # coping original agents list
        dsa_cost_list_prob_1.append(dsa_algorithm(agents1, dsa_prob_1))  # calling dsa with prob = 0.7
        dsa_cost_list_prob_2.append(dsa_algorithm(agents2, dsa_prob_2))  # calling dsa with prob = 0.4
        mgm_cost_list.append(mgm_algorithm(agents3))  # calling mgm algorithm
    return dsa_cost_list_prob_1, dsa_cost_list_prob_2, mgm_cost_list  # returns relevant lists


if __name__ == '__main__':
    prob_for_neighbor_problem1 = 0.2
    prob_for_neighbor_problem2 = 0.5
    dsa_prob_1 = 0.7
    dsa_prob_2 = 0.4
    dsa_cost_list_prob_1_1, dsa_cost_list_prob_2_1, mgm_cost_list_1 = initiate_problem(prob_for_neighbor_problem1, dsa_prob_1, dsa_prob_2)  # for problem 1 with prob = 0.2
    dsa_cost_list_prob_1_2, dsa_cost_list_prob_2_2, mgm_cost_list_2 = initiate_problem(prob_for_neighbor_problem2, dsa_prob_1, dsa_prob_2)  # for problem 2 with prob = 0.5
    plot_input_func(dsa_cost_list_prob_1_1, dsa_cost_list_prob_2_1, mgm_cost_list_1, dsa_cost_list_prob_1_2, dsa_cost_list_prob_2_2, mgm_cost_list_2)  # graph generator