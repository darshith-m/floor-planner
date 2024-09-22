'''Floor-planning using simulated annealing'''
import random
import math
import statistics
import copy
import argparse
from pathlib import Path
import re
from math import sqrt, exp, log
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
#random.seed(100)


class TreeNode:
    '''Class to store blocks in the form of Tree'''

    def __init__(self):
        self.key = None
        self.operand_1 = None
        self.operand_2 = None
        self.operator = None


class Block:
    '''Class to store details of blocks'''

    def __init__(self, is_soft):
        self.block_name = None
        # True for a soft-macro, None (or False) for a hard-macro
        self.is_soft = is_soft
        # For soft-macros only, otherwise None
        if self.is_soft:
            self.min_aspect_ratio = None    # Minimum aspect ratio
            self.max_aspect_ratio = None    # Maximum aspect ratio
        self.length = None
        self.width = None
        self.area = None
        # To print the coordinates of each block.
        self.x_coordinate = 0.0  # lower left
        self.y_coordinate = 0.0  # lower left


class FloorPlanner:
    '''Class to perform floor-planning'''

    def __init__(self, blocks, output_file):
        self.blocks = blocks    # List of all the blocks
        self.best_case_blocks = None    # Configuration of best case blocks
        self.ideal_cost = 0     # Least possible area
        for block_object in self.blocks.values():
            self.ideal_cost += block_object.area
        self.polish_expression = []     # Polish expression list
        self.initial_cost = 0.0         # To store initial cost after initial draw
        self.total_width = 0.0          # To store total width of placed blocks
        self.total_length = 0.0         # To store total length of placed blocks
        # To store the least area calculated by simulated annealing
        self.minimum_cost = 1e50
        self.best_polish_expression = None  # To store the best polish expression
        self.block_tree_dict = {}       # To store block tree dictionary objects
        self.root_node = None
        self.output_file = output_file  # Path to output file

    def matplotlib_draw(self):
        '''Function to draw floor-plan'''
        _, ax = plt.subplots(1)
        for block in self.blocks.values():
            # Draw rectangle for each block
            x_coordinate = block.x_coordinate
            y_coordinate = block.y_coordinate
            width = block.width
            length = block.length
            block_name = block.block_name
            rect = Rectangle((x_coordinate, y_coordinate), width,
                             length, block_name, fill=True, edgecolor="black")
            # Add the rectangle to the plot
            ax.add_patch(rect)
            # Annotate the block name
            x = x_coordinate + (width/2)
            y = y_coordinate+(length/2)
            ax.annotate(f"{block_name}", (x, y), color='black',
                        fontsize=5, ha='center', va='center')
        # Set the x and y limits
        plt.xlim(0, self.total_width*1.25)
        plt.ylim(0, self.total_length*1.25)
        plt.title("Floor-planning")
        plt.savefig(
            f"Floorplan: Area={self.total_width*self.total_length}.png")

    def calculate_length_width(self, operand, aspect_ratio):
        '''Function to calculate length and width from aspect ratio and area'''
        width = sqrt(aspect_ratio*self.blocks[operand].area)
        length = width/aspect_ratio
        return width, length

    def calculate_combination(self, operand_1, operand_2, operator):
        '''Function to combine soft macros'''
        # Store all possible combinations of length and width for operand 1
        operand_1_combinations = []
        if self.blocks[operand_1].is_soft:
            max_aspect_ratio = self.blocks[operand_1].max_aspect_ratio
            min_aspect_ratio = self.blocks[operand_1].min_aspect_ratio
            # Calculate length and width for maximum aspect ratio
            width1_max, length1_max = self.calculate_length_width(
                operand_1, max_aspect_ratio)
            operand_1_combinations.append((width1_max, length1_max))
            # Calculate length and width for minimum aspect ratio
            width1_min, length1_min = self.calculate_length_width(
                operand_1, min_aspect_ratio)
            operand_1_combinations.append((width1_min, length1_min))
            # Calculate length and width for aspect ratio of 1
            width1_1, length1_1 = self.calculate_length_width(operand_1, 1)
            if width1_1 is not None and length1_1 is not None:
                operand_1_combinations.append((width1_1, length1_1))
        # If the block is not a soft macro, store the width and length
        else:
            width = self.blocks[operand_1].width
            length = self.blocks[operand_1].length
            operand_1_combinations.append((width, length))
        # Store all possible combinations of length and width for operand 2
        operand_2_combinations = []
        if self.blocks[operand_2].is_soft:
            max_aspect_ratio = self.blocks[operand_2].max_aspect_ratio
            min_aspect_ratio = self.blocks[operand_2].min_aspect_ratio
            # Calculate length and width for maximum aspect ratio
            width2_max, length2_max = self.calculate_length_width(
                operand_2, max_aspect_ratio)
            operand_2_combinations.append((width2_max, length2_max))
            # Calculate length and width for minimum aspect ratio
            width2_min, length2_min = self.calculate_length_width(
                operand_2, min_aspect_ratio)
            operand_2_combinations.append((width2_min, length2_min))
            # Calculate length and width for aspect ratio of 1
            width2_1, length2_1 = self.calculate_length_width(operand_2, 1)
            if width2_1 is not None and length2_1 is not None:
                operand_2_combinations.append((width2_1, length2_1))
        # If the block is not a soft macro, store the width and length
        else:
            width = self.blocks[operand_2].width
            length = self.blocks[operand_2].length
            operand_2_combinations.append((width, length))

        #  Find the best combination of length and width
        index1 = 0
        index2 = 0
        while index1 < len(operand_1_combinations) and index2 < len(operand_2_combinations):
            length = 0
            width = 0
            if operator == "V":
                # Sort the combinations based on length
                operand_1_combinations.sort(key=lambda x: x[0])
                operand_2_combinations.sort(key=lambda x: x[0])
                # Find the maximum length and sum of widths
                length = max(
                    operand_1_combinations[index1][1], operand_2_combinations[index2][1])
                width = operand_1_combinations[index1][0] + \
                    operand_2_combinations[index2][0]
                current_index1 = index1
                current_index2 = index2
                # Check if the length of both blocks are same
                if operand_1_combinations[index1][1] == operand_2_combinations[index2][1]:
                    break
                if length == operand_1_combinations[index1][1]:
                    index1 += 1
                else:
                    index2 += 1
            elif operator == "H":
                # Sort the combinations based on width
                operand_1_combinations.sort(key=lambda x: x[1])
                operand_2_combinations.sort(key=lambda x: x[1])
                # Find the maximum width and sum of lengths
                length = operand_1_combinations[index1][1] + \
                    operand_2_combinations[index2][1]
                width = max(
                    operand_1_combinations[index1][0], operand_2_combinations[index2][0])
                current_index1 = index1
                current_index2 = index2
                # Check if the width of both blocks are same
                if operand_1_combinations[index1][0] == operand_2_combinations[index2][0]:
                    break
                if width == operand_1_combinations[index1][0]:
                    index1 += 1
                else:
                    index2 += 1
        # Update the width and length of the blocks
        width1, length1 = operand_1_combinations[current_index1]
        self.blocks[operand_1].width = width1
        self.blocks[operand_1].length = length1
        width2, length2 = operand_2_combinations[current_index2]
        self.blocks[operand_2].width = width2
        self.blocks[operand_2].length = length2
        return width, length

    def cost(self, polish_expression_copy, final_draw=False):
        '''Function to compute cost (area)'''
        # Keep a counter to create temporary block objects
        temp_block_counter = 0
        # Copy polish expression to avoid changing the original polish expression list
        polish_expression = polish_expression_copy.copy()
        # Check if the polish expression exists
        while len(polish_expression) > 1:
            # Iterate through polish expression
            for index, node in enumerate(polish_expression):
                # Initial length and width to 0 for every block
                length = 0.0
                width = 0.0
                # Check for an operator
                if node in ("H", "V"):
                    # Get the operator, operand 1 and operand 2 by popping from the list
                    operator = polish_expression.pop(index)
                    operand_1 = polish_expression.pop(index-1)
                    operand_2 = polish_expression.pop(index-2)
                    # Check if one of the blocks is soft macro
                    if self.blocks[operand_1].is_soft or self.blocks[operand_2].is_soft:
                        width, length = self.calculate_combination(
                            operand_1, operand_2, operator)
                    else:
                        # Perform vertical stacking
                        if operator == "V":
                            length = max(
                                self.blocks[operand_1].length, self.blocks[operand_2].length)
                            width = self.blocks[operand_1].width + \
                                self.blocks[operand_2].width
                        # Perform horizontal stacking
                        elif operator == "H":
                            length = self.blocks[operand_1].length + \
                                self.blocks[operand_2].length
                            width = max(
                                self.blocks[operand_1].width, self.blocks[operand_2].width)
                    # Create a temporary block which stores previously combined blocks
                    temp_block_counter += 1
                    temp_block = Block(is_soft=False)
                    temp_block.length = length
                    temp_block.width = width
                    temp_block.block_name = f"temp_block_{temp_block_counter}"
                    # Insert the temporary block in the polish expression
                    polish_expression.insert(index-2, temp_block.block_name)
                    # Store the temporary block in dictionary
                    self.blocks[temp_block.block_name] = temp_block
                    # Build tree only for the final draw
                    # print(f"Polish expression: {polish_expression}")
                    if final_draw:
                        block_tree = TreeNode()
                        block_tree.key = temp_block.block_name
                        block_tree.operand_1 = operand_1
                        block_tree.operand_2 = operand_2
                        block_tree.operator = operator
                        self.block_tree_dict[temp_block.block_name] = block_tree
                    break
        # Store total length and width after combining all blocks
        self.root_node = polish_expression[0]
        self.total_length = self.blocks[self.root_node].length
        self.total_width = self.blocks[self.root_node].width
        # Area after combining all blocks
        cost = self.total_length * self.total_width
        return cost

    def swap_operands(self, polish_expression):
        '''Function to perform pertubation-1 - swapping operands: (1H2 to 2H1) or (12H to 21H)'''
        swap_pairs = []
        # Pick operand pairs of the form - 1H2 or 2V3
        for i in range(1, len(polish_expression)-2):
            if polish_expression[i] == "H" or polish_expression[i] == "V":
                if (polish_expression[i+1] != "H" and polish_expression[i+1] != "V"):
                    if (polish_expression[i-1] != "H" and polish_expression[i-1] != "V"):
                        swap_pairs.append((i-1, i+1))
        # Pick consecutive operand pairs
        for i in range(0, len(polish_expression)-1):
            if (polish_expression[i] != "H" and polish_expression[i] != "V"):
                if (polish_expression[i-1] != "H" and polish_expression[i-1] != "V"):
                    swap_pairs.append((i-1, i))
        # Pick a random pair
        operand_index_1, operand_index_2 = random.choice(swap_pairs)
        # Swap operands
        polish_expression[operand_index_1], polish_expression[operand_index_2] = polish_expression[operand_index_2], polish_expression[operand_index_1]
        return polish_expression

    def complement_operator(self, polish_expression):
        '''Function to perform pertubation-2 - Complementing operands: (H to V) or (V to H)'''
        # Find indexes of operators
        operator_index = []
        for i, node in enumerate(polish_expression):
            if node == "H" or node == "V":
                operator_index.append(i)
        # Pick a random index of an operator
        random_index = random.choice(operator_index)
        # Complement the operator
        if polish_expression[random_index] == "H":
            polish_expression[random_index] = "V"
        elif polish_expression[random_index] == "V":
            polish_expression[random_index] = "H"
        return polish_expression

    def swap_operand_operator(self, polish_expression):
        '''Function to perform pertubation-3 - swapping operand with operator: (12H to 1H2)'''
        # Pick consecutive pair of operand and operator
        swap_pairs = []
        for i in range(len(polish_expression)-1):
            if (polish_expression[i] == "H" or polish_expression[i] == "V"):
                if (polish_expression[i-1] != "H" and polish_expression[i-1] != "V"):
                    swap_pairs.append((i, i-1))
                if (polish_expression[i+1] != "H" and polish_expression[i+1] != "V"):
                    swap_pairs.append((i, i+1))
        # Randomly pick a pair
        operator_index, operand_index = random.choice(swap_pairs)
        # Swap operand and operator
        polish_expression[operator_index], polish_expression[operand_index] = polish_expression[operand_index], polish_expression[operator_index]
        return polish_expression

    def rotate(self, polish_expression):
        '''Function to perform pertubation-4 - Rotating a block'''
        # Pick indexes of operands
        operands_list = []
        for i in range(0, len(polish_expression)-2):
            if (polish_expression[i] != "H" and polish_expression[i] != "V"):
                operands_list.append(i)
        # Randomly pick an operand
        operand_index = random.choice(operands_list)
        operand = polish_expression[operand_index]
        # Perform deepcopy to avoid changes in original block
        block_copy = copy.deepcopy(self.blocks[operand])
        # Perform swap
        width, length = block_copy.width, block_copy.length
        block_copy.width, block_copy.length = length, width
        # Assign it back to the operand's object
        self.blocks[operand] = copy.deepcopy(block_copy)
        return polish_expression

    def check_validity(self, polish_expression):
        '''Function to check validity of polish expression'''
        no_operands = 0
        no_operators = 0
        # The last element in polish expression has to be an operator
        if polish_expression[-1] != "H" and polish_expression[-1] != "V":
            return False
        for i, node in enumerate(polish_expression):
            # The third element in polish expression has to be an operator
            if i == 2:
                if polish_expression[i] != "H" and polish_expression[i] != "V":
                    return False
            # Check if the node is an operand and increment operand count
            if node not in ("H", "V"):
                no_operands += 1
            # Check if the node is an operator and increment operator count
            elif node in ("H", "V"):
                no_operators += 1
                # Two consecutive elements cannot have same operator
                if polish_expression[i] == polish_expression[i-1]:
                    return False
                # Count of operators and operands should never match
                if no_operators == no_operands:
                    return False
        return True

    def pertubation(self, polish_expression_copy):
        '''Function to randomly pick a pertubation'''
        # Copy polish expression to avoid changing the original polish expression list
        polish_expression = polish_expression_copy.copy()
        # Randomly pick a pertubation
        swap = random.randint(0, 3)
        # Pertubation - 1: swap operands
        if swap == 0:
            polish_expression = self.swap_operands(polish_expression)
            # Polish expression validity need not be checked, return True for validity
            return True, polish_expression
        # Pertubation - 2: complement operator
        if swap == 1:
            polish_expression = self.complement_operator(polish_expression)
            # Check validity
            valid = self.check_validity(polish_expression)
            return valid, polish_expression
        # Pertubation - 3: swap operand with operator
        if swap == 2:
            polish_expression = self.swap_operand_operator(polish_expression)
            # Check validity
            valid = self.check_validity(polish_expression)
            return valid, polish_expression
        # Pertubation - 4: Rotate a block
        if swap == 3:
            polish_expression = self.rotate(polish_expression)
            # Polish expression validity need not be checked, return True for validity
            return True, polish_expression

    def accept_move(self, delta_cost, temperature):
        '''Function to randomly accept move'''
        # Accept the move if cost has improved (its a downhill move!)
        if delta_cost < 0:
            return True
        # Accept the uphill move with a certain probability
        else:
            # Calculate the boltz probability
            boltz = exp((-delta_cost)/(temperature))
            # Randomly pick a number between 0 and 1
            r = random.uniform(0, 1)
            # If randomly picked probability is less than boltz probability, accept the uphill move
            return r < boltz

    def simulated_annealing(self):
        '''Function to perform simulated annealing'''
        # Defint the number of pertubations to perform per temperature
        num_moves_per_temp_step = 1000
        # Copy polish expression to avoid changing the original polish expression list
        polish_expression = self.polish_expression.copy()
        # Set the cost to initial cost
        current_cost = self.initial_cost
        delta_cost_list = []
        # Perform pertubations priot to simulated annealing to get an estimate of average delta cost
        total_iterations = num_moves_per_temp_step * len(self.blocks)
        for i in range(total_iterations):
            valid, new_polish_expression = self.pertubation(polish_expression)
            if valid:
                new_cost = self.cost(new_polish_expression)
                delta_cost = new_cost - current_cost
                current_cost = new_cost
                # Store only the uphill delta cost
                if delta_cost > 0:
                    delta_cost_list.append(delta_cost)
        # Average of delta cost
        delta_cost_mean = statistics.mean(delta_cost_list)
        std_dev = statistics.stdev(delta_cost_list)

        plt.figure()
        # Plotting
        plt.hist(delta_cost_list, bins=100, density=True, alpha=0.6, color='g')
        # Add a line plot for the normal distribution curve
        xmin, xmax = plt.xlim()
        x = [xmin + (xmax - xmin) * i / 100 for i in range(101)]
        p = [1 / (std_dev * math.sqrt(2 * math.pi)) * math.exp(- (xi -
                                                                delta_cost_mean) ** 2 / (2 * std_dev ** 2)) for xi in x]
        plt.plot(x, p, 'k', linewidth=2)
        title = "Fit results: mean = %.2f,  std = %.2f" % (
            delta_cost_mean, std_dev)
        plt.title(title)
        plt.xlabel('Uphill delta cost')
        plt.ylabel('Count')
        max_delta_cost = max(delta_cost_list)
        plt.savefig(f"{self.output_file}_delta_cost.png")
        with open(f"{self.output_file}_delta.txt", "w", encoding="utf-8") as delta_file:
            delta_file.write(f"Delta cost - Average = {delta_cost_mean}\n")
            delta_file.write(
                f"Delta cost - Minimum = {min(delta_cost_list)}\n")
            delta_file.write(f"Delta cost - Maximum = {max_delta_cost}\n")

        # Set initial temperature
        temperature_initial = (-max_delta_cost)/log(0.99)
        print(f"Intial temperature: {temperature_initial}")
        # Set freeze temperature
        temperature_freeze = 0.1
        print(f"current_cost: {current_cost}")
        print(f" Ideal cost: {self.ideal_cost}")
        polish_expression = self.polish_expression.copy()
        cost_list = []
        accept_list = []
        temperature_list = []
        temperature = temperature_initial
        # Start simulated annealing
        # Stop when temperature_initial reaches temperature_freeze
        # or current_cost is lesser than ideal cost
        while temperature > temperature_freeze and current_cost > self.ideal_cost:
            # Initialize count of accepted and rejected moves
            accept = 0
            reject = 0
            # Perform certain number of pertubations per temperature
            for i in range(num_moves_per_temp_step):
                # Perform pertubation and check if the new polish expression is valid
                valid, new_polish_expression = self.pertubation(
                    polish_expression)
                if valid:
                    # Compute new cost from the new polish expression
                    new_cost = self.cost(new_polish_expression)
                    # Compute change in cost
                    delta_cost = new_cost - current_cost
                    # Check if move can be accepted and store cost, polish expression, and blocks configuration
                    accept_flag = self.accept_move(delta_cost, temperature)
                    # Update accept and reject counts
                    if accept_flag:
                        accept += 1
                        # Update polish expression, cost, and blocks configuration
                        polish_expression = new_polish_expression.copy()
                        current_cost = new_cost
                        best_case_blocks = self.blocks.copy()
                    else:
                        reject += 1
            # Store current cost in a list
            cost_list.append(current_cost)
            # Store current temperature in a list
            temperature_list.append(temperature)
            # Set cooling schedule
            temperature = temperature*0.99
            # Compare to check if current cost is lower than minimum cost
            if current_cost < self.minimum_cost:
                # Update minimum cost, best polish expression, and best case blocks
                self.minimum_cost = current_cost
                self.best_polish_expression = polish_expression.copy()
                self.best_case_blocks = best_case_blocks.copy()
                # print("-------------------------------------------------")
                # print(f"Minimum cost = {self.minimum_cost}")
                # print(f"Polish expression: {self.best_polish_expression}")
                # for block in self.best_case_blocks.values():
                #    print(f"{block.block_name}-{block.length}-{block.width}")
                # print("-------------------------------------------------")
            accept_list.append((accept/(accept+reject))*100)

        print(f"Final - Minumum cost: {self.minimum_cost}")
        print(f"Final polish expression: {self.best_polish_expression}")

        # for block in self.best_case_blocks.values():
        #    print(f"{block.block_name}-{block.length}-{block.width}")

        # Plotting delta_cost_list
        plt.figure()  # Create a new figure
        plt.plot(cost_list)
        plt.title('Cost function vs Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.savefig("cost.png")  # Save the figure

        # Plotting accept_list
        plt.figure()  # Create a new figure for the second plot
        plt.plot(accept_list)
        plt.title('Acceptance percentage vs Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Acceptance percentage')
        plt.savefig("accept.png")  # Save the second figure

        plt.figure()  # Create a new figure for the second plot
        plt.plot(temperature_list)
        plt.title('Temperature vs Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Temperature')
        plt.savefig("temp.png")  # Save the second figure

    def initial_draw(self):
        '''Function to perform initial draw'''
        blocks_list = list(self.blocks.keys())
        # Append first 2 blocks in the list
        self.polish_expression.append(blocks_list[0])
        self.polish_expression.append(blocks_list[1])
        for i in range(2, len(blocks_list)):
            # Apped operator
            self.polish_expression.append("H")
            # Append operand
            block = blocks_list[i]
            self.polish_expression.append(block)
        # Append the last operator
        self.polish_expression.append("H")
        # Compute area for initial draw
        self.initial_cost = self.cost(self.polish_expression)

    def tree_traversal(self, root):
        '''Function to perform tree traversal to update block co-ordinates'''
        key = root.key
        operator = root.operator
        operand_1 = root.operand_1
        operand_2 = root.operand_2
        if key in self.block_tree_dict:
            # If the blocks are vertically stacked, update coordinates
            if operator == "V":
                self.blocks[operand_1].x_coordinate = self.blocks[key].x_coordinate
                self.blocks[operand_1].y_coordinate = self.blocks[key].y_coordinate
                self.blocks[operand_2].x_coordinate = self.blocks[key].x_coordinate + \
                    self.blocks[operand_1].width
                self.blocks[operand_2].y_coordinate = self.blocks[key].y_coordinate
            # If the blocks are horizonatally stacked, update coordinates
            elif root.operator == "H":
                self.blocks[operand_1].x_coordinate = self.blocks[key].x_coordinate
                self.blocks[operand_1].y_coordinate = self.blocks[key].y_coordinate
                self.blocks[operand_2].x_coordinate = self.blocks[key].x_coordinate
                self.blocks[operand_2].y_coordinate = self.blocks[key].y_coordinate + \
                    self.blocks[root.operand_1].length
            # Recursively traverse through the tree to update coordinates of all the blocks
            if operand_1 in self.block_tree_dict:
                self.tree_traversal(self.block_tree_dict[operand_1])
            if operand_2 in self.block_tree_dict:
                self.tree_traversal(self.block_tree_dict[operand_2])

    def final_draw(self):
        '''Function to update co-ordinates and print output file'''
        # Copy the best polish expression
        polish_expression = self.best_polish_expression.copy()
        # Remove temporary blocks
        self.blocks = {key: value
                       for key, value in self.best_case_blocks.items() if 'temp' not in key}
        # Compute area for final draw
        cost = self.cost(polish_expression, final_draw=True)
        # Set the root node
        root_node = self.root_node
        # Set the total length and width
        self.total_length = self.blocks[root_node].length
        self.total_width = self.blocks[root_node].width
        # Set the coordinates of the root node
        self.blocks[root_node].x_coordinate = 0
        self.blocks[root_node].y_coordinate = 0
        # Traverse through the tree to update coordinates of all the blocks
        self.tree_traversal(self.block_tree_dict[root_node])
        # Remove temporary blocks
        self.blocks = {key: value for key,
                       value in self.blocks.items() if 'temp' not in key}
        #self.matplotlib_draw()
        # Write the output file
        with open(f"{self.output_file}", "w", encoding="utf-8") as result_file:
            result_file.write(f"Final area: {cost}\n")
            result_file.write(f"Black area: {cost-self.ideal_cost}\n\n")
            result_file.write(
                "block_name lower_left(x,y)coordinate upper_right(x,y)coordinate\n")
            for node_name, node in self.blocks.items():
                result_file.write(
                    f"{node_name} ({round(node.x_coordinate,10)},{round(node.y_coordinate,10)}) ({round(node.x_coordinate+node.width,10)},{round(node.y_coordinate+node.length,10)})\n")
        result_file.close()

    def execute(self):
        '''Function to execute floor-planning'''
        # Step 1: Initial draw - Place the blocks in horizontally stacked way
        self.initial_draw()
        # Step 2: Perform simulated annealing
        self.simulated_annealing()
        # Step 3: Update coordinates of the blocks, and print the output file
        self.final_draw()


def parse_arguments():
    '''Function to parse command line arguments'''
    parser = argparse.ArgumentParser()
    # Argument to read block file
    parser.add_argument("--input", action="store", help="Reads *.block file.")
    parser.add_argument("--output", action="store",
                        help="Name of the output file.")
    args = parser.parse_args()
    return args


def read_block(path):
    '''Parse .block file'''
    blocks = {}
    # Iterate through all the lines of the netlist
    for line in path.open():
        # Check if the block is hard macro
        if "hardrectilinear" in line:
            block_search = re.search(
                r"(\w*)\s*(\w*)(?:\s*\d*\s*)\s*(\(\d+,\s*\d+\))\s*(\(\d+,\s*\d+\))\s*(\(\d+,\s*\d+\))\s*(\(\d*,\s*\d*\))", line)
            if block_search is not None:
                coordinate_str = block_search.group(5)
                # Extract the coordinates of the block
                numbers_str = coordinate_str.strip("()").split(",")
                # Convert the coordinates to integers
                numbers = [int(num) for num in numbers_str]
                # Create a block object
                block = Block(is_soft=False)
                # Store the block name, length, width, and area
                block.block_name = block_search.group(1)
                block.length = numbers[1]
                block.width = numbers[0]
                block.area = block.length * block.width
                # Store the block in the dictionary
                blocks[block_search.group(1)] = block
        # Check if the block is soft macro
        elif "softrectangular" in line:
            block_search = re.search(
                r"(\w*)\s*(\w*)\s*(\d*)\s*(\d+(?:\.\d+)?)\s*(\d+(?:\.\d+)?)", line)
            if block_search is not None:
                # Create a block object
                block = Block(is_soft=True)
                # Store the block name, area, minimum aspect ratio, and maximum aspect ratio
                block.block_name = block_search.group(1)
                block.area = float(block_search.group(3))
                block.min_aspect_ratio = float(block_search.group(4))
                block.max_aspect_ratio = float(block_search.group(5))
                # Store the block in the dictionary
                blocks[block_search.group(1)] = block
    return blocks


def main():
    '''Main function'''
    # Parse command line arguments
    inputs = parse_arguments()
    # Checks if the read_nldm argument is defined
    if inputs.input is not None and inputs.output is not None:
        # Path to the .block file
        path = inputs.input
        path_lib = Path(path)
        # Checks if the .block file exist
        if path_lib.exists():
            # Call read_block function if the .block file exists
            blocks = read_block(path_lib)
        else:
            print(".block file doesn'temperature_initial exist.")
    # Checks if the blocks is not empty
    if blocks is not None:
        # Create floorplanner object with parsed blocks and output file name
        floorplanner = FloorPlanner(blocks, output_file=inputs.output)
        # Execute floor-planning
        floorplanner.execute()


if __name__ == '__main__':
    main()
