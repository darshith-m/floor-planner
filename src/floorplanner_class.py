'''Floor-planning using simulated annealing'''
import copy
import random
from math import sqrt, exp, log
from block import Block
from tree import TreeNode
random.seed(100)

class FloorPlanner:
    '''Class to perform floor-planning'''

    def __init__(self, blocks, output_file):
        self.blocks = blocks    # List of all the blocks
        self.best_case_blocks = None    # Configuration of best case blocks
        self.ideal_cost = 0     # Least possible area
        self.soft_count = 0
        for block_object in self.blocks.values():
            self.ideal_cost += block_object.area
            if block_object.is_soft:
                self.soft_count += 1
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

    def calculate_length_width(self, operand, aspect_ratio):
        '''Function to calculate length and width from aspect ratio and area'''
        width = sqrt(aspect_ratio*self.blocks[operand].area)
        length = width/aspect_ratio
        return width, length

    def calculate_combination(self, operand_1, operand_2, operator):
        '''Function to combine soft macros'''
        # Store all possible combinations of length and width for operand 1
        operand_1_combinations = []
        if self.blocks[operand_1].is_soft and "temp" not in self.blocks[operand_1].block_name:
            self.blocks[operand_1].width = []
            self.blocks[operand_1].length = []
            # Check if the block has maximum aspect ratio
            if self.blocks[operand_1].max_aspect_ratio is not None:
                max_aspect_ratio = self.blocks[operand_1].max_aspect_ratio
                # Calculate length and width for maximum aspect ratio
                width1_max, length1_max = self.calculate_length_width(
                    operand_1, max_aspect_ratio)
                self.blocks[operand_1].width.append(width1_max)
                self.blocks[operand_1].length.append(length1_max)
                 # Calculate length and width for aspect ratio of 1
                width1_1, length1_1 = self.calculate_length_width(operand_1, 1)
                self.blocks[operand_1].width.append(width1_1)
                self.blocks[operand_1].length.append(length1_1)
            # Calculate length and width for minimum aspect ratio
            min_aspect_ratio = self.blocks[operand_1].min_aspect_ratio
            width1_min, length1_min = self.calculate_length_width(
                operand_1, min_aspect_ratio)
            self.blocks[operand_1].width.append(width1_min)
            self.blocks[operand_1].length.append(length1_min)
        # Store all possible combinations of length and width for operand 1
        width = self.blocks[operand_1].width[:]
        length = self.blocks[operand_1].length[:]
        zipped_list = zip(width, length)
        operand_1_combinations = list(zipped_list)
        # Store all possible combinations of length and width for operand 2
        operand_2_combinations = []
        # Check if the block has maximum aspect ratio
        if self.blocks[operand_2].is_soft and "temp" not in self.blocks[operand_2].block_name:
            self.blocks[operand_2].width = []
            self.blocks[operand_2].length = []
            if self.blocks[operand_2].max_aspect_ratio is not None:
                max_aspect_ratio = self.blocks[operand_2].max_aspect_ratio
                # Calculate length and width for maximum aspect ratio
                width2_max, length2_max = self.calculate_length_width(
                    operand_2, max_aspect_ratio)
                self.blocks[operand_2].width.append(width2_max)
                self.blocks[operand_2].length.append(length2_max)
                # Calculate length and width for aspect ratio of 1
                width2_1, length2_1 = self.calculate_length_width(operand_2, 1)
                self.blocks[operand_2].width.append(width2_1)
                self.blocks[operand_2].length.append(length2_1)
            # Calculate length and width for minimum aspect ratio
            min_aspect_ratio = self.blocks[operand_2].min_aspect_ratio
            width2_min, length2_min = self.calculate_length_width(
                operand_2, min_aspect_ratio)
            self.blocks[operand_2].width.append(width2_min)
            self.blocks[operand_2].length.append(length2_min)
        # Store all possible combinations of length and width for operand 2
        width = self.blocks[operand_2].width[:]
        length = self.blocks[operand_2].length[:]
        zipped_list = zip(width, length)
        operand_2_combinations = list(zipped_list)

        #  Find the best combination of length and width
        index1 = 0
        index2 = 0
        width_list = []
        length_list = []
        if operator == "V":
            # Sort the combinations based on length
            operand_1_combinations.sort(key=lambda x: x[0])
            operand_2_combinations.sort(key=lambda x: x[0])
            while index1 < len(operand_1_combinations) and index2 < len(operand_2_combinations):
                length = 0
                width = 0
                operand_1_length = operand_1_combinations[index1][1]
                operand_2_length = operand_2_combinations[index2][1]
                operand_1_width = operand_1_combinations[index1][0]
                operand_2_width = operand_2_combinations[index2][0]
                # Find the maximum length and sum of widths
                length = max(operand_1_length, operand_2_length)
                width = operand_1_width + operand_2_width
                length_list.append(length)
                width_list.append(width)
                # Check if the length of both blocks are same
                if operand_1_length == operand_2_length:
                    break
                if length == operand_1_length:
                    index1 += 1
                else:
                    index2 += 1
        elif operator == "H":
            # Sort the combinations based on width
            operand_1_combinations.sort(key=lambda x: x[1])
            operand_2_combinations.sort(key=lambda x: x[1])
            while index1 < len(operand_1_combinations) and index2 < len(operand_2_combinations):
                operand_1_length = operand_1_combinations[index1][1]
                operand_2_length = operand_2_combinations[index2][1]
                operand_1_width = operand_1_combinations[index1][0]
                operand_2_width = operand_2_combinations[index2][0]
                # Find the maximum width and sum of lengths
                length = operand_1_length + operand_2_length
                width = max(operand_1_width, operand_2_width)
                length_list.append(length)
                width_list.append(width)
                # Check if the width of both blocks are same
                if operand_1_width == operand_2_width:
                    break
                if width == operand_1_width:
                    index1 += 1
                else:
                    index2 += 1
        return width_list, length_list

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
                        temp_block = Block(is_soft=True)
                        temp_block.length = length[:]
                        temp_block.width = width[:]
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
                        temp_block = Block(is_soft=False)
                        temp_block.length = length
                        temp_block.width = width
                    temp_block_counter += 1
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
        if self.soft_count > 0:
            self.total_length = self.blocks[self.root_node].length[:]
            self.total_width = self.blocks[self.root_node].width[:]
            minimum_area = 1e10
            for width, length in zip(self.total_width, self.total_length):
                current_area = width * length
                if current_area < minimum_area:
                    minimum_area = current_area
            cost = minimum_area
        else:
            self.total_length = self.blocks[self.root_node].length
            self.total_width = self.blocks[self.root_node].width
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
        sequence = []
        for i, node in enumerate(polish_expression):
            if node in ("H", "V"):
                sequence.append(i)
                if i == len(polish_expression)-1:
                    operator_index.append(sequence)
            elif node not in ("H", "V"):
                if len(sequence) > 0:
                    operator_index.append(sequence)
                    sequence = []
        # Pick a random index of an operator
        random_indices = random.choice(operator_index)
        # Complement the operator
        for random_index in random_indices:
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
        # For soft blocks do not perform rotation
        if self.soft_count > 0:
            swap = random.randint(0, 2)
        # For hard blocks perform all pertubations
        else:
            swap = random.randint(0,3)
        # Pertubation - 1: swap operands
        if swap == 0:
            polish_expression = self.swap_operands(polish_expression)
            # Polish expression validity need not be checked, return True for validity
            valid = True
        # Pertubation - 2: complement operator
        elif swap == 1:
            polish_expression = self.complement_operator(polish_expression)
            # Check validity
            valid = self.check_validity(polish_expression)
        # Pertubation - 3: swap operand with operator
        elif swap == 2:
            polish_expression = self.swap_operand_operator(polish_expression)
            # Check validity
            valid = self.check_validity(polish_expression)
        # Pertubation - 4: Rotate a block
        elif swap == 3:
            polish_expression = self.rotate(polish_expression)
            # Polish expression validity need not be checked, return True for validity
            valid = True
        return valid, polish_expression

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
        if self.soft_count > 0:
            num_moves_per_temp_step = 100
        else:
            if len(self.blocks) <=20:
                num_moves_per_temp_step = 100
            elif len(self.blocks) <= 50:
                num_moves_per_temp_step = 200
            elif len(self.blocks) <= 100:
                num_moves_per_temp_step = 300
            else:
                num_moves_per_temp_step = 500
        # Defint the number of pertubations to perform per temperature
        # Copy polish expression to avoid changing the original polish expression list
        polish_expression = self.polish_expression.copy()
        # Set the cost to initial cost
        current_cost = self.initial_cost
        delta_cost_list = []
        # Perform pertubations prior to simulated annealing to get an estimate of average delta cost
        total_iterations = num_moves_per_temp_step * len(self.blocks)
        for _ in range(total_iterations):
            valid, new_polish_expression = self.pertubation(polish_expression)
            if valid:
                new_cost = self.cost(new_polish_expression)
                delta_cost = new_cost - current_cost
                current_cost = new_cost
                # Store only the uphill delta cost
                if delta_cost > 0:
                    delta_cost_list.append(delta_cost)
        # Average of delta cost
        max_delta_cost = max(delta_cost_list)
        # Set initial temperature
        temperature_initial = (-max_delta_cost)/log(0.99)
        # Set freeze temperature
        temperature_freeze = 0.1
        polish_expression = self.polish_expression.copy()
        temperature = temperature_initial
        # Start simulated annealing
        # Stop when temperature_initial reaches temperature_freeze
        # or current_cost is lesser than ideal cost
        while temperature > temperature_freeze and current_cost > self.ideal_cost:
            # Initialize count of accepted and rejected moves
            accept = 0
            reject = 0
            # Perform certain number of pertubations per temperature
            for _ in range(num_moves_per_temp_step):
                # Perform pertubation and check if the new polish expression is valid
                valid, new_polish_expression = self.pertubation(
                    polish_expression)
                if valid:
                    # Compute new cost from the new polish expression
                    new_cost = self.cost(new_polish_expression)
                    # Compute change in cost
                    delta_cost = new_cost - current_cost
                    # Check if move can be accepted based on delta cost and temperature
                    accept_flag = self.accept_move(delta_cost, temperature)
                    # Update accept and reject count
                    if accept_flag:
                        accept += 1
                        # Update polish expression, cost, and blocks configuration
                        polish_expression = new_polish_expression.copy()
                        current_cost = new_cost
                        best_case_blocks = self.blocks.copy()
                    else:
                        reject += 1
            # Set cooling schedule
            temperature = temperature*0.99
            # Compare to check if current cost is lower than minimum cost
            if current_cost < self.minimum_cost:
                # Update minimum cost, best polish expression, and best case blocks
                self.minimum_cost = current_cost
                self.best_polish_expression = polish_expression.copy()
                self.best_case_blocks = best_case_blocks.copy()

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
        # Get the key, operator, operand 1, and operand 2
        key = root.key
        operator = root.operator
        operand_1 = root.operand_1
        operand_2 = root.operand_2
        # Check if the root node is the key and soft count is greater than 0
        if key == self.root_node and self.soft_count > 0:
            minimum_area = 1e10
            # Find the minimum area
            for width, length in zip(self.total_width, self.total_length):
                current_area = width * length
                if current_area < minimum_area:
                    minimum_area = current_area
                    minimum_area_length = length
                    minimum_area_width = width
            # Store all possible combinations of length and width for operand 1
            operand1_combinations = zip(self.blocks[operand_1].width, self.blocks[operand_1].length)
            operand1_combinations = list(operand1_combinations)
            # Store all possible combinations of length and width for operand 2
            operand2_combinations = zip(self.blocks[operand_2].width, self.blocks[operand_2].length)
            operand2_combinations = list(operand2_combinations)
            # Perform vertical stacking
            if operator == "V":
                # Iterate through all possible combinations
                for width1, length1 in operand1_combinations:
                    for width2, length2 in operand2_combinations:
                        # Find the maximum length and sum of widths
                        length = max(length1, length2)
                        width = width1 + width2
                        # Check if the length and width are equal to minimum area length and width
                        if length == minimum_area_length and width == minimum_area_width:
                            # Update the width and length of the blocks
                            self.blocks[operand_1].width = width1
                            self.blocks[operand_1].length = length1
                            self.blocks[operand_2].width = width2
                            self.blocks[operand_2].length = length2
            # Perform horizontal stacking
            elif operator == "H":
                # Iterate through all possible combinations
                for width1, length1 in operand1_combinations:
                    for width2, length2 in operand2_combinations:
                        # Find the maximum width and sum of lengths
                        length = length1 + length2
                        width = max(width1, width2)
                        # Check if the length and width are equal to minimum area length and width
                        if length == minimum_area_length and width == minimum_area_width:
                            # Update the width and length of the blocks
                            self.blocks[operand_1].width = width1
                            self.blocks[operand_1].length = length1
                            self.blocks[operand_2].width = width2
                            self.blocks[operand_2].length = length2
            # Update the total length and width
            self.total_length = minimum_area_length
            self.total_width = minimum_area_width
        # Check if the key is not the root node and soft count is greater than 0
        elif key != self.root_node and self.soft_count > 0:
            # Store all possible combinations of length and width for operand 1
            operand1_combinations = zip(self.blocks[operand_1].width, self.blocks[operand_1].length)
            operand1_combinations = list(operand1_combinations)
            # Store all possible combinations of length and width for operand 2
            operand2_combinations = zip(self.blocks[operand_2].width, self.blocks[operand_2].length)
            operand2_combinations = list(operand2_combinations)
            # Perform vertical stacking
            if operator == "V":
                # Iterate through all possible combinations
                for width1, length1 in operand1_combinations:
                    for width2, length2 in operand2_combinations:
                        # Find the maximum length and sum of widths
                        length = max(length1, length2)
                        width = width1 + width2
                        # Check if the length and width are equal to the block's length and width
                        if length == self.blocks[key].length and width == self.blocks[key].width:
                            # Update the width and length of the blocks
                            self.blocks[operand_1].width = width1
                            self.blocks[operand_1].length = length1
                            self.blocks[operand_2].width = width2
                            self.blocks[operand_2].length = length2
            # Perform horizontal stacking
            elif operator == "H":
                # Iterate through all possible combinations
                for width1, length1 in operand1_combinations:
                    for width2, length2 in operand2_combinations:
                        # Find the maximum width and sum of lengths
                        length = length1 + length2
                        width = max(width1, width2)
                        # Check if the length and width are equal to the block's length and width
                        if length == self.blocks[key].length and width == self.blocks[key].width:
                            # Update the width and length of the blocks
                            self.blocks[operand_1].width = width1
                            self.blocks[operand_1].length = length1
                            self.blocks[operand_2].width = width2
                            self.blocks[operand_2].length = length2
        # Check if the key is in the block tree dictionary
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
                    self.blocks[operand_1].length
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
        #area = self.total_length * self.total_width
        black_area = cost - self.ideal_cost
        # black_area_percentage = (black_area/area)*100
        # # Write the output file
        with open(f"{self.output_file}", "w", encoding="utf-8") as result_file:
            result_file.write(f"Final area = {cost}\n")
            result_file.write(f"Black area = {black_area}\n\n")
            #result_file.write(f"Black area percentage: {black_area_percentage}\n\n")
            result_file.write(
                "block_name lower_left(x,y)coordinate upper_right(x,y)coordinate\n")
            for node_name, node in self.blocks.items():
                lower_x = round(node.x_coordinate,5)
                lower_y = round(node.y_coordinate,5)
                upper_x = round(node.x_coordinate+node.width,5)
                upper_y = round(node.y_coordinate+node.length,5)
                result_file.write(f"{node_name} ({lower_x},{lower_y}) ({upper_x},{upper_y})\n")
        result_file.close()

    def execute(self):
        '''Function to execute floor-planning'''
        # Step 1: Initial draw - Place the blocks in horizontally stacked way
        self.initial_draw()
        # Step 2: Perform simulated annealing
        self.simulated_annealing()
        # Step 3: Update coordinates of the blocks, and print the output file.
        self.final_draw()
