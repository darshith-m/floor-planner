"""Read block information from a .block file."""
import re
from block import Block

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
                if float(block_search.group(5)) != block.min_aspect_ratio:
                    block.max_aspect_ratio = float(block_search.group(5))
                # Store the block in the dictionary
                blocks[block_search.group(1)] = block
    return blocks
