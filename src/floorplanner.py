'''Main module for the floor-planning tool'''
import time
from pathlib import Path
from read_block import read_block
from floorplanner_class import FloorPlanner
from parse_arguments import parse_arguments

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
    start_time = time.time()
    main()
    end = time.time()
    print(f"Execution time: {end-start_time} seconds")
