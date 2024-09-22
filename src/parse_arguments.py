'''Module to parse command line arguments'''
import argparse

def parse_arguments():
    '''Function to parse command line arguments'''
    parser = argparse.ArgumentParser()
    # Argument to read block file
    parser.add_argument("--input", action="store", help="Reads *.block file.")
    parser.add_argument("--output", action="store",
                        help="Name of the output file.")
    args = parser.parse_args()
    return args
