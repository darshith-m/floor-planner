# Floorplanning for hard and soft macros

### Directory:
```bash
.
├── src                       # Source folder
│   ├── floorplanner.py           # Has the main function to start floorplanner
│   ├── floorplanner_class.py     # Contains floorplanner class
│   ├── block.py       			  # Contains block class to store block details
│	├── read_block.py			  # Contains block parsing function
│	├── tree.py					  # Contains tree class
│   └── parse_arguments.py        # Contains function to parse arguments for floorplanner.py
├── experiments	             # Attempted multi-threaded/ multi-process approches
├── hard_blocks              # Contains hard blocks netlists
├── soft_blocks              # Contains soft blocks netlists
├── hard_blocks              # Contains hard blocks netlists
├── output_soft              # Contains soft blocks output
├── output_hard              # Contains hard blocks output
└── README.md
```

### Commands:

1. Change the root directory to 'src' folder to execute the 'floorplanner.py' file.
------------------------------------------------------------
    cd src
------------------------------------------------------------

2. Command to generate circuit details output file.
----------------------------------------------------------------------------------
    python3.7 floorplanner.py -input <path/to/*.block> -output <*.out>
----------------------------------------------------------------------------------
