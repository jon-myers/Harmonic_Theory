import json

with open('branches/branches4.json') as json_file:
    branches = json.load(json_file)
    containments = []
    dims = []
    nums_of_decomp_branches = []
    decomp_branch_sizes = []
    roots = []
    assymetry = []
    stability = []
    routes = []
    loops = []
    rotation_shell_sizes = []
    rotation_shell_proportion = []
    multipath_shell_size = []
    multipath_shell_proportion = []


    for branch in branches:
        containments.append(branch['containments'])
        print(branch)
        print('\n\n')
