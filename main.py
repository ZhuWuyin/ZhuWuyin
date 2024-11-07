import os
import numpy as np
import multiprocessing as mp
from multiprocessing.managers import ListProxy
from Graph_Invariant import Graph_Invariant as GI
from pbar_B_GI import get_isomorphic, Batch_ADJ

def print_percentage(curr: int, total: int, end: str = "") -> None:
    if total <= 0:
        print(f"\rCount: {curr}", end=end)
    else :
        print("\rCount: {0}%".format(round(curr/total*100, 1)), end=end)

def get_matrices(file_path: str, matrices: ListProxy, total: int = -1):
    names = []
    with open(file_path, "r") as file:
        title = ""
        while True:
            title = file.readline()
            if title != "\n":
                temp = title.strip("\n .").split(" ")
                n = int(temp[-1])
                print("n =", n)
                break
        
        prev = 0
        curr = 0
        name = title.strip("\n")
        print_percentage(curr, total)
        while True:
            matrix = [file.readline().strip(" \n").split(" ") for i in range(n)]
            names.append(name)
            matrices.append(np.array(matrix, dtype = np.int8))
            empty_line = file.readline()
            name = file.readline().strip("\n")
            if curr >= prev+(prev/3)%100+2500:
                print_percentage(curr, total)
                prev = curr
            curr += 1
            if len(empty_line) == 0 or len(name) == 0:
                break
        print_percentage(curr, total, end = "\n")
    return (names, n)

def divide_task(start, length, count, lower_bound):
    step = (length - start) / count
    step = int(step) if step > lower_bound else lower_bound
    result = []
    for i in range(count):
        if start == length:
            break
        end = start + step
        end = end if end <= length else length
        if i == count - 1:
            end = length
        result.append(end)
        start = end
    return result

def get_task_range(tasks: list[int]) -> list[tuple[int]]:
    start = 0
    task_range = []
    for end in tasks:
        task_range.append((start, end))
        start = end
    return task_range

def get_input_int(s: str) -> int:
    while True:
        try :
            return int(input(s))
        except ValueError:
            print("Error input, try again ->")

def check_isomorphic(matrices: ListProxy, C, processor_count: int, lower_bound: int):
    tasks = divide_task(0, len(matrices), processor_count, lower_bound)
    task_range = get_task_range(tasks)
    manager = mp.Manager()
    SNF_matrices = manager.list()
    code_lst = manager.list()

    args = []
    for i in range(len(task_range)):
        r = task_range[i]
        flag = True
        if i == 0:
            flag = False
        args.append((matrices, r, C, SNF_matrices, flag, code_lst, i))
    
    with mp.Pool(processor_count) as pool:
        pool.starmap(Batch_ADJ, args)

    print("\nget_isomorphic")
    isomorphic = get_isomorphic(SNF_matrices)
    manager.shutdown()
    return isomorphic

if __name__ == "__main__":
    mp.freeze_support()
    processor_count = mp.cpu_count()
    lower_bound = 100

    while True:
        file_path = input("File Path: ").strip("\'\" ")
        if not os.path.exists(file_path):
            print("Error path, try again ->")
            continue
        break
    total = get_input_int("Total number of matrices: ")
    remaining_num = get_input_int("Number of remaining matrices: ")
    if remaining_num < 0:
        remaining_num = 0

    while True:
        print("\n---------------get_matrices---------------")
        matrices_manager = mp.Manager()
        matrices = matrices_manager.list()
        names, n = get_matrices(file_path, matrices, total)

        C_range = n**5
        C = GI.C_Generator(C_range, 4)
        isomorphic = check_isomorphic(matrices, C, processor_count, lower_bound)

        incorrect = []
        for i in isomorphic:
            name = names[i]
            matrix = matrices[i]
            incorrect.append(name+"\n"+'\n'.join([str(line).strip("[ ]") for line in matrix]))
        matrices_manager.shutdown()

        while True:
            try :
                with open("out{0}.txt".format(n), "w") as file:
                    file.write("\n"+"\n\n".join(incorrect))
                break
            except KeyboardInterrupt:
                print("Writing, please wait...")

        print("\nIncorrect:", len(incorrect))
        print("All Done")

        isomorphic_len = len(isomorphic)
        if isomorphic_len > remaining_num:
            file_path = "out{0}.txt".format(n)
            total = isomorphic_len
        else :
            break

    input("Press Enter to exit...")