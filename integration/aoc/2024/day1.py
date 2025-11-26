def part1(a: list, b: list) -> int:
    a = sorted(a)
    b = sorted(b)
    result = 0
    for lhs, rhs in zip(a, b):
        result += abs(lhs - rhs)
    return result

input = open("/home/gil/python-cpp/integration/aoc/2024/data/day1.txt", "r")
a = []
b = []
for line in input.readlines():
    els = line.split()
    a.append(int(els[0]))
    b.append(int(els[1]))

part1_solution = part1(a, b)
print(f"Solution day 1 - part 1: {part1_solution}")
assert part1_solution == 1579939

def part2(a: list, b: list) -> int:
    from collections import Counter

    lhs_counter = Counter(b)
    a_set = set(a)

    result = 0
    for el in a_set:
        result += el * lhs_counter[el]

    return result

part2_solution = part2(a, b)
print(f"Solution day 1 - part 2: {part2_solution}")
assert part2_solution == 20351745