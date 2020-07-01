import os
import marshal as pickle


def gen(fn):
    with open(fn) as f:
        lst = f.readlines()

    tree = {'$': {}}

    for l in lst:
        cur = tree['$']
        l = l.strip() + '#'
        print(l)
        for x in l:
            print(x)
            if x not in cur:
                cur[x] = {}
                print(tree)
            print(cur[x])
            cur = cur[x]
            print(cur)
            print(cur.keys())
    print(tree)
    if 'g' not in tree:
        print('1')
    if 'b' not in tree:
        print('2')

    return tree


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Build search tree from dictionary.')
    parser.add_argument('input_file', help='Input file name.')
    parser.add_argument('output_file', help='Output file name.')
    
    args = parser.parse_args()

    tree = gen(args.input_file)

    with open(args.output_file, 'wb') as f:
        pickle.dump(tree, f)
