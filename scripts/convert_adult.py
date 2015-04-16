import sys

__author__ = 'Sid'

if __name__ == "__main__":

    filename = str(sys.argv[1])
    fd = open(filename, "r")
    ft = open(filename[0:len(filename)-4]+"_conv.csv", "w")

    print "Processing ..."

    while True:

        line = fd.readline().strip()

        if not line or len(line) == 0:
            break

        if line == "\n":
            continue

        in_tokens = line.split(" ")
        out_tokens = ['0'] * 124

        out_tokens[0] = '1' if in_tokens[0].startswith("+") else "-1"

        for i in range(1, len(in_tokens)):

            split = in_tokens[i].split(":")
            out_tokens[int(split[0])+1] = '1'

        ft.write(",".join(out_tokens)+"\n")

    print "Done"






