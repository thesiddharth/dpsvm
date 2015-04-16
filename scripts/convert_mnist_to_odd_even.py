import sys

__author__ = 'Sid'

if __name__ == "__main__":

    filename = str(sys.argv[1])
    fd = open(filename, "r")
    ft = open(filename[0:len(filename)-4]+"_conv.csv", "w")

    print "Processing ..."

    while True:

        line = fd.readline()

        if not line or len(line) == 0:
            break

        if line == "\n":
            continue

        tokens = line.strip().split(",")
        first_token = '1' if int(tokens[0]) % 2 == 0 else '-1'

        tokens = map(lambda x: str(float(x)/255.0), tokens[1:])

        tokens.insert(0, first_token)

        ft.write(",".join(tokens)+"\n")

    print "Done"