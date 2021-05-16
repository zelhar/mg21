import argh


def importantCommand(name, title="mr"):
    """a greeting by name and title."""
    print("hello {} {} !".format(title, name))


def lessImportantCommand(x, y):
    """a very stupid command."""
    if x < y:
        print("fuck you!")
    else:
        print("I win")


# assembling parser
parser = argh.ArghParser()
parser.add_commands([importantCommand, lessImportantCommand])

if __name__ == "__main__":
    parser.dispatch()
