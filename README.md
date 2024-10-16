# ASCIIrcuit

Implements a Circuit class that can simulate digital circuits. The cool thing about the class is its class method `Circuit.from_string(string)`, that takes in some ASCII-art representation of a circuit and builds an instance of the class. The method `truth_table()` returns a MDTable, that `__repr__`s a Markdown table.

## Format

```
..........................................
.................12345678.................
Inputname.......0gatename0......Outputname
.................87654321.................
..........................................
```

Use `+` to cross wires without them touching and use `#` for branching and bends.

Examples:

```
...#not#........
...|...|........
A--#---+-------X
...|...|........
B--or--and-----Y
```

```
................
................
A--#-----------X
...#not#........
B--or--and-----Y
```

Both of these represent $X = A, Y = ¬A \land (A \lor B) \equiv ¬A \land B$, so $Y$ is True, if and only if $A$ is False and $B$ is True.

More examples in `main.py`

## Things I would do, if I could stick to a project (feel free to do a pull request):

- Turn it into a Obsidian plugin
- Some UX (and maybe even UI)
- Subcircuits: There should be a way to compact circuits down to custom gates. It shouldn't be too hard to do by appending a Gate constructed from the saved Circuit to the list of gates. (Maybe I shoud also merge the 3 very cursed global variables into one, or better yet, put them in a seperate file.)
- Recursive circuits: At the moment, for some recursive circuits, the algorithm never halts. I intend to fix that using an algorithm: It should first check for and report any cycles that don't go through a delay gate. Because this is a very basic component with no real representation in digital circuits, I want to do something fun: The bottom of the board should loop back to the top with delay. Delay here means to delay the input by one simulation tick, before returning it in the output. Because this forces more complex calculations to take place in multiple ticks, it would cause the simulation to always halt, and to report any possible loops.
