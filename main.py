import re
from typing import Callable


class Gate:
    inputs: list[str]
    outputs: list[str]
    operator: Callable[[tuple[bool]], tuple[bool]]

    def __init__(
        self,
        inputs: list[str],
        outputs: list[str],
        operator: Callable[[tuple[bool]], tuple[bool]],
    ) -> None:
        self.inputs = inputs
        self.outputs = outputs
        self.operator = operator


STANDARD_GATES: list[Gate] = {
    "not": Gate(["X0"], ["Y0"], lambda x: (not x[0],)),
    "and": Gate(["X0", "X1"], ["Y0"], lambda x: (x[0] and x[1],)),
    "or": Gate(["X0", "X1"], ["Y0"], lambda x: (x[0] or x[1],)),
    "nand": Gate(["X0", "X1"], ["Y0"], lambda x: (not (x[0] and x[1]),)),
    "nor": Gate(["X0", "X1"], ["Y0"], lambda x: (not (x[0] or x[1]),)),
    "xor": Gate(["X0", "X1"], ["Y0"], lambda x: (x[0] == x[1],)),
    "xnor": Gate(["X0", "X1"], ["Y0"], lambda x: (x[0] != x[1],)),
}

GLOBAL_GATES = STANDARD_GATES.copy()


class MDTable:
    columns: list[str]
    rows: list[dict[str, str]]
    max_column_width: dict[str, int]

    def __init__(self, columns):
        self.columns = [str(col) for col in columns]
        self.max_column_width = dict([(col, len(col)) for col in columns])
        self.rows = []

    def add_row(self, data: dict):
        row = dict()
        for key in self.columns:
            value = str(data[key] if key in data else "")
            row[key] = value
            if len(value) > self.max_column_width[key]:
                self.max_column_width[key] = len(value)
        self.rows.append(row)

    def __repr__(self) -> str:
        out = (
            "| "
            + " | ".join(
                [col.ljust(self.max_column_width[col]) for col in self.columns]
            )
            + " |\n"
        )
        out += (
            "|-"
            + "-|-".join(["-" * (self.max_column_width[col]) for col in self.columns])
            + "-|\n"
        )
        for row in self.rows:
            out += (
                "| "
                + " | ".join(
                    [row[col].ljust(self.max_column_width[col]) for col in self.columns]
                )
                + " |\n"
            )
        return out


class Circuit:
    inputs: dict[str, list[tuple[str, int]]]
    gates: dict[str, tuple[str, dict[int, tuple[str, int]]]]
    outputs: list[str]

    states: dict[str, bool]
    gate_input_states: dict[str, dict[int, bool]]
    wire_states: dict[int, bool]

    wire_state_mapping: dict[str, int]

    physical_wires: list[list[tuple[int, int]]]
    ascii_wires: list[list[str]]

    def __init__(
        self,
        inputs: dict[str, list[tuple[str, int]]],
        gates: dict[str, tuple[str, dict[int, tuple[str, int]]]],
        outputs: list[str],
        wire_state_mapping: dict[int, str],
    ):
        self.inputs = inputs
        self.gates = gates
        self.outputs = outputs
        self.wire_state_mapping = wire_state_mapping

    def reset(self):
        self.states = dict()
        self.gate_input_states = dict()
        self.wire_states = dict()

    def iterate(self, inputs: dict[str, bool], reset=True):
        if reset:
            self.reset()

        for key, value in inputs.items():
            self.wire_states[self.wire_state_mapping[key]] = value

        changed_values = []

        for name, value in inputs.items():
            if name not in self.states or self.states[name] != value:
                self.states[name] = value
                changed_values.append(("input", name))

        def change_gate_input(gate_name, input_port, value):
            if gate_name not in self.gate_input_states:
                self.gate_input_states[gate_name] = dict()
            if (
                input_port not in self.gate_input_states[gate_name]
                or self.gate_input_states[gate_name][input_port] != value
            ):
                if gate_name in self.outputs:
                    self.states[gate_name] = value
                    return
                self.gate_input_states[gate_name][input_port] = value
                if (
                    "gate_input",
                    gate_name,
                ) not in changed_values and len(
                    self.gate_input_states[gate_name]
                ) == len(GLOBAL_GATES[self.gates[gate_name][0]].inputs):
                    changed_values.append(("gate_input", gate_name))

        while len(changed_values):
            changed_value_type, changed_value_name = changed_values.pop(0)

            if changed_value_type == "input":
                for connected_gate in self.inputs[changed_value_name]:
                    change_gate_input(
                        connected_gate[0],
                        connected_gate[1],
                        self.states[changed_value_name],
                    )
                continue
            if changed_value_type == "gate_input":
                outputs = GLOBAL_GATES[self.gates[changed_value_name][0]].operator(
                    self.gate_input_states[changed_value_name]
                )
                for destinations, output in zip(
                    self.gates[changed_value_name][1].values(), outputs
                ):
                    for destination in destinations:
                        change_gate_input(*destination, output)
                continue

        return dict(
            [
                (output_name, int(self.states[output_name]))
                for output_name in self.outputs
            ]
        )

    def run_on_series(self, inputs: list[dict[str, bool]], steps=1, reset=True):
        if reset:
            self.reset()

        for step in range(steps):
            self.iterate(inputs[step], reset=False)

    def __repr__(self):
        return f"Inputs: {self.inputs}, Gates: {self.gates}, Outputs: {self.outputs}"

    @classmethod
    def from_wirenets(
        cls, wirenets: list[list[tuple[bool, str, int]]], nodes: list[tuple[str, str]]
    ):
        inputs = dict()
        gates = dict()
        outputs = []

        wire_state_mapping = dict()

        for node_id, node_info in nodes:
            node_type = node_info[0]
            node_info = node_info[1:]
            if node_type == ">":
                inputs[node_id] = []
            if node_type == "<":
                outputs.append(node_id)
            if node_type == "-":
                gates[node_id] = (
                    node_info,
                    dict(
                        [(i, []) for i in range(len(GLOBAL_GATES[node_info].outputs))]
                    ),  # {0: [], 1: []}
                )

        def add_connection(a, a_port, b, b_port):
            if a in inputs:
                inputs[a].append((b, b_port))
                return
            if a in gates:
                gates[a][1][a_port].append((b, b_port))
                return
            if a in outputs:
                raise ValueError("tried using output as input")

        for i, wirenet in enumerate(wirenets):
            ins = list(filter(lambda connection: connection[0], wirenet))
            if len(ins) != 1:
                raise ValueError(
                    f"the wirenet at index {i} has {len(ins)} inputs, expected 1"
                )
            value = ins[0]
            wire_state_mapping[ins[0][1]] = i

            for connection in wirenet:
                if not connection[0]:
                    add_connection(value[1], value[2], connection[1], connection[2])

        return cls(inputs, gates, outputs, wire_state_mapping)

    @classmethod
    def from_string(cls, string: str):
        string_array = string.strip().replace(".", " ").split("\n")

        width = len(string_array[0])
        height = len(string_array)

        for line in string_array:
            if len(line) != width:
                raise ValueError("input must be rectangular")

        def longest_alphanumeric_prefix(s):
            match = re.match(r"^[a-zA-Z0-9]+", s)
            return len(match.group(0)) if match else 0

        def longest_alphanumeric_suffix(s):
            match = re.search(r"[a-zA-Z0-9]+$", s)
            return len(match.group(0)) if match else 0

        def longest_lower_alphanumeric_substring(s):
            # Use regex to find all alphanumeric substrings
            matches = [match for match in re.finditer(r"[a-zA-Z0-9]+", s)]

            # Filter out matches that are at the start or end of the string
            filtered_matches = [
                (match.start(), match.end())
                for match in matches
                if match.start() > 0 and match.end() < len(s)
            ]

            # Return the filtered list or None if it's empty
            return filtered_matches if filtered_matches else None

        nodes = []  # [("A", ">A"),("xor1", "-xor"),]

        solder_blobs = dict()  # {((2, 5), (-1, 0)): (True, "A", 0)}

        gate_index = 0
        for y, line in enumerate(string_array):
            if length := longest_alphanumeric_prefix(line):
                solder_blobs[((length - 1, y), (-1, 0))] = (True, line[:length], 0)
                nodes.append((line[:length], ">" + line[:length]))
            if length := longest_alphanumeric_suffix(line):
                solder_blobs[((width - length, y), (1, 0))] = (
                    False,
                    line[-length:],
                    0,
                )
                nodes.append((line[-length:], "<" + line[-length:]))
            if (indices := longest_lower_alphanumeric_substring(line)) is not None:
                for start, end in indices:
                    gate_type = line[start:end]
                    gate_id = str(gate_index) + gate_type
                    nodes.append((gate_id, "-" + gate_type))
                    gate_index += 1

                    solder_blobs[((start, y), (1, 0))] = (
                        False,
                        gate_id,
                        0,
                    )
                    solder_blobs[((end - 1, y), (-1, 0))] = (
                        True,
                        gate_id,
                        0,
                    )
                    for x in range(1, end - start + 1):
                        solder_blobs[((start + x - 1, y), (0, 1))] = (
                            False,
                            gate_id,
                            x,
                        )
                        solder_blobs[((end - x, y), (0, -1))] = (
                            True,
                            gate_id,
                            x,
                        )

        array = [list(string) for string in string_array]

        physical_wirenets = []
        virtual_wirenets = []

        def eat_away(x, y):
            physical_wirenets.append([(x, y)])
            virtual_wirenets.append([])

            def connect_to(x, y, dx, dy):
                x, y = x + dx, y + dy

                if ((x, y), (dx, dy)) in solder_blobs:
                    virtual_wirenets[-1].append(solder_blobs[((x, y), (dx, dy))])

                if min(x, y) < 0 or x >= width or y >= height:
                    return

                if array[y][x] not in "-|+#":
                    return

                physical_wirenets[-1].append((x, y))

                if array[y][x] == "+":
                    if abs(dx):
                        array[y][x] = "|"
                        connect_to(x, y, dx, 0)
                    if abs(dy):
                        array[y][x] = "-"
                        connect_to(x, y, 0, dy)
                elif array[y][x] == "-":
                    if abs(dx):
                        array[y][x] = " "
                        connect_to(x, y, dx, 0)
                elif array[y][x] == "|":
                    if abs(dy):
                        array[y][x] = " "
                        connect_to(x, y, 0, dy)
                elif array[y][x] == "#":
                    array[y][x] = " "
                    connect_to(x, y, 1, 0)
                    connect_to(x, y, -1, 0)
                    connect_to(x, y, 0, 1)
                    connect_to(x, y, 0, -1)

            if array[y][x] == "+":
                array[y][x] = "-"
                connect_to(x, y, 0, 1)
                connect_to(x, y, 0, -1)
            elif array[y][x] == "-":
                array[y][x] = " "
                connect_to(x, y, 1, 0)
                connect_to(x, y, -1, 0)
            elif array[y][x] == "|":
                array[y][x] = " "
                connect_to(x, y, 0, 1)
                connect_to(x, y, 0, -1)
            elif array[y][x] == "#":
                array[y][x] = " "
                connect_to(x, y, 1, 0)
                connect_to(x, y, -1, 0)
                connect_to(x, y, 0, 1)
                connect_to(x, y, 0, -1)

        for y, line in enumerate(array):
            for x, char in enumerate(line):
                if char in "-|+#":
                    eat_away(x, y)
                    if array[y][x] in "-|":
                        eat_away(x, y)

        circuit = cls.from_wirenets(virtual_wirenets, nodes)

        circuit.physical_wires = physical_wirenets
        circuit.ascii_wires = [list(string) for string in string_array]

        return circuit

    def ascii_activation(self, f=0):
        raise NotImplementedError()
        plain = [list(string) for string in self.ascii_wires]
        for i, wire in enumerate(self.physical_wires):
            print(self.wire_states)
            if self.wire_states[i]:
                for x, y in wire:
                    if plain[y][x] in "-|":
                        plain[y][x] = "/\\"[(x + y + f) % 2]
        spicy = "\n".join(["".join(s) for s in plain])
        return spicy

    def truth_table(self):
        columns = []
        for input_id in self.inputs.keys():
            columns.append(input_id)
        for output_id in self.outputs:
            columns.append(output_id)

        table = MDTable(columns)

        line_count = 1 << len(self.inputs)
        for combination in range(line_count):
            inputs = dict()

            for i, input_id in enumerate(self.inputs.keys()):
                inputs[input_id] = combination >> (len(self.inputs) - i - 1) & 1

            outputs = self.iterate(inputs)

            table.add_row(inputs | outputs)

        return table


if __name__ == "__main__":
    string = """
. #---#  #--#       
A-+-#-xor# #xor--Sum
  | |    | |        
B-#-and# #-+#       
       |   #and#    
CIn----+---#   |    
       |     #-#    
       #-----or-COut
    """
    # """
    # x#-#
    # y+#and---and
    #  ||
    #  #+#
    #  |#or-----or
    #  ||
    #  #+#
    #  |#nand-nand
    #  ||
    #  #+#
    #  |#nor---nor
    #  ||
    #  #+#
    #  |#xor---xor
    #  ||
    #  #+#
    #   #xnor-xnor
    #     """

    circuit = Circuit.from_string(string)

    print(circuit.truth_table())

    # print(circuit.ascii_activation())
