# Must declare type of each field, but can't assign as part of declaration.
# use constructor to assign
import random


@value  # includes a default constructor
struct Grid(StringableRaising):
    var rows: Int
    var cols: Int
    var data: List[List[Int]]

    def __str__(grid: Grid) -> String:
        # Create an empty String
        str = String()

        # Iterate through rows 0 through rows-1
        for row in range(grid.rows):
            # Iterate through columns 0 through cols-1
            for col in range(grid.cols):
                if grid.data[row][col] == 1:
                    str += "*"  # If cell is populated, append an asterisk
                else:
                    str += " "  # If cell is not populated, append a space
            if row != grid.rows - 1:
                str += "\n"  # Add a newline between rows, but not at the end
        return str

    def __getitem__(self, row: Int, col: Int) -> Int:
        return self.data[row][col]

    def __setitem__(mut self, row: Int, col: Int, val: Int) -> None:
        self.data[row][col] = val

    @staticmethod
    def random(rows: Int, cols: Int) -> Self:
        random.seed()

        data = List[List[Int]]()

        for _row in range(rows):
            row_data = List[Int]()
            for _col in range(cols):
                row_data.append(Int(random.random_si64(0, 1)))
            data.append(row_data)

        return Self(rows, cols, data)

    def evolve(self) -> Self:
        next_generation = List[List[Int]]()

        for row in range(self.rows):
            row_data = List[Int]()

            # Calculate neighboring row indices, handling "wrap-around"
            row_above = (row - 1) % self.rows
            row_below = (row + 1) % self.rows

            for col in range(self.cols):
                # Calculate neighboring column indices, handling "wrap-around"
                col_left = (col - 1) % self.cols
                col_right = (col + 1) % self.cols

                # Determine number of populated cells around the current cell
                num_neighbors = (
                    self[row_above, col_left]
                    + self[row_above, col]
                    + self[row_above, col_right]
                    + self[row, col_left]
                    + self[row, col_right]
                    + self[row_below, col_left]
                    + self[row_below, col]
                    + self[row_below, col_right]
                )

                # Determine the state of the current cell for the next generation
                new_state = 0
                if self[row, col] == 1 and (
                    num_neighbors == 2 or num_neighbors == 3
                ):
                    new_state = 1
                elif self[row, col] == 0 and num_neighbors == 3:
                    new_state = 1
                row_data.append(new_state)

            next_generation.append(row_data)

        return Self(self.rows, self.cols, next_generation)
