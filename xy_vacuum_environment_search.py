import os.path
from tkinter import *
from agents import *
from search import *
import sys
import copy
from utils import PriorityQueue

"""
1- BFS: Breadth first search. Using tree or graph version, whichever makes more sense for the problem
2- DFS: Depth-First search. Again using tree or graph version.
3- UCS: Uniform-Cost-Search. Using the following cost function to optimise the path, from initial to current state.
4- A*:  Using A star search.
"""
searchTypes = ['None', 'BFS', 'DFS', 'UCS', 'A*']


class VacuumPlanning(Problem):
    """ The problem of find the next room to clean in a grid of m x n rooms.
    A state is represented by state of the grid. Each room is specified by index set
    (i, j), i in range(m) and j in range (n). Final goal is to find all dirty rooms. But
     we go by sub-goal, meaning finding next dirty room to clean, at a time."""

    def __init__(self, env, searchtype):
        """ Define goal state and initialize a problem
            initial is a pair (i, j) of where the agent is
            goal is next pair(k, l) where map[k][l] is dirty
        """
        self.solution = None
        self.env = env
        self.state = env.agent.location
        super().__init__(self.state)
        self.map = env.things
        self.searchType = searchtype
        self.agent = env.agent


    def generateSolution(self):
        """ generate search engien based on type of the search chosen by user"""
        self.env.read_env()
        self.state = env.agent.location
        super().__init__(self.state)
        if self.searchType == 'BFS':
            path, explored = breadth_first_graph_search(self)
            if path is not None:
                sol = path.solution()
                self.env.set_solution(sol)
                self.env.counter = len(sol)
                self.env.display_explored(explored)
                self.env.display_solution(sol)
            else:
                world = self.get_world()
                self.dirt_locations(world, env.width, env.height, 1)
        elif self.searchType == 'DFS':
            path, explored = depth_first_graph_search(self)
            if path is not None:
                sol = path.solution()
                self.env.set_solution(sol)
                self.env.counter = len(sol)
                self.env.display_explored(explored)
                self.env.display_solution(sol)
            else:
                world = self.get_world()
                self.dirt_locations(world, env.width, env.height, 1)
        elif self.searchType == 'UCS':
            path, explored = best_first_graph_search(self, lambda node: node.path_cost)
            if path is not None:
                sol = path.solution()
                self.env.set_solution(sol)
                self.env.counter = len(sol)
                self.env.display_explored(explored)
                self.env.display_solution(sol)
            else:
                world = self.get_world()
                self.dirt_locations(world, env.width, env.height, 1)
        elif self.searchType == 'A*':
            path, explored = astar_search(self)
            if path is not None:
                sol = path.solution()
                self.env.set_solution(sol)
                self.env.counter = len(sol)
                self.env.display_explored(explored)
                self.env.display_solution(sol)
            else:
                world = self.get_world()
                self.dirt_locations(world, env.width, env.height, 1)
        else:
            raise 'NameError'


    def generateNextSolution(self):
        self.generateSolution()

    def dirt_locations(self, world, w, h, yes_no):
        """
            Find dirt locations on the map.

            world is the list of objects in the map, w = width, h = height, yes_no is a 1 or 0 toggle
            which tells the function to display the dirt locations in a popup as these locations are the
            unreachable dirt.
        """
        self.dirt_list_locations = []
        world = np.array(world, dtype=list)
        world = world.reshape(w-2, h-2)

        i = 0
        while i < w-2:
            j = 0
            while j < h-2:
                if str(world[i][j]) == '[<Dirt>]':
                    self.dirt_list_locations.append((i+1, j+1))
                j = j + 1
            i = i + 1

        if yes_no == 1:
            if self.dirt_list_locations == []:
                pass
            else:
                print("The dirt room(s) at location(s)", self.dirt_list_locations, "were not reachable.")

                top = Toplevel(win)
                top.title("Unreachable Dirt")
                label = Label(top, text="The dirt room(s) at location(s) " + str(self.dirt_list_locations) + " were not reachable.", font='Times 14')
                label.pack()

                env.dirtCount = 0
                env.step()


    def get_world(self):
        """Returns all the items in the map"""
        result = []
        x_start, y_start = (1, 1)
        x_end, y_end = self.env.width-1, self.env.height-1
        for x in range(x_start, x_end):
            row = []
            for y in range(y_start, y_end):
                row.append(self.env.list_things_at((x, y)))
            result.append(row)
        return result

    def actions(self, state):
        """ Return the actions that can be executed in the given state.
        The result would be a list, since there are only four possible actions
        in any given state of the environment """

        possible_actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        possible_neighbors = self.env.things_near(state)

        # Check if there is a wall, if yes remove that action that faces the wall
        for child in possible_neighbors:
            x, y, z = child

            l = state

            if str(x) == "<Wall>":
                x1, y1 = l
                x2, y2 = y

                x3 = x1-x2
                y3 = y1-y2

                if x3 <= -1:
                    if 'LEFT' in possible_actions:
                        possible_actions.remove('LEFT')
                elif x3 >= 1:
                    if 'RIGHT' in possible_actions:
                        possible_actions.remove('RIGHT')
                elif y3 >= 1:
                    if 'UP' in possible_actions:
                        possible_actions.remove('UP')
                elif y3 <= -1:
                    if 'DOWN' in possible_actions:
                        possible_actions.remove('DOWN')

        return possible_actions



    def result(self, state, action):
        """ Given state and action, return a new state that is the result of the action.
        Action is assumed to be a valid action in the state """
        new_state = list(state)
        x, y = new_state


        if action == 'UP':
            y = y - 1
        elif action == 'DOWN':
            y = y + 1
        elif action == 'LEFT':
            x = x + 1
        elif action == 'RIGHT':
            x = x - 1

        new_state = [x, y]
        return new_state


    def goal_test(self, state):
        """ Given a state, return True if state is a goal state or False, otherwise """
        return self.env.some_things_at(state, Dirt)


    def path_cost(self, c, state1, action, state2):
        """To be used for UCS and A* search. Returns the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. For our problem
        state is (x, y) coordinate pair. To make our problem more interesting we are going to associate
        a height to each state as z = sqrt(x*x + y*y). This effectively means our grid is a bowl shape and
        the center of the grid is the center of the bowl. So now the distance between 2 states become the
        square of Euclidean distance as distance = (x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2"""

        x1, y1 = state1
        x2, y2 = state2

        z1 = np.sqrt(x1*x1 + y1*y1)
        z2 = np.sqrt(x2*x2 + y2*y2)

        #  Euclidean distance
        e_distance = (x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2
        return c + e_distance

    def h(self, node):
        """ to be used for A* search. Return the heuristic value for a given state. For this problem use minimum Manhattan
        distance to all the dirty rooms + absolute value of height distance as described above in path_cost() function. .
        """

        # Get dirt locations
        world = self.get_world()
        self.dirt_locations(world, env.width, env.height, 0)

        dirt = self.dirt_list_locations

        min_d = 1000000

        # Go through each dirt location and find the smallest distance.
        for (x, y) in dirt:
            x1, y1 = node.state
            x2, y2 = x, y
            z1 = np.sqrt(x1 * x1 + y1 * y1)
            z2 = np.sqrt(x2 * x2 + y2 * y2)

            e_distance = (x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2
            if e_distance < min_d:
                min_d = e_distance

        return min_d

# ______________________________________________________________________________


def agent_label(agt):
    """creates a label based on direction"""
    dir = agt.direction
    lbl = '^'
    if dir.direction == Direction.D:
        lbl = 'v'
    elif dir.direction == Direction.L:
        lbl = '<'
    elif dir.direction == Direction.R:
        lbl = '>'

    return lbl


def is_agent_label(lbl):
    """determines if the label is one of the labels tht agents have: ^ v < or >"""
    return lbl == '^' or lbl == 'v' or lbl == '<' or lbl == '>'


class Gui(VacuumEnvironment):
    """This is a two-dimensional GUI environment. Each location may be
    dirty, clean or can have a wall. The user can change these at each step.
    """
    xi, yi = (0, 0)

    perceptible_distance = 1

    def __init__(self, root, width, height):
        self.counter = 0
        self.dirt_location_list = []
        self.searchAgent = None
        print("creating xv with width ={} and height={}".format(width, height))
        super().__init__(width, height)

        self.agent = None
        self.root = root
        self.create_frames(height)
        self.create_buttons(width)
        self.create_walls()
        self.setupTestEnvironment()
        self.sol_path = []

    def setupTestEnvironment(self):
        """ first reset the agent"""

        if self.agent is not None:
            x = self.width // 2
            y = self.height // 2
            self.agent.location = (x, y)
            self.buttons[y][x].config(bg='white', text='^', state='normal')
            self.searchType = searchTypes[0]
            self.agent.performance = 0

        """next create a random number of block walls inside the grid as well"""
        roomCount = (self.width - 1) * (self.height - 1)
        blockCount = random.choice(range(roomCount//10, roomCount//5))
        for _ in range(blockCount):
            rownum = random.choice(range(1, self.height - 1))
            colnum = random.choice(range(1, self.width - 1))

            while rownum == self.width // 2 & colnum == self.height // 2:
                rownum = random.choice(range(1, self.height - 1))
                colnum = random.choice(range(1, self.width - 1))

            self.buttons[rownum][colnum].config(bg='red', text='W', disabledforeground='black')

        self.create_dirts()
        self.stepCount = 0
        self.searchType = None
        self.solution = []
        self.explored = set()
        self.read_env()

    def create_frames(self, h):
        """Adds h row frames to the GUI environment."""
        self.frames = []
        for _ in range(h):
            frame = Frame(self.root, bg='blue')
            frame.pack(side='bottom')
            self.frames.append(frame)

    def create_buttons(self, w):
        """Adds w buttons to the respective row frames in the GUI."""
        self.buttons = []
        for frame in self.frames:
            button_row = []
            for _ in range(w):
                button = Button(frame, bg='white', state='normal', height=1, width=1, padx=1, pady=1)
                button.config(command=lambda btn=button: self.toggle_element(btn))
                button.pack(side='left')
                button_row.append(button)
            self.buttons.append(button_row)

    def create_walls(self):
        """Creates the outer boundary walls which do not move. Also create a random number of
        internal blocks of walls."""
        for row, button_row in enumerate(self.buttons):
            if row == 0 or row == len(self.buttons) - 1:
                for button in button_row:
                    button.config(bg='red', text='W', state='disabled', disabledforeground='black')
            else:
                button_row[0].config(bg='red', text='W', state='disabled', disabledforeground='black')
                button_row[len(button_row) - 1].config(bg='red', text='W', state='disabled', disabledforeground='black')

    def create_dirts(self):
        """ set a small random number of rooms to be dirty at random location on the grid
        This function should be called after create_walls()"""
        self.read_env()   # this is needed to make sure wall objects are created
        roomCount = (self.width-1) * (self.height-1)
        self.dirtCount = random.choice(range(5, 15))
        dirtCreated = 0
        while dirtCreated != self.dirtCount:
            rownum = random.choice(range(1, self.height-1))
            colnum = random.choice(range(1, self.width-1))

            while rownum == self.width // 2 & colnum == self.height // 2:
                rownum = random.choice(range(1, self.height - 1))
                colnum = random.choice(range(1, self.width - 1))

            if self.some_things_at((colnum, rownum)):
                continue
            self.buttons[rownum][colnum].config(bg='grey')
            dirtCreated += 1

    def setSearchEngine(self, choice):
        """sets the chosen search engine for solving this problem"""
        self.searchType = choice
        self.searchAgent = VacuumPlanning(self, self.searchType)
        self.searchAgent.generateSolution()
        self.done = False

    def set_solution(self, sol):
        self.solution = list(reversed(sol))


    def display_explored(self, explored):
        """display explored slots in a light pink color"""
        if len(self.explored) > 0:  # means we have explored list from previous search. clear it.
            for (x, y) in self.explored:
                self.buttons[y][x].config(bg='white')

        self.explored = explored
        for (x, y) in explored:
            self.buttons[y][x].config(bg='pink')

    def display_solution(self, solution):
        """
            Display solution path in color cyan.

            Calculates solution path x y values and colors the button at that location cyan
        """

        ax, ay = self.agent.location
        length = len(solution)

        self.sol_path = []
        self.sol_path.append(self.agent.location)
        for child in solution:
            if length == 1:
                continue

            if child == 'UP':
                ay = ay - 1
            elif child == 'DOWN':
                ay = ay + 1
            elif child == 'LEFT':
                ax = ax + 1
            elif child == 'RIGHT':
                ax = ax - 1

            tmp = (ax, ay)
            self.sol_path.append(tmp)
            length = length - 1

        for (x, y) in self.sol_path:
            self.buttons[y][x].config(bg='cyan')



    def add_agent(self, agt, loc):
        """add an agent to the GUI"""
        self.add_thing(agt, loc)
        # Place the agent at the provided location.
        lbl = agent_label(agt)
        self.buttons[loc[1]][loc[0]].config(bg='white', text=lbl, state='normal')
        self.agent = agt

    def toggle_element(self, button):
        """toggle the element type on the GUI."""
        bgcolor = button['bg']
        txt = button['text']
        if is_agent_label(txt):
            if bgcolor == 'grey':
                button.config(bg='white', state='normal')
            else:
                button.config(bg='grey')
        else:
            if bgcolor == 'red':
                button.config(bg='grey', text='')
            elif bgcolor == 'grey':
                button.config(bg='white', text='', state='normal')
            elif bgcolor == 'white':
                button.config(bg='red', text='W')

    def execute_action(self, agent, action):
        """Determines the action the agent performs."""
        if self.searchAgent == None:
            print("Please select a search method in the menu.")

        xi, yi = agent.location
        color = 'cyan'

        if self.stepCount == 0:
            self.stepCount = 1

        self.counter = self.counter - 1

        if self.counter == 0:
            color = 'grey'

        if action == 'Suck':
            if self.stepCount == 1:
                self.agent.performance = -1
            else:
                print("Agent at location", xi, yi, "performs action", action)
                env.dirtCount = env.dirtCount - 1
                agent.performance += 100
                self.buttons[yi][xi].config(state='normal', bg='white')

        elif action == 'UP':
            print("Agent at location", xi, yi, "performs action DOWN")
            agent.performance -= 1
            self.buttons[yi][xi].config(text='', state='normal', bg='cyan')
            yi = yi - 1
            self.buttons[yi][xi].config(text='v', state='normal', bg=color)
            self.agent.location = xi, yi
        elif action == 'DOWN':
            print("Agent at location", xi, yi, "performs action UP")
            agent.performance -= 1
            self.buttons[yi][xi].config(text='', state='normal', bg='cyan')
            yi = yi + 1
            self.buttons[yi][xi].config(text='^', state='normal', bg=color)
            self.agent.location = xi, yi
        elif action == 'LEFT':
            print("Agent at location", xi, yi, "performs action RIGHT")
            agent.performance -= 1
            self.buttons[yi][xi].config(text='', state='normal', bg='cyan')
            xi = xi + 1
            self.buttons[yi][xi].config(text='>', state='normal', bg=color)
            self.agent.location = xi, yi
        elif action == 'RIGHT':
            print("Agent at location", xi, yi, "performs action LEFT")
            agent.performance -= 1
            self.buttons[yi][xi].config(text='', state='normal', bg='cyan')
            xi = xi - 1
            self.buttons[yi][xi].config(text='<', state='normal', bg=color)
            self.agent.location = xi, yi

        NumSteps_label.config(text=str(self.stepCount))
        TotalCost_label.config(text=str(self.agent.performance))

    def read_env(self):
        """read_env: This sets proper wall or Dirt status based on bg color"""
        """Reads the current state of the GUI environment."""
        self.dirtCount = 0
        for j, btn_row in enumerate(self.buttons):
            for i, btn in enumerate(btn_row):
                if (j != 0 and j != len(self.buttons) - 1) and (i != 0 and i != len(btn_row) - 1):
                    if self.some_things_at((i, j)):  # and (i, j) != agt_loc:
                        for thing in self.list_things_at((i, j)):
                            if not isinstance(thing, Agent):
                                self.delete_thing(thing)
                    if btn['bg'] == 'grey':  # adding dirt
                        self.add_thing(Dirt(), (i, j))
                        self.dirtCount += 1
                    elif btn['bg'] == 'red':  # adding wall
                        self.add_thing(Wall(), (i, j))

    def update_env(self):
        """Updates the GUI environment according to the current state."""
        self.read_env()
        self.step()
        if self.searchAgent != None:
            self.stepCount += 1

    def step(self):
        """updates the environment one step. Currently it is associated with one click of 'Step' button.
        """
        if env.dirtCount == 0:
            print("All cleanable rooms are cleaned. DONE!")
            self.done = True
            env.display_explored(set())

            # Clear explored path / solution path
            if len(self.sol_path) > 0:  # means we have explored list from previous search. So need to clear their visual fist
                for (x3, y3) in self.sol_path:
                    self.buttons[y3][x3].config(bg='white')
            return

        if len(self.solution) == 0:
            self.execute_action(self.agent, 'Suck')
            self.read_env()
            if env.dirtCount > 0 and self.searchAgent is not None:
                self.searchAgent.generateNextSolution()
                self.running = False
        else:
            move = self.solution.pop()
            self.execute_action(self.agent, move)


    def run(self, delay=0.25):
        """Run the Environment for given number of time steps,"""
        steps = 2000
        self.done = False

        for stepnum in range(steps):
            sleep(delay)
            if self.done:
                break
            self.update_env()
            Tk.update(self.root)




    def reset_env(self):
        """Resets the GUI and agents environment to the initial clear state."""
        self.running = False
        NumSteps_label.config(text=str(0))
        TotalCost_label.config(text=str(0))

        for j, btn_row in enumerate(self.buttons):
            for i, btn in enumerate(btn_row):
                if (j != 0 and j != len(self.buttons) - 1) and (i != 0 and i != len(btn_row) - 1):
                    if self.some_things_at((i, j)):
                        for thing in self.list_things_at((i, j)):
                            self.delete_thing(thing)
                    btn.config(bg='white', text='', state='normal')

        self.setupTestEnvironment()


"""
Our search Agents ignore ignore environment percepts for planning. The planning is done based ons static
 data from environment at the beginning. The environment if fully observable
 """
def XYSearchAgentProgram(percept):
    pass


class XYSearchAgent(Agent):
    """The modified SimpleRuleAgent for the GUI environment."""

    def __init__(self, program, loc):
        super().__init__(program)
        self.location = loc
        self.direction = Direction("up")
        self.searchType = searchTypes[0]
        self.stepCount = 0


if __name__ == "__main__":
    win = Tk()
    win.title("Searching Cleaning Robot")
    win.geometry("800x750+50+50")
    win.resizable(True, True)
    frame = Frame(win, bg='black')
    frame.pack(side='bottom')
    topframe = Frame(win, bg='black')
    topframe.pack(side='top')

    wid = 10
    if len(sys.argv) > 1:
        wid = int(sys.argv[1])

    hig = 10
    if len(sys.argv) > 2:
        hig = int(sys.argv[2])

    env = Gui(win, wid, hig)

    theAgent = XYSearchAgent(program=XYSearchAgentProgram, loc=(hig//2, wid//2))
    x, y = theAgent.location
    env.add_agent(theAgent, (y, x))

    NumSteps_label = Label(topframe, text='NumSteps: 0', bg='green', fg='white', bd=2, padx=2, pady=2)
    NumSteps_label.pack(side='left')
    TotalCost_label = Label(topframe, text='TotalCost: 0', bg='blue', fg='white', padx=2, pady=2)
    TotalCost_label.pack(side='right')
    reset_button = Button(frame, text='Reset', height=2, width=5, padx=2, pady=2)
    reset_button.pack(side='left')
    next_button = Button(frame, text='Next', height=2, width=5, padx=2, pady=2)
    next_button.pack(side='left')
    run_button = Button(frame, text='Run', height=2, width=5, padx=2, pady=2)
    run_button.pack(side='left')

    next_button.config(command=env.update_env)
    reset_button.config(command=env.reset_env)
    run_button.config(command=env.run)

    searchTypeStr = StringVar(win)
    searchTypeStr.set(searchTypes[0])
    searchTypeStr_dropdown = OptionMenu(frame, searchTypeStr, *searchTypes, command=env.setSearchEngine)
    searchTypeStr_dropdown.pack(side='left')

    win.mainloop()
