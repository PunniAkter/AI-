import time
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, colorchooser
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation
from PIL import Image, ImageTk
import threading
import random
import uuid
import os


class NQueens:
    def __init__(self, n):
        self.n = n
        self.solutions = []
        self.steps = 0
        self.execution_time = 0
        self.current_board = [[0 for _ in range(n)] for _ in range(n)]
        self.current_col = 0
        self.solution_path = []

    def reset(self):
        self.solutions = []
        self.steps = 0
        self.execution_time = 0
        self.current_board = [[0 for _ in range(self.n)] for _ in range(self.n)]
        self.current_col = 0
        self.solution_path = []

    def is_safe(self, board, row, col):
        for i in range(col):
            if board[row][i] == 1:
                return False
        for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
            if board[i][j] == 1:
                return False
        for i, j in zip(range(row, self.n), range(col, -1, -1)):
            if board[i][j] == 1:
                return False
        return True

    def solve_backtracking(self, track_path=False):
        self.reset()
        start_time = time.time()
        board = [[0 for _ in range(self.n)] for _ in range(self.n)]
        self._backtrack(board, 0, track_path)
        self.execution_time = time.time() - start_time
        return self.solutions

    def _backtrack(self, board, col, track_path=False):
        if col >= self.n:
            solution = [row[:] for row in board]
            self.solutions.append(solution)
            return True
        result = False
        for row in range(self.n):
            self.steps += 1
            if self.is_safe(board, row, col):
                board[row][col] = 1
                if track_path:
                    snapshot = [row[:] for row in board]
                    self.solution_path.append((snapshot, col, row))
                result = self._backtrack(board, col + 1, track_path) or result
                board[row][col] = 0
                if track_path:
                    snapshot = [row[:] for row in board]
                    self.solution_path.append((snapshot, col, -1))
        return result

    def count_conflicts(self, board, row, col):
        conflicts = 0
        for i in range(self.n):
            if i != col and board[row][i] == 1:
                conflicts += 1
        for i in range(self.n):
            if i != row and board[i][col] == 1:
                conflicts += 1
        for i in range(self.n):
            for j in range(self.n):
                if (i != row or j != col) and board[i][j] == 1:
                    if abs(i - row) == abs(j - col):
                        conflicts += 1
        return conflicts

    def solve_heuristic(self, track_path=False):
        self.reset()
        start_time = time.time()
        board = [[0 for _ in range(self.n)] for _ in range(self.n)]
        self._heuristic_backtrack(board, 0, track_path)
        self.execution_time = time.time() - start_time
        return self.solutions

    def _heuristic_backtrack(self, board, col, track_path=False):
        if col >= self.n:
            solution = [row[:] for row in board]
            self.solutions.append(solution)
            return True
        conflicts = []
        for row in range(self.n):
            self.steps += 1
            if self.is_safe(board, row, col):
                conflicts.append((row, self.count_conflicts(board, row, col)))
            else:
                conflicts.append((row, float('inf')))
        conflicts.sort(key=lambda x: x[1])
        result = False
        for row, conf in conflicts:
            if conf != float('inf'):
                board[row][col] = 1
                if track_path:
                    snapshot = [row[:] for row in board]
                    self.solution_path.append((snapshot, col, row))
                result = self._heuristic_backtrack(board, col + 1, track_path) or result
                board[row][col] = 0
                if track_path:
                    snapshot = [row[:] for row in board]
                    self.solution_path.append((snapshot, col, -1))
        return result

    def solve_genetic_algorithm(self, population_size=100, generations=1000):
        self.reset()
        start_time = time.time()
        population = [np.random.permutation(self.n).tolist() for _ in range(population_size)]
        best_fitness = float('-inf')
        best_solution = None

        for generation in range(generations):
            fitness_scores = []
            for chromosome in population:
                fitness = self._calculate_fitness(chromosome)
                fitness_scores.append(fitness)
                self.steps += 1
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = chromosome.copy()
                    if fitness == (self.n * (self.n - 1)) // 2:
                        break
            if best_fitness == (self.n * (self.n - 1)) // 2:
                break

            new_population = []
            elite_size = max(1, population_size // 10)
            sorted_indices = np.argsort(fitness_scores)[::-1]
            for i in range(elite_size):
                new_population.append(population[sorted_indices[i]].copy())

            while len(new_population) < population_size:
                parent1 = population[self._tournament_selection(fitness_scores)]
                parent2 = population[self._tournament_selection(fitness_scores)]
                child = self._order_crossover(parent1, parent2)
                if random.random() < 0.2:
                    idx1, idx2 = random.sample(range(self.n), 2)
                    child[idx1], child[idx2] = child[idx2], child[idx1]
                new_population.append(child)
            population = new_population

        if best_solution:
            board = [[0 for _ in range(self.n)] for _ in range(self.n)]
            for col, row in enumerate(best_solution):
                board[row][col] = 1
            if self.verify_solution(board):
                self.solutions.append(board)

        self.execution_time = time.time() - start_time
        return self.solutions

    def _calculate_fitness(self, chromosome):
        attacks = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if abs(chromosome[i] - chromosome[j]) == j - i:
                    attacks += 1
        max_attacks = (self.n * (self.n - 1)) // 2
        return max_attacks - attacks

    def _tournament_selection(self, fitness_scores, tournament_size=3):
        tournament_indices = random.sample(range(len(fitness_scores)), tournament_size)
        return max(tournament_indices, key=lambda idx: fitness_scores[idx])

    def _order_crossover(self, parent1, parent2):
        n = len(parent1)
        start, end = sorted(random.sample(range(n), 2))
        child = [-1] * n
        for i in range(start, end + 1):
            child[i] = parent1[i]
        remaining = [x for x in parent2 if x not in child[start:end + 1]]
        j = 0
        for i in range(n):
            if child[i] == -1:
                child[i] = remaining[j]
                j += 1
        return child

    def get_safe_positions(self, col):
        return [row for row in range(self.n) if self.is_safe(self.current_board, row, col)]

    def get_heuristic_ranked_positions(self, col):
        conflicts = [(row, self.count_conflicts(self.current_board, row, col))
                     for row in range(self.n) if self.is_safe(self.current_board, row, col)]
        conflicts.sort(key=lambda x: x[1])
        return [row for row, _ in conflicts]

    def place_queen_manually(self, row, col):
        if col < self.n and self.is_safe(self.current_board, row, col):
            self.current_board[row][col] = 1
            self.current_col += 1
            return True
        return False

    def is_solution_complete(self):
        return self.current_col >= self.n

    def get_current_board(self):
        return self.current_board

    def undo_last_move(self):
        if self.current_col > 0:
            self.current_col -= 1
            for row in range(self.n):
                if self.current_board[row][self.current_col] == 1:
                    self.current_board[row][self.current_col] = 0
                    break
            return True
        return False

    def verify_solution(self, board):
        for col in range(self.n):
            queen_count = sum(board[row][col] for row in range(self.n))
            if queen_count != 1:
                return False
        for row in range(self.n):
            queen_count = sum(board[row][col] for col in range(self.n))
            if queen_count != 1:
                return False
        for row in range(self.n):
            for col in range(self.n):
                if board[row][col] == 1:
                    for i, j in zip(range(row + 1, self.n), range(col + 1, self.n)):
                        if board[i][j] == 1:
                            return False
                    for i, j in zip(range(row - 1, -1, -1), range(col + 1, self.n)):
                        if board[i][j] == 1:
                            return False
                    for i, j in zip(range(row + 1, self.n), range(col - 1, -1, -1)):
                        if board[i][j] == 1:
                            return False
                    for i, j in zip(range(row - 1, -1, -1), range(col - 1, -1, -1)):
                        if board[i][j] == 1:
                            return False
        return True


class ThemeManager:
    def __init__(self):
        self.themes = {
            "Classic": ("#ffffff", "#4a4a4a", "#ffd700", "#00cc00", "#000000", "#f0f4f8"),
            "Ocean": ("#a6cced", "#1e5484", "#ffffff", "#00ffff", "#003366", "#e6f2ff"),
            "Forest": ("#c8e6c9", "#2e7d32", "#8d6e63", "#4caf50", "#33691e", "#ebf5eb"),
            "Royal": ("#e6d7ff", "#6a1b9a", "#ff5722", "#9c27b0", "#4a148c", "#f3e5f5"),
            "Desert": ("#f9e0c3", "#b35900", "#ffffff", "#ff9800", "#994d00", "#fff3e0"),
            "Neon": ("#222222", "#000000", "#39ff14", "#ff00ff", "#00ffff", "#121212"),
            "Coffee": ("#d7ccc8", "#5d4037", "#fafafa", "#8d6e63", "#3e2723", "#efebe9"),
            "Custom": ("#ffffff", "#4a4a4a", "#ffd700", "#00cc00", "#000000", "#f0f4f8")
        }
        self.current_theme = "Classic"

    def get_theme(self):
        return self.themes[self.current_theme]

    def set_theme(self, theme_name):
        if theme_name in self.themes:
            self.current_theme = theme_name

    def set_custom_theme(self, light_square, dark_square, queen_color, highlight_color, text_color, bg_color):
        self.themes["Custom"] = (light_square, dark_square, queen_color, highlight_color, text_color, bg_color)
        self.current_theme = "Custom"


class NQueensApp:
    def __init__(self, root):
        self.root = root
        self.root.title("N-Queens Solver - Enhanced Edition")
        self.root.geometry("1280x900")
        self.root.minsize(1000, 700)
        self.theme_manager = ThemeManager()
        self.apply_theme("Classic")

        self.board_size = tk.IntVar(value=8)
        self.algorithm = tk.StringVar(value="Backtracking")
        self.mode = tk.StringVar(value="Automatic")
        self.solver = NQueens(self.board_size.get())
        self.selected_solution = tk.IntVar(value=0)
        self.solution_count = tk.IntVar(value=0)
        self.animation_speed = tk.DoubleVar(value=0.5)
        self.animation_running = False
        self.theme = tk.StringVar(value="Classic")
        self.queen_style = tk.StringVar(value="Classic")
        self.show_threats = tk.BooleanVar(value=False)
        self.show_column_indicators = tk.BooleanVar(value=True)
        self.progress = tk.DoubleVar(value=0)
        self.is_solving = False
        self.anim_index = 0
        self.anim = None
        self.animation_lock = threading.Lock()

        self.queen_styles = {
            "Classic": "♛",
            "Modern": "♕",
            "Star": "★",
            "Crown": "👑",
            "Chess": "♚"
        }

        self.create_main_frames()
        self.create_control_frame()
        self.create_board_frame()
        self.create_status_frame()
        self.create_visualization_frame()
        self.update_board_display()

        self.root.bind("<Left>", lambda event: self.prev_solution())
        self.root.bind("<Right>", lambda event: self.next_solution())
        self.root.bind("<space>", lambda event: self.toggle_animation())
        self.root.bind("<Escape>", lambda event: self.reset())

    def apply_theme(self, theme_name):
        self.theme_manager.set_theme(theme_name)
        light_square, dark_square, queen_color, highlight_color, text_color, bg_color = self.theme_manager.get_theme()
        self.root.configure(bg=bg_color)
        style = ttk.Style()
        style.configure('TLabel', background=bg_color, foreground=text_color, font=('Arial', 12))
        style.configure('TButton', font=('Arial', 11, 'bold'), padding=8, background=bg_color, foreground=text_color)
        style.configure('Accent.TButton', background=highlight_color, foreground=text_color)
        style.configure('TCombobox', font=('Arial', 11), padding=5, fieldbackground=bg_color, foreground=text_color)
        style.configure('TFrame', background=bg_color)
        style.configure('TLabelframe', background=bg_color, foreground=text_color)
        style.configure('TLabelframe.Label', background=bg_color, foreground=text_color, font=('Arial', 12, 'bold'))
        style.configure('TCheckbutton', background=bg_color, foreground=text_color)
        style.configure('TScale', background=bg_color, troughcolor=light_square)

    def create_main_frames(self):
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.left_panel = ttk.Frame(self.main_frame)
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self.right_panel = ttk.Frame(self.main_frame)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

    def create_control_frame(self):
        control_frame = ttk.LabelFrame(self.left_panel, text="Controls", padding=(10, 5))
        control_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))

        row1 = ttk.Frame(control_frame)
        row1.pack(fill=tk.X, pady=5)
        ttk.Label(row1, text="Board Size:").pack(side=tk.LEFT, padx=(10, 5))
        size_combo = ttk.Combobox(row1, textvariable=self.board_size, values=list(range(4, 17)), width=4,
                                  state='readonly')
        size_combo.pack(side=tk.LEFT, padx=(0, 15))
        size_combo.bind("<<ComboboxSelected>>", self.on_size_changed)
        size_combo.bind('<Enter>', lambda e: size_combo.configure(foreground=self.theme_manager.get_theme()[3]))
        size_combo.bind('<Leave>', lambda e: size_combo.configure(foreground=self.theme_manager.get_theme()[4]))

        ttk.Label(row1, text="Algorithm:").pack(side=tk.LEFT, padx=(10, 5))
        algorithm_combo = ttk.Combobox(row1, textvariable=self.algorithm,
                                       values=["Backtracking", "Heuristic", "Genetic Algorithm"], width=15,
                                       state='readonly')
        algorithm_combo.pack(side=tk.LEFT, padx=(0, 15))
        algorithm_combo.bind('<Enter>', lambda e: algorithm_combo.configure(foreground=self.theme_manager.get_theme()[3]))
        algorithm_combo.bind('<Leave>',
                             lambda e: algorithm_combo.configure(foreground=self.theme_manager.get_theme()[4]))

        ttk.Label(row1, text="Mode:").pack(side=tk.LEFT, padx=(10, 5))
        mode_combo = ttk.Combobox(row1, textvariable=self.mode, values=["Automatic", "Manual", "Animation"], width=10,
                                  state='readonly')
        mode_combo.pack(side=tk.LEFT)
        mode_combo.bind("<<ComboboxSelected>>", self.on_mode_changed)
        mode_combo.bind('<Enter>', lambda e: mode_combo.configure(foreground=self.theme_manager.get_theme()[3]))
        mode_combo.bind('<Leave>', lambda e: mode_combo.configure(foreground=self.theme_manager.get_theme()[4]))

        row2 = ttk.Frame(control_frame)
        row2.pack(fill=tk.X, pady=5)
        self.solve_button = ttk.Button(row2, text="Solve", command=self.solve, style='Accent.TButton')
        self.solve_button.pack(side=tk.LEFT, padx=10)
        self.reset_button = ttk.Button(row2, text="Reset", command=self.reset)
        self.reset_button.pack(side=tk.LEFT, padx=10)
        self.export_button = ttk.Button(row2, text="Export Solution", command=self.export_solution)
        self.export_button.pack(side=tk.LEFT, padx=10)
        self.verify_button = ttk.Button(row2, text="Verify Solution", command=self.verify_solution)
        self.verify_button.pack(side=tk.LEFT, padx=10)

        row3 = ttk.Frame(control_frame)
        row3.pack(fill=tk.X, pady=5)
        self.prev_button = ttk.Button(row3, text="◄ Previous", command=self.prev_solution)
        self.prev_button.pack(side=tk.LEFT, padx=10)
        self.solution_label = ttk.Label(row3, text="Solution: 0/0", font=('Arial', 11, 'bold'))
        self.solution_label.pack(side=tk.LEFT, padx=15)
        self.next_button = ttk.Button(row3, text="Next ►", command=self.next_solution)
        self.next_button.pack(side=tk.LEFT, padx=10)

        self.manual_frame = ttk.LabelFrame(control_frame, text="Manual Controls", padding=(10, 5))
        manual_controls = ttk.Frame(self.manual_frame)
        manual_controls.pack(fill=tk.X, pady=5)
        self.hint_button = ttk.Button(manual_controls, text="Get Hint", command=self.get_hint)
        self.hint_button.pack(side=tk.LEFT, padx=10)
        self.undo_button = ttk.Button(manual_controls, text="Undo", command=self.undo_move)
        self.undo_button.pack(side=tk.LEFT, padx=10)
        self.auto_complete_button = ttk.Button(manual_controls, text="Auto Complete", command=self.auto_complete)
        self.auto_complete_button.pack(side=tk.LEFT, padx=10)

        self.animation_frame = ttk.LabelFrame(control_frame, text="Animation Controls", padding=(10, 5))
        anim_controls = ttk.Frame(self.animation_frame)
        anim_controls.pack(fill=tk.X, pady=5)
        ttk.Label(anim_controls, text="Speed:").pack(side=tk.LEFT, padx=(10, 5))
        speed_scale = ttk.Scale(anim_controls, from_=0.1, to=2.0, variable=self.animation_speed, orient=tk.HORIZONTAL,
                                length=150)
        speed_scale.pack(side=tk.LEFT, padx=(0, 10))
        self.anim_play_button = ttk.Button(anim_controls, text="▶ Play", command=self.toggle_animation)
        self.anim_play_button.pack(side=tk.LEFT, padx=5)
        self.anim_step_button = ttk.Button(anim_controls, text="Step →", command=self.animation_step)
        self.anim_step_button.pack(side=tk.LEFT, padx=5)

        theme_row = ttk.Frame(control_frame)
        theme_row.pack(fill=tk.X, pady=5)
        ttk.Label(theme_row, text="Theme:").pack(side=tk.LEFT, padx=(10, 5))
        theme_combo = ttk.Combobox(theme_row, textvariable=self.theme, values=list(self.theme_manager.themes.keys()),
                                   width=12, state='readonly')
        theme_combo.pack(side=tk.LEFT, padx=(0, 15))
        theme_combo.bind("<<ComboboxSelected>>", self.on_theme_changed)
        ttk.Label(theme_row, text="Queen Style:").pack(side=tk.LEFT, padx=(10, 5))
        queen_combo = ttk.Combobox(theme_row, textvariable=self.queen_style, values=list(self.queen_styles.keys()),
                                   width=10, state='readonly')
        queen_combo.pack(side=tk.LEFT, padx=(0, 15))
        queen_combo.bind("<<ComboboxSelected>>", lambda e: self.update_board_display())
        customize_button = ttk.Button(theme_row, text="Customize Theme", command=self.open_theme_customizer)
        customize_button.pack(side=tk.LEFT, padx=10)

        display_row = ttk.Frame(control_frame)
        display_row.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(display_row, text="Show Threats", variable=self.show_threats,
                        command=self.update_board_display).pack(side=tk.LEFT, padx=10)
        ttk.Checkbutton(display_row, text="Show Column Indicators", variable=self.show_column_indicators,
                        command=self.update_board_display).pack(side=tk.LEFT, padx=10)

        self.progress_bar = ttk.Progressbar(control_frame, variable=self.progress, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=10)

        if self.mode.get() == "Manual":
            self.manual_frame.pack(fill=tk.X, pady=5)
        elif self.mode.get() == "Animation":
            self.animation_frame.pack(fill=tk.X, pady=5)

    def create_board_frame(self):
        self.board_frame = ttk.LabelFrame(self.right_panel, text="Chess Board", padding=(15, 10))
        self.board_frame.pack(fill=tk.BOTH, expand=True)
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.board_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect('button_press_event', self.on_board_click)

    def create_status_frame(self):
        status_frame = ttk.LabelFrame(self.left_panel, text="Statistics", padding=(15, 10))
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))
        grid = ttk.Frame(status_frame)
        grid.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        ttk.Label(grid, text="Steps:", font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky=tk.W, pady=2)
        self.steps_label = ttk.Label(grid, text="0", font=('Arial', 10))
        self.steps_label.grid(row=0, column=1, sticky=tk.W, pady=2)
        ttk.Label(grid, text="Time:", font=('Arial', 10, 'bold')).grid(row=0, column=2, sticky=tk.W, padx=(20, 0),
                                                                      pady=2)
        self.time_label = ttk.Label(grid, text="0.000000 s", font=('Arial', 10))
        self.time_label.grid(row=0, column=3, sticky=tk.W, pady=2)

        ttk.Label(grid, text="Solutions:", font=('Arial', 10, 'bold')).grid(row=1, column=0, sticky=tk.W, pady=2)
        self.solutions_label = ttk.Label(grid, text="0", font=('Arial', 10))
        self.solutions_label.grid(row=1, column=1, sticky=tk.W, pady=2)
        ttk.Label(grid, text="Status:", font=('Arial', 10, 'bold')).grid(row=1, column=2, sticky=tk.W, padx=(20, 0),
                                                                        pady=2)
        self.status_label = ttk.Label(grid, text="Ready", font=('Arial', 10, 'italic'))
        self.status_label.grid(row=1, column=3, sticky=tk.W, pady=2)

        ttk.Label(grid, text="Animation:", font=('Arial', 10, 'bold')).grid(row=2, column=0, sticky=tk.W, pady=2)
        self.anim_status_label = ttk.Label(grid, text="Not running", font=('Arial', 10))
        self.anim_status_label.grid(row=2, column=1, sticky=tk.W, pady=2)
        ttk.Label(grid, text="Current step:", font=('Arial', 10, 'bold')).grid(row=2, column=2, sticky=tk.W,
                                                                              padx=(20, 0), pady=2)
        self.anim_step_label = ttk.Label(grid, text="0/0", font=('Arial', 10))
        self.anim_step_label.grid(row=2, column=3, sticky=tk.W, pady=2)

    def create_visualization_frame(self):
        vis_frame = ttk.LabelFrame(self.left_panel, text="Visualization", padding=(15, 10))
        vis_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        self.vis_fig, self.vis_ax = plt.subplots(figsize=(6, 4))
        self.vis_canvas = FigureCanvasTkAgg(self.vis_fig, master=vis_frame)
        self.vis_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.vis_ax.text(0.5, 0.5, "No data to visualize yet.\nSolve a puzzle to see visualizations.",
                        ha='center', va='center', fontsize=12, color='gray')
        self.vis_ax.set_xticks([])
        self.vis_ax.set_yticks([])
        self.vis_canvas.draw()

        vis_controls = ttk.Frame(vis_frame)
        vis_controls.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(vis_controls, text="Solution Distribution",
                   command=lambda: self.update_visualization('distribution')).pack(side=tk.LEFT, padx=5)
        ttk.Button(vis_controls, text="Queen Heatmap", command=lambda: self.update_visualization('heatmap')).pack(
            side=tk.LEFT, padx=5)

    def update_board_display(self, board=None, animate=False):
        with self.animation_lock:
            self.ax.clear()
            n = self.board_size.get()
            if board is None:
                board = self.solver.get_current_board() if self.mode.get() == "Manual" else [[0 for _ in range(n)] for _ in
                                                                                            range(n)]

            light_square, dark_square, queen_color, highlight_color, text_color, _ = self.theme_manager.get_theme()
            cmap = ListedColormap([light_square, dark_square])
            chess_board = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    chess_board[i, j] = (i + j) % 2
            self.ax.imshow(chess_board, cmap=cmap)

            for i in range(n + 1):
                self.ax.axhline(i - 0.5, color='#7f8c8d', linewidth=1)
                self.ax.axvline(i - 0.5, color='#7f8c8d', linewidth=1)

            if self.show_threats.get():
                for row in range(n):
                    for col in range(n):
                        if board[row][col] == 1:
                            for i, j in zip(range(row + 1, n), range(col + 1, n)):
                                self.ax.add_patch(
                                    plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=True, alpha=0.2, color='#ff4444'))
                            for i, j in zip(range(row - 1, -1, -1), range(col + 1, n)):
                                self.ax.add_patch(
                                    plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=True, alpha=0.2, color='#ff4444'))
                            for i, j in zip(range(row + 1, n), range(col - 1, -1, -1)):
                                self.ax.add_patch(
                                    plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=True, alpha=0.2, color='#ff4444'))
                            for i, j in zip(range(row - 1, -1, -1), range(col - 1, -1, -1)):
                                self.ax.add_patch(
                                    plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=True, alpha=0.2, color='#ff4444'))

            queen_positions = [(i, j) for i in range(n) for j in range(n) if board[i][j] == 1]
            if animate and queen_positions:
                def animate_queen(frame):
                    self.ax.text(queen_positions[frame][1], queen_positions[frame][0],
                                 self.queen_styles[self.queen_style.get()], fontsize=30,
                                 ha='center', va='center', color=queen_color)
                    self.canvas.draw()
                    return []

                self.anim = animation.FuncAnimation(self.fig, animate_queen, frames=len(queen_positions),
                                                   interval=300 / self.animation_speed.get(), blit=True, repeat=False)
            else:
                for i, j in queen_positions:
                    self.ax.text(j, i, self.queen_styles[self.queen_style.get()], fontsize=30,
                                 ha='center', va='center', color=queen_color)

            if self.mode.get() == "Manual" and self.solver.current_col < n and self.show_column_indicators.get():
                self.ax.text(self.solver.current_col, -0.5, f"↑\nCol {self.solver.current_col}",
                             ha='center', va='center', color='#e74c3c', fontsize=12, fontweight='bold')
                safe_positions = self.solver.get_safe_positions(self.solver.current_col)
                for row in safe_positions:
                    self.ax.add_patch(plt.Rectangle((self.solver.current_col - 0.5, row - 0.5),
                                                    1, 1, fill=False, edgecolor=highlight_color, lw=3))

            self.ax.set_xticks([])
            self.ax.set_yticks([])
            title = "N-Queens Puzzle" if self.mode.get() == "Automatic" and not self.solver.solutions else \
                f"{self.algorithm.get()} Solution {self.selected_solution.get() + 1}/{self.solution_count.get()}" if self.mode.get() == "Automatic" else \
                f"Manual Solving - Column {self.solver.current_col}" if self.mode.get() == "Manual" else \
                f"Animation - Step {self.anim_index}/{len(self.solver.solution_path)}"
            self.ax.set_title(title, fontsize=14, pad=15, fontweight='bold', color=text_color)
            self.canvas.draw()

    def update_visualization(self, vis_type):
        self.vis_ax.clear()
        light_square, _, _, _, text_color, _ = self.theme_manager.get_theme()

        if not self.solver.solutions:
            self.vis_ax.text(0.5, 0.5, "No solutions available.\nSolve the puzzle first.",
                             ha='center', va='center', fontsize=12, color=text_color)
            self.vis_ax.set_xticks([])
            self.vis_ax.set_yticks([])
            self.vis_canvas.draw()
            return

        if vis_type == 'distribution':
            solution_counts = []
            sizes = range(4, min(17, self.board_size.get() + 1))
            for n in sizes:
                solver = NQueens(n)
                solver.solve_backtracking()
                solution_counts.append(len(solver.solutions))
            self.vis_ax.bar(sizes, solution_counts, color=light_square, edgecolor=text_color)
            self.vis_ax.set_xlabel('Board Size 👎', color=text_color)
            self.vis_ax.set_ylabel('Number of Solutions', color=text_color)
            self.vis_ax.set_title('Solution Distribution', color=text_color)
            self.vis_ax.set_xticks(list(sizes))
            self.vis_ax.tick_params(colors=text_color)

        elif vis_type == 'heatmap':
            heatmap = np.zeros((self.board_size.get(), self.board_size.get()))
            for solution in self.solver.solutions:
                for i in range(self.board_size.get()):
                    for j in range(self.board_size.get()):
                        if solution[i][j] == 1:
                            heatmap[i][j] += 1
            im = self.vis_ax.imshow(heatmap, cmap='YlOrRd')
            self.vis_ax.set_title('Queen Placement Heatmap', color=text_color)
            self.vis_ax.set_xlabel('Column', color=text_color)
            self.vis_ax.set_ylabel('Row', color=text_color)
            self.vis_canvas.figure.colorbar(im, ax=self.vis_ax, label='Frequency')
            self.vis_ax.tick_params(colors=text_color)

        self.vis_canvas.draw()

    def solve(self):
        if self.is_solving:
            return
        self.is_solving = True
        self.status_label.config(text="Status: Solving...")
        self.progress.set(0)
        self.root.update()

        def solve_thread():
            try:
                n = self.board_size.get()
                algo = self.algorithm.get()
                track_path = self.mode.get() == "Animation"

                def update_progress():
                    if self.is_solving:
                        current_steps = self.solver.steps
                        max_steps = n * n * 100
                        progress = min((current_steps / max_steps) * 100, 95)
                        self.progress.set(progress)
                        self.steps_label.config(text=f"Steps: {current_steps}")
                        self.root.after(100, update_progress)

                self.root.after(100, update_progress)

                if algo == "Backtracking":
                    self.solver.solve_backtracking(track_path=track_path)
                elif algo == "Heuristic":
                    self.solver.solve_heuristic(track_path=track_path)
                else:
                    self.solver.solve_genetic_algorithm()

                self.root.after(0, lambda: self.post_solve())

            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"An error occurred: {str(e)}"))
                self.root.after(0, lambda: self.status_label.config(text="Status: Error"))
                self.is_solving = False

        threading.Thread(target=solve_thread, daemon=True).start()

    def post_solve(self):
        self.is_solving = False
        solutions_count = len(self.solver.solutions)
        self.solution_count.set(solutions_count)
        self.solutions_label.config(text=f"{solutions_count}")
        self.steps_label.config(text=f"Steps: {self.solver.steps}")
        self.time_label.config(text=f"Time: {self.solver.execution_time:.6f} s")
        self.progress.set(100)

        if solutions_count > 0:
            self.selected_solution.set(0)
            self.solution_label.config(text=f"Solution: 1/{solutions_count}")
            self.update_board_display(self.solver.solutions[0], animate=True)
            self.status_label.config(text="Status: Solution found")
            self.update_visualization('distribution')
        else:
            self.status_label.config(text="Status: No solution found")
        self.anim_index = 0
        self.anim_step_label.config(text=f"0/{len(self.solver.solution_path)}")

    def reset(self):
        if self.is_solving:
            return
        self.solver = NQueens(self.board_size.get())
        self.selected_solution.set(0)
        self.solution_count.set(0)
        self.progress.set(0)
        self.anim_index = 0
        self.animation_running = False
        self.anim_play_button.config(text="▶ Play")
        self.anim_status_label.config(text="Not running")
        self.anim_step_label.config(text="0/0")
        self.update_board_display()
        self.steps_label.config(text="Steps: 0")
        self.time_label.config(text="Time: 0.000000 s")
        self.solutions_label.config(text="0")
        self.solution_label.config(text="Solution: 0/0")
        self.status_label.config(text="Status: Ready")
        if self.anim:
            self.anim.event_source.stop()
            self.anim = None

    def prev_solution(self):
        if self.is_solving:
            return
        current = self.selected_solution.get()
        solutions_count = self.solution_count.get()
        if solutions_count > 0 and current > 0:
            current -= 1
            self.selected_solution.set(current)
            self.solution_label.config(text=f"Solution: {current + 1}/{solutions_count}")
            self.update_board_display(self.solver.solutions[current], animate=True)

    def next_solution(self):
        if self.is_solving:
            return
        current = self.selected_solution.get()
        solutions_count = self.solution_count.get()
        if solutions_count > 0 and current < solutions_count - 1:
            current += 1
            self.selected_solution.set(current)
            self.solution_label.config(text=f"Solution: {current + 1}/{solutions_count}")
            self.update_board_display(self.solver.solutions[current], animate=True)

    def on_size_changed(self, event):
        self.reset()
        self.solver = NQueens(self.board_size.get())
        self.update_board_display()

    def on_mode_changed(self, event):
        if self.mode.get() == "Manual":
            self.manual_frame.pack(fill=tk.X, pady=5)
            self.animation_frame.pack_forget()
        elif self.mode.get() == "Animation":
            self.animation_frame.pack(fill=tk.X, pady=5)
            self.manual_frame.pack_forget()
        else:
            self.manual_frame.pack_forget()
            self.animation_frame.pack_forget()
        self.reset()

    def on_theme_changed(self, event):
        self.apply_theme(self.theme.get())
        self.update_board_display()
        self.update_visualization('distribution' if self.solver.solutions else None)

    def open_theme_customizer(self):
        custom_window = tk.Toplevel(self.root)
        custom_window.title("Customize Theme")
        custom_window.geometry("400x600")
        custom_window.transient(self.root)
        custom_window.grab_set()

        colors = ['Light Square', 'Dark Square', 'Queen', 'Highlight', 'Text', 'Background']
        current_colors = list(self.theme_manager.get_theme())
        color_vars = [tk.StringVar(value=color) for color in current_colors]

        for i, (label, var) in enumerate(zip(colors, color_vars)):
            frame = ttk.Frame(custom_window)
            frame.pack(fill=tk.X, padx=10, pady=5)
            ttk.Label(frame, text=f"{label}:").pack(side=tk.LEFT)
            ttk.Entry(frame, textvariable=var, width=10).pack(side=tk.LEFT, padx=5)
            ttk.Button(frame, text="Choose",
                       command=lambda v=var: v.set(colorchooser.askcolor()[1] or v.get())).pack(side=tk.LEFT)

        def apply_custom():
            self.theme_manager.set_custom_theme(*[var.get() for var in color_vars])
            self.apply_theme("Custom")
            self.update_board_display()
            self.update_visualization('distribution' if self.solver.solutions else None)
            custom_window.destroy()

        ttk.Button(custom_window, text="Apply", command=apply_custom).pack(pady=10)
        ttk.Button(custom_window, text="Cancel", command=custom_window.destroy).pack(pady=5)

    def on_board_click(self, event):
        if self.is_solving or self.mode.get() != "Manual" or self.solver.is_solution_complete():
            return
        n = self.board_size.get()
        if event.xdata is not None and event.ydata is not None:
            col = self.solver.current_col
            row = int(event.ydata + 0.5)
            if 0 <= row < n:
                if self.solver.place_queen_manually(row, col):
                    if self.solver.is_solution_complete():
                        if self.solver.verify_solution(self.solver.current_board):
                            messagebox.showinfo("Success", "Congratulations! You solved the N-Queens problem!")
                            self.solver.solutions.append([row[:] for row in self.solver.current_board])
                            self.solution_count.set(1)
                            self.selected_solution.set(0)
                            self.solutions_label.config(text="1")
                            self.solution_label.config(text="Solution: 1/1")
                            self.update_visualization('distribution')
                        else:
                            messagebox.showwarning("Invalid Solution", "This configuration is not a valid solution!")
                    self.update_board_display(animate=True)
                else:
                    messagebox.showwarning("Invalid Move", "Cannot place queen at this position!")

    def get_hint(self):
        if self.is_solving or self.mode.get() != "Manual" or self.solver.is_solution_complete():
            return
        algo = self.algorithm.get()
        col = self.solver.current_col
        if algo == "Backtracking":
            safe_positions = self.solver.get_safe_positions(col)
            if safe_positions:
                messagebox.showinfo("Hint", f"Try placing a queen at row {safe_positions[0]}")
                self.flash_position(col, safe_positions[0])
        else:
            ranked_positions = self.solver.get_heuristic_ranked_positions(col)
            if ranked_positions:
                messagebox.showinfo("Hint", f"Best position is row {ranked_positions[0]} (fewest conflicts)")
                self.flash_position(col, ranked_positions[0])

    def flash_position(self, col, row):
        def flash(frame):
            with self.animation_lock:
                self.ax.clear()
                n = self.board_size.get()
                light_square, dark_square, queen_color, highlight_color, text_color, _ = self.theme_manager.get_theme()
                cmap = ListedColormap([light_square, dark_square])
                chess_board = np.zeros((n, n))
                for i in range(n):
                    for j in range(n):
                        chess_board[i, j] = (i + j) % 2
                self.ax.imshow(chess_board, cmap=cmap)

                for i in range(n + 1):
                    self.ax.axhline(i - 0.5, color='#7f8c8d', linewidth=1)
                    self.ax.axvline(i - 0.5, color='#7f8c8d', linewidth=1)

                for i in range(n):
                    for j in range(n):
                        if self.solver.current_board[i][j] == 1:
                            self.ax.text(j, i, self.queen_styles[self.queen_style.get()], fontsize=30,
                                         ha='center', va='center', color=queen_color)

                if frame % 2 == 0:
                    self.ax.add_patch(
                        plt.Rectangle((col - 0.5, row - 0.5), 1, 1, fill=True, alpha=0.3, color=highlight_color))
                else:
                    self.ax.add_patch(
                        plt.Rectangle((col - 0.5, row - 0.5), 1, 1, fill=False, edgecolor=highlight_color, lw=3))

                self.ax.set_xticks([])
                self.ax.set_yticks([])
                self.ax.set_title(f"Manual Solving - Column {self.solver.current_col}", fontsize=14, pad=15,
                                  fontweight='bold', color=text_color)
                self.canvas.draw()
                return []

        hint_animation = animation.FuncAnimation(self.fig, flash, frames=6, interval=200, blit=True)
        self.root.after(1200, lambda: hint_animation.event_source.stop())

    def undo_move(self):
        if self.is_solving or self.mode.get() != "Manual":
            return
        if self.solver.undo_last_move():
            self.update_board_display()
        else:
            messagebox.showwarning("Cannot Undo", "No moves to undo!")

    def auto_complete(self):
        if self.is_solving or self.mode.get() != "Manual" or self.solver.is_solution_complete():
            return
        self.is_solving = True
        self.status_label.config(text="Status: Auto-completing...")
        self.progress.set(0)

        def auto_complete_thread():
            try:
                board = [row[:] for row in self.solver.current_board]
                col = self.solver.current_col
                solver = NQueens(self.board_size.get())
                solver.current_board = board
                solver.current_col = col

                def update_progress():
                    if self.is_solving:
                        current_steps = solver.steps
                        max_steps = self.board_size.get() * self.board_size.get() * 10
                        progress = min((current_steps / max_steps) * 100, 95)
                        self.progress.set(progress)
                        self.steps_label.config(text=f"Steps: {current_steps}")
                        self.root.after(100, update_progress)

                self.root.after(100, update_progress)

                if self.algorithm.get() == "Backtracking":
                    solver.solve_backtracking()
                elif self.algorithm.get() == "Heuristic":
                    solver.solve_heuristic()
                else:
                    solver.solve_genetic_algorithm()

                self.root.after(0, lambda: self.post_auto_complete(solver))

            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"An error occurred: {str(e)}"))
                self.root.after(0, lambda: self.status_label.config(text="Status: Error"))
                self.is_solving = False

        threading.Thread(target=auto_complete_thread, daemon=True).start()

    def post_auto_complete(self, solver):
        self.is_solving = False
        if solver.solutions:
            self.solver.current_board = solver.solutions[0]
            self.solver.current_col = self.board_size.get()
            self.solver.solutions = solver.solutions
            self.solution_count.set(1)
            self.selected_solution.set(0)
            self.solutions_label.config(text="1")
            self.solution_label.config(text="Solution: 1/1")
            if self.solver.verify_solution(self.solver.current_board):
                messagebox.showinfo("Success", "Solution completed successfully!")
            else:
                messagebox.showwarning("Invalid Solution", "Auto-completed solution is invalid!")
            self.update_board_display(animate=True)
            self.update_visualization('distribution')
        else:
            messagebox.showwarning("No Solution", "Could not find a solution from this position!")
        self.steps_label.config(text=f"Steps: {solver.steps}")
        self.time_label.config(text=f"Time: {solver.execution_time:.6f} s")
        self.status_label.config(text="Status: Completed")
        self.progress.set(100)

    def toggle_animation(self):
        if self.is_solving or self.mode.get() != "Animation" or not self.solver.solution_path:
            return
        self.animation_running = not self.animation_running
        self.anim_play_button.config(text="❚❚ Pause" if self.animation_running else "▶ Play")
        self.anim_status_label.config(text="Running" if self.animation_running else "Paused")
        if self.animation_running:
            threading.Thread(target=self.run_animation, daemon=True).start()

    def run_animation(self):
        while self.animation_running and self.anim_index < len(self.solver.solution_path):
            self.root.after(0, self.animation_step)
            time.sleep(0.5 / self.animation_speed.get())
        if self.anim_index >= len(self.solver.solution_path):
            self.root.after(0, lambda: self.anim_status_label.config(text="Finished"))
            self.root.after(0, lambda: self.anim_play_button.config(text="▶ Play"))
            self.animation_running = False

    def animation_step(self):
        if self.is_solving or self.mode.get() != "Animation" or not self.solver.solution_path:
            return
        with self.animation_lock:
            if self.anim_index < len(self.solver.solution_path):
                board, col, row = self.solver.solution_path[self.anim_index]
                self.update_board_display(board, animate=True)
                self.anim_step_label.config(text=f"{self.anim_index + 1}/{len(self.solver.solution_path)}")
                self.anim_index += 1
            else:
                self.animation_running = False
                self.anim_play_button.config(text="▶ Play")
                self.anim_status_label.config(text="Finished")

    def export_solution(self):
        if not self.solver.solutions:
            messagebox.showwarning("No Solution", "No solution to export!")
            return
        solution = self.solver.solutions[self.selected_solution.get()]
        n = self.board_size.get()
        fig, ax = plt.subplots(figsize=(8, 8))
        light_square, dark_square, queen_color, _, text_color, _ = self.theme_manager.get_theme()
        cmap = ListedColormap([light_square, dark_square])
        chess_board = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                chess_board[i, j] = (i + j) % 2
        ax.imshow(chess_board, cmap=cmap)

        for i in range(n + 1):
            ax.axhline(i - 0.5, color='#7f8c8d', linewidth=1)
            ax.axvline(i - 0.5, color='#7f8c8d', linewidth=1)

        for i in range(n):
            for j in range(n):
                if solution[i][j] == 1:
                    ax.text(j, i, self.queen_styles[self.queen_style.get()], fontsize=30,
                            ha='center', va='center', color=queen_color)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"N-Queens Solution {self.selected_solution.get() + 1}", fontsize=14, pad=15, fontweight='bold',
                     color=text_color)

        filename = f"nqueens_solution_{uuid.uuid4()}.png"
        fig.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close(fig)
        messagebox.showinfo("Export Success", f"Solution exported as {filename}")

    def verify_solution(self):
        if not self.solver.solutions and self.mode.get() != "Manual":
            messagebox.showwarning("No Solution", "No solution to verify!")
            return
        board = self.solver.current_board if self.mode.get() == "Manual" else self.solver.solutions[
            self.selected_solution.get()]
        if self.solver.verify_solution(board):
            messagebox.showinfo("Verification", "Solution is valid!")
        else:
            messagebox.showwarning("Verification", "Solution is invalid!")


def main():
    root = tk.Tk()
    app = NQueensApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
