"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_winner(player) or game.is_loser(player):
        return game.utility(player)

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves - 2*opp_moves)

def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_winner(player) or game.is_loser(player):
        return game.utility(player)

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    factor1 = float(own_moves - 2*opp_moves)
    weight1 = 0.9

    factor2 = len(game.get_blank_spaces()) * game.move_count
    weight2 = 0.1
    return factor1 * weight1 + factor2 * weight2

def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_winner(player) or game.is_loser(player):
        return game.utility(player)

    # http://isolation-game-ai.herokuapp.com/Isolation-ProjectPaper.pdf
    #  (number of current player's moves - number of enemy's moves)/(num empty spaces +1)
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    empty_spaces = len(game.get_blank_spaces())
    return float((own_moves - opp_moves)/(empty_spaces + 1))


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout
        self.debug_mode = False


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def max_value(self, game, depth):
        """Select best move and maximum heuristic value.
        If specified depth is the first level, returns only best move.

        This is a modified version of MAX-VALUE in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        float
            The maximum heuristic value of the current game state to my agent;

        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        legal_moves = game.get_legal_moves()
        if not legal_moves or depth <= 0:
            return self.score(game, self)

        best_score = float("-inf")
        for move in legal_moves:
            best_score = max(best_score, self.min_value(game.forecast_move(move), depth-1))

        return best_score
        
    def min_value(self, game, depth):
        """Select best move and minimum heuristic value.
        If specified depth is the first level, returns only best move.

        This is a modified version of MIN-VALUE in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        float
            The minimum heuristic value of the current game state to my agent;

        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
            
        legal_moves = game.get_legal_moves()
        if not legal_moves or depth <= 0:
            return self.score(game, self)

        best_score = float("inf")
        for move in legal_moves:
            best_score = min(best_score, self.max_value(game.forecast_move(move), depth-1))

        return best_score
        
    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md
        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        best_move = (-1, -1)
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return best_move

        # Initialise a best move with the first option
        best_move = legal_moves[0]

        best_score = float("-inf")
        for move in legal_moves:
            score = self.min_value(game.forecast_move(move), depth-1)

            # If score is greater than the best score, take the move and score
            if best_score < score:
                best_score = score
                best_move = move

        return best_move

class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        
        """
        self.time_left = time_left
        self.search_depth = 1
        # print("*********************************************************************")

        legal_moves = game.get_legal_moves()
        #print("legal_moves=", legal_moves)
        if not legal_moves:
            #print("no more legal moves. Return (-1, -1)")
            return (-1, -1)
        elif len(legal_moves) == 1:
            # print("return only option ", legal_moves[0])
            return legal_moves[0]

        first_move, is_first_move = self.get_first_move(game)
        if is_first_move:
            return first_move

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = legal_moves[0]

        # Initte loop until hitting timer threshold
        #search_depth_threshold = 30
        while True:
            try:
                # Iterative deepening search
                # print("-----------------------------")
                # print("Iterative deepening search, search_depth:", self.search_depth)
                move = self.alphabeta(game, self.search_depth)
                if move == (-1, -1):
                    #print("(-1, -1) is returned from alphabeta at depth ", self.search_depth)
                    break
                
                best_move = move
                # print("updated best move:", best_move)
                # if search_depth_threshold < self.search_depth:
                #     break

                self.search_depth += 1

            except SearchTimeout:
                # When hitting timer threshold, stop iterative deepening search
                # print("SearchTimeout occurred.")
                break

        # Return the best move from the last completed search iteration
        #print("==========> Return the best move:", best_move)
        return best_move

    def get_first_move(self, game):
        """Select first move from the good opening book based on the lesson Solving 5x5 Isolation from Malcolm
            1. Move to the centre if possible
            2. Move to next to opponent
        """
        if game.move_count > 1:
            return (-1,-1), False

        h, w = game.height, game.width
        centre_move = ((h-1)//2, (w-1)//2)
        # Move to the centre if possible
        if game.move_is_legal(centre_move):
            return centre_move, True

        # First move for second player in case that the centre is occupired
        opp_loc = game.get_player_location(game.get_opponent(self))
        # Try to move next to opponent
        next_to_opp = (opp_loc[0]+1, opp_loc[1])
        if game.move_is_legal(next_to_opp):
            return next_to_opp, True
        else:
            return (opp_loc[0], opp_loc[1]+1), True

    def max_value(self, game, depth, alpha, beta):
        """Select best move and maximum heuristic value.
        If specified depth is the first level, returns only best move.

        This is a modified version of MAX-VALUE in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        float
            The maximum heuristic value of the current game state to my agent;
            Negative infinity if there are no legal moves;

        """
        #if self.debug_mode: print("+++++++++++++++ max_value(start):depth:",depth, " ++++++++++++++++++")
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        #print("max_value: depth==>", depth, ", alpha=", alpha, ", beta=", beta)
        legal_moves = game.get_legal_moves()
        if not legal_moves or depth <= 0:
            return self.score(game, self)

        best_score = float("-inf")
        #print("First value ==> best_move:", best_move, ", best_score:", best_score)
        #if self.debug_mode: print("legal_moves at depth(", depth, ")=", legal_moves)
        for move in legal_moves:
            #if self.debug_mode: print("******")
            #if self.debug_mode: print("try move:", m, "at depth(", depth, ")...")
            best_score = max(best_score, self.min_value(game.forecast_move(move), depth-1, alpha, beta))
            #if self.debug_mode: print("move:", m, "'s score: ",score ," at depth(", depth, ")...")

            # If score is greater than the best score, take the move and score
            # if best_score < score:
            #     best_score = score
                #if self.debug_mode: print("updating best_move:", best_move, ", best_score:", best_score)
            if best_score >= beta:
                #if self.debug_mode: print("beta cut-off, alpha=", alpha, ", beta=", beta)
                return best_score

            alpha = max(alpha, best_score)
            # if alpha < best_score:
            #     alpha = best_score
                #if self.debug_mode: print("updating alpha:", alpha)
            # if beta <= alpha:
            #     #if self.debug_mode: print("beta cut-off, alpha=", alpha, ", beta=", beta)
            #     break

        #if self.debug_mode: print("returning best_move(depth:", depth, "):", best_move, ", best_score:", best_score, "alpha=", alpha, ", beta=", beta)
        #if self.debug_mode: print("+++++++++++++++ max_value(end):depth:",depth, " ++++++++++++++++++")
        return best_score
        
    def min_value(self, game, depth, alpha, beta):
        """Select best move and minimum heuristic value.
        If specified depth is the first level, returns only best move.

        This is a modified version of MIN-VALUE in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        float
            The minimum heuristic value of the current game state to my agent;
            Infinity if there are no legal moves;

        """
        #if self.debug_mode: print("--------------------- min_value(start):depth:",depth, " -----------------------")
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
            
        #if self.debug_mode: print("min_value: depth==>", depth, ", alpha=", alpha, ", beta=", beta)
        legal_moves = game.get_legal_moves()
        if not legal_moves or depth <= 0:
            return self.score(game, self)

        best_score = float("inf")
        #if self.debug_mode: print("First value ==> best_move:", best_move, ", best_score:", best_score)
        #if self.debug_mode: print("legal_moves at depth(", depth, ")=", legal_moves)
        for move in legal_moves:
            #if self.debug_mode: print("******")
            #if self.debug_mode: print("try move:", m, "at depth(", depth, ")...")
            best_score = min(best_score, self.max_value(game.forecast_move(move), depth-1, alpha, beta))
            #if self.debug_mode: print("move:", m, "'s score: ",score ," at depth(", depth, ")...")

            # If score is less than the best score, take the move and score
            # if best_score > score:
            #     best_score = score
                #if self.debug_mode: print("updating best_move:", best_move, ", best_score:", best_score)

            if best_score <= alpha:
                #if self.debug_mode: print("alpha cut-off, alpha=", alpha, ", beta=", beta)
                return best_score

            beta = min(beta, best_score)
            # if beta > best_score:
            #     beta = best_score
                #if self.debug_mode: print("updating beta:", beta)
            #if self.debug_mode: print("best_move:", best_move, ", best_score:", best_score, "alpha=", alpha, ", beta=", beta)
            # if beta <= alpha:
            #     #if self.debug_mode: print("alpha cut-off, alpha=", alpha, ", beta=", beta)
            #     break

        #if self.debug_mode: print("returning best_move(depth:", depth, "):", best_move, ", best_score:", best_score, "alpha=", alpha, ", beta=", beta)
        #if self.debug_mode: print("--------------------- min_value(end):depth:",depth, " -----------------------")
        return best_score

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        float
            The heuristic value of the current game state to my agent;

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        best_move = (-1, -1)
        legal_moves = game.get_legal_moves()
        #print("legal_moves(alphabeta)=", legal_moves)
        if not legal_moves:
            #print("no more legal moves. Return (-1, -1)")
            return best_move

        # Initialise a best move with the first option
        best_move = legal_moves[0]
        best_score = float("-inf")
        for move in legal_moves:
            score = self.min_value(game.forecast_move(move), depth-1, alpha, beta)

            # If score is greater than the best score, take the move and score
            if best_score < score:
                best_score = score
                best_move = move

            alpha = max(alpha, best_score)
        #print("====> Return the best move(alphabeta):", best_move)
        return best_move
