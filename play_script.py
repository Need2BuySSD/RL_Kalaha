from kalaha_game import PlayableKalaha

if __name__ == "__main__":
    game = PlayableKalaha(pits_per_player=6, seeds_per_pit=6)
    game.play(input("Play against [P]layer or [AI]? "))