def print_color(message, color):
    # Function for displaying color messages
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "reset": "\033[0m",
    }
    print(f"{colors.get(color, colors['reset'])}{message}{colors['reset']}")
