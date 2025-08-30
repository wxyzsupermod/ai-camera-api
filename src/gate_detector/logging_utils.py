def log(msg: str, level: str = 'info', *, cfg_level: str = 'info') -> None:
    levels = ['debug', 'info', 'warn', 'error']
    if levels.index(level) >= levels.index(cfg_level):
        print(f'[{level.upper()}] {msg}')
