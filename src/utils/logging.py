import logging
import sys

_loggers = {}


def get_logger(name: str) -> logging.Logger:
    """Logger nomeado que funciona tanto standalone (script rodado sem nenhuma
    configuração prévia de logging) quanto sob um entry point que já chamou
    logging.basicConfig (main.py e o bloco __main__ de cada script em src/data/) —
    só anexa handler próprio quando o root logger ainda não tem nenhum, evitando que
    a mensagem se propague para o root e seja impressa em duplicidade."""
    if name in _loggers:
        return _loggers[name]
    logger = logging.getLogger(name)
    if not logger.handlers and not logging.root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    _loggers[name] = logger
    return logger
