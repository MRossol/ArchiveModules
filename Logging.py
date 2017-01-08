"""
Set up logger for logging
"""
import logging


def setup_logger(logger_name, log_file, level=logging.INFO, stream=True):
    """
    Set up logger for logging
    Parameters
    ----------
    logger_name: 'string'
        name of logger
    log_file: 'string'
        file path for logging file
    level: 'logging.type'
        Level of logging, default is INFO
    stream: 'bool'
        Stream logs as well as write to file

    Returns
    -------
    fileHandler: 'logging.FileHandler'
        Logging handler for file
    streamHandler: 'logging.StreamHandler'
        If stream is True, logging handler for streaming
    """
    # Create logger with name logger_name
    l = logging.getLogger(logger_name)
    # Format logger to output time - level - message
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # Create file handler
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fileHandler)
    # If desired create stream handler
    if stream:
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)
        l.addHandler(streamHandler)
