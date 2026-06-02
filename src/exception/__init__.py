import sys
import logging


def error_message_detail(error: Exception, error_detail: sys) -> str:
    """
    Extract detailed error info: script name, line number, message.
    Mirrors the sales-price-main pattern exactly.
    """
    _, _, exc_tb = error_detail.exc_info()

    if exc_tb is None:
        # No active traceback — format a basic message
        error_message = f"Error: {str(error)}"
        logging.error(error_message)
        return error_message

    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = (
        f"Error occurred in python script: [{file_name}] "
        f"at line number [{line_number}]: {str(error)}"
    )
    logging.error(error_message)
    return error_message


class DischargeAgentException(Exception):
    """
    Custom exception for the Discharge Summary Agent.
    Captures file name and line number automatically from the active traceback.

    Usage:
        try:
            ...
        except Exception as e:
            raise DischargeAgentException(e, sys)
    """

    def __init__(self, error_message: Exception | str, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self) -> str:
        return self.error_message
