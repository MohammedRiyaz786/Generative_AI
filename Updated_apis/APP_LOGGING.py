import logging
from colorama import Fore, Style, init
init(autoreset=True)

class ColoredFormatter(logging.Formatter):
    COLORS = {
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT,
    }

    def format(self, record):
        log_message = super().format(record)
        return f"{self.COLORS.get(record.levelname, '')}{log_message}{Style.RESET_ALL}"

# Set up logging
def setup_logging():
    try:
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        colored_formatter = ColoredFormatter(log_format)
        
        # Root logger setup
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        # Check if handlers are already configured
        if not root_logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(colored_formatter)
            
            # File handler
            file_handler = logging.FileHandler("logs/app.log")
            file_handler.setFormatter(logging.Formatter(log_format))
            
            # Add the handlers to the root logger
            root_logger.addHandler(console_handler)
            root_logger.addHandler(file_handler)
        
        print(f"{Fore.CYAN}Logging setup completed successfully{Style.RESET_ALL}")
        return True
    except Exception as e:
        print(e)
        return False
