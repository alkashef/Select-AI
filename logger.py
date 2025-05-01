"""
Logger module for centralized logging functionality.
"""
import os
import sys
import logging
from datetime import datetime
from typing import Optional, TextIO, Dict, Any
from dotenv import load_dotenv
from pathlib import Path


class Logger:
    """
    Centralized logging system that collects all logs, exceptions, and messages.
    Outputs to both console and a log file specified in the environment configuration.
    """
    _instance = None
    
    def __new__(cls) -> 'Logger':
        """
        Singleton pattern implementation to ensure only one logger instance exists.
        
        Returns:
            Logger: The single logger instance
        """
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self) -> None:
        """
        Initialize the logger with configurations from environment variables.
        """
        # Load environment variables
        env_path = os.path.join(os.path.dirname(__file__), 'config', '.env')
        load_dotenv(env_path)
        
        # Get log file path from environment or use default
        log_dir = os.getenv('LOG_DIR', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # Create log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_log_path = os.path.join(log_dir, f"borai_{timestamp}.log")
        self.log_file_path = os.getenv('LOG_FILE_PATH', default_log_path)
        
        # Configure the Python logging module
        self.logger = logging.getLogger('BorAI')
        self.logger.setLevel(logging.DEBUG)
        
        # File handler
        file_handler = logging.FileHandler(self.log_file_path)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Create formatter and add to handlers - modified to remove milliseconds and add square brackets
        formatter = logging.Formatter('[%(asctime)s] - %(name)s - %(levelname)s - %(message)s', 
                                     datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.info(f"Logger initialized. Logs will be saved to {self.log_file_path}")
    
    def info(self, message: str) -> None:
        """
        Log an info level message.
        
        Args:
            message: The message to log
        """
        self.logger.info(message)
    
    def debug(self, message: str) -> None:
        """
        Log a debug level message.
        
        Args:
            message: The message to log
        """
        self.logger.debug(message)
    
    def warning(self, message: str) -> None:
        """
        Log a warning level message.
        
        Args:
            message: The message to log
        """
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """
        Log an error level message.
        
        Args:
            message: The message to log
        """
        self.logger.error(message)
    
    def critical(self, message: str) -> None:
        """
        Log a critical level message.
        
        Args:
            message: The message to log
        """
        self.logger.critical(message)
    
    def exception(self, e: Exception, context: Optional[str] = None) -> None:
        """
        Log an exception with context information.
        
        Args:
            e: The exception to log
            context: Optional context information about where the exception occurred
        """
        context_info = f" in {context}" if context else ""
        self.logger.exception(f"Exception occurred{context_info}: {str(e)}")
    
    def log_operation(self, operation: str, status: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Log an operation with its status and additional details.
        
        Args:
            operation: The operation being performed
            status: The status of the operation (started, completed, failed)
            details: Additional details to log
        """
        message = f"Operation '{operation}' {status}"
        if details:
            message += f" - Details: {details}"
        self.info(message)


# Create a global instance that can be imported directly
logger = Logger()